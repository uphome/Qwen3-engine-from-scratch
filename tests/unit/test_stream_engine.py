"""SSE 流式：Engine 逐 token 推送 + 客户端取消清理（CPU tiny model）"""

import time

import torch

from server import Engine, Job


def make_stream_job(engine: Engine, max_new_tokens: int = 3) -> Job:
    job = Job(
        request_id=engine._next_id,
        input_ids=torch.arange(1, 4).unsqueeze(0),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )
    job.is_stream = True
    engine._next_id += 1
    return job


def test_stream_job_emits_tokens_then_sentinel(tiny_model, kv_pool_factory):
    pool = kv_pool_factory(num_blocks=32, max_seq_len=128)
    engine = Engine(tiny_model, pool, runner=None, batch_size=1)
    engine.start()
    try:
        job = make_stream_job(engine, max_new_tokens=3)
        engine.submit(job)

        tokens = []
        while True:
            tok = job.token_queue.get(timeout=60)
            if tok is None:            # 结束哨兵
                break
            tokens.append(tok)

        assert job.done.wait(5)
        assert job.error is None
        assert tokens == job.output_ids       # 推送顺序 == 最终生成序列
        assert len(tokens) == 3
        assert engine.metrics.finished_total == 1
    finally:
        engine.stop()


def test_cancel_stops_request_and_returns_blocks(tiny_model, kv_pool_factory):
    pool = kv_pool_factory(num_blocks=32, max_seq_len=256)
    engine = Engine(tiny_model, pool, runner=None, batch_size=1)
    engine.start()
    try:
        job = make_stream_job(engine, max_new_tokens=500)
        engine.submit(job)

        first = job.token_queue.get(timeout=60)
        assert first is not None

        job.cancelled.set()             # 模拟客户端断开

        deadline = time.time() + 15
        while time.time() < deadline and not job.done.is_set():
            time.sleep(0.05)
        assert job.done.is_set(), "取消后请求未及时清理"

        # 等引擎下一轮循环完成清理
        deadline = time.time() + 10
        while time.time() < deadline:
            if (not engine.scheduler.running and not engine.scheduler.waiting
                    and pool.free_count == pool.total_blocks):
                break
            time.sleep(0.05)

        assert not engine.scheduler.running
        assert not engine.scheduler.waiting
        assert pool.free_count == pool.total_blocks     # KV 块全部归还
        assert job.request_id not in engine._jobs
    finally:
        engine.stop()


def test_stream_and_non_stream_produce_same_tokens(tiny_model, kv_pool_factory):
    """同一 prompt 的流式 token 序列 == 贪心非流式输出"""
    pool = kv_pool_factory(num_blocks=32, max_seq_len=128)
    engine = Engine(tiny_model, pool, runner=None, batch_size=1)
    engine.start()
    try:
        # 非流式
        job1 = Job(request_id=engine._next_id,
                   input_ids=torch.arange(1, 4).unsqueeze(0),
                   max_new_tokens=5, temperature=0.0, top_k=0, top_p=1.0)
        engine._next_id += 1
        engine.submit(job1)
        assert job1.done.wait(60)
        assert job1.error is None

        # 流式（同输入）
        job2 = make_stream_job(engine, max_new_tokens=5)
        job2.input_ids = torch.arange(1, 4).unsqueeze(0)
        engine.submit(job2)
        tokens = []
        while True:
            tok = job2.token_queue.get(timeout=60)
            if tok is None:
                break
            tokens.append(tok)

        assert tokens == job1.output_ids
    finally:
        engine.stop()


def test_stream_job_rejected_wakes_handler(tiny_model, kv_pool_factory):
    """容量拒绝时必须往 token_queue 推哨兵，否则流式 handler 挂到超时"""
    pool = kv_pool_factory(num_blocks=1, max_seq_len=16)   # 只够 1 页
    engine = Engine(tiny_model, pool, runner=None, batch_size=1)
    engine.start()
    try:
        job = make_stream_job(engine, max_new_tokens=64)   # 需要多页
        engine.submit(job)
        tok = job.token_queue.get(timeout=30)
        assert tok is None
        assert job.error is not None and "池容量不足" in job.error
        assert engine.metrics.failed_total == 1
    finally:
        engine.stop()
