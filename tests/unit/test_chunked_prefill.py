"""chunked prefill：长 prompt 切片、状态机、与 decode 交替

纯调度层测试（不跑模型）：手动模拟每个 chunk 的 KV 写入 + 状态收尾。
"""

import torch

from qwen3 import Scheduler
from qwen3.request import Request, DECODING, PREFILLING


def make_req(rid: int, length: int, max_new: int = 8) -> Request:
    return Request(
        request_id=rid,
        input_ids=torch.arange(length).unsqueeze(0) + rid * 1000,
        max_new_tokens=max_new,
        eos_token_id=99999,
    )


def simulate_prefill(sched: Scheduler, batch, pool):
    """模拟一个 prefill chunk 的前向：写 KV → advance → on_step_done"""
    for req in batch.requests:
        n = req.prefill_end - req.prefill_offset
        req.kv_cache.update(
            0,
            torch.randn(1, pool.num_kv_heads, n, pool.head_dim),
            torch.randn(1, pool.num_kv_heads, n, pool.head_dim))
        req.kv_cache.advance_seq_len(n)
    sched.on_step_done(batch, torch.zeros(batch.size, 1, dtype=torch.long))


def test_long_prompt_split_into_chunks(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=128, max_seq_len=2048)
    sched = Scheduler(pool, batch_size=1, chunk_size=512)

    req = make_req(0, 1200)
    sched.add(req)

    lens = []
    while req.state != DECODING:
        batch = sched.schedule()
        assert batch.mode == "prefill"
        lens.append(batch.input_lens[0])
        simulate_prefill(sched, batch, pool)
        assert req.state in (PREFILLING, DECODING)

    assert lens == [512, 512, 176]          # 按 chunk 切片
    assert req.kv_cache.seq_len == 1200
    assert req.kv_cache.num_pages == 75     # ceil(1200/16)
    assert len(sched.running) == 1          # 完成后进入 decode


def test_chunk_size_zero_disables_split(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=128, max_seq_len=2048)
    sched = Scheduler(pool, batch_size=1, chunk_size=0)

    req = make_req(0, 1200)
    sched.add(req)
    batch = sched.schedule()
    assert batch.input_lens == [1200]       # 不分片
    simulate_prefill(sched, batch, pool)
    assert req.state == DECODING


def test_decode_interleaves_with_prefill_chunks(kv_pool_factory):
    """running 有请求时，prefill chunk 与 decode 交替，decode 不被整段卡住"""
    pool = kv_pool_factory(num_blocks=64, max_seq_len=1024)
    sched = Scheduler(pool, batch_size=2, chunk_size=64)

    # 先让短请求完成 prefill 进入 running
    short = make_req(0, 16, max_new=100)
    sched.add(short)
    batch = sched.schedule()
    assert batch.mode == "prefill"
    simulate_prefill(sched, batch, pool)
    assert short.state == DECODING

    # 长请求入队：后续应在 prefill chunk 与 decode 之间交替
    long_req = make_req(1, 200, max_new=8)
    sched.add(long_req)

    modes = []
    guard = 0
    while long_req.state != DECODING and guard < 50:
        batch = sched.schedule()
        modes.append(batch.mode)
        if batch.mode == "prefill":
            simulate_prefill(sched, batch, pool)
        else:
            # decode 短请求：写 1 token + 收尾
            for req in batch.requests:
                req.kv_cache.reserve_next()
                req.kv_cache.advance_seq_len(1)
            sched.on_step_done(batch, torch.zeros(batch.size, 1, dtype=torch.long))
        guard += 1

    # 交替节奏（起始相位取决于上一步）：相邻两步模式一定不同
    assert all(modes[i] != modes[i + 1] for i in range(len(modes) - 1)), modes
    assert modes.count("prefill") == 4      # 200 / 64 → 64,64,64,8
    assert modes.count("decode") == 4
    assert long_req.kv_cache.seq_len == 200


def test_pool_closed_after_chunked_prefill(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=128, max_seq_len=2048)
    sched = Scheduler(pool, batch_size=1, chunk_size=128)

    req = make_req(0, 300, max_new=2)
    sched.add(req)
    while req.state != DECODING:
        simulate_prefill(sched, sched.schedule(), pool)

    # 模拟 decode 直到完成（prefill 已产出首 token，max_new=2 → 再 1 步）
    while sched.has_pending():
        batch = sched.schedule()
        for r in batch.requests:
            r.kv_cache.reserve_next()
            r.kv_cache.advance_seq_len(1)
        sched.on_step_done(batch, torch.zeros(batch.size, 1, dtype=torch.long))

    assert req.state == "finished"
    assert pool.free_count + pool.cached_count == pool.total_blocks
