"""Engine 指标 + /metrics 文本渲染（CPU tiny model，无需真模型）"""


import torch

from server import Engine, Job, render_metrics


def make_job(engine: Engine, max_new_tokens: int = 3, length: int = 3) -> Job:
    job = Job(
        request_id=engine._next_id,
        input_ids=torch.arange(1, length + 1).unsqueeze(0),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )
    engine._next_id += 1
    return job


def test_engine_counts_and_ttft(tiny_model, kv_pool_factory):
    pool = kv_pool_factory(num_blocks=32, max_seq_len=128)
    engine = Engine(tiny_model, pool, runner=None, batch_size=2)
    engine.start()
    try:
        jobs = [make_job(engine) for _ in range(2)]
        for job in jobs:
            engine.submit(job)
        for job in jobs:
            assert job.done.wait(60), "生成超时"
            assert job.error is None, job.error

        m = engine.metrics
        assert m.requests_total == 2
        assert m.finished_total == 2
        assert m.failed_total == 0
        assert m.tokens_total == sum(len(j.output_ids) for j in jobs) == 6
        assert m.steps_total >= 2
        assert len(m.ttfts) == 2
        assert all(t > 0 for t in m.ttfts)
        assert all(j.ttft is not None and j.ttft > 0 for j in jobs)
    finally:
        engine.stop()


def test_engine_rejects_when_pool_too_small(tiny_model, kv_pool_factory):
    pool = kv_pool_factory(num_blocks=1, max_seq_len=16)   # 只够 1 页
    engine = Engine(tiny_model, pool, runner=None, batch_size=1)
    engine.start()
    try:
        job = make_job(engine, max_new_tokens=64, length=16)   # 需要多页
        engine.submit(job)
        assert job.done.wait(30)
        assert job.error is not None and "拒绝入队" in job.error
        assert engine.metrics.failed_total == 1
        assert engine.metrics.finished_total == 0
    finally:
        engine.stop()


def test_render_metrics_prometheus_format(tiny_model, kv_pool_factory):
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    engine = Engine(tiny_model, pool, runner=None, batch_size=2)
    m = engine.metrics
    m.requests_total = 5
    m.finished_total = 4
    m.failed_total = 1
    m.tokens_total = 128
    m.steps_total = 40
    m.last_batch_size = 2
    m.ttfts.extend([0.05, 0.08, 0.12, 0.3])
    pool.alloc(4)     # 占用 4 块 → used_ratio = 4/16

    text = render_metrics(engine, "qwen3-test")
    lines = text.splitlines()

    # 每个指标都有 HELP/TYPE，且样本值可被浮点解析
    for line in lines:
        assert line
        if line.startswith("#"):
            continue
        name_part, value = line.rsplit(" ", 1)
        float(value)

    assert "# TYPE engine_requests_total counter" in lines
    assert "engine_requests_total 5" in lines
    assert "engine_requests_finished_total 4" in lines
    assert "engine_requests_failed_total 1" in lines
    assert "engine_tokens_total 128" in lines
    assert "engine_kv_used_ratio 0.250000" in lines
    assert "engine_kv_free_blocks 12" in lines
    assert "engine_kv_total_blocks 16" in lines
    assert 'engine_ttft_seconds{quantile="0.5"}' in text
    assert "engine_ttft_seconds_count 4" in text
    assert 'engine_model_info{model="qwen3-test"} 1' in lines


def test_render_metrics_without_ttft_samples(tiny_model, kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    engine = Engine(tiny_model, pool, runner=None, batch_size=1)

    text = render_metrics(engine, "")
    assert "engine_ttft_seconds_count" not in text   # 无样本时不输出 count/sum
    assert "engine_model_info" not in text           # 无 model_id 时不输出
