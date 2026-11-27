"""训推同机监控接口：快照结构 / JSONL 追加 / 与 Prometheus 文本口径一致"""

import json
from collections import deque
from types import SimpleNamespace

import torch

from qwen3.monitoring import append_snapshot, snapshot, write_snapshot
from server import EngineMetrics, render_metrics


def make_engine():
    metrics = EngineMetrics()
    metrics.requests_total = 12
    metrics.finished_total = 10
    metrics.failed_total = 2
    metrics.tokens_total = 640
    metrics.steps_total = 88
    metrics.last_batch_size = 4
    metrics.ttfts.extend([0.05, 0.08, 0.12, 0.3])
    metrics.prefix_cache_hits_total = 3
    metrics.prefix_cache_hit_blocks_total = 48
    pool = SimpleNamespace(total_blocks=16, free_count=10,
                           cached_count=2, max_cached_blocks=12)
    scheduler = SimpleNamespace(running=[1, 2], waiting=deque([3]))
    return SimpleNamespace(metrics=metrics, pool=pool, scheduler=scheduler,
                           device=torch.device("cpu"))


def test_snapshot_structure_and_values():
    engine = make_engine()
    snap = snapshot(engine, "qwen3-test")

    assert snap["model"] == "qwen3-test"
    assert snap["requests"] == {"total": 12, "finished": 10, "failed": 2,
                                "tokens": 640}
    assert snap["scheduler"] == {"running": 2, "waiting": 1, "steps": 88,
                                 "last_batch_size": 4}
    # used = 16 - 10 = 6 → 6/16
    assert snap["kv"]["used_ratio"] == 0.375
    assert snap["kv"]["free_blocks"] == 10
    assert snap["kv"]["total_blocks"] == 16
    assert snap["kv"]["cached_blocks"] == 2
    assert snap["kv"]["cache_capacity"] == 12
    assert snap["ttft_s"]["count"] == 4
    assert snap["ttft_s"]["p50"] > 0
    assert snap["prefix_cache"] == {"hits": 3, "hit_blocks": 48,
                                     "evictions": 0}
    assert snap["gpu"] is None                     # CPU 设备无 GPU 字段
    assert snap["timestamp"] > 0
    assert snap["uptime_s"] >= 0

    # 必须可直接 JSON 序列化（训练端 json.dump 的前提）
    json.dumps(snap)


def test_ttft_percentile_matches_prometheus_text():
    """快照与 /metrics 文本的 TTFT 分位数口径必须一致（nearest）"""
    engine = make_engine()
    snap = snapshot(engine)

    text = render_metrics(engine, "qwen3-test")
    text_vals = {}
    for line in text.splitlines():
        if line.startswith("engine_ttft_seconds"):
            name, val = line.rsplit(" ", 1)
            text_vals[name] = float(val)

    assert snap["ttft_s"]["p50"] == round(
        text_vals['engine_ttft_seconds{quantile="0.5"}'], 6)
    assert snap["ttft_s"]["p95"] == round(
        text_vals['engine_ttft_seconds{quantile="0.95"}'], 6)
    assert snap["ttft_s"]["count"] == int(
        text_vals["engine_ttft_seconds_count"])


def test_append_snapshot_jsonl(tmp_path):
    engine = make_engine()
    path = tmp_path / "monitor" / "engine.jsonl"

    s1 = append_snapshot(str(path), engine, "qwen3-test", step=0, reward=0.5)
    s2 = append_snapshot(str(path), engine, "qwen3-test", step=1, reward=0.7)
    assert s1["step"] == 0 and s2["step"] == 1
    assert s1["reward"] == 0.5

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2                       # 每 step 一行
    rows = [json.loads(line) for line in lines]
    assert rows[0]["requests"]["total"] == 12
    assert rows[1]["step"] == 1
    assert rows[0]["model"] == "qwen3-test"


def test_write_snapshot_overwrite(tmp_path):
    engine = make_engine()
    path = tmp_path / "snap.json"
    write_snapshot(str(path), engine, "qwen3-test", step=9)
    write_snapshot(str(path), engine, "qwen3-test", step=10)

    data = json.loads(path.read_text())
    assert data["step"] == 10                    # 覆盖写，不是追加
    assert len(path.read_text().strip().splitlines()) > 1
    assert path.read_text().count('"requests"') == 1


def test_snapshot_works_without_lock():
    """duck typing：没有 lock/prefix 指标的自定义 metrics 也能采集"""
    metrics = SimpleNamespace(
        start_time=0.0, requests_total=1, finished_total=1, failed_total=0,
        tokens_total=8, steps_total=2, last_batch_size=1, ttfts=[0.1],
    )
    engine = SimpleNamespace(
        metrics=metrics,
        pool=SimpleNamespace(total_blocks=4, free_count=4,
                             cached_count=0, max_cached_blocks=3),
        scheduler=SimpleNamespace(running=[], waiting=deque()),
        device=torch.device("cpu"),
    )
    snap = snapshot(engine)
    assert snap["ttft_s"]["count"] == 1
    assert snap["prefix_cache"] == {"hits": 0, "hit_blocks": 0,
                                     "evictions": 0}
    assert snap["uptime_s"] > 0                  # start_time=0 → 运行时长很大
