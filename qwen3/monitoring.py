"""引擎运行状态快照 — 训推同机监控（JSONL 落盘 / HTTP JSON）

场景：训练框架（如 GRPO）与推理引擎**同进程**，训练侧每个 step 采集一次
引擎状态，追加写入 JSONL 文件；训练结束后离线绘图（pandas/wandb）。

用法::

    from qwen3.monitoring import append_snapshot

    for step in range(num_steps):
        rollouts = ...                 # 用 engine 生成
        append_snapshot("logs/engine.jsonl", engine,
                        model_id="qwen3-0.6b", step=step, reward=reward)

设计原则：
  - 只返回纯 Python dict（可直接 json.dump），不做差分/绘图（训练侧决定）
  - 不 import server.py（引擎只需 duck-typing：metrics / scheduler / pool / device）
  - 字段名稳定，新增字段只加不改
  - TTFT 分位数与 Prometheus `/metrics` 文本口径一致（nearest，非插值）
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


def _nearest_percentile(sorted_vals: list[float], q: float) -> float:
    """与 server.render_metrics 相同的分位数口径（nearest）"""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def snapshot(engine, model_id: str = "") -> dict[str, Any]:
    """采集一次引擎状态快照（纯 dict，可直接 json.dump）

    Args:
        engine: 推理引擎实例（server.Engine）。需要以下属性：
                - metrics:  EngineMetrics（计数/步数/TTFT 窗口）
                - scheduler: Scheduler（running/waiting）
                - pool:      KVCachePool（free_count/total_blocks/cached_count）
                - device:    torch.device（可选，用于 GPU 显存字段）
        model_id: 模型标识（写入快照，便于多实验区分）

    Returns:
        字段稳定的嵌套 dict，见模块文档/`API.md`。
    """
    m = engine.metrics
    sched = engine.scheduler
    pool = engine.pool

    used = pool.total_blocks - pool.free_count
    ratio = used / pool.total_blocks if pool.total_blocks else 0.0

    # TTFT 滑动窗口（Engine 线程会 append，HTTP/训练线程读取 → 加锁）
    lock = getattr(m, "lock", None)
    if lock is not None:
        with lock:
            ttft_vals = list(m.ttfts)
    else:
        ttft_vals = list(getattr(m, "ttfts", []))
    ttft_sorted = sorted(ttft_vals)

    # GPU 显存：同进程统计包含训练 + 推理（训推同机时最关心）
    gpu = None
    device = getattr(engine, "device", None)
    if device is not None and getattr(device, "type", None) == "cuda":
        try:
            import torch
            gpu = {
                "device": str(device),
                "memory_allocated_mb": round(
                    torch.cuda.memory_allocated(device) / 1024 ** 2, 1),
                "memory_reserved_mb": round(
                    torch.cuda.memory_reserved(device) / 1024 ** 2, 1),
            }
        except Exception:
            gpu = None

    return {
        "timestamp": time.time(),
        "uptime_s": round(time.perf_counter() - m.start_time, 3),
        "model": model_id,
        "requests": {
            "total": m.requests_total,
            "finished": m.finished_total,
            "failed": m.failed_total,
            "tokens": m.tokens_total,
        },
        "scheduler": {
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "steps": m.steps_total,
            "last_batch_size": m.last_batch_size,
        },
        "kv": {
            "used_ratio": round(ratio, 6),
            "free_blocks": pool.free_count,
            "total_blocks": pool.total_blocks,
            "cached_blocks": pool.cached_count,
            "cache_capacity": pool.max_cached_blocks,
        },
        "ttft_s": {
            "p50": round(_nearest_percentile(ttft_sorted, 0.50), 6),
            "p95": round(_nearest_percentile(ttft_sorted, 0.95), 6),
            "p99": round(_nearest_percentile(ttft_sorted, 0.99), 6),
            "count": len(ttft_sorted),
            "sum": round(sum(ttft_sorted), 6),
        },
        "prefix_cache": {
            "hits": getattr(m, "prefix_cache_hits_total", 0),
            "hit_blocks": getattr(m, "prefix_cache_hit_blocks_total", 0),
            "evictions": getattr(pool, "cache_evictions", 0),
        },
        "gpu": gpu,
    }


def append_snapshot(path: str, engine, model_id: str = "",
                    **extra: Any) -> dict[str, Any]:
    """采集快照并**追加一行 JSONL**（训练每个 step 一行，供离线绘图）

    extra: 训练侧附加字段（如 step=、loss=、reward=），会合并到该行顶层。
    Returns: 写入的 dict（便于训练端同时上报）。
    """
    snap = snapshot(engine, model_id)
    if extra:
        snap.update(extra)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")
    return snap


def write_snapshot(path: str, engine, model_id: str = "",
                   **extra: Any) -> dict[str, Any]:
    """采集快照并**覆盖写**单个 JSON 文件（一次性导出/调试用）"""
    snap = snapshot(engine, model_id)
    if extra:
        snap.update(extra)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return snap
