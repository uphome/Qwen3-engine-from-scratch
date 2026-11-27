#!/usr/bin/env python3
"""GRPO 训推同机监控示例（可直接运行，使用 fake engine 生成示例 JSONL）

真实训练时（同进程）::

    from server import Engine
    from qwen3.monitoring import append_snapshot

    engine = Engine(model, pool, runner, batch_size=8)   # 你的推理引擎
    engine.start()

    for step in range(num_steps):
        rollouts = ...                                   # 用 engine 生成
        append_snapshot("logs/engine.jsonl", engine,
                        model_id="qwen3-0.6b", step=step,
                        reward=mean_reward, loss=loss)

本脚本用 fake engine 演示字段与文件格式（不加载模型）::

    python integrations/grpo_example.py --steps 20 --out logs/engine.jsonl
"""

import argparse
import os
import random
import sys
from collections import deque
from types import SimpleNamespace

import torch

# 从仓库根导入 qwen3（脚本位于 integrations/ 子目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3.monitoring import append_snapshot


def make_fake_engine():
    """构造一个"形状正确"的 fake engine（字段与 server.Engine 一致）"""
    from server import EngineMetrics

    metrics = EngineMetrics()
    metrics.ttfts.extend([0.04, 0.05, 0.06, 0.08, 0.1])
    engine = SimpleNamespace(
        metrics=metrics,
        scheduler=SimpleNamespace(running=[], waiting=deque()),
        pool=SimpleNamespace(total_blocks=267, free_count=267,
                             cached_count=0, max_cached_blocks=200),
        device=torch.device("cpu"),
    )
    return engine


def main():
    ap = argparse.ArgumentParser(description="GRPO 训推同机监控示例")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", type=str, default="logs/engine.jsonl")
    args = ap.parse_args()

    engine = make_fake_engine()
    random.seed(0)

    for step in range(args.steps):
        # 模拟一个训练 step 的 rollout 统计（真实场景由引擎自己更新）
        m = engine.metrics
        n_req = random.randint(4, 16)
        m.requests_total += n_req
        m.finished_total += n_req
        m.tokens_total += n_req * random.randint(16, 64)
        m.steps_total += random.randint(20, 60)
        m.last_batch_size = random.randint(1, 8)
        m.prefix_cache_hits_total += random.randint(0, 2)
        m.prefix_cache_hit_blocks_total += random.randint(0, 20)
        m.ttfts.append(random.uniform(0.04, 0.12))
        engine.pool.free_count = random.randint(120, 240)
        engine.pool.cached_count = 267 - engine.pool.free_count
        engine.scheduler.waiting = deque([0] * random.randint(0, 3))
        engine.scheduler.running = [0] * random.randint(0, 6)

        snap = append_snapshot(
            args.out, engine, model_id="qwen3-0.6b-fake",
            step=step, reward=round(random.uniform(0, 1), 3))
        if step % 5 == 0:
            print(f"[step {step}] tokens/s≈{snap['requests']['tokens']} "
                  f"waiting={snap['scheduler']['waiting']} "
                  f"kv={snap['kv']['used_ratio']:.2f}")

    print(f"\n已写入 {args.steps} 行 -> {args.out}")
    print("离线绘图示例:")
    print("  import pandas as pd")
    print(f"  df = pd.read_json('{args.out}', lines=True)")
    print("  df[['step', 'requests', 'scheduler', 'kv']]")


if __name__ == "__main__":
    main()
