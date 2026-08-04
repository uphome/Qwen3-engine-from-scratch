"""
Phase 2 验证 — 批量 decode 正确性

对照:
  1. 调度器（批前向）跑出的输出 == 逐请求直接调 generate() 的输出（逐 token）
  2. 池的物理块分配/释放闭环（free_count 回到初始值）
  3. 状态机迁移合法
  4. 多 seed × 多 batch_size × 变长输入

用法:
    python test/test_batched.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool, Scheduler
from qwen3.request import Request
from generate import generate


def build_model(seed: int = 0):
    torch.manual_seed(seed)
    config = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=512,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model, config


def make_requests(config, n, seed=42, max_out=16):
    torch.manual_seed(seed)
    reqs = []
    for i in range(n):
        input_len = 8 + (i * 5) % 16
        input_ids = torch.randint(0, 512, (1, input_len))
        reqs.append(Request(
            request_id=i,
            input_ids=input_ids,
            max_new_tokens=max_out,
            eos_token_id=151645,   # vocab 内不存在 → 不触发 EOS
        ))
    return reqs


@torch.no_grad()
def run_scheduler(model, config, requests, batch_size=2):
    """调度主循环：prefill 逐请求 + decode 真批量，与 batched_generate.py 一致"""
    kv_pool = KVCachePool(
        num_blocks=64, num_layers=config.num_hidden_layers,
        block_size=16, num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim, device=torch.device("cpu"), dtype=torch.float32,
    )
    scheduler = Scheduler(kv_pool, batch_size=batch_size)
    for req in requests:
        scheduler.add(req)

    while scheduler.has_pending():
        batch = scheduler.schedule()
        assert batch is not None

        input_ids = batch.build_input_ids()
        kv_caches = batch.build_kv_caches()
        logits = model(input_ids, kv_cache=kv_caches)
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 贪心

        scheduler.on_step_done(batch, next_tokens)

    return scheduler, kv_pool


def check_batch(model, config, n_reqs, batch_size, seed):
    reqs = make_requests(config, n_reqs, seed=seed)
    scheduler, kv_pool = run_scheduler(model, config, reqs, batch_size=batch_size)

    for req in scheduler.finished:
        output, _ = generate(
            model, req.input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=0.0,
            eos_token_id=req.eos_token_id,
        )
        oracle = output[0, req.input_len:].tolist()
        assert req.output_ids == oracle, \
            f"[seed={seed} bs={batch_size}] Request {req.request_id}: " \
            f"调度器 {req.output_ids} != oracle {oracle}"

    assert kv_pool.free_count == kv_pool.total_blocks, \
        f"[seed={seed} bs={batch_size}] 池未归还: " \
        f"free={kv_pool.free_count}/{kv_pool.total_blocks}"
    assert len(scheduler.finished) == n_reqs
    assert scheduler.has_pending() is False

    print(f"  [seed={seed} bs={batch_size}] {n_reqs} requests OK, "
          f"pool {kv_pool.free_count}/{kv_pool.total_blocks}")


def main():
    model, config = build_model()

    # 组合: 请求数 × batch_size × seed（覆盖 prefill 补位/满批/饥饿各场景）
    cases = [
        (7, 2, 42),
        (9, 4, 42),
        (9, 4, 7),
        (5, 1, 0),    # batch_size=1: 等价串行
        (12, 4, 99),  # 超 batch 场景: 12 请求 4 批位
    ]
    for n, bs, seed in cases:
        check_batch(model, config, n, bs, seed)

    print(f"\n[PASS] 全部 {len(cases)} 组场景: 批 decode 输出与 oracle 逐 token 一致")
    print("[PASS] 池闭环 + 状态机迁移合法")


if __name__ == "__main__":
    main()
