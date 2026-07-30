"""
Lesson 2: 从零手写 Qwen3 模型 — 端到端推理（交互式）

用法:
    python main.py --model /data/hjt1/Qwen3-0.6B
    python main.py --model Qwen/Qwen3-8B --temperature 0

启动后进入交互循环，可反复输入 prompt 生成。输入 /quit 退出。
"""

import os
import time
import argparse
import traceback

import torch

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool
from qwen3 import load_weights_from_safetensors
from generate import generate
from chat_template import format_chat


def main():
    parser = argparse.ArgumentParser(description="Qwen3 交互式推理")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace 模型名或本地路径")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="单次生成的最大 token 数 (默认: 512)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (默认: 自动选择)")
    parser.add_argument("--pool-blocks", type=int, default=256,
                        help="KV cache 池的物理块数 (默认: 256, 每块 16 tokens = 支持 4096 token 序列)")
    args = parser.parse_args()

    # --- 设备 ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        device = torch.device(f"cuda:{free_mem.index(max(free_mem))}")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 60)
    print("Qwen3 交互式推理 — Paged KV Cache")
    print("=" * 60)
    print(f"Device: {device}, dtype: {dtype}")

    # --- 配置 ---
    config = Qwen3Config.from_pretrained(args.model)
    print(f"Model: {args.model}")
    print(f"  Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}, "
          f"Heads: {config.num_attention_heads}, KV Heads: {config.num_key_value_heads}")

    # --- 模型 ---
    model = Qwen3ForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,} ({param_count / 1e9:.2f}B)")
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    # --- Tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_token_id = 151645

    # --- KV Cache 共享池（全局创建一次）---
    block_size = 16
    num_blocks = max(args.pool_blocks, (args.max_tokens + block_size - 1) // block_size)
    kv_pool = KVCachePool(
        num_blocks=num_blocks,
        num_layers=config.num_hidden_layers,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=device,
        dtype=dtype,
    )
    pool_mem_mb = (kv_pool.k_buffer.numel() * kv_pool.k_buffer.element_size()
                   + kv_pool.v_buffer.numel() * kv_pool.v_buffer.element_size()) / 1024 ** 2
    max_seq = num_blocks * block_size
    print(f"  KV pool: {num_blocks} blocks × {block_size} tokens = {max_seq} tokens max, {pool_mem_mb:.1f} MB")
    print(f"  Pool usage: {num_blocks - kv_pool.free_count} / {num_blocks}")

    # --- 交互循环 ---
    print(f"\n{'=' * 60}")
    print("输入 prompt 开始生成，输入 /quit 退出")
    print(f"Sampling: T={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit", "/q"):
            print("退出")
            break

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond directly without thinking."},
            {"role": "user", "content": prompt},
        ]
        text = format_chat(messages)
        input_ids = torch.tensor([tokenizer.encode(text)], device=device)

        input_len = input_ids.shape[1]
        remaining = kv_pool.free_count
        needed = (input_len + args.max_tokens + block_size - 1) // block_size
        if needed > remaining:
            print(f"[WARN] 池剩余 {remaining} 块，需要 {needed} 块，可能 OOM")
            continue

        t0 = time.perf_counter()
        try:
            output_ids, stats = generate(
                model, input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=eos_token_id,
                kv_cache_pool=kv_pool,
            )
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            continue

        t_total = time.perf_counter() - t0

        new_tokens = output_ids[0][input_len:]
        output_text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

        print(f"\n{output_text}")

        n = len(new_tokens)
        step_times = stats["step_times"]
        blocks_used = stats.get("blocks_used", 0)
        prefill_ms = step_times[0] * 1000
        avg_ms = sum(step_times[1:]) / len(step_times[1:]) * 1000 if n > 0 and len(step_times) > 1 else 0

        print(f"\n  {n} tokens | {t_total:.1f}s | prefill {prefill_ms:.0f}ms | decode {avg_ms:.0f}ms/tok | pool {blocks_used}/{num_blocks}")


if __name__ == "__main__":
    main()
