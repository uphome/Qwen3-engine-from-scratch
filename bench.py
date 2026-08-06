"""
离线吞吐基准测试 — 手写 Qwen3 单序列串行推理

参考: bench.py (mini-sglang 的 continuous batching 基准)
适配: 手写 PyTorch Qwen3 框架 (qwen3/ + generate.py)
write by Claude code +deepseek4

用法:
    python bench.py --model /data/hjt1/Qwen3-0.6B
    python bench.py --model Qwen/Qwen3-0.6B --num-seqs 64 --max-input-len 2048
"""

import argparse
import statistics
import time
from random import randint, seed

import torch

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool
from qwen3 import load_weights_from_safetensors
from generate import generate


# ============================================================
# 辅助函数
# ============================================================

def resolve_device(device_str: str | None) -> torch.device:
    """选择设备: 指定设备 > 最空闲 GPU > CPU"""
    if device_str:
        return torch.device(device_str)

    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        best = free.index(max(free))
        return torch.device(f"cuda:{best}")

    return torch.device("cpu")


def make_random_input(vocab_size: int, seq_len: int) -> torch.Tensor:
    """生成随机 token ID，不依赖 tokenizer"""
    return torch.randint(0, min(vocab_size, 50000), (1, seq_len))


def percentile(data: list[float], p: float) -> float:
    """计算百分位数（无 numpy 依赖）"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    return sorted_data[f]


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """简单对齐表格"""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    lines = []
    # header
    lines.append("  " + "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    # rows
    for row in rows:
        lines.append("  " + "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: 手写 Qwen3 单序列串行吞吐测试"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径 (HF repo id 或本地目录)")
    parser.add_argument("--num-seqs", type=int, default=64,
                        help="总请求数 (default: 64)")
    parser.add_argument("--min-input-len", type=int, default=32,
                        help="输入长度下限 (default: 32)")
    parser.add_argument("--max-input-len", type=int, default=1024,
                        help="输入长度上限 (default: 1024)")
    parser.add_argument("--min-output-len", type=int, default=64,
                        help="输出长度下限 (default: 64)")
    parser.add_argument("--max-output-len", type=int, default=1024,
                        help="输出长度上限 (default: 1024)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="预热序列数 (default: 3)")
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备, e.g. 'cuda:1' (default: 自动选择最空闲的 GPU)")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子 (default: 0)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="温度, 0 = 贪心 (default: 0)")
    parser.add_argument("--profile", action="store_true",
                        help="启用 torch.profiler 导出 Chrome trace")
    parser.add_argument("--profile-output", type=str, default="trace.json",
                        help="Profiler trace 输出文件 (default: trace.json)")
    parser.add_argument("--profile-steps", type=int, default=10,
                        help="Profiler 采样的 decode step 数 (default: 10)")
    args = parser.parse_args()

    # 验证参数
    assert args.min_input_len <= args.max_input_len, \
        "min-input-len must be <= max-input-len"
    assert args.min_output_len <= args.max_output_len, \
        "min-output-len must be <= max-output-len"

    # ============================================================
    # 1. 设备 & 随机种子
    # ============================================================
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # 确保所有 CUDA API 调用操作正确的设备
    if device.type == "cuda":
        torch.cuda.set_device(device)

    seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("Benchmark: Qwen3 Serial Inference Throughput")
    print("=" * 72)
    print(f"Model:      {args.model}")
    print(f"Device:     {device}, dtype: {dtype}")
    print(f"Sequences:  {args.num_seqs}")
    print(f"Input len:  [{args.min_input_len}, {args.max_input_len}]")
    print(f"Output len: [{args.min_output_len}, {args.max_output_len}]")
    print(f"Warmup:     {args.warmup}")
    print(f"Seed:       {args.seed}")
    print(f"Sampling:   {'greedy' if args.temperature == 0 else f'T={args.temperature}'}")

    # ============================================================
    # 2. VRAM baseline
    # ============================================================
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        vram_baseline = torch.cuda.memory_allocated() / 1e9  # GB

    # ============================================================
    # 3. 加载模型
    # ============================================================
    print(f"\n[1/3] Loading model & weights...")
    config = Qwen3Config.from_pretrained(args.model)
    model = Qwen3ForCausalLM(config)
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count / 1e9:.2f}B")
    print(f"  Layers: {config.num_hidden_layers}, "
          f"Hidden: {config.hidden_size}, "
          f"Q heads: {config.num_attention_heads}, "
          f"KV heads: {config.num_key_value_heads}")

    if device.type == "cuda":
        vram_model = torch.cuda.memory_allocated() / 1e9
        print(f"  Model VRAM: {vram_model - vram_baseline:.2f} GB "
              f"(total: {vram_model:.2f} GB)")

    # --- KV Cache 共享池 ---
    block_size = 16
    max_seq_tokens = args.max_input_len + args.max_output_len
    num_blocks = (max_seq_tokens + block_size - 1) // block_size
    num_blocks = max(num_blocks, 128)
    kv_pool = KVCachePool(
        num_blocks=num_blocks,
        num_layers=config.num_hidden_layers,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_tokens,
    )
    pool_mem_mb = (kv_pool.k_buffer.numel() * kv_pool.k_buffer.element_size()
                   + kv_pool.v_buffer.numel() * kv_pool.v_buffer.element_size()) / 1024**2
    print(f"  KV pool: {num_blocks} blocks x {block_size} tokens, {pool_mem_mb:.1f} MB")

    # ============================================================
    # 4. 预热
    # ============================================================
    if args.warmup > 0:
        print(f"\n[2/3] Warmup ({args.warmup} sequences)...")

    for _ in range(args.warmup):
        # 用较短的序列预热，避免 OOM
        warm_input = make_random_input(config.vocab_size, min(64, args.max_input_len))
        warm_input = warm_input.to(device)
        warm_output_len = min(32, args.max_output_len)
        _, _ = generate(
            model, warm_input,
            max_new_tokens=warm_output_len,
            temperature=args.temperature,
            eos_token_id=-1,  # 不触发 EOS
            kv_cache_pool=kv_pool,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        vram_after_warmup = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after warmup: {vram_after_warmup:.2f} GB")

    # ============================================================
    # 5. 生成基准数据
    # ============================================================
    print(f"\n[3/3] Benchmark: {args.num_seqs} sequences...")

    # 预生成所有序列的配置，确保可复现
    seq_configs = []
    for _ in range(args.num_seqs):
        input_len = randint(args.min_input_len, args.max_input_len)
        output_len = randint(args.min_output_len, args.max_output_len)
        input_ids = make_random_input(config.vocab_size, input_len)
        seq_configs.append((input_ids, output_len))

    # ============================================================
    # 6. 基准测试循环
    # ============================================================
    results = []  # list of dict

    for seq_idx, (input_ids, target_output_len) in enumerate(seq_configs):
        input_ids = input_ids.to(device)
        input_len = input_ids.shape[1]

        # 端到端墙钟计时: perf_counter + synchronize（和 generate 内部 step_times 同口径）
        t_seq_start = time.perf_counter()

        output_ids, stats = generate(
            model, input_ids,
            max_new_tokens=target_output_len,
            temperature=args.temperature,
            eos_token_id=-1,  # 忽略 EOS，保证生成长度精确可控
            kv_cache_pool=kv_pool,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_gpu_ms = (time.perf_counter() - t_seq_start) * 1000

        # 提取指标
        n_generated = output_ids.shape[1] - input_len
        step_times = stats["step_times"]

        # Prefill = step 0
        prefill_ms = step_times[0] * 1000 if step_times else 0.0

        # Decode = steps 1..N
        decode_times = step_times[1:] if len(step_times) > 1 else []
        avg_decode_ms = (sum(decode_times) / len(decode_times) * 1000) if decode_times else 0.0

        results.append({
            "seq_idx": seq_idx,
            "input_len": input_len,
            "output_len": n_generated,
            "target_output_len": target_output_len,
            "total_gpu_ms": total_gpu_ms,
            "prefill_ms": prefill_ms,
            "avg_decode_ms_per_token": avg_decode_ms,
        })

        # 进度条
        if (seq_idx + 1) % max(1, args.num_seqs // 10) == 0:
            print(f"  [{seq_idx + 1}/{args.num_seqs}] ...")

    # ============================================================
    # 7. VRAM 统计
    # ============================================================
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        final_vram = torch.cuda.memory_allocated() / 1e9
        vram_overhead = peak_vram - vram_after_warmup

    # ============================================================
    # 8. 汇总 & 报告
    # ============================================================
    total_gpu_time_s = sum(r["total_gpu_ms"] for r in results) / 1000
    total_output_tokens = sum(r["output_len"] for r in results)
    total_input_tokens = sum(r["input_len"] for r in results)
    throughput = total_output_tokens / total_gpu_time_s if total_gpu_time_s > 0 else 0.0

    # 提取各维度数据
    input_lens = [r["input_len"] for r in results]
    output_lens = [r["output_len"] for r in results]
    prefill_ms_list = [r["prefill_ms"] for r in results]
    decode_ms_list = [r["avg_decode_ms_per_token"] for r in results]
    total_ms_list = [r["total_gpu_ms"] for r in results]

    print(f"\n{'=' * 72}")
    print("Benchmark Results")
    print("=" * 72)

    print(f"\n--- Summary ---")
    print(f"  Total GPU time:      {total_gpu_time_s:>8.2f} s")
    print(f"  Total input tokens:  {total_input_tokens:>8}")
    print(f"  Total output tokens: {total_output_tokens:>8}")
    print(f"  Throughput:          {throughput:>8.2f} tok/s")

    if device.type == "cuda":
        print(f"\n--- VRAM ---")
        print(f"  Before model:        {vram_baseline:>8.2f} GB")
        print(f"  After model:         {vram_model:>8.2f} GB")
        print(f"  Peak (inference):    {peak_vram:>8.2f} GB  (+{vram_overhead:.2f} GB)")
        print(f"  After benchmark:     {final_vram:>8.2f} GB")

    print(f"\n--- Per-Sequence Stats (N={args.num_seqs}) ---")
    stat_metrics = {
        "Input len":     input_lens,
        "Output len":    output_lens,
        "Prefill (ms)":  prefill_ms_list,
        "Decode (ms/tok)": decode_ms_list,
        "Total (ms)":    total_ms_list,
    }

    headers = ["", "mean", "min", "max", "p50", "p95", "p99"]
    rows = []
    for name, data in stat_metrics.items():
        if not data:
            continue
        is_int = name in ("Input len", "Output len")
        fmt = lambda v: f"{int(v)}" if is_int else f"{v:.1f}"
        row = [
            name,
            fmt(statistics.mean(data)),
            fmt(min(data)),
            fmt(max(data)),
            fmt(percentile(data, 50)),
            fmt(percentile(data, 95)),
            fmt(percentile(data, 99)),
        ]
        rows.append(row)

    print(format_table(headers, rows))

    # Prefill vs Decode 时间占比
    total_prefill_ms = sum(prefill_ms_list)
    total_decode_ms = sum(r["total_gpu_ms"] - r["prefill_ms"] for r in results)
    total_ms_all = total_prefill_ms + total_decode_ms
    if total_ms_all > 0:
        print(f"\n--- Time Breakdown ---")
        print(f"  Prefill: {total_prefill_ms / 1000:.2f}s ({total_prefill_ms / total_ms_all * 100:.1f}%)")
        print(f"  Decode:  {total_decode_ms / 1000:.2f}s ({total_decode_ms / total_ms_all * 100:.1f}%)")

    # ============================================================
    # 9. torch.profiler — Chrome trace 导出 + 算子级耗时分析
    # ============================================================
    if args.profile and device.type == "cuda":
        print(f"\n{'=' * 72}")
        print("torch.profiler — CUDA Kernel 级分析")
        print("=" * 72)

        # 构造一条中等长度的序列用于 profiling
        prof_input_len = min(args.max_input_len, 256)
        prof_output_len = args.profile_steps
        prof_input = make_random_input(config.vocab_size, prof_input_len).to(device)

        print(f"  Profiling: input={prof_input_len} tokens, output={prof_output_len} steps")
        print(f"  Trace file: {args.profile_output}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            _, _ = generate(
                model, prof_input,
                max_new_tokens=prof_output_len,
                temperature=args.temperature,
                eos_token_id=-1,
                kv_cache_pool=kv_pool,
            )

        # --- 算子耗时排名 ---
        print(f"\n  Top 25 operators by CUDA time:")
        print(f"  {'Name':<60} {'CUDA (ms)':>10}  {'%':>6}  {'Calls':>8}")
        print(f"  {'-' * 60} {'-' * 10}  {'-' * 6}  {'-' * 8}")

        events = prof.key_averages()
        total_cuda = sum(e.cuda_time_total for e in events if e.cuda_time_total > 0)

        for i, e in enumerate(
            sorted(events, key=lambda e: e.cuda_time_total, reverse=True)
        ):
            if i >= 25:
                break
            if e.cuda_time_total == 0:
                break
            pct = e.cuda_time_total / total_cuda * 100 if total_cuda > 0 else 0
            name = e.key[:58] + ".." if len(e.key) > 60 else e.key
            print(f"  {name:<60} {e.cuda_time_total / 1000:>10.3f}  {pct:>5.1f}%  {e.count:>8}")

        # --- 按算子类型分组（table 输出） ---
        print(f"\n  Operator type breakdown:")
        print(prof.key_averages().table(row_limit=20))

        # 导出 Chrome trace
        prof.export_chrome_trace(args.profile_output)
        print(f"\n  Chrome trace saved to: {args.profile_output}")
        print(f"  在 Chrome 中打开 chrome://tracing，加载此文件即可查看火焰图")

    print(f"\n{'=' * 72}")
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
