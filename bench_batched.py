"""
连续批处理 vs 串行推理基准 — continuous batching 吞吐对比

同一组请求，两种跑法:
  - serial:   逐个请求 generate()（无并发，基线）
  - batched:  调度器驱动，batch_size 从 1 扫到 --max-batch

指标口径（全部同一起点 = 请求 0 时刻全部到达，离线到达模式）:
  - Throughput      = 总输出 token / 总墙钟时间（continuous batching 的核心指标）
  - Decode ms/token = decode 步总耗时 / 总输出 token（每 token 的 decode 成本）
  - Latency p50/p95 = 每请求完成时刻的分布（请求间并发的直接体现）

用法:
    python bench_batched.py --model /data/hjt1/Qwen3-0.6B
    python bench_batched.py --model Qwen/Qwen3-0.6B --num-seqs 32 --max-batch 8
"""

import argparse
import os
import statistics
import time
from random import randint, seed

import torch

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool, Scheduler
from qwen3 import load_weights_from_safetensors
from qwen3.PagedKVcache import PagedKVCache
from qwen3.graph_runner import GraphRunner
from qwen3.request import Request
from generate import generate


def resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return torch.device(f"cuda:{free.index(max(free))}")
    return torch.device("cpu")


def percentile(data: list[float], p: float) -> float:
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
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    lines = ["  " + "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))]
    for row in rows:
        lines.append("  " + "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


# ============================================================
# 两种运行模式
# ============================================================

# 请求规格: (input_ids, max_new_tokens)。每个 run 内部新建 Request，
# 避免跨 run 复用同一批 Request 对象（num_generated/state 会污染后续 run）。

@torch.no_grad()
def run_serial(model, kv_pool, specs, temperature=0.0):
    """逐个请求 generate()——串行基线（请求共享池，公平对比）"""
    t0 = time.perf_counter()
    prefill_ms = 0.0
    decode_ms = 0.0
    latencies: list[float] = []
    n_tokens = 0

    for input_ids, max_new_tokens in specs:
        output_ids, stats = generate(
            model, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=-1,          # 不触发 EOS，生成长度精确可控
            kv_cache_pool=kv_pool,
        )
        if input_ids.is_cuda:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)  # 完成时刻（与批处理同口径）
        n_tokens += output_ids.shape[1] - input_ids.shape[1]
        if stats["step_times"]:
            prefill_ms += stats["step_times"][0] * 1000
            decode_ms += sum(stats["step_times"][1:]) * 1000

    total_s = time.perf_counter() - t0
    return {
        "mode": "serial", "batch_size": 1,
        "total_s": total_s, "n_tokens": n_tokens,
        "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "latencies": latencies,
    }


@torch.no_grad()
def run_batched(model, kv_pool, specs, batch_size, temperature=0.0,
                runner=None):
    """调度器驱动——continuous batching（与 batched_generate.py 同循环）

    runner: GraphRunner | None。非 None 时 decode 步走 CUDA Graph
            （replay），prefill 步保持 eager（model 前向）。
    """
    scheduler = Scheduler(kv_pool, batch_size=batch_size)
    for i, (input_ids, max_new_tokens) in enumerate(specs):
        scheduler.add(Request(
            request_id=i, input_ids=input_ids,
            max_new_tokens=max_new_tokens, eos_token_id=-1,
        ))

    t0 = time.perf_counter()
    prefill_ms = 0.0
    decode_ms = 0.0
    n_prefill_steps = 0
    n_decode_steps = 0
    finish_times: list[float] = []

    while scheduler.has_pending():
        batch = scheduler.schedule()
        t_step = time.perf_counter()

        if batch.mode == "decode" and runner is not None:
            # ---- decode 走 CUDA Graph：replay 替代 model 前向 ----
            # 与 model.forward 的 decode 分支等价，但 reserve/advance
            # 状态操作从模型内移到驱动循环（forward_decode 的契约）。
            caches = batch.build_kv_caches()
            for c in caches:
                c.reserve_next()          # 页预留（原 forward 内部做）
            start_pos = torch.tensor([c.seq_len for c in caches],
                                     dtype=torch.long,
                                     device=model.model.embed_tokens.weight.device)
            positions = start_pos.unsqueeze(1)                       # (k, 1)
            rows = torch.tensor([c.row_id for c in caches],
                                dtype=torch.int32, device=positions.device)
            logits = runner.replay(
                batch.build_input_ids(), positions, rows)
            logits = logits[:batch.size]          # (b,1,vocab) → 前 k 行有效
            for c in caches:
                c.advance_seq_len(1)      # 状态推进（原 forward 内部做）
            last_idx = torch.zeros(batch.size, dtype=torch.long,
                                   device=logits.device)
        else:
            input_ids = batch.build_input_ids()
            logits = model(input_ids, kv_cache=batch.build_kv_caches(),
                           input_lens=batch.input_lens)
            last_idx = torch.tensor(batch.input_lens, dtype=torch.long,
                                    device=logits.device) - 1
        next_tokens = logits[torch.arange(last_idx.shape[0]), last_idx] \
            .argmax(dim=-1, keepdim=True)      # 贪心，与串行同口径

        if logits.is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_step

        if batch.mode == "prefill":
            prefill_ms += dt * 1000
            n_prefill_steps += 1
        else:
            decode_ms += dt * 1000
            n_decode_steps += 1

        scheduler.on_step_done(batch, next_tokens)
        for req in batch.requests:
            if req.state == "finished":
                finish_times.append(time.perf_counter() - t0)

    total_s = time.perf_counter() - t0
    n_tokens = sum(r.num_generated for r in scheduler.finished)
    return {
        "mode": "batched", "batch_size": batch_size,
        "total_s": total_s, "n_tokens": n_tokens,
        "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "n_prefill_steps": n_prefill_steps, "n_decode_steps": n_decode_steps,
        "latencies": finish_times,
    }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: continuous batching vs serial"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--min-input-len", type=int, default=32)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--min-output-len", type=int, default=32)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-batch", type=int, default=8,
                        help="batch_size 扫描上限: 1,2,...,max-batch (default: 8)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="只跑指定 batch_size（跳过扫描）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", action="store_true",
                        help="跑正式基准前预热（推荐 GPU 上开启）")
    parser.add_argument("--no-graph", action="store_true",
                        help="禁用 CUDA Graph decode（默认开，QWEN3_CUDA_GRAPH=0 亦可）")
    args = parser.parse_args()

    assert args.min_input_len <= args.max_input_len
    assert args.min_output_len <= args.max_output_len

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("Benchmark: Continuous Batching vs Serial")
    print("=" * 72)
    print(f"Model:      {args.model}")
    print(f"Device:     {device}, dtype: {dtype}")
    print(f"Sequences:  {args.num_seqs}")
    print(f"Input len:  [{args.min_input_len}, {args.max_input_len}]")
    print(f"Output len: [{args.min_output_len}, {args.max_output_len}]")
    print(f"Seed:       {args.seed}")

    # ---- 加载模型 ----
    config = Qwen3Config.from_pretrained(args.model)
    model = Qwen3ForCausalLM(config)
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()
    print(f"\n[load] Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B, "
          f"Layers: {config.num_hidden_layers}")

    # ---- 请求规格（固定 seed 可复现；每个 run 内部新建 Request）----
    specs = []
    for _ in range(args.num_seqs):
        input_len = randint(args.min_input_len, args.max_input_len)
        output_len = randint(args.min_output_len, args.max_output_len)
        specs.append((torch.randint(0, config.vocab_size, (1, input_len),
                                    device=device), output_len))
    total_output = sum(s[1] for s in specs)
    print(f"[data] {args.num_seqs} requests, "
          f"expected output tokens: {total_output}")

    # ---- CUDA Graph（decode 图池）----
    # 开关：--no-graph 或 QWEN3_CUDA_GRAPH=0 关闭（默认开）。
    # 构造时捕获 buckets 张图（占位句柄常驻 pool，预留少量块 + 哑行 1 块）。
    batch_sizes = ([args.batch_size] if args.batch_size is not None
                   else list(range(1, args.max_batch + 1)))
    use_graph = (
        device.type == "cuda" and not args.no_graph
        and os.environ.get("QWEN3_CUDA_GRAPH", "1") != "0"
    )
    # buckets 用 2 的幂（vLLM 同款）：1,2,4,8,... ≥ max_batch 的最小幂。
    # decode 步任意请求数 k 落进 >=k 的最小桶，空槽哑行占位；
    # 避免逐 batch 捕获（max_batch=28 只需 6 张图而非 28 张）。
    graph_buckets = [1]
    if use_graph:
        while graph_buckets[-1] < max(batch_sizes):
            graph_buckets.append(graph_buckets[-1] * 2)

    # ---- KV 池（块数按最大并发 batch 备足；graph 占位句柄 + 哑行各 1 块）----
    block_size = 16
    max_seq_tokens = args.max_input_len + args.max_output_len
    max_batch = args.max_batch if args.batch_size is None else args.batch_size
    blocks_per_seq = (max_seq_tokens + block_size - 1) // block_size
    num_blocks = max(128, blocks_per_seq * min(max_batch, args.num_seqs)
                     + (max(graph_buckets) + 1 if use_graph else 0))
    kv_pool = KVCachePool(
        num_blocks=num_blocks, num_layers=config.num_hidden_layers,
        block_size=block_size, num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim, device=device, dtype=dtype,
        max_seq_len=max_seq_tokens,
    )
    print(f"[pool] {num_blocks} blocks x {block_size} "
          f"(>= {blocks_per_seq} blocks/seq x {max_batch} batch)")

    runner = None
    if use_graph:
        occupy = [PagedKVCache(kv_pool) for _ in range(max(graph_buckets))]
        runner = GraphRunner(model, kv_pool, occupy,
                             buckets=tuple(graph_buckets), reserve_pages=1)
        # 占位句柄 + 哑行各占 1 块，池归还检查要扣除
        n_reserved = max(graph_buckets) + 1
    else:
        n_reserved = 0

    # ---- 预热 ----
    if args.warmup:
        print("[warmup] ...")
        warm_specs = [(torch.randint(0, config.vocab_size,
                                     (1, min(64, args.max_input_len)),
                                     device=device), 16)]
        run_serial(model, kv_pool, warm_specs)
        run_batched(model, kv_pool, warm_specs, batch_size=2, runner=runner)

    # ---- 基准 ----
    results = [run_serial(model, kv_pool, specs)]
    for bs in batch_sizes:
        print(f"[run] batch_size={bs} ...")
        results.append(run_batched(model, kv_pool, specs, bs, runner=runner))
        assert kv_pool.free_count == kv_pool.total_blocks - n_reserved, \
            f"batch={bs} 池未归还: {kv_pool.free_count}/{kv_pool.total_blocks}"

    # ---- 报告 ----
    serial = results[0]
    print(f"\n{'=' * 72}")
    print("Results")
    print("=" * 72)

    headers = ["Mode", "Batch", "Steps", "Prefill(s)", "Decode(s)",
               "Total(s)", "Throughput", "Decode", "Lat p50", "Lat p95"]
    rows = []
    for r in results:
        tok_s = r["n_tokens"] / r["total_s"] if r["total_s"] else 0.0
        decode_per_tok = (r["decode_ms"] / r["n_tokens"]) if r["n_tokens"] else 0.0
        rows.append([
            r["mode"],
            str(r["batch_size"]),
            (f'{r["n_prefill_steps"]}+{r["n_decode_steps"]}'
             if "n_prefill_steps" in r else "-"),
            f'{r["prefill_ms"] / 1000:.2f}',
            f'{r["decode_ms"] / 1000:.2f}',
            f'{r["total_s"]:.2f}',
            f"{tok_s:.1f} tok/s",
            f"{decode_per_tok:.1f} ms/tok",
            f'{percentile(r["latencies"], 50):.2f}s',
            f'{percentile(r["latencies"], 95):.2f}s',
        ])
    print(format_table(headers, rows))

    # 加速比
    print(f"\n--- Speedup vs serial ---")
    serial_tok_s = serial["n_tokens"] / serial["total_s"]
    serial_dpt = serial["decode_ms"] / serial["n_tokens"]
    for r in results[1:]:
        tok_s = r["n_tokens"] / r["total_s"]
        dpt = r["decode_ms"] / r["n_tokens"]
        print(f"  batch={r['batch_size']:<2} "
              f"throughput {tok_s / serial_tok_s:>5.2f}x "
              f"| decode {serial_dpt / dpt:>5.2f}x per-token "
              f"({serial_dpt:.1f} -> {dpt:.1f} ms/tok)")

    print(f"\n{'=' * 72}")
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
