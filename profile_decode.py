"""
torch.profiler：decode 步 kernel 时间分布分析

用法（GPU，cuda_learn 环境）:
    python profile_decode.py --model /data/hjt1/Qwen3-0.6B
    python profile_decode.py --model /data/hjt1/Qwen3-0.6B --num-req 32 --batch 8 --top-k 25

原理:
    torch.profiler 在每次 CUDA kernel 启动时插入 Event，key_averages() 按
    kernel 名聚合，cuda_time_total 是 kernel 在 GPU 上的实际执行时间
    （不含 CPU 提交和排队等待）。

用途:
    - 回答"哪个算子真正占 GPU 时间"（避免瞎猜瓶颈）
    - 优化前后对比：CUDA Graph / 块表缓存 / 跳过 mask 等改动
      跑一次本脚本，看总 CUDA 时间与 top kernel 分布是否变化
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3 import (Qwen3Config, Qwen3ForCausalLM, KVCachePool,
                   load_weights_from_safetensors)
from qwen3.request import Request
from qwen3.scheduler import Scheduler


def resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return torch.device(f"cuda:{free.index(max(free))}")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Profile decode step kernel time")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-req", type=int, default=16, help="请求数")
    parser.add_argument("--batch", type=int, default=8, help="batch_size")
    parser.add_argument("--prompt-len", type=int, default=96, help="prompt 长度")
    parser.add_argument("--max-new", type=int, default=32, help="每请求生成上限")
    parser.add_argument("--top-k", type=int, default=15, help="显示 top N 个 kernel")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 72)
    print(f"Profile decode step | device={device} dtype={dtype}")
    print(f"  requests={args.num_req}, batch={args.batch}, "
          f"prompt={args.prompt_len}, max_new={args.max_new}")
    print("=" * 72)

    config = Qwen3Config.from_pretrained(args.model)
    model = Qwen3ForCausalLM(config)
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    pool = KVCachePool(1024, config.num_hidden_layers, 16,
                       config.num_key_value_heads, config.head_dim,
                       device=device, dtype=dtype)
    sched = Scheduler(pool, batch_size=args.batch)
    for i in range(args.num_req):
        sched.add(Request(i, torch.randint(0, 50000, (1, args.prompt_len), device=device),
                          max_new_tokens=args.max_new, eos_token_id=-1))

    def step_once():
        """跑一个调度步，返回该步是否 decode"""
        b = sched.schedule()
        logits = model(b.build_input_ids(), kv_cache=b.build_kv_caches(),
                       input_lens=b.input_lens)
        li = torch.tensor(b.input_lens, dtype=torch.long, device=logits.device) - 1
        sched.on_step_done(b, logits[torch.arange(li.shape[0]), li].argmax(-1, keepdim=True))
        return b.mode, b

    # 预热（Triton JIT 编译 + 跑到 decode 阶段）
    for _ in range(3):
        step_once()
    while True:
        mode, b = step_once()
        if mode == "decode":
            break

    # ---- 被剖析的代码：一次 decode forward ----
    input_ids = b.build_input_ids()
    caches = b.build_kv_caches()
    print(f"[profile] 一次 decode forward: input_ids={tuple(input_ids.shape)} "
          f"kv_caches={len(caches)} 个请求\n")

    activities = [torch.profiler.ProfilerActivity.CUDA]
    if device.type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
    ) as prof:
        logits = model(input_ids, kv_cache=caches, input_lens=b.input_lens)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- 按时间排 top N ----
    events = prof.key_averages()
    attr = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    total = sum(getattr(e, attr) for e in events if getattr(e, attr) > 0)
    print(f"总 {'CUDA' if device.type == 'cuda' else 'CPU'} 时间: "
          f"{total/1000:.2f} ms, 事件数: {len(events)}")
    print(f"{'name':<55}{'time(ms)':>10}{'%':>7}{'calls':>7}")
    print(f"{'-'*55}{'-'*10}{'-'*7}{'-'*7}")
    for e in sorted(events, key=lambda e: getattr(e, attr), reverse=True)[:args.top_k]:
        t = getattr(e, attr)
        if t == 0:
            break
        name = e.key[:53] + ".." if len(e.key) > 55 else e.key
        print(f"{name:<55}{t/1000:>10.2f}{t/total*100:>6.1f}%{e.count:>7}")

    print(f"\n提示: 墙钟 ≈ kernel 时间 + GPU 空等 CPU 的时间。"
          f"若墙钟 >> 本脚本的 kernel 总时间，瓶颈在 CPU 提交（kernel 启动间隙），"
          f"考虑 CUDA Graph；若接近，瓶颈在 GPU 计算。")


if __name__ == "__main__":
    main()
