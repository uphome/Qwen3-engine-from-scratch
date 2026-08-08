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

输出解读:
    总 CUDA 时间 = 一次 decode forward 里所有 kernel 在 GPU 上的执行时间之和
    墙钟（用户感知）≈ 总 CUDA 时间 + GPU 空等 CPU 的时间
      - 若墙钟 >> 总 CUDA 时间：瓶颈在 CPU 提交（每层 kernel 启动间隙）→ CUDA Graph
      - 若两者接近：瓶颈在 GPU 计算本身
"""

import argparse
import os
import sys

import torch

# 把项目根目录加进 sys.path，保证从任意目录运行都能 import qwen3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3 import (Qwen3Config, Qwen3ForCausalLM, KVCachePool,
                   load_weights_from_safetensors)
from qwen3.request import Request
from qwen3.scheduler import Scheduler


def resolve_device(device_str: str | None) -> torch.device:
    """选择设备: 指定设备 > 最空闲 GPU > CPU"""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        # 选剩余显存最多的 GPU（避免和其他任务抢占同一张卡）
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return torch.device(f"cuda:{free.index(max(free))}")
    return torch.device("cpu")


def report(prof, args, device, label):
    """打印 profiler 结果的 top-N kernel 表"""
    events = prof.key_averages()
    attr = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    total = sum(getattr(e, attr) for e in events if getattr(e, attr) > 0)
    print(f"[{label}] 总 {'CUDA' if device.type == 'cuda' else 'CPU'} 时间: "
          f"{total/1000:.2f} ms, 事件数: {len(events)}")
    print(f"{'name':<55}{'time(ms)':>10}{'%':>7}{'calls':>7}")
    print(f"{'-'*55}{'-'*10}{'-'*7}{'-'*7}")
    for e in sorted(events, key=lambda e: getattr(e, attr), reverse=True)[:args.top_k]:
        t = getattr(e, attr)
        if t == 0:
            break
        name = e.key[:53] + ".." if len(e.key) > 55 else e.key   # 长 kernel 名截断
        print(f"{name:<55}{t/1000:>10.2f}{t/total*100:>6.1f}%{e.count:>7}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Profile step kernel time")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-req", type=int, default=16, help="请求数")
    parser.add_argument("--batch", type=int, default=8, help="batch_size")
    parser.add_argument("--prompt-len", type=int, default=96, help="prompt 长度")
    parser.add_argument("--max-new", type=int, default=32, help="每请求生成上限")
    parser.add_argument("--top-k", type=int, default=15, help="显示 top N 个 kernel")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["decode", "prefill"], default="decode",
                        help="剖析哪个阶段：decode（默认）| prefill")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 72)
    print(f"Profile {args.mode} step | device={device} dtype={dtype}")
    print(f"  requests={args.num_req}, batch={args.batch}, "
          f"prompt={args.prompt_len}, max_new={args.max_new}")
    print(f"  QWEN3_FLASH_ATTN={os.environ.get('QWEN3_FLASH_ATTN', 'triton')}"
          f"（prefill 注意力：triton=融合 kernel / pytorch=标准实现）")
    print("=" * 72)

    # ---- 加载模型 ----
    config = Qwen3Config.from_pretrained(args.model)
    model = Qwen3ForCausalLM(config)
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    # CPU 上 profiler 只能记录 CPU 活动（没有 CUDA kernel 可测）
    activities = [torch.profiler.ProfilerActivity.CUDA]
    if device.type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU]

    def build_sched(pool):
        sched = Scheduler(pool, batch_size=args.batch)
        for i in range(args.num_req):
            sched.add(Request(i, torch.randint(0, 50000, (1, args.prompt_len), device=device),
                              max_new_tokens=args.max_new, eos_token_id=-1))
        return sched

    def step_once(sched):
        """跑一个调度步（prefill 或 decode），返回 (mode, batch)

        与 bench_batched.py 的驱动循环一致：调度 → 前向 → 采样 → 收尾。
        """
        b = sched.schedule()
        logits = model(b.build_input_ids(), kv_cache=b.build_kv_caches(),
                       input_lens=b.input_lens)
        li = torch.tensor(b.input_lens, dtype=torch.long, device=logits.device) - 1
        sched.on_step_done(b, logits[torch.arange(li.shape[0]), li].argmax(-1, keepdim=True))
        return b.mode, b

    # ---- 预热 ----
    # 触发 Triton kernel 的 JIT 编译（编译时间不能计入 profile）
    # 池子 1024 块足够 16 请求 × batch 8 使用
    warm_pool = KVCachePool(1024, config.num_hidden_layers, 16,
                            config.num_key_value_heads, config.head_dim,
                            device=device, dtype=dtype)
    warm_sched = build_sched(warm_pool)
    for _ in range(3):
        step_once(warm_sched)

    # ---- 被剖析的代码 ----
    if args.mode == "decode":
        # decode 模式：预热后继续跑到第一个 decode 批（prefill 的 K/V 已入 cache），
        # 然后 profile 一次 decode forward（S=1，推理主要耗时来源）
        while True:
            mode, b = step_once(warm_sched)
            if mode == "decode":
                break
        input_ids = b.build_input_ids()
        caches = b.build_kv_caches()
        print(f"[profile] 一次 decode forward: input_ids={tuple(input_ids.shape)} "
              f"kv_caches={len(caches)} 个请求\n")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            model(input_ids, kv_cache=caches, input_lens=b.input_lens)
            if device.type == "cuda":
                torch.cuda.synchronize()
        total = report(prof, args, device, "decode")
    else:
        # prefill 模式：fresh scheduler，第一次调度必为 prefill 批。
        # 预热已触发 JIT，这里 profile 的是干净的 prefill forward（右 pad 批）。
        pool = KVCachePool(1024, config.num_hidden_layers, 16,
                           config.num_key_value_heads, config.head_dim,
                           device=device, dtype=dtype)
        sched = build_sched(pool)
        b = sched.schedule()
        input_ids = b.build_input_ids()
        caches = b.build_kv_caches()
        print(f"[profile] 一次 prefill forward: input_ids={tuple(input_ids.shape)} "
              f"batch={b.size}, lens={b.input_lens}\n")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            model(input_ids, kv_cache=caches, input_lens=b.input_lens)
            if device.type == "cuda":
                torch.cuda.synchronize()
        total = report(prof, args, device, "prefill")

    print(f"\n提示: 墙钟 ≈ kernel 时间 + GPU 空等 CPU 的时间。"
          f"若墙钟 >> 本脚本的 kernel 总时间，瓶颈在 CPU 提交（kernel 启动间隙），"
          f"考虑 CUDA Graph；若接近，瓶颈在 GPU 计算。")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
