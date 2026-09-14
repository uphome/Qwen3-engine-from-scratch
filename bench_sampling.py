"""
采样微基准：旧逐请求 Python 采样 vs 新批量 GPU/Triton 采样。

不加载模型，只测采样算子本身的每步耗时与 kernel 启动开销。

用法：
    CUDA_VISIBLE_DEVICES=2 python bench_sampling.py \
        --batch-sizes 1,8,28 --vocab 151936 --steps 200
"""

import argparse
import time

import torch
import torch.nn.functional as F

from qwen3.sampling import sample_tokens


def old_sample_one(logits: torch.Tensor, temperature: float,
                   top_k: int, top_p: float) -> torch.Tensor:
    """旧版单请求采样（与 server.py / generate.py 原实现一致）。"""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        topk_vals = torch.topk(logits, min(top_k, logits.size(-1))).values
        logits = logits.masked_fill(logits < topk_vals[..., -1:], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        remove = mask.scatter(-1, sorted_idx, mask)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def old_sample_batch(logits: torch.Tensor, temperature: float,
                     top_k: int, top_p: float) -> torch.Tensor:
    B = logits.shape[0]
    out = torch.zeros(B, 1, dtype=torch.long, device=logits.device)
    for i in range(B):
        out[i, 0] = old_sample_one(logits[i], temperature, top_k, top_p)
    return out


def timeit(fn, steps: int):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000


def main():
    parser = argparse.ArgumentParser(description="Sampling microbenchmark")
    parser.add_argument("--batch-sizes", type=str, default="1,8,28")
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 72)
    print("Sampling microbenchmark")
    print(f"  device={device}, vocab={args.vocab}, steps={args.steps}, "
          f"T={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 72)

    for B in [int(x) for x in args.batch_sizes.split(",") if x]:
        torch.manual_seed(0)
        logits = torch.randn(B, args.vocab, device=device, dtype=dtype) * 2.0

        # 预热
        for _ in range(args.warmup):
            old_sample_batch(logits, args.temperature, args.top_k, args.top_p)
            sample_tokens(logits, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p)
        if device.type == "cuda":
            torch.cuda.synchronize()

        old_ms = timeit(
            lambda: old_sample_batch(logits, args.temperature,
                                     args.top_k, args.top_p),
            args.steps)
        new_ms = timeit(
            lambda: sample_tokens(logits, temperature=args.temperature,
                                  top_k=args.top_k, top_p=args.top_p),
            args.steps)

        print(f"batch={B:3d} | old={old_ms:8.3f} ms/step | "
              f"new={new_ms:8.3f} ms/step | speedup={old_ms/new_ms:6.2f}x")


if __name__ == "__main__":
    main()
