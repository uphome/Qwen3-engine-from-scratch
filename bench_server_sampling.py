"""
端到端服务端采样基准 — 对比旧/新采样实现的 HTTP 服务吞吐与延迟。

用法：
  # 先启动旧/新两个 server（不同端口），再分别运行本脚本
  python bench_server_sampling.py \
      --url http://127.0.0.1:8000/v1/chat/completions \
      --num-requests 64 --concurrency 8 --max-tokens 128 \
      --temperature 0.7 --top-k 50 --top-p 0.9

说明：
  - 每个请求都是独立的 HTTP chat completion
  - 服务端 Scheduler 按 batch-size 连续批处理
  - 指标：总完成 tokens / 墙钟 = 吞吐；每请求完成时刻 = 延迟
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(data, p):
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(data):
        return data[f] * (1 - c) + data[f + 1] * c
    return data[f]


def one_request(url, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return time.perf_counter() - t0, data


def main():
    parser = argparse.ArgumentParser(description="End-to-end sampling server benchmark")
    parser.add_argument("--url", type=str,
                        default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str,
                        default="请详细介绍一下人工智能的发展历史，至少写 200 字。")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "stream": False,
    }

    print("=" * 72)
    print("End-to-end sampling server benchmark")
    print(f"  url={args.url}")
    print(f"  requests={args.num_requests}, concurrency={args.concurrency}")
    print(f"  max_tokens={args.max_tokens}, T={args.temperature}, "
          f"top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 72)

    # 预热
    for _ in range(args.warmup):
        try:
            one_request(args.url, payload, args.timeout)
        except Exception as e:
            print(f"[warmup] 请求失败: {e}")
            return

    latencies = []
    total_completion = 0
    errors = 0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, args.url, payload, args.timeout)
                   for _ in range(args.num_requests)]
        for fut in as_completed(futures):
            try:
                dt, resp = fut.result()
                latencies.append(dt)
                usage = resp.get("usage") or {}
                total_completion += int(usage.get("completion_tokens", 0))
            except Exception as e:
                errors += 1
                print(f"[error] {e}")
    wall = time.perf_counter() - t0

    thr = total_completion / wall if wall > 0 else 0.0
    print(f"\nCompletion tokens: {total_completion}")
    print(f"Wall time:         {wall:.2f}s")
    print(f"Throughput:        {thr:.2f} tok/s")
    print(f"Errors:            {errors}")
    print(f"Latency p50:       {percentile(latencies, 50):.3f}s")
    print(f"Latency p95:       {percentile(latencies, 95):.3f}s")
    print(f"Latency p99:       {percentile(latencies, 99):.3f}s")
    if latencies:
        print(f"Latency mean:      {statistics.mean(latencies):.3f}s")
        print(f"Latency min/max:   {min(latencies):.3f}s / {max(latencies):.3f}s")


if __name__ == "__main__":
    main()
