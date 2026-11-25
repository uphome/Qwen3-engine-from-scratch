"""HTTP 服务压测 — 并发梯度 / QPS / TTFT / TPOT / 错误率 / 长稳

零依赖（urllib + concurrent.futures），直接压 OpenAI 兼容服务。

用法:
    # 先启动服务
    python server.py --model /path/to/Qwen3-0.6B --port 8000

    # 并发梯度扫描（流式，精确 TTFT）
    python bench_server.py --url http://127.0.0.1:8000 \
        --num-requests 64 --concurrency 1,2,4,8 --max-tokens 64 --stream

    # 非流式（端到端延迟）
    python bench_server.py --url http://127.0.0.1:8000 \
        --num-requests 32 --concurrency 4 --max-tokens 64

    # 长稳测试（30 分钟，固定并发 8，每 30s 打印窗口统计）
    python bench_server.py --url http://127.0.0.1:8000 \
        --duration 1800 --concurrency 8 --max-tokens 64 --stream \
        --json-out results/stability.json

    # 结果 JSON 供 scripts/make_bench_report.py 生成 Markdown
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] * (1 - c) + s[f + 1] * c
    return s[f]


def build_prompt(repeat: int) -> str:
    return "The quick brown fox jumps over the lazy dog. " * repeat


def one_request(url: str, prompt: str, max_tokens: int, temperature: float,
                stream: bool, timeout: float) -> dict:
    """单个请求：返回 {ttft, latency, tokens, error}"""
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=data,
        headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if stream:
                ttft = None
                text = ""
                usage = None
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    obj = json.loads(payload)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("text") or ""
                        if delta:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            text += delta
                latency = time.perf_counter() - t0
                tokens = (usage["completion_tokens"] if usage
                          else max(1, len(text)))
                return {"ttft": ttft if ttft is not None else latency,
                        "latency": latency, "tokens": tokens, "error": None}
            else:
                resp = json.loads(r.read())
                latency = time.perf_counter() - t0
                usage = resp.get("usage") or {}
                return {"ttft": latency, "latency": latency,
                        "tokens": usage.get("completion_tokens", 0),
                        "error": None}
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code}"
        try:
            msg += ": " + e.read().decode()[:200]
        except Exception:
            pass
        return {"ttft": None, "latency": time.perf_counter() - t0,
                "tokens": 0, "error": msg}
    except Exception as e:
        return {"ttft": None, "latency": time.perf_counter() - t0,
                "tokens": 0, "error": f"{type(e).__name__}: {e}"}


def summarize(results: list[dict], wall: float) -> dict:
    ok = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    n_tokens = sum(r["tokens"] for r in ok)
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    lats = [r["latency"] for r in ok]
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(errors),
        "error_rate": len(errors) / len(results) if results else 0.0,
        "wall_s": wall,
        "qps": len(ok) / wall if wall > 0 else 0.0,
        "tokens_per_s": n_tokens / wall if wall > 0 else 0.0,
        "n_tokens": n_tokens,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000,
        "ttft_p95_ms": percentile(ttfts, 95) * 1000,
        "lat_p50_ms": percentile(lats, 50) * 1000,
        "lat_p95_ms": percentile(lats, 95) * 1000,
        "lat_p99_ms": percentile(lats, 99) * 1000,
        "error_samples": [e["error"] for e in errors[:5]],
    }


def run_level(url, prompt, num_requests, concurrency, max_tokens,
              temperature, stream, timeout, progress=True) -> dict:
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request, url, prompt, max_tokens, temperature,
                          stream, timeout) for _ in range(num_requests)]
        for i, f in enumerate(as_completed(futs)):
            results.append(f.result())
            if progress and (i + 1) % max(1, num_requests // 10) == 0:
                print(f"    [{i + 1}/{num_requests}] ...", flush=True)
    wall = time.perf_counter() - t0
    out = summarize(results, wall)
    out.update({"concurrency": concurrency, "num_requests": num_requests,
                "stream": stream, "max_tokens": max_tokens})
    return out


def run_stability(url, prompt, concurrency, duration, max_tokens,
                  temperature, stream, timeout, window_s=30) -> dict:
    """长稳：固定并发持续发请求，每 window_s 打印窗口统计，结束返回总量"""
    deadline = time.perf_counter() + duration
    results = []
    windows = []
    lock = __import__("threading").Lock()

    def worker():
        while time.perf_counter() < deadline:
            r = one_request(url, prompt, max_tokens, temperature,
                            stream, timeout)
            with lock:
                results.append(r)      # 逐请求上报，窗口统计才能实时

    print(f"  [长稳] {duration}s, concurrency={concurrency}, "
          f"每 {window_s}s 打印窗口", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker) for _ in range(concurrency)]
        while any(not f.done() for f in futs):
            time.sleep(window_s)
            elapsed = time.perf_counter() - t0
            with lock:
                snap = list(results)
            ok = [r for r in snap if r["error"] is None]
            err = len(snap) - len(ok)
            n_tok = sum(r["tokens"] for r in ok)
            print(f"    t={elapsed:6.1f}s  req={len(snap):5d}  "
                  f"ok={len(ok):5d}  err={err:4d}  tok={n_tok}", flush=True)
            windows.append({"t": round(elapsed, 1), "requests": len(snap),
                            "ok": len(ok), "errors": err, "tokens": n_tok})
        for f in futs:
            f.result()
    wall = time.perf_counter() - t0

    out = summarize(results, wall)
    out.update({"concurrency": concurrency, "duration_s": duration,
                "stream": stream, "max_tokens": max_tokens,
                "windows": windows})
    return out


def main():
    parser = argparse.ArgumentParser(description="HTTP 服务压测（OpenAI 兼容）")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--concurrency", type=str, default="1,2,4,8",
                        help="并发档位，逗号分隔（长稳模式取第一个）")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-repeat", type=int, default=35,
                        help="prompt = 固定句子重复 N 次（≈N×10 tokens）")
    parser.add_argument("--stream", action="store_true",
                        help="流式请求（精确 TTFT）")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--duration", type=int, default=0,
                        help="长稳测试时长（秒）；>0 时忽略 num-requests")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    prompt = build_prompt(args.prompt_repeat)

    print("=" * 72)
    print("HTTP Server Benchmark")
    print("=" * 72)
    print(f"URL:         {args.url}")
    print(f"Prompt:      ~{len(prompt)} chars (repeat={args.prompt_repeat})")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Stream:      {args.stream}")
    if args.duration:
        print(f"Mode:        长稳 {args.duration}s, concurrency={levels[0]}")
    else:
        print(f"Requests:    {args.num_requests} per level")
        print(f"Concurrency: {levels}")

    results = []
    if args.duration:
        results.append(run_stability(
            args.url, prompt, levels[0], args.duration, args.max_tokens,
            args.temperature, args.stream, args.timeout))
    else:
        for c in levels:
            print(f"\n[run] concurrency={c} ...")
            results.append(run_level(
                args.url, prompt, args.num_requests, c, args.max_tokens,
                args.temperature, args.stream, args.timeout))

    # ---- 报告 ----
    print(f"\n{'=' * 72}")
    print("Results")
    print("=" * 72)
    headers = ["Conc", "OK/Total", "Err%", "QPS", "tok/s",
               "TTFT p50", "TTFT p95", "Lat p50", "Lat p95", "Lat p99"]
    rows = []
    for r in results:
        rows.append([
            str(r["concurrency"]),
            f'{r["ok"]}/{r["requests"]}',
            f'{r["error_rate"] * 100:.1f}%',
            f'{r["qps"]:.1f}',
            f'{r["tokens_per_s"]:.1f}',
            f'{r["ttft_p50_ms"]:.1f}ms',
            f'{r["ttft_p95_ms"]:.1f}ms',
            f'{r["lat_p50_ms"]:.0f}ms',
            f'{r["lat_p95_ms"]:.0f}ms',
            f'{r["lat_p99_ms"]:.0f}ms',
        ])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    for row in rows:
        print("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))

    errs = [e for r in results for e in r.get("error_samples", [])]
    if errs:
        print("\n错误样例:")
        for e in errs[:5]:
            print(f"  - {e}")

    if args.json_out:
        payload = {
            "meta": {"url": args.url, "prompt_chars": len(prompt),
                     "prompt_repeat": args.prompt_repeat,
                     "stream": args.stream, "max_tokens": args.max_tokens,
                     "duration": args.duration},
            "results": results,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[json] 结果已保存: {args.json_out}")

    print(f"\n{'=' * 72}")
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
