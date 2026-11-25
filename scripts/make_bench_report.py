#!/usr/bin/env python3
"""从压测 JSON 生成 BENCHMARK.md + 曲线图

用法:
    python scripts/make_bench_report.py \
        --engine-json results/engine.json \
        --server-json results/server.json \
        --out BENCHMARK.md --figdir benchmark
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, nd=1, suffix=""):
    return f"{v:.{nd}f}{suffix}"


def md_engine(data):
    lines = ["## 1. 连续批处理（bench_batched，引擎级）", ""]
    meta = data.get("meta", {})
    lines.append(f"> 模型 `{meta.get('model', '?')}`，{meta.get('num_seqs', '?')} 序列，"
                 f"input {meta.get('input_len')}，output {meta.get('output_len')}，"
                 f"CUDA Graph={meta.get('graph')}")
    lines.append(">")
    lines.append("> **TTFT 口径**：所有请求在同一时刻离线到达，因此 batched 的 TTFT"
                 "包含在 waiting 中的排队时间（batch 越小排队越长）；serial 为逐请求"
                 "执行（无排队，TTFT≈prefill 时间）。HTTP 压测的 TTFT 为真实并发视角。")
    lines += ["", "| Mode | Batch | Throughput | TTFT p50 | TTFT p95 | TPOT | "
              "Decode ms/tok | Lat p50 | Lat p95 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["results"]:
        n_decode = max(1, r["n_tokens"] - max(1, r["batch_size"]))
        tpot = r["decode_ms"] / n_decode
        tok_s = r["n_tokens"] / r["total_s"] if r["total_s"] else 0
        dpt = r["decode_ms"] / r["n_tokens"] if r["n_tokens"] else 0
        ttft = r.get("ttfts", [])
        p50 = sorted(ttft)[int(0.5 * (len(ttft) - 1))] * 1000 if ttft else 0
        p95 = sorted(ttft)[int(0.95 * (len(ttft) - 1))] * 1000 if ttft else 0
        lat = r.get("latencies", [])
        lp50 = sorted(lat)[int(0.5 * (len(lat) - 1))] * 1000 if lat else 0
        lp95 = sorted(lat)[int(0.95 * (len(lat) - 1))] * 1000 if lat else 0
        lines.append(
            f"| {r['mode']} | {r['batch_size']} | {fmt(tok_s)} tok/s | "
            f"{fmt(p50)}ms | {fmt(p95)}ms | {fmt(tpot, 2)}ms | "
            f"{fmt(dpt, 2)} ms/tok | {fmt(lp50/1000, 2)}s | {fmt(lp95/1000, 2)}s |")
    lines.append("")
    return lines


def md_server(data):
    lines = ["## 2. HTTP 服务（bench_server）", ""]
    meta = data.get("meta", {})
    mode = f"长稳 {meta.get('duration')}s" if meta.get("duration") else "并发梯度"
    lines.append(f"> {mode}，stream={meta.get('stream')}，"
                 f"max_tokens={meta.get('max_tokens')}，"
                 f"prompt≈{meta.get('prompt_chars')} chars")
    lines += ["", "| 并发 | OK/Total | 错误率 | QPS | tok/s | TTFT p50 | "
              "TTFT p95 | Lat p50 | Lat p95 | Lat p99 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["results"]:
        lines.append(
            f"| {r['concurrency']} | {r['ok']}/{r['requests']} | "
            f"{fmt(r['error_rate']*100)}% | {fmt(r['qps'])} | "
            f"{fmt(r['tokens_per_s'])} | {fmt(r['ttft_p50_ms'])}ms | "
            f"{fmt(r['ttft_p95_ms'])}ms | {fmt(r['lat_p50_ms'], 0)}ms | "
            f"{fmt(r['lat_p95_ms'], 0)}ms | {fmt(r['lat_p99_ms'], 0)}ms |")
    errs = [e for r in data["results"] for e in r.get("error_samples", [])]
    if errs:
        lines += ["", "错误样例：", ""]
        for e in errs[:5]:
            lines.append(f"- `{e}`")
    lines.append("")
    return lines


def md_stability(data):
    lines = ["## 3. 长稳测试", ""]
    meta = data.get("meta", {})
    lines.append(f"> 时长 {meta.get('duration')}s，并发 {data['results'][0]['concurrency']}，"
                 f"stream={meta.get('stream')}")
    r = data["results"][0]
    lines += ["", "| 请求数 | 错误率 | QPS | tok/s | TTFT p50 | TTFT p95 | "
              "Lat p50 | Lat p95 | Lat p99 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(
        f"| {r['ok']}/{r['requests']} | {fmt(r['error_rate']*100)}% | "
        f"{fmt(r['qps'])} | {fmt(r['tokens_per_s'])} | "
        f"{fmt(r['ttft_p50_ms'])}ms | {fmt(r['ttft_p95_ms'])}ms | "
        f"{fmt(r['lat_p50_ms'], 0)}ms | {fmt(r['lat_p95_ms'], 0)}ms | "
        f"{fmt(r['lat_p99_ms'], 0)}ms |")
    wins = r.get("windows") or []
    if wins:
        lines += ["", "窗口（每 30s）请求数与 tokens（无衰减 = 稳定）：", "",
                  "| t(s) | 累计请求 | 累计 tokens | 错误 |",
                  "|---:|---:|---:|---:|"]
        for w in wins:
            lines.append(f"| {w['t']} | {w['requests']} | {w['tokens']} | {w['errors']} |")
    lines.append("")
    return lines


def plot_stability(data, figdir):
    r = data["results"][0]
    wins = r.get("windows") or []
    if len(wins) < 2:
        return []
    ts = [w["t"] for w in wins]
    reqs = [w["requests"] for w in wins]
    toks = [w["tokens"] for w in wins]
    d_req = [reqs[0]] + [reqs[i] - reqs[i - 1] for i in range(1, len(reqs))]
    d_tok = [toks[0]] + [toks[i] - toks[i - 1] for i in range(1, len(toks))]
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(ts, d_tok, "o-", label="tokens/window")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("tokens per window")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ts, d_req, "s--", color="tab:orange", label="requests/window")
    ax2.set_ylabel("requests per window")
    ax1.set_title("Stability (120s, concurrency 8)")
    p = os.path.join(figdir, "stability.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return [p]


def plot_engine(data, figdir):
    rs = [r for r in data["results"] if r["mode"] == "batched"]
    if not rs:
        return []
    xs = [r["batch_size"] for r in rs]
    tok = [r["n_tokens"] / r["total_s"] for r in rs]
    ttft = [sorted(r.get("ttfts", [0]))[int(0.5 * (len(r.get("ttfts", [0])) - 1))] * 1000
            for r in rs]
    figs = []
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, tok, "o-")
    ax.set_xlabel("batch size")
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title("Continuous batching throughput")
    ax.grid(alpha=0.3)
    p = os.path.join(figdir, "engine_throughput.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    figs.append(p)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ttft, "s-")
    ax.set_xlabel("batch size")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("TTFT vs batch size")
    ax.grid(alpha=0.3)
    p = os.path.join(figdir, "engine_ttft.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    figs.append(p)
    return figs


def plot_server(data, figdir):
    rs = data["results"]
    if not rs:
        return []
    xs = [r["concurrency"] for r in rs]
    figs = []
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, [r["qps"] for r in rs], "o-", label="QPS")
    ax.plot(xs, [r["tokens_per_s"] for r in rs], "s-", label="tokens/s")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("rate")
    ax.set_title("HTTP server throughput")
    ax.grid(alpha=0.3); ax.legend()
    p = os.path.join(figdir, "server_qps.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    figs.append(p)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, [r["ttft_p50_ms"] for r in rs], "o-", label="TTFT p50")
    ax.plot(xs, [r["ttft_p95_ms"] for r in rs], "s--", label="TTFT p95")
    ax.plot(xs, [r["lat_p95_ms"] for r in rs], "^--", label="E2E p95")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("latency (ms)")
    ax.set_title("HTTP latency vs concurrency")
    ax.grid(alpha=0.3); ax.legend()
    p = os.path.join(figdir, "server_latency.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    figs.append(p)
    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-json", default=None)
    ap.add_argument("--server-json", default=None)
    ap.add_argument("--stability-json", default=None)
    ap.add_argument("--out", default="BENCHMARK.md")
    ap.add_argument("--figdir", default="benchmark")
    args = ap.parse_args()

    engine = load(args.engine_json)
    server = load(args.server_json)
    stability = load(args.stability_json)
    os.makedirs(args.figdir, exist_ok=True)
    figs = []
    if engine:
        figs += plot_engine(engine, args.figdir)
    if server:
        figs += plot_server(server, args.figdir)
    if stability:
        figs += plot_stability(stability, args.figdir)

    lines = ["# 压测报告（BENCHMARK）", "",
             "> 由 `scripts/make_bench_report.py` 从压测 JSON 自动生成。",
             ""]
    if engine:
        lines += md_engine(engine)
    if server:
        lines += md_server(server)
    if stability:
        lines += md_stability(stability)
    if figs:
        lines += ["## 曲线", ""]
        for p in figs:
            lines.append(f"![{os.path.basename(p)}]({p})")
            lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] 写入 {args.out}（{len(figs)} 张图 -> {args.figdir}/）")


if __name__ == "__main__":
    main()
