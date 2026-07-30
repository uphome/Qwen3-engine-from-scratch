# 性能记录

> 测试环境: NVIDIA A100-PCIE-40GB / CUDA 11.8 / PyTorch 2.2.2 / bfloat16
> 测试模型: Qwen3-0.6B (751M params, 28 layers, hidden=1024, Q heads=16, KV heads=8)
> 测试命令: `python bench.py --model /data/hjt1/Qwen3-0.6B --num-seqs 64 --warmup 3 --min-input-len 32 --max-input-len 1024 --min-output-len 64 --max-output-len 512 --seed 42`

## 汇总对比

| 版本 | 日期 | 改动 | Throughput (tok/s) | Decode (ms/tok) | Prefill (ms) | Peak VRAM (GB) |
|------|------|------|--------------------|-----------------|-------------|-----------------|
| v0.1 | 2026-07-28 | 纯 PyTorch, NaiveKVCache (torch.cat) | 24.35 | 26.7 | 79.5 | 3.90 |
| v0.2 | 2026-07-30 | PagedKVCache (共享池 + 块表, 单请求) | 25.26 | 36.6 | 602.1 | 3.89 |

## v0.1 详细 — 纯 PyTorch 手写, NaiveKVCache (torch.cat)

### 总体

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 31,458 |
| 总输出 tokens | 19,228 |
| 总 GPU 时间 | 789.7 s |

### 延迟分布

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 491 | 38 | 1012 | 415 | 971 | 998 |
| Output len | 300 | 80 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 79.5 | 25.8 | 179.6 | 76.0 | 155.7 | 166.5 |
| Decode (ms/tok) | 26.7 | 25.5 | 29.5 | 26.6 | 28.2 | 28.9 |
| Total (ms) | 12,339.0 | 3,031.5 | 21,876.6 | 11,989.9 | 20,971.4 | 21,866.4 |

### 时间占比

| Prefill | Decode |
|---------|--------|
| 5.09s (0.6%) | 784.61s (99.4%) |

### VRAM

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 3.01 GB | 3.90 GB | +0.88 GB |

## v0.2 详细 — PagedKVCache (共享物理块池, 单请求串行)

### 总体

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 31,458 |
| 总输出 tokens | 19,228 |
| 总 GPU 时间 | 761.3 s |
| KV pool | 128 blocks × 16 tokens, 224 MB |

### 延迟分布

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 491 | 38 | 1012 | 415 | 971 | 998 |
| Output len | 300 | 80 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 602.1 | 64.0 | 1697.9 | 557.5 | 1174.2 | 1437.1 |
| Decode (ms/tok) | 36.6 | 27.6 | 59.7 | 35.1 | 52.1 | 58.5 |
| Total (ms) | 11,895.5 | 2,596.7 | 25,235.5 | 11,050.6 | 22,200.4 | 25,207.7 |

### 时间占比

| Prefill | Decode |
|---------|--------|
| 38.54s (5.1%) | 722.77s (94.9%) |

### VRAM

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 3.01 GB | 3.89 GB | +0.88 GB |

### 备注

- Throughput 与 v0.1 持平 (25.26 vs 24.35 tok/s)，decode 略快，prefill 因 Python 逐 token 写入而变慢
- Prefill 优化方向：改 `update` 中 Python for 循环为向量化写入
- Decode 优化方向：`get_kv` 每次重建全量 K/V 的 `torch.cat` 可改为 PagedAttention kernel 直接读分页
