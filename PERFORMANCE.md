# 性能记录

> 测试环境: NVIDIA A100-PCIE-40GB / CUDA 11.8 / PyTorch 2.2.2 / bfloat16
> 测试模型: Qwen3-0.6B (751M params, 28 layers, hidden=1024, Q heads=16, KV heads=8)
> 测试命令: `python bench.py --model /data/hjt1/Qwen3-0.6B --num-seqs 64 --warmup 3 --min-input-len 32 --max-input-len 1024 --min-output-len 64 --max-output-len 512 --seed 42`
>
> ⚠ 重要更正：v0.1/v0.2 因 weights.py 漏传 dtype，权重实际为 float32（伪 bf16）；
>   显存 3.01 GB（=0.75B×4B）可证。v0.3 修复后为真 bf16（1.50 GB）。
>   v0.2 的 Decode 36.6 ms/tok 是 CUDA Event 量的纯 GPU 时间，真实墙钟 ~155 ms/tok
>   （Python 逐页循环的 CPU 开销被 Event 隐藏），消融实验见 v0.3 备注。

## 汇总对比

| 版本 | 日期 | 改动 | Throughput (tok/s) | Decode (ms/tok) | Prefill (ms) | Peak VRAM (GB) |
|------|------|------|--------------------|-----------------|-------------|-----------------|
| v0.1 | 2026-07-28 | 纯 PyTorch, NaiveKVCache (torch.cat) | 24.35 | 26.7 | 79.5 | 3.90 |
| v0.2 | 2026-07-30 | PagedKVCache (共享池 + 块表, 单请求) | 25.26 | 36.6* | 602.1 | 3.89 |
| v0.3 | 2026-08-04 | Triton decode kernel + 修复 weights/norm dtype | 28.89 | 31.7 | 797.3 | 2.06 |
| v0.4 | 2026-08-04 | prefill 标准 attention + update 向量化 | 27.49 | 36.1† | **50.0** | 2.06 |

* v0.2 的 Decode 为 CUDA Event 口径（纯 GPU 时间）；墙钟口径 ~155 ms/tok。
† v0.4 跑时 GPU 有外部负载，decode 数值偏高（消融 A/B/C 三组同步偏移 ~8%，代码无退化）。

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

## v0.3 详细 — Triton decode kernel + dtype 修复

### 总体

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 30,095 |
| 总输出 tokens | 19,271 |
| 总 GPU 时间 | 667.0 s |
| KV pool | 128 blocks × 16 tokens, 224 MB |

### 延迟分布

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 470 | 38 | 1012 | 403 | 970 | 998 |
| Output len | 301 | 76 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 797.3 | 92.7 | 1666.0 | 668.0 | 1587.1 | 1659.0 |
| Decode (ms/tok) | 31.7 | 29.9 | 41.7 | 31.6 | 33.3 | 39.0 |
| Total (ms) | 10,421.2 | 2,579.6 | 18,706.3 | 10,055.7 | 16,279.1 | 17,932.4 |

### 时间占比

| Prefill | Decode |
|---------|--------|
| 51.03s (7.7%) | 615.93s (92.3%) |

### VRAM

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 1.50 GB | 2.06 GB | +0.56 GB |

### 本轮修复

- `weights.py`: `model.to(device)` 漏 dtype → 权重一直 float32；改为 `model.to(device, dtype)`
- `norm.py`: RMSNorm 输出被 float32 weight 升回 float32；改为整体计算后统一 `.to(x.dtype)`
- `paged_attention.py`: `v_page` 升 fp32 统一计算精度
- Triton kernel: `tl.dot` 要求维度 ≥ 16，GQA group 仅 2/4，q tile 补齐 BLOCK_G=16

### 精度验证

- fp32 vs bf16 贪心输出: 40/40 token 一致（dtype 修复无精度退化）
- naive(cat) vs paged(Triton decode): 48/48 token 一致

### 消融实验（decode 提速归因）

同环境同 prompt，256 input + 64 decode 步，墙钟口径：

| 配置 | Decode (ms/tok) |
|------|-----------------|
| A: bf16 + Triton kernel | 31.99 |
| B: bf16 + PyTorch 逐页   | 176.56 |
| C: fp32 + PyTorch 逐页   | 155.64 |

结论：
- **decode 提速 100% 归因于 Triton kernel**（B vs A 差 5.5 倍），bf16 权重本身无贡献（B vs C 甚至微负）
- bf16 的收益是 VRAM 减半（3.89 → 2.06 GB），属带宽优化而非速度优化
- v0.2 的 36.6 ms/tok 是 CUDA Event 口径（纯 GPU 时间），Python 循环的 CPU 开销被隐藏；
  真实墙钟 ~155 ms/tok。Triton 路径 CPU 开销趋零，两种口径一致（≈32 ms）
- prefill 仍走 PyTorch 路径且变慢（602→797 ms），疑因 `v_page.float()` 每页转换 + bf16 小算子开销

## v0.4 详细 — prefill 优化（标准 attention + 向量化 update）

### 改动

1. **prefill 不再走逐页 PyTorch 实现**：K/V 本来就是连续张量，直接标准 attention
   （`_standard_attention`，vLLM 同款做法），顺带写入分页供 decode 使用，
   省去"写页 → 逐页读回重建"的无效往返
2. **`PagedKVCache.update` 向量化**：advanced indexing 一次性写入 S 个 token，
   替代逐 token Python 循环

### 效果（bench_prefill.py 专项，bf16，GPU 2）

| prefill 长度 | 耗时 | 对比 v0.3 旧路径 |
|------|------|------|
| 128 | 42.1 ms | - |
| 512 | 42.9 ms | - |
| 1024 | 43.6 ms | - |

prefill 时间 ≈ 常数（~43ms 固定开销主导：28 层逐层调度 + kernel 启动），
不再随序列长度线性增长。bench 全量：Prefill 797 → 50 ms（-94%），
占总时间 7.7% → 0.5%。

### 完整 benchmark（v0.4）

| 指标 | 数值 |
|------|------|
| Total GPU time | 700.9 s |
| Throughput | 27.49 tok/s |
| Prefill (mean) | 50.0 ms |
| Decode (mean) | 36.1 ms/tok（含外部负载，见汇总注释） |
| Peak VRAM | 2.06 GB |

### 备注

- prefill 优化后为常数开销，长 prompt 场景收益最大
- 下一步方向：continuous batching（多请求共享池并发调度）
