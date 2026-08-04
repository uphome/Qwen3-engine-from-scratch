# 性能记录

> 测试环境: NVIDIA A100-PCIE-40GB / CUDA 11.8 / PyTorch 2.2.2 / bfloat16
> 测试模型: Qwen3-0.6B (751M params, 28 layers, hidden=1024, Q heads=16, KV heads=8)
> 测试命令: `python bench.py --model /data/hjt1/Qwen3-0.6B --num-seqs 64 --warmup 3 --min-input-len 32 --max-input-len 1024 --min-output-len 64 --max-output-len 512 --seed 42`
> 计时口径: **真实墙钟**（perf_counter + torch.cuda.synchronize），端到端用户感知延迟
> 历史教训: 早期数据用 CPU 提交时间（无 synchronize），PyTorch 逐页路径被低估 ~4 倍，已废弃重跑

## 版本命名（语义化版本号）

主版本号表示**技术代际**，次版本号表示同代内的改进：

```
v0.1  NaiveKVCache 基线（无分页）
v0.2  PagedKVCache（存储层分页：共享池 + 块表，计算仍标准 attention）
v1.0  PagedAttention（计算层分页：Triton decode kernel + dtype 修复）
v1.1  优化版（prefill 标准 attention + 向量化 update + 计时修正，当前版本）
```

- **v0.x = 计算层未分页**（attention 仍走标准实现，分页只影响存储）
- **v1.x = PagedAttention 时代**（计算也分页，主版本跨入 1.0）
- v1.1 由历史提交 03d1007（prefill 优化）+ 86411cd（计时修正）合并而来，两者代码等价

## 汇总对比（公平口径，真实墙钟）

| 版本 | commit | 改动 | Throughput (tok/s) | Decode (ms/tok) | Prefill (ms) | Peak VRAM (GB) |
|------|--------|------|--------------------|-----------------|-------------|-----------------|
| v0.1 | 4a20a44 | 纯 PyTorch, NaiveKVCache (torch.cat) | **40.72** | **24.3** | 60.0 | 3.90 |
| v0.2 | 9bfbba3 | PagedKVCache + 重建连续 K/V | 27.13 | 34.6 | 562.0 | 3.89 |
| v1.0 | 3ec469b | Triton decode kernel + dtype 修复 | 27.16 | 34.6 | 864.6 | 2.06 |
| v1.1 | 当前 | Triton decode + 标准 prefill | 29.58 | 33.7 | **35.0** | 2.06 |

> v0.1/v0.2/v1.0 为历史 commit 检出 worktree、仅移植 synchronize 计时修复后
> 同环境重跑；v1.1 为当前版本实测。

### 关键发现

1. **单请求串行场景下 v0.1（naive cat）最快**（24.3ms vs 33.7ms）：
   分页/Triton 在 B=1 时反而慢 ~10ms——Triton grid = B×Hkv = 8 个 program，
   A100 108 个 SM 利用率 <8%；naive 的大 matmul 吃满 cuBLAS。
2. **分页/Triton 的价值在并发**：开销靠 continuous batching 摊薄，
   单请求串行是它们的劣势场景——这是做 continuous batching 的动机。
3. **v0.2 走的是"分页存储 + get_kv 重建连续 K/V + 标准 attention"**
   （每步 torch.cat 全量重建，+10ms vs v0.1），不是逐页循环；
   逐页循环（~150ms）是 v1.0 引入、v1.1 prefill 优化时移除的。
4. **prefill 优化收益最大**：v1.0 旧逐页 864.6ms → v1.1 标准 attention 35.0ms（-96%）。
5. **bf16 权重收益 = VRAM 减半**（3.90→2.06 GB），速度无贡献（v1.0 vs v0.2 decode 相同）。

### 为什么 v0.2/v1.0 的 prefill 这么慢

**v0.2（562ms）— 卡在 `update` 的逐 token Python 循环**

```python
# v0.2 的 update（9bfbba3）：
for t in range(S_new):                    # 平均 491 个 token 循环 491 次
    pos = write_pos + t
    block_idx = pos // block_size         # 每次 Python 除法/取模
    while len(self.block_table) <= block_idx:   # 每次都要检查页表
        self._allocate_block()
    pool.k_buffer[phys_id, layer, :, offset] = k_new[0, :, t, :]   # 1 次小 kernel 写 1 个 token
```

开销 = 28 层 × 491 token = **13,748 次 Python 循环**，每次循环包含
Python 解释器开销（~1µs）+ 一次 kernel 启动（~5-10µs，每次只写 1 个 token 的 K/V），
累积 → 562ms。注意此时 attention 本身是标准实现（连续 K/V），并不慢。

**v1.0（864.6ms）— 双重浪费：写进分页 + 逐页读回**

v1.0 的 prefill 走 `paged_attention()`，比 v0.2 多了一整趟无效往返：

```python
kv_cache.update(layer_idx, k_new, v_new)   # ① 逐 token 循环写入分页（~500ms）
...
for j, phys_id in enumerate(block_table):  # ② 又从分页逐页读回来！
    k_page = repeat_kv(pool.k_buffer[phys_id, layer_idx]...)   # repeat_kv → kernel
    s = matmul(q, k_page)                  # matmul → kernel
    exp / max / sum / matmul(p, v_page)    # 每页 ~6 次 kernel 启动
```

② 的额外开销：平均 ~31 页 × 28 层 = 868 次页循环，每页 6 次 kernel 启动
≈ **5000+ 次小 kernel**，且每页都是 batch=1 的小矩阵（tensor core 利用率 <1%）。
这解释了 864.6 vs 562 的 +300ms 差距。

**最讽刺的是**：prefill 的 K/V 本来就是连续张量，v1.0 却先把它拆散写进分页、
再逐页读回来重建——纯白折腾（"写页→逐页读回"的无效往返）。

**v1.1（35ms）— 两个修复合击**

| 阶段 | v0.2 | v1.0 | v1.1 |
|---|---|---|---|
| 写入分页 | 逐 token 循环（~500ms） | 逐 token 循环（~500ms） | **向量化 1 次写入**（advanced indexing） |
| 算 attention | 标准（连续 K/V，快） | 逐页读回（~300ms） | **标准（连续 K/V，快）** |
| **合计** | **562ms** | **864.6ms** | **35ms** |

- `update` 向量化（v1.1）→ 打掉 ① 的逐 token 循环（13,748 次 → 1 次）
- prefill 改 `_standard_attention`（v1.1）→ 打掉 ② 的逐页读回（根本不做分页计算）
- 35ms 只剩 28 层逐层调度的固定成本，不再随 prompt 长度线性增长

---

## v1.1 详细 — 当前版本（Triton decode + 标准 prefill）

### 总体

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 30,095 |
| 总输出 tokens | 19,271 |
| 总 GPU 时间 | 651.5 s |
| KV pool | 128 blocks × 16 tokens, 224 MB |

### 延迟分布

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 470 | 38 | 1012 | 403 | 970 | 998 |
| Output len | 301 | 76 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 35.0 | 30.0 | 51.0 | 33.3 | 48.0 | 50.7 |
| Decode (ms/tok) | 33.7 | 31.3 | 35.7 | 33.9 | 34.9 | 35.4 |
| Total (ms) | 10,179.2 | 2,528.5 | 18,088.5 | 10,089.0 | 16,759.8 | 17,846.9 |

### 时间占比

| Prefill | Decode |
|---------|--------|
| 2.24s (0.3%) | 649.23s (99.7%) |

### VRAM

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 1.50 GB | 2.06 GB | +0.56 GB |

### 架构（三段式 attention 路径）

```
attention forward
├── PagedKVCache 分支
│   ├── decode (S=1, CUDA, bf16)  → Triton kernel（qwen3/kernels/paged_attention.py）
│   ├── prefill (S>1)              → 标准 attention + 写入分页
│   └── decode 兜底 (CPU/fp32)     → PyTorch 逐页（paged_attention.py）
└── NaiveKVCache 分支              → cat 拼历史 + 标准 attention
```

### 本轮修复（dtype bugs）

- `weights.py`: `model.to(device)` 漏 dtype → 权重一直 float32（伪 bf16）；改为 `model.to(device, dtype)`
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
| A: bf16 + Triton kernel | 33.7 |
| B: bf16 + PyTorch 逐页   | 176.6 |
| C: fp32 + PyTorch 逐页   | 155.6 |

结论：
- **decode 提速 100% 归因于 Triton kernel**（B vs A 差 5.2 倍），bf16 权重本身无贡献（B vs C 甚至微负）
- bf16 的收益是 VRAM 减半（3.89 → 2.06 GB），属带宽优化而非速度优化
- Triton 路径 CPU 开销趋零，两种计时口径一致；PyTorch 逐页路径 CPU 提交与真实墙钟差 ~4 倍

### Prefill 优化（v1.1 两大改动）

1. **prefill 不再走逐页 PyTorch 实现**：K/V 本来就是连续张量，直接标准 attention
   （`_standard_attention`，vLLM 同款做法），顺带写入分页供 decode 使用
2. **`PagedKVCache.update` 向量化**：advanced indexing 一次性写入 S 个 token，
   替代逐 token Python 循环

效果（bench_prefill.py 专项，bf16）：

| prefill 长度 | 耗时 |
|------|------|
| 128 | 42.1 ms |
| 512 | 42.9 ms |
| 1024 | 43.6 ms |

prefill 时间 ≈ 常数（~43ms 固定开销主导：28 层逐层调度 + kernel 启动），
不再随序列长度线性增长。全量对比：864.6 → 35.0 ms（-96%）。

---

## v1.0 详细 — PagedAttention：Triton decode kernel + dtype 修复（重跑）

### 总体

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 30,095 |
| 总输出 tokens | 19,271 |
| 总 GPU 时间 | 709.6 s |

### 延迟分布

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 470 | 38 | 1012 | 403 | 970 | 998 |
| Output len | 301 | 76 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 864.6 | 93.7 | 2301.2 | 738.5 | 1832.8 | 2217.4 |
| Decode (ms/tok) | 34.6 | 30.9 | 50.4 | 31.9 | 47.8 | 49.6 |
| Total (ms) | 11,087.1 | 2,589.2 | 24,427.1 | 11,481.2 | 16,992.2 | 20,162.5 |

### 时间占比

| Prefill | Decode |
|---------|--------|
| 55.33s (7.8%) | 654.24s (92.2%) |

### VRAM

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 1.50 GB | 2.06 GB | +0.56 GB |

---

## v0.2 详细 — PagedKVCache + 重建连续 K/V（重跑）

### 总体

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 31,458 |
| 总输出 tokens | 19,228 |
| 总 GPU 时间 | 708.6 s |
| KV pool | 128 blocks × 16 tokens, 224 MB |

### 延迟分布

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 491 | 38 | 1012 | 415 | 971 | 998 |
| Output len | 300 | 80 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 562.0 | 64.6 | 1342.8 | 473.8 | 1082.4 | 1207.5 |
| Decode (ms/tok) | 34.6 | 26.6 | 43.0 | 34.5 | 40.9 | 42.7 |
| Total (ms) | 11,072.0 | 2,549.6 | 22,589.4 | 10,544.5 | 19,723.1 | 21,650.5 |

### 时间占比

| Prefill | Decode |
|---------|--------|
| 35.97s (5.1%) | 672.64s (94.9%) |

### VRAM

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 3.01 GB | 3.89 GB | +0.88 GB |

### 备注

- decode = `get_kv` 每步 torch.cat 全量重建连续 K/V + 标准 attention（+10ms vs v0.1）
- prefill = 逐 token Python 循环写入分页（562ms）

---

## v0.1 详细 — 纯 PyTorch 手写, NaiveKVCache (torch.cat)（重跑）

### 总体

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 31,458 |
| 总输出 tokens | 19,228 |
| 总 GPU 时间 | 472.2 s |

### 延迟分布

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 491 | 38 | 1012 | 415 | 971 | 998 |
| Output len | 300 | 80 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 60.0 | 23.9 | 124.0 | 47.6 | 111.1 | 118.7 |
| Decode (ms/tok) | 24.3 | 23.4 | 25.7 | 24.2 | 25.3 | 25.5 |
| Total (ms) | 7,377.6 | 1,983.2 | 12,567.5 | 7,377.3 | 12,192.6 | 12,441.1 |

### 时间占比

| Prefill | Decode |
|---------|--------|
| 3.84s (0.8%) | 468.33s (99.2%) |

### VRAM

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 3.01 GB | 3.90 GB | +0.88 GB |
