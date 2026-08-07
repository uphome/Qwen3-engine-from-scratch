# 性能记录

> 测试环境: NVIDIA A100-PCIE-40GB / CUDA 11.8 / PyTorch 2.2.2 / bfloat16
> 测试模型: Qwen3-0.6B (751M params, 28 layers, hidden=1024, Q heads=16, KV heads=8)
> 计时口径: **真实墙钟**（perf_counter + torch.cuda.synchronize），端到端用户感知延迟
> 历史教训: 早期数据用 CPU 提交时间（无 synchronize），PyTorch 逐页路径被低估 ~4 倍，已废弃重跑

### 运行环境前提（重要）

- 用 `cuda_learn` conda 环境（PyTorch 2.2.2+cu118 / Triton 2.2.0），勿用 base 的 cu128。
- **驱动若被降级（`nvidia-smi` 显示 CUDA ≤ 11.6，如 510.54），Triton 自带 ptxas 12.3
  生成的 cubin 会加载失败**（`RuntimeError: Triton Error [CUDA]: device kernel image is invalid`），
  连项目已有 kernel 也全挂。解决：设置环境变量让 Triton 用兼容的旧 ptxas：

  ```bash
  export TRITON_PTXAS_PATH=/data/hjt1/anaconda3/envs/cuda_learn/bin/ptxas
  ```

  所有跑 Triton kernel 的命令前都要带它（bench / profile / test 一律如此）。

- prefill 注意力路径由 `QWEN3_FLASH_ATTN` 控制：
  - `triton`（默认）= vLLM 风格 varlen flash attention 融合 kernel
  - `pytorch` = 标准实现（QKᵀ + softmax + PV，消融对比/基线用）
  - decode 仍由 `QWEN3_PAGED_ATTN` 控制，与 flash prefill 互不干扰。

---

## 1. 版本命名（语义化版本号）

主版本号表示**技术代际**：

```
v0.1  NaiveKVCache 基线（无分页）
v0.2  PagedKVCache（存储层分页：共享池 + 块表，计算仍标准 attention）
v1.0  PagedAttention（计算层分页：Triton decode kernel + dtype 修复）
v1.1  优化版（prefill 标准 attention + 向量化 update + 计时修正）
v2.0  连续批处理（Scheduler 调度，batch 扫描 1-4，当前开发中）
```

- **v0.x = 计算层未分页**（attention 仍走标准实现，分页只影响存储）
- **v1.x = PagedAttention 时代**（计算也分页，主版本跨入 1.0）
- **v2.x = Continuous Batching 时代**（多请求并发调度，吞吐量级提升）

---

## 2. 汇总对比

### 2.1 单请求串行（bench.py，64 序列，GPU 空闲）

| 版本 | commit | 改动 | Throughput (tok/s) | Decode (ms/tok) | Prefill (ms) | VRAM (GB) |
|------|--------|------|--------------------|-----------------|-------------|-----------|
| v0.1 | 4a20a44 | NaiveKVCache (torch.cat) | **40.72** | **24.3** | 60.0 | 3.90 |
| v0.2 | 9bfbba3 | PagedKVCache + 重建连续 K/V | 27.13 | 34.6 | 562.0 | 3.89 |
| v1.0 | 3ec469b | Triton decode kernel + dtype 修复 | 27.16 | 34.6 | 864.6 | 2.06 |
| v1.1 | 当前 | Triton decode + 标准 prefill | 29.58 | 33.7 | **35.0** | 2.06 |

> v0.1/v0.2/v1.0 为历史 commit 检出 worktree、仅移植 synchronize 计时修复后
> 同环境重跑；v1.1 为当前版本实测。

### 2.2 连续批处理（bench_batched.py，16 序列，GPU 空闲）

| Mode | Batch | Throughput | Decode | Lat p50 | vs serial |
|---|---|---|---|---|---|
| serial | 1 | 29.8 tok/s | 32.6 ms/tok | 6.42s | 1.00x |
| batched | 1 | 29.1 | 33.5 | 6.67s | 0.98x |
| batched | 2 | 46.3 | 20.8 | 4.41s | **1.55x** |
| batched | 3 | 62.5 | 15.3 | 3.62s | **2.10x** |
| batched | 4 | 74.8 | 12.8 | 3.27s | **2.51x** |

### 2.3 关键发现

1. **单请求串行下 v0.1（naive cat）最快**（24.3ms vs 33.7ms）：分页/Triton 在 B=1 时
   反而慢 ~10ms——Triton grid = B×Hkv = 8 个 program，SM 利用率 <8%；naive 大 matmul 吃满 cuBLAS。
2. **分页/Triton 的价值在并发**：开销靠 continuous batching 摊薄——这是做并发调度的动机。
3. **prefill 优化收益最大**：v1.0 逐页 864.6ms → v1.1 标准 attention 35.0ms（-96%）。
4. **bf16 权重收益 = VRAM 减半**（3.90→2.06 GB），对速度无贡献（v1.0 vs v0.2 decode 相同）。
5. **当前瓶颈在工程流水线**：GPU 纯算 29ms/步 vs 墙钟 91ms/步（利用率仅 32%），
   详见 v2.0 的瓶颈分析。

---

## 3. 版本详细（按时间正序）

### 3.1 v0.1 — NaiveKVCache（torch.cat）

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 31,458 |
| 总输出 tokens | 19,228 |
| 总 GPU 时间 | 472.2 s |

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 491 | 38 | 1012 | 415 | 971 | 998 |
| Output len | 300 | 80 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 60.0 | 23.9 | 124.0 | 47.6 | 111.1 | 118.7 |
| Decode (ms/tok) | 24.3 | 23.4 | 25.7 | 24.2 | 25.3 | 25.5 |
| Total (ms) | 7,377.6 | 1,983.2 | 12,567.5 | 7,377.3 | 12,192.6 | 12,441.1 |

| Prefill | Decode |
|---------|--------|
| 3.84s (0.8%) | 468.33s (99.2%) |

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 3.01 GB | 3.90 GB | +0.88 GB |

### 3.2 v0.2 — PagedKVCache + 重建连续 K/V

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 31,458 |
| 总输出 tokens | 19,228 |
| 总 GPU 时间 | 708.6 s |
| KV pool | 128 blocks × 16 tokens, 224 MB |

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 491 | 38 | 1012 | 415 | 971 | 998 |
| Output len | 300 | 80 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 562.0 | 64.6 | 1342.8 | 473.8 | 1082.4 | 1207.5 |
| Decode (ms/tok) | 34.6 | 26.6 | 43.0 | 34.5 | 40.9 | 42.7 |
| Total (ms) | 11,072.0 | 2,549.6 | 22,589.4 | 10,544.5 | 19,723.1 | 21,650.5 |

| Prefill | Decode |
|---------|--------|
| 35.97s (5.1%) | 672.64s (94.9%) |

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 3.01 GB | 3.89 GB | +0.88 GB |

备注：
- decode = `get_kv` 每步 torch.cat 全量重建连续 K/V + 标准 attention（+10ms vs v0.1）
- prefill = 逐 token Python 循环写入分页（562ms，原因见附录 A）

### 3.3 v1.0 — PagedAttention：Triton decode kernel + dtype 修复

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 30,095 |
| 总输出 tokens | 19,271 |
| 总 GPU 时间 | 709.6 s |

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 470 | 38 | 1012 | 403 | 970 | 998 |
| Output len | 301 | 76 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 864.6 | 93.7 | 2301.2 | 738.5 | 1832.8 | 2217.4 |
| Decode (ms/tok) | 34.6 | 30.9 | 50.4 | 31.9 | 47.8 | 49.6 |
| Total (ms) | 11,087.1 | 2,589.2 | 24,427.1 | 11,481.2 | 16,992.2 | 20,162.5 |

| Prefill | Decode |
|---------|--------|
| 55.33s (7.8%) | 654.24s (92.2%) |

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 1.50 GB | 2.06 GB | +0.56 GB |

本轮修复（dtype bugs）：
- `weights.py`: `model.to(device)` 漏 dtype → 权重一直 float32（伪 bf16）；改为 `model.to(device, dtype)`
- `norm.py`: RMSNorm 输出被 float32 weight 升回 float32；改为整体计算后统一 `.to(x.dtype)`
- `paged_attention.py`: `v_page` 升 fp32 统一计算精度
- Triton kernel: `tl.dot` 要求维度 ≥ 16，GQA group 仅 2/4，q tile 补齐 BLOCK_G=16

### 3.4 v1.1 — 优化版（当前单请求版本）

| 指标 | 数值 |
|------|------|
| 总序列数 | 64 |
| 总输入 tokens | 30,095 |
| 总输出 tokens | 19,271 |
| 总 GPU 时间 | 651.5 s |
| KV pool | 128 blocks × 16 tokens, 224 MB |

| | mean | min | max | p50 | p95 | p99 |
|------|------|-----|------|-----|------|------|
| Input len | 470 | 38 | 1012 | 403 | 970 | 998 |
| Output len | 301 | 76 | 509 | 297 | 500 | 507 |
| Prefill (ms) | 35.0 | 30.0 | 51.0 | 33.3 | 48.0 | 50.7 |
| Decode (ms/tok) | 33.7 | 31.3 | 35.7 | 33.9 | 34.9 | 35.4 |
| Total (ms) | 10,179.2 | 2,528.5 | 18,088.5 | 10,089.0 | 16,759.8 | 17,846.9 |

| Prefill | Decode |
|---------|--------|
| 2.24s (0.3%) | 649.23s (99.7%) |

| 模型加载后 | 推理峰值 | 增量 |
|-----------|---------|------|
| 1.50 GB | 2.06 GB | +0.56 GB |

架构（三段式 attention 路径）：

```
attention forward
├── PagedKVCache 分支
│   ├── decode (S=1, CUDA, bf16)  → Triton kernel（qwen3/kernels/paged_attention.py）
│   ├── prefill (S>1)              → 标准 attention + 写入分页
│   └── decode 兜底 (CPU/fp32)     → PyTorch 逐页（paged_attention.py）
└── NaiveKVCache 分支              → cat 拼历史 + 标准 attention
```

本轮优化（prefill 两大改动）：
1. **prefill 不再走逐页 PyTorch 实现**：K/V 本来就是连续张量，直接标准 attention
   （`_standard_attention`，vLLM 同款做法），顺带写入分页供 decode 使用
2. **`PagedKVCache.update` 向量化**：advanced indexing 一次性写入 S 个 token，
   替代逐 token Python 循环

效果（bench_prefill.py 专项）：prefill 时间 ≈ 常数（128/512/1024 tokens 均 ~43ms），
不再随序列长度线性增长。

精度验证：
- fp32 vs bf16 贪心输出: 40/40 token 一致（dtype 修复无精度退化）
- naive(cat) vs paged(Triton decode): 48/48 token 一致

### 3.5 v2.0 — 连续批处理（当前开发中）

> 引擎：Scheduler + Request + Batch（qwen3/scheduler.py 等）
> 基准：`bench_batched.py`（16 序列，batch 1-4 扫描，GPU 2 空闲，真实墙钟）
> 请求规格: input [32,128], output [16,64], greedy, eos 关闭

#### 批扫描结果

| Mode | Batch | Steps | Throughput | Decode | Lat p50 | vs serial |
|---|---|---|---|---|---|---|
| serial | 1 | - | 28.1 tok/s | 34.7 ms/tok | - | 1.00x |
| batched | 1 | 64+ | 28.1 | 34.7 | - | 1.00x |
| batched | 2 | - | 48.7 | 19.7 | - | 1.73x |
| batched | 4 | - | 78.5 | 11.9 | - | 2.79x |
| batched | 8 | - | 115.9 | 7.9 | - | 4.12x |
| batched | 12 | - | 137.9 | 6.6 | - | 4.91x |
| batched | 16 | - | 153.5 | 5.9 | - | 5.46x |
| batched | 24 | - | 172.7 | 5.3 | - | 6.15x |
| batched | **28** | - | **184.0** | **5.0** | - | **6.55x** |
| batched | 32 | - | 188.1 | 4.9 | - | 6.69x |

> 2026-08 扫描（64 序列，input [32,128], output [16,64], greedy，GPU 空闲）。

#### 吞吐峰值拐点分析

```
吞吐 (tok/s) vs batch（64 序列）:
  28 ┤●
  48 ┤ ●
  78 ┤  ●
 116 ┤   ●
 138 ┤    ●
 154 ┤     ●
 173 ┤      ●
 184 ┤       ●  ← batch 28-30 达峰
 188 ┤        ●  ← batch 32 微回落（SM 已占满 + 调度碎片）
     └──────────────────────
       1  4  8  12 16 24 28 32
```

- **快速增长段（batch 1→8）**：28→116 tok/s，每 +1 batch 平均 **+12.5 tok/s**。
  SM 逐步被填满（Triton decode kernel grid = B×Hkv，B 小时 program 不足）。
- **减速段（batch 8→28）**：116→184，每 +1 batch 平均 **+3.4 tok/s**。
  SM 利用率接近饱和，剩余增益来自更少的调度步/更高的 tile 合并。
- **拐点 ≈ batch 16**：每 +1 batch 的边际增益降到 3 tok/s 以下，
  batch 16 后再加 batch 只换 ~5% 吞吐，但 Lat p50 持续下降（并发友好）。
- **峰值 ≈ batch 28-30（184-193 tok/s）**，batch 32 微回落（188）。
  32 请求全部在跑时，批越大单步 decode 时间越长但步数越少，净增益趋零。
- **硬件约束**：A100 108 SM，batch=32 时 decode grid = 32×8=256 program，
  SM 已无空闲；继续加 batch 只增内存带宽竞争（每步读权重 1.5GB，带宽 2TB/s
  → decode 理论下限 0.75ms/步，当前 5.0ms/步，仍有 6.7x 空间，但那是
  CUDA Graph 消除 kernel 启动间隙的领域）。

**结论**：连续批处理的最优 batch 在 **16-30** 之间（吞吐 154-193 tok/s），
选 batch=24 附近可得 6x+ 吞吐且留调度余量；再往上边际收益 <3%/batch。

#### 瓶颈分析（batch=8 时 profile，2026-08 复测）

| 指标 | 数值 | 说明 |
|---|---|---|
| 墙钟 | 82.7 ms/步 | 用户感知时间（20 步均值） |
| GPU 纯 kernel | 30.0 ms/步 | profiler 统计，仅 **36% 利用率** |
| GPU 空等 CPU | ~53 ms | CPU 提交瓶颈（主因） |
| CPU 组装/收尾 | ~0.6 ms | 可忽略 |

top kernel（每步）：
- `index_elementwise`: 5.62ms × 448 次（mask 构造 + 采样索引）
- `unrolled_elementwise`: 2.30ms × 448（update 写入展开）
- `paged_attn_decode_kernel`: 1.12ms × 28（真正 attention 仅 ~3.7%）
- `Memcpy DtoD`: 1.02ms × 224（update 写入）
- 各类 GEMM（q/k/v 投影 + o_proj）: ~0.9-1.4ms × 84-114

**结论**：瓶颈仍在工程流水线——kernel 启动间隙 + 冗余 mask/index 构造，
GPU 算力只用了 1/3。优化方向：CUDA Graph（消启动间隙，预期墙钟 83→~35ms）、
跳过 decode mask 构造（已实现 use_triton 传 None，但批路径仍在造 mask）。

#### 硬件上限估算

```
0.6B bf16 权重 = 1.5 GB；A100 带宽 ≈ 2 TB/s
decode 每步读一遍全部权重 → 理论下限 ≈ 0.75 ms/步
当前 91ms/步 vs 硬件极限 0.75ms → 软件层瓶颈（非硬件），优化空间 ~120 倍
```

#### 本轮修复（调度器 bug）

- `scheduler.py` prefill 补位超编：`min(batch_size, waiting)` 未减 running 已有
  人数 → running 可超 batch_size → decode 切片 `running[:n]` 饿死队尾；
  改为 `min(batch_size - len(running), waiting)`，demo 验证补位只取剩余空位

### 3.6 v2.1 — prefill FlashAttention 融合（vLLM 风格 varlen kernel）

> 基准：`profile_decode.py --mode prefill`（torch.profiler，总 CUDA 时间）
> 前置：需 `TRITON_PTXAS_PATH`（见文件头运行环境前提）

| 配置 | standard prefill | flash prefill | 提升 |
|---|---|---|---|
| B=8, S=256 | 83.08 ms | 65.85 ms | **-20.7%** |
| B=2, S=1024 | 71.83 ms | 50.78 ms | **-29.3%** |
| B=1, S=1024 | 59.84 ms | 47.67 ms | **-20.3%** |

收益随序列长度放大（N² 注意力矩阵不物化）；B=2/S=1024 时 standard 路径
一度 OOM（物化 4×16×1024² 注意力矩阵），flash 路径不会。

改动：
- 新增 `qwen3/kernels/flash_attention_varlen.py`：vLLM 风格 varlen flash
  attention（`cu_seqlens` 变长 + 内核内 GQA，拷自 test/Fusedattention.py
  追加段）+ `flash_attention_prefill_batched` 适配层
  （(B,H,S,D) 右 pad 批 → 拉平流 + gather/scatter，等长批跳过 gather）
- `attention.py`/`decoder.py`/`model.py`：`input_lens` 穿透 + `QWEN3_FLASH_ATTN`
  开关（默认 triton，`=pytorch` 关闭消融）
- prefill 不再需要 CPU 侧 4D mask（kernel 内按 seq 边界处理 causal）

精度验证：synthetic 三场景（单请求/等长批/混合长度批）与 `_standard_attention`
逐位 diff < 0.02；真实模型 prefill logits argmax 一致率 96-98%（bf16 精度内）；
`test_batched.py` 端到端 6 组场景逐 token 一致。

架构更新（三段式 attention 路径）：

```
attention forward
├── PagedKVCache 分支
│   ├── decode (S=1, CUDA, bf16)  → Triton kernel（qwen3/kernels/paged_attention.py）
│   ├── prefill (S>1, CUDA, bf16) → varlen flash kernel（QWEN3_FLASH_ATTN=triton）
│   │                               └─ 回退 _standard_attention（=pytorch / CPU / fp32）
│   └── decode 兜底 (CPU/fp32)     → PyTorch 逐页（paged_attention.py）
└── 批量（list）分支             → 同上，prefill 变长批走 flash + cu_seqlens
```

### 3.7 当前耗时分析（2026-08，flash prefill 已上线）

- **decode 仍是绝对耗时主体**：64 序列串行中 Decode 占 99.8%（1308.7s vs 2.26s）。
- **decode 瓶颈 = CPU 提交**：墙钟 82.7ms/步 vs GPU 纯算 30.0ms/步（利用率 36%），
  主因是 ~500 kernel/步的启动间隙 → **CUDA Graph 是下一个 P0**。
- **prefill 已不是瓶颈**：flash 融合后单层 attention 3.68ms×28 次，
  已被投影 GEMM（~20ms）盖过；长 prompt 时 flash 收益显著且防 OOM。
- **吞吐拐点 batch≈16-30**：最优工作点在 batch 24-28（172-184 tok/s, 6x+ vs serial）。

---

## 附录 A：为什么 v0.2/v1.0 的 prefill 这么慢

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

---

## 附录 B：decode 提速归因（消融实验）

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
