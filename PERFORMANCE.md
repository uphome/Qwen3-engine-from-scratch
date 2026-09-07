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
- decode 步的 CUDA Graph 由 `QWEN3_CUDA_GRAPH` 控制（bench_batched 默认开，
  `0` 关闭；`--no-graph` 同效）：
  - 开 = decode 步走 GraphRunner 图池（replay 单次启动，batch 任意 < 桶）
  - 关 = eager forward_decode（Triton decode + update，v2.2 路径）
  - prefill 步永远 eager，不受此开关影响。

---

## 1. 版本命名（语义化版本号）

主版本号表示**技术代际**：

```
v0.1  NaiveKVCache 基线（无分页）
v0.2  PagedKVCache（存储层分页：共享池 + 块表，计算仍标准 attention）
v1.0  PagedAttention（计算层分页：Triton decode kernel + dtype 修复）
v1.1  优化版（prefill 标准 attention + 向量化 update + 计时修正）
v2.0  连续批处理（Scheduler 调度，batch 扫描 1-4）
v2.1  prefill FlashAttention 融合（vLLM 风格 varlen kernel）
v2.2  decode 批路径工程优化（GPU 常驻块表 + Triton K/V 写入）
v3.0  CUDA Graph decode（GraphRunner 图池）
v3.1  RoPE 预计算表 + CUDA Graph 内查表（P0-1，当前）
```

- **v0.x = 计算层未分页**（attention 仍走标准实现，分页只影响存储）
- **v1.x = PagedAttention 时代**（计算也分页，主版本跨入 1.0）
- **v2.x = Continuous Batching 时代**（多请求并发调度，吞吐量级提升）
- **v2.1 起 = prefill 计算融合**（QKᵀ/softmax/PV 折叠进单个 Triton kernel）
- **v2.2 = decode 工程流水线**（块表/seq_len GPU 常驻 + K/V 写入合并单 kernel，不引入新算法）
- **v3.0 = 执行模型代际**（decode 整步捕获成 CUDA Graph，eager → replay，
  新增执行层 GraphRunner，与调度层/存储层/计算层并列）
- **v3.1 = RoPE 图内化**（预计算完整 cos/sin 表，decode 查表进图，
  去掉每步图外 rotary_emb 与 cos/sin copy_）

---

## 2. 汇总对比

### 2.1 单请求串行（bench.py，64 序列，GPU 空闲）

| 版本 | commit | 改动 | Throughput (tok/s) | Decode (ms/tok) | Prefill (ms) | VRAM (GB) |
|------|--------|------|--------------------|-----------------|-------------|-----------|
| v0.1 | 4a20a44 | NaiveKVCache (torch.cat) | **40.72** | **24.3** | 60.0 | 3.90 |
| v0.2 | 9bfbba3 | PagedKVCache + 重建连续 K/V | 27.13 | 34.6 | 562.0 | 3.89 |
| v1.0 | 3ec469b | Triton decode kernel + dtype 修复 | 27.16 | 34.6 | 864.6 | 2.06 |
| v1.1 | 当前 | Triton decode + 标准 prefill | 29.58 | 33.7 | **35.0** | 2.06 |
| v2.1 | f3c2d40 | + flash prefill 融合 kernel | 29.27 | 34.1 | **35.3** | 2.06 |
| v2.2 | 4c4aa2e | + GPU 常驻块表 + Triton K/V 写入 | 31.3 | 31.8 | **35.3** | 2.06 |
| v3.0 | c1f94d5 | + CUDA Graph decode（串行也走图） | **129.82** | **7.3** | 35.6 | 2.06 |

> v0.1/v0.2/v1.0 为历史 commit 检出 worktree、仅移植 synchronize 计时修复后
> 同环境重跑；v1.1 为当时版本实测；v2.1/v2.2 为本次（2026-08）实测。
>
> **注意**：① v3.0 起 generate() 也走 CUDA Graph（main.py/bench.py 默认开，
> `--no-graph` 或 QWEN3_CUDA_GRAPH=0 关闭）——串行 decode 31.1 → 7.3 ms/tok，
> 吞吐 31.02 → 129.82 tok/s（4.2x）；② 图关闭时 v3.0 串行与 v2.2 持平
> （31.02/32.2，实测），说明收益全部来自 CPU 提交归零；③ v2.1/v2.2 的
> 单请求串行与 v1.1 基本持平（~29-31 vs 29.58 tok/s）——prefill 融合与
> decode 工程优化的收益原在**大 batch / 并发**，v3.0 让 B=1 也吃满
> （CPU 提交占比最高，图恰好归零）。

### 2.2 连续批处理（bench_batched.py，GPU 空闲）

v3.0 实测（64 序列，input [32,128], output [16,64], greedy，decode 走 CUDA Graph）：

| Mode | Batch | Throughput | Decode | vs serial |
|---|---|---|---|---|
| serial | 1 | 30.7 tok/s | 32.0 ms/tok | 1.00x |
| batched (图) | 1 | **147.7** | **6.3 ms/tok** | **4.74x** |
| batched (图) | 8 | **688.2** | **1.1 ms/tok** | **21.95x** |
| batched (图) | 16 | **1033.7** | **0.6 ms/tok** | **33.14x** |
| batched (图) | 28 | **1500.4** | **0.4 ms/tok** | **48.10x** |

> 同环境消融（--no-graph）：batch=8 无图 215.8 / 图 688.2（**3.2x**）；
> batch=28 无图 443.1 / 图 1500.4（**3.4x**）。v2.2 无图历史数据
> batch 8/16/28 = 194.6/330.8/513.7（量级一致）。完整扫描见 3.9。

### 2.3 关键发现

1. **单请求串行下 v0.1（naive cat）最快**（24.3ms vs 33.7ms）：分页/Triton 在 B=1 时
   反而慢 ~10ms——Triton grid = B×Hkv = 8 个 program，SM 利用率 <8%；naive 大 matmul 吃满 cuBLAS。
2. **分页/Triton 的价值在并发**：开销靠 continuous batching 摊薄——这是做并发调度的动机。
3. **prefill 优化收益最大**：v1.0 逐页 864.6ms → v1.1 标准 attention 35.0ms（-96%）。
4. **bf16 权重收益 = VRAM 减半**（3.90→2.06 GB），对速度无贡献（v1.0 vs v0.2 decode 相同）。
5. **吞吐拐点被 v2.2 改写**：块表常驻后 batch 8/16/28 达 194.6/330.8/513.7 tok/s，
   吞吐仍随 batch 增长（CPU 组装不再是瓶颈），拐点推后——完整扫描见 3.8。
6. **v3.0 消除 CPU 提交**：CUDA Graph 把 decode 步整图捕获，每步 ~33 次 kernel
   启动 → 1 次 replay。batch=8 墙钟 39.9 → ~2ms/步（~20x），batch=1 也吃满
   （31.1 → 6.3 ms/tok，5x）——图收益与 batch 无关，B=1 时占比更高。
7. **串行引擎同样受益**：generate() 集成图后（main.py/bench.py 默认开），
   单请求 decode 32.0 → 7.3 ms/tok，吞吐 31.02 → 129.82 tok/s（4.2x）——
   与 batch=1 图路径量级一致（6.3ms 略优于串行循环的 7.3ms）。
8. **新瓶颈 = 纯 GPU kernel 时间**：batch=28 时 0.4 ms/tok 已低于单请求带宽下限
   （0.75ms，权重 1.5GB / 2TB/s），批并行充分摊薄；继续提速需 TP/多卡或权重级优化。

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

### 3.5 v2.0 — 连续批处理

> 引擎：Scheduler + Request + Batch（qwen3/scheduler.py 等）
> 基准：`bench_batched.py`（64 序列，batch 1-32 扫描，GPU 空闲，真实墙钟）
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
当前 82.7ms/步 vs 硬件极限 0.75ms → 软件层瓶颈（非硬件），优化空间 ~110 倍
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

### 3.8 v2.2 — decode 批路径工程优化（GPU 常驻块表 + Triton K/V 写入）

> 基准：`profile_decode.py --mode decode`（batch=8）+ `bench_batched.py`（64 序列）
> 提交：db8407b（块表常驻）+ 4c4aa2e（Triton update）
> 定位：**不引入新算法**，纯工程流水线优化——消除批路径每层重复的块表组装杂活。

#### 改动（两个提交）

**db8407b — GPU-resident block table**
- `KVCachePool` 新增 GPU 常驻 `block_table_2d (max_requests, max_pages)` + `seq_lens`
  张量，请求按行槽位（row_id）管理（`alloc_row`/`free_row`）
- `reserve_next`：decode 步在层循环外预留页，28 层内块表稳定；
  修旧 bug（`seq_len%16==0` 在页已存在时仍重复分配）
- `update_kv_batch`：B 请求 K/V 写入合并为一次 advanced indexing（纯 GPU）
- `alloc` 复用已释放页清零（zero-initialized KV cache）

**4c4aa2e — Triton update_kv_batch**
- 新增 `update_kv_batch_kernel`（grid = B×num_kv_heads，每个 program 一次写 K+V）
- 替代 PyTorch advanced indexing（每层 2 次 `index_elementwise` + 中间地址计算 → 1 次 launch）

#### 实测对比（decode，batch=8，GPU 纯算）

| 里程碑 | decode 总 CUDA 时间 | `index_elementwise` 调用 |
|---|---|---|
| v2.1（flash prefill 后） | 29.7-30.5 ms | 448 + 224 次 |
| + 块表常驻（db8407b） | 19.1 ms | 56×2 次 |
| + Triton update（4c4aa2e） | **16.6 ms** | **0 次** |

**decode 墙钟 / 利用率**（batch=8，30 步均值）：

| 指标 | v2.1 | v2.2 | 变化 |
|---|---|---|---|
| 墙钟 | 82.7 ms/步 | **39.9 ms/步** | **-52%** |
| GPU 纯算 | 30.0 ms/步 | 16.6 ms/步 | -45% |
| GPU 利用率 | 36% | **42%** | +6pp |
| GPU 空等 CPU | ~53 ms | 23.3 ms | -56% |

#### 吞吐（64 序列，v2.1 vs v2.2）

| batch | v2.1 | v2.2 | 提升 |
|---|---|---|---|
| 8 | 115.9 tok/s | **194.6** | +68% |
| 16 | 153.5 | **330.8** | +116% |
| 28 | 184.0 | **513.7** | +179% |

#### 收益归因

- **块表组装消失**：批路径每层的 `torch.zeros + B 次行拷贝 + seq_len HtoD` 全部删除
  （`index_elementwise` 448 次 → 0）
- **K/V 写入合并**：每层 2 次 PyTorch scatter → 1 次 Triton launch（56×2 → 0 次
  `index_elementwise`）
- **CPU 提交减半**：墙钟 82.7→39.9ms，CPU 空等 53→23.3ms——工程杂活不再是
  decode 的主要开销，kernel 启动间隙成为剩余瓶颈 → CUDA Graph 收益更大

#### 正确性验证

- 块表常驻 + Triton update 均与 PyTorch 逐位一致；GPU batch vs 串行逐 token 16/16 一致
- `test_batched.py` 6 组场景全过

#### 剩余瓶颈（下一 P0）

decode 墙钟 39.9ms 里仍有 23.3ms 是 CPU 空等 GPU（kernel 启动间隙，~37 事件/步）。
块表常驻削掉了组装杂活，但每层仍 ~37 次 kernel 启动。**CUDA Graph 捕获整步
是下一个量级优化**（预期墙钟 39.9 → ~20ms，利用率 42% → 80%+）。

---

### 3.9 v3.0 — CUDA Graph decode（GraphRunner 图池）

> 基准：`bench_batched.py`（64 序列，input [32,128], output [16,64], greedy）
> 提交：c1f94d5（GraphRunner + driver 集成）
> 定位：**CPU 提交归零**——decode 整步捕获成 CUDA Graph，replay 单次启动。

#### 改动

**`qwen3/graph_runner.py`（新增，GraphRunner 图池）**
- 多 bucket 捕获（2 的幂：1,2,4,8,16,32，vLLM 同款），共享内存池
  （`graph_pool_handle`）捕获，避免每图独立池内存翻倍
- 运行期动态内容（input_ids / positions / cos / sin / rows）copy_ 进固定
  张量再 replay——图 replay 不传参数，指针全烧进图
- 空槽（k < bucket）填"哑行"row_id：pool 预分配 1 个物理页的行
  （seq_len=0，kernel 循环 0 页不读块表；物理页防 update 写 -1 越界）
- 捕获前置（两个易踩坑）：
  ① 占位句柄 seq_len=1 走 decode 分支——attention.py 以
     `caches[0].seq_len>0` 判断，否则走 prefill flash（其 `cu_seqlens.cpu()`
     是捕获禁区 → "operation not permitted when stream is capturing"）
  ② s_rows 必须填真实 row_id（全 0 会指向未分配行，block_table 全 -1
     → kernel 读 phys=-1 非法访问）

**`bench_batched.py`（driver 集成）**
- decode 步改走 `runner.replay`；reserve_next / advance_seq_len 从模型
  内部移到驱动循环（forward_decode 契约：图内零 Python 状态操作）
- 开关：`QWEN3_CUDA_GRAPH`（默认 1）/ `--no-graph`；prefill 保持 eager
- 池块数计入占位句柄 + 哑行（各 1 块），归还检查同步扣除

#### 实测（64 序列，decode 走图，batch 1-28 全扫描，2026-08-17 复测）

| batch | 吞吐 (tok/s) | Decode (ms/tok) | vs serial |
|---|---|---|---|
| 1 | 147.7 | 6.3 | 4.74x |
| 8 | 688.2 | 1.1 | 21.95x |
| 16 | 1033.7 | 0.6 | 33.14x |
| 28 | **1500.4** | **0.4** | **48.10x** |

同环境消融（--no-graph）：batch=8 无图 215.8 / 图 688.2（**3.2x**）；
batch=28 无图 443.1 / 图 1500.4（**3.4x**）。

#### 收益归因（vs v2.2 无图）

| batch | 无图 | 图 | 提升 |
|---|---|---|---|
| 8 | 215.8 | **688.2** | +219% |
| 16 | ~330（历史） | **1033.7** | +213% |
| 28 | 443.1 | **1500.4** | +239% |

- **CPU 提交归零**：每步 ~33 次 kernel 启动 + 启动间隙（v2.2 墙钟 39.9ms 中
  23.3ms 空等）→ 1 次 replay。batch=8 墙钟 ~40 → ~2ms（~20x）
- **B=1 也吃满**：单请求 31.1 → 6.3 ms/tok（5x）——B=1 时 CPU 提交占比最高，
  图收益与 batch 无关（v2.1/v2.2 的优化在 B=1 基本无感，v3.0 反超 naive cat
  v0.1 的 24.3ms）
- **剩余 = 纯 GPU kernel 时间**：batch=28 达 0.4 ms/tok，低于单请求权重带宽下限
  （0.75ms = 1.5GB / 2TB/s）——批并行已充分摊薄权重读，下一步是 TP/多卡

#### 正确性验证

- GraphRunner vs eager `forward_decode`：多 bucket（2/4/8）+ 空槽 + 逐 token
  argmax 完全一致（25 步 × 5 组 k）
- bench_batched 图路径与 --no-graph 调度 Steps 一致、无崩溃
- 验证过程踩过的坑（测试脚本层面）：decode 步前必须 `reserve_next` +
  GPU seq_lens 与 Python seq_len 同步（违反则 write_pos 越界到 -1 页）

#### 剩余瓶颈（下一 P0）

- ~~decode 每步仍在图外算 cos/sin（`rotary_emb`，一次 CPU 启动）~~ → 已由 v3.1 移入图内
- prefill 的 K/V 写入仍是 PyTorch advanced indexing（decode 已 Triton）
- chunked prefill 混批 / 优先级调度（P1-P2）

> 已完：generate()/main.py/bench.py 单请求路径集成图（decode 32 → 7.3 ms/tok，
> 4.2x）；图路径单请求也走批 kernel（合并 K/V 写入一并吃到）。

**B=1 提升空间**（图路径 6.3ms → 0.75ms 权重带宽下限之间还有 ~8x）：

| 构成 | 估计 | 说明 |
|---|---|---|
| 权重读（理论下限） | ~0.75ms | 1.5GB / 2TB/s，无法避免 |
| kernel 启动/间隙 | ~1-2ms | ~330 次发射，图内每次 ~1-3µs |
| GEMM 计算 | ~1-2ms | 小 GEMM 切块开销大，cuBLAS 吃不满 |
| norm/rope/attention 杂项 | ~2-3ms | q/k/v 三独立投影、逐层中间量读写 |

按性价比：**P0 kernel 融合**（q/k/v proj 3→1、MLP up+gate、norm+linear，预估
6.3 → 3-4ms）→ **P1 权重 INT8**（带宽减半，下限 0.75 → 0.4ms）→ **P2 TP 多卡**
（权重分卡带宽×N；注意量化/TP 对满并发 batch=28 收益反而更大——权重读被
batch 摊薄后剩余是计算/启动开销，融合对所有 batch 都受益）。


### 3.10 v3.1 — RoPE 预计算表移入 CUDA Graph（P0-1）

> 基准：`bench_batched.py`（64 序列，input [32,256], output [32,128], greedy，decode 走 CUDA Graph）
> 对比：旧图 `4f46e76` vs 新图 `5187b2a`，两者均为 **CUDA Graph decode**，不是 graph vs eager
> 定位：将 decode 每步图外 `rotary_emb()` 计算改为预计算 `cos/sin` 表 + 图内查表，
> 去掉每步的 RoPE 三角函数计算与 `cos/sin` `copy_`。

#### 改动

- `qwen3/rope.py`：预计算完整 `cos_table` / `sin_table`，`forward()` 改为纯 GPU gather
- `qwen3/config.py`：新增 `max_position_embeddings`，用于决定 RoPE 表长度
- `qwen3/model.py`：`forward_decode()` 改为接收 `position_ids`，内部通过 RoPE 表查表
- `qwen3/graph_runner.py`：移除 `s_cos` / `s_sin`，`replay(input_ids, positions, rows)`
- `generate.py` / `bench_batched.py` / `server.py`：调用方不再传 `cos/sin`

#### 实测（新旧图均为 CUDA Graph decode）

| Batch | 旧图 Throughput (tok/s) | 新图 Throughput (tok/s) | Throughput 提升 | 旧图 Decode (ms/tok) | 新图 Decode (ms/tok) | Decode 提升 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 142.3 | 150.1 | 1.055x | 6.44 | 6.19 | 1.039x |
| 2 | 232.9 | 246.8 | 1.060x | 3.71 | 3.59 | 1.033x |
| 3 | 332.4 | 331.4 | 0.997x | 2.53 | 2.49 | 1.017x |
| 4 | 421.4 | 423.0 | 1.004x | 1.93 | 1.89 | 1.020x |
| 5 | 493.8 | 501.0 | 1.015x | 1.61 | 1.58 | 1.015x |
| 6 | 531.4 | 568.1 | 1.069x | 1.37 | 1.35 | 1.016x |
| 7 | 615.5 | 639.9 | 1.040x | 1.19 | 1.18 | 1.013x |
| 8 | 683.2 | 689.6 | 1.009x | 1.07 | 1.05 | 1.013x |

> Decode ms/tok 按 `decode_time / 实际输出 tokens` 重新计算，比表格中的
> 四舍五入值更精确。新图 capture/instantiation time 约 `10.74s`，属一次性初始化成本；
> 旧图当时未单独记录 capture time。

#### 结论

- 小 batch（1-2）吞吐提升约 **5-6%**，decode 每 token 提升约 **3-4%**
- 中高 batch 吞吐提升约 **1-7%**，decode 每 token 提升约 **1-2%**
- 高 batch 下 CPU 启动开销已被并行摊薄，因此 RoPE 图内化收益变小
- 该优化主要减少 decode 路径的 CPU 侧 RoPE 计算与拷贝，属于图结构内的工程优化


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
