# 引擎对比：当前实现 vs 工业级推理引擎

> 本文件记录本项目（手写 Qwen3 引擎）与工业级推理引擎（vLLM / SGLang 等）
> 的架构与性能差距，**所有差距项即未来的 TODO 工作清单**，作为优化路线图。
> 更新日期: 2026-08-17
> 图例: [ ] = 未开始（TODO） | [x] = 已完成

---

## 0. TODO 总览（未来工作清单）

### 已完成
- [x] Continuous batching 基础（Scheduler + Request + Batch，batch 扫描 1-4）
- [x] prefill 补位超编 bug 修复（scheduler.py）
- [x] 计时口径统一为真实墙钟
- [x] prefill FlashAttention 融合（vLLM 风格 varlen kernel，v2.1）
- [x] 块表 GPU 常驻 + 28 层共享（block_table_2d + seq_lens 常驻张量，v2.2）
- [x] 块表缓存 / 仅申请新页时重建（reserve_next 层循环外预留，v2.2）
- [x] KV 块清零（alloc 复用页清零，zero-initialized KV cache，v2.2）
- [x] K/V 写入合并为 Triton 单 kernel（update_kv_batch_kernel，v2.2）
- [x] batch 16/32 扫描找吞吐拐点（v2.2 后 batch 8/16/28 = 194.6/330.8/513.7 tok/s）
- [x] CUDA Graph decode（GraphRunner 图池，2 的幂 bucket + 哑行占位，v3.0）
- [x] 批路径 decode 跳过 mask/position 构造（Triton 路径本就 None）

### 工程流水线（P0，收益最大）
- [ ] decode 的 cos/sin RoPE 计算移入图（当前每步 rotary_emb，尚有优化空间）

### 调度层（P1-P2）
- [ ] chunked prefill：长 prompt 切片与 decode 混批
- [ ] 优先级调度 + 抢占（短请求优先 / 长请求让步）
- [ ] 精确内存账本 + watermark 水位管理
- [ ] prefix caching：相同 prompt 前缀块哈希共享

### Kernel 层（P2-P3）
- [ ] split-K decode kernel（页并行，长上下文 SM 利用率）
- [ ] fp8 KV cache（带宽减半）
- [ ] 算子融合（RMSNorm/RoPE 进 kernel）
- [ ] prefill 写入用 Triton（变长 + 跨页，PagedKVCache.update 仍是 PyTorch）

### 工程层（P3）
- [ ] GPU 内采样（multinomial 进 kernel，消除 CPU 往返）
- [ ] 换出 (swap)：GPU↔CPU KV 块超卖显存

---

## 1. 当前引擎架构（已实现）

```
┌─────────────────────────────────────────────────────────┐
│  调度层  Scheduler（贪心 + 防饥饿）                        │
│    waiting → prefill 批 → running → decode 批 → finished  │
│    （FIFO 队列，无混批，无优先级）                          │
├─────────────────────────────────────────────────────────┤
│  存储层  PagedKVCache（共享物理池 + GPU 常驻 2D 块表）      │
│    k_buffer: [num_blocks, num_layers, Hkv, block_size, D] │
│    block_table_2d / seq_lens 常驻 GPU，row_id 行槽位寻址   │
│    块按需申请、完成即归还（池复用，复用页清零）             │
├─────────────────────────────────────────────────────────┤
│  计算层  三段式 attention                                  │
│    prefill  → varlen FlashAttention（cu_seqlens 变长）     │
│    decode   → CUDA Graph（GraphRunner 图池，v3.0）          │
│                ├ 图内: Triton decode + Triton update        │
│                └ 图外: reserve/advance/pos/rows（驱动循环） │
│    兜底     → PyTorch 逐页版（CPU/非 bf16）                │
├─────────────────────────────────────────────────────────┤
│  基准     bench.py（真实墙钟）/ bench_batched.py（批扫描）  │
└─────────────────────────────────────────────────────────┘
```

**已具备的工业级骨架**：共享池 + 常驻块表 + 请求级句柄、状态机校验、
防饥饿调度、KV 块归还复用 + 清零、连续批处理（batch_size 可扫）、
prefill 融合 + decode 工程优化（v2.1/v2.2）+ CUDA Graph 图池（v3.0）。

---

## 2. 逐层差距对比

### 2.1 调度层

| 能力 | 当前实现 | vLLM / 工业级 | 影响 |
|---|---|---|---|
| 调度策略 | 贪心 + FIFO | 优先级队列（公平/吞吐可配）+ 抢占 | 长请求可拖死短请求（无抢占） |
| 混批 | ❌ prefill 与 decode 不同批 | ✅ chunked prefill：长 prompt 切片与 decode 混批 | 长 prefill 会卡住整个批 |
| 内存账本 | 只检查 prefill 块数 | 精确的 seq→块映射 + watermark 水位 | 极端负载可能 OOM |
| 前缀复用 | ❌ | ✅ prefix caching：相同 prompt 前缀块哈希共享 | 多轮对话/system prompt 重复计算 |
| 换出 (swap) | ❌ | ✅ GPU↔CPU KV 块换出 | 无法超卖显存 |
| 批成员 | 固定 batch_size 上限 | 动态：完成即出、到达即进 | 相同（我们也有，但无抢占） |

### 2.2 Kernel 计算层

| 能力 | 当前实现 | vLLM / 工业级 | 影响 |
|---|---|---|---|
| decode kernel | Triton，grid = B×Hkv，页循环串行 | CUDA，split-K 页并行 + warp 级优化 | 长上下文时 SM 利用率低 |
| prefill kernel | varlen FlashAttention（v2.1 融合） | FlashAttention-2/3（更优 tiling） | 已融合，仍有优化空间 |
| CUDA Graph | ✅ GraphRunner 图池，decode 步零 Python 启动（v3.0） | ✅ 整步捕获成图，replay 单次启动 | **已对齐**：decode 墙钟 ~40 → 2ms 量级（batch=8） |
| KV 精度 | bf16 | fp8 KV cache（带宽减半） | 长上下文内存压力 |
| 算子融合 | 无 | RMSNorm/RoPE 融合进 kernel | kernel 启动次数 |
| 块表寻址 | ✅ GPU 常驻 2D 表 + row_id 行号寻址（v2.2） | GPU 常驻块表 / kernel 内寻址 | 已对齐 |

### 2.3 工程层

| 能力 | 当前实现 | vLLM / 工业级 | 影响 |
|---|---|---|---|
| KV 块清零 | ✅ alloc 复用页清零（v2.2） | ✅ zero-initialized KV cache | 已对齐（数据泄漏防护） |
| 精度控制 | bf16 存储 + fp32 累积 | 同款 + fp8 支持 | 相同哲学 ✓ |
| 采样 | Python 逐 token | CUDA kernel 内采样（multinomial） | CPU-GPU 往返 |
| 后端 | 单 NVIDIA CUDA | 多后端（ROCm/CPU/FPGA） | 可移植性 |
| 模型支持 | Qwen3 专用 | 全模型族（AutoModel 架构） | 通用性 |

---

## 3. 实测性能（当前结果，2026-08，v3.0）

> 环境：单卡 A100-40GB / CUDA 11.8 / PyTorch 2.2.2 / bf16 / 真实墙钟口径（synchronize）
> 模型：Qwen3-0.6B（0.75B, 28 层, hidden=1024, Q heads=16, KV heads=8）
> 对比参考：vLLM 数据来自 test/bench.py（模型/卡数未确认），仅作量级参考

### 3.1 单请求串行（bench.py，64 序列，GPU 空闲）

| 版本 | Decode (ms/tok) | Prefill (ms) | Throughput (tok/s) | VRAM |
|---|---|---|---|---|
| v0.1 Naive cat | 24.3 | 60.0 | 40.72 | 3.90 GB (fp32) |
| v0.2 分页存储+重建 | 34.6 | 562.0 | 27.13 | 3.89 GB (fp32) |
| v1.0 Triton decode | 34.6 | 864.6 | 27.16 | 2.06 GB |
| v1.1 标准 prefill | 33.7 | 35.0 | 29.58 | 2.06 GB |
| v2.2 无图 | 31.8 | 35.3 | 31.3 | 2.06 GB |
| **v3.0 CUDA Graph** | **7.3** | **35.6** | **129.82** | **2.06 GB** |

（v3.0 起 generate() 也走图——main.py/bench.py 默认开，--no-graph 关闭；
串行 decode 32.0 → 7.3 ms/tok，4.2x；图路径单请求走批 kernel）

### 3.2 连续批处理（bench_batched.py，64 序列，GPU 空闲）

| Mode | Batch | Throughput | Decode | vs serial |
|---|---|---|---|---|
| serial | 1 | 30.7 tok/s | 32.0 ms/tok | 1.00x |
| batched (无图) | 8 | 215.8 | 4.3 ms/tok | ~7x |
| batched (无图) | 28 | 443.1 | 1.9 ms/tok | ~14x |
| batched (图) | 1 | **147.7** | **6.3 ms/tok** | **4.74x** |
| batched (图) | 8 | **688.2** | **1.1 ms/tok** | **21.95x** |
| batched (图) | 16 | **1033.7** | **0.6 ms/tok** | **33.14x** |
| batched (图) | 28 | **1500.4** | **0.4 ms/tok** | **48.10x** |

CUDA Graph 收益（同环境消融）：batch=8 无图 215.8 / 图 688.2（**3.2x**），
batch=28 无图 443.1 / 图 1500.4（**3.4x**）；CPU 提交开销（每步 ~33 次启动）归零。

### 3.3 瓶颈分析（batch=8，v3.0 图路径后 profile）

| 指标 | v2.2 无图 | v3.0 图 | 说明 |
|---|---|---|---|
| 墙钟 | 39.9 ms/步 | ~2.0 ms/tok | CUDA Graph 消除 CPU 提交 |
| GPU 纯 kernel | 16.6 ms/步 | ~0.9 ms/tok | 图内仍是同样 kernel，但零启动间隙 |
| GPU 空等 CPU | ~23.3 ms | ~0 | **CPU 提交瓶颈已消除** |

v2.2 时代 top kernel（每步）：
- `unrolled_elementwise`: 1.55ms × 169（激活/归一化）
- GEMM（q/k/v 投影 + o_proj）: 1.42ms × 84
- `paged_attn_decode_kernel`: 1.14ms × 28（真正 attention ~6.8%）
- `update_kv_batch_kernel`: 0.16ms × 28（Triton K/V 写入，v2.2 新增）

**关键结论**：块表常驻 + Triton update（v2.2）消除 `index_elementwise`（448→0 次），
墙钟 91 → 39.9ms；CUDA Graph（v3.0）再消除 CPU 提交间隙（~23.3ms 空等），
batch=8 decode 墙钟 ~40 → 2ms 量级。剩余瓶颈转向纯 GPU kernel 时间
（权重带宽 0.75ms/步理论下限，当前 batch=28 已 0.4ms/tok < 单请求带宽下限，
批并行已充分摊薄）。

### 3.4 硬件上限估算（当前硬件下的天花板）

```
0.6B bf16 权重 = 1.5 GB；A100 带宽 ≈ 2 TB/s
decode 每步读一遍全部权重 → 理论下限 ≈ 0.75 ms/步（单请求）
当前 batch=28: 0.4 ms/tok（权重分摊后低于单请求下限 → 已达带宽效率区）
下一步上限：权重分区（TP/张量并行）或多卡 → 单请求逼近 0.75ms
```

**B=1 图路径 6.3ms 的构成分解**（提升空间分析）：

| 构成 | 估计 | 说明 |
|---|---|---|
| 权重读（理论下限） | ~0.75ms | 1.5GB / 2TB/s，无法避免 |
| kernel 启动/间隙 | ~1-2ms | 28 层 × ~12 个 kernel ≈ 330 次发射，图内每次 ~1-3µs |
| GEMM 计算 | ~1-2ms | 小 GEMM 切块开销大，cuBLAS 吃不满 |
| norm/rope/attention 杂项 | ~2-3ms | q/k/v 三独立投影、逐层中间量读写 |

→ 距 0.75ms 下限还有 ~8x 空间，按性价比：**kernel 融合（P0）→ INT8 权重（P1）→ TP（P2）**；
量化/TP 对满并发场景收益反而更大（权重读被 batch 摊薄后，剩余是计算/启动开销）。
```

### 3.5 已修复的问题

- `scheduler.py` prefill 补位超编 bug：`min(batch_size, waiting)` 未减 running 已有
  人数 → running 可超 batch_size → decode 切片饿死队尾；改为 `min(batch_size - len(running), waiting)`
- weights.py / norm.py dtype bug（v1.0 修复，见 PERFORMANCE.md）
- 计时口径：统一真实墙钟（synchronize），废弃 CPU 提交时间
- `reserve_next` 页分配 bug：旧条件 `seq_len%16==0` 在页已存在时仍重复分配（v2.2）
- 块表每层 Python 重建 + HtoD 拷贝（v2.2 改为 GPU 常驻 2D 表 + row_id 行号寻址）

---

## 4. 差距根因总结

```
算法层差距（较小）:
  - prefill 无 FlashAttention（已融合 v2.1）
  - decode kernel 无 split-K / warp 优化
  - 无 chunked prefill / prefix caching

工程层差距（v3.0 后已基本消除）:
  - ✅ CUDA Graph：decode 步整图捕获，replay 单次启动（v3.0）
  - ✅ decode 批路径 mask/position 构造跳过（Triton 路径本就 None）
  - prefill 的 K/V 写入仍是 PyTorch（decode 已 Triton，prefill 未覆盖）
```

---

## 5. 优化路线图（按性价比排序 = 未来执行顺序）

> 与第 0 节 TODO 清单一一对应，每项完成一个就在两处同时勾选。
> 建议顺序：先 P0 工程流水线（GPU 利用率 42% → 80%+），再 P1 调度，
> 最后 P2/P3 kernel 与高级特性。

| 优先级 | 优化 | 预期收益 | 难度 | 状态 |
|---|---|---|---|---|
| P0 | CUDA Graph 捕获 decode 步 | 39.9 → 2ms/步（~20 倍，已落地） | 高 | [x] |
| P0 | decode 跳过 mask/position 构造 | 砍掉无用 kernel | 低 | [x] |
| P0 | decode 的 cos/sin 移入图 | 省每步 rotary_emb | 低 | [ ] |
| P0 | QKV/MLP 投影融合（3→1 kernel） | B=1 decode 6.3 → ~3-4ms | 中 | [ ] |
| P1 | prefill K/V 写入 Triton（变长+跨页） | 消除 PyTorch scatter | 中 | [ ] |
| P1 | 权重 INT8 量化 | 带宽需求减半（0.75 → 0.4ms 下限） | 高 | [ ] |
| P2 | split-K decode kernel | 长上下文 decode 提速 | 中 | [ ] |
| P2 | chunked prefill 混批 | 长 prompt 不卡批 | 高 | [ ] |
| P2 | 优先级调度 + 抢占 | 短请求不饿死 | 中 | [ ] |
| P2 | TP 张量并行 | 权重分卡带宽×N；B=1 通信占比高，大 batch 收益大 | 高 | [ ] |
| P3 | prefix caching | 多轮对话省 prefill | 高 | [ ] |
| P3 | fp8 KV cache | 带宽减半 | 中 | [ ] |
| P3 | 采样进 GPU kernel | 消除 CPU 往返 | 中 | [ ] |
| P3 | 换出 (swap) | 显存超卖 | 高 | [ ] |

（已完成项见第 0 节：块表常驻 / 块表缓存 / KV 清零 / 吞吐拐点扫描 /
prefill FlashAttention / Triton K/V 写入 / CUDA Graph——v2.1/v2.2/v3.0
均已落地。）

---

## 6. 一句话定位

**当前引擎 = 结构完整的 continuous batching 教学引擎**：调度器、共享池、
GPU 常驻块表、Triton kernel（decode + prefill flash + K/V 写入）、批处理
全部到位且功能正确；v2.1/v2.2 已把 prefill 融合、块表常驻、KV 清零落地，
decode 墙钟从 91ms 降到 39.9ms。剩余差距集中在
**CUDA Graph（CPU 提交瓶颈）** 与**高级调度（混批 / 前缀复用 / 抢占）**。
GPU 算力用到约 42%，优化空间明确——**第 0 节 TODO 清单就是接下来的工作路线**。
