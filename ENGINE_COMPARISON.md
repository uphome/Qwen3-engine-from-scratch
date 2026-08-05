# 引擎对比：当前实现 vs 工业级推理引擎

> 本文件记录本项目（手写 Qwen3 引擎）与工业级推理引擎（vLLM / SGLang 等）
> 的架构与性能差距，**所有差距项即未来的 TODO 工作清单**，作为优化路线图。
> 更新日期: 2026-08-05
> 图例: [ ] = 未开始（TODO） | [x] = 已完成

---

## 0. TODO 总览（未来工作清单）

### 已完成
- [x] Continuous batching 基础（Scheduler + Request + Batch，batch 扫描 1-4）
- [x] prefill 补位超编 bug 修复（scheduler.py）
- [x] 计时口径统一为真实墙钟

### 工程流水线（P0，收益最大）
- [ ] CUDA Graph：decode 步整图捕获，消除每步几百次 kernel 启动（预期 91→~30ms/步）
- [ ] 块表缓存：仅申请新页时重建，28 层共享（消除 ~700 次 HtoD 拷贝/步）
- [ ] 块表 GPU 常驻：update 用 index_put_ 同步维护，消除 CPU→GPU 往返
- [ ] decode 跳过无用 mask/position 构造（Triton 路径不用 mask）
- [ ] KV 块清零（zero-initialized KV cache，上线安全必备）
- [ ] P0 完成后：batch 16/32 扫描找吞吐拐点（验证硬件上限）

### 调度层（P1-P2）
- [ ] chunked prefill：长 prompt 切片与 decode 混批
- [ ] 优先级调度 + 抢占（短请求优先 / 长请求让步）
- [ ] 精确内存账本 + watermark 水位管理
- [ ] prefix caching：相同 prompt 前缀块哈希共享

### Kernel 层（P2-P3）
- [ ] split-K decode kernel（页并行，长上下文 SM 利用率）
- [ ] prefill 用 FlashAttention（tiling + IO 优化）
- [ ] fp8 KV cache（带宽减半）
- [ ] 算子融合（RMSNorm/RoPE 进 kernel）

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
│  存储层  PagedKVCache（共享物理池 + 请求级块表）            │
│    k_buffer: [num_blocks, num_layers, Hkv, block_size, D] │
│    块按需申请、完成即归还（池复用）                         │
├─────────────────────────────────────────────────────────┤
│  计算层  三段式 attention                                  │
│    prefill  → 标准 attention（连续 K/V，vLLM 同款思路）     │
│    decode   → Triton kernel（GQA + online-softmax）       │
│    兜底     → PyTorch 逐页版（CPU/非 bf16）                │
├─────────────────────────────────────────────────────────┤
│  基准     bench.py（真实墙钟）/ bench_batched.py（批扫描）  │
└─────────────────────────────────────────────────────────┘
```

**已具备的工业级骨架**：共享池 + 块表 + 请求级句柄、状态机校验、
防饥饿调度、KV 块归还复用、连续批处理（batch_size 可扫）。

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
| prefill kernel | cuBLAS 标准 matmul | FlashAttention（tiling + IO 优化） | prefill 慢数倍 |
| CUDA Graph | ❌ 每步 28 层 × 多 kernel 启动 | ✅ 整步捕获成图，replay 单次启动 | **我们 91ms/步 vs GPU 纯算 29ms** |
| KV 精度 | bf16 | fp8 KV cache（带宽减半） | 长上下文内存压力 |
| 算子融合 | 无 | RMSNorm/RoPE 融合进 kernel | kernel 启动次数 |
| 块表寻址 | 每层 Python 重建 + HtoD 拷贝 | GPU 常驻块表 / kernel 内寻址 | **我们每步 ~700 次拷贝** |

### 2.3 工程层

| 能力 | 当前实现 | vLLM / 工业级 | 影响 |
|---|---|---|---|
| KV 块清零 | ❌ torch.empty 残留上一请求数据 | ✅ zero-initialized KV cache | 数据泄漏风险（上线前必修） |
| 精度控制 | bf16 存储 + fp32 累积 | 同款 + fp8 支持 | 相同哲学 ✓ |
| 采样 | Python 逐 token | CUDA kernel 内采样（multinomial） | CPU-GPU 往返 |
| 后端 | 单 NVIDIA CUDA | 多后端（ROCm/CPU/FPGA） | 可移植性 |
| 模型支持 | Qwen3 专用 | 全模型族（AutoModel 架构） | 通用性 |

---

## 3. 实测性能（当前结果，2026-08-05）

> 环境：单卡 A100-40GB / CUDA 11.8 / PyTorch 2.2.2 / bf16 / 真实墙钟口径（synchronize）
> 模型：Qwen3-0.6B（0.75B, 28 层, hidden=1024, Q heads=16, KV heads=8）
> 对比参考：vLLM 数据来自 test/bench.py（模型/卡数未确认），仅作量级参考

### 3.1 单请求串行（bench.py，64 序列，GPU 空闲）

| 版本 | Decode (ms/tok) | Prefill (ms) | Throughput (tok/s) | VRAM |
|---|---|---|---|---|
| v0.1 Naive cat | 24.3 | 60.0 | 40.72 | 3.90 GB (fp32) |
| v0.2 分页存储+重建 | 34.6 | 562.0 | 27.13 | 3.89 GB (fp32) |
| v1.0 Triton decode | 34.6 | 864.6 | 27.16 | 2.06 GB |
| **v1.1 当前** | **33.7** | **35.0** | **29.58** | **2.06 GB** |

### 3.2 连续批处理（bench_batched.py，16 序列，batch 扫描，GPU 空闲）

| Mode | Batch | Throughput | Decode | Lat p50 | vs serial |
|---|---|---|---|---|---|
| serial | 1 | 29.8 tok/s | 32.6 ms/tok | 6.42s | 1.00x |
| batched | 1 | 29.1 | 33.5 | 6.67s | 0.98x |
| batched | 2 | 46.3 | 20.8 | 4.41s | **1.55x** |
| batched | 3 | 62.5 | 15.3 | 3.62s | **2.10x** |
| batched | 4 | 74.8 | 12.8 | 3.27s | **2.51x** |

连续批处理收益确认：batch=4 时吞吐 2.51x、decode 2.55x。

### 3.3 瓶颈分析（batch=8 时 profile）

| 指标 | 数值 | 说明 |
|---|---|---|
| 墙钟 | 91.3 ms/步 | 用户感知时间 |
| GPU 纯 kernel | 29.4 ms/步 | profiler 统计，仅 32% 利用率 |
| GPU 空等 CPU | ~62 ms | **CPU 提交瓶颈**（主因） |
| CPU 组装/收尾 | ~0.6 ms | batch.build/on_step_done，可忽略 |

top kernel（每步）：
- `paged_attn_decode_kernel`: 1.06ms × 28 次（真正 attention 只有 ~3.6%）
- Memcpy HtoD: 0.96ms × 477 次（块表 torch.tensor 拷贝！）
- Memcpy DtoD: 1.09ms × 224 次（update 写入）
- index_elementwise: 5.61ms × 448 次（mask 构造 + 采样索引）

**关键结论**：瓶颈不在算法而在**工程流水线**——GPU 纯算 29ms/步，
墙钟 91ms/步，62ms 是 GPU 空等 CPU（每层重复的块表拷贝 + mask 构造 + kernel 启动）。
CPU 提交开销本身（墙钟-kernel）只有 0.06ms，是**kernel 间间隙**在等 CPU。

### 3.4 硬件上限估算（当前硬件下的天花板）

```
0.6B bf16 权重 = 1.5 GB；A100 带宽 ≈ 2 TB/s
decode 每步读一遍全部权重 → 理论下限 ≈ 0.75 ms/步
当前 91ms/步 vs 硬件极限 0.75ms → 差 ~120 倍（软件层瓶颈，非硬件）
优化后（CPU 提交追上 GPU）: batch=8 → ~30ms/步 → ~260 tok/s 量级
```

### 3.5 已修复的问题

- `scheduler.py` prefill 补位超编 bug：`min(batch_size, waiting)` 未减 running 已有
  人数 → running 可超 batch_size → decode 切片饿死队尾；改为 `min(batch_size - len(running), waiting)`
- weights.py / norm.py dtype bug（v1.0 修复，见 PERFORMANCE.md）
- 计时口径：统一真实墙钟（synchronize），废弃 CPU 提交时间

---

## 4. 差距根因总结

```
算法层差距（较小）:
  - prefill 无 FlashAttention
  - decode kernel 无 split-K / warp 优化
  - 无 chunked prefill / prefix caching

工程层差距（主因，占性能损失大头）:
  - 无 CUDA Graph：每步几百次 kernel 启动，GPU 大量空等
  - 块表每层 Python 重建 + HtoD 拷贝（~700 次/步）
  - decode 步构造无用的 mask（Triton 路径不用）
  - 无 GPU 常驻块表 / index_put_ 同步维护
```

---

## 5. 优化路线图（按性价比排序 = 未来执行顺序）

> 与第 0 节 TODO 清单一一对应，每项完成一个就在两处同时勾选。
> 建议顺序：先 P0 工程流水线（GPU 利用率 32% → 80%+），再 P1 调度，
> 最后 P2/P3 kernel 与高级特性。

| 优先级 | 优化 | 预期收益 | 难度 | 状态 |
|---|---|---|---|---|
| P0 | 块表缓存：仅申请新页时重建，28 层共享 | 砍掉大部分 HtoD 拷贝 | 低 | [ ] |
| P0 | decode 跳过 mask/position 构造 | 砍掉无用 kernel | 低 | [ ] |
| P0 | CUDA Graph 捕获 decode 步 | 91 → ~30ms/步（3 倍） | 高 | [ ] |
| P0 | KV 块清零（zero-init） | 上线安全 | 低 | [ ] |
| P0 | batch 16/32 扫描找吞吐拐点 | 验证硬件上限（优化后） | 低 | [ ] |
| P1 | 块表 GPU 常驻（index_put_ 同步） | 消除 HtoD | 中 | [ ] |
| P2 | split-K decode kernel | 长上下文 decode 提速 | 中 | [ ] |
| P2 | chunked prefill 混批 | 长 prompt 不卡批 | 高 | [ ] |
| P2 | 优先级调度 + 抢占 | 短请求不饿死 | 中 | [ ] |
| P3 | prefix caching | 多轮对话省 prefill | 高 | [ ] |
| P3 | fp8 KV cache | 带宽减半 | 中 | [ ] |
| P3 | 采样进 GPU kernel | 消除 CPU 往返 | 中 | [ ] |
| P3 | 换出 (swap) | 显存超卖 | 高 | [ ] |

---

## 6. 一句话定位

**当前引擎 = 结构完整的 continuous batching 教学引擎**：调度器、共享池、
块表、Triton kernel、批处理全部到位且功能正确；差距集中在
**工程流水线（CUDA Graph / 块表常驻 / 无用计算消除）** 与
**高级调度（混批 / 前缀复用 / 抢占）**。GPU 算力只用到约 1/3，
优化空间明确——**第 0 节 TODO 清单就是接下来的工作路线**。
