# 前缀缓存（Prefix Cache / 哈希 KV 复用）

> 目标：让 **KV cache 在不同对话/请求之间复用**——相同前缀不再重复 prefill。
> 当前状态：**阶段 1（完全命中）+ 阶段 2（部分命中）已落地**。

---

## 1. 为什么需要它

多轮对话、重复 system prompt、批量评测同一组 prompt 时，请求之间有大量
相同前缀。没有前缀缓存时，每个请求都要把整个 prompt 重新算一遍
（prefill），而 prefill 通常是首 token 延迟（TTFT）的主要来源。

```
请求 A: [system prompt 2048 tokens][user msg]  → prefill 全部 2064 tokens
请求 B: [system prompt 2048 tokens][user msg2] → 复用前 2048 tokens 的 KV，
                                                 只 prefill 后缀 16 tokens
```

---

## 2. 核心设计

### 2.1 块哈希链（block-level hash）

prompt 按 `block_size=16` 切块，哈希是**链式**的：

```python
h_0 = blake2b(parent=0,    tokens[0:16])
h_1 = blake2b(parent=h_0,  tokens[16:32])
h_2 = blake2b(parent=h_1,  tokens[32:48])
```

- 相同 token 前缀 → 哈希序列完全相同（可命中）
- 任意位置 token 不同 → 从该块开始哈希全部不同（不会误命中）
- **只对完整块计算**：不足 16 token 的尾部会与生成 token 混写，不能作为前缀缓存

实现：`qwen3/prefix_cache.py`（`hash_block` / `compute_prefix_hashes`）。

### 2.2 块池：空闲 / 缓存 / 引用计数

`KVCachePool` 的物理块有三种状态：

```
free_block_ids          空闲块（内容已清零，可分配）
_prefix_index            hash → 块（缓存块，保留 K/V 等待复用）
_cached_lru              ref=0 的缓存块（LRU 淘汰候选）
block_refs[blk]          引用计数（活跃请求持有 > 0）
```

关键规则：

| 事件 | 行为 |
|---|---|
| `alloc(n)` | 只从 free 池取（**永远不会拿到缓存块**，天然避免误清零） |
| 命中缓存 | `block_refs++`，块从 LRU 移出（使用期间不可淘汰） |
| 请求释放块（有 hash） | `ref--`；归零后**放回 LRU 保留 K/V** |
| 请求释放块（无 hash） | 清零回 free（decode 生产块） |
| 缓存超上限 | **链感知 LRU**：优先淘汰叶节点（无子块），避免切断前缀链；全链在缓存时退回 LRU 最旧 |
| 容量不足（活跃请求需要空间） | **强制淘汰 ref=0 缓存块**（链感知），再重试分配；`QWEN3_CACHE_EVICT_ON_DEMAND=0` 可关闭（退回直接拒绝） |

缓存上限：`max_cached_blocks = 75% × num_blocks`（留 25% 给活跃请求）。

### 2.3 完全命中（阶段 1）

条件：**prompt 长度块对齐**且所有完整块哈希全部命中。

```
Scheduler.add(req)
   │  match_prefix_partial(hashes)       # ref++
   ├─ 全部命中且对齐 → COW 最后一个块 → seq_len = L-1 → 直接 decode
   └─ 否则 → 部分命中 / 普通 prefill
```

**为什么 COW + seq_len = L-1**：缓存里没有保存 prompt 最后一个 token 的
logits，而第一个生成 token 由它决定。所以只复用前 L-1 个 token 的 KV，
第一次 decode 用 `prompt[L-1]`（position = L-1）重算该位置；这次会写
position L-1 的 K/V，必须写**私有块**（复制末块，约 18MB，~10µs）。

### 2.4 部分命中（阶段 2）

条件：最长前缀命中 `matched` 个完整块（`matched_tokens = 16 × n`），
且 `matched_tokens ≥ QWEN3_PREFIX_PARTIAL_MIN`（默认 1024，见 2.6）。

```
Scheduler.add(req)
   ├─ bind_prefix(matched)             # 共享前缀块（ref++）
   ├─ cache.seq_len = matched_tokens   # GPU seq_lens 同步
   ├─ req.prefill_offset = matched_tokens
   └─ 进入 waiting → prefill 批只送 input_ids[:, offset:]
```

后续链路自动衔接：

| 环节 | 改动 |
|---|---|
| `Batch.build_input_ids` | prefill 只取 `input_ids[:, prefill_offset:]`，右 pad |
| `Batch.input_lens` | 返回后缀长度（模型按它推进 seq_len） |
| `model.forward` | 显式 `mode` 判定（后缀 S=1 也不能误判 decode）；position_ids = `offset + arange(S)`；层前一次性预取历史 KV |
| `attention`（prefill） | 有历史 → `paged_attention_prefill_from_history`：q=后缀，K/V = 历史 + 后缀，因果 mask 按全局位置（`offset + i`） |
| `get_kv`（fast gather） | 每层一次高级索引 gather 历史（峰值 ~8MB/层，不做全历史副本） |

**KV 写入**：`PagedKVCache.update` 从 `seq_len = matched_tokens` 开始写新块，
共享前缀块保持只读；decode 从 L 继续写新块。

### 2.5 生命周期示例（部分命中）

```
A: prompt 2048（128 块）→ prefill → decode → finish
   → 128 块登记缓存（cached=128）

B: prompt = A 的 prompt + 16 新 token（129 块）
   → 最长前缀命中 128 块（ref++）
   → prefill 只算 16 个后缀 token（position 2048..2063）
   → decode 正常
   → finish：共享块回缓存，后缀块按 hash 规则处理
```

### 2.6 短历史回退阈值（工程取舍）

部分命中的固定成本 = 28 层权重读取 + 每层小 GEMM / attention 算子
（实测约 35ms）。历史较短时，全量 flash prefill 反而更快：

| 历史 tokens | 全量 prefill | 部分命中 | 加速 |
|---:|---:|---:|---:|
| 128 | 35.0 ms | 37.0 ms | 0.95x |
| 512 | 34.0 ms | 36.4 ms | 0.93x |
| 1024 | 35.2 ms | 37.5 ms | 0.94x |
| 2048 | 56.8 ms | 39.1 ms | **1.45x** |

（Qwen3-0.6B / A100 / 后缀 16 tokens / bf16；1024-1536 之间收益接近持平）

所以默认阈值 `QWEN3_PREFIX_PARTIAL_MIN=1024`：命中前缀低于 1024 token 时
释放命中块、回退全量 prefill，避免负收益；可通过环境变量调整（0 = 总是启用）。

---

## 3. 实测（Qwen3-0.6B，A100，CUDA Graph decode）

### 3.1 完全命中（阶段 1）

同一 256-token prompt（块对齐），贪心 16 tokens：

| 请求 | 路径 | TTFT | 输出 |
|---|---|---|---|
| A（冷启动） | prefill 256 + decode | 54.8 ms | ✓ |
| B（完全命中） | 跳过 prefill，直接 decode | **6.7 ms** | 与 A 一致 |
| | | **8.24x** | 命中 16 块 |

### 3.2 部分命中（阶段 2）

2048-token 前缀 + 16-token 后缀（模拟多轮对话），贪心 8 tokens：

| 请求 | 路径 | TTFT |
|---|---|---|
| A（冷启动） | 全量 prefill 2048 | 120 ms（含首次编译） |
| B（部分命中 2048/2064） | prefill 16 后缀 | **45.6 ms** |
| C（关闭缓存，全量 2064） | 全量 prefill | 59.5 ms |

**B vs C = 1.30x**；命中 128 块。

HTTP 端到端（1501-token prompt，/v1/completions，命中 93 块）：

| 请求 | 端到端延迟 |
|---|---|
| 第一次（冷） | 218 ms |
| 第二次（部分命中 1488/1501） | **140 ms（1.56x）** |

> 输出一致性说明：B 与 C 前 5 个 token 相同，第 5 个翻转。原因是
> bf16 下不同 attention 实现（PyTorch 历史拼接 vs flash 全量）的
> 数值差异累积，**不是逻辑错误**；CPU fp32 精确测试（全量 vs 部分命中）
> max diff < 1e-4、argmax 完全一致（见 `tests/model/test_prefix_prefill.py`）。

---

## 4. 可观测性

`GET /metrics`（`engine_prefix_cache_*`）：

| 指标 | 类型 | 含义 |
|---|---|---|
| `engine_prefix_cache_hits_total` | counter | 实际复用的命中请求数 |
| `engine_prefix_cache_hit_blocks_total` | counter | 复用块数 |
| `engine_prefix_cache_cached_blocks` | gauge | 当前可复用缓存块（ref=0） |
| `engine_prefix_cache_capacity_blocks` | gauge | 缓存上限（LRU 阈值） |
| `engine_prefix_cache_evictions_total` | counter | 容量压力下强制淘汰的缓存块数 |

---

## 5. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/unit/test_prefix_cache.py`（11） | 哈希链、完全命中/COW、部分命中/后缀组装、ref 共享、LRU、父链一致性、池闭环 |
| `tests/model/test_prefix_prefill.py`（4） | 部分命中 vs 全量 prefill logits（CPU fp32，含 1-token 后缀） |
| `tests/kernels/test_flash_prefill.py`（2, gpu） | flash 普通 varlen / 历史路径（q/kv 长度分离）数值正确性 |

---

## 6. 边界与限制

| 限制 | 说明 |
|---|---|
| 部分命中的性能依赖长历史 | 阈值默认 1024；短历史自动回退全量（避免负收益） |
| 历史读取逐层进行 | 不做全历史副本（预取会翻倍显存）；峰值 ~8MB/层 |
| LRU 淘汰 | 已实现链感知（优先淘汰链尾）；上限 75% |
| 容量记账 | 命中块从新块需求中扣除；完全命中时 COW 私有块计为新块 |
| 无跨进程持久化 | 缓存随进程生命周期 |
| 强制淘汰会牺牲缓存命中率 | 高负载时可用性优先（请求成功 > 缓存命中）；`QWEN3_CACHE_EVICT_ON_DEMAND=0` 可切回"拒绝"策略 |
| `generate()` 单请求路径 | 不走 Scheduler，不查缓存（请求结束仍登记块，受 LRU 上限约束） |
| 数值 | 前缀复用不保证 bit 级输出一致（bf16 不同 attention 实现） |

---

## 7. 阶段 3（进度）

1. ~~**链感知 LRU**~~：已实现（`_children` / `_block_parents`，优先淘汰叶节点）
2. **Triton 历史 kernel 接入**：`flash_attention_prefill_with_history` 已实现
   并通过数值测试；当前 q 很小（16）时 cuBLAS 的 PyTorch 路径更快，
   大后缀 / 大模型场景可切换
3. **部分命中指标**：`partial_hits` / 回退计数，便于观察阈值效果
4. **chunked prefill 协同**：长后缀切片与 decode 混批
5. **跨进程持久化**：磁盘 KV（多级缓存）

---

## 8. 与 vLLM 的对应

| 能力 | 本项目 | vLLM |
|---|---|---|
| 块哈希 | 链式 blake2b（16 字节） | 链式 hash |
| 共享粒度 | 16-token 块（对齐） | block（默认 16） |
| 引用计数 + LRU | ✅ | ✅ |
| 完全命中复用 | ✅ | ✅ |
| 部分命中（前缀 + 后缀） | ✅（PyTorch attention） | ✅（flash 内建 prefix） |
| COW 末块 | ✅（完全命中） | ✅ |
| 链感知淘汰 | ⏳ | 部分（引用计数 + LRU） |
| KV 持久化 | ❌ | ✅（多级缓存） |

---

## 9. 关键文件

| 文件 | 内容 |
|---|---|
| `qwen3/prefix_cache.py` | 块哈希链 |
| `qwen3/PagedKVcache.py` | 块池状态机（ref / 缓存 / LRU / COW / `get_kv_all`） |
| `qwen3/scheduler.py` | 最长前缀匹配、完全/部分命中、阈值回退 |
| `qwen3/request.py` | `prefix_hashes` / `prefill_offset` / `prefill_last_token` |
| `qwen3/batch.py` | prefill 后缀组装 |
| `qwen3/model.py` | `mode` 判定、position offset、历史预取 |
| `qwen3/attention.py` | prefill 历史 attention 路径 |
| `qwen3/paged_attention.py` | `paged_attention_prefill_from_history` |
| `qwen3/kernels/flash_attention_varlen.py` | q/kv 长度分离的 flash kernel（阶段 3 用） |
| `server.py` | 命中指标（`/metrics`） |
