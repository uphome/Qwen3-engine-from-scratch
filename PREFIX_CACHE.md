# 前缀缓存（Prefix Cache / 哈希 KV 复用）

> 目标：让 **KV cache 在不同对话/请求之间复用**——相同前缀不再重复 prefill。
> 本文记录设计与实现（阶段 1：完全命中），阶段 2（部分命中）见文末计划。

---

## 1. 为什么需要它

多轮对话、重复 system prompt、批量评测同一组 prompt 时，请求之间有大量
相同前缀。没有前缀缓存时，每个请求都要把整个 prompt 重新算一遍
（prefill），而 prefill 通常是首 token 延迟（TTFT）的主要来源。

目标效果：

```
请求 A: [system prompt 256 tokens][user msg]  → prefill 全部 268 tokens
请求 B: [system prompt 256 tokens][user msg2] → 复用前 256 tokens 的 KV
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

`KVCachePool` 的物理块现在有三种状态：

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
| 缓存超上限 | LRU 淘汰最旧块：删除 hash 索引 + 清零回 free |

缓存上限：`max_cached_blocks = num_blocks // 2`（阶段 1 固定）。

### 2.3 完全命中流程（阶段 1）

命中条件：**prompt 长度块对齐**（`len % 16 == 0`）且所有完整块哈希全部命中。

```
Scheduler.add(req)
   │  compute_prefix_hashes(prompt)
   │  pool.match_prefix(hashes)          # 全部命中 → ref++
   ├─ 未命中 → waiting（普通 prefill，prefill 完成后登记块哈希）
   └─ 命中 →
        COW 最后一个块：alloc 私有块 + copy_block + release 原块
        bind_prefix(blocks[:-1] + [私有块])
        seq_len = L - 1                   # 最后一个 token 待重算
        state = decoding                  # 跳过 prefill
        running.append(req)
```

### 2.4 为什么需要 COW + seq_len = L-1

完全命中时缓存里已有 `L` 个 token 的 KV，但**最后一个 prompt token 的
logits 没有保存**（prefill 时算过但没留下），而第一个生成 token 正是由
它决定的。所以：

1. 逻辑上只复用前 `L-1` 个 token 的 KV（`seq_len = L-1`）
2. 第一次 decode 输入 `prompt[L-1]`（`Request.prefill_last_token`），
   position = `L-1`，重算该位置得到首 token
3. 这次 decode 会写 position `L-1` 的 K/V → 必须写**私有块**，
   因此把最后一个共享块复制一份（COW，单块约 18MB，~10µs）
4. 首 token 产出后 `seq_len` 回到 `L`，后续全部走正常 decode

这样既不用改 attention（decode kernel 本来就支持读历史 paged KV），
也保证了共享块永远只读。

### 2.5 生命周期示例

```
A: add(prompt) → prefill 写入 2 块 → decode → finish
   → 2 块登记缓存（cached=2, free=14）

B: add(相同 prompt)
   → match 2 块（ref=1）
   → COW 末块（alloc 1, copy, release 原块 → 回缓存）
   → seq_len=31，直接进 running（跳过 prefill）
   → 首 decode 重算第 32 个 token → 后续正常
   → finish：共享块回缓存、COW 副本清零回 free
```

---

## 3. 实测（Qwen3-0.6B，A100，CUDA Graph decode）

同一 256-token prompt（块对齐），贪心 16 tokens：

| 请求 | 路径 | TTFT | 输出 |
|---|---|---|---|
| A（冷启动） | prefill 256 tokens + decode | 54.8 ms | ✓ |
| B（完全命中） | 跳过 prefill，直接 decode | **6.7 ms** | 与 A 完全一致 |
| | | **8.24x** | 命中 16 块 |

池闭环：`free=46 + cached=16 = 62 = 64 - 2`（2 块为 CUDA Graph 占位 + 哑行）。

> 复现脚本：`/tmp/opencode/verify_prefix_cache.py`（本地验证用，未入库）

---

## 4. 可观测性

`GET /metrics` 新增：

| 指标 | 类型 | 含义 |
|---|---|---|
| `engine_prefix_cache_hits_total` | counter | 完全命中请求数 |
| `engine_prefix_cache_hit_blocks_total` | counter | 复用块数 |
| `engine_prefix_cache_cached_blocks` | gauge | 当前可复用缓存块（ref=0） |
| `engine_prefix_cache_capacity_blocks` | gauge | 缓存上限（LRU 阈值） |

---

## 5. 测试

`tests/unit/test_prefix_cache.py`（8 个用例）：

- 哈希链：前缀相同后块一致、差异块发散、非对齐只算完整块
- 完全命中：跳过 prefill、块共享、COW 私有块、`seq_len=L-1`、首 token 输入
- 引用计数：两个请求同时命中，一方结束不回收块
- 部分命中 / 不同前缀：不命中且已取块正确归还
- LRU 淘汰：5 个块争 4 个缓存位，最旧淘汰后不再命中
- 混合流量池闭环：`free + cached == total`

---

## 6. 边界与限制（阶段 1）

| 限制 | 说明 | 阶段 2 |
|---|---|---|
| 只支持**完全命中** | prompt 必须块对齐且全部命中 | 部分命中：复用前 N 块，后缀重新 prefill |
| 不省 prefill 的**部分**场景 | 如多轮对话（历史前缀 + 新消息） | 核心收益场景 |
| 缓存上限固定 | `num_blocks // 2` | 可配置 / 按命中率自适应 |
| 无跨进程持久化 | 缓存随进程生命周期 | 可选：磁盘 KV 持久化 |
| COW 开销 | 每次命中复制 1 个块（~18MB） | 可改为仅重算最后一个 token 不写 KV |
| 仅 Scheduler/server 路径 | `generate()`（main.py 单请求）不走 Scheduler，不查缓存；请求结束仍会登记块（受 LRU 上限约束） | 可选：generate 接入查询 |

---

## 7. 阶段 2 计划（部分命中）

目标场景：多轮对话——`system + 历史对话` 是前缀，新用户消息是后缀。

需要：

1. **prefill 只送未命中后缀**：`input_ids[:, matched:]`，position 从
   `matched` 开始（RoPE 表天然支持任意位置）
2. **prefill attention 读历史 KV**：当前 `flash_attention_prefill_batched`
   只计算当前输入的 K/V。需要支持
   `q = 后缀, k/v = 历史 paged KV + 当前后缀`：
   - 先做 PyTorch 参考实现（正确性）
   - 再把分页寻址接入 Triton flash varlen kernel（性能）
3. **块对齐**：`matched = floor(prefix_len / 16) * 16`，尾部重新计算
4. **验收**：第二轮对话 TTFT 显著下降；输出与无缓存一致；命中率 / 池闭环

预计工作量：2-3 天（kernel 占一半）。

---

## 8. 与 vLLM 的对应

| 能力 | 本项目 | vLLM |
|---|---|---|
| 块哈希 | 链式 blake2b（16 字节） | 链式 hash |
| 共享粒度 | 16-token 块（对齐） | block（默认 16） |
| 引用计数 + LRU | ✅ | ✅ |
| 完全命中复用 | ✅ 阶段 1 | ✅ |
| 部分命中（chunked prefix） | ⏳ 阶段 2 | ✅ |
| COW 末块 | ✅（命中路径） | ✅（copy-on-write 块） |
| KV 持久化 | ❌ | ✅（多级缓存） |

---

## 9. 关键文件

| 文件 | 内容 |
|---|---|
| `qwen3/prefix_cache.py` | 块哈希链（`hash_block` / `compute_prefix_hashes`） |
| `qwen3/PagedKVcache.py` | 块池状态机（ref / 缓存 / LRU / COW） |
| `qwen3/scheduler.py` | 命中判定、跳过 prefill、块哈希登记 |
| `qwen3/request.py` | `prefix_hashes` / `prefill_last_token` |
| `server.py` | 命中指标（`/metrics`） |
| `tests/unit/test_prefix_cache.py` | 8 个用例 |
