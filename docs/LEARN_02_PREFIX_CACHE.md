# 学习篇 02：前缀缓存（哈希 KV 复用）

> 对应代码：`qwen3/prefix_cache.py`、`qwen3/PagedKVcache.py`、`qwen3/scheduler.py`
> （`Scheduler.add`）、`qwen3/batch.py`、`qwen3/model.py`、`qwen3/attention.py`、
> `qwen3/paged_attention.py`、`tests/unit/test_prefix_cache.py`
> 关联文档：[PREFIX_CACHE.md](../PREFIX_CACHE.md)（设计与实测）、
> [LEARN_06_BLOCK_POOL_STATE_MACHINE.md](LEARN_06_BLOCK_POOL_STATE_MACHINE.md)（块池三态/COW/淘汰的逐状态讲解）

---

## 0. 这篇讲什么

读完你应该能回答：

1. 为什么 prefix caching 以"块"为单位，而不是以"整个 prompt"为单位？
2. 块哈希为什么必须**链式**？只对完整块计算的原因？
3. 一个物理块被多个请求共享时，生命周期怎么管（ref/LRU）？
4. "完全命中"为什么要复制最后一个块（COW）？为什么 `seq_len = L-1`？
5. "部分命中"怎么让 attention 只算后缀、却能看到全部历史？
6. 为什么短历史要**回退**全量 prefill？
7. 链感知 LRU 解决什么问题？

前置知识：分页 KV cache（PagedKVcache 的块表/页）、RoPE 的位置语义、
causal attention。

---

## 1. 问题定义

假设两个请求：

```
A: [system prompt 2048 tokens][user msg1]
B: [system prompt 2048 tokens][user msg2]
```

没有前缀缓存时，B 要把 2048-token 的 system prompt **重新算一遍**（prefill）。
前缀缓存的目标：B 直接复用 A 留下的 2048 个 token 的 **K/V**，
只计算后面的 user msg。

关键约束：

- KV 以分页方式存储（块大小 16），不能"半个 token"复用
- 共享的块必须是**只读**的（其他请求还要读）
- 要能回收（LRU），还要能统计命中

---

## 2. 块哈希链

文件 `qwen3/prefix_cache.py`：

```python
def hash_block(parent_hash: int, tokens: list[int]) -> int:
    """链式块哈希：h = blake2b(parent_hash || token_ids)"""
    h = hashlib.blake2b(digest_size=16)
    h.update(parent_hash.to_bytes(16, "little"))
    for t in tokens:
        h.update(int(t).to_bytes(4, "little"))
    return int.from_bytes(h.digest(), "little")


def compute_prefix_hashes(token_ids: list[int], block_size: int) -> list[int]:
    ids = list(token_ids)
    n_full = len(ids) // block_size
    hashes, parent = [], 0
    for i in range(n_full):
        chunk = ids[i * block_size:(i + 1) * block_size]
        parent = hash_block(parent, chunk)
        hashes.append(parent)
    return hashes
```

### 2.1 为什么是链式

```
h0 = H(0,  tokens[0:16])
h1 = H(h0, tokens[16:32])
h2 = H(h1, tokens[32:48])
```

- **前缀相同** → 哈希序列完全相同 → 可以命中
- **任意位置不同** → 从该块开始哈希全部不同（链式传播）→ 不会误命中
- 如果只对单块内容做哈希（不链），两个不同 prompt 的第 3 块内容可能相同
  （例如都是空格），但前面的内容不同，`h2` 相同就会错误命中

### 2.2 为什么只对完整块

`n_full = len(ids) // block_size`，尾部不足 16 的 token 不参与缓存：

- 请求结束时，最后一个不完整块会被 decode 的 token 写满（内容混了生成 token）
- 如果缓存它，未来"相同前缀但更长"的请求会命中一个内容不完整的块
- 所以只有"恰好 16 个 prompt token、且之后不会再写"的块才能作为缓存

---

## 3. 块池状态机

`KVCachePool`（`qwen3/PagedKVcache.py`）的物理块有三种状态：

```
                 alloc(n)                    release_block(hash)
 free_block_ids ─────────►  active(ref>0) ───────────────────► cached(ref=0, 在 LRU)
      ▲                        │                                    │
      │                        │ release_block(hash=None)           │ _evict_cache / clear
      │                        ▼                                    │
      └──────────────────  _clear_and_free ◄────────────────────────┘
```

数据字段：

```python
self.block_refs = [0] * num_blocks          # 每块引用计数
self._prefix_index: dict[int, int] = {}     # hash -> block_id
self._block_hashes: dict[int, int] = {}     # block_id -> hash
self._cached_lru: OrderedDict[int, None] = OrderedDict()
self._children: dict[int, set[int]] = {}    # hash -> 子块 hash 集合（链感知）
self._block_parents: dict[int, int] = {}    # block_id -> 父 hash
self.max_cached_blocks = int(num_blocks * 0.75)
```

### 3.1 `alloc` 永远不会拿到缓存块

```python
def alloc(self, n):
    allocated = self.free_block_ids[-n:]
    del self.free_block_ids[-n:]
    for blk in allocated:
        self.k_buffer[blk].zero_()
        self.v_buffer[blk].zero_()
        self.block_refs[blk] = 1
    return allocated
```

缓存块**不在** `free_block_ids` 里，所以 `alloc` 不会误拿到一个"内容还有用"的块，
更不会把它清零。这是池设计最关键的一条不变量。

### 3.2 匹配：最长前缀

```python
def match_prefix_partial(self, hashes):
    blocks = []
    for h in hashes:
        blk = self._prefix_index.get(h)
        if blk is None:
            break
        self.block_refs[blk] += 1
        self._cached_lru.pop(blk, None)   # 使用期间不可淘汰
        blocks.append(blk)
    return blocks
```

- 遇到第一个 miss 就停（最长前缀）
- `ref++`：告诉池"这些块正被活跃请求使用"
- 从 LRU 移出：`_cached_lru` 只放"可淘汰"的块

**部分命中失败时的回滚** 在 `Scheduler.add` 里：

```python
for b, h in zip(matched, hashes):
    self.pool.release_block(b, h)     # ref--，回到 LRU
```

### 3.3 释放：三条分支

```python
def release_block(self, block_id, block_hash=None, parent_hash=0):
    self._release_ref(block_id)
    if self.block_refs[block_id] > 0:
        return                          # 还有别的请求在用
    if block_hash is not None:
        existing = self._prefix_index.get(block_hash)
        if existing is None:            # ① 请求自己 prefill 出的新块 → 登记缓存
            ...
        if existing == block_id:        # ② 本来就是缓存块（被命中复用）→ 恢复 LRU
            ...
        # ③ 同哈希已有另一个缓存块 → 本块重复，直接回收
    self._clear_and_free(block_id)
```

第 ① 和 ② 的区别是踩过坑的：命中复用的块**一直在 `_prefix_index` 里**
（内容有效，供其他请求继续命中），match 时只从 LRU 移出。释放时如果看到
`existing == block_id` 就"当成重复块清理"，会把缓存块清掉（命中率变 0）。
正确做法是把它放回 LRU。

### 3.4 链感知 LRU

普通 LRU 的问题：

```
缓存链 A → B → C，容量只够 2 块
淘汰最旧 A → 剩下 B、C
新请求匹配：从 A 开始查，A miss → 命中 0 块（后面的块白留）
```

链感知淘汰：优先淘汰**没有子块**的块（叶节点）：

```python
def _evict_cache(self):
    while len(self._cached_lru) > self.max_cached_blocks:
        victim = None
        for blk in self._cached_lru:                  # LRU 顺序
            h = self._block_hashes.get(blk)
            if h is not None and not self._children.get(h):
                victim = blk                          # 第一个"叶节点"
                break
        if victim is None:                            # 整条链都在缓存
            victim = next(iter(self._cached_lru))     # 退回 LRU 最旧
        ...
```

`_children` / `_block_parents` 在登记块时维护（`parent_hash` 由
`PagedKVCache.free()` 传入：块 i 的父就是块 i-1 的 hash）。

效果：A→B→C 链 + 无关块 D、上限 3 → 淘汰 C（链尾），保留 A、B，
前缀仍能命中 2 块（见 `test_chain_aware_eviction_keeps_prefix_chain`）。

---

## 4. 完全命中（阶段 1）

条件：**prompt 长度块对齐**（`len % 16 == 0`）且所有完整块全部命中。

`Scheduler.add` 的核心：

```python
matched = self.pool.match_prefix_partial(hashes)
if hashes and len(matched) == len(hashes) and req.input_len % bs == 0:
    # COW 最后一个块：首个 decode 要重算 prompt 末 token 的 logits
    last, last_hash = matched[-1], hashes[-1]
    priv = self.pool.alloc(1)[0]
    self.pool.copy_block(last, priv)
    self.pool.release_block(last, last_hash)
    blocks = matched[:-1] + [priv]
    req.kv_cache.bind_prefix(blocks, hashes[:-1] + [None])
    req.kv_cache.seq_len = req.input_len - 1
    self.pool.seq_lens[req.kv_cache.row_id] = req.input_len - 1
    req.prefill_last_token = int(req.input_ids[0, -1].item())
    req.start_prefill()
    req.enter_decoding()
    self.running.append(req)
    return len(blocks) - 1      # 复用块（COW 私有块算新块）
```

### 4.1 为什么需要 COW（copy-on-write）

缓存里只有 prompt 的 K/V，**没有保存最后一个 prompt token 的 logits**。
而第一个生成 token 正是由它决定的。所以：

1. 逻辑上只复用前 `L-1` 个 token（`seq_len = L-1`）
2. 第一次 decode 输入 `prompt[L-1]`，position = `L-1`，重算该位置得到首 token
3. 这次 decode 会**写** position `L-1` 的 K/V —— 如果写进共享块，
   会污染其他正在读同一块的请求
4. 所以把最后一个共享块**复制一份私有副本**（约 18MB，~10µs），
   写私有块；原共享块保持只读

### 4.2 `prefill_last_token` 与 `cur_token`

`Request.cur_token` 是 decode 批组装输入的地方：

```python
@property
def cur_token(self):
    tok = self.output_ids[-1] if self.output_ids else self.prefill_last_token
    assert tok is not None, "decode 前既无生成 token 也无 prompt 末 token"
    return torch.tensor([[tok]], device=self.input_ids.device)
```

正常请求：prefill 已产出首 token → `output_ids[-1]`。
完全命中请求：没有 prefill → 用 `prefill_last_token` 作为第一个 decode 输入。

### 4.3 为什么 `seq_len = L-1`

`reserve_next` 的公式：`need = ceil((seq_len + 1) / block_size)`。

- `seq_len = L-1` 时，`need = L/block_size` = 命中的块数 → 不分配新块，
  第一次 decode 写进 COW 私有块的正确位置（offset `(L-1)%16`）
- 第一次 decode 后 `seq_len = L`；再下一步才分配新块

### 4.4 完全命中的容量记账

`Scheduler.add` 返回 `len(blocks) - 1`（复用块数，COW 的私有块算新块），
`Engine._dispatch` 据此记账：

```python
job.blocks_reserved = max(0, needed - hit_blocks)
self._pending_blocks += job.blocks_reserved
```

完全命中时 `needed = ceil((L+max_new)/16)`，`hit = L/16 - 1` →
预留 `1 + ceil(max_new/16)` 块（COW + decode 空间）。

---

## 5. 部分命中（阶段 2）

场景：prompt = 历史前缀（2048）+ 新消息（16）。命中 128 块，后缀 16 个 token。

### 5.1 调度层：只送后缀

```python
if matched:
    new_pages = needed_full - len(matched)
    if new_pages > self.pool.free_count - reserved_blocks:
        raise RuntimeError(...)
    req.kv_cache.bind_prefix(matched, hashes[:len(matched)])
    offset = len(matched) * bs
    req.kv_cache.seq_len = offset
    self.pool.seq_lens[req.kv_cache.row_id] = offset
    req.prefill_offset = offset
    self.waiting.append(req)
    return len(matched)
```

三个关键字段：

| 字段 | 含义 |
|---|---|
| `cache.seq_len` | 已复用的历史 token 数（= offset），决定 KV 写入位置 |
| `req.prefill_offset` | 本请求 prefill 的起点（= offset） |
| `req.prefill_end` | 本步 prefill 的终点（chunked prefill 用；非 chunk 时 = input_len） |

### 5.2 Batch：只组装后缀

```python
# input_lens
return [r.prefill_end - r.prefill_offset for r in self.requests]

# build_input_ids
parts = [r.input_ids[0, r.prefill_offset:r.prefill_end]
         for r in self.requests]
S_max = max(p.shape[0] for p in parts)
padded = torch.full((self.size, S_max), self.PAD_ID, ...)
for i, p in enumerate(parts):
    padded[i, :p.shape[0]] = p
```

### 5.3 Model：位置要带偏移

`qwen3/model.py::Qwen3Model.forward` 的 prefill 分支：

```python
base = torch.arange(S, device=input_ids.device).unsqueeze(0)
if caches is not None:
    offsets = torch.tensor([c.seq_len for c in caches],
                           device=input_ids.device).unsqueeze(1)
    position_ids = base + offsets
else:
    position_ids = base.repeat(B, 1)
```

- 后缀第 i 个 token 的绝对位置 = `seq_len(已复用历史) + i`
- RoPE 表按绝对位置查表 → 与整段 prefill 的位置一致
- **1-token 后缀也不能误判 decode**：`model.forward` 用显式 `mode` 参数，
  而不是只看 `S == 1`（见第 6 节）

### 5.4 Attention：paged prefill 读历史

`qwen3/paged_attention.py::paged_attention_prefill_from_history`：

```python
k_all = torch.cat([hist_k, k_new], dim=2)      # (1,Hkv,hist+S,D)
v_all = torch.cat([hist_v, v_new], dim=2)
k = repeat_kv(k_all, groups)                   # GQA 扩头
v = repeat_kv(v_all, groups)

kv_len = hist_len + S
scores = torch.matmul(q, k.transpose(-1, -2)) * scaling   # (1,H,S,hist+S)

# 因果 mask：query 全局位置 hist+i 可见 col <= hist+i
row = torch.arange(S, device=q.device).view(1, 1, S, 1) + hist_len
col = torch.arange(kv_len, device=q.device).view(1, 1, 1, kv_len)
scores = scores.masked_fill(col > row, float("-inf"))

probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
out = torch.matmul(probs, v)
```

数学上等价于"把后缀拼到历史后面，做标准 causal attention"，只不过
历史 K/V 是从分页池读出来的。

历史从哪来？`PagedKVCache.get_kv(layer_idx)`：

```python
ids = torch.tensor(self.block_table, dtype=torch.long, device=kbuf.device)
k = kbuf[ids, layer_idx]                    # (P, Hkv, BS, D) 一次 gather
k = k.permute(1, 0, 2, 3).reshape(k.shape[1], -1, k.shape[3])[:, :self.seq_len]
return k.unsqueeze(0), v.unsqueeze(0)
```

- 一次高级索引取出所有页，再 permute+reshape 成连续序列
- 取代"逐页 Python cat"（28 层 × 17 次 cat 会让 CPU 提交成为瓶颈）
- 逐层调用，峰值只有一层的量（~8MB），**不做全历史副本**（那是 235MB/请求）

### 5.5 短历史回退

实测（0.6B/A100，后缀 16）：

| 历史 | 全量 prefill | 部分命中 | 加速 |
|---:|---:|---:|---:|
| 512 | 34.0ms | 36.4ms | 0.93x |
| 1024 | 35.2ms | 37.5ms | 0.94x |
| 2048 | 56.8ms | 39.1ms | **1.45x** |

部分命中的固定成本（28 层权重读取 + 每层小算子）在短历史时高于全量 flash。
所以 `Scheduler.add` 里：

```python
min_partial = int(os.environ.get("QWEN3_PREFIX_PARTIAL_MIN", "1024"))
if matched and len(matched) * bs < min_partial:
    for b, h in zip(matched, hashes):
        self.pool.release_block(b, h)
    matched = []
```

低于阈值 → 释放命中块，走普通 prefill（不做负收益优化）。
`QWEN3_PREFIX_PARTIAL_MIN=0` 可强制总是启用（对照实验用）。

---

## 6. `mode` 参数：为什么不能只看 `seq_len > 0`

阶段 2 引入了一个隐蔽问题：部分命中请求的 `seq_len = 2048`（>0），
后缀可能是 **1 个 token**（`S == 1`）。旧判定：

```python
if caches[0].seq_len > 0:      # 误判为 decode！
```

修复：`Qwen3Model.forward` / `Attention.forward` 增加 `mode` 参数：

```python
is_decode = (
    mode == "decode" if mode is not None else
    (caches[0].seq_len > 0 and S == 1 and input_lens is None)
)
```

- 调度方（`Engine._step`、`generate`、`bench_batched`）显式传 `mode`
- 自动推断只用于测试/脚本兜底
- 这是一个很好的面试点：**"状态字段"不能替代"显式意图"**

---

## 7. 生命周期与池不变量

请求结束时（`PagedKVCache.free`）：

```python
for i, blk in enumerate(self.block_table):
    h = self.block_hashes[i] if i < len(self.block_hashes) else None
    parent = (self.block_hashes[i - 1]
              if i > 0 and i - 1 < len(self.block_hashes) else 0)
    self._pool.release_block(blk, h, parent or 0)
```

- 完整 prompt 块：有 hash → 登记/恢复为缓存
- decode 生产块 / 不完整块：hash 为 None → 清零回 free

**不变量**（测试锁定）：

```
free_count + cached_count + active(ref>0) == total_blocks
```

`tests/unit/test_prefix_cache.py` 的多个用例结尾都断言它，防止：
- 块泄漏（既不在 free 也不在 cached，ref 也不为 0）
- 重复计数（同时出现在 free 和 cached）

---

## 7.5 容量压力下的强制淘汰（可用性 vs 缓存命中率）

前面的规则是"缓存满 → 淘汰最旧/链尾"；但**自由块不足时时怎么办**？
早期实现是：

```
Scheduler.add → 容量检查 → 不足 → 直接拒绝（503/OOM）
```

即使缓存池里有大量 `ref=0` 的块可以腾出来，也会拒绝请求。这是"可预测性优先"
的取舍（缓存是纯优化，不该改变行为）。

后来改成 **vLLM 同款策略：可用性优先**：

```python
def _ensure_capacity(self, req, need, reserved_blocks):
    available = self.pool.free_count - reserved_blocks
    if need <= available:
        return
    if self.evict_on_demand:                 # QWEN3_CACHE_EVICT_ON_DEMAND != "0"
        self.pool.evict_cache_for(need - available)
        available = self.pool.free_count - reserved_blocks
        if need <= available:
            return
    raise RuntimeError(...)                   # 缓存全淘汰完仍不足
```

`evict_cache_for` 的语义：

```python
def evict_cache_for(self, need):
    freed = 0
    while freed < need and self._evict_one():   # 链感知：叶节点优先
        freed += 1
    self.cache_evictions += freed
    return freed
```

关键保护：

1. **只淘汰 `_cached_lru` 里的块**（ref=0）——活跃请求持有的块（ref>0）
   天然不在 LRU，不会被误动
2. **链感知**：优先淘汰链尾，尽量保住长前缀链
3. **只淘汰到够用为止**（不是清空）
4. **开关**：`QWEN3_CACHE_EVICT_ON_DEMAND=0` 退回"直接拒绝"（可预测性优先）
5. **指标**：`engine_prefix_cache_evictions_total` 观察淘汰频率

测试（`tests/unit/test_cache_eviction.py`，6 个用例）：

| 用例 | 验证 |
|---|---|
| `test_evict_on_demand_frees_space` | 容量不足 → 淘汰后请求成功 |
| `test_evict_not_triggered_when_space_enough` | 空间足够 → 不淘汰 |
| `test_evict_does_not_touch_active_blocks` | ref>0 的块不被淘汰 |
| `test_evict_disabled_raises` | 开关关闭 → 拒绝（原行为） |
| `test_evict_insufficient_still_raises` | 缓存淘汰完仍不足 → 拒绝 |
| `test_eviction_metrics_visible` | 指标（JSON 快照 + Prometheus 文本）可见 |

**面试视角**：这是"缓存满了应该拒绝还是淘汰"的经典权衡——
离线批处理可重试（拒绝可接受），在线低延迟服务应优先保证请求成功（淘汰缓存）。
实现上要注意"淘汰与正在匹配"的顺序：`match_prefix_partial` 先 ref++，
命中块移出 LRU，淘汰不会影响当前请求。

## 8. 测试怎么覆盖

| 用例 | 验证点 |
|---|---|
| `test_hash_chain_prefix_property` | 前缀相同 → 前块 hash 相同；差异块发散；非对齐只算完整块 |
| `test_full_hit_skips_prefill_and_shares_blocks` | 完全命中：跳 prefill、块共享、COW、`seq_len=L-1`、首 token 输入 |
| `test_ref_count_while_shared_by_active_request` | 两个请求同时命中，一方结束不回收块 |
| `test_unaligned_prompt_partial_hit` | 部分命中：`prefill_offset`、Batch 只送后缀 |
| `test_different_second_block_partial_hit` | 最长前缀匹配（命中 1 块） |
| `test_no_prefix_miss` | 首块不同 → 不命中 |
| `test_lru_eviction` | 容量淘汰 |
| `test_chain_aware_eviction_keeps_prefix_chain` | 链感知：淘汰链尾，保留前缀链 |
| `test_pool_accounting_after_mixed_traffic` | 池不变量 |
| `tests/model/test_prefix_prefill.py` | 部分命中 logits == 全量（CPU fp32，max diff < 1e-4） |
| `tests/kernels/test_flash_prefill.py` | flash 历史路径 vs PyTorch 参考 |

---

## 9. 踩过的坑（真实）

1. **LRU 把链头淘汰** → 剩余块无法从 prompt 开头匹配，命中率骤降
   - 修复：链感知淘汰（叶节点优先）

2. **命中块释放时被当成重复块清理**（`existing == block_id` 分支缺失）
   - 现象：同一 prompt 第二次命中后，第三次 miss
   - 修复：恢复 LRU 而不是清零

3. **`clear_prefix_cache` 没清 `_cached_lru` 本身**
   - 现象：`free + cached + active > total`（块重复计数）
   - 修复：`while self._cached_lru: popitem()`

4. **COW 末块缺失**
   - 现象：完全命中后首个 token 输出错误/崩溃
   - 根因：写共享块 or 首 token 没有输入
   - 修复：COW + `prefill_last_token` + `seq_len=L-1`

5. **部分命中时 attention 仍走 flash（不读历史）**
   - 现象：输出与全量 prefill 不一致
   - 修复：`has_history → paged_attention_prefill_from_history`

6. **HTTP 层按全量容量预检**
   - 现象：可命中的请求被 503 拒绝
   - 修复：删除 HTTP 预检，交给 Scheduler 精确裁决

7. **bench_batched 缓存污染**
   - 现象：batch 扫描第二轮开始全部命中，吞吐虚高
   - 修复：每轮 `clear_prefix_cache()`

8. **bf16 数值差异导致个别 token 翻转**
   - 现象：部分命中与全量 prefill 的前几个 token 一致，后面某 token 不同
   - 原因：不同 attention 实现的 bf16 舍入差异，不是逻辑错误
   - 验证：CPU fp32 精确等价（diff < 1e-4）；文档如实记录

---

## 10. 动手实验

1. **观察命中**：启动 server，发两次相同长 prompt（>1024 token），
   看 `/metrics` 的 `engine_prefix_cache_hits_total` / `hit_blocks`
2. **消融**：`QWEN3_PREFIX_CACHE=0` 重启，重复实验（TTFT 回退）
3. **阈值**：`QWEN3_PREFIX_PARTIAL_MIN=0` 强制短历史部分命中，测负收益
4. **构造多轮对话**：第二轮 prompt = 第一轮完整 prompt + 新消息，
   对比 TTFT
5. **读测试**：`test_full_hit...` 和 `test_unaligned_prompt_partial_hit`
   画一遍块表变化（打印 `cache.block_table`）

---

## 11. 自测问题

1. 为什么块哈希要带 `parent_hash`？不带会怎样？
2. 一个缓存块被两个请求同时持有，free 时如何保证不提前回收？
3. 完全命中为什么不能直接 `seq_len = L`？
4. 部分命中的 `position_ids` 为什么要加 offset？RoPE 表怎么支持？
5. 短历史为什么部分命中反而更慢？阈值怎么来的？
6. 链感知淘汰和普通 LRU 的区别？没有它，长前缀会怎样？
7. 池不变量是什么？哪类 bug 会被它抓住？

---

## 12. 代码地图

| 功能 | 位置 |
|---|---|
| 块哈希链 | `qwen3/prefix_cache.py` |
| 池状态机/ref/LRU/链感知/COW | `qwen3/PagedKVcache.py::KVCachePool` |
| 完全/部分命中、阈值回退 | `qwen3/scheduler.py::Scheduler.add` |
| 后缀组装 | `qwen3/batch.py::Batch` |
| 位置偏移/历史预取 | `qwen3/model.py::Qwen3Model.forward` |
| 历史 attention | `qwen3/paged_attention.py::paged_attention_prefill_from_history` |
| 历史 gather | `qwen3/PagedKVcache.py::PagedKVCache.get_kv` |
| 测试 | `tests/unit/test_prefix_cache.py`、`tests/model/test_prefix_prefill.py`、`tests/kernels/test_flash_prefill.py` |
