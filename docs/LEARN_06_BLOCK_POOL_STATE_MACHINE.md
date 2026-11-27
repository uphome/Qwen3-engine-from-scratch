# 学习篇 06：KV 块池状态机

> 对应代码：`qwen3/PagedKVcache.py`（`KVCachePool` / `PagedKVCache`）、
> `qwen3/scheduler.py`、`qwen3/request.py`
> 测试：`tests/unit/test_prefix_cache.py`、`tests/unit/test_cache_eviction.py`、
> `tests/unit/test_kv_pool.py`、`tests/unit/test_scheduler.py`
> 关联文档：[LEARN_02_PREFIX_CACHE.md](LEARN_02_PREFIX_CACHE.md)（哈希与命中策略）、
> [LEARN_03_SCHEDULING.md](LEARN_03_SCHEDULING.md)（Request 状态机）

---

## 0. 先建立正确的心智模型

一个常见的误解是：

> “PagedKVCache 是一个状态机。”

更准确的说法是：

```
块表（数据结构）                     状态机（生命周期）
──────────────────────────────      ────────────────────────────────
逻辑页 → 物理块                      物理块：free / owned / shared / cached
PagedKVCache.block_table            请求：waiting → prefilling → decoding → finished
block_table_2d[row_id]              （Request 状态机，见 LEARN_03）
```

- **块表**回答“第 i 个逻辑页现在落在哪个物理块”（where）；
- **块池状态机**回答“这个物理块现在能不能分配/复用/淘汰/释放”（whether）；
- **Request 状态机**回答“这个请求现在该 prefill 还是 decode，什么时候结束”（when）。

`PagedKVCache` 本身只是一个**请求级句柄**：

```python
class PagedKVCache:
    def __init__(self, pool):
        self._pool = pool
        self.seq_len = 0
        self.block_table = []        # 逻辑页 → 物理块
        self.row_id = pool.alloc_row()
        self.block_table_tensor = pool.block_table_2d[self.row_id]
        self.num_pages = 0
        self.block_hashes = []       # 与 block_table 对齐（只有完整 prompt 块有值）
```

真正维护“块现在是什么状态”的是 `KVCachePool`。这篇文档只讲块池状态机，
但会把 Request/块表如何驱动它一起串起来。

读完你应该能回答：

1. 为什么只有块表不够，还需要引用计数和缓存状态？
2. 运行时四态（FREE / OWNED / SHARED / CACHED）之间怎么迁移？谁会触发？
3. `block_refs`、`_prefix_index`、`_cached_lru`、`_block_hashes` 各自解决什么问题？
4. 完全命中时 COW 在状态机上具体发生了什么？
5. 为什么容量淘汰不会误伤正在 decode 的请求？
6. 缓存块 `ref=0` 是否等于“可以立刻回收”？
7. 哪些不变量被测试锁定？怎么自己写一个检查器？

---

## 1. 为什么块表之外还需要状态机

假设只有块表，没有状态：

```
请求 A：block_table = [3, 7]
请求 B：block_table = [3, 7]       # 共享前缀，看起来没问题
```

但接下来会发生什么？

- A decode 时要写新 token，如果写进块 7，会**污染 B 正在读的 KV**；
- A 结束后，块 3/7 要不要立即清零？如果 B 还在用，不能清零；
- 如果保留内容给未来请求复用，保留多久？缓存满了淘汰谁？
- 如果两个物理块内容相同（哈希冲突/重复登记），该保留哪个？

这些都不是“映射关系”能回答的，需要**引用计数 + 缓存索引 + 淘汰策略**，
也就是一个资源生命周期状态机。

核心思想：

| 机制 | 回答的问题 |
|---|---|
| `block_refs` | 现在有多少个活跃请求正在读这个块？ |
| `_prefix_index` | 这个块的内容（哈希）还能被谁命中？ |
| `_cached_lru` | 内容可复用时，按什么顺序淘汰？ |
| 块表 | 某个请求的逻辑页具体指向哪个物理块？ |

---

## 2. 数据结构地图

`KVCachePool` 里与状态机相关的字段（`PagedKVcache.py:57-77`）：

```python
# 空闲块列表：只有真正可分配的块在这里
self.free_block_ids = list(range(num_blocks))

# 每块引用计数：0 = 空闲/缓存待用；>0 = 活跃请求持有
self.block_refs = [0] * num_blocks

# 哈希 → 物理块（内容索引；包含 cached 和“命中后正在用”的块）
self._prefix_index: dict[int, int] = {}

# 物理块 → 哈希（淘汰时反向删除索引用）
self._block_hashes: dict[int, int] = {}

# ref=0 且可淘汰的缓存块，按 LRU 顺序
self._cached_lru: OrderedDict[int, None] = OrderedDict()

# 链感知淘汰：父哈希 → 子哈希集合 / 块 → 父哈希
self._children: dict[int, set[int]] = {}
self._block_parents: dict[int, int] = {}

# 缓存上限（默认物理块的 75%）
self.max_cached_blocks = max(1, int(num_blocks * 0.75))
```

再加两套“请求级”资源：

```python
# GPU 常驻 2D 块表：每请求一行，kernel 直接读
self.block_table_2d = torch.full((max_requests, max_pages), -1, ...)
self.seq_lens = torch.zeros(max_requests, ...)
self._free_rows = list(range(max_requests))
```

它们的关系：

```
                 ┌────────────────────────────────────────────┐
                 │              KVCachePool                    │
                 │                                            │
  free_block_ids │  block_refs   _prefix_index   _cached_lru  │
  ┌───────────┐  │  ┌─────────┐  ┌───────────┐  ┌──────────┐  │
  │ 3, 9, 11  │  │  │ 0,1,2.. │  │ h0 → 3    │  │ 7 → None │  │
  └───────────┘  │  └─────────┘  │ h1 → 7    │  │ 12 → None│  │
                 │               └───────────┘  └──────────┘  │
                 │                                            │
                 │  block_table_2d: [max_requests, max_pages] │
                 │  row 0: [3, 7, -1, ...]                    │
                 │  row 1: [3, 7, 9, ...]                     │
                 └────────────────────────────────────────────┘
```

---

## 3. 核心模型：运行时四态（2×2）

把“引用计数”和“是否进入池级哈希索引”当成两个正交维度，会得到四个运行时状态：

| | 未进 `_prefix_index` | 已进 `_prefix_index` |
|---|---|---|
| **ref = 0** | **① FREE（空闲）** | **④ CACHED（缓存保留）** |
| **ref > 0** | **② OWNED（私有活跃）** | **③ SHARED（共享活跃）** |

为什么可以说“三态”也可以说“四态”：

- 从**分配/淘汰**角度看，② OWNED 和 ③ SHARED 都是 `ref>0`，都不能被
  `alloc` 拿到、也不能被淘汰，所以常被合并成一个 `active`；
- 从**前缀缓存**角度看，② 和 ③ 的能力不同：② 只属于生产它的请求，
  不能被其他请求命中；③ 已登记池级哈希，可以被继续匹配、`ref` 可以 >1。
- 注意：**不要用“是否正在使用”来区分 ② 和 ③**——两者都有活跃请求在用。
  真正的区别是“是否可共享/可命中”。

> 下文的“状态机”以四态为准；其他文档里的 `active` 是 ②+③ 的简写。

> 补充：你提到的“新产生”和“正在被使用”在代码里都属于 `ref>0`，
> 并不是一个稳定的二分——新产生的块也在被它的属主请求使用。
> 代码真正区分的是“是否已登记池级索引”。如果把“已 alloc 但还没写 K/V”
> 也算一态，那只是 `update()` / `reserve_next()` 内部的瞬态，
> 不会在调度步之间停留。

逐格解释：

### 3.1 ① FREE（空闲）

```python
ref = 0
不在 _prefix_index
不在 _cached_lru
在 free_block_ids
K/V 内容不可依赖（alloc 时统一清零；直接 `KVCachePool.free()` 的路径可能残留旧数据）
```

这是唯一能被 `alloc(n)` 拿到的状态。`alloc` **永远不会拿到缓存块**，
这是池设计最关键的不变量（`PagedKVcache.py:79-98`）：

```python
def alloc(self, n):
    if len(self.free_block_ids) < n:
        raise RuntimeError(...)
    allocated = self.free_block_ids[-n:]
    del self.free_block_ids[-n:]
    for blk in allocated:
        self.k_buffer[blk].zero_()
        self.v_buffer[blk].zero_()
        self.block_refs[blk] = 1
    return allocated
```

### 3.2 ② OWNED（私有活跃）

```python
ref > 0
不在 _prefix_index
不在 _cached_lru
```

请求 prefill/decode 自己写出来的块。它在**整个请求生命周期内都是 OWNED**，
不只是“刚分配的那一刻”：

- decode 生产块没有哈希，会一直 OWNED 到释放；
- 完整 prompt 块虽然在请求句柄的 `block_hashes` 里记了哈希，但**池级索引登记
  发生在请求结束、块被释放时**，所以活跃期间仍然是 OWNED；
- 它只属于生产它的请求，其他请求无法命中。

释放时：

```
OWNED --release(hash)──────────► CACHED（新登记）
OWNED --release(hash=None)─────► FREE
OWNED --release(重复哈希)──────► FREE
```

> 这也解释了一个常见疑问：“为什么 A 还在跑，B 不能直接命中 A 刚 prefill 的块？”
> 因为池级索引登记在 `release_block()`，而不是 prefill 完成时。

### 3.3 ③ SHARED（共享活跃）

```python
ref > 0
在 _prefix_index 里（哈希仍然有效）
不在 _cached_lru
```

请求命中了缓存块后，`match_prefix_partial()` 做两件事：

```python
self.block_refs[blk] += 1
self._cached_lru.pop(blk, None)     # 使用期间不可淘汰
```

关键点：**它没有把块从 `_prefix_index` 删除**。所以：

- 块对后来的请求仍然可见，`ref` 可以继续增长到 2、3……
- 但它已经不在 LRU，淘汰逻辑不会碰它；
- 多个请求可以同时持有它（共享只读前缀）；
- 最后一个请求释放、`ref` 归零后，它再回到 CACHED。

### 3.4 ④ CACHED（缓存保留）

```python
ref = 0
在 _prefix_index
在 _cached_lru
不在 free_block_ids
K/V 内容保留
```

这是“没有人使用，但存着以备下一次复用”的状态。两个触发会离开它：

- `match_prefix_partial()`：CACHED → SHARED（ref++，出 LRU）
- `_evict_one()` / `clear_prefix_cache()`：CACHED → FREE（清 KV，回池）

### 3.5 四态迁移图

```
                        alloc(n)
        ┌────────┐ ─────────────────► ┌──────────────────────┐
        │  FREE  │                    │  OWNED（私有活跃）    │
        │  ref=0 │ ◄───────────────── │  ref>0, 不在索引       │
        └────────┘ release(hash=None) └──────────┬───────────┘
             ▲           / 重复哈希                │
             │                                     │ release(hash)
             │                                     ▼
             │                       ┌──────────────────────┐
             │      evict / clear    │  CACHED（缓存保留）    │
             └────────────────────── │  ref=0, 在索引 + LRU  │
                                     └──────────┬───────────┘
                                                │ match：ref++，出 LRU
                                                ▼
                                     ┌──────────────────────┐
                                     │  SHARED（共享活跃）    │
                                     │  ref>0, 仍在索引       │
                                     └──────────────────────┘
                                          │      ▲
                                          └──────┘
                                  release 后 ref 仍 >0（自迁移）
```

两条容易写错的规则：

1. **没有 OWNED → SHARED 的直接迁移**。新块必须先 `release(hash)` 变成
   CACHED（ref=0），之后才可能被 `match` 成 SHARED。
2. **SHARED 不会变回 OWNED**。它要么在 ref>0 时保持 SHARED（自迁移），
   要么在最后一个引用释放后变 CACHED。

---

## 4. 状态迁移总表

| # | 迁移 | 触发点 | 代码 | 关键动作 |
|---|---|---|---|---|
| 1 | FREE → OWNED | 请求需要新页 | `KVCachePool.alloc()` | 从 free 表尾取块、清零 K/V、`ref=1` |
| 2 | CACHED → SHARED | 前缀命中 | `match_prefix_partial()` | `ref++`、从 `_cached_lru` 移除，保留索引 |
| 3 | SHARED → SHARED | 共享块被另一个请求释放，仍有读者 | `release_block()` 开头 | `ref--`，块仍在索引，不可淘汰 |
| 4 | OWNED → CACHED | 新生产块释放，`hash != None`，且同哈希无其他块 | `release_block()` 分支 ① | 登记哈希/父链、进 LRU |
| 5 | SHARED → CACHED | 命中复用的块最后一个引用释放 | `release_block()` 分支 ② | 只恢复 LRU（不改哈希/父链） |
| 6 | OWNED → FREE | 最后一个引用释放，且 `hash == None` | `release_block()` → `_clear_and_free()` | 清 K/V、`ref=0`、回 free 表 |
| 7 | OWNED → FREE | 最后一个引用释放，但同哈希已有另一个缓存块 | `release_block()` 分支 ③ | 当前块内容作废，直接回收 |
| 8 | CACHED → FREE | 缓存超上限（释放路径） | `_evict_cache()` → `_evict_one()` | 链感知淘汰、清 KV、回池 |
| 9 | CACHED → FREE | 容量压力（分配路径） | `evict_cache_for()` → `_evict_one()` | 只淘汰 ref=0、只淘汰到够用、计数 |
| 10 | CACHED → FREE | 手动清空/基准重置 | `clear_prefix_cache()` | 全部缓存回池 |
| 11 | 任意 → FREE | 直接归还（测试/哑行） | `KVCachePool.free(block_ids)` | 绕过前缀缓存登记，直接清索引 |

> 表中的 “hash” 指池级哈希；请求句柄里的 `block_hashes` 只是“释放时交给池的凭据”。

### 4.1 `release_block` 的三条分支

这是状态机里最绕的一段（`PagedKVCache.py:155-188`）：

```python
def release_block(self, block_id, block_hash=None, parent_hash=0):
    self._release_ref(block_id)
    if self.block_refs[block_id] > 0:
        return                              # 还有别的请求在用：SHARED 自迁移

    if block_hash is not None:
        existing = self._prefix_index.get(block_hash)
        if existing is None:
            # ① 请求自己 prefill 出的新块 → 登记为缓存
            self._prefix_index[block_hash] = block_id
            self._block_hashes[block_id] = block_hash
            self._block_parents[block_id] = parent_hash
            self._children.setdefault(parent_hash, set()).add(block_hash)
            self._cached_lru[block_id] = None
            self._evict_cache()
            return

        if existing == block_id:
            # ② 本来就是缓存块（被命中复用）→ 恢复 LRU
            # 哈希/父链在首次登记时已正确；这里不要用 parent_hash 覆盖
            self._cached_lru[block_id] = None
            self._evict_cache()
            return

        # ③ 同哈希已有另一个缓存块（内容重复）→ 本块直接回收
    self._clear_and_free(block_id)
```

三条分支分别对应：

| 分支 | 场景 | 结果 |
|---|---|---|
| ① `existing is None` | 请求自己 prefill 出的完整 prompt 块 | OWNED → CACHED（新登记） |
| ② `existing == block_id` | 命中复用的缓存块，ref 归零 | SHARED → CACHED（只恢复 LRU） |
| ③ `existing 是别的块` | 重复哈希 | OWNED → FREE（内容作废） |

**分支 ② 是踩过坑的**：命中复用的块一直在 `_prefix_index` 里，
如果把它当成“重复块”清理，缓存会瞬间失效（命中率归 0）。正确做法是
只把它放回 LRU。这也是为什么需要 `existing == block_id` 这个判断。

还有一个更隐蔽的坑：分支 ② **不能用调用方传入的 `parent_hash` 覆盖已有父链**。
COW 和回滚路径调用 `release_block(b, h)` 时没有传 parent（默认 0），早期实现
会把已登记块的 `_block_parents` 改成 0，但 `_children` 仍保留原关系——
两份链信息不一致，链感知淘汰会退化（块看起来永远是“非叶节点”）。
修复：分支 ② 只恢复 LRU，哈希/父链保持首次登记时的值。
回归测试：`test_cow_release_keeps_parent_chain`。

---

## 5. 完整生命周期：一个冷请求

设定：`block_size=16`，prompt=40 tokens，`max_new=16`，池子共 32 块。

### 5.1 `Scheduler.add`

```python
req.kv_cache = PagedKVCache(pool)          # 分配 row 槽位，块表还是空的
hashes = compute_prefix_hashes(...)        # 40 // 16 = 2 个完整块哈希 [h0, h1]
matched = pool.match_prefix_partial(hashes)  # 冷启动：matched = []
req.state = WAITING
```

此时：

```
block_table = []
num_pages   = 0
free_count  = 32
cached      = 0
```

注意：**冷启动的 waiting 请求只占 row 槽位，不占物理块**。容量预检按
`ceil((input_len + max_new) / block_size)` 记账，不是等真正分配才检查。

### 5.2 prefill

`PagedKVCache.update()` 根据 `end_pos = 40` 计算需要 3 页：

```
FREE → OWNED（b0, b1, b2）
block_table = [b0, b1, b2]
block_refs  = b0=1, b1=1, b2=1
free_count  = 29
```

prefill 完成后的收尾（`Scheduler.on_step_done`）把完整 prompt 块的哈希
记到请求句柄：

```python
cache.block_hashes = [h0, h1, None]
```

`None` 表示第 3 块不是完整 prompt 块（40 = 2×16 + 8），不能作为前缀缓存。

### 5.3 decode 跨页

prompt 40 + 已生成 8 = 48，正好用满 3 块；再生成第 49 个 token 时需要第 4 页：

```
reserve_next() → FREE → OWNED（b3）
block_table = [b0, b1, b2, b3]
free_count  = 28
```

### 5.4 finish

`Request.finish()` → `PagedKVCache.free()` 逐块释放：

```python
for i, blk in enumerate(self.block_table):
    h = self.block_hashes[i]
    parent = self.block_hashes[i - 1] if i > 0 else 0
    self._pool.release_block(blk, h, parent or 0)
```

结果：

| 块 | 内容 | hash | 迁移 | 说明 |
|---|---|---|---|---|
| b0 | prompt[0:16] | h0 | OWNED → CACHED | 完整 prompt 块，可复用 |
| b1 | prompt[16:32] | h1 | OWNED → CACHED | 完整 prompt 块，可复用 |
| b2 | prompt[32:40] + gen[0:8] | None | OWNED → FREE | 内容混合，不能缓存 |
| b3 | gen[8:16] | None | OWNED → FREE | decode 块，直接回收 |

最终：

```
free_count = 30
cached     = 2
free + cached = 32 = total_blocks
```

这个“结束后 `free + cached == total`”是测试里最常断言的池闭合条件
（没有活跃请求时）。

---

## 6. 共享与引用计数：两个请求同时命中

承接上面的缓存：池里有 b0(h0)、b1(h1) 两个 CACHED 块。

### 6.1 请求 B 完全命中（prompt 恰好 32 tokens）

```
1. match_prefix_partial([h0, h1])
   → b0, b1: CACHED → SHARED，ref 各 +1，移出 LRU
   → 但仍在 _prefix_index 中

2. COW 最后一个块：
   → alloc(1) 得到 privB：FREE → OWNED（ref=1）
   → copy_block(b1, privB)
   → release_block(b1, h1)：ref 1→0，existing == b1
       → SHARED → CACHED（放回 LRU）
   → bind_prefix([b0, privB], [h0, None])
```

此时：

```
block_refs: b0=1, b1=0(CACHED), privB=1
B.block_table = [b0, privB]
```

### 6.2 请求 C 也完全命中

```
match_prefix_partial([h0, h1])
  → b0: ref 1→2（共享！）
  → b1: ref 0→1（CACHED → SHARED）
  → COW b1 → privC
  → release b1：ref 1→0 → CACHED
C.block_table = [b0, privC]
```

此时：

```
b0: ref=2（B、C 共享只读前缀）
b1: ref=0, CACHED
privB / privC: ref=1，各自私有
```

### 6.3 依次结束

```
B.finish():
  b0: ref 2→1（还有 C 在用，不能回缓存）
  privB: ref 1→0，hash=None → FREE

C.finish():
  b0: ref 1→0，existing == b0 → CACHED
  privC: ref 1→0，hash=None → FREE
```

最终 b0、b1 都回到 CACHED，两个私有块回 FREE。池闭合。

> 结论：`ref` 是“活跃读者数”，不是“是否在缓存里”。
> 一个块可以同时是“缓存可见的（在 `_prefix_index`）”和“有活跃读者的（ref>0）”。

### 6.4 部分命中（prompt 不是块对齐）

例：缓存了 32-token 前缀（h0, h1），新 prompt = 32 个旧 token + 4 个新 token。

```
match_prefix_partial([h0, h1])
  → b0、b1：CACHED → SHARED（ref++，移出 LRU）
  → bind_prefix([b0, b1], [h0, h1])
  → seq_len = 32，prefill_offset = 32

prefill 后缀 4 token
  → 需要 ceil(36/16)=3 页，已有 2 页
  → alloc 第 3 页 b2：FREE → OWNED（私有活跃）

结束
  → b0、b1：release(hash) → SHARED → CACHED
  → b2：不是完整 prompt 块（36 = 2×16 + 4），hash=None → FREE
```

如果后缀刚好补齐一个完整块（例如 32 + 16），`on_step_done` 会为第 3 块
记录 h2，结束时它也会进 CACHED。短历史回退（`QWEN3_PREFIX_PARTIAL_MIN`）
本质上是把刚 match 的块立刻 `release` 回 CACHED，状态机不需要特殊分支。

---

## 7. COW 在状态机上到底发生了什么

完全命中时，为什么不能直接复用最后一块？因为：

- 缓存里只有 prompt 的 K/V，没有 prompt 最后一个 token 的 **logits**；
- 首个生成 token 需要重算 position `L-1` 的 logits；
- 重算会**写** position `L-1` 的 K/V；
- 如果写进共享块，会污染其他正在读同一块的请求。

所以 `Scheduler.add` 做了一次 COW（`scheduler.py:102-108`）：

```
状态序列：
  b_last: CACHED
    ── match ──► SHARED (ref=1, 出 LRU)
    ── alloc priv ──► priv: FREE → OWNED (ref=1)
    ── copy_block(b_last, priv) ──► K/V 复制
    ── release_block(b_last, h) ──► b_last: SHARED → CACHED (ref=0, 回 LRU)
    ── bind_prefix([...b_last 之外的块..., priv]) ──► 请求持有 priv
```

之后：

- `seq_len = L - 1`：第一次 decode 从 position `L-1` 开始；
- decode 写 priv 的 offset `(L-1) % 16`；
- 共享块 b0…b_{n-2} 保持只读，仍可被其他请求命中。

为什么只 COW 最后一块？因为前面的块不会再被写；只有最后一个不完整位置
（或者恰好块对齐时的最后一个块）会被 decode 覆盖。

---

## 8. 淘汰的两条路径

### 8.1 正常容量控制：`_evict_cache`

缓存块登记/恢复后，如果超过上限：

```python
def _evict_cache(self):
    while len(self._cached_lru) > self.max_cached_blocks:
        if not self._evict_one():
            break
```

`_evict_one()` 是链感知的：

```python
def _evict_one(self):
    if not self._cached_lru:
        return False
    victim = None
    for blk in self._cached_lru:              # LRU 顺序
        h = self._block_hashes.get(blk)
        if h is not None and not self._children.get(h):
            victim = blk                      # 优先叶节点
            break
    if victim is None:                        # 整条链都在缓存
        victim = next(iter(self._cached_lru)) # 退回最旧
    ...
    self._clear_and_free(victim)              # CACHED → FREE
    return True
```

为什么要优先淘汰叶节点？假设缓存链 A → B → C，容量只够 2 块：

```
淘汰 A（链头）→ 剩 B、C
新请求从 A 开始匹配 → A miss → 命中 0 块
B、C 白留

淘汰 C（链尾）→ 剩 A、B
新请求从 A 开始匹配 → 命中 2 块
```

`_children` / `_block_parents` 就是在维护这条链：

```python
self._children[parent_hash].add(block_hash)
self._block_parents[block_id] = parent_hash
```

### 8.2 容量压力：`evict_cache_for`

当活跃请求需要新块、但 free 不够时，`Scheduler._ensure_capacity` 会：

```python
available = self.pool.free_count - reserved_blocks
if need <= available:
    return
if self.evict_on_demand:                       # 默认开
    self.pool.evict_cache_for(need - available)
    available = self.pool.free_count - reserved_blocks
    if need <= available:
        return
raise RuntimeError(...)                        # 淘汰完仍不够
```

```python
def evict_cache_for(self, need):
    freed = 0
    while freed < need and self._evict_one():
        freed += 1
    self.cache_evictions += freed
    return freed
```

关键性质：

1. **只淘汰 CACHED（ref=0）**，绝不碰 ref>0 的 OWNED / SHARED 块；
2. **只淘汰到够用为止**，不会为一个请求清空整个缓存；
3. **顺序是“先匹配、后淘汰”**：`Scheduler.add` 先 `match_prefix_partial`
   （命中块 ref++ 并移出 LRU），再 `_ensure_capacity`。所以当前请求的命中块
   在淘汰发生前就已经被保护；
4. **`cache_evictions` 只统计压力淘汰**，正常超上限的 `_evict_cache` 不计入；
5. `QWEN3_CACHE_EVICT_ON_DEMAND=0` 可退回“直接拒绝”策略。

> 这就是上一篇讨论的“缓存满了应该拒绝还是淘汰”的落地：
> 默认可用性优先（请求成功 > 缓存命中率），但保留开关。

### 8.3 手动清空

`clear_prefix_cache()` 把 `_cached_lru` 里的所有块 CACHED → FREE，
用于基准测试每轮重置，避免同一组 prompt 命中污染对比。

---

## 9. 行槽位：第二套资源状态机

除了物理块，`max_requests` 个 **块表行槽位**（row）也是一份有限资源：

```python
self.block_table_2d = torch.full((max_requests, max_pages), -1, ...)
self.seq_lens = torch.zeros(max_requests, ...)
self._free_rows = list(range(max_requests))
```

生命周期：

```
FREE ROW ──alloc_row()──► ALLOCATED ROW ──free_row()──► FREE ROW
（_free_rows 中）          （PagedKVCache.row_id）     （回到 _free_rows）
```

- `PagedKVCache.__init__` 立即 `alloc_row()`，所以 waiting 请求也占一行；
- `alloc_row()` 会把 `seq_lens[row]` 清零；
- `free_row()` 只归还行号，**不清零整行内容**——读取时只读前 `num_pages` 项，
  下次分配会覆盖；
- `max_requests` 耗尽会报独立的 OOM（不是物理块 OOM）。

行槽位与物理块是两套独立的容量限制：

```
物理块不够 → "OOM: 请求 n 块，剩余 ..."
行槽位不够 → "OOM: 请求行槽位耗尽（max_requests=...）"
```

---

## 10. 失败/取消路径：状态机会被绕过吗？

### 10.1 `Scheduler.add` 的异常回滚

如果 `_ensure_capacity` 在匹配之后抛错：

```python
except Exception:
    for b, h in zip(matched, hashes):
        self.pool.release_block(b, h)       # SHARED → CACHED（ref 归零）
    if req.kv_cache is not None and not req.kv_cache.block_table:
        req.kv_cache.free()                 # 归还 row 槽位
    raise
```

块仍然走完整的 `release_block` 状态迁移，不会泄漏。

### 10.2 引擎异常/客户端断开

`server.py` 的错误/取消清理会：

```python
if req.kv_cache is not None:      # 归还资源（跳过 Request 状态机）
    req.kv_cache.free()
    req.kv_cache = None
```

注意这里跳过的是 **Request 状态机**（没有 `finish()`），但块池仍然走
`PagedKVCache.free()` → `release_block()` 的正常状态迁移。两套状态机不要混淆。

### 10.3 直接 `KVCachePool.free()`

这是更底层的“立即归还”，不做前缀缓存登记：

```python
def free(self, block_ids):
    for blk in block_ids:
        h = self._block_hashes.pop(blk, None)
        if h is not None and self._prefix_index.get(h) == blk:
            del self._prefix_index[h]
        self._cached_lru.pop(blk, None)
        self.block_refs[blk] = 0
    self.free_block_ids.extend(block_ids)
```

它主要给测试、CUDA Graph 哑行、旧接口用。正常请求释放请走
`PagedKVCache.free()`，否则完整 prompt 块不会进缓存。

---

## 11. 不变量清单

把下面这些当成“状态机的公理”。任何改动都应该先问：会不会破坏它们？

| # | 不变量 |
|---|---|
| I1 | 每个物理块恰好属于 FREE / OWNED / SHARED / CACHED 之一（唯一归属） |
| I2 | FREE 块：`ref=0`、不在 `_prefix_index`、不在 `_cached_lru` |
| I3 | `_cached_lru` 中的块一定 `ref=0`，且一定在 `_prefix_index` |
| I4 | `_prefix_index[h] = b` ⇒ `_block_hashes[b] = h`；父链关系一致 |
| I5 | 活跃请求的块（`ref>0`）永远不在 `_cached_lru`，不会被淘汰 |
| I6 | 没有活跃请求时：`free_count + cached_count == total_blocks` |
| I7 | `alloc()` 只从 `free_block_ids` 取块，绝不返回 CACHED 块 |
| I8 | 每个 `PagedKVCache` 恰好占一个 row，`free_row()` 只调用一次 |
| I9 | 同一个哈希最多对应一个物理块（重复的会被回收） |
| I10 | `sum(block_refs)` 可以大于活跃块数（共享），但每个块的 ref = 引用它的请求数 |

一个教学用的检查器（可直接在测试/REPL 里跑）：

```python
def check_pool(pool):
    free = set(pool.free_block_ids)
    cached = set(pool._cached_lru)
    indexed = set(pool._prefix_index.values())
    active = {i for i, r in enumerate(pool.block_refs) if r > 0}
    owned = active - indexed          # ② ref>0，未登记索引
    shared = active & indexed         # ③ ref>0，已登记索引

    assert len(pool.free_block_ids) == len(free), "free 表有重复块"
    assert free.isdisjoint(cached | owned | shared), "free 与其他状态重叠"
    assert cached.isdisjoint(owned | shared), "cached 与活跃状态重叠"
    assert owned.isdisjoint(shared), "块不能同时是 owned 和 shared"
    assert free | owned | shared | cached == set(range(pool.total_blocks))

    assert all(pool.block_refs[b] == 0 for b in free | cached)
    assert all(pool.block_refs[b] > 0 for b in owned | shared)
    assert cached <= indexed
    assert indexed <= cached | shared       # 索引里的块要么缓存，要么共享活跃
    for h, b in pool._prefix_index.items():
        assert pool._block_hashes[b] == h
```

> 注意 `cached <= set(_prefix_index.values())` 而不是相等：
> `_prefix_index` 里还可能有被命中的 SHARED 块（ref>0）。

---

## 12. 动手实验

### 12.1 直接观察四态

```python
import torch
from qwen3.PagedKVcache import KVCachePool, PagedKVCache
from qwen3.prefix_cache import compute_prefix_hashes

pool = KVCachePool(
    num_blocks=8, num_layers=1, block_size=16,
    num_kv_heads=2, head_dim=8,
    device="cpu", dtype=torch.float32, max_seq_len=128,
)

c = PagedKVCache(pool)
ids = list(range(32))
k = torch.randn(1, pool.num_kv_heads, 32, pool.head_dim)
v = torch.randn(1, pool.num_kv_heads, 32, pool.head_dim)
c.update(0, k, v)
c.advance_seq_len(32)
c.block_hashes[:2] = compute_prefix_hashes(ids, 16)   # 模拟 on_step_done
c.free()

print("free:", pool.free_count)
print("cached:", pool.cached_count)                    # 2
print("LRU:", list(pool._cached_lru))
print("refs:", pool.block_refs)

# 再命中
c2 = PagedKVCache(pool)
matched = pool.match_prefix_partial(compute_prefix_hashes(ids, 16))
indexed = set(pool._prefix_index.values())
print("matched:", matched)
print("refs:", [pool.block_refs[b] for b in matched])
print("cached:", pool.cached_count)                    # 0（都移出 LRU）
print("owned:", [i for i, r in enumerate(pool.block_refs)
                 if r > 0 and i not in indexed])       # 空
print("shared:", [i for i, r in enumerate(pool.block_refs)
                  if r > 0 and i in indexed])          # matched 的两个块
```

预期：命中后 `matched` 的两个块 `ref=1`，`cached_count` 变 0，
但 `_prefix_index` 里仍然有它们。

### 12.2 观察链感知淘汰

```python
pool.max_cached_blocks = 1
# 造一条 A → B → C 的缓存链，再触发 _evict_cache()
# 观察淘汰的是 C（叶节点），不是 A（链头）
pytest tests/unit/test_prefix_cache.py::test_chain_aware_eviction_keeps_prefix_chain -q
```

### 12.3 观察容量压力淘汰

```bash
# 默认：淘汰缓存块，请求成功
pytest tests/unit/test_cache_eviction.py::test_evict_on_demand_frees_space -q

# 关闭淘汰：请求被拒绝（原行为）
pytest tests/unit/test_cache_eviction.py::test_evict_disabled_raises -q
```

### 12.4 常用开关

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `QWEN3_PREFIX_CACHE` | `1` | 关闭后不计算/不匹配前缀缓存 |
| `QWEN3_PREFIX_PARTIAL_MIN` | `1024` | 部分命中的最短历史阈值（0 = 总是启用） |
| `QWEN3_CACHE_EVICT_ON_DEMAND` | `1` | 容量不足时是否强制淘汰 CACHED 块 |

---

## 13. 自测问题（附参考要点）

1. **为什么 `alloc()` 不先从缓存里淘汰再分配？**
   因为分配是机制，淘汰是策略；`alloc` 只认识 FREE 块，不知道该为谁腾空间。
   淘汰由 `Scheduler._ensure_capacity` 在“匹配之后”统一决策，避免在
   COW/匹配中途破坏状态。

2. **`ref=0` 的块一定可以立刻回收吗？**
   不一定。`ref=0` 且在 `_cached_lru` 的是 CACHED，可以复用；只有淘汰/清空
   才回 FREE。`ref=0` 且不在索引的是 FREE。

3. **命中后为什么只从 `_cached_lru` 移除，不删除 `_prefix_index`？**
   因为内容仍然有效，其他请求还可以继续命中；`ref++` 已经保证不会被淘汰。

4. **`_block_hashes` 为什么需要？`
   淘汰时只有块 ID，需要反查哈希才能删 `_prefix_index`、维护父链。

5. **完全命中时 COW 的每一步状态是什么？**
   CACHED → match → SHARED → alloc priv（FREE → OWNED）→ copy →
   release 原块（SHARED → CACHED）；priv 保持 OWNED，之后 decode 写 priv。

6. **容量压力淘汰为什么不会误伤当前请求？**
   `Scheduler.add` 先匹配（命中块 ref++、移出 LRU），后淘汰；淘汰只碰
   `_cached_lru`（ref=0），当前请求的块已不在其中。

7. **`cache_evictions` 统计的是什么？**
   只统计 `evict_cache_for` 的压力淘汰；正常的 `_evict_cache` 超上限淘汰不计入。

8. **为什么请求结束后 `free_count + cached_count == total_blocks`？**
   没有活跃请求时，所有块要么在 free 表，要么在缓存 LRU；这是池闭合条件。

9. **`PagedKVCache.free()` 能调用两次吗？**
   不能。第一次会把 `block_table` 清空、归还 row；第二次虽然不会重复释放物理块，
   但会**重复 `free_row()`**，导致行槽位被重复加入空闲表（静默破坏）。
   `Request.finish()` 用状态机保证只释放一次，异常路径也会把句柄置 `None`。

10. **缓存块会不会被下一个请求写坏？**
    不会。共享块只读；唯一可能被写的最后一块在完全命中时 COW 成私有副本，
    部分命中时新 token 写入新分配的块。

11. **为什么 OWNED 和 SHARED 不合并成一个 active？**
    从分配/淘汰看可以合并（两者都 `ref>0`、都不可淘汰），这就是“三态”视角；
    但从前缀缓存看，OWNED 还没进池级索引、不能被其他请求命中，SHARED 已经
    登记哈希、可以被继续匹配。四态把“是否可共享”显式表达出来，调试命中率
    和 ref 计数时不会混淆。

---

## 14. 代码地图

| 功能 | 位置 |
|---|---|
| 物理块分配/清零 | `qwen3/PagedKVcache.py::KVCachePool.alloc` |
| 前缀匹配（ref++/出 LRU） | `KVCachePool.match_prefix_partial` |
| 释放三分支（缓存/恢复/重复） | `KVCachePool.release_block` |
| 链感知淘汰单块 | `KVCachePool._evict_one` |
| 超上限淘汰 | `KVCachePool._evict_cache` |
| 容量压力淘汰 | `KVCachePool.evict_cache_for` |
| 手动清空 | `KVCachePool.clear_prefix_cache` |
| 行槽位 | `KVCachePool.alloc_row` / `free_row` |
| 请求级句柄/块表 | `qwen3/PagedKVcache.py::PagedKVCache` |
| 完全命中 + COW | `qwen3/scheduler.py::Scheduler.add` |
| 请求结束释放 | `qwen3/PagedKVcache.py::PagedKVCache.free` |
| Request 状态机 | `qwen3/request.py` |
| 块池测试 | `tests/unit/test_prefix_cache.py`、`test_cache_eviction.py`、`test_kv_pool.py` |

---

## 15. 一句话总结

> **块表负责“地址”，引用计数负责“读者”，缓存索引负责“内容复用”，
> 状态机负责“什么时候允许谁动它”。**
>
> 运行时有四态：`FREE`（空闲）→ `OWNED`（私有活跃）→ `CACHED`（缓存保留）
> ↔ `SHARED`（共享活跃）。每一次迁移都必须满足：
> 活跃块（OWNED/SHARED）不可淘汰、缓存块可命中但不可写、空闲块不含有效内容。
