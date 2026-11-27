"""
分页 KV Cache — 共享物理块池 + 请求级块表

架构：
  KVCachePool  — 全局唯一，管理 GPU 显存（k_buffer / v_buffer）
  PagedKVCache — 每个请求一个，从池中分配块

接口：
  - get_kv(layer_idx) → (k, v) | (None, None)
  - update(layer_idx, k_new, v_new)
  - advance_seq_len(n)
  - seq_len
"""

import torch
from collections import OrderedDict

from .kernels.paged_attention import update_kv_batch_triton


# 物理块运行时状态（由 free_block_ids / block_refs / _prefix_index / _cached_lru 派生）
# 详见 docs/LEARN_06_BLOCK_POOL_STATE_MACHINE.md
BLOCK_FREE = "free"        # 空闲，可分配
BLOCK_OWNED = "owned"      # ref>0，未登记索引（新 prefill/decode 块，不可共享）
BLOCK_SHARED = "shared"    # ref>0，已登记索引（命中缓存，可被其他请求共享）
BLOCK_CACHED = "cached"    # ref=0，保留 K/V 等待复用（可被命中/淘汰）


class KVCachePool:
    """全局物理块池——所有请求共享，进程启动时创建一次"""

    def __init__(self, num_blocks: int, num_layers: int, block_size: int,
                 num_kv_heads: int, head_dim: int, device=None, dtype=None,
                 max_seq_len: int = 2048, max_requests: int = 256):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # 单请求的最大序列长度（prompt + 生成）。决定每个请求块表张量的
        # 预留大小（max_pages = ceil(max_seq_len / block_size)），
        # 与池子总块数无关——一个请求不可能用完全部池子。
        self.max_seq_len = max_seq_len

        # GPU 常驻 2D 块表 + seq_len：所有请求的页表/长度共享一张大表，
        # 每请求占一行（row_id 槽位）。update/reserve 用 index 赋值同步，
        # decode 时层循环直接引用（零组装、零 HtoD，见 update_kv_batch）。
        max_pages = (max_seq_len + block_size - 1) // block_size
        self.block_table_2d = torch.full(
            (max_requests, max_pages), -1, dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(max_requests, dtype=torch.int32, device=device)
        self._free_rows = list(range(max_requests))
        self.max_requests = max_requests

        # KVCachePool 用 torch.empty 初始化，未写入槽位是垃圾数据（可能含 NaN/±inf）
        # → decode 时 QK^T 溢出 + mask 的 -inf 相加产生 NaN → 输出全 0。
        # 现象是"多跑几次随机失败"。
        # 零初始化：torch.empty 的未写入槽位是垃圾数据（可能含 NaN/±inf），
        # 即使被 mask 挡住也可能通过 +inf + (-inf) = NaN 污染计算。
        # 工业实现（vLLM --zero-initialized-kv-cache）同样清零池。
        self.k_buffer = torch.zeros(num_blocks, num_layers, num_kv_heads,
                                     block_size, head_dim, device=device, dtype=dtype)
        self.v_buffer = torch.zeros(num_blocks, num_layers, num_kv_heads,
                                     block_size, head_dim, device=device, dtype=dtype)

        self.free_block_ids = list(range(num_blocks))
        self.total_blocks = num_blocks

        # ---- 前缀缓存（阶段 1：完全命中）----
        # block_refs: 每块引用计数（0 = FREE/CACHED；>0 = OWNED/SHARED）
        # _prefix_index: 块哈希 -> 物理块（CACHED + SHARED；OWNED 在 release 时登记）
        # _block_hashes: 物理块 -> 哈希（CACHED + SHARED；淘汰时反向删除索引）
        # _cached_lru:   CACHED 块（ref=0，LRU 淘汰顺序）
        self.block_refs = [0] * num_blocks
        self._prefix_index: dict[int, int] = {}
        self._block_hashes: dict[int, int] = {}
        self._cached_lru: OrderedDict[int, None] = OrderedDict()
        # 链感知淘汰：hash → 子块 hash 集合 / 块 → 父 hash。
        # 淘汰时优先选"叶节点"（无子块），避免淘汰链头后整条前缀无法命中。
        self._children: dict[int, set[int]] = {}
        self._block_parents: dict[int, int] = {}
        # 缓存上限：75% 物理块（留 25% 给活跃请求的 prefill/decode）
        self.max_cached_blocks = max(1, int(num_blocks * 0.75))
        self.prefix_hits = 0          # 完全命中请求数（统计用）
        self.prefix_hit_blocks = 0    # 复用块数（统计用）
        self.cache_evictions = 0      # 容量压力下强制淘汰的缓存块数（统计用）

    def alloc(self, n: int):
        """FREE → OWNED：从空闲池取出 n 个物理块，返回块 ID 列表

        归还后重新分发的块先清零 K/V 内容：新请求的 prefill 只覆盖自己写入的
        槽位，若不清零，decode 的 seq_len 增长到未覆盖槽时会读到上一请求的
        残留 KV → 跨请求串扰（vLLM 的 --zero-initialized-kv-cache 同理）。
        首次分配的块在 __init__ 已是 torch.zeros，无需重复清零。

        注意：前缀缓存块**不在** free_block_ids 里（它们保留 K/V 等待复用），
        所以 alloc 永远不会拿到缓存块，也就不会误清零。
        """
        if len(self.free_block_ids) < n:
            raise RuntimeError(f"OOM: 请求 {n} 块, 剩余 {len(self.free_block_ids)} / {self.total_blocks}")
        allocated = self.free_block_ids[-n:]  # 拿出最后 n 个
        del self.free_block_ids[-n:]          # 直接在原列表上删除最后 n 个（原地操作）
        for blk in allocated:
            self.k_buffer[blk].zero_()
            self.v_buffer[blk].zero_()
            self.block_refs[blk] = 1
        return allocated

    def free(self, block_ids: list):
        """任意状态 → FREE：直接归还物理块（不登记前缀缓存）

        普通路径（测试 / 占位句柄 / 哑行）用；请求级释放走
        PagedKVCache.free() → release_block()（会把完整块登记为缓存）。
        """
        for blk in block_ids:
            h = self._block_hashes.pop(blk, None)
            if h is not None and self._prefix_index.get(h) == blk:
                del self._prefix_index[h]
            self._cached_lru.pop(blk, None)
            self.block_refs[blk] = 0
        self.free_block_ids.extend(block_ids)

    @property
    def free_count(self):
        return len(self.free_block_ids)

    def block_state(self, block_id: int) -> str:
        """返回物理块的运行时状态（只读调试/测试/监控用）

        四态（与 docs/LEARN_06_BLOCK_POOL_STATE_MACHINE.md 一致）：
          - BLOCK_FREE:   ref=0，在 free_block_ids，未登记索引
          - BLOCK_OWNED:  ref>0，未登记索引（新 prefill/decode 块，不可共享）
          - BLOCK_SHARED: ref>0，已登记索引（命中缓存，可被其他请求共享）
          - BLOCK_CACHED: ref=0，在 _cached_lru（保留 K/V，可被命中/淘汰）

        状态由现有字段派生，不单独存储，避免多份真相不一致。
        """
        if not 0 <= block_id < self.total_blocks:
            raise IndexError(
                f"block_id 越界: {block_id}（total_blocks={self.total_blocks}）")
        if block_id in self._cached_lru:
            return BLOCK_CACHED
        if self.block_refs[block_id] == 0:
            return BLOCK_FREE
        return BLOCK_SHARED if block_id in self._block_hashes else BLOCK_OWNED

    # ---- 前缀缓存：匹配 / 引用 / 淘汰 ----

    @property
    def cached_count(self) -> int:
        """当前 CACHED 块数（ref=0，未淘汰）"""
        return len(self._cached_lru)

    def match_prefix_partial(self, hashes: list[int]) -> list[int]:
        """最长前缀匹配：逐块查，遇到 miss 停止

        CACHED → SHARED：命中块 ref++，从 LRU 移出，但仍留在索引中。
        调用方负责：
          - 全命中 → 阶段 1 快速路径
          - 部分命中 → 绑定前缀块，prefill 后缀
          - 放弃使用 → 对每个块 release_block(blk, hashes[i])
        """
        blocks: list[int] = []
        for h in hashes:
            blk = self._prefix_index.get(h)
            if blk is None:
                break
            self.block_refs[blk] += 1
            self._cached_lru.pop(blk, None)
            blocks.append(blk)
        return blocks

    def match_prefix(self, hashes: list[int]) -> list[int] | None:
        """按哈希序列取缓存块并 ref++（要求全部命中）

        任一缺失返回 None（已取的块自动 release）。
        """
        blocks = self.match_prefix_partial(hashes)
        if len(blocks) != len(hashes):
            for b, h in zip(blocks, hashes):
                self.release_block(b, h)
            return None
        return blocks

    def release_block(self, block_id: int, block_hash: int | None = None,
                      parent_hash: int = 0):
        """请求释放块：ref--，归零后按状态迁移

        - OWNED + hash（同哈希无其他块）→ CACHED（新登记，保留 K/V）
        - SHARED + hash（命中复用）→ CACHED（恢复 LRU）
        - OWNED + hash=None → FREE（decode 块 / 不完整块）
        - OWNED + 同哈希已有其他块 → FREE（重复内容）

        parent_hash 用于维护哈希链（链感知淘汰：优先淘汰链尾）。
        注意命中复用的情况：块一直在 `_prefix_index` 里（内容有效），
        match 时只是从 LRU 队列移出；释放时 ref 归零 → 放回 LRU 即可。
        """
        self._release_ref(block_id)
        if self.block_refs[block_id] > 0:
            return
        if block_hash is not None:
            existing = self._prefix_index.get(block_hash)
            if existing is None:
                # 请求自己 prefill 出的新块：登记为缓存
                self._prefix_index[block_hash] = block_id
                self._block_hashes[block_id] = block_hash
                self._block_parents[block_id] = parent_hash
                self._children.setdefault(parent_hash, set()).add(block_hash)
                self._cached_lru[block_id] = None
                self._evict_cache()
                return
            if existing == block_id:
                # 原本就是缓存块（被命中复用）：只恢复 LRU 待用。
                # 哈希/父链在首次登记时已正确；match 只把块移出 LRU，
                # 未修改索引。COW/回滚路径可能传 parent_hash=0，
                # 若在这里覆盖会把父链打断（链感知淘汰退化）。
                self._cached_lru[block_id] = None
                self._evict_cache()
                return
            # 同哈希已有另一个缓存块（内容重复）→ 本块直接回收
        self._clear_and_free(block_id)

    def copy_block(self, src: int, dst: int):
        """复制物理块全部层 K/V（前缀缓存 COW 用；dst 由 alloc 得到，ref=1）"""
        self.k_buffer[dst].copy_(self.k_buffer[src])
        self.v_buffer[dst].copy_(self.v_buffer[src])

    def _release_ref(self, block_id: int):
        assert self.block_refs[block_id] > 0, \
            f"块 {block_id} 引用计数为 0，不能 release"
        self.block_refs[block_id] -= 1

    def _clear_and_free(self, block_id: int):
        self.k_buffer[block_id].zero_()
        self.v_buffer[block_id].zero_()
        self.block_refs[block_id] = 0
        self.free_block_ids.append(block_id)

    def clear_prefix_cache(self):
        """CACHED → FREE：清空前缀缓存（基准测试每轮重置，避免同一组 prompt 命中污染对比）"""
        while self._cached_lru:
            blk, _ = self._cached_lru.popitem(last=False)
            h = self._block_hashes.pop(blk, None)
            parent = self._block_parents.pop(blk, 0)
            if h is not None:
                if self._prefix_index.get(h) == blk:
                    del self._prefix_index[h]
                kids = self._children.get(parent)
                if kids is not None:
                    kids.discard(h)
                    if not kids:
                        del self._children[parent]
            self._clear_and_free(blk)

    def _evict_one(self) -> bool:
        """CACHED → FREE：淘汰一个缓存块（链感知：优先叶节点），返回是否成功

        链感知：优先淘汰"叶节点"（没有子块在缓存中的块），
        避免淘汰链头后剩余块无法从 prompt 开头匹配（命中率骤降）。
        若全部块都有子块（整条链都在缓存里），退回 LRU 最旧。
        """
        if not self._cached_lru:
            return False
        victim = None
        for blk in self._cached_lru:
            h = self._block_hashes.get(blk)
            if h is not None and not self._children.get(h):
                victim = blk
                break
        if victim is None:
            victim = next(iter(self._cached_lru))
        self._cached_lru.pop(victim, None)
        h = self._block_hashes.pop(victim, None)
        parent = self._block_parents.pop(victim, 0)
        if h is not None:
            if self._prefix_index.get(h) == victim:
                del self._prefix_index[h]
            kids = self._children.get(parent)
            if kids is not None:
                kids.discard(h)
                if not kids:
                    del self._children[parent]
        self._clear_and_free(victim)
        return True

    def _evict_cache(self):
        """CACHED → FREE：缓存超过上限时按 LRU/链感知淘汰（释放路径的正常容量控制）"""
        while len(self._cached_lru) > self.max_cached_blocks:
            if not self._evict_one():
                break

    def evict_cache_for(self, need: int) -> int:
        """CACHED → FREE：容量压力下强制淘汰，为活跃请求腾出空间

        Args:
            need: 需要增加的 free 块数
        Returns:
            实际释放的块数（可能 < need，说明缓存已空）
        仅淘汰 `_cached_lru` 中的块（CACHED, ref=0）；被活跃请求持有的块不受影响。
        """
        freed = 0
        while freed < need and self._evict_one():
            freed += 1
        self.cache_evictions += freed
        return freed

    # ---- 常驻 2D 块表：行槽位管理 ----

    def alloc_row(self) -> int:
        """为新请求分配一个常驻表行号（GPU 端 seq_lens 清零）"""
        if not self._free_rows:
            raise RuntimeError(
                f"OOM: 请求行槽位耗尽（max_requests={self.max_requests}），"
                f"增大 KVCachePool(max_requests=...)")
        row = self._free_rows.pop()
        self.seq_lens[row] = 0
        return row

    def free_row(self, row: int):
        """请求结束归还行号（表内容无需清零，num_pages 只读前 n 项）"""
        self._free_rows.append(row)

    def update_kv_batch(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
                        row_ids: torch.Tensor):
        """批量写 K/V（decode 批，纯 GPU，一次 Triton kernel 写 B 个请求各 1 token）

        k_new, v_new: (B, num_kv_heads, 1, head_dim)
        row_ids:      (B,) int32，各请求在常驻 2D 块表中的行号

        前提：本步已在层循环外 reserve_next()（页表已就绪），此处不做分配。
        实现：Triton kernel（grid = B×num_kv_heads）一次 launch 同时写 K 和 V，
        替代 PyTorch advanced indexing（每层 2 次 index_elementwise + 中间地址计算）。
        """
        k_in = k_new.squeeze(2)                          # (B, Hkv, D)
        v_in = v_new.squeeze(2)
        update_kv_batch_triton(
            k_in, v_in, self.k_buffer, self.v_buffer,
            self.seq_lens, self.block_table_2d, row_ids, layer_idx)


class PagedKVCache:
    """请求级 KV cache —— 轻量句柄

    用法:
        pool = KVCachePool(num_blocks=512, num_layers=28, ...)   # 全局创建一次
        cache = PagedKVCache(pool)                                # 每个请求创建一个
        # ... 正常使用 get_kv / update / advance_seq_len ...
        cache.free()                                              # 请求结束，归还块
    """

    def __init__(self, pool: KVCachePool):
        self._pool = pool
        self.seq_len = 0
        self.block_table = []    # 逻辑块表（Python list，兼容旧接口）

        # 常驻 2D 块表的行视图：pool.block_table_2d[row_id]（GPU 常驻，同步维护）。
        # kernel / update 直接读它，避免每层组装 (B, max_pages) 的 HtoD/D2D
        # 拷贝（profiler 显示批路径每层 zeros + B 次行拷贝 + seq_len HtoD）。
        self.row_id = pool.alloc_row()
        self.block_table_tensor = pool.block_table_2d[self.row_id]  # (max_pages,) 行视图
        self.num_pages = 0       # 当前已分配的页数（与 len(block_table) 同步）
        # 与 block_table 对齐的块哈希（仅"完整 prompt 块"有值）：
        # 请求结束 free 时，有哈希的块登记进前缀缓存（保留 K/V）。
        self.block_hashes: list[int | None] = []

    def _allocate_block(self):
        # 越界保护：超过 max_seq_len 对应的页数说明请求超长，
        # 显式报错而不是写穿张量（静默内存破坏）
        max_pages = self.block_table_tensor.shape[0]
        assert self.num_pages < max_pages, \
            f"请求超过 max_seq_len（{self._pool.max_seq_len} tokens / {max_pages} 页）"
        new = self._pool.alloc(1)
        self.block_table.append(new[0])
        self.block_table_tensor[self.num_pages] = new[0]   # 行视图 → 常驻 2D 表同步写
        self.block_hashes.append(None)
        self.num_pages += 1

    def bind_prefix(self, block_ids: list[int], hashes: list[int]):
        """绑定已命中的前缀块（块 ref 已由 pool.match_prefix 增加）

        用于前缀缓存完全命中：请求直接持有这些共享块（只读），
        seq_len 由调用方设为命中长度；decode 从新块继续写。
        """
        assert not self.block_table, "bind_prefix 只能在空块表上调用"
        assert len(block_ids) == len(hashes)
        assert len(block_ids) <= self.block_table_tensor.shape[0], "命中块数超过 max_pages"
        self.block_table = list(block_ids)
        self.block_hashes = list(hashes)
        self.num_pages = len(block_ids)
        for i, blk in enumerate(block_ids):
            self.block_table_tensor[i] = blk


    def reserve_next(self):
        """decode 步前预留页：本步将写 1 个新 token，若跨页边界则先申请。

        页分配从层循环内（旧 update 每层可能分配）提前到每步一次：
        申请后本步 28 层的块表完全稳定，update_kv_batch 只做纯 GPU 写入。
        条件推导：write_pos = seq_len，所需页 = ceil((seq_len+1)/16)；
        页已存在 iff num_pages >= ceil((seq_len+1)/16)。仅在不足时申请。
        """
        need = (self.seq_len + 1 + self._pool.block_size - 1) // self._pool.block_size
        if need > self.num_pages:
            self._allocate_block()

    def get_kv(self, layer_idx: int):
        """读取该层已缓存的所有 K, V → (1, Hkv, seq_len, D)

        GPU 上走"一次高级索引 gather"：k_buffer[ids, layer] 一次取出所有页，
        再 permute+reshape 成连续序列——替代逐页 Python 循环 + torch.cat
        （前缀缓存部分命中的 prefill 每层都要读历史，逐页 cat 会产生
        17×2 次小 kernel，28 层累计上千次，CPU 提交成为瓶颈）。
        """
        if self.seq_len == 0 or not self.block_table:
            return None, None

        kbuf = self._pool.k_buffer
        if kbuf.is_cuda:
            ids = torch.tensor(self.block_table, dtype=torch.long,
                               device=kbuf.device)
            k = kbuf[ids, layer_idx]                    # (P, Hkv, BS, D)
            v = self._pool.v_buffer[ids, layer_idx]
            k = k.permute(1, 0, 2, 3).reshape(
                k.shape[1], -1, k.shape[3])[:, :self.seq_len]
            v = v.permute(1, 0, 2, 3).reshape(
                v.shape[1], -1, v.shape[3])[:, :self.seq_len]
            return k.unsqueeze(0), v.unsqueeze(0)

        k_pages = [self._pool.k_buffer[pid, layer_idx] for pid in self.block_table]
        v_pages = [self._pool.v_buffer[pid, layer_idx] for pid in self.block_table]

        k = torch.cat(k_pages, dim=1)[:, :self.seq_len, :]   # [H, seq_len, D]
        v = torch.cat(v_pages, dim=1)[:, :self.seq_len, :]

        return k.unsqueeze(0), v.unsqueeze(0)   # [1, H, seq_len, D]

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """将新 token 的 K, V 写入页中（向量化，一次写入所有 token）

        k_new, v_new: (B, num_kv_heads, S_new, head_dim)

        核心思想：把"每个 token 写到哪个物理页的哪个偏移"从 Python 循环
        改成张量批量计算（advanced indexing），一次 kernel 写完全部 token。
        旧版逐 token 循环在 prefill（S_new 大）时是 28 层 × S_new 次
        Python 循环 + kernel 启动，向量化后循环次数降为 1。
        """
        _, _, S_new, _ = k_new.shape
        write_pos = self.seq_len
        end_pos = write_pos + S_new

        # ---- 1. 先一次性补齐页表（而不是写 token 过程中发现不够再补）----
        #     需要 ceil(end_pos / block_size) 个逻辑页；不够则从池申请
        need_blocks = (end_pos + self._pool.block_size - 1) // self._pool.block_size
        while len(self.block_table) < need_blocks:
            self._allocate_block()

        # ---- 2. 向量化定位：每个 token 的逻辑位置 → (物理页, 页内偏移) ----
        pos = torch.arange(write_pos, end_pos, device=k_new.device)  # 所有逻辑位置 [0, 1, ..., S_new-1]
        block_idx = pos // self._pool.block_size                     # 属于第几个逻辑页（每 block_size 个相同）
        offset = pos % self._pool.block_size                         # 页内偏移（0..block_size-1 循环）
        # 块表查表：逻辑页 → 物理页 ID（同一逻辑页的所有 token 映射到同一物理页）
        # 直接从 GPU 常驻张量切片（物理页号已经在 GPU 上，零 HtoD 拷贝）
        phys_ids = self.block_table_tensor[:self.num_pages][block_idx]

        # ---- 3. advanced indexing 一次写入全部 token ----
        # 左边: pool.k_buffer[phys_ids, layer_idx, :, offset]
        #   phys_ids / offset 都是 (S_new,) 索引张量 → PyTorch 逐元素配对:
        #   结果[i, h, j] = pool.k_buffer[phys_ids[i], layer_idx, h, offset[i], j]
        #   结果形状 (S_new, num_kv_heads, head_dim)，即第 i 个 token 的 K 写进
        #   它对应的 (物理页, 页内偏移)，一次 kernel 完成
        # 右边: k_new[0] 是 (num_kv_heads, S_new, head_dim)，permute(1,0,2)
        #   调换成 (S_new, num_kv_heads, head_dim) 与左边形状对齐
        self._pool.k_buffer[phys_ids, layer_idx, :, offset] = k_new[0].permute(1, 0, 2)
        self._pool.v_buffer[phys_ids, layer_idx, :, offset] = v_new[0].permute(1, 0, 2)

    def advance_seq_len(self, n: int = 1):
        self.seq_len += n
        # GPU 常驻 seq_len 同步（decode kernel / update_kv_batch 的数据源）
        self._pool.seq_lens[self.row_id] = self.seq_len

    def free(self):
        """归还所有物理块（OWNED/SHARED → CACHED/FREE）+ 归还常驻表行号"""
        if self.block_table:
            for i, blk in enumerate(self.block_table):
                h = self.block_hashes[i] if i < len(self.block_hashes) else None
                parent = (self.block_hashes[i - 1]
                          if i > 0 and i - 1 < len(self.block_hashes) else 0)
                self._pool.release_block(blk, h, parent or 0)
            self.block_table = []
            self.block_hashes = []
            self.num_pages = 0      # GPU 张量内容无需清零（只读前 num_pages 项）
        self.seq_len = 0
        self._pool.free_row(self.row_id)
