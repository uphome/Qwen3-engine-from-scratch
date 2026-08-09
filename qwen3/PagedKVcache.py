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

    def alloc(self, n: int):
        """从空闲池取出 n 个物理块，返回块 ID 列表

        归还后重新分发的块先清零 K/V 内容：新请求的 prefill 只覆盖自己写入的
        槽位，若不清零，decode 的 seq_len 增长到未覆盖槽时会读到上一请求的
        残留 KV → 跨请求串扰（vLLM 的 --zero-initialized-kv-cache 同理）。
        首次分配的块在 __init__ 已是 torch.zeros，无需重复清零。
        """
        if len(self.free_block_ids) < n:
            raise RuntimeError(f"OOM: 请求 {n} 块, 剩余 {len(self.free_block_ids)} / {self.total_blocks}")
        allocated = self.free_block_ids[-n:]  # 拿出最后 n 个
        del self.free_block_ids[-n:]          # 直接在原列表上删除最后 n 个（原地操作）
        for blk in allocated:
            self.k_buffer[blk].zero_()
            self.v_buffer[blk].zero_()
        return allocated

    def free(self, block_ids: list):
        """归还物理块"""
        self.free_block_ids.extend(block_ids)

    @property
    def free_count(self):
        return len(self.free_block_ids)

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
        """批量写 K/V（decode 批，纯 GPU，一次 kernel 写 B 个请求各 1 token）

        k_new, v_new: (B, num_kv_heads, 1, head_dim)
        row_ids:      (B,) int32，各请求在常驻 2D 块表中的行号

        前提：本步已在层循环外 reserve_next()（页表已就绪），此处不做分配。
        与旧版逐请求 Python 循环 update() 的区别：write_pos/物理页全部用
        GPU gather（seq_lens[row_ids] / block_table_2d[row_ids, block_idx]），
        B 个请求合并成一次 advanced-indexing 写入。
        """
        write_pos = self.seq_lens[row_ids]              # (B,) 各请求当前 seq_len
        block_idx = write_pos // self.block_size        # (B,) 逻辑页
        offset = write_pos % self.block_size            # (B,) 页内偏移
        phys_ids = self.block_table_2d[row_ids, block_idx]  # (B,) 物理页号（GPU gather）

        # k_new (B, Hkv, 1, D) → (B, Hkv, D)；左式 (B, Hkv, D)（advanced indexing）
        self.k_buffer[phys_ids, layer_idx, :, offset] = k_new.squeeze(2)
        self.v_buffer[phys_ids, layer_idx, :, offset] = v_new.squeeze(2)


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
        max_pages = (pool.max_seq_len + pool.block_size - 1) // pool.block_size
        self.block_table_tensor = pool.block_table_2d[self.row_id]  # (max_pages,) 行视图
        self.num_pages = 0       # 当前已分配的页数（与 len(block_table) 同步）

    def _allocate_block(self):
        # 越界保护：超过 max_seq_len 对应的页数说明请求超长，
        # 显式报错而不是写穿张量（静默内存破坏）
        max_pages = self.block_table_tensor.shape[0]
        assert self.num_pages < max_pages, \
            f"请求超过 max_seq_len（{self._pool.max_seq_len} tokens / {max_pages} 页）"
        new = self._pool.alloc(1)
        self.block_table.append(new[0])
        self.block_table_tensor[self.num_pages] = new[0]   # 行视图 → 常驻 2D 表同步写
        self.num_pages += 1


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
        """读取该层已缓存的所有 K, V"""
        if self.seq_len == 0 or not self.block_table:
            return None, None

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
        """归还所有物理块到共享池 + 归还常驻表行号"""
        if self.block_table:
            self._pool.free(self.block_table)
            self.block_table = []
            self.num_pages = 0      # GPU 张量内容无需清零（只读前 num_pages 项）
        self.seq_len = 0
        self._pool.free_row(self.row_id)
