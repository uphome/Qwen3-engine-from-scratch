"""
分页 KV Cache — 共享物理块池 + 请求级块表

架构：
  KVCachePool  — 全局唯一，管理 GPU 显存（k_buffer / v_buffer）
  PagedKVCache — 每个请求一个，从池中分配块，接口兼容 NaiveKVCache

与 NaiveKVCache 接口一致：
  - get_kv(layer_idx) → (k, v) | (None, None)
  - update(layer_idx, k_new, v_new)
  - advance_seq_len(n)
  - seq_len
"""

import torch


class KVCachePool:
    """全局物理块池——所有请求共享，进程启动时创建一次"""

    def __init__(self, num_blocks: int, num_layers: int, block_size: int,
                 num_kv_heads: int, head_dim: int, device=None, dtype=None):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.k_buffer = torch.empty(num_blocks, num_layers, num_kv_heads,
                                     block_size, head_dim, device=device, dtype=dtype)
        self.v_buffer = torch.empty(num_blocks, num_layers, num_kv_heads,
                                     block_size, head_dim, device=device, dtype=dtype)

        self.free_block_ids = list(range(num_blocks))
        self.total_blocks = num_blocks

    def alloc(self, n: int):
        """从空闲池取出 n 个物理块，返回块 ID 列表"""
        if len(self.free_block_ids) < n:
            raise RuntimeError(f"OOM: 请求 {n} 块, 剩余 {len(self.free_block_ids)} / {self.total_blocks}")
        allocated = self.free_block_ids[-n:]  # 拿出最后 n 个
        del self.free_block_ids[-n:]          # 直接在原列表上删除最后 n 个（原地操作）
        return allocated

    def free(self, block_ids: list):
        """归还物理块"""
        self.free_block_ids.extend(block_ids)

    @property
    def free_count(self):
        return len(self.free_block_ids)


class PagedKVCache:
    """请求级 KV cache —— 轻量句柄，接口兼容 NaiveKVCache

    用法:
        pool = KVCachePool(num_blocks=512, num_layers=28, ...)   # 全局创建一次
        cache = PagedKVCache(pool)                                # 每个请求创建一个
        # ... 正常使用 get_kv / update / advance_seq_len ...
        cache.free()                                              # 请求结束，归还块
    """

    def __init__(self, pool: KVCachePool):
        self._pool = pool
        self.seq_len = 0
        self.block_table = []    # [phys_id_0, phys_id_1, ...]

    def _allocate_block(self):
        new = self._pool.alloc(1)
        self.block_table.append(new[0])

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
        """
        _, _, S_new, _ = k_new.shape
        write_pos = self.seq_len
        end_pos = write_pos + S_new

        # 确保页表覆盖到 end_pos（页不够则从池申请）
        need_blocks = (end_pos + self._pool.block_size - 1) // self._pool.block_size
        while len(self.block_table) < need_blocks:
            self._allocate_block()

        # 向量化定位：每个 token 的逻辑位置 → (物理页, 页内偏移)
        # 用 advanced indexing 一次性写入，替代逐 token Python 循环
        pos = torch.arange(write_pos, end_pos, device=k_new.device)
        block_idx = pos // self._pool.block_size
        offset = pos % self._pool.block_size
        phys_ids = torch.tensor(self.block_table, device=k_new.device)[block_idx]

        # k_new[0]: (Hkv, S_new, D) → permute → (S_new, Hkv, D)
        # 与 pool.k_buffer[phys_ids, layer, :, offset] 的 (S_new, Hkv, D) 对齐
        self._pool.k_buffer[phys_ids, layer_idx, :, offset] = k_new[0].permute(1, 0, 2)
        self._pool.v_buffer[phys_ids, layer_idx, :, offset] = v_new[0].permute(1, 0, 2)

    def advance_seq_len(self, n: int = 1):
        self.seq_len += n

    def free(self):
        """归还所有物理块到共享池"""
        if self.block_table:
            self._pool.free(self.block_table)
            self.block_table = []
        self.seq_len = 0
