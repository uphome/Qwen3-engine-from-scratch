"""
KV Cache — 最朴素的实现：每层一个 list，新 token 的 KV 拼到后面
"""

import torch


class NaiveKVCache:
    """最简 KV Cache：每层存 (B, num_kv_heads, seq_len, head_dim) 的 K/V tensor"""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache = [None] * num_layers   # 每层一个 tensor 或 None
        self.v_cache = [None] * num_layers
        self.seq_len = 0                      # 当前缓存了多少个 token

    def get_kv(self, layer_id: int):
        """返回该层已缓存的 K, V（可能为 None）"""
        return self.k_cache[layer_id], self.v_cache[layer_id]

    def update(self, layer_id: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """将新 token 的 K, V 拼到缓存末尾"""
        # k_new, v_new: (B, num_kv_heads, S_new, head_dim)，S_new 通常为 1
        if self.k_cache[layer_id] is None:
            self.k_cache[layer_id] = k_new
            self.v_cache[layer_id] = v_new
        else:
            self.k_cache[layer_id] = torch.cat([self.k_cache[layer_id], k_new], dim=2)
            self.v_cache[layer_id] = torch.cat([self.v_cache[layer_id], v_new], dim=2)

    def advance_seq_len(self, n: int = 1):
        self.seq_len += n
