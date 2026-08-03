"""
Grouped Query Attention (GQA) + QK-Norm

GQA 是 MHA 和 MQA 的折中：
  - MHA (Multi-Head): 每个 Q 头有独立的 K, V 头 → 参数多，KV cache 大
  - MQA (Multi-Query): 所有 Q 头共享 1 组 K, V    → 参数少，但质量可能下降
  - GQA (Grouped):     每 G 个 Q 头共享 1 组 K, V  → 折中方案

Qwen3-8B: 32 个 Q heads, 8 个 KV heads → 每 4 个 Q 头共享一组 KV

QK-Norm (Qwen3 特有):
  在 RoPE 之前，对 Q 和 K 的每个头做 RMSNorm。稳定训练、提升质量。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import apply_rotary_pos_emb
from .PagedKVcache import PagedKVCache
from .paged_attention import paged_attention


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头复制 n_rep 次以匹配 Q 头数量 (用于 GQA)"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        # Q, K, V, O 线性投影
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm: 对每个头的 Q, K 做 RMSNorm (在 head_dim 上归一化)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,            # (batch, seq_len, hidden_size)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        attention_mask: torch.Tensor,            # (1, 1, seq_len, kv_len)
        kv_cache=None,                           # NaiveKVCache | None
        layer_idx: int = 0,                      # 当前是第几层
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # 投影 Q, K, V 并 reshape 成多头格式
        # (B, S, hidden) -> (B, S, heads, head_dim) -> (B, heads, S, head_dim)
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm (Qwen3 特有 — 在 RoPE 之前归一化 Q, K)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 应用 RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache: 分页则走 PagedAttention（逐物理页计算，不重建连续 K/V）
        if isinstance(kv_cache, PagedKVCache):
            attn_output = paged_attention(q, k, v, kv_cache, layer_idx,
                                          attention_mask, self.scaling)
            return self.o_proj(attn_output)

        # 朴素路径：先存新的（只存新 token 的 K,V），再拼旧的做 attention
        if kv_cache is not None:
            k_old, v_old = kv_cache.get_kv(layer_idx)
            kv_cache.update(layer_idx, k, v)    # 只存 k_new, v_new（S_new 个 token）
            if k_old is not None:
                # Decode 模式：拼上旧缓存，Q 才能看到所有历史
                k = torch.cat([k_old, k], dim=2)
                v = torch.cat([v_old, v], dim=2)

        # GQA: 将 KV 头复制以匹配 Q 头数量
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # 注意力计算: softmax(Q @ K^T / sqrt(d)) @ V
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # reshape 回 (B, S, hidden) 并投影输出
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn_output)
