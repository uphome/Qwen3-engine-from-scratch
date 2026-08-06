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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import apply_rotary_pos_emb
from .PagedKVcache import PagedKVCache
from .paged_attention import paged_attention
from .kernels.paged_attention import triton_paged_attention_decode, triton_paged_attention_decode_batch


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
        kv_cache=None,                           # PagedKVCache | list[PagedKVCache] | None
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
            # 环境变量 QWEN3_PAGED_ATTN=pytorch 可强制走 PyTorch 版（消融对比用）
            use_triton = (
                S == 1 and kv_cache.seq_len > 0 and q.is_cuda
                and os.environ.get("QWEN3_PAGED_ATTN", "triton") == "triton"
            )
            if use_triton:
                # decode：Triton kernel（bf16 GPU 专用，内部负责 update）
                attn_output = triton_paged_attention_decode(
                    q, k, v, kv_cache, layer_idx, self.scaling)
            elif S > 1:
                # prefill：K/V 本来就是连续张量，直接标准 attention（vLLM 同款做法），
                # 顺带写入分页供 decode 使用；不做"写页→逐页读回"的白折腾
                kv_cache.update(layer_idx, k, v)
                attn_output = self._standard_attention(q, k, v, attention_mask)
            else:
                # decode 兜底（CPU/非 bf16/强制 PyTorch）：逐页实现
                attn_output = paged_attention(q, k, v, kv_cache, layer_idx,
                                              attention_mask, self.scaling)
            return self.o_proj(attn_output)

        # 批量路径：kv_cache 是 list[PagedKVCache]，批内每请求一个句柄
        # （continuous batching 核心路径；单请求是 B=1 的特例）
        # 与单请求分支（上面）结构完全平行，只是把"1 个请求"换成"B 个请求"：
        #   decode 主路径 → Triton 一次 kernel 吃 B 请求（吞吐关键）
        #   prefill        → B-loop 写页 + 批 matmul 标准 attention
        #   decode 兜底    → 逐请求 paged_attention + cat 拼回
        if isinstance(kv_cache, (list, tuple)):
            caches = list(kv_cache)
            assert len(caches) == q.shape[0], "kv_caches 数量必须等于 batch"

            # ---- decode 批：Triton 一次 kernel 吃 B 个请求（吞吐核心路径）----
            # kernel 本体按 (B, num_kv_heads) 网格 + 2D 块表设计，天然支持批量；
            # batch wrapper 只多做两件事：① 拼 2D 块表（右 pad 0）② 收集 seq_len 数组。
            # 效果：B 个请求合并成一次 kernel 启动，且 grid = B×Hkv 个 program，
            #       解决单请求（B=1）时只有 8 个 program、SM 利用率 <8% 的问题。
            use_triton = (
                S == 1 and caches[0].seq_len > 0 and q.is_cuda
                and os.environ.get("QWEN3_PAGED_ATTN", "triton") == "triton"
            )
            if use_triton:
                attn_output = triton_paged_attention_decode_batch(
                    q, k, v, caches, layer_idx, self.scaling)
            elif S > 1:
                # ---- prefill 批：B-loop 写页 + 批 matmul 标准 attention ----
                # ① 逐请求写页（B-loop）：正确性优先。各请求块表长度不同，
                #    真 batched update 需要拼接批量写入，留作后续优化
                # ② 标准 attention 是批的：q/k/v 都是 (B, H, S, D)，
                #    matmul 一次算 B 个请求（大矩阵吃满 cuBLAS）——性能关键在这
                for i, c in enumerate(caches):
                    c.update(layer_idx, k[i:i + 1], v[i:i + 1])
                attn_output = self._standard_attention(q, k, v, attention_mask)
            else:
                # ---- decode 批兜底（CPU/非 bf16）：逐请求 PyTorch 逐页实现 ----
                # ① mask 切列：model.py 造的是 (B,1,S,max_kv) 矩形 mask
                #   （短请求补的列已置 -inf），但 paged_attention 要求 mask 的
                #   kv 维 == 实际可见长度，所以逐请求切成自己的 kv_len
                # ② torch.cat 拼回：B 个 (1,S,hidden) 输出 cat 成 (B,S,hidden)
                outs = []
                for i, c in enumerate(caches):
                    mask_i = attention_mask[i:i + 1, :, :, :c.seq_len + S]
                    outs.append(paged_attention(q[i:i + 1], k[i:i + 1], v[i:i + 1],
                                                c, layer_idx, mask_i, self.scaling))
                attn_output = torch.cat(outs, dim=0)
            return self.o_proj(attn_output)

        # 无 kv_cache（纯前向测试）：直接标准 attention
        attn_output = self._standard_attention(q, k, v, attention_mask)
        return self.o_proj(attn_output)

    def _standard_attention(self, q, k, v, attention_mask):
        """连续 K/V 的标准 attention: softmax(Q @ K^T / sqrt(d)) @ V"""
        # GQA: 将 KV 头复制以匹配 Q 头数量
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # 注意力计算: softmax(Q @ K^T / sqrt(d)) @ V
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # reshape 回 (B, S, hidden)
        B, S, _ = q.shape[0], q.shape[2], None
        return attn_output.transpose(1, 2).reshape(B, S, -1)
