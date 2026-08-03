"""
Triton PagedAttention — decode kernel（接入项目用）

与 test/paged_attention_triton.py 的教学版同一 kernel，这里做了一层
"项目接口"适配：直接吃 attention.py 传来的 (q, k_new, v_new, kv_cache)。

约束:
  - 仅 decode（S == 1，每个请求 1 个 query 位置）
  - 仅 bf16 + CUDA（kernel 里写回 o.to(tl.bfloat16)）
  - prefill 仍走 qwen3/paged_attention.py（PyTorch 版）

调用时机（attention.py 内）:
  decode 步: update 写入新 token 的 K/V → 逐物理页算 attention
  （本模块的 wrapper 内部负责 update，调用方无需关心）
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def paged_attn_decode_kernel(
    q_ptr,               # (B, num_heads, head_dim) bf16
    k_buffer_ptr,        # (num_blocks, num_kv_heads, block_size, head_dim) bf16
    v_buffer_ptr,        # (num_blocks, num_kv_heads, block_size, head_dim) bf16
    block_table_ptr,     # (B, max_pages) int32
    seq_len_ptr,         # (B,) int32
    out_ptr,             # (B, num_heads, head_dim) bf16
    stride_q_b, stride_q_h, stride_q_d,
    stride_kb_blk, stride_kb_h, stride_kb_p, stride_kb_d,
    stride_vb_blk, stride_vb_h, stride_vb_p, stride_vb_d,
    stride_bt_b, stride_bt_pg,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_kv_groups: tl.constexpr,
    scaling: tl.float32,
):
    pid = tl.program_id(0)
    b = pid // num_kv_heads
    hkv = pid % num_kv_heads

    L = tl.load(seq_len_ptr + b).to(tl.int32)
    num_pages = tl.maximum(0, (L + block_size - 1) // block_size)

    offs_g = tl.arange(0, num_kv_groups)
    offs_d = tl.arange(0, head_dim)
    offs_p = tl.arange(0, block_size)

    # GQA: 一次加载本 KV 头服务的 num_kv_groups 个 query
    q = tl.load(
        q_ptr
        + b * stride_q_b
        + (hkv * num_kv_groups + offs_g)[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=(offs_g[:, None] < num_kv_groups) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    m = tl.full((num_kv_groups,), float("-inf"), dtype=tl.float32)
    d_acc = tl.zeros((num_kv_groups,), dtype=tl.float32)
    o = tl.zeros((num_kv_groups, head_dim), dtype=tl.float32)

    for pg in range(num_pages):
        page_id = tl.load(
            block_table_ptr + b * stride_bt_b + pg * stride_bt_pg
        ).to(tl.int64)

        valid = tl.minimum(block_size, L - pg * block_size)
        token_mask = offs_p < valid

        k_ptrs = (
            k_buffer_ptr
            + page_id * stride_kb_blk
            + hkv * stride_kb_h
            + offs_p[:, None] * stride_kb_p
            + offs_d[None, :] * stride_kb_d
        )
        k = tl.load(
            k_ptrs,
            mask=(offs_p[:, None] < block_size) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        v_ptrs = (
            v_buffer_ptr
            + page_id * stride_vb_blk
            + hkv * stride_vb_h
            + offs_p[:, None] * stride_vb_p
            + offs_d[None, :] * stride_vb_d
        )
        v = tl.load(
            v_ptrs,
            mask=(offs_p[:, None] < block_size) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scaling
        scores = tl.where(token_mask[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        exp_scores = tl.exp(scores - m_new[:, None])
        o = o * alpha[:, None] + tl.dot(exp_scores, v)
        d_acc = d_acc * alpha + tl.sum(exp_scores, axis=1)
        m = m_new

    o = o / d_acc[:, None]

    out_base = out_ptr + b * stride_o_b + (hkv * num_kv_groups) * stride_o_h
    tl.store(
        out_base
        + offs_g[:, None] * stride_o_h
        + offs_d[None, :] * stride_o_d,
        o.to(tl.bfloat16),
        mask=(offs_g[:, None] < num_kv_groups) & (offs_d[None, :] < head_dim),
    )


def paged_attention_decode_triton(q, k_buffer, v_buffer, block_table, seq_len, block_size):
    """裸 kernel 入口（形状对齐 test 版）：q 为 (B, num_heads, head_dim)"""
    B, num_heads, head_dim = q.shape
    num_blocks, num_kv_heads, block_size_check, head_dim_check = k_buffer.shape
    assert block_size_check == block_size
    assert head_dim_check == head_dim
    assert v_buffer.shape == k_buffer.shape
    num_kv_groups = num_heads // num_kv_heads
    scaling = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    grid = (B * num_kv_heads,)

    paged_attn_decode_kernel[grid](
        q, k_buffer, v_buffer, block_table, seq_len, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_buffer.stride(0), k_buffer.stride(1), k_buffer.stride(2), k_buffer.stride(3),
        v_buffer.stride(0), v_buffer.stride(1), v_buffer.stride(2), v_buffer.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_heads, num_kv_heads, head_dim, block_size, num_kv_groups, scaling,
        num_warps=2,
        num_stages=2,
    )
    return out


def triton_paged_attention_decode(q, k_new, v_new, kv_cache, layer_idx, scaling):
    """项目接口：与 qwen3/paged_attention.py 同签名，但仅支持 decode（S == 1）

    Args:
        q:      (B, num_heads, 1, head_dim) bf16 当前步的 Q（已过 QK-Norm + RoPE）
        k_new:  (B, num_kv_heads, 1, head_dim) bf16
        v_new:  (B, num_kv_heads, 1, head_dim) bf16
        kv_cache: PagedKVCache（请求级句柄）
        layer_idx: 当前层号
        scaling: head_dim ** -0.5
    Returns:
        attn_output: (B, 1, num_heads * head_dim)，尚未过 o_proj
    """
    assert q.shape[2] == 1, "Triton kernel 仅支持 decode（S == 1）"
    assert q.dtype == torch.bfloat16, "Triton kernel 仅支持 bf16"

    # 1. 先把本步新 token 的 K/V 写入物理页（页满自动申请）
    kv_cache.update(layer_idx, k_new, v_new)

    # 2. 组装 kernel 输入（pool 是按层存的，取当前层视图）
    pool = kv_cache._pool
    k_buffer = pool.k_buffer[:, layer_idx]   # (num_blocks, num_kv_heads, block_size, head_dim)
    v_buffer = pool.v_buffer[:, layer_idx]

    # 块表：list → (1, num_pages) int32；seq_len 包含刚写入的新 token
    block_table = torch.tensor(kv_cache.block_table, dtype=torch.int32, device=q.device)
    block_table = block_table.unsqueeze(0)   # (1, num_pages)
    seq_len = torch.tensor([kv_cache.seq_len + 1], dtype=torch.int32, device=q.device)

    # 3. 跑 kernel（q 去掉 S 维）
    out = paged_attention_decode_triton(
        q.squeeze(2), k_buffer, v_buffer, block_table, seq_len, pool.block_size)

    # 4. 还原为 (B, 1, hidden) 给 o_proj
    B, num_heads, head_dim = q.shape[0], q.shape[1], q.shape[3]
    return out.reshape(B, 1, num_heads * head_dim)
