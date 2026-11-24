"""
vLLM 风格 varlen（变长）FlashAttention — prefill 阶段融合 kernel

来源：test/Fusedattention.py 追加段（910–1079 行，拷自 myvllm 项目）。
本项目只需要这两段（flash_attention_varlen_kernel + flash_attention_prefill），
store_kvcache / paged_attention_decode / Attention 类与本项目现有实现重复，不引入。

与教程 _attn_fwd 的关键区别：
  - 输入是拉平的 (total_tokens, H, D) + cu_seqlens（累计序列长度），原生支持变长
  - 内核内 GQA（kv_head_idx = off_h // (num_heads // num_kv_heads)），无需 repeat_kv
  - causal 掩码在 kernel 内按序列内相对位置处理，不需要 CPU 侧 4D mask
  - 用 tl.exp（非 log2 域 exp2），代码简单、精度略低
  - 无 M（logsumexp）输出——推理不需要 backward

调用时机（attention.py 内 prefill 分支）:
  q/k/v 是 (B, H, S, D) 右 pad 批 → 本模块的 flash_attention_prefill_batched
  负责 gather 成拉平流 + 造 cu_seqlens + scatter 回原布局，调用方无需关心。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs
    start_m = tl.program_id(0)  # block index
    off_h = tl.program_id(1)    # head index
    seq_idx = tl.program_id(2)  # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # Load sequence boundaries
    # q 与 kv 可以有不同长度（前缀缓存：q = 未命中后缀，kv = 历史 + 后缀）
    q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    q_len = q_end - q_start
    kv_start = tl.load(cu_seqlens_k_ptr + seq_idx)
    kv_end = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    kv_len = kv_end - kv_start
    # 后缀在 kv 中的起始偏移（无历史时为 0）
    q_offset = kv_len - q_len

    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= q_len:
        return

    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (q_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]

    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < q_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Number of blocks to process（按 kv 长度遍历）
    num_blocks = tl.cdiv(kv_len, BLOCK_N)

    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Mask for valid positions
        mask_n = offs_n < kv_len

        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (kv_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]

        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale

        # Apply causal mask: query 的全局 kv 位置 = 后缀偏移 + offs_m
        mask_causal = (offs_m[:, None] + q_offset) >= offs_n[None, :]
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # Rescale previous accumulator
        acc = acc * alpha[:, None]

        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (kv_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)

        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (q_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.

    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: q 的 cumulative sequence lengths
        scale: attention scale factor
        cu_seqlens_k: k/v 的 cumulative lengths（None = 与 q 相同；
                      前缀缓存时 kv = 历史 + 后缀 > q）

    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate output
    output = torch.empty_like(q)

    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs

    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # Number of sequences
    num_seqs = cu_seqlens.shape[0] - 1

    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        cu_seqlens if cu_seqlens_k is None else cu_seqlens_k,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


def flash_attention_prefill_batched(
    q, k, v, lens, scaling, num_heads, num_kv_heads, head_dim
):
    """项目接口：把 (B, H, S, D) 的右 pad prefill 批转成 varlen 拉平流喂 kernel

    Args（维度说明）:
        q:       (B, num_heads, S, D) bf16 当前批的 Q（已过 QK-Norm + RoPE）
        k:       (B, num_kv_heads, S, D) bf16 当前批的 K（未写页，原始连续张量）
        v:       (B, num_kv_heads, S, D) bf16 当前批的 V
        lens:    list[int]，批内各请求有效长度（右 pad 后必须传，对应 input_lens）
        scaling: head_dim ** -0.5
        num_heads / num_kv_heads / head_dim: 模型头参数
    Returns:
        attn_output: (B, S, num_heads * head_dim)，尚未过 o_proj
                    （与 _standard_attention 输出形状一致）

    流程：
      ① (B, H, S, D) 右 pad 批 → permute/reshape 成 (B*S, H, D) 行布局
      ② 按 lens 构造有效 token mask → gather 掉 pad → (total_tokens, H, D)
      ③ cu_seqlens = [0, l1, l1+l2, ...]，kernel 内按序列边界各自算
      ④ 输出 scatter 回 (B, S, H*D)
    优化：全部等长（或 B==1）时无 pad，跳过 gather，直接 reshape + 规则 cu_seqlens。
    """
    B, H, S, D = q.shape
    H_kv = k.shape[1]
    assert len(lens) == B, "lens 数量必须等于 batch"
    assert all(0 < l <= S for l in lens), "lens 越界（应满足 0 < lens_i <= S）"

    # ---- 1. 统一转成行布局 (B*S, H, D) ----
    q2 = q.permute(0, 2, 1, 3).reshape(B * S, H, D)   # 行 r = b*S + s
    k2 = k.permute(0, 2, 1, 3).reshape(B * S, H_kv, D)
    v2 = v.permute(0, 2, 1, 3).reshape(B * S, H_kv, D)

    # ---- 2. 等长批：无 pad，直接整块喂 ----
    if all(l == S for l in lens):
        q_f, k_f, v_f = q2, k2, v2
        # cu_seqlens = [0, S, 2S, ..., B*S]
        cu_seqlens = torch.arange(B + 1, device=q.device, dtype=torch.int32) * S
    else:
        # ---- 变长批：gather 掉右 pad 的无效 token ----
        s_idx = torch.arange(S, device=q.device)                       # (S,)
        lens_t = torch.tensor(lens, device=q.device, dtype=torch.long)  # (B,)
        valid = (s_idx[None, :] < lens_t[:, None]).reshape(-1)         # (B*S,) 布尔
        q_f, k_f, v_f = q2[valid], k2[valid], v2[valid]
        # cu_seqlens: 累计序列长度
        cu = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=q.device), 0, out=cu[1:])
        cu_seqlens = cu

    # ---- 3. 跑 kernel ----
    out_f = flash_attention_prefill(
        q_f, k_f, v_f, cu_seqlens, scaling, num_heads, num_kv_heads, head_dim)

    # ---- 4. scatter 回 (B, S, H*D) ----
    #    内核输出 (total_tokens, H, D)，等长时行序即 (b, s)，变长时按 valid 写回
    if all(l == S for l in lens):
        out = out_f.reshape(B, S, H, D)
    else:
        out = torch.zeros(B * S, H, D, dtype=q.dtype, device=q.device)
        out[valid] = out_f
        out = out.reshape(B, S, H, D)
    return out.reshape(B, S, H * D)


def flash_attention_prefill_with_history(
    q, k_new, v_new, lens, hist_kv, scaling, num_heads, num_kv_heads, head_dim
):
    """前缀缓存部分命中的 prefill：q = 未命中后缀，KV = 历史 + 后缀

    Args:
        q:           (B, H, S, D) 后缀 query（右 pad；lens 给出有效长度）
        k_new/v_new: (B, Hkv, S, D) 后缀 K/V
        lens:        list[int] 各请求后缀有效长度
        hist_kv:     list[(k, v)]，每请求历史 (1, Hkv, hist_i, D)（hist_i 可为 0）
        scaling:     head_dim ** -0.5

    Returns:
        (B, S, H*D)，与 flash_attention_prefill_batched 一致

    实现：历史 KV（gather 自分页池）与后缀拼接成连续张量，
    cu_seqlens_q / cu_seqlens_k 分别给出 q / kv 长度，
    kernel 内 causal 使用 q_offset = kv_len - q_len。
    """
    B, H, S, D = q.shape
    assert len(lens) == B and len(hist_kv) == B

    # ---- 1. q flatten（去 pad）----
    q2 = q.permute(0, 2, 1, 3).reshape(B * S, H, D)
    lens_t = torch.tensor(lens, dtype=torch.long, device=q.device)
    s_idx = torch.arange(S, device=q.device)
    valid_q = (s_idx[None, :] < lens_t[:, None]).reshape(-1)
    q_f = q2[valid_q]

    cu_q = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    torch.cumsum(lens_t.to(torch.int32), 0, out=cu_q[1:])

    # ---- 2. KV：历史 + 后缀（逐请求拼接，再 flatten）----
    k_parts, v_parts, kv_lens = [], [], []
    for i in range(B):
        hk, hv = hist_kv[i]
        li = lens[i]
        k_parts.append(torch.cat([hk[0], k_new[i, :, :li]], dim=1))
        v_parts.append(torch.cat([hv[0], v_new[i, :, :li]], dim=1))
        kv_lens.append(hk.shape[2] + li)

    k_all = torch.cat(k_parts, dim=1)          # (Hkv, total_kv, D)
    v_all = torch.cat(v_parts, dim=1)
    k_f = k_all.permute(1, 0, 2).contiguous()  # (total_kv, Hkv, D)
    v_f = v_all.permute(1, 0, 2).contiguous()

    cu_k = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    torch.cumsum(torch.tensor(kv_lens, dtype=torch.int32, device=q.device),
                 0, out=cu_k[1:])

    # ---- 3. kernel（q/kv 长度分离）----
    out_f = flash_attention_prefill(
        q_f, k_f, v_f, cu_q, scaling, num_heads, num_kv_heads, head_dim,
        cu_seqlens_k=cu_k)

    # ---- 4. scatter 回 (B, S, H*D) ----
    out = torch.zeros(B * S, H, D, dtype=q.dtype, device=q.device)
    out[valid_q] = out_f
    return out.reshape(B, S, H * D)
