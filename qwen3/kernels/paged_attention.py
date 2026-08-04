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
    BLOCK_G: tl.constexpr,       # q tile 行数（tl.dot 要求 >= 16，G 不足补齐）
    scaling: tl.float32,
):
    pid = tl.program_id(0)
    b = pid // num_kv_heads       # batch 索引
    hkv = pid % num_kv_heads      # KV 头索引

    L = tl.load(seq_len_ptr + b).to(tl.int32)   # 标量：本请求有效 token 数（含新 token）
    num_pages = tl.maximum(0, (L + block_size - 1) // block_size)   # 标量：本请求页数

    offs_g = tl.arange(0, BLOCK_G)        # (BLOCK_G=16,) 行 tile 固定 16 行，多余行算完丢弃
    offs_d = tl.arange(0, head_dim)       # (head_dim,) 列偏移
    offs_p = tl.arange(0, block_size)     # (block_size,) 页内 token 偏移

    # GQA: 一次加载本 KV 头服务的 num_kv_groups 个 query（补齐到 BLOCK_G 行）
    q = tl.load(
        q_ptr
        + b * stride_q_b
        + (hkv * num_kv_groups + offs_g)[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=(offs_g[:, None] < num_kv_groups) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)    # q: (BLOCK_G, head_dim) fp32，前 num_kv_groups 行有效

    m = tl.full((BLOCK_G,), float("-inf"), dtype=tl.float32)      # (BLOCK_G,) 每 q 行一个 running max
    d_acc = tl.zeros((BLOCK_G,), dtype=tl.float32)                # (BLOCK_G,) 每 q 行一个归一化分母
    o = tl.zeros((BLOCK_G, head_dim), dtype=tl.float32)           # (BLOCK_G, head_dim) 未归一化分子

    for pg in range(num_pages):
        page_id = tl.load(
            block_table_ptr + b * stride_bt_b + pg * stride_bt_pg
        ).to(tl.int64)    # 标量：逻辑页 pg → 物理页号

        valid = tl.minimum(block_size, L - pg * block_size)       # 标量：本页有效 token 数
        token_mask = offs_p < valid                               # (block_size,) 布尔

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
        ).to(tl.float32)   # k: (block_size, head_dim) fp32

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
        ).to(tl.float32)   # v: (block_size, head_dim) fp32

        scores = tl.dot(q, tl.trans(k)) * scaling    # (BLOCK_G, block_size)：q 行 × 页内 token 列
        scores = tl.where(token_mask[None, :], scores, float("-inf"))   # 无效槽位 → -inf

        m_new = tl.maximum(m, tl.max(scores, axis=1))   # (BLOCK_G,) 新全局 max
        alpha = tl.exp(m - m_new)                       # (BLOCK_G,) rescale 因子
        exp_scores = tl.exp(scores - m_new[:, None])    # (BLOCK_G, block_size) 本页权重
        o = o * alpha[:, None] + tl.dot(exp_scores, v)  # (BLOCK_G, head_dim) 分子累积
        d_acc = d_acc * alpha + tl.sum(exp_scores, axis=1)  # (BLOCK_G,) 分母累积
        m = m_new

    o = o / d_acc[:, None]    # (BLOCK_G, head_dim) 归一化输出

    out_base = out_ptr + b * stride_o_b + (hkv * num_kv_groups) * stride_o_h
    tl.store(
        out_base
        + offs_g[:, None] * stride_o_h
        + offs_d[None, :] * stride_o_d,
        o.to(tl.bfloat16),    # 只写前 num_kv_groups 行（mask 保证）
        mask=(offs_g[:, None] < num_kv_groups) & (offs_d[None, :] < head_dim),
    )


def paged_attention_decode_triton(q, k_buffer, v_buffer, block_table, seq_len, block_size):
    """裸 kernel 入口：把 (B, num_heads, head_dim) 的 q 和分页数据喂给 Triton kernel

    Args（维度说明）:
        q:          (B, num_heads, head_dim) bf16
                    批内每请求 1 个 query 位置（decode，S 维已 squeeze 掉）
        k_buffer:   (num_blocks, num_kv_heads, block_size, head_dim) bf16
                    当前层的 K 物理页（pool.k_buffer[:, layer_idx] 的视图）
        v_buffer:   (num_blocks, num_kv_heads, block_size, head_dim) bf16
                    当前层的 V 物理页
        block_table:(B, max_pages) int32
                    每行一个请求的逻辑页表（物理页号），右 pad 0
        seq_len:    (B,) int32
                    每请求有效 token 数（含刚写入的新 token）
        block_size: 标量，每块含多少个 token（16）
    Returns:
        out:        (B, num_heads, head_dim) bf16，kernel 输出（与 q 同形状）

    流程：grid = (B × num_kv_heads,) 个 program，
    每个 program 处理一个 (b, hkv)，按 block_table[b, pg] 定位物理页逐页计算。
    """
    B, num_heads, head_dim = q.shape
    num_blocks, num_kv_heads, block_size_check, head_dim_check = k_buffer.shape
    assert block_size_check == block_size
    assert head_dim_check == head_dim
    assert v_buffer.shape == k_buffer.shape
    num_kv_groups = num_heads // num_kv_heads      # 每个 KV 头服务的 Q 头数（GQA）
    scaling = 1.0 / math.sqrt(head_dim)
    BLOCK_G = 16   # tl.dot 要求所有维度 >= 16；GQA group 通常 2/4，补齐到 16

    out = torch.empty_like(q)                      # (B, num_heads, head_dim)
    grid = (B * num_kv_heads,)                     # 一维网格：B 请求 × Hkv 头

    paged_attn_decode_kernel[grid](
        q, k_buffer, v_buffer, block_table, seq_len, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_buffer.stride(0), k_buffer.stride(1), k_buffer.stride(2), k_buffer.stride(3),
        v_buffer.stride(0), v_buffer.stride(1), v_buffer.stride(2), v_buffer.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_heads, num_kv_heads, head_dim, block_size, num_kv_groups, BLOCK_G, scaling,
        num_warps=2,
        num_stages=2,
    )
    return out


def triton_paged_attention_decode(q, k_new, v_new, kv_cache, layer_idx, scaling):
    """项目接口（单请求版）：与 qwen3/paged_attention.py 同签名，但仅支持 decode（S == 1）

    Args（维度说明）:
        q:      (B, num_heads, 1, head_dim) bf16 当前步的 Q（已过 QK-Norm + RoPE）
                第 2 维是 S（=1），decode 只有 1 个 query 位置
        k_new:  (B, num_kv_heads, 1, head_dim) bf16 当前步新 token 的 K
        v_new:  (B, num_kv_heads, 1, head_dim) bf16 当前步新 token 的 V
        kv_cache: PagedKVCache（请求级句柄，内含共享池 + 块表）
        layer_idx: 当前层号
        scaling: head_dim ** -0.5
    Returns:
        attn_output: (B, 1, num_heads * head_dim)，尚未过 o_proj
                    （保留 S=1 维，与 prefill 路径的 (B, S, hidden) 契约一致）

    单请求 = 批量长度为 1 的特例（B 恒为 1）。
    """
    assert q.shape[2] == 1, "Triton kernel 仅支持 decode（S == 1）"
    assert q.dtype == torch.bfloat16, "Triton kernel 仅支持 bf16"

    # 1. 先把本步新 token 的 K/V 写入物理页（页满自动申请）
    #    k_new 形状 (1, Hkv, 1, D) 直接满足 update 的 (B, Hkv, S_new, D) 契约
    kv_cache.update(layer_idx, k_new, v_new)

    # 2. 组装 kernel 输入（pool 是按层存的，取当前层视图）
    pool = kv_cache._pool
    k_buffer = pool.k_buffer[:, layer_idx]   # (num_blocks, num_kv_heads, block_size, head_dim)
    v_buffer = pool.v_buffer[:, layer_idx]   # 同上

    # 块表：Python list → (1, num_pages) int32（B=1，unsqueeze 加批维）
    block_table = torch.tensor(kv_cache.block_table, dtype=torch.int32, device=q.device)
    block_table = block_table.unsqueeze(0)   # (1, num_pages)
    # seq_len 包含刚写入的新 token（旧缓存长度 + 1）
    seq_len = torch.tensor([kv_cache.seq_len + 1], dtype=torch.int32, device=q.device)  # (1,)

    # 3. 跑 kernel（q 去掉 S 维：(B, H, 1, D) → (B, H, D)）
    out = paged_attention_decode_triton(
        q.squeeze(2), k_buffer, v_buffer, block_table, seq_len, pool.block_size)

    # 4. 还原为 (B, 1, hidden) 给 o_proj
    B, num_heads, head_dim = q.shape[0], q.shape[1], q.shape[3]
    return out.reshape(B, 1, num_heads * head_dim)


def triton_paged_attention_decode_batch(q, k_new, v_new, kv_caches, layer_idx, scaling):
    """批量 decode 入口：B 个请求一次 kernel（continuous batching 核心路径）

    Args（维度说明）:
        q:      (B, num_heads, 1, head_dim) bf16 批内各请求的 Q（已过 QK-Norm + RoPE）
        k_new:  (B, num_kv_heads, 1, head_dim) bf16 批内各请求新 token 的 K
        v_new:  (B, num_kv_heads, 1, head_dim) bf16 批内各请求新 token 的 V
        kv_caches: list[PagedKVCache]，长度 == B，每个请求自己的句柄
        layer_idx: 当前层号
        scaling: head_dim ** -0.5
    Returns:
        attn_output: (B, 1, num_heads * head_dim)，尚未过 o_proj

    与单请求版的关系：单请求只是 B=1 的特例。kernel 本体按
    (B, num_kv_heads) 网格 + 2D 块表设计，天然支持批量；
    这里只负责把 B 个请求的块表拼成 (B, max_pages) 并右 pad。

    为什么"一次 kernel 吃 B 个请求"是吞吐关键：
      单请求 decode 时 kernel grid = 1×Hkv = 8 个 program，A100 108 个 SM
      利用率 <8%；批场景 grid = B×Hkv，B=64 时 512 个 program，SM 吃饱。
      分页/Triton 的开销靠并发摊薄——这正是 continuous batching 的意义。
    """
    assert q.shape[2] == 1, "Triton kernel 仅支持 decode（S == 1）"
    assert q.dtype == torch.bfloat16, "Triton kernel 仅支持 bf16"
    B = q.shape[0]                               # 批大小（请求数）
    assert len(kv_caches) == B, "kv_caches 数量必须等于 batch"

    # ---- 1. 逐请求写入新 token 的 K/V（B-loop）----
    #    每个请求的 K/V 写进自己的块表（页写满自动从池申请新块）。
    #    注意顺序：必须先 update 再收集块表——页满时 update 会使块表变长，
    #    先写才能拿到真实块表（否则第 2 步拼的 (B, max_pages) 会漏掉新页）。
    #    切 k_new[i:i+1] 保留 batch 维：k_new[i] 是 (Hkv, 1, D) 丢掉批维，
    #    而 update 要求 (B=1, Hkv, S_new=1, D)，所以必须切片而不是取下标。
    for i, c in enumerate(kv_caches):
        c.update(layer_idx, k_new[i:i + 1], v_new[i:i + 1])

    # ---- 2. 组装 2D 块表: (B, max_pages) int32，右 pad 0 ----
    #    kernel 需要矩形张量，但各请求页数不同（seq_len 不同）→ 右 pad 0。
    #    安全：kernel 每请求只循环 ceil(seq_len/block_size) 页（运行时边界），
    #    pad 的列永远不会被读到——与 model.py 的 mask 补列同一哲学。
    #    数据来源：每个 PagedKVCache 的 Python list 块表 → 转 GPU int32 张量，
    #    拷进第 i 行的前 len(block_table) 列，剩余列保持 0。
    pool = kv_caches[0]._pool                    # 共享全局池（所有请求同一实例）
    max_pages = max(len(c.block_table) for c in kv_caches)   # 批内最长页表
    block_table = torch.zeros(B, max_pages, dtype=torch.int32, device=q.device)  # (B, max_pages)
    for i, c in enumerate(kv_caches):
        if c.block_table:
            block_table[i, :len(c.block_table)] = \
                torch.tensor(c.block_table, dtype=torch.int32, device=q.device)

    # ---- 3. 批内各请求的 seq_len: (B,) int32 ----
    #    kernel 每请求用它算 num_pages 和尾页 valid mask。
    #    seq_len + 1：update 已写入新 token，所以有效长度 = 旧缓存 + 1。
    seq_len = torch.tensor([c.seq_len + 1 for c in kv_caches],
                           dtype=torch.int32, device=q.device)   # (B,)

    # ---- 4. 跑 kernel（q 去掉 S 维：(B, H, 1, D) → (B, H, D)）----
    #    k_buffer[:, layer_idx] 取当前层的物理页视图
    #    (num_blocks, num_layers, Hkv, block_size, D) → (num_blocks, Hkv, block_size, D)
    #    pool 是所有请求共享的——kernel 通过每行块表定位各请求的物理页，
    #    请求之间物理上共存于同一池，逻辑上互不可见（块表隔离）。
    out = paged_attention_decode_triton(
        q.squeeze(2), pool.k_buffer[:, layer_idx], pool.v_buffer[:, layer_idx],
        block_table, seq_len, pool.block_size)
    # out: (B, num_heads, head_dim)

    # ---- 5. 还原为 (B, 1, hidden) 给 o_proj ----
    #    kernel 输出 (B, H, D) → reshape 成 (B, 1, H*D)
    #    与单请求版接口一致（o_proj 期望 (B, S, hidden)，decode 时 S=1）
    num_heads, head_dim = q.shape[1], q.shape[3]
    return out.reshape(B, 1, num_heads * head_dim)
