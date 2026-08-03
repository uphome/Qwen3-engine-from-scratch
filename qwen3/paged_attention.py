"""
PagedAttention — 纯 PyTorch 教学实现

核心思想：
  标准实现把分页 K/V 重建为连续张量 (torch.cat) 再做 attention；
  PagedAttention 拿着块表 (block_table) 逐物理页计算，用 online-softmax
  边算边合并，不重建连续 K/V。数学上与标准实现完全等价。

  (Accumulative) online-softmax 合并（与 FlashAttention 相同）：
    每页只给出一部分分数的信息，靠三个累积状态跨页传递：
      m   = 迄今所有分数的最大值
      l   = 迄今 exp(score - m) 的和（归一化分母）
      acc = 迄今 exp(score - m) * v 的和（未归一化输出）
    页到来时按新的全局最大值 m' 统一缩放旧状态，保证数值稳定。

推导要点:
  out = A / L，其中 A = Σ_j exp(s_j)·v_j，L = Σ_j exp(s_j)
  L、A 可逐页累加；为防止 exp 溢出，引入 running max m，
  所有状态都表示在"已减去 m"的坐标下。
  新页 max m' 更大时，旧状态乘 exp(m − m') 换算回新坐标。
  最终 out = acc / l，m 在比值中约掉 → 与标准 softmax 等价。

  本文件是 vLLM CUDA kernel (vllm/attention/ops/paged_attn/paged_attention_kernel.cu)
  的 Python 模拟：逐页循环在 Python 层，性能不如 kernel，但逻辑一一对应。
"""

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA 扩头：KV 头复制 n_rep 份匹配 Q 头数"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def paged_attention(q, k_new, v_new, kv_cache, layer_idx, attention_mask, scaling=1.0):
    """直接在物理页上计算 attention，不重建连续 K/V。

    Args:
        q:            (B, num_heads, S, head_dim) 当前步的 Q（已过 QK-Norm + RoPE）
        k_new/v_new:  (B, num_kv_heads, S, head_dim) 当前步新 token 的 K/V
        kv_cache:     PagedKVCache（请求级句柄：共享池 + 块表）
        layer_idx:    当前层号
        attention_mask: (1, 1, S, kv_len)，kv_len = 已缓存长度 + S
        scaling:      head_dim ** -0.5
    Returns:
        attn_output:  (B, S, num_heads * head_dim)，尚未过 o_proj
    """
    B, H, S, D = q.shape
    Hk = k_new.shape[1]
    groups = H // Hk
    page_size = kv_cache._pool.block_size

    # ---- 0. 把本步新 token 的 K/V 写入物理页（kernel 里的 KV 存储阶段）----
    #      写入位置 = kv_cache.seq_len 起；页写满会自动从池申请新页
    kv_cache.update(layer_idx, k_new, v_new)

    pool = kv_cache._pool
    block_table = kv_cache.block_table            # [phys_id_0, phys_id_1, ...]
    kv_total = kv_cache.seq_len + S               # 有效 token 总数（含刚写入的）
    num_pages = len(block_table)                  #当前 块数长度 

    # ---- 1. 把 attention mask 补齐到"页槽位总数" ----
    #      最后一个物理页可能没写满，未写入的槽位置 -inf（softmax 权重为 0）
    total_slots = num_pages * page_size
    if attention_mask.shape[-1] < total_slots:
        mask_full = F.pad(attention_mask, (0, total_slots - attention_mask.shape[-1]),
                          value=float("-inf"))
    else:
        mask_full = attention_mask

    # ---- 2. online-softmax 的三个累积状态（每个 query 行各一份）----
    #      不变量：任意时刻 acc / l 都精确等于"已见所有页"的 softmax 输出
    #      状态处于"已减 m"坐标下 → exp 值域 ≤ 1，不会溢出
    #      统一在 float32 里合并，避免 bf16 精度损失（与标准 softmax(dtype=f32) 对齐）
    m = torch.full((B, H, S, 1), float("-inf"), dtype=torch.float32, device=q.device)
    l = torch.zeros((B, H, S, 1), dtype=torch.float32, device=q.device)
    acc = torch.zeros((B, H, S, D), dtype=torch.float32, device=q.device)

    # ---- 3. 逐物理页：部分分数 → online-softmax 合并 ----
    for j, phys_id in enumerate(block_table):  #第一个是 逻辑ID  第二个是物理ID
        # 该物理页当前层的 K/V: [Hk, page_size, D]，GQA 扩头 → [1, H, page_size, D]
        k_page = repeat_kv(pool.k_buffer[phys_id, layer_idx].unsqueeze(0), groups)
        v_page = repeat_kv(pool.v_buffer[phys_id, layer_idx].unsqueeze(0), groups)

        # 部分分数 s^(p) = Q @ K_pageᵀ * scale ∈ R^B（一页的局部得分）
        s = (torch.matmul(q, k_page.transpose(-2, -1)) * scaling).float()
        s = s + mask_full[:, :, :, j * page_size:(j + 1) * page_size]

        # online-softmax 合并（对照推导 ①②③④）：
        # ① 新全局最大值 m' = max(m, 本页最大值 m_p)
        m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        # ② rescale 因子 e^(m−m')：把旧状态从"减 m"坐标换算到"减 m'"坐标
        alpha = torch.exp(m - m_new)
        # 本页权重 e^(s−m')（被 mask 掉的槽位 exp(−∞)=0，自动不贡献）
        p = torch.exp(s - m_new)
        # ③ 分母累加：旧·缩放 + 本页 exp 和
        l = l * alpha + p.sum(dim=-1, keepdim=True)
        # ③ 分子累加：旧·缩放 + 本页 exp·v 和
        acc = acc * alpha + torch.matmul(p, v_page)
        # ④ 状态前滚
        m = m_new

    # ---- 4. 归一化 + 还原为 (B, S, hidden) ----
    attn_output = (acc / l).to(q.dtype)                   # [B, H, S, D]
    return attn_output.transpose(1, 2).reshape(B, S, -1)
