"""
采样模块 — 把采样逻辑尽量搬到 GPU，并减少 kernel 启动次数。

背景：
  当前采样散落在 generate.py / server.py / batched_generate.py 中，每个请求
  单独做 temperature / top-k / top-p / softmax / multinomial，B 个请求会放大
  成 8×B 次左右的小算子启动。

本模块提供：
  1. sample_tokens(): 统一入口，支持标量或每请求不同的采样参数。
  2. _sample_rows_torch(): 批量 PyTorch 版本，作为 CPU/回退路径。
  3. _sample_topk_kernel: Triton kernel，把 top-p + softmax + Gumbel-max
     采样融合成一次 kernel，配合 batch 化的 torch.topk 将每步采样压缩到
     约 2 次 GPU 启动。

随机性说明：
  Triton 路径使用 Gumbel-max 技巧：
    argmax_i (logits_i + Gumbel_noise_i) ~ Categorical(softmax(logits))
  数学分布与 torch.multinomial 等价，但随机数流不同；greedy / top_k=1 仍然
  与旧实现逐 token 一致。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # CPU 环境 / 未安装 Triton
    triton = None
    tl = None
    _HAS_TRITON = False


# Triton 路径支持的最大 top_k。top_k=0 或超过该值时回退到批量 torch 路径。
_MAX_TRITON_K = 1024

# 自动随机种子计数器：sample_tokens(seed=None) 时使用，保证每步随机流不同。
_SEED_COUNTER = 0


def _next_seed() -> int:
    global _SEED_COUNTER
    _SEED_COUNTER += 1
    return (torch.initial_seed() + _SEED_COUNTER * 0x9E3779B1) & 0x7FFFFFFF


def _as_batch_param(value, batch_size: int, device, dtype):
    """把标量 / list / tuple / tensor 统一成 (B,) 张量。"""
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(batch_size)
        return t
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, device=device, dtype=dtype)
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def _param_to_pylist(value, batch_size: int):
    """把参数转成 Python list，避免在 GPU tensor 上做 int()/item() 同步。"""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return None
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * batch_size


def _sample_rows_torch(logits: torch.Tensor, temperature: torch.Tensor,
                       top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """批量 PyTorch 版本：作为 CPU 路径与 Triton 不适用时的回退。"""
    B, V = logits.shape
    device = logits.device

    greedy = temperature <= 0
    safe_temp = torch.where(greedy, torch.ones_like(temperature), temperature)
    scaled = logits.float() / safe_temp[:, None]

    k_eff = torch.where(top_k > 0, top_k, torch.full_like(top_k, V))
    k_eff = k_eff.clamp(min=1, max=V)
    max_k = int(k_eff.max().item())

    topk_vals, topk_idx = torch.topk(scaled, max_k, dim=-1, largest=True, sorted=True)

    if max_k < V:
        col = torch.arange(max_k, device=device).unsqueeze(0)
        topk_vals = topk_vals.masked_fill(col >= k_eff.unsqueeze(1), float("-inf"))

    if bool((top_p < 1.0).any().item()):
        probs = F.softmax(topk_vals, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        remove = cumprobs > top_p[:, None]
        remove = remove.clone()
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        topk_vals = topk_vals.masked_fill(remove, float("-inf"))

    probs = F.softmax(topk_vals, dim=-1)
    sampled_local = torch.multinomial(probs, num_samples=1)
    sampled = topk_idx.gather(1, sampled_local)

    greedy_token = logits.argmax(dim=-1, keepdim=True)
    return torch.where(greedy.unsqueeze(1), greedy_token, sampled)


if _HAS_TRITON:
    @triton.jit
    def _sample_topk_kernel(
        topk_vals_ptr,       # (B, max_k)  logits 的 top-k 值（降序）
        topk_idx_ptr,        # (B, max_k)  对应 vocab 下标
        params_ptr,          # (4, B) float32: [temperature, k_eff, top_p, greedy]
        out_ptr,             # (B,) int64
        stride_vals_row,
        stride_idx_row,
        B,
        max_k,
        seed,
        BLOCK_K: tl.constexpr,
    ):
        """每个 program 处理一个请求。

        在 fusion kernel 内完成：
          temperature 缩放 → k_eff mask → softmax → top-p mask →
          Gumbel-max 采样 → gather 回 vocab token id
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < max_k

        vals = tl.load(
            topk_vals_ptr + row * stride_vals_row + offs,
            mask=mask, other=float("-inf")).to(tl.float32)
        idx = tl.load(
            topk_idx_ptr + row * stride_idx_row + offs,
            mask=mask, other=0)

        # params 布局 (4, B)：一次 HtoD 拷入全部逐请求采样参数。
        temp = tl.load(params_ptr + row).to(tl.float32)
        k_eff = tl.load(params_ptr + B + row).to(tl.int32)
        top_p = tl.load(params_ptr + 2 * B + row).to(tl.float32)
        greedy = tl.load(params_ptr + 3 * B + row).to(tl.int32)

        # temperature 缩放；greedy 行用 1.0 占位，避免除 0
        temp_safe = tl.where(temp > 0.0, temp, 1.0)
        x = vals / temp_safe

        # 只保留 top_k 范围内的候选
        valid = mask & (offs < k_eff)
        x = tl.where(valid, x, float("-inf"))

        # softmax
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        s = tl.sum(e, axis=0)
        p = e / s

        # top-p：keep[0]=True；keep[j] = cumsum(p[:j]) <= top_p
        c = tl.cumsum(p, axis=0)
        c_prev = c - p
        keep = (offs == 0) | (c_prev <= top_p)
        x = tl.where(keep, x, float("-inf"))

        # Gumbel-max：argmax(x + gumbel_noise)
        u = tl.rand(seed, row * BLOCK_K + offs)
        u = tl.maximum(u, 1.0e-7)
        g = -tl.log(-tl.log(u))
        score = x + g
        best = tl.argmax(score, axis=0)

        sampled = tl.load(topk_idx_ptr + row * stride_idx_row + best)
        greedy_idx = tl.load(topk_idx_ptr + row * stride_idx_row)
        out = tl.where(greedy != 0, greedy_idx, sampled)
        tl.store(out_ptr + row, out)


def _sample_tokens_triton(logits: torch.Tensor, temps_list, ks, ps_list,
                          seed: int | None) -> torch.Tensor | None:
    """尝试走 Triton 快速路径；不满足条件时返回 None，由调用方回退 torch。

    参数由 sample_tokens() 预先转成 Python list，避免 GPU tensor 上做 item() 同步。
    """
    if not _HAS_TRITON or not logits.is_cuda:
        return None

    B, V = logits.shape

    # top_k=0 或超过支持范围时回退。
    for k in ks:
        if k is None or int(k) <= 0 or int(k) > _MAX_TRITON_K:
            return None

    max_k = min(max(int(k) for k in ks), V)
    if max_k <= 0:
        return None

    device = logits.device
    # 一次 HtoD 把 temperature / k_eff / top_p / greedy 写入 (4, B)。
    params = torch.tensor(
        [   # row 0
            [float(t) for t in temps_list],
            # row 1
            [min(int(k), V) for k in ks],
            # row 2
            [float(p) for p in ps_list],
            # row 3
            [1.0 if float(t) <= 0 else 0.0 for t in temps_list],
        ],
        device=device, dtype=torch.float32,
    )

    # 只做一次 batch topk；temperature 缩放不改变 topk 顺序，可延后到 kernel 内。
    topk_vals, topk_idx = torch.topk(
        logits, max_k, dim=-1, largest=True, sorted=True)

    out = torch.empty(B, dtype=torch.int64, device=device)
    BLOCK_K = max(16, triton.next_power_of_2(max_k))
    if seed is None:
        seed = _next_seed()
    _sample_topk_kernel[(B,)](
        topk_vals, topk_idx, params, out,
        topk_vals.stride(0), topk_idx.stride(0),
        B, max_k, int(seed),
        BLOCK_K=BLOCK_K,
        num_warps=1,
    )
    return out.unsqueeze(1)


def sample_tokens(logits: torch.Tensor,
                  temperature=0.0,
                  top_k=0,
                  top_p=1.0,
                  seed: int | None = None) -> torch.Tensor:
    """对 (B, V) logits 批量采样，返回 (B, 1) token id。

    参数支持标量或长度 B 的 list/tuple/tensor。
    优先走 Triton kernel；不满足条件时回退到批量 PyTorch 实现。
    """
    if logits.ndim != 2:
        raise ValueError(f"logits 必须是 (B, V)，当前 shape={tuple(logits.shape)}")

    B, V = logits.shape
    device = logits.device

    # 先转成 Python list：Triton 路径不需要逐请求 item()/同步。
    temps_list = _param_to_pylist(temperature, B)
    ks_list = _param_to_pylist(top_k, B)
    ps_list = _param_to_pylist(top_p, B)

    if temps_list is not None and ks_list is not None and ps_list is not None:
        # 全 greedy：直接 argmax，最省。
        if all(float(t) <= 0 for t in temps_list):
            return logits.argmax(dim=-1, keepdim=True)

        # Triton 快速路径：top_p + softmax + Gumbel 采样融合成一次 kernel。
        triton_out = _sample_tokens_triton(
            logits, temps_list, ks_list, ps_list, seed)
        if triton_out is not None:
            return triton_out

    # 回退：批量 PyTorch 路径（支持 GPU tensor 参数 / top_k=0 / 超范围 top_k）。
    temperature_t = _as_batch_param(temperature, B, device, torch.float32)
    top_k_t = _as_batch_param(top_k, B, device, torch.long)
    top_p_t = _as_batch_param(top_p, B, device, torch.float32)
    return _sample_rows_torch(logits, temperature_t, top_k_t, top_p_t)
