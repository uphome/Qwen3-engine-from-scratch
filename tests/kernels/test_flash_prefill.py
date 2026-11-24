"""flash prefill kernel：普通 varlen / 带历史（q/kv 长度分离）数值正确性

覆盖：
  - flash_attention_prefill_batched vs PyTorch causal attention（普通路径）
  - flash_attention_prefill_with_history vs paged_attention_prefill_from_history
    （前缀缓存历史路径，q_len < kv_len）
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu

from qwen3.kernels.flash_attention_varlen import (
    flash_attention_prefill_batched,
    flash_attention_prefill_with_history,
)
from qwen3.paged_attention import paged_attention_prefill_from_history, repeat_kv


def reference_attention(q, k, v, scaling):
    """q (1,H,S,D) / k,v (1,Hkv,KV,D) → 标准 causal attention (1,S,H*D)

    q 的全局位置 = KV - S + i（历史 + 后缀）
    """
    H, Hkv = q.shape[1], k.shape[1]
    k = repeat_kv(k, H // Hkv)
    v = repeat_kv(v, H // Hkv)
    S, KV, D = q.shape[2], k.shape[2], q.shape[3]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scaling
    row = torch.arange(S, device=q.device).view(1, 1, S, 1) + (KV - S)
    col = torch.arange(KV, device=q.device).view(1, 1, 1, KV)
    scores = scores.masked_fill(col > row, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v)
    return out.transpose(1, 2).reshape(1, S, H * D)


def test_flash_prefill_matches_reference():
    torch.manual_seed(0)
    B, H, Hkv, S, D = 3, 4, 2, 5, 16
    lens = [5, 3, 1]
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    scaling = D ** -0.5

    out = flash_attention_prefill_batched(q, k, v, lens, scaling, H, Hkv, D)
    assert out.shape == (B, S, H * D)

    for i, li in enumerate(lens):
        ref = reference_attention(
            q[i:i + 1, :, :li], k[i:i + 1, :, :li], v[i:i + 1, :, :li], scaling)
        got = out[i:i + 1, :li]
        assert torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2), \
            f"request {i}: max diff {(got.float()-ref.float()).abs().max().item()}"


def test_flash_prefill_with_history_matches_reference():
    """q/kv 长度分离：hist 长度不同 + 后缀长度不同"""
    torch.manual_seed(0)
    B, H, Hkv, S, D = 2, 4, 2, 4, 16
    lens = [4, 3]
    hist_lens = [33, 17]
    scaling = D ** -0.5

    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k_new = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v_new = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    hist_kv = [
        (torch.randn(1, Hkv, hl, D, device="cuda", dtype=torch.bfloat16),
         torch.randn(1, Hkv, hl, D, device="cuda", dtype=torch.bfloat16))
        for hl in hist_lens
    ]

    out = flash_attention_prefill_with_history(
        q, k_new, v_new, lens, hist_kv, scaling, H, Hkv, D)
    assert out.shape == (B, S, H * D)

    for i, li in enumerate(lens):
        hk, hv = hist_kv[i]
        ref = reference_attention(
            q[i:i + 1, :, :li], torch.cat([hk, k_new[i:i + 1, :, :li]], 2),
            torch.cat([hv, v_new[i:i + 1, :, :li]], 2), scaling)
        got = out[i:i + 1, :li]
        assert torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2), \
            f"request {i}: max diff {(got.float()-ref.float()).abs().max().item()}"

        # PyTorch 参考实现与本 kernel 一致（历史路径双实现交叉验证）
        ref2 = paged_attention_prefill_from_history(
            q[i:i + 1, :, :li], k_new[i:i + 1, :, :li], v_new[i:i + 1, :, :li],
            hk, hv, scaling)
        assert torch.allclose(got.float(), ref2.float(), atol=3e-2, rtol=3e-2)
