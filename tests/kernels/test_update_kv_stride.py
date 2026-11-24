"""update_kv_batch Triton kernel：连续 / fused-qkv 非连续切片布局

回归背景（v3.2 fused qkv）：
  k/v 是 qkv_proj 输出的切片，batch 维 stride = (Hq+2*Hkv)*D，不再是
  Hkv*D。旧 kernel 按连续布局寻址，b>=1 的请求会读到 q 区（批量
  rollout 只有第一条正确）。本测试锁定"显式 stride"修复。
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

Hq, Hkv, D = 16, 8, 128
B = 3


def _make_env(tiny_config, device="cuda"):
    from qwen3 import KVCachePool
    from qwen3.PagedKVcache import PagedKVCache

    pool = KVCachePool(
        num_blocks=16, num_layers=2, block_size=16,
        num_kv_heads=Hkv, head_dim=D, device=device,
        dtype=torch.bfloat16, max_seq_len=64,
    )
    caches = []
    for _ in range(B):
        c = PagedKVCache(pool)
        c.seq_len = 5
        pool.seq_lens[c.row_id] = 5      # driver 约定：Python → GPU 同步
        c.reserve_next()                 # 写第 6 个 token 前预留页
        caches.append(c)
    rows = torch.tensor([c.row_id for c in caches], dtype=torch.int32, device=device)
    return pool, caches, rows


@pytest.mark.parametrize("layout", ["contiguous", "fused_slice"])
def test_update_kv_batch_writes_correct_positions(layout, gpu_device):
    device = gpu_device
    pool, caches, rows = _make_env(None)
    torch.manual_seed(0)

    if layout == "contiguous":
        k = torch.randn(B, Hkv, 1, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, Hkv, 1, D, device=device, dtype=torch.bfloat16)
        assert k.is_contiguous()
    else:
        # 模拟 fused qkv_proj: (B, 1, (Hq+2Hkv)*D) → k/v 切片 → (B,Hkv,1,D)
        fused = torch.randn(B, 1, (Hq + 2 * Hkv) * D, device=device,
                            dtype=torch.bfloat16)
        k = fused[..., Hq * D:(Hq + Hkv) * D].view(B, 1, Hkv, D).transpose(1, 2)
        v = fused[..., (Hq + Hkv) * D:].view(B, 1, Hkv, D).transpose(1, 2)
        assert not k.is_contiguous()
        assert k.stride(0) == (Hq + 2 * Hkv) * D   # batch stride ≠ Hkv*D

    pool.update_kv_batch(0, k, v, rows)

    for i, c in enumerate(caches):
        phys = int(pool.block_table_2d[c.row_id, 0])
        got_k = pool.k_buffer[phys, 0, :, 5, :]
        got_v = pool.v_buffer[phys, 0, :, 5, :]
        assert torch.equal(got_k, k[i].squeeze(1)), f"request {i} K 写错位置"
        assert torch.equal(got_v, v[i].squeeze(1)), f"request {i} V 写错位置"


def test_update_kv_batch_across_page_boundary(gpu_device):
    """写第 17 个 token 时落到第 2 页（块表项 1）"""
    device = gpu_device
    from qwen3 import KVCachePool
    from qwen3.PagedKVcache import PagedKVCache

    pool = KVCachePool(num_blocks=8, num_layers=1, block_size=16,
                       num_kv_heads=Hkv, head_dim=D, device=device,
                       dtype=torch.bfloat16, max_seq_len=64)
    c = PagedKVCache(pool)

    # 模拟 prefill 写入 16 个 token（占满第 1 页）
    c.update(0, torch.randn(1, Hkv, 16, D, device=device, dtype=torch.bfloat16),
             torch.randn(1, Hkv, 16, D, device=device, dtype=torch.bfloat16))
    c.advance_seq_len(16)
    assert c.num_pages == 1

    c.reserve_next()                       # 第 17 个 token 跨页
    assert c.num_pages == 2

    rows = torch.tensor([c.row_id], dtype=torch.int32, device=device)
    k = torch.randn(1, Hkv, 1, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, 1, D, device=device, dtype=torch.bfloat16)
    pool.update_kv_batch(0, k, v, rows)

    phys2 = int(pool.block_table_2d[c.row_id, 1])
    assert torch.equal(pool.k_buffer[phys2, 0, :, 0, :], k[0, :, 0, :])
