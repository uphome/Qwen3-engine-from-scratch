"""KVCachePool / PagedKVCache：块分配回收、行槽位、页表同步、写入读回"""

import pytest
import torch

from qwen3.PagedKVcache import (
    BLOCK_CACHED,
    BLOCK_FREE,
    BLOCK_OWNED,
    BLOCK_SHARED,
    PagedKVCache,
)
from qwen3.prefix_cache import compute_prefix_hashes

BLOCK = 16


def test_alloc_free_closed_loop(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8)
    assert pool.free_count == 8

    blocks = pool.alloc(3)
    assert len(blocks) == 3
    assert pool.free_count == 5
    assert len(set(blocks)) == 3

    pool.free(blocks)
    assert pool.free_count == 8


def test_alloc_more_than_available_raises(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=4)
    with pytest.raises(RuntimeError, match="OOM"):
        pool.alloc(5)


def test_alloc_zeroes_reused_blocks(kv_pool_factory):
    """复用块必须清零：防止上一请求残留 K/V 跨请求串扰"""
    pool = kv_pool_factory(num_blocks=2, max_seq_len=32)
    blocks = pool.alloc(1)
    b = blocks[0]
    pool.k_buffer[b].fill_(7.0)
    pool.v_buffer[b].fill_(-3.0)

    pool.free(blocks)
    again = pool.alloc(1)
    assert again[0] == b
    assert torch.count_nonzero(pool.k_buffer[b]) == 0
    assert torch.count_nonzero(pool.v_buffer[b]) == 0


def test_row_slots_alloc_free(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=32)
    r0 = pool.alloc_row()
    r1 = pool.alloc_row()
    assert r0 != r1
    assert pool.seq_lens[r0] == 0

    pool.seq_lens[r0] = 5
    pool.free_row(r0)
    r2 = pool.alloc_row()
    assert r2 == r0                    # LIFO 复用
    assert pool.seq_lens[r2] == 0      # 复用时 seq_len 清零


def test_reserve_next_crosses_page_boundary(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, block_size=BLOCK, max_seq_len=64)
    cache = PagedKVCache(pool)

    # seq_len=15 → 写第 16 个 token 仍在本页，不需要新页
    cache.seq_len = 15
    pool.seq_lens[cache.row_id] = 15
    cache._allocate_block()
    assert cache.num_pages == 1
    cache.reserve_next()
    assert cache.num_pages == 1

    # seq_len=16 → 下一个 token 落到第 2 页
    cache.seq_len = 16
    pool.seq_lens[cache.row_id] = 16
    cache.reserve_next()
    assert cache.num_pages == 2


def test_advance_seq_len_syncs_gpu_tensor(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    cache = PagedKVCache(pool)

    cache.reserve_next()
    cache.advance_seq_len(1)
    assert cache.seq_len == 1
    assert int(pool.seq_lens[cache.row_id]) == 1

    cache.advance_seq_len(4)
    assert cache.seq_len == 5
    assert int(pool.seq_lens[cache.row_id]) == 5


def test_prefill_update_and_get_kv_roundtrip(kv_pool_factory, tiny_config):
    """PyTorch 写入路径：写入 → 读回逐元素一致（跨页、变长）"""
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    cache = PagedKVCache(pool)
    Hkv, D = tiny_config.num_key_value_heads, tiny_config.head_dim

    S = 20    # 跨 2 页（block_size=16）
    k = torch.randn(1, Hkv, S, D)
    v = torch.randn(1, Hkv, S, D)
    cache.update(layer_idx=0, k_new=k, v_new=v)
    cache.advance_seq_len(S)

    assert cache.num_pages == 2
    k_all, v_all = cache.get_kv(layer_idx=0)
    assert k_all.shape == (1, Hkv, S, D)
    assert torch.allclose(k_all, k, atol=1e-6)
    assert torch.allclose(v_all, v, atol=1e-6)


def test_block_table_2d_mirrors_python_table(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    cache = PagedKVCache(pool)
    Hkv, D = pool.num_kv_heads, pool.head_dim
    cache.update(layer_idx=0,
                 k_new=torch.randn(1, Hkv, 20, D),
                 v_new=torch.randn(1, Hkv, 20, D))

    table_gpu = pool.block_table_2d[cache.row_id, :cache.num_pages].tolist()
    assert table_gpu == cache.block_table


def test_free_returns_blocks_and_row(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    cache = PagedKVCache(pool)
    Hkv, D = pool.num_kv_heads, pool.head_dim
    cache.update(0, torch.randn(1, Hkv, 20, D), torch.randn(1, Hkv, 20, D))
    cache.advance_seq_len(20)

    assert pool.free_count == 6
    cache.free()
    assert pool.free_count == 8
    assert cache.seq_len == 0
    assert cache.block_table == []
    assert cache.num_pages == 0


def test_overflow_max_seq_len_raises(kv_pool_factory):
    """请求超过 max_seq_len 必须显式报错，而不是写穿张量"""
    pool = kv_pool_factory(num_blocks=16, max_seq_len=32)
    cache = PagedKVCache(pool)
    Hkv, D = pool.num_kv_heads, pool.head_dim

    cache.update(0, torch.randn(1, Hkv, 32, D), torch.randn(1, Hkv, 32, D))
    cache.advance_seq_len(32)          # 2 页已写满
    with pytest.raises(AssertionError, match="max_seq_len"):
        cache.update(0, torch.randn(1, Hkv, 1, D), torch.randn(1, Hkv, 1, D))


def test_block_state_reports_four_runtime_states(kv_pool_factory):
    """FREE / OWNED / SHARED / CACHED 四态互斥且覆盖全部物理块"""
    pool = kv_pool_factory(num_blocks=8, block_size=BLOCK, max_seq_len=128)
    Hkv, D = pool.num_kv_heads, pool.head_dim

    # CACHED：完整 prefill 后释放，登记 2 个缓存块
    ids = list(range(2 * BLOCK))
    cache = PagedKVCache(pool)
    cache.update(0, torch.randn(1, Hkv, 2 * BLOCK, D),
                 torch.randn(1, Hkv, 2 * BLOCK, D))
    cache.advance_seq_len(2 * BLOCK)
    cache.block_hashes[:2] = compute_prefix_hashes(ids, BLOCK)
    cache.free()

    # SHARED：命中其中 1 块（ref=1，仍在索引里）
    hashes = compute_prefix_hashes(ids, BLOCK)
    shared = pool.match_prefix_partial(hashes[:1])[0]

    # OWNED：新请求 prefill 1 块（未登记索引）
    cache2 = PagedKVCache(pool)
    cache2.update(0, torch.randn(1, Hkv, BLOCK, D),
                  torch.randn(1, Hkv, BLOCK, D))
    owned = cache2.block_table[0]

    states = {b: pool.block_state(b) for b in range(pool.total_blocks)}
    assert states[owned] == BLOCK_OWNED
    assert states[shared] == BLOCK_SHARED
    cached = next(iter(pool._cached_lru))
    assert states[cached] == BLOCK_CACHED
    assert list(states.values()).count(BLOCK_FREE) == 5

    # 四态分区：每块恰好属于一个状态，并集为全部块
    buckets = {s: {b for b, v in states.items() if v == s}
               for s in set(states.values())}
    assert set(buckets) == {BLOCK_FREE, BLOCK_OWNED, BLOCK_SHARED, BLOCK_CACHED}
    assert set().union(*buckets.values()) == set(range(pool.total_blocks))
    assert sum(len(v) for v in buckets.values()) == pool.total_blocks

    with pytest.raises(IndexError, match="越界"):
        pool.block_state(-1)
    with pytest.raises(IndexError, match="越界"):
        pool.block_state(pool.total_blocks)
