"""容量压力下的缓存淘汰：按需淘汰 / 开关 / ref 保护 / 指标"""

import pytest
import torch

from qwen3 import Scheduler
from qwen3.prefix_cache import compute_prefix_hashes
from qwen3.request import Request, WAITING
from qwen3.monitoring import snapshot
from server import EngineMetrics, render_metrics

BS = 16


def make_req(rid: int, length: int, max_new: int = 8) -> Request:
    return Request(request_id=rid,
                   input_ids=torch.arange(length).unsqueeze(0) + rid * 10000,
                   max_new_tokens=max_new, eos_token_id=99999)


def seed_cache(pool, n_chains: int = 4, blocks_per_chain: int = 1):
    """直接登记 n 条缓存链（每条 blocks_per_chain 块），返回哈希列表"""
    total_blocks = n_chains * blocks_per_chain
    ids = pool.alloc(total_blocks)
    hashes_all = []
    for c in range(n_chains):
        base = [c * 10000 + i for i in range(BS * blocks_per_chain)]
        hashes = compute_prefix_hashes(base, BS)
        for j, h in enumerate(hashes):
            parent = hashes[j - 1] if j > 0 else 0
            pool.release_block(ids[c * blocks_per_chain + j], h, parent)
        hashes_all.append(hashes)
    return hashes_all


def test_evict_on_demand_frees_space(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=256)
    pool.max_cached_blocks = 4
    seed_cache(pool, n_chains=4, blocks_per_chain=1)
    assert pool.cached_count == 4
    assert pool.free_count == 4

    sched = Scheduler(pool, batch_size=1)
    # 96 token + 8 max_new → 7 块；free 只有 4 → 需淘汰 3 块
    req = make_req(0, 96, max_new=8)
    hit = sched.add(req)

    assert hit == 0
    assert req.state == WAITING
    assert pool.cache_evictions >= 1
    assert pool.free_count + pool.cached_count == pool.total_blocks
    # 淘汰后腾出的空间足够 prefill（7 块）
    assert pool.free_count >= 7


def test_evict_not_triggered_when_space_enough(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=16, max_seq_len=256)
    seed_cache(pool, n_chains=2, blocks_per_chain=1)
    ev_before = pool.cache_evictions

    sched = Scheduler(pool, batch_size=1)
    req = make_req(0, 32, max_new=8)          # 3 块，free 充足
    assert sched.add(req) == 0
    assert pool.cache_evictions == ev_before   # 未触发强制淘汰


def test_evict_does_not_touch_active_blocks(kv_pool_factory):
    """被活跃请求持有的块（ref>0）不在 LRU，强制淘汰不会动它们"""
    pool = kv_pool_factory(num_blocks=16, max_seq_len=256)
    pool.max_cached_blocks = 2

    # 一条 2 块的链，登记为缓存
    hashes = compute_prefix_hashes(list(range(BS * 2)), BS)
    ids = pool.alloc(2)
    for i, h in enumerate(hashes):
        pool.release_block(ids[i], h, hashes[i - 1] if i > 0 else 0)
    assert pool.cached_count == 2

    # 另一个请求命中这 2 块（ref→1，从 LRU 移出）
    held = pool.match_prefix_partial(hashes)
    assert len(held) == 2
    assert all(pool.block_refs[b] == 1 for b in held)
    assert pool.cached_count == 0

    # 再登记两个与 held 不重叠的缓存块，然后强制淘汰
    pool.max_cached_blocks = 4
    for base in (20000, 30000):
        hs = compute_prefix_hashes([base + i for i in range(BS)], BS)
        blk = pool.alloc(1)[0]
        pool.release_block(blk, hs[0], 0)
    assert pool.cached_count == 2

    ev = pool.cache_evictions
    freed = pool.evict_cache_for(999)

    assert freed == 2                              # 只淘汰了 LRU 里的块
    assert pool.cache_evictions == ev + freed
    assert all(pool.block_refs[b] == 1 for b in held)   # 被持有的块未动
    # 池不变量
    assert pool.free_count + pool.cached_count + sum(
        1 for r in pool.block_refs if r > 0) == pool.total_blocks


def test_evict_disabled_raises(kv_pool_factory, monkeypatch):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=256)
    pool.max_cached_blocks = 4
    seed_cache(pool, n_chains=4, blocks_per_chain=1)
    monkeypatch.setenv("QWEN3_CACHE_EVICT_ON_DEMAND", "0")
    sched = Scheduler(pool, batch_size=1)      # 构造时读开关
    req = make_req(0, 96, max_new=8)           # 需要 7 块，free 4

    with pytest.raises(RuntimeError, match="拒绝入队"):
        sched.add(req)
    assert pool.cache_evictions == 0


def test_evict_insufficient_still_raises(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=256)
    pool.max_cached_blocks = 2
    seed_cache(pool, n_chains=2, blocks_per_chain=1)   # free 6
    sched = Scheduler(pool, batch_size=1)
    # 需要 8 块（120 token + 8），free 6 + 缓存 2 = 8 → 刚好够；改为 136 → 9 块
    req = make_req(0, 136, max_new=8)          # ceil(144/16)=9 > 8
    with pytest.raises(RuntimeError, match="拒绝入队"):
        sched.add(req)
    # 缓存已被淘汰为空（尝试过），池不变量仍成立
    assert pool.free_count + pool.cached_count == pool.total_blocks


def test_eviction_metrics_visible(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=256)
    pool.max_cached_blocks = 4
    seed_cache(pool, n_chains=4, blocks_per_chain=1)
    sched = Scheduler(pool, batch_size=1)
    sched.add(make_req(0, 96, max_new=8))
    assert pool.cache_evictions > 0

    engine = type("E", (), {})()
    engine.metrics = EngineMetrics()
    engine.scheduler = sched
    engine.pool = pool
    engine.device = torch.device("cpu")
    snap = snapshot(engine)
    assert snap["prefix_cache"]["evictions"] == pool.cache_evictions
    assert "engine_prefix_cache_evictions_total" in render_metrics(engine)
