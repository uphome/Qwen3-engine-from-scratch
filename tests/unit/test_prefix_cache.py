"""前缀缓存（阶段 1：完全命中）：哈希链 / 命中快速路径 / 引用计数 / LRU

纯 KV 操作（不跑模型）：手动模拟 prefill 写页，验证块表与生命周期。
"""

import torch

from qwen3 import Scheduler
from qwen3.batch import Batch
from qwen3.prefix_cache import compute_prefix_hashes, hash_block
from qwen3.request import Request, DECODING, WAITING

BS = 16


# ---------------------------------------------------------------------------
# 哈希链
# ---------------------------------------------------------------------------

def test_hash_chain_prefix_property():
    ids = list(range(48))
    h1 = compute_prefix_hashes(ids, BS)

    # 前缀相同、后缀不同 → 前面块的哈希完全相同
    ids2 = ids[:32] + [1000 + i for i in range(16)]
    h2 = compute_prefix_hashes(ids2, BS)
    assert h1[:2] == h2[:2]
    assert h1[2] != h2[2]          # 差异块开始不同

    # 不同 token → 不同哈希
    assert compute_prefix_hashes([1] * 16, BS) != compute_prefix_hashes([2] * 16, BS)

    # 非对齐长度只返回完整块
    assert len(compute_prefix_hashes(list(range(20)), BS)) == 1
    assert len(compute_prefix_hashes(list(range(16)), BS)) == 1
    assert len(compute_prefix_hashes(list(range(15)), BS)) == 0


def test_hash_block_is_stable():
    assert hash_block(0, [1, 2, 3]) == hash_block(0, [1, 2, 3])
    assert hash_block(0, [1, 2, 3]) != hash_block(1, [1, 2, 3])


# ---------------------------------------------------------------------------
# 辅助：模拟一次完整 prefill（写页 + 状态收尾）
# ---------------------------------------------------------------------------

def complete_prefill(sched: Scheduler, req: Request, pool, write_len: int | None = None):
    n = req.input_len if write_len is None else write_len
    k = torch.randn(1, pool.num_kv_heads, n, pool.head_dim)
    v = torch.randn(1, pool.num_kv_heads, n, pool.head_dim)
    req.kv_cache.update(0, k, v)
    req.kv_cache.advance_seq_len(n)
    req.start_prefill()          # 模拟被调度器选中（waiting → prefilling）
    sched.on_step_done(Batch("prefill", [req]), torch.tensor([[1]]))


def finish_req(sched: Scheduler, req: Request):
    """模拟请求完成：状态迁移 + 回收块 + 从 running 移除（正常走 scheduler）"""
    req.finish()
    if req in sched.running:
        sched.running.remove(req)


def make_req(rid: int, ids: list[int], max_new: int = 4) -> Request:
    return Request(request_id=rid,
                   input_ids=torch.tensor([ids], dtype=torch.long),
                   max_new_tokens=max_new, eos_token_id=99999)


# ---------------------------------------------------------------------------
# 完全命中
# ---------------------------------------------------------------------------

def test_full_hit_skips_prefill_and_shares_blocks(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)

    ids = list(range(32))                     # 2 个完整块（块对齐）
    a = make_req(0, ids)
    assert sched.add(a) == 0                  # 首次：未命中
    complete_prefill(sched, a, pool)
    finish_req(sched, a)
    assert pool.cached_count == 2
    assert pool.free_count == 14
    cached_blocks = list(pool._cached_lru.keys())

    b = make_req(1, ids)
    hit = sched.add(b)
    assert hit == 1                            # 完全命中（2 块共享，COW 私有块算新块）
    assert b.state == DECODING                 # 跳过 prefill，直接进入 decode
    assert len(sched.running) == 1
    # 缓存里只保留前 L-1 个 token；最后一个 token 走首次 decode（重算 logits）
    assert b.kv_cache.seq_len == 31
    assert int(pool.seq_lens[b.kv_cache.row_id]) == 31
    assert b.kv_cache.num_pages == 2
    # 前 n-1 块共享；最后一块是 COW 私有副本（可写）
    assert b.kv_cache.block_table[:-1] == cached_blocks[:-1]
    assert b.kv_cache.block_table[-1] not in cached_blocks
    assert b.kv_cache.block_hashes == [b.prefix_hashes[0], None]
    assert b.prefill_last_token == ids[-1]
    assert b.cur_token.item() == ids[-1]

    # 首次 decode（写 position 31）不需要新块；跨到第 33 个 token 才分配
    b.kv_cache.reserve_next()
    assert b.kv_cache.num_pages == 2
    b.kv_cache.advance_seq_len(1)              # seq_len → 32
    b.kv_cache.reserve_next()
    assert b.kv_cache.num_pages == 3
    assert b.kv_cache.block_table[2] not in cached_blocks

    finish_req(sched, b)
    # 共享块与 COW 原块留在缓存；私有副本无哈希 → 回空闲池
    assert pool.cached_count == 2
    assert pool.free_count == 14
    assert pool.free_count + pool.cached_count == pool.total_blocks


def test_ref_count_while_shared_by_active_request(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)

    ids = list(range(32))
    a = make_req(0, ids)
    sched.add(a)
    complete_prefill(sched, a, pool)
    finish_req(sched, a)                                 # 登记缓存，ref=0

    b = make_req(1, ids)
    assert sched.add(b) == 1                   # 命中（末块 COW 计新块）
    c = make_req(2, ids)
    assert sched.add(c) == 1                   # 同时命中 → 共享首块 ref=2
    shared0 = b.kv_cache.block_table[0]
    assert c.kv_cache.block_table[0] == shared0
    assert pool.block_refs[shared0] == 2
    # 末块各自 COW，互不共享
    assert b.kv_cache.block_table[-1] != c.kv_cache.block_table[-1]

    finish_req(sched, b)                                 # ref=1：块不能被回收/淘汰
    assert pool.cached_count == 1                        # 只剩 COW 原块
    assert pool.block_refs[shared0] == 1

    finish_req(sched, c)                                 # ref=0 → 回到缓存
    assert pool.cached_count == 2
    assert pool.free_count + pool.cached_count == pool.total_blocks


def test_unaligned_prompt_partial_hit(kv_pool_factory, monkeypatch):
    """阶段 2：前缀命中 2 块，尾部 4 token 走 prefill（只送后缀）"""
    monkeypatch.setenv("QWEN3_PREFIX_PARTIAL_MIN", "0")
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)

    ids = list(range(32))
    a = make_req(0, ids)
    sched.add(a)
    complete_prefill(sched, a, pool)
    finish_req(sched, a)
    cached_blocks = list(pool._cached_lru.keys())

    b = make_req(1, ids + [100, 101, 102, 103])
    assert sched.add(b) == 2
    assert b.state == WAITING                   # 进入 prefill 批
    assert b.prefill_offset == 32
    assert b.kv_cache.seq_len == 32
    assert b.kv_cache.num_pages == 2
    assert b.kv_cache.block_table == cached_blocks
    assert all(pool.block_refs[blk] == 1 for blk in cached_blocks)

    batch = Batch("prefill", [b])
    assert batch.input_lens == [4]              # 只送未命中后缀
    assert batch.build_input_ids().tolist() == [[100, 101, 102, 103]]

    # 模拟 prefill 后缀（只写 4 个 token）后完成
    complete_prefill(sched, b, pool, write_len=4)
    finish_req(sched, b)
    assert pool.free_count + pool.cached_count == pool.total_blocks


def test_different_second_block_partial_hit(kv_pool_factory, monkeypatch):
    """第 2 块不同 → 只命中第 1 块（最长前缀匹配），后缀从 16 开始"""
    monkeypatch.setenv("QWEN3_PREFIX_PARTIAL_MIN", "0")
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)

    a = make_req(0, list(range(32)))
    sched.add(a)
    complete_prefill(sched, a, pool)
    finish_req(sched, a)

    b = make_req(1, list(range(16)) + [1000 + i for i in range(16)])
    assert sched.add(b) == 1
    assert b.state == WAITING
    assert b.prefill_offset == 16
    assert b.kv_cache.seq_len == 16
    assert b.kv_cache.num_pages == 1
    assert pool.cached_count == 1               # 命中的块被请求持有（移出 LRU）


def test_no_prefix_miss(kv_pool_factory):
    """首块就不同 → 完全未命中，普通 prefill"""
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)

    a = make_req(0, list(range(32)))
    sched.add(a)
    complete_prefill(sched, a, pool)
    finish_req(sched, a)

    b = make_req(1, [9000 + i for i in range(32)])
    assert sched.add(b) == 0
    assert b.state == WAITING
    assert b.prefill_offset == 0
    assert b.kv_cache.num_pages == 0


def test_lru_eviction(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    pool.max_cached_blocks = 4          # 显式限制，构造淘汰场景
    sched = Scheduler(pool, batch_size=1)

    prompts = [[1000 * i + j for j in range(16)] for i in range(5)]
    for i, ids in enumerate(prompts):
        req = make_req(i, ids)
        assert sched.add(req) == 0
        complete_prefill(sched, req, pool)
        req.finish()

    # 5 个块竞争 4 个缓存位 → 最旧的被淘汰
    assert pool.cached_count == 4
    assert pool.free_count + pool.cached_count == pool.total_blocks

    # 最旧的 prompt 已淘汰 → 重新请求不命中
    old = make_req(99, prompts[0])
    assert sched.add(old) == 0
    # 最新一个仍命中（单块 prompt：唯一块被 COW 复制，共享复用块数 = 0）
    newest = make_req(100, prompts[-1])
    assert sched.add(newest) == 0
    assert newest.state == DECODING
    finish_req(sched, newest)


def test_pool_accounting_after_mixed_traffic(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)

    shared = list(range(32))
    a = make_req(0, shared)
    sched.add(a)
    complete_prefill(sched, a, pool)
    finish_req(sched, a)

    for i in range(3):
        b = make_req(10 + i, shared)
        assert sched.add(b) == 1      # 命中：跳过 prefill，直接 decode
        finish_req(sched, b)

    assert pool.free_count + pool.cached_count == pool.total_blocks
    assert pool.cached_count == 2


def test_chain_aware_eviction_keeps_prefix_chain(kv_pool_factory):
    """淘汰优先选链尾：保留 A→B 前缀链，而不是淘汰链头导致命中 0 块"""
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    pool.max_cached_blocks = 3

    # 同一条 prompt 的 3 块哈希链 A → B → C
    hashes = compute_prefix_hashes(list(range(48)), BS)     # 3 块
    # 另一个 prompt 的 1 块（无关叶节点）
    other = compute_prefix_hashes([9000 + i for i in range(16)], BS)[0]

    blocks = pool.alloc(4)
    for i in range(3):
        parent = hashes[i - 1] if i > 0 else 0
        pool.release_block(blocks[i], hashes[i], parent)
    pool.release_block(blocks[3], other, 0)   # 第 4 块触发淘汰

    # 上限 3：应淘汰 C（链尾）或 D，保留 A、B（前缀链完整）
    hit = pool.match_prefix_partial(hashes)
    assert len(hit) == 2, f"链感知淘汰失效：命中 {len(hit)} 块（期望 2）"
    assert pool._children.get(hashes[0]), "A 的子块应为 B"
    for b, h in zip(hit, hashes):
        pool.release_block(b, h, hashes[hashes.index(h) - 1] if hashes.index(h) else 0)
