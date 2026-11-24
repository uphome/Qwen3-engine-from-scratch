"""Scheduler：组批策略、补位防超编、状态收尾、池闭环"""

import pytest
import torch

from qwen3 import Scheduler
from qwen3.request import Request, WAITING, PREFILLING, DECODING, FINISHED


def make_request(rid: int, length: int, max_new: int = 4) -> Request:
    return Request(
        request_id=rid,
        input_ids=torch.arange(length).unsqueeze(0),
        max_new_tokens=max_new,
        eos_token_id=9999,
    )


def test_add_allocates_handle_and_enters_waiting(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)
    sched.add(make_request(0, 10))

    req = sched.waiting[0]
    assert req.state == WAITING
    assert req.kv_cache is not None
    assert req.kv_cache.row_id >= 0


def test_add_rejects_when_pool_too_small(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=1, max_seq_len=64)   # 只够 1 页 = 16 token
    sched = Scheduler(pool, batch_size=1)

    sched.add(make_request(0, 12, max_new=4))             # (12+4)/16 = 1 页
    with pytest.raises(RuntimeError, match="拒绝入队"):
        sched.add(make_request(1, 17, max_new=4))         # 需要 2 页


def test_prefill_batch_size_respects_batch_size(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=32, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)
    for i in range(4):
        sched.add(make_request(i, 4))

    batch = sched.schedule()
    assert batch.mode == "prefill"
    assert batch.size == 2
    assert [r.request_id for r in batch.requests] == [0, 1]
    assert all(r.state == PREFILLING for r in batch.requests)


def test_on_step_done_prefill_moves_to_running(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=32, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)
    sched.add(make_request(0, 4))
    sched.add(make_request(1, 4))

    batch = sched.schedule()
    sched.on_step_done(batch, torch.tensor([[1], [2]]))

    assert len(sched.running) == 2
    assert all(r.state == DECODING for r in sched.running)
    assert sched.running[0].output_ids == [1]


def test_decode_batches_running_requests(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=32, max_seq_len=64)
    sched = Scheduler(pool, batch_size=2)
    sched.add(make_request(0, 4))
    sched.add(make_request(1, 4))
    sched.on_step_done(sched.schedule(), torch.tensor([[1], [2]]))

    batch = sched.schedule()
    assert batch.mode == "decode"
    assert batch.size == 2
    assert batch.input_lens == [1, 1]


def test_prefill_backfill_never_exceeds_batch_size(kv_pool_factory):
    """running 已有时补位只能占剩余空位（历史 bug：超编导致队尾饿死）"""
    pool = kv_pool_factory(num_blocks=64, max_seq_len=64)
    sched = Scheduler(pool, batch_size=4)
    for i in range(6):
        sched.add(make_request(i, 4))

    # 首轮 prefill 4 个 → running=4
    sched.on_step_done(sched.schedule(), torch.tensor([[1], [1], [1], [1]]))
    assert len(sched.running) == 4

    # running 已满 → 必须 decode，不能补位
    batch = sched.schedule()
    assert batch.mode == "decode"
    assert batch.size == 4

    # 一个请求完成后（running=3）→ 下一步补位只取 1 个
    batch = sched.schedule()
    sched.running[0].append(9999)          # 触发 eos
    sched.running[0].finish()
    sched.running.pop(0)
    assert len(sched.running) == 3

    batch = sched.schedule()
    assert batch.mode == "prefill"
    assert batch.size == 1                  # slots = 4 - 3


def test_finished_request_returns_blocks(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=16, max_seq_len=64)
    sched = Scheduler(pool, batch_size=1)
    sched.add(make_request(0, 4, max_new=1))

    # prefill 步：采样第 1 个 token，进入 decoding（尚未判定结束）
    batch = sched.schedule()
    sched.on_step_done(batch, torch.tensor([[7]]))
    assert len(sched.running) == 1

    # decode 步：再采样 1 个 → 达到 max_new_tokens → 出批
    batch = sched.schedule()
    sched.on_step_done(batch, torch.tensor([[8]]))

    assert len(sched.finished) == 1
    assert sched.finished[0].state == FINISHED
    assert pool.free_count == 16
    assert not sched.has_pending()


def test_full_loop_closed_pool(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=32, max_seq_len=64)
    sched = Scheduler(pool, batch_size=3)
    for i in range(5):
        sched.add(make_request(i, 4, max_new=2))

    step = 0
    while sched.has_pending():
        batch = sched.schedule()
        next_tokens = torch.zeros(batch.size, 1, dtype=torch.long)
        sched.on_step_done(batch, next_tokens)
        step += 1
        assert step < 100, "调度循环未收敛"

    assert len(sched.finished) == 5
    assert pool.free_count == 32
