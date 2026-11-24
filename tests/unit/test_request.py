"""Request 状态机：合法迁移、非法迁移、结束判定、输出拼接"""

import pytest
import torch

from qwen3.request import Request, WAITING, PREFILLING, DECODING, FINISHED


def make_req(max_new_tokens=4, eos_token_id=999) -> Request:
    return Request(
        request_id=0,
        input_ids=torch.arange(8).unsqueeze(0),   # (1, 8)
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


def test_initial_state_and_properties():
    req = make_req()
    assert req.state == WAITING
    assert req.input_len == 8
    assert req.total_len == 8
    assert req.num_generated == 0


def test_legal_transitions():
    req = make_req()
    req.kv_cache = None
    req.start_prefill()
    assert req.state == PREFILLING
    req.enter_decoding()
    assert req.state == DECODING
    req.finish()
    assert req.state == FINISHED


def test_illegal_transition_raises():
    req = make_req()
    with pytest.raises(AssertionError, match="非法状态迁移"):
        req.enter_decoding()          # waiting -> decoding 非法


def test_append_and_cur_token():
    req = make_req()
    req.append(42)
    assert req.output_ids == [42]
    assert req.num_generated == 1
    assert int(req.cur_token[0, 0]) == 42
    assert req.total_len == 9


def test_is_finished_on_max_new_tokens():
    req = make_req(max_new_tokens=2)
    req.append(1)
    assert not req.is_finished()
    req.append(2)
    assert req.is_finished()


def test_is_finished_on_eos():
    req = make_req(max_new_tokens=10, eos_token_id=151645)
    req.append(151645)
    assert req.is_finished()


def test_finish_frees_kv_cache(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=8, max_seq_len=64)
    from qwen3.PagedKVcache import PagedKVCache

    req = make_req()
    req.kv_cache = PagedKVCache(pool)
    Hkv, D = pool.num_kv_heads, pool.head_dim
    req.kv_cache.update(0, torch.randn(1, Hkv, 20, D), torch.randn(1, Hkv, 20, D))
    req.kv_cache.advance_seq_len(20)
    req.start_prefill()
    req.enter_decoding()

    assert pool.free_count == 6
    req.finish()
    assert pool.free_count == 8
    assert req.kv_cache is None


def test_to_output_ids_concatenates_prompt_and_generated():
    req = make_req()
    req.append(10)
    req.append(11)

    out = req.to_output_ids()
    assert out.shape == (1, 10)
    assert out[0, :8].tolist() == list(range(8))
    assert out[0, 8:].tolist() == [10, 11]


def test_to_output_ids_without_generation_returns_prompt():
    req = make_req()
    assert torch.equal(req.to_output_ids(), req.input_ids)
