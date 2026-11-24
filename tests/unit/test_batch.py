"""Batch：prefill 右 pad / decode 组装 / input_lens"""

import torch

from qwen3.batch import Batch
from qwen3.request import Request


def make_req(rid: int, length: int, generated=(1, 2)) -> Request:
    req = Request(
        request_id=rid,
        input_ids=torch.arange(length).unsqueeze(0) + rid * 100,
        max_new_tokens=8,
        eos_token_id=999,
    )
    if generated:
        req.output_ids = list(generated)
        req.num_generated = len(generated)
    return req


def test_prefill_pads_to_max_len():
    reqs = [make_req(0, 3, generated=()), make_req(1, 6, generated=())]
    batch = Batch("prefill", reqs)

    ids = batch.build_input_ids()
    assert ids.shape == (2, 6)
    assert ids[0, :3].tolist() == [0, 1, 2]
    assert ids[0, 3:].tolist() == [Batch.PAD_ID] * 3
    assert ids[1].tolist() == [100, 101, 102, 103, 104, 105]

    assert batch.input_lens == [3, 6]
    assert batch.size == 2


def test_decode_takes_last_token_per_request():
    reqs = [make_req(0, 4, generated=(7, 8)), make_req(1, 2, generated=(9,))]
    batch = Batch("decode", reqs)

    ids = batch.build_input_ids()
    assert ids.shape == (2, 1)
    assert ids.tolist() == [[8], [9]]
    assert batch.input_lens == [1, 1]


def test_build_kv_caches_passthrough():
    reqs = [make_req(0, 3), make_req(1, 5)]
    reqs[0].kv_cache = "cache-0"
    reqs[1].kv_cache = "cache-1"

    batch = Batch("decode", reqs)
    assert batch.build_kv_caches() == ["cache-0", "cache-1"]


def test_invalid_mode_raises():
    import pytest

    with pytest.raises(AssertionError):
        Batch("mixed", [])
