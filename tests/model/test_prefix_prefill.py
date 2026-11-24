"""前缀缓存部分命中：paged prefill 与全量 prefill 的 logits 等价（CPU/GPU 通用）"""

import pytest
import torch

from qwen3.PagedKVcache import PagedKVCache


def _new_cache(pool):
    return PagedKVCache(pool)


def test_partial_hit_logits_match_full_prefill(tiny_model, kv_pool_factory):
    """先 prefill 前缀（模拟历史请求），再 prefill 后缀（读历史）→
    后缀位置的 logits 与"整段一次 prefill"一致"""
    torch.manual_seed(0)
    total = 39
    prefix_len = 32                       # 块对齐（2 × 16）
    suffix_len = total - prefix_len
    ids = torch.arange(1, total + 1).unsqueeze(0)

    # 参考：整段一次 prefill
    pool_ref = kv_pool_factory(num_blocks=16, max_seq_len=128)
    cache_ref = _new_cache(pool_ref)
    logits_full = tiny_model(ids, kv_cache=[cache_ref], input_lens=[total],
                             mode="prefill")
    assert cache_ref.seq_len == total

    # 部分命中：前缀 prefill（写入缓存）→ 后缀 prefill（paged prefill 读历史）
    pool = kv_pool_factory(num_blocks=16, max_seq_len=128)
    cache = _new_cache(pool)
    tiny_model(ids[:, :prefix_len], kv_cache=[cache],
               input_lens=[prefix_len], mode="prefill")
    assert cache.seq_len == prefix_len

    logits_suffix = tiny_model(ids[:, prefix_len:], kv_cache=[cache],
                               input_lens=[suffix_len], mode="prefill")
    assert cache.seq_len == total
    assert logits_suffix.shape == (1, suffix_len, logits_full.shape[-1])

    diff = (logits_full[:, prefix_len:, :] - logits_suffix).abs().max().item()
    assert diff < 1e-4, f"部分命中 logits 偏差过大: {diff}"
    assert torch.equal(
        logits_full[:, prefix_len:, :].argmax(-1),
        logits_suffix.argmax(-1)), "argmax 不一致"


def test_partial_hit_single_token_suffix(tiny_model, kv_pool_factory):
    """后缀只有 1 个 token（S=1 的 prefill）也不能误走 decode 路径"""
    torch.manual_seed(0)
    ids = torch.arange(1, 34).unsqueeze(0)     # 33 tokens
    prefix_len, suffix_len = 32, 1

    pool_ref = kv_pool_factory(num_blocks=16, max_seq_len=128)
    cache_ref = _new_cache(pool_ref)
    logits_full = tiny_model(ids, kv_cache=[cache_ref], input_lens=[33],
                             mode="prefill")

    pool = kv_pool_factory(num_blocks=16, max_seq_len=128)
    cache = _new_cache(pool)
    tiny_model(ids[:, :prefix_len], kv_cache=[cache], input_lens=[prefix_len],
               mode="prefill")
    logits_suffix = tiny_model(ids[:, prefix_len:], kv_cache=[cache],
                               input_lens=[suffix_len], mode="prefill")

    diff = (logits_full[:, -1:, :] - logits_suffix).abs().max().item()
    assert diff < 1e-4, f"1-token 后缀 logits 偏差过大: {diff}"


@pytest.mark.parametrize("prefix_len,suffix_len", [(16, 5), (48, 20)])
def test_partial_hit_varied_lengths(tiny_model, kv_pool_factory,
                                    prefix_len, suffix_len):
    torch.manual_seed(0)
    total = prefix_len + suffix_len
    ids = torch.arange(1, total + 1).unsqueeze(0)

    pool_ref = kv_pool_factory(num_blocks=16, max_seq_len=128)
    logits_full = tiny_model(ids, kv_cache=[_new_cache(pool_ref)],
                             input_lens=[total], mode="prefill")

    pool = kv_pool_factory(num_blocks=16, max_seq_len=128)
    cache = _new_cache(pool)
    tiny_model(ids[:, :prefix_len], kv_cache=[cache], input_lens=[prefix_len],
               mode="prefill")
    logits_suffix = tiny_model(ids[:, prefix_len:], kv_cache=[cache],
                               input_lens=[suffix_len], mode="prefill")

    diff = (logits_full[:, prefix_len:, :] - logits_suffix).abs().max().item()
    assert diff < 1e-4
