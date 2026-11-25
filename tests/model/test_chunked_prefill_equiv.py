"""chunked prefill 数值等价：分块 prefill 的 logits == 整段 prefill（CPU/GPU 通用）"""

import torch

from qwen3.PagedKVcache import PagedKVCache


def test_chunked_prefill_matches_full(tiny_model, kv_pool_factory):
    torch.manual_seed(0)
    total, chunk = 40, 16            # 40 = 16 + 16 + 8
    ids = torch.arange(1, total + 1).unsqueeze(0)

    # 参考：整段一次 prefill
    pool_ref = kv_pool_factory(num_blocks=16, max_seq_len=128)
    logits_full = tiny_model(ids, kv_cache=[PagedKVCache(pool_ref)],
                             input_lens=[total], mode="prefill")

    # 分块 prefill（每块都读历史，走 paged prefill）
    pool = kv_pool_factory(num_blocks=16, max_seq_len=128)
    cache = PagedKVCache(pool)
    last = None
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        last = tiny_model(ids[:, start:end], kv_cache=[cache],
                          input_lens=[end - start], mode="prefill")
        assert cache.seq_len == end

    n_last = total - (total // chunk) * chunk or chunk
    diff = (logits_full[:, -n_last:, :] - last).abs().max().item()
    assert diff < 1e-4, f"分块 prefill logits 偏差过大: {diff}"
    assert torch.equal(
        logits_full[:, -n_last:, :].argmax(-1), last.argmax(-1))


def test_chunk_size_equals_prompt_is_single_pass(tiny_model, kv_pool_factory):
    torch.manual_seed(0)
    ids = torch.arange(1, 21).unsqueeze(0)      # 20 tokens
    pool_ref = kv_pool_factory(num_blocks=16, max_seq_len=128)
    full = tiny_model(ids, kv_cache=[PagedKVCache(pool_ref)],
                      input_lens=[20], mode="prefill")
    pool = kv_pool_factory(num_blocks=16, max_seq_len=128)
    one = tiny_model(ids, kv_cache=[PagedKVCache(pool)],
                     input_lens=[20], mode="prefill")
    assert torch.equal(full, one)
