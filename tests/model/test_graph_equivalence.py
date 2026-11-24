"""CUDA Graph 图池 vs eager forward_decode：多 bucket 逐 step logits 一致

覆盖：真实 row_id、空槽哑行、k=1 / k=2 两种桶、3 步状态推进。
"""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _make_env(model, tiny_config, k: int, device):
    """k 个真实请求句柄：模拟 prefill 完成（seq_len=1，K/V 为初始零值）"""
    from qwen3 import KVCachePool
    from qwen3.PagedKVcache import PagedKVCache

    pool = KVCachePool(
        num_blocks=4 + k, num_layers=tiny_config.num_hidden_layers,
        block_size=16, num_kv_heads=tiny_config.num_key_value_heads,
        head_dim=tiny_config.head_dim, device=device,
        dtype=torch.bfloat16, max_seq_len=64,
    )
    caches = [PagedKVCache(pool) for _ in range(k)]
    for c in caches:
        c.seq_len = 1
        pool.seq_lens[c.row_id] = 1
        c.reserve_next()
    return pool, caches


def _run_eager(model, caches, step_tokens, device):
    """eager：reserve → forward_decode（v3.1 起直接传 position_ids）→ advance"""
    outputs = []
    for toks in step_tokens:
        k = len(caches)
        assert len(toks) == k
        for c in caches:
            c.reserve_next()
        positions = torch.tensor([[c.seq_len] for c in caches],
                                 dtype=torch.long, device=device)
        rows = torch.tensor([c.row_id for c in caches],
                            dtype=torch.int32, device=device)
        ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(1)
        logits = model.forward_decode(ids, positions, rows, caches)
        for c in caches:
            c.advance_seq_len(1)
        outputs.append(logits.float().cpu())
    return outputs


def _run_graph(model, caches, step_tokens, buckets, device):
    from qwen3.graph_runner import GraphRunner
    from qwen3.PagedKVcache import PagedKVCache

    pool = caches[0]._pool
    occupy = [PagedKVCache(pool) for _ in range(max(buckets))]
    runner = GraphRunner(model, pool, occupy, buckets=buckets, reserve_pages=1)

    outputs = []
    for toks in step_tokens:
        k = len(caches)
        assert len(toks) == k
        for c in caches:
            c.reserve_next()
        positions = torch.tensor([[c.seq_len] for c in caches],
                                 dtype=torch.long, device=device)
        rows = torch.tensor([c.row_id for c in caches],
                            dtype=torch.int32, device=device)
        ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(1)
        logits = runner.replay(ids, positions, rows)[:k]
        for c in caches:
            c.advance_seq_len(1)
        outputs.append(logits.float().cpu())
    return outputs


@pytest.mark.parametrize("k,buckets", [(1, (1,)), (2, (1, 2))])
def test_graph_matches_eager(tiny_model, tiny_config, k, buckets, gpu_device):
    model = tiny_model.to(gpu_device).to(torch.bfloat16)
    steps = 3

    # 两条路径用相同随机输入（同 seed、同序列）
    torch.manual_seed(123)
    step_tokens = [torch.randint(0, tiny_config.vocab_size, (k,)).tolist()
                   for _ in range(steps)]

    pool_e, caches_e = _make_env(model, tiny_config, k, gpu_device)
    eager = _run_eager(model, caches_e, step_tokens, gpu_device)

    pool_g, caches_g = _make_env(model, tiny_config, k, gpu_device)
    graph = _run_graph(model, caches_g, step_tokens, buckets, gpu_device)

    for step, (a, b) in enumerate(zip(eager, graph)):
        assert a.shape == b.shape == (k, 1, tiny_config.vocab_size)
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), f"step {step} logits 不一致"

    # 池闭环：占位句柄 max(buckets) 页 + 哑行 1 页 + 每个真实请求 1 页（3 步内不跨页）
    n_reserved = max(buckets) + 1 + k
    assert pool_g.free_count == pool_g.total_blocks - n_reserved
