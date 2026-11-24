"""RoPE：预计算表数值 + 查表 + 旋转性质

约定：前后配对（rotate_half 交换前后两半），cos/sin 用 cat(freqs, freqs)。
"""

import math

import torch

from qwen3.rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


HEAD_DIM = 16
BASE = 1_000_000.0


def test_table_shape_and_position_zero():
    rope = RotaryEmbedding(HEAD_DIM, base=BASE, max_seq_len=32)
    assert rope.cos_table.shape == (32, HEAD_DIM)
    assert rope.sin_table.shape == (32, HEAD_DIM)

    # 位置 0：所有频率角度 = 0 → cos=1, sin=0
    assert torch.allclose(rope.cos_table[0], torch.ones(HEAD_DIM))
    assert torch.allclose(rope.sin_table[0], torch.zeros(HEAD_DIM))


def test_table_matches_manual_formula():
    """cos_table[m, i] = cos(m * base^(-2i/d))，且前后两半相同"""
    rope = RotaryEmbedding(HEAD_DIM, base=BASE, max_seq_len=32)
    m = 7
    for i in range(HEAD_DIM // 2):
        theta = BASE ** (-2.0 * i / HEAD_DIM)
        expected = math.cos(m * theta)
        assert math.isclose(float(rope.cos_table[m, i]), expected, rel_tol=1e-5)
        # 前后两半共享同一频率
        assert torch.equal(rope.cos_table[m, i], rope.cos_table[m, i + HEAD_DIM // 2])


def test_forward_gather_shape():
    rope = RotaryEmbedding(HEAD_DIM, base=BASE, max_seq_len=32)
    positions = torch.tensor([[0, 3, 7], [1, 2, 31]])
    cos, sin = rope(positions)

    assert cos.shape == (2, 3, HEAD_DIM)
    assert sin.shape == (2, 3, HEAD_DIM)
    assert torch.equal(cos[0, 1], rope.cos_table[3])


def test_rotate_half_semantics():
    """rotate_half([a0..a7, b0..b7]) = [-b0..-b7, a0..a7]"""
    x = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    out = rotate_half(x)
    assert torch.equal(out[0, :8], -x[0, 8:])
    assert torch.equal(out[0, 8:], x[0, :8])


def test_apply_at_position_zero_is_identity():
    rope = RotaryEmbedding(HEAD_DIM, base=BASE, max_seq_len=32)
    q = torch.randn(2, 4, 3, HEAD_DIM)
    k = torch.randn(2, 2, 3, HEAD_DIM)
    positions = torch.zeros(2, 3, dtype=torch.long)

    q_out, k_out = apply_rotary_pos_emb(q, k, *rope(positions))
    assert torch.allclose(q_out, q, atol=1e-6)
    assert torch.allclose(k_out, k, atol=1e-6)


def test_rotation_preserves_norm():
    """旋转是正交变换：每个 head_dim 向量的 L2 范数不变"""
    rope = RotaryEmbedding(HEAD_DIM, base=BASE, max_seq_len=64)
    q = torch.randn(1, 2, 5, HEAD_DIM)
    k = torch.randn(1, 1, 5, HEAD_DIM)
    positions = torch.arange(5).unsqueeze(0)

    q_out, k_out = apply_rotary_pos_emb(q, k, *rope(positions))
    assert torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k_out.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_relative_position_property():
    """RoPE 核心性质：q_m·k_n 只依赖相对位置 (m-n)

    同一对向量在 (m, n) 与 (m+d, n+d) 处的点积相同。
    """
    rope = RotaryEmbedding(HEAD_DIM, base=BASE, max_seq_len=64)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def dot(m, n):
        pos_q = torch.tensor([[m]])
        pos_k = torch.tensor([[n]])
        qr, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), *rope(pos_q))
        kr, _ = apply_rotary_pos_emb(torch.zeros_like(k), k, *rope(pos_k))
        return (qr * kr).sum().item()

    d = 5
    base = dot(3, 1)
    shifted = dot(3 + d, 1 + d)
    assert math.isclose(base, shifted, rel_tol=1e-4, abs_tol=1e-5)
