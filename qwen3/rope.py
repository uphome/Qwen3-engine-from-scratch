"""
Rotary Position Embedding (RoPE)

RoPE 核心思想：用旋转矩阵编码位置信息
  - 对 Q, K 的每对相邻维度 (x0, x1)，应用 2D 旋转：
    [x0', x1'] = [x0*cos(mθ) - x1*sin(mθ), x0*sin(mθ) + x1*cos(mθ)]
  - m 是位置索引，θ_i = base^(-2i/d) 是频率
  - 低维频率高（变化快），高维频率低（变化慢）

优势：
  - Q·K 的点积自然包含 (m-n) 的相对位置信息
  - 可以外推到训练时没见过的序列长度
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        # 频率: theta_i = 1 / (base^(2i/d)), i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (batch, seq_len) — 每个 token 的位置
        # 计算 angle = position * freq, 形状 (seq_len, head_dim/2)
        freqs = torch.outer(position_ids[0].float(), self.inv_freq)
        # 复制拼接成完整 head_dim: (seq_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # 返回 cos, sin, 形状 (1, seq_len, head_dim) — batch 维度广播
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将后半部分取负并与前半部分交换: [x1, x2] -> [-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q 和 K 应用旋转位置编码

    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: (1, seq_len, head_dim)  -> unsqueeze 成 (1, 1, seq_len, head_dim)
    """
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
