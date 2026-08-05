"""
Rotary Position Embedding (RoPE)

RoPE 核心思想：用旋转矩阵编码位置信息
  - 对 Q, K 的每对维度 (x_i, x_{i+64})，应用 2D 旋转：
    [x0', x1'] = [x0*cos(mθ) - x1*sin(mθ), x0*sin(mθ) + x1*cos(mθ)]
  - m 是位置索引，θ_i = base^(-2i/d) 是频率
  - 低维频率高（变化快，区分近邻 token），高维频率低（变化慢，长距离信息）

优势：
  - Q·K 的点积自然包含 (m-n) 的相对位置信息：
    q_m·k_n = (R_m·q)ᵀ·(R_n·k) = qᵀ·R_(n-m)·k，只依赖相对位置
  - 可以外推到训练时没见过的序列长度

两阶段分工（本文件的组织方式）:
  RotaryEmbedding      = 预计算角度表（有状态，全模型一份，与 Q/K 无关）
  apply_rotary_pos_emb = 消费角度表（无状态，每层 attention 调用，与位置无关）

  为什么拆开？角度只与位置有关、与层无关（28 层共享同一套频率），
  所以 model.py 在循环外算一次；而旋转每层的 Q/K 都不同，每层各做一次。
  合在一起的话每层都要重算三角函数，浪费 27 倍。

配套约定（重要，混用会静默算错）:
  - 配对方式：前后配对 —— 第 i 对维度 = (第 i 维, 第 i+64 维)
  - cos/sin 用 cat((freqs, freqs)) 构造：前后两半共享同一频率表
  - rotate_half 用"交换前后两半并取负前半"实现 —— 与 cat 的配对一一对应
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """阶段一：预计算角度表

    有状态部分只有 inv_freq（频率表，注册为 buffer 随模型迁移 dtype/device）。
    forward 只依赖 position_ids，产出 cos/sin 角度表 —— 不知道 Q/K 长什么样。
    """

    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        # 频率表: θ_i = 1 / (base^(2i/d)), i = 0, 1, ..., d/2-1
        #   - 只有 d/2 个频率：每"对"维度（2D 旋转）共享一个频率，d 维 = d/2 对
        #   - torch.arange(0, head_dim, 2) → [0, 2, 4, ..., d-2]（取偶数下标）
        #   - base 越大频率越低（旋转越慢），Qwen3 用 1e6，长上下文友好
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # register_buffer: 不占梯度、不参与参数计数；persistent=False 不进 state_dict
        # （inv_freq 可由参数推导，无需随权重保存）
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """位置 → 旋转角度 → cos/sin 角度表

        position_ids: (batch, seq_len) — 每个 token 的位置
          - 单请求 prefill: (1, S)，位置 0..S-1
          - 批 decode: (B, 1)，每行是各请求自己的 seq_len（各不相同！）
        """
        # 旋转角度 = 位置 × 频率:
        #   freqs[b, s, i] = position_ids[b, s] * inv_freq[i]  → (B, S, d/2)
        # 这是"外积"的批量推广（旧版 torch.outer(position_ids[0], inv_freq)
        # 只能处理单行，批 decode 每行位置不同必须逐行算）。
        # 注意: (B,S) @ (d/2,) 是矩阵向量乘（得 (B,)），不是外积 —— 必须用
        # einsum 显式声明输出形状 (B, S, d/2)。
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq)
        #不同batch 拥有不同的 token位置

        # 复制拼接成完整 head_dim: (B, S, head_dim)
        #   为什么 cat？角度总数（d/2 个）只有维度数（d 个）的一半 ——
        #   每对维度共享一个旋转角。cat 两遍让 [第 i 维] 和 [第 i+64 维]
        #   拿到同一个角度，正好匹配前后配对约定（rotate_half 与之配套）。
        emb = torch.cat((freqs, freqs), dim=-1)

        # 角度 → 三角函数。返回 (B, S, head_dim)：
        #   apply_rotary_pos_emb 里 unsqueeze(1) 成 (B, 1, S, head_dim)，
        #   与 q/k 的 (B, H, S, D) 沿头维广播
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """对每对维度旋转 90°（几何上 = 乘以虚数单位 i）

    前后配对约定下，"第 i 对 = (x_i, x_{i+64})"旋转 90° 得到 (-x_{i+64}, x_i)，
    即交换前后两半、前半取负:
      [x_0, ..., x_63, x_64, ..., x_127] → [-x_64, ..., -x_127, x_0, ..., x_63]

    它是旋转公式 x' = x·cosθ + rotate_half(x)·sinθ 里的"90° 转分量"——
    把 2D 旋转矩阵 [cosθ, −sinθ; sinθ, cosθ] 逐对的作用，
    变成一次交换 + 一次广播乘加（省去逐对拆分再拼接）。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """阶段二：执行旋转

    对 Q 和 K 应用旋转位置编码（每层 attention 调用，cos/sin 各层相同）

    q, k:   (batch, heads, seq_len, head_dim)
    cos, sin: (batch, seq_len, head_dim) → unsqueeze(1) 成 (B, 1, S, D)
              沿 heads 维广播（所有头共享同一位置编码）

    旋转公式（由 2D 旋转矩阵展开而来）:
      x' = x·cosθ + rotate_half(x)·sinθ
    等价于复数乘法 z·e^{iθ}（实部 + 虚轴分量），见文件头推导。
    """
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
