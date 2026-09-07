"""
SwiGLU MLP

标准 Transformer MLP:  y = W2 @ relu(W1 @ x)
SwiGLU MLP (Qwen3):    y = down_proj(silu(gate_proj(x)) * up_proj(x))

其中 silu(x) = x * sigmoid(x) (也叫 Swish)

gate 和 up 是两个独立的投影, gate 通过 silu 激活后作为 "门控" 来调节 up 的输出
这种设计比单层 relu 有更好的表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # gate 和 up 共享同一输入 x，融合成一个 gate_up_proj 大 GEMM。
        # 输出维度 = 2 * intermediate_size，forward 中再按 config 动态切回。
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 一次大 GEMM 后按中间维度切回 gate / up
        gate_up = self.gate_up_proj(x)
        gate = gate_up[..., :self.intermediate_size]
        up = gate_up[..., self.intermediate_size:]
        return self.down_proj(F.silu(gate) * up)
