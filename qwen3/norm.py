"""
RMSNorm — 比 LayerNorm 更高效的归一化

标准 LayerNorm: y = (x - mean) / std * gamma + beta    (需要 mean 和 std)
RMSNorm:        y = x / RMS(x) * gamma                 (只需要 RMS, 无 beta)

RMS(x) = sqrt(mean(x^2) + eps)

优势：省去均值计算和偏置参数，速度更快，效果相当
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先转 float32 做归一化（避免 bfloat16 精度问题），最后再转回原 dtype
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((x_float * rms) * self.weight).to(x.dtype)
