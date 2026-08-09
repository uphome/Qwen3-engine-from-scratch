"""
Transformer Decoder Layer

Pre-Norm 架构 (和 Post-Norm 的区别):

  Pre-Norm (Qwen3 使用):
    x → RMSNorm → Attention → + → RMSNorm → MLP → +
    |______________________|    |__________________|
          残差连接                    残差连接

  Post-Norm (原始 Transformer):
    x → Attention → + → LayerNorm → MLP → + → LayerNorm

Pre-Norm 训练更稳定，收敛更快，是现代 LLM 的标配
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .norm import RMSNorm
from .attention import Qwen3Attention
from .mlp import Qwen3MLP


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache=None,   # PagedKVCache | list[PagedKVCache] | None
        layer_idx: int = 0,
        input_lens: list[int] | None = None,
        row_ids: torch.Tensor | None = None,   # decode 批常驻 2D 块表行号（每步一次，各层共享）
    ) -> torch.Tensor:
        # 自注意力 + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask,
                                       kv_cache=kv_cache, layer_idx=layer_idx,
                                       input_lens=input_lens, row_ids=row_ids)
        hidden_states = residual + hidden_states

        # MLP + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
