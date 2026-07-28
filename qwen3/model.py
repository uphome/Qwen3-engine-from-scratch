"""
Qwen3Model — Transformer backbone
Qwen3ForCausalLM — 完整语言模型（backbone + lm_head）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3Config
from .norm import RMSNorm
from .rope import RotaryEmbedding
from .decoder import Qwen3DecoderLayer


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids: torch.Tensor,
                kv_cache=None) -> torch.Tensor:
        B, S = input_ids.shape

        # Token embedding: 每个 token ID -> 向量
        hidden_states = self.embed_tokens(input_ids)

        # 位置 ID
        if kv_cache is not None and kv_cache.seq_len > 0:
            # Decode 模式：新 token 的位置 = 已缓存的长度
            start_pos = kv_cache.seq_len
            position_ids = torch.arange(start_pos, start_pos + 1, device=input_ids.device).unsqueeze(0)
        else:
            # Prefill 模式：位置从 0 开始
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # 提前计算 RoPE cos/sin
        position_embeddings = self.rotary_emb(position_ids)

        # 注意力掩码
        if kv_cache is not None and kv_cache.seq_len > 0:
            # Decode 模式：Q 只有 1 个位置，可以看到所有缓存的 K + 自己的 K
            kv_len = kv_cache.seq_len + S
            causal_mask = torch.zeros(1, 1, S, kv_len, device=input_ids.device, dtype=hidden_states.dtype)
        else:
            # Prefill 模式：标准因果掩码
            causal_mask = torch.full((S, S), float("-inf"), device=input_ids.device, dtype=hidden_states.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # 逐层前向传播
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, causal_mask, position_embeddings,
                                  kv_cache=kv_cache, layer_idx=i)

        # 更新缓存计数（prefill 加 S，decode 加 1）
        if kv_cache is not None:
            kv_cache.advance_seq_len(S)

        # 最终归一化
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        # lm_head: 将 hidden_states 映射到 vocab_size 维度, 输出每个 token 的概率
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor,
                kv_cache=None) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        kv_cache: 传入则使用 KV cache（prefill 存，decode 读+存）
        returns: logits (batch, seq_len, vocab_size)
        """
        hidden_states = self.model(input_ids, kv_cache=kv_cache)
        return self.lm_head(hidden_states)
