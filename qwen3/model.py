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
                kv_cache=None, input_lens: list[int] | None = None) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        kv_cache: 单个 PagedKVCache（单请求）| list[PagedKVCache]（批）
                  | None。传入则使用 KV cache（prefill 存，decode 读+存）
        input_lens: 批内各请求的有效长度（prefill 批右 pad 后必须传入，
                    否则模型无法知道每请求的真实边界）。decode 批不需要。
        returns: logits (batch, seq_len, vocab_size)

        批量设计（continuous batching 基础）：
          - 核心思想：单请求 = 长度为 1 的批，一套代码覆盖两种场景
          - 批内每个请求的 seq_len 独立（缓存长度、可见范围、位置都不共享）
          - 三个关键点：
            ① position_ids 逐请求取自己的 start_pos（RoPE 位置不能共享）
            ② mask 逐请求限制可见列（防止请求间 KV 泄漏）
            ③ advance_seq_len 逐请求推进（各自维护进度）
        """
        B, S = input_ids.shape

        # ---- 统一 kv_cache 为 list ----
        # 单请求（B=1）与批量（B>1）共用同一套批逻辑，单请求只是批的特例。
        # 后面所有逻辑都按"批"处理，批大小为 1 时自然退化为单请求行为。
        # assert 是安全网：input_ids 有几个请求，就必须有几个缓存句柄。
        if kv_cache is None:
            caches = None
        elif isinstance(kv_cache, (list, tuple)):
            caches = list(kv_cache)
            assert len(caches) == B, f"kv_caches 数量 {len(caches)} 必须等于 batch {B}"
        else:
            caches = [kv_cache]

        # Token embedding: 每个 token ID -> 向量
        hidden_states = self.embed_tokens(input_ids)

        if caches is not None and caches[0].seq_len > 0:
            # ---- Decode 模式 ----
            # ① position_ids 必须逐请求独立：
            #    请求 A 缓存了 100 token → 新 token 位置 = 100
            #    请求 B 缓存了 50  token → 新 token 位置 = 50
            #    RoPE 的位置决定绝对位置信息，若共享 start_pos，
            #    请求 B 的 token 会被当成位置 100，相对位置全错。
            #    单请求时代是标量 start_pos，批量时代改为逐行张量。
            start_pos = torch.tensor([c.seq_len for c in caches],
                                     device=input_ids.device)
            position_ids = start_pos.unsqueeze(1).expand(B, S) #decode模式 S=1

            # ---- ② mask (B, 1, S, max_kv)：批安全的灵魂 ----
            # 问题：attention 的 K/V 张量在 batch 内必须矩形 (B, H, max_kv, D)，
            #       但各请求 kv_len 不同，短请求要"补齐"到 max_kv。
            #       补齐的列是别的请求的 KV——绝不能让它看到！
            # 解法：初始全 0（可见），逐请求把"自己 kv_len 之外"的列置 -inf，
            #       softmax 权重为 0 → 请求间逻辑隔离。
            #       例: 请求 A kv=110 → 全可见；请求 B kv=60 → 后 50 列 -inf。
            # 注意：decode 走 Triton kernel 时不看 mask（kernel 每请求单独跑，
            #       只读自己的块表），此 mask 是 PyTorch 兜底路径（CPU/prefill）用的。
            max_kv = max(c.seq_len + S for c in caches)
            causal_mask = torch.zeros(B, 1, S, max_kv, device=input_ids.device,
                                      dtype=hidden_states.dtype)
            for i, c in enumerate(caches):
                kv_len_i = c.seq_len + S
                if kv_len_i < max_kv:
                    causal_mask[i, :, :, kv_len_i:] = float("-inf")
            lens = [S] * B
        else:
            # ---- Prefill 批模式 ----
            # 批内各请求 prompt 长度可能不同（外部已右 pad 到 S_max，
            # input_lens 给出每请求真实长度）：
            # ① position_ids 逐请求 0..len_i-1（RoPE 位置不能共享）：
            #    pad 位随便填 0——它的输出会被采样丢弃，位置无关紧要
            # ② mask (B, 1, S, S) 逐请求限制可见范围（批安全的灵魂）：
            #    - 行/列 ≥ len_i 的区域（pad 或别的请求的 token）→ -inf
            #    - 纯 pad 行整行 -inf 会让 softmax 出 NaN，所以给 pad 行
            #      留第 0 列可见（自己请求的第一个 token，无跨请求泄漏）
            lens = input_lens if input_lens is not None else [S] * B
            # repeat 而非 expand：expand 是共享内存视图，逐行写 pad 会
            # 写穿到所有行（短请求的 pad 会把长请求的位置也改成 0）
            position_ids = torch.arange(S, device=input_ids.device) \
                .unsqueeze(0).repeat(B, 1)
            for i, l_i in enumerate(lens):
                if l_i < S:
                    position_ids[i, l_i:] = 0

            row = torch.arange(S, device=input_ids.device).view(1, S, 1)       # (1, S, 1)
            col = torch.arange(S, device=input_ids.device).view(1, 1, S)      # (1, 1, S)
            lens_t = torch.tensor(lens, device=input_ids.device).view(B, 1, 1)  # (B, 1, 1)
            # 有效区域: 行/列都在自己长度内 + 因果（col <= row）
            valid = (row < lens_t) & (col < lens_t) & (col <= row)
            # pad 行留第 0 列可见，防止整行 -inf → softmax NaN
            pad_keep = (row >= lens_t) & (col == 0)
            causal_mask = torch.full((B, S, S), float("-inf"),
                                     device=input_ids.device,
                                     dtype=hidden_states.dtype)
            causal_mask[valid | pad_keep] = 0
            causal_mask = causal_mask.unsqueeze(1)   # (B, 1, S, S)

        # 提前计算 RoPE cos/sin（position_ids 已是逐请求独立的）
        position_embeddings = self.rotary_emb(position_ids)

        # 逐层前向传播：caches（list）整包传给每层，
        # 层内根据 layer_idx 取每个请求自己的第 i 层缓存
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, causal_mask, position_embeddings,
                                  kv_cache=caches, layer_idx=i)

        # ---- ③ 逐请求推进缓存进度 ----
        # 每个请求的句柄是独立对象，必须逐个 advance（不能只推 caches[0]），
        # 否则下次 decode 的位置计算就全错了。
        # prefill 批还要按各请求自己的实际长度推进（lens），
        # 统一用批内统一 S 会把短请求的 seq_len 多推 pad 的长度。
        if caches is not None:
            for i, c in enumerate(caches):
                c.advance_seq_len(lens[i])

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
                kv_cache=None, input_lens: list[int] | None = None) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        kv_cache: 传入则使用 KV cache（prefill 存，decode 读+存）
        input_lens: prefill 批各请求的有效长度（右 pad 后必须传），decode 不需要
        returns: logits (batch, seq_len, vocab_size)
        """
        hidden_states = self.model(input_ids, kv_cache=kv_cache,
                                   input_lens=input_lens)
        return self.lm_head(hidden_states)
