"""
Qwen3Model — Transformer backbone
Qwen3ForCausalLM — 完整语言模型（backbone + lm_head）
"""

from __future__ import annotations

import os

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

        row_ids = None   # decode 批的常驻 2D 块表行号（prefill/兜底路径不用）
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

            # ---- ② mask：仅 PyTorch 兜底路径需要 ----
            # decode 走 Triton kernel 时不看 mask（kernel 每请求单独跑，只读
            # 自己的块表）→ 直接传 None，省掉 (B,1,S,max_kv) 构造的
            # index_elementwise（profiler 显示每步 448 次，占 18%）。
            # 仅在 CPU / 非 bf16 / 强制 QWEN3_PAGED_ATTN=pytorch 时构造。
            # 与 attention.py 的 use_triton 判断保持完全一致：
            # S==1（decode 步）+ seq_len>0（非 prefill）+ GPU + 未强制 pytorch
            use_triton = (
                S == 1 and caches[0].seq_len > 0 and input_ids.is_cuda
                and os.environ.get("QWEN3_PAGED_ATTN", "triton") == "triton"
            )
            row_ids = None
            if use_triton:
                causal_mask = None
                # ---- ③ 常驻 2D 块表：层循环外每步只做一次 ----
                # 预留本步新 token 的物理页（跨页边界时申请，每请求至多 1 页），
                # 保证 28 层循环内块表完全稳定；行号组装一次，各层共享。
                # kernel 直接引用 pool.block_table_2d / pool.seq_lens，
                # 层内零组装、零 HtoD（旧版每层 zeros + B 次行拷贝 + seq_len HtoD）。
                for c in caches:
                    c.reserve_next()
                row_ids = torch.tensor(
                    [c.row_id for c in caches],
                    dtype=torch.int32, device=input_ids.device)
            else:
                # 问题：attention 的 K/V 张量在 batch 内必须矩形 (B, H, max_kv, D)，
                #       但各请求 kv_len 不同，短请求要"补齐"到 max_kv。
                #       补齐的列是别的请求的 KV——绝不能让它看到！
                # 解法：初始全 0（可见），逐请求把"自己 kv_len 之外"的列置 -inf，
                #       softmax 权重为 0 → 请求间逻辑隔离。
                #       例: 请求 A kv=110 → 全可见；请求 B kv=60 → 后 50 列 -inf。
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

            # ---- ③ mask：仅标准 attention 兜底需要 ----
            # prefill 走 flash（QWEN3_FLASH_ATTN=triton 且 GPU）时，varlen kernel
            # 用 cu_seqlens 在 kernel 内处理 causal + 逐请求隔离，不需要 4D mask
            # （与 attention.py 的 use_flash 判断保持一致）→ 直接传 None，省掉
            # (B,1,S,S) 构造的 index_elementwise 开销。仅 standard 兜底需构造。
            use_flash = (
                input_ids.is_cuda
                and os.environ.get("QWEN3_FLASH_ATTN", "triton") != "pytorch"
            )
            if use_flash:
                causal_mask = None
            else:
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
        # row_ids：decode 批在层循环外组装一次的常驻 2D 块表行号，各层共享
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, causal_mask, position_embeddings,
                                  kv_cache=caches, layer_idx=i, input_lens=lens,
                                  row_ids=row_ids)

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

    def forward_decode(self, input_ids: torch.Tensor,
                       position_embeddings: tuple[torch.Tensor, torch.Tensor],
                       row_ids: torch.Tensor,
                       caches: list) -> torch.Tensor:
        """decode 步的"捕获友好"前向：纯 GPU 计算，零 Python 状态操作

        与 forward() 的 decode 分支等价，但**所有动态/状态部分由调用方负责**：
          - input_ids: (B, 1) 已含本步新 token（调用方 copy_ 进固定张量）
          - position_embeddings: (cos, sin) 已按本步 start_pos 算好（调用方传）
          - row_ids: (B,) 常驻 2D 块表行号（调用方组装）
          - caches:   list[PagedKVCache]（长度 == B）

        本函数内不做：reserve_next（页分配）、advance_seq_len（状态推进）、
        position_ids 构造、row_ids 组装——这些都有 Python 循环/分配/状态，
        是 CUDA Graph 捕获的禁区。调用方在 replay 前后处理。

        为什么能进图：
          - 层循环是 Python for，但每层只调固定形状的 GPU 算子（形状 B 恒定）
          - 无 torch.empty / tensor 构造（全部中间量由算子内部分配，预热后
            内存池稳定；唯一新增输出是每层的 hidden_states，形状固定）
          - row_ids / position_embeddings 都是外部传入的固定张量
        """
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, None, position_embeddings,
                                  kv_cache=caches, layer_idx=i, input_lens=None,
                                  row_ids=row_ids)
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

    def forward_decode(self, input_ids: torch.Tensor,
                       position_embeddings: tuple[torch.Tensor, torch.Tensor],
                       row_ids: torch.Tensor,
                       caches: list) -> torch.Tensor:
        """decode 步捕获友好的完整前向（backbone + lm_head）——供 CUDA Graph 捕获

        见 Qwen3Model.forward_decode 的说明：本函数零 Python 状态操作，
        reserve/advance/position/row_ids 全由调用方负责。返回 (B, 1, vocab) logits。
        """
        hidden_states = self.model.forward_decode(
            input_ids, position_embeddings, row_ids, caches)
        return self.lm_head(hidden_states)
