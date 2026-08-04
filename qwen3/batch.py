"""
Batch — 一个调度步内参与同一前向的请求集合

Phase 1 雏形：只打包请求列表 + 模式标记，驱动循环逐个请求处理。
Phase 2/3 将在这里组装批内张量：
  - input_ids  (B, S)          decode 批: S=1；prefill 批: 右 pad 到 S_max
  - position_ids (B, S)        每请求实际位置（pad 位为 0，被 mask 掉）
  - attention_mask (B, 1, S, kv_len)  批内每请求一行
  - block_table (B, max_pages) 收集每请求块表并右 pad
"""

from __future__ import annotations

import torch

from .request import Request


class Batch:
    """一个调度步的请求集合（要么全 prefill，要么全 decode）

    职责：把批内请求组装成一次 forward 需要的张量。
    组装发生在 schedule() 返回之后、model() 调用之前。

    Phase 2 现状：
      - prefill 批: size=1（逐请求 prefill，批量前向留到 Phase 3）
      - decode 批: 真批量 (B, 1)，position_ids/mask 在 model.py 里
        基于各请求 kv_cache.seq_len 构造
    """

    def __init__(self, mode: str, requests: list[Request]):
        assert mode in ("prefill", "decode")
        self.mode = mode
        self.requests: list[Request] = requests

    @property
    def size(self) -> int:
        return len(self.requests)

    def build_input_ids(self) -> torch.Tensor:
        """批内 input_ids: (B, S)

        decode 批: 每请求取 cur_token（上一步采样出的那个 token）→ (B, 1)
        prefill 批: 每请求取完整 prompt → (B, S_i)，Phase 3 前 size=1 不 pad
        """
        if self.mode == "decode":
            return torch.cat([r.cur_token for r in self.requests], dim=0)
        return torch.cat([r.input_ids for r in self.requests], dim=0)

    def build_kv_caches(self) -> list:
        """批内各请求的 KV cache 句柄（与 input_ids 行一一对应）"""
        return [r.kv_cache for r in self.requests]

    def __repr__(self):
        return (f"Batch(mode={self.mode}, size={self.size}, "
                f"ids={[r.request_id for r in self.requests]})")
