"""
Scheduler — continuous batching 的"主动决策者"

每个调度步做三件事:
  1. 组批   waiting 有请求 → 组 prefill 批；running 有请求 → 组 decode 批
  2. 前向   驱动循环拿到 batch 后做一次 forward
  3. 收尾   on_step_done: 采样结果回写请求，完成者踢出并归还 KV 块

调度策略（贪心 + 防饥饿）:
  - running 未满且有 waiting → prefill 批（补位，避免 decode 永远占着 GPU）
  - running 已满 → decode 批（推进所有 running）
  - running 空 → prefill 批（启动新请求）

队列:
  waiting  → 新请求（已分配 PagedKVCache 句柄，未占物理块）
  running  → prefill 完成、正在自回归的请求
  finished → 已完成，保留输出供收集
"""

from __future__ import annotations

import os
from collections import deque

import torch

from .batch import Batch
from .prefix_cache import compute_prefix_hashes
from .request import Request, WAITING
from .PagedKVcache import KVCachePool, PagedKVCache


class Scheduler:
    def __init__(self, pool: KVCachePool, batch_size: int = 1,
                 chunk_size: int | None = None):
        self.pool = pool
        self.batch_size = batch_size
        # chunked prefill：每步最多 prefill chunk_size 个 token（0 = 关闭）。
        # 默认 512（vLLM 同款量级）；长 prompt 切片后不会卡住 decode。
        if chunk_size is None:
            chunk_size = int(os.environ.get("QWEN3_CHUNK_SIZE", "512"))
        self.chunk_size = max(0, chunk_size)
        self._prefill_turn = False   # 上一步是否 prefill（与 decode 交替）
        # 容量不足时是否强制淘汰 ref=0 的缓存块（可用性优先，vLLM 同款思路）
        self.evict_on_demand = (
            os.environ.get("QWEN3_CACHE_EVICT_ON_DEMAND", "1") != "0"
        )
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.step = 0

    # ---- 入队 ----

    def _ensure_capacity(self, req: Request, need: int, reserved_blocks: int):
        """确保有 need 个新块可用；不足时先淘汰 ref=0 的缓存块再重试

        缓存淘汰只动 `_cached_lru`（ref=0），当前请求已匹配的块
        （ref>0）不受影响。QWEN3_CACHE_EVICT_ON_DEMAND=0 可关闭。
        """
        available = self.pool.free_count - reserved_blocks
        if need <= available:
            return
        if self.evict_on_demand:
            self.pool.evict_cache_for(need - available)
            available = self.pool.free_count - reserved_blocks
            if need <= available:
                return
        raise RuntimeError(
            f"[Scheduler] 池剩余 {self.pool.free_count} 块"
            f"（已入队 {reserved_blocks}，缓存淘汰后仍不足），"
            f"req {req.request_id} 需要 {need} 块，拒绝入队")

    def add(self, req: Request, reserved_blocks: int = 0) -> int:
        """新请求入队：分配 kv_cache 句柄；尝试前缀缓存（完全 / 部分命中）

        Args:
            reserved_blocks: 已入队未归还的块数记账（防批内超卖）
        Returns:
            命中的缓存块数（0 = 未命中，走正常 prefill）
        """
        bs = self.pool.block_size
        # 含 decode 空间的容量需求（预检保护；命中块会从需求中扣除）
        needed_full = (req.input_len + req.max_new_tokens + bs - 1) // bs
        req.kv_cache = PagedKVCache(self.pool)
        req.state = WAITING

        # 哈希链（QWEN3_PREFIX_CACHE=0 可关闭，消融/对照用）
        if os.environ.get("QWEN3_PREFIX_CACHE", "1") == "0":
            hashes: list[int] = []
        else:
            hashes = compute_prefix_hashes(req.input_ids[0].tolist(), bs)
        req.prefix_hashes = hashes

        matched = self.pool.match_prefix_partial(hashes)
        try:
            # ---- 阶段 1：完全命中（块对齐 + 全部命中）→ 跳过 prefill ----
            if hashes and len(matched) == len(hashes) and req.input_len % bs == 0:
                # COW 私有块计为新块：复用块数 = len(matched) - 1
                self._ensure_capacity(req, needed_full - (len(matched) - 1),
                                      reserved_blocks)
                # COW 最后一个块：首个 decode 要重算 prompt 末 token 的 logits
                last, last_hash = matched[-1], hashes[-1]
                priv = self.pool.alloc(1)[0]
                self.pool.copy_block(last, priv)
                self.pool.release_block(last, last_hash)
                blocks = matched[:-1] + [priv]
                req.kv_cache.bind_prefix(blocks, hashes[:-1] + [None])
                req.kv_cache.seq_len = req.input_len - 1
                self.pool.seq_lens[req.kv_cache.row_id] = req.input_len - 1
                req.prefill_last_token = int(req.input_ids[0, -1].item())
                req.start_prefill()
                req.enter_decoding()
                self.running.append(req)
                self.pool.prefix_hits += 1
                self.pool.prefix_hit_blocks += len(blocks)
                return len(blocks) - 1      # 复用块（COW 私有块算新块）

            # ---- 阶段 2：部分命中 → 复用前缀块，prefill 只算后缀 ----
            # 短历史回退：部分命中的固定成本（28 层权重读取 + 小 GEMM）
            # 在历史较短时高于全量 flash prefill（实测 0.6B/A100 交叉点
            # ~1k tokens）。低于阈值时释放命中块，走全量 prefill。
            min_partial = int(os.environ.get("QWEN3_PREFIX_PARTIAL_MIN", "1024"))
            if matched and len(matched) * bs < min_partial:
                for b, h in zip(matched, hashes):
                    self.pool.release_block(b, h)
                matched = []
            if matched:
                self._ensure_capacity(req, needed_full - len(matched),
                                      reserved_blocks)
                req.kv_cache.bind_prefix(matched, hashes[:len(matched)])
                offset = len(matched) * bs
                req.kv_cache.seq_len = offset
                self.pool.seq_lens[req.kv_cache.row_id] = offset
                req.prefill_offset = offset
                self.waiting.append(req)
                return len(matched)

            # ---- 未命中：普通 prefill ----
            self._ensure_capacity(req, needed_full, reserved_blocks)
            self.waiting.append(req)
            return 0

        except Exception:
            # 失败回滚：已匹配块放回缓存 + 归还句柄行（防长期服务槽位泄漏）
            for b, h in zip(matched, hashes):
                self.pool.release_block(b, h)
            if req.kv_cache is not None and not req.kv_cache.block_table:
                req.kv_cache.free()
            raise

    def has_pending(self) -> bool:
        """调度循环是否还要继续"""
        return bool(self.waiting) or bool(self.running)

    # ---- 每步调度 ----

    def schedule(self) -> Batch | None:
        """挑出本步执行的请求集合（要么全 prefill，要么全 decode）

        chunked prefill 的节奏：running 非空时 prefill / decode 交替，
        保证长 prompt 的切片不会让已运行请求的 decode 停顿整个 prefill。
        """
        if self.running:
            # running 有空位且 waiting 有人：prefill 补位（与 decode 交替）
            if (len(self.running) < self.batch_size and self.waiting
                    and not self._prefill_turn):
                return self._schedule_prefill()
            return self._schedule_decode()

        if self.waiting:
            return self._schedule_prefill()

        return None

    def _schedule_prefill(self) -> Batch:
        # Phase 3: 一次 prefill 最多 batch_size 个请求（右 pad 组批，见 Batch）。
        # prefill 完立即全部进 running，下一步就能与现有请求组批 decode。
        # ⚠ 补位只能装 running 剩下的空位：slots = batch_size - len(running)。
        #   若按 min(batch_size, waiting) 取数，running 已有人时补位会超编
        #   （如 batch_size=4, running=2, 补位取 4 → running=6），
        #   超编后 decode 的 running[:n] 切片会饿死队尾请求。
        slots = self.batch_size - len(self.running)
        n = min(slots, len(self.waiting))
        reqs = [self.waiting.popleft() for _ in range(n)]
        for req in reqs:
            req.start_prefill()
            # chunked prefill：本步只 prefill 一段（chunk_size=0 则整段）
            remaining = req.input_len - req.prefill_offset
            take = remaining if self.chunk_size <= 0 else min(self.chunk_size, remaining)
            req.prefill_end = req.prefill_offset + take
        self._prefill_turn = True
        return Batch("prefill", reqs)

    def _schedule_decode(self) -> Batch:
        # 批内所有 running 一起 decode（最多 batch_size 个，超出留到下一步）
        n = min(self.batch_size, len(self.running))
        self._prefill_turn = False
        return Batch("decode", self.running[:n])

    # ---- 收尾 ----

    def on_step_done(self, batch: Batch, next_tokens: torch.Tensor):
        """前向 + 采样完成后的逐请求收尾。

        next_tokens: (B, 1) 批内每个请求采样出的下一个 token
        """
        for i, req in enumerate(batch.requests):
            # chunked prefill：中间 chunk 的 logits 只是"半截上下文"的产物，
            # 不能作为首 token（否则输出会多出伪 token）；但收尾逻辑照常执行
            is_partial_chunk = (batch.mode == "prefill"
                                and req.prefill_end < req.input_len)
            if not is_partial_chunk:
                req.append(next_tokens[i].item())

            if batch.mode == "prefill":
                # 记录完整 prompt 块的哈希（free 时登记前缀缓存）
                cache = req.kv_cache
                for i, h in enumerate(req.prefix_hashes):
                    if i < len(cache.block_hashes):
                        cache.block_hashes[i] = h
                req.prefill_offset = req.prefill_end
                if req.prefill_offset < req.input_len:
                    # chunked prefill 还有剩余：放回队头继续（状态保持 prefilling）
                    self.waiting.appendleft(req)
                else:
                    # 整段 prefill 完成：进入自回归
                    req.enter_decoding()
                    self.running.append(req)
            else:
                # decode 步：结束后判定出批并归还 KV 块
                if req.is_finished():
                    req.finish()
                    self.running.remove(req)
                    self.finished.append(req)

    def __repr__(self):
        return (f"Scheduler(waiting={len(self.waiting)}, "
                f"running={len(self.running)}, finished={len(self.finished)}, "
                f"pool_free={self.pool.free_count})")
