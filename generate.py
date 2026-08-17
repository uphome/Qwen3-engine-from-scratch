"""
自回归生成 + 采样策略

LLM 生成文本的核心循环:
  1. 将整个序列送入模型, 得到 logits
  2. 取最后一个位置的 logits, 采样出下一个 token
  3. 将新 token 拼接到序列末尾
  4. 重复直到生成结束 token 或达到最大长度

采样策略:
  - Temperature: 控制随机性 (0 = 贪心, >0 = 缩放 logits)
  - Top-K: 只保留概率最高的 K 个 token
  - Top-P (Nucleus): 保留累积概率不超过 P 的 token

后续将在此模块中加入:
  - PagedAttention
  - 连续批处理 (Continuous Batching)
  - 算子融合 (Operator Fusion)
"""

import time

import torch
import torch.nn.functional as F

from qwen3 import Qwen3ForCausalLM, KVCachePool, PagedKVCache


@torch.no_grad()
def generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: int = 151645,
    kv_cache_pool: KVCachePool | None = None,
    graph_runner=None,
) -> tuple[torch.Tensor, dict]:
    """自回归生成 (有 KV cache)

    kv_cache_pool: 全局共享池（KVCachePool）。若为 None 则自动创建一个
                   本次请求专用的小池（不做跨请求复用）。
    graph_runner:  GraphRunner | None。非 None 时 decode 步走 CUDA Graph
                   （replay），prefill 步保持 eager（model 前向）。
    """
    generated = input_ids.clone()
    stats = {"step_times": [], "input_len": input_ids.shape[1]}

    # 创建 KV cache
    if kv_cache_pool is None:
        block_size = 16
        max_tokens = input_ids.shape[1] + max_new_tokens
        num_blocks = (max_tokens + block_size - 1) // block_size
        dtype = next(model.parameters()).dtype
        kv_cache_pool = KVCachePool(
            num_blocks=num_blocks,
            num_layers=model.config.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            device=input_ids.device,
            dtype=dtype,
        )

    kv_cache = PagedKVCache(kv_cache_pool)

    try:
        for step in range(max_new_tokens):
            t0 = time.perf_counter()

            if step == 0:
                # Prefill: 第一次送入完整 prompt，所有 K,V 写入 cache
                logits = model(generated, kv_cache=kv_cache)
            elif graph_runner is not None and input_ids.is_cuda:
                # Decode 走 CUDA Graph：replay 替代 model 前向。
                # 与 run_batched 的 decode 分支契约一致：reserve/advance
                # 是 Python 状态操作，必须在图外（forward_decode 内零状态）。
                kv_cache.reserve_next()          # 页预留（原 forward 内部做）
                start_pos = kv_cache.seq_len     # 图外读（replay 前）
                positions = torch.tensor([[start_pos]], dtype=torch.long,
                                         device=input_ids.device)
                cos, sin = model.model.rotary_emb(positions)
                rows = torch.tensor([kv_cache.row_id], dtype=torch.int32,
                                    device=input_ids.device)
                logits = graph_runner.replay(
                    generated[:, -1:], positions,
                    cos.to(torch.bfloat16), sin.to(torch.bfloat16), rows)
                logits = logits[:1]              # (b,1,vocab) → 前 k 行有效
                kv_cache.advance_seq_len(1)      # 状态推进（原 forward 内部做）
            else:
                # Decode: 只送入最后一个 token，其余 K,V 从 cache 读
                logits = model(generated[:, -1:], kv_cache=kv_cache)

            # CUDA 异步执行：必须同步才能测到真实墙钟（等 GPU 算完）。
            # 不同步的话 perf_counter 只测到"CPU 提交 kernel 的时间"，
            # 对 Python 逐页路径会低估数倍（GPU 还在后台排队执行）。
            if input_ids.is_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            stats["step_times"].append(dt)

            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :]  # shape: (batch, vocab_size)

            # === 采样策略 ===
            if temperature == 0:
                # 贪心解码: 直接选概率最高的 token
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature

                # Top-K: 只保留概率最高的 K 个 token
                if top_k > 0:
                    topk_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values
                    next_logits = next_logits.masked_fill(next_logits < topk_vals[..., -1:], float("-inf"))

                # Top-P (Nucleus): 保留累积概率不超过 P 的 token
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    remove = mask.scatter(-1, sorted_idx, mask)
                    next_logits = next_logits.masked_fill(remove, float("-inf"))

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == eos_token_id:
                break
    finally:
        stats["blocks_used"] = len(kv_cache.block_table)  # 记录峰值，在 free 之前
        kv_cache.free()   # 归还物理块到共享池

    return generated, stats
