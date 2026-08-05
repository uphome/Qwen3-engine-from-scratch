"""
批量生成驱动 — continuous batching 主循环

调度循环（每步一次 forward）:
    while scheduler.has_pending():
        batch = scheduler.schedule()        # ① 调度：挑本步请求
        input_ids = batch.build_input_ids() # ② 组装：(B, S)
        logits = model(input_ids, kv_cache=batch.build_kv_caches(),
                       input_lens=batch.input_lens)
        next_tokens = sample(logits)        # ③ 采样：每请求一个 token
        scheduler.on_step_done(batch, next_tokens)  # ④ 收尾：回写/出批/归还

Phase 3 现状:
  - prefill 批: 真批量 (B, S_max)，右 pad + 逐请求 position/mask
  - decode 批: 真批量 (B, 1)，一次 forward 推 B 个请求（CPU 上走
    PyTorch 逐页兜底；GPU + bf16 走 Triton kernel）

用法:
    python batched_generate.py --model /path/to/Qwen3-0.6B --num-seqs 4 --batch-size 2
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool, Scheduler
from qwen3 import load_weights_from_safetensors
from qwen3.request import Request


def make_requests(num_seqs: int, max_len: int, vocab_size: int,
                  device: torch.device, eos_token_id: int = -1) -> list[Request]:
    """构造离线请求集（随机 prompt，与 bench.py 同风格）"""
    requests = []
    for i in range(num_seqs):
        input_len = 32 + (i * 13) % max_len   # 变长输入
        input_ids = torch.randint(0, vocab_size, (1, input_len), device=device)
        requests.append(Request(
            request_id=i,
            input_ids=input_ids,
            max_new_tokens=64,
            eos_token_id=eos_token_id,
        ))
    return requests


@torch.no_grad()
def sample(logits: torch.Tensor, temperature: float = 0.0,
           top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    """从 (B, vocab) logits 采样，每行一个 token → (B, 1)"""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        vals = torch.topk(logits, min(top_k, logits.size(-1))).values
        logits = logits.masked_fill(logits < vals[..., -1:], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        remove = mask.scatter(-1, sorted_idx, mask)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser(description="Continuous batching 驱动（Phase 2）")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--pool-blocks", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = Qwen3Config.from_pretrained(args.model)
    model = Qwen3ForCausalLM(config)
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    kv_pool = KVCachePool(
        num_blocks=args.pool_blocks,
        num_layers=config.num_hidden_layers,
        block_size=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=device,
        dtype=dtype,
    )
    scheduler = Scheduler(kv_pool, batch_size=args.batch_size)

    for req in make_requests(args.num_seqs, 128, config.vocab_size, device,
                             eos_token_id=-1):
        scheduler.add(req)

    print(f"Scheduler: {scheduler}")
    t0 = time.perf_counter()
    n_steps = 0

    # ---- 调度主循环 ----
    while scheduler.has_pending():
        batch = scheduler.schedule()
        if batch is None:
            break

        input_ids = batch.build_input_ids()
        kv_caches = batch.build_kv_caches()
        logits = model(input_ids, kv_cache=kv_caches,
                       input_lens=batch.input_lens)   # 一次 forward 吃整批
        # 每请求取自己的最后一个位置:
        #   decode 批 S=1 → 位置 0；prefill 批右 pad → 各请求 input_len-1
        last_idx = torch.tensor(batch.input_lens, dtype=torch.long,
                                device=logits.device) - 1
        next_tokens = sample(logits[torch.arange(last_idx.shape[0]), last_idx])

        scheduler.on_step_done(batch, next_tokens)
        n_steps += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    # ---- 汇总 ----
    print(f"Scheduler final: {scheduler}")
    print(f"Steps: {n_steps}, Total: {t_total:.2f}s")
    for req in scheduler.finished:
        print(f"  {req}")


if __name__ == "__main__":
    main()
