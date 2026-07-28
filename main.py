"""
Lesson 2: 从零手写 Qwen3 模型 — 端到端推理

仅依赖 PyTorch，从零实现 Qwen3 的完整推理流程：
  1. 模型配置 (Qwen3Config)
  2. 基础组件 (RMSNorm, RoPE, SwiGLU MLP)
  3. 注意力机制 (GQA + QK-Norm)
  4. Transformer Decoder
  5. 权重加载 (safetensors 直读)
  6. 自回归生成

用法:
    python main.py --model Qwen/Qwen3-0.6B
    python main.py --model Qwen/Qwen3-8B --prompt "Tell me about AI" --temperature 0
"""

import os
import time
import argparse

import torch

from qwen3 import Qwen3Config, Qwen3ForCausalLM
from qwen3 import load_weights_from_safetensors
from generate import generate
from chat_template import format_chat


def main():
    parser = argparse.ArgumentParser(description="Lesson 2: 从零手写 Qwen3 推理")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace 模型名或本地路径, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="写一篇300字的文章。")
    parser.add_argument("--max-tokens", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备, e.g. 'cuda:1' (默认: 自动选择最空闲的 GPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("Lesson 2: 从零手写 Qwen3 — 端到端推理")
    print("=" * 60)

    # --- 选择设备 ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        best = free.index(max(free))
        device = torch.device(f"cuda:{best}")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}, dtype: {dtype}")

    # --- 读取配置 ---
    print(f"\n[1/4] 读取模型配置...")
    config = Qwen3Config.from_pretrained(args.model)
    print(f"  Model: {args.model}")
    print(f"  Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}, "
          f"Heads: {config.num_attention_heads}, KV Heads: {config.num_key_value_heads}")

    # --- 创建模型 ---
    print(f"\n[2/4] 创建模型结构...")
    model = Qwen3ForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,} ({param_count / 1e9:.2f}B)")

    # --- 加载权重 ---
    print(f"\n[3/4] 从 safetensors 加载权重...")
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    # --- 生成 ---
    print(f"\n[4/4] 生成文本...")
    # 不依赖 transformers：用 tokenizers 库直读 tokenizer.json
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    eos_token_id = 151645  # Qwen3 的 EOS token ID

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond directly without thinking."},
        {"role": "user", "content": args.prompt},
    ]
    text = format_chat(messages)
    input_ids = torch.tensor([tokenizer.encode(text).ids], device=device)

    sampling = "greedy" if args.temperature == 0 else f"T={args.temperature}"
    print(f"  Prompt: {args.prompt}")
    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Sampling: {sampling}, max_tokens: {args.max_tokens}")

    t0 = time.perf_counter()
    output_ids, stats = generate(
        model, input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )
    t_total = time.perf_counter() - t0

    # --- 输出 ---
    new_tokens = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens.tolist())

    print(f"\n{'=' * 60}")
    print("Generated:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    # --- 统计 ---
    n = len(new_tokens)
    step_times = stats["step_times"]
    print(f"\n统计:")
    print(f"  生成 tokens: {n}")
    print(f"  总耗时: {t_total:.2f}s")
    if n > 0:
        print(f"  平均每步: {sum(step_times) / len(step_times) * 1000:.1f} ms")
        print(f"  首步 (prefill): {step_times[0] * 1000:.1f} ms")
        if len(step_times) > 1:
            print(f"  末步: {step_times[-1] * 1000:.1f} ms")

    print(f"\nKV cache 已启用:")
    print(f"  Prefill: 处理 {stats['input_len']} tokens（存入 cache）")
    print(f"  Decode:  每次只处理 1 token（其余从 cache 读）")
    print(f"  每步时间应保持稳定，不随序列长度增长")


if __name__ == "__main__":
    main()
