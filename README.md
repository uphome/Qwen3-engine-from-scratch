# Qwen3-engine-from-scratch

从零构建 Qwen3 推理引擎。纯 PyTorch 手写，理解推理的每一行代码。后续用 Triton 逐算子替换，最终脱离 PyTorch。

## 快速开始

```bash
git clone https://github.com/uphomer/Qwen3-engine-from-scratch.git
cd Qwen3-engine-from-scratch

# 依赖: torch, safetensors, tokenizers
pip install torch safetensors tokenizers

# 下载 Qwen3-0.6B 并运行
python main.py --model /path/to/Qwen3-0.6B --prompt "你好" --temperature 0
```

## 项目结构

```
├── main.py               # CLI 入口
├── generate.py           # 自回归生成 + 采样（temperature/top-k/top-p）
├── chat_template.py      # Qwen3 Chat Format（<|im_start|>...）
├── bench.py              # 吞吐基准测试 + torch.profiler trace 导出
└── qwen3/                # 核心模型
    ├── config.py         # Qwen3Config — 纯数据类
    ├── norm.py           # RMSNorm
    ├── rope.py           # RotaryEmbedding
    ├── kv_cache.py       # NaiveKVCache
    ├── attention.py      # Qwen3Attention（GQA + QK-Norm）
    ├── mlp.py            # Qwen3MLP（SwiGLU）
    ├── decoder.py        # Qwen3DecoderLayer（Pre-Norm）
    ├── model.py          # Qwen3Model + Qwen3ForCausalLM
    └── weights.py        # safetensors 权重加载
```

## Benchmark

```bash
# 基础吞吐测试
python bench.py --model /path/to/Qwen3-0.6B --num-seqs 64

# 带 profiler trace，定位 kernel 级瓶颈
python bench.py --model /path/to/Qwen3-0.6B --profile --profile-output trace.json
```

已有指标：吞吐（tok/s）、延迟分布（p50/p95/p99）、VRAM 占用、Prefill/Decode 占比、算子级 CUDA 耗时排名。

## 致谢

- [Qwen3](https://github.com/QwenLM/Qwen3) — Qwen3 模型架构与权重
- [mini-sglang](https://github.com/sgl-project/sglang) — bench.py 参考 mini-sglang 的 benchmark 设计

## License

MIT
