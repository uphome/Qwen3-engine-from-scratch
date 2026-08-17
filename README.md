# Qwen3-engine-from-scratch

从零构建 Qwen3 推理引擎。纯 PyTorch 手写，理解推理的每一行代码。后续用 Triton 逐算子替换，最终脱离 PyTorch。

## 快速开始

```bash
git clone https://github.com/uphomer/Qwen3-engine-from-scratch.git
cd Qwen3-engine-from-scratch

# 依赖: torch, safetensors, tokenizers
pip install torch safetensors tokenizers

# 下载 Qwen3-0.6B 并运行（交互式：输入 prompt 生成，/quit 退出）
python main.py --model /path/to/Qwen3-0.6B --temperature 0
# 注：decode 默认走 CUDA Graph（v3.0）；--no-graph 或 QWEN3_CUDA_GRAPH=0 关闭
```

## 项目结构

```
├── main.py               # CLI 入口
├── generate.py           # 自回归生成 + 采样（temperature/top-k/top-p）
├── chat_template.py      # Qwen3 Chat Format（<|im_start|>...）
├── bench.py              # 吞吐基准测试 + torch.profiler trace 导出
├── bench_batched.py      # 连续批处理基准（Scheduler 调度，decode 走 CUDA Graph）
├── profile_decode.py     # prefill/decode 步 kernel 时间剖析
└── qwen3/                # 核心模型
    ├── config.py         # Qwen3Config — 纯数据类
    ├── norm.py           # RMSNorm
    ├── rope.py           # RotaryEmbedding
    ├── kv_cache.py       # NaiveKVCache
    ├── PagedKVcache.py   # 分页 KV（共享池 + GPU 常驻 2D 块表 + row_id 行槽位）
    ├── attention.py      # Qwen3Attention（GQA + QK-Norm，三段式：flash/varlen/decode）
    ├── mlp.py            # Qwen3MLP（SwiGLU）
    ├── decoder.py        # Qwen3DecoderLayer（Pre-Norm）
    ├── model.py          # Qwen3Model + Qwen3ForCausalLM（含 forward_decode）
    ├── scheduler.py      # 连续批处理调度器（waiting/running/finished）
    ├── batch.py          # Batch（prefill 右 pad / decode 组装）
    ├── request.py        # Request 状态机
    ├── graph_runner.py   # CUDA Graph 图池（decode 整步捕获，2 的幂 bucket）
    ├── kernels/          # Triton kernels（paged attention / flash varlen）
    └── weights.py        # safetensors 权重加载
```

## Benchmark

```bash
# 环境前提：若 nvidia-smi 显示驱动 CUDA <= 11.6（如 510.54），Triton 自带
# ptxas 12.3 生成的 cubin 加载失败，必须先指定兼容的旧 ptxas（见 PERFORMANCE.md）：
export TRITON_PTXAS_PATH=/data/hjt1/anaconda3/envs/cuda_learn/bin/ptxas

# prefill 注意力默认走 vLLM 风格 varlen flash attention 融合 kernel（QWEN3_FLASH_ATTN=triton）
# 关闭融合走标准实现（消融对比）：export QWEN3_FLASH_ATTN=pytorch

# 基础吞吐测试
python bench.py --model /path/to/Qwen3-0.6B --num-seqs 64

# 带 profiler trace，定位 kernel 级瓶颈
python bench.py --model /path/to/Qwen3-0.6B --profile --profile-output trace.json

# prefill/decode 步 kernel 时间分布（--mode prefill 剖析融合 kernel）
python profile_decode.py --model /path/to/Qwen3-0.6B --mode prefill --prompt-len 256

# 连续批处理基准（batch 1..N 扫描；decode 默认走 CUDA Graph 图池）
python bench_batched.py --model /path/to/Qwen3-0.6B --num-seqs 64 --max-batch 28 --warmup
# 关闭 CUDA Graph（消融对比）：--no-graph 或 export QWEN3_CUDA_GRAPH=0
python bench_batched.py --model /path/to/Qwen3-0.6B --batch-size 8 --no-graph

# 单请求基准（bench.py 串行也走图，与 main.py 同路径）
python bench.py --model /path/to/Qwen3-0.6B --num-seqs 64 --no-graph
```

已有指标：吞吐（tok/s）、延迟分布（p50/p95/p99）、VRAM 占用、Prefill/Decode 占比、算子级 CUDA 耗时排名。

## 致谢

- [Qwen3](https://github.com/QwenLM/Qwen3) — Qwen3 模型架构与权重
- [mini-sglang](https://github.com/sgl-project/sglang) — bench.py 参考 mini-sglang 的 benchmark 设计

## License

MIT
