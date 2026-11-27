# Qwen3-engine-from-scratch

从零构建 Qwen3 推理引擎。纯 PyTorch 手写，理解推理的每一行代码。后续用 Triton 逐算子替换，最终脱离 PyTorch。

## 快速开始

环境要求：Python >= 3.10、NVIDIA GPU（CUDA 11.x/12.x，支持 bf16）。

```bash
git clone https://github.com/uphome/Qwen3-engine-from-scratch.git
cd Qwen3-engine-from-scratch

# 1. 安装依赖（torch 走 cu118 wheel 源）
pip install -r requirements.txt

# 2. 下载 Qwen3-0.6B 权重（约 1.2GB，默认到 models/Qwen3-0.6B）
bash scripts/download_model.sh

# 3. 运行（交互式：输入 prompt 生成，/quit 退出）
python main.py --model models/Qwen3-0.6B --temperature 0
```

> decode 默认走 CUDA Graph（v3.0）；`--no-graph` 或 `QWEN3_CUDA_GRAPH=0` 关闭。
> 若 `nvidia-smi` 显示驱动 CUDA <= 11.6（如 510.54），需先设置 Triton ptxas（见下方 Benchmark 说明）。

## HTTP API 服务（OpenAI 兼容，零新依赖）

> 完整参数、响应格式、调用示例与错误码见 [API.md](API.md)。

```bash
# 启动服务（后台线程跑 Scheduler 连续批处理，纯标准库 http.server）
python server.py --model /path/to/Qwen3-0.6B --host 127.0.0.1 --port 8000

# 查询模型
curl http://127.0.0.1:8000/v1/models

# Prometheus 指标（请求数 / tokens / KV 占用 / TTFT summary）
curl http://127.0.0.1:8000/metrics

# Chat 补全（OpenAI chat.completion 格式，逐请求独立 temperature/top_k/top_p）
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":128,"temperature":0.7}'
# 思考链默认关闭（空思考块，直接回答）；要开启请加 "enable_thinking": true
# 函数调用：请求带 "tools" 即启用（OpenAI 标准格式，两轮 tool_calls 模式）

# 流式（SSE）：stream=true 逐 token 推送
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":true}'

# 纯 prompt 补全
curl -X POST http://127.0.0.1:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"prompt":"1+1=","max_tokens":64}'
```

特性与边界（学习版取舍）：

- `Engine` 后台单线程调度循环（复用 bench_batched 骨架），`queue.Queue` 收作业，
  Scheduler 连续批处理（batch 满 4 时新请求仍能入队，decode 步共享）
- 流式：`stream=true` 返回 SSE（`text/event-stream`），逐 token 推送；
  客户端断开自动取消并归还 KV
- 可观测：`GET /metrics`（Prometheus 文本格式，请求/tokens/KV/TTFT）
- 前缀缓存：KV 跨请求复用（完全命中 + 部分命中；完全命中 TTFT 8.2x，
  2k 历史部分命中 1.3-1.6x；链感知 LRU；设计见 [PREFIX_CACHE.md](PREFIX_CACHE.md)）
- chunked prefill：长 prompt 切片与 decode 交替（`QWEN3_CHUNK_SIZE`，默认 512），
  短请求 TTFT 改善 2.7x
- 训推同机监控：`qwen3.monitoring.append_snapshot()` 每步写 JSONL
  （同进程直接读引擎），或 `GET /metrics?format=json`
- KV 池按剩余显存自动裁剪（`--kv-ratio`，默认 0.6 留 40% 给 prefill 瞬时激活；
  OOM 就调低，如 `--kv-ratio 0.5`）；推理中途 OOM 会返回 503 且服务不崩
- 池满时新请求返回 503（不排队）
- 参数：`max_tokens`（默认 512）、`temperature`/`top_k`/`top_p` 每请求独立
- 4GB 卡实测参考：单请求 prompt ≤ 1500 tokens 稳定；并发建议 ≤4 个短 prompt
  （显存由 `batch-size`×`kv-ratio` 共同决定，可自行调节）

## 项目结构

```
├── main.py               # CLI 入口
├── generate.py           # 自回归生成 + 采样（temperature/top-k/top-p）
├── chat_template.py      # Qwen3 Chat Format（<|im_start|>...）
├── server.py             # OpenAI 兼容 HTTP API（连续批处理后台引擎）
├── bench.py              # 吞吐基准测试 + torch.profiler trace 导出
├── bench_batched.py      # 连续批处理基准（Scheduler 调度，decode 走 CUDA Graph）
├── bench_sampling.py     # GPU 采样微基准
├── bench_server.py       # HTTP 压测（并发梯度 / TTFT / 长稳）
├── bench_server_sampling.py  # HTTP 服务端采样压测（旧）
├── profile_decode.py     # prefill/decode 步 kernel 时间剖析
├── scripts/              # 工具脚本（模型下载 / 压测报告生成）
├── BENCHMARK.md          # 压测报告（自动生成）
├── benchmark/            # 压测曲线（PNG）
├── results/              # 压测原始 JSON
├── tests/                # 正式测试（pytest，CPU/GPU 分层）
└── qwen3/                # 核心模型
    ├── config.py         # Qwen3Config — 纯数据类
    ├── norm.py           # RMSNorm
    ├── rope.py           # RotaryEmbedding（v3.1 预计算表）
    ├── PagedKVcache.py   # 分页 KV（共享池 + GPU 常驻 2D 块表 + row_id 行槽位）
    ├── attention.py      # Qwen3Attention（GQA + QK-Norm + fused QKV）
    ├── mlp.py            # Qwen3MLP（SwiGLU + fused gate/up）
    ├── decoder.py        # Qwen3DecoderLayer（Pre-Norm）
    ├── model.py          # Qwen3Model + Qwen3ForCausalLM（含 forward_decode）
    ├── scheduler.py      # 连续批处理调度器（waiting/running/finished）
    ├── batch.py          # Batch（prefill 右 pad / decode 组装）
    ├── request.py        # Request 状态机
    ├── sampling.py       # GPU 批量采样（batch top-k + Triton Gumbel-max）
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

# 引擎压测：并发梯度 + TTFT/TPOT + JSON（报告用）
python bench_batched.py --model /path/to/Qwen3-0.6B --num-seqs 32 \
    --batch-sizes 1,2,4,8,16,28 --warmup --json-out results/engine.json

# HTTP 压测：并发梯度 / 流式 TTFT / 错误率（先启动 server.py）
python bench_server.py --url http://127.0.0.1:8000 --num-requests 32 \
    --concurrency 1,2,4,8 --max-tokens 64 --stream --json-out results/server.json

# 长稳测试（每 30s 窗口统计）
python bench_server.py --url http://127.0.0.1:8000 --duration 1800 \
    --concurrency 8 --max-tokens 64 --stream --json-out results/stability.json

# 生成压测报告（Markdown + 曲线图）
python scripts/make_bench_report.py --engine-json results/engine.json \
    --server-json results/server.json --stability-json results/stability.json \
    --out BENCHMARK.md --figdir benchmark
```

已有指标：吞吐（tok/s）、延迟分布（p50/p95/p99）、VRAM 占用、Prefill/Decode 占比、算子级 CUDA 耗时排名。

## 文档

- [docs/](docs/) — **学习文档（5 篇，结合代码逐段讲解）**：流式 / 前缀缓存 / 调度 / 测试压测 / 训推同机监控
- [API.md](API.md) — OpenAI 兼容 API 与 `/metrics`
- [PREFIX_CACHE.md](PREFIX_CACHE.md) — 哈希 KV 复用（完全/部分命中）
- [TESTING.md](TESTING.md) — 测试矩阵与压测复现
- [BENCHMARK.md](BENCHMARK.md) — 压测报告
- [ENGINE_COMPARISON.md](ENGINE_COMPARISON.md) — 与工业级引擎的差距与路线图
- [PERFORMANCE.md](PERFORMANCE.md) — 版本性能演进

## 致谢

- [Qwen3](https://github.com/QwenLM/Qwen3) — Qwen3 模型架构与权重
- [mini-sglang](https://github.com/sgl-project/sglang) — bench.py 参考 mini-sglang 的 benchmark 设计

## License

MIT
