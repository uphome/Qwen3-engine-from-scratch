# 测试说明（TESTING）

## 分层与标记

| 层 | 目录 | 标记 | 环境 |
|---|---|---|---|
| 单元 / 逻辑 | `tests/unit/` | 无 | CPU，秒级，CI 必跑 |
| Kernel 数值 | `tests/kernels/` | `gpu` | NVIDIA GPU + Triton |
| 执行路径一致性 | `tests/model/` | `gpu` + `slow` | GPU（CUDA Graph 捕获） |
| 端到端 / 压测 | 待建（W2+） | `gpu` / `slow` | GPU |

无 GPU 的环境会自动跳过 `gpu` 用例（conftest 钩子），CPU 子集可直接进 CI。

## 运行

```bash
# 全量（推荐，GPU 机器）
pytest -q

# 仅 CPU（CI 同款）
pytest -m "not gpu" -q

# 仅 GPU kernel 回归
pytest -m "gpu and not slow" -q

# 只跑某个文件
pytest tests/unit/test_scheduler.py -v
```

**重要环境变量**（驱动只支持 CUDA <= 11.6，如 510.54 时必须设置，否则 Triton
cubin 加载失败、GPU 用例报 "cpu tensor" 类假错误）：

```bash
export TRITON_PTXAS_PATH=/path/to/ptxas   # 如 conda 环境里的 ptxas
```

## 现有用例矩阵（W1）

| 文件 | 覆盖点 | 用例数 |
|---|---|---|
| `tests/unit/test_config.py` | config.json 读取、head_dim 推导、默认值 | 4 |
| `tests/unit/test_rope.py` | 预计算表数值、查表、rotate_half、位置 0 恒等、范数保持、相对位置性质 | 7 |
| `tests/unit/test_kv_pool.py` | 块分配/回收、复用清零、行槽位、跨页预留、seq_lens 同步、prefill 写入读回、池闭环、超长报错 | 9 |
| `tests/unit/test_request.py` | 状态机合法/非法迁移、结束判定、输出拼接、finish 归还块 | 9 |
| `tests/unit/test_batch.py` | prefill 右 pad、decode 组装、input_lens、kv_caches | 4 |
| `tests/unit/test_scheduler.py` | 组批、prefill 补位不超编、FIFO、完成回收、池闭环 | 7 |
| `tests/unit/test_chat_template.py` | Qwen3 特殊 token、thinking 开关、多轮 | 4 |
| `tests/unit/test_metrics.py` | Engine 计数/TTFT、容量拒绝计数、Prometheus 格式可解析 | 4 |
| `tests/unit/test_monitor.py` | 训推同机快照结构、JSONL 追加、与 Prometheus 口径一致 | 5 |
| `tests/unit/test_cache_eviction.py` | 容量不足时强制淘汰缓存块（开关/ref 保护/指标） | 6 |
| `tests/unit/test_stream_engine.py` | 流式 token 推送顺序、取消清理/归还、流式==非流式 | 3 |
| `tests/unit/test_prefix_cache.py` | 块哈希链、完全/部分命中、COW、后缀组装、引用计数、LRU、父链一致性、池闭环 | 11 |
| `tests/model/test_prefix_prefill.py` | 部分命中 vs 全量 prefill logits（CPU fp32） | 4 |
| `tests/kernels/test_flash_prefill.py` | flash 普通 varlen / 历史路径（q/kv 长度分离） | 2 |
| `tests/unit/test_chunked_prefill.py` | 长 prompt 切片、chunk 与 decode 交替、状态机、池闭环 | 4 |
| `tests/model/test_chunked_prefill_equiv.py` | 分块 prefill logits == 整段（CPU fp32） | 2 |
| `tests/kernels/test_update_kv_stride.py` | **v3.2 fused-qkv stride bug 回归**：连续/非连续切片、跨页写入 | 3 |
| `tests/model/test_graph_equivalence.py` | CUDA Graph 图池 vs eager 逐 step logits 一致（bucket 1/2、空槽、池闭环） | 2 |
| `tests/e2e/test_server_stream.py` | 真实 server SSE：流式==非流式、chunk 格式/usage、断开取消 | 3 |

## 约定

- **新功能必须带测试**：至少 1 个正向用例 + 1 个边界/回归用例。
- **GPU 用例写 `gpu` marker**：CI 无 GPU，靠它跳过；不要靠 `if cuda` 分支静默跳过。
- **真实数值对比优先**：kernel 改动对比 PyTorch 参考实现；执行路径改动对比 eager。
- **测试不依赖网络**：不下载模型；需要权重时用微型随机模型（`tiny_model` fixture）。

## 压测工具

| 工具 | 用途 | 输出 |
|---|---|---|
| `bench_batched.py` | 引擎级：并发梯度、TTFT/TPOT、p50/p95 | 终端表格 + `--json-out` |
| `bench_server.py` | HTTP：并发梯度 / 长稳、QPS、TTFT（流式）、错误率 | 终端表格 + `--json-out` |
| `scripts/make_bench_report.py` | 从 JSON 生成 `BENCHMARK.md` + 曲线图 | Markdown + PNG |

复现（当前 `BENCHMARK.md` 数据）：

```bash
# 1) 引擎
python bench_batched.py --model /data/hjt1/Qwen3-0.6B --num-seqs 32 \
    --min-input-len 32 --max-input-len 256 --min-output-len 32 --max-output-len 128 \
    --batch-sizes 1,2,4,8,16,28 --warmup --json-out results/engine.json

# 2) HTTP（另开终端启动 server.py）
python server.py --model /data/hjt1/Qwen3-0.6B --port 8767 --batch-size 8
python bench_server.py --url http://127.0.0.1:8767 --num-requests 32 \
    --concurrency 1,2,4,8 --max-tokens 64 --stream --json-out results/server.json

# 3) 长稳（120s 示例；正式可 1800s）
python bench_server.py --url http://127.0.0.1:8767 --duration 120 \
    --concurrency 8 --max-tokens 64 --stream --json-out results/stability.json

# 4) 报告
python scripts/make_bench_report.py --engine-json results/engine.json \
    --server-json results/server.json --stability-json results/stability.json \
    --out BENCHMARK.md --figdir benchmark
```

## 后续（按路线图）

- prefix caching（哈希 KV）：命中率 / 共享块生命周期 / 跨对话复用收益
- 压测：引擎 / HTTP 并发扫描、TTFT/TPOT、长稳压测
- 更多 e2e：tool calling、错误码边界（400/503）、`/v1/completions` 流式
