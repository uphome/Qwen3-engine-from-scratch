# 更新记录（CHANGELOG）

本文件记录工程侧的重大变更（功能 / 测试 / 工程规范）。
模型版本与性能演进见 [PERFORMANCE.md](PERFORMANCE.md)，架构差距见 [ENGINE_COMPARISON.md](ENGINE_COMPARISON.md)。

---

## 2026-11-24 — 工程化：测试地基 + /metrics + SSE 流式

### 新增：前缀缓存（Prefix Cache，阶段 1：完全命中）

- 块哈希链（`qwen3/prefix_cache.py`）：`h_i = blake2b(h_{i-1}, tokens[块])`
- 块池状态机：引用计数 + 缓存块（ref=0 保留 K/V）+ LRU 淘汰
- 完全命中快速路径：块对齐且全部命中 → 跳过 prefill，直接 decode
- COW 最后一个块（重算 prompt 末 token 的 logits，不写共享块）
- 实测（0.6B / 256-token prompt）：TTFT **54.8ms → 6.7ms（8.24x）**，输出一致
- 指标：`engine_prefix_cache_*`（命中数 / 复用块 / 缓存占用）
- 文档：[PREFIX_CACHE.md](PREFIX_CACHE.md)；测试：`tests/unit/test_prefix_cache.py`（8）
- 阶段 2（部分命中：后缀 prefill + attention 读历史 KV）待做

### 代码审查修复（前缀缓存 / SSE 交互）

- `bench_batched.py` / `test/test_batched.py`：池闭环断言计入前缀缓存块
  （`free + cached == total - 图预留`）——修复缓存块导致的"池未归还"误报
- 流式请求被容量拒绝时补推 `token_queue` 哨兵：否则 HTTP handler 会挂到
  600s 超时
- TTFT summary 加锁：避免 `/metrics` 线程排序时与 Engine 线程 append 竞争
- `KVCachePool.free()` 防御性清理缓存索引（防外部误 free 缓存块后悬空）
- 前缀缓存 COW 分配失败时归还已匹配块（防御性回滚）

### 新增：pytest 测试体系（61 用例）

- `tests/unit/`（46）：config / rope / kv_pool / request / batch / scheduler /
  chat_template / metrics / stream_engine
- `tests/kernels/`（3, gpu）：`update_kv_batch` 连续 + fused-qkv 非连续切片
  回归（锁定 v3.2 stride bug）+ 跨页写入
- `tests/model/`（2, gpu+slow）：CUDA Graph 图池 vs eager 逐 step logits 一致
- `tests/e2e/`（3, gpu+slow）：真实 server SSE —— 流式 == 非流式、chunk 格式、
  客户端断开取消
- `conftest.py`：无 GPU 自动跳过、多卡自动选最空闲、微型模型 fixture
- `TESTING.md`：分层 / 标记 / 运行方式 / 用例矩阵 / 约定

### 新增：`/metrics`（Prometheus，零依赖）

- 请求/完成/失败、tokens、调度步、批大小、`running`/`waiting`
- KV 池占用率 / 空闲块 / 总块
- TTFT summary：滑动窗口 256 样本，p50/p95/p99 + count/sum
- 口径：启动 warmup（request_id=-1）不计入请求指标

### 新增：SSE 流式（`stream=true`，OpenAI 兼容）

- `POST /v1/chat/completions` / `POST /v1/completions` 支持 `stream`
- `text/event-stream` 逐 token 推送：首 chunk（role）→ content deltas →
  finish_reason →（可选 `stream_options.include_usage`）→ `data: [DONE]`
- 增量解码：累计 ids decode 后取后缀，多字节 UTF-8 不跨 token 拆坏
- 客户端断开 → `BrokenPipe` → 标记取消 → Engine 下一轮清理并归还 KV 块
- Engine：`Job.token_queue` 逐 token 推送（`None` 为结束哨兵）

### 修复：多卡环境 Triton 误报 "cpu tensor?"

- 现象：server 在 cuda:1 启动时 GraphRunner 捕获崩溃
  `Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`
- 原因：Triton launch 按**当前 CUDA device**（默认 0）校验指针，与 cuda:1
  上的张量不匹配
- 修复：`main.py` / `server.py` / `bench_batched.py` / `profile_decode.py`
  统一 `torch.cuda.set_device`；Engine 后台线程入口同样显式设置

### 工程规范

- `requirements.txt` / `requirements-dev.txt` / `pyproject.toml`（pytest markers、ruff）
- `LICENSE`（MIT）、`scripts/download_model.sh`（Qwen3-0.6B）
- `.gitignore`：`models/`、`traces/`、`results/raw/`、pytest/ruff 缓存
- GitHub Actions CI：CPU torch + `ruff check` + `pytest -m "not gpu"`
- README 快速开始改为 3 条命令（安装 → 下载模型 → 运行）
- 全仓 ruff 清理（20 处：f-string 前缀、未使用导入/变量）

---

## 2026-11-16 — v3.3 GPU 采样

- `qwen3/sampling.py`：batch top-k + Triton Gumbel-max 采样
  （batch=28 采样从 18.7ms/步 → 0.25ms/步）
- 每请求独立 temperature / top_k / top_p

## 2026-09 — v3.1 / v3.2

- v3.1：RoPE 预计算表 + 图内查表（decode 每步省一次 rotary_emb）
- v3.2：QKV/MLP 投影融合（qkv_proj / gate_up_proj，decode 每 token ~5%）
