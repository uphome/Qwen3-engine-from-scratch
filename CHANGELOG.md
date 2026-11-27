# 更新记录（CHANGELOG）

本文件记录工程侧的重大变更（功能 / 测试 / 工程规范）。
模型版本与性能演进见 [PERFORMANCE.md](PERFORMANCE.md)，架构差距见 [ENGINE_COMPARISON.md](ENGINE_COMPARISON.md)。

---

## 2026-11-27 — 容量不足时强制淘汰缓存（可用性优先）

- `KVCachePool.evict_cache_for(n)`：容量压力下按链感知淘汰 ref=0 缓存块
- `Scheduler._ensure_capacity`：容量不足时先淘汰再重试分配；
  `QWEN3_CACHE_EVICT_ON_DEMAND=0` 可退回"直接拒绝"（可预测性优先）
- 指标：`engine_prefix_cache_evictions_total` / JSON 快照 `prefix_cache.evictions`
- 测试：`tests/unit/test_cache_eviction.py`（6）
- 文档：PREFIX_CACHE / LEARN_02 / API / TESTING

---

## 2026-11-27 — 训推同机监控接口

- 新增 `qwen3/monitoring.py`：`snapshot` / `append_snapshot` / `write_snapshot`
  - 同进程训练（GRPO）每个 step 采集一次引擎状态，JSONL 追加落盘
  - 字段：requests / scheduler / kv / ttft / prefix_cache / gpu
  - 与 `/metrics` 文本同口径（TTFT nearest 分位数）
- `GET /metrics?format=json`：外部进程直接拿到同样结构的 JSON
- `integrations/grpo_example.py`：可运行的训练侧示例（fake engine 生成 JSONL）
- `tests/unit/test_monitor.py`（5）：结构、JSONL、分位口径、duck typing
- 文档：`docs/LEARN_05_TRAINING_MONITORING.md`；API/README/TESTING 同步

---

## 2026-11-25 — 学习文档（4 篇）

- `docs/LEARN_01_STREAMING.md`：SSE 协议、跨线程 token 队列、增量解码、断开取消
- `docs/LEARN_02_PREFIX_CACHE.md`：块哈希链、ref/链感知 LRU、完全/部分命中、COW
- `docs/LEARN_03_SCHEDULING.md`：状态机、组批、chunked prefill、容量记账
- `docs/LEARN_04_TESTING_BENCH.md`：四层测试、CI、TTFT/TPOT 口径、压测与长稳
- 每篇：概念 → 代码逐段 → 设计取舍 → 踩坑复盘 → 动手实验 → 自测问题
- `docs/README.md` 索引；README 增加文档入口

---

## 2026-11-25 — 功能补齐：流式增量解码 / 链感知 LRU / chunked prefill

### HTTP 流式增量解码

- 之前每 token 对**累计 ids** 全量 decode（O(n²)）；改为只 decode 未输出 token
- BPE 跨 token 的 UTF-8 字节：decode 出现 U+FFFD 时暂存，待补齐再输出
  （e2e 断言无替换字符）；8 token 上限防异常卡死

### 前缀缓存：链感知 LRU

- 淘汰时优先选**叶节点**（无子块），避免淘汰链头后整条前缀无法命中
- 维护 `hash -> 子块集合 / 块 -> 父 hash`；测试：A→B→C 链 + 无关块、
  上限 3 时淘汰 C，保留 A/B（命中 2 块而非 0）

### chunked prefill（长 prompt 切片）

- `Scheduler(chunk_size=512)`（`QWEN3_CHUNK_SIZE` 可调，0 = 关闭）
- 长 prompt 分多个调度步 prefill，**prefill chunk 与 decode 交替**，
  长 prefill 不再让已运行请求停顿整段
- 中间 chunk 不采样（避免伪 token）；TTFT 只在最后一个 chunk 记录
- 复用阶段 2 的 paged prefill（chunk 读历史 KV）+ position offset
- 实测（2048 长 prompt + 短请求并发，禁用前缀缓存隔离）：
  · 短请求 TTFT **182.5 → 66.9ms（2.73x）**
  · 短请求完成时间 382 → 413ms（0.92x，decode 与 chunk 交替的权衡）
  · 长请求 TTFT 182 → 239ms（0.76x，chunk 间插入 decode）
  · 输出与整段 prefill 完全一致（测试覆盖）

---

## 2026-11-25 — 压测体系（引擎 / HTTP / 长稳 / 报告）

### 新增

- `bench_batched.py`：TTFT（含离线排队，注明口径）、TPOT、`--json-out`、
  `--batch-sizes`（按梯度而非全扫描）；每轮 run 前清空前缀缓存（防污染）；
  修复 `--batch-sizes` 未参与池容量计算导致的 OOM
- `bench_server.py`：HTTP 压测（零依赖）
  - 并发梯度、QPS、tok/s、流式精确 TTFT、端到端延迟、错误率
  - 长稳模式（`--duration` + 每 30s 窗口统计，逐请求实时上报）
- `scripts/make_bench_report.py`：JSON → `BENCHMARK.md` + 曲线图（matplotlib）
- `BENCHMARK.md`：首次完整报告（引擎/HTTP/长稳 + 5 张图）
  - 引擎 32 序列：batch=28 **1592.8 tok/s**、TTFT p50 167.8ms、TPOT 0.51ms
  - HTTP 并发 1→8：131.8 → **668.5 tok/s**，0 错误，TTFT 42.7 → 88.1ms
  - 长稳 120s（并发 8）：**1206 请求 0 错误**、514 tok/s、窗口吞吐无衰减

---

## 2026-11-24 — 工程化：测试地基 + /metrics + SSE 流式

### 新增：前缀缓存（Prefix Cache，阶段 1+2：完全 / 部分命中）

- 块哈希链（`qwen3/prefix_cache.py`）：`h_i = blake2b(h_{i-1}, tokens[块])`
- 块池状态机：引用计数 + 缓存块（ref=0 保留 K/V）+ LRU 淘汰
- 完全命中快速路径：块对齐且全部命中 → 跳过 prefill，直接 decode
- COW 最后一个块（重算 prompt 末 token 的 logits，不写共享块）
- 实测（0.6B / 256-token prompt）：TTFT **54.8ms → 6.7ms（8.24x）**，输出一致
- 指标：`engine_prefix_cache_*`（命中数 / 复用块 / 缓存占用）
- 文档：[PREFIX_CACHE.md](PREFIX_CACHE.md)；测试：`tests/unit/test_prefix_cache.py`（8）
#### 阶段 2：部分命中（多轮对话）

- 最长前缀匹配（`match_prefix_partial`）：复用前 N 个完整块
- `Request.prefill_offset` + `Batch` 只送未命中后缀（position_ids 带偏移）
- attention 历史路径：`paged_attention_prefill_from_history`（q=后缀，KV=历史+后缀，
  因果 mask 按全局位置）；`get_kv_all` 层前一次性 gather 历史（消除逐层小算子）
- 短历史回退：`QWEN3_PREFIX_PARTIAL_MIN`（默认 1024）以下走全量 prefill
  （实测交叉点 ~1k tokens，避免负收益）
- flash varlen kernel 支持 q/kv 长度分离（`cu_seqlens_k` + causal offset），
  历史路径 Triton 实现已就绪并通过数值测试（阶段 3 切换）
- 容量记账：命中块从新块需求扣除；完全命中 COW 私有块计新块；
  `Scheduler.add` 接收 `reserved_blocks` 防批内超卖

实测（0.6B，2048 前缀 + 16 后缀）：TTFT 59.8 → 40.1ms（1.49x）；
sweep 交叉点：hist 1024 = 1.04x，hist 2048 = 1.57x。
已知：bf16 下部分命中与全量 prefill 可能有个别贪心 token 翻转（数值差异）；
CPU fp32 精确测试 max diff < 1e-4。

### 阶段 2 代码审查修复

- **bench_batched 缓存污染**：同一组 prompt 在 batch 扫描中会命中上一轮缓存，
  吞吐数据失真；每轮 `run_batched` 前 `pool.clear_prefix_cache()`
- `clear_prefix_cache` 修复：清空时必须同时从 LRU 移除（否则块重复计数，
  池不变量 `free+cached+active == total` 被破坏）
- **server HTTP 层容量预检删除**：按全量 prefill 预检会误拒可命中的请求
  （503）；容量交给 Scheduler 精确裁决
- **历史 KV 预取回退为逐层读取**：一次性 gather 所有层会让显存翻倍
  （2k 历史 ≈ 235MB/请求），改为每层 fast gather（峰值 ~8MB）
- 旧 `test/test_batched.py` / `profile_decode.py` 显式传 `mode`：
  自动推断在 decode 输入带 `input_lens` 时会误判 prefill（mask=None 崩溃）
- 完全命中且复用块为 0（单块 prompt）时 metrics 仍计命中；
  warmup 不计入
- Scheduler 容量错误消息与检查口径统一（含 decode 空间）

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
