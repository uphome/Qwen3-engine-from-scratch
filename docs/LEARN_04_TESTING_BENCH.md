# 学习篇 04：测试体系与压测

> 对应代码：`tests/`、`pyproject.toml`、`.github/workflows/ci.yml`、
> `bench_batched.py`、`bench_server.py`、`scripts/make_bench_report.py`
> 关联文档：[TESTING.md](../TESTING.md)、[BENCHMARK.md](../BENCHMARK.md)

---

## 0. 这篇讲什么

读完你应该能回答：

1. 一个推理引擎应该测哪几层？每层测什么、不测什么？
2. 没有 GPU 的 CI 怎么跑？GPU 测试怎么组织？
3. 如何证明"优化前后数值等价"？
4. TTFT / TPOT / 吞吐 / 端到端延迟，各自的口径是什么？什么时候会"看起来怪"？
5. 压测脚本怎么写才能不被缓存/排队等效应污染？
6. 长稳测试怎么统计窗口、怎么看衰减？

---

## 1. 测试金字塔（本项目的四层）

```
            e2e（真实 server 子进程，慢，少量）
          ─────────────────────────────
        model（执行路径一致性 / 数值等价，GPU）
      ───────────────────────────────────
    kernels（Triton vs PyTorch 参考，GPU）
  ─────────────────────────────────────
unit（配置/状态机/池/调度/模板，CPU，快）
```

| 层 | 目录 | 标记 | 运行环境 | 回答什么问题 |
|---|---|---|---|---|
| unit | `tests/unit/` | 无 | CPU，秒级 | 逻辑/状态机/不变量对不对 |
| kernel | `tests/kernels/` | `gpu` | GPU | 手写 kernel 与参考实现数值是否一致 |
| model | `tests/model/` | `gpu`/`slow` | GPU（或 CPU tiny） | 图/eager、缓存/全量、分块/整段是否等价 |
| e2e | `tests/e2e/` | `gpu`+`slow` | 真实模型 + 子进程 | HTTP 契约（流式/错误码/断开）是否成立 |

**关键取舍**：

- 上层测试（e2e）只测"契约"，不重复测数值（数值在 kernel/model 层）
- 下层测试（unit）不依赖模型权重（用微型随机模型/纯 KV 操作），保证 CI 可跑
- GPU 测试不进 CI，但**本地必须可一键跑**（`pytest -m gpu`）

---

## 2. conftest 的三个基础设施

`tests/conftest.py`：

### 2.1 GPU 自动跳过

```python
def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="需要 CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
```

- CI（无 GPU）不会因为缺 GPU 而失败
- 本地（有 GPU）自动全跑
- 不要写 `if not cuda: return`（会"静默通过"）；用 marker + skip

### 2.2 自动选最空闲 GPU

```python
@pytest.fixture(scope="session")
def gpu_device() -> str:
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    best = free.index(max(free))
    torch.cuda.set_device(best)
    return f"cuda:{best}"
```

共享机器上 GPU 0 常被占满；测试自动落到空闲卡。`set_device` 同时修正
Triton kernel 的当前设备（否则多卡会报 `cpu tensor?`，见学习篇 01 的教训）。

### 2.3 微型模型与池工厂

```python
@pytest.fixture
def tiny_config():
    return Qwen3Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=16, num_hidden_layers=2, vocab_size=512, ...)

@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return Qwen3ForCausalLM(tiny_config).eval()

@pytest.fixture
def kv_pool_factory(tiny_config):
    def _make(device="cpu", num_blocks=32, max_seq_len=128, block_size=16):
        ...
    return _make
```

好处：单测 <1s、可复现（固定 seed）、不依赖下载模型。

---

## 3. 每层怎么写（各举一例）

### 3.1 unit：状态机/池不变量

```python
def test_pool_closed_after_chunked_prefill(kv_pool_factory):
    pool = kv_pool_factory(num_blocks=128, max_seq_len=2048)
    sched = Scheduler(pool, batch_size=1, chunk_size=128)
    req = make_req(0, 300, max_new=2)
    sched.add(req)
    while req.state != DECODING:
        simulate_prefill(sched, sched.schedule(), pool)
    while sched.has_pending():
        batch = sched.schedule()
        for r in batch.requests:
            r.kv_cache.reserve_next()
            r.kv_cache.advance_seq_len(1)
        sched.on_step_done(batch, torch.zeros(batch.size, 1, dtype=torch.long))
    assert req.state == "finished"
    assert pool.free_count + pool.cached_count == pool.total_blocks
```

**模式**：不跑模型，手动模拟 `update/advance`，只验证调度与池状态。

### 3.2 kernel：与参考实现比

```python
def test_flash_prefill_with_history_matches_reference():
    ...
    out = flash_attention_prefill_with_history(q, k_new, v_new, lens, hist_kv, ...)
    for i, li in enumerate(lens):
        ref = reference_attention(...)        # PyTorch 朴素实现
        got = out[i:i + 1, :li]
        assert torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)
```

**模式**：手写参考实现（清晰的 causal + GQA），同输入比数值（bf16 容差 3e-2）。

### 3.3 model：数值等价

```python
def test_partial_hit_logits_match_full_prefill(tiny_model, kv_pool_factory):
    logits_full = tiny_model(ids, kv_cache=[new_cache], input_lens=[39], mode="prefill")
    # 先 prefill 前 32（模拟历史），再 prefill 后 7（读历史）
    tiny_model(ids[:, :32], kv_cache=[cache], input_lens=[32], mode="prefill")
    logits_suffix = tiny_model(ids[:, 32:], kv_cache=[cache], input_lens=[7], mode="prefill")
    diff = (logits_full[:, 32:, :] - logits_suffix).abs().max().item()
    assert diff < 1e-4
```

**模式**：CPU fp32 上做到精确等价（<1e-4），GPU bf16 上另有容差测试。
这类测试是"优化不改变语义"的证据。

### 3.4 e2e：真实服务契约

三个用例（学习篇 01 第 6 节）：
- 流式拼接 == 非流式
- chunk 格式 / usage / `[DONE]`
- 客户端断开 → 服务端清理

**模式**：`subprocess` 起真实 server，用标准库 HTTP 客户端，验证外部可观测行为。

---

## 4. CI：为什么只跑 CPU

`.github/workflows/ci.yml`：

```yaml
- name: Install dependencies (CPU)
  run: |
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
    pip install triton==2.2.0 safetensors tokenizers pytest ruff
- name: Lint (ruff)
  run: ruff check .
- name: Unit tests (CPU; GPU 用例自动跳过)
  run: pytest -m "not gpu" -q
```

- GitHub 免费 runner 没有 GPU
- CPU 子集覆盖：配置/状态机/池/调度/模板/采样回退
- GPU 测试在本地/服务器跑（文档写明命令）

**为什么不把 CUDA 测试放进 CI**：不是“偷懒”，而是成本/收益——GPU runner 贵；
关键数值测试用 CPU tiny model + GPU 本地执行已足够。

---

## 5. 引擎压测（`bench_batched.py`）

### 5.1 它测什么

- **离线批到达**：所有请求在 t=0 同时到达
- 扫描 batch_size → 吞吐/延迟曲线
- CUDA Graph 开/关消融
- 每次 run 从同一组 specs 开始（可复现）

### 5.2 指标口径（重要）

```python
# TTFT：prefill 批完成时刻 - t0（所有请求同时到达）
now = time.perf_counter()
ttfts.extend([now - t0] * batch.size)
```

- **TTFT 含排队**：batch=1、32 个请求时，第 32 个请求的首 token 要等前面
  31 个逐个 prefill → TTFT p50 可能几秒。这是"离线同批到达"的真实体验，
  报告里必须注明
- **TPOT**：`decode_ms / (n_tokens - batch_size)`（去掉每请求首 token）
- **decode ms/tok**：`decode_ms / n_tokens`（含首 token，偏大一点）

### 5.3 每次 run 清空缓存（防污染）

```python
def run_batched(...):
    # 基准扫描用同一组 prompt：若不清空前缀缓存，第二轮起会全部命中，
    # 吞吐被"缓存复用"污染（测不到纯调度/图性能）
    kv_pool.clear_prefix_cache()
    scheduler = Scheduler(kv_pool, batch_size=batch_size)
```

这是真实踩坑：不清缓存时 batch=2/4 的吞吐虚高（prompt 完全命中）。

### 5.4 JSON 输出（供报告消费）

```python
payload = {"meta": {...}, "results": results}
json.dump(payload, open(args.json_out, "w"), indent=2)
```

`results` 里保留 `latencies` / `ttfts` 原始列表，报告脚本自己算分位数。

### 5.5 池容量必须覆盖最大 batch

```python
max_batch = max(batch_sizes)     # 不是 args.max_batch！
```

这是另一个真实 bug：`--batch-sizes 1,2,4,8,16,28` 时若仍按默认
`max_batch=8` 算池，batch=28 会 OOM。

---

## 6. HTTP 压测（`bench_server.py`）

### 6.1 并发模型

```python
with ThreadPoolExecutor(max_workers=concurrency) as ex:
    futs = [ex.submit(one_request, url, prompt, ...) for _ in range(num_requests)]
    for f in as_completed(futs):
        results.append(f.result())
```

- 每个 worker 一个请求；服务端并发数 ≈ 客户端并发数
- 每个档位独立计时（墙钟 / OK 请求数 = QPS）
- 错误（HTTP 4xx/5xx/超时）计入错误率，不中断压测

### 6.2 流式 TTFT 怎么测

```python
for raw in r:
    line = raw.decode().strip()
    if not line.startswith("data: "): continue
    payload = line[6:]
    if payload == "[DONE]": break
    obj = json.loads(payload)
    if obj.get("usage"): usage = obj["usage"]
    choices = obj.get("choices") or []
    if choices:
        delta = choices[0].get("text") or ""
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - t0    # 首帧时刻
            text += delta
```

- 非流式只能测"端到端延迟"；**流式才有真实首包时刻（TTFT）**
- `stream_options.include_usage` 让最后一个 chunk 带精确 token 数

### 6.3 长稳模式的坑

第一版：

```python
def worker():
    local = []
    while time.perf_counter() < deadline:
        local.append(one_request(...))
    with lock:
        results.extend(local)      # 只在结束时上报
```

后果：主循环每 30s 采样时 `results` 还是空的（窗口统计失效）。
修复：**逐请求上报**：

```python
def worker():
    while time.perf_counter() < deadline:
        r = one_request(...)
        with lock:
            results.append(r)
```

窗口统计（`run_stability`）：

```python
windows.append({"t": round(elapsed, 1), "requests": len(snap),
                "ok": len(ok), "errors": err, "tokens": n_tok})
```

报告脚本把累计值差分 → 每窗口 tokens/requests，用来看"是否衰减"。

---

## 7. 报告生成（`scripts/make_bench_report.py`）

输入：`results/engine.json`、`results/server.json`、`results/stability.json`

输出：

- `BENCHMARK.md`：三张表（引擎/HTTP/长稳）+ 曲线
- `benchmark/*.png`：吞吐-并发、TTFT-并发、长稳窗口

```python
ap.add_argument("--engine-json")
ap.add_argument("--server-json")
ap.add_argument("--stability-json")
```

**为什么用 JSON + 脚本**：

- 数据与展示分离：重新画图/换模板不用重跑压测
- 可复现：面试官可以拿原始 JSON 自己算
- 历史对比：不同提交的 JSON 可以并排比较

---

## 8. 怎么解读压测结果

### 8.1 吞吐 vs 延迟不是一回事

| 指标 | 优化目标 |
|---|---|
| tok/s（吞吐） | 大 batch、连续批处理、图 |
| TTFT（首包） | 短排队、chunked prefill |
| TPOT（每 token） | decode kernel 效率 |
| 端到端 p95 | 公平调度、防长请求阻塞 |

### 8.2 注意口径

- 引擎压测（离线同批到达）：TTFT 含排队，batch 越小越夸张
- HTTP 压测（在线并发）：TTFT 是真实首包
- 对比不同系统前，先确认：请求长度分布、输出长度、并发模型、是否缓存命中

### 8.3 本项目的一组参考数字

| 场景 | 结果 |
|---|---|
| 引擎 batch=28 | 1592.8 tok/s，TPOT 0.51ms |
| HTTP 并发 8 | 668.5 tok/s，TTFT p50 88.1ms，0 错误 |
| 长稳 120s（并发 8） | 1206 请求，0 错误，窗口无衰减 |

---

## 9. 踩过的坑（真实）

1. **bench 缓存污染**：同一组 prompt 第二轮全部命中 → 清缓存
2. **`--batch-sizes` 未参与池容量** → OOM
3. **长稳窗口批量上报** → 窗口统计为空
4. **TTFT 口径未注明** → 读者误以为 batch=1 的首包要 7 秒
5. **测试文件同名**（`tests/unit/test_chunked_prefill.py` 与
   `tests/model/test_chunked_prefill.py`）→ pytest 导入冲突；
   重命名为 `_equiv`
6. **测试直接 `finish()` 而未从 running 移除** → 调度器状态不一致；
   用辅助函数 `finish_req` 模拟正常流程
7. **手动 prefill 模拟时没调 `start_prefill`** → 状态机 waiting→decoding 断言失败

---

## 10. 动手实验

1. **跑四层**：`pytest -m "not gpu" -q`、`pytest -m "gpu and not slow" -q`、
   `pytest -m "gpu and slow" -q`、`pytest -q`
2. **加一个测试**：给任意新功能写"正向 + 边界"两条用例，放在正确的层
3. **跑压测**：按 [TESTING.md](../TESTING.md) 的四条命令，生成自己的 `BENCHMARK.md`
4. **改坏一个优化**：把 `_prefill_turn` 去掉，重跑 chunked 测试，看哪个断言失败
5. **对比消融**：`--no-graph` vs 默认，观察吞吐差 3x（学习篇 03）

---

## 11. 自测问题

1. 为什么 kernel 测试要自己写参考实现，而不是相信 flash？容差怎么定？
2. CPU CI 能覆盖哪些 bug？哪些覆盖不了？
3. TTFT 在引擎压测里为什么含排队？HTTP 压测为什么不含？
4. 为什么 `bench_batched` 每轮清缓存？不清会怎样？
5. 长稳测试的窗口统计为什么要"逐请求上报"？
6. 一个"数值等价"测试通过，能说明什么？不能说明什么？

---

## 12. 代码地图

| 功能 | 位置 |
|---|---|
| fixtures / markers / GPU 选择 | `tests/conftest.py`、`pyproject.toml` |
| 单元测试 | `tests/unit/` |
| kernel 测试 | `tests/kernels/` |
| 数值等价测试 | `tests/model/` |
| e2e 测试 | `tests/e2e/` |
| CI | `.github/workflows/ci.yml` |
| 引擎压测 | `bench_batched.py` |
| HTTP 压测 / 长稳 | `bench_server.py` |
| 报告生成 | `scripts/make_bench_report.py` |
