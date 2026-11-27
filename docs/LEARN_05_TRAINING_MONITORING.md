# 学习篇 05：训推同机监控（引擎快照接口）

> 对应代码：`qwen3/monitoring.py`、`server.py`（`/metrics?format=json`）、
> `integrations/grpo_example.py`、`tests/unit/test_monitor.py`
> 关联文档：[API.md](../API.md) 2.1、[CHANGELOG](../CHANGELOG.md)

---

## 0. 这篇讲什么

场景：训练框架（如 GRPO）与推理引擎**同进程**。训练侧想每个 step 采集一次
引擎运行状态，**追加写入 JSONL**，训练结束后离线绘图（pandas/wandb）。
不要求实时监控/告警。

读完你应该能回答：

1. 为什么同进程时可以直接读 `engine`，而不是走 HTTP？
2. 快照应该包含哪些字段？每类字段能解释什么现象？
3. 为什么不建议在接口里做差分/绘图？
4. JSONL 每 step 一行的好处？和 Prometheus 文本的关系？
5. 同进程训推共享 GPU 时，哪些字段会被训练干扰、怎么解释？

---

## 1. 整体结构

```
训练循环（GRPO）                     推理引擎（同进程）
  for step in range(N):
    rollouts = engine...  ─────────►  Scheduler 调度 / prefill / decode
    append_snapshot(jsonl, engine,   │
                    step=..., ...)   │ metrics: 计数/步数/TTFT 窗口
        │                            │ scheduler: running/waiting
        │ 读 metrics/scheduler/pool   │ pool: free/cached/total
        ▼                            ▼
  logs/engine.jsonl  ── 训练结束 ──►  pandas 绘图
```

接口只有一个核心函数：

```python
from qwen3.monitoring import snapshot, append_snapshot, write_snapshot

snap = snapshot(engine, model_id="qwen3-0.6b")      # 纯 dict
append_snapshot("logs/engine.jsonl", engine, step=0, reward=0.5)   # JSONL 追加
write_snapshot("logs/last.json", engine)             # 覆盖写单文件
```

设计原则：

- **只返回 dict**：差分、滑动平均、绘图全部由训练侧决定
- **不 import server.py**：引擎对象 duck-typing（`metrics/scheduler/pool/device`），
  测试里可以塞 fake engine
- **字段稳定**：新增字段只加不改，训练侧脚本不会突然 KeyError
- **两套消费方式**：同进程直接调用；外部进程走 `GET /metrics?format=json`

---

## 2. 快照字段：每个字段能解释什么

```json
{
  "timestamp": 1795...,
  "uptime_s": 3600.2,
  "model": "qwen3-0.6b",

  "requests": {"total": 1206, "finished": 1206, "failed": 0, "tokens": 77184},

  "scheduler": {"running": 0, "waiting": 0, "steps": 8120, "last_batch_size": 1},

  "kv": {"used_ratio": 0.42, "free_blocks": 155, "total_blocks": 267,
          "cached_blocks": 96, "cache_capacity": 200},

  "ttft_s": {"p50": 0.081, "p95": 0.121, "p99": 0.15, "count": 1198, "sum": 97.3},

  "prefix_cache": {"hits": 120, "hit_blocks": 1536},

  "gpu": {"device": "cuda:1", "memory_allocated_mb": 412.3,
           "memory_reserved_mb": 512.0}
}
```

| 字段 | 解释什么 |
|---|---|
| `requests.total/finished/failed` | 趋势差分出 QPS、错误率；`failed` 增长 = 容量或推理异常 |
| `requests.tokens` | 差分出 tokens/s（rollout 吞吐） |
| `scheduler.running` | 正在 decode 的请求数（并发度） |
| `scheduler.waiting` | **背压信号**：持续增长说明提交过快或调度跟不上 |
| `scheduler.steps` | 调度步数（差分出步频） |
| `kv.used_ratio` | 池接近 1 → 新请求开始 503；训练侧应限流 |
| `kv.cached_blocks` | 前缀缓存占用（GRPO 重复 prompt 时很高） |
| `ttft_s` | 最近 256 个样本的首 token 延迟（p50/p95/p99 + count） |
| `prefix_cache.hits` | 命中请求数；解释吞吐波动（命中高 → prefill 省算力） |
| `gpu.memory_allocated_mb` | **同进程统计**：训练 + 推理的总显存；训推同卡时看是否接近上限 |

> `ttft_s` 是滑动窗口，不是全局统计；`count` 告诉你样本量。
> 分位数口径与 `/metrics` 文本一致（nearest，非插值），避免两套数打架。

---

## 3. 为什么同进程要直接读，而不是 HTTP

| 方式 | 开销 | 额外依赖 |
|---|---|---|
| 直接 `snapshot(engine)` | 微秒级（读内存 + 少量计算） | 无 |
| `GET /metrics?format=json` | 本机 HTTP ~0.1-1ms | 需要 server 启动、端口 |

同进程训练每 step 采集一次，直接读内存最省事；HTTP 端点保留给
**外部工具**（看板、shell 脚本、另一个进程）使用。

`/metrics?format=json` 与现有 Prometheus 文本共用同一批字段（TTFT 口径一致），
文本输出完全不变（向后兼容）。

---

## 4. JSONL 为什么适合训练监控

```python
append_snapshot("logs/engine.jsonl", engine, step=0, reward=0.5)
append_snapshot("logs/engine.jsonl", engine, step=1, reward=0.7)
```

- 每 step 一行 JSON：追加写，**不需要读旧数据**，崩溃也保留已写部分
- 训练结束一行命令读出：

```python
import pandas as pd
df = pd.read_json("logs/engine.jsonl", lines=True)
df["tokens_per_s"] = df["requests"].apply(lambda r: r["tokens"]).diff() / df["uptime_s"].diff()

df.plot(x="step", y=["tokens_per_s"])
```

- 与覆盖写（`write_snapshot`）相比：JSONL 保留历史，适合画曲线；
  覆盖写只适合"看当前一眼"

`extra` 参数允许把训练侧字段合并进同一行：

```python
append_snapshot(path, engine, step=step, loss=loss, reward=reward, lr=lr)
```

这样绘图时训练指标和引擎指标天然对齐（同一 step、同一时间戳）。

---

## 5. 实现细节（`qwen3/monitoring.py`）

### 5.1 为什么不依赖 server.py

```python
def snapshot(engine, model_id=""):
    m = engine.metrics
    sched = engine.scheduler
    pool = engine.pool
    ...
```

只访问 duck-typed 属性。好处：

- 训练侧可以在测试/离线场景塞 fake engine（见 `tests/unit/test_monitor.py`）
- 避免 import server 时附带 HTTP/日志初始化
- 未来引擎重构（拆模块）不影响监控接口

### 5.2 TTFT 加锁读取

```python
lock = getattr(m, "lock", None)
if lock is not None:
    with lock:
        ttft_vals = list(m.ttfts)
else:
    ttft_vals = list(getattr(m, "ttfts", []))
```

Engine 线程会 `ttfts.append`，训练线程读取；`deque` 迭代时被修改会抛
`RuntimeError`，所以加锁快照。没有 lock 的自定义 metrics 也能工作。

### 5.3 GPU 字段

```python
if device is not None and getattr(device, "type", None) == "cuda":
    gpu = {
        "device": str(device),
        "memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
        "memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
    }
```

- `memory_allocated` 是**进程级**统计：训推同进程时同时包含训练和推理
- 这是同机场景最关心的：推理 KV 池 + 训练激活 + 优化器状态是否接近显存上限
- CPU 设备时字段为 `None`（训练侧要判空）

### 5.4 JSONL 追加

```python
def append_snapshot(path, engine, model_id="", **extra):
    snap = snapshot(engine, model_id)
    if extra:
        snap.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")
    return snap
```

- 每次打开/关闭文件（训练 step 通常秒级，开销可忽略）
- `ensure_ascii=False`：中文模型名/标签可读
- 自动建目录

---

## 6. 训练侧集成示例

```python
from qwen3.monitoring import append_snapshot

for step in range(num_steps):
    rollouts = generate_rollouts(engine, prompts)     # 你的 rollout 逻辑

    # 可选：背压（用 waiting / kv 判断）
    snap = append_snapshot("logs/engine.jsonl", engine,
                           model_id="qwen3-0.6b",
                           step=step, reward=rollouts.mean_reward)
    if snap["scheduler"]["waiting"] > 16:
        time.sleep(0.2)                               # 队列太长，缓一缓
```

完整可运行示例：`integrations/grpo_example.py`（fake engine，生成示例 JSONL）。

---

## 7. 同机训练的注意事项

1. **GPU 竞争**：训练和推理共享卡时，引擎的 TTFT/吞吐会随训练步波动——
   这本身是有用信息（训练侧能看到推理被训练拖慢多少），但解释时要结合训练状态
2. **采样频率**：同进程每 step 采一次即可；不要在一个 step 里高频采样
   （`ttfts` 是滑动窗口，高频采样意义不大）
3. **累计计数器**：训练侧做差分时，注意引擎重启会归零
   （`cur < prev` 时重置基线）
4. **分位数不要平均**：多引擎实例时每实例单独记录，不要对 p50 取平均
5. **训练结束的最后一 step**：可能还有未完成的请求（waiting/running > 0），
   快照会反映出来；训练侧要决定是否等待引擎清空

---

## 8. 测试

`tests/unit/test_monitor.py`（5 个用例）：

| 用例 | 验证 |
|---|---|
| `test_snapshot_structure_and_values` | 字段结构、KV 比例、TTFT count、JSON 可序列化 |
| `test_ttft_percentile_matches_prometheus_text` | 快照与 `/metrics` 文本的 TTFT 口径一致 |
| `test_append_snapshot_jsonl` | 每 step 一行、extra 合并、可读回 |
| `test_write_snapshot_overwrite` | 覆盖写、单文件 |
| `test_snapshot_works_without_lock` | duck typing（无 lock/prefix 字段） |

---

## 9. 自测问题

1. 同进程监控为什么优先直接读 `engine`，而不是 HTTP？
2. `waiting` 持续增长说明什么？训练侧应该怎么处理？
3. `ttft_s` 为什么带 `count`？为什么是滑动窗口？
4. `gpu.memory_allocated_mb` 在同进程时统计的是什么？
5. JSONL 与覆盖写 JSON 各适合什么场景？
6. 为什么接口不做差分/绘图？

---

## 10. 代码地图

| 功能 | 位置 |
|---|---|
| 快照/JSONL/覆盖写 | `qwen3/monitoring.py` |
| HTTP JSON 端点 | `server.py::OpenAIHandler.do_GET`（`/metrics?format=json`） |
| 训练侧示例 | `integrations/grpo_example.py` |
| 测试 | `tests/unit/test_monitor.py` |
