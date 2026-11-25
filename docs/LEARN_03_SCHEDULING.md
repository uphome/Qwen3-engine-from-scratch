# 学习篇 03：调度与 chunked prefill

> 对应代码：`qwen3/scheduler.py`、`qwen3/request.py`、`qwen3/batch.py`、
> `server.py::Engine`、`bench_batched.py::run_batched`、
> `tests/unit/test_chunked_prefill.py`
> 关联文档：[ENGINE_COMPARISON.md](../ENGINE_COMPARISON.md)、[CHANGELOG](../CHANGELOG.md)

---

## 0. 这篇讲什么

读完你应该能回答：

1. continuous batching 的调度器每个 step 到底在做什么？
2. prefill 批和 decode 批为什么不能混在一起（在本项目里）？
3. `on_step_done` 三种分支的职责边界？
4. chunked prefill 怎么切片？为什么与 decode 交替？
5. 为什么中间 chunk 不能采样、TTFT 不能记？
6. 长 prompt 切片后，位置编码/KV 写入怎么对齐？
7. 容量记账（`blocks_reserved`）解决什么问题？

前置知识：prefill / decode 的区别、KV cache、causal attention。

---

## 1. 两种调度模式

| 模式 | 场景 | 特点 |
|---|---|---|
| 静态批处理 | 离线、长度相近 | 简单，但短请求等长请求 |
| **continuous batching** | 在线服务 | 每步重新组批：完成的踢出、新到的加入 |

本项目的调度器每个 step：

```
schedule() → 组一个 Batch（prefill 或 decode）
model/batch 前向 → 采样
on_step_done() → 写回 token、完成者出批、推进状态
```

循环在 `Engine.run`（常驻服务）和 `bench_batched.run_batched`（基准）里，
两者是**同构**的（后面 7 节对比）。

---

## 2. Request 状态机

`qwen3/request.py`：

```python
_VALID_TRANSITIONS = {
    WAITING: (PREFILLING,),
    PREFILLING: (DECODING, PREFILLING),   # 第二项是 chunked prefill 的续传
    DECODING: (DECODING, FINISHED),
    FINISHED: (),
}
```

状态含义：

| 状态 | 含义 |
|---|---|
| `waiting` | 已入队，等被调度做 prefill |
| `prefilling` | 本步参与 prefill（chunked 时可能持续多步） |
| `decoding` | prefill 完成，每步 decode 一个 token |
| `finished` | EOS/达上限，KV 已归还 |

所有迁移都经过 `_transition()` 校验：

```python
def _transition(self, new_state):
    assert new_state in _VALID_TRANSITIONS[self.state], \
        f"[Request {self.request_id}] 非法状态迁移: {self.state} -> {new_state}"
    self.state = new_state
```

**为什么要严格校验**：调度器写错顺序（例如没 prefill 就 decode）会在
assert 处立刻暴露，而不是带病跑出错误结果再排查。测试里专门有一条
`test_illegal_transition_raises`。

---

## 3. 三个队列与决策树

```python
self.waiting: deque[Request] = deque()   # 新请求 / chunk 续传（队头）
self.running: list[Request] = []          # 已 prefill、正在 decode
self.finished: list[Request] = []         # 已完成（保留输出供收集）
```

`Scheduler.schedule()`：

```python
if self.running:
    # running 有空位且 waiting 有人：prefill 补位（与 decode 交替）
    if (len(self.running) < self.batch_size and self.waiting
            and not self._prefill_turn):
        return self._schedule_prefill()
    return self._schedule_decode()

if self.waiting:
    return self._schedule_prefill()

return None
```

决策表：

| running | waiting | 上一步 | 结果 |
|---|---|---|---|
| 空 | 有 | - | prefill |
| 非空 | 无 | - | decode |
| 非空 | 有 | 上一步 decode | prefill 补位 |
| 非空 | 有 | 上一步 prefill | decode（防 prefill 连续占满） |
| 非空 | 有 且 running 已满 | - | decode |

**补位防超编**（历史 bug）：

```python
slots = self.batch_size - len(self.running)
n = min(slots, len(self.waiting))
```

如果按 `min(batch_size, len(waiting))` 取，`running=2, batch=4, waiting=4`
会一次补 4 个 → running=6 → decode 的 `running[:n]` 切片永远饿死队尾。

---

## 4. 组批：prefill 右 pad / decode 拼 token

`Batch`（`qwen3/batch.py`）：

```python
def build_input_ids(self):
    if self.mode == "decode":
        return torch.cat([r.cur_token for r in self.requests], dim=0)  # (B,1)

    parts = [r.input_ids[0, r.prefill_offset:r.prefill_end]
             for r in self.requests]
    S_max = max(p.shape[0] for p in parts)
    padded = torch.full((self.size, S_max), self.PAD_ID, ...)
    for i, p in enumerate(parts):
        padded[i, :p.shape[0]] = p
    return padded
```

- **prefill**：各请求长度不同 → 右 pad 到 `S_max`；`input_lens` 给出有效长度
- **decode**：每请求取 `cur_token`（上一步采样结果），形状 `(B,1)`

`input_lens`：

```python
if self.mode == "decode":
    return [1] * self.size
return [r.prefill_end - r.prefill_offset for r in self.requests]
```

`prefill_end - prefill_offset` 同时覆盖两种场景：
- 普通 prefill：`0 .. input_len`
- 前缀缓存部分命中：`matched .. input_len`
- chunked prefill：当前 chunk 的区间

---

## 5. on_step_done：三种收尾

```python
for i, req in enumerate(batch.requests):
    is_partial_chunk = (batch.mode == "prefill"
                        and req.prefill_end < req.input_len)
    if not is_partial_chunk:
        req.append(next_tokens[i].item())

    if batch.mode == "prefill":
        # 记录完整 prompt 块的哈希（free 时登记前缀缓存）
        ...
        req.prefill_offset = req.prefill_end
        if req.prefill_offset < req.input_len:
            self.waiting.appendleft(req)      # 还有 chunk
        else:
            req.enter_decoding()
            self.running.append(req)
    else:
        if req.is_finished():
            req.finish()
            self.running.remove(req)
            self.finished.append(req)
```

三个细节：

1. **prefill 分支不判断 `is_finished`**：prefill 只产出一个首 token，
   是否结束由后续 decode 步判断（`is_finished` 需要看最后一个 token 是否 EOS）
2. **中间 chunk 不 append**（`is_partial_chunk`）：见第 6 节
3. **`appendleft`**：chunk 续传放队头，保证长请求连续推进；
   与 decode 的交替由 `_prefill_turn` 控制，不会饿死别的请求

---

## 6. chunked prefill

### 6.1 为什么需要

长 prompt（如 4k token）整段 prefill 要几百 ms。在这期间：

- 已在 running 的短请求无法 decode（token 间隔被拉长）
- 新到达的短请求 TTFT 变差

chunked prefill 把长 prompt 切成 `chunk_size=512` 的片段，每步只 prefill 一段，
**段与段之间穿插 decode**。

### 6.2 切片

`_schedule_prefill`：

```python
for req in reqs:
    req.start_prefill()
    remaining = req.input_len - req.prefill_offset
    take = remaining if self.chunk_size <= 0 else min(self.chunk_size, remaining)
    req.prefill_end = req.prefill_offset + take
self._prefill_turn = True
```

`chunk_size` 默认 512，可用 `QWEN3_CHUNK_SIZE` 调整（0 = 关闭）。
注意 `prefill_offset` 可能非 0（前缀缓存部分命中的起点）。

### 6.3 与 decode 交替

`_prefill_turn` 标记上一步是否是 prefill：

```
prefill(chunk1) → decode → prefill(chunk2) → decode → ...
```

效果（2048 长 prompt + 短请求并发，隔离前缀缓存的实测）：

| 指标 | 整段 prefill | chunked | 变化 |
|---|---:|---:|---:|
| 短请求 TTFT | 182.5 ms | 66.9 ms | **2.73x** |
| 短请求完成 | 382 ms | 413 ms | 0.92x（权衡） |
| 长请求 TTFT | 182 ms | 239 ms | 0.76x（权衡） |

**权衡解释**：chunked 让短请求的首 token 不用等整段长 prefill（TTFT 大改善），
但长请求的 chunk 步会占用原本用于 decode 的时间（短请求完成时间略增、
长请求 TTFT 略增）。这是"不混批"版本的固有代价；vLLM 在同一步里同时
prefill chunk + decode，会更优（本项目的阶段 3 可做）。

### 6.4 中间 chunk 不能采样（真实 bug）

第一版实现中，每个 chunk 都走 `sample_tokens` 并 `append`：

- 中间 chunk 的 logits 是"只有半截上下文"的产物，不是真正的首 token
- 后果：输出多出伪 token、`num_generated` 提前增加、finish 判断错乱

修复（`on_step_done`）：

```python
is_partial_chunk = (batch.mode == "prefill" and req.prefill_end < req.input_len)
if not is_partial_chunk:
    req.append(next_tokens[i].item())
```

**注意**：只跳过 `append`，**收尾逻辑照常执行**（更新 `prefill_offset`、
放回 waiting）。第一版修复时误用了 `continue`，直接跳过了收尾 → 请求永远
卡在 waiting（`schedule()` 返回 None）。

### 6.5 TTFT 记录要跳过中间 chunk

`Engine._step` 里 TTFT 记录：

```python
is_first_token = (batch.mode == "decode"
                  or req.prefill_end >= req.input_len)
if (job is not None and job.request_id >= 0
        and job.ttft is None and req.num_generated == 0
        and is_first_token):
    job.ttft = now - job.submit_time
```

不修的话，长请求的 TTFT 会记在第一个 chunk 完成时（虚低），
与"真正的首 token 时刻"不符。

### 6.6 位置编码与 KV 写入

chunk 2 的前向：

- `cache.seq_len` = chunk 1 已写入的长度（`advance_seq_len` 在 model 内累加）
- `position_ids = arange(S) + seq_len`（model 的 prefill 分支）
- `cache.update(layer, k, v)` 从 `seq_len` 继续写
- attention 有历史 → `paged_attention_prefill_from_history`（学习篇 02 第 5 节）

所以 chunked prefill 与 prefix caching 复用的是**同一套"后缀 prefill + 读历史"**机制。

### 6.7 数值等价

`tests/model/test_chunked_prefill_equiv.py`：

- 整段 prefill 40 token
- 分块 16/16/8
- 最后一块的 logits 与整段后 8 行逐元素对比（CPU fp32，diff < 1e-4）

调度层测试 `tests/unit/test_chunked_prefill.py`：

- 1200 token / chunk 512 → `input_lens == [512, 512, 176]`
- `chunk_size=0` → 一次整段
- 与 decode 交替的模式（相邻两步模式一定不同）
- 池闭环（`free + cached == total`）

---

## 7. 两个同构调度循环

### 7.1 常驻服务（`server.py::Engine.run`）

```python
while True:
    self._cleanup_cancelled()              # 清理断开请求
    if not self.scheduler.has_pending():
        job = self.q.get()                 # 阻塞等新作业
        if job is None: return
        self._dispatch(job)
    while True:                            # drain 队列
        try: job = self.q.get_nowait()
        except queue.Empty: break
        if job is None: return
        self._dispatch(job)
    self._step()                           # 一个调度步
```

### 7.2 基准（`bench_batched.py::run_batched`）

```python
while scheduler.has_pending():
    batch = scheduler.schedule()
    t_step = time.perf_counter()
    if batch.mode == "decode" and runner is not None:
        ... runner.replay(...)             # 走 CUDA Graph
    else:
        logits = model(...)                # prefill / 无图 decode
    next_tokens = ...
    if logits.is_cuda: torch.cuda.synchronize()
    dt = time.perf_counter() - t_step
    scheduler.on_step_done(batch, next_tokens)
```

两者共享 `Scheduler`/`Batch`/`on_step_done`，区别只在：
- 服务版有队列、取消、指标、流式推送
- 基准版直接同步计时

**理解这一点**：调度逻辑只有一份，服务和基准只是两个驱动循环。

---

## 8. 容量记账（防超卖）

`Engine._dispatch`：

```python
needed = self._needed_blocks(input_len, job.max_new_tokens)   # 含 decode 空间
hit_blocks = self.scheduler.add(req, self._pending_blocks)    # 内部精确检查
job.blocks_reserved = max(0, needed - hit_blocks)
self._pending_blocks += job.blocks_reserved
```

- `_pending_blocks`：所有"已入队未归还"请求的块数之和
- `Scheduler.add(reserved_blocks=self._pending_blocks)`：检查时扣掉已记账的块，
  防止两个请求同时通过检查后一起 OOM
- 命中请求只记新块（`needed - hit_blocks`）
- 完成/异常/取消时 `_pending_blocks -= job.blocks_reserved`（必须用同一个值，
  否则记账漂移）

---

## 9. 踩过的坑（真实）

1. **补位超编** → `running` 超过 `batch_size`，队尾饿死
2. **chunk 中间采样** → 伪 token
3. **chunk 收尾被 `continue` 跳过** → 请求卡在 waiting
4. **TTFT 记在第一个 chunk** → 指标虚低
5. **HTTP 层全量容量预检** → 命中请求被 503（学习篇 02 第 9 节）
6. **`--batch-sizes` 没参与池容量计算** → 压测 OOM
7. **长稳窗口 worker 批量上报** → 窗口统计失效（压测文档）

---

## 10. 动手实验

1. **观察调度**：在 `Engine._step` 加一行打印 `batch.mode, batch.size, [r.prefill_offset for r in batch.requests]`，跑长 prompt + 短请求
2. **切 chunk 大小**：`QWEN3_CHUNK_SIZE=128/512/2048`，对比短请求 TTFT 与完成时间
3. **关掉 chunk**：`QWEN3_CHUNK_SIZE=0`，观察长 prompt 对短请求的阻塞
4. **读测试**：`test_decode_interleaves_with_prefill_chunks` 画一遍模式序列
5. **验证数值**：跑 `test_chunked_prefill_equiv`，再故意把 position offset 去掉，看 diff

---

## 11. 自测问题

1. `schedule()` 在 `running` 非空、`waiting` 有请求、上一步是 prefill 时做什么？为什么？
2. 为什么 prefill 批右 pad，而 decode 批不用 pad？
3. chunked prefill 的中间 chunk 为什么不能采样？不做修复会出现什么现象？
4. `prefill_offset` 有哪两个来源？
5. 长 prompt + 短请求并发时，chunked 为什么改善短请求 TTFT 却可能让完成时间变差？
6. `_pending_blocks` 的作用？如果没有它会怎样？
7. 服务版和基准版的调度循环有什么共同点？

---

## 12. 代码地图

| 功能 | 位置 |
|---|---|
| 状态机 | `qwen3/request.py` |
| 调度决策/组批/收尾 | `qwen3/scheduler.py` |
| 批组装 | `qwen3/batch.py` |
| 常驻调度循环/取消/记账 | `server.py::Engine` |
| 基准调度循环 | `bench_batched.py::run_batched` |
| chunked 测试 | `tests/unit/test_chunked_prefill.py`、`tests/model/test_chunked_prefill_equiv.py` |
