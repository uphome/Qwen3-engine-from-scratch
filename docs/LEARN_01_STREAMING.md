# 学习篇 01：SSE 流式输出与增量解码

> 对应代码：`server.py`（`_stream_generation` / `_sse_start` / `_sse_send` /
> `Engine._step` / `Engine._cleanup_cancelled`）、`tests/e2e/test_server_stream.py`
> 关联文档：[API.md](../API.md) 2.2、[CHANGELOG](../CHANGELOG.md)

---

## 0. 这篇讲什么

读完你应该能回答：

1. SSE 到底是什么协议？为什么不用 WebSocket？
2. 推理 token 在 **Engine 后台线程** 产生，怎么送到 **HTTP 线程**？
3. 为什么"流式"不是简单的 `for token: print`——增量解码难在哪？
4. 客户端一断开，服务端怎么知道、怎么停止白算？
5. 每个事件的格式为什么是那样？

前置知识：HTTP/1.1 基础（响应头、连接关闭）、Python 线程与 `queue.Queue`、
BPE 分词的基本概念。

---

## 1. 为什么需要流式

非流式请求的生命周期：

```
客户端 POST ──► 服务端 prefill（可能几百 ms）──► decode 全部 token ──► 一次性返回
                                                      ▲
                                              用户这段时间什么都看不到
```

流式把"等全部生成完"变成"首 token 一出来就推送"：

```
POST ──► prefill ──► token1 ──► token2 ──► ... ──► [DONE]
              ▲          ▲
        首包延迟(TTFT) 每个 token 立即到达客户端
```

对 0.6B 模型，350-token prompt 的 TTFT 约 40ms、后续每 token 6-10ms；
生成长文本时，流式对**感知延迟**的改善是数量级的。

---

## 2. SSE 协议：真的只是拼字符串

SSE（Server-Sent Events）的响应是一个**普通 HTTP 响应**，只是：

1. `Content-Type: text/event-stream`
2. 响应体按 `data: <payload>\n\n` 分帧
3. 服务端不设 `Content-Length`，**连接关闭代表结束**（close-delimited）
4. 服务端每帧后 **flush**，客户端逐帧读取

`curl -N` 可以直接看到原始帧：

```bash
curl -N -X POST http://127.0.0.1:8767/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":16,"stream":true}'
```

```
data: {"id":"chatcmpl-000","object":"chat.completion.chunk",...,"delta":{"role":"assistant","content":""}}

data: {"id":"chatcmpl-000",...,"delta":{"content":"我是"},"finish_reason":null}

data: {...,"delta":{},"finish_reason":"stop"}

data: [DONE]
```

没有 WebSocket、没有分块编码（chunked）、没有异步框架——因为**只有服务端单向推送**，
SSE 就是最小方案。用 `ThreadingHTTPServer` 的每连接一线程模型天然支持长响应。

---

## 3. 架构：两个线程 + 一个队列

这是整个流式实现最核心的设计。相关代码：

- `Job`（`server.py`）：请求的运行时状态
- `Engine._step`（后台线程）：每步产出 token
- `_stream_generation`（HTTP 线程）：读 token、写 SSE

### 3.1 Job 里的三个流式字段

```python
class Job:
    def __init__(...):
        ...
        self.is_stream = False                   # 是否流式请求
        self.token_queue: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()       # 客户端断开 / 取消
```

- `is_stream`：决定 Engine 是否往队列推 token
- `token_queue`：**生产者（Engine）→ 消费者（HTTP 线程）** 的单向通道
- `cancelled`：消费者反向通知生产者"别算了"

### 3.2 Engine 侧：每步推 token

`Engine._step` 在采样之后、收尾之前：

```python
# 流式推送本步新 token（on_step_done 之前/之后均可；先推送
# 保证 HTTP 线程尽早拿到首包）
for i, req in enumerate(batch.requests):
    job = self._jobs.get(req.request_id)
    if job is not None and job.is_stream:
        job.token_queue.put(int(next_tokens[i].item()))
```

关键点：

- `next_tokens` 是**整个 batch** 的采样结果（`(B,1)`），需要按请求逐个取
- 推送发生在 `on_step_done` **之前**：越早 put，HTTP 线程越早 flush
- `is_stream` 判断避免给非流式请求做无用功

### 3.3 三种"结束"都要推哨兵

消费者用 `None` 作为"没有更多 token"的哨兵。代码里有三处：

```python
# ① 正常完成（Engine._step 完成分支）
if job.is_stream:
    job.token_queue.put(None)     # 流结束哨兵

# ② 推理异常（Engine._step except 分支）
if job.is_stream:
    job.token_queue.put(None)

# ③ 取消（Engine._cleanup_cancelled）
if job.is_stream:
    job.token_queue.put(None)
```

**这是踩过坑的地方**：最初只做了 ①，结果流式请求被容量拒绝时
（`_dispatch` 的失败分支直接 `job.done.set()`，没有推哨兵），
HTTP 线程会一直阻塞在 `queue.get(timeout=WAIT_TIMEOUT)`，挂 600 秒。
现在 `_dispatch` 的两个失败分支也补了 `job.token_queue.put(None)`。

**经验**：只要引入"阻塞读队列"的消费者，就要检查**所有**可能终止生产者的路径
（成功、异常、取消、拒绝入队）。

### 3.4 为什么不共享一个 list / 用回调？

| 方案 | 问题 |
|---|---|
| 共享 list + 忙轮询 | 空转烧 CPU；sleep 增加延迟 |
| 回调（Engine 直接调 HTTP 写） | Engine 线程写 socket 会被慢客户端阻塞，拖垮整个调度循环 |
| `queue.Queue` | 解耦：Engine 只负责生产，HTTP 线程负责网络 IO；消费者慢不会阻塞生产者（队列有缓冲） |

---

## 4. `_stream_generation` 逐段精读

### 4.1 准备与"先等第一个 token"

```python
job, input_len = self._prepare_job(prompt_text, body)
job.is_stream = True
srv.engine.submit(job)

try:
    first = job.token_queue.get(timeout=WAIT_TIMEOUT)
except queue.Empty:
    job.cancelled.set()
    raise OpenAIError(503, "生成超时（请求仍在后台继续，结果丢弃）")

if first is None:                  # 未开始产出：错误还在 HTTP 头之前
    if job.error:
        raise OpenAIError(503, job.error)
    self._sse_start()
    self._sse_send("[DONE]")
    return
```

**为什么先等第一个 token 再发响应头？**

HTTP 响应头一旦发出（`send_response(200)`），状态码就固定了。
如果请求被容量拒绝（503）或推理异常，我们希望返回**正常的 JSON 错误**，
而不是"200 + 流里一个 error 事件"。

所以顺序是：

```
submit → 等第一个 token（或 None 哨兵）
   ├─ 拿到 None 且 job.error → 还没发头 → 正常 HTTP 503
   └─ 拿到真实 token → 已确认能生成 → 发 200 + SSE 头 → 边收边发
```

代价：首 token 之前多一次等待（本来也要等 prefill），没有额外开销。

### 4.2 响应头

```python
def _sse_start(self):
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
    self.send_header("Cache-Control", "no-cache")
    self.send_header("Connection", "close")
    self.send_header("X-Accel-Buffering", "no")   # nginx 等反代不缓冲
    self.send_header("Access-Control-Allow-Origin", "*")
    self.end_headers()
    self.close_connection = True
```

逐行解释：

| 头 | 作用 |
|---|---|
| `text/event-stream` | 声明 SSE |
| `no-cache` | 禁止中间层缓存流 |
| `Connection: close` | **没有 Content-Length 时用连接关闭表示结束**（HTTP/1.1 合法） |
| `X-Accel-Buffering: no` | 告诉 nginx 不要缓冲（否则代理会攒一大块再发） |
| `close_connection = True` | 让 `BaseHTTPRequestHandler` 写完就关，不尝试 keep-alive |

> 另一种做法是手动实现 HTTP chunked 编码（`Transfer-Encoding: chunked`），
> 但 close-delimited 更简单且 curl / OpenAI SDK 都兼容。

### 4.3 发送一帧

```python
def _sse_send(self, payload):
    text = (payload if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False))
    self.wfile.write(f"data: {text}\n\n".encode("utf-8"))
    self.wfile.flush()      # 关键：不等缓冲区满，立刻推给客户端
```

- `ensure_ascii=False`：中文不转义，帧更小、可读
- `flush()` 是流式的灵魂：`wfile` 是带缓冲的，不 flush 会攒到 8KB 才发
- `data:` + 两个换行是 SSE 的帧边界

### 4.4 事件格式（`emit` 闭包）

```python
def emit(delta: dict, finish_reason=None):
    if chat:
        obj = {"id": chunk_id, "object": "chat.completion.chunk",
               "created": created, "model": srv.model_id,
               "choices": [{"index": 0, "delta": delta,
                            "finish_reason": finish_reason}]}
    else:
        obj = {"id": chunk_id, "object": "text_completion",
               "created": created, "model": srv.model_id,
               "choices": [{"index": 0, "text": delta.get("content", ""),
                            "finish_reason": finish_reason}]}
    self._sse_send(obj)
```

- `chat.completion.chunk` 用 `delta`；`text_completion` 用 `text` 增量
- 首帧 `emit({"role": "assistant", "content": ""})` 告诉客户端"助手开始说话了"
- 结束帧 `emit({}, finish_reason)`：`delta` 为空、给出停止原因（`stop`/`length`）
- `stream_options.include_usage` 时再追加一个 `choices: []` 的 usage 帧（OpenAI 同款）

### 4.5 增量解码（重点）

**朴素实现**（本项目最初的写法）：

```python
ids = []
while tok is not None:
    ids.append(tok)
    text = tokenizer.decode(ids, skip_special_tokens=True)   # 每次都全量 decode
    delta = text[len(sent_text):]
    ...
```

问题：第 n 个 token 时要 decode 长度为 n 的序列 → 总计 O(n²)。
64 token 的请求还行；几百 token + 高并发时，tokenizer 的 Python 调用会变成瓶颈。

**第二个问题**：BPE token 是**字节级**的。一个中文字符（UTF-8 3 字节）
可能被拆成 2-3 个 token；单独 decode 第一个 token 会得到 `�`（U+FFFD，
替换字符）。所以不能简单 `decode([tok])` 直接推。

**当前实现**（`_stream_generation`）：

```python
pending: list[int] = []
n_tokens = 0
tok = first
while tok is not None:
    pending.append(tok)
    n_tokens += 1
    text = srv.tokenizer.decode(pending, skip_special_tokens=True)
    if "\ufffd" not in text:
        if text:
            emit({"content": text})
        pending = []
    elif len(pending) >= 8:
        # 防御：异常 token 序列迟迟不完整，强制输出避免卡住
        emit({"content": text})
        pending = []
    tok = job.token_queue.get(timeout=WAIT_TIMEOUT)

if pending:                    # 生成结束时仍未补齐（极少见）
    tail = srv.tokenizer.decode(pending, skip_special_tokens=True)
    if tail:
        emit({"content": tail})
```

逐行理解：

| 步骤 | 含义 |
|---|---|
| `pending.append(tok)` | 把新 token 放进"待解码缓冲区"（通常只有 1 个） |
| `decode(pending)` | 只 decode 缓冲区，O(1)（正常情况） |
| `"\ufffd" not in text` | 没有替换字符 = UTF-8 字节完整，可以安全输出 |
| `pending = []` | 完整了就清空，下一个 token 重新开始 |
| 含 U+FFFD | 字节不完整 → **保留**，等下一个 token 拼成完整字符 |
| `len(pending) >= 8` | 防御：若某 token 序列一直不完整（坏模型/特殊 token），不能无限等 |
| 循环结束后处理 `pending` | 生成结束时残留的不完整字节，尽量输出 |

**为什么正常情况是 O(1)**：绝大多数英文 token、完整的中文 token 单独 decode
就是合法字符串；只有跨 token 的多字节字符才会进 `pending` 等一次。
`pending` 通常长度 1-2，不会累积。

**验证**：`tests/e2e/test_server_stream.py` 的 `test_stream_matches_non_stream`
除了比较流式/非流式文本，还断言：

```python
assert "\ufffd" not in stream_text, "增量解码出现替换字符（UTF-8 被拆坏）"
```

### 4.6 错误、断开、超时

```python
try:
    ...  # 上面的循环
except (BrokenPipeError, ConnectionResetError):
    job.cancelled.set()
    log.log(REQUEST_LEVEL, f"客户端断开 req#{job.request_id}，取消生成")
except queue.Empty:
    job.cancelled.set()
    log.warning(f"流式生成超时 req#{job.request_id}（已取消）")
```

- `BrokenPipeError`：客户端关页面/杀 curl，下一次 `wfile.write` 立刻抛
- `ConnectionResetError`：对端 RST
- `queue.Empty`：单 token 间隔超过 `WAIT_TIMEOUT`（600s），视为卡死

**注意**：`job.cancelled.set()` 只是"贴 flag"，真正的清理在 Engine 侧（下一节）。

---

## 5. 取消闭环：从 BrokenPipe 到 KV 归还

```
HTTP 线程                          Engine 线程（每轮循环顶部）
_stream_generation                  run()
  wfile.write ──► BrokenPipe
  job.cancelled.set() ──────────►   _cleanup_cancelled()
                                      遍历 waiting/running
                                      若 cancelled:
                                        - 从队列移除
                                        - kv_cache.free()（归还块/行）
                                        - _pending_blocks -= blocks_reserved
                                        - token_queue.put(None)
                                        - job.done.set()
```

`_cleanup_cancelled` 代码（`Engine`）：

```python
def _cleanup_cancelled(self):
    for container in (self.scheduler.waiting, self.scheduler.running):
        for req in list(container):
            job = self._jobs.get(req.request_id)
            if job is None or not job.cancelled.is_set():
                continue
            container.remove(req)
            if req.kv_cache is not None:
                req.kv_cache.free()
                req.kv_cache = None
            self._jobs.pop(req.request_id, None)
            self._pending_blocks -= job.blocks_reserved
            if job.is_stream:
                job.token_queue.put(None)
            job.done.set()
```

**为什么在 Engine 侧清理，而不是 HTTP 线程直接 free？**

- `kv_cache.free()` 会改池状态（`block_refs`、LRU、free 列表），必须与调度器**串行**
- HTTP 线程只贴 `cancelled` 标志，Engine 在自己的循环里统一处理 → 无锁

**验证**：`test_client_disconnect_cancels_generation` 用 `http.client` 读 3 行后
`conn.close()`，然后轮询 `/metrics` 直到 `engine_running == 0` 且
`engine_kv_used_ratio < 0.05`（块已归还）。

---

## 6. 测试怎么写的（`tests/e2e/test_server_stream.py`）

### 6.1 fixture：启动真实 server

```python
@pytest.fixture(scope="module")
def server_url():
    if not os.path.isdir(MODEL):
        pytest.skip(...)
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "server.py", ...])
    # 轮询 /v1/models 直到 ready（最多 180s）
    yield url
    proc.terminate()
```

- 用 `subprocess` 起**真实进程**（不是进程内构造 handler）：覆盖启动、加载、
  图捕获、线程模型的完整链路
- `scope="module"`：三个用例共享一个 server（启动 ~30s，不能每个用例重启）

### 6.2 三个用例分别验证什么

| 用例 | 验证点 |
|---|---|
| `test_stream_matches_non_stream` | 流式拼接文本 == 非流式；无 U+FFFD |
| `test_stream_chunk_format` | `object` 类型、首帧 role、finish_reason、usage 帧、`[DONE]` |
| `test_client_disconnect_cancels_generation` | 断开后 running/waiting 归零、KV 归还 |

读取 SSE 的辅助函数：

```python
def _sse_events(url, body):
    text, events, done = "", [], False
    with _post_json(url + "/v1/chat/completions", body) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            obj = json.loads(payload)
            events.append(obj)
            if obj.get("choices"):      # usage 帧的 choices 为空
                text += obj["choices"][0]["delta"].get("content", "")
    return text, events, done
```

---

## 7. 踩过的坑（都真实发生过）

1. **容量拒绝 → 流式挂起 600s**
   - 原因：`_dispatch` 失败分支只 `done.set()`，没推 `token_queue` 哨兵
   - 修复：所有失败路径都 `put(None)`
   - 教训：引入阻塞队列后，检查所有"不再生产"的路径

2. **O(n²) 累计 decode**
   - 原因：每 token 全量 `decode(ids)` + 字符串切片
   - 修复：`pending` 增量解码

3. **UTF-8 被拆坏**
   - 现象：中文流式输出出现 `�`
   - 根因：单 token 是半个字符的字节
   - 修复：U+FFFD 检测 + 暂存等补齐

4. **首 token 之前就发 200**
   - 后果：容量拒绝只能伪装成流内 error 事件，客户端拿不到 503
   - 修复：先等第一个 token/哨兵，再发响应头

5. **`XAceel-Buffering` 缺失时经 nginx 后"不流式"**
   - 现象：本地 curl 是流式，经过反代变成一次性输出
   - 修复：`X-Accel-Buffering: no`

---

## 8. 动手实验

1. **观察帧**：启动 server，`curl -N` 看原始帧；对比 `--no-buffer` 与默认
2. **观察 TTFT**：`/metrics` 的 `engine_ttft_seconds` 与 curl 首包时间对齐
3. **断开取消**：`curl -N --max-time 1` 打一个长请求，观察 server 日志
   （`-v` 启动可看到"取消 req#..."）和 `/metrics` 的 running 归零
4. **破坏增量解码**：把 `pending` 逻辑临时改回"每 token decode 单 id"，
   用中文 prompt 观察 `�`；再改回来
5. **压测对照**：跑 `bench_server.py --stream` vs 不带 `--stream`，
   对比 TTFT p50（流式才能测到精确首包）

---

## 9. 自测问题

1. 为什么 `_sse_start` 要设置 `Connection: close`？不设会怎样？
2. 如果客户端在第 3 个 token 后断开，服务端从哪一行代码感知？清理是谁做的？
3. 增量解码为什么不能直接 `decode([tok])`？`pending` 一般有多长？
4. 流式请求被容量拒绝时，客户端收到的是 503 还是 200+error 事件？为什么？
5. `emit` 的 `finish_reason` 和 usage 帧分别在哪一步发送？

---

## 10. 代码地图

| 功能 | 位置 |
|---|---|
| Job 字段（token_queue/cancelled） | `server.py::Job` |
| 每步推 token / 三种终止哨兵 | `server.py::Engine._step`、`_dispatch`、`_cleanup_cancelled` |
| SSE 头/帧 | `server.py::OpenAIHandler._sse_start`、`_sse_send` |
| 流式主循环/增量解码 | `server.py::OpenAIHandler._stream_generation` |
| 首 token 前错误处理 | `OpenAIHandler._prepare_job` + `_stream_generation` 开头 |
| 测试 | `tests/e2e/test_server_stream.py` |
