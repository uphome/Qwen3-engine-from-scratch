# Qwen3 Engine HTTP API 使用文档

本项目自研推理引擎的 HTTP API 服务（`server.py`），接口兼容 OpenAI 格式，
纯 Python 标准库实现，零额外框架依赖。

---

## 1. 快速开始

```bash
cd Qwen3-engine-from-scratch
conda activate qwen3        # 或使用任意装了 torch/transformers 的环境
python server.py --model E:/codeall/Qwen3-0.6B --port 8000
```

看到以下输出即成功：

```
Qwen3 HTTP API Server — OpenAI 兼容
Device: cuda:0, dtype: torch.bfloat16
  KV pool: 632 块 × 16 tokens ...
  CUDA Graph: 开启 (buckets=[1, 2, 4] ...)
  Warmup: 完成
Listening on http://127.0.0.1:8000
```

启动参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必填 | 模型路径或 HF 模型名 |
| `--host` | `127.0.0.1` | 监听地址（局域网访问设 `0.0.0.0`） |
| `--port` | `8000` | 监听端口 |
| `--batch-size` | `4` | 连续批处理最大并发（4GB 卡建议 ≤4） |
| `--max-tokens` | `512` | 单请求生成 token 上限 |
| `--kv-ratio` | `0.6` | KV 池占剩余显存比例（OOM 调低） |
| `--no-graph` | 关 | 禁用 CUDA Graph decode |
| `-v` / `--verbose` | - | 控制台显式 DEBUG 级别（默认 INFO） |
| `--no-log-file` | 关 | 只打控制台，不写日志文件 |

日志：控制台 + `logs/server.log`（UTF-8，5MB×3 滚动，已 gitignore）。

---

## 2. 接口总览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/v1/models` | 查询可用模型 |
| `POST` | `/v1/chat/completions` | 对话补全（messages 格式） |
| `POST` | `/v1/completions` | 纯文本补全（prompt 格式） |

---

## 3. Chat 补全（主要接口）

### 请求

```http
POST /v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "qwen3-0.6b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "什么是机器学习？用简单的话回答"}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9
}
```

字段说明（除 `messages` 外均可省略）：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model` | string | - | 忽略（单模型服务） |
| `messages` | array | 必填 | 对话历史，role 取 `system`/`user`/`assistant` |
| `max_tokens` | int | 512 | 生成上限，最大 `--max-tokens` |
| `temperature` | float | 0.7 | 采样温度，0 = 贪心 |
| `top_k` | int | 50 | top-k 截断 |
| `top_p` | float | 0.9 | nucleus 采样 |
| `enable_thinking` | bool | false | 是否开启思考链（`false` = 空思考块直接回答；`true` = 走 Qwen3 原生 ` thinking` 推理） |
| `tools` | array | 无 | 函数定义列表（OpenAI 格式），提供则走 Tool Calling |
| `stream` | bool | false | 暂不支持，传 `true` 返回 400 |

### 响应

```json
{
  "id": "chatcmpl-003",
  "object": "chat.completion",
  "created": 1786961314,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "机器学习是……"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 16, "completion_tokens": 128, "total_tokens": 144}
}
```

`finish_reason`：
- `stop` — 模型输出了结束符 `<|im_end|>`
- `length` — 达到 `max_tokens` 上限
- `tool_calls` — 请求了函数调用（见下）

### 3.3 函数调用（Tool Calling）

Qwen3 原生支持，请求里带 `tools` 即启用。两轮调用模式：

**第 1 轮**：带 `tools` 请求 → 模型返回 `finish_reason: "tool_calls"`，
`message.tool_calls` 里是函数名 + JSON 参数：

```json
{
  "id": "chatcmpl-000", "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant", "content": null,
      "tool_calls": [{
        "id": "call_0000", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 162, "completion_tokens": 21, "total_tokens": 183}
}
```

调用方自己执行 `get_weather`，然后把结果作为 `role: "tool"` 消息回传，
（`tools` 必须继续带上）进行第 2 轮，模型根据结果产出最终回答：

```json
{
  "messages": [
    {"role": "user", "content": "What is the weather in Beijing?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_0000", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_0000", "content": "{\"temperature\": 25, \"condition\": \"sunny\"}"}
  ],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "query the weather of a city",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "max_tokens": 150,
  "temperature": 0
}
```

`tools` 为 OpenAI 标准函数定义格式；内部用模型官方 Jinja 模板渲染
（`<tools>` / `<tool_response>` 标签），与训练格式一致。

---

## 4. 纯文本补全

```http
POST /v1/completions
Content-Type: application/json
```

```json
{"prompt": "1+1=", "max_tokens": 64, "temperature": 0}
```

响应结构与 chat 一致，`choices[0].text` 为生成文本。

---

## 5. 调用示例

### 5.1 curl

```bash
# 查模型
curl http://127.0.0.1:8000/v1/models

# 对话
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```

> Windows 控制台直接传中文可能乱码。中文请求建议把 JSON 存成 UTF-8 文件：
> ```bash
> curl -X POST http://127.0.0.1:8000/v1/chat/completions \
>   -H "Content-Type: application/json" \
>   --data-binary @req.json
> ```

### 5.2 Python（标准库）

```python
import json
import urllib.request

body = json.dumps({
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128,
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"})

with urllib.request.urlopen(req) as r:
    data = json.load(r)

print(data["choices"][0]["message"]["content"])
```

### 5.3 openai SDK

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

### 5.4 requests

```bash
pip install requests
```

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={"messages": [{"role": "user", "content": "你好"}]},
    timeout=300,
)
print(resp.json()["choices"][0]["message"]["content"])
```

---

## 6. 错误处理

所有错误统一 OpenAI 格式：

```json
{"error": {"message": "错误描述", "type": "invalid_request_error", "code": 400}}
```

| 状态码 | 触发场景 |
|---|---|
| `400` | 请求体不是合法 JSON / 缺少 `messages` / `stream` 暂不支持 / prompt 超过 2048 tokens |
| `404` | 路径不存在 |
| `500` | 服务器内部错误（带 traceback） |
| `503` | KV 池容量不足或推理中途 OOM（服务不会崩溃，可继续调用） |

生成耗时可能较长（长 prompt + 大 `max_tokens` 在 4GB 卡上可能 1-2 分钟），
客户端超时建议设 300 秒以上。

---

## 6.1 日志系统

每条日志格式：`时间 | 级别 | 线程 | 内容`，控制台与 `logs/server.log` 双写
（文件固定 UTF-8，Windows 控制台中文乱码时看文件）。

| 级别 | 用途 | 示例 |
|---|---|---|
| `DEBUG` | 底层调试（`-v` 打开） | kernel 级信息 |
| `INFO` | 启动/配置/关闭 | 池容量、模型信息、Listening... |
| `REQUEST` | 每请求一条（含访问日志和引擎入队/完成） | `POST /v1/chat/completions -> 200` |
| `WARNING` | 资源裁剪、预热超时、翻译异常为客户端 503 | 显存自适应裁剪 |
| `ERROR` | 完整 traceback（仅写文件） | 推理步骤失败 |

查看方式：

```bash
tail -f logs/server.log                    # 实时跟踪
grep " ERROR " logs/server.log             # 只看错误
grep " REQUEST " logs/server.log           # 只看请求
```

## 6.2 错误处理与恢复

- **请求级错误**（400/404/503）：正常业务路径，返回 OpenAI 错误格式
  `{"error":{"message","type","code"}}`，不影响服务。
- **推理失败**（Engine 线程）：`try/except` 隔离整批请求，全部标记失败返回
  503、归还 KV 块与记账，**完整 traceback 进日志文件**，客户端只收到概括信息
  （不泄露内部堆栈）。OOM 会给出调参建议。
- **HTTP 内部错误**（500）：未预期异常，`log.exception` 记录，连接返回 500。
- **请求超时**（>600s）：返回 503（请求仍会算完，结果丢弃）。
- 服务本身不因单个请求失败退出；CUDA 显存异常后 `empty_cache` 相关逻辑保证
  后续请求还能继续。

---

## 7. 4GB 显卡实际运行边界（RTX 3050 实测）

| 场景 | 结果 |
|---|---|
| 单请求 prompt ≤ 1500 tokens + 512 输出 | 稳定 |
| 并发 3-4 个短 prompt（~150 tokens） | 稳定，批处理共享 decode |
| 单请求 prompt > 2048 tokens | 返回 400（`MAX_INPUT_LEN` 硬上限） |
| 高并发长 prompt | 可能 OOM，返回 503（报错不崩溃） |

显存不足时：`--kv-ratio 0.5`、调低 `--batch-size 2`、减小 `--max-tokens`。

---

## 8. 设计说明（学习用）

- `Engine` 后台单线程跑连续批处理（复用 `bench_batched.py` 骨架），
  `queue.Queue` 收请求、`Scheduler` 组批、`threading.Event` 通知 HTTP 线程
- decode 走 CUDA Graph（bucket 1/2/4），prefill 走 eager
- 每请求独立 `temperature`/`top_k`/`top_p`（批内逐请求采样）
- `KV 池`按 `--kv-ratio` × 剩余显存自适应分配，防启动 OOM
- 已知取舍：无流式 SSE（v2）、池满不排队直接 503、无 OpenAPI 文档

---

## 9. 常见问题

**Q: 中文乱码？** Windows 控制台编码问题，用 UTF-8 文件传输 JSON body。

**Q: 怎么让局域网其他机器访问？** `--host 0.0.0.0`，然后访问 `http://<你的IP>:8000`。

**Q: 想关掉 CUDA Graph？** `--no-graph` 或环境变量 `QWEN3_CUDA_GRAPH=0`（用于对比速度）。