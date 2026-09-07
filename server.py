"""
Qwen3 推理引擎 HTTP API 服务 — OpenAI 兼容（纯标准库，零新依赖）

架构:
  - Engine:      后台单线程调度循环（复用 bench_batched.py 的连续批处理骨架）
                 queue.Queue 收作业，Scheduler 组批，逐请求采样，
                 完成后置 threading.Event 通知 HTTP 线程
  - HTTP 层:     ThreadingHTTPServer + BaseHTTPRequestHandler
                 GET  /v1/models
                 POST /v1/chat/completions   （OpenAI chat 格式）
                 POST /v1/completions        （纯 prompt 格式）

用法:
    python server.py --model E:\\codeall\\Qwen3-0.6B
    curl http://127.0.0.1:8000/v1/models
    curl -X POST http://127.0.0.1:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{"messages":[{"role":"user","content":"你好"}]}'

说明:
  - 池满时新请求返回 503（不做排队），stream 流式暂不支持（v2）
  - 每请求独立 temperature/top_k/top_p；decode 默认走 CUDA Graph
"""

import argparse
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler

import torch
import torch.nn.functional as F

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool, Scheduler
from qwen3 import load_weights_from_safetensors
from qwen3.PagedKVcache import PagedKVCache
from qwen3.graph_runner import GraphRunner
from qwen3.request import Request
from chat_template import format_chat

EOS_TOKEN_ID = 151645
MAX_INPUT_LEN = 2048        # 池容量假设的输入余量（prompt 上限）
BLOCK_SIZE = 16
WAIT_TIMEOUT = 600          # HTTP 线程等结果的超时（秒）

# Qwen3 tool calling：模型输出 <tool_call>{"name":..., "arguments":{...}}</tool_call>
# （官方模板同款；正则非贪婪 + re.S 匹配多行 JSON）
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

# ============================================================
# 日志系统：logger.* 替代 print（统一级别/时间戳/线程名/落盘）
# ============================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)-18s | %(message)s"
LOG_DATE = "%H:%M:%S"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# 每请求一条的访问日志级别（介于 INFO/WARNING 之间，方便单独查看/过滤）
REQUEST_LEVEL = 25
logging.addLevelName(REQUEST_LEVEL, "REQUEST")


def setup_logging(verbosity: int = 0, log_file: bool = True) -> logging.Logger:
    """初始化根日志：控制台 + 可选滚动文件（UTF-8，Windows 中文不乱码）。

    级别：默认 INFO；--verbose 传 1 → DEBUG；0 且非 TTY → WARNING（安静）。
    """
    level = logging.DEBUG if verbosity else logging.INFO
    if verbosity == 0 and not sys.stderr.isatty():
        level = logging.WARNING

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)                    # handler 级别各自再控制
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(LOG_FORMAT, LOG_DATE)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(LOG_DIR, "server.log"),
            maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger("qwen3.server")


# 模块级 logger（其他 handler 用），main() 里 setup_logging 后再用
log = logging.getLogger("qwen3.server")


def resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return torch.device(f"cuda:{free.index(max(free))}")
    return torch.device("cpu")


def kv_bytes_per_block(config, block_size: int) -> int:
    """单个物理块的显存字节数（bf16 K + V）"""
    return (config.num_hidden_layers * config.num_key_value_heads
            * block_size * config.head_dim * 2 * 2)


def adaptive_num_blocks(requested: int, config, block_size: int,
                        device, ratio: float = 0.9) -> int:
    """显存自适应裁剪：4GB 小卡上池子建太大直接 OOM（k_buffer 一次性分配）。

    ratio 留 10% 给 decode 激活 / CUDA Graph 工作区。
    """
    if device.type != "cuda":
        return requested
    torch.cuda.empty_cache()
    free_bytes = torch.cuda.mem_get_info(device)[0]
    capacity = int(free_bytes * ratio / kv_bytes_per_block(config, block_size))
    if requested > capacity:
        log.warning("显存自适应: 请求 %d 块, 剩余显存仅够 %d 块, 已裁剪", requested, capacity)
        return max(1, capacity)
    return requested


def sample_one(logits: torch.Tensor, temperature: float,
               top_k: int, top_p: float) -> torch.Tensor:
    """单请求采样（generate.py 的采样逻辑抽成函数，参数随请求独立）"""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        topk_vals = torch.topk(logits, min(top_k, logits.size(-1))).values
        logits = logits.masked_fill(logits < topk_vals[..., -1], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        remove = mask.scatter(-1, sorted_idx, mask)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ============================================================
# Engine — 后台调度线程
# ============================================================

class Job:
    """一个 API 请求的完整状态：HTTP 线程提交，Engine 线程消费"""

    def __init__(self, request_id: int, input_ids: torch.Tensor,
                 max_new_tokens: int, temperature: float,
                 top_k: int, top_p: float, eos_token_id: int = EOS_TOKEN_ID):
        self.request_id = request_id
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id
        self.req: Request | None = None    # Scheduler.add 时注入
        self.output_ids: list[int] = []
        self.finish_reason: str = ""
        self.error: str | None = None
        self.done = threading.Event()


class Engine:
    """后台线程：queue 收作业 → Scheduler 连续批处理 → 逐请求采样 → 通知完成

    调度循环 = bench_batched.py 的 run_batched 循环（schedule → 前向 →
    采样 → on_step_done）改常驻版：无活可干时阻塞在 q.get()（无忙轮询），
    有请求就不断调度直到清空。
    """

    def __init__(self, model, pool, runner, batch_size: int):
        self.model = model
        self.pool = pool
        self.runner = runner            # GraphRunner | None（decode 走图）
        self.scheduler = Scheduler(pool, batch_size=batch_size)
        self.device = pool.k_buffer.device
        self.q = queue.Queue()
        self._jobs: dict[int, Job] = {}     # request_id → job（finished 后移除）
        self._pending_blocks = 0            # 已入队未归还的块数记账
        self._next_id = 0
        self._thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        """哨兵退出（HTTP 线程已在收尾时才调）"""
        self.q.put(None)

    def submit(self, job: Job):
        self.q.put(job)

    # ---- 后台循环 ----

    def run(self):
        while True:
            # 无活可干 → 阻塞等新作业（避免忙轮询烧 CPU/GPU）
            if not self.scheduler.has_pending():
                job = self.q.get()
                if job is None:
                    return
                self._dispatch(job)
            # drain 队列里积压的所有作业
            while True:
                try:
                    job = self.q.get_nowait()
                except queue.Empty:
                    break
                if job is None:
                    return
                self._dispatch(job)
            self._step()

    def _needed_blocks(self, input_len: int, max_new_tokens: int) -> int:
        bs = self.pool.block_size
        return (input_len + max_new_tokens + bs - 1) // bs

    def _dispatch(self, job: Job):
        """入队前容量校验（含已入队未归还的记账，防批内超卖）+ 组装 Request"""
        input_len = job.input_ids.shape[1]
        needed = self._needed_blocks(input_len, job.max_new_tokens)
        available = self.pool.free_count - self._pending_blocks
        if needed > available:
            job.error = (
                f"池容量不足: 剩余 {self.pool.free_count} 块, 已入队占用 "
                f"{self._pending_blocks} 块, 请求需要 {needed} 块"
                f"（当前单请求约可用 {available * self.pool.block_size} tokens）")
            job.done.set()
            return
        try:
            req = Request(
                request_id=job.request_id, input_ids=job.input_ids,
                max_new_tokens=job.max_new_tokens, eos_token_id=job.eos_token_id,
            )
            self.scheduler.add(req)      # 分配 PagedKVCache 行槽位
        except RuntimeError as e:
            job.error = f"入队失败: {e}"
            job.done.set()
            return
        job.req = req
        self._jobs[job.request_id] = job
        self._pending_blocks += needed
        log.log(REQUEST_LEVEL, f"入队 req#{job.request_id}: "
                 f"in={input_len}, max={job.max_new_tokens}, T={job.temperature}")

    @torch.no_grad()
    def _step(self):
        """一个调度步（与 bench_batched.py run_batched 循环同构）"""
        batch = self.scheduler.schedule()
        if batch is None:
            return
        try:
            if batch.mode == "decode" and self.runner is not None:
                # ---- decode 走 CUDA Graph：replay 替代 model 前向 ----
                caches = batch.build_kv_caches()
                for c in caches:
                    c.reserve_next()              # 页预留（原 forward 内部做）
                start_pos = torch.tensor([c.seq_len for c in caches],
                                         dtype=torch.long, device=self.device)
                positions = start_pos.unsqueeze(1)                  # (k, 1)
                rows = torch.tensor([c.row_id for c in caches],
                                    dtype=torch.int32, device=self.device)
                logits = self.runner.replay(
                    batch.build_input_ids(), positions, rows)
                logits = logits[:batch.size]      # (b,1,vocab) → 前 k 行有效
                for c in caches:
                    c.advance_seq_len(1)          # 状态推进（原 forward 内部做）
                last_idx = torch.zeros(batch.size, dtype=torch.long,
                                       device=logits.device)
            else:
                # ---- prefill / 无图 eager ----
                logits = self.model(batch.build_input_ids(),
                                    kv_cache=batch.build_kv_caches(),
                                    input_lens=batch.input_lens)
                last_idx = torch.tensor(batch.input_lens, dtype=torch.long,
                                        device=logits.device) - 1

            # 逐请求采样（参数随请求独立；B ≤ batch_size，开销可忽略）
            next_tokens = torch.zeros(batch.size, 1, dtype=torch.long,
                                      device=logits.device)
            for i, req in enumerate(batch.requests):
                job = self._jobs.get(req.request_id)
                if job is None:                   # 理论不可达，防御
                    next_tokens[i, 0] = logits[i, last_idx[i]].argmax()
                    continue
                next_tokens[i, 0] = sample_one(
                    logits[i, last_idx[i]], job.temperature, job.top_k, job.top_p)

            self.scheduler.on_step_done(batch, next_tokens)

            # 完成者：收集输出 + 归还记账 + 通知 HTTP 线程
            for req in batch.requests:
                if req.state == "finished":
                    # 从 scheduler.finished 中移除，避免长期服务中
                    # 已结束 Request 的 input_ids 等 GPU 张量被无限保留导致显存上涨
                    if req in self.scheduler.finished:
                        self.scheduler.finished.remove(req)
                    job = self._jobs.pop(req.request_id, None)
                    if job is None:
                        continue
                    job.output_ids = list(req.output_ids)
                    job.finish_reason = ("length" if req.num_generated
                                         >= req.max_new_tokens else "stop")
                    self._pending_blocks -= self._needed_blocks(
                        req.input_len, req.max_new_tokens)
                    log.log(REQUEST_LEVEL, f"完成 req#{req.request_id}: "
                             f"gen={req.num_generated}, reason={job.finish_reason}, "
                             f"pool_free={self.pool.free_count}")
                    job.done.set()
        except Exception as e:
            # 异常隔离：批内所有请求置错 + 归还记账 + 归还资源，服务不崩。
            # 完整 traceback 进日志文件；客户端只收到一条概括性错误。
            job_ids = [r.request_id for r in batch.requests]
            log.exception(f"推理步骤失败 (batch={batch.mode}, "
                          f"reqs={job_ids}): {e}")
            for req in batch.requests:
                job = self._jobs.pop(req.request_id, None)
                if job is not None:
                    self._pending_blocks -= self._needed_blocks(
                        req.input_len, req.max_new_tokens)
                    if not job.done.is_set():
                        job.error = _describe_error(e, f"req#{req.request_id}")
                        job.done.set()
                if req.kv_cache is not None:      # 归还资源（跳过状态机）
                    req.kv_cache.free()
                    req.kv_cache = None
                if req in self.scheduler.running:
                    self.scheduler.running.remove(req)


def _describe_error(e: Exception, context: str) -> str:
    """把底层异常翻译成对 API 调用方可读的消息（不泄露内部 traceback）"""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return (f"{context}: 显存不足 (OOM)。请调小 --max-tokens / --batch-size "
                f"或降低 --kv-ratio，稍后重试")
    return f"{context}: 推理异常: {type(e).__name__}: {e}"


def _extract_tool_calls(text: str) -> list[dict] | None:
    """从生成文本提取 <tool_call> JSON 块 → OpenAI tool_calls 格式

    模型的工具调用格式（官方模板约定）:
        <tool_call>
        {"name": "get_weather", "arguments": {"city": "Beijing"}}
        </tool_call>
    返回 None（无调用）或形如:
        [{"id": "call_0000", "type": "function",
          "function": {"name": ..., "arguments": "<json字符串>"}}]
    """
    calls = []
    for i, m in enumerate(TOOL_CALL_RE.finditer(text)):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments", {})
            if not isinstance(name, str):
                continue
            calls.append({
                "id": f"call_{i:04d}",
                "type": "function",
                "function": {"name": name,
                             "arguments": json.dumps(args, ensure_ascii=False)},
            })
        except json.JSONDecodeError:
            log.warning("tool_call JSON 解析失败: %s...", m.group(1)[:120])
    return calls or None


# ============================================================
# HTTP 层
# ============================================================

class OpenAIError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "Qwen3Engine/0.1"
    protocol_version = "HTTP/1.1"

    # ---- 响应工具 ----

    def log_request(self, code: int, size: int = -1):
        """每请求一行访问日志（覆盖默认：时间/级别由 logger 统一管理）"""
        log.log(REQUEST_LEVEL, "%s %s -> %d",
                self.command, self.path.split("?")[0], code)

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, etype: str = "invalid_request_error"):
        self._send_json(code, {"error": {"message": message, "type": etype,
                                         "code": code}})

    # ---- 路由 ----

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            srv = self.server
            self._send_json(200, {
                "object": "list",
                "data": [{"id": srv.model_id, "object": "model",
                          "created": srv.created, "owned_by": "qwen3-engine"}],
            })
        else:
            self._error(404, f"未找到 {self.path}", "not_found")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._error(404, f"未找到 {self.path}", "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            log.log(REQUEST_LEVEL, f"{path} 请求体不是合法 JSON")
            self._error(400, "请求体不是合法 JSON")
            return
        try:
            if path == "/v1/chat/completions":
                self._handle_chat(body)
            else:
                self._handle_completion(body)
        except OpenAIError as e:
            log.log(REQUEST_LEVEL, f"{path} 业务错误: HTTP {e.code}: {e}")
            self._error(e.code, str(e))
        except Exception as e:
            log.exception(f"{path} 未预期异常: {e}")
            self._error(500, f"服务器内部错误: {type(e).__name__}: {e}",
                        "internal_error")

    # ---- 核心处理 ----

    def _run_generation(self, prompt_text: str, body: dict) -> dict:
        """公共生成路径：prompt 文本 → tokenize → 入队 → 等结果 → 返回 token 级数据"""
        srv = self.server
        if body.get("stream"):
            raise OpenAIError(400, "stream 流式输出暂不支持（v2）")

        input_ids = torch.tensor([srv.tokenizer.encode(prompt_text)],
                                 device=srv.device)
        input_len = input_ids.shape[1]
        if input_len > MAX_INPUT_LEN:
            raise OpenAIError(400, f"prompt 过长: {input_len} tokens > {MAX_INPUT_LEN}")

        max_tokens = min(int(body.get("max_tokens", srv.max_tokens)),
                         srv.max_tokens)
        temperature = float(body.get("temperature", 0.7))
        top_k = int(body.get("top_k", 50))
        top_p = float(body.get("top_p", 0.9))
        if temperature < 0:
            raise OpenAIError(400, "temperature 不能为负")

        # 容量预检（粗略；精确裁决在 Engine._dispatch 的记账里）
        needed = (input_len + max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        if needed > srv.pool.free_count:
            raise OpenAIError(
                503, f"池剩余 {srv.pool.free_count} 块, 请求需要 {needed} 块; "
                     f"请调小 max_tokens（当前约可用 {srv.pool.free_count * BLOCK_SIZE} tokens）")

        job = Job(request_id=srv.engine._next_id, input_ids=input_ids,
                  max_new_tokens=max_tokens, temperature=temperature,
                  top_k=top_k, top_p=top_p)
        srv.engine._next_id += 1
        t0 = time.perf_counter()
        srv.engine.submit(job)
        if not job.done.wait(WAIT_TIMEOUT):
            raise OpenAIError(503, "生成超时（请求仍在后台继续，结果丢弃）")
        if job.error:
            raise OpenAIError(503, job.error)

        elapsed = time.perf_counter() - t0
        log.log(REQUEST_LEVEL, f"HTTP req#{job.request_id}: in={input_len}, "
                 f"gen={len(job.output_ids)}, {elapsed:.1f}s, "
                 f"reason={job.finish_reason}")
        return {"request_id": job.request_id, "input_len": input_len,
                "output_ids": job.output_ids,
                "finish_reason": job.finish_reason}

    def _handle_chat(self, body: dict):
        srv = self.server
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIError(400, "缺少 messages")
        # enable_thinking: 默认关（空思考块）。true 则走原生思考链。
        enable_thinking = bool(body.get("enable_thinking", False))
        tools = body.get("tools")

        if tools:
            # tool calling 路径：复用模型官方 Jinja 模板（处理 tools 渲染、
            # tool_response 合并、历史 tool_call 恢复），保证与训练格式一致
            try:
                prompt_text = srv.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    tools=tools, enable_thinking=enable_thinking)
            except Exception as e:
                raise OpenAIError(400, f"tools 渲染失败: {e}")
        else:
            try:
                prompt_text = format_chat(messages,
                                          enable_thinking=enable_thinking)
            except KeyError as e:
                raise OpenAIError(400, f"messages 格式错误: {e}")

        result = self._run_generation(prompt_text, body)

        output_text = srv.tokenizer.decode(result["output_ids"],
                                           skip_special_tokens=True)
        n = len(result["output_ids"])

        # 提取 <tool_call> 块 → OpenAI tool_calls 格式；正文剔除该块
        tool_calls = _extract_tool_calls(output_text)
        if tool_calls:
            content = TOOL_CALL_RE.sub("", output_text).strip()
            message = {"role": "assistant", "content": content or None,
                       "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": output_text}
            finish_reason = result["finish_reason"]

        self._send_json(200, {
            "id": f"chatcmpl-{result['request_id']:03d}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": srv.model_id,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": result["input_len"],
                      "completion_tokens": n,
                      "total_tokens": result["input_len"] + n},
        })

    def _handle_completion(self, body: dict):
        srv = self.server
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise OpenAIError(400, "缺少 prompt（字符串）")
        result = self._run_generation(prompt, body)

        output_text = srv.tokenizer.decode(result["output_ids"],
                                           skip_special_tokens=True)
        n = len(result["output_ids"])
        self._send_json(200, {
            "id": f"cmpl-{result['request_id']:03d}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": srv.model_id,
            "choices": [{"index": 0, "text": output_text,
                         "finish_reason": result["finish_reason"]}],
            "usage": {"prompt_tokens": result["input_len"],
                      "completion_tokens": n,
                      "total_tokens": result["input_len"] + n},
        })


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3 HTTP API 服务 (OpenAI 兼容)")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace 模型名或本地路径")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="连续批处理最大批大小 (4GB 卡建议 ≤4)")
    parser.add_argument("--kv-ratio", type=float, default=0.6,
                        help="KV 池占用剩余显存的比例 (0.6 = 留 40% 给 "
                             "prefill/decode 瞬时激活, 4GB 卡 OOM 就调低)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="单请求生成上限 (默认: 512)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-graph", action="store_true",
                        help="禁用 CUDA Graph decode（默认开，QWEN3_CUDA_GRAPH=0 亦可）")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="日志级别加强 (-v = DEBUG；默认 INFO)")
    parser.add_argument("--no-log-file", action="store_true",
                        help="只打印控制台，不写 logs/server.log")
    args = parser.parse_args()

    setup_logging(args.verbose, log_file=not args.no_log_file)

    # ---- 设备 / 模型 ----
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    log.info("=" * 60)
    log.info("Qwen3 HTTP API Server %s OpenAI 兼容", "—")
    log.info("=" * 60)
    log.info("Device: %s, dtype: %s", device, dtype)

    config = Qwen3Config.from_pretrained(args.model)
    model = Qwen3ForCausalLM(config)
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()
    log.info("Model: %s", args.model)
    log.info("  Layers: %d, Hidden: %d, Heads: %d, KV Heads: %d",
             config.num_hidden_layers, config.hidden_size,
             config.num_attention_heads, config.num_key_value_heads)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- CUDA Graph（decode 图池，buckets 2 的幂）----
    use_graph = (
        device.type == "cuda" and not args.no_graph
        and os.environ.get("QWEN3_CUDA_GRAPH", "1") != "0"
    )
    graph_buckets = [1]
    if use_graph:
        while graph_buckets[-1] < args.batch_size:
            graph_buckets.append(graph_buckets[-1] * 2)

    # ---- KV 池：容量 = 每请求块数 × 批大小 + 图预留；显存不足自动裁剪 ----
    blocks_per_seq = (MAX_INPUT_LEN + args.max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = blocks_per_seq * args.batch_size
    if use_graph:
        num_blocks += max(graph_buckets) + 1        # 占位句柄 + 哑行
    num_blocks = adaptive_num_blocks(num_blocks, config, BLOCK_SIZE, device,
                                     ratio=args.kv_ratio)
    kv_pool = KVCachePool(
        num_blocks=num_blocks, num_layers=config.num_hidden_layers,
        block_size=BLOCK_SIZE, num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim, device=device, dtype=dtype,
        max_seq_len=MAX_INPUT_LEN + args.max_tokens,
    )
    log.info("  KV pool: %d 块 x %d tokens (>= %d 块/请求 x %d 批) = %d tokens 容量, 每块 %.2f MiB",
             num_blocks, BLOCK_SIZE, blocks_per_seq, args.batch_size,
             num_blocks * BLOCK_SIZE,
             kv_bytes_per_block(config, BLOCK_SIZE) / 1024 ** 2)

    runner = None
    if use_graph:
        occupy = [PagedKVCache(kv_pool) for _ in range(max(graph_buckets))]
        runner = GraphRunner(model, kv_pool, occupy,
                             buckets=tuple(graph_buckets), reserve_pages=1)
        log.info("  CUDA Graph: 开启 (buckets=%s, 预留 %d 块; --no-graph 关闭)",
                 graph_buckets, max(graph_buckets) + 1)

    # ---- Engine 后台线程 ----
    engine = Engine(model, kv_pool, runner, batch_size=args.batch_size)
    engine.start()

    # ---- 预热（kernel 编译 + 图捕获已在构造时完成；再走一遍完整调度路径）----
    warm_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    warm_job = Job(request_id=-1, input_ids=warm_ids, max_new_tokens=8,
                   temperature=0, top_k=50, top_p=0.9, eos_token_id=-1)
    engine.submit(warm_job)
    if warm_job.done.wait(120):
        log.info("  Warmup: 完成 (32 in / 8 out)")
    else:
        log.warning("  Warmup 超时，继续启动")

    # ---- HTTP 服务 ----
    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    server.engine = engine
    server.pool = kv_pool
    server.tokenizer = tokenizer
    server.model_id = os.path.basename(os.path.normpath(args.model)).lower()
    server.created = int(time.time())
    server.max_tokens = args.max_tokens
    server.device = device

    log.info("")
    log.info("Listening on http://%s:%d", args.host, args.port)
    log.info("  GET  /v1/models")
    log.info("  POST /v1/chat/completions   (OpenAI chat 格式)")
    log.info("  POST /v1/completions        (纯 prompt 格式)")
    log.info("  Ctrl+C 退出")
    log.info("=" * 60)
    log.info("[--verbose 打开 DEBUG; 日志文件: %s]",
             os.path.join(LOG_DIR, "server.log"))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("关闭中 ...")
    finally:
        server.shutdown()
        engine.stop()


if __name__ == "__main__":
    main()
