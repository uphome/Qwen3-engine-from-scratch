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
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler

import torch
from urllib.parse import parse_qs, urlparse

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool, Scheduler
from qwen3.monitoring import snapshot as engine_snapshot
from qwen3 import load_weights_from_safetensors
from qwen3.PagedKVcache import PagedKVCache
from qwen3.graph_runner import GraphRunner
from qwen3.request import Request
from qwen3.sampling import sample_tokens
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
        self.submit_time = time.perf_counter()   # TTFT 起点
        self.ttft: float | None = None           # 首 token 延迟（秒）
        self.blocks_reserved = 0                 # Engine 记账（新块需求，命中时扣除）
        self.is_stream = False                   # 流式请求：逐 token 推送到 token_queue
        self.token_queue: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()       # 客户端断开 / 取消
        self.done = threading.Event()


class EngineMetrics:
    """进程内指标：Engine 后台线程写、/metrics HTTP 线程读

    线程安全说明：int 计数在 GIL 下读写原子；deque 快照在极端并发下
    可能轻微不一致（监控指标允许，不需要加锁）。
    """

    TTFT_WINDOW = 256      # TTFT 滑动窗口样本数（summary 分位数近似）

    def __init__(self):
        self.lock = threading.Lock()      # 保护 ttfts（跨线程读/写）
        self.start_time = time.perf_counter()
        self.requests_total = 0       # 收到的 API 请求（不含 warmup）
        self.finished_total = 0       # 成功完成
        self.failed_total = 0         # 失败（容量拒绝 / 推理异常）
        self.tokens_total = 0         # 累计生成 token 数
        self.steps_total = 0          # 调度步数
        self.last_batch_size = 0      # 最近一步批大小
        self.ttfts: deque[float] = deque(maxlen=self.TTFT_WINDOW)
        self.prefix_cache_hits_total = 0        # 前缀缓存完全命中请求数
        self.prefix_cache_hit_blocks_total = 0  # 复用的缓存块数


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
        self.metrics = EngineMetrics()
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
        # 后台线程的 CUDA current device 默认是 0：显式对齐到池所在设备，
        # 否则多卡环境下 Triton kernel launch 会按 device 0 校验指针，
        # 报 "Pointer argument ... cannot be accessed from Triton (cpu tensor?)"。
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        while True:
            # 每轮先清理客户端已断开的请求（不白算）
            self._cleanup_cancelled()
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

    def _cleanup_cancelled(self):
        """移除客户端已断开的请求：归还 KV 块 + 记账 + 通知流式端退出"""
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
                log.log(REQUEST_LEVEL, f"取消 req#{req.request_id}"
                        f"（客户端断开），已归还 KV 块")

    def _dispatch(self, job: Job):
        """入队前容量校验（含已入队未归还的记账，防批内超卖）+ 组装 Request"""
        track = job.request_id >= 0        # warmup job（-1）不计入指标
        if track:
            self.metrics.requests_total += 1
        input_len = job.input_ids.shape[1]
        needed = self._needed_blocks(input_len, job.max_new_tokens)
        # 容量预检交给 Scheduler.add（前缀命中时不需要全量新块，
        # 在调度层按"命中块数 + 已入队记账"精确判断）
        try:
            req = Request(
                request_id=job.request_id, input_ids=job.input_ids,
                max_new_tokens=job.max_new_tokens, eos_token_id=job.eos_token_id,
            )
            hit_blocks = self.scheduler.add(req, self._pending_blocks)
            # 完全命中会直接进入 decoding（单块 prompt 的复用块数可能为 0），
            # 部分命中进 waiting 且 hit_blocks > 0
            if track and (hit_blocks or req.state == "decoding"):
                self.metrics.prefix_cache_hits_total += 1
                self.metrics.prefix_cache_hit_blocks_total += hit_blocks
        except RuntimeError as e:
            if track:
                self.metrics.failed_total += 1
            job.error = f"入队失败: {e}"
            job.done.set()
            job.token_queue.put(None)   # 流式 handler 阻塞在 queue 上，需唤醒
            return
        job.req = req
        self._jobs[job.request_id] = job
        # 记账 = 总需求 - 前缀复用块（命中请求只占新块）
        job.blocks_reserved = max(0, needed - hit_blocks)
        self._pending_blocks += job.blocks_reserved
        log.log(REQUEST_LEVEL, f"入队 req#{job.request_id}: "
                 f"in={input_len}, max={job.max_new_tokens}, T={job.temperature}")

    @torch.no_grad()
    def _step(self):
        """一个调度步（与 bench_batched.py run_batched 循环同构）"""
        batch = self.scheduler.schedule()
        if batch is None:
            return
        self.metrics.steps_total += 1
        self.metrics.last_batch_size = batch.size
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
                                    input_lens=batch.input_lens,
                                    mode=batch.mode)
                last_idx = torch.tensor(batch.input_lens, dtype=torch.long,
                                        device=logits.device) - 1

            # 批量 GPU 采样（减少逐请求算子启动；参数仍随请求独立）
            batch_jobs = [self._jobs.get(req.request_id) for req in batch.requests]
            temperatures = [(j.temperature if j is not None else 0.0) for j in batch_jobs]
            top_ks = [(j.top_k if j is not None else 0) for j in batch_jobs]
            top_ps = [(j.top_p if j is not None else 1.0) for j in batch_jobs]
            batch_logits = logits[torch.arange(batch.size, device=logits.device), last_idx]
            next_tokens = sample_tokens(
                batch_logits, temperature=temperatures,
                top_k=top_ks, top_p=top_ps).long()

            # TTFT：本步将产出各请求的首 token（prefill 批；num_generated 仍为 0）
            now = time.perf_counter()
            for job, req in zip(batch_jobs, batch.requests):
                # chunked prefill 的中间 chunk 不产出首 token：
                # 只有最后一个 chunk（或 decode 批）才记录 TTFT
                is_first_token = (batch.mode == "decode"
                                  or req.prefill_end >= req.input_len)
                if (job is not None and job.request_id >= 0
                        and job.ttft is None and req.num_generated == 0
                        and is_first_token):
                    job.ttft = now - job.submit_time
                    with self.metrics.lock:
                        self.metrics.ttfts.append(job.ttft)

            # 流式推送本步新 token（on_step_done 之前/之后均可；先推送
            # 保证 HTTP 线程尽早拿到首包）
            for i, req in enumerate(batch.requests):
                job = self._jobs.get(req.request_id)
                if job is not None and job.is_stream:
                    job.token_queue.put(int(next_tokens[i].item()))

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
                    self._pending_blocks -= job.blocks_reserved
                    if job.request_id >= 0:      # warmup 不计入指标
                        self.metrics.finished_total += 1
                        self.metrics.tokens_total += len(job.output_ids)
                    if job.is_stream:
                        job.token_queue.put(None)     # 流结束哨兵
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
            self.metrics.failed_total += sum(
                1 for r in batch.requests if r.request_id >= 0)
            for req in batch.requests:
                job = self._jobs.pop(req.request_id, None)
                if job is not None:
                    self._pending_blocks -= job.blocks_reserved
                    if not job.done.is_set():
                        job.error = _describe_error(e, f"req#{req.request_id}")
                        job.done.set()
                    if job.is_stream:
                        job.token_queue.put(None)
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

def render_metrics(engine, model_id: str = "") -> str:
    """把 Engine 进程内状态渲染成 Prometheus 文本格式（text exposition 0.0.4）

    零依赖：只是拼字符串，Prometheus 每 15s 来抓一次。
    """
    m = engine.metrics
    sched = engine.scheduler
    pool = engine.pool
    used = pool.total_blocks - pool.free_count
    ratio = used / pool.total_blocks if pool.total_blocks else 0.0

    out: list[str] = []

    def emit(name: str, typ: str, help_: str, samples: list[str]):
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} {typ}")
        out.extend(samples)

    emit("engine_requests_total", "counter",
         "收到的 API 请求数（不含 warmup）",
         [f"engine_requests_total {m.requests_total}"])
    emit("engine_requests_finished_total", "counter",
         "成功完成的请求数",
         [f"engine_requests_finished_total {m.finished_total}"])
    emit("engine_requests_failed_total", "counter",
         "失败的请求数（容量拒绝 / 推理异常）",
         [f"engine_requests_failed_total {m.failed_total}"])
    emit("engine_tokens_total", "counter",
         "累计生成的 token 数",
         [f"engine_tokens_total {m.tokens_total}"])
    emit("engine_steps_total", "counter",
         "调度步数（prefill + decode，含启动 warmup）",
         [f"engine_steps_total {m.steps_total}"])
    emit("engine_running", "gauge",
         "正在 decode 的请求数",
         [f"engine_running {len(sched.running)}"])
    emit("engine_waiting", "gauge",
         "等待调度的请求数",
         [f"engine_waiting {len(sched.waiting)}"])
    emit("engine_batch_size", "gauge",
         "最近一个调度步的批大小",
         [f"engine_batch_size {m.last_batch_size}"])
    emit("engine_kv_used_ratio", "gauge",
         "KV 池物理块占用率 (0-1)",
         [f"engine_kv_used_ratio {ratio:.6f}"])
    emit("engine_kv_free_blocks", "gauge",
         "KV 池空闲物理块数",
         [f"engine_kv_free_blocks {pool.free_count}"])
    emit("engine_kv_total_blocks", "gauge",
         "KV 池物理块总数",
         [f"engine_kv_total_blocks {pool.total_blocks}"])
    emit("engine_prefix_cache_hits_total", "counter",
         "前缀缓存完全命中的请求数",
         [f"engine_prefix_cache_hits_total {m.prefix_cache_hits_total}"])
    emit("engine_prefix_cache_hit_blocks_total", "counter",
         "复用的缓存块数",
         [f"engine_prefix_cache_hit_blocks_total {m.prefix_cache_hit_blocks_total}"])
    emit("engine_prefix_cache_cached_blocks", "gauge",
         "当前可复用的缓存块数（ref=0）",
         [f"engine_prefix_cache_cached_blocks {pool.cached_count}"])
    emit("engine_prefix_cache_capacity_blocks", "gauge",
         "缓存块上限（LRU 淘汰阈值）",
         [f"engine_prefix_cache_capacity_blocks {pool.max_cached_blocks}"])
    emit("engine_prefix_cache_evictions_total", "counter",
         "容量压力下强制淘汰的缓存块数",
         [f"engine_prefix_cache_evictions_total "
          f"{getattr(pool, 'cache_evictions', 0)}"])
    emit("engine_uptime_seconds", "gauge",
         "进程运行时长（秒）",
         [f"engine_uptime_seconds {time.perf_counter() - m.start_time:.3f}"])

    # TTFT summary：分位数（滑动窗口）+ count/sum（Prometheus summary 惯例）
    ttft_samples: list[str] = []
    with m.lock:                 # 快照，避免与 Engine 线程 append 竞争
        ttft_vals = list(m.ttfts)
    if ttft_vals:
        vals = sorted(ttft_vals)
        n = len(vals)
        for q in (0.5, 0.95, 0.99):
            idx = min(n - 1, int(q * (n - 1)))
            ttft_samples.append(f'engine_ttft_seconds{{quantile="{q}"}} {vals[idx]:.6f}')
        ttft_samples.append(f"engine_ttft_seconds_count {n}")
        ttft_samples.append(f"engine_ttft_seconds_sum {sum(vals):.6f}")
    emit("engine_ttft_seconds", "summary",
         "首 token 延迟（提交 → 首个生成 token）", ttft_samples)

    if model_id:
        emit("engine_model_info", "gauge",
             "模型元信息（值恒为 1）",
             [f'engine_model_info{{model="{model_id}"}} 1'])

    return "\n".join(out) + "\n"


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

    def _send_text(self, code: int, body: str,
                   content_type: str = "text/plain; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

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
        elif path == "/metrics":
            srv = self.server
            # ?format=json：供训练框架（同机/外部）直接 json.loads；
            # 默认仍是 Prometheus 文本（向后兼容）
            query = parse_qs(urlparse(self.path).query)
            fmt = query.get("format", ["prometheus"])[0].lower()
            if fmt == "json":
                self._send_json(200, engine_snapshot(srv.engine, srv.model_id))
            else:
                self._send_text(
                    200, render_metrics(srv.engine, srv.model_id),
                    "text/plain; version=0.0.4; charset=utf-8")
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

    def _prepare_job(self, prompt_text: str, body: dict) -> tuple[Job, int]:
        """tokenize + 参数校验 + 容量预检 + 创建 Job（不提交）"""
        srv = self.server
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

        # 容量由 Engine/Scheduler 精确裁决（前缀命中时不需要全量新块，
        # HTTP 层按全量预检会误拒可命中的请求）

        job = Job(request_id=srv.engine._next_id, input_ids=input_ids,
                  max_new_tokens=max_tokens, temperature=temperature,
                  top_k=top_k, top_p=top_p)
        srv.engine._next_id += 1
        return job, input_len

    def _run_generation(self, prompt_text: str, body: dict) -> dict:
        """非流式公共生成路径：tokenize → 入队 → 等结果 → 返回 token 级数据"""
        srv = self.server
        job, input_len = self._prepare_job(prompt_text, body)

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

    # ---- SSE 流式 ----

    def _sse_start(self):
        """开始 SSE 响应。无 Content-Length → 用 close-delimited 响应
        （写完 [DONE] 后关闭连接），HTTP/1.1 合法且客户端（curl/SDK）兼容。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")   # nginx 等反代不缓冲
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse_send(self, payload):
        """发送一条 SSE 事件（dict → JSON；str 原样，如 [DONE]）"""
        text = (payload if isinstance(payload, str)
                else json.dumps(payload, ensure_ascii=False))
        self.wfile.write(f"data: {text}\n\n".encode("utf-8"))
        self.wfile.flush()      # 关键：不等缓冲区满，立刻推给客户端

    def _stream_generation(self, prompt_text: str, body: dict, chat: bool):
        """流式生成：Engine 每产出一个 token 就推一个 SSE 事件

        首 token 先等（提交失败/容量拒绝时还没发 SSE 头，可正常回 HTTP 错误），
        之后边收边发；客户端断开 → cancel，Engine 下一轮清理并归还 KV。
        """
        srv = self.server
        job, input_len = self._prepare_job(prompt_text, body)
        job.is_stream = True
        t0 = time.perf_counter()
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

        self._sse_start()
        created = int(time.time())
        chunk_id = (f"chatcmpl-{job.request_id:03d}" if chat
                    else f"cmpl-{job.request_id:03d}")

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

        if chat:
            emit({"role": "assistant", "content": ""})

        # 流式增量解码：只 decode 尚未输出的 token（通常 1 个），
        # 避免每个 token 都对累计 ids 全量 decode（O(n²)）。
        # BPE token 可能是不完整 UTF-8 字节（中文/emoji 跨 token）：
        # decode 结果含 U+FFFD 时先暂存，等后续 token 补齐字节再输出。
        pending: list[int] = []
        n_tokens = 0
        finish_reason = "stop"
        tok = first
        try:
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

            if job.error:                    # 推流中途异常：发 error 事件后收尾
                self._sse_send({"error": {"message": job.error,
                                          "type": "server_error", "code": 503}})
            else:
                finish_reason = job.finish_reason or "stop"
                emit({}, finish_reason)

            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            if include_usage:
                self._sse_send({
                    "id": chunk_id,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": created, "model": srv.model_id,
                    "choices": [],
                    "usage": {"prompt_tokens": input_len,
                              "completion_tokens": n_tokens,
                              "total_tokens": input_len + n_tokens},
                })
            self._sse_send("[DONE]")

            elapsed = time.perf_counter() - t0
            log.log(REQUEST_LEVEL, f"HTTP stream req#{job.request_id}: "
                     f"in={input_len}, gen={n_tokens}, {elapsed:.1f}s, "
                     f"reason={finish_reason}")
        except (BrokenPipeError, ConnectionResetError):
            job.cancelled.set()
            log.log(REQUEST_LEVEL, f"客户端断开 req#{job.request_id}，取消生成")
        except queue.Empty:
            job.cancelled.set()
            log.warning(f"流式生成超时 req#{job.request_id}（已取消）")

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

        if body.get("stream"):
            self._stream_generation(prompt_text, body, chat=True)
            return

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
        if body.get("stream"):
            self._stream_generation(prompt, body, chat=False)
            return
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
    if device.type == "cuda":
        torch.cuda.set_device(device)   # Triton kernel / 图捕获按当前 device 校验指针
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
                   temperature=0.7, top_k=50, top_p=0.9, eos_token_id=-1)
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
    log.info("  GET  /metrics              (Prometheus 指标)")
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
