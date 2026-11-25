"""HTTP 服务 SSE 端到端：流式一致性 / chunk 格式 / 客户端断开取消

需要真实模型（默认 /data/hjt1/Qwen3-0.6B，可用 QWEN3_TEST_MODEL 覆盖）。
标记 gpu + slow：CI 跳过，本地全量 pytest / -m "gpu and slow" 运行。
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("QWEN3_TEST_MODEL", "/data/hjt1/Qwen3-0.6B")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    if not os.path.isdir(MODEL):
        pytest.skip(f"真实模型不存在: {MODEL}（设置 QWEN3_TEST_MODEL 指向模型目录）")

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--model", MODEL, "--port", str(port),
         "--batch-size", "2", "--max-tokens", "32", "--no-log-file"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("server 进程提前退出")
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2).read()
                break
            except Exception:
                time.sleep(1)
        else:
            pytest.fail("server 启动超时（180s）")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _post_json(url: str, body: dict, timeout: float = 60):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _sse_events(url: str, body: dict):
    """逐条读取 SSE 事件，返回 (文本拼接, 事件列表, 是否收到 [DONE])"""
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
            if obj.get("choices"):      # usage chunk 的 choices 为空
                text += obj["choices"][0]["delta"].get("content", "")
    return text, events, done


def _plain_chat(url: str, body: dict) -> str:
    with _post_json(url + "/v1/chat/completions", body) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


MESSAGES = [{"role": "user", "content": "用一句话介绍你自己"}]


def test_stream_matches_non_stream(server_url):
    """同参数下流式拼接文本 == 非流式返回"""
    stream_text, _, done = _sse_events(
        server_url, {"messages": MESSAGES, "max_tokens": 16,
                     "temperature": 0, "stream": True})
    normal_text = _plain_chat(
        server_url, {"messages": MESSAGES, "max_tokens": 16, "temperature": 0})

    assert done
    assert stream_text == normal_text
    assert "\ufffd" not in stream_text, "增量解码出现替换字符（UTF-8 被拆坏）"


def test_stream_chunk_format(server_url):
    """OpenAI chat.completion.chunk 结构 + finish_reason + usage chunk"""
    _, events, done = _sse_events(
        server_url, {"messages": MESSAGES, "max_tokens": 8, "temperature": 0,
                     "stream": True,
                     "stream_options": {"include_usage": True}})

    assert done
    assert events[0]["object"] == "chat.completion.chunk"
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"
    assert all(e["object"] == "chat.completion.chunk" for e in events)

    # 倒数第二个是 usage chunk（choices 为空），之前是 finish_reason chunk
    usage = events[-1]
    assert usage["choices"] == []
    assert usage["usage"]["completion_tokens"] > 0
    assert usage["usage"]["total_tokens"] == (
        usage["usage"]["prompt_tokens"] + usage["usage"]["completion_tokens"])

    finish = events[-2]["choices"][0]
    assert finish["finish_reason"] in ("stop", "length")


def test_client_disconnect_cancels_generation(server_url):
    """客户端中途断开 → 引擎取消请求、归还 KV、队列清空"""
    port = int(server_url.rsplit(":", 1)[1])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    body = json.dumps({
        "messages": [{"role": "user", "content": "写一篇很长的文章"}],
        "max_tokens": 512, "temperature": 0.7, "stream": True,
    })
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200

    # 读前几条事件后立刻断开（模拟用户关闭页面）
    for _ in range(3):
        if not resp.readline():
            break
    conn.close()

    # 等待引擎清理（客户端断开 → BrokenPipe → cancelled → 下一轮 cleanup）
    deadline = time.time() + 15
    metrics = ""
    while time.time() < deadline:
        with urllib.request.urlopen(server_url + "/metrics", timeout=5) as r:
            metrics = r.read().decode()
        vals = {}
        for line in metrics.splitlines():
            if line.startswith(("engine_running ", "engine_waiting ",
                                "engine_kv_used_ratio ")):
                name, v = line.split()
                vals[name] = float(v)
        if vals.get("engine_running") == 0 and vals.get("engine_waiting") == 0:
            break
        time.sleep(0.5)

    assert vals["engine_running"] == 0
    assert vals["engine_waiting"] == 0
    # KV 占用回到图预留的基线（< 5%），说明被取消请求的块已归还
    assert vals["engine_kv_used_ratio"] < 0.05
