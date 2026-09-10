"""内置 LLM 链路真实 HTTP 端到端测试 (Opt-LLM-01)。

与单元测试的区别: 用真实 TCP 端口上的 OpenAI 兼容 mock 服务器
(ThreadingHTTPServer) 走完整 ``httpx → HTTP → JSON/SSE 解析 → usage 记账``
链路, 而非 mock httpx 客户端。覆盖:

- 成功路径: ``llm_enhance`` 摘要被改写 + token usage 记账;
- 流式路径: ``llm_enhance_stream`` SSE 分片拼接 + usage 记账;
- 降级路径: HTTP 500 / 连接拒绝 → 静默回退规则摘要;
- 门控: openai 后端无 key 时完全不发请求;
- 引擎集成: ``FeedbackEngine`` preload → run() 全链路 (含 _llm_loaded 门控)。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from f1opt.config import Settings
from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.feedback import engine as fb_engine
from f1opt.feedback.engine import FeedbackEngine, llm_enhance, llm_enhance_stream
from f1opt.feedback.engine import TokenUsageTracker

MARKER = "[MOCK-LLM-E2E]"
CONTENT = MARKER + " 推头对策: 前翼-1, 后翼+1, 刹车点后移 5 米。"
USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


class _MockHandler(BaseHTTPRequestHandler):
    """可编程 OpenAI 兼容 mock: mode ∈ {ok, sse, error_500}。"""

    server_version = "MockOllama/1.0"

    def log_message(self, *args) -> None:  # 静默
        pass

    def do_GET(self) -> None:  # noqa: N802
        # preload_llm 可达性探测 (Opt-LLM-02): Ollama 原生 /api/tags 与
        # OpenAI 兼容 /v1/models 均须可用。
        if self.path in ("/api/tags", "/v1/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": "mock-llm", "object": "model", "owned_by": "mock"}],
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        self.server.requests += 1
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "bad json"})
            return
        mode = self.server.mode
        if mode == "error_500":
            self._send(500, {"error": "mock internal error"})
            return
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for part in [MARKER, " 推头对策:", " 前翼-1, 后翼+1。"]:
                chunk = {
                    "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            final = {"choices": [], "usage": USAGE}
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._send(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": CONTENT}}],
                "usage": USAGE,
            },
            ctype="application/json",
        )

    def _send(self, code: int, obj: dict, ctype: str = "application/json") -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _MockHandler)
        self.requests = 0
        self.mode = "ok"


@pytest.fixture()
def mock_server():
    srv = _MockServer()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def local_endpoint(monkeypatch, mock_server):
    """把 local 后端指到临时 mock 端口, 返回 (server, base_url)。"""
    url = f"http://127.0.0.1:{mock_server.server_address[1]}/v1/chat/completions"
    monkeypatch.setitem(fb_engine._LLM_ENDPOINTS, "local", url)
    return mock_server, url


def _local_settings() -> Settings:
    return Settings(
        llm_backend="local", llm_api_key="", llm_model="mock-llm", llm_reflection=False
    )


def _base_feedback() -> dict:
    return {"summary": "规则引擎摘要", "dimensions": [], "sources": []}


class TestLlmEnhanceLive:
    def test_success_rewrites_summary_and_records_usage(self, local_endpoint) -> None:
        srv, _ = local_endpoint
        tk = TokenUsageTracker()
        out = llm_enhance(_base_feedback(), "为什么推头？", _local_settings(), tracker=tk)
        assert out["summary"].startswith(MARKER)
        assert out["dimensions"] == []  # 结构化维度永不被 LLM 改动
        assert srv.requests == 1
        per = tk.per_backend()
        assert per.get("local", {}).get("successful_calls", 0) == 1

    def test_http_500_falls_back_silently(self, local_endpoint) -> None:
        srv, _ = local_endpoint
        srv.mode = "error_500"
        fb = _base_feedback()
        out = llm_enhance(fb, "q", _local_settings(), tracker=TokenUsageTracker())
        assert out["summary"] == "规则引擎摘要"

    def test_connection_refused_falls_back_silently(self, monkeypatch) -> None:
        # 指向必然无人监听的回环端口。
        monkeypatch.setitem(
            fb_engine._LLM_ENDPOINTS, "local", "http://127.0.0.1:1/v1/chat/completions"
        )
        out = llm_enhance(_base_feedback(), "q", _local_settings(), tracker=TokenUsageTracker())
        assert out["summary"] == "规则引擎摘要"

    def test_openai_without_key_never_calls(self, local_endpoint) -> None:
        srv, _ = local_endpoint
        cfg = Settings(
            llm_backend="openai", llm_api_key="", llm_model="gpt-4o-mini",
            llm_reflection=False,
        )
        out = llm_enhance(_base_feedback(), "q", cfg, tracker=TokenUsageTracker())
        assert out["summary"] == "规则引擎摘要"
        assert srv.requests == 0  # 完全不发起请求

    def test_backend_none_gated(self, local_endpoint) -> None:
        srv, _ = local_endpoint
        cfg = Settings(llm_backend="none", llm_reflection=False)
        out = llm_enhance(_base_feedback(), "q", cfg, tracker=TokenUsageTracker())
        assert out["summary"] == "规则引擎摘要"
        assert srv.requests == 0


class TestLlmStreamLive:
    def test_stream_yields_deltas_and_usage(self, local_endpoint) -> None:
        srv, _ = local_endpoint
        tk = TokenUsageTracker()
        cfg = _local_settings()
        deltas = list(llm_enhance_stream(_base_feedback(), "q", cfg, tracker=tk))
        assert deltas, "流式未产出任何分片"
        assert "".join(deltas).startswith(MARKER)
        assert srv.requests == 1
        per = tk.per_backend()
        assert per.get("local", {}).get("streamed_calls", 0) == 1

    def test_stream_error_yields_nothing(self, local_endpoint) -> None:
        srv, _ = local_endpoint
        srv.mode = "error_500"
        deltas = list(llm_enhance_stream(_base_feedback(), "q", _local_settings()))
        assert deltas == []


class TestLlmHeaders:
    """Opt-LLM-03 回归: 空 key 不得产生 ``Bearer `` (httpx 拒收的非法头)。"""

    def test_empty_key_omits_authorization(self) -> None:
        h = fb_engine._llm_headers("")
        assert "Authorization" not in h
        assert h["Content-Type"] == "application/json"

    def test_whitespace_key_omits_authorization(self) -> None:
        assert "Authorization" not in fb_engine._llm_headers("   ")

    def test_nonempty_key_sends_bearer(self) -> None:
        h = fb_engine._llm_headers("sk-test")
        assert h["Authorization"] == "Bearer sk-test"


class TestLlmChainDiagnosis:
    """Opt-LLM-01: 逐段复现 llm_enhance 内部步骤, 失败时精确定位故障段。

    llm_enhance 的静默降级会吞掉异常; 本测试把每一步拆开断言,
    云端失败时可直接读出是 哪一环 (配置 / 端点解析 / HTTP / 解析)。
    """

    def test_stepwise_chain(self, local_endpoint) -> None:
        srv, url = local_endpoint
        cfg = _local_settings()
        # 1. 配置
        assert cfg.llm_backend == "local", f"backend={cfg.llm_backend!r}"
        assert cfg.llm_model == "mock-llm"
        # 2. 端点解析 (与 llm_enhance 同一来源)
        endpoint = fb_engine._LLM_ENDPOINTS.get(cfg.llm_backend)
        assert endpoint == url, f"endpoint={endpoint!r}"
        # 3. 原始 HTTP POST (与 llm_enhance 同构)
        import httpx

        payload = {
            "model": cfg.llm_model,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
            ],
            "temperature": 0.3,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.post(
                    endpoint,
                    # 与生产一致: 走 _llm_headers (Opt-LLM-03 修复点)。
                    headers=fb_engine._llm_headers(cfg.llm_api_key),
                    json=payload,
                )
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"STEP3 httpx POST raised {type(exc).__name__}: {exc}")
        assert r.status_code == 200, f"STEP3 http {r.status_code}: {r.text[:200]}"
        # 4. 响应解析
        try:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"STEP4 parse raised {type(exc).__name__}: {exc}")
        assert content.startswith(MARKER), f"STEP4 content={content[:80]!r}"
        # 5. tracker 记账
        tk = TokenUsageTracker()
        tk.record(cfg.llm_backend, cfg.llm_model, fb_engine._extract_usage(data), success=True, streamed=False)
        assert tk.per_backend().get("local", {}).get("successful_calls") == 1
        assert srv.requests >= 1
        # 6. 完整 llm_enhance (若前 5 步全过而它仍回退, 逐行读 engine 源码)
        out = llm_enhance(_base_feedback(), "q", cfg, tracker=TokenUsageTracker())
        assert out["summary"].startswith(MARKER), (
            f"STEP6 llm_enhance 未生效: summary={out['summary'][:80]!r}"
        )


class TestEngineIntegrationLive:
    def test_preload_then_run_uses_llm(self, local_endpoint) -> None:
        """preload → run() 全链路: 摘要被 LLM 改写, 门控真实生效。"""
        engine = FeedbackEngine(config=_local_settings())
        pre = engine.preload_llm()
        assert pre["loaded"] is True
        assert pre["backend"] == "local"
        out = engine.run(
            [], DEFAULT_SETUP.model_dump(), "shanghai", question="为什么推头？"
        )
        assert out["summary"].startswith(MARKER)
        engine.unload_llm()

    def test_without_preload_llm_never_runs(self, local_endpoint) -> None:
        """未 preload 时 run() 保持规则摘要 (游戏内省内存的门控语义)。"""
        srv, _ = local_endpoint
        engine = FeedbackEngine(config=_local_settings())
        out = engine.run(
            [], DEFAULT_SETUP.model_dump(), "shanghai", question="为什么推头？"
        )
        assert not out["summary"].startswith(MARKER)
        assert srv.requests == 0
