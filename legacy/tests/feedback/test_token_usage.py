"""Tests for LLM token usage tracking (Iter-138)."""
from __future__ import annotations

import pytest

from f1opt.config import Settings
from f1opt.feedback.engine import (
    FeedbackEngine,
    TokenUsageTracker,
    _extract_usage,
    _extract_usage_from_stream_chunk,
    get_default_token_tracker,
    llm_enhance,
    llm_enhance_stream,
)


# --- TokenUsageTracker unit tests ------------------------------------------ #
class TestTokenUsageTracker:
    def test_record_success_logs_tokens(self) -> None:
        t = TokenUsageTracker()
        rec = t.record(
            "openai", "gpt-4o-mini",
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            success=True,
        )
        assert rec is not None
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50
        assert rec.total_tokens == 150
        assert rec.success is True
        assert rec.streamed is False

    def test_record_streamed_flag(self) -> None:
        t = TokenUsageTracker()
        t.record("openai", "m", {"prompt_tokens": 10, "completion_tokens": 5,
                                  "total_tokens": 15}, streamed=True)
        totals = t.totals()
        assert totals["streamed_calls"] == 1

    def test_record_failed_call_with_no_usage_logs_zero_tokens(self) -> None:
        t = TokenUsageTracker()
        rec = t.record("openai", "m", None, success=False)
        assert rec is not None
        assert rec.prompt_tokens == 0
        assert rec.success is False
        totals = t.totals()
        assert totals["calls"] == 1
        assert totals["failed_calls"] == 1
        assert totals["successful_calls"] == 0

    def test_record_success_with_no_usage_skips(self) -> None:
        """A successful call with no usage dict is NOT logged (nothing to track)."""
        t = TokenUsageTracker()
        rec = t.record("openai", "m", None, success=True)
        assert rec is None
        assert t.totals()["calls"] == 0

    def test_record_malformed_usage_treated_as_zero(self) -> None:
        t = TokenUsageTracker()
        rec = t.record("openai", "m", {"prompt_tokens": "bad", "completion_tokens": None})
        assert rec is not None
        assert rec.prompt_tokens == 0
        assert rec.completion_tokens == 0

    def test_record_total_defaults_to_sum_when_missing(self) -> None:
        t = TokenUsageTracker()
        rec = t.record("openai", "m", {"prompt_tokens": 30, "completion_tokens": 20})
        assert rec is not None
        assert rec.total_tokens == 50

    def test_totals_aggregates_across_calls(self) -> None:
        t = TokenUsageTracker()
        t.record("openai", "m", {"prompt_tokens": 100, "completion_tokens": 50,
                                  "total_tokens": 150})
        t.record("openai", "m", {"prompt_tokens": 200, "completion_tokens": 100,
                                  "total_tokens": 300})
        tot = t.totals()
        assert tot["prompt_tokens"] == 300
        assert tot["completion_tokens"] == 150
        assert tot["total_tokens"] == 450
        assert tot["calls"] == 2
        assert tot["successful_calls"] == 2
        assert tot["failed_calls"] == 0

    def test_totals_empty_returns_zeros(self) -> None:
        t = TokenUsageTracker()
        tot = t.totals()
        assert tot["prompt_tokens"] == 0
        assert tot["calls"] == 0
        assert tot["successful_calls"] == 0
        assert tot["failed_calls"] == 0
        assert tot["streamed_calls"] == 0

    def test_per_backend_breakdown(self) -> None:
        t = TokenUsageTracker()
        t.record("openai", "m1", {"prompt_tokens": 100, "completion_tokens": 50,
                                   "total_tokens": 150})
        t.record("local", "m2", {"prompt_tokens": 10, "completion_tokens": 5,
                                  "total_tokens": 15})
        pb = t.per_backend()
        assert set(pb.keys()) == {"openai", "local"}
        assert pb["openai"]["total_tokens"] == 150
        assert pb["local"]["total_tokens"] == 15

    def test_recent_returns_last_n(self) -> None:
        t = TokenUsageTracker(max_records=100)
        for i in range(10):
            t.record("openai", "m", {"prompt_tokens": i, "completion_tokens": 0,
                                      "total_tokens": i})
        recs = t.recent(3)
        assert len(recs) == 3
        assert recs[-1].prompt_tokens == 9
        assert recs[0].prompt_tokens == 7

    def test_recent_n_zero_returns_empty(self) -> None:
        t = TokenUsageTracker()
        t.record("openai", "m", {"prompt_tokens": 1, "completion_tokens": 0,
                                  "total_tokens": 1})
        assert t.recent(0) == []

    def test_reset_clears_all(self) -> None:
        t = TokenUsageTracker()
        t.record("openai", "m", {"prompt_tokens": 100, "completion_tokens": 50,
                                  "total_tokens": 150})
        t.reset()
        assert t.totals()["calls"] == 0

    def test_max_records_bounds_history(self) -> None:
        t = TokenUsageTracker(max_records=3)
        for i in range(5):
            t.record("openai", "m", {"prompt_tokens": i, "completion_tokens": 0,
                                      "total_tokens": i})
        assert t.totals()["calls"] == 3
        recs = t.recent(10)
        assert recs[0].prompt_tokens == 2
        assert recs[-1].prompt_tokens == 4

    def test_max_records_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="max_records"):
            TokenUsageTracker(max_records=0)

    def test_cost_estimate_default_rates(self) -> None:
        t = TokenUsageTracker()
        t.record("openai", "gpt-4o-mini",
                 {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                  "total_tokens": 2_000_000})
        cost = t.cost_estimate()
        assert abs(cost["openai"] - 0.75) < 1e-9

    def test_cost_estimate_local_is_zero(self) -> None:
        t = TokenUsageTracker()
        t.record("local", "llama3.1",
                 {"prompt_tokens": 500, "completion_tokens": 200,
                  "total_tokens": 700})
        cost = t.cost_estimate()
        assert cost["local"] == 0.0

    def test_cost_estimate_custom_rates(self) -> None:
        t = TokenUsageTracker()
        t.record("openai", "m",
                 {"prompt_tokens": 1_000_000, "completion_tokens": 0,
                  "total_tokens": 1_000_000})
        cost = t.cost_estimate(rates={"openai": (1.0, 2.0)})
        assert abs(cost["openai"] - 1.0) < 1e-9

    def test_thread_safety_smoke(self) -> None:
        """Concurrent records from multiple threads don't lose entries."""
        import threading

        t = TokenUsageTracker()
        N_THREADS = 8
        N_PER = 100

        def _worker() -> None:
            for _ in range(N_PER):
                t.record("openai", "m",
                         {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2})

        threads = [threading.Thread(target=_worker) for _ in range(N_THREADS)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert t.totals()["calls"] == N_THREADS * N_PER


# --- usage extraction helpers ---------------------------------------------- #
class TestUsageExtraction:
    def test_extract_usage_present(self) -> None:
        data = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                                          "total_tokens": 8}}
        u = _extract_usage(data)
        assert u == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

    def test_extract_usage_absent(self) -> None:
        assert _extract_usage({"choices": []}) is None
        assert _extract_usage({}) is None

    def test_extract_usage_not_dict(self) -> None:
        assert _extract_usage({"usage": "not a dict"}) is None

    def test_extract_usage_from_stream_chunk_present(self) -> None:
        obj = {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                         "total_tokens": 2}}
        u = _extract_usage_from_stream_chunk(obj)
        assert u is not None and u["total_tokens"] == 2

    def test_extract_usage_from_stream_chunk_absent(self) -> None:
        assert _extract_usage_from_stream_chunk(
            {"choices": [{"delta": {"content": "x"}}]}
        ) is None
        assert _extract_usage_from_stream_chunk({}) is None


# --- llm_enhance integration (mocked httpx) -------------------------------- #
def _feedback() -> dict:
    return {
        "summary": "rule-based",
        "dimensions": [{"name": "balance", "value": "neutral",
                        "evidence": "g_lat=2.0", "advice": None}],
        "setup_suggestions": [],
        "sources": [{"frame_t": 1.0, "field": "speed", "value": 200.0}],
    }


def _settings(backend: str = "openai", key: str = "sk-fake") -> Settings:
    return Settings(llm_backend=backend, llm_api_key=key)


def _install_nonstream_mock(monkeypatch: pytest.MonkeyPatch, usage: dict | None):
    """Mock httpx.Client so post() returns a response with the given usage."""
    import httpx

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            data = {"choices": [{"message": {"content": "LLM rewrite"}}]}
            if usage is not None:
                data["usage"] = usage
            return data

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def post(self, url: str, **k: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)


def test_llm_enhance_records_usage_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nonstream_mock(
        monkeypatch,
        {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    )
    tk = TokenUsageTracker()
    out = llm_enhance(_feedback(), "q?", _settings(), tracker=tk)
    assert out["summary"] == "LLM rewrite"
    tot = tk.totals()
    assert tot["calls"] == 1
    assert tot["prompt_tokens"] == 120
    assert tot["completion_tokens"] == 40
    assert tot["total_tokens"] == 160
    assert tot["successful_calls"] == 1
    assert tot["failed_calls"] == 0


def test_llm_enhance_records_failed_call_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class _BoomClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _BoomClient:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def post(self, url: str, **k: object):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", _BoomClient)
    tk = TokenUsageTracker()
    out = llm_enhance(_feedback(), "q?", _settings(), tracker=tk)
    assert out["summary"] == "rule-based"
    tot = tk.totals()
    assert tot["calls"] == 1
    assert tot["failed_calls"] == 1
    assert tot["successful_calls"] == 0
    assert tot["prompt_tokens"] == 0


def test_llm_enhance_no_usage_still_success_but_unlogged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend that omits usage: success but no record (nothing to track)."""
    _install_nonstream_mock(monkeypatch, usage=None)
    tk = TokenUsageTracker()
    out = llm_enhance(_feedback(), "q?", _settings(), tracker=tk)
    assert out["summary"] == "LLM rewrite"
    assert tk.totals()["calls"] == 0


def test_llm_enhance_default_tracker_when_none_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When tracker=None, the module-level default tracker is used."""
    _install_nonstream_mock(
        monkeypatch,
        {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    )
    default = get_default_token_tracker()
    default.reset()
    llm_enhance(_feedback(), "q?", _settings())  # tracker=None
    assert default.totals()["calls"] == 1
    assert default.totals()["total_tokens"] == 10


# --- llm_enhance_stream integration (mocked SSE with usage chunk) ---------- #
def _sse_lines_with_usage(chunks: list[str], usage: dict | None) -> list[str]:
    import json as _json
    lines: list[str] = []
    for c in chunks:
        payload = _json.dumps({"choices": [{"delta": {"content": c}}]})
        lines.append(f"data: {payload}")
        lines.append("")
    if usage is not None:
        payload = _json.dumps({"choices": [], "usage": usage})
        lines.append(f"data: {payload}")
        lines.append("")
    lines.append("data: [DONE]")
    return lines


def _install_stream_mock(monkeypatch: pytest.MonkeyPatch, lines: list[str]):
    import httpx

    class _Resp:
        def __init__(self) -> None:
            self._lines = lines

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield from self._lines

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def stream(self, method: str, url: str, **k: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)


def test_llm_enhance_stream_records_usage_from_final_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _sse_lines_with_usage(
        ["Hello", " world"],
        {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    )
    _install_stream_mock(monkeypatch, lines)
    tk = TokenUsageTracker()
    chunks = list(llm_enhance_stream(_feedback(), "q?", _settings(), tracker=tk))
    assert chunks == ["Hello", " world"]
    tot = tk.totals()
    assert tot["calls"] == 1
    assert tot["prompt_tokens"] == 80
    assert tot["completion_tokens"] == 20
    assert tot["total_tokens"] == 100
    assert tot["streamed_calls"] == 1
    assert tot["successful_calls"] == 1


def test_llm_enhance_stream_no_usage_chunk_unlogged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream without a final usage chunk: success but no record."""
    lines = _sse_lines_with_usage(["A", "B"], usage=None)
    _install_stream_mock(monkeypatch, lines)
    tk = TokenUsageTracker()
    list(llm_enhance_stream(_feedback(), "q?", _settings(), tracker=tk))
    assert tk.totals()["calls"] == 0


def test_llm_enhance_stream_network_error_logs_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class _BoomClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _BoomClient:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def stream(self, method: str, url: str, **k: object):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", _BoomClient)
    tk = TokenUsageTracker()
    chunks = list(llm_enhance_stream(_feedback(), "q?", _settings(), tracker=tk))
    assert chunks == []
    tot = tk.totals()
    assert tot["calls"] == 1
    assert tot["failed_calls"] == 1
    assert tot["streamed_calls"] == 1


# --- FeedbackEngine.token_usage integration -------------------------------- #
def test_engine_token_usage_accessor(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nonstream_mock(
        monkeypatch,
        {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
    )
    eng = FeedbackEngine(config=_settings())
    eng.set_token_tracker(TokenUsageTracker())  # isolate from module default
    before = eng.token_usage()
    assert before["calls"] == 0
    # Invoke the LLM enhance path directly via the engine's tracker.
    from f1opt.feedback.engine import llm_enhance
    llm_enhance(_feedback(), "q?", eng.config, tracker=eng.token_tracker)
    after = eng.token_usage()
    assert after["calls"] == 1
    assert after["total_tokens"] == 75
    # cost estimate is queryable.
    cost = eng.token_cost_estimate()
    assert "openai" in cost


def test_engine_reset_token_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nonstream_mock(
        monkeypatch,
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    eng = FeedbackEngine(config=_settings())
    eng.set_token_tracker(TokenUsageTracker())
    from f1opt.feedback.engine import llm_enhance
    llm_enhance(_feedback(), "q?", eng.config, tracker=eng.token_tracker)
    assert eng.token_usage()["calls"] == 1
    eng.reset_token_usage()
    assert eng.token_usage()["calls"] == 0


def test_engine_token_usage_per_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nonstream_mock(
        monkeypatch,
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    eng = FeedbackEngine(config=_settings())
    eng.set_token_tracker(TokenUsageTracker())
    from f1opt.feedback.engine import llm_enhance
    llm_enhance(_feedback(), "q?", eng.config, tracker=eng.token_tracker)
    pb = eng.token_usage_per_backend()
    assert "openai" in pb
    assert pb["openai"]["total_tokens"] == 15
