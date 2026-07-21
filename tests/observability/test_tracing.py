"""Tests for :mod:`f1opt.observability.tracing`.

The tracing module wraps OpenTelemetry so callers can use ``span`` /
``get_tracer`` unconditionally. These tests exercise the no-op path
(the default when ``F1OPT_OTEL_EXPORT`` is unset) plus the OTel-enabled
path using a fake tracer, without requiring a real OTel collector.
"""

from __future__ import annotations

from typing import Any

import pytest

from f1opt.observability import tracing


@pytest.fixture(autouse=True)
def _clear_otel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tracing is disabled unless a test opts in."""
    monkeypatch.delenv("F1OPT_OTEL_EXPORT", raising=False)


# --------------------------------------------------------------------------- #
# is_tracing_enabled
# --------------------------------------------------------------------------- #
def test_is_tracing_enabled_default_false() -> None:
    assert tracing.is_tracing_enabled() is False


def test_is_tracing_enabled_when_export_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F1OPT_OTEL_EXPORT", "http://localhost:4317")
    assert tracing.is_tracing_enabled() is True


# --------------------------------------------------------------------------- #
# get_tracer / no-op tracer
# --------------------------------------------------------------------------- #
def test_get_tracer_returns_noop_when_disabled() -> None:
    tracer = tracing.get_tracer()
    assert isinstance(tracer, tracing._NoopTracer)


def test_noop_tracer_span_is_safe() -> None:
    """The no-op span supports the full span API without raising."""
    tracer = tracing.get_tracer()
    with tracer.start_as_current_span("op", attributes={"k": "v"}) as s:
        s.set_attribute("a", 1)
        s.record_exception(ValueError("boom"))
        s.set_status("ok")
        s.end()
    # start_span also returns a usable no-op span.
    span = tracer.start_span("op2")
    assert isinstance(span, tracing._NoopSpan)


# --------------------------------------------------------------------------- #
# span() context manager
# --------------------------------------------------------------------------- #
def test_span_noop_path_yields_usable_span() -> None:
    with tracing.span("predict", track="suzuka", iterations=80) as s:
        # No-op span must accept attribute/exception calls silently.
        s.set_attribute("track", "suzuka")
    assert isinstance(s, tracing._NoopSpan)


def test_span_uses_real_tracer_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """When get_tracer returns a non-noop tracer, span delegates to it."""
    calls: dict[str, Any] = {}

    class _FakeSpan:
        def __enter__(self) -> _FakeSpan:
            return self

        def __exit__(self, *exc: Any) -> None:
            calls["exited"] = True

    class _FakeTracer:
        def start_as_current_span(self, name: str, attributes: Any = None) -> _FakeSpan:
            calls["name"] = name
            calls["attributes"] = attributes
            return _FakeSpan()

    monkeypatch.setattr(tracing, "get_tracer", lambda: _FakeTracer())

    with tracing.span("search", track="monza") as s:
        assert isinstance(s, _FakeSpan)

    assert calls["name"] == "search"
    assert calls["attributes"] == {"track": "monza"}
    assert calls["exited"] is True


def test_span_falls_back_to_noop_on_tracer_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the real tracer raises, span must degrade to a no-op, not propagate."""

    class _ExplodingTracer:
        def start_as_current_span(self, name: str, attributes: Any = None) -> Any:
            raise RuntimeError("otel exploded")

    monkeypatch.setattr(tracing, "get_tracer", lambda: _ExplodingTracer())

    with tracing.span("boom") as s:
        assert isinstance(s, tracing._NoopSpan)


# --------------------------------------------------------------------------- #
# _otel_available
# --------------------------------------------------------------------------- #
def test_otel_available_true_when_installed() -> None:
    """opentelemetry is a project dependency, so it must be importable."""
    assert tracing._otel_available() is True


# --------------------------------------------------------------------------- #
# real-tracer path (OTel installed + export configured)
# --------------------------------------------------------------------------- #
def test_get_tracer_real_when_enabled_and_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With export set and OTel installed, a real (non-noop) tracer is used."""
    monkeypatch.setenv("F1OPT_OTEL_EXPORT", "http://localhost:4317")
    tracer = tracing.get_tracer()
    assert not isinstance(tracer, tracing._NoopTracer)
    # The real tracer exposes the OTel span API.
    assert hasattr(tracer, "start_as_current_span")


def test_get_tracer_falls_back_when_trace_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure obtaining the real tracer degrades to a no-op tracer."""
    monkeypatch.setenv("F1OPT_OTEL_EXPORT", "http://localhost:4317")
    monkeypatch.setattr(tracing, "_otel_available", lambda: True)

    import opentelemetry.trace as _trace

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no tracer")

    monkeypatch.setattr(_trace, "get_tracer", _boom)
    assert isinstance(tracing.get_tracer(), tracing._NoopTracer)


# --------------------------------------------------------------------------- #
# shutdown_tracing
# --------------------------------------------------------------------------- #
def test_shutdown_tracing_noop_when_disabled() -> None:
    """shutdown_tracing is a safe no-op when tracing is disabled."""
    # Should simply return without raising.
    tracing.shutdown_tracing()


def test_shutdown_tracing_enabled_path_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enabled shutdown path runs without raising when no SDK provider set."""
    monkeypatch.setenv("F1OPT_OTEL_EXPORT", "http://localhost:4317")
    # No SDK TracerProvider is installed as the global provider in tests, so
    # this exercises the isinstance branch and returns cleanly.
    tracing.shutdown_tracing()
