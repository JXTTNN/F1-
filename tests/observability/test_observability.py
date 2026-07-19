"""Tests for :mod:`f1opt.observability` (structlog config + metrics registry).

Iter-06: verifies that:
- :func:`configure_structlog` runs without raising.
- :func:`get_logger` returns an object exposing the stdlib-compatible
  ``debug`` / ``info`` / ``warning`` / ``exception`` methods.
- A JSON-configured structlog logger emits structured JSON containing the
  ``event``, ``level`` and ``timestamp`` fields on stdout.
- :class:`MetricsRegistry` records latency samples and reports correct
  rolling-window stats (min/p50/p95/max/count) plus listener counters.
"""

from __future__ import annotations

import json

from f1opt.observability.logging import configure_structlog, get_logger
from f1opt.observability.metrics import LatencyHistogram, MetricsRegistry


# --------------------------------------------------------------------------- #
# structlog configuration
# --------------------------------------------------------------------------- #
def test_configure_structlog_no_raise() -> None:
    """configure_structlog() must not raise for both json and console modes."""
    configure_structlog(json=True)
    configure_structlog(json=False)


def test_get_logger_returns_logger() -> None:
    """get_logger returns an object with stdlib-compatible log methods."""
    configure_structlog()
    log = get_logger("test_logger_methods")
    assert hasattr(log, "debug")
    assert hasattr(log, "info")
    assert hasattr(log, "warning")
    assert hasattr(log, "error")
    assert hasattr(log, "exception")
    assert callable(log.debug)
    assert callable(log.info)


def test_log_emits_json_fields(capsys: object) -> None:
    """A JSON-configured logger emits JSON containing event/level/timestamp."""
    configure_structlog(json=True)
    log = get_logger("test_json_emit")
    log.info("event_test")
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    out = captured.out
    # The JSONRenderer output is a single JSON object line.
    assert "event_test" in out
    assert "event" in out
    assert "timestamp" in out
    # The output must be valid JSON (one line per record).
    line = out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "event_test"
    assert record["level"] == "info"
    assert "timestamp" in record


# --------------------------------------------------------------------------- #
# MetricsRegistry
# --------------------------------------------------------------------------- #
def test_metrics_registry_empty_stats() -> None:
    """A fresh registry reports zero counts and zero latency stats."""
    reg = MetricsRegistry()
    snap = reg.snapshot(listener=None)
    assert snap["listener"] == {
        "received": 0,
        "dropped": 0,
        "parse_errors": 0,
        "regressions": 0,
        "gaps": 0,
        "validation_failures": 0,
        "flagged_samples": 0,
    }
    for name in ("predict", "search", "feedback"):
        assert snap["latency"][name] == {
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "count": 0,
        }
    assert snap["uptime_s"] >= 0.0


def test_metrics_registry_record_and_stats() -> None:
    """Recording samples updates count and latency stats."""
    reg = MetricsRegistry()
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        reg.predict.record(v)
    stats = reg.predict.stats()
    assert stats["count"] == 5
    assert stats["min"] == 0.1
    assert stats["max"] == 0.5
    assert 0.2 <= stats["p50"] <= 0.4
    assert stats["p95"] >= stats["p50"]


def test_metrics_registry_snapshot_with_listener() -> None:
    """snapshot() reflects listener counter attributes when a listener is given."""

    class FakeListener:
        received = 42
        dropped = 3
        parse_errors = 1
        regressions = 0
        gaps = 0
        validation_failures = 5

    reg = MetricsRegistry()
    snap = reg.snapshot(FakeListener())
    assert snap["listener"]["received"] == 42
    assert snap["listener"]["dropped"] == 3
    assert snap["listener"]["parse_errors"] == 1
    assert snap["listener"]["validation_failures"] == 5


def test_latency_histogram_rolling_window() -> None:
    """The deque maxlen=100 drops the oldest samples beyond 100."""
    hist = LatencyHistogram()
    for i in range(150):
        hist.record(float(i) / 100.0)
    stats = hist.stats()
    assert stats["count"] == 100  # capped at maxlen
    # The oldest 50 samples (0.00..0.49) were dropped; min is now 0.50.
    assert stats["min"] == 0.50
    assert stats["max"] == 1.49
