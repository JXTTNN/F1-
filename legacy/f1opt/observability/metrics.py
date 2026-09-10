"""Runtime metrics registry for F1OPT (Iter-06).

Provides :class:`MetricsRegistry` — a lightweight, process-wide collector of
latency histograms for the ``/api/predict``, ``/api/search`` and
``/api/feedback`` endpoints, plus an uptime gauge. Exposed via the
``GET /api/metrics`` endpoint.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyHistogram:
    """Rolling-window latency stats (last N samples)."""

    samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )

    def record(self, seconds: float) -> None:
        """Append a latency sample (seconds)."""
        self.samples.append(seconds)

    def stats(self) -> dict[str, Any]:
        """Return ``{min, p50, p95, max, count}`` over the current window."""
        if not self.samples:
            return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
        s = sorted(self.samples)
        n = len(s)
        return {
            "min": s[0],
            "p50": s[n // 2],
            "p95": s[int(n * 0.95)] if n > 1 else s[0],
            "max": s[-1],
            "count": n,
        }


class MetricsRegistry:
    """Process-wide metrics: latency histograms for predict/search/feedback."""

    def __init__(self) -> None:
        self.predict = LatencyHistogram()
        self.search = LatencyHistogram()
        self.feedback = LatencyHistogram()
        self.start_time = time.perf_counter()

    def uptime_s(self) -> float:
        """Seconds since this registry was constructed."""
        return time.perf_counter() - self.start_time

    def snapshot(self, listener: Any) -> dict[str, Any]:
        """Return the full metrics dict (listener counters + latency + uptime).

        ``listener`` may be ``None`` (e.g. when the UDP listener failed to
        bind); in that case all listener counters are reported as 0.
        """
        listener_metrics: dict[str, Any] = {
            "received": 0,
            "dropped": 0,
            "parse_errors": 0,
            "regressions": 0,
            "gaps": 0,
            "validation_failures": 0,
            "flagged_samples": 0,
        }
        if listener is not None:
            for k in listener_metrics:
                listener_metrics[k] = getattr(listener, k, 0)
        return {
            "listener": listener_metrics,
            "latency": {
                "predict": self.predict.stats(),
                "search": self.search.stats(),
                "feedback": self.feedback.stats(),
            },
            "uptime_s": self.uptime_s(),
        }


__all__ = ["LatencyHistogram", "MetricsRegistry"]
