"""Performance profiling middleware and decorators."""

from __future__ import annotations

import time
import functools
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from f1opt.observability.logging import get_logger

log = get_logger(__name__)


class LatencyHistogram:
    """Simple latency histogram with p50/p95/p99 tracking."""

    def __init__(self, max_samples: int = 10000) -> None:
        self._samples: list[float] = []
        self._max_samples = max_samples
        self.count = 0

    def record(self, elapsed_s: float) -> None:
        self.count += 1
        if len(self._samples) < self._max_samples:
            self._samples.append(elapsed_s)
        else:
            idx = self.count % self._max_samples
            if idx < len(self._samples):
                self._samples[idx] = elapsed_s

    @property
    def p50(self) -> float:
        return self._percentile(50.0)

    @property
    def p95(self) -> float:
        return self._percentile(95.0)

    @property
    def p99(self) -> float:
        return self._percentile(99.0)

    @property
    def avg(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def _percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        k = (len(sorted_samples) - 1) * p / 100.0
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_samples):
            return sorted_samples[f] + c * (sorted_samples[f + 1] - sorted_samples[f])
        return sorted_samples[f]

    def stats(self) -> dict[str, float]:
        return {
            "count": self.count,
            "avg_s": self.avg,
            "p50_s": self.p50,
            "p95_s": self.p95,
            "p99_s": self.p99,
            "samples": len(self._samples),
        }


_histograms: dict[str, LatencyHistogram] = defaultdict(LatencyHistogram)


def profile(func: Callable, *, name: str | None = None) -> Callable:
    label = name or func.__qualname__
    _hist = _attached[label]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _hist.record(elapsed)
        if elapsed > 0.01:
            log.debug("%s: %.1f ms", label, elapsed * 1000)
        return result
    return wrapper


def get_histogram(name: str) -> LatencyHistogram:
    return _attached[name]


def all_histograms() -> dict[str, dict[str, float]]:
    return {name: h.stats() for name, h in _attached.items()}
