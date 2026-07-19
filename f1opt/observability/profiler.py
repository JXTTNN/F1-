"""Performance profiling + latency budget infrastructure.

Lightweight, pure-python (stdlib only) timing/budgeting helpers used to keep
the optimizer's interactive paths within their latency budgets.

References (textbook formulas, no papers):
    - Knuth TAOCP vol 2 (statistics of running times).
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

__all__ = [
    "PerformanceProfiler",
    "OperationTimer",
    "LatencyBudget",
    "MemoryTracker",
    "DEFAULT_LATENCY_BUDGETS_MS",
]

# Module-level default latency budgets (ms) for the optimizer's hot paths.
DEFAULT_LATENCY_BUDGETS_MS: dict[str, float] = {
    "predict_lap_time": 50.0,
    "search_setup": 500.0,
    "generate_feedback": 200.0,
    "parse_packet": 1.0,
    "align_frames": 200.0,
    "bayesian_search": 2000.0,
    "batch_predict": 200.0,
}


# --------------------------------------------------------------------------- #
# PerformanceProfiler
# --------------------------------------------------------------------------- #
class PerformanceProfiler:
    """Measure + report performance of named operations."""

    def __init__(self) -> None:
        # name -> deque of elapsed_ms (capped to last 1000 samples).
        self._samples: dict[str, deque[float]] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager that times a block, recording to the histogram."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record(name, elapsed_ms)

    def _record(self, name: str, elapsed_ms: float) -> None:
        if name not in self._samples:
            self._samples[name] = deque(maxlen=1000)
        self._samples[name].append(elapsed_ms)

    def time_function(self, func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Call ``func``, return (result, elapsed_s)."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        return result, time.perf_counter() - start

    def benchmark(
        self,
        func: Callable,
        args_list: list,
        warmup: int = 1,
        repeats: int = 5,
    ) -> dict[str, Any]:
        """Benchmark ``func`` over ``args_list``; return timing statistics."""
        # Warmup runs (discarded).
        for _ in range(max(0, warmup)):
            for args in args_list:
                args_tuple = args if isinstance(args, tuple) else (args,)
                func(*args_tuple)

        all_times_ms: list[float] = []
        for _ in range(max(1, repeats)):
            for args in args_list:
                args_tuple = args if isinstance(args, tuple) else (args,)
                start = time.perf_counter()
                func(*args_tuple)
                all_times_ms.append((time.perf_counter() - start) * 1000.0)

        return self._summarize(func.__name__, all_times_ms, len(args_list) * repeats)

    @staticmethod
    def _summarize(name: str, times_ms: list[float], n_calls: int) -> dict[str, Any]:
        sorted_t = sorted(times_ms)
        n = len(sorted_t)

        def _pct(p: float) -> float:
            if n == 0:
                return 0.0
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return sorted_t[idx]

        return {
            "name": name,
            "n_calls": n_calls,
            "total_time_s": sum(times_ms) / 1000.0,
            "avg_time_ms": statistics.mean(times_ms) if times_ms else 0.0,
            "p50_ms": _pct(0.50),
            "p95_ms": _pct(0.95),
            "p99_ms": _pct(0.99),
            "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        }

    def report(self) -> dict[str, dict[str, float]]:
        """Return all recorded timings: {name: {count, avg_ms, p50, p95, p99}}."""
        out: dict[str, dict[str, float]] = {}
        for name, samples in self._samples.items():
            if not samples:
                continue
            out[name] = self._summarize(
                name, list(samples), len(samples)
            )
            # Trim to the public contract.
            out[name] = {
                "count": float(len(samples)),
                "avg_ms": out[name]["avg_time_ms"],
                "p50_ms": out[name]["p50_ms"],
                "p95_ms": out[name]["p95_ms"],
                "p99_ms": out[name]["p99_ms"],
            }
        return out

    def reset(self) -> None:
        self._samples.clear()

    def summary_text(self) -> str:
        """Human-readable Chinese summary."""
        lines = []
        for name, stats in self.report().items():
            lines.append(
                f"操作 {name} 调用 {int(stats['count'])} 次, "
                f"平均 {stats['avg_ms']:.2f}ms, P95 {stats['p95_ms']:.2f}ms"
            )
        return "\n".join(lines) if lines else "无性能记录"


# --------------------------------------------------------------------------- #
# OperationTimer
# --------------------------------------------------------------------------- #
class OperationTimer:
    """Lightweight timer with lap support."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._last = self._start
        self._laps: dict[str, float] = {}

    def lap(self, name: str) -> float:
        """Record a lap, return elapsed seconds since last lap."""
        now = time.perf_counter()
        elapsed = now - self._last
        self._laps[name] = elapsed
        self._last = now
        return elapsed

    def total(self) -> float:
        """Total elapsed seconds since timer start."""
        return time.perf_counter() - self._start

    def laps(self) -> dict[str, float]:
        return dict(self._laps)


# --------------------------------------------------------------------------- #
# LatencyBudget
# --------------------------------------------------------------------------- #
class LatencyBudget:
    """Check whether operations meet latency budgets (ms)."""

    def __init__(self, budgets: dict[str, float] | None = None) -> None:
        self.budgets = dict(budgets) if budgets else dict(DEFAULT_LATENCY_BUDGETS_MS)

    def check(self, name: str, elapsed_ms: float) -> dict[str, Any]:
        """Check a single timing against its budget."""
        budget = self.budgets.get(name)
        if budget is None:
            return {
                "name": name,
                "elapsed_ms": float(elapsed_ms),
                "budget_ms": None,
                "within_budget": True,
                "headroom_ms": None,
                "headroom_pct": None,
            }
        within = elapsed_ms <= budget
        headroom_ms = budget - elapsed_ms
        return {
            "name": name,
            "elapsed_ms": float(elapsed_ms),
            "budget_ms": float(budget),
            "within_budget": within,
            "headroom_ms": float(headroom_ms),
            "headroom_pct": float(headroom_ms / budget) if budget > 0 else 0.0,
        }

    def check_all(self, timings: dict[str, float]) -> list[dict[str, Any]]:
        """Check multiple timings against budgets."""
        return [self.check(name, ms) for name, ms in timings.items()]

    def violations(self, timings: dict[str, float]) -> list[str]:
        """Return names of violated budgets."""
        return [
            name for name, ms in timings.items()
            if name in self.budgets and ms > self.budgets[name]
        ]


# --------------------------------------------------------------------------- #
# MemoryTracker
# --------------------------------------------------------------------------- #
class MemoryTracker:
    """Track memory usage (RSS + VMS)."""

    def snapshot(self) -> dict[str, float]:
        """Return current memory snapshot (MB)."""
        rss_mb = vms_mb = 0.0
        try:
            import resource
            # ru_maxrss is in KB on Linux.
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_mb = usage.ru_maxrss / 1024.0
            vms_mb = rss_mb  # ru_maxrss is peak RSS; approximate VMS.
        except (ImportError, AttributeError):
            pass
        # Python object count estimate (rough).
        import gc
        python_objects_estimate = len(gc.get_objects())
        return {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "python_objects_estimate": float(python_objects_estimate),
        }

    def delta(self, func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, float]]:
        """Call ``func``, return (result, {before_mb, after_mb, delta_mb})."""
        before = self.snapshot()
        result = func(*args, **kwargs)
        after = self.snapshot()
        return result, {
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
            "delta_mb": after["rss_mb"] - before["rss_mb"],
        }
