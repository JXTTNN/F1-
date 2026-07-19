"""Tests for the performance profiler."""

from __future__ import annotations

import time

import pytest

from f1opt.observability.profiler import (
    DEFAULT_LATENCY_BUDGETS_MS,
    LatencyBudget,
    MemoryTracker,
    OperationTimer,
    PerformanceProfiler,
)


# --------------------------------------------------------------------------- #
# PerformanceProfiler
# --------------------------------------------------------------------------- #
class TestPerformanceProfiler:
    def test_measure_records_timing(self) -> None:
        p = PerformanceProfiler()
        with p.measure("op1"):
            time.sleep(0.01)
        report = p.report()
        assert "op1" in report
        assert report["op1"]["avg_ms"] > 0

    def test_measure_name_present_in_report(self) -> None:
        p = PerformanceProfiler()
        with p.measure("my_op"):
            pass
        assert "my_op" in p.report()

    def test_time_function_returns_result_and_elapsed(self) -> None:
        p = PerformanceProfiler()

        def add(a, b):
            return a + b

        result, elapsed = p.time_function(add, 2, 3)
        assert result == 5
        assert elapsed >= 0

    def test_time_function_elapsed_positive(self) -> None:
        p = PerformanceProfiler()

        def slow():
            time.sleep(0.005)
            return "done"

        _, elapsed = p.time_function(slow)
        assert elapsed > 0

    def test_benchmark_returns_stats_dict(self) -> None:
        p = PerformanceProfiler()

        def f(x):
            return x * 2

        stats = p.benchmark(f, [1, 2, 3], warmup=1, repeats=3)
        assert "name" in stats
        assert "p50_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert stats["n_calls"] == 9  # 3 args * 3 repeats

    def test_benchmark_avg_positive(self) -> None:
        p = PerformanceProfiler()

        def f(x):
            time.sleep(0.001)
            return x

        stats = p.benchmark(f, [1, 2], warmup=0, repeats=2)
        assert stats["avg_time_ms"] > 0

    def test_report_returns_dict(self) -> None:
        p = PerformanceProfiler()
        with p.measure("a"):
            pass
        with p.measure("b"):
            pass
        r = p.report()
        assert "a" in r and "b" in r

    def test_reset_clears_all(self) -> None:
        p = PerformanceProfiler()
        with p.measure("x"):
            pass
        p.reset()
        assert p.report() == {}

    def test_summary_text_contains_chinese(self) -> None:
        p = PerformanceProfiler()
        with p.measure("op"):
            pass
        text = p.summary_text()
        assert "操作" in text or "调用" in text

    def test_summary_text_contains_name(self) -> None:
        p = PerformanceProfiler()
        with p.measure("named_op"):
            pass
        assert "named_op" in p.summary_text()

    def test_measure_fast_operation_records(self) -> None:
        """Very fast operation (no sleep) still records a timing."""
        p = PerformanceProfiler()
        with p.measure("fast"):
            x = 1 + 1  # noqa: F841
        assert "fast" in p.report()


# --------------------------------------------------------------------------- #
# OperationTimer
# --------------------------------------------------------------------------- #
class TestOperationTimer:
    def test_total_positive(self) -> None:
        t = OperationTimer()
        time.sleep(0.005)
        assert t.total() > 0

    def test_lap_returns_elapsed(self) -> None:
        t = OperationTimer()
        time.sleep(0.005)
        elapsed = t.lap("first")
        assert elapsed > 0

    def test_laps_returns_dict(self) -> None:
        t = OperationTimer()
        t.lap("a")
        t.lap("b")
        laps = t.laps()
        assert "a" in laps and "b" in laps
        assert laps["a"] > 0 and laps["b"] > 0


# --------------------------------------------------------------------------- #
# LatencyBudget
# --------------------------------------------------------------------------- #
class TestLatencyBudget:
    def test_check_within_budget(self) -> None:
        b = LatencyBudget({"op": 100.0})
        result = b.check("op", 50.0)
        assert result["within_budget"] is True

    def test_check_over_budget(self) -> None:
        b = LatencyBudget({"op": 100.0})
        result = b.check("op", 150.0)
        assert result["within_budget"] is False

    def test_check_unknown_budget_returns_within(self) -> None:
        b = LatencyBudget({"op": 100.0})
        result = b.check("unknown_op", 999.0)
        # Unknown budgets are considered within budget (no constraint).
        assert result["within_budget"] is True

    def test_violations_returns_list(self) -> None:
        b = LatencyBudget({"a": 10.0, "b": 100.0})
        v = b.violations({"a": 15.0, "b": 50.0})
        assert "a" in v
        assert "b" not in v

    def test_headroom_computed(self) -> None:
        b = LatencyBudget({"op": 100.0})
        result = b.check("op", 30.0)
        assert result["headroom_ms"] == pytest.approx(70.0)
        assert result["headroom_pct"] == pytest.approx(0.7, abs=0.01)

    def test_check_all_returns_list(self) -> None:
        b = LatencyBudget({"a": 10.0, "b": 100.0})
        results = b.check_all({"a": 5.0, "b": 50.0})
        assert len(results) == 2
        assert all("within_budget" in r for r in results)

    def test_default_budgets_present(self) -> None:
        assert "predict_lap_time" in DEFAULT_LATENCY_BUDGETS_MS
        assert "search_setup" in DEFAULT_LATENCY_BUDGETS_MS


# --------------------------------------------------------------------------- #
# MemoryTracker
# --------------------------------------------------------------------------- #
class TestMemoryTracker:
    def test_snapshot_returns_dict(self) -> None:
        m = MemoryTracker()
        s = m.snapshot()
        assert "rss_mb" in s
        assert "vms_mb" in s
        assert "python_objects_estimate" in s

    def test_delta_returns_tuple(self) -> None:
        m = MemoryTracker()

        def make_list():
            return list(range(1000))

        result, mem = m.delta(make_list)
        assert result == list(range(1000))
        assert "before_mb" in mem
        assert "after_mb" in mem
        assert "delta_mb" in mem
