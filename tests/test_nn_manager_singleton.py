"""setup_sim 单例 + sim_optimizer 的健壮性测试。

覆盖：

1. **并发安全**：FastAPI 默认线程池并发处理请求，首次加载须由模块级锁
   串行化，不得重复构造 SetupSimModel。
2. **失败不重试**：加载失败（构造抛异常）后用哨兵缓存结果，
   不在每个请求上重新读盘解析。
3. **模拟优化器的降级**：模型不可用 / 赛道未覆盖时逐位返回输入。
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from setup_tuner.domain.setup import CarSetup
from setup_tuner.engine import setup_sim as sim_mod
from setup_tuner.engine.setup_sim import (
    get_setup_sim,
    reset_setup_sim,
)
from setup_tuner.engine.sim_optimizer import sim_refine


@pytest.fixture(autouse=True)
def _reset_singleton():
    """每个用例前后都重置单例，避免跨用例污染。"""
    reset_setup_sim()
    yield
    reset_setup_sim()


class TestSetupSimSingleton:
    """setup_sim 侧单例的并发与失败缓存。"""

    def test_failure_is_cached_not_retried(self, monkeypatch) -> None:
        """加载失败只尝试一次，后续调用不重复构造。"""
        calls = {"n": 0}

        def _boom(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            raise RuntimeError("模拟加载失败")

        monkeypatch.setattr(sim_mod, "SetupSimModel", _boom)

        for _ in range(5):
            model = get_setup_sim()
            assert model.available is False
        assert calls["n"] == 1, f"加载失败被重试了 {calls['n']} 次，应恰好 1 次"
        assert sim_mod._SIM is sim_mod._LOAD_FAILED

    def test_success_is_cached(self, monkeypatch) -> None:
        """构造成功后缓存实例，后续调用不再构造。"""
        calls = {"n": 0}
        sentinel = object()

        def _fake(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return sentinel

        monkeypatch.setattr(sim_mod, "SetupSimModel", _fake)

        assert get_setup_sim() is sentinel
        assert get_setup_sim() is sentinel
        assert calls["n"] == 1

    def test_reset_allows_reload(self, monkeypatch) -> None:
        """reset 后允许重新加载（失败 → 成功可恢复）。"""
        calls = {"n": 0}

        def _flaky(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("首次失败")
            return object()

        monkeypatch.setattr(sim_mod, "SetupSimModel", _flaky)

        assert get_setup_sim().available is False
        reset_setup_sim()
        assert get_setup_sim() is not None
        assert calls["n"] == 2

    def test_concurrent_first_load_builds_once(self, monkeypatch) -> None:
        """并发首调只构造一次（锁 + 双重检查）。"""
        import time

        calls = {"n": 0}
        lock = threading.Lock()
        sentinel = object()

        def _slow(*args: Any, **kwargs: Any) -> Any:
            with lock:
                calls["n"] += 1
            time.sleep(0.02)
            return sentinel

        monkeypatch.setattr(sim_mod, "SetupSimModel", _slow)

        barrier = threading.Barrier(8)
        results: list[Any] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            mgr = get_setup_sim()
            with results_lock:
                results.append(mgr)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is sentinel for r in results)
        assert calls["n"] == 1, f"并发首调构造了 {calls['n']} 次，应恰好 1 次"

    def test_failure_sentinel_is_not_none(self) -> None:
        """失败哨兵不得为 None，否则会与"未加载"混淆而反复重试。"""
        assert sim_mod._LOAD_FAILED is not None


class TestSimRefineDegradation:
    """模拟优化器在模型不可用/赛道未覆盖时必须逐位中性。"""

    def test_returns_input_unchanged_without_model(self, monkeypatch) -> None:
        """模型不可用 → delta 与输入逐位一致，available=False。"""
        reset_setup_sim()

        class _Unavailable:
            available = False
            reason = "测试：模型不可用"
            track_ids: list[str] = []

            def covers(self, track_id: str) -> bool:
                return False

        monkeypatch.setattr(sim_mod, "SetupSimModel", lambda *a, **k: _Unavailable())
        reset_setup_sim()
        delta0 = {f.name: 0.5 for f in __import__(
            "setup_tuner.domain.setup", fromlist=["ALL_SETUP_FIELDS"],
        ).ALL_SETUP_FIELDS}
        # 注意：get_setup_sim 在 sim_optimizer 内被调用
        import setup_tuner.engine.sim_optimizer as opt_mod

        monkeypatch.setattr(
            opt_mod, "get_setup_sim",
            lambda *a, **k: _Unavailable(),
        )
        result = sim_refine(
            delta0, CarSetup.default().to_dict(), "suzuka", dx=None,
        )
        assert result.available is False
        assert result.delta == delta0
        assert "模型不可用" in result.reason

    def test_unknown_track_returns_input_unchanged(self, monkeypatch) -> None:
        """赛道不在覆盖范围 → 原样返回。"""
        import setup_tuner.engine.sim_optimizer as opt_mod

        class _Model:
            available = True
            reason = "ok"
            track_ids = ["suzuka"]

            def covers(self, track_id: str) -> bool:
                return track_id in self.track_ids

            def predict_delta_s(self, track_id: str, setup_norm: dict) -> float | None:
                return 0.0

        monkeypatch.setattr(opt_mod, "get_setup_sim", lambda *a, **k: _Model())
        delta0 = {f.name: 0.0 for f in __import__(
            "setup_tuner.domain.setup", fromlist=["ALL_SETUP_FIELDS"],
        ).ALL_SETUP_FIELDS}
        result = sim_refine(
            delta0, CarSetup.default().to_dict(), "no_such_track", dx=None,
        )
        assert result.available is False
        assert result.delta == delta0
        assert "覆盖范围" in result.reason
