"""engine._get_nn_manager 单例的健壮性测试。

覆盖本轮加固的两个点：

1. **并发安全**：FastAPI 默认线程池并发处理请求，
   首次加载须由模块级锁串行化，不得重复构造 NNModelManager。
2. **失败不重试**：加载失败后用哨兵缓存结果，
   不在每个请求上重新 import + 构造（PyTorch 可用但权重缺失时
   仍会构造完整 F1SetupNet，代价很高）。
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from setup_tuner.engine import engine
from setup_tuner.engine.engine import _LOAD_FAILED, _get_nn_manager, reset_nn_manager


@pytest.fixture(autouse=True)
def _reset_singleton():
    """每个用例前后都重置单例，避免跨用例污染。"""
    reset_nn_manager()
    yield
    reset_nn_manager()


class TestNoNnManagerSingleton:
    """engine 侧单例的并发与失败缓存。"""

    def test_returns_none_when_manager_unavailable(self, monkeypatch) -> None:
        """NNModelManager 构造抛异常时对外表现为 None。"""
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("模拟加载失败")

        monkeypatch.setattr("setup_tuner.engine.nn_model.NNModelManager", _boom)
        assert _get_nn_manager() is None

    def test_failure_is_cached_not_retried(self, monkeypatch) -> None:
        """加载失败只尝试一次，后续调用不重复构造。"""
        calls = {"n": 0}

        def _boom(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            raise RuntimeError("模拟加载失败")

        monkeypatch.setattr("setup_tuner.engine.nn_model.NNModelManager", _boom)

        for _ in range(5):
            assert _get_nn_manager() is None

        assert calls["n"] == 1, f"加载失败被重试了 {calls['n']} 次，应恰好 1 次"
        # 内部确实缓存了失败哨兵
        assert engine._NN_MANAGER is _LOAD_FAILED

    def test_success_is_cached(self, monkeypatch) -> None:
        """构造成功后缓存实例，后续调用不再构造。"""
        calls = {"n": 0}
        sentinel = object()

        def _fake(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return sentinel

        monkeypatch.setattr("setup_tuner.engine.nn_model.NNModelManager", _fake)

        assert _get_nn_manager() is sentinel
        assert _get_nn_manager() is sentinel
        assert calls["n"] == 1

    def test_reset_allows_reload(self, monkeypatch) -> None:
        """reset 后允许重新加载（失败 → 成功可恢复）。"""
        calls = {"n": 0}

        def _flaky(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("首次失败")
            return object()

        monkeypatch.setattr("setup_tuner.engine.nn_model.NNModelManager", _flaky)

        assert _get_nn_manager() is None
        reset_nn_manager()
        assert _get_nn_manager() is not None
        assert calls["n"] == 2

    def test_concurrent_first_load_builds_once(self, monkeypatch) -> None:
        """并发首调只构造一次（锁 + 双重检查）。

        构造函数内加小睡，放大竞态窗口：无锁实现会构造多次。
        """
        import time

        calls = {"n": 0}
        lock = threading.Lock()
        sentinel = object()

        def _slow(*args: Any, **kwargs: Any) -> Any:
            with lock:
                calls["n"] += 1
            time.sleep(0.02)
            return sentinel

        monkeypatch.setattr("setup_tuner.engine.nn_model.NNModelManager", _slow)

        barrier = threading.Barrier(8)
        results: list[Any] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            mgr = _get_nn_manager()
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

    def test_none_result_is_not_confused_with_failure_sentinel(self) -> None:
        """失败哨兵不得为 None，否则会与"未加载"混淆而反复重试。"""
        assert _LOAD_FAILED is not None
