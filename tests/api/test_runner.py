"""Tests for :mod:`f1opt.api.runner` (console entrypoint).

The runner boots uvicorn using the configured host/port. These tests
stub out ``uvicorn.run`` and structlog configuration so no server is
actually started, and assert the runner wires the settings through
correctly.
"""

from __future__ import annotations

from typing import Any

import pytest

from f1opt.api import runner


def test_run_invokes_uvicorn_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """run() configures logging and boots uvicorn with the app + settings."""
    captured: dict[str, Any] = {}

    def _fake_uvicorn_run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    configured: dict[str, bool] = {"called": False}

    def _fake_configure() -> None:
        configured["called"] = True

    monkeypatch.setattr(runner.uvicorn, "run", _fake_uvicorn_run)
    monkeypatch.setattr(runner, "configure_structlog", _fake_configure)

    runner.run()

    assert configured["called"] is True
    assert captured["app"] == "f1opt.api.app:app"

    settings = runner.get_settings()
    assert captured["kwargs"]["host"] == settings.api_host
    assert captured["kwargs"]["port"] == settings.api_port


def test_run_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host/port come from settings, which read F1OPT_API_* env vars."""
    captured: dict[str, Any] = {}

    monkeypatch.setenv("F1OPT_API_HOST", "0.0.0.0")
    monkeypatch.setenv("F1OPT_API_PORT", "9123")
    # get_settings is lru_cache'd; clear so the env override is picked up.
    runner.get_settings.cache_clear()

    monkeypatch.setattr(runner, "configure_structlog", lambda: None)
    monkeypatch.setattr(
        runner.uvicorn,
        "run",
        lambda app, **kwargs: captured.update(app=app, **kwargs),
    )

    try:
        runner.run()
        assert captured["host"] == "0.0.0.0"
        assert captured["port"] == 9123
    finally:
        runner.get_settings.cache_clear()
