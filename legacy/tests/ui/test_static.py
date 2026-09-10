"""Tests for the static UI mount on the FastAPI app.

Verifies that ``f1opt/ui/static/index.html`` is served at ``/`` by the
``StaticFiles`` mount added in :mod:`f1opt.api.app`, and that the mount does
NOT shadow the ``/api`` REST routes.
"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from f1opt.api.app import create_app


def _app() -> FastAPI:
    """A fresh app with the UDP listener disabled (no port binding)."""
    return create_app(start_listener=False)


async def test_root_serves_index_html() -> None:
    """GET / returns 200 and serves the dashboard HTML."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/")
        assert r.status_code == 200
        assert "F1 2026 调教优化系统" in r.text
        # The 调教搜索 button (added in Iter-02 Task 2.4) is rendered.
        assert "调教搜索" in r.text
        # The single-file bundle ships its own JS+CSS (no external CDN).
        assert "<script" in r.text
        assert "<style" in r.text


async def test_api_health_not_shadowed_by_static() -> None:
    """Static mount at / must not shadow /api/health."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"]


async def test_static_assets_not_at_api_path() -> None:
    """A non-existent static path 404s without touching /api."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/does-not-exist.xyz")
        # StaticFiles returns 404 for missing files; the API routes are intact.
        assert r.status_code == 404


async def test_root_has_iteration_history_and_export() -> None:
    """The dashboard ships the Iter-03 UI additions: an iteration-history panel
    (迭代历史) and a setup export button (导出调教)."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/")
        assert r.status_code == 200
        assert "F1 2026 调教优化系统" in r.text
        assert "迭代历史" in r.text
        assert "导出调教" in r.text
        # Iteration-history panel + export wiring are present in the bundle.
        assert 'id="iter-list"' in r.text
        assert 'id="iter-detail"' in r.text
        assert 'id="export-setup-btn"' in r.text
        # JS helpers backing the new features.
        assert "setupToGameFormat" in r.text
        assert "loadIterations" in r.text
        assert "f1-25-setup-v1" in r.text


async def test_root_has_raf_throttling_and_samples_export() -> None:
    """Iter-13/14 UI additions: Parquet samples export button + rAF render
    throttling for the 60Hz telemetry stream."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/")
        assert r.status_code == 200
        # Iter-13: samples export button + handler.
        assert 'id="export-samples-btn"' in r.text
        assert "/api/samples/parquet" in r.text
        # Iter-14: requestAnimationFrame render throttling for telemetry frames.
        assert "requestAnimationFrame(renderPendingFrame)" in r.text
        assert "pendingRenderFrame" in r.text
        assert "renderScheduled" in r.text

