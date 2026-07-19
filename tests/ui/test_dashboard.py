"""Tests for the intelligence dashboard HTML and its wiring to the extended API.

Verifies that ``f1opt/ui/static/dashboard.html`` is served by the extended
app's ``StaticFiles`` mount and that it exposes every extended API endpoint
(Bayesian / Pareto search, strategy planning, lap + teammate comparison,
weather impact, extended health). The dashboard ships its own embedded CSS +
JS with no external CDN, so the assertions check the rendered HTML payload.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from f1opt.api.extended_app import create_extended_app

_DASHBOARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "f1opt"
    / "ui"
    / "static"
    / "dashboard.html"
)


def _app() -> FastAPI:
    """A fresh extended app with the UDP listener disabled (no port binding)."""
    return create_extended_app(start_listener=False)


def _dashboard_html() -> str:
    """Read the dashboard HTML directly from disk (fallback for content tests)."""
    return _DASHBOARD_PATH.read_text(encoding="utf-8")


async def _get(client: AsyncClient, path: str) -> str:
    r = await client.get(path)
    assert r.status_code == 200, f"GET {path} -> {r.status_code}"
    return r.text


# --------------------------------------------------------------------------- #
# Served via StaticFiles mount
# --------------------------------------------------------------------------- #
async def test_dashboard_html_is_served_at_root_path() -> None:
    """GET /dashboard.html returns 200 and the intelligence dashboard HTML."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        text = await _get(client, "/dashboard.html")
        assert "F1 2026 智能分析中心" in text


async def test_dashboard_html_does_not_shadow_api_routes() -> None:
    """Serving /dashboard.html must not shadow the extended /api routes."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/health/extended")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


# --------------------------------------------------------------------------- #
# Header + tabs
# --------------------------------------------------------------------------- #
async def test_dashboard_has_title_and_version_badge() -> None:
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        text = await _get(client, "/dashboard.html")
    assert "F1 2026 智能分析中心" in text
    assert 'id="version-badge"' in text
    assert "智能分析中心" in text


async def test_dashboard_has_all_five_nav_tabs() -> None:
    """All five navigation tabs are rendered in the header nav."""
    text = _dashboard_html()
    for name in ["赛道策略", "调教搜索", "圈速对比", "天气影响", "健康状态"]:
        assert name in text, f"missing tab name: {name}"
    # Tab buttons carry the data-tab attribute used by the active-state logic.
    assert 'data-tab="strategy"' in text
    assert 'data-tab="search"' in text
    assert 'data-tab="compare"' in text
    assert 'data-tab="weather"' in text
    assert 'data-tab="health"' in text


async def test_dashboard_has_embedded_script_and_style() -> None:
    """The single-file bundle ships its own JS+CSS (no external CDN)."""
    app = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        text = await _get(client, "/dashboard.html")
    assert "<script" in text
    assert "</script>" in text
    assert "<style" in text
    assert "</style>" in text
    # No external CDN dependency.
    assert "https://cdn" not in text
    assert "https://unpkg.com" not in text


# --------------------------------------------------------------------------- #
# Extended API endpoint wiring (every fetch URL must appear in the bundle)
# --------------------------------------------------------------------------- #
async def test_dashboard_calls_strategy_plan_endpoint() -> None:
    text = _dashboard_html()
    assert "/api/strategy/plan" in text


async def test_dashboard_calls_bayesian_search_endpoint() -> None:
    text = _dashboard_html()
    assert "/api/bayesian-search" in text


async def test_dashboard_calls_pareto_search_endpoint() -> None:
    text = _dashboard_html()
    assert "/api/pareto-search" in text


async def test_dashboard_calls_compare_laps_endpoint() -> None:
    text = _dashboard_html()
    assert "/api/compare/laps" in text


async def test_dashboard_calls_compare_teammates_endpoint() -> None:
    text = _dashboard_html()
    assert "/api/compare/teammates" in text


async def test_dashboard_calls_weather_impact_endpoint() -> None:
    text = _dashboard_html()
    assert "/api/weather/impact" in text


async def test_dashboard_calls_health_extended_endpoint() -> None:
    text = _dashboard_html()
    assert "/api/health/extended" in text


# --------------------------------------------------------------------------- #
# JS feature surface
# --------------------------------------------------------------------------- #
async def test_dashboard_has_svg_chart_rendering() -> None:
    """SVG charts are built dynamically (createElementNS with the SVG namespace
    and table bodies selected via querySelector)."""
    text = _dashboard_html()
    assert "http://www.w3.org/2000/svg" in text
    assert "createElementNS" in text
    assert 'querySelector("tbody")' in text or "querySelector(" in text


async def test_dashboard_has_tab_active_state_logic() -> None:
    """The tabs use an `active` class toggled by an activateTab handler."""
    text = _dashboard_html()
    assert "classList.toggle" in text
    assert 'classList.add("active"' in text or 'classList.toggle("active"' in text
    assert "activateTab" in text


async def test_dashboard_has_error_handling_toast() -> None:
    """Fetch failures surface a toast (no bare alert())."""
    text = _dashboard_html()
    assert 'id="toast"' in text
    assert "showToast" in text
    # Avoid alert() — prefer the toast UI.
    assert "alert(" not in text


# --------------------------------------------------------------------------- #
# Bayesian search UI
# --------------------------------------------------------------------------- #
async def test_dashboard_has_bayesian_search_ui_elements() -> None:
    """Bayesian search exposes an acquisition-function <select> and an
    n_iterations <input type=range> slider."""
    text = _dashboard_html()
    assert 'id="bayes-acq"' in text
    assert 'id="bayes-iter"' in text
    assert 'type="range"' in text
    # Acquisition options cover ei / ucb / pi.
    assert 'value="ei"' in text
    assert 'value="ucb"' in text
    assert 'value="pi"' in text


async def test_dashboard_has_bayesian_seed_input() -> None:
    """A seed input is exposed for reproducible Bayesian searches."""
    text = _dashboard_html()
    assert 'id="bayes-seed"' in text


# --------------------------------------------------------------------------- #
# Pareto search UI
# --------------------------------------------------------------------------- #
async def test_dashboard_has_pareto_scatter_plot_element() -> None:
    """A dedicated <svg> element is reserved for the Pareto-front scatter."""
    text = _dashboard_html()
    assert 'id="pareto-scatter"' in text
    assert "renderScatter" in text


async def test_dashboard_has_pareto_subtab_switching() -> None:
    """Search tab exposes Bayesian / Pareto sub-tabs with active state."""
    text = _dashboard_html()
    assert 'data-subtab="bayes"' in text
    assert 'data-subtab="pareto"' in text
    assert "activateSubtab" in text


# --------------------------------------------------------------------------- #
# Other tabs UI
# --------------------------------------------------------------------------- #
async def test_dashboard_has_strategy_form_inputs() -> None:
    """Strategy tab exposes track / total_laps / fuel_load inputs and the
    compounds checkbox group."""
    text = _dashboard_html()
    assert 'id="str-track"' in text
    assert 'id="str-laps"' in text
    assert 'id="str-fuel"' in text
    assert 'id="str-compounds"' in text
    # All five compounds are selectable.
    for c in ["soft", "medium", "hard", "intermediate", "wet"]:
        assert f'value="{c}"' in text


async def test_dashboard_has_compare_textareas() -> None:
    """Comparison tab has a laps JSON textarea plus teammate textareas."""
    text = _dashboard_html()
    assert 'id="cmp-input"' in text
    assert 'id="tm-driver"' in text
    assert 'id="tm-teammate"' in text


async def test_dashboard_has_weather_sliders() -> None:
    """Weather tab has sliders for ambient / track temp, humidity,
    precipitation and wind speed."""
    text = _dashboard_html()
    for i in ["w-amb", "w-trk", "w-hum", "w-pre", "w-wnd"]:
        assert f'id="{i}"' in text
    assert "renderRadialGauge" in text


async def test_dashboard_has_health_refresh_button_and_modules_target() -> None:
    """Health tab exposes a refresh button and a modules-loaded chip target."""
    text = _dashboard_html()
    assert 'id="h-refresh"' in text
    assert 'id="h-modules"' in text
    assert 'id="h-status-val"' in text


# --------------------------------------------------------------------------- #
# fetchJSON helper
# --------------------------------------------------------------------------- #
async def test_dashboard_has_fetchjson_helper() -> None:
    """A shared fetchJSON helper centralises all API calls."""
    text = _dashboard_html()
    assert "async function fetchJSON" in text
    assert "function showToast" in text
