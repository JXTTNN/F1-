"""Tests for the extended API router (f1opt.api.extended / extended_app).

Endpoints are exercised with ``httpx.AsyncClient`` over ``ASGITransport``. The
app is built once per module via ``create_extended_app()`` (UDP listener
disabled). The surrogate model is already trained on disk
(``data_store/models/segment_surrogate.pt``); the autouse fixture just resets
the model cache for deterministic predictions.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from f1opt.api.extended_app import create_extended_app
from f1opt.data.setup_schema import CarSetup


@pytest.fixture(scope="module", autouse=True)
def _ensure_model_ready() -> None:
    try:
        from f1opt.model.surrogate import reset_default_model_cache

        reset_default_model_cache()
    except Exception:
        pass


@pytest.fixture(scope="module")
def app() -> FastAPI:
    return create_extended_app()


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _lap(
    lap_time: float,
    s1: float,
    s2: float,
    s3: float,
    avg_speed: float | None = None,
    max_speed: float | None = None,
) -> dict[str, Any]:
    lap: dict[str, Any] = {"lap_time": lap_time, "sector_times": [s1, s2, s3]}
    if avg_speed is not None:
        lap["avg_speed"] = avg_speed
    if max_speed is not None:
        lap["max_speed"] = max_speed
    return lap


# --------------------------------------------------------------------------- #
# POST /api/bayesian-search
# --------------------------------------------------------------------------- #
async def test_bayesian_search_valid_track(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/bayesian-search",
            json={"track_id": "melbourne", "n_iterations": 5, "seed": 42},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "recommended_setup" in body
    assert isinstance(body["recommended_setup"], dict)


async def test_bayesian_search_recommended_setup_is_carsetup_dict(
    app: FastAPI,
) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/bayesian-search",
            json={"track_id": "melbourne", "n_iterations": 4, "seed": 7},
        )
    assert r.status_code == 200
    setup = r.json()["recommended_setup"]
    assert isinstance(setup, dict)
    CarSetup(**setup)  # reconstructs a valid CarSetup without error


async def test_bayesian_search_invalid_track_graceful(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/bayesian-search",
            json={"track_id": "nonexistent_track", "n_iterations": 3},
        )
    # Graceful: surrogate fallback keeps it 200 (422 acceptable per contract).
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert "recommended_setup" in r.json()


async def test_bayesian_search_n_iterations_3(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/bayesian-search",
            json={"track_id": "melbourne", "n_iterations": 3},
        )
    assert r.status_code == 200, r.text
    assert r.json()["iterations"] == 3


async def test_bayesian_search_has_predicted_gain(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/bayesian-search",
            json={"track_id": "monaco", "n_iterations": 3},
        )
    assert r.status_code == 200
    assert isinstance(r.json()["predicted_gain_s"], float)


async def test_bayesian_search_invalid_acquisition_returns_400(
    app: FastAPI,
) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/bayesian-search",
            json={"track_id": "melbourne", "n_iterations": 3, "acquisition": "nope"},
        )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# POST /api/pareto-search
# --------------------------------------------------------------------------- #
async def test_pareto_search(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/pareto-search",
            json={"track_id": "melbourne", "n_iterations": 3, "seed": 42},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "pareto_front_size" in body
    assert isinstance(body["pareto_front_size"], int)


async def test_pareto_search_best_lap_setup_is_dict(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/pareto-search",
            json={"track_id": "melbourne", "n_iterations": 3},
        )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["best_lap_time_setup"], dict)
    CarSetup(**body["best_lap_time_setup"])
    assert isinstance(body["best_tire_wear_setup"], dict)
    assert isinstance(body["knee_setup"], dict)


async def test_pareto_search_has_history(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/pareto-search",
            json={"track_id": "suzuka", "n_iterations": 3},
        )
    assert r.status_code == 200
    assert isinstance(r.json()["history"], list)


# --------------------------------------------------------------------------- #
# POST /api/strategy/plan
# --------------------------------------------------------------------------- #
async def test_strategy_plan(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/strategy/plan",
            json={"track_id": "melbourne", "total_laps": 30, "fuel_load_kg": 80.0},
        )
    assert r.status_code == 200, r.text
    assert "strategy_type" in r.json()


async def test_strategy_plan_total_laps_zero(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/strategy/plan",
            json={"track_id": "melbourne", "total_laps": 0, "fuel_load_kg": 80.0},
        )
    # 0 is valid (ge=0) -> 200 with an empty plan (422 also acceptable).
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert "strategy_type" in r.json()


async def test_strategy_plan_recommendation_reason_nonempty(
    app: FastAPI,
) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/strategy/plan",
            json={"track_id": "monza", "total_laps": 25, "fuel_load_kg": 75.0},
        )
    assert r.status_code == 200
    reason = r.json()["recommendation_reason"]
    assert isinstance(reason, str) and reason.strip() != ""


async def test_strategy_plan_available_compounds(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/strategy/plan",
            json={
                "track_id": "spa",
                "total_laps": 20,
                "fuel_load_kg": 70.0,
                "available_compounds": ["medium", "hard"],
            },
        )
    assert r.status_code == 200
    assert r.json()["strategy_type"] in ("0-stop", "1-stop", "2-stop")


async def test_strategy_plan_negative_laps_rejected(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post(
            "/api/strategy/plan",
            json={"track_id": "melbourne", "total_laps": -1, "fuel_load_kg": 80.0},
        )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# POST /api/compare/laps
# --------------------------------------------------------------------------- #
async def test_compare_laps_with_reference_and_3(app: FastAPI) -> None:
    ref = _lap(90.0, 30.0, 30.0, 30.0)
    laps = [
        _lap(91.0, 30.5, 30.0, 30.5),
        _lap(89.5, 29.5, 30.0, 30.0),
        _lap(90.5, 30.0, 30.5, 30.0),
    ]
    async with _client(app) as client:
        r = await client.post(
            "/api/compare/laps", json={"reference_lap": ref, "laps": laps}
        )
    assert r.status_code == 200, r.text
    assert len(r.json()["comparisons"]) == 3


async def test_compare_laps_empty(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post("/api/compare/laps", json={"laps": []})
    assert r.status_code == 200
    body = r.json()
    assert body["comparisons"] == []
    assert body["ranking"] == []
    assert body["best_lap_index"] is None


async def test_compare_laps_ranking_sorted_by_lap_time(app: FastAPI) -> None:
    laps = [
        _lap(92.0, 30, 31, 31),
        _lap(89.0, 29, 30, 30),
        _lap(90.5, 30, 30, 30.5),
    ]
    async with _client(app) as client:
        r = await client.post("/api/compare/laps", json={"laps": laps})
    assert r.status_code == 200
    times = [e["lap_time"] for e in r.json()["ranking"]]
    assert times == sorted(times)


async def test_compare_laps_best_lap_index(app: FastAPI) -> None:
    laps = [
        _lap(92.0, 30, 31, 31),
        _lap(89.0, 29, 30, 30),
        _lap(90.5, 30, 30, 30.5),
    ]
    async with _client(app) as client:
        r = await client.post("/api/compare/laps", json={"laps": laps})
    assert r.status_code == 200
    # 89.0 is fastest and was originally at index 1.
    assert r.json()["best_lap_index"] == 1


# --------------------------------------------------------------------------- #
# POST /api/compare/teammates
# --------------------------------------------------------------------------- #
async def test_compare_teammates(app: FastAPI) -> None:
    driver = [_lap(90.0, 30, 30, 30), _lap(89.5, 29.5, 30, 30)]
    teammate = [_lap(90.3, 30, 30.2, 30.1), _lap(90.1, 30, 30, 30.1)]
    async with _client(app) as client:
        r = await client.post(
            "/api/compare/teammates",
            json={"driver_laps": driver, "teammate_laps": teammate},
        )
    assert r.status_code == 200, r.text
    assert "gap_best_s" in r.json()


# --------------------------------------------------------------------------- #
# POST /api/narrate
# --------------------------------------------------------------------------- #
async def test_narrate_with_dimensions(app: FastAPI) -> None:
    dims = [
        {
            "name": "balance",
            "value": "understeer",
            "evidence": "data",
            "advice": "增加前翼",
        },
        {
            "name": "lap_time_potential",
            "value": "~0.5s above reference",
            "evidence": "",
            "advice": "",
        },
    ]
    async with _client(app) as client:
        r = await client.post("/api/narrate", json={"dimensions": dims})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "narration" in body and "summary" in body
    assert isinstance(body["narration"], str)
    assert isinstance(body["summary"], str)
    assert body["narration"] != ""
    assert body["summary"] != ""


async def test_narrate_empty(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.post("/api/narrate", json={"dimensions": []})
    assert r.status_code == 200
    assert r.json()["narration"] == ""


async def test_narrate_with_archetype(app: FastAPI) -> None:
    dims = [
        {
            "name": "balance",
            "value": "oversteer",
            "evidence": "",
            "advice": "增加后翼",
        }
    ]
    async with _client(app) as client:
        r = await client.post(
            "/api/narrate",
            json={"dimensions": dims, "archetype": "AGGRESSIVE"},
        )
    assert r.status_code == 200
    assert r.json()["narration"] != ""


# --------------------------------------------------------------------------- #
# GET /api/weather/impact
# --------------------------------------------------------------------------- #
async def test_weather_impact_dry(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.get(
            "/api/weather/impact", params={"precipitation_mm": 0}
        )
    assert r.status_code == 200, r.text
    assert r.json()["grip_multiplier"] == pytest.approx(1.0, abs=0.01)


async def test_weather_impact_wet(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.get(
            "/api/weather/impact", params={"precipitation_mm": 8}
        )
    assert r.status_code == 200
    assert r.json()["grip_multiplier"] < 1.0


async def test_weather_impact_setup_adjustments_list(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.get(
            "/api/weather/impact", params={"precipitation_mm": 8}
        )
    assert r.status_code == 200
    adj = r.json()["setup_adjustments"]
    assert isinstance(adj, list)
    assert len(adj) > 0


async def test_weather_impact_compound_wet(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.get(
            "/api/weather/impact", params={"precipitation_mm": 8}
        )
    assert r.status_code == 200
    assert r.json()["compound_recommendation"] == "wet"


# --------------------------------------------------------------------------- #
# GET /api/health/extended
# --------------------------------------------------------------------------- #
async def test_health_extended(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.get("/api/health/extended")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "modules_loaded" in body
    assert isinstance(body["modules_loaded"], list)
    assert len(body["modules_loaded"]) > 0


async def test_health_extended_model_available_bool(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.get("/api/health/extended")
    assert r.status_code == 200
    assert isinstance(r.json()["model_available"], bool)


async def test_health_extended_has_version_and_test_count(app: FastAPI) -> None:
    async with _client(app) as client:
        r = await client.get("/api/health/extended")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
    assert isinstance(body["test_count_estimate"], int)


# --------------------------------------------------------------------------- #
# Meta: no 500 errors on valid input across endpoints
# --------------------------------------------------------------------------- #
async def test_no_500_errors_on_valid_input(app: FastAPI) -> None:
    async with _client(app) as client:
        post_calls = [
            ("/api/bayesian-search", {"track_id": "melbourne", "n_iterations": 3}),
            ("/api/pareto-search", {"track_id": "melbourne", "n_iterations": 3}),
            (
                "/api/strategy/plan",
                {"track_id": "melbourne", "total_laps": 20, "fuel_load_kg": 70.0},
            ),
            ("/api/compare/laps", {"laps": [_lap(90, 30, 30, 30)]}),
            (
                "/api/compare/teammates",
                {
                    "driver_laps": [_lap(90, 30, 30, 30)],
                    "teammate_laps": [_lap(91, 30, 30, 31)],
                },
            ),
            (
                "/api/narrate",
                {
                    "dimensions": [
                        {"name": "grip", "value": "良好", "evidence": "", "advice": ""}
                    ]
                },
            ),
        ]
        for path, payload in post_calls:
            resp = await client.post(path, json=payload)
            assert resp.status_code < 500, (
                f"POST {path} returned {resp.status_code}: {resp.text}"
            )
        for path in ("/api/weather/impact", "/api/health/extended"):
            resp = await client.get(path)
            assert resp.status_code < 500, (
                f"GET {path} returned {resp.status_code}: {resp.text}"
            )
