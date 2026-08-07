"""Extended API router for F1OPT.

Adds Bayesian/Pareto setup search, race-strategy planning, lap & teammate
comparison, NLG narration, and weather-impact endpoints on top of the core
:mod:`f1opt.api.app` without modifying it. Mounted via
:func:`f1opt.api.extended_app.create_extended_app`.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from f1opt import __version__
from f1opt.data.setup_schema import DEFAULT_SETUP, CarSetup
from f1opt.data.track_evolution import WeatherCondition, WeatherImpactModel
from f1opt.feedback.comparison import LapComparator, TeammateComparison
from f1opt.feedback.nlg import FeedbackNarrator, ToneAdapter
from f1opt.model.bayesian import bayesian_search_setup
from f1opt.model.pareto import MultiObjectiveOptimizer
from f1opt.model.strategy import RaceStrategyPlanner

__all__ = ["router"]

#: Normalized CarSetup search space (21 dims, each in [0, 1]).
_SETUP_BOUNDS: list[list[float]] = [[0.0, 1.0]] * 21

#: Approximate total test count reported by /api/health/extended.
_TEST_COUNT_ESTIMATE = 837

_VALID_ACQUISITIONS = ("ei", "ucb", "pi")


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
class BayesianSearchRequest(BaseModel):
    """``POST /api/bayesian-search`` body."""

    track_id: str
    baseline_setup: dict[str, Any] | None = None
    driver_profile: dict[str, Any] | None = None
    n_iterations: int = 15
    acquisition: str = "ei"
    seed: int = 42


class ParetoSearchRequest(BaseModel):
    """``POST /api/pareto-search`` body."""

    track_id: str
    objectives: list[str] = Field(
        default_factory=lambda: ["lap_time", "tire_wear"]
    )
    n_iterations: int = 20
    seed: int = 42


class StrategyPlanRequest(BaseModel):
    """``POST /api/strategy/plan`` body."""

    track_id: str
    total_laps: int = Field(ge=0)
    fuel_load_kg: float = Field(ge=0)
    available_compounds: list[str] = Field(
        default_factory=lambda: ["soft", "medium", "hard"]
    )


class CompareLapsRequest(BaseModel):
    """``POST /api/compare/laps`` body."""

    reference_lap: dict[str, Any] | None = None
    laps: list[dict[str, Any]] = Field(default_factory=list)


class CompareTeammatesRequest(BaseModel):
    """``POST /api/compare/teammates`` body."""

    driver_laps: list[dict[str, Any]]
    teammate_laps: list[dict[str, Any]]


class NarrateRequest(BaseModel):
    """``POST /api/narrate`` body."""

    dimensions: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "zh"
    archetype: str | None = None


class BatchFeedbackRequest(BaseModel):
    """``POST /api/feedback/batch`` body: 批量反馈处理 (Iter-183)."""

    sessions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of feedback sessions, each with frames, setup, track_id, question",
    )
    track_id: str = "monza"
    question: str | None = None


router = APIRouter()


def _setup_to_dict(setup: Any) -> dict[str, Any]:
    """Serialize a ``CarSetup`` (or pass through a dict) to a JSON-safe dict."""
    if isinstance(setup, CarSetup):
        return setup.model_dump()
    return dict(setup)


# --------------------------------------------------------------------------- #
# 1. Bayesian setup search
# --------------------------------------------------------------------------- #
@router.post("/api/bayesian-search")
async def bayesian_search(body: BayesianSearchRequest) -> dict[str, Any]:
    """Bayesian-optimized setup search via the DNN surrogate objective."""
    if body.acquisition not in _VALID_ACQUISITIONS:
        raise HTTPException(
            status_code=400, detail=f"invalid acquisition: {body.acquisition}"
        )
    if body.baseline_setup is not None:
        try:
            baseline = CarSetup(**body.baseline_setup)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        baseline = DEFAULT_SETUP
    result = bayesian_search_setup(
        body.track_id,
        baseline=baseline,
        driver_profile=body.driver_profile,
        n_iterations=body.n_iterations,
        acquisition=body.acquisition,
        seed=body.seed,
    )
    result["recommended_setup"] = _setup_to_dict(result.get("recommended_setup"))
    return result


# --------------------------------------------------------------------------- #
# 2. Multi-objective Pareto search
# --------------------------------------------------------------------------- #
@router.post("/api/pareto-search")
async def pareto_search(body: ParetoSearchRequest) -> dict[str, Any]:
    """NSGA-II-style multi-objective Pareto setup search."""
    opt = MultiObjectiveOptimizer(
        _SETUP_BOUNDS,
        objectives=body.objectives,
        n_iterations=body.n_iterations,
        seed=body.seed,
    )
    res = opt.search(body.track_id)
    pareto_front = res["pareto_front"]
    front_size = len(pareto_front.compute_front())
    return {
        "pareto_front_size": front_size,
        "best_lap_time_setup": _setup_to_dict(res["best_lap_time_setup"]),
        "best_tire_wear_setup": _setup_to_dict(res["best_tire_wear_setup"]),
        "knee_setup": _setup_to_dict(res["knee_setup"]),
        "history": res["history"],
    }


# --------------------------------------------------------------------------- #
# 3. Race strategy planning
# --------------------------------------------------------------------------- #
@router.post("/api/strategy/plan")
async def strategy_plan(body: StrategyPlanRequest) -> dict[str, Any]:
    """Plan the optimal 0/1/2-stop race strategy for a track."""
    planner = RaceStrategyPlanner(
        body.track_id,
        total_laps=body.total_laps,
        fuel_load_kg=body.fuel_load_kg,
    )
    return planner.optimal_strategy(body.available_compounds)


# --------------------------------------------------------------------------- #
# 4. Lap comparison
# --------------------------------------------------------------------------- #
@router.post("/api/compare/laps")
async def compare_laps(body: CompareLapsRequest) -> dict[str, Any]:
    """Compare laps against an optional reference and rank by lap time."""
    comparator = LapComparator(body.reference_lap)
    comparisons = comparator.compare_multi(body.laps)
    ranked = comparator.rank_laps(body.laps)
    ranking = [
        {"index": i, "lap_time": lap.get("lap_time")} for i, lap in ranked
    ]
    best_lap_index = ranking[0]["index"] if ranking else None
    return {
        "comparisons": comparisons,
        "ranking": ranking,
        "best_lap_index": best_lap_index,
    }


# --------------------------------------------------------------------------- #
# 5. Teammate comparison
# --------------------------------------------------------------------------- #
@router.post("/api/compare/teammates")
async def compare_teammates(body: CompareTeammatesRequest) -> dict[str, Any]:
    """Driver vs teammate head-to-head comparison."""
    comp = TeammateComparison(body.driver_laps, body.teammate_laps)
    return comp.head_to_head()


# --------------------------------------------------------------------------- #
# 6. NLG narration
# --------------------------------------------------------------------------- #
@router.post("/api/narrate")
async def narrate(body: NarrateRequest) -> dict[str, str]:
    """Narrate structured feedback dimensions into natural-language prose."""
    narrator = FeedbackNarrator(language=body.language)
    narration = narrator.narrate_all(body.dimensions)
    if body.archetype and narration:
        narration = ToneAdapter(body.archetype).adapt(narration)
    summary = narrator.summarize_session(
        {"dimensions": body.dimensions, "setup_suggestions": []}
    )
    return {"narration": narration, "summary": summary}


# --------------------------------------------------------------------------- #
# 9. Batch feedback processing (Iter-183)
# --------------------------------------------------------------------------- #
@router.post("/api/feedback/batch")
async def feedback_batch(body: BatchFeedbackRequest) -> dict[str, Any]:
    """Iter-183: Process multiple feedback sessions in a single request.

    Accepts a list of session dicts, each with optional ``frames``, ``setup``,
    ``track_id``, and ``question``. Returns a list of feedback results with
    per-session timing. This is useful for batch processing lap data after a
    session.
    """
    from f1opt.feedback.engine import FeedbackEngine

    engine = FeedbackEngine()
    results: list[dict[str, Any]] = []
    total_time = 0.0
    start_all = time.time()

    for i, session in enumerate(body.sessions):
        frames = session.get("frames", [])
        setup = session.get("setup", DEFAULT_SETUP.model_dump())
        tid = session.get("track_id", body.track_id)
        q = session.get("question", body.question)

        t0 = time.perf_counter()
        try:
            result = engine.run(
                frames=frames,
                setup=setup,
                track_id=tid,
                question=q,
            )
            elapsed = time.perf_counter() - t0
            results.append({
                "index": i,
                "status": "success",
                "elapsed_s": round(elapsed, 3),
                "feedback": result,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append({
                "index": i,
                "status": "error",
                "elapsed_s": round(elapsed, 3),
                "error": str(exc),
            })
        total_time += elapsed

    return {
        "total_sessions": len(body.sessions),
        "results": results,
        "total_elapsed_s": round(time.perf_counter() - start_all, 3),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
    }


# --------------------------------------------------------------------------- #
# 7. Weather impact
# --------------------------------------------------------------------------- #
@router.get("/api/weather/impact")
async def weather_impact(
    ambient_temp_c: float = Query(25.0, ge=-30.0, le=60.0),
    track_temp_c: float = Query(30.0, ge=-30.0, le=90.0),
    humidity_pct: float = Query(50.0, ge=0.0, le=100.0),
    precipitation_mm: float = Query(0.0, ge=0.0),
    wind_speed_ms: float = Query(0.0, ge=0.0),
) -> dict[str, Any]:
    """Analyse weather impact on grip, lap time, compound and setup."""
    weather = WeatherCondition(
        ambient_temp_c=ambient_temp_c,
        track_temp_c=track_temp_c,
        humidity_pct=humidity_pct,
        precipitation_mm=precipitation_mm,
        wind_speed_ms=wind_speed_ms,
    )
    model = WeatherImpactModel()
    return {
        "grip_multiplier": model.grip_multiplier(weather),
        "lap_time_delta": model.lap_time_delta(weather, 90.0),
        "compound_recommendation": weather.compound_recommendation(),
        "setup_adjustments": model.setup_adjustment_recommendations(
            weather, "medium"
        ),
    }


# --------------------------------------------------------------------------- #
# 8. Extended health check
# --------------------------------------------------------------------------- #
@router.get("/api/health/extended")
async def health_extended() -> dict[str, Any]:
    """Extended health check: loaded modules + surrogate availability."""
    candidate_modules = [
        "f1opt.model.bayesian",
        "f1opt.model.pareto",
        "f1opt.model.strategy",
        "f1opt.feedback.comparison",
        "f1opt.feedback.nlg",
        "f1opt.data.track_evolution",
    ]
    modules_loaded: list[str] = []
    for name in candidate_modules:
        try:
            __import__(name)
            modules_loaded.append(name)
        except ImportError:
            pass
    model_available = False
    model_version = "unknown"
    try:
        from f1opt.model.surrogate import MODEL_VERSION, _get_default_model

        _get_default_model()
        model_available = True
        model_version = MODEL_VERSION
    except Exception:
        model_available = False
    return {
        "status": "ok",
        "version": __version__,
        "modules_loaded": modules_loaded,
        "model_available": bool(model_available),
        "model_version": model_version,
        "test_count_estimate": _TEST_COUNT_ESTIMATE,
    }
