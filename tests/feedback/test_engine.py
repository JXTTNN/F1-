"""Tests for :mod:`f1opt.feedback.engine` (Iter-03, 10-dimension coverage).

Builds a synthetic 600-frame (10s @ 60Hz) telemetry window with scripted values
to trigger:

- understeer (high steer + low g_lat mid-corner),
- a rear-tyre-wear imbalance (oversteer indicator),
- a brake lockup pattern (brake=1.0, speed barely dropping), and
- steering corrections (oscillating steer in a corner window).

Verifies the public contract:

- ``generate_feedback`` returns ``{summary, dimensions, setup_suggestions,
  sources}`` with ``dimensions`` covering ALL 10 ``FEEDBACK_DIMENSIONS`` in
  order.
- Each dimension carries evidence; numeric claims trace to ``sources``.
- The ``setup_advice`` dimension consumes ``search_setup`` (predicted_gain /
  param names appear in its advice).
- The ``braking`` dimension flags lockup risk (advice mentions 锁死/lockup).
- The driver's Chinese question is answered (evidence-grounded).
- Empty frames / invalid track_id degrade gracefully (10 dims still emitted).
- The LLM path is gated by ``config.llm_backend`` + API key (no network by
  default; falls back on missing key / network error).

Performance note: ``setup_advice`` calls ``search_setup(iterations=25, seed=42)``
which trains/loads the cached surrogate (~30ms). The whole module stays well
under 10s; no mocking is required.
"""

from __future__ import annotations

import math
import re

import pytest

from f1opt.config import Settings
from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.driver.profile import AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE
from f1opt.feedback import (
    FEEDBACK_DIMENSIONS,
    FeedbackEngine,
    generate_feedback,
    get_session,
    reset_sessions,
)
from f1opt.feedback.engine import extract_metrics, llm_enhance
from f1opt.feedback.prompts import USER_PROMPT_TEMPLATE, format_driver_profile


# --------------------------------------------------------------------------- #
# Synthetic frame factory (unified 60Hz aligner frame keys)
# --------------------------------------------------------------------------- #
def _frame(i: int, **overrides: float) -> dict[str, float]:
    t = i / 60.0
    f: dict[str, float] = {
        "session_time": t,
        "speed": 250.0 + 5.0 * (i % 60),
        "throttle": 0.8,
        "brake": 0.0,
        "steer": 0.0,
        "gear": 6,
        "rpm": 9000,
        "tyre_temp_fl": 90.0,
        "tyre_temp_fr": 91.0,
        "tyre_temp_rl": 92.0,
        "tyre_temp_rr": 93.0,
        "g_lat": 0.0,
        "g_long": 0.0,
        "g_vert": 1.0,
        "world_x": 0.0,
        "world_y": 0.0,
        "world_z": 0.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "velocity_z": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "ers_store": 1_000_000.0,
        "ers_deploy_mode": 0,
        "drs_allowed": 0,
        "fuel_in_tank": 30.0,
        "fuel_remaining_laps": 5.0,
        # Running lap timer: 86.5s -> 96.5s over the 10s window (last = lap time).
        "lap_time": 86.5 + t,
        "lap_distance": float(i),
        "tyre_wear_fl": 5.0,
        "tyre_wear_fr": 5.0,
        "tyre_wear_rl": 15.0,
        "tyre_wear_rr": 16.0,
    }
    f.update(overrides)
    return f


def _scripted_frames() -> list[dict[str, float]]:
    """600 frames with understeer + tyre-wear imbalance + lockup + corrections.

    - 100..160: brake lockup (brake=1.0, speed barely dropping).
    - 200..400: understeer (high steer + low g_lat mid-corner).
    - 400..500: steering corrections (oscillating low-amplitude steer).
    """
    frames: list[dict[str, float]] = []
    for i in range(600):
        if 100 <= i < 160:
            frames.append(
                _frame(i, brake=1.0, speed=200.0 - 0.01 * (i - 100), throttle=0.0)
            )
        elif 200 <= i < 400:
            frames.append(_frame(i, steer=0.8, g_lat=1.0, brake=0.0, throttle=0.5))
        elif 400 <= i < 500:
            s = 0.3 * math.sin(2 * math.pi * (i - 400) / 20.0)
            g = 1.5 + 0.8 * math.sin(2 * math.pi * (i - 400) / 20.0)
            frames.append(_frame(i, steer=s, g_lat=g, throttle=0.6))
        else:
            frames.append(_frame(i))
    return frames


def _lockup_frames() -> list[dict[str, float]]:
    """Dedicated lockup pattern: full brake, speed essentially constant."""
    frames: list[dict[str, float]] = []
    for i in range(120):
        frames.append(_frame(i, brake=1.0, speed=200.0, throttle=0.0))
    return frames


def _full_lap_frames() -> list[dict[str, float]]:
    """Frames whose lap_distance spans a full lap (sector boundaries crossable)."""
    frames: list[dict[str, float]] = []
    for i in range(600):
        frames.append(_frame(i, lap_distance=float(i) * 10.0))
    return frames


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _construct_with(setup_dict: dict, name: str, after: float) -> CarSetup:
    """Construct CarSetup overriding one field; raises if ``after`` is invalid."""
    spec = SETUP_FIELDS[name]
    typed = int(round(after)) if spec.kind == "int" else float(after)
    return CarSetup(**{**setup_dict, name: typed})


def _dim_by_name(out: dict, name: str) -> dict:
    return next(d for d in out["dimensions"] if d["name"] == name)


# --------------------------------------------------------------------------- #
# FEEDBACK_DIMENSIONS constant
# --------------------------------------------------------------------------- #
def test_feedback_dimensions_constant_has_16_entries() -> None:
    assert len(FEEDBACK_DIMENSIONS) == 19  # Iter-256: +active_aero_usage
    assert FEEDBACK_DIMENSIONS == [
        "balance",
        "grip",
        "tyres",
        "braking",
        "ers_deployment",
        "drs_usage",
        "throttle_brake_smoothness",
        "confidence",
        "lap_time_potential",
        "sector_compare",
        "setup_advice",
        "corner_analysis",  # Iter-164.14
        "fuel_consumption",  # Iter-203
        "throttle_brake_overlap",  # Iter-210
        "aero_balance",  # Iter-214
        "brake_temp",  # Iter-222
        "tyre_temp_gradient",  # Iter-227
        "grip_consistency",  # Iter-241
        "active_aero_usage",  # Iter-256
    ]


# --------------------------------------------------------------------------- #
# Contract shape
# --------------------------------------------------------------------------- #
def test_generate_feedback_returns_contract_shape() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    assert set(out.keys()) == {"summary", "dimensions", "setup_suggestions", "sources", "_quality"}
    assert isinstance(out["summary"], str) and out["summary"]
    assert isinstance(out["dimensions"], list)
    assert isinstance(out["setup_suggestions"], list)
    assert isinstance(out["sources"], list)


def test_dimensions_all_10_names_in_order() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    names = [d["name"] for d in out["dimensions"]]
    assert names == FEEDBACK_DIMENSIONS
    assert len(names) == 19  # Iter-256: +active_aero_usage
    # Each dimension has the required keys.
    for d in out["dimensions"]:
        assert set(d.keys()) == {"name", "value", "evidence", "advice"}


def test_dimensions_include_lap_time_potential() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    ltp = _dim_by_name(out, "lap_time_potential")
    assert ltp["value"]
    assert ltp["evidence"]


def test_active_aero_usage_dimension() -> None:
    """Iter-256: 主动空力使用维度按 X/Z-Mode 帧占比产出量化结论。"""
    frames = []
    for i in range(200):
        # 前 60 帧 X-Mode 激活 (低阻直道), 后 140 帧 Z-Mode 激活 (高下压弯道)
        if i < 60:
            frames.append(_frame(i, active_aero_x=1.0, active_aero_z=0.0))
        else:
            frames.append(_frame(i, active_aero_x=0.0, active_aero_z=1.0))
    out = generate_feedback(frames, DEFAULT_SETUP.model_dump(), "monza")
    dim = _dim_by_name(out, "active_aero_usage")
    assert dim["name"] == "active_aero_usage"
    assert "X-Mode 30%" in dim["value"]
    assert "Z-Mode 70%" in dim["value"]
    assert dim["advice"]


def test_active_aero_usage_insufficient_when_absent() -> None:
    """Iter-256: 无主动空力数据时该维度返回 数据不足 而不崩溃。"""
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    dim = _dim_by_name(out, "active_aero_usage")
    assert dim["value"] == "数据不足"


def test_at_least_one_setup_suggestion_with_valid_after() -> None:
    setup = DEFAULT_SETUP.model_dump()
    out = generate_feedback(_scripted_frames(), setup, "melbourne")
    assert len(out["setup_suggestions"]) >= 1
    for s in out["setup_suggestions"]:
        assert {"name", "before", "after", "unit", "expected_gain", "reason"} <= set(s.keys())
        assert s["name"] in SETUP_FIELDS
        spec = SETUP_FIELDS[s["name"]]
        assert spec.min <= s["after"] <= spec.max
        # Constructing a CarSetup with the suggested field must not raise.
        _construct_with(setup, s["name"], s["after"])
        # before must equal the current setup value.
        assert s["before"] == pytest.approx(float(setup[s["name"]]))


def test_sources_non_empty_and_well_formed() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    assert len(out["sources"]) > 0
    for s in out["sources"]:
        assert set(s.keys()) == {"frame_t", "field", "value"}
        assert isinstance(s["frame_t"], float)
        assert isinstance(s["field"], str) and s["field"]
        assert isinstance(s["value"], (int, float))


def test_sources_back_reason_frame_refs() -> None:
    """Every ``t=X.XXs`` reference in a suggestion reason appears in sources."""
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    source_ts = {round(s["frame_t"], 2) for s in out["sources"]}
    for s in out["setup_suggestions"]:
        for m in re.finditer(r"t=([0-9]+\.[0-9]+)s", s["reason"]):
            assert round(float(m.group(1)), 2) in source_ts, (
                f"reason references t={m.group(1)}s not present in sources"
            )


def test_dimension_evidence_frame_refs_trace_to_sources() -> None:
    """For 3 telemetry-derived dims, every ``t=X.XXs`` in evidence is in sources
    and evidence is non-empty when data is available."""
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    source_ts = {round(s["frame_t"], 2) for s in out["sources"]}
    checked = 0
    for name in ("balance", "tyres", "braking"):
        dim = _dim_by_name(out, name)
        assert dim["evidence"], f"{name} evidence empty with data available"
        for m in re.finditer(r"t=([0-9]+\.[0-9]+)s", dim["evidence"]):
            assert round(float(m.group(1)), 2) in source_ts, (
                f"{name} evidence references t={m.group(1)}s not in sources"
            )
        checked += 1
    assert checked == 3


# --------------------------------------------------------------------------- #
# Metric extraction (direct)
# --------------------------------------------------------------------------- #
def test_extract_metrics_understeer_triggered() -> None:
    metrics = extract_metrics(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    u = metrics["values"].get("understeer_indicator")
    assert u is not None and u > 0.4
    # Evidence refs for understeer must be present.
    assert "understeer_steer" in metrics["refs"]
    assert "understeer_g_lat" in metrics["refs"]
    assert metrics["sources"], "sources must be non-empty for scripted frames"


def test_extract_metrics_tyre_wear_imbalance() -> None:
    metrics = extract_metrics(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    twb = metrics["values"]["tyre_wear_balance"]
    assert twb["rear_avg"] > twb["front_avg"]
    assert metrics["values"]["oversteer_indicator"] > 0.4


def test_extract_metrics_lockup_and_confidence_computed() -> None:
    metrics = extract_metrics(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    assert metrics["values"]["lockup_proxy"] > 0.3
    assert "steering_correction_freq" in metrics["values"]
    assert "g_lat_stability" in metrics["values"]
    assert "confidence_score" in metrics["values"]
    assert "lockup_brake" in metrics["refs"]
    assert "steering_correction" in metrics["refs"]


def test_extract_metrics_sector_times_derivable_for_full_lap() -> None:
    metrics = extract_metrics(_full_lap_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    sectors = metrics["values"].get("sector_times")
    assert sectors is not None and len(sectors) == 3
    assert all(s > 0 for s in sectors)
    assert "sector_s1" in metrics["refs"]


# --------------------------------------------------------------------------- #
# Dimension-level behaviour
# --------------------------------------------------------------------------- #
def test_feedback_has_understeer_balance_dimension_and_front_wing_suggestion() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    balance = _dim_by_name(out, "balance")
    assert "understeer" in balance["value"]
    sug_names = {s["name"] for s in out["setup_suggestions"]}
    assert "front_wing" in sug_names


def test_setup_advice_dimension_references_model_result() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    sa = _dim_by_name(out, "setup_advice")
    advice = sa["advice"] or ""
    value = sa["value"] or ""
    # Advice is non-empty and either mentions predicted_gain or a setup param
    # name (or there are rule-based setup_suggestions as fallback).
    assert advice or out["setup_suggestions"]
    blob = advice + " " + value
    assert (
        "predicted_gain" in blob
        or any(s["name"] in blob for s in out["setup_suggestions"])
        or any(name in blob for name in SETUP_FIELDS)
    ), f"setup_advice does not reference model result: {blob!r}"
    # Evidence documents the search_setup call.
    assert "search_setup" in sa["evidence"]


def test_setup_advice_after_values_are_valid_carsetup() -> None:
    """The model-recommended diff fields must be constructible as CarSetup
    values (they come from CarSetup.from_vector so always valid; this guards
    the integration end-to-end)."""
    from f1opt.model.optimizer import search_setup

    result = search_setup(
        "melbourne",
        driver_profile=None,
        baseline=CarSetup(**DEFAULT_SETUP.model_dump()),
        iterations=40,
        seed=42,
    )
    # Every changed field's after value must construct a valid CarSetup.
    base = DEFAULT_SETUP.model_dump()
    for d in result.diff:
        _construct_with(base, d["name"], d["after"])


def test_lockup_detection_flags_braking_dimension() -> None:
    """Full brake + barely-decreasing speed => braking advice mentions lockup."""
    out = generate_feedback(_lockup_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    braking = _dim_by_name(out, "braking")
    assert braking["advice"] is not None
    assert "锁死" in braking["advice"] or "lockup" in braking["advice"].lower()
    assert braking["evidence"]  # evidence grounded in brake/speed frames


def test_confidence_dimension_present_with_evidence() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    conf = _dim_by_name(out, "confidence")
    assert "corrections" in conf["value"]
    assert "stability" in conf["value"]
    assert conf["evidence"]


def test_sector_compare_dimension_uses_nominal_when_not_separable() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    sec = _dim_by_name(out, "sector_compare")
    # scripted frames' lap_distance does not span 2/3 of the track => fallback.
    assert "nominal" in sec["value"] or "not separable" in sec["value"]


def test_sector_compare_dimension_derives_splits_for_full_lap() -> None:
    out = generate_feedback(_full_lap_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    sec = _dim_by_name(out, "sector_compare")
    assert "S1=" in sec["value"] and "nom" in sec["value"]
    assert sec["evidence"]


# --------------------------------------------------------------------------- #
# Question handling
# --------------------------------------------------------------------------- #
def test_question_answered_in_summary() -> None:
    out = generate_feedback(
        _scripted_frames(),
        DEFAULT_SETUP.model_dump(),
        "melbourne",
        question="为什么 T1 入弯推头",
    )
    summary = out["summary"]
    assert summary
    # The canned answer addresses the understeer question and references a metric.
    assert "推头" in summary or "understeer" in summary.lower()
    assert "steer" in summary or "g_lat" in summary or "indicator" in summary


def test_question_about_lockup_answered() -> None:
    out = generate_feedback(
        _lockup_frames(),
        DEFAULT_SETUP.model_dump(),
        "melbourne",
        question="为什么 T1 制动锁死",
    )
    summary = out["summary"]
    assert "锁死" in summary or "lockup" in summary.lower()


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_empty_frames_no_crash() -> None:
    out = generate_feedback([], DEFAULT_SETUP.model_dump(), "melbourne")
    assert set(out.keys()) == {"summary", "dimensions", "setup_suggestions", "sources", "_quality"}
    names = [d["name"] for d in out["dimensions"]]
    assert names == FEEDBACK_DIMENSIONS
    assert out["sources"] == []
    assert out["setup_suggestions"] == []
    # Data-dependent dims flag insufficient data.
    for name in ("balance", "grip", "tyres", "braking", "confidence"):
        assert "数据不足" in _dim_by_name(out, name)["value"]
    # lap_time_potential still reports the reference; setup_advice still runs.
    assert "reference" in _dim_by_name(out, "lap_time_potential")["value"].lower()
    assert _dim_by_name(out, "setup_advice")["advice"] is not None
    # Summary notes insufficient data.
    assert "数据不足" in out["summary"] or "insufficient" in out["summary"].lower()


def test_invalid_track_id_no_crash() -> None:
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "not_a_real_track")
    assert set(out.keys()) == {"summary", "dimensions", "setup_suggestions", "sources", "_quality"}
    names = [d["name"] for d in out["dimensions"]]
    assert names == FEEDBACK_DIMENSIONS
    assert out["summary"]
    # setup_advice still produces something (search_setup handles unknown track).
    sa = _dim_by_name(out, "setup_advice")
    assert sa["value"]


# --------------------------------------------------------------------------- #
# LLM gating (no network by default; fallback on missing key / error)
# --------------------------------------------------------------------------- #
def test_default_engine_runs_without_network() -> None:
    """Default config (llm_backend='none') must not touch the network."""
    out = generate_feedback(_scripted_frames(), DEFAULT_SETUP.model_dump(), "melbourne")
    assert out["summary"]  # rule-based summary intact


def test_llm_enhance_falls_back_on_missing_key() -> None:
    settings = Settings(llm_backend="openai", llm_api_key="")
    feedback = {
        "summary": "rule-based summary",
        "dimensions": [],
        "setup_suggestions": [],
        "sources": [],
    }
    out = llm_enhance(feedback, "why understeer?", settings)
    assert out["summary"] == "rule-based summary"  # unchanged


def test_preload_llm_local_backend_no_key_required(monkeypatch) -> None:
    """local 后端 (Ollama) 无需 API key；Ollama 可达时 preload 返回 loaded=True。"""
    import httpx

    # Mock Ollama 可达性检查 (/api/tags) 为成功。
    def _fake_get(self, url, **kw):
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", _fake_get)
    engine = FeedbackEngine(Settings(llm_backend="local", llm_api_key=""))
    result = engine.preload_llm()
    assert result["loaded"] is True
    assert result["backend"] == "local"


def test_preload_llm_local_backend_unreachable(monkeypatch) -> None:
    """Iter-254: Ollama 不可达时 preload 应返回 loaded=False (而非假成功)。"""
    import httpx

    def _fake_get_fail(self, url, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.Client, "get", _fake_get_fail)
    engine = FeedbackEngine(Settings(llm_backend="local", llm_api_key=""))
    result = engine.preload_llm()
    assert result["loaded"] is False
    assert "not reachable" in result["reason"]


def test_preload_llm_openai_backend_requires_key() -> None:
    """openai 云端后端无 key 时 preload_llm 应返回 loaded=False。"""
    engine = FeedbackEngine(Settings(llm_backend="openai", llm_api_key=""))
    result = engine.preload_llm()
    assert result["loaded"] is False


def test_llm_enhance_falls_back_on_unknown_backend() -> None:
    settings = Settings(llm_backend="bogus", llm_api_key="sk-fake")
    feedback = {
        "summary": "rule-based summary",
        "dimensions": [],
        "setup_suggestions": [],
        "sources": [],
    }
    out = llm_enhance(feedback, None, settings)
    assert out["summary"] == "rule-based summary"


def test_llm_enhance_falls_back_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    class _BoomClient:
        def __init__(self, *a: object, **k: object) -> None:
            raise RuntimeError("no network")

        def __enter__(self) -> _BoomClient:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

    monkeypatch.setattr(httpx, "Client", _BoomClient)
    settings = Settings(llm_backend="openai", llm_api_key="sk-fake")
    feedback = {
        "summary": "rule-based summary",
        "dimensions": [{"name": "balance", "value": "x", "evidence": "y", "advice": None}],
        "setup_suggestions": [],
        "sources": [{"frame_t": 1.0, "field": "speed", "value": 100.0}],
    }
    out = llm_enhance(feedback, "why?", settings)
    assert out["summary"] == "rule-based summary"  # network failed -> fallback


def test_llm_enhance_honors_config_llm_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Iter-257: llm_enhance 应使用 config.llm_model (而非硬编码默认模型)。"""
    import httpx

    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "LLM rewrite"}}]}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def post(self, url: str, **k: object) -> _Resp:
            captured["json"] = k.get("json")
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    settings = Settings(llm_backend="openai", llm_api_key="sk-fake", llm_model="my-custom-model")
    feedback = {
        "summary": "rule-based",
        "dimensions": [],
        "setup_suggestions": [],
        "sources": [],
    }
    out = llm_enhance(feedback, "why?", settings)
    assert out["summary"] == "LLM rewrite"
    payload = captured.get("json")
    assert isinstance(payload, dict)
    assert payload["model"] == "my-custom-model"


def test_llm_enhance_reflection_refines_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Iter-258: 开启 llm_reflection 时做第二轮自评修正, 返回修正后的答案。"""
    import httpx

    calls: list[str] = []

    class _Resp:
        def __init__(self, content: str) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": self._content}}]}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def post(self, url: str, **k: object) -> _Resp:
            calls.append("post")
            if len(calls) == 1:
                return _Resp("draft answer")
            return _Resp("refined answer")

    monkeypatch.setattr(httpx, "Client", _Client)
    settings = Settings(
        llm_backend="openai", llm_api_key="sk-fake", llm_reflection=True
    )
    feedback = {
        "summary": "rule-based",
        "dimensions": [],
        "setup_suggestions": [],
        "sources": [{"frame_t": 1.0, "field": "speed", "value": 200.0}],
    }
    out = llm_enhance(feedback, "why?", settings)
    assert out["summary"] == "refined answer"
    assert len(calls) == 2  # 首轮 + 反思轮


def test_llm_enhance_no_reflection_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Iter-258: 默认 llm_reflection=False 只做单轮调用 (保持轻量)。"""
    import httpx

    calls: list[str] = []

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "single answer"}}]}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def post(self, url: str, **k: object) -> _Resp:
            calls.append("post")
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    settings = Settings(llm_backend="openai", llm_api_key="sk-fake")
    feedback = {
        "summary": "rule-based",
        "dimensions": [],
        "setup_suggestions": [],
        "sources": [],
    }
    out = llm_enhance(feedback, "why?", settings)
    assert out["summary"] == "single answer"
    assert len(calls) == 1  # 默认单轮


# --------------------------------------------------------------------------- #
# Re-exports
# --------------------------------------------------------------------------- #
def test_package_reexports() -> None:
    from f1opt.feedback import FEEDBACK_DIMENSIONS as FD2
    from f1opt.feedback import FeedbackEngine as FE2
    from f1opt.feedback import generate_feedback as gf2

    assert FE2 is FeedbackEngine
    assert gf2 is generate_feedback
    assert FD2 is FEEDBACK_DIMENSIONS


# --------------------------------------------------------------------------- #
# Driver-profile personalisation (Iter-05)
# --------------------------------------------------------------------------- #
def test_feedback_aggressive_vs_conservative_advice_differs() -> None:
    """aggressive vs conservative profiles must yield distinct advice text in
    at least the braking / balance / confidence dimensions (the scripted frames
    trigger lockup + understeer + corrections so all three carry data)."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()
    out_a = generate_feedback(frames, setup, "monza", driver_profile=AGGRESSIVE_PROFILE)
    out_c = generate_feedback(frames, setup, "monza", driver_profile=CONSERVATIVE_PROFILE)

    differing = 0
    for name in ("braking", "balance", "confidence"):
        adv_a = _dim_by_name(out_a, name)["advice"] or ""
        adv_c = _dim_by_name(out_c, name)["advice"] or ""
        assert adv_a, f"{name}: aggressive advice unexpectedly empty"
        assert adv_c, f"{name}: conservative advice unexpectedly empty"
        assert adv_a != adv_c, (
            f"{name}: advice identical for aggressive vs conservative:\n"
            f"  aggressive={adv_a!r}\n  conservative={adv_c!r}"
        )
        differing += 1
    assert differing >= 3


def test_feedback_driver_profile_none_backward_compatible() -> None:
    """driver_profile=None must reproduce the pre-Iter-05 behaviour exactly:
    passing None explicitly equals omitting the argument."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()
    out_implicit = generate_feedback(frames, setup, "melbourne")
    out_explicit_none = generate_feedback(
        frames, setup, "melbourne", driver_profile=None
    )
    # Both advice fields and the full payload must match (deterministic, seed=42).
    for name in ("braking", "balance", "confidence", "setup_advice"):
        assert (_dim_by_name(out_implicit, name)["advice"] or "") == (
            _dim_by_name(out_explicit_none, name)["advice"] or ""
        ), f"{name}: None-profile path drifted from default"
    assert out_implicit == out_explicit_none


def test_feedback_setup_advice_passes_driver_profile() -> None:
    """setup_advice dimension forwards the profile to search_setup: the
    evidence string records the resolved style tag (AGGRESSIVE/CONSERVATIVE)."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()
    out_a = generate_feedback(frames, setup, "melbourne", driver_profile=AGGRESSIVE_PROFILE)
    out_c = generate_feedback(frames, setup, "melbourne", driver_profile=CONSERVATIVE_PROFILE)
    ev_a = _dim_by_name(out_a, "setup_advice")["evidence"]
    ev_c = _dim_by_name(out_c, "setup_advice")["evidence"]
    assert "search_setup" in ev_a and "driver_profile=AGGRESSIVE" in ev_a, ev_a
    assert "search_setup" in ev_c and "driver_profile=CONSERVATIVE" in ev_c, ev_c
    # And a None profile records NONE (backward-compat path).
    out_n = generate_feedback(frames, setup, "melbourne", driver_profile=None)
    ev_n = _dim_by_name(out_n, "setup_advice")["evidence"]
    assert "driver_profile=NONE" in ev_n, ev_n


def test_prompt_includes_driver_profile() -> None:
    """USER_PROMPT_TEMPLATE renders a driver_profile paragraph; aggressive
    profile surfaces the style label + aggression scalar, None yields the
    default (no personalisation) marker."""
    rendered = USER_PROMPT_TEMPLATE.format(
        question="为什么推头",
        granularity="overall",
        granularity_hint="",
        summary="summary text",
        dimensions="- balance: x",
        metrics_summary="t=1.0s speed=100.0",
        driver_profile=format_driver_profile(AGGRESSIVE_PROFILE),
    )
    assert "driver_profile" in rendered
    assert "aggressive" in rendered
    assert "aggression_score=0.90" in rendered
    # None profile -> explicit default marker.
    rendered_none = USER_PROMPT_TEMPLATE.format(
        question="q",
        granularity="overall",
        granularity_hint="",
        summary="s",
        dimensions="d",
        metrics_summary="m",
        driver_profile=format_driver_profile(None),
    )
    assert "default (no personalisation)" in rendered_none


def test_feedback_smoothness_personalized_by_profile() -> None:
    """throttle_brake_smoothness advice is personalised too: aggressive
    (throttle_smoothness=0.20 -> 渐进度不足) vs conservative (0.85 -> 较平顺)
    produce distinct advice clauses."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()
    out_a = generate_feedback(frames, setup, "melbourne", driver_profile=AGGRESSIVE_PROFILE)
    out_c = generate_feedback(frames, setup, "melbourne", driver_profile=CONSERVATIVE_PROFILE)
    adv_a = _dim_by_name(out_a, "throttle_brake_smoothness")["advice"] or ""
    adv_c = _dim_by_name(out_c, "throttle_brake_smoothness")["advice"] or ""
    assert adv_a and adv_c
    assert adv_a != adv_c
    # Aggressive low-smoothness clause mentions 油门渐进度不足.
    assert "渐进度不足" in adv_a


# --------------------------------------------------------------------------- #
# Multi-turn conversation memory (Iter-07)
# --------------------------------------------------------------------------- #
# Keep the process-wide session registry clean across the conversation tests
# so they cannot leak state into one another (or into other modules).
@pytest.fixture(autouse=True)
def _clean_conversation_sessions() -> None:
    reset_sessions()
    yield
    reset_sessions()


def test_generate_feedback_with_session_id_multi_turn_context_reference() -> None:
    """Two consecutive calls with the same session_id: the second question
    contains a demonstrative (那个) and must be prefixed with ``[引用上文]``
    carrying a snippet of the FIRST question's text. The first turn (no prior
    history) must NOT carry the prefix."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()

    # Turn 1: ask about understeer. No prior history -> no [引用上文] prefix.
    out1 = generate_feedback(
        frames, setup, "melbourne", question="为什么 T1 入弯推头", session_id="iter7-multi"
    )
    assert "[引用上文]" not in out1["summary"]

    # Turn 2: a follow-up using a demonstrative (那个) referencing turn 1.
    out2 = generate_feedback(
        frames, setup, "melbourne", question="那个怎么解决", session_id="iter7-multi"
    )
    assert "[引用上文]" in out2["summary"], out2["summary"]
    # The snippet must carry the prior question's keyword (推头).
    assert "推头" in out2["summary"]
    # The session registry now holds both turns for this id.
    sess = get_session("iter7-multi")
    assert len(sess.history) == 4  # 2 user + 2 assistant turns
    assert sess.history[0]["role"] == "user"
    assert sess.history[0]["content"] == "为什么 T1 入弯推头"


def test_generate_feedback_session_id_none_backward_compatible() -> None:
    """session_id=None must reproduce the pre-Iter-07 behaviour exactly:

    - identical output to omitting the kwarg entirely,
    - summary carries NO ``[引用上文]`` prefix (conversation is None),
    - no session is registered in the global registry.
    """
    from f1opt.feedback.conversation import _sessions

    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()

    out_omitted = generate_feedback(
        frames, setup, "melbourne", question="为什么推头"
    )
    out_none = generate_feedback(
        frames, setup, "melbourne", question="为什么推头", session_id=None
    )
    # Byte-identical payload (deterministic, seed=42, no LLM, no memory).
    assert out_omitted == out_none
    # No context prefix leaks into the None-path summary.
    assert "[引用上文]" not in out_none["summary"]
    # No session was created in the registry.
    assert _sessions == {}


def test_generate_feedback_first_turn_with_session_has_no_reference_prefix() -> None:
    """Even with a session_id and a demonstrative-laden question, the FIRST
    turn has no prior history, so the ``[引用上文]`` prefix must NOT appear.
    The turn is still recorded so the next call can reference it."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()
    out = generate_feedback(
        frames,
        setup,
        "melbourne",
        question="那个推头怎么解决",
        session_id="iter7-first",
    )
    # First turn -> no prior context to reference.
    assert "[引用上文]" not in out["summary"]
    # But the turn was still recorded for future turns.
    sess = get_session("iter7-first")
    assert len(sess.history) == 2  # 1 user + 1 assistant
    assert sess.history[0] == {"role": "user", "content": "那个推头怎么解决"}


def test_generate_feedback_different_session_ids_are_isolated() -> None:
    """Two different session_ids must NOT share context: a demonstrative
    follow-up under session B (which has no prior history) must NOT pick up
    the question recorded under session A."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()

    # Session A: prime it with an understeer question.
    generate_feedback(
        frames, setup, "melbourne", question="为什么推头", session_id="iter7-A"
    )
    # Session B: a fresh follow-up using a demonstrative.
    out_b = generate_feedback(
        frames, setup, "melbourne", question="那个怎么解决", session_id="iter7-B"
    )
    # B has no prior history -> no [引用上文] prefix.
    assert "[引用上文]" not in out_b["summary"]
    # And the two sessions are distinct instances with isolated histories.
    sess_a = get_session("iter7-A")
    sess_b = get_session("iter7-B")
    assert sess_a is not sess_b
    assert len(sess_a.history) == 2  # A has the primed turn
    assert len(sess_b.history) == 2  # B only has its own turn


def test_generate_feedback_non_demonstrative_question_records_but_no_prefix() -> None:
    """A follow-up question WITHOUT a demonstrative (刚才/那个/它/上面) must
    still be recorded in the session (so later demonstrative turns can
    reference it) but must NOT receive the ``[引用上文]`` prefix."""
    frames = _scripted_frames()
    setup = DEFAULT_SETUP.model_dump()
    sid = "iter7-nondemo"

    # Turn 1: a plain (non-demonstrative) question.
    out1 = generate_feedback(
        frames, setup, "melbourne", question="为什么推头", session_id=sid
    )
    assert "[引用上文]" not in out1["summary"]
    # Turn 2: another plain question — recorded but unprefixed.
    out2 = generate_feedback(
        frames, setup, "melbourne", question="如何改善轮胎", session_id=sid
    )
    assert "[引用上文]" not in out2["summary"]
    # Both turns recorded; the session now has 4 entries.
    sess = get_session(sid)
    assert len(sess.history) == 4
    assert sess.history[2]["content"] == "如何改善轮胎"
    # Turn 3: a demonstrative follow-up — NOW it prefixes with turn 2's text.
    out3 = generate_feedback(
        frames, setup, "melbourne", question="它怎么处理", session_id=sid
    )
    assert "[引用上文]" in out3["summary"]
    assert "如何改善轮胎" in out3["summary"]


# --------------------------------------------------------------------------- #
# Iter-134: LLM streaming output support
# --------------------------------------------------------------------------- #
class TestIter134Streaming:
    """Streaming LLM output (Iter-134).

    Covers ``_parse_sse_data_line``, ``llm_enhance_stream`` / ``_async`` gating
    + chunk yielding, ``FeedbackEngine.run_stream`` / ``_async`` (rule-based +
    LLM paths), and the public ``generate_feedback_stream`` / ``_async`` entry
    points. ``httpx`` is mocked via ``monkeypatch`` to simulate SSE streaming
    responses (``data: {"choices": [{"delta": {"content": "..."}}]}`` lines
    terminated by ``data: [DONE]``).
    """

    # --- helpers ----------------------------------------------------------- #
    @staticmethod
    def _feedback() -> dict:
        return {
            "summary": "rule-based summary",
            "dimensions": [
                {"name": "balance", "value": "neutral", "evidence": "g_lat=2.0", "advice": None}
            ],
            "setup_suggestions": [],
            "sources": [{"frame_t": 1.0, "field": "speed", "value": 200.0}],
        }

    @staticmethod
    def _settings(backend: str = "openai", key: str = "sk-fake") -> Settings:
        return Settings(llm_backend=backend, llm_api_key=key)

    @staticmethod
    def _frames() -> list[dict]:
        return [
            {
                "session_time": 0.0,
                "speed": 200.0,
                "throttle": 0.8,
                "brake": 0.0,
                "steer": 0.0,
                "g_lat": 0.5,
                "lap_time": 90.0,
                "lap_distance": 0.0,
            },
        ]

    @staticmethod
    def _sse_lines(chunks: list[str]) -> list[str]:
        import json as _json

        lines: list[str] = []
        for c in chunks:
            payload = _json.dumps({"choices": [{"delta": {"content": c}}]})
            lines.append(f"data: {payload}")
            lines.append("")
        lines.append("data: [DONE]")
        return lines

    @staticmethod
    def _install_sync_mock(
        monkeypatch: pytest.MonkeyPatch,
        lines: list[str],
        capture: dict | None = None,
    ) -> None:
        import httpx

        class _Resp:
            def __init__(self) -> None:
                self._lines = lines

            def raise_for_status(self) -> None:
                return None

            def iter_lines(self):  # generator
                yield from self._lines

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: object) -> bool:
                return False

        class _Client:
            def __init__(self, *a: object, **k: object) -> None:
                pass

            def __enter__(self) -> _Client:
                return self

            def __exit__(self, *a: object) -> bool:
                return False

            def stream(self, method: str, url: str, **k: object) -> _Resp:
                if capture is not None:
                    capture["url"] = url
                    capture["json"] = k.get("json", {})
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Client)

    @staticmethod
    def _install_async_mock(
        monkeypatch: pytest.MonkeyPatch,
        lines: list[str],
        capture: dict | None = None,
    ) -> None:
        import httpx

        class _Resp:
            def __init__(self) -> None:
                self._lines = lines

            def raise_for_status(self) -> None:
                return None

            async def aiter_lines(self):  # async generator
                for ln in self._lines:
                    yield ln

            async def __aenter__(self) -> _Resp:
                return self

            async def __aexit__(self, *a: object) -> bool:
                return False

        class _Client:
            def __init__(self, *a: object, **k: object) -> None:
                pass

            async def __aenter__(self) -> _Client:
                return self

            async def __aexit__(self, *a: object) -> bool:
                return False

            def stream(self, method: str, url: str, **k: object) -> _Resp:
                if capture is not None:
                    capture["url"] = url
                    capture["json"] = k.get("json", {})
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)

    # --- _parse_sse_data_line --------------------------------------------- #
    def test_parse_sse_valid_returns_content(self) -> None:
        from f1opt.feedback.engine import _parse_sse_data_line

        assert _parse_sse_data_line('{"choices": [{"delta": {"content": "hi"}}]}') == "hi"

    def test_parse_sse_done_empty_nonjson_return_none(self) -> None:
        from f1opt.feedback.engine import _parse_sse_data_line

        assert _parse_sse_data_line("[DONE]") is None
        assert _parse_sse_data_line("") is None
        assert _parse_sse_data_line("not json") is None

    def test_parse_sse_missing_keys_role_only_empty_content_return_none(self) -> None:
        from f1opt.feedback.engine import _parse_sse_data_line

        assert _parse_sse_data_line('{"choices": []}') is None
        assert _parse_sse_data_line('{"choices": [{"delta": {}}]}') is None
        assert _parse_sse_data_line('{"choices": [{"delta": {"role": "assistant"}}]}') is None
        assert _parse_sse_data_line('{"choices": [{"delta": {"content": ""}}]}') is None
        assert _parse_sse_data_line('{"no_choices": 1}') is None

    # --- llm_enhance_stream gating ---------------------------------------- #
    def test_llm_enhance_stream_gating_empty_streams(self) -> None:
        from f1opt.feedback.engine import llm_enhance_stream

        fb = self._feedback()
        assert list(llm_enhance_stream(fb, "q?", self._settings(backend="none", key=""))) == []
        assert list(llm_enhance_stream(fb, "q?", self._settings(backend="openai", key=""))) == []
        assert list(llm_enhance_stream(fb, "q?", self._settings(backend="bogus", key="sk-x"))) == []

    # --- llm_enhance_stream yields chunks --------------------------------- #
    def test_llm_enhance_stream_yields_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from f1opt.feedback.engine import llm_enhance_stream

        capture: dict = {}
        self._install_sync_mock(
            monkeypatch, self._sse_lines(["Hello", " world", "!"]), capture
        )
        chunks = list(llm_enhance_stream(self._feedback(), "q?", self._settings()))
        assert chunks == ["Hello", " world", "!"]
        assert capture["url"].endswith("/chat/completions")
        assert capture["json"].get("stream") is True
        assert isinstance(capture["json"].get("messages"), list)
        assert len(capture["json"]["messages"]) >= 2

    # --- llm_enhance_stream silent fallback on network error -------------- #
    def test_llm_enhance_stream_silent_fallback_network_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from f1opt.feedback.engine import llm_enhance_stream

        class _BoomClient:
            def __init__(self, *a: object, **k: object) -> None:
                pass

            def __enter__(self) -> _BoomClient:
                return self

            def __exit__(self, *a: object) -> bool:
                return False

            def stream(self, method: str, url: str, **k: object) -> object:
                raise ConnectionError("network down")

        monkeypatch.setattr(httpx, "Client", _BoomClient)
        chunks = list(llm_enhance_stream(self._feedback(), "q?", self._settings()))
        assert chunks == []

    # --- llm_enhance_stream_async gating + chunks ------------------------- #
    async def test_llm_enhance_stream_async_gating_and_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from f1opt.feedback.engine import llm_enhance_stream_async

        fb = self._feedback()
        assert [
            c async for c in llm_enhance_stream_async(
                fb, "q?", self._settings(backend="none", key="")
            )
        ] == []
        assert [
            c async for c in llm_enhance_stream_async(
                fb, "q?", self._settings(backend="openai", key="")
            )
        ] == []
        assert [
            c async for c in llm_enhance_stream_async(
                fb, "q?", self._settings(backend="bogus", key="sk-x")
            )
        ] == []
        capture: dict = {}
        self._install_async_mock(monkeypatch, self._sse_lines(["A", "B", "C"]), capture)
        chunks = [c async for c in llm_enhance_stream_async(fb, "q?", self._settings())]
        assert chunks == ["A", "B", "C"]
        assert capture["url"].endswith("/chat/completions")
        assert capture["json"].get("stream") is True

    # --- FeedbackEngine.run_stream rule-based (no LLM) -------------------- #
    def test_run_stream_rule_based_only_done_event(self) -> None:
        engine = FeedbackEngine(config=self._settings(backend="none", key=""))
        events = list(
            engine.run_stream(self._frames(), DEFAULT_SETUP.model_dump(), "melbourne")
        )
        assert len(events) == 1
        assert events[0]["type"] == "done"
        assert isinstance(events[0]["feedback"], dict)
        assert events[0]["feedback"]["summary"]

    # --- FeedbackEngine.run_stream with mocked LLM ------------------------ #
    def test_run_stream_with_llm_yields_chunks_and_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture: dict = {}
        self._install_sync_mock(monkeypatch, self._sse_lines(["LLM ", "summary"]), capture)
        engine = FeedbackEngine(config=self._settings(backend="openai", key="sk-fake"))
        engine.preload_llm()
        events = list(
            engine.run_stream(self._frames(), DEFAULT_SETUP.model_dump(), "melbourne")
        )
        chunks = [e for e in events if e["type"] == "chunk"]
        done = [e for e in events if e["type"] == "done"]
        assert len(chunks) == 2
        assert [e["text"] for e in chunks] == ["LLM ", "summary"]
        assert len(done) == 1
        assert done[0]["feedback"]["summary"] == "LLM summary"

    # --- generate_feedback_stream public entry point ---------------------- #
    def test_generate_feedback_stream_public_entry_point(self) -> None:
        from f1opt.feedback import generate_feedback_stream

        events = list(
            generate_feedback_stream(self._frames(), DEFAULT_SETUP.model_dump(), "melbourne")
        )
        assert len(events) >= 1
        assert events[-1]["type"] == "done"
        assert isinstance(events[-1]["feedback"], dict)

    # --- generate_feedback_stream_async public entry point ---------------- #
    async def test_generate_feedback_stream_async_public_entry_point(self) -> None:
        from f1opt.feedback import generate_feedback_stream_async

        events = [
            e
            async for e in generate_feedback_stream_async(
                self._frames(), DEFAULT_SETUP.model_dump(), "melbourne"
            )
        ]
        assert len(events) >= 1
        assert events[-1]["type"] == "done"
        assert isinstance(events[-1]["feedback"], dict)
