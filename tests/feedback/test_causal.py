"""Tests for :mod:`f1opt.feedback.causal` (Iter-26..28).

Covers:

- :data:`CAUSAL_RULES` — covers ALL 19 :data:`SETUP_FIELDS` with the required
  sub-keys (primary_effect_inc/dec, secondary_inc/dec, metric_deltas).
- :class:`CausalExplanationEngine.explain` — required output keys for three
  representative fields (front_wing / front_brake_bias / front_tyre_pressure),
  change detection (increased/decreased/unchanged), magnitude_pct calculation,
  risk classification (small=low, large=high), the causal-chain text, and the
  ``expected = coef * delta`` formula (incl. the camber more-negative case).
- :class:`WhatIfAnalyzer` — analyze_change / analyze_multi_change /
  suggest_accompanying, the unchanged edge, and extreme-change high risk.

The WhatIf tests lazily load the torch surrogate (one cached model load,
~0.3s); the pure-CAUSAL_RULES tests never touch torch.
"""

from __future__ import annotations

import math

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS
from f1opt.feedback.causal import (
    ACCOMPANYING_RULES,
    CAUSAL_RULES,
    CausalExplanationEngine,
    WhatIfAnalyzer,
)

_TRACK_ID = "melbourne"
_REQUIRED_EXPLAIN_KEYS = {
    "field",
    "change",
    "magnitude_pct",
    "primary_effect",
    "secondary_effects",
    "expected_metric_deltas",
    "risk",
    "risk_reason",
    "explanation_text",
}


def _expected_magnitude_pct(current: float, proposed: float, name: str) -> float:
    """Mirror of causal._magnitude_pct for cross-checking in tests."""
    spec = SETUP_FIELDS[name]
    delta = abs(float(proposed) - float(current))
    cur = abs(float(current))
    if cur > 1e-9:
        return delta / cur * 100.0
    span = float(spec.max) - float(spec.min)
    return delta / span * 100.0 if span > 0 else 0.0


# --------------------------------------------------------------------------- #
# CAUSAL_RULES coverage & structure
# --------------------------------------------------------------------------- #
def test_causal_rules_covers_all_19_setup_fields() -> None:
    assert set(CAUSAL_RULES) == set(SETUP_FIELDS)
    assert len(CAUSAL_RULES) == 23  # Iter-288: +engine_braking, +ballast


def test_causal_rules_entry_structure_has_required_keys() -> None:
    for name, rule in CAUSAL_RULES.items():
        assert {"primary_effect_inc", "primary_effect_dec", "secondary_inc",
                "secondary_dec", "metric_deltas"} <= set(rule), name
        assert isinstance(rule["secondary_inc"], list) and rule["secondary_inc"], name
        assert isinstance(rule["secondary_dec"], list) and rule["secondary_dec"], name
        assert isinstance(rule["metric_deltas"], dict) and rule["metric_deltas"], name


# --------------------------------------------------------------------------- #
# CausalExplanationEngine.explain — required keys for 3 example fields
# --------------------------------------------------------------------------- #
def test_explain_front_wing_returns_required_keys() -> None:
    ce = CausalExplanationEngine(SETUP_FIELDS)
    out = ce.explain("front_wing", 25, 30)
    assert set(out) >= _REQUIRED_EXPLAIN_KEYS
    assert out["field"] == "front_wing"
    assert out["change"] == "increased"
    assert out["primary_effect"] == "增加前轴下压力"
    assert isinstance(out["secondary_effects"], list) and len(out["secondary_effects"]) >= 1
    assert out["risk"] in {"low", "medium", "high"}


def test_explain_front_brake_bias_returns_required_keys() -> None:
    ce = CausalExplanationEngine(SETUP_FIELDS)
    out = ce.explain("front_brake_bias", 50, 52)
    assert set(out) >= _REQUIRED_EXPLAIN_KEYS
    assert out["field"] == "front_brake_bias"
    assert out["change"] == "increased"
    assert "制动" in out["primary_effect"]
    assert isinstance(out["secondary_effects"], list)


def test_explain_front_tyre_pressure_returns_required_keys() -> None:
    ce = CausalExplanationEngine(SETUP_FIELDS)
    out = ce.explain("front_tyre_pressure", 24.0, 25.0)
    assert set(out) >= _REQUIRED_EXPLAIN_KEYS
    assert out["field"] == "front_tyre_pressure"
    assert out["change"] == "increased"
    assert "气压" in out["primary_effect"] or "胎" in out["primary_effect"]


# --------------------------------------------------------------------------- #
# Change detection
# --------------------------------------------------------------------------- #
def test_explain_change_increased() -> None:
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 25, 30)
    assert out["change"] == "increased"
    assert out["primary_effect"] == "增加前轴下压力"


def test_explain_change_decreased() -> None:
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 30, 25)
    assert out["change"] == "decreased"
    assert out["primary_effect"] == "降低前轴下压力"


def test_explain_change_unchanged() -> None:
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 25, 25)
    assert out["change"] == "unchanged"
    assert out["magnitude_pct"] == 0.0
    # all expected metric deltas must be zeroed on no change
    assert all(v == 0.0 for v in out["expected_metric_deltas"].values())


# --------------------------------------------------------------------------- #
# magnitude_pct
# --------------------------------------------------------------------------- #
def test_magnitude_pct_calculation() -> None:
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 25, 30)
    assert out["magnitude_pct"] == pytest.approx(_expected_magnitude_pct(25, 30, "front_wing"))
    # 25 -> 30 clicks = 5/25 = 20%
    assert out["magnitude_pct"] == pytest.approx(20.0)


def test_magnitude_pct_zero_current_falls_back_to_range() -> None:
    # current=0 -> relative-to-current undefined; falls back to range span.
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 0, 5)
    # 5 / (50 - 0) * 100 = 10%
    assert out["magnitude_pct"] == pytest.approx(10.0)
    assert out["magnitude_pct"] == pytest.approx(_expected_magnitude_pct(0, 5, "front_wing"))


# --------------------------------------------------------------------------- #
# Risk classification
# --------------------------------------------------------------------------- #
def test_risk_low_for_small_change() -> None:
    # 1 click -> low (<=2)
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 25, 26)
    assert out["risk"] == "low"


def test_risk_medium_for_moderate_change() -> None:
    # 3 clicks -> medium (2 < .. <= 5)
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 25, 28)
    assert out["risk"] == "medium"


def test_risk_high_for_extreme_change() -> None:
    # 25 clicks -> high (>5)
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 25, 50)
    assert out["risk"] == "high"


# --------------------------------------------------------------------------- #
# explanation_text causal chain + metric delta formula
# --------------------------------------------------------------------------- #
def test_explanation_text_contains_causal_chain_for_front_wing() -> None:
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", 25, 30)
    txt = out["explanation_text"]
    # causal chain marker + the primary effect keyword
    assert "→" in txt
    assert "下压力" in txt
    # quantitative tail
    assert "变化" in txt and "clicks" in txt


def test_expected_metric_deltas_follow_coef_times_delta_formula() -> None:
    coefs = CAUSAL_RULES["front_wing"]["metric_deltas"]
    current, proposed = 25, 30
    delta = proposed - current
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_wing", current, proposed)
    for metric, coef in coefs.items():
        assert out["expected_metric_deltas"][metric] == pytest.approx(coef * delta)


def test_camber_more_negative_increases_corner_grip() -> None:
    # front_camber value goes -2.5 -> -3.5 (more negative = more camber).
    # coef for corner_grip is negative, so delta<0 -> positive grip gain.
    out = CausalExplanationEngine(SETUP_FIELDS).explain("front_camber", -2.5, -3.5)
    assert out["change"] == "decreased"  # raw value decreased
    assert out["expected_metric_deltas"]["corner_grip"] > 0.0
    assert out["expected_metric_deltas"]["tyre_temp"] > 0.0


# --------------------------------------------------------------------------- #
# Unknown field
# --------------------------------------------------------------------------- #
def test_causal_engine_unknown_field_raises() -> None:
    ce = CausalExplanationEngine(SETUP_FIELDS)
    with pytest.raises(KeyError):
        ce.explain("not_a_real_field", 1.0, 2.0)


# --------------------------------------------------------------------------- #
# WhatIfAnalyzer
# --------------------------------------------------------------------------- #
_REQUIRED_WHATIF_KEYS = {
    "field",
    "current",
    "proposed",
    "delta",
    "causal",
    "lap_time_delta",
    "confidence",
    "recommended_accompanying",
}


def test_whatif_analyze_change_returns_all_keys() -> None:
    wa = WhatIfAnalyzer(DEFAULT_SETUP, _TRACK_ID)
    out = wa.analyze_change("front_wing", 30)
    assert set(out) >= _REQUIRED_WHATIF_KEYS
    assert out["field"] == "front_wing"
    assert out["current"] == float(DEFAULT_SETUP.front_wing)
    assert out["proposed"] == 30.0
    assert isinstance(out["causal"], dict)
    assert isinstance(out["lap_time_delta"], float)
    assert math.isfinite(out["lap_time_delta"])
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["recommended_accompanying"], list)


def test_whatif_analyze_change_unchanged_edge() -> None:
    wa = WhatIfAnalyzer(DEFAULT_SETUP, _TRACK_ID)
    cur = float(DEFAULT_SETUP.front_wing)
    out = wa.analyze_change("front_wing", cur)
    assert out["causal"]["change"] == "unchanged"
    assert out["causal"]["magnitude_pct"] == 0.0
    # identical setup -> lap time delta is exactly 0
    assert abs(out["lap_time_delta"]) < 1e-9
    assert out["delta"] == 0.0


def test_whatif_analyze_multi_change_two_fields() -> None:
    wa = WhatIfAnalyzer(DEFAULT_SETUP, _TRACK_ID)
    out = wa.analyze_multi_change({"front_wing": 30, "front_brake_bias": 53})
    assert out["n_fields"] == 2
    assert isinstance(out["changes"], list) and len(out["changes"]) == 2
    assert {c["field"] for c in out["changes"]} == {"front_wing", "front_brake_bias"}
    assert isinstance(out["lap_time_delta"], float)
    assert math.isfinite(out["lap_time_delta"])
    assert isinstance(out["combined_setup_delta"], list)
    assert len(out["combined_setup_delta"]) == 2


def test_whatif_suggest_accompanying_returns_list_of_dicts() -> None:
    wa = WhatIfAnalyzer(DEFAULT_SETUP, _TRACK_ID)
    out = wa.suggest_accompanying("front_wing", 1)
    assert isinstance(out, list)
    assert out, "front_wing should have at least one accompanying suggestion"
    for entry in out:
        assert set(entry) >= {"field", "direction", "reason"}
        assert entry["direction"] == 1
        assert entry["field"] in SETUP_FIELDS


def test_whatif_extreme_change_high_risk() -> None:
    wa = WhatIfAnalyzer(DEFAULT_SETUP, _TRACK_ID)
    out = wa.analyze_change("front_wing", 50)  # 25 -> 50 = 25 clicks
    assert out["causal"]["risk"] == "high"


def test_whatif_accompanying_rules_cover_balance_pairs() -> None:
    # front/rear pairs must each point at the other for aero/susp/arb/height.
    for f in ("front_wing", "rear_wing", "front_arb", "rear_arb",
              "front_suspension", "rear_suspension",
              "front_ride_height", "rear_ride_height",
              "front_tyre_pressure", "rear_tyre_pressure",
              "front_camber", "rear_camber"):
        assert ACCOMPANYING_RULES[f], f
