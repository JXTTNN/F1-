"""Unit tests for setup presets + feedback-driven auto-tuner.

Covers :mod:`f1opt.model.presets`:

- :class:`SetupPresets` — per-track-type baselines, condition / compound /
  aggressive / conservative / driver-archetype variants and ``list_presets``.
- :class:`SetupAutoTuner` — feedback-driven tuning for balance / tyres /
  braking / confidence, ``tune_diff`` change log, ``apply_constraints`` via
  :class:`~f1opt.model.suspension.SetupHarmonics`, ``confidence_score`` and
  edge cases (empty feedback, determinism).

Tests are pure-python and deterministic; the optional
``lap_time_potential="gap"`` optimizer path is not exercised here to keep the
suite independent of surrogate / scipy training.
"""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.data.track_evolution import WeatherCondition
from f1opt.model.presets import SetupAutoTuner, SetupPresets

# One track per track_type, for the per-track-type coverage tests.
_TRACK_BY_TYPE = {
    "high_speed_low_downforce": "monza",
    "street": "monaco",
    "high_downforce": "hungaroring",
    "medium": "melbourne",
    "mixed": "suzuka",
}

# A setup that already passes every SetupHarmonics check (springs/ARB/ride
# height within the model's [1, 11] game-unit window, harmonious, rake in the
# 5–15 mm window, camber correctly aligned). Used for the "unchanged" and
# determinism edge cases where apply_constraints must be a no-op.
CONSISTENT_SETUP = CarSetup(
    front_wing=25,
    rear_wing=27,
    active_aero_mode=1,
    x_mode_activations=2,
    on_throttle_diff=80,
    off_throttle_diff=55,
    front_camber=-3.50,
    rear_camber=-2.00,
    front_toe=0.05,
    rear_toe=0.20,
    front_suspension=6,
    rear_suspension=6,
    front_arb=6,
    rear_arb=6,
    front_ride_height=4,
    rear_ride_height=7,
    brake_pressure=100,
    front_brake_bias=55,
    front_tyre_pressure=24.0,
    rear_tyre_pressure=20.5,
    fuel_load=30.0,
)


def _is_valid_setup(setup: CarSetup) -> bool:
    """Every field within its legal range and aligned to its step."""
    for name, spec in SETUP_FIELDS.items():
        v = getattr(setup, name)
        if v < spec.min or v > spec.max:
            return False
    return True


# === SetupPresets.for_track: per track_type validity =======================
@pytest.mark.parametrize("track_type,track_id", list(_TRACK_BY_TYPE.items()))
def test_for_track_returns_valid_setup_for_each_track_type(
    track_type: str, track_id: str
) -> None:
    """for_track returns a valid CarSetup for every track_type."""
    presets = SetupPresets()
    setup = presets.for_track(track_id)
    assert isinstance(setup, CarSetup)
    assert _is_valid_setup(setup)


def test_for_track_high_speed_low_downforce_has_low_front_wing() -> None:
    """Monza-style preset runs a low front wing for top speed."""
    setup = SetupPresets().for_track("monza")
    assert setup.front_wing <= 15
    assert setup.front_wing < DEFAULT_SETUP.front_wing


def test_for_track_high_downforce_has_high_front_wing() -> None:
    """Monaco-style preset runs a high front wing for cornering grip."""
    setup = SetupPresets().for_track("hungaroring")
    assert setup.front_wing >= 40
    assert setup.front_wing > DEFAULT_SETUP.front_wing


# === SetupPresets.for_condition ============================================
def test_for_condition_wet_raises_ride_height() -> None:
    """Wet condition raises both ride heights vs the dry track baseline."""
    presets = SetupPresets()
    wet = WeatherCondition(precipitation_mm=8.0, track_temp_c=20.0)
    base = presets.for_track("melbourne")
    adjusted = presets.for_condition(wet, "melbourne")
    assert adjusted.front_ride_height > base.front_ride_height
    assert adjusted.rear_ride_height > base.rear_ride_height


def test_for_condition_wet_increases_rear_wing() -> None:
    """Wet condition adds rear wing to compensate for lost grip."""
    presets = SetupPresets()
    wet = WeatherCondition(precipitation_mm=8.0, track_temp_c=20.0)
    base = presets.for_track("melbourne")
    adjusted = presets.for_condition(wet, "melbourne")
    assert adjusted.rear_wing > base.rear_wing


def test_for_condition_hot_lowers_pressures() -> None:
    """Hot condition lowers tyre pressures to limit thermal growth."""
    presets = SetupPresets()
    hot = WeatherCondition(track_temp_c=40.0, ambient_temp_c=35.0)
    base = presets.for_track("melbourne")
    adjusted = presets.for_condition(hot, "melbourne")
    assert adjusted.front_tyre_pressure < base.front_tyre_pressure
    assert adjusted.rear_tyre_pressure < base.rear_tyre_pressure


# === SetupPresets.for_compound =============================================
def test_for_compound_soft_vs_hard_pressures_differ() -> None:
    """Soft and hard compounds produce different tyre pressures."""
    presets = SetupPresets()
    base = presets.for_track("monza")  # front_camber=-3.20, not at a limit
    soft = presets.for_compound("soft", base)
    hard = presets.for_compound("hard", base)
    assert soft.front_tyre_pressure < hard.front_tyre_pressure
    assert soft.rear_tyre_pressure < hard.rear_tyre_pressure


def test_for_compound_soft_vs_hard_camber_differ() -> None:
    """Soft compound runs more negative camber than hard."""
    presets = SetupPresets()
    base = presets.for_track("monza")
    soft = presets.for_compound("soft", base)
    hard = presets.for_compound("hard", base)
    # More negative camber = smaller (more negative) value.
    assert soft.front_camber < hard.front_camber


# === aggressive / conservative variants ====================================
def test_aggressive_variant_is_stiffer_than_base() -> None:
    """Aggressive variant stiffens the suspension vs the base."""
    presets = SetupPresets()
    base = DEFAULT_SETUP
    agg = presets.aggressive_variant(base)
    assert agg.front_suspension > base.front_suspension
    assert agg.rear_suspension > base.rear_suspension


def test_conservative_variant_is_softer_than_base() -> None:
    """Conservative variant softens the suspension vs the base."""
    presets = SetupPresets()
    base = DEFAULT_SETUP
    cons = presets.conservative_variant(base)
    assert cons.front_suspension < base.front_suspension
    assert cons.rear_suspension < base.rear_suspension


# === driver archetypes =====================================================
def test_for_driver_archetype_aggressive_stiffer_rear() -> None:
    """AGGRESSIVE archetype stiffens the rear axle."""
    presets = SetupPresets()
    base = DEFAULT_SETUP
    agg = presets.for_driver_archetype("AGGRESSIVE", base)
    assert agg.rear_suspension > base.rear_suspension
    assert agg.rear_arb > base.rear_arb


def test_for_driver_archetype_development_softer() -> None:
    """DEVELOPMENT archetype softens the car (more forgiving)."""
    presets = SetupPresets()
    base = DEFAULT_SETUP
    dev = presets.for_driver_archetype("DEVELOPMENT", base)
    assert dev.front_suspension < base.front_suspension
    assert dev.rear_suspension < base.rear_suspension


def test_for_driver_archetype_tire_whisperer_less_camber() -> None:
    """TIRE_WHISPERER archetype runs less (less negative) front camber."""
    presets = SetupPresets()
    base = DEFAULT_SETUP
    tw = presets.for_driver_archetype("TIRE_WHISPERER", base)
    # Less negative camber => greater (closer to zero) value.
    assert tw.front_camber > base.front_camber


# === list_presets ==========================================================
def test_list_presets_returns_non_empty_with_required_keys() -> None:
    """list_presets returns a non-empty list of dicts with the required keys."""
    presets = SetupPresets()
    items = presets.list_presets()
    assert isinstance(items, list)
    assert len(items) >= 5
    for item in items:
        assert {"name", "track_type", "description", "setup_diff"} <= set(item)
        assert isinstance(item["setup_diff"], list)


# === SetupAutoTuner: tune ==================================================
def test_tune_understeer_increases_front_wing() -> None:
    """Understeer feedback increases front wing for more front grip."""
    tuner = SetupAutoTuner(
        DEFAULT_SETUP,
        "melbourne",
        [{"name": "balance", "value": "understeer", "advice": "pushing wide"}],
    )
    tuned = tuner.tune()
    assert tuned.front_wing > DEFAULT_SETUP.front_wing


def test_tune_oversteer_decreases_front_wing() -> None:
    """Oversteer feedback decreases front wing to calm the front end."""
    tuner = SetupAutoTuner(
        DEFAULT_SETUP,
        "melbourne",
        [{"name": "balance", "value": "oversteer", "advice": "rear stepping out"}],
    )
    tuned = tuner.tune()
    assert tuned.front_wing < DEFAULT_SETUP.front_wing


def test_tune_overheating_tyres_lowers_pressures() -> None:
    """Overheating feedback lowers tyre pressures."""
    tuner = SetupAutoTuner(
        DEFAULT_SETUP,
        "melbourne",
        [{"name": "tyres", "value": "overheating", "advice": "temps climbing"}],
    )
    tuned = tuner.tune()
    assert tuned.front_tyre_pressure < DEFAULT_SETUP.front_tyre_pressure
    assert tuned.rear_tyre_pressure < DEFAULT_SETUP.rear_tyre_pressure


# === SetupAutoTuner: tune_diff =============================================
def test_tune_diff_returns_change_log_with_required_keys() -> None:
    """tune_diff returns a list of {field, before, after, reason} dicts."""
    tuner = SetupAutoTuner(
        DEFAULT_SETUP,
        "melbourne",
        [{"name": "balance", "value": "understeer", "advice": ""}],
    )
    diff = tuner.tune_diff()
    assert isinstance(diff, list)
    assert len(diff) > 0
    for entry in diff:
        assert {"field", "before", "after", "reason"} <= set(entry)
        assert isinstance(entry["reason"], str) and entry["reason"]


# === SetupAutoTuner: apply_constraints =====================================
def test_apply_constraints_returns_valid_setup() -> None:
    """apply_constraints always returns a valid CarSetup."""
    tuner = SetupAutoTuner(DEFAULT_SETUP, "melbourne", [])
    constrained = tuner.apply_constraints(DEFAULT_SETUP)
    assert isinstance(constrained, CarSetup)
    assert _is_valid_setup(constrained)


# === SetupAutoTuner: confidence_score ======================================
def test_confidence_score_in_unit_interval() -> None:
    """confidence_score is always within [0, 1]."""
    tuner = SetupAutoTuner(
        DEFAULT_SETUP,
        "melbourne",
        [{"name": "balance", "value": "understeer"}],
    )
    score = tuner.confidence_score()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_confidence_score_higher_with_more_feedback_dims() -> None:
    """More recognised feedback dimensions yield a higher confidence score."""
    one_dim = SetupAutoTuner(
        DEFAULT_SETUP, "melbourne", [{"name": "balance", "value": "understeer"}]
    )
    three_dims = SetupAutoTuner(
        DEFAULT_SETUP,
        "melbourne",
        [
            {"name": "balance", "value": "understeer"},
            {"name": "tyres", "value": "overheating"},
            {"name": "braking", "value": "lockup"},
        ],
    )
    assert three_dims.confidence_score() > one_dim.confidence_score()
    assert one_dim.confidence_score() > 0.0


# === Edge cases ============================================================
def test_empty_feedback_tune_returns_setup_unchanged() -> None:
    """With empty feedback a consistent setup is returned unchanged."""
    tuner = SetupAutoTuner(CONSISTENT_SETUP, "melbourne", [])
    tuned = tuner.tune()
    assert tuned.model_dump() == CONSISTENT_SETUP.model_dump()
    # No changes were recorded.
    assert tuner.tune_diff() == []


def test_determinism_same_inputs_same_result() -> None:
    """Identical inputs produce identical tuned setups (no gap feedback)."""
    feedback = [
        {"name": "balance", "value": "understeer"},
        {"name": "tyres", "value": "overheating"},
    ]
    tuner_a = SetupAutoTuner(CONSISTENT_SETUP, "melbourne", feedback)
    tuner_b = SetupAutoTuner(CONSISTENT_SETUP, "melbourne", feedback)
    a = tuner_a.tune()
    b = tuner_b.tune()
    assert a.model_dump() == b.model_dump()
    assert tuner_a.tune_diff() == tuner_b.tune_diff()


def test_tune_twice_on_same_tuner_is_stable() -> None:
    """Calling tune() twice on the same tuner yields the same result."""
    tuner = SetupAutoTuner(
        CONSISTENT_SETUP,
        "melbourne",
        [{"name": "confidence", "value": "low"}],
    )
    first = tuner.tune()
    second = tuner.tune()
    assert first.model_dump() == second.model_dump()
