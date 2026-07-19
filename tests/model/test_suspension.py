"""Suspension + vehicle dynamics sub-model unit tests (Iter-21).

Covers the three classes in :mod:`f1opt.model.suspension`:

- :class:`SuspensionModel` — spring/ARB/ride-height scaling, rake, corner
  weights, roll & pitch stiffness distribution, heave natural frequency,
  damping ratio and lateral load-transfer distribution.
- :class:`VehicleDynamicsModel` — steady-state balance, stability factor, yaw
  inertia, yaw response time, trail-brake balance, aero sensitivity and the
  Chinese setup-balance diagnosis.
- :class:`SetupHarmonics` — spring/ARB harmony, ride-height/rake window, camber
  alignment, plus edge-case robustness and determinism.

Tests are pure-python and deterministic.
"""

from __future__ import annotations

import math

import pytest

from f1opt.model.suspension import (
    ARB_STIFFNESS_MAX_N_M_PER_DEG,
    ARB_STIFFNESS_MIN_N_M_PER_DEG,
    RIDE_HEIGHT_MAX_MM,
    RIDE_HEIGHT_MIN_MM,
    SPRING_RATE_MAX_N_PER_MM,
    SPRING_RATE_MIN_N_PER_MM,
    SetupHarmonics,
    SuspensionModel,
    VehicleDynamicsModel,
)

# A representative baseline setup using CarSetup field names. Spring/ARB/ride
# height game units are in [1, 11]; camber/toe are already in degrees.
BASE_SETUP: dict = {
    "front_suspension": 6,
    "rear_suspension": 4,
    "front_arb": 7,
    "rear_arb": 5,
    "front_ride_height": 4,
    "rear_ride_height": 8,
    "front_camber": -3.5,
    "rear_camber": -2.0,
    "front_toe": 0.05,
    "rear_toe": 0.20,
}

CAR_MASS_KG = 798.0
MASS_PER_CORNER_KG = CAR_MASS_KG / 4.0


# === SuspensionModel: spring / ARB / ride height ===========================
def test_spring_rate_front_in_valid_range() -> None:
    """Front spring rate falls within the mapped [80, 200] N/mm window."""
    model = SuspensionModel(BASE_SETUP)
    k = model.spring_rate("front")
    assert isinstance(k, float)
    assert SPRING_RATE_MIN_N_PER_MM <= k <= SPRING_RATE_MAX_N_PER_MM


def test_spring_rate_rear_in_valid_range() -> None:
    """Rear spring rate falls within the mapped [80, 200] N/mm window."""
    model = SuspensionModel(BASE_SETUP)
    k = model.spring_rate("rear")
    assert SPRING_RATE_MIN_N_PER_MM <= k <= SPRING_RATE_MAX_N_PER_MM


def test_spring_rate_scales_correctly_at_bounds() -> None:
    """Game unit 1 -> 80 N/mm, game unit 11 -> 200 N/mm (linear)."""
    soft = SuspensionModel({**BASE_SETUP, "front_suspension": 1})
    stiff = SuspensionModel({**BASE_SETUP, "front_suspension": 11})
    assert soft.spring_rate("front") == pytest.approx(SPRING_RATE_MIN_N_PER_MM)
    assert stiff.spring_rate("front") == pytest.approx(SPRING_RATE_MAX_N_PER_MM)
    # Midpoint of [1, 11] maps to the midpoint of [80, 200].
    mid = SuspensionModel({**BASE_SETUP, "front_suspension": 6})
    assert mid.spring_rate("front") == pytest.approx(140.0)


def test_arb_stiffness_scales_with_game_units() -> None:
    """ARB stiffness maps [1, 11] -> [10, 100] N·m/deg, monotonically."""
    soft = SuspensionModel({**BASE_SETUP, "front_arb": 1})
    stiff = SuspensionModel({**BASE_SETUP, "front_arb": 11})
    assert soft.arb_stiffness("front") == pytest.approx(ARB_STIFFNESS_MIN_N_M_PER_DEG)
    assert stiff.arb_stiffness("front") == pytest.approx(ARB_STIFFNESS_MAX_N_M_PER_DEG)
    # Midpoint 6 -> 55 N·m/deg.
    mid = SuspensionModel({**BASE_SETUP, "front_arb": 6})
    assert mid.arb_stiffness("front") == pytest.approx(55.0)
    assert soft.arb_stiffness("front") < mid.arb_stiffness("front") < stiff.arb_stiffness("front")


def test_ride_height_in_valid_mm_range() -> None:
    """Ride height maps [1, 11] -> [5, 40] mm for both axles."""
    model = SuspensionModel(BASE_SETUP)
    for kind in ("front", "rear"):
        rh = model.ride_height(kind)
        assert isinstance(rh, float)
        assert RIDE_HEIGHT_MIN_MM <= rh <= RIDE_HEIGHT_MAX_MM
    low = SuspensionModel({**BASE_SETUP, "front_ride_height": 1, "rear_ride_height": 1})
    high = SuspensionModel({**BASE_SETUP, "front_ride_height": 11, "rear_ride_height": 11})
    assert low.ride_height("front") == pytest.approx(RIDE_HEIGHT_MIN_MM)
    assert high.ride_height("rear") == pytest.approx(RIDE_HEIGHT_MAX_MM)


def test_rake_angle_positive_when_rear_higher() -> None:
    """Rake angle is positive (degrees) when rear ride height > front."""
    model = SuspensionModel(BASE_SETUP)  # rear (8) > front (4) in game units
    rake = model.rake_angle()
    assert isinstance(rake, float)
    assert rake > 0.0
    # Sanity: a 35 mm difference over 3 m is well under a degree.
    extreme = SuspensionModel(
        {**BASE_SETUP, "front_ride_height": 1, "rear_ride_height": 11}
    )
    assert extreme.rake_angle() > rake
    assert extreme.rake_angle() < 1.5  # 35 mm over 3 m ≈ 0.67°


def test_corner_weights_sum_to_total_weight() -> None:
    """The four corner weights (N) sum to mass * g."""
    model = SuspensionModel(BASE_SETUP)
    cw = model.corner_weights(CAR_MASS_KG, cg_longitudinal_pct=0.52)
    assert set(cw.keys()) == {"fl", "fr", "rl", "rr"}
    assert math.isclose(sum(cw.values()), CAR_MASS_KG * 9.81, rel_tol=1e-9)
    # Front bias > 0.5 means each front corner is heavier than each rear corner.
    assert cw["fl"] > cw["rl"]


def test_roll_stiffness_distribution_in_unit_interval() -> None:
    """Roll stiffness distribution is in [0, 1] and reflects front share."""
    model = SuspensionModel(BASE_SETUP)
    dist = model.roll_stiffness_distribution()
    assert isinstance(dist, float)
    assert 0.0 <= dist <= 1.0
    # Front springs + ARB both stiffer than rear here -> front-biased.
    assert dist > 0.5


def test_pitch_stiffness_distribution_in_unit_interval() -> None:
    """Pitch stiffness distribution (spring-only) is in [0, 1]."""
    model = SuspensionModel(BASE_SETUP)
    dist = model.pitch_stiffness_distribution()
    assert 0.0 <= dist <= 1.0
    assert math.isclose(dist + (1 - dist), 1.0)


def test_natural_frequency_in_f1_range() -> None:
    """Heave natural frequency lands in the F1-typical [3, 10] Hz band."""
    model = SuspensionModel(BASE_SETUP)
    for kind in ("front", "rear"):
        f = model.natural_frequency(kind, MASS_PER_CORNER_KG)
        assert isinstance(f, float)
        assert 3.0 <= f <= 10.0


def test_damping_ratio_finite_and_near_critical() -> None:
    """Damping ratio is finite and near the 0.7 default."""
    model = SuspensionModel(BASE_SETUP)
    zeta = model.damping_ratio()
    assert isinstance(zeta, float)
    assert math.isfinite(zeta)
    assert 0.5 < zeta < 0.9


def test_load_transfer_distribution_in_unit_interval() -> None:
    """Lateral load-transfer distribution is in [0, 1] and tracks roll dist."""
    model = SuspensionModel(BASE_SETUP)
    for lat_g in (0.0, 1.0, 2.5, 4.0):
        ltd = model.load_transfer_distribution(lat_g)
        assert 0.0 <= ltd <= 1.0
    # At 0G, pure elastic distribution equals roll stiffness distribution.
    assert math.isclose(
        model.load_transfer_distribution(0.0), model.roll_stiffness_distribution()
    )


# === VehicleDynamicsModel ==================================================
def test_steady_state_balance_returns_finite() -> None:
    """Understeer gradient (rad/g) is a finite float."""
    vdm = VehicleDynamicsModel(BASE_SETUP)
    k = vdm.steady_state_balance(lat_g=1.0, speed_ms=50.0)
    assert isinstance(k, float)
    assert math.isfinite(k)


def test_steady_state_balance_sign_reflects_setup() -> None:
    """A front-stiff setup yields a more positive (understeer) gradient."""
    understeer_setup = {
        **BASE_SETUP,
        "front_suspension": 11, "rear_suspension": 1,
        "front_arb": 11, "rear_arb": 1,
        "front_camber": -2.5, "rear_camber": -1.0,
    }
    oversteer_setup = {
        **BASE_SETUP,
        "front_suspension": 1, "rear_suspension": 11,
        "front_arb": 1, "rear_arb": 11,
        "front_camber": -3.5, "rear_camber": -2.0,
    }
    k_under = VehicleDynamicsModel(understeer_setup).steady_state_balance(1.0, 50.0)
    k_over = VehicleDynamicsModel(oversteer_setup).steady_state_balance(1.0, 50.0)
    assert k_under > k_over


def test_stability_factor_returns_valid_string() -> None:
    """stability_factor returns one of the three classification strings."""
    vdm = VehicleDynamicsModel(BASE_SETUP)
    stab = vdm.stability_factor()
    assert stab in {"understeer", "oversteer", "neutral"}


def test_yaw_inertia_positive_finite() -> None:
    """Yaw moment of inertia follows m·(wb²+tw²)/12 and is positive."""
    vdm = VehicleDynamicsModel(BASE_SETUP, mass_kg=798.0, track_width_m=2.0, wheelbase_m=3.5)
    iz = vdm.yaw_inertia()
    assert isinstance(iz, float)
    assert iz > 0.0
    expected = 798.0 * (3.5 ** 2 + 2.0 ** 2) / 12.0
    assert math.isclose(iz, expected, rel_tol=1e-12)


def test_response_time_in_realistic_band() -> None:
    """Yaw response time constant falls in the F1 [0.02, 0.3] s band."""
    vdm = VehicleDynamicsModel(BASE_SETUP)
    for step in (0.0, 1.0, 2.0, 3.0):
        tau = vdm.response_time(step)
        assert isinstance(tau, float)
        assert 0.02 <= tau <= 0.3


def test_response_time_grows_with_lateral_step() -> None:
    """Higher lateral step (tyre load sensitivity) lengthens the response."""
    vdm = VehicleDynamicsModel(BASE_SETUP)
    assert vdm.response_time(0.0) < vdm.response_time(3.0)


def test_trail_brake_balance_structure_and_safety_flag() -> None:
    """trail_brake_balance returns the required keys with a bool safety flag."""
    vdm = VehicleDynamicsModel(BASE_SETUP)
    out = vdm.trail_brake_balance(brake_bias=0.55, lat_g=1.0)
    assert set(out.keys()) == {"front_share", "rear_share", "trail_brake_safe"}
    assert isinstance(out["front_share"], float)
    assert isinstance(out["rear_share"], float)
    assert isinstance(out["trail_brake_safe"], bool)
    assert math.isclose(out["front_share"] + out["rear_share"], 1.0)
    # Low front bias at high lateral g is unsafe (rear would lock first).
    unsafe = vdm.trail_brake_balance(brake_bias=0.45, lat_g=2.0)
    assert unsafe["trail_brake_safe"] is False


def test_aero_sensitivity_returns_dict_and_downforce_grows_with_speed() -> None:
    """aero_sensitivity returns downforce + balance_shift; downforce grows with v."""
    vdm = VehicleDynamicsModel(BASE_SETUP)
    low = vdm.aero_sensitivity(30.0)
    high = vdm.aero_sensitivity(80.0)
    assert set(low.keys()) == {"downforce_total_n", "balance_shift"}
    assert isinstance(low["downforce_total_n"], float)
    assert isinstance(low["balance_shift"], float)
    assert low["downforce_total_n"] < high["downforce_total_n"]
    assert low["downforce_total_n"] >= 0.0


# === Setup balance diagnosis ===============================================
def test_setup_balance_diagnosis_has_all_keys_and_chinese_strings() -> None:
    """Diagnosis returns all 4 keys with non-empty Chinese strings."""
    vdm = VehicleDynamicsModel(BASE_SETUP)
    diag = vdm.setup_balance_diagnosis()
    assert set(diag.keys()) == {"mechanical_balance", "aero_balance", "overall", "recommendation"}
    for value in diag.values():
        assert isinstance(value, str)
        assert len(value) > 0


# === SetupHarmonics ========================================================
def test_check_spring_arb_harmony_structure() -> None:
    """check_spring_arb_harmony returns ok + warnings list."""
    harm = SetupHarmonics(BASE_SETUP)
    out = harm.check_spring_arb_harmony()
    assert set(out.keys()) == {"ok", "warnings"}
    assert isinstance(out["ok"], bool)
    assert isinstance(out["warnings"], list)
    # A deliberately mismatched setup (stiff spring + soft ARB) must warn.
    mismatched = SetupHarmonics(
        {**BASE_SETUP, "front_suspension": 11, "front_arb": 1,
         "rear_suspension": 11, "rear_arb": 1}
    )
    bad = mismatched.check_spring_arb_harmony()
    assert bad["ok"] is False
    assert len(bad["warnings"]) >= 1


def test_check_ride_height_rake_flags_extreme_rake() -> None:
    """Extreme rake (rear much higher than front) is flagged as not-ok."""
    extreme = SetupHarmonics(
        {**BASE_SETUP, "front_ride_height": 1, "rear_ride_height": 11}
    )
    out = extreme.check_ride_height_rake()
    assert set(out.keys()) == {"ok", "rake_mm", "warnings"}
    assert isinstance(out["rake_mm"], float)
    assert out["ok"] is False
    assert out["rake_mm"] > 15.0
    assert len(out["warnings"]) >= 1
    # A moderate, in-window rake is ok.
    moderate = SetupHarmonics(
        {**BASE_SETUP, "front_ride_height": 5, "rear_ride_height": 8}
    )
    ok_out = moderate.check_ride_height_rake()
    assert ok_out["ok"] is True
    assert 5.0 <= ok_out["rake_mm"] <= 15.0


def test_check_camber_alignment_flags_front_less_negative() -> None:
    """Front camber less negative than rear is flagged."""
    bad = SetupHarmonics(
        {**BASE_SETUP, "front_camber": -2.0, "rear_camber": -3.5}
    )
    out = bad.check_camber_alignment()
    assert set(out.keys()) == {"ok", "warnings"}
    assert out["ok"] is False
    assert any("外倾" in w for w in out["warnings"])
    # A well-aligned setup (front more negative) is ok.
    good = SetupHarmonics(
        {**BASE_SETUP, "front_camber": -3.5, "rear_camber": -2.0}
    )
    assert good.check_camber_alignment()["ok"] is True


def test_all_checks_aggregates() -> None:
    """all_checks runs the three sub-checks and aggregates ok + warning count."""
    harm = SetupHarmonics(BASE_SETUP)
    out = harm.all_checks()
    assert set(out.keys()) == {"ok", "checks", "total_warnings"}
    assert set(out["checks"].keys()) == {
        "spring_arb_harmony", "ride_height_rake", "camber_alignment",
    }
    assert isinstance(out["total_warnings"], int)
    assert out["total_warnings"] >= 0
    # total_warnings equals the sum of per-check warning counts.
    summed = sum(len(c["warnings"]) for c in out["checks"].values())
    assert out["total_warnings"] == summed
    assert out["ok"] == all(c["ok"] for c in out["checks"].values())


# === Edge cases & determinism ==============================================
def test_boundary_setups_do_not_crash() -> None:
    """Setups at the extreme game-unit boundaries (all 1s / all 11s) don't crash."""
    fields = [
        "front_suspension", "rear_suspension", "front_arb", "rear_arb",
        "front_ride_height", "rear_ride_height",
    ]
    geo = {"front_camber": -3.5, "rear_camber": -2.0, "front_toe": 0.05, "rear_toe": 0.20}
    for boundary in (1, 11):
        setup = {f: boundary for f in fields} | geo
        susp = SuspensionModel(setup)
        susp.spring_rate("front")
        susp.arb_stiffness("rear")
        susp.ride_height("front")
        susp.rake_angle()
        susp.corner_weights(CAR_MASS_KG)
        susp.roll_stiffness_distribution()
        susp.natural_frequency("front", MASS_PER_CORNER_KG)
        susp.damping_ratio()
        susp.load_transfer_distribution(1.0)
        vdm = VehicleDynamicsModel(setup)
        assert math.isfinite(vdm.steady_state_balance(1.0, 50.0))
        assert vdm.stability_factor() in {"understeer", "oversteer", "neutral"}
        assert vdm.response_time(1.0) > 0.0
        vdm.aero_sensitivity(50.0)
        vdm.setup_balance_diagnosis()
        SetupHarmonics(setup).all_checks()


def test_out_of_range_values_are_clamped_not_raised() -> None:
    """Values outside [1, 11] are clamped rather than raising."""
    setup = {**BASE_SETUP, "front_suspension": -5.0, "rear_arb": 999.0}
    susp = SuspensionModel(setup)
    assert susp.spring_rate("front") == pytest.approx(SPRING_RATE_MIN_N_PER_MM)
    assert susp.arb_stiffness("rear") == pytest.approx(ARB_STIFFNESS_MAX_N_M_PER_DEG)


def test_determinism_same_setup_same_results() -> None:
    """The same setup dict yields identical results across instances."""
    a = SuspensionModel(BASE_SETUP)
    b = SuspensionModel(dict(BASE_SETUP))
    assert a.spring_rate("front") == b.spring_rate("front")
    assert a.arb_stiffness("rear") == b.arb_stiffness("rear")
    assert a.rake_angle() == b.rake_angle()
    assert a.roll_stiffness_distribution() == b.roll_stiffness_distribution()
    va, vb = VehicleDynamicsModel(BASE_SETUP), VehicleDynamicsModel(BASE_SETUP)
    assert va.steady_state_balance(1.0, 50.0) == vb.steady_state_balance(1.0, 50.0)
    assert va.response_time(1.0) == vb.response_time(1.0)
    assert va.setup_balance_diagnosis() == vb.setup_balance_diagnosis()
    assert SetupHarmonics(BASE_SETUP).all_checks() == SetupHarmonics(BASE_SETUP).all_checks()


def test_conceptual_alias_keys_accepted() -> None:
    """The conceptual aliases (front_spring / front_anti_roll_bar) also work."""
    aliased: dict = {
        "front_spring": 6, "rear_spring": 4,
        "front_anti_roll_bar": 7, "rear_anti_roll_bar": 5,
        "front_ride_height": 4, "rear_ride_height": 8,
        "front_camber": -3.5, "rear_camber": -2.0,
        "front_toe": 0.05, "rear_toe": 0.20,
    }
    via_alias = SuspensionModel(aliased)
    via_canonical = SuspensionModel(BASE_SETUP)
    assert via_alias.spring_rate("front") == via_canonical.spring_rate("front")
    assert via_alias.arb_stiffness("rear") == via_canonical.arb_stiffness("rear")
    assert via_alias.rake_angle() == via_canonical.rake_angle()


def test_load_transfer_distribution_neutral_setup_is_balanced() -> None:
    """A symmetric (front==rear) setup gives a ~50/50 transfer distribution."""
    symmetric = {
        **BASE_SETUP,
        "front_suspension": 6, "rear_suspension": 6,
        "front_arb": 6, "rear_arb": 6,
    }
    model = SuspensionModel(symmetric)
    assert math.isclose(model.roll_stiffness_distribution(), 0.5, abs_tol=1e-9)
    assert math.isclose(model.load_transfer_distribution(2.0), 0.5, abs_tol=1e-9)
