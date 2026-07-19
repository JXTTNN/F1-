"""Physics sub-model unit tests (Iter-15 .. Iter-20).

Covers all six physics sub-models in :mod:`f1opt.model.physics`:

- smoke tests for each model class (returns valid type);
- tire thermal: grip_factor peak at 90 °C, dropoff at extremes, in [0, 1];
- tire degradation: monotonic increase with slip, thermal activation > 100 °C,
  wear→laptime penalty anchors and blowout;
- aero: downforce/drag scale with v^2, ground-effect stall below 5 mm, lower
  ride height gives more downforce (above stall), aero balance range/sign;
- powertrain: ERS modes monotonic, fuel penalty linear, boost modes monotonic
  with the 0/0.05/0.12/0.20 anchors;
- brake: bias classification, lockup risk rises with temp and slip, in [0, 1];
- mass: weight transfer linear in lat_g, total mass = 798 + fuel, fuel burn
  scales with track length.

Tests run fast (pure-python, no torch) and are deterministic.
"""

from __future__ import annotations

import pytest

from f1opt.model.physics import (
    BLOWOUT_PENALTY_S,
    BOOST_BENEFIT_S,
    BRAKE_HEAT_COEFF,
    CAR_MIN_MASS_KG,
    FUEL_PENALTY_PER_KG,
    GRIP_AT_WINDOW_EDGE,
    PEAK_GRIP_TEMP_C,
    AeroModel,
    BrakeThermalModel,
    MassModel,
    PowertrainModel,
    TireDegradationModel,
    TireThermalModel,
    TireThermalState,
    TireWearState,
)


# === Smoke tests (one per model class) =====================================
def test_tire_thermal_model_smoke() -> None:
    """TireThermalModel.temperature returns a dict of per-tire temps."""
    model = TireThermalModel()
    state = TireThermalState()
    out = model.temperature(state, ambient_track_temp_c=40.0, slip_work=50_000.0, duration_s=10.0)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"FL", "FR", "RL", "RR"}
    for _tid, axle in out.items():
        assert isinstance(axle, dict)
        assert set(axle.keys()) == {"surface", "core"}
        assert isinstance(axle["surface"], float)
        assert isinstance(axle["core"], float)


def test_tire_degradation_model_smoke() -> None:
    """TireDegradationModel.wear_lap returns a non-negative float."""
    model = TireDegradationModel()
    state = TireWearState()
    inc = model.wear_lap(state, slip_angle_deg=3.0, tyre_load_g=1.5, track_temp_c=40.0)
    assert isinstance(inc, float)
    assert inc >= 0.0


def test_aero_model_smoke() -> None:
    """AeroModel.downforce returns a finite float."""
    model = AeroModel()
    f = model.downforce(front_wing=25, rear_wing=27, ride_height_f=10.0, ride_height_r=12.0, speed_ms=55.0)
    assert isinstance(f, float)
    assert f > 0.0


def test_powertrain_model_smoke() -> None:
    """PowertrainModel.ers_deploy_per_lap returns a non-negative float."""
    model = PowertrainModel()
    val = model.ers_deploy_per_lap(2, "medium")
    assert isinstance(val, float)
    assert val >= 0.0


def test_brake_model_smoke() -> None:
    """BrakeThermalModel.brake_temp returns a dict with front/rear floats."""
    model = BrakeThermalModel()
    out = model.brake_temp(brake_pressure=100.0, brake_bias=0.55, ambient_c=30.0, lap_frac_braking=0.2)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"front", "rear"}
    assert isinstance(out["front"], float)
    assert isinstance(out["rear"], float)


def test_mass_model_smoke() -> None:
    """MassModel.weight_transfer returns a non-negative float."""
    model = MassModel()
    wt = model.weight_transfer(mass_kg=900.0, cg_height_m=0.3, track_width_m=2.0, lat_g=3.0)
    assert isinstance(wt, float)
    assert wt > 0.0


# === Tire thermal: grip_factor =============================================
def test_grip_factor_peak_at_90c() -> None:
    """grip_factor is maximal at 90 °C and equals 1.0 there."""
    model = TireThermalModel()
    assert model.grip_factor(PEAK_GRIP_TEMP_C) == pytest.approx(1.0)
    # Strictly greater than nearby off-peak temperatures.
    assert model.grip_factor(90.0) > model.grip_factor(80.0)
    assert model.grip_factor(90.0) > model.grip_factor(100.0)


def test_grip_factor_window_edges() -> None:
    """grip_factor equals ~0.7 at the 70 °C and 110 °C window edges."""
    model = TireThermalModel()
    assert model.grip_factor(70.0) == pytest.approx(GRIP_AT_WINDOW_EDGE)
    assert model.grip_factor(110.0) == pytest.approx(GRIP_AT_WINDOW_EDGE)


def test_grip_factor_dropoff_at_extremes() -> None:
    """grip_factor drops below 0.7 outside the window and toward 0 at extremes."""
    model = TireThermalModel()
    assert model.grip_factor(30.0) < GRIP_AT_WINDOW_EDGE
    assert model.grip_factor(150.0) < GRIP_AT_WINDOW_EDGE
    # Cold/hot extremes approach zero.
    assert model.grip_factor(0.0) == pytest.approx(0.0)
    assert model.grip_factor(200.0) == pytest.approx(0.0)


def test_grip_factor_always_in_unit_range() -> None:
    """grip_factor is in [0, 1] across a broad temperature sweep."""
    model = TireThermalModel()
    for t in [-20, 0, 10, 30, 50, 70, 80, 90, 100, 110, 130, 150, 200, 500]:
        g = model.grip_factor(float(t))
        assert 0.0 <= g <= 1.0, f"grip_factor({t})={g} out of [0,1]"


def test_tire_thermal_heats_with_slip_work() -> None:
    """More slip work raises surface temperature above a no-slip baseline."""
    model = TireThermalModel()
    base = TireThermalState()
    hot = TireThermalState()
    model.temperature(base, ambient_track_temp_c=40.0, slip_work=0.0, duration_s=10.0)
    model.temperature(hot, ambient_track_temp_c=40.0, slip_work=200_000.0, duration_s=10.0)
    assert hot.surface_temps["FL"] > base.surface_temps["FL"]


def test_tire_thermal_dissipates_toward_ambient() -> None:
    """Without slip work, a hot tire cools toward ambient."""
    model = TireThermalModel()
    state = TireThermalState()
    # Start hot (default 90 °C), cool toward a cold ambient with no slip work.
    start = state.surface_temps["FL"]
    model.temperature(state, ambient_track_temp_c=20.0, slip_work=0.0, duration_s=60.0)
    assert state.surface_temps["FL"] < start


def test_tire_thermal_zero_duration_returns_current() -> None:
    """duration_s <= 0 returns the current state unchanged."""
    model = TireThermalModel()
    state = TireThermalState()
    before = state.surface_temps["FL"]
    out = model.temperature(state, ambient_track_temp_c=20.0, slip_work=100_000.0, duration_s=0.0)
    assert out["FL"]["surface"] == pytest.approx(before)
    assert state.surface_temps["FL"] == pytest.approx(before)


# === Tire degradation ======================================================
def test_wear_monotonic_in_slip() -> None:
    """Per-lap wear increases monotonically with slip angle."""
    model = TireDegradationModel()
    prev = -1.0
    for slip in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        inc = model.wear_lap(TireWearState(), slip_angle_deg=slip, tyre_load_g=1.5, track_temp_c=40.0)
        assert inc > prev, f"wear not monotonic at slip={slip}"
        prev = inc


def test_wear_thermal_activation_above_100c() -> None:
    """Wear is higher with thermal activation (track temp 110 °C) than without (90 °C)."""
    model = TireDegradationModel()
    cold = model.wear_lap(TireWearState(), slip_angle_deg=4.0, tyre_load_g=1.5, track_temp_c=90.0)
    hot = model.wear_lap(TireWearState(), slip_angle_deg=4.0, tyre_load_g=1.5, track_temp_c=110.0)
    assert hot > cold, f"thermal activation failed: hot={hot}, cold={cold}"


def test_wear_increases_with_load() -> None:
    """Wear increases linearly with vertical load."""
    model = TireDegradationModel()
    low = model.wear_lap(TireWearState(), slip_angle_deg=3.0, tyre_load_g=1.0, track_temp_c=40.0)
    high = model.wear_lap(TireWearState(), slip_angle_deg=3.0, tyre_load_g=3.0, track_temp_c=40.0)
    assert high > low


def test_wear_to_laptime_penalty_anchors() -> None:
    """Penalty is 0 at 0 %, ~0.8 s at 50 %, and the blowout value at/above 90 %."""
    model = TireDegradationModel()
    assert model.wear_to_laptime_penalty(0.0) == pytest.approx(0.0)
    assert model.wear_to_laptime_penalty(50.0) == pytest.approx(0.8, abs=1e-9)
    assert model.wear_to_laptime_penalty(90.0) == BLOWOUT_PENALTY_S
    assert model.wear_to_laptime_penalty(95.0) == BLOWOUT_PENALTY_S


def test_wear_to_laptime_penalty_monotonic_below_blowout() -> None:
    """Penalty is monotonically increasing for wear below the blowout threshold."""
    model = TireDegradationModel()
    prev = -1.0
    for w in [0.0, 10.0, 25.0, 40.0, 50.0, 70.0, 89.0]:
        p = model.wear_to_laptime_penalty(w)
        assert p > prev
        prev = p


def test_wear_state_accumulates() -> None:
    """wear_lap accumulates the increment into the state's wear_pct."""
    model = TireDegradationModel()
    state = TireWearState()
    inc = model.wear_lap(state, slip_angle_deg=3.0, tyre_load_g=1.5, track_temp_c=40.0)
    assert state.wear_pct == pytest.approx(inc)
    assert inc > 0.0


# === Aerodynamics ==========================================================
def test_downforce_scales_with_v_squared() -> None:
    """Doubling speed quadruples downforce (exact v^2 scaling)."""
    model = AeroModel()
    f1 = model.downforce(25, 27, ride_height_f=10.0, ride_height_r=12.0, speed_ms=25.0)
    f2 = model.downforce(25, 27, ride_height_f=10.0, ride_height_r=12.0, speed_ms=50.0)
    assert f2 == pytest.approx(4.0 * f1)


def test_drag_scales_with_v_squared() -> None:
    """Doubling speed quadruples drag (exact v^2 scaling)."""
    model = AeroModel()
    d1 = model.drag(25, 27, ride_height_avg=11.0, speed_ms=25.0)
    d2 = model.drag(25, 27, ride_height_avg=11.0, speed_ms=50.0)
    assert d2 == pytest.approx(4.0 * d1)


def test_ground_effect_stall_below_5mm() -> None:
    """Downforce collapses below the 5 mm stall: 4 mm < 6 mm despite being lower."""
    model = AeroModel()
    df_stalled = model.downforce(25, 27, ride_height_f=4.0, ride_height_r=4.0, speed_ms=55.0)
    df_attached = model.downforce(25, 27, ride_height_f=6.0, ride_height_r=6.0, speed_ms=55.0)
    assert df_stalled < df_attached, "ground-effect stall below 5 mm not reproduced"


def test_ground_effect_lower_more_downforce_above_stall() -> None:
    """Above the stall threshold, lower ride height yields more downforce."""
    model = AeroModel()
    df_low = model.downforce(25, 27, ride_height_f=8.0, ride_height_r=8.0, speed_ms=55.0)
    df_high = model.downforce(25, 27, ride_height_f=20.0, ride_height_r=20.0, speed_ms=55.0)
    assert df_low > df_high


def test_downforce_increases_with_wing_angle() -> None:
    """More wing angle (front + rear) produces more downforce."""
    model = AeroModel()
    low = model.downforce(5, 5, ride_height_f=10.0, ride_height_r=10.0, speed_ms=55.0)
    high = model.downforce(45, 45, ride_height_f=10.0, ride_height_r=10.0, speed_ms=55.0)
    assert high > low


def test_downforce_balance_range_and_sign() -> None:
    """Aero balance is in [-1, 1]; positive when front > rear, negative otherwise."""
    model = AeroModel()
    assert model.downforce_balance(8000.0, 4000.0) == pytest.approx(1.0 / 3.0)
    assert model.downforce_balance(4000.0, 8000.0) == pytest.approx(-1.0 / 3.0)
    assert model.downforce_balance(5000.0, 5000.0) == pytest.approx(0.0)
    # Extremes clamp to ±1.
    assert model.downforce_balance(10000.0, 0.0) == pytest.approx(1.0)
    assert model.downforce_balance(0.0, 10000.0) == pytest.approx(-1.0)
    # Zero total returns 0 (no division by zero).
    assert model.downforce_balance(0.0, 0.0) == 0.0


def test_aero_negative_speed_clamped_to_zero() -> None:
    """Negative speed is clamped to 0 → zero downforce and drag."""
    model = AeroModel()
    assert model.downforce(25, 27, 10.0, 12.0, speed_ms=-50.0) == 0.0
    assert model.drag(25, 27, 11.0, speed_ms=-50.0) == 0.0


# === Powertrain / ERS ======================================================
def test_ers_modes_monotonic() -> None:
    """ERS deployment energy is monotonically increasing in mode 0..3."""
    model = PowertrainModel()
    vals = [model.ers_deploy_per_lap(m, "medium") for m in range(4)]
    assert vals == sorted(vals)
    assert vals[0] == 0.0
    assert vals[-1] > vals[0]


def test_ers_layout_factor_varies() -> None:
    """High-speed layouts deploy more ERS energy than twisty ones at the same mode."""
    model = PowertrainModel()
    fast = model.ers_deploy_per_lap(3, "high_speed_low_downforce")
    twisty = model.ers_deploy_per_lap(3, "high_downforce")
    assert fast > twisty


def test_ers_mode_clamped() -> None:
    """Out-of-range modes are clamped to [0, 3]."""
    model = PowertrainModel()
    assert model.ers_deploy_per_lap(-5, "medium") == model.ers_deploy_per_lap(0, "medium")
    assert model.ers_deploy_per_lap(99, "medium") == model.ers_deploy_per_lap(3, "medium")


def test_laptime_benefit_scales_with_energy() -> None:
    """More deployed energy yields more laptime benefit on the same track."""
    model = PowertrainModel()
    assert model.laptime_benefit_kj_to_s(200.0, 5000.0) > model.laptime_benefit_kj_to_s(100.0, 5000.0)


def test_fuel_effect_linear() -> None:
    """Fuel laptime penalty is linear (constant slope) above the minimum fuel."""
    model = PowertrainModel()
    # Equal fuel deltas → equal penalty deltas (slope = FUEL_PENALTY_PER_KG).
    d1 = model.fuel_effect_laptime(70.0) - model.fuel_effect_laptime(50.0)
    d2 = model.fuel_effect_laptime(50.0) - model.fuel_effect_laptime(30.0)
    assert d1 == pytest.approx(d2)
    assert d1 == pytest.approx(FUEL_PENALTY_PER_KG * 20.0)
    # Below the minimum fuel the penalty is zero.
    assert model.fuel_effect_laptime(0.0) == 0.0
    assert model.fuel_effect_laptime(5.0) == 0.0


def test_fuel_effect_clamped_to_tank() -> None:
    """Fuel load is clamped to the 110 kg tank capacity."""
    model = PowertrainModel()
    assert model.fuel_effect_laptime(200.0) == model.fuel_effect_laptime(110.0)


def test_boost_modes_monotonic_and_anchors() -> None:
    """Boost laptime benefit is monotonic in mode with the 0/0.05/0.12/0.20 anchors."""
    model = PowertrainModel()
    vals = [model.boost_mode_laptime(m) for m in range(4)]
    assert vals == sorted(vals)
    assert vals == [BOOST_BENEFIT_S[m] for m in range(4)]
    assert vals == [0.0, 0.05, 0.12, 0.20]


def test_boost_mode_clamped() -> None:
    """Out-of-range boost modes are clamped to [0, 3]."""
    model = PowertrainModel()
    assert model.boost_mode_laptime(-1) == 0.0
    assert model.boost_mode_laptime(9) == 0.20


# === Brake thermal =========================================================
def test_brake_bias_balance_classification() -> None:
    """Bias classification: <0.50 understeer, 0.50-0.56 neutral, >0.56 oversteer."""
    model = BrakeThermalModel()
    assert model.brake_bias_balance(0.48) == "understeer"
    assert model.brake_bias_balance(0.50) == "neutral"
    assert model.brake_bias_balance(0.53) == "neutral"
    assert model.brake_bias_balance(0.56) == "neutral"
    assert model.brake_bias_balance(0.58) == "oversteer"


def test_brake_temp_front_heavier_with_more_bias() -> None:
    """Higher front bias shifts more brake heat to the front axle."""
    model = BrakeThermalModel()
    low = model.brake_temp(100.0, 0.50, 30.0, 0.2)
    high = model.brake_temp(100.0, 0.58, 30.0, 0.2)
    assert high["front"] > low["front"]
    assert high["rear"] < low["rear"]


def test_brake_temp_scales_with_pressure_and_braking() -> None:
    """Brake temp rises with brake pressure and braking fraction."""
    model = BrakeThermalModel()
    base = model.brake_temp(80.0, 0.55, 30.0, 0.1)["front"]
    hard = model.brake_temp(100.0, 0.55, 30.0, 0.3)["front"]
    assert hard > base
    assert base > 30.0  # above ambient


def test_lockup_risk_increases_with_temp() -> None:
    """Lockup risk rises with brake temperature (slip and bias held fixed)."""
    model = BrakeThermalModel()
    cold = model.lockup_risk(temp_c=200.0, bias=0.55, slip=5.0)
    hot = model.lockup_risk(temp_c=700.0, bias=0.55, slip=5.0)
    assert hot > cold


def test_lockup_risk_increases_with_slip() -> None:
    """Lockup risk rises with wheel slip (temp and bias held fixed)."""
    model = BrakeThermalModel()
    low = model.lockup_risk(temp_c=400.0, bias=0.55, slip=2.0)
    high = model.lockup_risk(temp_c=400.0, bias=0.55, slip=9.0)
    assert high > low


def test_lockup_risk_in_unit_range() -> None:
    """Lockup risk is clamped to [0, 1] across extreme inputs."""
    model = BrakeThermalModel()
    for temp in [-100, 0, 100, 500, 1000, 5000]:
        for slip in [-5, 0, 5, 20, 100]:
            for bias in [-0.2, 0.5, 0.6, 1.5]:
                r = model.lockup_risk(float(temp), float(bias), float(slip))
                assert 0.0 <= r <= 1.0


def test_brake_heat_coeff_used() -> None:
    """Sanity: at full pressure/fraction, front temp = ambient + BRAKE_HEAT_COEFF * bias."""
    model = BrakeThermalModel()
    out = model.brake_temp(100.0, 0.55, 30.0, 1.0)
    assert out["front"] == pytest.approx(30.0 + BRAKE_HEAT_COEFF * 0.55)


# === Mass & weight transfer ================================================
def test_weight_transfer_linear_in_lat_g() -> None:
    """Lateral load transfer is linear in lateral acceleration."""
    model = MassModel()
    wt1 = model.weight_transfer(900.0, 0.3, 2.0, 1.0)
    wt2 = model.weight_transfer(900.0, 0.3, 2.0, 2.0)
    wt3 = model.weight_transfer(900.0, 0.3, 2.0, 3.0)
    assert wt2 == pytest.approx(2.0 * wt1)
    assert wt3 == pytest.approx(3.0 * wt1)


def test_weight_transfer_zero_lat_g() -> None:
    """Zero lateral acceleration yields zero load transfer."""
    model = MassModel()
    assert model.weight_transfer(900.0, 0.3, 2.0, 0.0) == 0.0


def test_total_mass_is_min_plus_fuel() -> None:
    """Total mass = 798 kg minimum car mass + clamped fuel load."""
    model = MassModel()
    assert model.total_mass(0.0) == pytest.approx(CAR_MIN_MASS_KG)
    assert model.total_mass(30.0) == pytest.approx(CAR_MIN_MASS_KG + 30.0)
    # Clamped to the 110 kg tank.
    assert model.total_mass(200.0) == pytest.approx(CAR_MIN_MASS_KG + 110.0)


def test_fuel_burn_scales_with_track_length() -> None:
    """Doubling track length doubles per-lap fuel burn (same fuel load)."""
    model = MassModel()
    short = model.fuel_burn_per_lap(30.0, 5000.0)
    long = model.fuel_burn_per_lap(30.0, 10000.0)
    assert long == pytest.approx(2.0 * short)


def test_fuel_burn_positive_and_reasonable() -> None:
    """Per-lap fuel burn is positive and in a physically reasonable range for a 5 km lap."""
    model = MassModel()
    burn = model.fuel_burn_per_lap(30.0, 5000.0)
    assert burn > 0.0
    # ~1.6 L/lap at 0.75 kg/L ≈ 1.2 kg/lap baseline, plus a small mass factor.
    assert 1.0 < burn < 1.5
