"""Tests for the Pacejka Magic Formula tire model."""

from __future__ import annotations

import math

import pytest

from f1opt.model.tire_dynamics import (
    COMPOUND_PARAMS,
    MagicFormulaTire,
    TireSet,
)


# --------------------------------------------------------------------------- #
# MagicFormulaTire — pure longitudinal
# --------------------------------------------------------------------------- #
class TestPureLongitudinal:
    def test_zero_slip_zero_force(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        assert t.pure_longitudinal(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_peaks_around_optimal_slip_ratio(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        opt = t.optimal_slip_ratio()
        # Optimal slip ratio in expected F1 range.
        assert 0.05 <= opt <= 0.15
        # Force at optimal is the peak.
        f_opt = t.pure_longitudinal(opt)
        f_below = t.pure_longitudinal(opt * 0.5)
        f_above = t.pure_longitudinal(opt * 2.0)
        assert f_opt >= f_below
        assert f_opt >= f_above

    def test_force_increases_then_decreases(self) -> None:
        """Magic Formula: force rises, peaks, then falls (not monotonic)."""
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        forces = [t.pure_longitudinal(sr) for sr in [0.0, 0.05, 0.1, 0.2, 0.5, 0.9]]
        # Peak is somewhere in the middle, not at extremes.
        assert max(forces) > forces[0]
        assert max(forces) > forces[-1]

    def test_load_scales_force(self) -> None:
        """D = mu * Fz, so force scales roughly linearly with load."""
        t1 = MagicFormulaTire(load_n=2000.0, compound="soft")
        t2 = MagicFormulaTire(load_n=4000.0, compound="soft")
        f1 = t1.pure_longitudinal(0.1)
        f2 = t2.pure_longitudinal(0.1)
        # f2 should be roughly 2x f1 (slightly less due to load sensitivity).
        assert f2 > f1 * 1.8

    def test_compound_affects_peak(self) -> None:
        """Soft compound has higher peak mu than hard."""
        t_soft = MagicFormulaTire(load_n=4000.0, compound="soft")
        t_hard = MagicFormulaTire(load_n=4000.0, compound="hard")
        f_soft = t_soft.pure_longitudinal(t_soft.optimal_slip_ratio())
        f_hard = t_hard.pure_longitudinal(t_hard.optimal_slip_ratio())
        assert f_soft > f_hard


# --------------------------------------------------------------------------- #
# MagicFormulaTire — pure lateral
# --------------------------------------------------------------------------- #
class TestPureLateral:
    def test_zero_slip_zero_force(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        assert t.pure_lateral(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_peaks_around_optimal_slip_angle(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        opt = t.optimal_slip_angle()
        assert 4.0 <= opt <= 10.0
        f_opt = t.pure_lateral(opt)
        f_below = t.pure_lateral(opt * 0.5)
        f_above = t.pure_lateral(opt * 2.0)
        assert f_opt >= f_below
        assert f_opt >= f_above

    def test_camber_affects_lateral(self) -> None:
        """Mild negative camber (near -3.5 deg) increases lateral grip."""
        t_neutral = MagicFormulaTire(load_n=4000.0, camber_deg=0.0, compound="soft")
        t_camber = MagicFormulaTire(load_n=4000.0, camber_deg=-3.5, compound="soft")
        # At a moderate slip angle, cambered tire should produce more lateral force.
        assert t_camber.pure_lateral(6.0) > t_neutral.pure_lateral(6.0)


# --------------------------------------------------------------------------- #
# Combined slip
# --------------------------------------------------------------------------- #
class TestCombinedForce:
    def test_returns_tuple(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        fx, fy = t.combined_force(0.1, 6.0)
        assert isinstance(fx, float)
        assert isinstance(fy, float)

    def test_combined_within_friction_circle(self) -> None:
        """|combined| should not exceed peak pure force (friction ellipse)."""
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        fx, fy = t.combined_force(0.1, 6.0)
        fx0 = t.pure_longitudinal(0.1)
        fy0 = t.pure_lateral(6.0)
        # Each component reduced by the other's presence.
        assert abs(fx) <= abs(fx0) + 1e-6
        assert abs(fy) <= abs(fy0) + 1e-6

    def test_pure_slip_no_coupling(self) -> None:
        """With only longitudinal slip, Fx ≈ pure_longitudinal."""
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        fx, fy = t.combined_force(0.1, 0.0)
        assert fx == pytest.approx(t.pure_longitudinal(0.1), rel=1e-3)
        assert abs(fy) < 1.0  # ~0


# --------------------------------------------------------------------------- #
# Grip modifiers
# --------------------------------------------------------------------------- #
class TestGripModifiers:
    def test_grip_vs_temp_peak_at_90(self) -> None:
        t = MagicFormulaTire(compound="soft")  # peak_temp 90
        assert t.grip_vs_temp(90.0) == pytest.approx(1.0, abs=1e-6)
        assert t.grip_vs_temp(70.0) < 1.0
        assert t.grip_vs_temp(110.0) < 1.0

    def test_grip_vs_temp_falloff(self) -> None:
        t = MagicFormulaTire(compound="soft")
        # At peak, full grip.
        g_peak = t.grip_vs_temp(90.0)
        # Well outside window (peak+2*window=130), grip ~0.
        g_hot = t.grip_vs_temp(130.0)
        assert g_peak > g_hot
        assert g_hot < 0.2

    def test_grip_vs_load_decreasing(self) -> None:
        t = MagicFormulaTire(compound="soft")
        # More load → slightly less mu (load sensitivity).
        assert t.grip_vs_load(2000.0) > t.grip_vs_load(6000.0)

    def test_grip_vs_camber_peak_near_minus_3_5(self) -> None:
        t = MagicFormulaTire(compound="soft")
        assert t.grip_vs_camber(-3.5) == pytest.approx(1.0, abs=1e-6)
        # Far from peak, grip lower.
        assert t.grip_vs_camber(0.0) < 1.0
        assert t.grip_vs_camber(5.0) < 1.0

    def test_grip_vs_camber_in_range(self) -> None:
        t = MagicFormulaTire(compound="soft")
        for c in [-15.0, -3.5, 0.0, 5.0, 15.0]:
            g = t.grip_vs_camber(c)
            assert 0.6 <= g <= 1.0


# --------------------------------------------------------------------------- #
# Aligning moment + optima
# --------------------------------------------------------------------------- #
class TestAligningAndOptima:
    def test_aligning_torque_zero_at_zero_slip(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        assert t.self_aligning_torque(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_aligning_torque_finite(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        mz = t.self_aligning_torque(6.0)
        assert math.isfinite(mz)

    def test_optimal_slip_ratio_range(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        opt = t.optimal_slip_ratio()
        assert 0.05 <= opt <= 0.15

    def test_optimal_slip_angle_range(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        opt = t.optimal_slip_angle()
        assert 4.0 <= opt <= 10.0


# --------------------------------------------------------------------------- #
# TireSet
# --------------------------------------------------------------------------- #
class TestTireSet:
    def test_default_init(self) -> None:
        ts = TireSet(compound="soft", track_temp_c=30.0)
        assert len(ts.tires) == 4
        assert ts.compound == "soft"

    def test_update_all_four(self) -> None:
        ts = TireSet()
        ts.update(
            slip_ratios=[0.05, 0.05, 0.08, 0.08],
            slip_angles=[6.0, 6.0, 4.0, 4.0],
            loads_n=[3500, 3500, 4500, 4500],
            cambers_deg=[-3.5, -3.5, -2.0, -2.0],
            temps_c=[90, 90, 95, 95],
        )
        assert ts.tires[0].slip_ratio == 0.05
        assert ts.tires[2].load_n == 4500.0

    def test_update_wrong_length_raises(self) -> None:
        ts = TireSet()
        with pytest.raises(ValueError):
            ts.update([0.1] * 3, [1.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)

    def test_total_longitudinal_is_sum(self) -> None:
        ts = TireSet()
        ts.update([0.1] * 4, [0.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        expected = sum(t.pure_longitudinal() for t in ts.tires)
        assert ts.total_longitudinal() == pytest.approx(expected, rel=1e-6)

    def test_total_lateral_is_sum(self) -> None:
        ts = TireSet()
        ts.update([0.0] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        expected = sum(t.pure_lateral() for t in ts.tires)
        assert ts.total_lateral() == pytest.approx(expected, rel=1e-6)

    def test_lateral_balance_in_range(self) -> None:
        ts = TireSet()
        ts.update([0.0] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        bal = ts.lateral_balance()
        assert -1.0 <= bal <= 1.0

    def test_lateral_balance_zero_lateral(self) -> None:
        """When no lateral slip, balance is 0 (well-defined)."""
        ts = TireSet()
        ts.update([0.1] * 4, [0.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        assert ts.lateral_balance() == 0.0

    def test_traction_capacity_keys(self) -> None:
        ts = TireSet()
        cap = ts.traction_capacity()
        assert {"fx_max", "fy_max", "combined_max"} <= set(cap.keys())
        assert cap["fx_max"] > 0
        assert cap["fy_max"] > 0

    def test_wear_rate_positive(self) -> None:
        ts = TireSet()
        ts.update([0.1] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        assert ts.wear_rate() > 0

    def test_wear_rate_increases_with_slip(self) -> None:
        ts_low = TireSet()
        ts_low.update([0.02] * 4, [2.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        ts_high = TireSet()
        ts_high.update([0.2] * 4, [12.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        assert ts_high.wear_rate() > ts_low.wear_rate()

    def test_overheating_risk_in_range(self) -> None:
        ts = TireSet()
        ts.update([0.1] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [95] * 4)
        risk = ts.overheating_risk()
        assert 0.0 <= risk <= 1.0

    def test_overheating_risk_increases_with_temp(self) -> None:
        ts_cool = TireSet()
        ts_cool.update([0.1] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [95] * 4)
        ts_hot = TireSet()
        ts_hot.update([0.1] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [120] * 4)
        assert ts_hot.overheating_risk() > ts_cool.overheating_risk()

    def test_overheating_risk_zero_below_100(self) -> None:
        ts = TireSet()
        ts.update([0.1] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [80] * 4)
        assert ts.overheating_risk() == 0.0

    def test_grip_balance_diagnosis_understeer(self) -> None:
        """Front tires with low load → less grip → understeer tendency."""
        ts = TireSet()
        # Front much less loaded than rear.
        ts.update(
            slip_ratios=[0.0] * 4,
            slip_angles=[6.0] * 4,
            loads_n=[2000, 2000, 5000, 5000],  # front light, rear heavy
            cambers_deg=[0.0] * 4,
            temps_c=[90] * 4,
        )
        diag = ts.grip_balance_diagnosis()
        assert "推头" in diag

    def test_grip_balance_diagnosis_oversteer(self) -> None:
        """Rear tires with low load → less grip → oversteer tendency."""
        ts = TireSet()
        ts.update(
            slip_ratios=[0.0] * 4,
            slip_angles=[6.0] * 4,
            loads_n=[5000, 5000, 2000, 2000],  # front heavy, rear light
            cambers_deg=[-3.5, -3.5, 0.0, 0.0],
            temps_c=[90] * 4,
        )
        diag = ts.grip_balance_diagnosis()
        assert "甩尾" in diag

    def test_grip_balance_diagnosis_balanced(self) -> None:
        ts = TireSet()
        ts.update([0.0] * 4, [6.0] * 4, [4000] * 4, [-3.0] * 4, [90] * 4)
        diag = ts.grip_balance_diagnosis()
        assert diag == "平衡良好"

    def test_to_dict(self) -> None:
        ts = TireSet()
        d = ts.to_dict()
        assert "compound" in d
        assert "tires" in d
        assert len(d["tires"]) == 4


# --------------------------------------------------------------------------- #
# Compound params + edge cases
# --------------------------------------------------------------------------- #
class TestCompoundAndEdge:
    def test_compound_params_complete(self) -> None:
        for name in ["soft", "medium", "hard", "intermediate", "wet"]:
            assert name in COMPOUND_PARAMS
            p = COMPOUND_PARAMS[name]
            assert p.mu_peak > 0
            assert p.b_long > 0
            assert p.b_lat > 0

    def test_unknown_compound_falls_back(self) -> None:
        t = MagicFormulaTire(compound="nonexistent")
        assert t.compound == "soft"

    def test_extreme_slip_clamped(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        # Extreme slip_ratio beyond [-1, 1] is clamped, no crash.
        f = t.pure_longitudinal(5.0)
        assert math.isfinite(f)

    def test_extreme_slip_angle_clamped(self) -> None:
        t = MagicFormulaTire(load_n=4000.0, compound="soft")
        f = t.pure_lateral(100.0)
        assert math.isfinite(f)

    def test_zero_load_zero_force(self) -> None:
        t = MagicFormulaTire(load_n=0.0, compound="soft")
        assert t.pure_longitudinal(0.1) == 0.0
        assert t.pure_lateral(6.0) == 0.0

    def test_determinism(self) -> None:
        t1 = MagicFormulaTire(load_n=4000.0, compound="soft", temp_c=90.0)
        t2 = MagicFormulaTire(load_n=4000.0, compound="soft", temp_c=90.0)
        assert t1.pure_longitudinal(0.1) == t2.pure_longitudinal(0.1)
        assert t1.pure_lateral(6.0) == t2.pure_lateral(6.0)

    def test_temp_affects_force(self) -> None:
        """Hot tire (130°C, beyond window) produces less force than peak-temp tire."""
        t_peak = MagicFormulaTire(load_n=4000.0, temp_c=90.0, compound="soft")
        t_hot = MagicFormulaTire(load_n=4000.0, temp_c=130.0, compound="soft")
        f_peak = t_peak.pure_lateral(6.0)
        f_hot = t_hot.pure_lateral(6.0)
        assert f_peak > f_hot
