"""Tests for track evolution + weather models."""

from __future__ import annotations

import pytest

from f1opt.data.track_evolution import (
    SessionTimeline,
    TrackConditionSnapshot,
    TrackEvolutionModel,
    WeatherCondition,
    WeatherForecast,
    WeatherImpactModel,
    WindModel,
)


# --------------------------------------------------------------------------- #
# TrackEvolutionModel
# --------------------------------------------------------------------------- #
class TestTrackEvolution:
    def test_grip_level_in_range(self) -> None:
        m = TrackEvolutionModel("melbourne", "race")
        for lap in [0, 5, 10, 20, 40]:
            g = m.grip_level(lap)
            assert 0.0 <= g <= 1.0

    def test_grip_level_monotonic_increase(self) -> None:
        m = TrackEvolutionModel("melbourne", "race")
        g0 = m.grip_level(0)
        g10 = m.grip_level(10)
        g30 = m.grip_level(30)
        assert g0 <= g10 <= g30

    def test_rubber_buildup_asymptotic(self) -> None:
        m = TrackEvolutionModel("melbourne", "practice")
        assert m.rubber_buildup(0) == pytest.approx(0.0, abs=1e-6)
        assert m.rubber_buildup(100) < 1.0  # asymptotes below 1
        assert m.rubber_buildup(50) > m.rubber_buildup(10)

    def test_track_temp_progression_shape(self) -> None:
        m = TrackEvolutionModel("melbourne", "race")
        t0 = m.track_temp_progression(0, 25.0)
        t_mid = m.track_temp_progression(30, 25.0)
        t_late = m.track_temp_progression(55, 25.0)
        # Mid-session peak, then cool.
        assert t_mid > t0
        assert t_mid >= t_late

    def test_lap_time_delta_negative_with_evolution(self) -> None:
        m = TrackEvolutionModel("melbourne", "race")
        d0 = m.lap_time_delta_from_evolution(0)
        d30 = m.lap_time_delta_from_evolution(30)
        assert d0 == pytest.approx(0.0, abs=1e-6)
        assert d30 < 0  # faster with evolution

    def test_optimal_lap_window_per_session(self) -> None:
        for sess, expected in [
            ("practice", (15, 40)),
            ("qualifying", (3, 12)),
            ("sprint", (8, 18)),
            ("race", (10, 30)),
        ]:
            m = TrackEvolutionModel("x", sess)
            assert m.optimal_lap_window() == expected

    def test_marbles_penalty_increases_with_distance(self) -> None:
        m = TrackEvolutionModel("x", "race")
        assert m.marbles_offline_grip_penalty(0.0) == 0.0
        assert m.marbles_offline_grip_penalty(1.0) < 0.0
        assert m.marbles_offline_grip_penalty(2.0) <= m.marbles_offline_grip_penalty(1.0)
        # Caps at -0.15.
        assert m.marbles_offline_grip_penalty(10.0) >= -0.15

    def test_unknown_session_defaults_to_race(self) -> None:
        m = TrackEvolutionModel("x", "unknown_session")
        assert m.session_type == "race"


# --------------------------------------------------------------------------- #
# TrackConditionSnapshot + SessionTimeline
# --------------------------------------------------------------------------- #
class TestSnapshotAndTimeline:
    def test_snapshot_to_dict(self) -> None:
        s = TrackConditionSnapshot(grip_level=0.9, track_temp_c=35.0)
        d = s.to_dict()
        assert d["grip_level"] == 0.9
        assert d["track_temp_c"] == 35.0
        assert "wetness" in d

    def test_timeline_record_and_at(self) -> None:
        tl = SessionTimeline("x")
        tl.record(1, TrackConditionSnapshot(grip_level=0.85))
        tl.record(10, TrackConditionSnapshot(grip_level=0.95))
        assert tl.at(1).grip_level == 0.85
        assert tl.at(10).grip_level == 0.95

    def test_timeline_interpolation(self) -> None:
        tl = SessionTimeline("x")
        tl.record(1, TrackConditionSnapshot(grip_level=0.80))
        tl.record(11, TrackConditionSnapshot(grip_level=0.90))
        mid = tl.at(6)
        # Linear interpolation: 0.85 at lap 6.
        assert mid.grip_level == pytest.approx(0.85, abs=0.01)

    def test_timeline_empty_returns_default(self) -> None:
        tl = SessionTimeline("x")
        s = tl.at(5)
        assert isinstance(s, TrackConditionSnapshot)

    def test_timeline_trend(self) -> None:
        tl = SessionTimeline("x")
        tl.record(1, TrackConditionSnapshot(grip_level=0.80))
        tl.record(10, TrackConditionSnapshot(grip_level=0.95))
        assert tl.trend("grip_level") == "improving"
        tl2 = SessionTimeline("x")
        tl2.record(1, TrackConditionSnapshot(wetness=0.5))
        tl2.record(10, TrackConditionSnapshot(wetness=0.1))
        assert tl2.trend("wetness") == "improving"


# --------------------------------------------------------------------------- #
# WeatherCondition
# --------------------------------------------------------------------------- #
class TestWeatherCondition:
    def test_dry_wetness_zero_precip(self) -> None:
        w = WeatherCondition(precipitation_mm=0.0, humidity_pct=50.0)
        assert w.wetness() < 0.15
        assert w.is_dry()

    def test_wet_high_precip(self) -> None:
        w = WeatherCondition(precipitation_mm=10.0)
        assert w.wetness() >= 0.6
        assert w.is_wet()

    def test_intermediate(self) -> None:
        w = WeatherCondition(precipitation_mm=2.0)
        assert w.is_intermediate()

    def test_compound_recommendation_dry_hot(self) -> None:
        w = WeatherCondition(precipitation_mm=0.0, track_temp_c=40.0)
        assert w.compound_recommendation() == "hard"

    def test_compound_recommendation_dry_cold(self) -> None:
        w = WeatherCondition(precipitation_mm=0.0, track_temp_c=10.0)
        assert w.compound_recommendation() == "soft"

    def test_compound_recommendation_wet(self) -> None:
        w = WeatherCondition(precipitation_mm=8.0)
        assert w.compound_recommendation() == "wet"

    def test_compound_recommendation_intermediate(self) -> None:
        w = WeatherCondition(precipitation_mm=2.0)
        assert w.compound_recommendation() == "intermediate"

    def test_to_dict(self) -> None:
        w = WeatherCondition()
        d = w.to_dict()
        assert "wetness" in d
        assert "ambient_temp_c" in d


# --------------------------------------------------------------------------- #
# WeatherImpactModel
# --------------------------------------------------------------------------- #
class TestWeatherImpact:
    def test_grip_multiplier_dry(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=0.0)
        assert m.grip_multiplier(w) == pytest.approx(1.0, abs=1e-6)

    def test_grip_multiplier_wet_lower(self) -> None:
        m = WeatherImpactModel()
        w_dry = WeatherCondition(precipitation_mm=0.0)
        w_wet = WeatherCondition(precipitation_mm=8.0)
        assert m.grip_multiplier(w_wet) < m.grip_multiplier(w_dry)

    def test_lap_time_delta_zero_dry(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=0.0)
        assert m.lap_time_delta(w, 90.0) == 0.0

    def test_lap_time_delta_positive_wet(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=8.0)
        assert m.lap_time_delta(w, 90.0) > 0

    def test_downforce_loss_zero_dry(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=0.0)
        assert m.downforce_loss(w) == pytest.approx(0.0, abs=1e-6)

    def test_downforce_loss_positive_wet(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=8.0)
        assert m.downforce_loss(w) > 0

    def test_tire_temp_impact_hot_positive(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(ambient_temp_c=40.0, precipitation_mm=0.0)
        assert m.tire_temp_impact(w) > 0

    def test_tire_temp_impact_wet_negative(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(ambient_temp_c=20.0, precipitation_mm=8.0)
        assert m.tire_temp_impact(w) < 0

    def test_visibility_impact_in_range(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(visibility_m=5000.0)
        v = m.visibility_impact(w)
        assert 0.0 <= v <= 1.0

    def test_setup_adjustments_dry_hot(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=0.0, track_temp_c=40.0)
        recs = m.setup_adjustment_recommendations(w, "high_downforce")
        assert any(r["field"] == "front_tyre_pressure" for r in recs)

    def test_setup_adjustments_wet(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=8.0)
        recs = m.setup_adjustment_recommendations(w, "high_downforce")
        fields = [r["field"] for r in recs]
        assert "front_ride_height" in fields
        assert "rear_wing" in fields
        assert "front_suspension" in fields

    def test_setup_adjustments_heavy_wet_has_arb(self) -> None:
        m = WeatherImpactModel()
        w = WeatherCondition(precipitation_mm=12.0)  # wet
        recs = m.setup_adjustment_recommendations(w, "street")
        assert any(r["field"] == "front_arb" for r in recs)


# --------------------------------------------------------------------------- #
# WeatherForecast
# --------------------------------------------------------------------------- #
class TestWeatherForecast:
    def test_forecast_initial(self) -> None:
        w = WeatherCondition(ambient_temp_c=20.0)
        f = WeatherForecast(w)
        assert f.forecast_at(0).ambient_temp_c == 20.0

    def test_forecast_interpolation(self) -> None:
        f = WeatherForecast(WeatherCondition(ambient_temp_c=20.0))
        f.add_change(10, WeatherCondition(ambient_temp_c=30.0))
        mid = f.forecast_at(5)
        assert mid.ambient_temp_c == pytest.approx(25.0, abs=0.5)

    def test_will_change_dry_to_wet(self) -> None:
        f = WeatherForecast(WeatherCondition(precipitation_mm=0.0))
        f.add_change(20, WeatherCondition(precipitation_mm=8.0))
        assert f.will_change_dry_to_wet()

    def test_will_change_wet_to_dry(self) -> None:
        f = WeatherForecast(WeatherCondition(precipitation_mm=8.0))
        f.add_change(20, WeatherCondition(precipitation_mm=0.0))
        assert f.will_change_wet_to_dry()

    def test_no_change_returns_false(self) -> None:
        f = WeatherForecast(WeatherCondition(precipitation_mm=0.0))
        assert not f.will_change_dry_to_wet()

    def test_strategy_recommendation_dry_to_wet(self) -> None:
        f = WeatherForecast(WeatherCondition(precipitation_mm=0.0))
        f.add_change(20, WeatherCondition(precipitation_mm=8.0))
        rec = f.strategy_recommendation()
        assert "雨" in rec

    def test_strategy_recommendation_stable(self) -> None:
        f = WeatherForecast(WeatherCondition(precipitation_mm=0.0))
        rec = f.strategy_recommendation()
        assert "稳定" in rec


# --------------------------------------------------------------------------- #
# WindModel
# --------------------------------------------------------------------------- #
class TestWindModel:
    def test_headwind_component(self) -> None:
        m = WindModel()
        # Wind from same direction as car = headwind.
        h = m.headwind_component(10.0, wind_dir_deg=0.0, car_dir_deg=0.0)
        assert h == pytest.approx(10.0, abs=0.1)

    def test_tailwind_component(self) -> None:
        m = WindModel()
        # Wind from opposite direction = tailwind.
        h = m.headwind_component(10.0, wind_dir_deg=180.0, car_dir_deg=0.0)
        assert h == pytest.approx(-10.0, abs=0.1)

    def test_crosswind_component(self) -> None:
        m = WindModel()
        c = m.crosswind_component(10.0, wind_dir_deg=90.0, car_dir_deg=0.0)
        assert abs(c) == pytest.approx(10.0, abs=0.1)

    def test_lap_time_wind_effect_finite(self) -> None:
        m = WindModel()
        e = m.lap_time_wind_effect(10.0, wind_dir_deg=45.0, track_bearing_deg=0.0)
        assert isinstance(e, float)

    def test_lap_time_wind_effect_headwind_slower(self) -> None:
        m = WindModel()
        # Strong headwind → positive (slower).
        e = m.lap_time_wind_effect(15.0, wind_dir_deg=0.0, track_bearing_deg=0.0)
        assert e > 0

    def test_aero_balance_shift_in_range(self) -> None:
        m = WindModel()
        for cross in [-30, -10, 0, 10, 30]:
            assert -1.0 <= m.aero_balance_shift(cross) <= 1.0
