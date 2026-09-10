"""Tests for f1opt.model.weather (Iter-12)."""

from __future__ import annotations

from f1opt.model.weather import (
    COMPOUND_CROSSOVER,
    WeatherModel,
    WeatherState,
)


# --------------------------------------------------------------------------- #
# WeatherState
# --------------------------------------------------------------------------- #
class TestWeatherState:
    def test_default_is_dry(self) -> None:
        s = WeatherState()
        assert s.is_dry
        assert s.rain_category == "dry"

    def test_rain_categories(self) -> None:
        assert WeatherState(rain_intensity_mmh=0.0).rain_category == "dry"
        assert WeatherState(rain_intensity_mmh=1.0).rain_category == "drizzle"
        assert WeatherState(rain_intensity_mmh=5.0).rain_category == "light"
        assert WeatherState(rain_intensity_mmh=15.0).rain_category == "moderate"
        assert WeatherState(rain_intensity_mmh=30.0).rain_category == "heavy"

    def test_wet_not_dry(self) -> None:
        assert not WeatherState(rain_intensity_mmh=5.0).is_dry
        assert not WeatherState(track_wetness=0.5).is_dry


# --------------------------------------------------------------------------- #
# Wetness dynamics
# --------------------------------------------------------------------------- #
class TestWetnessDynamics:
    def test_rain_increases_wetness(self) -> None:
        wm = WeatherModel()
        wm.step(rain_mmh=10.0, minutes=10.0)
        assert wm.state.track_wetness > 0.0

    def test_no_rain_decreases_wetness(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.8, track_temp_c=40.0))
        wm.step(rain_mmh=0.0, minutes=30.0)
        assert wm.state.track_wetness < 0.8

    def test_wetness_clamped_to_1(self) -> None:
        wm = WeatherModel()
        wm.step(rain_mmh=50.0, minutes=60.0)
        assert wm.state.track_wetness <= 1.0

    def test_wetness_clamped_to_0(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.1, track_temp_c=50.0))
        wm.step(rain_mmh=0.0, minutes=120.0)
        assert wm.state.track_wetness >= 0.0

    def test_hot_track_dries_faster(self) -> None:
        wm_hot = WeatherModel(initial=WeatherState(track_wetness=0.8, track_temp_c=50.0))
        wm_cool = WeatherModel(initial=WeatherState(track_wetness=0.8, track_temp_c=20.0))
        wm_hot.step(rain_mmh=0.0, minutes=20.0)
        wm_cool.step(rain_mmh=0.0, minutes=20.0)
        assert wm_hot.state.track_wetness < wm_cool.state.track_wetness

    def test_track_temp_drops_with_rain(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_temp_c=40.0))
        wm.step(rain_mmh=20.0, minutes=60.0)
        assert wm.state.track_temp_c < 40.0

    def test_track_temp_floor(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_temp_c=15.0))
        wm.step(rain_mmh=50.0, minutes=120.0)
        assert wm.state.track_temp_c >= 12.0


# --------------------------------------------------------------------------- #
# Lap time penalty
# --------------------------------------------------------------------------- #
class TestLapTimePenalty:
    def test_dry_slick_zero_penalty(self) -> None:
        wm = WeatherModel()  # dry
        assert wm.lap_time_penalty("soft") == 0.0
        assert wm.lap_time_penalty("medium") == 0.0
        assert wm.lap_time_penalty("hard") == 0.0

    def test_wet_compound_has_base_penalty(self) -> None:
        """Wet/intermediate have base penalty even in optimal window."""
        wm = WeatherModel(initial=WeatherState(track_wetness=0.5))
        # intermediate optimal window is 0.2-0.65, wetness=0.5 in window
        assert wm.lap_time_penalty("intermediate") > 0.0

    def test_wrong_tire_huge_penalty(self) -> None:
        """Slicks on very wet track → aquaplaning huge penalty."""
        wm = WeatherModel(initial=WeatherState(track_wetness=0.8))
        slick_pen = wm.lap_time_penalty("soft")
        wet_pen = wm.lap_time_penalty("wet")
        assert slick_pen > wet_pen * 5  # slicks catastrophically worse

    def test_intermediate_better_than_slick_in_wet(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.4))
        assert wm.lap_time_penalty("intermediate") < wm.lap_time_penalty("soft")

    def test_wet_better_than_intermediate_in_very_wet(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.9))
        assert wm.lap_time_penalty("wet") < wm.lap_time_penalty("intermediate")

    def test_unknown_compound_zero_penalty(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.5))
        assert wm.lap_time_penalty("nonexistent") == 0.0

    def test_penalty_non_negative(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.3))
        for c in ("soft", "medium", "hard", "intermediate", "wet"):
            assert wm.lap_time_penalty(c) >= 0.0


# --------------------------------------------------------------------------- #
# Compound recommendation
# --------------------------------------------------------------------------- #
class TestRecommendCompound:
    def test_dry_recommends_dry_compound(self) -> None:
        wm = WeatherModel()
        assert wm.recommend_compound() == "medium"

    def test_medium_wet_recommends_intermediate(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.4))
        assert wm.recommend_compound() == "intermediate"

    def test_very_wet_recommends_wet(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.8))
        assert wm.recommend_compound() == "wet"

    def test_recommended_has_lowest_penalty(self) -> None:
        """Recommended compound should have lowest penalty among all."""
        for w in (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
            wm = WeatherModel(initial=WeatherState(track_wetness=w))
            rec = wm.recommend_compound()
            rec_pen = wm.lap_time_penalty(rec)
            for c in ("soft", "medium", "hard", "intermediate", "wet"):
                if c == rec:
                    continue
                assert rec_pen <= wm.lap_time_penalty(c) + 0.01, (
                    f"w={w}: {rec} pen={rec_pen} worse than {c} "
                    f"pen={wm.lap_time_penalty(c)}"
                )


# --------------------------------------------------------------------------- #
# Visibility & follow loss
# --------------------------------------------------------------------------- #
class TestVisibilityFollow:
    def test_dry_full_visibility(self) -> None:
        wm = WeatherModel()
        assert wm.visibility_score() == 1.0
        assert wm.follow_loss_factor() == 1.0

    def test_wet_reduces_visibility(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.8))
        assert wm.visibility_score() < 1.0

    def test_wet_increases_follow_loss(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.8))
        assert wm.follow_loss_factor() > 1.0

    def test_visibility_bounded(self) -> None:
        for w in (0.0, 0.5, 1.0):
            wm = WeatherModel(initial=WeatherState(track_wetness=w))
            v = wm.visibility_score()
            assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------------- #
# Crossover data integrity
# --------------------------------------------------------------------------- #
class TestCrossoverData:
    def test_all_compounds_have_window(self) -> None:
        for c in ("soft", "medium", "hard", "intermediate", "wet"):
            assert c in COMPOUND_CROSSOVER

    def test_windows_ordered(self) -> None:
        """Wet tire window starts higher than intermediate window."""
        assert COMPOUND_CROSSOVER["wet"][0] >= COMPOUND_CROSSOVER["intermediate"][1] - 0.1

    def test_dry_tires_window_starts_at_zero(self) -> None:
        for c in ("soft", "medium", "hard"):
            assert COMPOUND_CROSSOVER[c][0] == 0.0


# --------------------------------------------------------------------------- #
# Summary & reset
# --------------------------------------------------------------------------- #
class TestSummaryReset:
    def test_summary_keys(self) -> None:
        wm = WeatherModel()
        s = wm.summary()
        required = {"rain_intensity_mmh", "rain_category", "track_wetness",
                    "track_temp_c", "is_dry", "recommended_compound",
                    "visibility", "follow_loss_factor"}
        assert required.issubset(s.keys())

    def test_reset_restores_initial(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.5, track_temp_c=25.0))
        wm.step(rain_mmh=20.0, minutes=30.0)  # mutate
        wm.reset()
        assert wm.state.track_wetness == 0.5
        assert wm.state.track_temp_c == 25.0

    def test_zero_minutes_no_change(self) -> None:
        wm = WeatherModel(initial=WeatherState(track_wetness=0.3))
        wm.step(rain_mmh=10.0, minutes=0.0)
        assert wm.state.track_wetness == 0.3


# --------------------------------------------------------------------------- #
# Realistic rain scenario
# --------------------------------------------------------------------------- #
class TestRainScenario:
    def test_rain_then_drying_crossover(self) -> None:
        """模拟一场雨: wetness 升到全湿, 推荐全雨胎; 然后雨停, 推荐切回半雨胎."""
        wm = WeatherModel(initial=WeatherState(track_temp_c=30.0))
        # 雨开始
        wm.step(rain_mmh=15.0, minutes=10.0)
        assert wm.recommend_compound() == "wet"
        # 雨停, 干燥 60 分钟
        wm.step(rain_mmh=0.0, minutes=60.0)
        # 应已切到 intermediate 或 medium
        rec = wm.recommend_compound()
        assert rec in ("intermediate", "medium")
