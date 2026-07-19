"""Tests for LiveStrategyAdvisor (Iter-23)."""

from __future__ import annotations

from f1opt.model.strategy_optimizer import LiveStrategyAdvisor, LiveStrategyDecision


def _advisor() -> LiveStrategyAdvisor:
    return LiveStrategyAdvisor(track_id="monza", total_laps=58, pit_loss_s=23.0)


# --------------------------------------------------------------------------- #
# SC/VSC pit decision
# --------------------------------------------------------------------------- #
class TestSafetyCarPitDecision:
    def test_sc_active_pits_with_old_tire(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=15,
            sc_active=True, sc_remaining_laps=3,
            position=4, gap_ahead_s=2.5, gap_behind_s=8.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
        )
        assert d.should_pit is True
        assert "SC" in d.reason or "cheap" in d.reason
        assert d.urgency >= 0.9
        assert d.new_compound == "hard"
        assert d.remaining_pit_laps == ()
        assert d.remaining_compounds == ()

    def test_sc_active_no_pit_with_fresh_tire(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=1,
            sc_active=True, sc_remaining_laps=3,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
        )
        assert d.should_pit is False

    def test_vsc_uses_higher_discount(self) -> None:
        adv = _advisor()
        # VSC discount 0.55 → effective loss 12.65
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=15,
            sc_active=True, sc_remaining_laps=3,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            is_vsc=True,
        )
        assert d.should_pit is True
        # VSC reason should mention 12.65 not 4.6
        assert "12.6" in d.reason or "12.7" in d.reason

    def test_sc_no_pit_when_only_2_laps_left(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=56, current_compound="medium", tire_age_laps=15,
            sc_active=True, sc_remaining_laps=3,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
        )
        # remaining=2, no pit
        assert d.should_pit is False


# --------------------------------------------------------------------------- #
# Tire cliff avoidance
# --------------------------------------------------------------------------- #
class TestCliffAvoidance:
    def test_high_wear_triggers_pit(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="soft", tire_age_laps=18,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            tire_wear_pct=75.0,
        )
        assert d.should_pit is True
        assert "cliff" in d.reason.lower()
        assert d.urgency >= 0.8

    def test_moderate_wear_no_pit(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=10,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            tire_wear_pct=40.0,
        )
        assert d.should_pit is False


# --------------------------------------------------------------------------- #
# Weather crossover
# --------------------------------------------------------------------------- #
class TestWeatherCrossover:
    def test_dry_to_wet_crossover_pits(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=10,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            weather_recommended_compound="wet",
        )
        assert d.should_pit is True
        assert "crossover" in d.reason.lower() or "weather" in d.reason.lower()
        assert d.new_compound == "wet"

    def test_same_compound_no_pit(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=10,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            weather_recommended_compound="medium",  # same as current
        )
        assert d.should_pit is False

    def test_wet_to_dry_crossover_pits(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=30, current_compound="intermediate", tire_age_laps=8,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(45,), remaining_compounds=("medium",),
            weather_recommended_compound="medium",
        )
        assert d.should_pit is True
        assert d.new_compound == "medium"


# --------------------------------------------------------------------------- #
# Undercut logic
# --------------------------------------------------------------------------- #
class TestUndercut:
    def test_undercut_attempt_when_close_ahead(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=18, current_compound="medium", tire_age_laps=15,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=1.5, gap_behind_s=20.0,
            remaining_pit_laps=(20,), remaining_compounds=("hard",),
            tire_wear_pct=50.0,
        )
        assert d.should_pit is True
        assert "undercut" in d.reason.lower()
        assert d.urgency >= 0.5

    def test_no_undercut_when_far_ahead(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=18, current_compound="medium", tire_age_laps=15,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=40.0, gap_behind_s=20.0,
            remaining_pit_laps=(20,), remaining_compounds=("hard",),
            tire_wear_pct=50.0,
        )
        assert d.should_pit is False

    def test_no_undercut_with_fresh_tire(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=18, current_compound="medium", tire_age_laps=2,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=1.0, gap_behind_s=20.0,
            remaining_pit_laps=(20,), remaining_compounds=("hard",),
            tire_wear_pct=10.0,
        )
        assert d.should_pit is False


# --------------------------------------------------------------------------- #
# Late race hold position
# --------------------------------------------------------------------------- #
class TestLateRace:
    def test_late_race_no_pit(self) -> None:
        """Late race with no remaining pit stops and high wear — no pit."""
        adv = _advisor()
        # lap=56, total=58, remaining=2 → late race triggers before cliff
        d = adv.decide(
            lap=56, current_compound="medium", tire_age_laps=15,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(), remaining_compounds=(),
            tire_wear_pct=80.0,
        )
        assert d.should_pit is False
        assert "late" in d.reason.lower() or "hold" in d.reason.lower()

    def test_very_late_no_pit_even_high_wear(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=56, current_compound="medium", tire_age_laps=15,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(), remaining_compounds=(),
            tire_wear_pct=80.0,
        )
        assert d.should_pit is False
        assert "late" in d.reason.lower() or "hold" in d.reason.lower()


# --------------------------------------------------------------------------- #
# Defensive logic
# --------------------------------------------------------------------------- #
class TestDefensive:
    def test_defensive_when_close_behind_and_fresh_tire(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=5,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=15.0, gap_behind_s=2.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            tire_wear_pct=15.0,
        )
        assert d.should_pit is False
        assert "defensive" in d.reason.lower()

    def test_no_defensive_when_far_behind(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=5,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=15.0, gap_behind_s=30.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            tire_wear_pct=15.0,
        )
        # Defensive doesn't trigger → default behavior
        assert d.should_pit is False
        assert "planned" in d.reason.lower() or "following" in d.reason.lower()


# --------------------------------------------------------------------------- #
# Default behavior
# --------------------------------------------------------------------------- #
class TestDefaultBehavior:
    def test_default_no_pit_when_nothing_special(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=5,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=15.0, gap_behind_s=30.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            tire_wear_pct=15.0,
        )
        assert d.should_pit is False
        assert "planned" in d.reason.lower()

    def test_decision_has_all_fields(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=5,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=15.0, gap_behind_s=30.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            tire_wear_pct=15.0,
        )
        assert isinstance(d, LiveStrategyDecision)
        assert isinstance(d.should_pit, bool)
        assert isinstance(d.reason, str)
        assert isinstance(d.remaining_pit_laps, tuple)
        assert isinstance(d.remaining_compounds, tuple)
        assert 0.0 <= d.urgency <= 1.0
        assert isinstance(d.estimated_gain_s, float)


# --------------------------------------------------------------------------- #
# Gain estimation sanity
# --------------------------------------------------------------------------- #
class TestGainEstimation:
    def test_sc_pit_gain_positive_when_old_tire(self) -> None:
        adv = _advisor()
        # SC pit discount 0.20, so effective loss = 4.6
        # Tire age 15 → penalty_per_lap = 0.6, remaining 38 → 22.8
        # Warmup 0.4, sc_saving = 23 - 4.6 = 18.4
        # gain = 22.8 - 0.4 + 18.4 = 40.8
        d = adv.decide(
            lap=20, current_compound="medium", tire_age_laps=15,
            sc_active=True, sc_remaining_laps=3,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
        )
        assert d.estimated_gain_s > 0.0
        assert d.estimated_gain_s > 15.0  # at least 15s gain

    def test_cliff_gain_positive_when_many_laps_left(self) -> None:
        adv = _advisor()
        d = adv.decide(
            lap=15, current_compound="soft", tire_age_laps=15,
            sc_active=False, sc_remaining_laps=0,
            position=4, gap_ahead_s=10.0, gap_behind_s=10.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
            tire_wear_pct=75.0,
        )
        # remaining = 43, cliff penalty 1.2/lap × 38 = 45.6
        # minus warmup 0.4, minus pit_loss 23 → 22.2
        assert d.estimated_gain_s > 10.0
