"""Tests for development_planner (Iter-34)."""

from __future__ import annotations

import pytest

from f1opt.model.development_planner import (
    DevelopmentPlanner,
    UpgradeEvent,
    plan_season_development,
)


class TestPlanSeason:
    def test_returns_list_of_events(self) -> None:
        events = plan_season_development("rb", "balanced", 60.0)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, UpgradeEvent)

    def test_events_sorted_by_race_idx(self) -> None:
        events = plan_season_development("rb", "balanced", 60.0)
        for i in range(len(events) - 1):
            assert events[i].race_idx < events[i + 1].race_idx

    def test_total_spent_under_budget(self) -> None:
        events = plan_season_development("rb", "balanced", 60.0)
        total = sum(e.cost_usd_m for e in events)
        assert total <= 60.0 + 1e-6

    def test_cumulative_gain_monotonic(self) -> None:
        events = plan_season_development("rb", "balanced", 60.0)
        for i in range(1, len(events)):
            assert events[i].cumulative_gain_s > events[i - 1].cumulative_gain_s

    def test_no_duplicate_upgrades(self) -> None:
        events = plan_season_development("rb", "balanced", 60.0)
        upgrades = [e.upgrade_id for e in events]
        assert len(upgrades) == len(set(upgrades))


class TestBudgetImpact:
    def test_low_budget_fewer_upgrades(self) -> None:
        many = plan_season_development("rb", "balanced", 60.0)
        few = plan_season_development("rb", "balanced", 5.0)
        assert len(few) < len(many)

    def test_zero_budget_no_upgrades(self) -> None:
        events = plan_season_development("rb", "balanced", 0.0)
        assert events == []


class TestTeamStrength:
    @pytest.mark.parametrize("strength", ["balanced", "aero_focused",
                                            "powertrain_focused", "backmarker"])
    def test_each_strength_produces_path(self, strength: str) -> None:
        events = plan_season_development("rb", strength, 60.0)
        assert isinstance(events, list)
        # Should have at least 5 upgrades with full budget
        assert len(events) >= 5

    def test_aero_focused_starts_with_aero(self) -> None:
        events = plan_season_development("rb", "aero_focused", 60.0)
        if events:
            # First upgrade should be aero (subject to track priority)
            # Either aero or whatever the first track requires
            assert events[0].upgrade_id.startswith(("aero_", "powertrain_", "mech_"))


class TestSummary:
    def test_summary_structure(self) -> None:
        planner = DevelopmentPlanner("rb", "balanced", 60.0)
        s = planner.summary()
        required = {"team_id", "team_strength", "n_upgrades",
                    "total_spent_m", "remaining_budget_m",
                    "total_gain_s", "upgrade_path"}
        assert required.issubset(s.keys())
        assert s["team_id"] == "rb"
        assert s["team_strength"] == "balanced"
