"""Tests for cost_cap (Iter-33)."""

from __future__ import annotations

import pytest

from f1opt.model.cost_cap import (
    CostCapBudget,
    UpgradeDecision,
    allocate_budget,
)


class TestBudget:
    def test_initial_budget(self) -> None:
        b = CostCapBudget(team_id="rb")
        assert b.remaining_budget_m > 0
        assert b.total_spent_m == 0

    def test_spend_reduces_budget(self) -> None:
        b = CostCapBudget(team_id="rb", initial_budget_m=10.0)
        before = b.remaining_budget_m
        assert b.spend(race_idx=1, upgrade_id="aero_front_wing")
        assert b.remaining_budget_m == before - 1.8

    def test_cannot_afford_raises(self) -> None:
        b = CostCapBudget(team_id="rb", initial_budget_m=2.0)
        # powertrain_ice costs 5.5
        assert not b.can_afford("powertrain_ice")
        assert not b.spend(race_idx=1, upgrade_id="powertrain_ice")

    def test_unknown_upgrade_raises(self) -> None:
        b = CostCapBudget(team_id="rb")
        with pytest.raises(ValueError):
            b.evaluate_upgrade("unknown_upgrade")


class TestEvaluation:
    def test_evaluate_returns_decision(self) -> None:
        b = CostCapBudget(team_id="rb")
        d = b.evaluate_upgrade("aero_floor")
        assert isinstance(d, UpgradeDecision)
        assert d.upgrade_id == "aero_floor"
        assert d.cost_usd_m == 2.5
        assert d.expected_gain_s == 0.45

    def test_cost_per_tenth_positive(self) -> None:
        b = CostCapBudget(team_id="rb")
        d = b.evaluate_upgrade("aero_floor")
        assert d.cost_per_tenth_s > 0
        # cost 2.5 / (0.45 * 10) = 0.556
        assert d.cost_per_tenth_s == pytest.approx(0.556, rel=1e-2)


class TestRecommendation:
    def test_recommend_returns_best_efficiency(self) -> None:
        b = CostCapBudget(team_id="rb", initial_budget_m=60.0)
        r = b.recommend_next_upgrade()
        assert r is not None
        assert isinstance(r, UpgradeDecision)

    def test_recommend_no_repeat(self) -> None:
        b = CostCapBudget(team_id="rb", initial_budget_m=60.0)
        first = b.recommend_next_upgrade()
        b.spend(race_idx=1, upgrade_id=first.upgrade_id)
        second = b.recommend_next_upgrade()
        assert second is not None
        assert second.upgrade_id != first.upgrade_id

    def test_recommend_none_when_broke(self) -> None:
        b = CostCapBudget(team_id="rb", initial_budget_m=0.5)
        r = b.recommend_next_upgrade()
        # Nothing affordable
        assert r is None


class TestCumulative:
    def test_total_gain_increases_with_spend(self) -> None:
        b = CostCapBudget(team_id="rb", initial_budget_m=60.0)
        before = b.total_performance_gain_s()
        b.spend(race_idx=1, upgrade_id="aero_floor")
        after = b.total_performance_gain_s()
        assert after > before
        assert after == pytest.approx(before + 0.45, rel=1e-3)

    def test_summary_structure(self) -> None:
        b = CostCapBudget(team_id="rb", initial_budget_m=60.0)
        b.spend(race_idx=1, upgrade_id="aero_floor")
        s = b.summary()
        required = {"team_id", "initial_budget_m", "total_spent_m",
                    "remaining_budget_m", "n_upgrades", "total_gain_s",
                    "upgrades_done"}
        assert required.issubset(s.keys())
        assert s["n_upgrades"] == 1
        assert s["upgrades_done"] == ["aero_floor"]


class TestAllocateBudget:
    @pytest.mark.parametrize("strategy", ["balanced", "aero_focused",
                                            "powertrain_focused", "backmarker"])
    def test_returns_full_list(self, strategy: str) -> None:
        result = allocate_budget(strategy)
        assert len(result) == 11  # all 11 upgrades
        assert len(set(result)) == 11  # no duplicates

    def test_aero_focused_starts_with_aero(self) -> None:
        result = allocate_budget("aero_focused")
        assert result[0].startswith("aero_")

    def test_backmarker_starts_with_cheap(self) -> None:
        result = allocate_budget("backmarker")
        # First item should be cheap (front_wing 1.8 or rear_wing 1.5)
        from f1opt.model.cost_cap import _UPGRADE_COSTS
        assert _UPGRADE_COSTS[result[0]] <= 2.0
