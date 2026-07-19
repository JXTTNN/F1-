"""Tests for :mod:`f1opt.feedback.comparison` (comparative lap/sector analysis).

Covers the four public classes:

- :class:`LapComparator` — reference comparison, sector deltas,
  strength/weakness detection, verdict, no-reference handling, batch compare,
  lap ranking.
- :class:`SectorAnalyzer` — averages/bests, theoretical best (perfect lap),
  non-negative delta, weak/strong sector, potential gain, corner-count-adjusted
  strength map.
- :class:`TeammateComparison` — head-to-head keys, gap computation, sectors
  won, consistency comparison, qualifying-gap prediction.
- :class:`SetupChangeImpact` — impact keys, significance flag, verdict text,
  consistency delta, and empty-input edge cases.
"""

from __future__ import annotations

import pytest

from f1opt.feedback.comparison import (
    LapComparator,
    SectorAnalyzer,
    SetupChangeImpact,
    TeammateComparison,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _lap(
    lap_time: float,
    sectors: list[float],
    avg_speed: float = 200.0,
    max_speed: float = 300.0,
) -> dict:
    return {
        "lap_time": lap_time,
        "sector_times": list(sectors),
        "avg_speed": avg_speed,
        "max_speed": max_speed,
    }


def _teammate_comparison() -> TeammateComparison:
    """A driver-vs-teammate scenario with a clear sector winner each sector.

    driver sector bests: s1=29, s2=30, s3=30
    teammate sector bests: s1=30, s2=29, s3=30.5
    -> driver wins s1 & s3 (2), teammate wins s2 (1), no ties.
    driver best lap = 89.0, teammate best lap = 90.5 -> gap_best = -1.5.
    """
    return TeammateComparison(
        driver_laps=[_lap(90.0, [30, 30, 30]), _lap(89.0, [29, 30, 30])],
        teammate_laps=[_lap(91.0, [31, 29, 31]), _lap(90.5, [30, 30, 30.5])],
    )


# --------------------------------------------------------------------------- #
# LapComparator
# --------------------------------------------------------------------------- #
_COMPARE_KEYS = {
    "lap_time_delta",
    "sector_deltas",
    "speed_deltas",
    "strength_sector",
    "weakness_sector",
    "verdict",
}


def test_compare_with_reference_returns_all_keys() -> None:
    ref = _lap(90.0, [30, 30, 30])
    out = LapComparator(ref).compare(_lap(89.0, [29, 30, 30]))
    assert _COMPARE_KEYS <= set(out)


def test_compare_faster_lap_negative_delta() -> None:
    ref = _lap(90.0, [30, 30, 30])
    out = LapComparator(ref).compare(_lap(89.0, [29, 30, 30]))
    assert out["lap_time_delta"] < 0
    assert out["lap_time_delta"] == pytest.approx(-1.0)


def test_compare_sector_deltas_correct_length() -> None:
    ref = _lap(90.0, [30, 30, 30])
    out = LapComparator(ref).compare(_lap(90.0, [29, 30, 31]))
    assert len(out["sector_deltas"]) == 3
    assert out["sector_deltas"] == pytest.approx([-1.0, 0.0, 1.0])


def test_compare_strength_weakness_sectors_identified() -> None:
    ref = _lap(90.0, [30, 30, 30])
    # s1 faster (-2), s2 unchanged, s3 slower (+2) vs ref.
    out = LapComparator(ref).compare(_lap(90.0, [28, 30, 32]))
    assert out["strength_sector"] == 1
    assert out["weakness_sector"] == 3


def test_compare_verdict_contains_fast_or_slow() -> None:
    ref = _lap(90.0, [30, 30, 30])
    faster = LapComparator(ref).compare(_lap(89.0, [29, 30, 30]))
    slower = LapComparator(ref).compare(_lap(91.0, [30, 30, 31]))
    assert "快" in faster["verdict"]
    assert "慢" in slower["verdict"]


def test_compare_no_reference_returns_relative_deltas() -> None:
    cmp = LapComparator(None)
    out = cmp.compare(_lap(90.0, [30, 30, 30]))
    # None handled gracefully: all keys present, neutral deltas.
    assert _COMPARE_KEYS <= set(out)
    assert isinstance(out["sector_deltas"], list) and len(out["sector_deltas"]) == 3
    assert out["lap_time_delta"] == 0.0
    assert out["verdict"] == "无参考圈速"


def test_compare_multi_returns_list() -> None:
    ref = _lap(90.0, [30, 30, 30])
    laps = [_lap(89.0, [29, 30, 30]), _lap(91.0, [30, 30, 31])]
    out = LapComparator(ref).compare_multi(laps)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(_COMPARE_KEYS <= set(r) for r in out)


def test_rank_laps_sorted_ascending_by_lap_time() -> None:
    laps = [
        _lap(92.0, [30, 31, 31]),
        _lap(89.0, [29, 30, 30]),
        _lap(90.0, [30, 30, 30]),
    ]
    ranked = LapComparator().rank_laps(laps)
    times = [lap["lap_time"] for _, lap in ranked]
    assert times == sorted(times)
    # original index of the fastest lap (89.0) is 1
    assert ranked[0][0] == 1
    assert ranked[0][1]["lap_time"] == 89.0


# --------------------------------------------------------------------------- #
# SectorAnalyzer
# --------------------------------------------------------------------------- #
def test_sector_averages_and_bests_correct() -> None:
    out = SectorAnalyzer().analyze([[30, 30, 30], [28, 32, 30], [29, 31, 29]])
    # sector 1: avg 29, min 28 ; sector 3: min 29
    assert out["sector_averages"][0] == pytest.approx(29.0)
    assert out["sector_bests"][0] == pytest.approx(28.0)
    assert out["sector_bests"][2] == pytest.approx(29.0)


def test_theoretical_best_equals_sum_of_bests() -> None:
    out = SectorAnalyzer().analyze([[30, 30, 30], [28, 32, 30], [29, 31, 29]])
    assert out["theoretical_best"] == pytest.approx(sum(out["sector_bests"]))
    # sector bests = [28, 30, 29] -> 87
    assert out["theoretical_best"] == pytest.approx(87.0)


def test_theoretical_best_delta_nonneg() -> None:
    out = SectorAnalyzer().analyze([[30, 30, 30], [28, 32, 30], [29, 31, 29]])
    assert out["theoretical_best_delta"] >= 0
    # actual best lap = min(90, 90, 89) = 89 ; theoretical = 87 -> delta = 2
    assert out["theoretical_best_delta"] == pytest.approx(2.0)


def test_weak_sector_in_range() -> None:
    out = SectorAnalyzer().analyze([[30, 30, 30], [28, 32, 30], [29, 31, 29]])
    assert out["weak_sector"] in {1, 2, 3}
    assert out["strong_sector"] in {1, 2, 3}


def test_potential_gain_nonneg() -> None:
    out = SectorAnalyzer().analyze([[30, 30, 30], [28, 32, 30], [29, 31, 29]])
    assert out["potential_gain_s"] >= 0
    # gain = sum(avg - best) = (29-28)+(31-30)+(29.667-29) = 1 + 1 + 0.667
    assert out["potential_gain_s"] == pytest.approx(
        sum(
            a - b
            for a, b in zip(out["sector_averages"], out["sector_bests"], strict=True)
        )
    )


def test_corner_strength_map_returns_three_sectors() -> None:
    out = SectorAnalyzer().corner_strength_map(
        [[30, 30, 30], [28, 32, 30], [29, 31, 29]], [6, 8, 5]
    )
    assert isinstance(out, dict)
    assert len(out) == 3
    assert set(out) == {1, 2, 3}
    for v in out.values():
        assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------------- #
# TeammateComparison
# --------------------------------------------------------------------------- #
_H2H_KEYS = {
    "driver_best",
    "teammate_best",
    "driver_avg",
    "teammate_avg",
    "gap_best_s",
    "gap_avg_s",
    "sectors_won_driver",
    "sectors_won_teammate",
    "consistency_comparison",
    "verdict",
}


def test_head_to_head_returns_all_keys() -> None:
    out = _teammate_comparison().head_to_head()
    assert _H2H_KEYS <= set(out)


def test_gap_best_s_computed_correctly() -> None:
    out = _teammate_comparison().head_to_head()
    # driver_best 89.0, teammate_best 90.5 -> gap = -1.5 (driver faster)
    assert out["gap_best_s"] == pytest.approx(-1.5)


def test_sectors_won_counts_sum_to_three() -> None:
    out = _teammate_comparison().head_to_head()
    assert out["sectors_won_driver"] + out["sectors_won_teammate"] == 3


def test_consistency_comparison_has_better_field() -> None:
    out = _teammate_comparison().head_to_head()
    cc = out["consistency_comparison"]
    assert "better" in cc
    assert cc["better"] in {"driver", "teammate"}
    assert "driver_cv" in cc and "teammate_cv" in cc


def test_qualifying_prediction_returns_required_keys() -> None:
    out = _teammate_comparison().qualifying_prediction()
    assert {"predicted_pole_s", "confidence", "reasoning"} <= set(out)
    assert out["predicted_pole_s"] >= 0.0
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["reasoning"], str) and out["reasoning"]


# --------------------------------------------------------------------------- #
# SetupChangeImpact
# --------------------------------------------------------------------------- #
_IMPACT_KEYS = {
    "lap_time_delta_avg",
    "lap_time_delta_best",
    "sector_deltas",
    "consistency_delta",
    "verdict",
    "significant",
}


def test_impact_returns_all_keys() -> None:
    before = [_lap(92.0, [31, 31, 30]), _lap(91.5, [30, 31, 30.5])]
    after = [_lap(90.0, [30, 30, 30]), _lap(90.5, [30, 30, 30.5])]
    out = SetupChangeImpact(before, after).impact()
    assert _IMPACT_KEYS <= set(out)
    assert len(out["sector_deltas"]) == 3


def test_significant_true_for_big_improvement() -> None:
    before = [_lap(92.0, [31, 31, 30]), _lap(92.5, [31, 31, 30.5])]
    after = [_lap(90.0, [30, 30, 30]), _lap(90.5, [30, 30, 30.5])]
    out = SetupChangeImpact(before, after).impact()
    # after_avg ~ 90.25, before_avg ~ 92.25 -> delta ~ -2.0 -> significant
    assert out["significant"] is True
    assert out["lap_time_delta_avg"] < 0


def test_impact_verdict_improvement() -> None:
    before = [_lap(92.0, [31, 31, 30]), _lap(92.5, [31, 31, 30.5])]
    after = [_lap(90.0, [30, 30, 30]), _lap(90.5, [30, 30, 30.5])]
    out = SetupChangeImpact(before, after).impact()
    assert "调教改进" in out["verdict"]


def test_impact_verdict_no_change() -> None:
    before = [_lap(90.0, [30, 30, 30])]
    after = [_lap(90.0, [30, 30, 30])]
    out = SetupChangeImpact(before, after).impact()
    assert "无明显变化" in out["verdict"]
    assert out["significant"] is False


def test_impact_consistency_delta_negative_when_after_more_consistent() -> None:
    before = [
        _lap(92.0, [31, 31, 30]),
        _lap(95.0, [32, 32, 31]),
        _lap(90.0, [30, 30, 30]),
    ]
    after = [
        _lap(90.0, [30, 30, 30]),
        _lap(90.1, [30, 30, 30.1]),
        _lap(90.05, [30, 30, 30.05]),
    ]
    out = SetupChangeImpact(before, after).impact()
    # after is far more consistent -> consistency_delta (after_cv - before_cv) < 0
    assert out["consistency_delta"] < 0


# --------------------------------------------------------------------------- #
# Edge cases — empty inputs must be handled gracefully
# --------------------------------------------------------------------------- #
def test_empty_laps_handled_gracefully() -> None:
    sa = SectorAnalyzer()
    # SectorAnalyzer.analyze on empty input
    analyzed = sa.analyze([])
    assert "theoretical_best" in analyzed  # did not crash
    assert analyzed["theoretical_best"] == 0.0

    # corner_strength_map with empty sector_times still returns 3 sectors
    csm = sa.corner_strength_map([], [6, 8, 5])
    assert len(csm) == 3

    # TeammateComparison with no laps
    tc = TeammateComparison([], [])
    h2h = tc.head_to_head()
    assert "verdict" in h2h
    assert h2h["verdict"] == "数据不足"
    qp = tc.qualifying_prediction()
    assert {"predicted_pole_s", "confidence", "reasoning"} <= set(qp)

    # SetupChangeImpact with no laps
    sci = SetupChangeImpact([], [])
    imp = sci.impact()
    assert _IMPACT_KEYS <= set(imp)
    assert imp["significant"] is False
    assert "无明显变化" in imp["verdict"]


def test_rank_laps_empty() -> None:
    assert LapComparator().rank_laps([]) == []
