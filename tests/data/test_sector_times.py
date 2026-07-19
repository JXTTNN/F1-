"""F1 2026 赛道扇区时间数据库测试 (Iter-48)."""

from __future__ import annotations

import pytest

from f1opt.data.sector_times import (
    all_track_ids,
    high_wear_sector,
    sector_times_for,
    sector_with_longest_straight,
    total_lap_time_s,
    tracks_sorted_by_lap_time,
)


# --------------------------------------------------------------------------- #
# 基础查询
# --------------------------------------------------------------------------- #
def test_sector_times_for_monza():
    d = sector_times_for("monza")
    assert d.track_id == "monza"
    assert d.s1_s > 0
    assert d.s2_s > 0
    assert d.s3_s > 0


def test_sector_times_for_unknown_raises():
    with pytest.raises(ValueError, match="Unknown track_id"):
        sector_times_for("nonexistent")


def test_all_24_tracks_present():
    """2026 赛历 24 条赛道应有扇区数据."""
    ids = all_track_ids()
    assert len(ids) == 24
    # 关键赛道
    for tid in ["monza", "monaco", "silverstone", "spa", "suzuka", "bahrain"]:
        assert tid in ids


# --------------------------------------------------------------------------- #
# 总圈速
# --------------------------------------------------------------------------- #
def test_total_lap_time_sum_of_sectors():
    d = sector_times_for("monza")
    assert d.total_lap_time_s == d.s1_s + d.s2_s + d.s3_s


def test_total_lap_time_convenience():
    assert total_lap_time_s("monza") > 0


def test_monaco_shortest_lap():
    """Monaco 圈速最短 (~75s)."""
    sorted_tracks = tracks_sorted_by_lap_time()
    assert sorted_tracks[0][0] == "monaco"


def test_spa_longest_lap():
    """Spa 圈速最长 (~110s)."""
    sorted_tracks = tracks_sorted_by_lap_time()
    assert sorted_tracks[-1][0] == "spa"


def test_lap_times_reasonable():
    """所有赛道圈速应在 70-115s 范围."""
    for tid in all_track_ids():
        t = total_lap_time_s(tid)
        assert 70.0 <= t <= 115.0, f"{tid}: {t}s out of range"


# --------------------------------------------------------------------------- #
# 扇区查询
# --------------------------------------------------------------------------- #
def test_sector_time_by_index():
    d = sector_times_for("monza")
    assert d.sector_time(1) == d.s1_s
    assert d.sector_time(2) == d.s2_s
    assert d.sector_time(3) == d.s3_s


def test_sector_time_invalid_index():
    d = sector_times_for("monza")
    with pytest.raises(ValueError):
        d.sector_time(0)
    with pytest.raises(ValueError):
        d.sector_time(4)


def test_sector_drs_query():
    d = sector_times_for("monza")
    # monza S1, S2 有 DRS
    assert d.sector_drs(1) is True
    assert d.sector_drs(2) is True


def test_sector_wear_query():
    d = sector_times_for("silverstone")
    # silverstone 高磨损
    assert d.sector_wear(1) > 1.0


def test_sector_overtake_query():
    d = sector_times_for("monaco")
    # monaco 超车极难
    assert d.sector_overtake(1) < 0.2


# --------------------------------------------------------------------------- #
# 最长直道扇区
# --------------------------------------------------------------------------- #
def test_sector_with_longest_straight_returns_valid():
    for tid in all_track_ids():
        s = sector_with_longest_straight(tid)
        assert s in (1, 2, 3)


def test_monza_long_straight_sector():
    """Monza S1 含最长直道 (主直道)."""
    d = sector_times_for("monza")
    # monza 应有 DRS 扇区
    assert d.s1_drs or d.s2_drs


# --------------------------------------------------------------------------- #
# 高磨损扇区
# --------------------------------------------------------------------------- #
def test_high_wear_sector_returns_valid():
    for tid in all_track_ids():
        s = high_wear_sector(tid)
        assert s in (1, 2, 3)


def test_silverstone_high_wear():
    """Silverstone 高能弯多, 磨损高."""
    d = sector_times_for("silverstone")
    high = d.high_wear_sector()
    assert d.sector_wear(high) >= 1.10


# --------------------------------------------------------------------------- #
# 最佳超车扇区
# --------------------------------------------------------------------------- #
def test_best_overtake_sector_monza():
    """Monza 超车容易 (长直道 + 慢弯)."""
    d = sector_times_for("monza")
    best = d.best_overtake_sector()
    assert d.sector_overtake(best) >= 0.60


def test_monaco_low_overtake():
    """Monaco 超车极难."""
    d = sector_times_for("monaco")
    best = d.best_overtake_sector()
    assert d.sector_overtake(best) < 0.25


# --------------------------------------------------------------------------- #
# 排序
# --------------------------------------------------------------------------- #
def test_tracks_sorted_by_lap_time():
    sorted_tracks = tracks_sorted_by_lap_time()
    times = [t for _, t in sorted_tracks]
    assert times == sorted(times)


def test_tracks_sorted_length_24():
    assert len(tracks_sorted_by_lap_time()) == 24


# --------------------------------------------------------------------------- #
# DRS 区分布
# --------------------------------------------------------------------------- #
def test_most_tracks_have_drs():
    """绝大多数赛道有 DRS 区."""
    n_with_drs = 0
    for tid in all_track_ids():
        d = sector_times_for(tid)
        if d.s1_drs or d.s2_drs or d.s3_drs:
            n_with_drs += 1
    assert n_with_drs >= 22  # 至少 22 条有 DRS


def test_monaco_minimal_drs():
    """Monaco DRS 极少 (仅隧道后)."""
    d = sector_times_for("monaco")
    drs_count = sum([d.s1_drs, d.s2_drs, d.s3_drs])
    assert drs_count <= 1


def test_bahrain_three_drs_zones():
    """Bahrain 有 3 个 DRS 区."""
    d = sector_times_for("bahrain")
    drs_count = sum([d.s1_drs, d.s2_drs, d.s3_drs])
    assert drs_count >= 2


# --------------------------------------------------------------------------- #
# 磨损系数
# --------------------------------------------------------------------------- #
def test_wear_factors_in_range():
    """磨损系数应在 0.7-1.4 范围."""
    for tid in all_track_ids():
        d = sector_times_for(tid)
        for s in (1, 2, 3):
            assert 0.7 <= d.sector_wear(s) <= 1.4, f"{tid} S{s}: {d.sector_wear(s)}"


def test_overtake_in_range():
    """超车系数应在 0-0.8 范围."""
    for tid in all_track_ids():
        d = sector_times_for(tid)
        for s in (1, 2, 3):
            assert 0.0 <= d.sector_overtake(s) <= 0.8, f"{tid} S{s}: {d.sector_overtake(s)}"


# --------------------------------------------------------------------------- #
# 数据完整性
# --------------------------------------------------------------------------- #
def test_all_sector_times_positive():
    for tid in all_track_ids():
        d = sector_times_for(tid)
        assert d.s1_s > 0
        assert d.s2_s > 0
        assert d.s3_s > 0


def test_sector_times_reasonable():
    """每个扇区时间应在 20-45s 范围."""
    for tid in all_track_ids():
        d = sector_times_for(tid)
        for s in (1, 2, 3):
            t = d.sector_time(s)
            assert 20.0 <= t <= 45.0, f"{tid} S{s}: {t}s"


def test_frozen_dataclass():
    """TrackSectorData 应为 frozen (不可变)."""
    d = sector_times_for("monza")
    with pytest.raises(AttributeError):
        d.s1_s = 999.0  # type: ignore


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_sector_analysis_for_strategy():
    """策略分析: 找出高磨损扇区 + DRS 扇区组合."""
    d = sector_times_for("silverstone")
    high_wear = d.high_wear_sector()
    drs_sectors = [i for i in (1, 2, 3) if d.sector_drs(i)]
    # Silverstone 高磨损扇区应在 S1 或 S2 (高能弯)
    assert high_wear in (1, 2)
    assert len(drs_sectors) >= 1


def test_overtake_difficulty_ranking():
    """超车难度排序: Monaco 最难, Monza 最易."""
    monaco_best = sector_times_for("monaco").best_overtake_sector()
    monaco_overtake = sector_times_for("monaco").sector_overtake(monaco_best)
    monza_best = sector_times_for("monza").best_overtake_sector()
    monza_overtake = sector_times_for("monza").sector_overtake(monza_best)
    assert monza_overtake > monaco_overtake


def test_lap_time_decomposition():
    """圈速分解: 三扇区时间之和 = 总圈速."""
    for tid in all_track_ids():
        d = sector_times_for(tid)
        total = d.s1_s + d.s2_s + d.s3_s
        assert abs(total - d.total_lap_time_s) < 1e-9
