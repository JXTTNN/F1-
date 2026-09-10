"""Iter-57 测试: F1 2026 DRS 圈速耦合."""

from __future__ import annotations

import timeit

from f1opt.model.drs_coupling import (
    DRS_ZONES_2026,
    all_drs_tracks,
    drs_available,
    drs_lap_gain_s,
    drs_speed_delta_kmh,
    drs_zone_gain_s,
    get_drs_zone_data,
    max_drs_gain_s,
    n_drs_zones,
)


# --------------------------------------------------------------------------- #
# DRS 赛道数据完整性
# --------------------------------------------------------------------------- #
class TestDRSZoneData:
    def test_all_24_tracks_present(self):
        """24 赛道全部有 DRS 数据."""
        assert len(DRS_ZONES_2026) == 24

    def test_known_tracks_drs_counts(self):
        """关键赛道 DRS 区数量正确."""
        cases = [
            ("suzuka", 1),      # 单 DRS 区
            ("monaco", 1),      # 单短 DRS 区
            ("monza", 2),
            ("melbourne", 4),   # 4 DRS 区
            ("singapore", 4),   # 4 DRS 区
            ("losail", 1),
        ]
        for tid, expected in cases:
            assert n_drs_zones(tid) == expected, f"{tid}: {n_drs_zones(tid)} != {expected}"

    def test_zone_data_fields_positive(self):
        """所有赛道 DRS 数据字段为正."""
        for tid, data in DRS_ZONES_2026.items():
            assert data.n_zones >= 1, f"{tid}: n_zones={data.n_zones}"
            assert data.total_zone_length_m > 0, f"{tid}: total={data.total_zone_length_m}"
            assert data.avg_zone_length_m > 0, f"{tid}: avg={data.avg_zone_length_m}"

    def test_avg_zone_consistent_with_total(self):
        """avg_zone = total / n_zones."""
        for _tid, data in DRS_ZONES_2026.items():
            expected_avg = data.total_zone_length_m / data.n_zones
            assert abs(data.avg_zone_length_m - expected_avg) < 0.01

    def test_unknown_track_returns_default(self):
        """未知赛道返回默认 DRS 数据."""
        d = get_drs_zone_data("nonexistent")
        assert d.n_zones == 2
        assert d.total_zone_length_m > 0

    def test_max_lap_gain_s_positive(self):
        """所有赛道最大 DRS 收益为正."""
        for tid in DRS_ZONES_2026:
            assert max_drs_gain_s(tid) > 0


# --------------------------------------------------------------------------- #
# DRS 可用性规则 (FIA 2026)
# --------------------------------------------------------------------------- #
class TestDRSAvailability:
    def test_qualifying_always_available(self):
        """排位赛: DRS 全程可用 (无需前车)."""
        avail, reason = drs_available(
            lap=1, gap_to_ahead_s=None, session_type="qualifying"
        )
        assert avail
        assert "qualifying" in reason

    def test_qualifying_available_no_gap(self):
        """排位赛: 即使无前车 (gap=None) 也可用."""
        avail, _ = drs_available(
            lap=5, gap_to_ahead_s=None, session_type="qualifying"
        )
        assert avail

    def test_race_lap1_disabled(self):
        """正赛第 1 圈 DRS 禁用."""
        avail, reason = drs_available(
            lap=1, gap_to_ahead_s=0.5, session_type="race"
        )
        assert not avail
        assert "lap 1" in reason

    def test_race_lap2_enabled_with_close_gap(self):
        """正赛第 2 圈 + 前车 1s 内 → DRS 可用."""
        avail, _ = drs_available(
            lap=2, gap_to_ahead_s=0.8, session_type="race"
        )
        assert avail

    def test_race_no_car_ahead_disabled(self):
        """正赛无前车 → DRS 不可用."""
        avail, reason = drs_available(
            lap=5, gap_to_ahead_s=None, session_type="race"
        )
        assert not avail
        assert "No car" in reason

    def test_race_gap_too_large_disabled(self):
        """正赛前车 > 1s → DRS 不可用."""
        avail, _ = drs_available(
            lap=5, gap_to_ahead_s=2.0, session_type="race"
        )
        assert not avail

    def test_wet_disabled(self):
        """湿地 (wetness > 0.30) DRS 禁用."""
        avail, reason = drs_available(
            lap=5, gap_to_ahead_s=0.5, session_type="race", wetness=0.5
        )
        assert not avail
        assert "wet" in reason

    def test_wet_threshold_boundary(self):
        """wetness = 0.30 边界 (仍可用)."""
        avail, _ = drs_available(
            lap=5, gap_to_ahead_s=0.5, session_type="race", wetness=0.30
        )
        assert avail  # 0.30 不超过阈值

    def test_sc_disabled_2_laps(self):
        """SC 重启后 2 圈 DRS 禁用."""
        # SC 在第 10 圈结束: 第 11, 12 圈禁用, 第 13 圈可用
        avail_11, _ = drs_available(
            lap=11, gap_to_ahead_s=0.5, session_type="race", sc_just_ended_lap=10
        )
        avail_12, _ = drs_available(
            lap=12, gap_to_ahead_s=0.5, session_type="race", sc_just_ended_lap=10
        )
        avail_13, _ = drs_available(
            lap=13, gap_to_ahead_s=0.5, session_type="race", sc_just_ended_lap=10
        )
        assert not avail_11
        assert not avail_12
        assert avail_13


# --------------------------------------------------------------------------- #
# DRS 圈速收益
# --------------------------------------------------------------------------- #
class TestDRSGain:
    def test_zone_gain_proportional_to_length(self):
        """DRS 区收益与长度成正比."""
        short = drs_zone_gain_s(300.0)
        long_ = drs_zone_gain_s(1000.0)
        assert long_ > short
        assert abs(long_ - 3.33 * short) < 0.01  # 1000/300 ratio

    def test_zone_gain_zero_for_short(self):
        """短直道 (< 200m) DRS 收益为 0."""
        assert drs_zone_gain_s(150.0) == 0.0
        assert drs_zone_gain_s(199.0) == 0.0

    def test_zone_gain_positive_for_valid(self):
        """有效 DRS 区收益为正."""
        assert drs_zone_gain_s(500.0) > 0
        assert drs_zone_gain_s(800.0) > 0

    def test_lap_gain_qualifying_more_than_race(self):
        """排位赛 DRS 收益 >= 正赛 (排位无 gap 限制)."""
        q_gain = drs_lap_gain_s(
            "monza", lap=1, gap_to_ahead_s=None, session_type="qualifying"
        )
        r_gain = drs_lap_gain_s(
            "monza", lap=5, gap_to_ahead_s=0.5, session_type="race"
        )
        assert q_gain > 0
        assert r_gain > 0
        # 同赛道同区数, 收益应相等
        assert abs(q_gain - r_gain) < 0.001

    def test_lap_gain_zero_when_unavailable(self):
        """DRS 不可用时收益为 0."""
        # 第 1 圈正赛
        assert drs_lap_gain_s("monza", lap=1, session_type="race") == 0.0
        # 无前车
        assert drs_lap_gain_s("monza", lap=5, gap_to_ahead_s=None, session_type="race") == 0.0
        # 湿地
        assert drs_lap_gain_s("monza", lap=5, wetness=0.5, session_type="race") == 0.0

    def test_monza_more_gain_than_monaco(self):
        """Monza (长直道) DRS 收益 > Monaco (短直道)."""
        monza = drs_lap_gain_s("monza", lap=5, gap_to_ahead_s=0.5, session_type="race")
        monaco = drs_lap_gain_s("monaco", lap=5, gap_to_ahead_s=0.5, session_type="race")
        assert monza > monaco

    def test_baku_high_gain_long_straight(self):
        """Baku (超长主直道) DRS 收益高."""
        baku = drs_lap_gain_s("baku", lap=5, gap_to_ahead_s=0.5, session_type="race")
        # Baku 总 DRS 长度 2400m, 收益应 > 0.5s
        assert baku > 0.5

    def test_gain_realistic_magnitude(self):
        """DRS 收益量级合理 (EA F1 2026: 0.3-1.2s/圈)."""
        for tid in ["monza", "baku", "melbourne", "singapore"]:
            gain = drs_lap_gain_s(
                tid, lap=5, gap_to_ahead_s=0.5, session_type="race"
            )
            assert 0.2 < gain < 1.5, f"{tid}: gain={gain:.3f}s"

    def test_n_active_zones_limit(self):
        """n_active_zones 限制激活区数."""
        full = drs_lap_gain_s(
            "melbourne", lap=5, gap_to_ahead_s=0.5, session_type="race"
        )
        limited = drs_lap_gain_s(
            "melbourne", lap=5, gap_to_ahead_s=0.5,
            session_type="race", n_active_zones=1
        )
        assert limited < full
        # 1 区 vs 4 区: 比例 ~1/4
        assert abs(limited / full - 0.25) < 0.01


# --------------------------------------------------------------------------- #
# 性能
# --------------------------------------------------------------------------- #
class TestPerformance:
    def test_drs_lap_gain_under_10_us(self):
        """DRS 圈速收益计算 < 10 us."""
        n = 50000
        t = timeit.timeit(
            lambda: drs_lap_gain_s(
                "monza", lap=5, gap_to_ahead_s=0.5, session_type="race"
            ),
            number=n,
        )
        per_call_us = t / n * 1e6
        assert per_call_us < 10.0, f"Too slow: {per_call_us:.2f} us/call"

    def test_drs_available_under_5_us(self):
        """DRS 可用性判断 < 5 us."""
        n = 50000
        t = timeit.timeit(
            lambda: drs_available(5, 0.5, "race", 0.0, 0),
            number=n,
        )
        per_call_us = t / n * 1e6
        assert per_call_us < 5.0, f"Too slow: {per_call_us:.2f} us/call"


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
class TestConvenience:
    def test_all_drs_tracks_returns_24(self):
        assert len(all_drs_tracks()) == 24

    def test_drs_speed_delta_realistic(self):
        """DRS 速度差 10-15 km/h (EA F1 2026)."""
        d = drs_speed_delta_kmh()
        assert 10.0 <= d <= 15.0

    def test_max_drs_gain_monza_high(self):
        """Monza 最大 DRS 收益高 (长直道)."""
        gain = max_drs_gain_s("monza")
        assert gain > 0.5
