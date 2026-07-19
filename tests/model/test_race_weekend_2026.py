"""Iter-63 测试: EA F1 2026 完整比赛周末仿真器."""

from __future__ import annotations

import timeit

from f1opt.data.ea_f1_2026_benchmark import EA_F1_2026_LAP_TIME_BENCHMARK
from f1opt.model.lap_simulator_2026 import (
    StintPlan2026,
)
from f1opt.model.race_weekend_2026 import (
    RaceWeekend2026,
    SessionResult2026,
    WeekendReport2026,
)
from f1opt.model.safety_car import SafetyCarModel, SafetyCarPeriod


# --------------------------------------------------------------------------- #
# 周末编排基础
# --------------------------------------------------------------------------- #
class TestWeekendOrchestration:
    """完整周末编排必须包含 5 环节."""

    def test_run_returns_weekend_report(self):
        w = RaceWeekend2026(track_id="monza", race_total_laps=53)
        r = w.run()
        assert isinstance(r, WeekendReport2026)
        assert r.track_id == "monza"

    def test_all_five_sessions_present(self):
        r = RaceWeekend2026(track_id="monza").run()
        assert set(r.sessions.keys()) == {"fp1", "fp2", "fp3", "qualifying", "race"}

    def test_each_session_is_session_result(self):
        r = RaceWeekend2026(track_id="monza").run()
        for name, s in r.sessions.items():
            assert isinstance(s, SessionResult2026), f"{name} 不是 SessionResult2026"
            assert s.name == name

    def test_fp1_fp2_fp3_lap_counts(self):
        """FP1=20, FP2=15, FP3=12 圈 (EA F1 2026 practice stint lengths)."""
        r = RaceWeekend2026(track_id="monza").run()
        assert r.sessions["fp1"].laps == 20
        assert r.sessions["fp2"].laps == 15
        assert r.sessions["fp3"].laps == 12

    def test_fp_compounds(self):
        """FP1=medium, FP2=soft, FP3=medium."""
        r = RaceWeekend2026(track_id="monza").run()
        assert r.sessions["fp1"].compound == "medium"
        assert r.sessions["fp2"].compound == "soft"
        assert r.sessions["fp3"].compound == "medium"

    def test_qualifying_three_flying_laps(self):
        r = RaceWeekend2026(track_id="monza").run()
        assert r.sessions["qualifying"].laps == 3
        assert len(r.qualifying_laps) == 3

    def test_race_uses_plan_length(self):
        plan = StintPlan2026(("medium", "soft"), (26, 26))
        w = RaceWeekend2026(track_id="monza", race_plan=plan, race_total_laps=52)
        r = w.run()
        assert r.sessions["race"].laps == 52
        assert len(r.race_results) == 52


# --------------------------------------------------------------------------- #
# EA F1 2026 物理量级 (FP / Qualifying / Race 排序)
# --------------------------------------------------------------------------- #
class TestEAF12026PhysicsMagnitude:
    """各环节圈速必须符合 EA F1 2026 物理."""

    def test_qualifying_faster_than_fp2(self):
        """排位 (PARTY + QUALIFYING PU + 低油量 + 新软胎) 必须 < FP2 (RICH + ATTACK)."""
        r = RaceWeekend2026(track_id="monza").run()
        assert r.qualifying_best_s < r.sessions["fp2"].best_lap_s

    def test_fp2_faster_than_fp1(self):
        """FP2 (低油量 + ATTACK + RICH) < FP1 (高油量 + BALANCED)."""
        r = RaceWeekend2026(track_id="monza").run()
        assert r.sessions["fp2"].best_lap_s < r.sessions["fp1"].best_lap_s

    def test_qualifying_best_close_to_benchmark(self):
        """排位最佳圈应接近 EA F1 2026 benchmark (排位=满 SoC + 9MJ + PARTY + 30kg).

        偏差来自 30kg 油量 (相对 reference 50kg 减 20kg ≈ -0.7s) +
        QUALIFYING PU (9MJ vs 6MJ ref, 净 +0.27s 更快) + PARTY (-0.4s).
        综合: 排位圈速应明显快于 benchmark (约 -1.4s).
        """
        r = RaceWeekend2026(track_id="monza").run()
        bench = EA_F1_2026_LAP_TIME_BENCHMARK["monza"]
        # 排位应快于 benchmark (PU + fuel + tire 全优化)
        assert r.qualifying_best_s < bench
        # 偏差在合理范围 (1-3s 改进)
        delta = bench - r.qualifying_best_s
        assert 0.5 < delta < 3.5, f"排位改进 {delta:.3f}s 超出预期"

    def test_race_avg_slower_than_qualifying(self):
        """正赛平均圈 (高油量 + BALANCED + 跟车 + 进站) > 排位最佳."""
        r = RaceWeekend2026(track_id="monza").run()
        assert r.sessions["race"].avg_lap_s > r.qualifying_best_s

    def test_race_lap_count_matches_total_laps(self):
        w = RaceWeekend2026(
            track_id="monaco",
            race_plan=StintPlan2026(("medium", "soft"), (39, 39)),
            race_total_laps=78,
        )
        r = w.run()
        assert len(r.race_results) == 78

    def test_pit_stops_recorded(self):
        """2-stop 策略应有 2 条进站记录."""
        w = RaceWeekend2026(
            track_id="monza",
            race_plan=StintPlan2026(("medium", "soft", "medium"), (18, 18, 17)),
            race_total_laps=53,
        )
        r = w.run()
        assert len(r.race_pit_records) == 2

    def test_no_pit_stops_for_one_stop_plan(self):
        """1-stop 策略应有 1 条进站记录."""
        w = RaceWeekend2026(
            track_id="monza",
            race_plan=StintPlan2026(("medium", "soft"), (26, 27)),
            race_total_laps=53,
        )
        r = w.run()
        assert len(r.race_pit_records) == 1


# --------------------------------------------------------------------------- #
# 多赛道兼容性
# --------------------------------------------------------------------------- #
class TestMultipleTracks:
    """周末仿真必须在多赛道上稳定运行."""

    def test_sprint_track(self):
        """Spielberg (短圈速赛道) 周末完整运行."""
        w = RaceWeekend2026(
            track_id="spielberg",
            race_plan=StintPlan2026(("medium", "soft"), (36, 35)),
            race_total_laps=71,
        )
        r = w.run()
        assert r.sessions["race"].laps == 71
        assert r.qualifying_best_s < 70.0  # Spielberg benchmark 64.5s

    def test_long_track(self):
        """Spa (长圈速赛道) 周末完整运行."""
        w = RaceWeekend2026(
            track_id="spa",
            race_plan=StintPlan2026(("medium", "soft"), (22, 22)),
            race_total_laps=44,
        )
        r = w.run()
        assert r.qualifying_best_s < 105.0  # Spa benchmark 104.5s
        assert r.sessions["race"].laps == 44

    def test_street_circuit(self):
        """Monaco (街道赛) 周末完整运行."""
        w = RaceWeekend2026(
            track_id="monaco",
            race_plan=StintPlan2026(("soft", "medium"), (39, 39)),
            race_total_laps=78,
        )
        r = w.run()
        assert r.qualifying_best_s < 75.0  # Monaco benchmark 73.0s


# --------------------------------------------------------------------------- #
# SC/VSC 整合
# --------------------------------------------------------------------------- #
class TestSafetyCarWeekend:
    """SC 注入周末正赛."""

    def test_sc_in_race(self):
        """SC 期间正赛圈速变慢."""
        sc = SafetyCarModel(periods=[
            SafetyCarPeriod(start_lap=10, end_lap=13, kind="sc"),
        ])
        w_normal = RaceWeekend2026(
            track_id="monza",
            race_plan=StintPlan2026(("medium", "soft"), (26, 27)),
            race_total_laps=53,
        )
        w_sc = RaceWeekend2026(
            track_id="monza",
            race_plan=StintPlan2026(("medium", "soft"), (26, 27)),
            race_total_laps=53,
            race_safety_car=sc,
        )
        r_normal = w_normal.run()
        r_sc = w_sc.run()
        # SC 期间 (lap 10-13) 圈速明显变慢
        sc_laps_sc = [r.lap_time_s for r in r_sc.race_results[9:13]]
        sc_laps_normal = [r.lap_time_s for r in r_normal.race_results[9:13]]
        avg_sc = sum(sc_laps_sc) / len(sc_laps_sc)
        avg_normal = sum(sc_laps_normal) / len(sc_laps_normal)
        assert avg_sc > avg_normal + 5.0  # SC ×1.30 应明显变慢

    def test_sc_free_pit_discount_in_weekend(self):
        """SC 期间进站应享受折扣."""
        sc = SafetyCarModel(periods=[
            SafetyCarPeriod(start_lap=25, end_lap=28, kind="sc"),
        ])
        # 进站圈 = 26 (stint 1 末圈), 落在 SC 内
        w = RaceWeekend2026(
            track_id="monza",
            race_plan=StintPlan2026(("medium", "soft"), (26, 27)),
            race_total_laps=53,
            race_safety_car=sc,
        )
        r = w.run()
        # 第一条进站记录应享受 SC 折扣 (< 1.0)
        assert len(r.race_pit_records) >= 1
        first_pit = r.race_pit_records[0]
        assert first_pit["sc_discount"] < 1.0


# --------------------------------------------------------------------------- #
# 湿地周末
# --------------------------------------------------------------------------- #
class TestWetWeekend:
    """湿地周末仿真."""

    def test_wet_weekend_runs(self):
        w = RaceWeekend2026(track_id="monza", wet=True)
        r = w.run()
        assert r is not None

    def test_wet_qualifying_slower(self):
        """湿地排位应慢于干地排位."""
        r_dry = RaceWeekend2026(track_id="monza").run()
        r_wet = RaceWeekend2026(track_id="monza", wet=True).run()
        assert r_wet.qualifying_best_s > r_dry.qualifying_best_s

    def test_wet_race_slower(self):
        """湿地正赛总时长应长于干地."""
        r_dry = RaceWeekend2026(track_id="monza").run()
        r_wet = RaceWeekend2026(track_id="monza", wet=True).run()
        assert r_wet.race_total_s > r_dry.race_total_s


# --------------------------------------------------------------------------- #
# 车队/车手偏移
# --------------------------------------------------------------------------- #
class TestTeamAndDriverOffset:
    """车队性能偏移应影响周末成绩."""

    def test_top_team_faster(self):
        """顶队 (负偏移) 排位应快于后段车队 (正偏移)."""
        r_top = RaceWeekend2026(
            track_id="monza", car_performance_offset_s=-0.5
        ).run()
        r_back = RaceWeekend2026(
            track_id="monza", car_performance_offset_s=0.5
        ).run()
        assert r_top.qualifying_best_s < r_back.qualifying_best_s

    def test_driver_offset_affects_all_sessions(self):
        """车手偏移应影响所有环节."""
        r_fast = RaceWeekend2026(
            track_id="monza", driver_skill_offset_s=-0.3
        ).run()
        r_slow = RaceWeekend2026(
            track_id="monza", driver_skill_offset_s=0.3
        ).run()
        assert r_fast.qualifying_best_s < r_slow.qualifying_best_s
        assert r_fast.sessions["fp1"].best_lap_s < r_slow.sessions["fp1"].best_lap_s
        assert r_fast.race_total_s < r_slow.race_total_s


# --------------------------------------------------------------------------- #
# 报告 API
# --------------------------------------------------------------------------- #
class TestReportAPI:
    """WeekendReport2026 API 测试."""

    def test_summary_structure(self):
        r = RaceWeekend2026(track_id="monza").run()
        s = r.summary()
        assert s["track_id"] == "monza"
        assert "qualifying_best_s" in s
        assert "race_total_s" in s
        assert "race_laps" in s
        assert "race_pit_stops" in s
        assert "sessions" in s
        for name in ("fp1", "fp2", "fp3", "qualifying", "race"):
            assert name in s["sessions"]
            assert "laps" in s["sessions"][name]
            assert "best_lap_s" in s["sessions"][name]
            assert "compound" in s["sessions"][name]

    def test_qualifying_best_s_property(self):
        r = RaceWeekend2026(track_id="monza").run()
        assert r.qualifying_best_s == r.sessions["qualifying"].best_lap_s

    def test_race_total_s_property(self):
        r = RaceWeekend2026(track_id="monza").run()
        assert r.race_total_s == r.sessions["race"].total_time_s


# --------------------------------------------------------------------------- #
# 性能 (EA F1 2026 专业车队标准: < 5ms/weekend)
# --------------------------------------------------------------------------- #
class TestPerformance:
    """完整周末仿真性能 < 5ms (专业车队实时策略评估标准)."""

    def test_full_weekend_under_5ms(self):
        w = RaceWeekend2026(track_id="monza")
        # warmup
        w.run()
        elapsed = timeit.timeit(lambda: w.run(), number=10)
        per_call_ms = (elapsed / 10) * 1000.0
        assert per_call_ms < 5.0, f"周末仿真 {per_call_ms:.3f}ms > 5ms 阈值"

    def test_full_weekend_under_2ms_warm(self):
        """热缓存下应 < 2ms (lru_cache 命中)."""
        w = RaceWeekend2026(track_id="monza")
        # 多次 warmup 充分热缓存
        for _ in range(5):
            w.run()
        elapsed = timeit.timeit(lambda: w.run(), number=20)
        per_call_ms = (elapsed / 20) * 1000.0
        assert per_call_ms < 2.0, f"热缓存周末仿真 {per_call_ms:.3f}ms > 2ms 阈值"
