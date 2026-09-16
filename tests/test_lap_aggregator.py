"""整圈遥测聚合器 + 遥测规则 6/7/10/15 生效回归测试。

背景：``/suggest`` 此前只把「每类包最新一帧」喂给规则引擎，
``engine._derive_telemetry_dx`` 的规则 6（on_straight）/7（max_steer）/
10（avg_steer）/15（max_speed）因为拿不到整圈统计而**永不触发**。
``telemetry/lap_aggregator.py`` 补齐这些量，本文件锁定其正确性与规则联动。
"""

from __future__ import annotations

import pytest

from setup_tuner.engine.engine import _derive_telemetry_dx
from setup_tuner.report.builder import extract_telemetry_summary
from setup_tuner.telemetry.lap_aggregator import LapAggregator


def _frame(speed: float, throttle: float, brake: float = 0.0, steer: float = 0.0) -> dict:
    return {
        "m_speed": speed,
        "m_throttle": throttle,
        "m_brake": brake,
        "m_steer": steer,
        "m_tyresSurfaceTemperature": [100, 102, 96, 98],
        "m_tyresPressure": [23.0, 23.2, 21.5, 21.7],
        "m_brakesTemperature": [420, 430, 380, 390],
    }


def _feed(agg: LapAggregator, frames: list[dict]) -> None:
    for f in frames:
        agg.on_telemetry(f)


# ===========================================================================
# 1. 聚合器本身
# ===========================================================================
class TestLapAggregator:
    """逐帧累积与整圈固化。"""

    def test_empty_snapshot(self) -> None:
        """没有任何帧时不产出快照。"""
        agg = LapAggregator()
        assert agg.best_snapshot() is None
        assert agg.last_completed_lap() is None
        assert agg.frames_total == 0

    def test_accumulates_scalars_and_wheels(self) -> None:
        """速度极值/均值、转向绝对值均值与极值、胎温胎压均值。"""
        agg = LapAggregator()
        _feed(agg, [_frame(100, 0.5, 0.2, 0.1), _frame(300, 0.9, 0.4, -0.4)])
        snap = agg.snapshot()
        assert snap["lap_frames"] == 2
        assert snap["max_speed"] == 300.0
        assert snap["avg_speed"] == 200.0
        assert snap["avg_steer"] == 0.25  # (0.1 + 0.4) / 2
        assert snap["max_steer"] == 0.4
        assert snap["avg_throttle"] == 0.7
        assert snap["max_brake"] == 0.4
        assert snap["m_tyresSurfaceTemperature"] == [100.0, 102.0, 96.0, 98.0]

    def test_on_straight_flag(self) -> None:
        """直道判定：转向回正 + 大油门。"""
        agg = LapAggregator()
        agg.on_telemetry(_frame(300, 0.98, 0.0, 0.01))
        assert agg.snapshot()["on_straight"] is True
        agg.on_telemetry(_frame(120, 0.98, 0.0, 0.30))
        assert agg.snapshot()["on_straight"] is False

    def test_lap_rollover_freezes_previous_lap(self) -> None:
        """圈号变化时固化上一圈并重置累积。"""
        agg = LapAggregator()
        agg.on_lap_data({"m_currentLapNum": 1, "m_sector": 0})
        _feed(agg, [_frame(200, 0.8), _frame(250, 0.9)])
        agg.on_lap_data({"m_currentLapNum": 2, "m_sector": 1})
        completed = agg.last_completed_lap()
        assert completed is not None
        assert completed["lap_frames"] == 2
        assert completed["max_speed"] == 250.0
        # 新圈重新累积
        assert agg.snapshot()["lap_frames"] == 0
        assert agg.snapshot()["lap_number"] == 2

    def test_sector_is_one_based(self) -> None:
        """扇区统一为 1 基（UDP 0/1/2 → 1/2/3）。"""
        agg = LapAggregator()
        agg.on_lap_data({"m_currentLapNum": 1, "m_sector": 0})
        assert agg.snapshot()["sector"] == 1
        agg.on_lap_data({"m_currentLapNum": 1, "m_sector": 2})
        assert agg.snapshot()["sector"] == 3

    def test_reset(self) -> None:
        """reset 后回到初始状态。"""
        agg = LapAggregator()
        _feed(agg, [_frame(200, 0.8)])
        agg.reset()
        assert agg.best_snapshot() is None
        assert agg.frames_total == 0


# ===========================================================================
# 2. 规则 6/7/10/15 是否真的被点亮
# ===========================================================================
class TestTelemetryRulesNowFire:
    """整圈统计接入后，此前永不触发的遥测规则必须生效。"""

    def test_rules_never_fire_without_lap_stats(self) -> None:
        """对照：只喂单帧（旧行为）时，规则 6/7/10/15 均不触发。"""
        all_latest = {6: _frame(180, 0.95, 0.0, 0.4)}
        summary = extract_telemetry_summary(all_latest)
        dx = _derive_telemetry_dx(summary)
        assert dx["turnin_req"] == 0.0        # 规则 7 需要 max_steer
        assert dx["hi_speed_stab_req"] == 0.0  # 规则 6/10/15 需要 on_straight/avg_steer/max_speed

    def test_rule6_and_rule15_fire_on_slow_straight_lap(self) -> None:
        """全油门直道但速度偏低 → 规则 6 与 15 各扣 0.3。"""
        agg = LapAggregator()
        _feed(agg, [_frame(180, 0.97, 0.0, 0.0) for _ in range(10)])
        summary = extract_telemetry_summary({6: _frame(180, 0.97)}, agg.best_snapshot())
        assert summary["on_straight"] is True
        assert summary["max_speed"] == 180.0
        dx = _derive_telemetry_dx(summary)
        assert dx["hi_speed_stab_req"] == -0.6  # 规则6 -0.3 + 规则15 -0.3
        assert dx["turnin_req"] == 0.0

    def test_rule7_and_rule10_fire_on_corner_heavy_lap(self) -> None:
        """持续大转向的圈 → 规则 7（turnin）与规则 10（弯中不稳）触发。"""
        agg = LapAggregator()
        _feed(agg, [_frame(300, 0.4, 0.1, 0.4) for _ in range(10)])
        summary = extract_telemetry_summary({6: _frame(300, 0.4)}, agg.best_snapshot())
        assert summary["avg_steer"] == 0.4
        assert summary["max_steer"] == 0.4
        assert summary["max_speed"] == 300.0
        dx = _derive_telemetry_dx(summary)
        assert dx["turnin_req"] == 0.3            # 规则 7
        assert dx["hi_speed_stab_req"] == 0.3     # 规则 10（规则15 因 max_speed=300 不触发）

    def test_rule5_fires_with_one_based_sector(self) -> None:
        """扇区 1 基后规则 5（出弯油门低）才能成立。"""
        # 3 号扇区 + 低油门
        dx = _derive_telemetry_dx({"sector": 3, "m_throttle": 0.1})
        assert dx["exit_traction_req"] == 0.3
        # 旧口径（0 基原始值 2）不触发
        assert _derive_telemetry_dx({"sector": 2, "m_throttle": 0.1})["exit_traction_req"] == 0.0


# ===========================================================================
# 扇区转换单点化（防再次分叉）
# ===========================================================================
class TestSectorConversionSingleSource:
    """``m_sector`` 的 0 基→1 基转换必须只有 ``packets.to_sector_1based`` 一处实现。

    历史背景：ws.py / report.builder.py / lap_aggregator.py 曾各自手写 ``+1``，
    其中 lap_aggregator 还额外做了 ``min(3, ...)`` 截断，三处口径需人工对齐。
    ``to_sector_1based`` 的文档字符串明确写了「统一在此转换，避免每个消费方
    各自 +1 而再次错位」——本测试把这句话变成可执行约束。
    """

    @pytest.mark.parametrize("raw,expected", [
        (0, 1), (1, 2), (2, 3),      # 规范值域 0/1/2 → 1/2/3
        (3, 3),                       # 越界钳到上限
        (-1, 1), (-99, 1),            # 越界钳到下限
        (None, None),                 # 缺失
    ])
    def test_aggregator_matches_authoritative_conversion(self, raw, expected) -> None:
        """聚合器的扇区结果必须与 ``to_sector_1based`` 逐值一致。"""
        from setup_tuner.telemetry.packets import to_sector_1based

        assert to_sector_1based(raw) == expected

        agg = LapAggregator()
        agg.on_lap_data({"m_currentLapNum": 1, "m_sector": raw})
        assert agg.snapshot()["sector"] == expected

    @pytest.mark.parametrize("raw", [True, False])
    def test_bool_is_treated_as_missing(self, raw: bool) -> None:
        """bool 是 int 的子类，但不得被当作扇区数值（避免 S2/S1 误判）。"""
        from setup_tuner.telemetry.packets import to_sector_1based

        assert to_sector_1based(raw) is None
        agg = LapAggregator()
        agg.on_lap_data({"m_currentLapNum": 1, "m_sector": raw})
        assert agg.snapshot()["sector"] is None

    def test_no_duplicate_conversion_in_ws_layer(self) -> None:
        """ws.py 不得再定义自己的扇区转换包装函数（应直接用权威实现）。"""
        import inspect

        from setup_tuner.api import ws

        assert not hasattr(ws, "_sector_1based"), (
            "ws.py 重新引入了 _sector_1based 包装，请直接用 to_sector_1based"
        )
        source = inspect.getsource(ws)
        assert "int(raw) + 1" not in source, "ws.py 出现手工 +1 扇区转换"
