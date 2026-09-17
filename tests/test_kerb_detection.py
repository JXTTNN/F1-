"""路肩检测（按弯归因）测试 —— task-82。

相对法（实测标定）：逐弯悬挂加速度 vs 全圈中位数，显著偏高的弯 = 压路肩弯。
真实数据验证（Monza 87k 帧）：检出 T2/T5/T10（三个减速弯）+ T7，与赛历吻合。
"""

from __future__ import annotations

import pytest

from setup_tuner.engine.engine import DIAG_DIMS, _apply_kerb_rules
from setup_tuner.telemetry.lap_aggregator import LapAggregator


def _lap(lap_num: int, dist: float, last_ms: int = 0) -> dict:
    return {
        "m_currentLapNum": lap_num,
        "m_lapDistance": dist,
        "m_lastLapTimeInMS": last_ms,
        "m_sector": 1,
        "m_currentLapInvalid": 0,
    }


def _motion(left_accel: float, right_accel: float) -> dict:
    """构造已解析的 MotionEx 帧（车轮序 0=RL 1=RR 2=FL 3=FR）。"""
    return {
        "m_suspensionAcceleration": [
            left_accel, right_accel, left_accel, right_accel,
        ],
        "m_suspensionPosition": [0.05, 0.05, 0.05, 0.05],
        "m_frontAeroHeight": 0.03,
        "m_rearAeroHeight": 0.05,
    }


def _locator_three_corners(dist: float) -> int | None:
    """0-400m=T1（路肩弯）、400-800m=T2（平整）、800-1000m=T3（平整）。"""
    if dist < 0:
        return None
    if dist < 400:
        return 1
    if dist < 800:
        return 2
    return 3


def _feed_frame(agg: LapAggregator, lap: int, dist: float, accel: float) -> None:
    """喂一帧：LapData（位置）+ Telemetry（计帧）+ MotionEx（粗糙度）。

    注意：聚合器的圈帧计数 ``_n`` 由 on_telemetry 递增——只喂 MotionEx
    时整圈快照永不固化（真实链路中两者总是成对到达）。
    """
    agg.on_lap_data(_lap(lap, dist=dist))
    agg.on_telemetry({"m_speed": 200.0, "m_throttle": 0.6, "m_brake": 0.05,
                      "m_steer": 0.2})
    agg.on_motion_ex(_motion(left_accel=accel, right_accel=accel))


def _run_lap(agg: LapAggregator, lap: int, kerb_frames: int, calm_frames: int) -> None:
    """喂一圈：T1 内 kerb_frames 帧高粗糙度，T2/T3 内 calm_frames 帧平整。"""
    _feed_frame(agg, lap, dist=10.0, accel=8000.0 if kerb_frames else 150.0)
    for _ in range(max(0, kerb_frames - 1)):
        _feed_frame(agg, lap, dist=100.0, accel=8000.0)
    for i in range(calm_frames):
        # T2（平整段）与 T3 各半，幅度低且稳定
        dist = 500.0 if i % 2 == 0 else 900.0
        _feed_frame(agg, lap, dist=dist, accel=200.0)
    _feed_frame(agg, lap, dist=990.0, accel=150.0)


def _finish_lap(agg: LapAggregator, next_lap: int):
    """喂下一圈的第一个 LapData（圈号变化）→ 固化上一圈，返回快照。"""
    _feed_frame(agg, next_lap, dist=10.0, accel=150.0)
    return agg.take_completed_lap()


class TestAggregatorKerbAttribution:
    """按弯路肩归因：相对法检出显著偏粗糙的弯。"""

    def test_kerb_corner_detected_and_ranked(self) -> None:
        agg = LapAggregator()
        agg.set_track_context(1000.0, _locator_three_corners)
        _run_lap(agg, lap=1, kerb_frames=60, calm_frames=120)
        snap = _finish_lap(agg, next_lap=2)

        assert snap is not None
        kerb = snap.get("kerb_corners")
        assert kerb, "应检出压路肩的弯"
        assert kerb[0]["corner"] == 1, "T1（路肩弯）应排第一"
        assert kerb[0]["ratio"] >= 1.6
        # T2/T3 平整段不应上榜
        assert all(k["corner"] != 2 for k in kerb)
        assert all(k["corner"] != 3 for k in kerb)

    def test_smooth_lap_no_kerb(self) -> None:
        """全程平整（无高粗糙度帧）→ 不应有路肩弯。"""
        agg = LapAggregator()
        agg.set_track_context(1000.0, _locator_three_corners)
        _run_lap(agg, lap=1, kerb_frames=0, calm_frames=150)
        snap = _finish_lap(agg, next_lap=2)
        assert snap is not None
        assert not snap.get("kerb_corners"), "全平整圈不应检出路肩弯"

    def test_track_context_persists_across_laps(self) -> None:
        """回归：赛道上下文**必须跨圈保持**——曾被误放逐圈重置表里，
        第二圈开始定位器变 None，路肩统计整体失效。"""
        agg = LapAggregator()
        agg.set_track_context(1000.0, _locator_three_corners)
        _run_lap(agg, lap=1, kerb_frames=60, calm_frames=120)
        first = _finish_lap(agg, next_lap=2)
        assert first and first.get("kerb_corners"), "第 1 圈应有路肩检出"

        # 不再重新注入上下文，直接喂第二圈
        _run_lap(agg, lap=2, kerb_frames=60, calm_frames=120)
        second = _finish_lap(agg, next_lap=3)
        assert second is not None
        assert second.get("kerb_corners"), (
            "第 2 圈也必须有路肩检出（上下文跨圈保持）"
        )

    def test_no_locator_means_no_kerb_stats(self) -> None:
        """未注入上下文（如纯重放旧数据）→ 不产生路肩字段，也不崩。"""
        agg = LapAggregator()
        _feed_frame(agg, 1, dist=10.0, accel=9000.0)
        for _ in range(50):
            agg.on_motion_ex(_motion(left_accel=9000.0, right_accel=9000.0))
        _feed_frame(agg, 2, dist=990.0, accel=9000.0)
        snap = agg.take_completed_lap()
        assert snap is not None
        assert "kerb_corners" not in snap


class TestEngineKerbRule:
    """引擎规则17：压路肩 → 抬离地 / 悬挂余量（路肩不得不压，针对性优化）。"""

    def test_severe_kerb_raises_ride_height(self) -> None:
        dx = dict.fromkeys(DIAG_DIMS, 0.0)
        _apply_kerb_rules(
            {"kerb_corners": [
                {"corner": 5, "ratio": 10.4, "side": "right"},
                {"corner": 2, "ratio": 1.8, "side": "left"},
            ]},
            dx,
        )
        assert dx["ride_height_req"] == pytest.approx(0.30)
        assert dx["hi_speed_stab_req"] == pytest.approx(0.10)

    def test_mild_kerb_raises_ride_height_only(self) -> None:
        dx = dict.fromkeys(DIAG_DIMS, 0.0)
        _apply_kerb_rules(
            {"kerb_corners": [{"corner": 9, "ratio": 1.7, "side": "left"}]},
            dx,
        )
        assert dx["ride_height_req"] == pytest.approx(0.15)
        assert dx["hi_speed_stab_req"] == pytest.approx(0.0)

    def test_no_kerb_data_is_noop(self) -> None:
        dx = dict.fromkeys(DIAG_DIMS, 0.0)
        _apply_kerb_rules({}, dx)
        _apply_kerb_rules({"kerb_corners": []}, dx)
        _apply_kerb_rules({"kerb_corners": [{"corner": 1}]}, dx)  # 缺 ratio
        assert all(v == 0.0 for v in dx.values())

    def test_negative_ratio_is_ignored(self) -> None:
        """脏数据（负/零 ratio）不参与判定。"""
        dx = dict.fromkeys(DIAG_DIMS, 0.0)
        _apply_kerb_rules({"kerb_corners": [{"corner": 1, "ratio": -3.0}]}, dx)
        _apply_kerb_rules({"kerb_corners": [{"corner": 1, "ratio": 0.0}]}, dx)
        assert all(v == 0.0 for v in dx.values())
