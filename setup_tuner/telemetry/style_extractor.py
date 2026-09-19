"""车手风格提取器 —— 把逐帧遥测累积成 12 维风格向量（task-62 M2）。

与 :class:`telemetry.lap_aggregator.LapAggregator` 同模式：逐帧流式累积、
圈号变化时固化整圈向量，不落任何原始遥测。成本同量级 ≈2 µs/帧。

11 维由单圈遥测直接计算；第 12 维（圈速一致性）是跨圈量，
由 ``Store``/更新器依据 ``lap_record`` 历史圈时方差计算后追加。

输出向量各维均归一化到 [0, 1]（含义见 ``STYLE_DIMS``）。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

STYLE_DIMS: list[str] = [
    "steer_aggression",      # 1 攻弯强度（平均|转向|）
    "steer_smoothness",      # 2 转向平滑度（转向变化率的反向）
    "throttle_aggression",   # 3 油门激进度
    "brake_aggression",      # 4 刹车激进度（峰值）
    "trail_braking",         # 5 循迹刹车占比 ★车手差异最大
    "throttle_onset",        # 6 出弯给油时机（低速段全油门占比）
    "tyre_management",       # 7 轮胎管理（胎温离散度的反向）
    "slip_ratio",            # 8 滑移迹象（大油门但加速弱）
    "straight_speed_ratio",  # 9 直道末速比
    "brake_thermal",         # 10 制动热负荷
    "slow_corner_ratio",     # 11 慢弯占比（弯型分布近似）
    "lap_consistency",       # 12 圈速一致性（跨圈，Store 层填充）
]

_STYLE_STEER_RATE_NORM = 0.08   # 每帧转向变化率归一化基准
_TYRE_DISPERSION_NORM = 30.0    # 四轮胎温极差归一化基准（℃）
_SLIP_ACCEL_MIN = 0.3           # 大油门下的最小每帧加速度（km/h/帧）
_TOP_SPEED_NORM = 340.0         # 直道末速归一化基准（km/h）
_TRAIL_THROTTLE_MAX = 0.2
_TRAIL_BRAKE_MIN = 0.3
_SLIP_THROTTLE_MIN = 0.6
_SLOW_CORNER_SPEED_MAX = 120.0


class StyleExtractor:
    """逐帧累积单圈风格统计，圈号变化时固化上一圈向量。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()
        self._lap_number: int | None = None
        self._last_completed: list[float] | None = None

    def _reset(self) -> None:
        self._n = 0
        self._steer_abs_sum = 0.0
        self._steer_rate_sum = 0.0
        self._throttle_sum = 0.0
        self._brake_max = 0.0
        self._trail_frames = 0
        self._onset_full_frames = 0
        self._onset_low_speed_frames = 0
        self._slip_frames = 0
        self._slow_frames = 0
        self._speed_max = 0.0
        self._prev_speed: float | None = None
        self._prev_steer_abs: float | None = None
        self._tyre_hi = -1e9
        self._tyre_lo = 1e9
        self._brake_temp_sum = 0.0

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    # ------------------------------------------------------------------ #
    def on_lap_data(self, lap: dict[str, Any]) -> None:
        """接收 Packet 2：跟踪圈号，圈号变化时固化上一圈向量。"""
        lap_no = lap.get("m_currentLapNum")
        lap_no = int(lap_no) if isinstance(lap_no, (int, float)) else None
        with self._lock:
            if (lap_no is not None and self._lap_number is not None
                    and lap_no != self._lap_number and self._n > 0):
                self._last_completed = self._build_locked()
                self._reset()
            if lap_no is not None:
                self._lap_number = lap_no

    def on_telemetry(self, frame: dict[str, Any]) -> None:
        """接收 Packet 6：累积单帧统计。"""
        speed = self._as_float(frame.get("m_speed"))
        throttle = self._as_float(frame.get("m_throttle"))
        brake = self._as_float(frame.get("m_brake"))
        steer = self._as_float(frame.get("m_steer"))

        with self._lock:
            self._n += 1
            if steer is not None:
                steer_abs = abs(steer)
                self._steer_abs_sum += steer_abs
                if self._prev_steer_abs is not None:
                    self._steer_rate_sum += abs(steer_abs - self._prev_steer_abs)
                self._prev_steer_abs = steer_abs
            if speed is not None:
                if speed > self._speed_max:
                    self._speed_max = speed
                if speed < _SLOW_CORNER_SPEED_MAX:
                    self._slow_frames += 1
                if self._prev_speed is not None:
                    accel = speed - self._prev_speed
                    if (throttle is not None
                            and throttle >= _SLIP_THROTTLE_MIN
                            and accel < _SLIP_ACCEL_MIN):
                        self._slip_frames += 1
            if throttle is not None:
                self._throttle_sum += throttle
            if brake is not None:
                if brake > self._brake_max:
                    self._brake_max = brake
                if (throttle is not None and throttle < _TRAIL_THROTTLE_MAX
                        and brake >= _TRAIL_BRAKE_MIN):
                    self._trail_frames += 1
            if speed is not None and throttle is not None:
                half_top = 0.5 * max(self._speed_max, 1.0)
                if speed < half_top:
                    self._onset_low_speed_frames += 1
                    if throttle >= 0.9:
                        self._onset_full_frames += 1
            temps = frame.get("m_tyresSurfaceTemperature")
            if isinstance(temps, list) and len(temps) >= 4:
                try:
                    values = [float(v) for v in temps[:4]]
                    self._tyre_hi = max(self._tyre_hi, max(values))
                    self._tyre_lo = min(self._tyre_lo, min(values))
                except (TypeError, ValueError):
                    # 非数值胎温：本帧跳过该信号；记录一次避免风格向量静默失真
                    logger.debug("轮胎温度非数值，本帧跳过：%r", temps)
            btemps = frame.get("m_brakesTemperature")
            if isinstance(btemps, list) and len(btemps) >= 4:
                try:
                    self._brake_temp_sum += sum(float(v) for v in btemps[:4]) / 4.0
                except (TypeError, ValueError):
                    # 非数值刹车温度：本帧跳过该信号
                    logger.debug("刹车温度非数值，本帧跳过：%r", btemps)
            self._prev_speed = speed

    # ------------------------------------------------------------------ #
    def _build_locked(self) -> list[float]:
        n = self._n or 1

        def clip01(v: float) -> float:
            return round(max(0.0, min(1.0, v)), 4)

        avg_steer = self._steer_abs_sum / n
        avg_rate = self._steer_rate_sum / n
        avg_throttle = self._throttle_sum / n
        avg_btemp = self._brake_temp_sum / n
        dispersion = max(0.0, self._tyre_hi - self._tyre_lo) \
            if self._tyre_hi > -1e8 else 0.0
        return [
            clip01(avg_steer / 0.5),                                              # 1
            clip01(1.0 - avg_rate / _STYLE_STEER_RATE_NORM),                      # 2
            clip01(avg_throttle),                                                 # 3
            clip01(self._brake_max),                                              # 4
            clip01(self._trail_frames / n),                                       # 5
            clip01(self._onset_full_frames / max(self._onset_low_speed_frames, 1)),  # 6
            clip01(1.0 - dispersion / _TYRE_DISPERSION_NORM),                     # 7
            clip01(self._slip_frames / n),                                        # 8
            clip01(self._speed_max / _TOP_SPEED_NORM),                            # 9
            clip01(avg_btemp / 1000.0),                                           # 10
            clip01(self._slow_frames / n),                                        # 11
        ]

    def snapshot(self) -> list[float] | None:
        """当前这一圈「到目前为止」的风格向量（无帧时返回 None）。"""
        with self._lock:
            if self._n == 0:
                return None
            return self._build_locked()

    def take_completed(self) -> list[float] | None:
        """取出上一圈固化的风格向量（取出后清空）。"""
        with self._lock:
            vector, self._last_completed = self._last_completed, None
            return vector

    def reset(self) -> None:
        """清空全部状态。"""
        with self._lock:
            self._reset()
            self._lap_number = None
            self._last_completed = None
