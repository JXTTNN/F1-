"""整圈遥测聚合器 —— 把逐帧遥测累积成引擎需要的整圈统计量。

背景（2026-09 审计结论）：
    ``/suggest`` 此前只把「每类包最新一帧」交给规则引擎
    （``report/builder.extract_telemetry_summary``），因此
    ``engine/engine.py::_derive_telemetry_dx`` 的 15 条遥测规则中有 5 条拿不到所需字段，
    永远无法触发：

    ==========================  ====================================
    规则                        所需字段（单帧路径从不提供）
    ==========================  ====================================
    6  直道速度低                ``on_straight``
    7  入弯响应差                ``max_steer``
    10 弯中不稳定                ``avg_steer``
    15 直道极速低                ``max_speed``
    5  出弯油门低                ``sector``（且早期是 0/1/2 与 3 比较错位）
    ==========================  ====================================

    本聚合器在接收侧持续累积这些量，产出引擎可直接消费的整圈统计；
    环内成本云端实测约 1.1–2.0 µs/帧 → 90 秒单圈约 6–11 ms（约 0.01% 单核）。

线程模型：
    ``on_telemetry`` / ``on_lap_data`` 由 UDP 接收线程调用，
    ``snapshot`` 由 API 线程调用；内部用 ``threading.Lock`` 保护。
"""

from __future__ import annotations

import threading
from typing import Any

# 判定「当前处于直道」的阈值（转向接近回正 + 大油门）
_STRAIGHT_STEER_MAX = 0.05
_STRAIGHT_THROTTLE_MIN = 0.9


class LapAggregator:
    """逐帧累积整圈遥测统计，并在圈号变化时固化上一圈快照。

    产出字段（``snapshot()``）与 ``engine`` 读取的键名保持一致：
        ``max_speed`` / ``avg_speed`` / ``avg_steer`` / ``max_steer`` /
        ``avg_throttle`` / ``max_brake`` / ``m_throttle`` / ``m_brake`` /
        ``on_straight`` / ``straight_ratio`` /
        ``m_tyresSurfaceTemperature`` / ``m_tyresInnerTemperature`` /
        ``m_brakesTemperature`` / ``m_tyresPressure``（均为整圈均值）/
        ``sector``（1 基）/ ``lap_number`` / ``lap_frames``。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_acc()
        self._lap_number: int | None = None
        self._sector: int | None = None
        self._on_straight: bool = False
        self._last_completed: dict[str, Any] | None = None
        self._frames_total: int = 0

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _reset_acc(self) -> None:
        self._n = 0
        self._speed_sum = 0.0
        self._speed_max = 0.0
        self._steer_abs_sum = 0.0
        self._steer_abs_max = 0.0
        self._throttle_sum = 0.0
        self._brake_sum = 0.0
        self._brake_max = 0.0
        self._straight_frames = 0
        self._wheel_sum = {
            "m_tyresSurfaceTemperature": [0.0, 0.0, 0.0, 0.0],
            "m_tyresInnerTemperature": [0.0, 0.0, 0.0, 0.0],
            "m_brakesTemperature": [0.0, 0.0, 0.0, 0.0],
            "m_tyresPressure": [0.0, 0.0, 0.0, 0.0],
        }
        self._wheel_n = dict.fromkeys(self._wheel_sum, 0)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _accumulate_wheels(self, frame: dict[str, Any]) -> None:
        for key, target in self._wheel_sum.items():
            values = frame.get(key)
            if isinstance(values, list) and len(values) >= 4:
                for i in range(4):
                    target[i] += float(values[i])
                self._wheel_n[key] += 1

    def _build_snapshot_locked(self) -> dict[str, Any]:
        n = self._n or 1
        snapshot: dict[str, Any] = {
            "max_speed": round(self._speed_max, 3),
            "avg_speed": round(self._speed_sum / n, 3),
            "avg_steer": round(self._steer_abs_sum / n, 4),
            "max_steer": round(self._steer_abs_max, 4),
            "avg_throttle": round(self._throttle_sum / n, 4),
            "avg_brake": round(self._brake_sum / n, 4),
            "max_brake": round(self._brake_max, 4),
            "straight_ratio": round(self._straight_frames / n, 4),
            "on_straight": self._on_straight,
            "lap_number": self._lap_number,
            "lap_frames": self._n,
            "sector": self._sector,
        }
        for key, target in self._wheel_sum.items():
            count = self._wheel_n[key]
            if count:
                snapshot[key] = [round(v / count, 3) for v in target]
        return snapshot

    # ------------------------------------------------------------------ #
    # 输入
    # ------------------------------------------------------------------ #
    def on_lap_data(self, lap: dict[str, Any]) -> None:
        """接收 Packet 2 (LapData)：更新圈号 / 扇区，圈号变化时固化上一圈。"""
        sector_raw = lap.get("m_sector")
        sector = None
        if isinstance(sector_raw, (int, float)):
            sector = max(1, min(3, int(sector_raw) + 1))
        lap_no = lap.get("m_currentLapNum")
        lap_no = int(lap_no) if isinstance(lap_no, (int, float)) else None

        with self._lock:
            if sector is not None:
                self._sector = sector
            if lap_no is not None and self._lap_number is not None and lap_no != self._lap_number:
                # 完成一圈：固化快照并重置累积
                if self._n > 0:
                    self._last_completed = self._build_snapshot_locked()
                self._reset_acc()
            if lap_no is not None:
                self._lap_number = lap_no

    def on_telemetry(self, frame: dict[str, Any]) -> None:
        """接收 Packet 6 (CarTelemetry)：累积单帧统计。"""
        speed = self._as_float(frame.get("m_speed"))
        throttle = self._as_float(frame.get("m_throttle"))
        brake = self._as_float(frame.get("m_brake"))
        steer = self._as_float(frame.get("m_steer"))

        with self._lock:
            self._n += 1
            self._frames_total += 1
            if speed is not None:
                self._speed_sum += speed
                if speed > self._speed_max:
                    self._speed_max = speed
            if throttle is not None:
                self._throttle_sum += throttle
            if brake is not None:
                self._brake_sum += brake
                if brake > self._brake_max:
                    self._brake_max = brake
            if steer is not None:
                steer_abs = abs(steer)
                self._steer_abs_sum += steer_abs
                if steer_abs > self._steer_abs_max:
                    self._steer_abs_max = steer_abs
            if (steer is not None and throttle is not None
                    and abs(steer) <= _STRAIGHT_STEER_MAX
                    and throttle >= _STRAIGHT_THROTTLE_MIN):
                self._straight_frames += 1
                self._on_straight = True
            else:
                self._on_straight = False
            self._accumulate_wheels(frame)

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        """当前这一圈「到目前为止」的统计快照。"""
        with self._lock:
            return self._build_snapshot_locked()

    def last_completed_lap(self) -> dict[str, Any] | None:
        """上一圈的完整统计快照（尚未跑完任何一圈时为 None）。"""
        with self._lock:
            return dict(self._last_completed) if self._last_completed else None

    def best_snapshot(self) -> dict[str, Any] | None:
        """优先返回上一整圈快照；没有则返回当前圈至今的快照（帧数 > 0 时）。"""
        completed = self.last_completed_lap()
        if completed is not None:
            return completed
        with self._lock:
            if self._n == 0:
                return None
            return self._build_snapshot_locked()

    def reset(self) -> None:
        """清空全部状态（供测试与切换赛道时使用）。"""
        with self._lock:
            self._reset_acc()
            self._lap_number = None
            self._sector = None
            self._on_straight = False
            self._last_completed = None
            self._frames_total = 0

    @property
    def frames_total(self) -> int:
        """累计接收的遥测帧数（用于健康检查/诊断）。"""
        return self._frames_total
