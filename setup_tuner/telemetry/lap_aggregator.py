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

import logging
import threading
from typing import Any

from .packets import to_sector_1based

logger = logging.getLogger(__name__)

# 判定「当前处于直道」的阈值（转向接近回正 + 大油门）
_STRAIGHT_STEER_MAX = 0.05
_STRAIGHT_THROTTLE_MIN = 0.9

# 底板（plank）离地高度判定阈值，单位米。
# Source: EA F1 25 UDP Telemetry Specification, Packet 13 (MotionEx) —
# ``m_frontAeroHeight`` / ``m_rearAeroHeight`` 为 "plank edge height above road
# surface"（底板前后缘离地高度）。
# F1 规则要求底板磨块（skid block）厚度 10mm，且离地间隙被压到约 1"/5mm 量级
# 即视为底板触地（bottoming / 刮底）；此处以「最小值落入该带内」作为触发信号。
_PLANK_BOTTOMING_MIN_M = 0.002
_PLANK_BOTTOMING_MAX_M = 0.012
# 悬挂行程（m_suspensionPosition）被压到接近下限时同样视为触底迹象
_SUSPENSION_BOTTOMING_MAX_M = 0.005

# ── 路肩（kerb）检测 ────────────────────────────────────────────────────
# 信号：MotionEx ``m_suspensionAcceleration``（四轮悬挂加速度）。
# 实测（data/recordings 真实圈，87k 帧）该信号绝对量纲噪声极大
# （p50≈175、p90≈1230、p99≈10978、max≈178799），**绝对阈值法不可用**
# ——74% 的帧都会越过 500。故改用**相对法**：
#   逐帧取四轮 |加速度| 最大值作为「路面粗糙度」，
#   按弯聚合后与全圈各弯中位数比较，显著偏高的弯即压路肩弯。
# 实测验证（Monza 11 弯）：检出 T5(Roggia 减速弯) 2.9x / T10(Ascari 减速弯)
# 2.9x / T7(Lesmo2) 2.8x / T2(Rettifilo 减速弯) 2.0x —— 与赛历吻合。
_KERB_ROUGHNESS_RATIO = 1.6   # 相对全圈中位数的倍数阈值
_KERB_MIN_FRAMES = 15         # 单弯最少帧数，避免样本过少误报

# ── Packet 7 (CarStatus) 轮胎配方枚举 ───────────────────────────────────
# Source: EA F1 25 UDP Telemetry Specification — m_actualTyreCompound
#   16 = C5（最软） 17 = C4  18 = C3  19 = C2  20 = C1（最硬）
#   21 = C0（超硬） 22 = C6（超软）
#   7  = Inter（中性胎） 8 = Wet（全雨胎）
#   9/10/11 = 经典软/中/硬（F1 经典配方）
# 软胎工作窗口低、升温快 → 胎温告警阈值应下调；
# 硬胎工作窗口高、升温慢 → 阈值应上调。
_SOFT_COMPOUNDS = frozenset({16, 17, 22, 9})
_HARD_COMPOUNDS = frozenset({20, 21, 19, 11})
_WET_COMPOUNDS = frozenset({7, 8})

_COMPOUND_NAMES: dict[int, str] = {
    7: "Inter", 8: "Wet",
    9: "Soft(Classic)", 10: "Medium(Classic)", 11: "Hard(Classic)",
    16: "C5", 17: "C4", 18: "C3", 19: "C2", 20: "C1", 21: "C0", 22: "C6",
}


def _compound_label(code: int | None) -> str:
    """轮胎配方代码 → 可读名称（未知代码返回 ``Unknown(n)``）。"""
    if code is None:
        return "Unknown"
    return _COMPOUND_NAMES.get(code, f"Unknown({code})")


def _pick_int(value: Any, fallback: int | None) -> int | None:
    """取整数状态量：非法值（None/bool/非数值）时保留上一次有效值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return int(value)


def _pick_float(value: Any, fallback: float | None) -> float | None:
    """取浮点状态量：非法值时保留上一次有效值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


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
        # 赛道上下文：跨圈保持（_reset_acc 不重置它），由 set_track_context 注入
        self._track_length: float | None = None
        self._corner_locator: Any = None
        self._reset_acc()
        self._lap_time_ms: int | None = None
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
        # Packet 13 累积量：底板离地高度极值 / 触底帧计数
        self._motion_n = 0
        self._front_height_min: float | None = None
        self._rear_height_min: float | None = None
        self._front_height_sum = 0.0
        self._rear_height_sum = 0.0
        self._plank_bottoming_frames = 0
        self._suspension_height_min: float | None = None
        # 路肩检测：按弯粗糙度累积（逐圈重置）。
        # 注意：赛道上下文（_track_length / _corner_locator）**不在此重置**——
        # 它由 set_track_context() 注入后必须跨圈保持，否则每圈都被清空。
        self._lap_distance: float | None = None
        self._corner_rough_sum: dict[int, float] = {}
        self._corner_rough_n: dict[int, int] = {}
        self._corner_rough_left: dict[int, float] = {}
        self._corner_rough_right: dict[int, float] = {}
        # Packet 7 累积量：轮胎配方/胎龄/燃油/ERS/刹车平衡
        # 这些是"整圈不变或单调变化"的状态量，取最后一次有效值即可。
        self._status_frames = 0
        self._tyre_compound: int | None = None
        self._visual_tyre_compound: int | None = None
        self._tyres_age_laps: int | None = None
        self._fuel_in_tank: float | None = None
        self._fuel_remaining_laps: float | None = None
        self._front_brake_bias: float | None = None
        self._ers_store_energy: float | None = None
        self._ers_deploy_mode: int | None = None
        self._traction_control: int | None = None
        self._anti_lock_brakes: int | None = None
        self._fuel_mix: int | None = None
        self._max_rpm: int | None = None
        self._drs_allowed: int | None = None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _kerb_corners_locked(self) -> list[dict[str, Any]]:
        """挑出「压路肩」的弯（调用方须持锁）。

        相对法：逐弯路面粗糙度均值 vs 全圈各弯中位数。绝对值噪声大，
        但「相对偏粗糙」稳定——路肩把车轮顶起，悬挂加速度显著高于正常路面。

        Returns:
            按 ratio 降序的列表，每项 ``{corner, ratio, frames, side}``；
            弯数不足 3 个或基线为 0 时返回空（中位数无意义）。
        """
        means = {
            c: self._corner_rough_sum[c] / n
            for c, n in self._corner_rough_n.items()
            if n >= _KERB_MIN_FRAMES
        }
        if len(means) < 3:
            return []
        ordered = sorted(means.values())
        base = ordered[len(ordered) // 2]
        if base <= 0:
            return []
        out: list[dict[str, Any]] = []
        for corner, mean in means.items():
            ratio = mean / base
            if ratio >= _KERB_ROUGHNESS_RATIO:
                left = self._corner_rough_left.get(corner, 0.0)
                right = self._corner_rough_right.get(corner, 0.0)
                out.append({
                    "corner": corner,
                    "ratio": round(ratio, 2),
                    "frames": self._corner_rough_n[corner],
                    "side": "left" if left >= right else "right",
                })
        out.sort(key=lambda d: (-d["ratio"], d["corner"]))
        return out

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
            "lap_time_ms": self._lap_time_ms,
        }
        for key, target in self._wheel_sum.items():
            count = self._wheel_n[key]
            if count:
                snapshot[key] = [round(v / count, 3) for v in target]
        # Packet 13：底板离地高度统计（规则9 刮底检测的输入）
        if self._motion_n:
            n_motion = self._motion_n
            snapshot["plank_front_height_min"] = round(self._front_height_min or 0.0, 5)
            snapshot["plank_rear_height_min"] = round(self._rear_height_min or 0.0, 5)
            snapshot["plank_front_height_avg"] = round(self._front_height_sum / n_motion, 5)
            snapshot["plank_rear_height_avg"] = round(self._rear_height_sum / n_motion, 5)
            snapshot["plank_bottoming_ratio"] = round(
                self._plank_bottoming_frames / n_motion, 5,
            )
            snapshot["plank_bottoming"] = self._plank_bottoming_frames > 0
            snapshot["motion_ex_frames"] = n_motion
            if self._suspension_height_min is not None:
                snapshot["suspension_height_min"] = round(
                    self._suspension_height_min, 5,
                )
            # 路肩：按弯粗糙度相对全圈中位数，挑出压路肩弯（有些路肩不得不压）
            kerb = self._kerb_corners_locked()
            if kerb:
                snapshot["kerb_corners"] = kerb
        # Packet 7：车辆状态（轮胎配方/胎龄/燃油/ERS/刹车平衡）
        if self._status_frames:
            snapshot["car_status_frames"] = self._status_frames
            if self._tyre_compound is not None:
                snapshot["tyre_compound"] = self._tyre_compound
                snapshot["tyre_compound_name"] = _compound_label(self._tyre_compound)
                # 碳陶瓷/软胎等工作窗口差异由引擎侧按配方选择阈值
                snapshot["is_soft_compound"] = self._tyre_compound in _SOFT_COMPOUNDS
                snapshot["is_hard_compound"] = self._tyre_compound in _HARD_COMPOUNDS
            if self._visual_tyre_compound is not None:
                snapshot["visual_tyre_compound"] = self._visual_tyre_compound
            if self._tyres_age_laps is not None:
                snapshot["tyres_age_laps"] = self._tyres_age_laps
            if self._fuel_in_tank is not None:
                snapshot["fuel_in_tank"] = round(self._fuel_in_tank, 3)
            if self._fuel_remaining_laps is not None:
                snapshot["fuel_remaining_laps"] = round(self._fuel_remaining_laps, 3)
            if self._front_brake_bias is not None:
                # 游戏内实际刹车平衡（% 前轴）：判断调教写入值是否真被生效
                snapshot["front_brake_bias"] = round(self._front_brake_bias, 3)
            if self._ers_store_energy is not None:
                snapshot["ers_store_energy"] = round(self._ers_store_energy, 3)
            if self._ers_deploy_mode is not None:
                snapshot["ers_deploy_mode"] = self._ers_deploy_mode
            if self._traction_control is not None:
                snapshot["traction_control"] = self._traction_control
            if self._anti_lock_brakes is not None:
                snapshot["anti_lock_brakes"] = self._anti_lock_brakes
            if self._fuel_mix is not None:
                snapshot["fuel_mix"] = self._fuel_mix
            if self._max_rpm is not None:
                snapshot["max_rpm"] = self._max_rpm
            if self._drs_allowed is not None:
                snapshot["drs_allowed"] = self._drs_allowed
        return snapshot

    def set_track_context(
        self, track_length: float | None, corner_locator: Any = None,
    ) -> None:
        """注入赛道上下文，用于把帧归因到弯道（路肩检测）。

        Args:
            track_length: 赛道长度（米，来自 Session 包 ``m_trackLength``）。
            corner_locator: 可调用对象 ``f(lap_distance_m) -> corner_number|None``。
                由调用方（app / exporter）用 :func:`domain.corner_locator.locate_corner`
                构造——**telemetry 层不反向依赖 domain 层**，故用回调解耦。
                为 None 时不做按弯归因（路肩统计自动缺省为空）。

        注入后 ``on_motion_ex`` 会把每帧路面粗糙度按弯累积，圈末在快照里
        输出 ``kerb_corners``（显著偏粗糙的弯 = 压路肩弯）。
        """
        with self._lock:
            self._track_length = track_length
            self._corner_locator = corner_locator

    # ------------------------------------------------------------------ #
    # 输入
    # ------------------------------------------------------------------ #
    def on_lap_data(self, lap: dict[str, Any]) -> None:
        """接收 Packet 2 (LapData)：更新圈号 / 扇区，圈号变化时固化上一圈。"""
        # 扇区统一走 to_sector_1based（0 基 → 1 基），避免此处再次手写 +1
        # 而与 ws.py / report.builder 的转换发生偏移。
        sector = to_sector_1based(lap.get("m_sector"))
        lap_no = lap.get("m_currentLapNum")
        lap_no = int(lap_no) if isinstance(lap_no, (int, float)) else None

        with self._lock:
            # 圈内距离：路肩检测按弯归因的唯一位置来源（MotionEx 不带距离）
            dist = lap.get("m_lapDistance")
            if isinstance(dist, (int, float)) and not isinstance(dist, bool):
                self._lap_distance = float(dist)
            if sector is not None:
                self._sector = sector
            # 记录最近一次"上圈完成圈时"（m_lastLapTimeInMS），圈号变化时归入上一圈
            last_lap_ms = lap.get("m_lastLapTimeInMS")
            if isinstance(last_lap_ms, (int, float)):
                self._lap_time_ms = int(last_lap_ms)
            if lap_no is not None and self._lap_number is not None and lap_no != self._lap_number:
                # 完成一圈：固化快照并重置累积
                if self._n > 0:
                    self._last_completed = self._build_snapshot_locked()
                self._reset_acc()
                self._lap_time_ms = None
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

    def on_motion_ex(self, frame: dict[str, Any]) -> None:
        """接收 Packet 13 (MotionEx)：累积底板离地高度与悬挂行程极值。

        这是「规则9 刮底检测」唯一的真实信号源。规范中
        ``m_frontAeroHeight`` / ``m_rearAeroHeight`` 即底板前/后缘离地高度，
        悬挂位置 ``m_suspensionPosition`` 可佐证悬挂触底。

        判定策略（保守，避免误报）：
        - 只要某一帧底板前后缘**任一**最小值落入 ``[_PLANK_BOTTOMING_MIN_M,
          _PLANK_BOTTOMING_MAX_M]`` 区间，即计一次触底帧；
        - 整圈 ``plank_bottoming`` 为真，表示这一圈至少发生过一次底板触地。
        """
        front = self._as_float(frame.get("m_frontAeroHeight"))
        rear = self._as_float(frame.get("m_rearAeroHeight"))

        suspension_min: float | None = None
        susp = frame.get("m_suspensionPosition")
        if isinstance(susp, list) and susp:
            candidates = [
                float(v) for v in susp
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if candidates:
                suspension_min = min(candidates)

        # 路肩信号：四轮悬挂加速度，分左右侧取最大（判定压哪一侧的路肩）。
        # 车轮序（官方规范）：0=RL 1=RR 2=FL 3=FR → 左=RL/FL，右=RR/FR。
        rough_left: float | None = None
        rough_right: float | None = None
        accel = frame.get("m_suspensionAcceleration")
        if isinstance(accel, list) and len(accel) == 4:
            vals: list[float] = []
            for v in accel:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(abs(float(v)))
            if len(vals) == 4:
                rough_left = max(vals[0], vals[2])
                rough_right = max(vals[1], vals[3])

        # 所有信号都没有 → 该帧无有效内容，不计入也不改变状态
        if (front is None and rear is None and suspension_min is None
                and rough_left is None):
            return

        with self._lock:
            self._motion_n += 1
            if front is not None:
                self._front_height_sum += front
                if self._front_height_min is None or front < self._front_height_min:
                    self._front_height_min = front
            if rear is not None:
                self._rear_height_sum += rear
                if self._rear_height_min is None or rear < self._rear_height_min:
                    self._rear_height_min = rear
            if suspension_min is not None:
                if (self._suspension_height_min is None
                        or suspension_min < self._suspension_height_min):
                    self._suspension_height_min = suspension_min
            heights = [h for h in (front, rear) if h is not None]
            if heights:
                lowest = min(heights)
                if _PLANK_BOTTOMING_MIN_M <= lowest <= _PLANK_BOTTOMING_MAX_M:
                    self._plank_bottoming_frames += 1
                elif lowest < _PLANK_BOTTOMING_MIN_M:
                    # 离地高度低于物理下限（罕见，通常为悬空/异常数据）——
                    # 若悬挂同时压到接近下限，仍判为触底。
                    if (suspension_min is not None
                            and suspension_min <= _SUSPENSION_BOTTOMING_MAX_M):
                        self._plank_bottoming_frames += 1
            elif (suspension_min is not None
                    and suspension_min <= _SUSPENSION_BOTTOMING_MAX_M):
                # 本帧无底板离地高度，但悬挂行程已压到接近下限 —— 视为触底迹象。
                self._plank_bottoming_frames += 1

            # ── 路肩归因：把本帧路面粗糙度记到所属弯 ──
            if (rough_left is not None and rough_right is not None
                    and self._corner_locator is not None
                    and self._track_length
                    and self._lap_distance is not None):
                try:
                    corner = self._corner_locator(self._lap_distance)
                except Exception:
                    # 定位失败不致命（该帧不参与路肩统计），但必须留痕
                    logger.warning("路肩归因定位失败", exc_info=True)
                    corner = None
                if corner is not None:
                    rough = max(rough_left, rough_right)
                    self._corner_rough_sum[corner] = (
                        self._corner_rough_sum.get(corner, 0.0) + rough
                    )
                    self._corner_rough_n[corner] = (
                        self._corner_rough_n.get(corner, 0) + 1
                    )
                    # 记录两侧各自累积，用于判定该弯主要压哪一侧
                    if rough_left >= rough_right:
                        self._corner_rough_left[corner] = (
                            self._corner_rough_left.get(corner, 0.0) + rough_left
                        )
                    else:
                        self._corner_rough_right[corner] = (
                            self._corner_rough_right.get(corner, 0.0) + rough_right
                        )

    def on_car_status(self, frame: dict[str, Any]) -> None:
        """接收 Packet 7 (CarStatus)：累积轮胎配方 / 胎龄 / 燃油 / ERS / 刹车平衡。

        这些字段此前被 ``parse_car_status`` 解析出来却无人消费，导致：
        - 胎温/胎压阈值无法按配方（软/中/硬/雨胎）区分；
        - 游戏内实际刹车平衡（``m_frontBrakeBias``）无从核对调教是否生效；
        - 燃油与 ERS 状态在报告中完全缺失。

        状态量多为"整圈不变或单调变化"，取最后一次有效值即可。
        """
        with self._lock:
            self._status_frames += 1
            self._tyre_compound = _pick_int(
                frame.get("m_actualTyreCompound"), self._tyre_compound,
            )
            self._visual_tyre_compound = _pick_int(
                frame.get("m_visualTyreCompound"), self._visual_tyre_compound,
            )
            self._tyres_age_laps = _pick_int(
                frame.get("m_tyresAgeLaps"), self._tyres_age_laps,
            )
            self._fuel_in_tank = _pick_float(
                frame.get("m_fuelInTank"), self._fuel_in_tank,
            )
            self._fuel_remaining_laps = _pick_float(
                frame.get("m_fuelRemainingLaps"), self._fuel_remaining_laps,
            )
            self._front_brake_bias = _pick_float(
                frame.get("m_frontBrakeBias"), self._front_brake_bias,
            )
            self._ers_store_energy = _pick_float(
                frame.get("m_ersStoreEnergy"), self._ers_store_energy,
            )
            self._ers_deploy_mode = _pick_int(
                frame.get("m_ersDeployMode"), self._ers_deploy_mode,
            )
            self._traction_control = _pick_int(
                frame.get("m_tractionControl"), self._traction_control,
            )
            self._anti_lock_brakes = _pick_int(
                frame.get("m_antiLockBrakes"), self._anti_lock_brakes,
            )
            self._fuel_mix = _pick_int(frame.get("m_fuelMix"), self._fuel_mix)
            self._max_rpm = _pick_int(frame.get("m_maxRPM"), self._max_rpm)
            self._drs_allowed = _pick_int(
                frame.get("m_drsAllowed"), self._drs_allowed,
            )

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

    def take_completed_lap(self) -> dict[str, Any] | None:
        """取出上一整圈快照（取出后清空；无则 None）。供落库线程消费。"""
        with self._lock:
            snapshot, self._last_completed = self._last_completed, None
            return dict(snapshot) if snapshot else None

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
