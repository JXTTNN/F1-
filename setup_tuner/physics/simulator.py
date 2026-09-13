"""圈速仿真器 — 整合6大子模型，逐段仿真计算圈速."""
from __future__ import annotations

import math
from dataclasses import dataclass

from setup_tuner.domain.setup import CarSetup
from setup_tuner.domain.track import Track
from setup_tuner.physics.aero_model import AeroModel
from setup_tuner.physics.brake_model import BrakeModel
from setup_tuner.physics.diff_model import DiffModel
from setup_tuner.physics.suspension_model import SuspensionModel
from setup_tuner.physics.track_model import (
    CornerSegment,
    StraightSegment,
    TrackModel,
)
from setup_tuner.physics.tyre_model import TyreModel

# ── 物理常数 ──
G = 9.81
CAR_MASS = 798.0
ENGINE_POWER = 750.0  # kW
ENGINE_FORCE_MAX = 15000.0
ROLLING_RESISTANCE = 0.015
FUEL_CONSUMPTION_RATE = 1.5
MIN_GRIP = 1.9
# ── 仿真参数 ──
SLIP_ANGLE = 3.0
OPTIMAL_AERO_BAL = 45.0
TEMP_SLIP_HEAT = 3.0
TEMP_COOLING = 0.02
WEIGHT_DIST_FRONT = 0.5
SECTOR_COUNT = 3
TOP_SPEED_STEP = 1.0
TOP_SPEED_MAX = 200.0
MIN_SPEED = 10.0
WEAR_LEN_REF = 250.0
WEAR_LAT_FACTOR = 0.1
CORNER_ITERS = 3


@dataclass
class CornerResult:
    """单弯仿真结果."""
    number: int
    name: str
    speed_entry: float
    speed_apex: float
    speed_exit: float
    lateral_accel: float
    tyre_wear: float
    stability: float
    balance: float


@dataclass
class LapResult:
    """圈速仿真结果."""
    lap_time: float
    sector_times: list[float]
    corner_results: list[CornerResult]
    total_tyre_wear: float
    avg_stability: float
    avg_balance: float
    top_speed: float
    fuel_consumption: float


class Simulator:
    """圈速仿真器 — 整合6大子模型，逐段仿真计算圈速."""

    def __init__(self, setup: CarSetup, track: Track,
                 tyre_type: str = "medium", track_temp: float = 30.0,
                 ambient_temp: float = 20.0, fuel_load: float = 100.0) -> None:
        self.setup = setup
        self.track = track
        self.tyre_type = tyre_type
        self.track_temp = track_temp
        self.ambient_temp = ambient_temp
        self.fuel_load = fuel_load
        s = setup
        self.aero = AeroModel(s.front_wing, s.rear_wing)
        self.tyre_f = TyreModel(tyre_type, s.front_left_tyre_pressure, s.front_camber)
        self.tyre_r = TyreModel(tyre_type, s.rear_left_tyre_pressure, s.rear_camber)
        self.suspension = SuspensionModel(
            s.front_suspension, s.rear_suspension, s.front_anti_roll_bar,
            s.rear_anti_roll_bar, s.front_ride_height, s.rear_ride_height)
        self.diff = DiffModel(s.on_throttle_diff, s.off_throttle_diff)
        self.brake = BrakeModel(s.brake_pressure, s.brake_bias)
        # 飞行圈轮胎已预热至工作温度
        self.temp_f: float = self.tyre_f.params["temp_opt"]
        self.temp_r: float = self.tyre_r.params["temp_opt"]
        self._top_speed: float = self._compute_top_speed()

    def simulate_lap(self) -> LapResult:
        """执行完整圈速仿真."""
        track_out = TrackModel(self.track).compute(max_lateral_accel=2.5)
        corners, straights = track_out.corners, track_out.straights
        corner_results = [self._simulate_corner(c) for c in corners]
        seg_times = self._compute_segment_times(corners, straights, corner_results)
        return self._assemble_result(corner_results, seg_times)

    # ── 弯道仿真 ──
    def _simulate_corner(self, corner: CornerSegment) -> CornerResult:
        v_apex = self._compute_corner_max_speed(corner)
        lat_g = (v_apex * v_apex) / (corner.radius * G) if corner.radius > 0 else 0.0
        wear = self._compute_tyre_wear(corner, lat_g, v_apex)
        stability = self._compute_stability(lat_g, v_apex)
        balance = self._compute_balance(v_apex)
        self._update_tyre_temp(lat_g)
        return CornerResult(corner.number, corner.name, v_apex, v_apex, v_apex,
                            lat_g, wear, stability, balance)

    def _compute_corner_max_speed(self, corner: CornerSegment) -> float:
        if corner.radius <= 0:
            return MIN_SPEED
        v_est = corner.speed_max
        for _ in range(CORNER_ITERS):
            a_lat_max = self._compute_max_lateral_accel(v_est)
            v_max = math.sqrt(a_lat_max * G * corner.radius)
            v_est = (v_est + v_max) / 2.0
        return max(v_max, MIN_SPEED)

    def _compute_max_lateral_accel(self, speed: float) -> float:
        """a_lat_max = μ_avg * (1 + downforce / (m*g))."""
        a = self.aero.compute(speed, self.setup.front_ride_height, self.setup.rear_ride_height)
        fl = CAR_MASS * G * WEIGHT_DIST_FRONT + a.downforce_front
        rl = CAR_MASS * G * (1 - WEIGHT_DIST_FRONT) + a.downforce_rear
        ft = self.tyre_f.compute(self.temp_f, fl, SLIP_ANGLE)
        rt = self.tyre_r.compute(self.temp_r, rl, SLIP_ANGLE)
        mu = max((ft.grip_coefficient + rt.grip_coefficient) / 2.0, MIN_GRIP)
        return mu * (1.0 + a.downforce_total / (CAR_MASS * G))

    def _compute_tyre_wear(self, corner: CornerSegment, lat_g: float, speed: float) -> float:
        a = self.aero.compute(speed, self.setup.front_ride_height, self.setup.rear_ride_height)
        load = CAR_MASS * G * WEIGHT_DIST_FRONT + a.downforce_front
        t = self.tyre_f.compute(self.temp_f, load, SLIP_ANGLE)
        return t.wear_rate * (corner.length / WEAR_LEN_REF) * (1.0 + lat_g * WEAR_LAT_FACTOR)

    def _compute_stability(self, lat_g: float, speed: float) -> float:
        d = self.diff.compute()
        susp = self.suspension.compute(lat_g)
        a = self.aero.compute(speed, self.setup.front_ride_height, self.setup.rear_ride_height)
        diff_stab = (d.corner_entry_stability + d.corner_exit_traction) / 2.0
        susp_stab = 1.0 - susp.bottoming_risk
        aero_dev = abs(a.aero_balance - OPTIMAL_AERO_BAL)
        aero_stab = max(0.0, 1.0 - aero_dev / OPTIMAL_AERO_BAL)
        return 0.4 * diff_stab + 0.3 * susp_stab + 0.3 * aero_stab

    def _compute_balance(self, speed: float) -> float:
        a = self.aero.compute(speed, self.setup.front_ride_height, self.setup.rear_ride_height)
        fl = CAR_MASS * G * WEIGHT_DIST_FRONT + a.downforce_front
        rl = CAR_MASS * G * (1 - WEIGHT_DIST_FRONT) + a.downforce_rear
        ft = self.tyre_f.compute(self.temp_f, fl, SLIP_ANGLE)
        rt = self.tyre_r.compute(self.temp_r, rl, SLIP_ANGLE)
        fg = max(ft.grip_coefficient, MIN_GRIP) * fl
        rg = max(rt.grip_coefficient, MIN_GRIP) * rl
        total = fg + rg
        return (rg - fg) / total if total > 0 else 0.0

    def _update_tyre_temp(self, lat_g: float) -> None:
        heat = lat_g * TEMP_SLIP_HEAT
        self.temp_f += heat - (self.temp_f - self.ambient_temp) * TEMP_COOLING
        self.temp_r += heat - (self.temp_r - self.ambient_temp) * TEMP_COOLING

    # ── 直道仿真 ──
    def _compute_segment_times(self, corners: list[CornerSegment],
                                straights: list[StraightSegment],
                                cr: list[CornerResult]) -> list[float]:
        times: list[float] = []
        for i, corner in enumerate(corners):
            v = max(cr[i].speed_apex, MIN_SPEED)
            times.append(corner.length / v)
            if i < len(straights):
                v_entry = cr[i].speed_exit
                v_exit = cr[i + 1].speed_entry if i + 1 < len(cr) else v_entry
                times.append(self._compute_straight_time(v_entry, v_exit, straights[i].length))
        return times

    def _compute_straight_time(self, v_entry: float, v_exit: float, length: float) -> float:
        if length <= 0:
            return 0.0
        v_avg = (v_entry + v_exit) / 2.0
        a_acc = self._compute_acceleration(v_entry, v_avg)
        a_dec = self._compute_deceleration(v_avg)
        if a_acc <= 0:
            return length / max(v_entry, MIN_SPEED)
        if a_dec <= 0:
            return length / max(v_avg, MIN_SPEED)
        v_peak_sq = (2.0 * length * a_acc * a_dec + a_dec * v_entry**2
                     + a_acc * v_exit**2) / (a_acc + a_dec)
        v_peak = min(math.sqrt(max(v_peak_sq, 0.0)), self._top_speed)
        v_peak = max(v_peak, max(v_entry, v_exit))
        d_acc = (v_peak**2 - v_entry**2) / (2.0 * a_acc)
        d_dec = (v_peak**2 - v_exit**2) / (2.0 * a_dec)
        d_cruise = max(0.0, length - d_acc - d_dec)
        t_cruise = d_cruise / v_peak if v_peak > 0 else 0.0
        return (v_peak - v_entry) / a_acc + t_cruise + (v_peak - v_exit) / a_dec

    def _compute_acceleration(self, v_low: float, v_high: float) -> float:
        v_avg = max((v_low + v_high) / 2.0, MIN_SPEED)
        a = self.aero.compute(v_avg, self.setup.front_ride_height, self.setup.rear_ride_height)
        f_eng = min(ENGINE_POWER * 1000.0 / v_avg, ENGINE_FORCE_MAX)
        f_traction = MIN_GRIP * (CAR_MASS * G + a.downforce_total)
        f_drive = min(f_eng, f_traction)
        return (f_drive - a.drag_force - ROLLING_RESISTANCE * CAR_MASS * G) / CAR_MASS

    def _compute_deceleration(self, speed: float) -> float:
        a = self.aero.compute(speed, self.setup.front_ride_height, self.setup.rear_ride_height)
        ft = self.tyre_f.compute(self.temp_f, CAR_MASS * G * WEIGHT_DIST_FRONT + a.downforce_front)
        rt = self.tyre_r.compute(self.temp_r, CAR_MASS * G * (1 - WEIGHT_DIST_FRONT) + a.downforce_rear)
        b = self.brake.compute(downforce_front=a.downforce_front, downforce_rear=a.downforce_rear,
                               tyre_mu_front=ft.grip_coefficient, tyre_mu_rear=rt.grip_coefficient)
        max_dec = min(b.deceleration_max, max(ft.grip_coefficient, MIN_GRIP) * G,
                      max(rt.grip_coefficient, MIN_GRIP) * G)
        return max(max_dec, 0.1)

    def _compute_top_speed(self) -> float:
        v = 30.0
        while v < TOP_SPEED_MAX:
            a = self.aero.compute(v, self.setup.front_ride_height, self.setup.rear_ride_height)
            f_eng = min(ENGINE_POWER * 1000.0 / v, ENGINE_FORCE_MAX)
            if f_eng <= a.drag_force + ROLLING_RESISTANCE * CAR_MASS * G:
                break
            v += TOP_SPEED_STEP
        return v

    # ── 结果组装 ──
    def _assemble_result(self, cr: list[CornerResult], seg_times: list[float]) -> LapResult:
        lap_time = sum(seg_times)
        n = len(cr)
        avg_stab = sum(r.stability for r in cr) / n if n else 0.0
        avg_bal = sum(r.balance for r in cr) / n if n else 0.0
        return LapResult(lap_time, self._compute_sector_times(seg_times), cr,
                         sum(r.tyre_wear for r in cr), avg_stab, avg_bal,
                         self._top_speed, FUEL_CONSUMPTION_RATE)

    @staticmethod
    def _compute_sector_times(seg_times: list[float]) -> list[float]:
        n = len(seg_times)
        if n == 0:
            return [0.0, 0.0, 0.0]
        per = max(1, n // SECTOR_COUNT)
        return [sum(seg_times[s * per:(s + 1) * per if s < SECTOR_COUNT - 1 else n])
                for s in range(SECTOR_COUNT)]
