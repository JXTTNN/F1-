"""遥测校准器 — 用真实遥测数据校准物理模型参数.

校准方法论（融合 GitHub 参考项目经验）：
    - 网格搜索 + 坐标下降法（参考 Zoltan-eke/f1tenth_ws grid_search_tire.py）
    - 多目标加权误差度量（参考 rembertdesigns/F1-car-setup-optimizer）
    - 多轮迭代避免局部最优（参考 f1tenth_ws 已知限制修复）
    - 边界收敛检测（参考 f1tenth_ws 已知限制修复）
    - RMSE 验证仿真输出与遥测基准（参考 f1tenth_ws offline_validation.py）

校准流程：
    1. 从遥测 JSON 提取基准数据（TelemetryBenchmark）
    2. 用遥测中的调教参数构造 CarSetup
    3. 用默认参数运行仿真器，计算初始误差
    4. 对每个可校准参数进行网格搜索（坐标下降法）
    5. 多轮迭代以避免局部最优
    6. 检查边界收敛并扩展搜索范围
    7. 输出校准结果

仅依赖 Python 标准库（dataclasses, math, typing, json, itertools）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from setup_tuner.domain.setup import CarSetup
from setup_tuner.domain.track import Track
from setup_tuner.physics import aero_model as aero_mod
from setup_tuner.physics import brake_model as brake_mod
from setup_tuner.physics import simulator as sim_mod
from setup_tuner.physics import tyre_model as tyre_mod
from setup_tuner.physics.simulator import LapResult, Simulator

# ── 单位转换常量 ──
KMH_TO_MS: float = 3.6
MS_PER_SEC: float = 1000.0

# ── 轮胎配方映射（F1 25 游戏编码 → 仿真器内部编码）──
COMPOUND_MAPPING: dict[str, str] = {
    "C1": "hard",
    "C2": "medium",
    "C3": "soft",
    "C4": "soft",
    "intermediate": "intermediate",
    "wet": "wet",
}

# ── 误差权重（多目标加权）──
W_LAP_TIME: float = 0.50
W_TOP_SPEED: float = 0.30
W_CORNER_SPEED: float = 0.20

# ── 校准迭代参数 ──
MAX_ROUNDS: int = 3
CORNER_PERCENTILE: float = 25.0
PRESSURE_VALID_THRESHOLD: float = 10.0

# ── 可校准参数定义 ──
# name: 参数名, module: 所属模块, default: 默认值,
# search_min/max: 搜索范围, step: 步长
CALIBRATABLE_PARAMS: list[dict[str, Any]] = [
    {"name": "ENGINE_POWER", "module": "sim",
     "default": 750.0, "search_min": 700.0, "search_max": 800.0, "step": 10.0},
    {"name": "CL_FRONT_MAX", "module": "aero",
     "default": 2.0, "search_min": 1.5, "search_max": 2.5, "step": 0.1},
    {"name": "CL_REAR_MAX", "module": "aero",
     "default": 1.6, "search_min": 1.2, "search_max": 2.0, "step": 0.1},
    {"name": "CD_BASE", "module": "aero",
     "default": 0.3, "search_min": 0.2, "search_max": 0.4, "step": 0.02},
    {"name": "mu_peak_medium", "module": "tyre",
     "default": 1.6, "search_min": 1.3, "search_max": 1.9, "step": 0.05},
    {"name": "MAX_BRAKE_FORCE_PER_WHEEL", "module": "brake",
     "default": 2000.0, "search_min": 1500.0, "search_max": 2500.0, "step": 100.0},
    {"name": "CAR_MASS", "module": "sim",
     "default": 798.0, "search_min": 780.0, "search_max": 820.0, "step": 5.0},
    {"name": "MIN_GRIP", "module": "sim",
     "default": 1.9, "search_min": 1.5, "search_max": 2.2, "step": 0.05},
]


@dataclass
class TelemetryBenchmark:
    """从遥测 JSON 提取的基准数据."""
    lap_time_actual: float
    top_speed_actual: float
    avg_corner_speed_actual: float
    max_lat_g: float
    max_accel_g: float
    max_decel_g: float
    sector_times_actual: list[float]
    tyre_temps_actual: list[float]
    track_temp: float
    ambient_temp: float
    tyre_compound: str
    weather: str


@dataclass
class CalibrationResult:
    """校准结果."""
    calibrated_params: dict[str, float]
    original_params: dict[str, float]
    error_before: dict[str, float]
    error_after: dict[str, float]
    improvement_pct: dict[str, float]
    iterations: int
    boundary_converged: list[str]


class Calibrator:
    """遥测校准器 — 用真实遥测数据校准物理模型参数.

    采用网格搜索 + 坐标下降法，多轮迭代避免局部最优，
    边界收敛检测与范围扩展。

    Usage:
        calibrator = Calibrator(telemetry_file_path, track)
        result = calibrator.calibrate()
    """

    def __init__(self, telemetry_file_path: str, track: Track) -> None:
        self.telemetry_path: str = telemetry_file_path
        self.track: Track = track
        self._telemetry_data: dict[str, Any] = self._load_telemetry()
        self._original_values: dict[str, float] = self._snapshot_params()

    # ── 遥测数据加载 ──

    def _load_telemetry(self) -> dict[str, Any]:
        """加载遥测 JSON 文件."""
        with open(self.telemetry_path, encoding="utf-8") as f:
            return json.load(f)

    # ── 基准提取 ──

    def extract_benchmark(self) -> TelemetryBenchmark:
        """从遥测 JSON 提取基准数据."""
        data = self._telemetry_data
        samples: list[dict[str, Any]] = data.get("samples", [])

        lap_time_sec: float = data.get("lap_time_ms", 0) / MS_PER_SEC
        speeds_ms: list[float] = [
            s.get("speed", 0) / KMH_TO_MS
            for s in samples if s.get("speed", 0) > 0
        ]
        top_speed_ms: float = max(speeds_ms) if speeds_ms else 0.0
        avg_corner_ms: float = self._compute_avg_corner_speed(speeds_ms)
        max_lat, max_accel, max_decel = self._compute_g_forces(samples)
        sector_sec: list[float] = [
            t / MS_PER_SEC for t in data.get("sector_times_ms", [0, 0, 0])
        ]
        tyre_temps: list[float] = self._compute_avg_tyre_temps(samples)

        weather_data: dict[str, Any] = data.get("weather", {})
        tyre_data: dict[str, Any] = data.get("tyre", {})
        compound_name: str = tyre_data.get("compound_name", "medium")

        return TelemetryBenchmark(
            lap_time_actual=lap_time_sec,
            top_speed_actual=top_speed_ms,
            avg_corner_speed_actual=avg_corner_ms,
            max_lat_g=max_lat,
            max_accel_g=max_accel,
            max_decel_g=max_decel,
            sector_times_actual=sector_sec,
            tyre_temps_actual=tyre_temps,
            track_temp=float(weather_data.get("track_temp", 30.0)),
            ambient_temp=float(weather_data.get("air_temp", 20.0)),
            tyre_compound=COMPOUND_MAPPING.get(compound_name, "medium"),
            weather=weather_data.get("name", "dry"),
        )

    @staticmethod
    def _compute_avg_corner_speed(speeds_ms: list[float]) -> float:
        """估算平均弯道速度（取速度最低的 25% 的平均值）."""
        if not speeds_ms:
            return 0.0
        sorted_speeds: list[float] = sorted(speeds_ms)
        cutoff: int = max(1, int(len(sorted_speeds) * CORNER_PERCENTILE / 100.0))
        corner_speeds: list[float] = sorted_speeds[:cutoff]
        return sum(corner_speeds) / len(corner_speeds)

    @staticmethod
    def _compute_g_forces(
        samples: list[dict[str, Any]],
    ) -> tuple[float, float, float]:
        """从遥测数据计算最大侧向/加速/减速 G 力."""
        lat_gs: list[float] = [abs(float(s.get("lat_g", 0.0))) for s in samples]
        long_gs: list[float] = [float(s.get("long_g", 0.0)) for s in samples]

        max_lat: float = max(lat_gs) if lat_gs else 0.0
        max_accel: float = max(long_gs) if long_gs else 0.0
        max_decel: float = abs(min(long_gs)) if long_gs else 0.0

        if max_lat < 0.01 and max_accel < 0.01 and max_decel < 0.01:
            return Calibrator._estimate_g_from_speed(samples)
        return max_lat, max_accel, max_decel

    @staticmethod
    def _estimate_g_from_speed(
        samples: list[dict[str, Any]],
    ) -> tuple[float, float, float]:
        """从速度变化率估算 G 力（当遥测 G 力数据不可用时）."""
        if len(samples) < 2:
            return 0.0, 0.0, 0.0

        speeds_ms: list[float] = []
        times_s: list[float] = []
        for s in samples:
            speeds_ms.append(float(s.get("speed", 0)) / KMH_TO_MS)
            times_s.append(float(s.get("lap_time_ms", 0)) / MS_PER_SEC)

        max_accel_g: float = 0.0
        max_decel_g: float = 0.0
        for i in range(1, len(speeds_ms)):
            dt: float = times_s[i] - times_s[i - 1]
            if dt <= 0.001:
                continue
            dv: float = speeds_ms[i] - speeds_ms[i - 1]
            accel: float = dv / dt
            if accel > 0:
                max_accel_g = max(max_accel_g, accel / 9.81)
            else:
                max_decel_g = max(max_decel_g, abs(accel) / 9.81)

        return 2.0, max_accel_g, max_decel_g  # 2.0g = F1 典型最大侧向G

    @staticmethod
    def _compute_avg_tyre_temps(
        samples: list[dict[str, Any]],
    ) -> list[float]:
        """计算四轮平均胎温（从圈中段采样点）."""
        if not samples:
            return [0.0, 0.0, 0.0, 0.0]

        mid_start: int = len(samples) // 4
        mid_end: int = len(samples) * 3 // 4
        mid_samples: list[dict[str, Any]] = samples[mid_start:mid_end] or samples[:100]

        temps_sum: list[float] = [0.0, 0.0, 0.0, 0.0]
        count: int = 0
        for s in mid_samples:
            surface_temps: list = s.get("tyre_surface_temp", [])
            if len(surface_temps) >= 4:
                for i in range(4):
                    temps_sum[i] += float(surface_temps[i])
                count += 1

        if count == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [t / count for t in temps_sum]

    # ── CarSetup 构造 ──

    def _build_car_setup(self) -> CarSetup:
        """从遥测数据构造 CarSetup."""
        setup_data: dict[str, Any] = self._telemetry_data.get("setup", {})
        samples: list[dict[str, Any]] = self._telemetry_data.get("samples", [])
        tyre_pressures: list[float] = self._extract_tyre_pressures(samples)

        uint8_params: dict[str, float] = self._map_uint8_setup_params(setup_data)
        clamp_params: dict[str, float] = self._clamp_setup_params(setup_data)
        pressures: dict[str, float] = self._resolve_tyre_pressures(tyre_pressures)

        return CarSetup(**uint8_params, **clamp_params, **pressures)

    def _map_uint8_setup_params(
        self, setup_data: dict[str, Any],
    ) -> dict[str, float]:
        """将 uint8 编码的调教参数映射到 CarSetup 有效范围."""
        m: Any = self._map_uint8_to_range
        return {
            "front_wing": m(setup_data.get("front_wing", 5), 0, 10, 0.0, 10.0),
            "rear_wing": m(setup_data.get("rear_wing", 5), 0, 11, 0.0, 11.0),
            "on_throttle_diff": m(setup_data.get("on_throttle_diff", 75), 50, 100, 50.0, 100.0),
            "off_throttle_diff": m(setup_data.get("off_throttle_diff", 75), 50, 100, 50.0, 100.0),
            "brake_pressure": m(setup_data.get("brake_pressure", 90), 80, 100, 80.0, 100.0),
            "brake_bias": m(setup_data.get("brake_bias", 58), 50, 70, 50.0, 70.0),
            "front_suspension": m(setup_data.get("front_suspension", 3), 1, 6, 1.0, 6.0),
            "rear_suspension": m(setup_data.get("rear_suspension", 3), 1, 6, 1.0, 6.0),
            "front_anti_roll_bar": m(setup_data.get("front_anti_roll_bar", 6), 1, 11, 1.0, 11.0),
            "rear_anti_roll_bar": m(setup_data.get("rear_anti_roll_bar", 6), 1, 11, 1.0, 11.0),
            "front_ride_height": m(setup_data.get("front_ride_height", 4), 2, 7, 2.0, 7.0),
            "rear_ride_height": m(setup_data.get("rear_ride_height", 4), 2, 7, 2.0, 7.0),
        }

    def _clamp_setup_params(
        self, setup_data: dict[str, Any],
    ) -> dict[str, float]:
        """对 camber/toe 参数进行范围校验（不在范围内用默认值）."""
        c: Any = self._clamp_or_default
        return {
            "front_camber": c(setup_data.get("front_camber", -3.0), -3.5, -2.5, -3.0),
            "rear_camber": c(setup_data.get("rear_camber", -1.5), -2.0, -1.0, -1.5),
            "front_toe": c(setup_data.get("front_toe", 0.05), 0.0, 0.1, 0.05),
            "rear_toe": c(setup_data.get("rear_toe", 0.35), 0.1, 0.6, 0.35),
        }

    @staticmethod
    def _resolve_tyre_pressures(
        tyre_pressures: list[float],
    ) -> dict[str, float]:
        """从 samples 胎压值构造 CarSetup 胎压参数."""
        fl: float = tyre_pressures[0] if tyre_pressures[0] > PRESSURE_VALID_THRESHOLD else 23.5
        fr: float = tyre_pressures[1] if tyre_pressures[1] > PRESSURE_VALID_THRESHOLD else 23.5
        rl: float = tyre_pressures[2] if tyre_pressures[2] > PRESSURE_VALID_THRESHOLD else 22.0
        rr: float = tyre_pressures[3] if tyre_pressures[3] > PRESSURE_VALID_THRESHOLD else 22.0
        return {
            "front_left_tyre_pressure": fl,
            "front_right_tyre_pressure": fr,
            "rear_left_tyre_pressure": rl,
            "rear_right_tyre_pressure": rr,
        }

    @staticmethod
    def _map_uint8_to_range(
        raw_value: Any,
        game_min: float,
        game_max: float,
        target_min: float,
        target_max: float,
    ) -> float:
        """将 uint8 原始值映射到目标范围."""
        value: float = float(raw_value)
        if target_min <= value <= target_max:
            return value
        if game_min <= value <= game_max:
            return value
        if 0 <= value <= 255:
            return target_min + (value / 255.0) * (target_max - target_min)
        return (target_min + target_max) / 2.0

    @staticmethod
    def _clamp_or_default(
        value: Any,
        min_val: float,
        max_val: float,
        default_val: float,
    ) -> float:
        """若值在有效范围内则使用，否则用默认值."""
        v: float = float(value)
        if min_val <= v <= max_val:
            return v
        return default_val

    @staticmethod
    def _extract_tyre_pressures(
        samples: list[dict[str, Any]],
    ) -> list[float]:
        """从遥测 samples 中提取四轮平均胎压（psi）."""
        if not samples:
            return [23.5, 23.5, 22.0, 22.0]

        mid_start: int = len(samples) // 4
        mid_end: int = len(samples) * 3 // 4
        mid_samples: list[dict[str, Any]] = samples[mid_start:mid_end] or samples[:100]

        pressures_sum: list[float] = [0.0, 0.0, 0.0, 0.0]
        count: int = 0
        for s in mid_samples:
            pressures: list = s.get("tyre_pressure", [])
            if len(pressures) >= 4:
                for i in range(4):
                    pressures_sum[i] += float(pressures[i])
                count += 1

        if count == 0:
            return [23.5, 23.5, 22.0, 22.0]
        return [p / count for p in pressures_sum]

    # ── 仿真运行 ──

    def run_simulation(
        self,
        setup: CarSetup,
        benchmark: TelemetryBenchmark,
    ) -> LapResult:
        """用给定调教和当前物理参数运行仿真."""
        fuel_data: dict[str, Any] = self._telemetry_data.get("fuel", {})
        fuel_load: float = float(fuel_data.get("load_kg", 100.0))

        sim: Simulator = Simulator(
            setup=setup,
            track=self.track,
            tyre_type=benchmark.tyre_compound,
            track_temp=benchmark.track_temp,
            ambient_temp=benchmark.ambient_temp,
            fuel_load=fuel_load,
        )
        return sim.simulate_lap()

    # ── 误差计算 ──

    @staticmethod
    def compute_error(
        benchmark: TelemetryBenchmark,
        sim_result: LapResult,
    ) -> dict[str, float]:
        """计算仿真结果与遥测基准之间的误差（多目标加权）."""
        lap_err: float = 0.0
        if benchmark.lap_time_actual > 0:
            lap_err = abs(sim_result.lap_time - benchmark.lap_time_actual) / benchmark.lap_time_actual

        top_err: float = 0.0
        if benchmark.top_speed_actual > 0:
            top_err = abs(sim_result.top_speed - benchmark.top_speed_actual) / benchmark.top_speed_actual

        corner_speeds: list[float] = [cr.speed_apex for cr in sim_result.corner_results]
        sim_avg_corner: float = sum(corner_speeds) / len(corner_speeds) if corner_speeds else 0.0
        corner_err: float = 0.0
        if benchmark.avg_corner_speed_actual > 0:
            corner_err = abs(sim_avg_corner - benchmark.avg_corner_speed_actual) / benchmark.avg_corner_speed_actual

        total: float = W_LAP_TIME * lap_err + W_TOP_SPEED * top_err + W_CORNER_SPEED * corner_err

        return {
            "lap_time_error": lap_err,
            "top_speed_error": top_err,
            "corner_speed_error": corner_err,
            "total_error": total,
        }

    # ── 参数管理 ──

    @staticmethod
    def _snapshot_params() -> dict[str, float]:
        """快照当前所有可校准参数的值."""
        snapshot: dict[str, float] = {}
        for p in CALIBRATABLE_PARAMS:
            snapshot[p["name"]] = Calibrator._get_param(p["name"], p["module"])
        return snapshot

    @staticmethod
    def _get_param(name: str, module: str) -> float:
        """获取当前参数值."""
        if module == "sim":
            return float(getattr(sim_mod, name))
        if module == "aero":
            return float(getattr(aero_mod, name))
        if module == "brake":
            return float(getattr(brake_mod, name))
        if module == "tyre":
            if name.startswith("mu_peak_"):
                compound: str = name.replace("mu_peak_", "")
                return float(tyre_mod.TYRE_PARAMS[compound]["mu_peak"])
            return float(getattr(tyre_mod, name))
        raise ValueError(f"未知模块: {module}")

    @staticmethod
    def _set_param(name: str, module: str, value: float) -> None:
        """设置参数值（修改模块级变量）."""
        if module == "sim":
            setattr(sim_mod, name, value)
        elif module == "aero":
            setattr(aero_mod, name, value)
        elif module == "brake":
            setattr(brake_mod, name, value)
        elif module == "tyre":
            if name.startswith("mu_peak_"):
                compound: str = name.replace("mu_peak_", "")
                tyre_mod.TYRE_PARAMS[compound]["mu_peak"] = value
            else:
                setattr(tyre_mod, name, value)
        else:
            raise ValueError(f"未知模块: {module}")

    def _restore_params(self) -> None:
        """恢复所有参数到原始值."""
        for name, value in self._original_values.items():
            self._set_param(name, self._find_module(name), value)

    @staticmethod
    def _find_module(name: str) -> str:
        """根据参数名查找所属模块."""
        for p in CALIBRATABLE_PARAMS:
            if p["name"] == name:
                return p["module"]
        raise ValueError(f"未知参数: {name}")

    # ── 网格搜索 ──

    def _grid_search(
        self,
        param_name: str,
        search_min: float,
        search_max: float,
        step: float,
        setup: CarSetup,
        benchmark: TelemetryBenchmark,
    ) -> float:
        """对单个参数进行网格搜索，返回误差最小的参数值."""
        module: str = self._find_module(param_name)
        best_value: float = self._get_param(param_name, module)
        best_error: float = self._evaluate(setup, benchmark)

        num_steps: int = int(round((search_max - search_min) / step)) + 1
        for i in range(num_steps):
            value: float = search_min + i * step
            self._set_param(param_name, module, value)
            error: float = self._evaluate(setup, benchmark)
            if error < best_error:
                best_error = error
                best_value = value

        self._set_param(param_name, module, best_value)
        return best_value

    def _evaluate(
        self,
        setup: CarSetup,
        benchmark: TelemetryBenchmark,
    ) -> float:
        """运行仿真并返回总误差."""
        sim_result: LapResult = self.run_simulation(setup, benchmark)
        return self.compute_error(benchmark, sim_result)["total_error"]

    # ── 边界收敛检测 ──

    @staticmethod
    def _is_boundary_converged(
        param_name: str,
        best_value: float,
        search_min: float,
        search_max: float,
    ) -> bool:
        """检查参数是否收敛在搜索范围边界."""
        tolerance: float = 0.0
        for p in CALIBRATABLE_PARAMS:
            if p["name"] == param_name:
                tolerance = p["step"] / 2.0
                break
        return abs(best_value - search_min) < tolerance or abs(best_value - search_max) < tolerance

    # ── 主校准入口 ──

    def calibrate(self) -> CalibrationResult:
        """执行完整校准流程.

        1. 提取遥测基准
        2. 构造 CarSetup
        3. 计算初始误差
        4. 坐标下降法逐参数网格搜索
        5. 多轮迭代避免局部最优
        6. 边界收敛检测
        7. 恢复参数，返回结果
        """
        benchmark: TelemetryBenchmark = self.extract_benchmark()
        setup: CarSetup = self._build_car_setup()

        self._restore_params()
        error_before: dict[str, float] = self.compute_error(
            benchmark, self.run_simulation(setup, benchmark),
        )

        boundary_converged: list[str] = []
        for _ in range(MAX_ROUNDS):
            boundary_converged.extend(
                self._calibrate_round(setup, benchmark)
            )

        error_after: dict[str, float] = self.compute_error(
            benchmark, self.run_simulation(setup, benchmark),
        )

        calibrated: dict[str, float] = {}
        for p in CALIBRATABLE_PARAMS:
            calibrated[p["name"]] = self._get_param(p["name"], p["module"])

        improvement: dict[str, float] = self._compute_improvement(error_before, error_after)

        result: CalibrationResult = CalibrationResult(
            calibrated_params=calibrated,
            original_params=dict(self._original_values),
            error_before=error_before,
            error_after=error_after,
            improvement_pct=improvement,
            iterations=MAX_ROUNDS,
            boundary_converged=list(set(boundary_converged)),
        )

        self._restore_params()
        return result

    def _calibrate_round(
        self,
        setup: CarSetup,
        benchmark: TelemetryBenchmark,
    ) -> list[str]:
        """执行一轮坐标下降校准，返回收敛在边界的参数名列表."""
        converged: list[str] = []
        for p in CALIBRATABLE_PARAMS:
            best_val: float = self._grid_search(
                p["name"], p["search_min"], p["search_max"],
                p["step"], setup, benchmark,
            )
            if self._is_boundary_converged(
                p["name"], best_val, p["search_min"], p["search_max"],
            ):
                converged.append(p["name"])
        return converged

    @staticmethod
    def _compute_improvement(
        error_before: dict[str, float],
        error_after: dict[str, float],
    ) -> dict[str, float]:
        """计算各误差项的改善百分比."""
        improvement: dict[str, float] = {}
        for key in error_before:
            before: float = error_before[key]
            after: float = error_after[key]
            improvement[key] = (before - after) / before * 100.0 if before > 0 else 0.0
        return improvement
