"""JSON 遥测导入器 — 从 F1TelemetryCollector 导出的 JSON 文件加载圈遥测数据。

解析 ``D:\\F1TelemetryCollector\\dist\\data`` 下的单圈 JSON 文件，
计算整圈统计摘要（平均速度/最大速度/平均油门/平均刹车/胎温均值/胎压均值等），
并转换为 engine._derive_telemetry_dx / _derive_telemetry_gain 可用的遥测字典。

关键发现：
- setup 中的胎压字段为科学记数法（UDP 原始异常值），必须从 samples 中取胎压；
- setup 中的调教参数为 UDP 原始值（未做值域转换）；
- G 力数据全部为 0，importer 不依赖 G 力数据。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# LapTelemetrySummary dataclass
# --------------------------------------------------------------------------- #
@dataclass
class LapTelemetrySummary:
    """整圈遥测统计摘要。

    包含圈基本信息、天气/轮胎/燃油/驾驶风格等元数据，
    以及从 samples 逐帧数据计算出的统计量（均值/极值）。
    """

    # 圈基本信息
    lap_number: int
    lap_time_ms: int
    lap_time_str: str
    track_id: int
    track_name: str
    lap_valid: bool
    sample_count: int
    sector_times_ms: list[int]

    # 天气
    weather_code: int
    weather_name: str
    track_temp: float
    air_temp: float

    # 轮胎
    tyre_compound: str
    tyre_age_laps: int

    # 速度统计 (km/h)
    avg_speed: float
    max_speed: float

    # 油门/刹车/转向统计 (0-1)
    avg_throttle: float
    avg_brake: float
    max_brake: float
    avg_steer: float  # 平均转向绝对值
    max_steer: float  # 最大转向绝对值

    # 四轮温度/胎压统计（官方车轮顺序 [RL, RR, FL, FR]）
    avg_tyre_surface_temp: list[float]
    avg_tyre_inner_temp: list[float]
    avg_brake_temp: list[float]
    avg_tyre_pressure: list[float]  # 从 samples 取！
    max_tyre_surface_temp: list[float]
    max_brake_temp: list[float]
    min_tyre_pressure: list[float]
    max_tyre_pressure: list[float]

    # 调教/燃油/驾驶风格
    setup_raw: dict[str, float]
    fuel_load_kg: float
    fuel_in_tank_kg: float
    driver_style: dict[str, Any]

    def to_telemetry_dict(self) -> dict[str, Any]:
        """转换为 engine._derive_telemetry_dx / _derive_telemetry_gain 可用的遥测字典。

        输出字段名与 engine.py 中 _derive_telemetry_dx / _derive_telemetry_gain
        所读取的 key 保持一致。
        """
        return {
            "m_tyresSurfaceTemperature": self.avg_tyre_surface_temp,
            "m_tyresInnerTemperature": self.avg_tyre_inner_temp,
            "m_brakesTemperature": self.avg_brake_temp,
            "m_tyresPressure": self.avg_tyre_pressure,
            "m_throttle": self.avg_throttle,
            "m_brake": self.avg_brake,
            "m_speed": self.avg_speed,
            "weather": self.weather_name,
            "m_weather": self.weather_code,
            "track_temp": self.track_temp,
            "air_temp": self.air_temp,
            "tyre_compound": self.tyre_compound,
            "sector": 0,  # 暂用 0（整圈摘要无扇区区分）
            "lap_distance": 0.0,  # 暂用 0（整圈摘要无单帧距离）
            # task-104 扩展：转向与速度极值统计（供 _derive_telemetry_dx 新规则使用）
            "avg_steer": self.avg_steer,
            "max_steer": self.max_steer,
            "max_speed": self.max_speed,
        }


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def _avg_or_zero(values: list[float]) -> float:
    """计算均值，空列表返回 0.0。"""
    return statistics.fmean(values) if values else 0.0


def _max_or_zero(values: list[float]) -> float:
    """计算最大值，空列表返回 0.0。"""
    return max(values) if values else 0.0


def _min_or_zero(values: list[float]) -> float:
    """计算最小值，空列表返回 0.0。"""
    return min(values) if values else 0.0


def _abs_avg(values: list[float]) -> float:
    """计算绝对值均值，空列表返回 0.0。"""
    return statistics.fmean(abs(v) for v in values) if values else 0.0


def _abs_max(values: list[float]) -> float:
    """计算绝对值最大值，空列表返回 0.0。"""
    return max((abs(v) for v in values), default=0.0)


def _per_wheel_avg(samples: list[dict[str, Any]], key: str) -> list[float]:
    """从 samples 中提取四轮列表字段，计算每轮均值（保持官方顺序 [RL, RR, FL, FR]）。"""
    if not samples:
        return [0.0, 0.0, 0.0, 0.0]
    # 收集四轮各自的所有帧值
    wheel_data: list[list[float]] = [[] for _ in range(4)]
    for s in samples:
        vals = s.get(key)
        if isinstance(vals, list) and len(vals) >= 4:
            for i in range(4):
                wheel_data[i].append(float(vals[i]))
    return [_avg_or_zero(w) for w in wheel_data]


def _per_wheel_max(samples: list[dict[str, Any]], key: str) -> list[float]:
    """从 samples 中提取四轮列表字段，计算每轮最大值（保持官方顺序 [RL, RR, FL, FR]）。"""
    if not samples:
        return [0.0, 0.0, 0.0, 0.0]
    wheel_data: list[list[float]] = [[] for _ in range(4)]
    for s in samples:
        vals = s.get(key)
        if isinstance(vals, list) and len(vals) >= 4:
            for i in range(4):
                wheel_data[i].append(float(vals[i]))
    return [_max_or_zero(w) for w in wheel_data]


def _per_wheel_min(samples: list[dict[str, Any]], key: str) -> list[float]:
    """从 samples 中提取四轮列表字段，计算每轮最小值（保持官方顺序 [RL, RR, FL, FR]）。"""
    if not samples:
        return [0.0, 0.0, 0.0, 0.0]
    wheel_data: list[list[float]] = [[] for _ in range(4)]
    for s in samples:
        vals = s.get(key)
        if isinstance(vals, list) and len(vals) >= 4:
            for i in range(4):
                wheel_data[i].append(float(vals[i]))
    return [_min_or_zero(w) for w in wheel_data]


# --------------------------------------------------------------------------- #
# 核心导入函数
# --------------------------------------------------------------------------- #
def _extract_scalar_fields(
    samples: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """从 samples 中提取标量字段（速度/油门/刹车/转向）。

    Returns:
        包含 ``speeds``/``throttles``/``brakes``/``steers`` 四个列表的字典。
    """
    return {
        "speeds": [float(s.get("speed", 0)) for s in samples],
        "throttles": [float(s.get("throttle", 0.0)) for s in samples],
        "brakes": [float(s.get("brake", 0.0)) for s in samples],
        "steers": [float(s.get("steer", 0.0)) for s in samples],
    }


def _compute_wheel_stats(
    samples: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """从 samples 计算四轮温度/胎压统计（官方车轮顺序 [RL, RR, FL, FR]）。

    胎压从 samples 取（setup 中为科学记数法异常值）。

    Returns:
        包含 8 个四轮列表的字典（avg/max/min 各项）。
    """
    return {
        "avg_tyre_surface": _per_wheel_avg(samples, "tyre_surface_temp"),
        "avg_tyre_inner": _per_wheel_avg(samples, "tyre_inner_temp"),
        "avg_brake_temp": _per_wheel_avg(samples, "brake_temp"),
        "avg_tyre_pressure": _per_wheel_avg(samples, "tyre_pressure"),
        "max_tyre_surface": _per_wheel_max(samples, "tyre_surface_temp"),
        "max_brake_temp": _per_wheel_max(samples, "brake_temp"),
        "min_tyre_pressure": _per_wheel_min(samples, "tyre_pressure"),
        "max_tyre_pressure": _per_wheel_max(samples, "tyre_pressure"),
    }


def import_lap_json(filepath: str) -> LapTelemetrySummary:
    """从 JSON 文件路径导入单圈遥测数据并计算统计摘要。

    Args:
        filepath: JSON 文件绝对路径。

    Returns:
        :class:`LapTelemetrySummary` 整圈统计摘要。

    Raises:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: JSON 格式错误。
        KeyError: 缺少必需字段。
    """
    path = Path(filepath)
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    samples: list[dict[str, Any]] = data.get("samples", [])
    weather: dict[str, Any] = data.get("weather", {})
    tyre: dict[str, Any] = data.get("tyre", {})
    fuel: dict[str, Any] = data.get("fuel", {})
    setup: dict[str, Any] = data.get("setup", {})
    driver_style: dict[str, Any] = data.get("driver_style", {})

    scalars = _extract_scalar_fields(samples)
    wheels = _compute_wheel_stats(samples)

    return LapTelemetrySummary(
        lap_number=int(data.get("lap_number", 0)),
        lap_time_ms=int(data.get("lap_time_ms", 0)),
        lap_time_str=str(data.get("lap_time_str", "")),
        track_id=int(data.get("track_id", 0)),
        track_name=str(data.get("track_name", "")),
        lap_valid=bool(data.get("lap_valid", False)),
        sample_count=int(data.get("sample_count", len(samples))),
        sector_times_ms=list(data.get("sector_times_ms", [])),
        weather_code=int(weather.get("code", 0)),
        weather_name=str(weather.get("name", "")),
        track_temp=float(weather.get("track_temp", 0)),
        air_temp=float(weather.get("air_temp", 0)),
        tyre_compound=str(tyre.get("compound_name", "")),
        tyre_age_laps=int(tyre.get("age_laps", 0)),
        avg_speed=_avg_or_zero(scalars["speeds"]),
        max_speed=_max_or_zero(scalars["speeds"]),
        avg_throttle=_avg_or_zero(scalars["throttles"]),
        avg_brake=_avg_or_zero(scalars["brakes"]),
        max_brake=_max_or_zero(scalars["brakes"]),
        avg_steer=_abs_avg(scalars["steers"]),
        max_steer=_abs_max(scalars["steers"]),
        avg_tyre_surface_temp=wheels["avg_tyre_surface"],
        avg_tyre_inner_temp=wheels["avg_tyre_inner"],
        avg_brake_temp=wheels["avg_brake_temp"],
        avg_tyre_pressure=wheels["avg_tyre_pressure"],
        max_tyre_surface_temp=wheels["max_tyre_surface"],
        max_brake_temp=wheels["max_brake_temp"],
        min_tyre_pressure=wheels["min_tyre_pressure"],
        max_tyre_pressure=wheels["max_tyre_pressure"],
        setup_raw={k: float(v) for k, v in setup.items()},
        fuel_load_kg=float(fuel.get("load_kg", 0.0)),
        fuel_in_tank_kg=float(fuel.get("in_tank_kg", 0.0)),
        driver_style=dict(driver_style),
    )


def import_laps_from_directory(dirpath: str) -> list[LapTelemetrySummary]:
    """从目录导入所有圈遥测数据（按 lap_number 排序）。

    扫描目录下所有 ``lap_*.json`` 文件，逐个解析为 :class:`LapTelemetrySummary`，
    结果按 ``lap_number`` 升序排列。

    Args:
        dirpath: 包含圈 JSON 文件的目录路径。

    Returns:
        按 lap_number 排序的 :class:`LapTelemetrySummary` 列表。
    """
    directory = Path(dirpath)
    json_files = sorted(directory.glob("lap_*.json"))
    summaries: list[LapTelemetrySummary] = []
    for jf in json_files:
        try:
            summary = import_lap_json(str(jf))
            summaries.append(summary)
        except (json.JSONDecodeError, KeyError):
            # 跳过损坏的 JSON 文件，不中断整体导入
            continue
    summaries.sort(key=lambda s: s.lap_number)
    return summaries