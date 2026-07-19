"""F1 天气与赛道湿润模型 (Iter-12).

真实 F1 雨战是最复杂的工程场景之一 — 轮胎选择 "crossover" (干胎↔半雨胎↓全雨胎
的切换点) 决定圈速差可达 5-20 s. Pirelli 公开数据 + F1 车队工程实践:

- **雨强** (mm/h): 0 干, <2.5 小雨, 2.5-10 中雨, >10 大雨.
- **赛道湿润度** ``track_wetness`` (0-1): 随降雨累积, 随赛道温度 + 时间衰减.
- **Crossover 点** (Pirelli 公开):
    - slick → intermediate: wetness ≈ 0.30
    - intermediate → wet:    wetness ≈ 0.65
    - wet → 极端 (aquaplaning): wetness > 0.90
- **圈速惩罚**: 错误轮胎在错误湿润度下损失巨大 (slick on 0.6 wet = +15 s/lap).
- **能见度**: 前车 spray 拖慢后车 (跟车损失放大).
- **赛道温度**: 降雨时下降, 影响胎温窗口.

公开 API:
    - :class:`WeatherState` — 单点天气状态.
    - :class:`WeatherModel` — 时间演化 + 圈速惩罚 + 轮胎推荐.
    - :data:`COMPOUND_CROSSOVER` — Pirelli crossover 湿润度阈值.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# 物理常量 (Pirelli 公开数据 + F1 工程估算)
# --------------------------------------------------------------------------- #
# 雨强分级 (mm/h)
_RAIN_LIGHT = 2.5       # 小雨
_RAIN_MODERATE = 10.0   # 中雨
_RAIN_HEAVY = 25.0      # 大雨

# 湿润度动力学
_WETNESS_ACCUM_RATE = 0.020   # 每 mm/h 雨每分钟湿润度增量
_WETNESS_DRY_RATE = 0.015     # 干燥时每分钟 (赛道温度驱动) 衰减
_WETNESS_DRY_TEMP_FACTOR = 0.4  # 赛道温度对干燥的放大 (每 10°C above 30)

# Crossover 湿润度阈值 (Pirelli 公开工程数据)
COMPOUND_CROSSOVER: dict[str, tuple[float, float]] = {
    # compound: (optimal_wetness_low, optimal_wetness_high)
    "soft": (0.0, 0.15),
    "medium": (0.0, 0.18),
    "hard": (0.0, 0.20),
    "intermediate": (0.20, 0.65),
    "wet": (0.60, 1.00),
}

# 圈速惩罚系数 (秒, 在最优窗外每 0.1 wetness 偏离)
_PENALTY_PER_0_1_WETNESS = {
    "soft": 2.5,         # 干胎在湿地极快退化 (aquaplaning)
    "medium": 2.3,
    "hard": 2.1,
    "intermediate": 1.2, # 半雨胎窗外较宽
    "wet": 0.8,          # 全雨胎窗外最宽
}

# 最优窗内仍有基础惩罚 (相对干地圈速, 秒) — 雨地永远慢于干地
_BASE_PENALTY_IN_WINDOW = {
    "soft": 0.0,
    "medium": 0.0,
    "hard": 0.0,
    "intermediate": 1.8,   # 半雨胎最优仍慢 1.8s
    "wet": 4.5,            # 全雨胎最优慢 4.5s
}

# 能见度 (spray) 跟车损失放大
_SPRAY_FOLLOW_FACTOR = 2.5  # wetness=1 时跟车损失 ×3.5

# 赛道温度: 降雨每 mm/h 降低赛道温度 (°C), 上限
_TRACK_TEMP_DROP_PER_RAIN_MMH = 0.4
_TRACK_TEMP_MIN = 12.0


# --------------------------------------------------------------------------- #
# WeatherState
# --------------------------------------------------------------------------- #
@dataclass
class WeatherState:
    """单点天气状态."""

    rain_intensity_mmh: float = 0.0   # 雨强 (mm/h)
    track_wetness: float = 0.0        # 赛道湿润度 (0-1)
    track_temp_c: float = 35.0        # 赛道温度 (°C)
    ambient_temp_c: float = 25.0      # 环境温度 (°C)
    wind_speed_ms: float = 5.0        # 风速 (m/s)
    wind_dir_deg: float = 0.0         # 风向 (度, 0=顺时针)

    @property
    def is_dry(self) -> bool:
        return self.rain_intensity_mmh < 0.1 and self.track_wetness < 0.15

    @property
    def rain_category(self) -> str:
        r = self.rain_intensity_mmh
        if r < 0.1:
            return "dry"
        if r < _RAIN_LIGHT:
            return "drizzle"
        if r < _RAIN_MODERATE:
            return "light"
        if r < _RAIN_HEAVY:
            return "moderate"
        return "heavy"


# --------------------------------------------------------------------------- #
# WeatherModel
# --------------------------------------------------------------------------- #
@dataclass
class WeatherModel:
    """天气时间演化 + 圈速惩罚 + 轮胎推荐.

    用法::

        wm = WeatherModel(initial=WeatherState(track_temp_c=30.0))
        # 模拟一场雨
        wm.step(rain_mmh=8.0, minutes=5.0)
        penalty = wm.lap_time_penalty(compound="intermediate")
        rec = wm.recommend_compound()
    """

    initial: WeatherState = field(default_factory=WeatherState)
    state: WeatherState = field(init=False)

    def __post_init__(self) -> None:
        self.state = WeatherState(
            rain_intensity_mmh=self.initial.rain_intensity_mmh,
            track_wetness=self.initial.track_wetness,
            track_temp_c=self.initial.track_temp_c,
            ambient_temp_c=self.initial.ambient_temp_c,
            wind_speed_ms=self.initial.wind_speed_ms,
            wind_dir_deg=self.initial.wind_dir_deg,
        )

    # ------------------------------------------------------------------ #
    def step(self, rain_mmh: float, minutes: float) -> WeatherState:
        """推进 ``minutes`` 分钟, 给定雨强, 更新湿润度与赛道温度.

        - 湿润度: 降雨累积上升, 无雨时按赛道温度衰减.
        - 赛道温度: 降雨时下降 (蒸发冷却), 无雨时缓慢回升.
        """
        if minutes <= 0:
            return self.state
        # 湿润度变化
        if rain_mmh > 0.0:
            accum = _WETNESS_ACCUM_RATE * rain_mmh * minutes
            self.state.track_wetness = min(1.0, self.state.track_wetness + accum)
        else:
            # 干燥: 赛道温度越高干燥越快
            temp_factor = 1.0 + _WETNESS_DRY_TEMP_FACTOR * max(
                0.0, (self.state.track_temp_c - 30.0) / 10.0
            )
            dry = _WETNESS_DRY_RATE * temp_factor * minutes
            self.state.track_wetness = max(0.0, self.state.track_wetness - dry)
        # 赛道温度变化
        if rain_mmh > 0.0:
            drop = _TRACK_TEMP_DROP_PER_RAIN_MMH * rain_mmh * (minutes / 60.0)
            self.state.track_temp_c = max(
                _TRACK_TEMP_MIN, self.state.track_temp_c - drop
            )
        else:
            # 无雨缓慢回升向环境温度
            self.state.track_temp_c += 0.5 * (minutes / 60.0) * (
                self.state.ambient_temp_c + 5.0 - self.state.track_temp_c
            )
        self.state.rain_intensity_mmh = float(rain_mmh)
        return self.state

    # ------------------------------------------------------------------ #
    def lap_time_penalty(self, compound: str, lap_time_s: float = 90.0) -> float:
        """返回当前湿润度下该 compound 的圈速惩罚 (秒, ≥0).

        错误轮胎在错误湿润度下惩罚巨大; 最优窗内仅有基础雨地惩罚.
        """
        w = self.state.track_wetness
        window = COMPOUND_CROSSOVER.get(compound)
        if window is None:
            return 0.0
        low, high = window
        # 基础雨地惩罚 (在窗内)
        penalty = _BASE_PENALTY_IN_WINDOW.get(compound, 0.0)
        # 窗外惩罚: 越偏离最优窗惩罚越大
        per_0_1 = _PENALTY_PER_0_1_WETNESS.get(compound, 1.5)
        if w < low:
            deviation = (low - w) / 0.1
            penalty += deviation * per_0_1
        elif w > high:
            deviation = (w - high) / 0.1
            # 干胎在高湿润度下触发 aquaplaning (额外指数惩罚)
            if compound in ("soft", "medium", "hard") and w > 0.4:
                penalty += deviation * per_0_1 * 2.5  # aquaplaning
            elif compound == "intermediate" and w > 0.75:
                # 半雨胎在极湿区也触发 aquaplaning (Pirelli: inters 极限 ~0.70)
                penalty += deviation * per_0_1 * 2.0
            else:
                penalty += deviation * per_0_1
        return float(max(0.0, penalty))

    # ------------------------------------------------------------------ #
    def recommend_compound(self) -> str:
        """基于当前湿润度推荐最优 compound (最低圈速惩罚).

        工程方法: 取所有候选 compound 中惩罚最低者; 干地返回 medium (平衡).
        """
        w = self.state.track_wetness
        if w < 0.15:
            return "medium"
        # 取最低惩罚 compound (真实车队 crossover 决策)
        candidates = ("soft", "medium", "hard", "intermediate", "wet")
        best = min(candidates, key=lambda c: self.lap_time_penalty(c))
        # 干地附近若 slick 与 medium 持平, 偏好 medium (耐用)
        if w < 0.20 and best in ("soft", "hard"):
            return "medium"
        return best

    # ------------------------------------------------------------------ #
    def follow_loss_factor(self) -> float:
        """跟车损失放大因子 (spray 降低能见度).

        干地 = 1.0 (无放大), 全湿 = _SPRAY_FOLLOW_FACTOR+1.
        """
        w = self.state.track_wetness
        return 1.0 + w * _SPRAY_FOLLOW_FACTOR

    # ------------------------------------------------------------------ #
    def visibility_score(self) -> float:
        """能见度评分 (0-1, 1=最佳). 全湿 ~0.2."""
        w = self.state.track_wetness
        return float(max(0.1, 1.0 - w * 0.8))

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """返回当前天气摘要."""
        return {
            "rain_intensity_mmh": round(self.state.rain_intensity_mmh, 2),
            "rain_category": self.state.rain_category,
            "track_wetness": round(self.state.track_wetness, 3),
            "track_temp_c": round(self.state.track_temp_c, 1),
            "ambient_temp_c": round(self.state.ambient_temp_c, 1),
            "wind_speed_ms": round(self.state.wind_speed_ms, 1),
            "is_dry": self.state.is_dry,
            "recommended_compound": self.recommend_compound(),
            "visibility": round(self.visibility_score(), 2),
            "follow_loss_factor": round(self.follow_loss_factor(), 2),
        }

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """重置到 initial 状态."""
        self.state = WeatherState(
            rain_intensity_mmh=self.initial.rain_intensity_mmh,
            track_wetness=self.initial.track_wetness,
            track_temp_c=self.initial.track_temp_c,
            ambient_temp_c=self.initial.ambient_temp_c,
            wind_speed_ms=self.initial.wind_speed_ms,
            wind_dir_deg=self.initial.wind_dir_deg,
        )
