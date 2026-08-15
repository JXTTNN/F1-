"""F1 2026 轮胎性能曲线模型 (Iter-47).

Pirelli 2026 轮胎性能曲线: 圈速随轮胎年龄变化的精确模型.
不是简单线性退化, 而是三段式曲线:

1. **Warmup 阶段** (0..W): 新胎需暖胎, 圈速略慢.
2. **Optimal 阶段** (W..C): 最佳工作窗口, 圈速稳定 + 缓慢退化.
3. **Cliff 阶段** (C..∞): 临界点后磨耗加速, 圈速陡降.

数学模型 (基于 Pirelli 公开数据 + 车队 simulator 量级):
- Warmup: lap_time = base + warmup_penalty * (1 - lap/W)
- Optimal: lap_time = base + deg_per_lap * (lap - W)
- Cliff: lap_time = base + deg_per_lap * (C - W) + cliff_slope * (lap - C)^1.5

Pirelli 2026 化合物 (C0-C5 + intermediate + wet):
- C0 (最硬): warmup 慢, 退化极慢, cliff 晚
- C5 (最软): warmup 快, 退化快, cliff 早

数据来源: Pirelli 2026 pre-event technical notes + F1 车队 simulator 量级估计.
所有数值是车队 simulator 合理工程估计, 不代表真实车队内部数据.

公开 API:
    - :class:`TirePerformanceCurve` — 单化合物曲线.
    - :func:`lap_time_delta_s` — 圈速 delta 计算.
    - :func:`tire_curve_for` — 查询化合物曲线.
    - :func:`optimal_stint_length` — 最优 stint 长度.
    - :func:`compare_compounds` — 化合物对比.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Pirelli 2026 化合物曲线参数
# --------------------------------------------------------------------------- #
# (warmup_laps, cliff_lap, deg_per_lap_s, cliff_slope, base_offset_s)
# - warmup_laps: 暖胎圈数
# - cliff_lap: 临界圈数 (此圈后磨耗加速)
# - deg_per_lap_s: optimal 阶段每圈退化 (s)
# - cliff_slope: 临界后退化斜率 (s/lap^1.5)
# - base_offset_s: 基础圈速偏移 (相对 medium, 负=快) — post-warmup flying lap delta
#
# Iter-111 物理精度校准 (Pirelli 2026 pre-event notes + EA F1 2026 量级):
# - hard alias base_offset +0.6→+0.4: 权威 fresh flying-lap delta +0.3~+0.5s
# - soft alias base_offset -0.4: 权威 fresh flying-lap delta -0.3~-0.5s (已正确)
# - c0 +0.8: 权威 +0.6~+0.9s (已正确)
# - c5 -0.5: 权威 -0.4~-0.6s (已正确)
_TIRE_CURVE_PARAMS: dict[str, tuple[int, int, float, float, float]] = {
    # Slick compounds (C0-C5)
    "c0": (3, 35, 0.018, 0.15, 0.8),    # 最硬, 慢但持久
    "c1": (3, 32, 0.022, 0.18, 0.5),
    "c2": (2, 30, 0.028, 0.22, 0.25),   # ~ medium
    "c3": (2, 28, 0.035, 0.28, 0.0),    # medium baseline
    "c4": (2, 25, 0.045, 0.35, -0.25),  # ~ soft
    "c5": (1, 20, 0.060, 0.45, -0.5),   # 最软, 快但短命
    # 别名 (兼容 EA F1 命名)
    "hard": (3, 35, 0.020, 0.17, 0.4),     # ≈ C1, Iter-111: +0.6→+0.4 (权威 +0.3~+0.5)
    "medium": (2, 30, 0.030, 0.25, 0.0),   # ≈ C3
    "soft": (1, 22, 0.050, 0.40, -0.4),    # ≈ C5
    # 雨胎
    "intermediate": (2, 25, 0.035, 0.30, 0.5),
    "wet": (3, 30, 0.025, 0.20, 2.5),      # 雨胎基础慢
}

# Iter-111: warmup 惩罚系数 0.3→0.18. 旧版 0.3×w 给 medium 0.6s / hard 0.9s
# out-lap 暖胎惩罚, 远超真实 F1 量级 (medium ~0.3-0.4s, hard ~0.4-0.5s, soft ~0.2s).
# 系数 0.18 给: soft 0.18s, medium 0.36s, hard 0.54s — 物理可信. 不影响 reference
# state 校准 (_REF_TIRE_DELTA 同步变化, 在 reference 处相互抵消, benchmark 验证不变).
# 影响: 优化器在 age=0 边缘的 compound 选择更准; out-lap 圈速更接近真实.
_WARMUP_PEN_PER_LAP_S = 0.18


@dataclass(frozen=True)
class TirePerformanceCurve:
    """单化合物性能曲线.

    - ``warmup_laps``: 暖胎圈数 (此期间圈速略慢).
    - ``cliff_lap``: 临界圈数 (此圈后磨耗加速).
    - ``deg_per_lap_s``: optimal 阶段每圈退化 (s).
    - ``cliff_slope``: 临界后退化斜率.
    - ``base_offset_s``: 基础圈速偏移 (相对 medium baseline, 负=快).
    """

    compound: str
    warmup_laps: int
    cliff_lap: int
    deg_per_lap_s: float
    cliff_slope: float
    base_offset_s: float

    # ------------------------------------------------------------------ #
    def lap_time_delta_s(self, tire_age_laps: float) -> float:
        """计算给定轮胎年龄的圈速 delta (s).

        delta = 相对 fresh medium baseline 的额外圈速损失.
        正值 = 比基准慢.

        Args:
            tire_age_laps: 轮胎已跑圈数 (0 = 新胎).

        Returns:
            圈速 delta (s). 新胎 warmup 阶段略正, optimal 缓慢上升, cliff 陡升.
        """
        age = max(0, int(tire_age_laps))
        w = self.warmup_laps
        c = self.cliff_lap

        if age <= 0:
            # 第 0 圈: 新胎, warmup 损失最大
            return self.base_offset_s + _WARMUP_PEN_PER_LAP_S * w

        if age <= w:
            # Warmup 阶段: 线性减少 warmup 损失
            warmup_pen = _WARMUP_PEN_PER_LAP_S * w * (1 - age / w)
            return self.base_offset_s + warmup_pen

        if age <= c:
            # Optimal 阶段: 线性退化
            deg = self.deg_per_lap_s * (age - w)
            return self.base_offset_s + deg

        # Cliff 阶段: 陡降
        optimal_deg = self.deg_per_lap_s * (c - w)
        cliff_deg = self.cliff_slope * ((age - c) ** 1.5)
        return self.base_offset_s + optimal_deg + cliff_deg

    # ------------------------------------------------------------------ #
    def is_in_warmup(self, tire_age_laps: int) -> bool:
        return tire_age_laps < self.warmup_laps

    def is_in_optimal(self, tire_age_laps: int) -> bool:
        return self.warmup_laps <= tire_age_laps < self.cliff_lap

    def is_past_cliff(self, tire_age_laps: int) -> bool:
        return tire_age_laps >= self.cliff_lap

    def stage(self, tire_age_laps: int) -> str:
        """返回当前阶段: warmup / optimal / cliff."""
        if self.is_in_warmup(tire_age_laps):
            return "warmup"
        if self.is_in_optimal(tire_age_laps):
            return "optimal"
        return "cliff"

    # ------------------------------------------------------------------ #
    def optimal_stint_length(self) -> int:
        """最优 stint 长度 — cliff 前的圈数 (最大化利用 optimal 阶段)."""
        return self.cliff_lap

    def max_competitive_stint_length(self) -> int:
        """最大有竞争力 stint 长度 — cliff 后 3 圈 (临界后仍可短暂维持)."""
        return self.cliff_lap + 3

    # ------------------------------------------------------------------ #
    def total_degradation_over_stint(self, stint_laps: int) -> float:
        """整个 stint 的累计退化损失 (s)."""
        return sum(self.lap_time_delta_s(i) for i in range(stint_laps))

    def avg_lap_time_delta_s(self, stint_laps: int) -> float:
        """stint 平均圈速 delta."""
        if stint_laps <= 0:
            return 0.0
        return self.total_degradation_over_stint(stint_laps) / stint_laps


# --------------------------------------------------------------------------- #
# 曲线缓存
# --------------------------------------------------------------------------- #
_CURVE_CACHE: dict[str, TirePerformanceCurve] = {}


def tire_curve_for(compound: str) -> TirePerformanceCurve:
    """查询化合物性能曲线.

    Args:
        compound: 化合物名 (c0-c5 / hard / medium / soft / intermediate / wet).

    Returns:
        :class:`TirePerformanceCurve` 实例.

    Raises:
        ValueError: 未知化合物.
    """
    if compound in _CURVE_CACHE:
        return _CURVE_CACHE[compound]
    if compound not in _TIRE_CURVE_PARAMS:
        raise ValueError(f"Unknown compound: {compound!r}")
    w, c, deg, slope, base = _TIRE_CURVE_PARAMS[compound]
    curve = TirePerformanceCurve(
        compound=compound,
        warmup_laps=w,
        cliff_lap=c,
        deg_per_lap_s=deg,
        cliff_slope=slope,
        base_offset_s=base,
    )
    _CURVE_CACHE[compound] = curve
    return curve


def lap_time_delta_s(compound: str, tire_age_laps: float) -> float:
    """便捷函数: 计算化合物在给定年龄的圈速 delta."""
    return tire_curve_for(compound).lap_time_delta_s(tire_age_laps)


def optimal_stint_length(compound: str) -> int:
    """便捷函数: 最优 stint 长度."""
    return tire_curve_for(compound).optimal_stint_length()


def compare_compounds(compound_a: str, compound_b: str, stint_laps: int) -> dict[str, object]:
    """对比两个化合物在给定 stint 长度下的性能.

    Returns:
        包含 avg_delta、total_deg、recommendation 的字典.
    """
    ca = tire_curve_for(compound_a)
    cb = tire_curve_for(compound_b)
    avg_a = ca.avg_lap_time_delta_s(stint_laps)
    avg_b = cb.avg_lap_time_delta_s(stint_laps)

    if avg_a < avg_b:
        better = compound_a
        margin = avg_b - avg_a
    else:
        better = compound_b
        margin = avg_a - avg_b

    return {
        "compound_a": compound_a,
        "compound_b": compound_b,
        "avg_delta_a_s": avg_a,
        "avg_delta_b_s": avg_b,
        "total_deg_a_s": ca.total_degradation_over_stint(stint_laps),
        "total_deg_b_s": cb.total_degradation_over_stint(stint_laps),
        "better_compound": better,
        "margin_s": margin,
        "stint_laps": stint_laps,
    }


def all_compounds() -> list[str]:
    """所有支持的化合物名."""
    return list(_TIRE_CURVE_PARAMS.keys())


def cliff_lap_for(compound: str) -> int:
    """便捷: 查询化合物临界圈数."""
    return tire_curve_for(compound).cliff_lap
