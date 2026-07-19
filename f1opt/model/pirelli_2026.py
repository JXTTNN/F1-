"""F1 2026 — Pirelli 轮胎规格 (Iter-28).

FIA 2026 Pirelli 轮胎规格 (与 F1 2026 EA Sports 游戏一致):

1. **干地 slick 范围**: C0-C5 (6 种, 从硬到软).
   - C0 = 最硬 (Spa/Monza 等高速低磨蚀)
   - C5 = 最软 (Monaco/Singapore 等低速高 grip)
2. **每场选定 3 种**: Pirelli 每场从 6 种中选 3 种作为该场 "soft/medium/hard"
   颜色分配 (例如 Monaco 用 C3-C4-C5, Monza 用 C0-C1-C2).
3. **2026 新增 wet 干湿两用**: 全新设计, 不需暖胎, 即装即用.
4. **intermediate 重新设计**: 工作窗口更宽, crossover 更精准.
5. **颜色规则**: 红=soft (该场最软), 黄=medium (中), 白=hard (该场最硬).

公开 API:
    - :class:`Pirelli2026Range` — 单场 Pirelli 选胎方案.
    - :func:`tire_compound_for_track` — 按赛道返回 Pirelli 选定 C-range.
"""

from __future__ import annotations

from dataclasses import dataclass


# Pirelli 2026 化合物物理参数 (6 slick + intermediate + wet)
# 单位: warmup_laps, steady_rate_s/lap, cliff_threshold_pct, cliff_rate_s
@dataclass(frozen=True)
class PirelliCompound2026:
    """Pirelli 2026 化合物参数."""

    code: str  # C0-C5 / intermediate / wet
    warmup_laps: float
    warmup_penalty_s: float
    steady_rate_s: float
    cliff_threshold_pct: float
    cliff_rate_s: float
    temp_optimal_c: float
    temp_window_c: float
    grip_factor: float
    """相对 C3=1.0 的抓地力系数. C5 高, C0 低."""
    wear_rate_per_lap: float
    """基础磨损 % / lap, 不含赛道磨蚀."""


# Pirelli 2026 6 slick 化合物 + 2 雨胎
_PIRELLI_2026_RANGE: dict[str, PirelliCompound2026] = {
    "C0": PirelliCompound2026(
        code="C0", warmup_laps=3.0, warmup_penalty_s=0.80,
        steady_rate_s=0.022, cliff_threshold_pct=88.0, cliff_rate_s=0.8,
        temp_optimal_c=110.0, temp_window_c=18.0,
        grip_factor=0.92, wear_rate_per_lap=2.2,
    ),
    "C1": PirelliCompound2026(
        code="C1", warmup_laps=2.8, warmup_penalty_s=0.75,
        steady_rate_s=0.028, cliff_threshold_pct=85.0, cliff_rate_s=0.9,
        temp_optimal_c=107.0, temp_window_c=17.0,
        grip_factor=0.95, wear_rate_per_lap=2.5,
    ),
    "C2": PirelliCompound2026(
        code="C2", warmup_laps=2.5, warmup_penalty_s=0.65,
        steady_rate_s=0.035, cliff_threshold_pct=82.0, cliff_rate_s=1.0,
        temp_optimal_c=105.0, temp_window_c=16.0,
        grip_factor=0.98, wear_rate_per_lap=2.9,
    ),
    "C3": PirelliCompound2026(
        code="C3", warmup_laps=2.2, warmup_penalty_s=0.60,
        steady_rate_s=0.045, cliff_threshold_pct=78.0, cliff_rate_s=1.1,
        temp_optimal_c=100.0, temp_window_c=15.0,
        grip_factor=1.00, wear_rate_per_lap=3.4,
    ),
    "C4": PirelliCompound2026(
        code="C4", warmup_laps=1.8, warmup_penalty_s=0.50,
        steady_rate_s=0.060, cliff_threshold_pct=72.0, cliff_rate_s=1.4,
        temp_optimal_c=95.0, temp_window_c=13.0,
        grip_factor=1.04, wear_rate_per_lap=4.2,
    ),
    "C5": PirelliCompound2026(
        code="C5", warmup_laps=1.5, warmup_penalty_s=0.45,
        steady_rate_s=0.075, cliff_threshold_pct=66.0, cliff_rate_s=1.7,
        temp_optimal_c=92.0, temp_window_c=12.0,
        grip_factor=1.08, wear_rate_per_lap=5.2,
    ),
    "intermediate": PirelliCompound2026(
        code="intermediate", warmup_laps=1.0, warmup_penalty_s=0.30,
        steady_rate_s=0.10, cliff_threshold_pct=55.0, cliff_rate_s=2.0,
        temp_optimal_c=70.0, temp_window_c=15.0,
        grip_factor=0.78, wear_rate_per_lap=6.5,
    ),
    "wet": PirelliCompound2026(
        code="wet", warmup_laps=0.8, warmup_penalty_s=0.25,
        steady_rate_s=0.06, cliff_threshold_pct=80.0, cliff_rate_s=1.0,
        temp_optimal_c=55.0, temp_window_c=20.0,
        grip_factor=0.65, wear_rate_per_lap=3.0,
    ),
}


@dataclass(frozen=True)
class Pirelli2026Range:
    """单场 Pirelli 选胎方案.

    Attributes:
        track_id: 赛道 id.
        soft_code: 该场红胎对应的 C-code (C0-C5).
        medium_code: 该场黄胎.
        hard_code: 该场白胎.
    """

    track_id: str
    soft_code: str
    medium_code: str
    hard_code: str

    def compound_for_color(self, color: str) -> PirelliCompound2026:
        """按颜色 (soft/medium/hard) 返回实际 C-code 参数."""
        if color == "soft":
            return _PIRELLI_2026_RANGE[self.soft_code]
        if color == "medium":
            return _PIRELLI_2026_RANGE[self.medium_code]
        if color == "hard":
            return _PIRELLI_2026_RANGE[self.hard_code]
        if color in ("intermediate", "wet"):
            return _PIRELLI_2026_RANGE[color]
        raise ValueError(f"Unknown color/compound: {color!r}")

    def all_colors(self) -> dict[str, PirelliCompound2026]:
        """返回该场三色胎参数."""
        return {
            "soft": self.compound_for_color("soft"),
            "medium": self.compound_for_color("medium"),
            "hard": self.compound_for_color("hard"),
        }


# Pirelli 2026 各场选胎方案 (基于 Pirelli 公开 pre-event notes 估算)
# 不同赛道选用 C0-C5 中的 3 种作为该场 soft/medium/hard
_TRACK_TIRE_SELECTION: dict[str, tuple[str, str, str]] = {
    # (soft, medium, hard) — 该场实际 C-code
    # 高速低磨蚀赛道用最硬 C0-C2
    "monza": ("C2", "C1", "C0"),
    "spa": ("C3", "C2", "C1"),
    "baku": ("C3", "C2", "C1"),
    "jeddah": ("C3", "C2", "C1"),
    "las_vegas": ("C3", "C2", "C1"),
    # 高磨蚀赛道用中等 C2-C4
    "silverstone": ("C4", "C3", "C2"),
    "suzuka": ("C4", "C3", "C2"),
    "barcelona": ("C4", "C3", "C2"),
    "austin": ("C4", "C3", "C2"),
    "shanghai": ("C4", "C3", "C2"),
    # 中等赛道用 C2-C4
    "melbourne": ("C4", "C3", "C2"),
    "sakhir": ("C4", "C3", "C2"),
    "miami": ("C4", "C3", "C2"),
    "montreal": ("C4", "C3", "C2"),
    "sao_paulo": ("C4", "C3", "C2"),
    "lusail": ("C4", "C3", "C2"),
    "yas_marina": ("C4", "C3", "C2"),
    "madrid": ("C4", "C3", "C2"),
    "spielberg": ("C4", "C3", "C2"),
    "mexico_city": ("C4", "C3", "C2"),
    # 低速高 grip 用最软 C3-C5
    "monaco": ("C5", "C4", "C3"),
    "singapore": ("C5", "C4", "C3"),
    "zandvoort": ("C5", "C4", "C3"),
    "hungaroring": ("C5", "C4", "C3"),
}


def tire_compound_for_track(track_id: str) -> Pirelli2026Range:
    """按赛道返回 Pirelli 2026 选胎方案."""
    selection = _TRACK_TIRE_SELECTION.get(track_id)
    if selection is None:
        # 默认中等
        selection = ("C4", "C3", "C2")
    return Pirelli2026Range(
        track_id=track_id,
        soft_code=selection[0],
        medium_code=selection[1],
        hard_code=selection[2],
    )


def all_pirelli_compounds() -> dict[str, PirelliCompound2026]:
    """返回 Pirelli 2026 全部 8 种化合物参数."""
    return dict(_PIRELLI_2026_RANGE)


def pirelli_compound(code: str) -> PirelliCompound2026:
    """按 code 返回化合物参数."""
    if code not in _PIRELLI_2026_RANGE:
        raise ValueError(f"Unknown Pirelli compound code: {code!r}")
    return _PIRELLI_2026_RANGE[code]
