"""F1 2026 车手信心与车队士气模型 (Iter-52).

EA Sports F1 2026 生涯模式引入"车手信心"和"车队士气"系统:

**车手信心 (Driver Confidence)** 0..100:
- 影响圈速: 高信心 → 更激进 → 更快 (但风险增加).
- 影响超车: 高信心车手更敢尝试.
- 影响一致性: 低信心 → 圈速波动大.
- 信心理由: 好成绩上升, 事故/坏成绩下降.

**车队士气 (Team Morale)** 0..100:
- 影响 R&D 效率: 高士气 → 升级完成更快.
- 影响车手信心: 高士气 → 车手信心上限提高.
- 影响可靠性: 低士气 → 机械故障概率略增.
- 士气理由: 好成绩上升, 内部矛盾/坏成绩下降.

EA F1 2026 信心/士气事件:
- 比赛胜利: 信心 +8, 士气 +6
- 领奖台: 信心 +5, 士气 +4
- 得分: 信心 +2, 士气 +2
- 退赛: 信心 -6, 士气 -4
- 事故: 信心 -4, 士气 -2
- 队友内斗: 信心 -2, 士气 -5
- 升级完成: 士气 +3
- 车手续约: 信心 +3, 士气 +2

公开 API:
    - :class:`DriverConfidence` — 车手信心状态.
    - :class:`TeamMorale` — 车队士气状态.
    - :func:`confidence_lap_time_delta_s` — 信心对圈速影响.
    - :func:`morale_rd_efficiency_factor` — 士气对 R&D 效率.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# 枚举: 事件类型
# --------------------------------------------------------------------------- #
class ConfidenceEvent(Enum):
    """影响车手信心的事件."""

    RACE_WIN = "race_win"
    PODIUM = "podium"
    POINTS = "points"
    RETIREMENT = "retirement"
    CRASH = "crash"
    TEAMMATE_BATTLE = "teammate_battle"
    GOOD_QUALIFYING = "good_qualifying"
    BAD_QUALIFYING = "bad_qualifying"
    CONTRACT_RENEWAL = "contract_renewal"


class MoraleEvent(Enum):
    """影响车队士气的事件."""

    RACE_WIN = "race_win"
    PODIUM = "podium"
    POINTS = "points"
    RETIREMENT = "retirement"
    DOUBLE_DNF = "double_dnf"          # 双退
    UPGRADE_COMPLETE = "upgrade_complete"
    TEAMMATE_CONFLICT = "teammate_conflict"
    GOOD_PR = "good_pr"                # 正面公关
    BAD_PR = "bad_pr"                  # 负面公关
    SPONSOR_SIGNING = "sponsor_signing"


# 事件影响值
_CONFIDENCE_IMPACT: dict[ConfidenceEvent, float] = {
    ConfidenceEvent.RACE_WIN: 8.0,
    ConfidenceEvent.PODIUM: 5.0,
    ConfidenceEvent.POINTS: 2.0,
    ConfidenceEvent.RETIREMENT: -6.0,
    ConfidenceEvent.CRASH: -4.0,
    ConfidenceEvent.TEAMMATE_BATTLE: -2.0,
    ConfidenceEvent.GOOD_QUALIFYING: 3.0,
    ConfidenceEvent.BAD_QUALIFYING: -2.0,
    ConfidenceEvent.CONTRACT_RENEWAL: 3.0,
}

_MORALE_IMPACT: dict[MoraleEvent, float] = {
    MoraleEvent.RACE_WIN: 6.0,
    MoraleEvent.PODIUM: 4.0,
    MoraleEvent.POINTS: 2.0,
    MoraleEvent.RETIREMENT: -4.0,
    MoraleEvent.DOUBLE_DNF: -8.0,
    MoraleEvent.UPGRADE_COMPLETE: 3.0,
    MoraleEvent.TEAMMATE_CONFLICT: -5.0,
    MoraleEvent.GOOD_PR: 2.0,
    MoraleEvent.BAD_PR: -3.0,
    MoraleEvent.SPONSOR_SIGNING: 2.0,
}


# --------------------------------------------------------------------------- #
# DriverConfidence
# --------------------------------------------------------------------------- #
@dataclass
class DriverConfidence:
    """车手信心状态 (0..100).

    - ``value``: 当前信心值.
    - ``history``: 信心变化历史 [(event, delta, new_value)].
    """

    driver_id: str
    value: float = 50.0
    history: list[tuple[str, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.value = max(0.0, min(100.0, self.value))

    def apply_event(self, event: ConfidenceEvent) -> float:
        """应用事件, 返回信心变化."""
        delta = _CONFIDENCE_IMPACT[event]
        old = self.value
        self.value = max(0.0, min(100.0, self.value + delta))
        actual_delta = self.value - old
        self.history.append((event.value, actual_delta, self.value))
        return actual_delta

    def decay_toward_baseline(self, baseline: float = 50.0, rate: float = 0.05) -> None:
        """信心向基线缓慢回归 (每场比赛)."""
        self.value += (baseline - self.value) * rate
        self.value = max(0.0, min(100.0, self.value))

    @property
    def level(self) -> str:
        """信心等级: low / medium / high / very_high."""
        if self.value < 25:
            return "low"
        if self.value <= 50:
            return "medium"
        if self.value <= 75:
            return "high"
        return "very_high"

    @property
    def lap_time_delta_s(self) -> float:
        """信心对圈速影响 (s). 高信心 → 负 (更快)."""
        return confidence_lap_time_delta_s(self.value)


def confidence_lap_time_delta_s(confidence: float) -> float:
    """信心值 → 圈速 delta (s).

    - 100 信心: -0.30 s (最快)
    - 50 信心: 0.0 s (基准)
    - 0 信心: +0.40 s (最慢)

    线性插值, 但低信心端略陡 (信心低时影响更大).
    """
    c = max(0.0, min(100.0, confidence))
    if c >= 50:
        # 50..100: 0 → -0.30
        return -0.30 * (c - 50) / 50
    # 0..50: +0.40 → 0
    return 0.40 * (50 - c) / 50


# --------------------------------------------------------------------------- #
# TeamMorale
# --------------------------------------------------------------------------- #
@dataclass
class TeamMorale:
    """车队士气状态 (0..100)."""

    team_id: str
    value: float = 60.0
    history: list[tuple[str, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.value = max(0.0, min(100.0, self.value))

    def apply_event(self, event: MoraleEvent) -> float:
        """应用事件, 返回士气变化."""
        delta = _MORALE_IMPACT[event]
        old = self.value
        self.value = max(0.0, min(100.0, self.value + delta))
        actual_delta = self.value - old
        self.history.append((event.value, actual_delta, self.value))
        return actual_delta

    def decay_toward_baseline(self, baseline: float = 60.0, rate: float = 0.03) -> None:
        """士气向基线缓慢回归."""
        self.value += (baseline - self.value) * rate
        self.value = max(0.0, min(100.0, self.value))

    @property
    def level(self) -> str:
        if self.value < 25:
            return "critical"
        if self.value < 50:
            return "low"
        if self.value < 75:
            return "good"
        return "excellent"

    @property
    def rd_efficiency_factor(self) -> float:
        """士气对 R&D 效率系数 (1.0 = 标准)."""
        return morale_rd_efficiency_factor(self.value)

    @property
    def reliability_factor(self) -> float:
        """士气对可靠性系数 (1.0 = 标准, 低士气略降)."""
        if self.value >= 50:
            return 1.0
        # 低士气降低可靠性
        return 1.0 - (50 - self.value) / 100 * 0.05  # 最多降 2.5%


def morale_rd_efficiency_factor(morale: float) -> float:
    """士气值 → R&D 效率系数.

    - 100 士气: 1.20 (升级快 20%)
    - 60 士气: 1.0 (基准)
    - 0 士气: 0.70 (升级慢 30%)
    """
    m = max(0.0, min(100.0, morale))
    if m >= 60:
        # 60..100: 1.0 → 1.20
        return 1.0 + 0.20 * (m - 60) / 40
    # 0..60: 0.70 → 1.0
    return 0.70 + 0.30 * m / 60


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def apply_race_result(
    confidence: DriverConfidence,
    morale: TeamMorale,
    position: int,
    retired: bool = False,
) -> None:
    """根据比赛结果更新信心与士气.

    Args:
        confidence: 车手信心.
        morale: 车队士气.
        position: 完赛名次 (1=胜), 退赛时忽略.
        retired: 是否退赛.
    """
    if retired:
        confidence.apply_event(ConfidenceEvent.RETIREMENT)
        morale.apply_event(MoraleEvent.RETIREMENT)
        return

    if position == 1:
        confidence.apply_event(ConfidenceEvent.RACE_WIN)
        morale.apply_event(MoraleEvent.RACE_WIN)
    elif position <= 3:
        confidence.apply_event(ConfidenceEvent.PODIUM)
        morale.apply_event(MoraleEvent.PODIUM)
    elif position <= 10:
        confidence.apply_event(ConfidenceEvent.POINTS)
        morale.apply_event(MoraleEvent.POINTS)
    # 11+ 不变


def apply_qualifying_result(
    confidence: DriverConfidence,
    qualifying_position: int,
    total_drivers: int = 20,
) -> None:
    """根据排位结果更新信心."""
    # 前 1/3 为好, 后 1/3 为差
    if qualifying_position <= total_drivers / 3:
        confidence.apply_event(ConfidenceEvent.GOOD_QUALIFYING)
    elif qualifying_position > total_drivers * 2 / 3:
        confidence.apply_event(ConfidenceEvent.BAD_QUALIFYING)
