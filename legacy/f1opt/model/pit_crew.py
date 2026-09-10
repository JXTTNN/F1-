"""F1 2026 Pit Crew Performance 模型 (Iter-39).

真实 F1 各车队进站效率差异显著:
- **Red Bull Racing**: 2021-2024 多次创下 < 2.0s 进站记录 (最快 1.82s).
- **Mercedes/Ferrari/McLaren**: 2.3-2.5s 平均.
- **后段车队 (Sauber/Haas/RB)**: 2.8-3.5s, 偶发慢停 (>4s).

进站时间 = 进出维修区 (固定 ~17-19s) + 停车换胎 (变量 1.8-4.0s).

影响策略:
- 慢队倾向于少进站 (1-stop vs 2-stop), 因为每次进站多损失 1s.
- 快队可以多进站换取新鲜轮胎 (undercut 战术更有效).

EA Sports F1 2026 游戏官方 Pit Crew 评级:
- 每车队 0-99 评分, 90+ = 顶尖 (RBR/Mercedes), 70-80 = 中段, <70 = 后段.

数据来源 (Iter-39):
- F1 官方进站时间统计 2023-2025 (公开).
- EA Sports F1 2026 车队评级.

公开 API:
    - :func:`pit_crew_rating` — 返回车队 pit crew 评分 (0-99).
    - :func:`pit_stop_time_s` — 返回车队停车换胎时间 (s).
    - :func:`effective_pit_loss_s` — 返回车队在某赛道的有效进站总损失.
"""

from __future__ import annotations

import random

from f1opt.data.teams_2026 import get_team_profile_2026

# --------------------------------------------------------------------------- #
# 车队 pit crew 评分 (0-99, EA F1 2026 游戏官方评级外推)
# --------------------------------------------------------------------------- #
_PIT_CREW_RATING: dict[str, int] = {
    "rbr": 96,   # Red Bull — 历史最快进站 (1.82s 记录)
    "mcl": 94,   # McLaren — 2024-2025 进步显著
    "mer": 93,   # Mercedes — 稳定 2.3s
    "fer": 92,   # Ferrari — 偶发失误但平均快
    "amr": 88,   # Aston Martin — 中上
    "wil": 84,   # Williams — 中等
    "alp": 82,   # Alpine — 中等
    "rb": 86,    # Racing Bulls — 与 RBR 同体系, 训练有素
    "aud": 78,   # Audi — 后段
    "has": 76,   # Haas — 最慢之一
    "cad": 74,   # Cadillac — 新车队, 进站磨合期
}


# --------------------------------------------------------------------------- #
# 时间换算
# --------------------------------------------------------------------------- #
# 基础停车换胎时间 (s): rating 99 → 1.8s, rating 70 → 3.2s
_RATING_TO_STOP_TIME_A = 1.8  # rating=99 时的基础时间
_RATING_TO_STOP_TIME_B = 4.7  # 斜率: 时间 = A + (99-rating) * (B-A)/(99-70)
_RATING_MAX = 99
_RATING_MIN = 70

# 进站失误概率 (慢停 >4s) 与 rating 反相关
_SLOW_STOP_PROB_BASE = 0.02  # 顶尖队 2% 慢停概率
_SLOW_STOP_PROB_MAX = 0.10   # 后段 10% 慢停概率
_SLOW_STOP_EXTRA_TIME_MIN = 1.5  # 慢停额外时间
_SLOW_STOP_EXTRA_TIME_MAX = 4.0

# 完美进站 (<2.0s) 概率 (仅顶尖队)
_PERFECT_STOP_PROB = 0.15
_PERFECT_STOP_TIME_BONUS = 0.3  # 完美进站再快 0.3s


def pit_crew_rating(team_id: str) -> int:
    """返回车队 pit crew 评分 (0-99)."""
    if team_id not in _PIT_CREW_RATING:
        # 验证 team_id 存在
        get_team_profile_2026(team_id)
        return 80  # 默认中段
    return _PIT_CREW_RATING[team_id]


def _rating_to_base_stop_time(rating: int) -> float:
    """评分 → 基础停车换胎时间 (s)."""
    # rating 99 → 1.8s, rating 70 → 4.7s (线性插值)
    if rating >= _RATING_MAX:
        return _RATING_TO_STOP_TIME_A
    if rating <= _RATING_MIN:
        return _RATING_TO_STOP_TIME_B
    ratio = (_RATING_MAX - rating) / (_RATING_MAX - _RATING_MIN)
    return _RATING_TO_STOP_TIME_A + ratio * (_RATING_TO_STOP_TIME_B - _RATING_TO_STOP_TIME_A)


def pit_stop_time_s(team_id: str, seed: int | None = None) -> float:
    """返回单次进站停车换胎时间 (s), 含随机性.

    顶级队 ~1.8-2.3s, 后段 ~2.8-4.0s, 偶发慢停 +1.5-4s.
    """
    rating = pit_crew_rating(team_id)
    base = _rating_to_base_stop_time(rating)
    rng = random.Random(seed)

    # 高斯噪声 (±0.15s)
    noise = rng.gauss(0.0, 0.15)
    stop_time = base + noise

    # 完美进站 (仅 rating >= 90)
    if rating >= 90 and rng.random() < _PERFECT_STOP_PROB:
        stop_time -= _PERFECT_STOP_TIME_BONUS

    # 慢停概率 (rating 越低概率越高)
    slow_prob = _SLOW_STOP_PROB_BASE + (
        (_RATING_MAX - max(rating, _RATING_MIN)) / (_RATING_MAX - _RATING_MIN)
    ) * (_SLOW_STOP_PROB_MAX - _SLOW_STOP_PROB_BASE)
    if rng.random() < slow_prob:
        extra = rng.uniform(_SLOW_STOP_EXTRA_TIME_MIN, _SLOW_STOP_EXTRA_TIME_MAX)
        stop_time += extra

    # 物理下限 (现代 F1 不可能 < 1.5s)
    return max(1.5, float(stop_time))


def expected_pit_stop_time_s(team_id: str) -> float:
    """返回期望进站停车换胎时间 (s, 无随机性).

    用于策略优化 (确定性估计).
    """
    rating = pit_crew_rating(team_id)
    base = _rating_to_base_stop_time(rating)

    # 期望慢停额外时间
    slow_prob = _SLOW_STOP_PROB_BASE + (
        (_RATING_MAX - max(rating, _RATING_MIN)) / (_RATING_MAX - _RATING_MIN)
    ) * (_SLOW_STOP_PROB_MAX - _SLOW_STOP_PROB_BASE)
    expected_slow_extra = slow_prob * (
        (_SLOW_STOP_EXTRA_TIME_MIN + _SLOW_STOP_EXTRA_TIME_MAX) / 2
    )

    # 期望完美进站折扣 (仅 rating >= 90)
    expected_perfect_bonus = 0.0
    if rating >= 90:
        expected_perfect_bonus = _PERFECT_STOP_PROB * _PERFECT_STOP_TIME_BONUS

    return max(1.5, base + expected_slow_extra - expected_perfect_bonus)


def effective_pit_loss_s(team_id: str, track_id: str) -> float:
    """返回车队在某赛道的有效进站总损失 (s).

    = 进出维修区 (track 固定) + 停车换胎 (team-specific 期望时间).

    注意: 这里返回的是"期望"值, 实际仿真应用 :func:`pit_stop_time_s` 加随机性.
    """
    from f1opt.data.track_engineering import get_track_engineering
    try:
        eng = get_track_engineering(track_id)
        pit_lane_loss = eng.pit_loss_s
    except (KeyError, ValueError):
        # fallback 到默认
        pit_lane_loss = 23.0
    return pit_lane_loss + expected_pit_stop_time_s(team_id)


def all_pit_crew_ratings() -> dict[str, int]:
    """返回全部 11 队 pit crew 评分."""
    return dict(_PIT_CREW_RATING)


def pit_crew_ranking() -> list[tuple[str, int]]:
    """返回 pit crew 评分排名 (team_id, rating), 降序."""
    ranked = sorted(_PIT_CREW_RATING.items(), key=lambda x: -x[1])
    return ranked
