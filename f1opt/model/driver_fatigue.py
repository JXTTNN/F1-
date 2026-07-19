"""F1 2026 车手疲劳模型 (Iter-43).

真实 F1 高温赛道对车手体能挑战极大:
- **Singapore Marina Bay**: 舱温 32°C, 湿度 80%, 60 圈夜间赛 → 后半程圈速 +0.3-0.5s.
- **Qatar Losail**: 40°C+ 环境温度, 2023 多名车手因脱水退赛.
- **Miami**: 高温高湿, 57 圈.
- **Bahrain/Suzuka**: 中等挑战.
- **Spa/Silverstone**: 凉爽, 疲劳影响小.

疲劳影响:
- **圈速**: 后半程每圈慢 0.1-0.5s (与体能评分反相关).
- **失误率**: 疲劳车手失误概率 +50-200%.
- **一致性**: 圈速波动加大.
- **雨战专注**: 疲劳影响雨战判断.

车手体能评分 (EA F1 2026 游戏官方):
- 0-99 分, 顶尖 (Verstappen, Hamilton, Norris) 90+.
- 新秀和老将 (Alonso) 体能可能低, 但经验弥补.
- 体能差的車手在高温赛道后半程明显掉速.

物理模型:
- 疲劳度 = f(环境温度, 湿度, 比赛进度, 车手体能).
- 疲劳度 0..1, 0 = 完全清醒, 1 = 极度疲劳.
- 圈速惩罚 = 疲劳度 × max_penalty_s × track_difficulty.

EA Sports F1 2026 游戏官方 Stamina/Fitness 评级.

公开 API:
    - :class:`DriverFatigueModel` — 疲劳状态机.
    - :func:`track_fatigue_difficulty` — 赛道疲劳难度系数.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 赛道疲劳难度系数 (0..1, 1 = 极端)
# --------------------------------------------------------------------------- #
_TRACK_FATIGUE_DIFFICULTY: dict[str, float] = {
    # 极端 (高温高湿)
    "singapore": 1.00,    # 夜赛但 32°C + 80% 湿度
    "losail": 0.95,       # 2023 多人脱水退赛
    "miami": 0.85,
    "bahrain": 0.80,
    "jeddah": 0.75,
    "las_vegas": 0.70,    # 夜赛但干燥
    "yas_marina": 0.70,
    # 中等
    "suzuka": 0.55,       # 凉爽但高 G 力
    "melbourne": 0.50,
    "shanghai": 0.55,
    "austin": 0.60,
    "interlagos": 0.55,
    "monza": 0.45,
    "spa": 0.50,          # 长距离但凉爽
    "silverstone": 0.45,
    "barcelona": 0.50,
    "budapest": 0.60,     # 夏季高温
    "hungaroring": 0.60,
    "zandvoort": 0.50,
    "amsterdam": 0.50,
    # 低 (街道赛 / 凉爽 / 短)
    "monaco": 0.35,       # 短距离 + 低速
    "baku": 0.45,
    "madrid": 0.50,
    "montreal": 0.40,
}
_DEFAULT_DIFFICULTY = 0.50


def track_fatigue_difficulty(track_id: str) -> float:
    """赛道疲劳难度系数 0..1."""
    return _TRACK_FATIGUE_DIFFICULTY.get(track_id, _DEFAULT_DIFFICULTY)


# --------------------------------------------------------------------------- #
# 疲劳参数
# --------------------------------------------------------------------------- #
_MAX_FATIGUE_PENALTY_S = 0.50  # 极端疲劳单圈最大惩罚
_FATIGUE_BUILDUP_PER_LAP = 0.012  # 每圈疲劳累积 (难度 1.0)
_FATIGUE_RECOVERY_SC = -0.10  # SC 期间每圈恢复
_MISTAKE_PROB_BASE = 0.002  # 基础失误概率
_MISTAKE_PROB_FATIGUE_FACTOR = 3.0  # 疲劳车手失误概率倍数


# --------------------------------------------------------------------------- #
# DriverFatigueModel
# --------------------------------------------------------------------------- #
@dataclass
class DriverFatigueModel:
    """车手疲劳状态机 (Iter-43).

    用法::

        model = DriverFatigueModel(
            driver_fitness=90,  # 车手体能评分 0-99
            track_id="singapore",
            ambient_temp_c=32.0,
            humidity_pct=80.0,
            total_laps=60,
        )
        # 每圈更新
        for lap in range(60):
            model.update_lap()
            penalty = model.lap_penalty_s()
            mistake_prob = model.mistake_probability()
    """

    driver_fitness: int = 80
    """车手体能评分 0-99 (EA F1 2026 Stamina)."""
    track_id: str = "monza"
    ambient_temp_c: float = 25.0
    humidity_pct: float = 50.0
    total_laps: int = 58

    # 状态
    fatigue: float = field(init=False, default=0.0)
    laps_completed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.driver_fitness = max(0, min(99, int(self.driver_fitness)))
        self.fatigue = 0.0
        self.laps_completed = 0

    # ------------------------------------------------------------------ #
    def _fatigue_buildup_rate(self) -> float:
        """每圈疲劳累积率 (受赛道难度 + 环境 + 体能影响)."""
        base = _FATIGUE_BUILDUP_PER_LAP * track_fatigue_difficulty(self.track_id)
        # 环境温度影响 (>25°C 加速)
        temp_factor = 1.0 + max(0.0, (self.ambient_temp_c - 25.0) / 15.0)
        # 湿度影响 (>50% 加速)
        humidity_factor = 1.0 + max(0.0, (self.humidity_pct - 50.0) / 100.0)
        # 体能影响 (体能 99 = 0.7×, 体能 50 = 1.5×)
        fitness_factor = 1.5 - (self.driver_fitness / 99.0) * 0.8
        return base * temp_factor * humidity_factor * fitness_factor

    # ------------------------------------------------------------------ #
    def update_lap(self) -> float:
        """更新一圈后的疲劳度.

        Returns:
            更新后的疲劳度 0..1.
        """
        rate = self._fatigue_buildup_rate()
        self.fatigue += rate
        # 物理上限
        self.fatigue = min(1.0, self.fatigue)
        self.laps_completed += 1
        return self.fatigue

    # ------------------------------------------------------------------ #
    def update_sc_lap(self) -> float:
        """SC 期间更新一圈 (恢复疲劳)."""
        self.fatigue += _FATIGUE_RECOVERY_SC
        self.fatigue = max(0.0, self.fatigue)
        self.laps_completed += 1
        return self.fatigue

    # ------------------------------------------------------------------ #
    def lap_penalty_s(self) -> float:
        """当前疲劳导致的圈速惩罚 s."""
        return self.fatigue * _MAX_FATIGUE_PENALTY_S

    def mistake_probability(self) -> float:
        """当前疲劳下的单圈失误概率."""
        return _MISTAKE_PROB_BASE * (1.0 + self.fatigue * _MISTAKE_PROB_FATIGUE_FACTOR)

    def consistency_factor(self) -> float:
        """疲劳影响下的一致性因子 (1.0 = 正常, <1 = 波动)."""
        return max(0.5, 1.0 - self.fatigue * 0.5)

    # ------------------------------------------------------------------ #
    def expected_total_fatigue_penalty_s(self) -> float:
        """整场预计疲劳总惩罚 s (后半程)."""
        # 假设线性累积到 0.7 (典型后半程)
        avg_fatigue = 0.35  # 平均疲劳
        # 后半程 (laps/2 圈) 受惩罚
        half_race = self.total_laps // 2
        return avg_fatigue * _MAX_FATIGUE_PENALTY_S * half_race

    # ------------------------------------------------------------------ #
    def state(self) -> dict[str, float | int | str]:
        """返回当前状态摘要."""
        if self.fatigue < 0.2:
            phase = "fresh"
        elif self.fatigue < 0.5:
            phase = "moderate"
        elif self.fatigue < 0.8:
            phase = "tired"
        else:
            phase = "exhausted"
        return {
            "driver_fitness": self.driver_fitness,
            "track_id": self.track_id,
            "ambient_temp_c": self.ambient_temp_c,
            "humidity_pct": self.humidity_pct,
            "laps_completed": self.laps_completed,
            "fatigue": round(self.fatigue, 4),
            "lap_penalty_s": round(self.lap_penalty_s(), 4),
            "mistake_probability": round(self.mistake_probability(), 6),
            "consistency_factor": round(self.consistency_factor(), 4),
            "phase": phase,
        }
