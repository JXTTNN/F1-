"""F1 轮胎进站策略优化器 (Iter-16).

真实 F1 车队 (Ferrari, Mercedes, Red Bull) 每场比赛通过 "strategy simulator"
网格搜索最优进站策略 — 哪圈进站 + 用哪种 compound. 决策变量:

- **进站次数**: 0 (无进站, 干地短赛/湿地) / 1 / 2 / 3.
- **每次进站圈数**: pit_laps tuple.
- **每个 stint 化合物**: compounds tuple (长度 = pit_laps+1).

约束 (FIA 2026 体育规则 §30.5):
- 必须使用至少 2 种不同干地 compound (除非湿地/全程一胎).
- 干地正赛最少 1 次进站换胎.

目标: 最小化总比赛时间 (圈速 + 进站损失 + 轮胎退化).

公开 API:
    - :class:`StrategySimulator` — 单车轻量仿真 (不模拟对手, 仅圈速+进站).
    - :class:`StrategyOptimizer` — 网格搜索 + 启发式修剪.
    - :func:`optimize_strategy` — 便捷函数.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from f1opt.data.setup_schema import CarSetup
from f1opt.model.lap_simulator import LapTimeSimulator
from f1opt.model.weather import WeatherModel

# --------------------------------------------------------------------------- #
# 物理常量
# --------------------------------------------------------------------------- #
_DEFAULT_PIT_LOSS_S = 23.0
_PIT_LOSS_BY_TRACK: dict[str, float] = {
    "monaco": 21.0, "monza": 23.0, "spa": 22.0, "silverstone": 23.0,
    "suzuka": 22.0, "bahrain": 24.0, "jeddah": 22.0, "melbourne": 23.0,
    "singapore": 22.0, "yas_marini": 23.0, "baku": 22.0, "miami": 23.0,
    "barcelona": 23.0, "austin": 23.0, "interlagos": 22.0, "losail": 23.0,
    "shanghai": 24.0, "budapest": 22.0, "amsterdam": 23.0, "montreal": 23.0,
    "las_vegas": 23.0, "madrid": 23.0,
}

# 轮胎特性 (从 tire_stint 数据, stint 推荐长度)
_COMPOUND_STINT_RANGE: dict[str, tuple[int, int]] = {
    "soft": (8, 18),
    "medium": (15, 28),
    "hard": (22, 38),
    "intermediate": (10, 20),
    "wet": (8, 18),
}

# 干地候选 compound (FIA 必须用 2 种干地胎规则)
_DRY_COMPOUNDS = ("soft", "medium", "hard")
_WET_COMPOUNDS = ("intermediate", "wet")


def _pit_loss_for(track_id: str) -> float:
    return _PIT_LOSS_BY_TRACK.get(track_id, _DEFAULT_PIT_LOSS_S)


# --------------------------------------------------------------------------- #
# StrategySimulator — 单车轻量仿真
# --------------------------------------------------------------------------- #
@dataclass
class StrategySimulator:
    """单车单圈仿真 (不模拟对手, 仅圈速 + 进站损失).

    比 :class:`RaceSimulation` 快 ~50×, 用于策略网格搜索.
    """

    setup: CarSetup
    track_id: str
    total_laps: int
    pit_loss_s: float = 0.0  # 自动从 track_id 推断
    driver_aggression: float = 0.7
    driver_smoothness: float = 0.7
    driver_consistency: float = 0.7
    initial_fuel_kg: float = 110.0
    brake_temp_c: float = 500.0
    track_temp_c: float = 35.0
    weather: WeatherModel | None = None
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        if self.pit_loss_s <= 0:
            self.pit_loss_s = _pit_loss_for(self.track_id)
        # 缓存: compound → 完整 stint 圈速列表 (Iter-16 性能优化)
        self._stint_cache: dict[str, list[float]] = {}

    def _get_stint_lap_times(self, compound: str) -> list[float]:
        """返回该 compound 从新胎开始的逐圈圈速 (缓存)."""
        if compound not in self._stint_cache:
            sim = LapTimeSimulator(
                setup=self.setup,
                track_id=self.track_id,
                compound=compound,
                driver_aggression=self.driver_aggression,
                driver_smoothness=self.driver_smoothness,
                driver_consistency=self.driver_consistency,
                initial_fuel_kg=self.initial_fuel_kg,
                brake_temp_c=self.brake_temp_c,
                track_temp_c=self.track_temp_c,
                weather=self.weather,
            )
            stint = sim.simulate_stint(self.total_laps)
            self._stint_cache[compound] = [lp["lap_time"] for lp in stint]
        return self._stint_cache[compound]

    # ------------------------------------------------------------------ #
    def evaluate(self, pit_laps: tuple[int, ...],
                 compounds: tuple[str, ...]) -> dict[str, Any]:
        """评估一套策略, 返回总时间与详情 (使用 stint 缓存加速)."""
        if len(compounds) != len(pit_laps) + 1:
            raise ValueError(
                f"compounds ({len(compounds)}) must be len(pit_laps)+1 "
                f"({len(pit_laps)+1})"
            )
        # 检查进站圈递增
        for i in range(len(pit_laps) - 1):
            if pit_laps[i] >= pit_laps[i + 1]:
                raise ValueError(f"pit_laps must be strictly increasing: {pit_laps}")

        rng = random.Random(self.rng_seed)
        total_time = 0.0
        lap_times: list[float] = []
        # 每个 stint 的起止圈 (1-indexed)
        stint_starts = [1] + [pl + 1 for pl in pit_laps]
        stint_ends = list(pit_laps) + [self.total_laps]
        for stint_idx, (c, start, end) in enumerate(
            zip(compounds, stint_starts, stint_ends, strict=True)
        ):
            stint_len = end - start + 1
            if stint_len <= 0:
                continue
            # 取缓存的圈速 (从新胎开始)
            cached = self._get_stint_lap_times(c)
            for k in range(stint_len):
                base_lap = cached[k] if k < len(cached) else cached[-1]
                noise = rng.gauss(0.0, (1.0 - self.driver_consistency) * 0.3)
                lap_time = base_lap + noise
                lap_times.append(lap_time)
                total_time += lap_time
            # 进站损失 (该 stint 末尾进站, 最后 stint 除外)
            if stint_idx < len(pit_laps):
                total_time += self.pit_loss_s
                lap_times[-1] += self.pit_loss_s

        return {
            "total_time": float(total_time),
            "lap_times": lap_times,
            "pit_laps": pit_laps,
            "compounds": compounds,
            "n_stops": len(pit_laps),
            "best_lap": float(min(lap_times)) if lap_times else 0.0,
            "worst_lap": float(max(lap_times)) if lap_times else 0.0,
            "avg_lap": float(total_time / max(1, self.total_laps)),
        }


# --------------------------------------------------------------------------- #
# StrategyOptimizer — 网格搜索
# --------------------------------------------------------------------------- #
@dataclass
class StrategyCandidate:
    """单套策略评估结果."""

    pit_laps: tuple[int, ...]
    compounds: tuple[str, ...]
    total_time: float
    n_stops: int
    avg_lap: float
    best_lap: float
    worst_lap: float


@dataclass
class StrategyOptimizer:
    """网格搜索最优进站策略.

    用法::

        opt = StrategyOptimizer(
            setup=DEFAULT_SETUP, track_id="monza", total_laps=53,
        )
        result = opt.optimize()
        # result.best = StrategyCandidate(pit_laps=(22, 41),
        #     compounds=("medium","hard","medium"), ...)
    """

    setup: CarSetup
    track_id: str
    total_laps: int
    driver_aggression: float = 0.7
    driver_smoothness: float = 0.7
    driver_consistency: float = 0.7
    initial_fuel_kg: float = 110.0
    track_temp_c: float = 35.0
    weather: WeatherModel | None = None
    # 搜索参数
    n_stops_options: tuple[int, ...] = (1, 2, 3)
    pit_window_step: int = 3  # 进站圈步长 (越小越精确, 越慢)
    rng_seed: int | None = None
    # 评估结果
    all_candidates: list[StrategyCandidate] = field(default_factory=list)
    best: StrategyCandidate | None = None

    # ------------------------------------------------------------------ #
    def optimize(self) -> StrategyCandidate:
        """执行网格搜索, 返回最优策略."""
        self.all_candidates = []
        # 决定候选 compound (干地 vs 湿地)
        compounds_pool: tuple[str, ...]
        if self.weather is not None and self.weather.state.track_wetness > 0.3:
            compounds_pool = _WET_COMPOUNDS
        else:
            compounds_pool = _DRY_COMPOUNDS

        rng = random.Random(self.rng_seed)

        # 构建仿真器 (复用)
        sim = StrategySimulator(
            setup=self.setup,
            track_id=self.track_id,
            total_laps=self.total_laps,
            driver_aggression=self.driver_aggression,
            driver_smoothness=self.driver_smoothness,
            driver_consistency=self.driver_consistency,
            initial_fuel_kg=self.initial_fuel_kg,
            track_temp_c=self.track_temp_c,
            weather=self.weather,
            rng_seed=self.rng_seed,
        )

        # 枚举每套 n_stops 策略
        for n_stops in self.n_stops_options:
            if n_stops == 0:
                # 无进站: 只 1 个 compound
                for c in compounds_pool:
                    self._try_candidate(sim, (), (c,))
                continue
            # 生成进站圈候选 (网格搜索)
            pit_laps_candidates = self._gen_pit_laps_candidates(n_stops)
            for pit_laps in pit_laps_candidates:
                # 生成 compound 组合 (n_stops+1 个 stint)
                for compounds in self._gen_compound_combos(n_stops + 1, compounds_pool, rng):
                    self._try_candidate(sim, pit_laps, compounds)

        # 选最优
        if not self.all_candidates:
            # fallback: 1-stop medium→hard at mid
            mid = self.total_laps // 2
            self._try_candidate(sim, (mid,), ("medium", "hard"))
        self.best = min(self.all_candidates, key=lambda c: c.total_time)
        return self.best

    def _try_candidate(self, sim: StrategySimulator,
                       pit_laps: tuple[int, ...],
                       compounds: tuple[str, ...]) -> None:
        try:
            r = sim.evaluate(pit_laps=pit_laps, compounds=compounds)
            self.all_candidates.append(StrategyCandidate(
                pit_laps=pit_laps,
                compounds=compounds,
                total_time=r["total_time"],
                n_stops=r["n_stops"],
                avg_lap=r["avg_lap"],
                best_lap=r["best_lap"],
                worst_lap=r["worst_lap"],
            ))
        except (ValueError, KeyError):
            pass  # 跳过非法策略

    def _gen_pit_laps_candidates(self, n_stops: int) -> list[tuple[int, ...]]:
        """生成所有合法进站圈组合 (递增, 在 stint 推荐长度内)."""
        if n_stops == 0:
            return [()]
        # 单次进站: 全程 / 2 ± window
        if n_stops == 1:
            mid = self.total_laps // 2
            candidates: list[tuple[int, ...]] = []
            for pl in range(max(3, mid - 8), min(self.total_laps - 3, mid + 9),
                            self.pit_window_step):
                candidates.append((pl,))
            return candidates
        # 2 次: 第一站在 1/3, 第二站在 2/3
        if n_stops == 2:
            third = self.total_laps // 3
            two_thirds = 2 * self.total_laps // 3
            candidates = []
            for p1 in range(max(3, third - 6), min(self.total_laps - 6, third + 7),
                             self.pit_window_step):
                for p2 in range(max(p1 + 5, two_thirds - 6),
                                min(self.total_laps - 3, two_thirds + 7),
                                self.pit_window_step):
                    candidates.append((p1, p2))
            return candidates
        # 3 次: 均分 4 段
        quarter = self.total_laps // 4
        candidates = []
        for p1 in range(max(3, quarter - 4), quarter + 5, self.pit_window_step):
            for p2 in range(max(p1 + 4, 2 * quarter - 4), 2 * quarter + 5,
                            self.pit_window_step):
                for p3 in range(max(p2 + 4, 3 * quarter - 4), 3 * quarter + 5,
                                self.pit_window_step):
                    candidates.append((p1, p2, p3))
        return candidates

    def _gen_compound_combos(
        self, n_stints: int, pool: tuple[str, ...],
        rng: random.Random,
    ) -> list[tuple[str, ...]]:
        """生成 compound 组合 (限制总数避免爆炸).

        干地: 必须至少 2 种 (FIA 规则), 否则 penalty.
        """
        if n_stints == 1:
            return [(c,) for c in pool]
        if n_stints == 2:
            out: list[tuple[str, ...]] = []
            for c1 in pool:
                for c2 in pool:
                    if c1 != c2 or pool == _WET_COMPOUNDS:
                        out.append((c1, c2))
            return out
        # n_stints >= 3: 第一/中段/最后; 中段用最耐用 compound 简化
        out = []
        # 中段只取 medium/hard (干地) 或 wet (湿地) 以减少组合爆炸
        mid_pool = ("medium", "hard") if pool == _DRY_COMPOUNDS else pool
        for c_start in pool:
            for c_end in pool:
                for c_mid in mid_pool:
                    combo = (c_start,) + (c_mid,) * (n_stints - 2) + (c_end,)
                    # FIA 规则: 干地至少 2 种
                    if pool == _DRY_COMPOUNDS and len(set(combo)) == 1:
                        continue
                    out.append(combo)
        # 截断避免过多
        if len(out) > 30:
            rng.shuffle(out)
            out = out[:30]
        return out

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        if self.best is None:
            self.optimize()
        assert self.best is not None
        return {
            "track_id": self.track_id,
            "total_laps": self.total_laps,
            "best_strategy": {
                "pit_laps": self.best.pit_laps,
                "compounds": self.best.compounds,
                "n_stops": self.best.n_stops,
                "total_time": round(self.best.total_time, 3),
                "avg_lap": round(self.best.avg_lap, 3),
            },
            "n_candidates_evaluated": len(self.all_candidates),
            "top5": [
                {
                    "pit_laps": c.pit_laps,
                    "compounds": c.compounds,
                    "total_time": round(c.total_time, 3),
                    "n_stops": c.n_stops,
                }
                for c in sorted(self.all_candidates, key=lambda x: x.total_time)[:5]
            ],
        }


def optimize_strategy(
    setup: CarSetup,
    track_id: str,
    total_laps: int,
    weather: WeatherModel | None = None,
    driver_aggression: float = 0.7,
    seed: int | None = None,
) -> StrategyCandidate:
    """便捷函数: 单次调用返回最优策略."""
    opt = StrategyOptimizer(
        setup=setup, track_id=track_id, total_laps=total_laps,
        weather=weather, driver_aggression=driver_aggression, rng_seed=seed,
    )
    return opt.optimize()


# --------------------------------------------------------------------------- #
# LiveStrategyAdvisor (Iter-23)
# --------------------------------------------------------------------------- #
@dataclass
class LiveStrategyDecision:
    """实时策略建议."""

    should_pit: bool
    """是否本圈进站."""
    new_compound: str | None
    """若进站, 切换到的化合物 (None = 不进站)."""
    reason: str
    """决策原因 (人类可读)."""
    remaining_pit_laps: tuple[int, ...]
    """剩余 (原计划) 进站圈. 若本次进站, 该 lap 从 tuple 中移除."""
    remaining_compounds: tuple[str, ...]
    """剩余 (原计划) 化合物. 若本次进站, 该 compound 从 tuple 中移除."""
    urgency: float
    """决策紧迫度 0..1 (0 = 谨慎, 1 = 必须立即进站)."""
    estimated_gain_s: float
    """预计本次决策收益 (相对不进站), 正 = 进站更快."""


@dataclass
class LiveStrategyAdvisor:
    """实时策略顾问 (Iter-23).

    车队 strategy wall 每圈调用此顾问, 给定当前比赛状态, 判断是否本圈进站.

    决策逻辑 (F1 真实车队 wall 决策原则):

    1. **SC/VSC 期间 + 轮胎非新胎**: 立即进站 (cheap pit, 时间损失仅 20-55%).
    2. **轮胎接近悬崖 (wear > 70%)**: 进站 (避免 1-2 s/lap 悬崖惩罚).
    3. **湿地推荐化合物变化 (Pirelli crossover)**: 进站换胎.
    4. **Undercut 机会 (gap_ahead < 1.5 * pit_loss + 2)**: 进站超越前车.
    5. **末段 (remaining <= 5) + 当前胎 OK**: 不进站 (维持位置).
    6. **保守 (gap_behind < pit_loss + 2)**: 维持当前策略, 防御后车.

    用法::

        advisor = LiveStrategyAdvisor(track_id="monza", total_laps=58,
                                       pit_loss_s=23.0)
        decision = advisor.decide(
            lap=20, current_compound="medium", tire_age_laps=15,
            sc_active=True, sc_remaining_laps=3,
            position=4, gap_ahead_s=2.5, gap_behind_s=8.0,
            remaining_pit_laps=(40,), remaining_compounds=("hard",),
        )
        if decision.should_pit:
            print("PIT! reason:", decision.reason)
    """

    track_id: str
    total_laps: int
    pit_loss_s: float = _DEFAULT_PIT_LOSS_S
    sc_pit_discount: float = 0.20  # SC 期间 pit loss 仅 20%
    vsc_pit_discount: float = 0.55
    cliff_wear_threshold_pct: float = 70.0
    undercut_gap_threshold_s: float = 36.0  # ~1.5 × pit_loss
    defensive_gap_threshold_s: float = 25.0

    # ------------------------------------------------------------------ #
    def decide(
        self,
        lap: int,
        current_compound: str,
        tire_age_laps: int,
        sc_active: bool,
        sc_remaining_laps: int,
        position: int,
        gap_ahead_s: float,
        gap_behind_s: float,
        remaining_pit_laps: tuple[int, ...],
        remaining_compounds: tuple[str, ...],
        weather_recommended_compound: str | None = None,
        tire_wear_pct: float = 0.0,
        is_vsc: bool = False,
    ) -> LiveStrategyDecision:
        """实时策略建议."""
        remaining = self.total_laps - lap
        discount = self.vsc_pit_discount if is_vsc else self.sc_pit_discount
        effective_pit_loss = self.pit_loss_s * (discount if sc_active else 1.0)
        next_compound = remaining_compounds[0] if remaining_compounds else current_compound

        # 1. SC/VSC 期间 + 轮胎非新胎 → 立即进站
        if sc_active and tire_age_laps >= 3 and remaining >= 3:
            gain = self._estimate_sc_pit_gain(
                tire_age_laps, remaining, current_compound, next_compound,
                effective_pit_loss,
            )
            return LiveStrategyDecision(
                should_pit=True,
                new_compound=next_compound,
                reason=(
                    f"SC/VSC cheap pit (loss {effective_pit_loss:.1f}s "
                    f"vs {self.pit_loss_s:.1f}s)"
                ),
                remaining_pit_laps=remaining_pit_laps[1:],
                remaining_compounds=remaining_compounds[1:],
                urgency=0.95,
                estimated_gain_s=gain,
            )

        # 2. 轮胎接近悬崖 → 进站
        if tire_wear_pct >= self.cliff_wear_threshold_pct and remaining >= 3:
            gain = self._estimate_cliff_avoidance_gain(
                tire_wear_pct, remaining, current_compound, next_compound,
                self.pit_loss_s,
            )
            return LiveStrategyDecision(
                should_pit=True,
                new_compound=next_compound,
                reason=f"Tire cliff avoidance (wear {tire_wear_pct:.0f}%)",
                remaining_pit_laps=remaining_pit_laps[1:],
                remaining_compounds=remaining_compounds[1:],
                urgency=0.85,
                estimated_gain_s=gain,
            )

        # 3. 湿地推荐化合物变化 → 进站换胎 (Pirelli crossover)
        if (weather_recommended_compound is not None
                and weather_recommended_compound != current_compound
                and remaining >= 3):
            return LiveStrategyDecision(
                should_pit=True,
                new_compound=weather_recommended_compound,
                reason=f"Weather crossover: {current_compound} → {weather_recommended_compound}",
                remaining_pit_laps=remaining_pit_laps[1:],
                remaining_compounds=remaining_compounds[1:],
                urgency=0.90,
                estimated_gain_s=8.0,  # wet/dry mismatch huge gain
            )

        # 4. Undercut 机会 (前车接近 + 我们轮胎已用一定圈数)
        # Undercut = 提前进站用新胎做出快圈超越前车
        min_tire_age_for_undercut = 12  # 至少用 12 圈才考虑 undercut
        if (gap_ahead_s < self.undercut_gap_threshold_s
                and tire_age_laps >= min_tire_age_for_undercut
                and remaining_pit_laps  # 还有计划进站
                and remaining >= 5
                and remaining_pit_laps[0] - lap <= 5):  # 接近原计划进站圈
            gain = self._estimate_undercut_gain(
                gap_ahead_s, tire_age_laps, effective_pit_loss,
            )
            return LiveStrategyDecision(
                should_pit=True,
                new_compound=next_compound,
                reason=f"Undercut attempt (gap_ahead {gap_ahead_s:.1f}s)",
                remaining_pit_laps=remaining_pit_laps[1:],
                remaining_compounds=remaining_compounds[1:],
                urgency=0.60,
                estimated_gain_s=gain,
            )

        # 5. 末段不进站
        if remaining <= 5:
            return LiveStrategyDecision(
                should_pit=False,
                new_compound=None,
                reason=f"Late race ({remaining} laps left), hold position",
                remaining_pit_laps=remaining_pit_laps,
                remaining_compounds=remaining_compounds,
                urgency=0.10,
                estimated_gain_s=0.0,
            )

        # 6. 防御后车 (gap_behind 接近 + 我们 tire 新)
        if (gap_behind_s < self.defensive_gap_threshold_s
                and tire_age_laps <= 10
                and remaining_pit_laps):
            # 推迟进站, 防御 undercut
            return LiveStrategyDecision(
                should_pit=False,
                new_compound=None,
                reason=f"Defensive (gap_behind {gap_behind_s:.1f}s), hold track pos",
                remaining_pit_laps=remaining_pit_laps,
                remaining_compounds=remaining_compounds,
                urgency=0.30,
                estimated_gain_s=0.0,
            )

        # 7. 默认: 按原计划
        return LiveStrategyDecision(
            should_pit=False,
            new_compound=None,
            reason="Following planned strategy",
            remaining_pit_laps=remaining_pit_laps,
            remaining_compounds=remaining_compounds,
            urgency=0.20,
            estimated_gain_s=0.0,
        )

    # ------------------------------------------------------------------ #
    def _estimate_sc_pit_gain(
        self,
        tire_age_laps: int,
        remaining: int,
        current_compound: str,
        next_compound: str,
        effective_pit_loss: float,
    ) -> float:
        """估算 SC 期间进站收益 (相对不进站).

        新胎相对旧胎每圈快 0.05-0.10 s (取决于当前胎磨损程度).
        """
        # 旧胎每圈额外损失 (粗略: 0.04 s/lap × tire_age)
        old_tire_penalty_per_lap = 0.04 * min(tire_age_laps, 25)
        # 新胎前几圈有暖胎惩罚 (~0.4 s 平均)
        warmup_cost = 0.4
        # SC pit loss 节省
        sc_saving = self.pit_loss_s - effective_pit_loss
        # 总收益
        gain = (old_tire_penalty_per_lap * remaining
                - warmup_cost
                + sc_saving)
        return max(0.0, gain)

    def _estimate_cliff_avoidance_gain(
        self,
        tire_wear_pct: float,
        remaining: int,
        current_compound: str,
        next_compound: str,
        pit_loss: float,
    ) -> float:
        """估算避免悬崖进站的收益."""
        # 悬崖期每圈损失 1.0-1.6 s
        cliff_penalty_per_lap = 1.2
        # 估计还有多少圈会进入悬崖
        laps_in_cliff = max(0, remaining - 5)  # 保守估计
        warmup_cost = 0.4
        gain = (cliff_penalty_per_lap * laps_in_cliff
                - warmup_cost
                - pit_loss)
        return max(0.0, gain)

    def _estimate_undercut_gain(
        self,
        gap_ahead_s: float,
        tire_age_laps: int,
        effective_pit_loss: float,
    ) -> float:
        """估算 undercut 收益."""
        # 新胎每圈快 0.04 * tire_age (老胎越老收益越大)
        per_lap_gain = 0.04 * min(tire_age_laps, 25)
        # 假设能做出 5 个快圈
        undercut_window = 5
        gain = per_lap_gain * undercut_window - effective_pit_loss + gap_ahead_s * 0.3
        return gain
