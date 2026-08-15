"""F1 赛季端到端仿真器 (Iter-18 / Iter-21).

整合 :class:`RaceSimulation` + :class:`Championship` + :class:`QualifyingSession`
模拟完整 F1 赛季:

1. 输入: 22 车手 + 11 车队 + 24 场赛历.
2. 每场:
   - :class:`QualifyingSession` 三阶段排位赛决定发车位 (Iter-21).
   - 用 :class:`RaceSimulation` 跑完正赛.
   - 把完赛名次记录到 :class:`Championship`.
3. 输出: 双世界冠军 + 完整积分榜.

公开 API:
    - :class:`SeasonDriver` — 单车手赛季配置.
    - :class:`SeasonSimulator` — 完整赛季仿真.
    - :func:`simulate_season` — 便捷函数.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from f1opt.data.setup_schema import DEFAULT_SETUP, CarSetup
from f1opt.model.championship import (
    SEASON_2026_CALENDAR,
    Championship,
    ConstructorStanding,
    DriverStanding,
)
from f1opt.model.qualifying import (
    DriverQualifyingInput,
    QualifyingSession,
)
from f1opt.model.race_simulator import (
    RaceCar,
    RaceSimulation,
    RaceStrategy,
)
from f1opt.model.weather import WeatherModel, WeatherState


# --------------------------------------------------------------------------- #
# SeasonDriver
# --------------------------------------------------------------------------- #
def _affinity_for(driver_id: str, track_id: str) -> float:
    """查询车手-赛道亲和度 (Iter-38). 安全 fallback 0.0."""
    try:
        from f1opt.data.driver_track_affinity import driver_track_affinity
        return driver_track_affinity(driver_id, track_id)
    except (ValueError, ImportError):
        return 0.0


@dataclass
class SeasonDriver:
    """单车手赛季配置."""

    driver_id: str
    driver_name: str
    team_id: str
    setup: CarSetup = field(default_factory=lambda: DEFAULT_SETUP)
    driver_aggression: float = 0.7
    driver_smoothness: float = 0.7
    driver_consistency: float = 0.7
    driver_tire_management: float = 0.5
    """车手轮胎管理风格 0..1 (Iter-22)."""
    car_performance_offset_s: float = 0.0
    """车队赛车性能偏移 s/lap (Iter-36). 来自 teams_2026."""
    driver_track_affinity_s: float = 0.0
    """车手-赛道亲和度 s/lap (Iter-38). 正=快, 负=慢.
    默认 0; 季中仿真器会按当前 track_id 动态查询."""
    # 默认策略 (具体赛事可覆盖)
    default_strategy: RaceStrategy = field(default_factory=lambda: RaceStrategy(
        pit_laps=(20, 40), compounds=("medium", "hard", "medium")
    ))


# --------------------------------------------------------------------------- #
# SeasonSimulator
# --------------------------------------------------------------------------- #
@dataclass
class SeasonSimulator:
    """完整 F1 赛季仿真.

    用法::

        drivers = [SeasonDriver(driver_id=f"d{i:02d}", driver_name=f"D{i+1}",
                                team_id=f"t{i//2:02d}") for i in range(22)]
        teams = [(f"t{i:02d}", f"Team {i}") for i in range(11)]
        sim = SeasonSimulator(drivers=drivers, teams=teams, seed=42)
        result = sim.run()
        # result["champions"]["drivers_champion"]
    """

    drivers: list[SeasonDriver]
    teams: list[tuple[str, str]]  # (team_id, team_name)
    calendar: tuple[str, ...] = SEASON_2026_CALENDAR
    total_laps_per_race: int = 58  # 默认, 实际可按赛道查
    seed: int | None = None
    weather_wetness_range: tuple[float, float] = (0.0, 0.15)  # 大部分干地

    def __post_init__(self) -> None:
        if len(self.drivers) != 22:
            raise ValueError(f"Need 22 drivers, got {len(self.drivers)}")
        if len(self.teams) != 11:
            raise ValueError(f"Need 11 teams, got {len(self.teams)}")

    # ------------------------------------------------------------------ #
    def run(self) -> dict[str, Any]:
        """运行完整赛季, 返回最终结果."""
        rng = random.Random(self.seed)
        # 构建 Championship
        driver_standings = [
            DriverStanding(d.driver_id, d.driver_name, d.team_id)
            for d in self.drivers
        ]
        team_standings = [
            ConstructorStanding(tid, tname) for tid, tname in self.teams
        ]
        champ = Championship(
            drivers=driver_standings,
            teams=team_standings,
            calendar=self.calendar,
        )

        # 每场赛季积分跟踪 (用于排序种子)
        driver_points: dict[str, float] = {d.driver_id: 0.0 for d in self.drivers}

        race_summaries: list[dict[str, Any]] = []

        for race_idx, track_id in enumerate(self.calendar):
            race_seed = (self.seed or 0) + race_idx
            # === 三阶段排位赛 (Iter-21) ===
            qualy_inputs = [
                DriverQualifyingInput(
                    driver_id=d.driver_id,
                    driver_name=d.driver_name,
                    team_id=d.team_id,
                    setup=d.setup,
                    skill=(d.driver_consistency * 0.4
                           + d.driver_smoothness * 0.3
                           + d.driver_aggression * 0.3),
                    aggression=d.driver_aggression,
                    smoothness=d.driver_smoothness,
                    consistency=d.driver_consistency,
                )
                for d in self.drivers
            ]
            # 排位赛种子独立每场 (含 momentum 的影响通过 points 传递)
            qualy_sess = QualifyingSession(
                track_id=track_id,
                drivers=qualy_inputs,
                seed=race_seed * 7 + 1,
            )
            grid = qualy_sess.run()
            grid_order = [r.driver_id for r in grid]

            # === 构建 RaceCar ===
            cars: list[RaceCar] = []
            for grid_pos, d_id in enumerate(grid_order, start=1):
                sd = next(d for d in self.drivers if d.driver_id == d_id)
                # Q2 通过者使用其 Q2 最快圈所用胎作为正赛首段胎
                qualy_result = next(r for r in grid if r.driver_id == d_id)
                if qualy_result.grid_position <= 10 and \
                        qualy_result.q2_tire_for_race != "soft":
                    # 用 Q2 决定的轮胎作为首段
                    first_compound = qualy_result.q2_tire_for_race
                    strategy = RaceStrategy(
                        pit_laps=sd.default_strategy.pit_laps,
                        compounds=(first_compound,) + sd.default_strategy.compounds[1:],
                    )
                else:
                    strategy = sd.default_strategy
                car = RaceCar(
                    driver_id=sd.driver_id,
                    driver_name=sd.driver_name,
                    setup=sd.setup,
                    grid_position=grid_pos,
                    strategy=strategy,
                    driver_aggression=sd.driver_aggression,
                    driver_smoothness=sd.driver_smoothness,
                    driver_consistency=sd.driver_consistency,
                    driver_tire_management=sd.driver_tire_management,
                    car_performance_offset_s=sd.car_performance_offset_s,
                    driver_track_affinity_s=_affinity_for(sd.driver_id, track_id),
                    team_id=sd.team_id,
                )
                cars.append(car)

            # === 随机天气 (大部分干地, 10% 雨) ===
            weather: WeatherModel | None = None
            if rng.random() < 0.10:
                w = rng.uniform(*self.weather_wetness_range)
                # 偶发大雨
                if rng.random() < 0.2:
                    w = rng.uniform(0.5, 0.9)
                weather = WeatherModel(initial=WeatherState(
                    track_wetness=w,
                    track_temp_c=30.0 if w > 0.3 else 35.0,
                ))

            # === 运行正赛 ===
            sim = RaceSimulation(
                track_id=track_id,
                cars=cars,
                total_laps=self.total_laps_per_race,
                seed=race_seed,
                weather=weather,
                weather_rain_mmh=5.0 if weather is not None else 0.0,
            )
            results = sim.run()

            # === 记录到 Championship ===
            champ_results = []
            for pos, car in results:
                dnf = car.retired
                laps = car.laps_completed
                champ_results.append({
                    "driver_id": car.driver_id,
                    "team_id": next(
                        d.team_id for d in self.drivers if d.driver_id == car.driver_id
                    ),
                    "position": None if dnf else pos,
                    "laps_completed": laps,
                })
            champ.record_race(
                race_id=track_id,
                results=champ_results,
                total_laps=self.total_laps_per_race,
            )

            # 更新 driver_points (用于下站 momentum)
            for cr in champ_results:
                pos = cr["position"]
                if pos is not None and int(pos) <= 10:
                    from f1opt.model.championship import RACE_POINTS
                    driver_points[str(cr["driver_id"])] += RACE_POINTS[int(pos) - 1]

            # 简短摘要
            race_summaries.append({
                "race_idx": race_idx,
                "track_id": track_id,
                "winner": results[0][1].driver_name if not results[0][1].retired
                          else next(r[1].driver_name for r in results if not r[1].retired),
                "n_retirements": sum(1 for _, c in results if c.retired),
                "weather": "wet" if weather is not None else "dry",
                "n_sc_periods": sim.safety_car.summary()["n_periods"],
            })

        return {
            "champions": champ.champions(),
            "summary": champ.summary(),
            "race_summaries": race_summaries,
            "final_driver_standings": champ.driver_standings(),
            "final_constructor_standings": champ.constructor_standings(),
        }


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
def simulate_season(
    n_drivers: int = 22,
    n_teams: int = 11,
    seed: int | None = None,
    calendar: tuple[str, ...] = SEASON_2026_CALENDAR,
) -> dict[str, Any]:
    """便捷函数: 生成默认 22 车 11 队赛季并仿真."""
    drivers = [
        SeasonDriver(
            driver_id=f"d{i:02d}",
            driver_name=f"Driver {i + 1}",
            team_id=f"t{i // 2:02d}",
            driver_aggression=0.4 + (i % 5) * 0.1,
            driver_smoothness=0.5 + (i % 4) * 0.1,
            driver_consistency=0.6 + (i % 3) * 0.1,
            driver_tire_management=0.4 + (i % 6) * 0.1,  # 0.4-0.9 spread
        )
        for i in range(n_drivers)
    ]
    teams = [(f"t{i:02d}", f"Team {i + 1}") for i in range(n_teams)]
    sim = SeasonSimulator(drivers=drivers, teams=teams, calendar=calendar, seed=seed)
    return sim.run()


def build_2026_season_drivers() -> list[SeasonDriver]:
    """用真实 F1 2026 车手档案 + 车队性能构建 SeasonDriver 列表 (Iter-36).

    整合 :mod:`f1opt.data.drivers_2026` 与 :mod:`f1opt.data.teams_2026`,
    让赛季仿真器使用真实车手能力评分 + 真实车队赛车性能偏移.

    返回 22 位真实 F1 2026 车手配置.
    """
    from f1opt.data.drivers_2026 import all_drivers_2026
    from f1opt.data.teams_2026 import pace_offset_for_team

    season_drivers: list[SeasonDriver] = []
    for d in all_drivers_2026():
        try:
            offset = pace_offset_for_team(d.team_id)
        except ValueError:
            offset = 0.0
        season_drivers.append(SeasonDriver(
            driver_id=d.driver_id,
            driver_name=d.driver_name,
            team_id=d.team_id,
            driver_aggression=d.aggression,
            driver_smoothness=d.smoothness,
            driver_consistency=d.driver_consistency,
            driver_tire_management=d.driver_tire_management,
            car_performance_offset_s=offset,
        ))
    return season_drivers


def build_2026_season_teams() -> list[tuple[str, str]]:
    """返回 F1 2026 真实 11 支车队 (team_id, team_name) (Iter-36)."""
    from f1opt.data.teams_2026 import all_teams_2026_profiles
    return [(t.team_id, t.team_name) for t in all_teams_2026_profiles()]


def simulate_season_2026(seed: int | None = None) -> dict[str, Any]:
    """用真实 F1 2026 车手 + 车队数据仿真完整赛季 (Iter-36).

    返回赛季结果 (世界冠军 + 积分榜 + 每场摘要).
    """
    sim = SeasonSimulator(
        drivers=build_2026_season_drivers(),
        teams=build_2026_season_teams(),
        seed=seed,
    )
    return sim.run()
