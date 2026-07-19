"""F1 2026 — Sprint 周末格式 (Iter-32).

FIA 2026 Sprint 周末格式 (与 F1 2026 EA Sports 一致):

1. **Sprint 周末共 6 场**: China, Miami, Montreal, Silverstone, Zandvoort, Singapore.
2. **周五**: FP1 (60min) + Sprint Qualifying (3 阶段, 决定 Sprint 发车).
3. **周六**: Sprint (100km, ~30min, 无强制进站) + 正赛 Qualifying.
4. **周日**: 正赛 (完整距离).
5. **Sprint 积分**: 1-8 名 8-7-6-5-4-3-2-1 分 (短赛).
6. **2026 新规**: Sprint 不再决定周日正赛发车位 (2024 改革).

公开 API:
    - :class:`SprintWeekendSimulator` — 整 Sprint 周末仿真.
    - :func:`simulate_sprint_weekend` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from f1opt.data.setup_schema import CarSetup
from f1opt.model.championship import RACE_POINTS, SPRINT_POINTS
from f1opt.model.qualifying import (
    DriverQualifyingInput,
    QualifyingSession,
)
from f1opt.model.race_simulator import (
    RaceCar,
    RaceSimulation,
    RaceStrategy,
)

_SPRINT_DISTANCE_KM = 100.0
_SPRINT_LAPS_DEFAULT = 19  # ~100km / 5.3km avg


@dataclass
class SprintResult:
    """Sprint 赛结果."""

    track_id: str
    sprint_grid: list[str]
    """Sprint 发车顺序 (driver_id)."""
    sprint_classification: list[tuple[int, str, str]]
    """(position, driver_id, driver_name) — 含 DNF."""
    sprint_points: dict[str, int]
    """driver_id → sprint 积分."""


@dataclass
class SprintWeekendSimulator:
    """Sprint 周末仿真器 (Iter-32)."""

    track_id: str
    drivers: list[DriverQualifyingInput]
    """全部 20 车手."""
    race_setups: dict[str, CarSetup]
    """driver_id → 正赛 setup."""
    sprint_total_laps: int = _SPRINT_LAPS_DEFAULT
    race_total_laps: int = 58
    seed: int | None = None
    championship_context: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def run_sprint(self) -> SprintResult:
        """运行 Sprint 周末的 Sprint 部分."""
        # 1. Sprint Qualifying (3 阶段)
        sq_sess = QualifyingSession(
            track_id=self.track_id,
            drivers=self.drivers,
            seed=(self.seed or 0) + 100,
        )
        sq_grid = sq_sess.run()
        sprint_grid_order = [r.driver_id for r in sq_grid]

        # 2. Sprint Race (无强制进站, 单 stint)
        # Sprint strategy: 0 stops, 1 compound全程
        sprint_strategies = {
            d_id: RaceStrategy(pit_laps=(), compounds=("medium",))
            for d_id in sprint_grid_order
        }
        cars: list[RaceCar] = []
        for grid_pos, d_id in enumerate(sprint_grid_order, start=1):
            d_input = next(d for d in self.drivers if d.driver_id == d_id)
            setup = self.race_setups.get(d_id, d_input.setup)
            cars.append(RaceCar(
                driver_id=d_id,
                driver_name=d_input.driver_name,
                setup=setup,
                grid_position=grid_pos,
                strategy=sprint_strategies[d_id],
                driver_aggression=d_input.aggression,
                driver_smoothness=d_input.smoothness,
                driver_consistency=d_input.consistency,
            ))

        # Sprint 通常无 SC + 干地 (简化)
        sim = RaceSimulation(
            track_id=self.track_id,
            cars=cars,
            total_laps=self.sprint_total_laps,
            seed=(self.seed or 0) + 200,
        )
        results = sim.run()

        # 3. 分配 Sprint 积分 (1-8 名)
        sprint_points: dict[str, int] = {d_id: 0 for d_id in sprint_grid_order}
        classification: list[tuple[int, str, str]] = []
        for pos, car in results:
            classification.append((pos, car.driver_id, car.driver_name))
            if not car.retired and pos <= 8:
                sprint_points[car.driver_id] = SPRINT_POINTS[pos - 1]

        return SprintResult(
            track_id=self.track_id,
            sprint_grid=sprint_grid_order,
            sprint_classification=classification,
            sprint_points=sprint_points,
        )

    # ------------------------------------------------------------------ #
    def run_full_weekend(self) -> dict[str, Any]:
        """运行完整 Sprint 周末: Sprint + 正赛 Qualifying + 正赛."""
        # Sprint
        sprint_result = self.run_sprint()

        # 正赛 Qualifying
        rq_sess = QualifyingSession(
            track_id=self.track_id,
            drivers=self.drivers,
            seed=(self.seed or 0) + 300,
        )
        race_grid = rq_sess.run()
        race_grid_order = [r.driver_id for r in race_grid]

        # 正赛 — 完整距离
        cars: list[RaceCar] = []
        for grid_pos, d_id in enumerate(race_grid_order, start=1):
            d_input = next(d for d in self.drivers if d.driver_id == d_id)
            setup = self.race_setups.get(d_id, d_input.setup)
            # Sprint 周末正赛默认 1-stop
            strategy = RaceStrategy(
                pit_laps=(self.race_total_laps // 2,),
                compounds=("medium", "hard"),
            )
            cars.append(RaceCar(
                driver_id=d_id,
                driver_name=d_input.driver_name,
                setup=setup,
                grid_position=grid_pos,
                strategy=strategy,
                driver_aggression=d_input.aggression,
                driver_smoothness=d_input.smoothness,
                driver_consistency=d_input.consistency,
            ))

        race_sim = RaceSimulation(
            track_id=self.track_id,
            cars=cars,
            total_laps=self.race_total_laps,
            seed=(self.seed or 0) + 400,
        )
        race_results = race_sim.run()

        # 正赛积分
        race_points: dict[str, int] = {d_id: 0 for d_id in race_grid_order}
        race_classification: list[tuple[int, str, str]] = []
        for pos, car in race_results:
            race_classification.append((pos, car.driver_id, car.driver_name))
            if not car.retired and pos <= 10:
                race_points[car.driver_id] = RACE_POINTS[pos - 1]

        # 总积分 = Sprint + 正赛
        total_points = {
            d_id: sprint_result.sprint_points.get(d_id, 0)
                  + race_points.get(d_id, 0)
            for d_id in race_grid_order
        }

        return {
            "track_id": self.track_id,
            "sprint": sprint_result,
            "race_qualifying_grid": race_grid_order,
            "race_classification": race_classification,
            "sprint_points": sprint_result.sprint_points,
            "race_points": race_points,
            "total_points": total_points,
            "weekend_winner": max(total_points.items(), key=lambda x: x[1])[0],
        }


def simulate_sprint_weekend(
    track_id: str,
    drivers: list[DriverQualifyingInput],
    race_setups: dict[str, CarSetup] | None = None,
    seed: int | None = None,
    sprint_total_laps: int = _SPRINT_LAPS_DEFAULT,
    race_total_laps: int = 58,
) -> dict[str, Any]:
    """便捷函数: 运行完整 Sprint 周末."""
    if race_setups is None:
        race_setups = {d.driver_id: d.setup for d in drivers}
    sim = SprintWeekendSimulator(
        track_id=track_id,
        drivers=drivers,
        race_setups=race_setups,
        seed=seed,
        sprint_total_laps=sprint_total_laps,
        race_total_laps=race_total_laps,
    )
    return sim.run_full_weekend()
