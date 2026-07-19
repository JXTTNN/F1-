"""EA F1 2026 完整比赛周末仿真器 (Iter-63).

编排完整 F1 2026 比赛周末: FP1 / FP2 / FP3 / 排位赛 / 正赛.
每环节独立仿真, 汇总为周末报告. 严格契合 EA F1 2026 物理引擎.

**周末结构 (FIA 2026)**:
- FP1: 自由练习 1 (60 min, 长 stint, 高油量, 评估轮胎)
- FP2: 自由练习 2 (60 min, 排位模拟, 低油量, PARTY 模式)
- FP3: 自由练习 3 (60 min, 正赛模拟, 中油量)
- 排位赛: Q1/Q2/Q3 (单圈极速, 低油量, PARTY 模式, DRS 全程)
- 正赛: 多 stint + SC/VSC + 进站策略

公开 API:
    - :class:`RaceWeekend2026` — 完整周末仿真器.
    - :class:`WeekendReport2026` — 周末报告 (含各环节结果).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from f1opt.model.fuel_model import FuelMode
from f1opt.model.lap_simulator_2026 import (
    LapConfig2026,
    LapResult2026,
    LapSimulator2026,
    MultiStintSimulator2026,
    StintPlan2026,
    simulate_lap_2026,
)
from f1opt.model.pu_2026 import BATTERY_CAPACITY_MJ, PUDeployMode
from f1opt.model.safety_car import SafetyCarModel


# --------------------------------------------------------------------------- #
# 周末报告
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionResult2026:
    """单环节结果."""

    name: str  # "fp1" / "fp2" / "fp3" / "qualifying" / "race"
    laps: int
    best_lap_s: float
    avg_lap_s: float
    total_time_s: float
    compound: str
    notes: str = ""


@dataclass
class WeekendReport2026:
    """完整周末报告 (EA F1 2026)."""

    track_id: str
    team_id: str | None
    sessions: dict[str, SessionResult2026] = field(default_factory=dict)
    race_results: list[LapResult2026] = field(default_factory=list)
    race_pit_records: list[dict[str, Any]] = field(default_factory=list)
    qualifying_laps: list[LapResult2026] = field(default_factory=list)

    @property
    def qualifying_best_s(self) -> float:
        r = self.sessions.get("qualifying")
        return r.best_lap_s if r else 0.0

    @property
    def race_total_s(self) -> float:
        r = self.sessions.get("race")
        return r.total_time_s if r else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "team_id": self.team_id,
            "qualifying_best_s": self.qualifying_best_s,
            "race_total_s": self.race_total_s,
            "race_laps": len(self.race_results),
            "race_pit_stops": len(self.race_pit_records),
            "sessions": {
                name: {
                    "laps": s.laps,
                    "best_lap_s": s.best_lap_s,
                    "avg_lap_s": s.avg_lap_s,
                    "compound": s.compound,
                }
                for name, s in self.sessions.items()
            },
        }


# --------------------------------------------------------------------------- #
# RaceWeekend2026
# --------------------------------------------------------------------------- #
@dataclass
class RaceWeekend2026:
    """EA F1 2026 完整比赛周末仿真器.

    编排 FP1/FP2/FP3/排位/正赛, 每环节独立仿真, 汇总周末报告.

    用法::

        weekend = RaceWeekend2026(
            track_id="monza",
            team_id="rbr",
            race_plan=StintPlan2026(("medium","soft"), (30, 23)),
            race_safety_car=scm,
        )
        report = weekend.run()
        print(f"排位最佳: {report.qualifying_best_s:.3f}s")
        print(f"正赛总时: {report.race_total_s:.1f}s")
    """

    track_id: str
    team_id: str | None = None
    # 正赛配置
    race_plan: StintPlan2026 = field(
        default_factory=lambda: StintPlan2026(("medium", "soft"), (26, 26))
    )
    race_safety_car: SafetyCarModel | None = None
    race_total_laps: int = 52
    # 车手/赛车偏移
    driver_skill_offset_s: float = 0.0
    car_performance_offset_s: float = 0.0
    # 环境
    wet: bool = False
    track_temp_c: float = 30.0  # EA F1 2026 赛道温度 (°C)
    ambient_temp_c: float = 25.0  # 环境温度 (°C)

    _report: WeekendReport2026 | None = None

    # ------------------------------------------------------------------ #
    def run(self) -> WeekendReport2026:
        """运行完整周末, 返回报告."""
        self._report = WeekendReport2026(
            track_id=self.track_id, team_id=self.team_id
        )
        self._run_fp1()
        self._run_fp2()
        self._run_fp3()
        self._run_qualifying()
        self._run_race()
        return self._report

    # ------------------------------------------------------------------ #
    def _stint_result(
        self, name: str, laps: int, compound: str, fuel_kg: float,
        pu_mode: PUDeployMode, fuel_mode: FuelMode, session_type: str,
        notes: str = "",
    ) -> SessionResult2026:
        """运行单 stint 环节, 返回 SessionResult2026."""
        sim = LapSimulator2026(
            track_id=self.track_id, total_laps=laps, compound=compound,
            initial_fuel_kg=fuel_kg, pu_mode=pu_mode, fuel_mode=fuel_mode,
            wet=self.wet, session_type=session_type,
            driver_skill_offset_s=self.driver_skill_offset_s,
            car_performance_offset_s=self.car_performance_offset_s,
            track_temp_c=self.track_temp_c,
            ambient_temp_c=self.ambient_temp_c,
        )
        stint = sim.simulate_stint()
        times = [r.lap_time_s for r in stint]
        return SessionResult2026(
            name=name, laps=len(stint),
            best_lap_s=min(times) if times else 0.0,
            avg_lap_s=sum(times) / len(times) if times else 0.0,
            total_time_s=sum(times),
            compound=compound, notes=notes,
        )

    def _run_fp1(self) -> None:
        """FP1: 长 stint, 高油量, medium 胎, BALANCED 模式 (评估轮胎)."""
        r = self._stint_result(
            "fp1", laps=20, compound="medium", fuel_kg=110.0,
            pu_mode=PUDeployMode.BALANCED, fuel_mode=FuelMode.NORMAL,
            session_type="race", notes="高油量 stint 评估",
        )
        self._report.sessions["fp1"] = r

    def _run_fp2(self) -> None:
        """FP2: 排位模拟, 中油量, soft 胎, ATTACK 模式."""
        r = self._stint_result(
            "fp2", laps=15, compound="soft", fuel_kg=60.0,
            pu_mode=PUDeployMode.ATTACK, fuel_mode=FuelMode.RICH,
            session_type="qualifying", notes="排位模拟",
        )
        self._report.sessions["fp2"] = r

    def _run_fp3(self) -> None:
        """FP3: 正赛模拟, 中油量, medium 胎, BALANCED 模式."""
        r = self._stint_result(
            "fp3", laps=12, compound="medium", fuel_kg=80.0,
            pu_mode=PUDeployMode.BALANCED, fuel_mode=FuelMode.NORMAL,
            session_type="race", notes="正赛模拟",
        )
        self._report.sessions["fp3"] = r

    def _run_qualifying(self) -> None:
        """排位赛: 3 圈 flying lap, 低油量, PARTY 模式, QUALIFYING PU, DRS 全程.

        EA F1 2026 排位: 每圈满 SoC + 9MJ deploy + PARTY 燃油 + DRS 全程.
        取最佳圈为排位成绩.
        """
        laps: list[LapResult2026] = []
        for lap in range(1, 4):
            cfg = LapConfig2026(
                track_id=self.track_id,
                compound="soft",
                tire_age_laps=0,  # 每圈新软胎 (排位多套胎)
                current_fuel_kg=30.0,  # 低油量
                fuel_mode=FuelMode.PARTY,
                pu_mode=PUDeployMode.QUALIFYING,
                pu_soc_mj=BATTERY_CAPACITY_MJ,  # 每圈满 SoC
                wet=self.wet,
                gap_to_ahead_s=1.5,  # 净空
                session_type="qualifying",
                lap=lap,
                driver_skill_offset_s=self.driver_skill_offset_s,
                car_performance_offset_s=self.car_performance_offset_s,
                track_temp_c=self.track_temp_c,
                ambient_temp_c=self.ambient_temp_c,
                lap_in_stint=3,  # 排位圈 = 已暖胎状态 (out-lap 后)
            )
            laps.append(simulate_lap_2026(cfg))

        times = [r.lap_time_s for r in laps]
        self._report.sessions["qualifying"] = SessionResult2026(
            name="qualifying", laps=3,
            best_lap_s=min(times), avg_lap_s=sum(times) / len(times),
            total_time_s=sum(times), compound="soft",
            notes=f"3 flying laps, best={min(times):.3f}s",
        )
        self._report.qualifying_laps = laps

    def _run_race(self) -> None:
        """正赛: 多 stint + SC/VSC + 进站策略 (EA F1 2026 race physics)."""
        sim = MultiStintSimulator2026(
            track_id=self.track_id, plan=self.race_plan,
            initial_fuel_kg=110.0, pu_mode=PUDeployMode.BALANCED,
            fuel_mode=FuelMode.NORMAL, wet=self.wet,
            gap_to_ahead_s=0.8,  # race 跟车
            safety_car=self.race_safety_car, team_id=self.team_id,
            driver_skill_offset_s=self.driver_skill_offset_s,
            car_performance_offset_s=self.car_performance_offset_s,
            track_temp_c=self.track_temp_c,
            ambient_temp_c=self.ambient_temp_c,
        )
        race = sim.simulate_race()
        times = [r.lap_time_s for r in race]
        self._report.sessions["race"] = SessionResult2026(
            name="race", laps=len(race),
            best_lap_s=min(times) if times else 0.0,
            avg_lap_s=sum(times) / len(times) if times else 0.0,
            total_time_s=sum(times),
            compound="+".join(self.race_plan.compounds),
            notes=f"{self.race_plan.n_stops} stops",
        )
        self._report.race_results = race
        self._report.race_pit_records = list(sim._pit_records)
