"""F1 锦标赛积分模型 (Iter-17).

F1 双重积分制 (FIA 2026 体育规则 §6.4):

- **正赛积分** (Top 10): 25-18-15-12-10-8-6-4-2-1.
- **最快圈积分**: +1 (需完赛且在 Top 10, 2019-2024 规则; 2025 起取消).
- **冲刺赛积分** (Top 8): 8-7-6-5-4-3-2-1.
- **半分规则**: 完成圈数 < 75% 但 ≥ 2 圈, 积分减半 (Red Flag 提前结束).
- **零分**: P11+ 或 DNF.

构造函数输入赛季 24 场 (2026 赛历), 模拟每场后累计车手 + 车队积分,
输出最终车手世界冠军 + 车队世界冠军.

公开 API:
    - :data:`RACE_POINTS` / :data:`SPRINT_POINTS` — FIA 积分表.
    - :class:`Championship` — 完整赛季仿真.
    - :class:`DriverStanding` / :class:`ConstructorStanding` — 单实体积分.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# FIA 积分表 (2026 体育规则)
# --------------------------------------------------------------------------- #
RACE_POINTS: tuple[int, ...] = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
SPRINT_POINTS: tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2, 1)
FASTEST_LAP_POINT = 1  # 2024 及之前; 2025 取消
_HALF_POINTS_FACTOR = 0.5

# 2026 赛季 24 场赛历 (FIA 官方)
SEASON_2026_CALENDAR: tuple[str, ...] = (
    "melbourne", "shanghai", "suzuka", "bahrain", "jeddah",
    "miami", "monaco", "montreal", "barcelona", "silverstone",
    "spa", "budapest", "amsterdam", "monza", "baku",
    "singapore", "austin", "losail", "madrid", "interlagos",
    "las_vegas", "yas_marina", "madrid_e", "suzuka_e",  # 后两场为示例占位
)[:24]  # 严格 24 场


# --------------------------------------------------------------------------- #
# Standings
# --------------------------------------------------------------------------- #
@dataclass
class DriverStanding:
    """车手积分状态."""

    driver_id: str
    driver_name: str
    team_id: str = ""
    points: float = 0.0
    race_results: list[dict[str, Any]] = field(default_factory=list)
    wins: int = 0
    podiums: int = 0
    poles: int = 0
    fastest_laps: int = 0
    dnfs: int = 0

    def add_result(self, position: int | None, points: float,
                   fastest_lap: bool = False, pole: bool = False,
                   dnf: bool = False, race_id: str = "") -> None:
        self.points += points
        self.race_results.append({
            "race_id": race_id, "position": position, "points": points,
            "fastest_lap": fastest_lap, "pole": pole, "dnf": dnf,
        })
        if position == 1:
            self.wins += 1
        if position is not None and position <= 3:
            self.podiums += 1
        if pole:
            self.poles += 1
        if fastest_lap:
            self.fastest_laps += 1
        if dnf:
            self.dnfs += 1


@dataclass
class ConstructorStanding:
    """车队积分状态."""

    team_id: str
    team_name: str
    points: float = 0.0
    wins: int = 0
    podiums: int = 0
    poles: int = 0

    def add_points(self, points: float, position: int | None = None) -> None:
        self.points += points
        if position == 1:
            self.wins += 1
        if position is not None and position <= 3:
            self.podiums += 1


# --------------------------------------------------------------------------- #
# 积分计算
# --------------------------------------------------------------------------- #
def points_for_position(
    position: int | None,
    n_laps_completed: int,
    total_laps: int,
    fastest_lap: bool = False,
    is_sprint: bool = False,
    apply_fastest_lap_point: bool = False,
) -> float:
    """计算单场积分.

    - position=None (DNF): 0 分.
    - 半分规则: 完赛圈数 < 75% 总圈数但 ≥ 2 → 积分 ×0.5.
    - fastest_lap: +1 (仅 Top 10 完赛, 且 apply_fastest_lap_point=True).
    """
    if position is None:
        return 0.0
    points_table = SPRINT_POINTS if is_sprint else RACE_POINTS
    if position < 1 or position > len(points_table):
        return 0.0
    base = float(points_table[position - 1])
    # 半分规则
    if n_laps_completed < total_laps * 0.75 and n_laps_completed >= 2:
        base *= _HALF_POINTS_FACTOR
    # 最快圈 (仅正赛, Top 10)
    if (fastest_lap and apply_fastest_lap_point and not is_sprint
            and position <= 10):
        base += FASTEST_LAP_POINT
    return base


# --------------------------------------------------------------------------- #
# Championship
# --------------------------------------------------------------------------- #
@dataclass
class Championship:
    """F1 赛季仿真: 累计车手 + 车队积分, 输出双世界冠军.

    用法::

        champ = Championship(drivers=[...], teams=[...], calendar=SEASON_2026_CALENDAR)
        champ.record_race(race_id="melbourne", results=[
            {"driver_id": "d1", "team_id": "ferrari", "position": 1, ...},
            ...
        ], total_laps=58)
        champ.standings()  # → {drivers: [...], constructors: [...]}
    """

    drivers: list[DriverStanding]
    teams: list[ConstructorStanding]
    calendar: tuple[str, ...] = SEASON_2026_CALENDAR
    races_completed: int = 0

    def __post_init__(self) -> None:
        self._driver_by_id = {d.driver_id: d for d in self.drivers}
        self._team_by_id = {t.team_id: t for t in self.teams}

    # ------------------------------------------------------------------ #
    def record_race(
        self,
        race_id: str,
        results: list[dict[str, Any]],
        total_laps: int,
        is_sprint: bool = False,
        apply_fastest_lap_point: bool = False,
    ) -> None:
        """记录一场比赛结果, 累计积分."""
        # 找最快圈车手 (若适用)
        fastest_lap_driver = None
        if apply_fastest_lap_point and not is_sprint:
            for r in results:
                if r.get("fastest_lap"):
                    fastest_lap_driver = r["driver_id"]
                    break
        for r in results:
            d_id = r["driver_id"]
            team_id = r.get("team_id", "")
            pos = r.get("position")  # None = DNF
            laps = r.get("laps_completed", total_laps if pos else 0)
            fl = (fastest_lap_driver == d_id) and pos is not None and pos <= 10
            pts = points_for_position(
                position=pos,
                n_laps_completed=laps,
                total_laps=total_laps,
                fastest_lap=fl,
                is_sprint=is_sprint,
                apply_fastest_lap_point=apply_fastest_lap_point,
            )
            # 车手积分
            if d_id in self._driver_by_id:
                d = self._driver_by_id[d_id]
                d.add_result(
                    position=pos, points=pts, fastest_lap=fl,
                    pole=r.get("pole", False), dnf=(pos is None),
                    race_id=race_id,
                )
            # 车队积分 (车手积分累加)
            if team_id and team_id in self._team_by_id:
                self._team_by_id[team_id].add_points(pts, position=pos)
        self.races_completed += 1

    # ------------------------------------------------------------------ #
    def driver_standings(self) -> list[DriverStanding]:
        """车手积分榜 (降序)."""
        return sorted(self.drivers, key=lambda d: -d.points)

    def constructor_standings(self) -> list[ConstructorStanding]:
        """车队积分榜 (降序)."""
        return sorted(self.teams, key=lambda t: -t.points)

    def standings(self) -> dict[str, list]:
        return {
            "drivers": self.driver_standings(),
            "constructors": self.constructor_standings(),
        }

    def champions(self) -> dict[str, Any]:
        """返回双世界冠军 (积分最高)."""
        ds = self.driver_standings()
        cs = self.constructor_standings()
        return {
            "drivers_champion": ds[0] if ds else None,
            "constructors_champion": cs[0] if cs else None,
            "races_completed": self.races_completed,
        }

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        ds = self.driver_standings()
        cs = self.constructor_standings()
        return {
            "races_completed": self.races_completed,
            "total_races": len(self.calendar),
            "drivers_champion": {
                "name": ds[0].driver_name, "points": ds[0].points,
                "wins": ds[0].wins,
            } if ds else None,
            "constructors_champion": {
                "name": cs[0].team_name, "points": cs[0].points,
                "wins": cs[0].wins,
            } if cs else None,
            "top5_drivers": [
                {"name": d.driver_name, "points": d.points,
                 "wins": d.wins, "podiums": d.podiums, "dnfs": d.dnfs}
                for d in ds[:5]
            ],
            "top5_constructors": [
                {"name": t.team_name, "points": t.points, "wins": t.wins}
                for t in cs[:5]
            ],
        }
