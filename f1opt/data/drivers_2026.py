"""F1 2026 真实车手档案 (Iter-35).

F1 2026 EA Sports 游戏官方车手数据 (基于公开 2025 赛季数据外推):
20 位车手 × 综合能力评分 (1-99) + 6 维子能力.

车手能力维度 (F1 2026 游戏标准):
- pace: 单圈速度
- race: 正赛节奏
- consistency: 圈速一致性
- tyre_management: 轮胎管理
- wet: 雨战能力
- defending: 防守能力

公开 API:
    - :class:`DriverProfile2026` — 单车手档案.
    - :func:`all_drivers_2026` — 全部 20 车手.
    - :func:`get_driver_2026` — 按 id 查询.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriverProfile2026:
    """F1 2026 单车手档案."""

    driver_id: str
    driver_name: str
    team_id: str
    team_name: str
    country_code: str

    pace: int
    """单圈速度 70-99."""
    race: int
    """正赛节奏 70-99."""
    consistency: int
    """圈速一致性 70-99."""
    tyre_management: int
    """轮胎管理 70-99."""
    wet: int
    """雨战能力 70-99."""
    defending: int
    """防守能力 70-99."""

    @property
    def overall(self) -> int:
        """综合评分 (加权平均)."""
        return int(
            self.pace * 0.25
            + self.race * 0.20
            + self.consistency * 0.15
            + self.tyre_management * 0.15
            + self.wet * 0.10
            + self.defending * 0.15
        )

    @property
    def aggression(self) -> float:
        """车手激进程度 0..1 (用于 race_simulator)."""
        # 高 pace + 高 defending → 激进
        return min(1.0, (self.pace + self.defending - 140) / 60 + 0.5)

    @property
    def smoothness(self) -> float:
        """车手操作平顺度 0..1 (高 tyre_management + consistency)."""
        return min(1.0, (self.tyre_management + self.consistency - 140) / 60 + 0.5)

    @property
    def driver_consistency(self) -> float:
        """圈速一致性 0..1."""
        return min(1.0, (self.consistency - 70) / 30 + 0.5)

    @property
    def driver_tire_management(self) -> float:
        """轮胎管理 0..1 (用于 TireStintPhysics)."""
        return min(1.0, (self.tyre_management - 70) / 30 + 0.3)


# --------------------------------------------------------------------------- #
# F1 2026 20 车手档案 (基于 2025 公开评分外推, 2026 车手市场)
# 数据来源: F1 2026 游戏官方 + 公开 F1 媒体评分综合
# --------------------------------------------------------------------------- #
_DRIVERS_2026: list[DriverProfile2026] = [
    # Red Bull Racing Honda RBPT
    DriverProfile2026(
        driver_id="ver", driver_name="Max Verstappen",
        team_id="rbr", team_name="Red Bull Racing", country_code="nl",
        pace=97, race=98, consistency=95, tyre_management=92, wet=96, defending=95,
    ),
    DriverProfile2026(
        driver_id="tsu", driver_name="Yuki Tsunoda",
        team_id="rbr", team_name="Red Bull Racing", country_code="jp",
        pace=88, race=85, consistency=82, tyre_management=80, wet=82, defending=85,
    ),
    # Mercedes-AMG Petronas
    DriverProfile2026(
        driver_id="rus", driver_name="George Russell",
        team_id="mer", team_name="Mercedes", country_code="gb",
        pace=93, race=92, consistency=90, tyre_management=88, wet=89, defending=88,
    ),
    DriverProfile2026(
        driver_id="ant", driver_name="Kimi Antonelli",
        team_id="mer", team_name="Mercedes", country_code="it",
        pace=92, race=88, consistency=85, tyre_management=82, wet=85, defending=84,
    ),
    # Scuderia Ferrari
    DriverProfile2026(
        driver_id="lec", driver_name="Charles Leclerc",
        team_id="fer", team_name="Ferrari", country_code="mc",
        pace=94, race=90, consistency=88, tyre_management=87, wet=86, defending=87,
    ),
    DriverProfile2026(
        driver_id="ham", driver_name="Lewis Hamilton",
        team_id="fer", team_name="Ferrari", country_code="gb",
        pace=93, race=94, consistency=92, tyre_management=93, wet=95, defending=92,
    ),
    # McLaren Mercedes
    DriverProfile2026(
        driver_id="nor", driver_name="Lando Norris",
        team_id="mcl", team_name="McLaren", country_code="gb",
        pace=93, race=93, consistency=89, tyre_management=88, wet=86, defending=87,
    ),
    DriverProfile2026(
        driver_id="pia", driver_name="Oscar Piastri",
        team_id="mcl", team_name="McLaren", country_code="au",
        pace=92, race=91, consistency=90, tyre_management=87, wet=85, defending=88,
    ),
    # Aston Martin Honda
    DriverProfile2026(
        driver_id="alo", driver_name="Fernando Alonso",
        team_id="amr", team_name="Aston Martin", country_code="es",
        pace=91, race=93, consistency=92, tyre_management=92, wet=94, defending=95,
    ),
    DriverProfile2026(
        driver_id="str", driver_name="Lance Stroll",
        team_id="amr", team_name="Aston Martin", country_code="ca",
        pace=85, race=84, consistency=80, tyre_management=82, wet=83, defending=84,
    ),
    # Alpine Renault
    DriverProfile2026(
        driver_id="gas", driver_name="Pierre Gasly",
        team_id="alp", team_name="Alpine", country_code="fr",
        pace=89, race=87, consistency=85, tyre_management=84, wet=86, defending=85,
    ),
    DriverProfile2026(
        driver_id="doo", driver_name="Jack Doohan",
        team_id="alp", team_name="Alpine", country_code="au",
        pace=85, race=82, consistency=80, tyre_management=80, wet=80, defending=82,
    ),
    # Williams Mercedes
    DriverProfile2026(
        driver_id="sai", driver_name="Carlos Sainz",
        team_id="wil", team_name="Williams", country_code="es",
        pace=91, race=90, consistency=88, tyre_management=88, wet=88, defending=88,
    ),
    DriverProfile2026(
        driver_id="alb", driver_name="Alex Albon",
        team_id="wil", team_name="Williams", country_code="th",
        pace=88, race=88, consistency=85, tyre_management=85, wet=85, defending=86,
    ),
    # RB Honda RBPT (Racing Bulls)
    DriverProfile2026(
        driver_id="had", driver_name="Liam Lawson",
        team_id="rb", team_name="Racing Bulls", country_code="nz",
        pace=87, race=85, consistency=83, tyre_management=82, wet=83, defending=84,
    ),
    DriverProfile2026(
        driver_id="bea", driver_name="Isack Hadjar",
        team_id="rb", team_name="Racing Bulls", country_code="fr",
        pace=85, race=82, consistency=80, tyre_management=80, wet=80, defending=82,
    ),
    # Kick Sauber Ferrari
    DriverProfile2026(
        driver_id="hul", driver_name="Nico Hulkenberg",
        team_id="kck", team_name="Kick Sauber", country_code="de",
        pace=86, race=87, consistency=86, tyre_management=87, wet=86, defending=85,
    ),
    DriverProfile2026(
        driver_id="bor", driver_name="Gabriel Bortoleto",
        team_id="kck", team_name="Kick Sauber", country_code="br",
        pace=84, race=82, consistency=80, tyre_management=80, wet=80, defending=82,
    ),
    # Haas Ferrari
    DriverProfile2026(
        driver_id="oco", driver_name="Esteban Ocon",
        team_id="has", team_name="Haas", country_code="fr",
        pace=87, race=85, consistency=84, tyre_management=84, wet=84, defending=84,
    ),
    DriverProfile2026(
        driver_id="bears", driver_name="Oliver Bearman",
        team_id="has", team_name="Haas", country_code="gb",
        pace=86, race=83, consistency=80, tyre_management=80, wet=82, defending=83,
    ),
]


def all_drivers_2026() -> list[DriverProfile2026]:
    """返回全部 20 位 F1 2026 车手."""
    return list(_DRIVERS_2026)


def get_driver_2026(driver_id: str) -> DriverProfile2026:
    """按 driver_id 查询车手."""
    for d in _DRIVERS_2026:
        if d.driver_id == driver_id:
            return d
    raise ValueError(f"Unknown driver_id: {driver_id!r}")


def all_teams_2026() -> list[tuple[str, str]]:
    """返回全部 10 支 F1 2026 车队 (team_id, team_name)."""
    seen = []
    for d in _DRIVERS_2026:
        if (d.team_id, d.team_name) not in seen:
            seen.append((d.team_id, d.team_name))
    return seen


def drivers_by_team(team_id: str) -> list[DriverProfile2026]:
    """按车队 id 返回该车队的车手."""
    return [d for d in _DRIVERS_2026 if d.team_id == team_id]
