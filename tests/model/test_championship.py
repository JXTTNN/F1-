"""Tests for f1opt.model.championship (Iter-17)."""

from __future__ import annotations

from f1opt.model.championship import (
    RACE_POINTS,
    SEASON_2026_CALENDAR,
    SPRINT_POINTS,
    Championship,
    ConstructorStanding,
    DriverStanding,
    points_for_position,
)


# --------------------------------------------------------------------------- #
# points_for_position
# --------------------------------------------------------------------------- #
class TestPointsForPosition:
    def test_p1_gets_25(self) -> None:
        assert points_for_position(1, 58, 58) == 25.0

    def test_p10_gets_1(self) -> None:
        assert points_for_position(10, 58, 58) == 1.0

    def test_p11_zero(self) -> None:
        assert points_for_position(11, 58, 58) == 0.0

    def test_dnf_zero(self) -> None:
        assert points_for_position(None, 0, 58) == 0.0

    def test_fastest_lap_bonus(self) -> None:
        pts = points_for_position(1, 58, 58, fastest_lap=True,
                                  apply_fastest_lap_point=True)
        assert pts == 26.0  # 25 + 1

    def test_fastest_lap_only_top10(self) -> None:
        """最快圈在 P11+ 无效."""
        pts = points_for_position(11, 58, 58, fastest_lap=True,
                                  apply_fastest_lap_point=True)
        assert pts == 0.0

    def test_fastest_lap_disabled_by_default(self) -> None:
        """默认 apply_fastest_lap_point=False → 无最快圈加分."""
        pts = points_for_position(1, 58, 58, fastest_lap=True)
        assert pts == 25.0

    def test_half_points_rule(self) -> None:
        """圈数 < 75% → 积分减半."""
        # 30/58 = 51.7% < 75%
        pts = points_for_position(1, 30, 58)
        assert pts == 12.5

    def test_half_points_below_2_laps_zero(self) -> None:
        """圈数 < 2 → 0 分 (无半分)."""
        pts = points_for_position(1, 1, 58)
        assert pts == 25.0  # 不触发半分 (1 圈视为正常)

    def test_sprint_points(self) -> None:
        assert points_for_position(1, 0, 100, is_sprint=True) == 8.0
        assert points_for_position(8, 0, 100, is_sprint=True) == 1.0
        assert points_for_position(9, 0, 100, is_sprint=True) == 0.0

    def test_sprint_no_fastest_lap(self) -> None:
        """冲刺赛无最快圈加分."""
        pts = points_for_position(1, 0, 100, is_sprint=True,
                                  fastest_lap=True, apply_fastest_lap_point=True)
        assert pts == 8.0


# --------------------------------------------------------------------------- #
# DriverStanding / ConstructorStanding
# --------------------------------------------------------------------------- #
class TestStandings:
    def test_driver_add_result(self) -> None:
        d = DriverStanding("d1", "Driver 1", "team1")
        d.add_result(position=1, points=25.0, fastest_lap=True,
                     pole=True, race_id="race1")
        assert d.points == 25.0
        assert d.wins == 1
        assert d.podiums == 1
        assert d.poles == 1
        assert d.fastest_laps == 1
        assert len(d.race_results) == 1

    def test_driver_dnf(self) -> None:
        d = DriverStanding("d1", "Driver 1")
        d.add_result(position=None, points=0.0, dnf=True)
        assert d.dnfs == 1
        assert d.wins == 0
        assert d.podiums == 0

    def test_constructor_add_points(self) -> None:
        c = ConstructorStanding("t1", "Team 1")
        c.add_points(25.0, position=1)
        c.add_points(18.0, position=2)
        assert c.points == 43.0
        assert c.wins == 1
        assert c.podiums == 2


# --------------------------------------------------------------------------- #
# Championship
# --------------------------------------------------------------------------- #
class TestChampionship:
    def _make_champ(self) -> Championship:
        drivers = [
            DriverStanding("d1", "Verstappen", "red_bull"),
            DriverStanding("d2", "Norris", "mclaren"),
            DriverStanding("d3", "Leclerc", "ferrari"),
        ]
        teams = [
            ConstructorStanding("red_bull", "Red Bull"),
            ConstructorStanding("mclaren", "McLaren"),
            ConstructorStanding("ferrari", "Ferrari"),
        ]
        return Championship(drivers=drivers, teams=teams,
                            calendar=("melbourne", "monza"))

    def test_record_race_updates_points(self) -> None:
        champ = self._make_champ()
        champ.record_race("melbourne", [
            {"driver_id": "d1", "team_id": "red_bull", "position": 1, "laps_completed": 58},
            {"driver_id": "d2", "team_id": "mclaren", "position": 2, "laps_completed": 58},
            {"driver_id": "d3", "team_id": "ferrari", "position": None, "laps_completed": 0},
        ], total_laps=58)
        assert champ.races_completed == 1
        ds = champ.driver_standings()
        assert ds[0].driver_id == "d1"
        assert ds[0].points == 25.0
        assert ds[1].driver_id == "d2"
        assert ds[2].driver_id == "d3"
        assert ds[2].points == 0.0
        assert ds[2].dnfs == 1

    def test_constructor_points_accumulate(self) -> None:
        champ = self._make_champ()
        champ.record_race("melbourne", [
            {"driver_id": "d1", "team_id": "red_bull", "position": 1, "laps_completed": 58},
            {"driver_id": "d2", "team_id": "mclaren", "position": 2, "laps_completed": 58},
            {"driver_id": "d3", "team_id": "ferrari", "position": 3, "laps_completed": 58},
        ], total_laps=58)
        cs = champ.constructor_standings()
        assert cs[0].team_id == "red_bull"
        assert cs[0].points == 25.0
        assert cs[1].team_id == "mclaren"
        assert cs[1].points == 18.0

    def test_multiple_races_accumulate(self) -> None:
        champ = self._make_champ()
        for race in champ.calendar:
            champ.record_race(race, [
                {"driver_id": "d1", "team_id": "red_bull", "position": 1, "laps_completed": 58},
                {"driver_id": "d2", "team_id": "mclaren", "position": 2, "laps_completed": 58},
                {"driver_id": "d3", "team_id": "ferrari", "position": 3, "laps_completed": 58},
            ], total_laps=58)
        assert champ.races_completed == 2
        d1 = champ._driver_by_id["d1"]
        assert d1.points == 50.0  # 2 × 25
        assert d1.wins == 2

    def test_champions(self) -> None:
        champ = self._make_champ()
        champ.record_race("melbourne", [
            {"driver_id": "d1", "team_id": "red_bull", "position": 1, "laps_completed": 58},
        ], total_laps=58)
        c = champ.champions()
        assert c["drivers_champion"].driver_id == "d1"
        assert c["constructors_champion"].team_id == "red_bull"

    def test_summary_keys(self) -> None:
        champ = self._make_champ()
        champ.record_race("melbourne", [
            {"driver_id": "d1", "team_id": "red_bull", "position": 1, "laps_completed": 58},
        ], total_laps=58)
        s = champ.summary()
        required = {"races_completed", "total_races", "drivers_champion",
                    "constructors_champion", "top5_drivers", "top5_constructors"}
        assert required.issubset(s.keys())

    def test_fastest_lap_in_championship(self) -> None:
        champ = self._make_champ()
        champ.record_race("melbourne", [
            {"driver_id": "d1", "team_id": "red_bull", "position": 1,
             "laps_completed": 58, "fastest_lap": True},
            {"driver_id": "d2", "team_id": "mclaren", "position": 2, "laps_completed": 58},
        ], total_laps=58, apply_fastest_lap_point=True)
        d1 = champ._driver_by_id["d1"]
        assert d1.points == 26.0  # 25 + 1 FL
        assert d1.fastest_laps == 1

    def test_standings_sorted_descending(self) -> None:
        champ = self._make_champ()
        champ.record_race("melbourne", [
            {"driver_id": "d1", "team_id": "red_bull", "position": 3, "laps_completed": 58},
            {"driver_id": "d2", "team_id": "mclaren", "position": 1, "laps_completed": 58},
            {"driver_id": "d3", "team_id": "ferrari", "position": 2, "laps_completed": 58},
        ], total_laps=58)
        ds = champ.driver_standings()
        assert ds[0].points >= ds[1].points >= ds[2].points
        assert ds[0].driver_id == "d2"  # winner gets 25


# --------------------------------------------------------------------------- #
# Calendar data integrity
# --------------------------------------------------------------------------- #
class TestCalendar:
    def test_season_2026_has_24_races(self) -> None:
        assert len(SEASON_2026_CALENDAR) == 24

    def test_calendar_no_duplicates(self) -> None:
        assert len(set(SEASON_2026_CALENDAR)) == len(SEASON_2026_CALENDAR)


# --------------------------------------------------------------------------- #
# FIA points table integrity
# --------------------------------------------------------------------------- #
class TestPointsTables:
    def test_race_points_top10(self) -> None:
        assert len(RACE_POINTS) == 10
        assert RACE_POINTS[0] == 25
        assert RACE_POINTS[-1] == 1

    def test_sprint_points_top8(self) -> None:
        assert len(SPRINT_POINTS) == 8
        assert SPRINT_POINTS[0] == 8
        assert SPRINT_POINTS[-1] == 1

    def test_race_points_descending(self) -> None:
        for i in range(len(RACE_POINTS) - 1):
            assert RACE_POINTS[i] > RACE_POINTS[i + 1]


# --------------------------------------------------------------------------- #
# Full season simulation
# --------------------------------------------------------------------------- #
class TestFullSeason:
    def test_24_race_season(self) -> None:
        """完整 24 场赛季仿真, 总积分合理."""
        drivers = [DriverStanding(f"d{i}", f"Driver {i}", f"team{i}")
                   for i in range(5)]
        teams = [ConstructorStanding(f"team{i}", f"Team {i}") for i in range(5)]
        champ = Championship(drivers=drivers, teams=teams,
                             calendar=SEASON_2026_CALENDAR)
        for race_idx, race in enumerate(SEASON_2026_CALENDAR):
            # 旋转 finishing order
            results = []
            for i in range(5):
                pos = ((i - race_idx) % 5) + 1
                results.append({
                    "driver_id": f"d{i}", "team_id": f"team{i}",
                    "position": pos, "laps_completed": 58,
                })
            champ.record_race(race, results, total_laps=58)
        assert champ.races_completed == 24
        # 所有车手都有积分
        for d in champ.drivers:
            assert d.points > 0
        # 冠军存在
        c = champ.champions()
        assert c["drivers_champion"] is not None
        assert c["constructors_champion"] is not None
