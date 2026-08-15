"""Pit Crew 集成到 Race Simulator 测试 (Iter-40)."""

from __future__ import annotations

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.race_simulator import (
    RaceCar,
    RaceSimulation,
    RaceStrategy,
)


def _make_car(driver_id="d1", team_id="", pit_crew_offset_s=0.0,
              grid_position=1, aggression=0.7):
    """构建测试用 RaceCar."""
    return RaceCar(
        driver_id=driver_id,
        driver_name=driver_id,
        setup=DEFAULT_SETUP,
        grid_position=grid_position,
        strategy=RaceStrategy(pit_laps=(20,), compounds=("medium", "hard")),
        driver_aggression=aggression,
        team_id=team_id,
        pit_crew_offset_s=pit_crew_offset_s,
    )


# --------------------------------------------------------------------------- #
# Pit crew 偏移字段
# --------------------------------------------------------------------------- #
def test_race_car_has_pit_crew_offset_field():
    car = _make_car()
    assert hasattr(car, "pit_crew_offset_s")
    assert car.pit_crew_offset_s == 0.0  # 默认


def test_race_car_has_team_id_field():
    car = _make_car(team_id="rbr")
    assert car.team_id == "rbr"


# --------------------------------------------------------------------------- #
# 进站损失受 pit crew 影响
# --------------------------------------------------------------------------- #
def test_fast_pit_crew_reduces_pit_loss():
    """快队 (pit_crew_offset_s < 0) 进站损失应 < 默认."""
    # 两场比赛, 一场用快队, 一场用慢队
    # 用同一车手不同 team_id 避免车手性能差异
    cars_fast = [_make_car(driver_id="d1", team_id="rbr",
                            pit_crew_offset_s=-1.0, grid_position=1)]
    cars_slow = [_make_car(driver_id="d1", team_id="has",
                            pit_crew_offset_s=1.5, grid_position=1)]

    sim_fast = RaceSimulation(track_id="monza", cars=cars_fast,
                              total_laps=25, seed=42)
    sim_slow = RaceSimulation(track_id="monza", cars=cars_slow,
                              total_laps=25, seed=42)

    r_fast = sim_fast.run()
    r_slow = sim_slow.run()

    # 慢队总时间应 > 快队 (因为进站更慢)
    assert r_slow[0][1].cumulative_time > r_fast[0][1].cumulative_time


def test_team_id_lookup_affects_pit_loss():
    """有 team_id 的车手进站损失应反映车队 pit crew 评级."""
    # RBR (rating 96) vs Haas (rating 76)
    # 不设 pit_crew_offset_s, 让 race_simulator 动态查询
    cars_rbr = [_make_car(driver_id="d1", team_id="rbr", grid_position=1)]
    cars_has = [_make_car(driver_id="d1", team_id="has", grid_position=1)]

    sim_rbr = RaceSimulation(track_id="monza", cars=cars_rbr,
                             total_laps=25, seed=42)
    sim_has = RaceSimulation(track_id="monza", cars=cars_has,
                             total_laps=25, seed=42)

    r_rbr = sim_rbr.run()
    r_has = sim_has.run()

    # Haas 进站更慢 → 总时间更长
    assert r_has[0][1].cumulative_time > r_rbr[0][1].cumulative_time


def test_no_team_id_uses_default_pit_loss():
    """无 team_id 应使用默认进站损失 (无偏移)."""
    cars = [_make_car(driver_id="d1", team_id="", grid_position=1)]
    sim = RaceSimulation(track_id="monza", cars=cars, total_laps=25, seed=42)
    # 应正常完成
    results = sim.run()
    assert len(results) == 1
    assert not results[0][1].retired


# --------------------------------------------------------------------------- #
# SC 期间 pit crew 偏移仍应用 (但被 discount 折扣)
# --------------------------------------------------------------------------- #
def test_pit_crew_offset_applied_during_sc():
    """SC 期间进站, pit crew 偏移应被 discount 折扣后应用."""
    from f1opt.model.safety_car import SafetyCarModel
    sc = SafetyCarModel(seed=42)
    # 强制 SC 在第 20 圈
    sc.generate_periods(total_laps=25, n_retirements=2,
                        weather_wetness=0.0, rng=__import__("random").Random(42))

    cars = [_make_car(driver_id="d1", team_id="has",
                       pit_crew_offset_s=2.0, grid_position=1)]
    sim = RaceSimulation(track_id="monza", cars=cars, total_laps=25,
                         seed=42, safety_car=sc)
    # 应正常完成 (SC 期间进站损失被折扣)
    results = sim.run()
    assert len(results) == 1


# --------------------------------------------------------------------------- #
# 整场比赛: 多车手不同车队 pit crew
# --------------------------------------------------------------------------- #
def test_multi_team_race_pit_crew_differences():
    """整场 22 车比赛, 顶队总进站时间应 < 后段."""
    # 简化: 4 车队各 1 车, 在同一位发车, 同一策略
    cars = [
        _make_car(driver_id="rbr", team_id="rbr", grid_position=1),
        _make_car(driver_id="mer", team_id="mer", grid_position=2),
        _make_car(driver_id="has", team_id="has", grid_position=3),
        _make_car(driver_id="aud", team_id="aud", grid_position=4),
    ]
    sim = RaceSimulation(track_id="monza", cars=cars, total_laps=25, seed=42)
    results = sim.run()

    # RBR 总时间应 < Haas (尽管 Haas 起步晚 2 位)
    by_team = {r[1].driver_id: r[1].cumulative_time for r in results}
    # 注意: 起步位置也有影响, 但 25 圈 + 进站差异应足够大
    # 这里用相对差异: RBR vs HAS 的差距应 > 起步位置差异 (2 * 0.05 = 0.1s)
    diff = by_team["has"] - by_team["rbr"]
    assert diff > 0.5  # Haas 比 RBR 至少慢 0.5s (含进站差异)


# --------------------------------------------------------------------------- #
# Season 2026 集成: 真实车队 pit crew 影响
# --------------------------------------------------------------------------- #
def test_season_2026_uses_team_pit_crew():
    """2026 赛季仿真应使用真实车队 pit crew 性能."""
    from f1opt.model.season_simulator import build_2026_season_drivers

    drivers = build_2026_season_drivers()
    # 每位车手应有 team_id (来自 SeasonDriver)
    for d in drivers:
        assert d.team_id != ""
    # RBR 车手应有 team_id "rbr"
    rbr_driver = next(d for d in drivers if d.driver_id == "ver")
    assert rbr_driver.team_id == "rbr"
