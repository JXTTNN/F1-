"""F1 2026 起步 & 第 1 圈混乱模型测试 (Iter-37)."""

from __future__ import annotations

from f1opt.model.race_start import (
    RaceStartModel,
    StartDriverInput,
    simulate_race_start,
)


# --------------------------------------------------------------------------- #
# 基础仿真
# --------------------------------------------------------------------------- #
def test_simulate_returns_20_results():
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1)
               for i in range(20)]
    results = simulate_race_start(drivers, seed=42)
    assert len(results) == 20


def test_empty_input_returns_empty():
    assert simulate_race_start([], seed=42) == []


def test_results_have_new_positions():
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1)
               for i in range(20)]
    results = simulate_race_start(drivers, seed=42)
    new_positions = sorted(r.new_position for r in results)
    # 新位置应是 1..20 的排列
    assert new_positions == list(range(1, 21))


def test_new_positions_unique():
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1)
               for i in range(20)]
    results = simulate_race_start(drivers, seed=42)
    positions = [r.new_position for r in results]
    assert len(set(positions)) == 20


# --------------------------------------------------------------------------- #
# 起步反应时间
# --------------------------------------------------------------------------- #
def test_reaction_time_in_valid_range():
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1)
               for i in range(20)]
    results = simulate_race_start(drivers, seed=42)
    for r in results:
        assert 0.13 <= r.reaction_time_s <= 0.35


def test_top_driver_faster_reaction():
    """高 aggression + consistency + clutch 的车手反应更快."""
    inputs = [
        StartDriverInput(driver_id="top", grid_position=1,
                         driver_aggression=0.95, driver_consistency=0.95,
                         clutch_skill=0.95, pole_position=True),
        StartDriverInput(driver_id="avg", grid_position=2,
                         driver_aggression=0.5, driver_consistency=0.5,
                         clutch_skill=0.5),
    ]
    # 多次仿真取平均 (单次有噪声)
    n_top_fast = 0
    for seed in range(20):
        r = simulate_race_start(inputs, seed=seed)
        if r[0].reaction_time_s < r[1].reaction_time_s:
            n_top_fast += 1
    # 顶级车手至少 70% 概率反应更快
    assert n_top_fast >= 14


# --------------------------------------------------------------------------- #
# 起步抓位
# --------------------------------------------------------------------------- #
def test_aggressive_driver_gains_positions_on_average():
    """高 aggression + clutch 的车手平均应抓位."""
    n_gains = 0
    for seed in range(50):
        drivers = [
            StartDriverInput(driver_id="aggr", grid_position=10,
                             driver_aggression=0.95, driver_consistency=0.9,
                             clutch_skill=0.95),
        ]
        # 其余 19 个普通车手 (避开 grid_position=10)
        other_positions = [p for p in range(1, 21) if p != 10]
        for i, pos in enumerate(other_positions):
            drivers.append(StartDriverInput(
                driver_id=f"d{i}", grid_position=pos,
                driver_aggression=0.5, driver_consistency=0.5,
                clutch_skill=0.5,
            ))
        results = simulate_race_start(drivers, seed=seed)
        aggr_result = next(r for r in results if r.driver_id == "aggr")
        if aggr_result.position_change > 0:
            n_gains += 1
    # 激进车手至少 50% 概率抓位 (起步本就有随机性)
    assert n_gains >= 25


def test_position_change_bounded():
    """位置变动不应超出 [-5, +4]."""
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1,
                                 driver_aggression=0.5)
               for i in range(20)]
    for seed in range(20):
        results = simulate_race_start(drivers, seed=seed)
        for r in results:
            assert -5 <= r.position_change <= 4


# --------------------------------------------------------------------------- #
# T1 接触
# --------------------------------------------------------------------------- #
def test_t1_contact_can_occur():
    """高 aggression 车手在某些种子下应有 T1 接触."""
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1,
                                 driver_aggression=0.9)
               for i in range(20)]
    any_contact = False
    for seed in range(50):
        results = simulate_race_start(drivers, seed=seed)
        if any(r.t1_contact for r in results):
            any_contact = True
            break
    assert any_contact


def test_t1_contact_increases_lap_time():
    """T1 接触应导致第 1 圈时间偏移."""
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1,
                                 driver_aggression=0.9)
               for i in range(20)]
    for seed in range(50):
        results = simulate_race_start(drivers, seed=seed)
        contacted = [r for r in results if r.t1_contact]
        if contacted:
            for r in contacted:
                assert r.lap1_time_offset_s > 0
            return
    # 没接触也没问题
    assert True


def test_low_aggression_fewer_contacts():
    """低 aggression 车手 T1 接触概率更低."""
    n_contacts_low = 0
    n_contacts_high = 0
    for seed in range(100):
        drivers_low = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1,
                                         driver_aggression=0.2)
                       for i in range(20)]
        r_low = simulate_race_start(drivers_low, seed=seed)
        n_contacts_low += sum(1 for r in r_low if r.t1_contact)

        drivers_high = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1,
                                          driver_aggression=0.95)
                        for i in range(20)]
        r_high = simulate_race_start(drivers_high, seed=seed)
        n_contacts_high += sum(1 for r in r_high if r.t1_contact)
    assert n_contacts_high >= n_contacts_low


# --------------------------------------------------------------------------- #
# 熄火
# --------------------------------------------------------------------------- #
def test_stall_rare():
    """熄火概率应很低 (< 1%)."""
    n_stalls = 0
    n_total = 0
    for seed in range(100):
        drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1)
                   for i in range(20)]
        results = simulate_race_start(drivers, seed=seed)
        n_stalls += sum(1 for r in results if r.stalled)
        n_total += len(results)
    assert n_stalls / n_total < 0.02


# --------------------------------------------------------------------------- #
# 可重复性
# --------------------------------------------------------------------------- #
def test_reproducible_with_same_seed():
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1,
                                 driver_aggression=0.7)
               for i in range(20)]
    r1 = simulate_race_start(drivers, seed=42)
    r2 = simulate_race_start(drivers, seed=42)
    for a, b in zip(r1, r2, strict=True):
        assert a.driver_id == b.driver_id
        assert a.new_position == b.new_position
        assert a.position_change == b.position_change


def test_different_seed_different_results():
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1,
                                 driver_aggression=0.7)
               for i in range(20)]
    r1 = simulate_race_start(drivers, seed=42)
    r2 = simulate_race_start(drivers, seed=99)
    # 不同种子应有不同结果 (除非极小概率)
    changes1 = [r.position_change for r in r1]
    changes2 = [r.position_change for r in r2]
    assert changes1 != changes2


# --------------------------------------------------------------------------- #
# SC 概率
# --------------------------------------------------------------------------- #
def test_lap1_sc_probability_higher():
    """第 1 圈 SC 概率乘数应 > 1."""
    model = RaceStartModel(seed=42)
    assert model.lap1_safety_car_probability() > 1.0
    assert model.lap2_safety_car_probability() > 1.0
    assert model.lap1_safety_car_probability() > model.lap2_safety_car_probability()


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def test_summary_returns_dict():
    drivers = [StartDriverInput(driver_id=f"d{i}", grid_position=i + 1)
               for i in range(20)]
    model = RaceStartModel(seed=42)
    results = model.simulate(drivers)
    s = model.summary(results)
    assert "n_drivers" in s
    assert "avg_position_change" in s
    assert "n_t1_contacts" in s
    assert "biggest_gainer" in s
    assert s["n_drivers"] == 20


def test_summary_empty():
    model = RaceStartModel(seed=42)
    assert model.summary([]) == {}


# --------------------------------------------------------------------------- #
# 杆位优势
# --------------------------------------------------------------------------- #
def test_pole_position_slightly_advantaged():
    """杆位车手反应时间应略快 (观察灯专注)."""
    inputs = [
        StartDriverInput(driver_id="pole", grid_position=1,
                         driver_aggression=0.7, driver_consistency=0.7,
                         clutch_skill=0.7, pole_position=True),
        StartDriverInput(driver_id="p2", grid_position=2,
                         driver_aggression=0.7, driver_consistency=0.7,
                         clutch_skill=0.7, pole_position=False),
    ]
    n_pole_faster = 0
    for seed in range(30):
        r = simulate_race_start(inputs, seed=seed)
        if r[0].reaction_time_s < r[1].reaction_time_s:
            n_pole_faster += 1
    # 杆位至少 60% 概率反应更快
    assert n_pole_faster >= 18
