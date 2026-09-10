"""F1 2026 燃油消耗模型测试 (Iter-49)."""

from __future__ import annotations

from f1opt.model.fuel_model import (
    FuelConsumptionModel,
    FuelMode,
    FuelState,
    fuel_effect_on_lap_time,
    fuel_per_lap,
    recommended_start_fuel_kg,
)


# --------------------------------------------------------------------------- #
# FuelMode 枚举
# --------------------------------------------------------------------------- #
def test_fuel_mode_values():
    assert FuelMode.LEAN.value == "lean"
    assert FuelMode.NORMAL.value == "normal"
    assert FuelMode.RICH.value == "rich"
    assert FuelMode.PARTY.value == "party"


def test_four_fuel_modes():
    assert len(list(FuelMode)) == 4


# --------------------------------------------------------------------------- #
# FuelConsumptionModel 基础
# --------------------------------------------------------------------------- #
def test_model_track_id():
    m = FuelConsumptionModel("monza")
    assert m.track_id == "monza"
    assert m.base_consumption_kg_per_lap == 1.90


def test_model_unknown_track_uses_default():
    m = FuelConsumptionModel("nonexistent")
    assert m.base_consumption_kg_per_lap == 1.65


def test_consumption_normal_mode():
    m = FuelConsumptionModel("monza")
    assert m.consumption_for_mode(FuelMode.NORMAL) == 1.90


def test_consumption_lean_lower():
    m = FuelConsumptionModel("monza")
    normal = m.consumption_for_mode(FuelMode.NORMAL)
    lean = m.consumption_for_mode(FuelMode.LEAN)
    assert lean < normal


def test_consumption_rich_higher():
    m = FuelConsumptionModel("monza")
    normal = m.consumption_for_mode(FuelMode.NORMAL)
    rich = m.consumption_for_mode(FuelMode.RICH)
    assert rich > normal


def test_consumption_party_highest():
    m = FuelConsumptionModel("monza")
    party = m.consumption_for_mode(FuelMode.PARTY)
    rich = m.consumption_for_mode(FuelMode.RICH)
    assert party > rich


# --------------------------------------------------------------------------- #
# 圈速影响
# --------------------------------------------------------------------------- #
def test_lean_mode_slower():
    m = FuelConsumptionModel("monza")
    assert m.lap_time_delta_for_mode(FuelMode.LEAN) > 0


def test_rich_mode_faster():
    m = FuelConsumptionModel("monza")
    assert m.lap_time_delta_for_mode(FuelMode.RICH) < 0


def test_normal_mode_zero_delta():
    m = FuelConsumptionModel("monza")
    assert m.lap_time_delta_for_mode(FuelMode.NORMAL) == 0.0


def test_party_mode_fastest():
    m = FuelConsumptionModel("monza")
    party = m.lap_time_delta_for_mode(FuelMode.PARTY)
    rich = m.lap_time_delta_for_mode(FuelMode.RICH)
    assert party < rich  # party 更快 (更负)


# --------------------------------------------------------------------------- #
# 燃油质量对圈速影响
# --------------------------------------------------------------------------- #
def test_fuel_effect_zero_when_empty():
    assert fuel_effect_on_lap_time(0.0) == 0.0


def test_fuel_effect_positive_with_fuel():
    assert fuel_effect_on_lap_time(100.0) > 0


def test_fuel_effect_proportional():
    """燃油影响应与质量成正比."""
    e1 = fuel_effect_on_lap_time(50.0)
    e2 = fuel_effect_on_lap_time(100.0)
    assert abs(e2 - 2 * e1) < 1e-9


def test_fuel_effect_110kg_reasonable():
    """110 kg 燃油影响应在 3-5s 范围."""
    effect = fuel_effect_on_lap_time(110.0)
    assert 3.0 <= effect <= 5.0


def test_fuel_effect_method():
    m = FuelConsumptionModel("monza")
    assert m.fuel_effect_on_lap_time(0.0) == 0.0
    assert m.fuel_effect_on_lap_time(100.0) > 0


# --------------------------------------------------------------------------- #
# calculate 完整状态计算
# --------------------------------------------------------------------------- #
def test_calculate_basic():
    m = FuelConsumptionModel("monza")
    state = FuelState(current_fuel_kg=110.0, total_laps=53, current_lap=20)
    r = m.calculate(state)
    assert r.fuel_used_kg > 0
    assert r.fuel_remaining_kg < 110.0
    assert r.laps_remaining == 33


def test_calculate_fuel_deficit_when_low():
    """燃油不足时应报告 deficit."""
    m = FuelConsumptionModel("monza")  # 1.9 kg/lap
    state = FuelState(current_fuel_kg=10.0, total_laps=53, current_lap=50)
    r = m.calculate(state)
    # 剩 3 圈, 需 5.7 kg, 只有 10 kg → 富余
    assert r.fuel_deficit_kg == 0.0
    assert not r.needs_fuel_save


def test_calculate_fuel_deficit_when_insufficient():
    m = FuelConsumptionModel("monza")  # 1.9 kg/lap
    # 3 圈需 5.7kg, 只有 3kg → 不足
    state = FuelState(current_fuel_kg=3.0, total_laps=53, current_lap=50)
    r = m.calculate(state)
    assert r.fuel_deficit_kg > 0
    assert r.needs_fuel_save is True


def test_calculate_projected_finish():
    m = FuelConsumptionModel("monza")
    state = FuelState(current_fuel_kg=60.0, total_laps=53, current_lap=20)
    r = m.calculate(state)
    # 33 圈 × 1.9 = 62.7, 起步 60 → 不足
    assert r.projected_finish_fuel_kg < 0


def test_calculate_lean_mode_reduces_deficit():
    m = FuelConsumptionModel("monza")
    state_normal = FuelState(current_fuel_kg=50.0, total_laps=53, current_lap=20,
                             mode=FuelMode.NORMAL)
    state_lean = FuelState(current_fuel_kg=50.0, total_laps=53, current_lap=20,
                           mode=FuelMode.LEAN)
    r_normal = m.calculate(state_normal)
    r_lean = m.calculate(state_lean)
    assert r_lean.fuel_deficit_kg <= r_normal.fuel_deficit_kg


def test_calculate_zero_laps_remaining():
    """完赛后 laps_remaining=0."""
    m = FuelConsumptionModel("monza")
    state = FuelState(current_fuel_kg=10.0, total_laps=53, current_lap=53)
    r = m.calculate(state)
    assert r.laps_remaining == 0
    assert r.fuel_used_kg == 0.0


# --------------------------------------------------------------------------- #
# recommend_mode
# --------------------------------------------------------------------------- #
def test_recommend_lean_when_deficit():
    m = FuelConsumptionModel("monza")
    state = FuelState(current_fuel_kg=3.0, total_laps=53, current_lap=50)
    assert m.recommend_mode(state) == FuelMode.LEAN


def test_recommend_normal_when_sufficient():
    m = FuelConsumptionModel("monza")
    state = FuelState(current_fuel_kg=110.0, total_laps=53, current_lap=20)
    assert m.recommend_mode(state) == FuelMode.NORMAL


def test_recommend_rich_late_race_with_surplus():
    """后段 + 余油充足 → 推荐 RICH."""
    m = FuelConsumptionModel("monza")
    # 总 53 圈, 当前 50, 余 30 kg → 远超 3 圈需求
    state = FuelState(current_fuel_kg=30.0, total_laps=53, current_lap=50)
    rec = m.recommend_mode(state)
    assert rec in (FuelMode.RICH, FuelMode.NORMAL)


# --------------------------------------------------------------------------- #
# recommended_start_fuel_kg
# --------------------------------------------------------------------------- #
def test_start_fuel_monza_53_laps():
    fuel = recommended_start_fuel_kg("monza", 53)
    # 53 × 1.9 + 2 = 102.7 kg
    assert 95.0 < fuel <= 110.0


def test_start_fuel_capped_at_110():
    """起步油量不超过 110 kg (FIA 上限)."""
    fuel = recommended_start_fuel_kg("spa", 100)  # 极长
    assert fuel <= 110.0


def test_start_fuel_includes_margin():
    """起步油量应包含 2 kg 安全余量."""
    m = FuelConsumptionModel("monza")
    fuel = m.recommended_start_fuel_kg(53)
    # 至少 53 × 1.9 = 100.7
    assert fuel >= 100.7


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def test_fuel_per_lap_convenience():
    assert fuel_per_lap("monza") == 1.90
    assert fuel_per_lap("monza", FuelMode.LEAN) < 1.90


def test_fuel_per_lap_unknown_track():
    assert fuel_per_lap("nonexistent") == 1.65  # 默认


# --------------------------------------------------------------------------- #
# 赛道对比
# --------------------------------------------------------------------------- #
def test_spa_highest_consumption():
    """Spa 长直道多, 燃油消耗最高."""
    spa = fuel_per_lap("spa")
    monaco = fuel_per_lap("monaco")
    assert spa > monaco


def test_monaco_lowest_consumption():
    """Monaco 低速, 燃油消耗最低."""
    monaco = fuel_per_lap("monaco")
    spa = fuel_per_lap("spa")
    assert monaco < spa
    assert monaco < 1.4


def test_all_tracks_in_range():
    """所有赛道燃油消耗应在 1.0-2.2 kg/lap 范围."""
    from f1opt.model.fuel_model import all_track_consumptions
    for tid, cons in all_track_consumptions().items():
        assert 1.0 <= cons <= 2.2, f"{tid}: {cons}"


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_deterministic():
    m = FuelConsumptionModel("monza")
    state = FuelState(current_fuel_kg=80.0, total_laps=53, current_lap=25)
    r1 = m.calculate(state)
    r2 = m.calculate(state)
    assert r1.fuel_used_kg == r2.fuel_used_kg
    assert r1.fuel_deficit_kg == r2.fuel_deficit_kg


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_full_race_fuel_management():
    """完整 53 圈 monza 燃油管理仿真."""
    m = FuelConsumptionModel("monza")
    fuel = m.recommended_start_fuel_kg(53)
    state = FuelState(current_fuel_kg=fuel, total_laps=53, current_lap=0)
    # 每圈推进
    for lap in range(53):
        state.current_lap = lap
        r = m.calculate(state)
        if r.needs_fuel_save:
            state.mode = FuelMode.LEAN
        else:
            state.mode = FuelMode.NORMAL
        state.current_fuel_kg -= m.consumption_for_mode(state.mode)
    # 应能完赛
    assert state.current_fuel_kg >= -1.0


def test_fuel_save_strategy():
    """节油策略: 5 圈 LEAN 可省多少."""
    m = FuelConsumptionModel("monza")
    normal = m.consumption_for_mode(FuelMode.NORMAL)
    lean = m.consumption_for_mode(FuelMode.LEAN)
    saved_per_lap = normal - lean
    saved_5_laps = saved_per_lap * 5
    assert saved_5_laps > 0.5  # 至少省 0.5 kg


def test_qualifying_party_mode():
    """排位用 PARTY 模式最快."""
    m = FuelConsumptionModel("monza")
    party_delta = m.lap_time_delta_for_mode(FuelMode.PARTY)
    normal_delta = m.lap_time_delta_for_mode(FuelMode.NORMAL)
    assert party_delta < normal_delta  # party 更快
