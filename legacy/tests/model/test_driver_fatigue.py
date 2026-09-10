"""F1 2026 车手疲劳模型测试 (Iter-43)."""

from __future__ import annotations

from f1opt.model.driver_fatigue import (
    DriverFatigueModel,
    track_fatigue_difficulty,
)


# --------------------------------------------------------------------------- #
# 赛道难度
# --------------------------------------------------------------------------- #
def test_singapore_hardest_track():
    assert track_fatigue_difficulty("singapore") == 1.00


def test_losail_extreme():
    assert track_fatigue_difficulty("losail") >= 0.90


def test_monaco_easier():
    """Monaco 短距离低速, 疲劳影响小."""
    assert track_fatigue_difficulty("monaco") < 0.5


def test_unknown_track_default():
    assert track_fatigue_difficulty("nonexistent") == 0.50


def test_difficulty_in_range():
    for track in ["singapore", "monaco", "monza", "spa", "bahrain"]:
        d = track_fatigue_difficulty(track)
        assert 0.0 <= d <= 1.0


# --------------------------------------------------------------------------- #
# 疲劳累积
# --------------------------------------------------------------------------- #
def test_fatigue_starts_at_zero():
    model = DriverFatigueModel(driver_fitness=80, track_id="monza")
    assert model.fatigue == 0.0


def test_fatigue_increases_with_laps():
    model = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                ambient_temp_c=32, humidity_pct=80)
    model.update_lap()
    assert model.fatigue > 0.0
    f1 = model.fatigue
    model.update_lap()
    assert model.fatigue > f1


def test_fatigue_capped_at_1():
    """疲劳度上限 1.0."""
    model = DriverFatigueModel(driver_fitness=50, track_id="singapore",
                                ambient_temp_c=40, humidity_pct=90,
                                total_laps=60)
    for _ in range(100):
        model.update_lap()
    assert model.fatigue <= 1.0


def test_fit_driver_fatigues_slower():
    """体能好的车手疲劳更慢."""
    model_fit = DriverFatigueModel(driver_fitness=95, track_id="singapore",
                                    ambient_temp_c=32, humidity_pct=80)
    model_unfit = DriverFatigueModel(driver_fitness=50, track_id="singapore",
                                      ambient_temp_c=32, humidity_pct=80)
    for _ in range(20):
        model_fit.update_lap()
        model_unfit.update_lap()
    assert model_fit.fatigue < model_unfit.fatigue


def test_hot_track_fatigues_faster():
    """高温赛道疲劳更快."""
    model_cool = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                     ambient_temp_c=20, humidity_pct=50)
    model_hot = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                    ambient_temp_c=40, humidity_pct=90)
    for _ in range(20):
        model_cool.update_lap()
        model_hot.update_lap()
    assert model_hot.fatigue > model_cool.fatigue


def test_humid_track_fatigues_faster():
    """高湿度加速疲劳."""
    model_dry = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                    ambient_temp_c=32, humidity_pct=30)
    model_humid = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                      ambient_temp_c=32, humidity_pct=90)
    for _ in range(20):
        model_dry.update_lap()
        model_humid.update_lap()
    assert model_humid.fatigue > model_dry.fatigue


# --------------------------------------------------------------------------- #
# SC 恢复
# --------------------------------------------------------------------------- #
def test_sc_reduces_fatigue():
    """SC 期间恢复疲劳."""
    model = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                ambient_temp_c=32, humidity_pct=80)
    for _ in range(30):
        model.update_lap()
    f_before = model.fatigue
    model.update_sc_lap()
    assert model.fatigue < f_before


# --------------------------------------------------------------------------- #
# 圈速惩罚
# --------------------------------------------------------------------------- #
def test_lap_penalty_zero_when_fresh():
    model = DriverFatigueModel(driver_fitness=80, track_id="monza")
    assert model.lap_penalty_s() == 0.0


def test_lap_penalty_increases_with_fatigue():
    model = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                ambient_temp_c=32, humidity_pct=80)
    p0 = model.lap_penalty_s()
    for _ in range(30):
        model.update_lap()
    assert model.lap_penalty_s() > p0


def test_lap_penalty_bounded():
    """圈速惩罚有上限 0.5s."""
    model = DriverFatigueModel(driver_fitness=50, track_id="singapore",
                                ambient_temp_c=40, humidity_pct=90)
    for _ in range(100):
        model.update_lap()
    assert model.lap_penalty_s() <= 0.5


# --------------------------------------------------------------------------- #
# 失误概率
# --------------------------------------------------------------------------- #
def test_mistake_prob_increases_with_fatigue():
    model = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                ambient_temp_c=32, humidity_pct=80)
    p0 = model.mistake_probability()
    for _ in range(30):
        model.update_lap()
    assert model.mistake_probability() > p0


def test_mistake_prob_base_low():
    """基础失误概率应很低."""
    model = DriverFatigueModel(driver_fitness=80, track_id="monza")
    assert model.mistake_probability() < 0.01


# --------------------------------------------------------------------------- #
# 一致性因子
# --------------------------------------------------------------------------- #
def test_consistency_factor_decreases_with_fatigue():
    model = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                ambient_temp_c=32, humidity_pct=80)
    c0 = model.consistency_factor()
    for _ in range(30):
        model.update_lap()
    assert model.consistency_factor() < c0


def test_consistency_factor_bounded():
    """一致性因子下限 0.5."""
    model = DriverFatigueModel(driver_fitness=50, track_id="singapore",
                                ambient_temp_c=40, humidity_pct=90)
    for _ in range(100):
        model.update_lap()
    assert model.consistency_factor() >= 0.5


# --------------------------------------------------------------------------- #
# 状态摘要
# --------------------------------------------------------------------------- #
def test_state_returns_dict():
    model = DriverFatigueModel(driver_fitness=80, track_id="monza")
    s = model.state()
    assert "fatigue" in s
    assert "lap_penalty_s" in s
    assert "phase" in s
    assert s["phase"] == "fresh"


def test_state_phase_progression():
    model = DriverFatigueModel(driver_fitness=50, track_id="singapore",
                                ambient_temp_c=40, humidity_pct=90)
    # fresh
    assert model.state()["phase"] == "fresh"
    # 累积到 moderate
    while model.fatigue < 0.2:
        model.update_lap()
    assert model.state()["phase"] == "moderate"
    # 累积到 tired
    while model.fatigue < 0.5:
        model.update_lap()
    assert model.state()["phase"] == "tired"
    # 累积到 exhausted
    while model.fatigue < 0.8:
        model.update_lap()
    assert model.state()["phase"] == "exhausted"


# --------------------------------------------------------------------------- #
# 整场预计惩罚
# --------------------------------------------------------------------------- #
def test_expected_total_penalty_positive():
    model = DriverFatigueModel(driver_fitness=80, track_id="singapore",
                                total_laps=60)
    assert model.expected_total_fatigue_penalty_s() > 0


def test_expected_total_penalty_longer_race_higher():
    """更长比赛预计总惩罚更高."""
    short = DriverFatigueModel(driver_fitness=80, track_id="monza", total_laps=40)
    long = DriverFatigueModel(driver_fitness=80, track_id="monza", total_laps=70)
    assert long.expected_total_fatigue_penalty_s() > short.expected_total_fatigue_penalty_s()


# --------------------------------------------------------------------------- #
# 圈数追踪
# --------------------------------------------------------------------------- #
def test_laps_completed_increments():
    model = DriverFatigueModel(driver_fitness=80, track_id="monza")
    assert model.laps_completed == 0
    model.update_lap()
    assert model.laps_completed == 1
    model.update_lap()
    assert model.laps_completed == 2


def test_sc_lap_increments_counter():
    model = DriverFatigueModel(driver_fitness=80, track_id="monza")
    model.update_sc_lap()
    assert model.laps_completed == 1


# --------------------------------------------------------------------------- #
# 边界: 体能评分钳制
# --------------------------------------------------------------------------- #
def test_fitness_clamped_to_0_99():
    model_low = DriverFatigueModel(driver_fitness=-10, track_id="monza")
    model_high = DriverFatigueModel(driver_fitness=200, track_id="monza")
    assert model_low.driver_fitness == 0
    assert model_high.driver_fitness == 99
