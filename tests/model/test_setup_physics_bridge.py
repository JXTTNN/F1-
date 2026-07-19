"""setup_physics_bridge 单元测试 (Iter-66).

覆盖:
- 物理一致性: 最优 setup 在各赛道类型上 penalty=0; DEFAULT_SETUP 在 medium 上 =0
- 物理方向: Monza 高翼应慢 (低 downforce 赛道); Monaco 高翼应快 (高 downforce)
- 物理真值: setup_lap_time 在 reference setup + 50kg fuel 下 ≈ benchmark (0% err)
- Setup 敏感性: ±5 档 rear_wing 扰动 std > 0.05s (DNN 可学习信号)
- 跨赛道差异: 同一 setup 在不同赛道类型上 penalty 显著不同
- API 一致性: setup_to_lap_config / evaluate_setup / setup_lap_time 互洽
- fuel_load 1:1 映射到 current_fuel_kg
- 未知赛道不崩溃 (回退 medium 缩放)
"""

from __future__ import annotations

import numpy as np
import pytest

from f1opt.data.ea_f1_2026_benchmark import EA_F1_2026_LAP_TIME_BENCHMARK
from f1opt.data.setup_schema import DEFAULT_SETUP, CarSetup
from f1opt.data.tracks import ALL_TRACKS, TRACKS_BY_ID
from f1opt.model.lap_simulator_2026 import LapConfig2026, LapResult2026
from f1opt.model.setup_physics_bridge import (
    _BASE_SENSITIVITY_S_PER_CLICK,
    _TRACK_TYPE_OPTIMA,
    _TRACK_TYPE_SCALE,
    evaluate_setup,
    optimal_setup_for_track,
    optimal_setup_for_track_type,
    setup_lap_time,
    setup_penalty_s,
    setup_to_lap_config,
)


# --- 物理一致性: 最优 setup penalty = 0 ----------------------------------------
@pytest.mark.parametrize("track_type", list(_TRACK_TYPE_OPTIMA.keys()))
def test_optimal_setup_zero_penalty(track_type: str) -> None:
    """各赛道类型的参考最优 setup 在该类型赛道上 penalty 接近 0.

    Iter-164.15: 逐赛道工程参数调整 (_track_engineering_adjusted_optima) 让
    同类型赛道的最优值略有偏移 (downforce/tire_wear/brake_wear 调制), 故
    track_type 级最优 setup 在特定赛道上不再严格 penalty=0, 而是落在一个
    小的工程调整范围内 (≤1.0s, 典型 <0.5s). 这反映真实 F1: 同类型赛道仍需
    逐场微调.
    """
    opt = optimal_setup_for_track_type(track_type)  # type: ignore[arg-type]
    sample_track = next(t for t in ALL_TRACKS if t.track_type == track_type)
    pen = setup_penalty_s(opt, sample_track.track_id)
    # Iter-164.15: 允许工程调整带来的小 penalty (≤1.0s)
    assert pen <= 1.0, (
        f"{track_type} 最优 setup penalty={pen:.6f}s > 1.0s "
        f"(工程调整范围过大, track={sample_track.track_id})"
    )


def test_default_setup_zero_penalty_on_medium() -> None:
    """DEFAULT_SETUP 是 medium 类型最优 (校准锚点), 在 melbourne 上 penalty 接近 0.

    Iter-164.15: Melbourne 的工程调整 (tire_wear=1.10, brake_wear=0.75) 让
    camber/pressure 最优值微偏, DEFAULT_SETUP 不再严格 penalty=0.
    """
    pen = setup_penalty_s(DEFAULT_SETUP, "melbourne")
    assert pen <= 0.5, (
        f"DEFAULT_SETUP penalty={pen:.6f}s > 0.5s on melbourne"
    )


def test_default_setup_positive_penalty_off_medium() -> None:
    """DEFAULT_SETUP 在非 medium 赛道上 penalty > 0 (远离该类型最优)."""
    for tid in ["monza", "monaco", "hungaroring", "spa"]:
        pen = setup_penalty_s(DEFAULT_SETUP, tid)
        assert pen > 0.5, f"{tid} penalty={pen:.3f}s 过低 (期望 > 0.5)"


# --- 物理方向: setup 改变应使圈速按物理预期变化 --------------------------------
def test_monza_high_wing_slower() -> None:
    """Monza (low-downforce 赛道) 高尾翼 = 高阻力 = 慢."""
    low_wing = CarSetup(**{**DEFAULT_SETUP.model_dump(), "front_wing": 8, "rear_wing": 6})
    high_wing = CarSetup(**{**DEFAULT_SETUP.model_dump(), "front_wing": 8, "rear_wing": 30})
    t_low = setup_lap_time(low_wing, "monza")
    t_high = setup_lap_time(high_wing, "monza")
    assert t_high > t_low + 0.5, (
        f"monza high_wing={t_high:.3f}s 未显著慢于 low_wing={t_low:.3f}s"
    )


def test_monaco_high_wing_faster_or_comparable() -> None:
    """Monaco (街道/高 downforce) 最优 setup 已是高翼; 进一步加翼未必更快,
    但至少在最优附近时不应大幅慢于低翼 (Monaco 直道短, 翼面阻力代价小)."""
    opt = optimal_setup_for_track_type("street")
    # 最优 monaco setup 圈速
    t_opt = setup_lap_time(opt, "monaco")
    # 把翼面降到 monza 风格 (低翼) -> monaco 上慢得多 (慢弯抓地不足)
    low_wing = CarSetup(**{**opt.model_dump(), "front_wing": 8, "rear_wing": 6})
    t_low = setup_lap_time(low_wing, "monaco")
    assert t_low > t_opt + 1.0, (
        f"monaco 低翼={t_low:.3f}s 未显著慢于最优={t_opt:.3f}s "
        f"(慢弯抓地不足应导致大圈速代价)"
    )


def test_setup_penalty_monotone_with_deviation() -> None:
    """setup 偏离最优越远, penalty 单调增加 (V 形惩罚)."""
    opt = optimal_setup_for_track_type("medium")
    penalties = []
    for delta in [0, 5, 10, 20, 30]:
        s = opt.model_copy(update={"rear_wing": max(0, min(50, opt.rear_wing + delta))})
        penalties.append(setup_penalty_s(s, "melbourne"))
    # 单调递增
    for i in range(1, len(penalties)):
        assert penalties[i] > penalties[i - 1], (
            f"penalty 未单调递增: {[f'{p:.3f}' for p in penalties]}"
        )


# --- 物理真值: reference setup + 50kg fuel ≈ benchmark -----------------------
@pytest.mark.parametrize("track_id", [
    "melbourne", "monza", "monaco", "hungaroring", "spa",
    "silverstone", "suzuka", "jeddah", "yas_marina",
])
def test_optimal_setup_matches_benchmark(track_id: str) -> None:
    """各赛道最优 setup (50kg fuel) 通过物理引擎评估, 圈速 ≈ benchmark (误差 <0.2%).

    Iter-164.15: 用 optimal_setup_for_track (逐赛道工程参数感知最优) 替代
    optimal_setup_for_track_type (类型级最优), 让工程调整后的最优 setup 仍
    匹配 benchmark.
    """
    opt = optimal_setup_for_track(track_id)
    # fuel_load 已在 optimal_setup_for_track 内设为 50 (reference)
    lt = setup_lap_time(opt, track_id)
    bench = EA_F1_2026_LAP_TIME_BENCHMARK[track_id]
    err_pct = 100.0 * abs(lt - bench) / bench
    assert err_pct < 0.2, (
        f"{track_id} sim={lt:.3f}s bench={bench:.3f}s err={err_pct:.3f}% >= 0.2%"
    )


def test_fuel_load_direct_mapping() -> None:
    """setup.fuel_load 1:1 映射到 LapConfig2026.current_fuel_kg."""
    s_light = DEFAULT_SETUP.model_copy(update={"fuel_load": 10.0})
    s_heavy = DEFAULT_SETUP.model_copy(update={"fuel_load": 100.0})
    cfg_light = setup_to_lap_config(s_light, "melbourne")
    cfg_heavy = setup_to_lap_config(s_heavy, "melbourne")
    assert cfg_light.current_fuel_kg == pytest.approx(10.0)
    assert cfg_heavy.current_fuel_kg == pytest.approx(100.0)
    # 重燃油应更慢
    assert setup_lap_time(s_heavy, "melbourne") > setup_lap_time(s_light, "melbourne")


def test_car_offset_reflects_penalty() -> None:
    """LapConfig2026.car_performance_offset_s = setup_penalty_s."""
    s = DEFAULT_SETUP.model_copy(update={"rear_wing": 45})  # 远离 medium 最优 27
    cfg = setup_to_lap_config(s, "melbourne")
    pen = setup_penalty_s(s, "melbourne")
    assert cfg.car_performance_offset_s == pytest.approx(pen, rel=1e-9)


# --- Setup 敏感性 (DNN 训练信号) ----------------------------------------------
@pytest.mark.parametrize("track_id", [
    "monza", "monaco", "hungaroring", "spa", "melbourne", "silverstone",
])
def test_setup_sensitivity_strong_signal(track_id: str) -> None:
    """±5 档 rear_wing 扰动应使圈速 std > 0.05s (DNN 可学习信号)."""
    track = TRACKS_BY_ID[track_id]
    base = optimal_setup_for_track_type(track.track_type)
    rng = np.random.default_rng(42)
    times = []
    for _ in range(20):
        rw = max(0, min(50, base.rear_wing + int(rng.integers(-5, 6))))
        s = base.model_copy(update={"rear_wing": rw, "fuel_load": 50.0})
        times.append(setup_lap_time(s, track_id))
    std = float(np.std(times))
    assert std > 0.05, (
        f"{track_id} rear_wing ±5 扰动 std={std:.4f}s <= 0.05s (信号过弱)"
    )


def test_setup_sensitivity_multi_param() -> None:
    """同时扰动多个参数, 圈速方差显著大于单参数 (证明多参数信号叠加)."""
    rng = np.random.default_rng(0)
    base = optimal_setup_for_track_type("medium")
    # 单参数 (只动 rear_wing)
    single_times = []
    for _ in range(20):
        rw = max(0, min(50, base.rear_wing + int(rng.integers(-5, 6))))
        single_times.append(setup_lap_time(
            base.model_copy(update={"rear_wing": rw, "fuel_load": 50.0}), "melbourne"
        ))
    # 多参数 (同时动 wing + suspension + camber)
    multi_times = []
    for _ in range(20):
        s = base.model_copy(update={
            "rear_wing": max(0, min(50, base.rear_wing + int(rng.integers(-5, 6)))),
            "front_suspension": max(1, min(50, base.front_suspension + int(rng.integers(-5, 6)))),
            "front_camber": max(-3.5, min(-2.5, base.front_camber + 0.05 * int(rng.integers(-5, 6)))),
            "fuel_load": 50.0,
        })
        multi_times.append(setup_lap_time(s, "melbourne"))
    assert np.std(multi_times) > np.std(single_times) * 1.2, (
        f"多参数扰动 std={np.std(multi_times):.4f}s 未显著大于单参数 "
        f"std={np.std(single_times):.4f}s"
    )


# --- 跨赛道差异 ---------------------------------------------------------------
def test_same_setup_different_track_different_penalty() -> None:
    """同一 setup (DEFAULT_SETUP) 在不同赛道类型上 penalty 显著不同."""
    pen_medium = setup_penalty_s(DEFAULT_SETUP, "melbourne")  # = 0 (medium 锚点)
    pen_monza = setup_penalty_s(DEFAULT_SETUP, "monza")       # > 0 (低 downforce)
    pen_monaco = setup_penalty_s(DEFAULT_SETUP, "monaco")     # > 0 (街道)
    assert pen_medium < pen_monza
    assert pen_medium < pen_monaco
    assert pen_monza > 3.0, f"monza penalty={pen_monza:.3f}s 过低 (期望 > 3s)"


def test_optimal_setup_differs_across_track_types() -> None:
    """不同赛道类型的最优 setup 在 aero 上显著不同."""
    opt_low_df = optimal_setup_for_track_type("high_speed_low_downforce")
    opt_high_df = optimal_setup_for_track_type("high_downforce")
    opt_street = optimal_setup_for_track_type("street")
    # 低 downforce 赛道最优翼面 << 高 downforce 赛道最优翼面
    assert opt_low_df.rear_wing < opt_high_df.rear_wing
    assert opt_high_df.rear_wing < opt_street.rear_wing


# --- API 一致性 ---------------------------------------------------------------
def test_setup_to_lap_config_returns_lap_config() -> None:
    """setup_to_lap_config 返回 LapConfig2026 实例, track_id 透传."""
    cfg = setup_to_lap_config(DEFAULT_SETUP, "monza")
    assert isinstance(cfg, LapConfig2026)
    assert cfg.track_id == "monza"
    assert cfg.current_fuel_kg == pytest.approx(DEFAULT_SETUP.fuel_load)


def test_evaluate_setup_returns_lap_result() -> None:
    """evaluate_setup 返回 LapResult2026, 含 car_offset_s 反映 setup penalty."""
    s = DEFAULT_SETUP.model_copy(update={"rear_wing": 40})  # 偏离 medium 最优
    r = evaluate_setup(s, "melbourne")
    assert isinstance(r, LapResult2026)
    assert r.car_offset_s == pytest.approx(setup_penalty_s(s, "melbourne"), rel=1e-9)
    assert r.lap_time_s > 50.0
    # benchmark + car_offset + fuel_delta (DEFAULT fuel=30 vs ref 50) + 其他 ≈ lap_time
    # car_offset 应 > 0 (偏离最优)
    assert r.car_offset_s > 0.5


def test_setup_lap_time_matches_evaluate() -> None:
    """setup_lap_time = evaluate_setup().lap_time_s."""
    s = DEFAULT_SETUP.model_copy(update={"front_wing": 35, "rear_wing": 35})
    for tid in ["monza", "melbourne", "monaco"]:
        direct = setup_lap_time(s, tid)
        via_eval = evaluate_setup(s, tid).lap_time_s
        assert direct == pytest.approx(via_eval, abs=1e-9)


def test_kwargs_passthrough_compound_and_wet() -> None:
    """setup_to_lap_config 接受 compound / wet 等 kwargs 覆写."""
    cfg = setup_to_lap_config(DEFAULT_SETUP, "melbourne", compound="soft", wet=True)
    assert cfg.compound == "soft"
    assert cfg.wet is True


# --- 未知赛道不崩溃 -----------------------------------------------------------
def test_unknown_track_falls_back_to_medium() -> None:
    """未知 track_id 应回退到 medium 缩放, 不抛异常."""
    pen = setup_penalty_s(DEFAULT_SETUP, "definitely_not_a_track")
    # DEFAULT_SETUP 是 medium 最优 -> 未知赛道用 medium 缩放 -> penalty = 0
    assert pen == pytest.approx(0.0, abs=1e-9)
    # 仍能评估圈速 (lap_simulator 对未知 track 用 90s 兜底)
    lt = setup_lap_time(DEFAULT_SETUP, "definitely_not_a_track")
    assert isinstance(lt, float)
    assert 50.0 < lt < 200.0


# --- 内部表一致性 -------------------------------------------------------------
def test_sensitivity_table_covers_all_non_fuel_fields() -> None:
    """_BASE_SENSITIVITY 覆盖 18 项非 fuel 参数 (19 - fuel_load)."""
    # fuel_load 不在敏感度表 (燃油通过 current_fuel_kg 直接映射)
    assert "fuel_load" not in _BASE_SENSITIVITY_S_PER_CLICK
    # 18 项非 fuel 参数全部覆盖
    assert len(_BASE_SENSITIVITY_S_PER_CLICK) == 18
    # 所有值 > 0 (惩罚必须非零)
    for name, s in _BASE_SENSITIVITY_S_PER_CLICK.items():
        assert s > 0.0, f"{name} 敏感度={s} <= 0"


def test_track_type_scale_covers_all_types_and_fields() -> None:
    """_TRACK_TYPE_SCALE 覆盖 5 种赛道类型 × 18 项参数."""
    assert set(_TRACK_TYPE_SCALE.keys()) == set(_TRACK_TYPE_OPTIMA.keys())
    for tt, scale in _TRACK_TYPE_SCALE.items():
        assert set(scale.keys()) == set(_BASE_SENSITIVITY_S_PER_CLICK.keys()), (
            f"{tt} scale 缺字段"
        )
        for v in scale.values():
            assert v > 0.0


def test_optima_table_matches_default_setup_on_medium() -> None:
    """medium 最优 setup 在 17 项非 fuel 参数上 = DEFAULT_SETUP (校准锚点)."""
    medium_opt = _TRACK_TYPE_OPTIMA["medium"]
    for name in _BASE_SENSITIVITY_S_PER_CLICK:
        assert medium_opt[name] == pytest.approx(getattr(DEFAULT_SETUP, name)), (
            f"{name}: medium 最优={medium_opt[name]} != DEFAULT={getattr(DEFAULT_SETUP, name)}"
        )
