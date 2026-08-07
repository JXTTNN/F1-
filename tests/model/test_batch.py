"""批量 / 缓存 / 网格 / 敏感度 / Pareto 推理辅助单元测试 (perf 层).

覆盖 :mod:`f1opt.model.batch` 全部公开函数与 :mod:`f1opt.model.surrogate`
新增的 ``predict_lap_time_cached`` LRU 缓存层. 全部确定性; 缓存测试用 autouse
fixture 在每条测试前后清空缓存, 避免模块全局状态串扰.
"""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.driver.profile import AGGRESSIVE_PROFILE
from f1opt.model.batch import (
    batch_predict_full,
    batch_predict_lap_times,
    predict_lap_time_grid,
    sensitivity_analysis,
    setup_pareto_front,
)
from f1opt.model.surrogate import (
    _PREDICT_CACHE_MAXSIZE,
    _PREDICT_CACHE_STATS,
    _predict_cache,
    clear_predict_cache,
    driver_vector,
    predict_full,
    predict_lap_time,
    predict_lap_time_cached,
)

TRACK = "melbourne"
# lap_time 合理 F1 区间 (与 surrogate._MIN/_MAX_LAP_TIME 一致).
MIN_LAP = 50.0
MAX_LAP = 250.0


# --- 固件 -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_predict_cache() -> None:
    """每条测试前后清空 surrogate 缓存 + 计数, 保证隔离."""
    clear_predict_cache()
    yield
    clear_predict_cache()


def _drv_tuple(profile: object = None) -> tuple[float, ...]:
    """把 driver_vector 归一为纯 Python float tuple (供缓存键使用)."""
    return tuple(float(v) for v in driver_vector(profile))


def _varied_setups(n: int) -> list[CarSetup]:
    """构造 n 份差异 setup (front_wing 0..n-1 档), 保证圈速互不相同."""
    return [
        DEFAULT_SETUP.model_copy(update={"front_wing": min(50, i)})
        for i in range(n)
    ]


# --- 批量圈速 ---------------------------------------------------------------
def test_batch_predict_lap_times_returns_n_positive_floats_in_range() -> None:
    """5 个 setup 批量预测返回 5 个正 float, 全部落在合理圈速区间."""
    setups = _varied_setups(5)
    times = batch_predict_lap_times(setups, TRACK)
    assert len(times) == 5
    assert all(isinstance(t, float) for t in times)
    assert all(t > 0.0 for t in times)
    assert all(MIN_LAP < t < MAX_LAP for t in times)


def test_batch_predict_matches_single_predict_lap_time_within_tol() -> None:
    """批量路径与逐条 predict_lap_time 在 1e-4 内一致 (同一默认模型)."""
    setups = _varied_setups(5)
    batch = batch_predict_lap_times(setups, TRACK)
    single = [predict_lap_time(s, TRACK) for s in setups]
    for b, s in zip(batch, single, strict=True):
        assert b == pytest.approx(s, abs=1e-4)


def test_batch_predict_handles_single_element_list() -> None:
    """长度 1 的 CarSetup 列表返回长度 1 的 float 列表, 与单条一致."""
    setups = [DEFAULT_SETUP]
    out = batch_predict_lap_times(setups, TRACK)
    assert len(out) == 1
    assert isinstance(out[0], float)
    assert out[0] == pytest.approx(predict_lap_time(DEFAULT_SETUP, TRACK), abs=1e-4)


# --- 批量富预测 -------------------------------------------------------------
def test_batch_predict_full_returns_dicts_with_required_keys() -> None:
    """batch_predict_full 返回 5 个含 lap_time/sectors/responses 的字典."""
    setups = _varied_setups(5)
    out = batch_predict_full(setups, TRACK)
    assert len(out) == 5
    for d in out:
        assert isinstance(d, dict)
        assert isinstance(d["lap_time"], float)
        assert MIN_LAP < d["lap_time"] < MAX_LAP
        assert isinstance(d["sectors"], list) and len(d["sectors"]) == 3
        assert all(isinstance(s, float) and s > 0.0 for s in d["sectors"])
        assert isinstance(d["responses"], dict) and len(d["responses"]) >= 7
        assert "model_version" in d


def test_batch_predict_full_matches_single_predict_full() -> None:
    """batch_predict_full 各字段与逐条 predict_full 一致 (圈速容差 1e-4)."""
    setups = _varied_setups(4)
    batch = batch_predict_full(setups, TRACK, AGGRESSIVE_PROFILE)
    single = [predict_full(s, TRACK, AGGRESSIVE_PROFILE) for s in setups]
    for b, s in zip(batch, single, strict=True):
        assert b["lap_time"] == pytest.approx(s["lap_time"], abs=1e-4)


# --- 网格扫描 ---------------------------------------------------------------
def test_predict_lap_time_grid_front_wing_sweep_returns_valid_floats() -> None:
    """front_wing 0..50 step 5 (11 档) 网格扫描返回 11 个合法 float."""
    values = list(range(0, 51, 5))  # 0,5,...,50 -> 11 values
    times = predict_lap_time_grid("front_wing", values, DEFAULT_SETUP, TRACK)
    assert len(times) == 11
    assert all(isinstance(t, float) for t in times)
    assert all(MIN_LAP < t < MAX_LAP for t in times)


def test_predict_lap_time_grid_empty_values_returns_empty() -> None:
    """空 values 返回空列表."""
    assert predict_lap_time_grid("front_wing", [], DEFAULT_SETUP, TRACK) == []


# --- 敏感度 -----------------------------------------------------------------
def test_sensitivity_analysis_returns_21_nonnegative_floats() -> None:
    """sensitivity_analysis 返回 21 键字典, 值均为 >= 0 的 float."""
    sens = sensitivity_analysis(DEFAULT_SETUP, TRACK)
    assert set(sens.keys()) == set(SETUP_FIELDS.keys())
    assert len(sens) == 21
    assert all(isinstance(v, float) for v in sens.values())
    assert all(v >= 0.0 for v in sens.values())


def test_sensitivity_analysis_has_at_least_one_nonzero_field() -> None:
    """至少一个字段敏感度 > 0 (训练后模型对 setup 扰动有响应)."""
    sens = sensitivity_analysis(DEFAULT_SETUP, TRACK)
    assert any(v > 0.0 for v in sens.values()), "所有字段敏感度均为 0"


def test_sensitivity_analysis_delta_steps_2_returns_valid_structure() -> None:
    """delta_steps=2 仍返回 19 键, 值均为 >= 0 的 float (结构不变)."""
    sens = sensitivity_analysis(DEFAULT_SETUP, TRACK, delta_steps=2)
    assert set(sens.keys()) == set(SETUP_FIELDS.keys())
    assert len(sens) == 21
    assert all(isinstance(v, float) and v >= 0.0 for v in sens.values())


# --- Pareto 前沿 ------------------------------------------------------------
def test_setup_pareto_front_returns_non_dominated_subset() -> None:
    """6 个 setup 的 Pareto 前沿为索引子集, 且每个返回索引均非被支配."""
    setups = _varied_setups(6)
    front = setup_pareto_front(setups, TRACK)
    assert isinstance(front, list)
    assert 1 <= len(front) <= 6
    assert all(0 <= i < 6 for i in front)
    # 前沿索引升序且唯一
    assert front == sorted(set(front))
    # weight=0 -> 单目标圈速; 非支配 = 圈速严格最小者集合 (无其他严格更小).
    times = batch_predict_lap_times(setups, TRACK)
    for i in front:
        for j in range(6):
            if j == i:
                continue
            assert not (times[j] < times[i]), (
                f"front 成员 {i} 被更低圈速的 {j} 支配"
            )


def test_setup_pareto_front_weight_zero_includes_min_lap_time_index() -> None:
    """tire_wear_weight=0 (单目标圈速) 前沿必含圈速最小者的索引."""
    setups = [
        DEFAULT_SETUP.model_copy(update={"fuel_load": f})
        for f in (10.0, 30.0, 50.0, 70.0, 90.0, 110.0)
    ]
    times = batch_predict_lap_times(setups, TRACK)
    min_idx = times.index(min(times))
    front = setup_pareto_front(setups, TRACK, tire_wear_weight=0.0)
    assert min_idx in front


def test_setup_pareto_front_two_objective_with_tire_weight() -> None:
    """tire_wear_weight>0 双目标前沿: 唯一圈速最小者 (严格更低) 必非支配."""
    # fuel_load 严格递增 -> 圈速严格递增 (燃油惩罚 0.03s/kg 恒在先验中),
    # 故 index 0 为唯一严格最小圈速, 双目标下不可能被任何 j>0 支配.
    setups = [
        DEFAULT_SETUP.model_copy(update={"fuel_load": f})
        for f in (10.0, 30.0, 50.0, 70.0, 90.0, 110.0)
    ]
    times = batch_predict_lap_times(setups, TRACK)
    min_idx = times.index(min(times))
    assert min_idx == 0
    front = setup_pareto_front(setups, TRACK, tire_wear_weight=1.0)
    assert min_idx in front  # 圈速严格最小者不可能被支配
    assert front == sorted(set(front))


# --- 空列表 -----------------------------------------------------------------
def test_empty_list_inputs_return_empty_for_all_batch_functions() -> None:
    """所有批量函数对空列表输入返回 []."""
    assert batch_predict_lap_times([], TRACK) == []
    assert batch_predict_full([], TRACK) == []
    assert setup_pareto_front([], TRACK) == []


# --- 缓存层 -----------------------------------------------------------------
def test_predict_lap_time_cached_matches_predict_lap_time() -> None:
    """缓存版与原版 predict_lap_time 返回一致 (同一默认模型)."""
    sv = tuple(DEFAULT_SETUP.to_vector())
    dv = _drv_tuple()
    cached = predict_lap_time_cached(sv, TRACK, dv)
    direct = predict_lap_time(DEFAULT_SETUP, TRACK)
    assert isinstance(cached, float)
    assert cached == pytest.approx(direct, abs=1e-6)


def test_predict_lap_time_cached_second_call_is_cache_hit() -> None:
    """第二次相同键调用命中缓存 (hits 计数自增, misses 不变)."""
    sv = tuple(DEFAULT_SETUP.to_vector())
    dv = _drv_tuple()
    predict_lap_time_cached(sv, TRACK, dv)
    assert _PREDICT_CACHE_STATS["misses"] == 1
    assert _PREDICT_CACHE_STATS["hits"] == 0
    second = predict_lap_time_cached(sv, TRACK, dv)
    assert _PREDICT_CACHE_STATS["hits"] == 1
    assert _PREDICT_CACHE_STATS["misses"] == 1
    assert second == pytest.approx(predict_lap_time(DEFAULT_SETUP, TRACK), abs=1e-6)


def test_clear_predict_cache_resets_stats_and_cache() -> None:
    """clear_predict_cache 清空缓存并把 hits/misses 归零."""
    sv = tuple(DEFAULT_SETUP.to_vector())
    dv = _drv_tuple()
    predict_lap_time_cached(sv, TRACK, dv)
    predict_lap_time_cached(sv, TRACK, dv)
    assert _PREDICT_CACHE_STATS["hits"] == 1
    assert len(_predict_cache) == 1
    clear_predict_cache()
    assert _predict_cache == {}
    assert _PREDICT_CACHE_STATS == {"hits": 0, "misses": 0}


def test_cache_respects_maxsize_eviction() -> None:
    """插入超过 maxsize 个不同键后, 缓存大小不超过 maxsize (LRU 淘汰)."""
    base_vec = list(DEFAULT_SETUP.to_vector())
    dv = _drv_tuple()
    n = _PREDICT_CACHE_MAXSIZE + 88  # 600 > 512
    for i in range(n):
        # 扰动最后一维 (fuel_load 归一化), 保持在 [0,1] 内 -> from_vector 合法.
        last = i / float(n)
        sv = tuple(base_vec[:-1] + [last])
        predict_lap_time_cached(sv, TRACK, dv)
    assert len(_predict_cache) <= _PREDICT_CACHE_MAXSIZE
    assert len(_predict_cache) == _PREDICT_CACHE_MAXSIZE
    assert _PREDICT_CACHE_STATS["misses"] == n
    assert _PREDICT_CACHE_STATS["hits"] == 0


def test_cache_key_includes_track_id_no_stale_hit() -> None:
    """不同 track_id 产生不同键 -> 不命中, 各自返回该赛道的真实圈速."""
    sv = tuple(DEFAULT_SETUP.to_vector())
    dv = _drv_tuple()
    t_mel = predict_lap_time_cached(sv, "melbourne", dv)
    t_mon = predict_lap_time_cached(sv, "monza", dv)
    # 两条赛道圈速不同 (长度/类型先验不同).
    assert t_mel != t_mon
    # 各自与直接预测一致 (未被对方缓存污染).
    assert t_mel == pytest.approx(predict_lap_time(DEFAULT_SETUP, "melbourne"), abs=1e-6)
    assert t_mon == pytest.approx(predict_lap_time(DEFAULT_SETUP, "monza"), abs=1e-6)
    # 两次均为 miss (键不同).
    assert _PREDICT_CACHE_STATS["misses"] == 2
    assert _PREDICT_CACHE_STATS["hits"] == 0


def test_cache_key_includes_driver_vector() -> None:
    """不同 driver 向量产生不同键 -> 不命中 (driver 影响预测)."""
    sv = tuple(DEFAULT_SETUP.to_vector())
    dv_zero = _drv_tuple(None)
    dv_aggr = _drv_tuple(AGGRESSIVE_PROFILE)
    predict_lap_time_cached(sv, TRACK, dv_zero)
    predict_lap_time_cached(sv, TRACK, dv_aggr)
    # 两个不同 driver 键 -> 两次 miss.
    assert _PREDICT_CACHE_STATS["misses"] == 2
    assert _PREDICT_CACHE_STATS["hits"] == 0
    assert len(_predict_cache) == 2
