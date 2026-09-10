"""调教全流程集成测试 (Iter-72: 输入→分析→输出→反馈闭环全赛道验证).

覆盖 EA F1 2026 全 5 种赛道类型 + 别名赛道, 验证:
- search_setup 在所有赛道类型上 gain >= 0 (基线保障).
- recommended_lap_time 落在物理合理区间.
- response_profile 完整填充 (7 项响应).
- holistic=True 注入物理默认胎耗权重.
- observation_buffer 反馈闭环生效 (feedback_corrected=True).
- 别名赛道 (sakhir/sao_paulo/lusail) 与规范名一致.
"""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, CarSetup
from f1opt.model.optimizer import search_setup
from f1opt.model.online_correction import ObservationBuffer, add_observation
from f1opt.model.surrogate import predict_lap_time

# 每种赛道类型选一个代表 + 3 个别名赛道, 共 8 赛道 (覆盖全类型 + 别名回归).
_REPRESENTATIVE_TRACKS = [
    ("melbourne", "medium"),          # medium 代表
    ("monza", "high_speed_low_downforce"),  # 高速低下压力代表
    ("monaco", "street"),             # 街道代表
    ("hungaroring", "high_downforce"),  # 高下压力代表
    ("suzuka", "mixed"),              # 混合代表
    ("sakhir", "medium"),             # 别名 (→ bahrain)
    ("sao_paulo", "mixed"),           # 别名 (→ interlagos)
    ("lusail", "medium"),             # 别名 (→ losail)
]

# 次优基线 (高下压力粗调, 模拟真实练习赛起点).
_SUBOPT_BASELINE = DEFAULT_SETUP.model_copy(update={"front_wing": 33, "rear_wing": 35})


@pytest.mark.parametrize("track_id,track_type", _REPRESENTATIVE_TRACKS)
def test_full_pipeline_gain_nonnegative(track_id: str, track_type: str) -> None:
    """所有赛道类型: search_setup gain >= 0 (基线保障, Iter-67)."""
    result = search_setup(track_id, baseline=_SUBOPT_BASELINE, iterations=25, seed=0)
    assert result.predicted_gain_s >= -0.01, (
        f"{track_id}: gain={result.predicted_gain_s:.4f} < 0 (基线保障失效)"
    )


@pytest.mark.parametrize("track_id,track_type", _REPRESENTATIVE_TRACKS)
def test_full_pipeline_lap_time_in_bounds(track_id: str, track_type: str) -> None:
    """所有赛道类型: recommended_lap_time 落在 [50, 200]s 物理区间."""
    result = search_setup(track_id, baseline=_SUBOPT_BASELINE, iterations=25, seed=0)
    assert 50.0 < result.recommended_lap_time < 200.0, (
        f"{track_id}: rec_lap={result.recommended_lap_time} 超出物理区间"
    )
    assert 50.0 < result.baseline_lap_time < 200.0


@pytest.mark.parametrize("track_id,track_type", _REPRESENTATIVE_TRACKS)
def test_full_pipeline_response_profile_complete(track_id: str, track_type: str) -> None:
    """所有赛道类型: response_profile 含完整 7 项响应 (Iter-69 整体性)."""
    result = search_setup(track_id, baseline=_SUBOPT_BASELINE, iterations=25, seed=0)
    expected = {
        "speed_avg", "speed_max", "slip_angle", "tyre_load_spread",
        "rake", "tyre_temp", "g_lat_max",
    }
    assert set(result.response_profile.keys()) == expected
    assert set(result.baseline_response_profile.keys()) == expected


@pytest.mark.parametrize("track_id,track_type", _REPRESENTATIVE_TRACKS)
def test_full_pipeline_holistic_injects_weight(track_id: str, track_type: str) -> None:
    """所有赛道类型: holistic=True 注入物理默认胎耗权重 0.3 (Iter-69)."""
    result = search_setup(
        track_id, baseline=_SUBOPT_BASELINE, iterations=25, seed=0, holistic=True,
    )
    assert result.tire_wear_weight == 0.3
    assert result.tire_wear > 0.0


@pytest.mark.parametrize("track_id,track_type", _REPRESENTATIVE_TRACKS)
def test_full_pipeline_feedback_loop_active(track_id: str, track_type: str) -> None:
    """所有赛道类型: 传入含观测的 buffer 时 feedback_corrected=True (Iter-71)."""
    buf = ObservationBuffer()
    dnn_pred = predict_lap_time(DEFAULT_SETUP, track_id, None)
    add_observation(buf, DEFAULT_SETUP, track_id, None, dnn_pred - 0.5)

    result = search_setup(
        track_id, baseline=_SUBOPT_BASELINE, iterations=25, seed=0,
        observation_buffer=buf,
    )
    assert result.feedback_corrected is True, (
        f"{track_id}: 反馈闭环未激活"
    )

    # 对比: 无 buffer 时 feedback_corrected=False.
    no_buf = search_setup(
        track_id, baseline=_SUBOPT_BASELINE, iterations=25, seed=0,
    )
    assert no_buf.feedback_corrected is False


@pytest.mark.parametrize("track_id,track_type", _REPRESENTATIVE_TRACKS)
def test_full_pipeline_recommended_is_valid_setup(track_id: str, track_type: str) -> None:
    """所有赛道类型: recommended 可重构为合法 CarSetup."""
    result = search_setup(track_id, baseline=_SUBOPT_BASELINE, iterations=25, seed=0)
    rec = CarSetup(**result.recommended)
    assert isinstance(rec, CarSetup)
    # 推荐与基线有 diff (除非基线已最优, 此时 gain=0, diff 为空).
    if result.predicted_gain_s > 0.01:
        assert len(result.diff) > 0


def test_alias_tracks_consistent_with_canonical() -> None:
    """别名赛道 (sakhir/sao_paulo/lusail) 的预测与规范名一致 (Iter-67 回归)."""
    from f1opt.data.ea_f1_2026_benchmark import benchmark_lap_time_s

    for alias, canonical in [("sakhir", "bahrain"), ("sao_paulo", "interlagos"),
                              ("lusail", "losail")]:
        # benchmark 一致
        assert benchmark_lap_time_s(alias) == benchmark_lap_time_s(canonical)
        # DNN 预测一致 (同赛道物理)
        lt_alias = predict_lap_time(DEFAULT_SETUP, alias, None)
        lt_canon = predict_lap_time(DEFAULT_SETUP, canonical, None)
        assert lt_alias == pytest.approx(lt_canon, abs=0.01)
