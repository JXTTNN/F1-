"""在线残差修正层测试 (Iter-70 反馈闭环: 遥测反哺 surrogate).

覆盖:
- ObservationBuffer FIFO 容量管理.
- add_observation 记录观测 + 计算 DNN 预测残差.
- corrected_lap_time 无观测时退回纯 DNN 预测.
- corrected_lap_time 近距离观测修正 DNN 预测 (朝观测方向).
- corrected_lap_time 远距离/不同赛道观测不修正 (权重过低).
- 多条观测的加权平均修正.
"""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.online_correction import (
    ObservationBuffer,
    add_observation,
    corrected_lap_time,
)
from f1opt.model.surrogate import predict_lap_time


def test_buffer_fifo_capacity() -> None:
    """ObservationBuffer 容量满后 FIFO 淘汰最旧观测."""
    buf = ObservationBuffer(capacity=3)
    assert len(buf) == 0
    for i in range(5):
        add_observation(buf, DEFAULT_SETUP, "melbourne", None, 80.0 + i)
    assert len(buf) == 3  # 容量 3, 5 条后剩 3 条
    # 最旧 2 条 (80.0, 81.0) 被淘汰, 剩 82.0, 83.0, 84.0.
    obs = buf.observations_for_track("melbourne")
    observed_laps = sorted(o.observed_lap for o in obs)
    assert observed_laps == [82.0, 83.0, 84.0]


def test_buffer_track_filtering() -> None:
    """observations_for_track 仅返回指定赛道的观测."""
    buf = ObservationBuffer(capacity=10)
    add_observation(buf, DEFAULT_SETUP, "melbourne", None, 80.0)
    add_observation(buf, DEFAULT_SETUP, "monza", None, 85.0)
    add_observation(buf, DEFAULT_SETUP, "melbourne", None, 81.0)
    assert len(buf.observations_for_track("melbourne")) == 2
    assert len(buf.observations_for_track("monza")) == 1
    assert len(buf.observations_for_track("spa")) == 0


def test_corrected_no_observation_returns_dnn() -> None:
    """无观测时 corrected_lap_time == 纯 DNN 预测 (无修正)."""
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    dnn_pred = predict_lap_time(setup, "melbourne", None)
    corrected = corrected_lap_time(setup, "melbourne", None, buf)
    assert corrected == pytest.approx(dnn_pred, abs=1e-6)


def test_corrected_nearby_observation_shifts_toward_observed() -> None:
    """近距离观测修正 DNN 预测朝观测方向.

    用 DEFAULT_SETUP 作为观测 setup, 实测圈速 = DNN 预测 + 1.0s (模拟 DNN
    在该 setup 上低估 1s). 修正后对相同 setup 的预测应接近 DNN+1.0.
    """
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    dnn_pred = predict_lap_time(setup, "melbourne", None)
    observed = dnn_pred + 1.0  # DNN 低估 1s
    add_observation(buf, setup, "melbourne", None, observed)

    corrected = corrected_lap_time(setup, "melbourne", None, buf)
    # 完全相同 setup -> 权重=1 -> 修正 = 残差 = +1.0.
    assert corrected == pytest.approx(dnn_pred + 1.0, abs=0.05)


def test_corrected_different_track_no_correction() -> None:
    """不同赛道观测不修正 (track 精确匹配)."""
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    dnn_pred_melb = predict_lap_time(setup, "melbourne", None)
    # 在 monza 记录观测, melbourne 预测不应被修正.
    add_observation(buf, setup, "monza", None, 999.0)  # 极端观测

    corrected = corrected_lap_time(setup, "melbourne", None, buf)
    assert corrected == pytest.approx(dnn_pred_melb, abs=1e-6)


def test_corrected_far_observation_low_weight() -> None:
    """远距离 setup 观测权重低, 修正幅度小.

    在极端 setup (前后翼拉满) 上记录观测, 对 DEFAULT_SETUP 的修正应接近 0
    (setup 距离大, Gaussian 权重指数衰减).
    """
    buf = ObservationBuffer()
    target = DEFAULT_SETUP
    dnn_pred = predict_lap_time(target, "melbourne", None)
    # 极端 setup (前后翼拉满, 与 DEFAULT_SETUP 差距大).
    far_setup = DEFAULT_SETUP.model_copy(update={"front_wing": 50, "rear_wing": 50})
    far_observed = predict_lap_time(far_setup, "melbourne", None) + 5.0  # 大残差
    add_observation(buf, far_setup, "melbourne", None, far_observed)

    corrected = corrected_lap_time(target, "melbourne", None, buf)
    # 修正幅度应远小于 5.0 (远距离权重低).
    correction = corrected - dnn_pred
    assert abs(correction) < 1.0, f"远距离观测修正过大: {correction:.4f}s"


def test_corrected_multiple_observations_weighted_average() -> None:
    """多条观测的加权平均修正.

    两条近距离观测 (残差 +1s 和 +3s), 修正应介于 1 和 3 之间 (加权平均).
    """
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    dnn_pred = predict_lap_time(setup, "melbourne", None)
    add_observation(buf, setup, "melbourne", None, dnn_pred + 1.0)
    add_observation(buf, setup, "melbourne", None, dnn_pred + 3.0)

    corrected = corrected_lap_time(setup, "melbourne", None, buf)
    # 两条相同 setup 观测, 权重相等 -> 修正 = (1+3)/2 = 2.0.
    assert corrected == pytest.approx(dnn_pred + 2.0, abs=0.1)


def test_feedback_loop_improves_prediction_accuracy() -> None:
    """反馈闭环端到端: 加入真实观测后, 预测精度提升.

    模拟: DNN 在某 setup 上预测 = 真实 + 误差. 加入该 setup 的真实观测后,
    corrected_lap_time 的误差应 < 纯 DNN 误差.
    """
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    track = "hungaroring"
    dnn_pred = predict_lap_time(setup, track, None)
    # 模拟真实圈速 = DNN 预测 - 0.5s (DNN 高估 0.5s).
    true_lap = dnn_pred - 0.5

    # 反馈前: DNN 误差 = 0.5s.
    dnn_error = abs(dnn_pred - true_lap)

    # 加入真实观测.
    add_observation(buf, setup, track, None, true_lap)
    corrected = corrected_lap_time(setup, track, None, buf)

    # 反馈后: corrected 应接近 true_lap (误差 << 0.5s).
    corrected_error = abs(corrected - true_lap)
    assert corrected_error < dnn_error * 0.3, (
        f"反馈未提升精度: dnn_error={dnn_error:.4f}s corrected_error={corrected_error:.4f}s"
    )


# --- Iter-79 观测质量加权 ---------------------------------------------------
def test_quality_weighting_high_quality_dominates() -> None:
    """高质量观测应主导修正 (相同残差, quality=1.0 影响远大于 quality=0.1)."""
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    track = "melbourne"
    dnn_pred = predict_lap_time(setup, track, None)

    # 两条观测, 相同 setup (相同距离权重), 不同 quality
    # 高质量: 残差 +1.0s; 低质量: 残差 +5.0s
    add_observation(buf, setup, track, None, dnn_pred + 1.0, quality=1.0)
    add_observation(buf, setup, track, None, dnn_pred + 5.0, quality=0.1)

    corrected = corrected_lap_time(setup, track, None, buf)
    # 加权: (1.0*1.0 + 0.1*5.0) / (1.0+0.1) = 1.5/1.1 = 1.364s
    # 高质量主导, 修正应接近 +1.0 而非 +5.0
    assert 0.5 < (corrected - dnn_pred) < 2.0, (
        f"高质量未主导: correction={corrected-dnn_pred:.3f}s (应接近 1.0)"
    )


def test_quality_clamped_to_unit_interval() -> None:
    """quality 被 clamp 到 [0,1]."""
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    dnn_pred = predict_lap_time(setup, "melbourne", None)

    # quality > 1 被 clamp 到 1.0
    add_observation(buf, setup, "melbourne", None, dnn_pred + 1.0, quality=5.0)
    obs = buf.observations_for_track("melbourne")[0]
    assert obs.quality == 1.0

    # quality < 0 被 clamp 到 0.0
    buf2 = ObservationBuffer()
    add_observation(buf2, setup, "melbourne", None, dnn_pred + 1.0, quality=-1.0)
    obs2 = buf2.observations_for_track("melbourne")[0]
    assert obs2.quality == 0.0


def test_zero_quality_observation_ignored() -> None:
    """quality=0 的观测被忽略 (权重为 0, 不影响修正)."""
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    track = "melbourne"
    dnn_pred = predict_lap_time(setup, track, None)

    # quality=0 的极端观测 (残差 +10s) 应被忽略
    add_observation(buf, setup, track, None, dnn_pred + 10.0, quality=0.0)

    corrected = corrected_lap_time(setup, track, None, buf)
    # quality=0 -> 权重=0 -> 总权重 < _MIN_TOTAL_WEIGHT -> 退回纯 DNN
    assert corrected == pytest.approx(dnn_pred, abs=1e-6)


# --- Iter-80 异常观测检测 ---------------------------------------------------
def test_outlier_observation_filtered() -> None:
    """残差 > 5.0s 的 outlier 观测被过滤 (交通/黄旗污染圈不拉偏修正).

    Iter-164.21: 阈值从 3.0 提高到 5.0.
    """
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    track = "melbourne"
    dnn_pred = predict_lap_time(setup, track, None)

    # 正常观测: 残差 +0.5s
    add_observation(buf, setup, track, None, dnn_pred + 0.5, quality=1.0)
    # outlier 观测: 残差 +7.0s (交通损失, > 5.0 阈值)
    add_observation(buf, setup, track, None, dnn_pred + 7.0, quality=1.0)

    corrected = corrected_lap_time(setup, track, None, buf)
    # outlier 被过滤, 只剩正常观测 -> 修正 = +0.5s, 不是加权平均 (+3.75s)
    assert corrected == pytest.approx(dnn_pred + 0.5, abs=0.1), (
        f"outlier 未被过滤: correction={corrected-dnn_pred:.3f}s (应接近 +0.5)"
    )


def test_all_outlier_observations_fall_back_to_dnn() -> None:
    """所有观测都是 outlier 时退回纯 DNN 预测 (无有效观测).

    Iter-164.21: 阈值从 3.0 提高到 5.0, outlier 残差需 > 5.0s.
    """
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    track = "melbourne"
    dnn_pred = predict_lap_time(setup, track, None)

    # 两条 outlier 观测 (残差 +6.0s, -7.0s, 均 > 5.0 阈值)
    add_observation(buf, setup, track, None, dnn_pred + 6.0, quality=1.0)
    add_observation(buf, setup, track, None, dnn_pred - 7.0, quality=1.0)

    corrected = corrected_lap_time(setup, track, None, buf)
    # 全部被过滤 -> 总权重=0 < _MIN_TOTAL_WEIGHT -> 退回纯 DNN
    assert corrected == pytest.approx(dnn_pred, abs=1e-6)


def test_mixed_outlier_and_normal_observations() -> None:
    """混合观测: outlier 被过滤, 正常观测参与修正.

    Iter-164.21: 阈值从 3.0 提高到 5.0, outlier 残差需 > 5.0s.
    """
    buf = ObservationBuffer()
    setup = DEFAULT_SETUP
    track = "melbourne"
    dnn_pred = predict_lap_time(setup, track, None)

    # 3 条观测: 2 正常 (残差 +0.5, +0.7) + 1 outlier (残差 +8.0, > 5.0 阈值)
    add_observation(buf, setup, track, None, dnn_pred + 0.5, quality=1.0)
    add_observation(buf, setup, track, None, dnn_pred + 0.7, quality=1.0)
    add_observation(buf, setup, track, None, dnn_pred + 8.0, quality=1.0)

    corrected = corrected_lap_time(setup, track, None, buf)
    # outlier 过滤后, 2 条正常观测加权平均 = +0.6s
    assert corrected == pytest.approx(dnn_pred + 0.6, abs=0.1)
