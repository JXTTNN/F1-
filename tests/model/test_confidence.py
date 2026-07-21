"""调教分析侧: 置信度评估测试 (Iter-76)."""

from __future__ import annotations

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.data.tracks import TRACKS_BY_ID
from f1opt.model.confidence import confidence_label, prediction_confidence
from f1opt.model.setup_physics_bridge import optimal_setup_for_track_type


def test_confidence_in_unit_interval() -> None:
    """置信度在 [0, 1] 区间."""
    for track in ["melbourne", "monza", "monaco", "hungaroring", "spa"]:
        c = prediction_confidence(DEFAULT_SETUP, track)
        assert 0.0 <= c <= 1.0, f"{track}: confidence={c} 超出 [0,1]"


def test_optimal_setup_high_confidence() -> None:
    """赛道类型最优 setup 置信度 > 0.8 (penalty=0, V 谷底)."""
    for track_id in ["melbourne", "barcelona", "spielberg"]:
        track = TRACKS_BY_ID[track_id]
        opt = optimal_setup_for_track_type(track.track_type)
        c = prediction_confidence(opt, track_id)
        assert c > 0.8, f"{track_id}: optimal setup confidence={c:.3f} 应 > 0.8"


def test_suboptimal_setup_lower_confidence() -> None:
    """次优 setup 置信度 < 最优 setup (penalty 大 -> 置信度低)."""
    track = TRACKS_BY_ID["melbourne"]
    opt = optimal_setup_for_track_type(track.track_type)
    subopt = opt.model_copy(update={"front_wing": 50, "rear_wing": 50})  # 极端偏离
    c_opt = prediction_confidence(opt, "melbourne")
    c_sub = prediction_confidence(subopt, "melbourne")
    assert c_opt > c_sub, (
        f"最优 confidence ({c_opt:.3f}) 应 > 次优 ({c_sub:.3f})"
    )


def test_extreme_setup_low_confidence() -> None:
    """极端 setup (所有参数拉满) 置信度 < 0.5 (外推严重)."""
    extreme = DEFAULT_SETUP.model_copy(update={
        "front_wing": 50, "rear_wing": 50, "front_arb": 50, "rear_arb": 50,
        "front_camber": -5.0, "rear_camber": -5.0,
    })
    c = prediction_confidence(extreme, "melbourne")
    assert c < 0.5, f"极端 setup confidence={c:.3f} 应 < 0.5"


def test_confidence_label_thresholds() -> None:
    """confidence_label 阈值正确."""
    assert confidence_label(0.9) == "high"
    assert confidence_label(0.8) == "high"
    assert confidence_label(0.79) == "medium"
    assert confidence_label(0.5) == "medium"
    assert confidence_label(0.49) == "low"
    assert confidence_label(0.1) == "low"


def test_confidence_monotonic_with_penalty() -> None:
    """置信度随 penalty 单调递减 (偏离最优越远越不置信)."""
    track = TRACKS_BY_ID["melbourne"]
    opt = optimal_setup_for_track_type(track.track_type)
    confidences = []
    # 逐步增加 front_wing 偏离
    for fw in [opt.front_wing, opt.front_wing + 3, opt.front_wing + 6,
               opt.front_wing + 10, opt.front_wing + 15]:
        s = opt.model_copy(update={"front_wing": fw})
        confidences.append(prediction_confidence(s, "melbourne"))
    # 应单调递减
    for i in range(len(confidences) - 1):
        assert confidences[i] >= confidences[i + 1], (
            f"confidence 非单调递减: {confidences}"
        )
