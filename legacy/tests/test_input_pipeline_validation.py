"""调教输入侧完整性测试 (Iter-74: driver_profile 准确性 + track context 完整性).

验证调教输入的两大维度:
- **track_context**: 全 24 赛道非零, track_type one-hot 正确, 别名一致.
- **driver_profile**: DNN 对 driver profile 敏感 (aggr/cons 差异化), 优化器
  对不同 driver 给出不同推荐, 提取的物理方向正确.
"""

from __future__ import annotations

import numpy as np
import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS
from f1opt.data.tracks import ALL_TRACKS
from f1opt.driver.profile import (
    AGGRESSIVE_PROFILE,
    CONSERVATIVE_PROFILE,
    DEFAULT_PROFILE,
    DriverProfile,
    extract_driver_profile,
)
from f1opt.model.optimizer import search_setup
from f1opt.model.surrogate import TRACK_CONTEXT_DIM, predict_lap_time, track_context


# --- track_context 完整性 ---------------------------------------------------
def test_track_context_dim_is_10() -> None:
    """track_context 维度 = 10 (length + corners + sprint + 5 one-hot + elev + unknown)."""
    assert TRACK_CONTEXT_DIM == 10


def test_all_24_tracks_have_nonzero_context() -> None:
    """全 24 赛道 track_context 非零 (无 unknown flag, 无全零向量)."""
    for track in ALL_TRACKS:
        ctx = track_context(track.track_id)
        assert ctx[-1] < 0.5, f"{track.track_id}: unknown flag set"
        assert np.any(ctx[:-1] != 0), f"{track.track_id}: zero context vector"


def test_track_type_onehot_correct() -> None:
    """track_context 的 track_type one-hot 与 Track.track_type 一致."""
    type_names = [
        "high_speed_low_downforce", "street", "high_downforce", "medium", "mixed",
    ]
    for track in ALL_TRACKS:
        ctx = track_context(track.track_id)
        onehot = ctx[3:8]
        expected_idx = type_names.index(track.track_type)
        assert onehot[expected_idx] == 1.0, (
            f"{track.track_id}: track_type one-hot 错误 "
            f"(expected idx {expected_idx}={track.track_type})"
        )
        assert onehot.sum() == 1.0, f"{track.track_id}: track_type one-hot 多于一位为 1"


@pytest.mark.parametrize(
    "alias,canonical",
    [("sakhir", "bahrain"), ("sao_paulo", "interlagos"), ("lusail", "losail")],
)
def test_track_context_alias_consistent(alias: str, canonical: str) -> None:
    """别名 vs 规范名 track_context 完全一致 (Iter-72/73 双向解析)."""
    ctx_a = track_context(alias)
    ctx_c = track_context(canonical)
    assert np.allclose(ctx_a, ctx_c), (
        f"{alias} vs {canonical}: context 不一致 (max diff={np.max(np.abs(ctx_a-ctx_c)):.4f})"
    )


# --- driver_profile 敏感性 --------------------------------------------------
@pytest.mark.parametrize("track_id", ["melbourne", "monza", "monaco", "hungaroring"])
def test_dnn_sensitive_to_driver_profile(track_id: str) -> None:
    """DNN 对 driver_profile 敏感: AGGR 比 CONS 快.

    物理基础: 激进车手 (晚刹/高承诺/ERS 攻击) 圈速更快; 保守车手虽平顺一致
    但极限承诺低. DEFAULT_PROFILE (全零) 代表"无风格信息", 不参与快慢比较.
    """
    lt_aggr = predict_lap_time(DEFAULT_SETUP, track_id, AGGRESSIVE_PROFILE)
    lt_cons = predict_lap_time(DEFAULT_SETUP, track_id, CONSERVATIVE_PROFILE)
    assert lt_aggr < lt_cons, (
        f"{track_id}: AGGR ({lt_aggr:.3f}) 应比 CONS ({lt_cons:.3f}) 快"
    )
    # DNN 对 driver_profile 非平凡响应 (aggr != cons, 差异显著)
    assert abs(lt_aggr - lt_cons) > 0.3, (
        f"{track_id}: AGGR vs CONS 差异 {abs(lt_aggr-lt_cons):.3f}s 过小 (DNN 不敏感)"
    )


def test_optimizer_differentiates_drivers() -> None:
    """优化器对不同 driver 给出不同推荐 (≥2 参数差异超半档).

    真实车队工作流: 不同车手风格需要不同调教 (如激进车手需更硬悬挂应对晚刹).
    """
    subopt = DEFAULT_SETUP.model_copy(update={"front_wing": 33, "rear_wing": 35})
    r_aggr = search_setup(
        "hungaroring", baseline=subopt,
        driver_profile=AGGRESSIVE_PROFILE, iterations=40, seed=0,
    )
    r_cons = search_setup(
        "hungaroring", baseline=subopt,
        driver_profile=CONSERVATIVE_PROFILE, iterations=40, seed=0,
    )
    diff_params = [
        n for n, spec in SETUP_FIELDS.items()
        if abs(r_aggr.recommended[n] - r_cons.recommended[n]) > spec.step * 0.5
    ]
    assert len(diff_params) >= 2, (
        f"AGGR vs CONS 仅 {len(diff_params)} 个参数不同: {diff_params}"
    )


def test_aggressive_faster_than_conservative() -> None:
    """激进车手推荐圈速 < 保守车手推荐圈速 (物理一致性)."""
    subopt = DEFAULT_SETUP.model_copy(update={"front_wing": 33, "rear_wing": 35})
    r_aggr = search_setup(
        "melbourne", baseline=subopt,
        driver_profile=AGGRESSIVE_PROFILE, iterations=40, seed=0,
    )
    r_cons = search_setup(
        "melbourne", baseline=subopt,
        driver_profile=CONSERVATIVE_PROFILE, iterations=40, seed=0,
    )
    assert r_aggr.recommended_lap_time < r_cons.recommended_lap_time


def test_extract_driver_profile_physical_direction() -> None:
    """extract_driver_profile 从合成帧提取的画像物理方向正确.

    激进帧 (高 g_lat, sharp brake) -> aggression_score > 保守帧.
    激进帧 (ers_deploy_mode=1) -> ers_usage_intensity > 保守帧.
    """
    def aggressive_frames(n=600):
        frames = []
        for i in range(n):
            brake = 1.0 if (i % 60) < 5 else 0.0
            frames.append({
                "session_time": i / 60.0, "speed": 250.0,
                "throttle": 0.9, "brake": brake, "steer": 0.0,
                "g_lat": 4.5, "g_long": 0.0, "g_vert": 1.0,
                "lap_distance": float(i), "lap_time": 80.0,
                "ers_store": 1000000.0 - i * 100, "ers_deploy_mode": 1,
                "drs_allowed": 1, "fuel_in_tank": 30.0,
            })
        return frames

    def conservative_frames(n=600):
        frames = []
        for i in range(n):
            frames.append({
                "session_time": i / 60.0, "speed": 200.0,
                "throttle": 0.5, "brake": 0.1, "steer": 0.0,
                "g_lat": 2.0, "g_long": 0.0, "g_vert": 1.0,
                "lap_distance": float(i), "lap_time": 85.0,
                "ers_store": 1000000.0, "ers_deploy_mode": 0,
                "drs_allowed": 0, "fuel_in_tank": 30.0,
            })
        return frames

    aggr = extract_driver_profile(aggressive_frames(), track_length_m=5000)
    cons = extract_driver_profile(conservative_frames(), track_length_m=5000)

    assert aggr.aggression_score > cons.aggression_score, (
        f"激进帧 aggression ({aggr.aggression_score:.3f}) 应 > 保守帧 ({cons.aggression_score:.3f})"
    )
    assert aggr.ers_usage_intensity > cons.ers_usage_intensity, (
        f"激进帧 ers_usage ({aggr.ers_usage_intensity:.3f}) 应 > 保守帧 ({cons.ers_usage_intensity:.3f})"
    )


def test_driver_profile_vector_roundtrip() -> None:
    """DriverProfile.to_vector / from_vector 互逆 (8 维 [0,1])."""
    for profile in [DEFAULT_PROFILE, AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE]:
        vec = profile.to_vector()
        assert len(vec) == 8
        assert all(0.0 <= v <= 1.0 for v in vec)
        restored = DriverProfile.from_vector(vec)
        assert restored.to_vector() == pytest.approx(vec, abs=1e-9)
