"""DriverProfile 单元测试 (Iter-02 Task 2.1).

覆盖: 脚本化激进/保守圈提取、向量往返、空帧与 None 字段容错、
手工样例差异化、证据字典结构。
"""

from __future__ import annotations

import copy

import pytest

from f1opt.driver import (
    AGGRESSIVE_PROFILE,
    CONSERVATIVE_PROFILE,
    DEFAULT_PROFILE,
    DriverProfile,
    extract_driver_profile,
)

TRACK_LENGTH = 5000.0
N_FRAMES = 600


def _make_lap(kind: str, n: int = N_FRAMES, track_length: float = TRACK_LENGTH) -> list[dict]:
    """生成一段脚本化单圈遥测帧 (60Hz, lap_distance 单调 0..track_length)。

    aggressive: 直道全油门带几次深浅不一的油门脉冲 + 0.65 处猛踩晚制动 +
    弯中高 g_lat (5g) + DRS/ERS 大量使用。
    conservative: 平顺渐收油门 + 0.20 处早缓制动 (爬坡到 0.4) + 弯中低 g_lat
    (1g) + DRS/ERS 极少使用。
    """
    frames: list[dict] = []
    for i in range(n):
        t = i / 60.0
        frac = i / (n - 1)
        lap_distance = frac * track_length
        if kind == "aggressive":
            # 油门: 直道全开, 两次深浅不一的 lift-and-stab 制造变化的正向梯度。
            if frac < 0.30:
                throttle = 1.0
            elif frac < 0.32:
                throttle = 0.2
            elif frac < 0.50:
                throttle = 1.0
            elif frac < 0.52:
                throttle = 0.6
            elif frac < 0.65:
                throttle = 1.0
            else:
                throttle = 0.3
            # 晚制动: 0.65 处从 0 一步跳到 1.0。
            brake = 1.0 if frac >= 0.65 else 0.0
            # 高承诺弯中: 大转向 + 高 g_lat。
            if 0.66 < frac < 0.85:
                steer = 0.6
                g_lat = 5.0
            else:
                steer = 0.0
                g_lat = 0.0
            ers_deploy_mode = 1 if frac < 0.60 else 0
            ers_store = 1.0 - 0.5 * frac  # 持续下降
            drs_allowed = 1 if (frac < 0.20 or 0.40 < frac < 0.50) else 0
        else:  # conservative
            # 平顺油门: 渐收进弯, 平稳回补。
            if frac < 0.18:
                throttle = 1.0
            elif frac < 0.30:
                throttle = 1.0 - (frac - 0.18) / 0.12 * 0.7
            elif frac < 0.45:
                throttle = 0.3
            else:
                throttle = min(1.0, 0.3 + (frac - 0.45) / 0.15 * 0.7)
            # 早缓制动: 0.20 起在 0.02 区间内线性爬到 0.4。
            if frac < 0.20:
                brake = 0.0
            else:
                brake = min(0.4, (frac - 0.20) / 0.02 * 0.4)
            # 低承诺弯中: 小转向 + 低 g_lat。
            if 0.22 < frac < 0.40:
                steer = 0.4
                g_lat = 1.0
            else:
                steer = 0.0
                g_lat = 0.0
            ers_deploy_mode = 0
            ers_store = 0.5 + 0.1 * frac  # 基本持平/略升
            drs_allowed = 1 if frac < 0.10 else 0
        frames.append(
            {
                "session_time": t,
                "lap_distance": lap_distance,
                "throttle": throttle,
                "brake": brake,
                "steer": steer,
                "g_lat": g_lat,
                "ers_deploy_mode": ers_deploy_mode,
                "ers_store": ers_store,
                "drs_allowed": drs_allowed,
            }
        )
    return frames


# --- 脚本化圈提取 ----------------------------------------------------------
def test_aggressive_lap_high_aggression_late_braking() -> None:
    """激进圈: aggression_score > 0.6 且 brake_point_norm < 0.5 (晚制动)。"""
    frames = _make_lap("aggressive")
    prof = extract_driver_profile(frames, track_length_m=TRACK_LENGTH)
    assert prof.aggression_score > 0.6
    assert prof.brake_point_norm < 0.5


def test_conservative_lap_low_aggression_early_braking() -> None:
    """保守圈: aggression_score < 0.4 且 brake_point_norm > 0.5 (早制动)。"""
    frames = _make_lap("conservative")
    prof = extract_driver_profile(frames, track_length_m=TRACK_LENGTH)
    assert prof.aggression_score < 0.4
    assert prof.brake_point_norm > 0.5


def test_extracted_profile_all_fields_in_range() -> None:
    """提取出的 8 个字段均落在 [0,1]。"""
    prof = extract_driver_profile(_make_lap("aggressive"), track_length_m=TRACK_LENGTH)
    for name in (
        "brake_point_norm",
        "throttle_smoothness",
        "steer_smoothness",
        "corner_balance_pref",
        "aggression_score",
        "consistency_score",
        "ers_usage_intensity",
        "drs_usage_efficiency",
    ):
        v = getattr(prof, name)
        assert 0.0 <= v <= 1.0, f"{name}={v} 越界"


# --- 向量往返 --------------------------------------------------------------
def test_to_vector_length_and_range() -> None:
    """to_vector 返回 8 个 [0,1] 浮点。"""
    vec = AGGRESSIVE_PROFILE.to_vector()
    assert len(vec) == 8
    assert all(isinstance(v, float) for v in vec)
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_from_vector_round_trip() -> None:
    """to_vector -> from_vector 还原同一向量 (值已落在 [0,1])。"""
    for prof in (DEFAULT_PROFILE, AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE):
        vec = prof.to_vector()
        rebuilt = DriverProfile.from_vector(vec)
        assert rebuilt.to_vector() == pytest.approx(vec, abs=1e-12)


def test_from_vector_clamps_out_of_range() -> None:
    """from_vector 把越界值钳位到 [0,1]。"""
    rebuilt = DriverProfile.from_vector([-0.5, 1.5, 2.0, -1.0, 0.3, 0.7, 0.2, 0.9])
    vec = rebuilt.to_vector()
    assert vec[0] == 0.0
    assert vec[1] == 1.0
    assert vec[2] == 1.0
    assert vec[3] == 0.0


def test_from_vector_wrong_length_raises() -> None:
    """向量长度不符抛出 ValueError。"""
    with pytest.raises(ValueError):
        DriverProfile.from_vector([0.0] * 7)


# --- 空帧与 None 字段容错 --------------------------------------------------
def test_empty_frames_returns_zeros() -> None:
    """空帧返回全零画像, to_vector = 8 个 0, 不崩溃。"""
    prof = extract_driver_profile([])
    assert prof.to_vector() == [0.0] * 8
    assert prof.evidence() == {}


def test_none_fields_skipped_gracefully() -> None:
    """部分帧 g_lat 为 None 时被跳过, 不崩溃且仍产出合法画像。"""
    frames = _make_lap("aggressive")
    for i, f in enumerate(frames):
        if i % 3 == 0:
            f["g_lat"] = None
    prof = extract_driver_profile(frames, track_length_m=TRACK_LENGTH)
    # 不崩溃, 所有字段仍在 [0,1]。
    assert all(0.0 <= v <= 1.0 for v in prof.to_vector())
    # g_lat 相关指标仍计算 (非 None 帧足够)。
    assert 0.0 <= prof.corner_balance_pref <= 1.0


def test_all_none_brake_no_onset() -> None:
    """全帧无 brake 数据时 brake_point_norm 退化为 0, 不崩溃。"""
    frames = _make_lap("aggressive")
    for f in frames:
        f["brake"] = None
    prof = extract_driver_profile(frames, track_length_m=TRACK_LENGTH)
    assert prof.brake_point_norm == 0.0


# --- 手工样例差异化 --------------------------------------------------------
def test_exemplars_differ_in_enough_fields() -> None:
    """AGGRESSIVE 与 CONSERVATIVE 至少 5 个字段不同。"""
    fields = [
        "brake_point_norm",
        "throttle_smoothness",
        "steer_smoothness",
        "corner_balance_pref",
        "aggression_score",
        "consistency_score",
        "ers_usage_intensity",
        "drs_usage_efficiency",
    ]
    differing = [
        name
        for name in fields
        if getattr(AGGRESSIVE_PROFILE, name) != getattr(CONSERVATIVE_PROFILE, name)
    ]
    assert len(differing) >= 5


def test_exemplars_vector_difference_signal() -> None:
    """两样例 to_vector 至少 3 个维度相差 >= 0.3。"""
    a = AGGRESSIVE_PROFILE.to_vector()
    c = CONSERVATIVE_PROFILE.to_vector()
    big_diffs = [abs(x - y) for x, y in zip(a, c, strict=True) if abs(x - y) >= 0.3]
    assert len(big_diffs) >= 3


def test_default_profile_is_zeros() -> None:
    """DEFAULT_PROFILE 为全零。"""
    assert DEFAULT_PROFILE.to_vector() == [0.0] * 8


# --- 证据 ------------------------------------------------------------------
def test_evidence_non_empty_and_well_formed() -> None:
    """非空帧提取后 evidence 非空, 每条含 frame_t/field/value。"""
    frames = _make_lap("aggressive")
    prof = extract_driver_profile(frames, track_length_m=TRACK_LENGTH)
    ev = prof.evidence()
    assert ev, "evidence 不应为空"
    for metric, entry in ev.items():
        assert {"frame_t", "field", "value"} <= set(entry.keys()), metric
        assert isinstance(entry["frame_t"], float)
        assert isinstance(entry["field"], str)
        assert isinstance(entry["value"], float)
        assert 0.0 <= entry["value"] <= 1.0


def test_evidence_excluded_from_serialization() -> None:
    """证据不应出现在 pydantic 序列化模型中。"""
    prof = extract_driver_profile(_make_lap("aggressive"), track_length_m=TRACK_LENGTH)
    dumped = prof.model_dump()
    assert "_evidence" not in dumped
    assert "evidence" not in dumped


def test_exemplar_profiles_have_no_evidence() -> None:
    """手工样例不带证据 (evidence 为空)。"""
    assert AGGRESSIVE_PROFILE.evidence() == {}
    assert CONSERVATIVE_PROFILE.evidence() == {}


def test_extract_does_not_mutate_input() -> None:
    """提取不修改输入帧列表。"""
    frames = _make_lap("aggressive")
    snapshot = copy.deepcopy(frames)
    extract_driver_profile(frames, track_length_m=TRACK_LENGTH)
    assert frames == snapshot
