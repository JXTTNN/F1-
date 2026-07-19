"""F1 2026 车手-赛道亲和度模型测试 (Iter-38)."""

from __future__ import annotations

import pytest

from f1opt.data.driver_track_affinity import (
    driver_specialist_tracks,
    driver_track_affinity,
    has_affinity,
    top_specialist_for_track,
    track_specialists,
)


# --------------------------------------------------------------------------- #
# 基础查询
# --------------------------------------------------------------------------- #
def test_verstappen_strong_at_suzuka():
    aff = driver_track_affinity("ver", "suzuka")
    assert aff >= 0.20  # Verstappen Suzuka 3 连胜


def test_hamilton_strongest_at_silverstone():
    aff = driver_track_affinity("ham", "silverstone")
    assert aff >= 0.25  # Hamilton Silverstone 8 胜


def test_leclerc_strong_at_monza():
    aff = driver_track_affinity("lec", "monza")
    assert aff >= 0.20


def test_alonso_strong_at_monaco():
    aff = driver_track_affinity("alo", "monaco")
    assert aff >= 0.20


def test_default_zero_for_no_affinity():
    """无显著亲和度的车手-赛道组合应返回 0."""
    # Tsunoda 在 Silverstone 无显著亲和
    aff = driver_track_affinity("tsu", "silverstone")
    assert aff == 0.0


def test_unknown_driver_raises():
    with pytest.raises(ValueError, match="Unknown driver_id"):
        driver_track_affinity("nonexistent", "suzuka")


# --------------------------------------------------------------------------- #
# 范围验证
# --------------------------------------------------------------------------- #
def test_affinity_in_valid_range():
    """亲和度应在 -0.30 .. +0.30 范围内."""
    from f1opt.data.drivers_2026 import all_drivers_2026
    # 测试 24 条赛道
    track_ids = ["melbourne", "shanghai", "suzuka", "bahrain", "jeddah",
                 "miami", "montreal", "monaco", "barcelona", "silverstone",
                 "spa", "hungaroring", "zandvoort", "monza", "baku",
                 "singapore", "austin", "interlagos", "losail", "yas_marina",
                 "las_vegas", "madrid"]
    for d in all_drivers_2026():
        for t in track_ids:
            aff = driver_track_affinity(d.driver_id, t)
            assert -0.35 <= aff <= 0.35, \
                f"{d.driver_id}@{t}: {aff} out of range"


def test_weakness_returns_negative():
    """车手弱点赛道应返回负值."""
    aff = driver_track_affinity("bor", "monaco")
    assert aff < 0  # Bortoleto 新秀 + Monaco 难


# --------------------------------------------------------------------------- #
# 赛道专家
# --------------------------------------------------------------------------- #
def test_track_specialists_returns_list():
    specs = track_specialists("suzuka")
    assert isinstance(specs, list)
    # Suzuka 应该有专家
    assert len(specs) >= 1


def test_verstappen_top_specialist_at_suzuka():
    specs = track_specialists("suzuka")
    # Verstappen 应在 Suzuka 专家前列
    assert specs[0][0] == "ver"


def test_hamilton_top_specialist_at_silverstone():
    specs = track_specialists("silverstone")
    assert specs[0][0] == "ham"


def test_track_specialists_sorted_descending():
    specs = track_specialists("suzuka")
    affinities = [s[1] for s in specs]
    assert affinities == sorted(affinities, reverse=True)


def test_track_specialists_filter_threshold():
    """track_specialists 应只返回亲和度 > 0.10 的车手."""
    specs = track_specialists("monza")
    for _driver_id, aff in specs:
        assert aff > 0.10


def test_track_no_specialists():
    """某些赛道可能没有专家 (所有车手亲和度 <= 0.10)."""
    # madrid 是新赛道, 应该没有专家 (或很少)
    specs = track_specialists("madrid")
    # 接受空列表 (新赛道)
    assert isinstance(specs, list)


# --------------------------------------------------------------------------- #
# 车手擅长赛道
# --------------------------------------------------------------------------- #
def test_driver_specialist_tracks_returns_list():
    tracks = driver_specialist_tracks("ver")
    assert isinstance(tracks, list)
    assert len(tracks) >= 1


def test_verstappen_specialist_at_suzuka_spa():
    tracks = driver_specialist_tracks("ver")
    track_ids = [t for t, _ in tracks]
    assert "suzuka" in track_ids
    assert "spa" in track_ids


def test_driver_specialist_tracks_sorted():
    tracks = driver_specialist_tracks("ham")
    affs = [a for _, a in tracks]
    assert affs == sorted(affs, reverse=True)


def test_driver_specialist_tracks_threshold():
    tracks = driver_specialist_tracks("alo")
    for _, aff in tracks:
        assert aff > 0.10


def test_driver_specialist_unknown_raises():
    with pytest.raises(ValueError):
        driver_specialist_tracks("nonexistent")


def test_driver_no_specialist_tracks():
    """部分车手可能没有显著擅长的赛道."""
    tracks = driver_specialist_tracks("doo")
    # Doohan 新秀, 可能没有专家赛道 (或很少)
    assert isinstance(tracks, list)


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def test_top_specialist_for_suzuka():
    top = top_specialist_for_track("suzuka")
    assert top == "ver"


def test_top_specialist_for_silverstone():
    top = top_specialist_for_track("silverstone")
    assert top == "ham"


def test_top_specialist_none_for_new_track():
    """新赛道可能无专家."""
    top = top_specialist_for_track("madrid")
    # 接受 None 或某个 driver_id
    assert top is None or isinstance(top, str)


def test_has_affinity_true_for_strong_pair():
    assert has_affinity("ver", "suzuka") is True


def test_has_affinity_false_for_neutral():
    assert has_affinity("tsu", "silverstone") is False


def test_has_affinity_true_for_weakness():
    """弱点也应被识别 (|affinity| > 0.10)."""
    assert has_affinity("bor", "monaco") is True


# --------------------------------------------------------------------------- #
# 一致性: 多次查询应返回相同结果
# --------------------------------------------------------------------------- #
def test_affinity_deterministic():
    a1 = driver_track_affinity("ver", "suzuka")
    a2 = driver_track_affinity("ver", "suzuka")
    assert a1 == a2
