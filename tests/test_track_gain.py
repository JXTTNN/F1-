"""赛道敏感度系数回归测试 —— 建议必须「因赛道而异」（S1）。

背景（2026-09 审计）：``generate_suggestion`` 虽然接收 ``track_id``，
但规则引擎的 ``Dx × C`` 与赛道无关（track_id 只传给当时不可用的 NN 分支），
云端实测 **suzuka 与 monza 在相同症状下 setup_delta 逐位相同** ——
24 条赛道共用一套建议。``domain/track_coefficients.py`` 引入按 ``track_type``
分组的温和敏感度倍数（0.85–1.25，只改幅度不改方向）。
"""

from __future__ import annotations

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.track import get_all_tracks, get_track_by_id
from setup_tuner.domain.track_coefficients import (
    NEUTRAL_GAIN,
    gain_for_track_type,
    track_gain_for,
)
from setup_tuner.engine.engine import compute_setup_delta, generate_suggestion

_SYMPTOMS = [("understeer", 3), ("exit_wheelspin", 2)]


# ===========================================================================
# 1. 系数本身的约束
# ===========================================================================
class TestGainDefinition:
    """系数覆盖面与取值区间。"""

    def test_covers_all_params(self) -> None:
        """每类赛道的系数都覆盖全部 21 参数。"""
        for track_type in ("medium", "mixed", "street", "high_downforce",
                           "high_speed_low_downforce"):
            gain = gain_for_track_type(track_type)
            assert set(gain) == {f.name for f in ALL_SETUP_FIELDS}

    def test_moderate_range_only(self) -> None:
        """系数限制在 0.85–1.25：只改敏感度，不做激进缩放。"""
        for track_type in ("street", "high_downforce", "high_speed_low_downforce"):
            for param, factor in gain_for_track_type(track_type).items():
                assert 0.85 <= factor <= 1.25, f"{track_type}/{param} 系数越界：{factor}"

    def test_neutral_types_are_identity(self) -> None:
        """medium / mixed 不做臆测，全部保持 1.0。"""
        assert gain_for_track_type("medium") == NEUTRAL_GAIN
        assert gain_for_track_type("mixed") == NEUTRAL_GAIN

    def test_unknown_track_is_neutral(self) -> None:
        """未知赛道返回中性系数（不臆测）。"""
        assert track_gain_for("__nope__") == NEUTRAL_GAIN

    def test_all_tracks_resolvable(self) -> None:
        """24 条赛道都能取到系数，且只含合法参数键。"""
        valid = {f.name for f in ALL_SETUP_FIELDS}
        for track in get_all_tracks():
            gain = track_gain_for(track.track_id)
            assert set(gain) <= valid


# ===========================================================================
# 2. 建议确实因赛道而异
# ===========================================================================
class TestSuggestionVariesByTrack:
    """同症状不同赛道 → 结果不同（Q1 已修复）。"""

    def test_suzuka_differs_from_monza(self) -> None:
        """suzuka(mixed) 与 monza(high_speed_low_downforce) 结果必须不同。"""
        suzuka = get_track_by_id("suzuka")
        monza = get_track_by_id("monza")
        assert suzuka is not None and monza is not None
        assert suzuka.track_type != monza.track_type

        setup = CarSetup.default().to_dict()
        a = generate_suggestion(_SYMPTOMS, setup, "suzuka", None, model_type="rule")
        b = generate_suggestion(_SYMPTOMS, setup, "monza", None, model_type="rule")
        assert a["setup_delta"] != b["setup_delta"], "不同赛道仍输出相同建议"

    def test_monza_amplifies_aero_and_damps_ride_height(self) -> None:
        """monza 翼片敏感度↑、离地间隙敏感度↓（低下压力赛道特性）。"""
        gain = track_gain_for("monza")
        assert gain["front_wing"] > 1.0
        assert gain["rear_wing"] > 1.0
        assert gain["front_ride_height"] < 1.0

    def test_street_circuit_amplifies_ride_height(self) -> None:
        """街道赛（monaco）离地间隙最敏感、翼片最不敏感。"""
        gain = track_gain_for("monaco")
        assert gain["front_ride_height"] > 1.0
        assert gain["front_wing"] < 1.0

    def test_direction_not_flipped(self) -> None:
        """赛道系数只缩放幅度：非饱和参数上的调整方向不变。"""
        setup = CarSetup.default().to_dict()
        neutral = compute_setup_delta(
            generate_suggestion(_SYMPTOMS, setup, "suzuka", None, model_type="rule")["dx"],
            setup,
        )
        neutral_gen = {f.name: f for f in ALL_SETUP_FIELDS}
        for track_id in ("monza", "monaco", "hungaroring"):
            scaled = generate_suggestion(_SYMPTOMS, setup, track_id, None, model_type="rule")
            for name, delta in scaled["setup_delta"].items():
                base = neutral[name]
                if base == 0.0 or delta == 0.0:
                    continue
                # 未饱和的参数方向必须一致
                if abs(abs(base) - neutral_gen[name].max_delta) < 1e-9:
                    continue
                assert (delta > 0) == (base > 0), f"{track_id}/{name} 方向被翻转"

    def test_deterministic(self) -> None:
        """相同输入两次结果完全一致（确定性保持）。"""
        setup = CarSetup.default().to_dict()
        first = generate_suggestion(_SYMPTOMS, setup, "monza", None, model_type="rule")
        second = generate_suggestion(_SYMPTOMS, setup, "monza", None, model_type="rule")
        assert first["setup_delta"] == second["setup_delta"]
