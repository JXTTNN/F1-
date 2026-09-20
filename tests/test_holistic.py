"""整体思维调教分析层测试（``engine/holistic``）。

锁定四件事：

1. **赛道需求画像由真实弯道数据算出** —— 不同弯型构成的赛道必须得到不同画像
   （这正是修掉"同 track_type 分组导致 Monza 与 Jeddah 输出逐位相同"的关键）。
2. **同一症状在不同类别弯道上产生不同的调教含义** —— 慢弯重机械抓地/牵引，
   快弯重空气动力学/高速稳定性。这是"准确结合各弯道特性"的可验证形式。
3. **跨弯道类别的参数意图冲突可被识别**（慢弯要软、快弯要硬这类不可兼得）。
4. **整体性收口有效** —— 胎压左右对称、前后翼平衡窗口、改动预算。
"""

from __future__ import annotations

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.domain.track import get_track_by_id
from setup_tuner.engine.diagnostic import DIAG_DIMS
from setup_tuner.engine.holistic import (
    FAST,
    MEDIUM,
    SLOW,
    class_weighted_dx,
    class_weights,
    corner_class,
    holistic_coherence,
    track_demand,
)

_ALL_TRACKS = [
    "melbourne", "shanghai", "suzuka", "sakhir", "jeddah", "miami", "montreal",
    "monaco", "barcelona", "spielberg", "silverstone", "spa", "hungaroring",
    "zandvoort", "monza", "madrid", "baku", "singapore", "austin",
    "mexico_city", "sao_paulo", "las_vegas", "lusail", "yas_marina",
]


def _corners_of_class(track_id: str, klass: str) -> list[int]:
    track = get_track_by_id(track_id)
    assert track is not None
    return [c.number for c in track.corners if corner_class(c.corner_type) == klass]


# ===========================================================================
# 1. 赛道需求画像
# ===========================================================================
class TestTrackDemand:
    """需求画像必须来自真实弯道数据。"""

    @pytest.mark.parametrize("track_id", _ALL_TRACKS)
    def test_demand_available_for_all_tracks(self, track_id: str) -> None:
        d = track_demand(track_id)
        assert d.corner_count == len(get_track_by_id(track_id).corners)
        assert d.slow + d.medium + d.fast == d.corner_count

    @pytest.mark.parametrize("track_id", _ALL_TRACKS)
    def test_shares_sum_to_one(self, track_id: str) -> None:
        d = track_demand(track_id)
        assert d.slow_share + d.medium_share + d.fast_share == pytest.approx(1.0)

    @pytest.mark.parametrize("track_id", _ALL_TRACKS)
    def test_indices_in_unit_range(self, track_id: str) -> None:
        d = track_demand(track_id)
        for v in (d.traction_index, d.aero_index, d.braking_index):
            assert 0.0 <= v <= 1.0

    def test_profile_differs_between_dissimilar_tracks(self) -> None:
        """弯型构成不同的赛道必须得到不同画像（否则等于没接弯道特性）。"""
        monza = track_demand("monza")
        jeddah = track_demand("jeddah")
        assert monza.describe() != jeddah.describe()
        # Jeddah 快弯多、尾速高 → 空力需求应高于 Monza
        assert jeddah.aero_index > monza.aero_index

    def test_traction_index_tracks_slow_corners(self) -> None:
        """慢弯占比高的赛道牵引需求更高。"""
        monaco = track_demand("monaco")
        zandvoort = track_demand("zandvoort")
        assert monaco.slow_share > zandvoort.slow_share
        assert monaco.traction_index > zandvoort.traction_index

    def test_longest_sequence_is_meaningful(self) -> None:
        """存在连续弯段的赛道最长段应 >= 2，且不超过总弯数。"""
        d = track_demand("jeddah")
        assert 2 <= d.longest_sequence <= d.corner_count


# ===========================================================================
# 2. 弯道类别改变调教含义（"结合各弯道特性"）
# ===========================================================================
class TestCornerClassWeighting:
    """同一症状在慢弯与快弯必须得到不同的 Dx 形状。"""

    def test_corner_class_mapping(self) -> None:
        assert corner_class("slow") == SLOW
        assert corner_class("medium") == MEDIUM
        assert corner_class("fast") == FAST
        assert corner_class("SLOW ") == SLOW
        assert corner_class("未知") == MEDIUM  # 保守回退

    def test_weights_differ_across_classes(self) -> None:
        w_slow = class_weights("midcorner_understeer", SLOW)
        w_fast = class_weights("midcorner_understeer", FAST)
        assert w_slow and w_fast
        # 慢弯：机械前抓地权重更高；快弯：高速稳定性权重更高
        assert w_slow["front_grip_req"] > w_fast["front_grip_req"]
        assert w_fast["hi_speed_stab_req"] > w_slow["hi_speed_stab_req"]

    def test_same_symptom_slow_vs_fast_gives_different_dx(self) -> None:
        """同症状在慢弯/快弯上 → 不同的 Dx（这是本层存在的意义）。"""
        slow_n = _corners_of_class("zandvoort", SLOW)[0]
        fast_n = _corners_of_class("zandvoort", FAST)[0]

        def fb(n: int) -> list[dict]:
            return [{"corner_number": n, "symptom": "midcorner_understeer",
                     "strength": 3, "category": "apex"}]

        dx_slow = class_weighted_dx(fb(slow_n), "zandvoort").dx
        dx_fast = class_weighted_dx(fb(fast_n), "zandvoort").dx
        assert dx_slow != dx_fast
        # 慢弯：机械前抓地占比更高
        assert dx_slow["front_grip_req"] > dx_fast["front_grip_req"]
        # 快弯：高速稳定性占比更高
        assert dx_fast["hi_speed_stab_req"] > dx_slow["hi_speed_stab_req"]

    def test_by_class_breakdown_is_attributed(self) -> None:
        slow_n = _corners_of_class("zandvoort", SLOW)[0]
        fast_n = _corners_of_class("zandvoort", FAST)[0]
        r = class_weighted_dx([
            {"corner_number": slow_n, "symptom": "understeer", "strength": 3,
             "category": "apex"},
            {"corner_number": fast_n, "symptom": "oversteer", "strength": 3,
             "category": "apex"},
        ], "zandvoort")
        assert r.by_class[SLOW]["front_grip_req"] > 0
        assert r.by_class[FAST]["rear_grip_req"] > 0
        # 分解之和等于最终 Dx（收口前）
        for dim in DIAG_DIMS:
            total = sum(r.by_class[k][dim] for k in (SLOW, MEDIUM, FAST))
            assert total > 0 or r.dx[dim] == 0

    def test_global_feedback_uses_medium_class(self) -> None:
        """无弯道号（全局症状）按中速类处理，不崩溃。"""
        r = class_weighted_dx(
            [{"corner_number": None, "symptom": "lap_slow", "strength": 3,
              "category": "global"}], "suzuka")
        assert r.dx["tyre_life_req"] >= 0

    def test_duplicate_feedback_keeps_strongest_only(self) -> None:
        """同一 (弯号, 症状) 重复提交不得线性放大（沿用既有聚合口径）。"""
        fb = {"corner_number": 1, "symptom": "understeer", "strength": 3,
              "category": "apex"}
        one = class_weighted_dx([fb], "suzuka").dx
        many = class_weighted_dx([fb] * 20, "suzuka").dx
        assert one == many

    def test_unknown_symptom_ignored(self) -> None:
        r = class_weighted_dx(
            [{"corner_number": 1, "symptom": "__nope__", "strength": 3}], "suzuka")
        assert all(v == 0 for v in r.dx.values())

    def test_deterministic(self) -> None:
        fb = [{"corner_number": 1, "symptom": "oversteer", "strength": 4,
               "category": "exit"}]
        assert class_weighted_dx(fb, "spa").dx == class_weighted_dx(fb, "spa").dx


# ===========================================================================
# 3. 跨类别冲突
# ===========================================================================
class TestCrossClassConflicts:
    """慢弯与快弯对同一参数的正反要求必须被识别出来。"""

    def test_conflict_detectable(self) -> None:
        """"慢弯刮底 + 快弯出弯甩尾"在后轴参数上正相反 —— 这是真实权衡。"""
        slow_n = _corners_of_class("zandvoort", SLOW)[0]
        fast_n = _corners_of_class("zandvoort", FAST)[0]
        r = class_weighted_dx([
            {"corner_number": slow_n, "symptom": "bottoming", "strength": 5,
             "category": "apex"},
            {"corner_number": fast_n, "symptom": "exit_oversteer", "strength": 5,
             "category": "exit"},
        ], "zandvoort")
        assert r.conflicts, "应识别出跨类别参数冲突"
        for c in r.conflicts:
            assert c.slow_intent * c.fast_intent < 0
            assert c.param in {f.name for f in ALL_SETUP_FIELDS}
            assert c.describe()

    def test_no_conflict_for_single_class(self) -> None:
        """只有一类弯的反馈时不存在跨类别冲突。"""
        slow_n = _corners_of_class("zandvoort", SLOW)[0]
        r = class_weighted_dx(
            [{"corner_number": slow_n, "symptom": "bottoming", "strength": 5}],
            "zandvoort")
        assert not r.conflicts


# ===========================================================================
# 4. 整体性收口
# ===========================================================================
class TestHolisticCoherence:
    """收口必须真的改变结果，并给出可读的权衡说明。"""

    def test_tyre_pressures_made_symmetric(self) -> None:
        delta = {"front_left_tyre_pressure": 0.4,
                 "front_right_tyre_pressure": -0.2,
                 "rear_left_tyre_pressure": 0.3,
                 "rear_right_tyre_pressure": 0.1}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert out["front_left_tyre_pressure"] == out["front_right_tyre_pressure"]
        assert out["rear_left_tyre_pressure"] == out["rear_right_tyre_pressure"]
        assert any("胎压" in n for n in notes)

    def test_wing_balance_window_enforced(self) -> None:
        delta = {"front_wing": 5.0, "rear_wing": -5.0}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert abs(out["front_wing"] - out["rear_wing"]) <= 3.0
        assert any("前后翼" in n for n in notes)

    def test_change_budget_enforced(self) -> None:
        delta = {f.name: 0.5 for f in ALL_SETUP_FIELDS}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert sum(1 for v in out.values() if v) <= 12
        assert any("改动预算" in n for n in notes)

    def test_small_change_set_untouched(self) -> None:
        """改动很少时不应被无谓收口。"""
        delta = {"front_wing": 1.0}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert out["front_wing"] == 1.0
        assert not any("改动预算" in n for n in notes)

    # ---- task-82：悬挂几何 / 防倾杆 / rake / 压路肩冲突 ----

    def test_camber_pair_only_larger_side_shrinks(self) -> None:
        """外倾配对：只回收变化更大的一侧，**不得在零变化侧凭空造值**。

        对称等比收口曾把 rear_camber 从 0 变成非零（无出处）——已改为
        「回收大侧」，保证出处可追溯。
        """
        delta = {"front_camber": -0.45, "rear_camber": 0.0}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert out["rear_camber"] == 0.0, "不应在零变化侧凭空造出反向变化"
        assert abs(out["front_camber"] - out["rear_camber"]) <= 0.30
        assert any("外倾" in n for n in notes)

    def test_toe_pair_within_window_untouched(self) -> None:
        """束角变化在窗口内 → 不收口。"""
        delta = {"front_toe": 0.02, "rear_toe": 0.01}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert out["front_toe"] == 0.02
        assert not any("束角" in n for n in notes)

    def test_arb_pair_larger_side_shrinks(self) -> None:
        """防倾杆前后差值超窗 → 回收大侧到窗口内。"""
        delta = {"front_anti_roll_bar": 4.0, "rear_anti_roll_bar": -1.0}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert abs(out["front_anti_roll_bar"] - out["rear_anti_roll_bar"]) <= 2.0
        assert any("防倾杆" in n for n in notes)

    def test_rake_change_window(self) -> None:
        """前后离地变化（rake）超窗 → 收口。"""
        delta = {"front_ride_height": -3.0, "rear_ride_height": 3.0}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert abs(out["rear_ride_height"] - out["front_ride_height"]) <= 3.0
        assert any("rake" in n for n in notes)

    def test_kerb_blocks_ride_height_reduction(self) -> None:
        """压路肩冲突收口：遥测检出需要更高离地 → 撤回降低离地的建议。

        路肩不得不压——此时降低离地等于让车更吃不了路肩。
        """
        dx = {"ride_height_req": 0.30}  # 引擎规则17：重度路肩
        delta = {"front_ride_height": -2.0, "rear_wing": 1.0}
        out, notes = holistic_coherence(delta, dx, track_demand("suzuka"))
        assert out["front_ride_height"] == 0.0, "降低离地的建议应被撤回"
        assert out["rear_wing"] == 1.0, "其它建议不受影响"
        assert any("路肩" in n for n in notes)

    def test_no_ride_height_signal_keeps_reduction(self) -> None:
        """无路肩/刮底信号时降低离地不受限（正常优化路径）。"""
        delta = {"front_ride_height": -2.0}
        out, notes = holistic_coherence(delta, {}, track_demand("suzuka"))
        assert out["front_ride_height"] == -2.0
        assert not any("路肩" in n for n in notes)

    def test_traction_track_notes(self) -> None:
        """慢弯占比高的赛道应给出牵引优先的说明。"""
        _, notes = holistic_coherence({}, {}, track_demand("monaco"))
        assert any("牵引" in n for n in notes)

    def test_aero_track_notes(self) -> None:
        _, notes = holistic_coherence({}, {}, track_demand("jeddah"))
        assert any("下压力" in n for n in notes)

    def test_long_sequence_notes(self) -> None:
        """连续弯段长的赛道应提示整车平衡一致性。"""
        _, notes = holistic_coherence({}, {}, track_demand("jeddah"))
        assert any("连弯" in n for n in notes)

    def test_returns_new_dict(self) -> None:
        """不得原地修改入参（避免污染上游）。"""
        delta = {"front_wing": 9.0, "rear_wing": -9.0}
        snapshot = dict(delta)
        holistic_coherence(delta, {}, track_demand("suzuka"))
        assert delta == snapshot


# ===========================================================================
# 5. 与引擎/接口的接线（弯道号必须真的传到模型，不再被丢弃）
# ===========================================================================
class TestWiringIntoEngine:
    """``generate_suggestion`` 接到逐弯反馈后必须走弯道特性路径。"""

    def test_unknown_track_degrades_gracefully(self) -> None:
        """未知/合成赛道必须返回中性画像而非抛错。

        这是一次真实回归：接入初期 ``track_demand`` 对未知赛道抛 KeyError，
        打挂了 23 个既有引擎测试（它们用 'test_track' 等合成 id）。
        引擎对未知赛道历来宽容（``_derive_track_gain`` 也是中性），不得收紧。
        """
        d = track_demand("__no_such_track__")
        assert d.corner_count == 0
        assert d.traction_index == d.aero_index == d.braking_index == 0.0
        _, notes = holistic_coherence({}, {}, d)
        assert notes == []

    def test_generate_suggestion_returns_holistic_block(self) -> None:
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        track = get_track_by_id("zandvoort")
        assert track is not None
        slow_n = next(c.number for c in track.corners if c.corner_type == SLOW)
        fast_n = next(c.number for c in track.corners if c.corner_type == FAST)
        res = generate_suggestion(
            symptoms=[],
            current_setup=CarSetup().to_dict(),
            track_id="zandvoort",
            feedbacks=[
                {"corner_number": slow_n, "symptom": "bottoming", "strength": 3,
                 "category": "apex"},
                {"corner_number": fast_n, "symptom": "exit_oversteer",
                 "strength": 3, "category": "exit"},
            ],
        )
        h = res["holistic"]
        assert h["demand"]
        assert len(h["corner_notes"]) == 2, "逐弯反馈应各留一条加权依据"
        assert h["conflicts"], "慢弯刮底 vs 快弯出弯甩尾应识别出跨类别冲突"
        # 关键：弯道号确实参与了模型 —— 慢弯与快弯各自被标注为对应类别
        joined = " ".join(h["corner_notes"])
        assert f"T{slow_n}（slow）" in joined
        assert f"T{fast_n}（fast）" in joined

    def test_feedback_path_differs_from_symptom_path(self) -> None:
        """走逐弯反馈路径与走聚合症状路径的结果必须不同（否则等于没接线）。"""
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        setup = CarSetup().to_dict()
        track = get_track_by_id("zandvoort")
        assert track is not None
        slow_n = next(c.number for c in track.corners if c.corner_type == SLOW)
        by_feedback = generate_suggestion(
            symptoms=[], current_setup=setup, track_id="zandvoort",
            feedbacks=[{"corner_number": slow_n, "symptom": "bottoming",
                        "strength": 3, "category": "apex"}],
        )
        by_symptom = generate_suggestion(
            symptoms=[("bottoming", 3, "apex")], current_setup=setup,
            track_id="zandvoort",
        )
        assert by_feedback["dx"] != by_symptom["dx"]

    def test_no_feedbacks_still_works(self) -> None:
        """未提供逐弯反馈时退回原路径，且仍带 holistic 块（画像提示）。"""
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        res = generate_suggestion(
            symptoms=[("understeer", 2)], current_setup=CarSetup().to_dict(),
            track_id="monaco",
        )
        assert res["holistic"]["corner_notes"] == []
        assert res["holistic"]["coherence_notes"]


class TestReportCarriesHolistic:
    """报告必须携带整体分析（否则前端无从解释"为什么这么调"）。"""

    def test_report_contains_holistic(self) -> None:
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion
        from setup_tuner.report.builder import build_report

        res = generate_suggestion(
            symptoms=[("understeer", 2)], current_setup=CarSetup().to_dict(),
            track_id="jeddah",
        )
        report = build_report(suggestion_result=res, track_id="jeddah")
        assert "holistic" in report
        assert report["holistic"]["demand"]


# ===========================================================================
# 2026-09-20 回归：悬挂几何必须参与（用户实测反馈"悬挂几何什么的都没有"）
# ===========================================================================
class TestMechanicalParticipationAllTracks:
    """任何赛道、任何典型症状都必须给出悬挂/几何/防倾杆改动。

    背景：机械抓地参与度门槛原为 `traction_index >= 0.35`，导致中/低牵引
    赛道（suzuka 0.22 / spa 0.26 / hungaroring 0.29 / silverstone 0.17）
    的整圈最优解把机械项全部清零 —— 36 个场景里 10 个完全无机械项。
    门槛已降为 0.10（全赛道触发），需求信号扩到 抓地/牵引/入弯/高速稳定/重刹。
    本类锁定该行为：**不得回退成"只改空力/差速/胎压"**。
    """

    _MECHANICAL = (
        "front_suspension", "rear_suspension",
        "front_anti_roll_bar", "rear_anti_roll_bar",
        "front_ride_height", "rear_ride_height",
    )
    _TRACKS = ("suzuka", "spa", "hungaroring", "silverstone", "monza", "monaco")
    _SYMPTOMS = (
        "understeer", "oversteer", "exit_wheelspin",
        "midcorner_traction", "high_speed_instability",
    )

    _STIFFNESS = (
        "front_suspension", "rear_suspension",
        "front_anti_roll_bar", "rear_anti_roll_bar",
    )

    def test_every_scenario_has_mechanical_change(self) -> None:
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        setup = CarSetup.default().to_dict()
        missing: list[str] = []
        # 模型类型用 rule：本用例锁定的是**规则/收口路径**的整体性，
        # 不带神经网络模拟（否则模型可用性会影响结论）。
        for track in self._TRACKS:
            for symptom in self._SYMPTOMS:
                res = generate_suggestion(
                    [(symptom, 3)], setup, track, None, model_type="rule",
                )
                delta = res["setup_delta"]
                if not any(abs(delta.get(p, 0.0)) > 1e-9 for p in self._MECHANICAL):
                    missing.append(f"{track}/{symptom}")
        assert not missing, (
            f"以下场景完全没有悬挂/几何/防倾杆改动（整体性回归）：{missing}；"
            "检查 holistic_coherence 的机械抓地参与度规则与"
            "_TRACTION_PARTICIPATION_MIN 门槛"
        )

    def test_stiffness_params_participate(self) -> None:
        """**刚度类**（悬挂 + 防倾杆）必须参与 —— 只给前束/外倾不算调了底盘。

        用户实测反馈："调教整体性思维太差了，悬挂几何什么的都没有"。
        另一种表现形式：机械项里只剩前束角（纯几何），底盘刚度没动。
        修复：恢复时若仍无刚度类，则由耦合矩阵按主导机械需求推导
        （`_derive_mechanical_from_matrix`）。

        **必须两条路径都测**（负向验证发现的坑）：纯症状路径的规则 delta 本来
        就常带刚度项，矩阵推导用不上；真正需要它的是**反馈路径**
        （`class_weighted_dx` 按弯道重加权后，机械项可能只剩一个前束角）——
        而那正是 UI/API 的生产路径。只测纯症状路径的用例挡不住回退。
        """
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        setup = CarSetup.default().to_dict()
        missing: list[str] = []
        for track in self._TRACKS:
            for symptom in self._SYMPTOMS:
                # ① 纯症状路径
                plain = generate_suggestion(
                    [(symptom, 3)], setup, track, None, model_type="rule",
                )["setup_delta"]
                if not any(abs(plain.get(p, 0.0)) > 1e-9 for p in self._STIFFNESS):
                    missing.append(f"{track}/{symptom}[症状路径]")
                # ② 反馈路径（生产路径：带弯道号的逐弯反馈）
                fb = generate_suggestion(
                    [(symptom, 3)], setup, track, None, model_type="rule",
                    feedbacks=[{"corner_number": 3, "symptom": symptom, "strength": 3}],
                )["setup_delta"]
                if not any(abs(fb.get(p, 0.0)) > 1e-9 for p in self._STIFFNESS):
                    missing.append(f"{track}/{symptom}[反馈路径]")
        assert not missing, f"以下场景没有任何刚度类改动：{missing}"

    def test_derive_mechanical_from_matrix_is_bounded(self) -> None:
        """矩阵推导的刚度改动必须受 max_delta / 档位 / 上下限约束。"""
        from setup_tuner.domain.setup import get_field
        from setup_tuner.engine.holistic import _derive_mechanical_from_matrix

        derived = _derive_mechanical_from_matrix(
            {"exit_traction_req": 3.0, "front_grip_req": 0.5},
        )
        assert derived, "有明确机械需求时应推导出刚度改动"
        for name, value in derived.items():
            spec = get_field(name)
            assert abs(value) <= spec.max_delta + 1e-9, f"{name} 超出 max_delta"
            steps = round((value - spec.min_val) / spec.step)
            assert abs(value - (spec.min_val + steps * spec.step)) < 1e-6, (
                f"{name} 未对齐档位"
            )
        # 无机械需求 → 空
        assert _derive_mechanical_from_matrix({}) == {}
        assert _derive_mechanical_from_matrix({"front_grip_req": 0.0}) == {}

    def test_participation_is_budget_neutral(self, monkeypatch) -> None:
        """机械抓地参与度收口是**置换而非追加**：开关该规则，总改动幅度不变。

        做法：分别在「参与度规则生效（门槛 0.10）」与「规则关闭（门槛设 99）」
        两种配置下跑同一场景，断言两者总幅度一致 —— 若实现改成"追加机械项"，
        开启规则那一侧的总幅度会变大，本用例即变红。
        """
        import setup_tuner.engine.holistic as holistic_mod
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        setup = CarSetup.default().to_dict()

        def total(track: str, symptom: str) -> tuple[float, bool]:
            res = generate_suggestion(
                [(symptom, 3)], setup, track, None, model_type="rule",
            )
            delta = res["setup_delta"]
            has_mech = any(abs(delta.get(p, 0.0)) > 1e-9 for p in self._MECHANICAL)
            return sum(abs(v) for v in delta.values()), has_mech

        for track in ("suzuka", "spa", "monaco"):
            for symptom in ("understeer", "exit_wheelspin", "high_speed_instability"):
                with_rules, mech_on = total(track, symptom)
                monkeypatch.setattr(holistic_mod, "_TRACTION_PARTICIPATION_MIN", 99.0)
                without_rules, _mech_off = total(track, symptom)
                monkeypatch.undo()
                assert mech_on, f"{track}/{symptom} 开启规则后仍无机械项"
                # 置换而非追加：开启参与度规则**不得**让总改动幅度变大
                # （实现会从非机械项回收等量幅度；档位向下取整可能略微少一点，
                #   方向永远是"不增加"，与"湿地总幅度不增加"同一约定）
                assert with_rules <= without_rules + 1e-6, (
                    f"{track}/{symptom} 参与度规则放大了总改动幅度："
                    f"{with_rules} > {without_rules}（应为置换而非追加）"
                )
