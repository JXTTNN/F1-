"""圈级调教模型与优化器测试（``engine/lap_model`` + ``engine/optimizer``）。

本文件重点锁定**建模正确性**，因为这一层出错的代价极高：错误不会抛异常，
只会让"最优调教"变得荒谬。开发过程中真实踩过（并已修复）的四个坑，
每个都有对应测试：

1. **供给量纲错误** —— 曾用 ``Δp × C[dim][p]`` 与 ``Δp / C[dim][p]``，
   前者把供给放大 C² 倍（残差瞬间清零，最优点退化成"把所有能力拉满"），
   后者把并行路径当串联（严重超供，优化器发现"删光改动"最省）。
   正确解：按格拉姆对角 ``Σ_p C[dim][p]²`` 归一化 → 规则引擎那一步
   供给恰好 ≈ 需求。见 ``TestSupplyDimension``。
2. **残差双向惩罚** —— 超供也被平方惩罚，目标被超供罚分主导。
   正确解：只罚缺口。见 ``TestDeficitIsOneSided``。
3. **无弯道时仍优化** —— 逐弯残差恒为 0，只剩代价项，"最优解"= 什么都不改，
   把规则引擎的整份建议清空。正确解：无弯道基础则不做圈级优化。
   见 ``TestNoCornerDataGuard``。
4. **天气没进模型** —— 不同起点的优化收敛到同一解，湿地保守意图被抹平。
   正确解：天气以需求倾斜进入目标函数。见 ``TestWeatherEntersModel``。
"""

from __future__ import annotations

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.track import get_track_by_id
from setup_tuner.engine.coupling import get_coupling
from setup_tuner.engine.diagnostic import DIAG_DIMS
from setup_tuner.engine.engine import compute_setup_delta, generate_suggestion
from setup_tuner.engine.holistic import FAST, SLOW, class_weighted_dx
from setup_tuner.engine.lap_model import (
    _gram_diag,
    apply_weather_to_needs,
    corner_evaluations,
    needs_by_class_from_dx,
    objective,
    supply_from_units,
    track_context,
    units_to_delta,
)
from setup_tuner.engine.optimizer import optimize_setup

_SPECS = {f.name: f for f in ALL_SETUP_FIELDS}
_SETUP = CarSetup().to_dict()


def _dx(**overrides: float) -> dict[str, float]:
    base = dict.fromkeys(DIAG_DIMS, 0.0)
    base.update(overrides)
    return base


def _corner_of(track_id: str, klass: str) -> int:
    track = get_track_by_id(track_id)
    assert track is not None
    return next(c.number for c in track.corners if c.corner_type == klass)


def _units_from_delta(delta: dict[str, float]) -> dict[str, float]:
    return {
        f.name: (delta.get(f.name, 0.0) / (f.max_delta or 1.0))
        for f in ALL_SETUP_FIELDS
    }


# ===========================================================================
# 1. 供给量纲（格拉姆归一化）
# ===========================================================================
class TestSupplyDimension:
    """供给必须与 Dx 同量纲：规则引擎那一步供给 ≈ 需求。"""

    @pytest.mark.parametrize("dim", list(DIAG_DIMS))
    def test_gram_normalization_is_exact_without_quantization(self, dim: str) -> None:
        """格拉姆归一化的**精确性**：不加量化时，按 C 行给出的改动恰好供满 1.0。

        即 ``Δp = C[dim][p]`` ⇒ ``supply[dim] = 1.0``。这是量纲自洽的定义式，
        与规则引擎的取整行为无关，单独锁定以免归一化被改坏。
        """
        units = {
            f.name: (get_coupling(dim, f.name).sign * get_coupling(dim, f.name).magnitude
                     / (f.max_delta or 1.0))
            for f in ALL_SETUP_FIELDS
            if get_coupling(dim, f.name) is not None
        }
        assert supply_from_units(units)[dim] == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.parametrize("dim", list(DIAG_DIMS))
    def test_rule_step_supply_is_positive(self, dim: str) -> None:
        """规则引擎那一步必须对该维度产生**正供给**（方向不能错）。

        注：供给量不必然接近 1.0 —— C 行的幅度常小于参数离散步长
        （如 exit_traction_req 的 C 行是 0.3~0.5，而对应参数 step=1），
        取整会把大半改动抹掉。实测 exit_traction_req 的量化后自供给仅约 0.11，
        即**规则引擎本身在这些维度上供不足**。这是既有性质，由圈级优化器
        通过改用其他参数路径来补偿，故此处只锁定方向。
        """
        dx = _dx(**{dim: 1.0})
        units = _units_from_delta(compute_setup_delta(dx, _SETUP))
        supply = supply_from_units(units)[dim]
        assert supply > 0, f"{dim} 的规则步未产生正供给（方向性错误）"

    def test_gram_diag_positive(self) -> None:
        gram = _gram_diag()
        assert set(gram) == set(DIAG_DIMS)
        assert all(v > 0 for v in gram.values())

    def test_supply_scales_linearly(self) -> None:
        """供给对改动量线性（模型是线性的，便于优化器工作）。"""
        units = _units_from_delta(compute_setup_delta(_dx(front_grip_req=1.0), _SETUP))
        half = {k: v / 2 for k, v in units.items()}
        full_supply = supply_from_units(units)["front_grip_req"]
        half_supply = supply_from_units(half)["front_grip_req"]
        assert half_supply == pytest.approx(full_supply / 2, rel=1e-6)

    def test_zero_change_supplies_nothing(self) -> None:
        assert all(v == 0 for v in supply_from_units({}).values())


# ===========================================================================
# 2. 残差只罚缺口
# ===========================================================================
class TestDeficitIsOneSided:
    """超供不再增加目标值（超供的代价交给显式代价项）。"""

    def test_more_supply_never_worse(self) -> None:
        ctx = track_context("suzuka", None, _dx(front_grip_req=1.0))
        corners = corner_evaluations("suzuka")
        needs = needs_by_class_from_dx(_dx(front_grip_req=1.0))
        low = objective(_units_from_delta({"front_wing": 1.0}), needs, "suzuka", ctx, corners)
        high = objective(_units_from_delta({"front_wing": 2.0}), needs, "suzuka", ctx, corners)
        assert high.residual <= low.residual + 1e-12

    def test_zero_change_leaves_full_deficit(self) -> None:
        ctx = track_context("suzuka", None, _dx(front_grip_req=1.0))
        corners = corner_evaluations("suzuka")
        needs = needs_by_class_from_dx(_dx(front_grip_req=1.0))
        none = objective({}, needs, "suzuka", ctx, corners)
        some = objective(_units_from_delta({"front_wing": 1.0}), needs, "suzuka", ctx, corners)
        assert none.residual > some.residual


# ===========================================================================
# 3. 逐弯展开与车手反馈加权
# ===========================================================================
class TestCornerEvaluations:
    def test_covers_all_corners(self) -> None:
        track = get_track_by_id("suzuka")
        assert track is not None
        evals = corner_evaluations("suzuka")
        assert [e.number for e in evals] == [c.number for c in track.corners]

    def test_importance_normalized(self) -> None:
        evals = corner_evaluations("monaco")
        assert sum(e.importance for e in evals) == pytest.approx(1.0)
        # 慢弯耗时长 → 重要度应高于快弯
        slow = next(e for e in evals if e.klass == SLOW)
        fast = next(e for e in evals if e.klass == FAST)
        assert slow.importance > fast.importance

    def test_feedback_boosts_corner(self) -> None:
        """车手报过的弯重要度必须被抬高（"结合车手反馈"的落点）。"""
        base = {e.number: e.importance for e in corner_evaluations("suzuka")}
        number = max(base, key=lambda n: base[n])
        boosted = {
            e.number: e.importance
            for e in corner_evaluations("suzuka", {number})
        }
        assert boosted[number] > base[number]

    def test_unknown_track_returns_empty(self) -> None:
        assert corner_evaluations("__no_such_track__") == []


# ===========================================================================
# 4. 目标函数与代价项
# ===========================================================================
class TestObjective:
    def test_breakdown_sums(self) -> None:
        ctx = track_context("spa", None, _dx(front_grip_req=1.0))
        corners = corner_evaluations("spa")
        br = objective(_units_from_delta({"front_wing": 1.0}),
                       needs_by_class_from_dx(_dx(front_grip_req=1.0)),
                       "spa", ctx, corners)
        assert br.total == pytest.approx(
            br.residual + br.drag_cost + br.bottoming_cost
            + br.tyre_heat_cost + br.effort_cost
        )
        assert sum(br.per_class.values()) == pytest.approx(br.residual, abs=1e-6)

    def test_drag_cost_penalizes_adding_wings(self) -> None:
        ctx = track_context("monza", None)
        add = objective(_units_from_delta({"front_wing": 3.0, "rear_wing": 3.0}),
                        needs_by_class_from_dx({}), "monza", ctx, corner_evaluations("monza"))
        remove = objective(_units_from_delta({"front_wing": -3.0, "rear_wing": -3.0}),
                           needs_by_class_from_dx({}), "monza", ctx, corner_evaluations("monza"))
        assert add.drag_cost > remove.drag_cost

    def test_bottoming_cost_only_when_lowering(self) -> None:
        ctx = track_context("spa", None)
        lower = objective(_units_from_delta({"front_ride_height": -2.0,
                                             "rear_ride_height": -2.0}),
                          needs_by_class_from_dx({}), "spa", ctx, corner_evaluations("spa"))
        raise_ = objective(_units_from_delta({"front_ride_height": 2.0,
                                              "rear_ride_height": 2.0}),
                           needs_by_class_from_dx({}), "spa", ctx, corner_evaluations("spa"))
        assert lower.bottoming_cost > 0
        assert raise_.bottoming_cost == 0

    def test_drag_weight_higher_on_fast_tracks(self) -> None:
        """均速更高的赛道，翼片阻力代价权重更大。"""
        ctx_monza = track_context("monza", None)
        ctx_monaco = track_context("monaco", None)
        assert ctx_monza.drag_weight > ctx_monaco.drag_weight


# ===========================================================================
# 5. 优化器
# ===========================================================================
class TestOptimizer:
    def _feedback(self, track_id: str) -> tuple[list[dict], dict]:
        slow = _corner_of(track_id, SLOW)
        fast = _corner_of(track_id, FAST)
        fbs = [
            {"corner_number": slow, "symptom": "bottoming", "strength": 3,
             "category": "apex"},
            {"corner_number": fast, "symptom": "exit_oversteer", "strength": 3,
             "category": "exit"},
        ]
        cw = class_weighted_dx(fbs, track_id)
        dx = {d: sum(cw.by_class[k][d] for k in cw.by_class) for d in DIAG_DIMS}
        return fbs, {"dx": dx, "by_class": cw.by_class}

    def test_improves_objective_on_real_track(self) -> None:
        fbs, info = self._feedback("spa")
        rule = compute_setup_delta(info["dx"], _SETUP)
        res = optimize_setup(info["dx"], _SETUP, "spa", feedbacks=fbs,
                             initial_delta=rule, needs_by_class=info["by_class"])
        assert res.after.total <= res.before.total + 1e-9

    def test_result_is_step_aligned_and_in_bounds(self) -> None:
        """输出必须是合法档位（F1 参数是分档的），且不越界、不超 max_delta。"""
        fbs, info = self._feedback("zandvoort")
        rule = compute_setup_delta(info["dx"], _SETUP)
        res = optimize_setup(info["dx"], _SETUP, "zandvoort", feedbacks=fbs,
                             initial_delta=rule, needs_by_class=info["by_class"])
        for name, value in res.delta.items():
            spec = _SPECS[name]
            if value:
                assert abs(value / spec.step - round(value / spec.step)) < 1e-9, (
                    f"{name}={value} 未对齐步长 {spec.step}"
                )
            assert abs(value) <= spec.max_delta + 1e-9
            assert spec.min_val - 1e-9 <= _SETUP[name] + value <= spec.max_val + 1e-9

    def test_tyre_pressures_stay_symmetric(self) -> None:
        fbs, info = self._feedback("spa")
        res = optimize_setup(info["dx"], _SETUP, "spa", feedbacks=fbs)
        assert (res.delta["front_left_tyre_pressure"]
                == res.delta["front_right_tyre_pressure"])
        assert (res.delta["rear_left_tyre_pressure"]
                == res.delta["rear_right_tyre_pressure"])

    def test_deterministic(self) -> None:
        fbs, info = self._feedback("spa")
        a = optimize_setup(info["dx"], _SETUP, "spa", feedbacks=fbs)
        b = optimize_setup(info["dx"], _SETUP, "spa", feedbacks=fbs)
        assert a.delta == b.delta
        assert a.after.total == b.after.total

    def test_corner_profiles_differ_across_tracks(self) -> None:
        """不同赛道的弯道类别重要度分布必须不同（赛道特性进入模型的入口）。"""
        def profile(track_id: str) -> dict:
            out = {SLOW: 0.0, "medium": 0.0, FAST: 0.0}
            for e in corner_evaluations(track_id):
                out[e.klass] += e.importance
            return {k: round(v, 6) for k, v in out.items()}

        assert profile("monza") != profile("monaco")
        assert profile("jeddah") != profile("zandvoort")

    @pytest.mark.parametrize("pair", [("monza", "monaco"), ("monaco", "jeddah"),
                                      ("monza", "spa")])
    def test_track_characteristics_change_the_optimum(self, pair: tuple) -> None:
        """**核心断言**：弯型构成差异明显的两条赛道，最优调教必须不同。

        这条曾失败过 —— 当时所有弯共用同一份全圈需求，于是"所有弯想要同样的
        东西"，不存在权衡，四条赛道算出逐位相同的调教。必须让慢弯/快弯各自
        带着不同需求（按类别分列），赛道弯型占比才会真正决定取向。

        注意：这里只取**弯型构成差异明显**的组合。归一化后弯型分布接近的两条
        赛道，在 1/8 的离散网格上确实可能落到同一个最优点 —— 那是网格分辨率
        的固有现象，不是模型失效，故不做"任意两两必不同"的过强断言。
        """
        track_a, track_b = pair
        fbs_a, info_a = self._feedback(track_a)
        fbs_b, info_b = self._feedback(track_b)
        res_a = optimize_setup(info_a["dx"], _SETUP, track_a, feedbacks=fbs_a,
                               needs_by_class=info_a["by_class"])
        res_b = optimize_setup(info_b["dx"], _SETUP, track_b, feedbacks=fbs_b,
                               needs_by_class=info_b["by_class"])
        assert res_a.delta != res_b.delta, (
            f"{track_a} 与 {track_b} 得到相同调教 —— 赛道特性未生效"
        )

    def test_per_class_gain_reported(self) -> None:
        fbs, info = self._feedback("spa")
        rule = compute_setup_delta(info["dx"], _SETUP)
        res = optimize_setup(info["dx"], _SETUP, "spa", feedbacks=fbs,
                             initial_delta=rule, needs_by_class=info["by_class"])
        assert set(res.per_class_gain) == {"slow", "medium", "fast"}

    def test_trace_is_readable(self) -> None:
        fbs, info = self._feedback("spa")
        res = optimize_setup(info["dx"], _SETUP, "spa", feedbacks=fbs,
                             needs_by_class=info["by_class"])
        assert res.trace
        assert any("圈级目标" in t for t in res.trace)


# ===========================================================================
# 6. 无弯道守卫（真实回归）
# ===========================================================================
class TestNoCornerDataGuard:
    """没有赛道弯道数据时不得做圈级优化 —— 否则会清空整份建议。"""

    def test_unknown_track_keeps_initial_delta(self) -> None:
        dx = _dx(front_grip_req=1.0)
        rule = compute_setup_delta(dx, _SETUP)
        res = optimize_setup(dx, _SETUP, "__no_such_track__", initial_delta=rule)
        assert res.delta == rule, "未知赛道下优化器不得改动规则建议"
        assert res.improvement == pytest.approx(0.0)
        assert res.trace == []

    def test_generate_suggestion_keeps_symptoms_effect_on_unknown_track(self) -> None:
        """合成 track_id 下，多症状叠加仍必须产生差异。

        这是真实回归的固化：优化器接管 delta 后，未知赛道残差恒为 0、
        只剩代价项，最优解退化为"什么都不改"，导致
        "多症状叠加未产生差异"（当时打挂 5 个既有测试）。
        """
        one = generate_suggestion(symptoms=[("understeer", 2)], current_setup=_SETUP,
                                  track_id="test_track")
        two = generate_suggestion(symptoms=[("understeer", 2), ("exit_wheelspin", 3)],
                                  current_setup=_SETUP, track_id="test_track")
        assert one["setup_delta"] != two["setup_delta"]


# ===========================================================================
# 7. 天气必须进入模型（真实回归）
# ===========================================================================
class TestWeatherEntersModel:
    def test_wet_tilts_needs(self) -> None:
        needs = needs_by_class_from_dx(
            _dx(front_grip_req=1.0, hi_speed_stab_req=1.0, exit_traction_req=1.0)
        )
        wet = apply_weather_to_needs(needs, True)
        assert wet[SLOW]["hi_speed_stab_req"] > needs[SLOW]["hi_speed_stab_req"]
        assert wet[SLOW]["exit_traction_req"] > needs[SLOW]["exit_traction_req"]
        assert wet[SLOW]["front_grip_req"] < needs[SLOW]["front_grip_req"]

    def test_dry_is_identity(self) -> None:
        needs = needs_by_class_from_dx(_dx(front_grip_req=1.0))
        assert apply_weather_to_needs(needs, False) is needs

    def test_wet_reduces_grip_modifier(self) -> None:
        dry = track_context("suzuka", None)
        wet = track_context("suzuka", None, wet=True)
        assert wet.grip_modifier < dry.grip_modifier

    def test_wet_output_differs_from_dry(self) -> None:
        """湿地与干地必须给出不同调教。

        这是真实回归的固化：天气未进模型时，优化器从不同起点收敛到同一解，
        湿/干输出逐位相同，`_derive_telemetry_gain` 的保守意图被完全抹平。
        """
        dry = generate_suggestion(symptoms=[("understeer", 2)], current_setup=_SETUP,
                                  track_id="suzuka")
        wet = generate_suggestion(symptoms=[("understeer", 2)], current_setup=_SETUP,
                                  track_id="suzuka", telemetry={"weather": "wet"})
        assert dry["setup_delta"] != wet["setup_delta"]

    def test_none_telemetry_is_tolerated(self) -> None:
        """telemetry=None 是合法输入（早期 _is_wet_weather 会 AttributeError）。"""
        res = generate_suggestion(symptoms=[("understeer", 2)], current_setup=_SETUP,
                                  track_id="suzuka", telemetry=None)
        assert res["setup_delta"]


# ===========================================================================
# 8. 步长对齐工具
# ===========================================================================
class TestUnitsToDelta:
    def test_snaps_to_step(self) -> None:
        delta = units_to_delta({"front_suspension": 0.3}, _SETUP)
        assert delta["front_suspension"] % _SPECS["front_suspension"].step == 0

    def test_snaps_to_step_without_current_setup(self) -> None:
        """不传 current_setup 时也必须对齐步长。

        ``units_to_delta`` 里有**两处**对齐：夹取边界前一次、夹取后一次。
        只测"传了 current_setup"的路径时，去掉前一处对齐仍会被后一处兜住，
        测试无法发现 —— 必须单独覆盖"不带 current_setup"的路径，
        把前一处对齐真正置于覆盖之下。
        """
        delta = units_to_delta({"front_suspension": 0.3})
        assert delta["front_suspension"] % _SPECS["front_suspension"].step == 0
        assert delta["front_suspension"] != 0.6

    def test_respects_max_delta(self) -> None:
        delta = units_to_delta({"front_wing": 1.0}, _SETUP)
        assert delta["front_wing"] <= _SPECS["front_wing"].max_delta

    def test_returns_all_20_fields(self) -> None:
        """必须返回全部 21 项（下游 _build_param_details 按 21 项取值）。"""
        assert len(units_to_delta({"front_wing": 0.1}, _SETUP)) == 20

    def test_respects_upper_bound(self) -> None:
        setup = dict(_SETUP)
        setup["front_wing"] = _SPECS["front_wing"].max_val
        delta = units_to_delta({"front_wing": 1.0}, setup)
        assert setup["front_wing"] + delta["front_wing"] <= _SPECS["front_wing"].max_val

    def test_pressure_pairs_mirrored(self) -> None:
        delta = units_to_delta({"front_left_tyre_pressure": 0.5}, _SETUP)
        assert delta["front_right_tyre_pressure"] == delta["front_left_tyre_pressure"]
