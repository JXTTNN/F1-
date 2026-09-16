"""反馈聚合回归测试 —— 防止 Dx 随反馈条数线性放大。

背景（2026-09 审计）：``/suggest`` 早期把每条反馈都作为独立症状交给
``compute_dx``，而 ``compute_dx`` 对同症状是代数求和。云端实测同一条反馈
重复 20 次会把 ``|Dx|max`` 从 2.4 推到 48，21 个参数里 16 个被 ``max_delta``
顶满 —— 建议不再随输入变化，且反馈越多越极端。

``report/builder.aggregate_feedback_symptoms`` 按 ``(弯道, 症状)`` 取最强强度，
既保留"多个弯都推头"的信息，又消除重复点击造成的虚假放大。
"""

from __future__ import annotations

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.engine.diagnostic import compute_dx
from setup_tuner.engine.engine import generate_suggestion
from setup_tuner.report.builder import (
    aggregate_feedback_symptoms,
    feedbacks_to_symptoms,
)


def _fb(corner: int | None, symptom: str, strength: int) -> dict:
    return {
        "corner_number": corner,
        "symptom": symptom,
        "strength": strength,
        "track_id": "suzuka",
    }


# ===========================================================================
# 1. 聚合语义
# ===========================================================================
class TestAggregation:
    """按 (弯道, 症状) 聚合，取最强强度。"""

    def test_duplicates_collapse_to_max_strength(self) -> None:
        """同一弯同一症状重复提交 → 只保留一条，强度取最大值。"""
        feedbacks = [_fb(1, "understeer", 2), _fb(1, "understeer", 5), _fb(1, "understeer", 3)]
        assert aggregate_feedback_symptoms(feedbacks) == [("understeer", 5)]

    def test_different_corners_kept_separately(self) -> None:
        """不同弯道的同一症状各保留一条（保留"多个弯都推头"的信息）。"""
        feedbacks = [_fb(1, "understeer", 3), _fb(7, "understeer", 2)]
        assert aggregate_feedback_symptoms(feedbacks) == [("understeer", 3), ("understeer", 2)]

    def test_global_and_corner_are_distinct_keys(self) -> None:
        """全局症状（corner=None）与具体弯道是不同键。"""
        feedbacks = [_fb(None, "tyre_wear", 2), _fb(3, "tyre_wear", 4)]
        assert aggregate_feedback_symptoms(feedbacks) == [("tyre_wear", 2), ("tyre_wear", 4)]

    def test_skips_invalid_rows(self) -> None:
        """缺失 symptom / strength 的行被跳过。"""
        feedbacks = [{"corner_number": 1}, {"corner_number": 2, "symptom": "understeer"}]
        assert aggregate_feedback_symptoms(feedbacks) == []

    def test_preserves_first_appearance_order(self) -> None:
        """保持首次出现顺序（保证 Dx 计算可复现）。"""
        feedbacks = [_fb(2, "oversteer", 3), _fb(1, "understeer", 3)]
        assert aggregate_feedback_symptoms(feedbacks) == [("oversteer", 3), ("understeer", 3)]

    def test_raw_projection_unchanged_for_backcompat(self) -> None:
        """``feedbacks_to_symptoms`` 保持原语义（逐条投影，不聚合）。"""
        feedbacks = [_fb(1, "understeer", 2), _fb(1, "understeer", 5)]
        assert feedbacks_to_symptoms(feedbacks) == [("understeer", 2), ("understeer", 5)]


# ===========================================================================
# 2. 对 Dx / 建议的影响
# ===========================================================================
class TestDxNoLongerExplodes:
    """聚合后 Dx 不再随重复条数放大。"""

    def test_dx_stable_under_duplicates(self) -> None:
        """同一反馈 ×20 与 ×1 的 Dx 完全一致。"""
        one = compute_dx(aggregate_feedback_symptoms([_fb(1, "understeer", 3)]))
        many = compute_dx(
            aggregate_feedback_symptoms([_fb(1, "understeer", 3)] * 20)
        )
        assert one == many

    def test_old_behaviour_would_have_exploded(self) -> None:
        """对照：不做聚合时 Dx 会线性放大（记录该问题已修复）。"""
        raw_one = compute_dx(feedbacks_to_symptoms([_fb(1, "understeer", 3)]))
        raw_many = compute_dx(feedbacks_to_symptoms([_fb(1, "understeer", 3)] * 20))
        assert raw_many["front_grip_req"] == raw_one["front_grip_req"] * 20

    def test_suggestion_not_saturated_by_duplicates(self) -> None:
        """重复反馈不再把参数全部顶到 max_delta。"""
        setup = CarSetup.default().to_dict()
        spec = {f.name: f for f in ALL_SETUP_FIELDS}
        dup20 = aggregate_feedback_symptoms([_fb(1, "understeer", 3)] * 20)
        result = generate_suggestion(dup20, setup, "suzuka", None, model_type="rule")
        saturated = sum(
            1 for name, delta in result["setup_delta"].items()
            if abs(abs(delta) - spec[name].max_delta) < 1e-9
        )
        assert saturated <= 6, f"重复反馈仍导致 {saturated}/21 参数饱和"
