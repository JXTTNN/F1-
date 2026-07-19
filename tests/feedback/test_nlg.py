"""Tests for :mod:`f1opt.feedback.nlg` (natural-language generation).

Covers the four public classes:

- :class:`FeedbackNarrator` — narrate_dimension / narrate_all / narrate_setup_change
  / summarize_session, including priority ordering and the empty-list edge.
- :class:`ToneAdapter` — adapt / prefix per driver archetype.
- :class:`ExplanationGenerator` — explain_why / explain_how_to_fix /
  rank_explanations, including the empty-metrics edge.
- :class:`ConversationFlow` — opening / acknowledge / transition / closing /
  clarify.

All tests are pure unit tests: no LLM calls, no telemetry, no torch.
"""

from __future__ import annotations

from f1opt.data.tracks import get_track
from f1opt.feedback.nlg import (
    ConversationFlow,
    ExplanationGenerator,
    FeedbackNarrator,
    ToneAdapter,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _all_ten_dims() -> list[dict]:
    """Return all 10 dimension names with realistic-ish values for narrate_all."""
    return [
        {"name": "setup_advice", "value": "模型推荐 2 项改动", "evidence": "", "advice": None},
        {"name": "sector_compare", "value": "S1=30s S2=32s S3=33s", "evidence": "", "advice": None},
        {"name": "lap_time_potential", "value": "~0.5s above reference", "evidence": "", "advice": None},
        {"name": "confidence", "value": "confidence 0.55", "evidence": "", "advice": None},
        {"name": "throttle_brake_smoothness", "value": "throttle smoothness 0.40", "evidence": "", "advice": None},
        {"name": "ers_drs", "value": "DRS activations: 3", "evidence": "", "advice": None},
        {"name": "braking", "value": "lockup proxy 0.50", "evidence": "", "advice": None},
        {"name": "tyres", "value": "wear FL=5%; temp spread 8C", "evidence": "", "advice": None},
        {"name": "grip", "value": "max g_lat 3.50G", "evidence": "", "advice": None},
        {"name": "balance", "value": "understeer (推头)", "evidence": "", "advice": "增加前翼"},
    ]


# --------------------------------------------------------------------------- #
# FeedbackNarrator.narrate_dimension
# --------------------------------------------------------------------------- #
def test_narrate_dimension_balance_returns_non_empty() -> None:
    narrator = FeedbackNarrator()
    dim = {"name": "balance", "value": "understeer (entry)", "evidence": "", "advice": None}
    out = narrator.narrate_dimension(dim)
    assert isinstance(out, str)
    assert out
    # Balance understeer narration should reference the push/understeer theme.
    assert "推头" in out or "前轮" in out


def test_narrate_dimension_tyres_mentions_temperature() -> None:
    narrator = FeedbackNarrator()
    dim = {
        "name": "tyres",
        "value": "overheating (FL 105°C)",
        "evidence": "",
        "advice": None,
    }
    out = narrator.narrate_dimension(dim)
    assert out
    assert "温度" in out


def test_narrate_dimension_lap_time_potential_mentions_gap() -> None:
    narrator = FeedbackNarrator()
    dim = {
        "name": "lap_time_potential",
        "value": "~0.5s above reference",
        "evidence": "",
        "advice": None,
    }
    out = narrator.narrate_dimension(dim)
    assert out
    assert "0.5" in out
    assert "参考" in out


def test_narrate_dimension_data_insufficient_handled() -> None:
    narrator = FeedbackNarrator()
    dim = {"name": "balance", "value": "数据不足", "evidence": "", "advice": None}
    out = narrator.narrate_dimension(dim)
    assert out
    assert "数据不足" in out


# --------------------------------------------------------------------------- #
# FeedbackNarrator.narrate_all
# --------------------------------------------------------------------------- #
def test_narrate_all_coherent_paragraph_long_for_ten_dims() -> None:
    narrator = FeedbackNarrator()
    out = narrator.narrate_all(_all_ten_dims())
    assert isinstance(out, str)
    assert len(out) > 100


def test_narrate_all_orders_by_priority_balance_first() -> None:
    """balance/grip/tyres (tier 0) must be narrated before setup_advice (tier 3)."""
    narrator = FeedbackNarrator()
    dims = [
        {"name": "setup_advice", "value": "模型推荐 2 项改动", "evidence": "", "advice": None},
        {"name": "balance", "value": "understeer (推头)", "evidence": "", "advice": None},
        {"name": "tyres", "value": "temp spread 8C", "evidence": "", "advice": None},
    ]
    out = narrator.narrate_all(dims)
    assert out
    # balance narration (推头) appears before setup_advice narration (调教).
    assert 0 <= out.find("推头") < out.find("调教")


def test_narrate_all_uses_transitions() -> None:
    narrator = FeedbackNarrator()
    out = narrator.narrate_all(_all_ten_dims())
    assert "首先" in out
    assert "最后" in out


def test_narrate_all_empty_dimensions_returns_empty_string() -> None:
    narrator = FeedbackNarrator()
    assert narrator.narrate_all([]) == ""


# --------------------------------------------------------------------------- #
# FeedbackNarrator.narrate_setup_change
# --------------------------------------------------------------------------- #
def test_narrate_setup_change_mentions_before_after_and_gain() -> None:
    narrator = FeedbackNarrator()
    suggestion = {
        "pname": "front_wing",
        "before": 25,
        "after": 28,
        "expected_gain": "~0.3s/lap",
        "reason": "增加前轴下压力",
    }
    out = narrator.narrate_setup_change(suggestion)
    assert out
    assert "25" in out
    assert "28" in out
    assert "0.3" in out  # the gain
    assert "前翼" in out  # Chinese param name


def test_narrate_setup_change_includes_reason() -> None:
    narrator = FeedbackNarrator()
    suggestion = {
        "pname": "rear_arb",
        "before": 10,
        "after": 8,
        "expected_gain": "~0.1s",
        "reason": "软化后轴",
    }
    out = narrator.narrate_setup_change(suggestion)
    assert "原因" in out
    assert "软化后轴" in out


# --------------------------------------------------------------------------- #
# FeedbackNarrator.summarize_session
# --------------------------------------------------------------------------- #
def test_summarize_session_returns_two_to_three_sentences() -> None:
    narrator = FeedbackNarrator()
    feedback = {
        "dimensions": [
            {"name": "lap_time_potential", "value": "~0.5s above reference", "evidence": "", "advice": None},
            {"name": "balance", "value": "understeer (推头)", "evidence": "", "advice": None},
            {"name": "tyres", "value": "temp spread 8C", "evidence": "", "advice": None},
        ],
        "setup_suggestions": [
            {"pname": "front_wing", "before": 25, "after": 28, "expected_gain": "~0.3s", "reason": "x"},
        ],
    }
    out = narrator.summarize_session(feedback)
    assert out
    sentence_count = out.count("。")
    assert 2 <= sentence_count <= 3, f"expected 2-3 sentences, got {sentence_count}: {out!r}"


def test_summarize_session_empty_feedback_still_returns_string() -> None:
    narrator = FeedbackNarrator()
    out = narrator.summarize_session({})
    assert out
    assert out.count("。") >= 2


# --------------------------------------------------------------------------- #
# ToneAdapter
# --------------------------------------------------------------------------- #
def test_tone_adapter_aggressive_vs_development_differ() -> None:
    text = "前轴抓地不足，建议增加前翼。"
    aggressive = ToneAdapter("AGGRESSIVE_OVERTAKER").adapt(text)
    development = ToneAdapter("DEVELOPMENT").adapt(text)
    assert aggressive != development
    assert aggressive
    assert development
    assert "进攻型" in aggressive
    assert "新晋" in development


def test_tone_adapter_prefix_non_empty() -> None:
    for archetype in ("AGGRESSIVE_OVERTAKER", "DEVELOPMENT", "RACE_CRAFT", "TIRE_WHISPERER"):
        assert ToneAdapter(archetype).prefix(), f"empty prefix for {archetype}"


def test_tone_adapter_tire_whisperer_emphasizes_tyres() -> None:
    text = "圈速还有提升空间。"
    out = ToneAdapter("TIRE_WHISPERER").adapt(text)
    assert "胎" in out


# --------------------------------------------------------------------------- #
# ExplanationGenerator.explain_why
# --------------------------------------------------------------------------- #
def test_explain_why_returns_multi_sentence_string() -> None:
    eg = ExplanationGenerator()
    out = eg.explain_why("推头", {"front_wing": 22, "front_tyre_temp": 82})
    assert isinstance(out, str)
    assert out
    assert out.count("。") >= 2


def test_explain_why_understeer_mentions_front_wing_topic() -> None:
    eg = ExplanationGenerator()
    out = eg.explain_why("推头", {"front_wing": 22})
    assert out
    assert "前翼" in out or "下压力" in out or "前轮" in out


def test_explain_why_empty_metrics_still_returns_string() -> None:
    eg = ExplanationGenerator()
    out = eg.explain_why("推头", {})
    assert isinstance(out, str)
    assert out
    assert out.count("。") >= 2


def test_explain_why_oversteer_handled() -> None:
    eg = ExplanationGenerator()
    out = eg.explain_why("甩尾", {"rear_wing": 20})
    assert out
    assert "甩尾" in out or "后轴" in out


# --------------------------------------------------------------------------- #
# ExplanationGenerator.explain_how_to_fix
# --------------------------------------------------------------------------- #
def test_explain_how_to_fix_returns_list_of_strings() -> None:
    eg = ExplanationGenerator()
    out = eg.explain_how_to_fix("推头", {"front_wing": 24})
    assert isinstance(out, list)
    assert len(out) >= 1
    for item in out:
        assert isinstance(item, str)
        assert item


def test_explain_how_to_fix_understeer_includes_front_wing_action() -> None:
    eg = ExplanationGenerator()
    out = eg.explain_how_to_fix("推头", {"front_wing": 24})
    assert any("前翼" in item for item in out), f"no front_wing action in {out!r}"


def test_explain_how_to_fix_lockup_includes_brake_action() -> None:
    eg = ExplanationGenerator()
    out = eg.explain_how_to_fix("锁死", {"front_brake_bias": 53})
    assert any("制动" in item or "刹车" in item for item in out)


# --------------------------------------------------------------------------- #
# ExplanationGenerator.rank_explanations
# --------------------------------------------------------------------------- #
def test_rank_explanations_sorts_by_relevance_descending() -> None:
    eg = ExplanationGenerator()
    explanations = ["原因A", "原因B", "原因C"]
    relevance = [0.3, 0.9, 0.6]
    ranked = eg.rank_explanations(explanations, relevance)
    assert ranked == ["原因B", "原因C", "原因A"]


def test_rank_explanations_empty_inputs_returns_empty() -> None:
    eg = ExplanationGenerator()
    assert eg.rank_explanations([], []) == []


# --------------------------------------------------------------------------- #
# ConversationFlow
# --------------------------------------------------------------------------- #
def test_conversation_flow_opening_mentions_track() -> None:
    cf = ConversationFlow()
    out = cf.opening("melbourne")
    assert out
    circuit = get_track("melbourne").circuit_name
    assert circuit in out or "melbourne" in out.lower()


def test_conversation_flow_opening_uses_driver_name() -> None:
    cf = ConversationFlow()
    out = cf.opening("melbourne", driver_name="Lewis")
    assert "Lewis" in out


def test_conversation_flow_acknowledge_non_empty() -> None:
    cf = ConversationFlow()
    out = cf.acknowledge("为什么推头")
    assert out
    assert "推头" in out


def test_conversation_flow_transition_mentions_both_topics() -> None:
    cf = ConversationFlow()
    out = cf.transition("平衡", "制动")
    assert "平衡" in out
    assert "制动" in out


def test_conversation_flow_closing_non_empty() -> None:
    cf = ConversationFlow()
    out = cf.closing()
    assert out
    assert out.count("。") >= 1


def test_conversation_flow_clarify_asks_question() -> None:
    cf = ConversationFlow()
    out = cf.clarify("那个")
    assert out
    assert "?" in out or "？" in out


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_same_inputs_same_output() -> None:
    narrator = FeedbackNarrator()
    dim = {"name": "balance", "value": "understeer (推头)", "evidence": "x", "advice": "增加前翼"}
    assert narrator.narrate_dimension(dim) == narrator.narrate_dimension(dim)
    assert narrator.narrate_all(_all_ten_dims()) == narrator.narrate_all(_all_ten_dims())

    eg = ExplanationGenerator()
    assert eg.explain_why("推头", {"front_wing": 22}) == eg.explain_why("推头", {"front_wing": 22})
    assert eg.explain_how_to_fix("推头", {"front_wing": 24}) == eg.explain_how_to_fix("推头", {"front_wing": 24})
    assert eg.rank_explanations(["a", "b"], [0.1, 0.2]) == eg.rank_explanations(["a", "b"], [0.1, 0.2])

    cf = ConversationFlow()
    assert cf.opening("melbourne") == cf.opening("melbourne")
