"""Tests for setup safety filter (Opt-Safety)."""

import pytest

from f1opt.feedback.safety import (
    check_parameter_safety,
    check_feedback_consistency,
    validate_suggestions,
    filter_harmful_suggestions,
    SafetyReport,
    PARAM_LIMITS,
)


class TestParameterSafety:
    def test_within_limits_safe(self):
        result = check_parameter_safety("front_wing", 10, 15)
        assert result.is_safe
        assert result.original_after == 15
        assert result.clamped_after == 15

    def test_below_min_clamped(self):
        result = check_parameter_safety("front_wing", 5, -5)
        assert result.clamped_after == 0
        assert result.reason

    def test_above_max_clamped(self):
        result = check_parameter_safety("front_wing", 45, 60)
        assert result.clamped_after == 50
        assert result.reason

    def test_healthy_delta_no_reason(self):
        result = check_parameter_safety("front_wing", 10, 12)
        assert result.is_safe
        assert not result.reason

    def test_large_delta_has_warning(self):
        result = check_parameter_safety("front_wing", 10, 30)
        assert result.clamped_after != 30 or result.reason

    def test_unknown_parameter_allowed(self):
        result = check_parameter_safety("unknown_param", 0, 100)
        assert result.is_safe
        assert "Unknown parameter" in result.reason


class TestFeedbackConsistency:
    def test_no_issues_no_warnings(self):
        suggestions = [{"name": "front_wing", "before": 10, "after": 11}]
        warnings = check_feedback_consistency(suggestions, [])
        assert len(warnings) == 0

    def test_understeer_inconsistent_with_rear_wing_increase(self):
        # 增后翼会恶化转向不足 (后下压过重，前轮相对负荷不足)
        suggestions = [{"name": "rear_wing", "before": 10, "after": 15}]
        warnings = check_feedback_consistency(suggestions, ["understeer_in"])
        assert len(warnings) == 1
        assert "worsen" in warnings[0].lower() or "恶化" in warnings[0]

    def test_understeer_consistent_with_rear_wing_decrease(self):
        # 减后翼是正确的调校方向，不应触发矛盾
        suggestions = [{"name": "rear_wing", "before": 15, "after": 10}]
        warnings = check_feedback_consistency(suggestions, ["understeer_in"])
        assert len(warnings) == 0

    def test_tire_wear_pressure_decrease_worsens(self):
        suggestions = [{"name": "front_tyre_pressure", "before": 22, "after": 20}]
        warnings = check_feedback_consistency(suggestions, ["tire_wear"])
        assert len(warnings) == 1

    def test_tire_wear_pressure_increase_ok(self):
        suggestions = [{"name": "front_tyre_pressure", "before": 20, "after": 22}]
        warnings = check_feedback_consistency(suggestions, ["tire_wear"])
        assert len(warnings) == 0


class TestValidateSuggestions:
    def test_all_safe_suggestions(self):
        suggestions = [
            {"name": "front_wing", "before": 10, "after": 12},
            {"name": "rear_wing", "before": 15, "after": 16},
        ]
        report = validate_suggestions(suggestions)
        assert report.all_safe
        assert len(report.checks) == 2

    def test_clamps_harmful_changes(self):
        suggestions = [{"name": "front_wing", "before": 10, "after": 100}]
        report = validate_suggestions(suggestions)
        # When delta > 2x safe_delta, we clamp to before + safe_delta (protective)
        # safe_delta for front_wing is 6, so 10 + 6 = 16
        assert suggestions[0]["after"] == 16  # before + safe_delta

    def test_rejects_extreme_changes(self):
        suggestions = [{"name": "rear_wing", "before": 10, "after": 50}]
        filtered, report = filter_harmful_suggestions(suggestions)
        assert len(filtered) == 0 or "rear_wing" not in [s["name"] for s in filtered]


class TestSafetyReport:
    def test_all_safe_report(self):
        report = SafetyReport()
        assert report.all_safe
        assert report.summary() == "All adjustments within safe bounds."

    def test_report_with_rejections(self):
        report = SafetyReport()
        report.rejected_params = ["test_param"]
        assert not report.all_safe
        assert "Rejected 1 harmful adjustment" in report.summary()
