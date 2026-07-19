"""Tests for the model validation + verification suite.

Covers :class:`SurrogateValidator`, :class:`PhysicsValidator` and
:class:`SetupSanityChecker` from :mod:`f1opt.model.validation`.
"""

from __future__ import annotations

import re

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.model.validation import (
    PhysicsValidator,
    SetupSanityChecker,
    SurrogateValidator,
)

# Matches at least one CJK Unified Ideograph (used to assert Chinese output).
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# --------------------------------------------------------------------------- #
# SurrogateValidator
# --------------------------------------------------------------------------- #
class TestSurrogateValidator:
    def test_validate_monotonicity_returns_dict_with_required_keys(self) -> None:
        sv = SurrogateValidator()
        out = sv.validate_monotonicity("front_wing")
        assert isinstance(out, dict)
        for key in ("field", "track_id", "monotonic", "direction", "samples"):
            assert key in out

    def test_validate_monotonicity_samples_has_five_entries(self) -> None:
        sv = SurrogateValidator()
        out = sv.validate_monotonicity("front_wing")
        assert isinstance(out["samples"], list)
        assert len(out["samples"]) == 5
        for entry in out["samples"]:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_validate_range_all_in_range_or_reports_count(self) -> None:
        sv = SurrogateValidator()
        out = sv.validate_range("melbourne")
        assert isinstance(out, dict)
        assert "all_in_range" in out and "out_of_range_count" in out
        # Either everything is in range, or the out-of-range count is reported.
        assert out["all_in_range"] is True or out["out_of_range_count"] > 0

    def test_validate_sector_consistency(self) -> None:
        sv = SurrogateValidator()
        out = sv.validate_sector_consistency("melbourne")
        assert isinstance(out, dict)
        assert "consistent" in out and "max_error_s" in out
        # Sectors sum to lap_time within tolerance, OR the inconsistency is flagged.
        assert out["consistent"] is True or out["max_error_s"] >= 0.01

    def test_validate_response_ranges_has_tyre_temp_and_slip_angle(self) -> None:
        sv = SurrogateValidator()
        out = sv.validate_response_ranges()
        assert isinstance(out, dict)
        responses = out["responses"]
        assert isinstance(responses, dict)
        assert "tyre_temp" in responses
        assert "slip_angle" in responses

    def test_cross_track_consistency_returns_dict(self) -> None:
        sv = SurrogateValidator()
        out = sv.cross_track_consistency()
        assert isinstance(out, dict)
        assert "consistent" in out and "samples" in out

    def test_full_report_returns_dict_with_passed_and_summary(self) -> None:
        sv = SurrogateValidator()
        out = sv.full_report()
        assert isinstance(out, dict)
        assert "passed" in out and "summary" in out
        assert isinstance(out["passed"], bool)
        assert isinstance(out["summary"], str) and len(out["summary"]) > 0

    def test_validate_monotonicity_for_front_wing_specifically(self) -> None:
        sv = SurrogateValidator()
        out = sv.validate_monotonicity("front_wing")
        assert out["field"] == "front_wing"
        assert out["track_id"] == "melbourne"
        assert isinstance(out["monotonic"], bool)
        assert out["direction"] in ("increasing", "decreasing", "mixed")

    def test_cross_track_monaco_vs_monza_delta_nonzero(self) -> None:
        sv = SurrogateValidator()
        out = sv.cross_track_consistency()
        assert "monaco_vs_monza_delta" in out
        assert out["monaco_vs_monza_delta"] != 0.0

    def test_determinism_same_inputs_same_results(self) -> None:
        sv = SurrogateValidator()
        a = sv.validate_range("melbourne", seed=7)
        b = sv.validate_range("melbourne", seed=7)
        assert a == b
        c = sv.validate_monotonicity("front_wing")
        d = sv.validate_monotonicity("front_wing")
        assert c == d


# --------------------------------------------------------------------------- #
# PhysicsValidator
# --------------------------------------------------------------------------- #
class TestPhysicsValidator:
    def test_validate_aero_returns_dict(self) -> None:
        pv = PhysicsValidator()
        out = pv.validate_aero()
        assert isinstance(out, dict)
        assert "downforce_quadratic" in out and "drag_quadratic" in out
        assert isinstance(out["downforce_quadratic"], bool)

    def test_validate_tire_thermal_peak_at_90(self) -> None:
        pv = PhysicsValidator()
        out = pv.validate_tire_thermal()
        assert out["peak_at_90"] is True

    def test_validate_tire_dynamics_zero_at_zero_slip(self) -> None:
        pv = PhysicsValidator()
        out = pv.validate_tire_dynamics()
        assert out["zero_at_zero_slip"] is True

    def test_validate_suspension_corner_weights_sum_correct(self) -> None:
        pv = PhysicsValidator()
        out = pv.validate_suspension()
        assert out["corner_weights_sum_correct"] is True

    def test_full_report_has_summary(self) -> None:
        pv = PhysicsValidator()
        out = pv.full_report()
        assert isinstance(out, dict)
        assert "summary" in out and "passed" in out
        assert isinstance(out["passed"], bool)
        assert isinstance(out["summary"], str) and len(out["summary"]) > 0

    def test_validate_energy_consistency_balanced(self) -> None:
        pv = PhysicsValidator()
        out = pv.validate_energy_consistency()
        assert isinstance(out, dict)
        assert "balanced" in out
        assert out["balanced"] is True


# --------------------------------------------------------------------------- #
# SetupSanityChecker
# --------------------------------------------------------------------------- #
class TestSetupSanityChecker:
    def test_check_range_compliance_returns_list(self) -> None:
        ss = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        out = ss.check_range_compliance()
        assert isinstance(out, list)
        assert all(isinstance(w, str) for w in out)

    def test_check_track_appropriateness_returns_list(self) -> None:
        ss = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        out = ss.check_track_appropriateness()
        assert isinstance(out, list)
        assert all(isinstance(w, str) for w in out)

    def test_check_internal_consistency_returns_list(self) -> None:
        ss = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        out = ss.check_internal_consistency()
        assert isinstance(out, list)
        assert all(isinstance(w, str) for w in out)

    def test_overall_warnings_returns_list(self) -> None:
        ss = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        out = ss.overall_warnings()
        assert isinstance(out, list)
        assert all(isinstance(w, str) for w in out)

    def test_is_sane_returns_bool(self) -> None:
        ss = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        out = ss.is_sane()
        assert isinstance(out, bool)

    def test_recommendation_returns_nonempty_chinese_string(self) -> None:
        ss = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        out = ss.recommendation()
        assert isinstance(out, str)
        assert len(out) > 0
        assert _CJK_RE.search(out) is not None

    def test_extreme_setup_all_max_warnings_nonempty(self) -> None:
        extreme = CarSetup.from_vector([1.0] * len(SETUP_FIELDS))
        ss = SetupSanityChecker(extreme, "monza")
        warnings = ss.overall_warnings()
        assert len(warnings) > 0
        # An all-max setup at a low-downforce track is not sane.
        assert ss.is_sane() is False

    def test_default_setup_few_or_no_warnings(self) -> None:
        ss = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        overall = ss.overall_warnings()
        # DEFAULT_SETUP should only produce a handful of (range-compliance) warnings
        # and remain internally sane for melbourne.
        assert len(overall) < 10
        assert ss.is_sane() is True

    def test_recommendation_extreme_setup_nonempty_chinese(self) -> None:
        extreme = CarSetup.from_vector([1.0] * len(SETUP_FIELDS))
        ss = SetupSanityChecker(extreme, "monaco")
        out = ss.recommendation()
        assert isinstance(out, str) and len(out) > 0
        assert _CJK_RE.search(out) is not None


# --------------------------------------------------------------------------- #
# Cross-cutting determinism (validator-level)
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_physics_validator_deterministic(self) -> None:
        pv1 = PhysicsValidator()
        pv2 = PhysicsValidator()
        assert pv1.validate_aero() == pv2.validate_aero()
        assert pv1.validate_suspension() == pv2.validate_suspension()

    def test_setup_sanity_checker_deterministic(self) -> None:
        ss1 = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        ss2 = SetupSanityChecker(DEFAULT_SETUP, "melbourne")
        assert ss1.overall_warnings() == ss2.overall_warnings()
        assert ss1.recommendation() == ss2.recommendation()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
