"""Tests for :mod:`f1opt.feedback.quality` (Iter-146)."""
from __future__ import annotations

from f1opt.feedback.quality import (
    ResponseQualityReport,
    _check_actionability,
    _check_completeness,
    _check_groundedness,
    _extract_numeric_claims,
    assess_response_quality,
)


class TestExtractNumericClaims:
    def test_english_units(self) -> None:
        claims = _extract_numeric_claims("speed 320.5 kph, temp 95°C, bias 56%")
        assert len(claims) >= 2
        assert any(v == 320.5 and u == "kph" for v, u in claims)

    def test_chinese_units(self) -> None:
        claims = _extract_numeric_claims("提升 0.3秒，温度 15度")
        assert len(claims) >= 2
        assert any(v == 0.3 and u == "秒" for v, u in claims)

    def test_no_claims(self) -> None:
        claims = _extract_numeric_claims("The car feels good today.")
        assert len(claims) == 0


class TestGroundedness:
    def test_all_claims_verified(self) -> None:
        sources = [
            {"frame_t": 1.0, "field": "speed", "value": 320.5},
            {"frame_t": 2.0, "field": "tyre_temp", "value": 95.0},
        ]
        score, issues = _check_groundedness(
            "Speed is 320.5 kph, tyre temp is 95°C", sources
        )
        assert score >= 0.5

    def test_no_claims_no_penalty(self) -> None:
        score, issues = _check_groundedness("Looking good", [])
        assert score == 1.0
        assert len(issues) == 0

    def test_unverified_claims_flagged(self) -> None:
        sources = [{"frame_t": 1.0, "field": "speed", "value": 300.0}]
        score, issues = _check_groundedness(
            "Speed is 999 kph and temp is 120°C", sources
        )
        assert score < 1.0
        assert len(issues) > 0


class TestCompleteness:
    def test_all_dimensions_filled(self) -> None:
        feedback = {
            "dimensions": [
                {"name": "balance", "value": "Neutral"},
                {"name": "grip", "value": "Good grip"},
                {"name": "tyres", "value": "Wear 15%"},
            ]
        }
        score, issues = _check_completeness(feedback)
        assert score == 1.0

    def test_empty_dimensions_detected(self) -> None:
        feedback = {
            "dimensions": [
                {"name": "balance", "value": "N/A"},
                {"name": "grip", "value": ""},
                {"name": "tyres", "value": "Good"},
            ]
        }
        score, issues = _check_completeness(feedback)
        assert score == 1.0 / 3.0

    def test_no_dimensions(self) -> None:
        score, issues = _check_completeness({})
        assert score == 0.0


class TestActionability:
    def test_highly_actionable(self) -> None:
        score, issues = _check_actionability(
            "建议调整前翼角度 +2度，降低后胎压 0.5 bar，预计提升 0.3秒"
        )
        assert score >= 0.7

    def test_not_actionable(self) -> None:
        score, issues = _check_actionability(
            "The car balance is neutral. Everything looks good."
        )
        assert score == 0.0

    def test_limited_actionable(self) -> None:
        score, issues = _check_actionability(
            "建议调整前翼角度，车平衡良好"
        )
        assert 0.0 < score < 1.0


class TestAssessResponseQuality:
    def test_full_report(self) -> None:
        feedback = {
            "summary": "建议调整前翼增加 2度，预计提升 0.3秒。胎温 85°C 正常。",
            "dimensions": [
                {"name": "balance", "value": "Mild understeer"},
                {"name": "grip", "value": "Adequate"},
                {"name": "tyres", "value": "Normal"},
                {"name": "braking", "value": "Stable"},
                {"name": "ers_drs", "value": "Optimal"},
                {"name": "throttle_brake_smoothness", "value": "Smooth"},
                {"name": "confidence", "value": "High"},
                {"name": "lap_time_potential", "value": "0.3s gain"},
                {"name": "sector_compare", "value": "S1 slow"},
                {"name": "setup_advice", "value": "Front wing +2"},
            ],
        }
        sources = [
            {"frame_t": 1.0, "field": "tyre_temp", "value": 85.0},
            {"frame_t": 2.0, "field": "speed", "value": 320.0},
        ]
        report = assess_response_quality(feedback, sources)
        assert isinstance(report, ResponseQualityReport)
        assert 0.0 <= report.groundedness <= 1.0
        assert 0.0 <= report.completeness <= 1.0
        assert 0.0 <= report.actionability <= 1.0
        assert 0.0 <= report.overall <= 1.0
        assert report.label in ("excellent", "good", "fair", "poor")
        assert isinstance(report.issues, list)

    def test_no_sources(self) -> None:
        feedback = {"summary": "OK", "dimensions": []}
        report = assess_response_quality(feedback, None)
        assert report.groundedness == 1.0

    def test_custom_weights(self) -> None:
        feedback = {
            "summary": "建议调整前翼增加 2度",
            "dimensions": [
                {"name": "balance", "value": "Mild understeer"},
                {"name": "grip", "value": "Good"},
            ],
        }
        r1 = assess_response_quality(feedback, weights=(0.5, 0.3, 0.2))
        r2 = assess_response_quality(feedback, weights=(0.2, 0.3, 0.5))
        assert r1.overall != r2.overall  # Different weights => different overall

    def test_poor_feedback(self) -> None:
        feedback = {
            "summary": "Speed is 500 kph but no evidence",
            "dimensions": [
                {"name": "balance", "value": "N/A"},
                {"name": "grip", "value": ""},
            ],
        }
        sources = [{"frame_t": 1.0, "field": "speed", "value": 300.0}]
        report = assess_response_quality(feedback, sources)
        assert report.label == "poor"
        assert report.overall < 0.4