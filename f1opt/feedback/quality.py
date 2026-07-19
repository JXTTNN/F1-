"""LLM response quality assessment (Iter-146).

EA F1 2026 professional standard: every LLM-generated feedback response must
be assessed for groundedness, completeness, and actionability. This module
provides automated quality scoring that can be embedded in the feedback
pipeline — both for the rule-based path (self-check) and the LLM-enhanced
path (external-audit).

Key assessments:

- **Groundedness** (0.0–1.0): Do numeric claims in the response trace back to
  telemetry evidence? Penalises invented numbers and hallucinated values.
- **Completeness** (0.0–1.0): How many of the 10 feedback dimensions have
  meaningful non-empty entries?
- **Actionability** (0.0–1.0): Does the response contain concrete, specific
  advice (setup changes, lap-time estimates, sector-specific guidance)?
- **Overall** (0.0–1.0): Weighted average of the three scores.

Usage::

    from f1opt.feedback.quality import assess_response_quality
    report = assess_response_quality(feedback_dict, telemetry_sources)
    print(report.overall, report.label)
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass


@dataclass
class ResponseQualityReport:
    """Structured quality scores for a feedback response (Iter-146)."""

    groundedness: float
    """0.0–1.0: how well numeric claims are traceable to evidence."""

    completeness: float
    """0.0–1.0: fraction of dimensions with non-empty entries."""

    actionability: float
    """0.0–1.0: whether the response gives concrete, specific advice."""

    overall: float
    """0.0–1.0: weighted average (0.4×groundedness + 0.3×completeness + 0.3×actionability)."""

    label: str
    """Quality label: ``"excellent"`` (>=0.8), ``"good"`` (>=0.6),
    ``"fair"`` (>=0.4), ``"poor"`` (<0.4)."""

    issues: list[str]
    """Human-readable list of quality concerns found."""


# Patterns for detecting numeric claims in response text.
_NUMERIC_PATTERN = _re.compile(
    r"(\d+\.?\d*)\s*(s|秒|kph|km/h|mph|deg|°|C|℃|℃|%|G|g|bar|psi|mm|cm|m|kg|N|Nm|kW|hp)"
)
# Chinese numeric patterns: "0.3秒", "15度", "2.5 G"
_CN_NUMERIC_PATTERN = _re.compile(
    r"(\d+\.?\d*)\s*(秒|度|公里|米|厘米|毫米|牛|千瓦|马力)"
)


def _extract_numeric_claims(text: str) -> list[tuple[float, str]]:
    """Extract all numeric claims (value, unit) from response text."""
    claims: list[tuple[float, str]] = []
    for m in _NUMERIC_PATTERN.finditer(text):
        claims.append((float(m.group(1)), m.group(2)))
    for m in _CN_NUMERIC_PATTERN.finditer(text):
        claims.append((float(m.group(1)), m.group(2)))
    return claims


def _extract_evidence_values(sources: list[dict]) -> set[str]:
    """Extract all distinct evidence snippets for groundedness comparison."""
    tokens: set[str] = set()
    for s in sources:
        val = s.get("value")
        if val is not None:
            if isinstance(val, float):
                tokens.add(f"{val:.2f}")
                tokens.add(f"{val:.1f}")
                tokens.add(str(int(val)))
            else:
                tokens.add(str(val))
    return tokens


def _check_groundedness(text: str, sources: list[dict]) -> tuple[float, list[str]]:
    """Check whether numeric claims can be traced to telemetry evidence."""
    claims = _extract_numeric_claims(text)
    if not claims:
        return 1.0, []  # No numeric claims = no hallucination risk

    evidence = _extract_evidence_values(sources)
    issues: list[str] = []
    matched = 0
    for val, unit in claims:
        val_str = f"{val:.1f}"
        # Check if any evidence token contains a similar value
        found = any(
            str(val_str) in ev or ev in str(val_str)
            for ev in evidence
        ) or any(
            # Check ±1% tolerance for exact match
            abs(float(ev) - val) < 0.01 * max(abs(val), 1e-6)
            for ev in evidence
            if ev.replace(".", "").replace("-", "").isdigit()
        )
        if found:
            matched += 1
        else:
            issues.append(f"Unverified claim: {val}{unit}")

    if not claims:
        return 1.0, issues
    score = matched / len(claims)
    return score, issues


def _check_completeness(feedback: dict) -> tuple[float, list[str]]:
    """Check how many feedback dimensions have meaningful content."""
    dims = feedback.get("dimensions", [])
    if not dims:
        return 0.0, ["No feedback dimensions found"]

    total = len(dims)
    filled = 0
    issues: list[str] = []
    for d in dims:
        name = d.get("name", "unknown")
        value = d.get("value", "")
        # Meaningful = non-empty and not just "N/A" / "OK" / "-"
        if value and value.strip() not in ("N/A", "OK", "-", "—", "未检测", "无"):
            filled += 1
        else:
            issues.append(f"Dimension '{name}' has no meaningful value")

    score = filled / total if total > 0 else 0.0
    return score, issues


def _check_actionability(text: str) -> tuple[float, list[str]]:
    """Check whether the response contains concrete, actionable advice."""
    actionable_patterns = [
        _re.compile(r"建议|推荐|调整|增加|减少|升高|降低|修改|设[置定]"),
        _re.compile(r"try|adjust|increase|decrease|set|change|modify|recommend"),
        _re.compile(r"前翼|后翼|差速器|防倾杆|弹簧|离地间隙|外倾角|前束|胎压|制动"),
        _re.compile(r"front.wing|rear.wing|diff|anti.roll|spring|ride.height|camber|toe|tyre.pressure|brake"),
        _re.compile(r"\d+\.?\d*\s*(s|秒)\s*(gain|提升|gain|improvement|faster)"),
        _re.compile(r"predicted_gain|lap_time_potential|预计.*提升"),
    ]
    issues: list[str] = []
    hits = 0
    for pat in actionable_patterns:
        if pat.search(text):
            hits += 1

    if hits == 0:
        issues.append("No actionable advice found")
        score = 0.0
    elif hits >= 3:
        score = 1.0
    elif hits >= 2:
        score = 0.7
    else:
        score = 0.4
        issues.append("Limited actionable advice")

    return score, issues


def assess_response_quality(
    feedback: dict,
    telemetry_sources: list[dict] | None = None,
    *,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> ResponseQualityReport:
    """Assess the quality of a feedback response (Iter-146).

    Evaluates three dimensions and produces a structured report suitable for
    logging, dashboard display, and automated quality gating.

    Args:
        feedback: The feedback dict returned by ``FeedbackEngine.run()``
            (must contain ``summary`` and ``dimensions`` keys).
        telemetry_sources: Optional list of telemetry evidence dicts
            (``{frame_t, field, value}``) for groundedness checking.
            When ``None``, groundedness defaults to 1.0.
        weights: ``(groundedness_weight, completeness_weight, actionability_weight)``.
            Default ``(0.4, 0.3, 0.3)``.

    Returns:
        :class:`ResponseQualityReport` with scores and issues.
    """
    all_issues: list[str] = []
    text = feedback.get("summary", "")

    # Groundedness
    if telemetry_sources and len(telemetry_sources) > 0:
        g_score, g_issues = _check_groundedness(text, telemetry_sources)
    else:
        g_score, g_issues = 1.0, []
    all_issues.extend(g_issues)

    # Completeness
    c_score, c_issues = _check_completeness(feedback)
    all_issues.extend(c_issues)

    # Actionability
    a_score, a_issues = _check_actionability(text)
    all_issues.extend(a_issues)

    overall = weights[0] * g_score + weights[1] * c_score + weights[2] * a_score

    if overall >= 0.8:
        label = "excellent"
    elif overall >= 0.6:
        label = "good"
    elif overall >= 0.4:
        label = "fair"
    else:
        label = "poor"

    return ResponseQualityReport(
        groundedness=g_score,
        completeness=c_score,
        actionability=a_score,
        overall=overall,
        label=label,
        issues=all_issues,
    )


__all__ = [
    "ResponseQualityReport",
    "assess_response_quality",
]