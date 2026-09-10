"""Setup safety filter — reject harmful parameter adjustments (Opt-Safety).

When the LLM generates setup suggestions, this module validates each proposed
change against physical limits, track-specific constraints, and user feedback
consistency. Changes that would be physically impossible, dangerous to the car,
or contradictory to the user's reported issues are rejected with an explanation.

The filter operates as a post-processor on the LLM output before it reaches
the frontend. It never modifies approved suggestions — only removes or flags
harmful ones.

Updated 2026-09-09: PARAM_LIMITS and FEEDBACK_CONTRADICTIONS now use real
SETUP_FIELDS names (front_wing, rear_wing, front_brake_bias, on_throttle_diff
etc.) and are derived from the unified issue registry where available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from f1opt.feedback.registry import get_contradictions_for_issue

log = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a single parameter safety check."""

    param_name: str
    original_after: float
    clamped_after: float
    is_safe: bool
    reason: str = ""


@dataclass
class SafetyReport:
    """Aggregated safety report for a full suggestion set."""

    checks: list[SafetyCheckResult] = field(default_factory=list)
    rejected_params: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def all_safe(self) -> bool:
        return len(self.rejected_params) == 0

    def summary(self) -> str:
        if self.all_safe and not self.warnings:
            return "All adjustments within safe bounds."
        parts = []
        if self.rejected_params:
            parts.append(f"Rejected {len(self.rejected_params)} harmful adjustment(s): {', '.join(self.rejected_params)}")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s): {'; '.join(self.warnings[:3])}")
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Physical limits for F1 2026 setup parameters (conservative bounds)
# Names now match setup_schema.SETUP_FIELDS exactly.
# --------------------------------------------------------------------------- #

PARAM_LIMITS: dict[str, dict[str, float]] = {
    # Aerodynamics
    "front_wing": {"min": 0, "max": 50, "step": 1, "safe_delta_per_lap": 6},
    "rear_wing": {"min": 0, "max": 50, "step": 1, "safe_delta_per_lap": 6},
    "front_ride_height": {"min": 10, "max": 80, "step": 1, "safe_delta_per_lap": 5},
    "rear_ride_height": {"min": 10, "max": 80, "step": 1, "safe_delta_per_lap": 5},
    # Suspension
    "front_suspension_stiffness": {"min": 1, "max": 100, "step": 1, "safe_delta_per_lap": 10},
    "rear_suspension_stiffness": {"min": 1, "max": 100, "step": 1, "safe_delta_per_lap": 10},
    "front_arb": {"min": 1, "max": 100, "step": 1, "safe_delta_per_lap": 8},
    "rear_arb": {"min": 1, "max": 100, "step": 1, "safe_delta_per_lap": 8},
    "front_camber": {"min": -5.0, "max": 0, "step": 0.1, "safe_delta_per_lap": 0.5},
    "rear_camber": {"min": -5.0, "max": 0, "step": 0.1, "safe_delta_per_lap": 0.5},
    "front_toe": {"min": -3.0, "max": 3.0, "step": 0.1, "safe_delta_per_lap": 0.3},
    "rear_toe": {"min": -3.0, "max": 3.0, "step": 0.1, "safe_delta_per_lap": 0.3},
    # Differential
    "on_throttle_diff": {"min": 50, "max": 100, "step": 1, "safe_delta_per_lap": 8},
    "off_throttle_diff": {"min": 50, "max": 100, "step": 1, "safe_delta_per_lap": 8},
    # Brakes
    "brake_pressure": {"min": 50, "max": 100, "step": 1, "safe_delta_per_lap": 5},
    "front_brake_bias": {"min": 45, "max": 65, "step": 0.5, "safe_delta_per_lap": 1.0},
    # Tyres
    "front_tyre_pressure": {"min": 15.0, "max": 28.0, "step": 0.1, "safe_delta_per_lap": 0.5},
    "rear_tyre_pressure": {"min": 15.0, "max": 28.0, "step": 0.1, "safe_delta_per_lap": 0.5},
    # Engine / ERS
    "fuel_mix": {"min": 1, "max": 3, "step": 1, "safe_delta_per_lap": 1},
    "ers_deployment_mode": {"min": 0, "max": 3, "step": 1, "safe_delta_per_lap": 1},
}


# --------------------------------------------------------------------------- #
# Feedback-consistency rules (derived from ISSUE_REGISTRY)
# We build a flat FEEDBACK_CONTRADICTIONS for backward compatibility, but the
# source of truth is ISSUE_REGISTRY contradictions. Any key not present falls
# back to an empty list (no contradiction).
# --------------------------------------------------------------------------- #

FEEDBACK_CONTRADICTIONS: dict[str, list[dict[str, Any]]] = {}


def _refresh_contradictions() -> None:
    """Rebuild FEEDBACK_CONTRADICTIONS from the registry."""
    FEEDBACK_CONTRADICTIONS.clear()
    for phase, issues in getattr(
        __import__("f1opt.feedback.registry", fromlist=["ISSUE_REGISTRY"]),
        "ISSUE_REGISTRY",
        {},
    ).items():
        for issue_id, data in issues.items():
            contras = data.get("contradictions", [])
            if contras:
                FEEDBACK_CONTRADICTIONS[issue_id] = contras


_refresh_contradictions()


def check_parameter_safety(
    param_name: str,
    before: float,
    after: float,
) -> SafetyCheckResult:
    """Check if a single parameter adjustment is physically safe.

    Args:
        param_name: Setup parameter name.
        before: Current value.
        after: Proposed new value.

    Returns:
        SafetyCheckResult with is_safe=True if within bounds, or clamped value.
    """
    limits = PARAM_LIMITS.get(param_name)
    if limits is None:
        # Unknown parameter — allow but warn
        return SafetyCheckResult(
            param_name=param_name,
            original_after=after,
            clamped_after=after,
            is_safe=True,
            reason="Unknown parameter, no limits defined",
        )

    # Clamp to physical bounds
    clamped = max(limits["min"], min(limits["max"], after))

    # Check delta magnitude — warn if changing too aggressively in one step
    delta = abs(after - before)
    safe_delta = limits.get("safe_delta_per_lap", float("inf"))
    reason = ""

    if delta > safe_delta * 2:
        # Hard reject: changing by more than 2x the safe delta per step
        clamped = before + (safe_delta if after > before else -safe_delta)
        clamped = max(limits["min"], min(limits["max"], clamped))
        reason = f"Delta {delta:.2f} exceeds 2x safe limit ({safe_delta:.2f}). Clamped to prevent mechanical stress."
        log.warning("Safety filter: %s delta %.2f > 2x safe_delta %.2f, clamped to %.2f",
                     param_name, delta, safe_delta, clamped)
    elif delta > safe_delta:
        # Soft warning: allow but flag
        reason = f"Delta {delta:.2f} exceeds single-step safe limit ({safe_delta:.2f}). Monitor closely."

    if clamped != after:
        log.info("Safety filter: %s clamped from %.2f to %.2f (bounds: %.2f-%.2f)",
                 param_name, after, clamped, limits["min"], limits["max"])

    return SafetyCheckResult(
        param_name=param_name,
        original_after=after,
        clamped_after=clamped,
        is_safe=True if not reason else False,
        reason=reason,
    )


def check_feedback_consistency(
    suggestions: list[dict[str, Any]],
    user_issues: list[str],
) -> list[str]:
    """Check if proposed suggestions contradict the user's reported issues.

    Args:
        suggestions: List of suggestion dicts with 'name' or 'field', 'after', 'before'.
        user_issues: List of feedback issue IDs (e.g., 'understeer_in').

    Returns:
        List of warning messages for contradictory suggestions.
    """
    warnings: list[str] = []

    for issue_id in user_issues:
        contradictions = get_contradictions_for_issue(issue_id) or \
            FEEDBACK_CONTRADICTIONS.get(issue_id, [])
        for contra in contradictions:
            for sug in suggestions:
                param_name = sug.get("name") or sug.get("field", "")
                if param_name != contra["param"]:
                    continue
                before = float(sug.get("before", 0))
                after = float(sug.get("after", 0))
                direction = "increase" if after > before else "decrease" if after < before else "same"
                if direction == contra["direction"]:
                    msg = (
                        f"WARNING: {contra['reason']}。"
                        f"User reported '{issue_id}', but suggestion changes "
                        f"{contra['param']} {direction} ({before:.2f} → {after:.2f}). "
                        f"This may worsen the reported issue."
                    )
                    warnings.append(msg)
                    log.warning("Feedback contradiction: %s", msg)

    return warnings


def validate_suggestions(
    suggestions: list[dict[str, Any]],
    user_issues: list[str] | None = None,
) -> SafetyReport:
    """Full safety validation on a set of LLM-generated suggestions.

    Args:
        suggestions: List of suggestion dicts from the LLM output.
        user_issues: Optional list of user-reported feedback issue IDs.

    Returns:
        SafetyReport with clamped values applied and warnings collected.
    """
    report = SafetyReport()

    if not suggestions:
        return report

    # 1. Parameter-level safety checks
    for sug in suggestions:
        name = sug.get("name", "") or sug.get("field", "")
        before = float(sug.get("before", 0))
        after = float(sug.get("after", 0))

        result = check_parameter_safety(name, before, after)
        report.checks.append(result)

        # Apply clamping
        sug["after"] = result.clamped_after

        if not result.is_safe:
            report.rejected_params.append(name)
            sug["_safety_note"] = result.reason

        if result.reason and result.is_safe:
            report.warnings.append(result.reason)

    # 2. Feedback consistency checks
    if user_issues:
        fb_warnings = check_feedback_consistency(suggestions, user_issues)
        report.warnings.extend(fb_warnings)
        for sug in suggestions:
            for warn in fb_warnings:
                if (sug.get("name", "") or sug.get("field", "")) in warn:
                    sug["_safety_warning"] = warn

    return report


def filter_harmful_suggestions(
    suggestions: list[dict[str, Any]],
    user_issues: list[str] | None = None,
) -> tuple[list[dict[str, Any]], SafetyReport]:
    """Filter out harmful suggestions and clamp borderline ones.

    Returns:
        Tuple of (filtered_suggestions, safety_report).
    """
    report = validate_suggestions(suggestions, user_issues)

    # Remove suggestions that were hard-rejected
    filtered = []
    for sug in suggestions:
        note = sug.get("_safety_note", "")
        if "exceeds 2x safe limit" in note:
            log.info("Filtered harmful suggestion: %s (%s)", sug.get("name") or sug.get("field"), note)
            continue
        filtered.append(sug)

    return filtered, report