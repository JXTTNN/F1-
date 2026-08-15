"""Telemetry data quality scoring (Iter-159).

EA F1 2026 professional standard: before feeding telemetry data into the
model pipeline, the team needs a quick, objective quality score that
summarizes the reliability of the data stream. This module provides a
composite quality score based on multiple dimensions:

- **Packet loss rate**: fraction of expected packets that were lost.
- **Field completeness**: fraction of expected fields that have non-null values.
- **Timestamp regularity**: how evenly spaced the timestamps are (low jitter).
- **Anomaly rate**: fraction of frames flagged as anomalous by the
  :class:`~f1opt.telemetry.analytics.AnomalyDetector`.
- **Value range compliance**: fraction of values within expected physical ranges.

Each dimension produces a sub-score in [0, 1], and the overall score is a
weighted average. The module also produces a human-readable quality label
and a list of specific issues found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DataQualityReport",
    "score_data_quality",
]

# Expected physical ranges for key telemetry fields (value, min, max).
# Values outside these ranges indicate sensor errors or data corruption.
_EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    "speed": (0.0, 350.0),          # km/h
    "throttle": (0.0, 1.0),         # normalized
    "brake": (0.0, 1.0),            # normalized
    "steer": (-1.0, 1.0),           # normalized
    "g_lat": (-10.0, 10.0),         # g
    "g_long": (-10.0, 10.0),        # g
    "rpm": (0.0, 20000.0),          # rpm
    "gear": (0, 8),                 # F1 gears 1-8 (0=neutral)
    "fuel_remaining": (0.0, 200.0), # kg
    "lap_distance": (0.0, 10000.0), # meters
    "tyre_temp_fl": (0.0, 200.0),   # Celsius
    "tyre_temp_fr": (0.0, 200.0),
    "tyre_temp_rl": (0.0, 200.0),
    "tyre_temp_rr": (0.0, 200.0),
}

# Weights for each quality dimension (must sum to 1.0).
_WEIGHTS: dict[str, float] = {
    "packet_loss": 0.30,
    "completeness": 0.25,
    "regularity": 0.15,
    "anomaly_rate": 0.15,
    "range_compliance": 0.15,
}


@dataclass
class DataQualityReport:
    """Composite data quality report (Iter-159).

    Attributes:
        overall: Overall quality score in [0, 1] (1.0 = perfect).
        label: Human-readable label: "excellent" (>=0.9), "good" (>=0.7),
            "fair" (>=0.5), "poor" (<0.5).
        packet_loss_score: Sub-score for packet loss (1.0 = no loss).
        completeness_score: Sub-score for field completeness (1.0 = all present).
        regularity_score: Sub-score for timestamp regularity (1.0 = perfect).
        anomaly_score: Sub-score for anomaly rate (1.0 = no anomalies).
        range_compliance_score: Sub-score for value range compliance.
        issues: List of specific issues found (strings).
        n_frames: Number of frames analyzed.
    """
    overall: float
    label: str
    packet_loss_score: float
    completeness_score: float
    regularity_score: float
    anomaly_score: float
    range_compliance_score: float
    issues: list[str] = field(default_factory=list)
    n_frames: int = 0


def _compute_packet_loss_score(
    n_expected: int, n_received: int, issues: list[str]
) -> float:
    """Score based on packet loss rate (1.0 = no loss, 0.0 = total loss)."""
    if n_expected <= 0:
        return 1.0
    loss_rate = 1.0 - (n_received / n_expected)
    if loss_rate > 0.05:
        issues.append(f"Packet loss rate {loss_rate:.1%} exceeds 5% threshold")
    return max(0.0, 1.0 - loss_rate)


def _compute_completeness_score(
    frames: list[dict], expected_fields: list[str], issues: list[str]
) -> float:
    """Score based on fraction of non-null expected field values."""
    if not frames or not expected_fields:
        return 1.0
    total = 0
    present = 0
    missing_fields: set[str] = set()
    for frame in frames:
        for field_name in expected_fields:
            total += 1
            val = frame.get(field_name)
            if val is not None:
                present += 1
            else:
                missing_fields.add(field_name)
    if missing_fields:
        issues.append(
            f"Missing fields: {', '.join(sorted(missing_fields)[:5])}"
            f"{'...' if len(missing_fields) > 5 else ''}"
        )
    return present / total if total > 0 else 1.0


def _compute_regularity_score(
    timestamps: list[float], issues: list[str]
) -> float:
    """Score based on timestamp regularity (low jitter = high score)."""
    if len(timestamps) < 3:
        return 1.0
    deltas = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
        if timestamps[i + 1] > timestamps[i]
    ]
    if not deltas:
        return 0.0
    mean_delta = sum(deltas) / len(deltas)
    if mean_delta <= 0:
        return 0.0
    # Coefficient of variation (CV = std/mean) measures jitter
    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    std = variance ** 0.5
    cv = std / mean_delta
    # CV=0 → perfect regularity (score 1.0), CV>=1 → poor (score 0.0)
    score = max(0.0, 1.0 - cv)
    if cv > 0.5:
        issues.append(
            f"Timestamp jitter high (CV={cv:.2f}, mean_dt={mean_delta:.4f}s)"
        )
    return score


def _compute_anomaly_score(
    n_anomalies: int, n_frames: int, issues: list[str]
) -> float:
    """Score based on anomaly rate (1.0 = no anomalies)."""
    if n_frames <= 0:
        return 1.0
    rate = n_anomalies / n_frames
    if rate > 0.10:
        issues.append(f"Anomaly rate {rate:.1%} exceeds 10% threshold")
    return max(0.0, 1.0 - rate)


def _compute_range_compliance_score(
    frames: list[dict], issues: list[str]
) -> float:
    """Score based on fraction of values within expected physical ranges."""
    if not frames:
        return 1.0
    total = 0
    in_range = 0
    out_of_range_fields: dict[str, int] = {}
    for frame in frames:
        for field_name, (lo, hi) in _EXPECTED_RANGES.items():
            val = frame.get(field_name)
            if val is None:
                continue
            total += 1
            try:
                v = float(val)
                if lo <= v <= hi:
                    in_range += 1
                else:
                    out_of_range_fields[field_name] = (
                        out_of_range_fields.get(field_name, 0) + 1
                    )
            except (TypeError, ValueError):
                pass
    if out_of_range_fields:
        worst = max(out_of_range_fields, key=lambda k: out_of_range_fields[k])
        issues.append(f"Out-of-range values in: {worst}")
    return in_range / total if total > 0 else 1.0


def _label_from_score(score: float) -> str:
    if score >= 0.9:
        return "excellent"
    if score >= 0.7:
        return "good"
    if score >= 0.5:
        return "fair"
    return "poor"


def score_data_quality(
    frames: list[dict],
    *,
    n_expected_packets: int | None = None,
    expected_fields: list[str] | None = None,
    n_anomalies: int = 0,
    timestamp_field: str = "session_time",
) -> DataQualityReport:
    """Compute a composite data quality score (Iter-159).

    Args:
        frames: List of telemetry frame dicts to evaluate.
        n_expected_packets: Expected number of packets (for loss calculation).
            If ``None``, defaults to ``len(frames)`` (no loss detected).
        expected_fields: List of field names expected to be present in each
            frame. If ``None``, uses the keys from :data:`_EXPECTED_RANGES`.
        n_anomalies: Number of anomalous frames detected by AnomalyDetector.
        timestamp_field: Field name containing the timestamp for regularity
            calculation (default ``"session_time"``).

    Returns:
        :class:`DataQualityReport` with overall score, sub-scores, and issues.
    """
    issues: list[str] = []
    n_frames = len(frames)

    if n_expected_packets is None:
        n_expected_packets = n_frames

    # Sub-scores
    packet_loss_score = _compute_packet_loss_score(
        n_expected_packets, n_frames, issues
    )
    completeness_score = _compute_completeness_score(
        frames, expected_fields or list(_EXPECTED_RANGES.keys()), issues
    )

    timestamps = [
        float(f[timestamp_field])
        for f in frames
        if f.get(timestamp_field) is not None
    ]
    timestamps.sort()
    regularity_score = _compute_regularity_score(timestamps, issues)

    anomaly_score = _compute_anomaly_score(n_anomalies, n_frames, issues)
    range_compliance_score = _compute_range_compliance_score(frames, issues)

    # Weighted average
    overall = (
        _WEIGHTS["packet_loss"] * packet_loss_score
        + _WEIGHTS["completeness"] * completeness_score
        + _WEIGHTS["regularity"] * regularity_score
        + _WEIGHTS["anomaly_rate"] * anomaly_score
        + _WEIGHTS["range_compliance"] * range_compliance_score
    )

    return DataQualityReport(
        overall=overall,
        label=_label_from_score(overall),
        packet_loss_score=packet_loss_score,
        completeness_score=completeness_score,
        regularity_score=regularity_score,
        anomaly_score=anomaly_score,
        range_compliance_score=range_compliance_score,
        issues=issues,
        n_frames=n_frames,
    )
