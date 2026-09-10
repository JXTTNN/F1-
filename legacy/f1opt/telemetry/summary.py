"""Real-time telemetry statistics summary (Iter-143).

Online running statistics, per-field anomaly detection, and telemetry
health scoring for the EA F1 2026 dashboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass as _dataclass
from typing import Any

__all__ = [
    "FieldStats",
    "HealthReport",
    "RunningStats",
    "TelemetryHealth",
]


class RunningStats:
    """Online mean/variance via Welford's algorithm."""

    def __init__(self) -> None:
        self.n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    @property
    def mean(self) -> float:
        return self._mean if self.n > 0 else 0.0

    @property
    def variance(self) -> float:
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance) if self.n > 1 else 0.0

    @property
    def min(self) -> float:
        return self._min if self.n > 0 else 0.0

    @property
    def max(self) -> float:
        return self._max if self.n > 0 else 0.0

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean, "std": self.std,
            "min": self.min, "max": self.max, "n": self.n,
        }


_TRACKED_FIELDS: tuple[str, ...] = (
    "speed", "throttle", "brake", "steer", "rpm", "gear",
    "g_lat", "g_long", "g_vert",
    "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
    "tyre_wear_fl", "tyre_wear_fr", "tyre_wear_rl", "tyre_wear_rr",
    "ers_store", "fuel_in_tank", "lap_time", "lap_distance",
)


class FieldStats:
    """Per-field running statistics for a telemetry frame stream."""

    def __init__(self, min_samples: int = 30) -> None:
        self._stats: dict[str, RunningStats] = {
            f: RunningStats() for f in _TRACKED_FIELDS
        }
        self._min_samples = min_samples
        self._total_frames = 0

    def update(self, frame: dict[str, float]) -> None:
        self._total_frames += 1
        for field in _TRACKED_FIELDS:
            val = frame.get(field)
            if val is not None and isinstance(val, (int, float)):
                self._stats[field].update(float(val))

    def summary(self) -> dict[str, dict[str, float]]:
        return {f: s.to_dict() for f, s in self._stats.items()}

    def anomalies(
        self, n_sigma: float = 3.0, fields: list[str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        target = fields if fields is not None else list(_TRACKED_FIELDS)
        result: dict[str, list[dict[str, Any]]] = {}
        for f in target:
            s = self._stats[f]
            if s.n < self._min_samples or s.std < 1e-9:
                continue
            deviation = abs(s.mean) / s.std
            if deviation > n_sigma:
                result[f] = [{
                    "field": f, "mean": s.mean, "std": s.std,
                    "min": s.min, "max": s.max, "n_sigma": deviation,
                }]
        return result

    def reset(self) -> None:
        self._total_frames = 0
        for s in self._stats.values():
            s.reset()

    @property
    def total_frames(self) -> int:
        return self._total_frames


@_dataclass
class HealthReport:
    health_score: float
    packet_loss_rate: float
    frame_gap_rate: float
    anomaly_count: int
    total_frames: int
    fields_anomalous: list[str]
    summary: str


class TelemetryHealth:
    """Combined health score for the telemetry feed."""

    def __init__(
        self,
        loss_weight: float = 0.4,
        gap_weight: float = 0.3,
        anomaly_weight: float = 0.3,
    ) -> None:
        self._loss_weight = loss_weight
        self._gap_weight = gap_weight
        self._anomaly_weight = anomaly_weight

    def compute(
        self,
        loss_rate: float,
        gap_rate: float,
        anomaly_count: int,
        total_frames: int,
        anomalous_fields: list[str],
    ) -> HealthReport:
        loss_penalty = min(1.0, loss_rate * 10.0)
        gap_penalty = min(1.0, gap_rate * 10.0)
        anomaly_ratio = anomaly_count / max(total_frames, 1)
        anomaly_penalty = min(1.0, anomaly_ratio * 20.0)

        score = 1.0 - (
            self._loss_weight * loss_penalty
            + self._gap_weight * gap_penalty
            + self._anomaly_weight * anomaly_penalty
        )
        score = max(0.0, min(1.0, score))

        parts: list[str] = []
        if score >= 0.9:
            parts.append("Excellent")
        elif score >= 0.7:
            parts.append("Good")
        elif score >= 0.5:
            parts.append("Fair")
        else:
            parts.append("Poor")
        if loss_rate > 0.01:
            parts.append(f"{loss_rate:.1%} packet loss")
        if gap_rate > 0.01:
            parts.append(f"{gap_rate:.1%} frame gaps")
        if anomalous_fields:
            parts.append(f"anomalies: {', '.join(anomalous_fields[:5])}")

        return HealthReport(
            health_score=score,
            packet_loss_rate=loss_rate,
            frame_gap_rate=gap_rate,
            anomaly_count=anomaly_count,
            total_frames=total_frames,
            fields_anomalous=anomalous_fields,
            summary="; ".join(parts) if parts else "Healthy",
        )
