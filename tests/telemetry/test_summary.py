"""Tests for :mod:`f1opt.telemetry.summary`.

Covers the online telemetry statistics helpers that previously had no
dedicated coverage:

- :class:`RunningStats` — Welford online mean/variance/min/max, plus the
  empty-state defaults, ``reset`` and ``to_dict`` serialization.
- :class:`FieldStats` — per-field aggregation over telemetry frames,
  non-numeric / missing field handling, anomaly detection and reset.
- :class:`TelemetryHealth` — combined health score, penalty saturation,
  score clamping and the human-readable summary string.
"""

from __future__ import annotations

import math

import pytest

from f1opt.telemetry.summary import (
    FieldStats,
    HealthReport,
    RunningStats,
    TelemetryHealth,
)


# --------------------------------------------------------------------------- #
# RunningStats
# --------------------------------------------------------------------------- #
def test_running_stats_empty_defaults() -> None:
    """A fresh RunningStats reports zeroed stats (no division by zero)."""
    s = RunningStats()
    assert s.n == 0
    assert s.mean == 0.0
    assert s.variance == 0.0
    assert s.std == 0.0
    assert s.min == 0.0
    assert s.max == 0.0


def test_running_stats_single_sample() -> None:
    """With one sample, variance/std are 0 but mean/min/max equal the value."""
    s = RunningStats()
    s.update(4.0)
    assert s.n == 1
    assert s.mean == 4.0
    assert s.variance == 0.0
    assert s.std == 0.0
    assert s.min == 4.0
    assert s.max == 4.0


def test_running_stats_matches_numpy_style_moments() -> None:
    """Mean/variance match the population moments computed directly."""
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    s = RunningStats()
    for x in data:
        s.update(x)

    expected_mean = sum(data) / len(data)
    expected_var = sum((x - expected_mean) ** 2 for x in data) / len(data)

    assert s.n == len(data)
    assert math.isclose(s.mean, expected_mean, rel_tol=1e-12)
    assert math.isclose(s.variance, expected_var, rel_tol=1e-9)
    assert math.isclose(s.std, math.sqrt(expected_var), rel_tol=1e-9)
    assert s.min == 2.0
    assert s.max == 9.0


def test_running_stats_reset() -> None:
    """reset() returns the accumulator to its empty state."""
    s = RunningStats()
    for x in (1.0, 2.0, 3.0):
        s.update(x)
    s.reset()
    assert s.n == 0
    assert s.mean == 0.0
    assert s.min == 0.0
    assert s.max == 0.0
    # Usable again after reset.
    s.update(10.0)
    assert s.n == 1
    assert s.mean == 10.0


def test_running_stats_to_dict() -> None:
    """to_dict exposes the stable summary keys."""
    s = RunningStats()
    for x in (1.0, 3.0):
        s.update(x)
    d = s.to_dict()
    assert set(d) == {"mean", "std", "min", "max", "n"}
    assert d["mean"] == 2.0
    assert d["min"] == 1.0
    assert d["max"] == 3.0
    assert d["n"] == 2


# --------------------------------------------------------------------------- #
# FieldStats
# --------------------------------------------------------------------------- #
def test_field_stats_tracks_known_fields() -> None:
    """update() feeds numeric tracked fields into their accumulators."""
    fs = FieldStats()
    fs.update({"speed": 100.0, "throttle": 0.5})
    fs.update({"speed": 200.0, "throttle": 1.0})

    summary = fs.summary()
    assert fs.total_frames == 2
    assert summary["speed"]["n"] == 2
    assert summary["speed"]["mean"] == 150.0
    assert summary["speed"]["min"] == 100.0
    assert summary["speed"]["max"] == 200.0
    assert summary["throttle"]["n"] == 2


def test_field_stats_ignores_missing_and_non_numeric() -> None:
    """Missing keys, None and non-numeric values are skipped."""
    fs = FieldStats()
    fs.update({"speed": 120.0, "gear": None, "rpm": "fast"})
    fs.update({"speed": 130.0})  # no gear / rpm at all

    summary = fs.summary()
    assert fs.total_frames == 2
    assert summary["speed"]["n"] == 2
    # Non-numeric / missing fields never accumulated a sample.
    assert summary["gear"]["n"] == 0
    assert summary["rpm"]["n"] == 0


def test_field_stats_bool_is_accepted_as_numeric() -> None:
    """bool is a subclass of int and is treated as a numeric sample."""
    fs = FieldStats()
    fs.update({"brake": True})
    fs.update({"brake": False})
    assert fs.summary()["brake"]["n"] == 2


def test_field_stats_anomalies_requires_min_samples() -> None:
    """Fields below the min-sample threshold never flag anomalies."""
    fs = FieldStats(min_samples=30)
    for _ in range(10):
        fs.update({"speed": 100.0})
    assert fs.anomalies() == {}


def test_field_stats_anomalies_detects_large_deviation() -> None:
    """A field whose mean is many sigma from zero is flagged."""
    fs = FieldStats(min_samples=5)
    # mean ~100 with tiny std => abs(mean)/std is huge => anomaly.
    for i in range(50):
        fs.update({"speed": 100.0 + (i % 2) * 0.01})

    result = fs.anomalies(n_sigma=3.0)
    assert "speed" in result
    entry = result["speed"][0]
    assert entry["field"] == "speed"
    assert entry["n_sigma"] > 3.0
    assert set(entry) == {"field", "mean", "std", "min", "max", "n_sigma"}


def test_field_stats_anomalies_field_filter() -> None:
    """The ``fields`` argument restricts which fields are inspected."""
    fs = FieldStats(min_samples=5)
    for _ in range(20):
        fs.update({"speed": 100.0, "throttle": 100.0})

    only_speed = fs.anomalies(n_sigma=1.0, fields=["speed"])
    assert set(only_speed) <= {"speed"}
    assert "throttle" not in only_speed


def test_field_stats_anomalies_skips_zero_std() -> None:
    """Constant fields (std ~ 0) are not flagged despite non-zero mean."""
    fs = FieldStats(min_samples=5)
    for _ in range(20):
        fs.update({"speed": 100.0})  # perfectly constant => std == 0
    assert fs.anomalies(n_sigma=1.0) == {}


def test_field_stats_reset() -> None:
    """reset() clears frame counter and every field accumulator."""
    fs = FieldStats()
    for _ in range(5):
        fs.update({"speed": 100.0})
    fs.reset()
    assert fs.total_frames == 0
    assert fs.summary()["speed"]["n"] == 0


# --------------------------------------------------------------------------- #
# TelemetryHealth
# --------------------------------------------------------------------------- #
def test_telemetry_health_perfect_feed() -> None:
    """No loss/gaps/anomalies yields a perfect score and Healthy summary."""
    report = TelemetryHealth().compute(
        loss_rate=0.0,
        gap_rate=0.0,
        anomaly_count=0,
        total_frames=1000,
        anomalous_fields=[],
    )
    assert isinstance(report, HealthReport)
    assert report.health_score == 1.0
    assert report.summary == "Excellent"
    assert report.packet_loss_rate == 0.0
    assert report.total_frames == 1000


def test_telemetry_health_score_decreases_with_loss() -> None:
    """Packet loss lowers the score and is reported in the summary."""
    report = TelemetryHealth().compute(
        loss_rate=0.05,
        gap_rate=0.0,
        anomaly_count=0,
        total_frames=1000,
        anomalous_fields=[],
    )
    assert report.health_score < 1.0
    assert "packet loss" in report.summary


def test_telemetry_health_penalties_saturate_and_clamp() -> None:
    """Extreme inputs saturate penalties; score is clamped to [0, 1]."""
    report = TelemetryHealth().compute(
        loss_rate=1.0,
        gap_rate=1.0,
        anomaly_count=10_000,
        total_frames=100,
        anomalous_fields=["speed", "throttle"],
    )
    assert report.health_score == 0.0
    assert report.summary.startswith("Poor")


def test_telemetry_health_summary_labels() -> None:
    """Score bands map to the expected qualitative labels."""
    th = TelemetryHealth()

    # score = 1 - 0.4*min(1, 0.05*10) = 1 - 0.4*0.5 = 0.8 -> "Good"
    good = th.compute(
        loss_rate=0.05, gap_rate=0.0, anomaly_count=0,
        total_frames=1000, anomalous_fields=[],
    )
    assert good.health_score == pytest.approx(0.8)
    assert good.summary.startswith("Good")

    # score = 1 - 0.4*min(1, 0.1*10) = 1 - 0.4 = 0.6 -> "Fair"
    fair = th.compute(
        loss_rate=0.1, gap_rate=0.0, anomaly_count=0,
        total_frames=1000, anomalous_fields=[],
    )
    assert fair.health_score == pytest.approx(0.6)
    assert fair.summary.startswith("Fair")


def test_telemetry_health_lists_anomalous_fields_capped() -> None:
    """The summary lists at most five anomalous fields."""
    fields = [f"f{i}" for i in range(8)]
    report = TelemetryHealth().compute(
        loss_rate=0.0,
        gap_rate=0.0,
        anomaly_count=1,
        total_frames=1000,
        anomalous_fields=fields,
    )
    assert report.fields_anomalous == fields
    assert "anomalies:" in report.summary
    # Only the first five field names are embedded in the summary text.
    assert "f4" in report.summary
    assert "f5" not in report.summary.split("anomalies:")[1]


def test_telemetry_health_zero_total_frames_no_div_error() -> None:
    """total_frames=0 must not raise (guarded division)."""
    report = TelemetryHealth().compute(
        loss_rate=0.0,
        gap_rate=0.0,
        anomaly_count=5,
        total_frames=0,
        anomalous_fields=[],
    )
    assert 0.0 <= report.health_score <= 1.0


def test_telemetry_health_custom_weights() -> None:
    """Custom weights change the relative contribution of each penalty."""
    loss_only = TelemetryHealth(loss_weight=1.0, gap_weight=0.0, anomaly_weight=0.0)
    report = loss_only.compute(
        loss_rate=0.1,  # loss_penalty saturates to 1.0
        gap_rate=1.0,   # ignored (weight 0)
        anomaly_count=1000,  # ignored (weight 0)
        total_frames=10,
        anomalous_fields=[],
    )
    assert report.health_score == 0.0
