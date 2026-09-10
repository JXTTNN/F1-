"""Tests for :mod:`f1opt.telemetry.quality_score` (Iter-159)."""
from __future__ import annotations

from f1opt.telemetry.quality_score import DataQualityReport, score_data_quality


def _make_good_frames(n: int = 10) -> list[dict]:
    """Create n well-formed frames with regular timestamps."""
    return [
        {
            "session_time": i / 60.0,
            "speed": 200.0 + i,
            "throttle": 0.5,
            "brake": 0.0,
            "steer": 0.0,
            "g_lat": 1.0,
            "g_long": 0.5,
            "rpm": 8000.0 + i * 100,
            "gear": 3,
            "fuel_in_tank": 50.0 - i * 0.1,
            "fuel_remaining_laps": 5.0,
            "ers_store": 1_000_000.0,
            "ers_deploy_mode": 1,
            "lap_distance": float(i * 100),
            "tyre_temp_fl": 95.0,
            "tyre_temp_fr": 95.0,
            "tyre_temp_rl": 95.0,
            "tyre_temp_rr": 95.0,
        }
        for i in range(n)
    ]


class TestScoreDataQuality:
    def test_perfect_data(self) -> None:
        """完美数据 → overall 接近 1.0."""
        frames = _make_good_frames(20)
        report = score_data_quality(frames)
        assert isinstance(report, DataQualityReport)
        assert report.overall > 0.9
        assert report.label in ("excellent", "good")

    def test_empty_frames(self) -> None:
        """空帧列表 → overall = 1.0 (无数据无问题)."""
        report = score_data_quality([])
        assert report.overall >= 0.9
        assert report.n_frames == 0

    def test_packet_loss_detected(self) -> None:
        """有丢包 → packet_loss_score 降低."""
        frames = _make_good_frames(10)
        report = score_data_quality(frames, n_expected_packets=20)
        assert report.packet_loss_score < 0.6
        assert any("Packet loss" in i for i in report.issues)

    def test_missing_fields_detected(self) -> None:
        """缺少字段 → completeness_score 降低."""
        frames = _make_good_frames(10)
        for f in frames:
            del f["speed"]
        report = score_data_quality(frames, expected_fields=["speed", "throttle"])
        assert report.completeness_score < 1.0
        assert any("Missing" in i for i in report.issues)

    def test_irregular_timestamps(self) -> None:
        """不规则时间戳 → regularity_score 降低."""
        frames = _make_good_frames(10)
        # Make timestamps irregular
        for i, f in enumerate(frames):
            f["session_time"] = i * (0.01 if i % 2 == 0 else 0.05)
        report = score_data_quality(frames)
        assert report.regularity_score < 0.8

    def test_high_anomaly_rate(self) -> None:
        """高异常率 → anomaly_score 降低."""
        frames = _make_good_frames(20)
        report = score_data_quality(frames, n_anomalies=10)
        assert report.anomaly_score < 0.6
        assert any("Anomaly" in i for i in report.issues)

    def test_out_of_range_values(self) -> None:
        """超出范围的值 → range_compliance_score 降低."""
        frames = _make_good_frames(10)
        for f in frames:
            f["speed"] = 500.0  # > 350 max
        report = score_data_quality(frames)
        assert report.range_compliance_score < 1.0
        assert any("Out-of-range" in i for i in report.issues)

    def test_no_anomalies(self) -> None:
        """无异常 → anomaly_score = 1.0."""
        frames = _make_good_frames(10)
        report = score_data_quality(frames, n_anomalies=0)
        assert report.anomaly_score == 1.0

    def test_no_packet_loss(self) -> None:
        """无丢包 → packet_loss_score = 1.0."""
        frames = _make_good_frames(10)
        report = score_data_quality(frames, n_expected_packets=10)
        assert report.packet_loss_score == 1.0

    def test_overall_in_range(self) -> None:
        """overall 在 [0, 1] 范围内."""
        frames = _make_good_frames(10)
        report = score_data_quality(frames)
        assert 0.0 <= report.overall <= 1.0

    def test_label_thresholds(self) -> None:
        """label 与 overall 分数对应."""
        report = DataQualityReport(
            overall=0.95, label="excellent",
            packet_loss_score=1.0, completeness_score=1.0,
            regularity_score=1.0, anomaly_score=1.0,
            range_compliance_score=1.0,
        )
        assert report.label == "excellent"

    def test_issues_list_populated(self) -> None:
        """有问题时 issues 列表非空."""
        frames = _make_good_frames(10)
        report = score_data_quality(frames, n_expected_packets=20, n_anomalies=5)
        assert len(report.issues) > 0

    def test_issues_empty_for_good_data(self) -> None:
        """好数据 issues 列表为空."""
        frames = _make_good_frames(20)
        report = score_data_quality(frames)
        # Good data should have no issues (or very few)
        assert len(report.issues) == 0

    def test_custom_timestamp_field(self) -> None:
        """自定义时间戳字段."""
        frames = _make_good_frames(10)
        for f in frames:
            f["custom_time"] = f.pop("session_time")
        report = score_data_quality(frames, timestamp_field="custom_time")
        assert report.n_frames == 10

    def test_null_values_in_fields(self) -> None:
        """字段为 None 时 completeness 降低."""
        frames = _make_good_frames(10)
        for f in frames:
            f["speed"] = None
        report = score_data_quality(frames, expected_fields=["speed"])
        assert report.completeness_score == 0.0

    def test_n_frames_recorded(self) -> None:
        """n_frames 正确记录."""
        frames = _make_good_frames(15)
        report = score_data_quality(frames)
        assert report.n_frames == 15
