"""Unit tests for :mod:`f1opt.telemetry.replay`.

Covers the four public classes:

* :class:`TelemetryReplay` — frame emission, seeking, pause/resume,
  progress, duration, finished state, speed scaling, and the empty-frames
  edge case.
* :class:`SessionRecorder` — frame/lap/metadata recording, dict + JSON
  roundtrips, and summary statistics.
* :class:`SessionImporter` — CSV parsing, MoTeC-style text parsing, and
  format detection.
* :class:`SessionExporter` — CSV export and markdown summary report.

Timing-sensitive replay tests use a high ``speed`` multiplier (e.g. 100x)
so the wall-clock wait for frames to become "due" is sub-millisecond,
keeping the suite fast and deterministic on CI.
"""

from __future__ import annotations

import csv
import io
import json
import time

import pytest

from f1opt.telemetry.replay import (
    SessionExporter,
    SessionImporter,
    SessionRecorder,
    TelemetryReplay,
)

_DT = 1.0 / 60.0  # 60 Hz frame interval.


# --------------------------------------------------------------------------- #
# Frame / session builders
# --------------------------------------------------------------------------- #
def make_frame(
    t: float,
    *,
    speed: float = 200.0,
    throttle: float = 0.7,
    brake: float = 0.0,
    steer: float = 0.0,
    gear: int = 5,
    rpm: float = 9000.0,
    drs: int = 0,
    lap_number: int = 1,
) -> dict:
    """Build a synthetic unified frame dict at session_time ``t``."""
    return {
        "session_time": float(t),
        "speed": float(speed),
        "throttle": float(throttle),
        "brake": float(brake),
        "steer": float(steer),
        "gear": int(gear),
        "rpm": float(rpm),
        "drs": int(drs),
        "lap_number": int(lap_number),
    }


def spaced_frames(times: list[float], **overrides) -> list[dict]:
    """Build a list of frames at the given session_time values."""
    return [make_frame(t, **overrides) for t in times]


# --------------------------------------------------------------------------- #
# TelemetryReplay
# --------------------------------------------------------------------------- #
class TestTelemetryReplay:
    def test_next_frame_returns_frames_in_order(self) -> None:
        # 5 frames spaced 0.1s apart; speed=100 → all due in ~4ms wall.
        frames = spaced_frames([0.0, 0.1, 0.2, 0.3, 0.4])
        replay = TelemetryReplay(frames, speed=100.0)
        replay.start()
        time.sleep(0.005)  # plenty for all to be due at 100x
        drained: list[dict] = []
        while (f := replay.next_frame()) is not None:
            drained.append(f)
        assert len(drained) == len(frames)
        times = [f["session_time"] for f in drained]
        assert times == [0.0, 0.1, 0.2, 0.3, 0.4]

    def test_next_frame_returns_none_when_finished(self) -> None:
        frames = spaced_frames([0.0, 0.01])
        replay = TelemetryReplay(frames, speed=100.0)
        replay.start()
        time.sleep(0.005)
        # Drain all frames.
        while replay.next_frame() is not None:
            pass
        assert replay.next_frame() is None

    def test_progress_goes_from_zero_to_one(self) -> None:
        frames = spaced_frames([0.0, 0.1, 0.2, 0.3, 0.4])
        replay = TelemetryReplay(frames, speed=100.0)
        replay.start()
        # Immediately after start, progress should be near 0.
        assert replay.progress() == pytest.approx(0.0, abs=0.05)
        time.sleep(0.01)  # well past total duration at 100x
        while replay.next_frame() is not None:
            pass
        assert replay.progress() == pytest.approx(1.0)

    def test_current_session_time_increases(self) -> None:
        frames = spaced_frames([0.0, 1.0, 2.0])
        replay = TelemetryReplay(frames, speed=10.0)
        replay.start()
        t0 = replay.current_session_time()
        time.sleep(0.02)
        t1 = replay.current_session_time()
        assert t1 > t0

    def test_duration_computed_correctly(self) -> None:
        frames = spaced_frames([1.0, 2.5, 4.0])
        replay = TelemetryReplay(frames)
        assert replay.duration() == pytest.approx(3.0)

    def test_frames_remaining_decreases(self) -> None:
        frames = spaced_frames([0.0, 0.05, 0.10, 0.15, 0.20])
        replay = TelemetryReplay(frames, speed=200.0)
        replay.start()
        time.sleep(0.005)
        initial = replay.frames_remaining()
        assert initial == len(frames)
        replay.next_frame()
        assert replay.frames_remaining() == len(frames) - 1
        replay.next_frame()
        assert replay.frames_remaining() == len(frames) - 2

    def test_is_finished_true_after_all_frames(self) -> None:
        frames = spaced_frames([0.0, 0.01, 0.02])
        replay = TelemetryReplay(frames, speed=100.0)
        replay.start()
        time.sleep(0.005)
        assert not replay.is_finished()
        while replay.next_frame() is not None:
            pass
        assert replay.is_finished()

    def test_seek_jumps_to_position(self) -> None:
        frames = spaced_frames([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        replay = TelemetryReplay(frames, speed=1.0)
        replay.start()
        # Seek to 0.3 — next frame should be the one at 0.3 (or later).
        replay.seek(0.3)
        f = replay.next_frame()
        assert f is not None
        assert f["session_time"] == pytest.approx(0.3)
        # Subsequent frames should follow in order from 0.4.
        # At speed=1.0 the next frame (0.4) isn't immediately due, so we
        # only assert that the index advanced past 0.3.
        assert replay.frames_remaining() == len(frames) - 4  # 0.4, 0.5 left

    def test_pause_and_resume(self) -> None:
        frames = spaced_frames([0.0, 0.05, 0.10, 0.15, 0.20])
        replay = TelemetryReplay(frames, speed=10.0)
        replay.start()
        # Drain first frame (at t=0, always due).
        first = replay.next_frame()
        assert first is not None
        # Pause: while paused, next_frame returns None.
        replay.pause()
        time.sleep(0.01)
        assert replay.next_frame() is None
        # Resume: subsequent frames become due again.
        replay.resume()
        time.sleep(0.05)  # at 10x → 0.5s session time, all remaining due
        drained = []
        while (f := replay.next_frame()) is not None:
            drained.append(f)
        # Should have drained all the remaining frames (4 of 5).
        assert len(drained) == 4

    def test_speed_2_makes_replay_faster(self) -> None:
        # 5 frames spaced 0.1s apart; total duration 0.4s.
        # At 2x speed, all 5 should drain in ~0.2s wall (well under 0.4s).
        frames = spaced_frames([0.0, 0.1, 0.2, 0.3, 0.4])
        replay = TelemetryReplay(frames, speed=2.0)
        replay.start()
        start_wall = time.monotonic()
        drained: list[dict] = []
        while len(drained) < len(frames):
            f = replay.next_frame()
            if f is not None:
                drained.append(f)
                continue
            time.sleep(0.001)
        elapsed = time.monotonic() - start_wall
        assert len(drained) == len(frames)
        # At 2x speed, expected ~0.2s; assert comfortably under real-time 0.4s.
        assert elapsed < 0.35

    def test_empty_frames_immediately_finished(self) -> None:
        replay = TelemetryReplay([])
        assert replay.is_finished() is True
        assert replay.duration() == 0.0
        assert replay.frames_remaining() == 0
        # next_frame before start returns None.
        assert replay.next_frame() is None
        assert replay.progress() == 1.0


# --------------------------------------------------------------------------- #
# SessionRecorder
# --------------------------------------------------------------------------- #
class TestSessionRecorder:
    def test_record_frame_appends(self) -> None:
        rec = SessionRecorder("melbourne", {"wing": 5}, driver_name="VER")
        assert len(rec.frames) == 0
        rec.record_frame(make_frame(0.0))
        rec.record_frame(make_frame(0.1))
        assert len(rec.frames) == 2
        assert rec.frames[0]["session_time"] == 0.0
        assert rec.frames[1]["session_time"] == 0.1

    def test_record_frame_auto_timestamps_when_missing(self) -> None:
        rec = SessionRecorder("monza", {})
        # Frame without session_time → auto-stamped 0.0 (first frame).
        rec.record_frame({"speed": 100.0})
        assert rec.frames[0]["session_time"] == 0.0
        # Second frame without session_time → last + 1/60.
        rec.record_frame({"speed": 110.0})
        assert rec.frames[1]["session_time"] == pytest.approx(_DT)

    def test_record_lap_stores_laps(self) -> None:
        rec = SessionRecorder("silverstone", {})
        rec.record_lap(1, 95.123, [30.0, 32.0, 33.123])
        rec.record_lap(2, 94.5, [29.5, 31.8, 33.2])
        assert len(rec.laps) == 2
        assert rec.laps[0]["lap_number"] == 1
        assert rec.laps[0]["lap_time"] == pytest.approx(95.123)
        assert rec.laps[0]["sector_times"] == [30.0, 32.0, 33.123]
        assert rec.laps[1]["lap_number"] == 2

    def test_set_metadata_stores_metadata(self) -> None:
        rec = SessionRecorder("monaco", {})
        rec.set_metadata("weather", "dry")
        rec.set_metadata("compound", "medium")
        rec.set_metadata("air_temp", 24.5)
        assert rec.metadata["weather"] == "dry"
        assert rec.metadata["compound"] == "medium"
        assert rec.metadata["air_temp"] == 24.5

    def test_to_dict_has_required_keys(self) -> None:
        rec = SessionRecorder("spa", {"wing": 4}, driver_name="HAM")
        rec.record_frame(make_frame(0.0))
        rec.record_lap(1, 105.0, [35.0, 35.0, 35.0])
        rec.set_metadata("weather", "wet")
        d = rec.to_dict()
        required = {
            "track_id", "setup", "driver_name", "frames",
            "laps", "metadata", "duration", "frame_count", "lap_count",
        }
        assert required <= set(d)
        assert d["track_id"] == "spa"
        assert d["frame_count"] == 1
        assert d["lap_count"] == 1
        assert d["driver_name"] == "HAM"
        assert d["metadata"] == {"weather": "wet"}

    def test_to_dict_from_dict_roundtrip_preserves_frames(self) -> None:
        rec = SessionRecorder("suzuka", {"wing": 3}, driver_name="LEC")
        for t in [0.0, 0.1, 0.2, 0.3]:
            rec.record_frame(make_frame(t, speed=250.0 + t))
        rec.record_lap(1, 92.0, [30.0, 31.0, 31.0])
        rec.set_metadata("weather", "dry")

        rec2 = SessionRecorder.from_dict(rec.to_dict())
        assert rec2.track_id == "suzuka"
        assert rec2.driver_name == "LEC"
        assert rec2.setup == {"wing": 3}
        assert len(rec2.frames) == 4
        assert rec2.frames[0]["session_time"] == 0.0
        assert rec2.frames[3]["speed"] == pytest.approx(250.3)
        assert len(rec2.laps) == 1
        assert rec2.laps[0]["lap_time"] == pytest.approx(92.0)
        assert rec2.metadata["weather"] == "dry"

    def test_to_json_from_json_roundtrip(self) -> None:
        rec = SessionRecorder("zandvoort", {"wing": 6}, driver_name="SAI")
        rec.record_frame(make_frame(0.0))
        rec.record_frame(make_frame(0.5))
        rec.record_lap(1, 90.0, [30.0, 30.0, 30.0])
        rec.set_metadata("compound", "soft")

        json_str = rec.to_json()
        assert isinstance(json_str, str)
        # Must be valid JSON.
        parsed = json.loads(json_str)
        assert parsed["track_id"] == "zandvoort"

        rec2 = SessionRecorder.from_json(json_str)
        assert rec2.track_id == "zandvoort"
        assert rec2.driver_name == "SAI"
        assert len(rec2.frames) == 2
        assert rec2.frames[1]["session_time"] == pytest.approx(0.5)
        assert rec2.metadata["compound"] == "soft"
        assert len(rec2.laps) == 1

    def test_summary_has_required_keys(self) -> None:
        rec = SessionRecorder("cota", {"wing": 4})
        for t in [0.0, 0.1, 0.2]:
            rec.record_frame(make_frame(t))
        rec.record_lap(1, 100.0, [33.0, 33.0, 34.0])
        rec.record_lap(2, 98.0, [32.0, 33.0, 33.0])
        rec.record_lap(3, 99.0, [33.0, 33.0, 33.0])
        rec.set_metadata("weather", "dry")
        s = rec.summary()
        required = {
            "track_id", "lap_count", "best_lap_time", "avg_lap_time",
            "frame_count", "duration", "metadata",
        }
        assert required <= set(s)
        assert s["track_id"] == "cota"
        assert s["lap_count"] == 3
        assert s["frame_count"] == 3
        assert s["best_lap_time"] == pytest.approx(98.0)
        assert s["avg_lap_time"] == pytest.approx((100.0 + 98.0 + 99.0) / 3.0)
        assert s["duration"] == pytest.approx(0.2)
        assert s["metadata"] == {"weather": "dry"}

    def test_summary_with_no_laps_returns_zeros(self) -> None:
        rec = SessionRecorder("baku", {})
        rec.record_frame(make_frame(0.0))
        s = rec.summary()
        assert s["lap_count"] == 0
        assert s["best_lap_time"] == 0.0
        assert s["avg_lap_time"] == 0.0


# --------------------------------------------------------------------------- #
# SessionImporter
# --------------------------------------------------------------------------- #
class TestSessionImporter:
    def _sample_csv(self, n_rows: int = 3) -> str:
        rows = [
            "session_time,speed,throttle,brake,steer,gear,rpm,drs,lap_number",
        ]
        for i in range(n_rows):
            t = i * 0.1
            rows.append(
                f"{t},{200 + i},0.7,0.0,0.0,5,9000,0,1"
            )
        return "\n".join(rows) + "\n"

    def test_from_csv_parses_valid_csv(self) -> None:
        importer = SessionImporter()
        rec = importer.from_csv(self._sample_csv(3))
        assert isinstance(rec, SessionRecorder)
        assert len(rec.frames) == 3
        # Frame fields are coerced to the right types.
        f0 = rec.frames[0]
        assert f0["session_time"] == 0.0
        assert f0["speed"] == 200.0
        assert f0["gear"] == 5
        assert f0["drs"] == 0
        assert f0["lap_number"] == 1

    def test_from_csv_frame_count_matches_rows(self) -> None:
        importer = SessionImporter()
        for n in (1, 5, 10):
            rec = importer.from_csv(self._sample_csv(n))
            assert len(rec.frames) == n, f"expected {n} frames, got {len(rec.frames)}"

    def test_from_motec_telemetry_parses_text(self) -> None:
        motec_text = (
            "track=monza\n"
            "driver=VER\n"
            "[Frame]\n"
            "session_time=0.0\n"
            "speed=300.0\n"
            "throttle=0.9\n"
            "\n"
            "[Frame]\n"
            "session_time=0.1\n"
            "speed=305.0\n"
            "throttle=0.95\n"
        )
        importer = SessionImporter()
        rec = importer.from_motec_telemetry(motec_text)
        assert isinstance(rec, SessionRecorder)
        assert rec.track_id == "monza"
        assert rec.driver_name == "VER"
        assert len(rec.frames) == 2
        assert rec.frames[0]["session_time"] == 0.0
        assert rec.frames[0]["speed"] == 300.0
        assert rec.frames[1]["session_time"] == pytest.approx(0.1)
        assert rec.frames[1]["throttle"] == pytest.approx(0.95)

    def test_detect_format_csv(self) -> None:
        importer = SessionImporter()
        assert importer.detect_format(self._sample_csv(2)) == "csv"

    def test_detect_format_json(self) -> None:
        importer = SessionImporter()
        assert importer.detect_format('{"track_id": "x", "frames": []}') == "json"
        assert importer.detect_format("[1, 2, 3]") == "json"

    def test_detect_format_motec(self) -> None:
        importer = SessionImporter()
        motec = (
            "[Frame]\n"
            "session_time=0.0\n"
            "speed=200.0\n"
        )
        assert importer.detect_format(motec) == "motec"

    def test_detect_format_unknown(self) -> None:
        importer = SessionImporter()
        assert importer.detect_format("this is just garbage text") == "unknown"
        assert importer.detect_format("") == "unknown"
        assert importer.detect_format("   \n  \n") == "unknown"


# --------------------------------------------------------------------------- #
# SessionExporter
# --------------------------------------------------------------------------- #
class TestSessionExporter:
    def _populated_session(self) -> SessionRecorder:
        rec = SessionRecorder("monza", {"wing": 2}, driver_name="VER")
        for i, t in enumerate([0.0, 0.1, 0.2, 0.3]):
            rec.record_frame(make_frame(t, speed=250.0 + i))
        rec.record_lap(1, 82.5, [27.0, 27.5, 28.0])
        rec.record_lap(2, 81.8, [26.8, 27.4, 27.6])
        rec.set_metadata("weather", "dry")
        rec.set_metadata("compound", "soft")
        return rec

    def test_to_csv_produces_valid_csv_with_header(self) -> None:
        rec = self._populated_session()
        exporter = SessionExporter(rec)
        csv_str = exporter.to_csv()
        # Parse the CSV.
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) >= 1  # at least the header
        header = rows[0]
        # Canonical columns should appear in the header.
        for col in ("session_time", "speed", "throttle", "brake", "gear", "rpm"):
            assert col in header, f"missing column {col} in header"
        # Data rows (one per frame).
        assert len(rows) == 1 + len(rec.frames)
        # First data row's session_time should be 0.0.
        st_idx = header.index("session_time")
        assert float(rows[1][st_idx]) == pytest.approx(0.0)

    def test_to_csv_roundtrips_through_importer(self) -> None:
        rec = self._populated_session()
        exporter = SessionExporter(rec)
        csv_str = exporter.to_csv()
        # Re-import and verify frame count matches.
        importer = SessionImporter()
        rec2 = importer.from_csv(csv_str)
        assert len(rec2.frames) == len(rec.frames)
        # Spot-check a value.
        assert rec2.frames[0]["speed"] == pytest.approx(250.0)

    def test_to_summary_markdown_contains_track_and_lap_info(self) -> None:
        rec = self._populated_session()
        md = SessionExporter(rec).to_summary_markdown()
        assert isinstance(md, str)
        assert "# Telemetry Session Summary" in md
        assert "monza" in md
        assert "VER" in md
        # Lap section with at least one row.
        assert "## Laps" in md
        assert "82.500" in md or "82.5" in md
        # Metadata section.
        assert "## Metadata" in md
        assert "weather" in md
        assert "dry" in md

    def test_to_json_delegates_to_session(self) -> None:
        rec = self._populated_session()
        exporter = SessionExporter(rec)
        assert exporter.to_json() == rec.to_json()
        # And it roundtrips.
        rec2 = SessionRecorder.from_json(exporter.to_json())
        assert rec2.track_id == rec.track_id
        assert len(rec2.frames) == len(rec.frames)


# --------------------------------------------------------------------------- #
# Cross-class integration: importer → recorder → exporter → importer
# --------------------------------------------------------------------------- #
class TestRoundtripIntegration:
    def test_csv_roundtrip_preserves_frame_count(self) -> None:
        original = SessionRecorder("silverstone", {"wing": 5})
        for i in range(8):
            original.record_frame(make_frame(i * 0.05, speed=200.0 + i))
        csv_str = SessionExporter(original).to_csv()
        rec = SessionImporter().from_csv(csv_str)
        assert len(rec.frames) == len(original.frames)
        assert rec.frames[0]["session_time"] == pytest.approx(0.0)
        assert rec.frames[-1]["session_time"] == pytest.approx(7 * 0.05)

    def test_motec_roundtrip_preserves_frame_count(self) -> None:
        motec_lines = ["track=spa", "driver=HAM", ""]
        for i in range(5):
            motec_lines.append("[Frame]")
            motec_lines.append(f"session_time={i * 0.1:.3f}")
            motec_lines.append(f"speed={220.0 + i:.1f}")
            motec_lines.append("")
        rec = SessionImporter().from_motec_telemetry("\n".join(motec_lines))
        assert rec.track_id == "spa"
        assert rec.driver_name == "HAM"
        assert len(rec.frames) == 5
        assert rec.frames[2]["session_time"] == pytest.approx(0.2)
