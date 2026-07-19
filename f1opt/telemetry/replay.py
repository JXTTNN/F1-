"""Telemetry replay + session import/export for the F1 setup optimizer.

Four public classes:

* :class:`TelemetryReplay` — replays a recorded list of unified telemetry
  frame dicts at real-time (``speed=1.0``) or accelerated / decelerated
  speed (``speed=2.0`` plays 2x faster; ``0.5`` plays at half speed). Frames
  are emitted by :meth:`TelemetryReplay.next_frame` only once their
  ``session_time`` is due on the wall clock, allowing a downstream consumer
  to simulate a live UDP feed from recorded data.

* :class:`SessionRecorder` — accumulates frames, completed laps and free-form
  metadata (weather, tyre compound, etc.) for a single session, with full
  JSON serialization / reconstruction roundtrips.

* :class:`SessionImporter` — parses external telemetry formats (CSV rows or
  simplified MoTeC-style ``key=value`` text exports) into a
  :class:`SessionRecorder`, with a small format sniffing helper.

* :class:`SessionExporter` — serializes a :class:`SessionRecorder` back out
  to CSV, JSON, or a human-readable markdown summary report.

The frames manipulated here follow the unified layout produced by
:class:`f1opt.telemetry.aligner.TelemetryAligner` (each dict carries at
least a ``session_time`` key) but the module is defensive: missing fields,
empty inputs and non-monotonic timestamps return sensible defaults rather
than raise.
"""

from __future__ import annotations

import bisect
import csv
import io
import json
import time
from typing import Any

#: Default 60 Hz frame interval used when a recorded frame lacks
#: ``session_time`` and no prior frame exists to increment from.
_DEFAULT_FRAME_DT: float = 1.0 / 60.0

#: Default placeholder track id / setup used when importing sessions whose
#: source format does not carry that information (CSV, MoTeC text).
_DEFAULT_TRACK_ID: str = "imported"
_DEFAULT_SETUP: dict[str, Any] = {}


def _frame_session_time(frame: dict[str, Any]) -> float:
    """Return ``frame['session_time']`` as a float (0.0 if missing)."""
    v = frame.get("session_time")
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# TelemetryReplay
# --------------------------------------------------------------------------- #
class TelemetryReplay:
    """Replay recorded telemetry frames at real-time or scaled speed.

    ``frames`` is a list of unified frame dicts (each carrying at least
    ``session_time``). The list is sorted by ``session_time`` on construction
    so callers do not have to pre-sort.

    ``speed`` scales the wall-clock-to-session-time mapping: ``1.0`` is
    real-time, ``2.0`` plays twice as fast (1 wall second = 2 session
    seconds), ``0.5`` plays at half speed. ``speed <= 0`` is treated as
    ``1.0`` (real-time) to avoid a division-by-zero / negative-time footgun.

    Usage::

        replay = TelemetryReplay(frames, speed=2.0)
        replay.start()
        while (frame := replay.next_frame()) is not None:
            consume(frame)
    """

    def __init__(self, frames: list[dict], speed: float = 1.0) -> None:
        self._frames: list[dict] = sorted(
            (dict(f) for f in frames), key=_frame_session_time
        )
        # Defensive: non-positive speed would freeze or reverse time.
        self._speed: float = float(speed) if speed and speed > 0 else 1.0
        self._index: int = 0
        self._started: bool = False
        self._paused: bool = False
        # Wall-clock anchor: when replay (or the current segment after a
        # seek) began. ``current_session_time`` is computed as
        # ``_start_session_time + (effective_now - _start_wall - _pause_offset) * speed``.
        self._start_wall: float = 0.0
        self._start_session_time: float = 0.0
        self._pause_offset: float = 0.0
        self._pause_start: float | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Begin replay. Records the wall-clock anchor at the first frame's
        ``session_time``. Idempotent: calling ``start()`` twice resets the
        replay position to the beginning.
        """
        self._started = True
        self._index = 0
        self._start_wall = time.monotonic()
        self._start_session_time = self._first_session_time()
        self._pause_offset = 0.0
        self._pause_start = None
        self._paused = False

    def pause(self) -> None:
        """Pause replay. While paused, :meth:`next_frame` returns ``None``
        and :meth:`current_session_time` is frozen."""
        if not self._started or self._paused:
            return
        self._paused = True
        self._pause_start = time.monotonic()

    def resume(self) -> None:
        """Resume a paused replay."""
        if not self._paused:
            return
        assert self._pause_start is not None
        self._pause_offset += time.monotonic() - self._pause_start
        self._pause_start = None
        self._paused = False

    def seek(self, session_time: float) -> None:
        """Jump to ``session_time`` (seconds). The next :meth:`next_frame`
        call will return the earliest frame whose ``session_time`` is
        ``>= session_time``. Resets the wall-clock anchor so the seeked
        position is reached immediately."""
        target = float(session_time)
        # Re-anchor: from now on, ``target`` is "now" in session time.
        self._start_session_time = target
        self._start_wall = time.monotonic()
        self._pause_offset = 0.0
        if self._paused:
            self._pause_start = time.monotonic()
        # Advance index to the first frame at/after target.
        times = [_frame_session_time(f) for f in self._frames]
        self._index = bisect.bisect_left(times, target)
        if not self._started:
            # Seek before start: mark as started so next_frame can emit.
            self._started = True

    # ------------------------------------------------------------------ #
    # Frame emission
    # ------------------------------------------------------------------ #
    def next_frame(self) -> dict | None:
        """Return the next frame whose ``session_time`` is due on the wall
        clock, or ``None`` if no frame is currently due (or replay is
        paused / finished / not started). Frames are returned in
        ``session_time`` order; each frame is emitted exactly once."""
        if not self._started or self._paused:
            return None
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        if _frame_session_time(frame) <= self.current_session_time():
            self._index += 1
            return frame
        return None

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #
    def progress(self) -> float:
        """Fraction of replay completed in ``[0, 1]``.

        For non-empty replays: ``(current_session_time - first) / duration``.
        For zero-duration replays (single frame or empty): ``1.0`` once
        finished, else ``0.0``.
        """
        if not self._frames:
            return 1.0
        dur = self.duration()
        if dur <= 0.0:
            return 1.0 if self.is_finished() else 0.0
        frac = (self.current_session_time() - self._first_session_time()) / dur
        return max(0.0, min(1.0, frac))

    def current_session_time(self) -> float:
        """Current replay position in session seconds."""
        if not self._started:
            return self._first_session_time()
        # While paused, freeze at the pause-start wall moment.
        now = (
            self._pause_start
            if self._paused and self._pause_start is not None
            else time.monotonic()
        )
        elapsed_wall = now - self._start_wall - self._pause_offset
        return self._start_session_time + elapsed_wall * self._speed

    def duration(self) -> float:
        """Total session duration: last frame ``session_time`` minus first.

        Returns ``0.0`` for empty or single-frame inputs.
        """
        if len(self._frames) < 2:
            return 0.0
        return _frame_session_time(self._frames[-1]) - _frame_session_time(
            self._frames[0]
        )

    def frames_remaining(self) -> int:
        """Number of frames not yet emitted."""
        return max(0, len(self._frames) - self._index)

    def is_finished(self) -> bool:
        """True when all frames have been emitted (or there were none)."""
        if not self._frames:
            return True
        if not self._started:
            return False
        return self._index >= len(self._frames)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _first_session_time(self) -> float:
        if not self._frames:
            return 0.0
        return _frame_session_time(self._frames[0])


# --------------------------------------------------------------------------- #
# SessionRecorder
# --------------------------------------------------------------------------- #
class SessionRecorder:
    """Record a live telemetry session for later replay / export.

    Accumulates per-frame telemetry, completed laps and free-form metadata
    (weather, tyre compound, etc.) for a single ``(track_id, setup, driver)``
    triple. Fully JSON-serializable via :meth:`to_json` /
    :meth:`from_json`.
    """

    def __init__(
        self,
        track_id: str,
        setup: dict,
        driver_name: str = "",
    ) -> None:
        self.track_id: str = track_id
        self.setup: dict[str, Any] = dict(setup)
        self.driver_name: str = driver_name
        self._frames: list[dict[str, Any]] = []
        self._laps: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_frame(self, frame: dict) -> None:
        """Append a frame. If ``session_time`` is missing, it is auto-stamped
        as ``last_session_time + 1/60`` (or ``0.0`` for the first frame)."""
        f = dict(frame)
        if "session_time" not in f or f["session_time"] is None:
            if self._frames:
                last = _frame_session_time(self._frames[-1])
                f["session_time"] = last + _DEFAULT_FRAME_DT
            else:
                f["session_time"] = 0.0
        else:
            # Coerce to float for downstream consistency.
            f["session_time"] = _frame_session_time(f)
        self._frames.append(f)

    def record_lap(
        self,
        lap_number: int,
        lap_time: float,
        sector_times: list[float],
    ) -> None:
        """Record a completed lap (lap number, total time, sector splits)."""
        self._laps.append(
            {
                "lap_number": int(lap_number),
                "lap_time": float(lap_time),
                "sector_times": [float(s) for s in sector_times],
            }
        )

    def set_metadata(self, key: str, value: Any) -> None:
        """Record a free-form metadata field (weather, compound, etc.)."""
        self._metadata[key] = value

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def frames(self) -> list[dict[str, Any]]:
        return self._frames

    @property
    def laps(self) -> list[dict[str, Any]]:
        return self._laps

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def duration(self) -> float:
        """Last-frame session_time minus first-frame session_time (0 if <2)."""
        if len(self._frames) < 2:
            return 0.0
        return _frame_session_time(self._frames[-1]) - _frame_session_time(
            self._frames[0]
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the full session to a plain dict."""
        return {
            "track_id": self.track_id,
            "setup": self.setup,
            "driver_name": self.driver_name,
            "frames": self._frames,
            "laps": self._laps,
            "metadata": self._metadata,
            "duration": self.duration(),
            "frame_count": len(self._frames),
            "lap_count": len(self._laps),
        }

    def to_json(self) -> str:
        """JSON serialization of :meth:`to_dict`."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> SessionRecorder:
        """Reconstruct a :class:`SessionRecorder` from :meth:`to_dict` output."""
        rec = cls(
            track_id=str(data.get("track_id", _DEFAULT_TRACK_ID)),
            setup=dict(data.get("setup", {})),
            driver_name=str(data.get("driver_name", "")),
        )
        rec._frames = [dict(f) for f in data.get("frames", [])]
        rec._laps = [dict(lap) for lap in data.get("laps", [])]
        rec._metadata = dict(data.get("metadata", {}))
        return rec

    @classmethod
    def from_json(cls, json_str: str) -> SessionRecorder:
        """Reconstruct from a JSON string produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """Return a compact summary dict (best/avg lap, frame count, ...)."""
        lap_times = [
            float(lap["lap_time"])
            for lap in self._laps
            if lap.get("lap_time")
        ]
        if lap_times:
            best = min(lap_times)
            avg = sum(lap_times) / len(lap_times)
        else:
            best = 0.0
            avg = 0.0
        return {
            "track_id": self.track_id,
            "lap_count": len(self._laps),
            "best_lap_time": best,
            "avg_lap_time": avg,
            "frame_count": len(self._frames),
            "duration": self.duration(),
            "metadata": dict(self._metadata),
        }


# --------------------------------------------------------------------------- #
# SessionImporter
# --------------------------------------------------------------------------- #
class SessionImporter:
    """Import telemetry from external formats (CSV, MoTeC-style text, JSON)."""

    #: CSV columns expected by :meth:`from_csv` (in canonical order).
    CSV_COLUMNS: tuple[str, ...] = (
        "session_time",
        "speed",
        "throttle",
        "brake",
        "steer",
        "gear",
        "rpm",
        "drs",
        "lap_number",
    )

    def __init__(self) -> None:
        # No state — importers are stateless helpers.
        pass

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #
    def from_csv(self, csv_str: str) -> SessionRecorder:
        """Parse CSV with the canonical telemetry columns.

        The first row must be a header naming a subset of
        :data:`CSV_COLUMNS`. Each subsequent row becomes one frame. Missing
        columns default to ``0``; unrecognized columns are preserved on the
        frame dict (so round-tripping a richer CSV does not lose data).
        """
        rec = SessionRecorder(
            track_id=_DEFAULT_TRACK_ID, setup=dict(_DEFAULT_SETUP)
        )
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        if not rows:
            return rec
        header = [h.strip() for h in rows[0]]
        col_index: dict[str, int] = {name: i for i, name in enumerate(header)}
        for raw_row in rows[1:]:
            if not raw_row or all(not str(c).strip() for c in raw_row):
                continue
            frame: dict[str, Any] = {}
            for col in self.CSV_COLUMNS:
                if col in col_index and col_index[col] < len(raw_row):
                    cell = raw_row[col_index[col]].strip()
                    if cell == "":
                        continue
                    frame[col] = self._coerce_csv_cell(col, cell)
            # Preserve any extra columns verbatim.
            for name, i in col_index.items():
                if name in self.CSV_COLUMNS:
                    continue
                if i < len(raw_row):
                    frame[name] = raw_row[i].strip()
            if "session_time" not in frame:
                # Auto-stamp via recorder (incremental 1/60).
                pass
            rec.record_frame(frame)
        return rec

    @staticmethod
    def _coerce_csv_cell(col: str, cell: str) -> Any:
        """Coerce a CSV cell to the appropriate python type per column."""
        if col in ("gear", "drs", "lap_number"):
            try:
                return int(float(cell))
            except ValueError:
                return 0
        try:
            return float(cell)
        except ValueError:
            return cell

    # ------------------------------------------------------------------ #
    # MoTeC-style text export
    # ------------------------------------------------------------------ #
    def from_motec_telemetry(self, motec_text: str) -> SessionRecorder:
        """Parse a simplified MoTeC-style text export.

        Each frame is a block of ``key=value`` lines. Frames are separated
        by blank lines or a ``[Frame]`` / ``[Sample]`` header line. Values
        are coerced to ``float`` when numeric, else kept as strings.

        Example input::

            [Frame]
            session_time=0.0
            speed=200.0
            throttle=0.5

            [Frame]
            session_time=0.0167
            speed=201.0
            throttle=0.55
        """
        rec = SessionRecorder(
            track_id=_DEFAULT_TRACK_ID, setup=dict(_DEFAULT_SETUP)
        )
        # Optional track / driver / setup metadata parsed from header lines
        # of the form ``track=...`` / ``driver=...`` appearing before the
        # first frame block.
        current: dict[str, Any] = {}
        saw_frame = False
        for raw_line in motec_text.splitlines():
            line = raw_line.strip()
            if not line:
                if current:
                    rec.record_frame(current)
                    current = {}
                continue
            if line.startswith("[") and line.endswith("]"):
                tag = line[1:-1].strip().lower()
                if tag in ("frame", "sample", "lap"):
                    if current:
                        rec.record_frame(current)
                        current = {}
                    saw_frame = True
                # Unknown section header — ignore (treated as separator).
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not saw_frame:
                # Pre-frame header metadata.
                if key.lower() == "track":
                    rec.track_id = value
                elif key.lower() == "driver":
                    rec.driver_name = value
                elif key.lower() == "setup":
                    # Treat as a free-form setup string.
                    rec.setup["raw"] = value
                # else: ignore unknown header lines.
                continue
            current[key] = self._coerce_value(value)
        if current:
            rec.record_frame(current)
        return rec

    @staticmethod
    def _coerce_value(value: str) -> Any:
        """Coerce a MoTeC ``key=value`` value to float when numeric."""
        try:
            return float(value)
        except ValueError:
            return value

    # ------------------------------------------------------------------ #
    # Format detection
    # ------------------------------------------------------------------ #
    def detect_format(self, content: str) -> str:
        """Return ``"csv"`` / ``"motec"`` / ``"json"`` / ``"unknown"``.

        Heuristics:

        * ``"json"`` — content parses as JSON and is an object or array.
        * ``"csv"`` — first non-empty line contains a comma, and the
          header row matches at least two of :data:`CSV_COLUMNS`.
        * ``"motec"`` — content contains ``key=value`` lines (with ``=``
          and a bare identifier before it).
        * ``"unknown"`` — otherwise.
        """
        s = content.strip()
        if not s:
            return "unknown"
        # JSON?
        if s[0] in "{[":
            try:
                obj = json.loads(s)
            except (json.JSONDecodeError, ValueError):
                obj = None
            if isinstance(obj, (dict, list)):
                return "json"
        # CSV? — first non-empty line must contain a comma and match >=2 cols.
        first_line = s.splitlines()[0].strip()
        if "," in first_line:
            header_tokens = {t.strip().lower() for t in first_line.split(",")}
            known = {c.lower() for c in self.CSV_COLUMNS}
            if len(header_tokens & known) >= 2:
                return "csv"
        # MoTeC? — at least one ``identifier=value`` line.
        for line in s.splitlines():
            stripped = line.strip()
            if "=" in stripped:
                key, _, val = stripped.partition("=")
                key = key.strip()
                # Identifier-like key (letters/digits/underscore), and the
                # line is not a [section] header (already handled above).
                if key and all(
                    ch.isalnum() or ch in "_." for ch in key
                ) and val != "":
                    return "motec"
        return "unknown"


# --------------------------------------------------------------------------- #
# SessionExporter
# --------------------------------------------------------------------------- #
class SessionExporter:
    """Export a :class:`SessionRecorder` to CSV, JSON, or markdown summary."""

    def __init__(self, session: SessionRecorder) -> None:
        self.session = session

    # ------------------------------------------------------------------ #
    def to_csv(self) -> str:
        """CSV with all frame fields. The header is the union of keys across
        all frames (``session_time`` always first); missing fields are
        written as empty cells."""
        frames = self.session.frames
        if not frames:
            # Still emit a header so consumers can parse it.
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(list(SessionImporter.CSV_COLUMNS))
            return buf.getvalue()
        # Preserve canonical column order, then append any extra keys.
        canonical = list(SessionImporter.CSV_COLUMNS)
        extras: list[str] = []
        seen: set[str] = set(canonical)
        for f in frames:
            for k in f:
                if k not in seen:
                    extras.append(k)
                    seen.add(k)
        columns = canonical + extras
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        for f in frames:
            row = []
            for col in columns:
                v = f.get(col)
                if v is None:
                    row.append("")
                else:
                    row.append(v)
            writer.writerow(row)
        return buf.getvalue()

    def to_json(self) -> str:
        """Full JSON serialization (delegates to :meth:`SessionRecorder.to_json`)."""
        return self.session.to_json()

    def to_summary_markdown(self) -> str:
        """Markdown summary report: track, driver, lap stats, frame count,
        and a metadata table."""
        s = self.session.summary()
        lines: list[str] = []
        lines.append("# Telemetry Session Summary")
        lines.append("")
        lines.append(f"- **Track:** `{self.session.track_id}`")
        driver = self.session.driver_name or "—"
        lines.append(f"- **Driver:** {driver}")
        lines.append(f"- **Frames:** {s['frame_count']}")
        lines.append(f"- **Laps:** {s['lap_count']}")
        lines.append(f"- **Duration (s):** {s['duration']:.3f}")
        best = s["best_lap_time"]
        avg = s["avg_lap_time"]
        lines.append(
            f"- **Best lap time (s):** {best:.3f}" if best else "- **Best lap time:** —"
        )
        lines.append(
            f"- **Avg lap time (s):** {avg:.3f}" if avg else "- **Avg lap time:** —"
        )
        lines.append("")
        # Per-lap table.
        if self.session.laps:
            lines.append("## Laps")
            lines.append("")
            lines.append("| Lap | Time (s) | Sectors (s) |")
            lines.append("| --- | --- | --- |")
            for lap in self.session.laps:
                sectors = ", ".join(
                    f"{float(x):.3f}" for x in lap.get("sector_times", [])
                )
                lines.append(
                    f"| {lap.get('lap_number', '?')} "
                    f"| {float(lap.get('lap_time', 0.0)):.3f} "
                    f"| {sectors} |"
                )
            lines.append("")
        # Metadata table.
        if s["metadata"]:
            lines.append("## Metadata")
            lines.append("")
            lines.append("| Key | Value |")
            lines.append("| --- | --- |")
            for k, v in s["metadata"].items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        return "\n".join(lines)


__all__ = [
    "TelemetryReplay",
    "SessionRecorder",
    "SessionImporter",
    "SessionExporter",
]
