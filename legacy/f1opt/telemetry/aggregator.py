"""Complete-lap aggregation to Parquet for clean training samples.

The :class:`LapAggregator` is an async subscriber compatible with
:class:`f1opt.telemetry.listener.TelemetryListener`. It tracks per-car lap
progress, accumulates per-frame telemetry statistics, and — when a lap
completes cleanly (no flashback, no invalid flag) — appends a summary row to
an in-memory buffer that can be flushed to a Parquet file.

A lap is considered ``clean`` iff no frame regression (flashback) was observed
between its start and end. Flashback is detected via the
:class:`~f1opt.telemetry.validation.FrameTracker` monitoring
``m_overallFrameIdentifier`` (which does NOT reset on flashback, unlike
``m_frameIdentifier``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1opt.observability.logging import get_logger

from .packets import PacketHeader
from .validation import FrameTracker, flag_severity, merge_flags

log = get_logger(__name__)

_SCHEMA = pa.schema([
    pa.field("session_uid", pa.uint64()),
    pa.field("car_index", pa.uint8()),
    pa.field("lap_number", pa.uint8()),
    pa.field("lap_time_ms", pa.uint32()),
    pa.field("overall_frame_start", pa.uint32()),
    pa.field("overall_frame_end", pa.uint32()),
    pa.field("session_time_start", pa.float32()),
    pa.field("session_time_end", pa.float32()),
    pa.field("num_samples", pa.uint32()),
    pa.field("avg_speed", pa.float32()),
    pa.field("avg_throttle", pa.float32()),
    pa.field("avg_brake", pa.float32()),
    pa.field("avg_ers_deploy", pa.float32()),
    pa.field("avg_active_aero_x", pa.float32()),  # Iter-191: F1 2026 X-Mode
    pa.field("avg_active_aero_z", pa.float32()),  # Iter-191: F1 2026 Z-Mode
    pa.field("max_tyre_wear", pa.float32()),
    pa.field("track_id", pa.int8()),
    pa.field("weather", pa.uint8()),
    pa.field("clean", pa.bool_()),
    pa.field("invalid_reason", pa.string()),
    pa.field("quality_flag", pa.string()),
])


@dataclass
class _LapState:
    """In-progress lap aggregation state for one car."""

    car_index: int
    lap_number: int
    overall_frame_start: int
    session_time_start: float
    speed_sum: float = 0.0
    throttle_sum: float = 0.0
    brake_sum: float = 0.0
    ers_deploy_sum: float = 0.0
    active_aero_x_sum: float = 0.0  # Iter-280: X-Mode (mode==1) 帧累计
    active_aero_z_sum: float = 0.0  # Iter-280: Z-Mode (mode==0) 帧累计
    car_telemetry2_count: int = 0  # Iter-280: Packet 16 样本数 (主动空力)
    car_status_count: int = 0  # Iter-255: CarStatus 样本数 (与 num_samples 不同率)
    max_tyre_wear: float = 0.0
    num_samples: int = 0
    dirty: bool = False  # set True if flashback detected mid-lap
    invalid_reason: str | None = None  # set if a field-level validation failed
    worst_flag: str = "OK"  # worst sample flag seen during this lap


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Iter-124: Coerce ``v`` to float; ``None`` / non-numeric → ``default``.

    Defensive helper for telemetry fields that may be ``None`` (explicit null
    in parsed packet) or missing. Prevents ``float(None)`` TypeError that
    would crash the aggregator subscriber loop.
    """
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _ensure_quality_flag_column(table: pa.Table) -> pa.Table:
    """Add a null ``quality_flag`` column to a table lacking it.

    Backward compat: Parquet files written before the ``quality_flag`` column
    existed are upgraded with a null column on read/concat so schema-aligned
    operations (concat, re-write) succeed. Nulls are defaulted to ``"OK"``
    by :meth:`LapAggregator.all_rows`.
    """
    if "quality_flag" in table.column_names:
        return table
    return table.append_column(
        pa.field("quality_flag", pa.string()),
        pa.nulls(table.num_rows, type=pa.string()),
    )


class LapAggregator:
    """Aggregate complete, clean laps from telemetry and persist to Parquet.

    Usage::

        agg = LapAggregator("data_store/laps.parquet")
        listener = TelemetryListener(port=20777)
        listener.subscribe(agg)
        await listener.start()
        # ... run ...
        await listener.stop()
        agg.flush()

    ``clean`` semantics: a lap is ``clean=True`` iff it is not flashback-dirty
    AND the worst per-frame ``quality_flag`` severity does not exceed
    ``clean_threshold``. The threshold is a flag name whose coarse severity
    (see :func:`f1opt.telemetry.validation.flag_severity`) sets the bar:

    - ``clean_threshold="SUSPECT"`` (default, severity 1): ``OK`` and any
      ``SUSPECT_*`` lap is clean; only ``INVALID`` laps are ``clean=False``.
    - ``clean_threshold="OK"`` (severity 0): only all-``OK`` laps are clean;
      any ``SUSPECT_*`` or ``INVALID`` lap is ``clean=False``.
    - ``clean_threshold="INVALID"`` (severity 2): even ``INVALID`` laps are
      clean (disables the flag-based gate; only flashback-dirty is excluded).

    Laps with flag severity **greater than** the threshold severity are
    ``clean=False``.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        clean_threshold: str = "SUSPECT",
    ) -> None:
        self.output_path = Path(output_path)
        self._clean_threshold = clean_threshold
        # (session_uid, car_index) -> in-progress lap state
        self._laps: dict[tuple[int, int], _LapState] = {}
        self._frame_tracker = FrameTracker()
        # session_uid -> last Session packet fields (for track_id / weather)
        self._last_session: dict[int, dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] = []
        # Iter-120: in-memory cache of the finalized Parquet table. Avoids O(N)
        # disk read on every flush() (old code did read+concat+rewrite = O(N)
        # read + O(N) write per flush). Now: first flush reads disk (O(N)), caches
        # the merged table; subsequent flushes concat with cache (O(batch)) and
        # write (O(N), unavoidable with Parquet single-file model). Eliminates the
        # O(N) read, halving the per-flush cost.
        self._cached_table: pa.Table | None = None

    # ------------------------------------------------------------------ #
    # Subscriber entry point
    # ------------------------------------------------------------------ #
    async def __call__(
        self, header: PacketHeader, parsed: dict[str, Any], raw: bytes
    ) -> None:
        """Subscriber entry point — called by TelemetryListener per packet."""
        sid = header.session_uid
        ofid = header.overall_frame_identifier

        # Track frame regressions (flashback). overall_frame_identifier is
        # monotonic across flashbacks, so a regression indicates reordering
        # or a flashback replay window.
        regressed, _, _ = self._frame_tracker.observe(sid, ofid)
        if regressed:
            # Mark all in-progress laps for this session as dirty.
            for key, state in self._laps.items():
                if key[0] == sid:
                    state.dirty = True

        # Field-level validation: a frame with an out-of-range field taints
        # all in-progress laps for this session (records invalid_reason).
        vinfo = parsed.get("__validation__")
        if isinstance(vinfo, dict) and vinfo.get("ok") is False:
            reason = vinfo.get("reason") or "unknown validation failure"
            for key, state in self._laps.items():
                if key[0] == sid and state.invalid_reason is None:
                    state.invalid_reason = reason

        # Field-level flag closure: merge the frame's ``_flag`` (stamped by
        # the listener via flag_sample) into the per-lap worst flag so the
        # lap row exposes ``quality_flag`` and ``clean`` reflects it. A
        # flagged sample is marked, not silently written.
        flag = parsed.get("_flag")
        if flag is not None:
            for key, state in self._laps.items():
                if key[0] == sid:
                    state.worst_flag = merge_flags([state.worst_flag, flag])

        if header.packet_id == 1:  # Session
            self._last_session[sid] = parsed
        elif header.packet_id == 2:  # LapData
            self._on_lap_data(header, parsed)
        elif header.packet_id == 6:  # CarTelemetry
            self._on_car_telemetry(header, parsed)
        elif header.packet_id == 7:  # CarStatus
            self._on_car_status(header, parsed)
        elif header.packet_id == 16:  # CarTelemetryData2 (Iter-280)
            self._on_car_telemetry_2(header, parsed)
        elif header.packet_id == 10:  # CarDamage
            self._on_car_damage(header, parsed)

    # ------------------------------------------------------------------ #
    # Per-packet handlers
    # ------------------------------------------------------------------ #
    def _on_lap_data(self, header: PacketHeader, parsed: dict[str, Any]) -> None:
        sid = header.session_uid
        cars = parsed.get("m_lapData") or []
        for car_idx, c in enumerate(cars):
            current_lap = c.get("m_currentLapNum", 0)
            last_lap_time = c.get("m_lastLapTimeInMS", 0)
            invalid = c.get("m_currentLapInvalid", 0)
            key = (sid, car_idx)

            prev = self._laps.get(key)
            if prev is None:
                # First observation for this car — initialize.
                self._laps[key] = _LapState(
                    car_index=car_idx,
                    lap_number=current_lap,
                    overall_frame_start=header.overall_frame_identifier,
                    session_time_start=header.session_time,
                )
            elif current_lap > prev.lap_number:
                # Lap transition: the previous lap has completed.
                if not prev.dirty and invalid == 0 and last_lap_time > 0:
                    row = self._build_row(header, prev, last_lap_time)
                    self._rows.append(row)
                # Start the new lap.
                self._laps[key] = _LapState(
                    car_index=car_idx,
                    lap_number=current_lap,
                    overall_frame_start=header.overall_frame_identifier,
                    session_time_start=header.session_time,
                )
            elif current_lap < prev.lap_number:
                # Lap number went backwards (flashback to an earlier lap).
                # Discard the in-progress state and start fresh.
                self._laps[key] = _LapState(
                    car_index=car_idx,
                    lap_number=current_lap,
                    overall_frame_start=header.overall_frame_identifier,
                    session_time_start=header.session_time,
                )
            # else: same lap — continue accumulating via the per-frame handlers.

    def _on_car_telemetry(
        self, header: PacketHeader, parsed: dict[str, Any]
    ) -> None:
        sid = header.session_uid
        cars = parsed.get("m_carTelemetryData") or []
        for car_idx, c in enumerate(cars):
            state = self._laps.get((sid, car_idx))
            if state is None:
                continue
            # Iter-124: _safe_float handles None / non-numeric defensively.
            state.speed_sum += _safe_float(c.get("m_speed"))
            state.throttle_sum += _safe_float(c.get("m_throttle"))
            state.brake_sum += _safe_float(c.get("m_brake"))
            state.num_samples += 1

    def _on_car_status(
        self, header: PacketHeader, parsed: dict[str, Any]
    ) -> None:
        sid = header.session_uid
        cars = parsed.get("m_carStatusData") or []
        for car_idx, c in enumerate(cars):
            state = self._laps.get((sid, car_idx))
            if state is None:
                continue
            # Iter-124: _safe_float handles None (explicit null) defensively.
            state.ers_deploy_sum += _safe_float(c.get("m_ersDeployedThisLap"))
            state.car_status_count += 1  # Iter-255: 独立计数, 与 num_samples 不同率

    def _on_car_telemetry_2(
        self, header: PacketHeader, parsed: dict[str, Any]
    ) -> None:
        """Iter-280: Packet 16 (CarTelemetryData2) — F1 2026 主动空力模式累计."""
        sid = header.session_uid
        cars = parsed.get("m_carTelemetryData2") or []
        for car_idx, c in enumerate(cars):
            state = self._laps.get((sid, car_idx))
            if state is None:
                continue
            # m_activeAeroMode: 0=Corner(Z-Mode) / 1=Straight(X-Mode)
            mode = int(_safe_float(c.get("m_activeAeroMode")))
            state.active_aero_x_sum += 1.0 if mode == 1 else 0.0
            state.active_aero_z_sum += 1.0 if mode == 0 else 0.0
            state.car_telemetry2_count += 1

    def _on_car_damage(
        self, header: PacketHeader, parsed: dict[str, Any]
    ) -> None:
        sid = header.session_uid
        cars = parsed.get("m_carDamageData") or []
        for car_idx, c in enumerate(cars):
            state = self._laps.get((sid, car_idx))
            if state is None:
                continue
            wears = c.get("m_tyresWear") or []
            if wears:
                # Iter-124: filter None / non-numeric wear values defensively.
                valid_wears = [_safe_float(w, default=-1.0) for w in wears]
                valid_wears = [w for w in valid_wears if w >= 0.0]
                if valid_wears:
                    state.max_tyre_wear = max(state.max_tyre_wear, max(valid_wears))

    # ------------------------------------------------------------------ #
    # Row construction + persistence
    # ------------------------------------------------------------------ #
    def _build_row(
        self, header: PacketHeader, state: _LapState, lap_time_ms: int
    ) -> dict[str, Any]:
        sid = header.session_uid
        sess = self._last_session.get(sid, {})
        n = max(state.num_samples, 1)  # Iter-124: guard against div-by-zero
        invalid_reason = state.invalid_reason
        worst_flag = state.worst_flag
        # clean = not flashback-dirty AND worst flag severity within threshold.
        flag_ok = flag_severity(worst_flag) <= flag_severity(self._clean_threshold)
        # Iter-124: num_samples=0 means no telemetry frames were observed for
        # this lap (e.g. packet loss, very short lap). Mark as invalid so it
        # doesn't pollute training data with zeroed averages.
        if state.num_samples == 0:
            invalid_reason = invalid_reason or "no_telemetry_samples"
            flag_ok = False
        # Iter-124: _safe_float for session fields (track_id / weather may be None).
        track_id = _safe_float(sess.get("m_trackId"), default=-1.0)
        weather = _safe_float(sess.get("m_weather"), default=0.0)
        return {
            "session_uid": sid,
            "car_index": state.car_index,
            "lap_number": state.lap_number,
            "lap_time_ms": lap_time_ms,
            "overall_frame_start": state.overall_frame_start,
            "overall_frame_end": header.overall_frame_identifier,
            "session_time_start": state.session_time_start,
            "session_time_end": header.session_time,
            "num_samples": state.num_samples,
            "avg_speed": state.speed_sum / n,
            "avg_throttle": state.throttle_sum / n,
            "avg_brake": state.brake_sum / n,
            # Iter-255: ERS/主动空力均来自 CarStatus (20Hz), 用 car_status_count 而非 num_samples
            "avg_ers_deploy": state.ers_deploy_sum / max(state.car_status_count, 1),
            "avg_active_aero_x": state.active_aero_x_sum / max(state.car_telemetry2_count, 1),  # Iter-280
            "avg_active_aero_z": state.active_aero_z_sum / max(state.car_telemetry2_count, 1),  # Iter-280
            "max_tyre_wear": state.max_tyre_wear,
            "track_id": int(track_id),
            "weather": int(weather),
            "clean": (not state.dirty) and flag_ok,
            "invalid_reason": invalid_reason,
            "quality_flag": worst_flag,
        }

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Return a copy of the aggregated (not-yet-flushed) rows."""
        return list(self._rows)

    def all_rows(self) -> list[dict[str, Any]]:
        """Return all lap rows: on-disk Parquet (flushed) + in-memory (pending).

        Deduplicated by (session_uid, car_index, lap_number, overall_frame_start)
        so a row pending in memory that also exists on disk (e.g. flushed twice)
        is not double-counted. Used by ``GET /api/samples``.

        Iter-120: uses ``_cached_table`` (in-memory Arrow table from last flush)
        when available, avoiding O(N) Parquet disk read+decompress. Falls back
        to disk read only on first call or after :meth:`close`.
        """
        merged: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        # Iter-120: prefer in-memory cache (O(N) but no disk I/O) over disk read.
        if self._cached_table is not None:
            table = _ensure_quality_flag_column(self._cached_table)
            for row in table.to_pylist():
                if row.get("quality_flag") is None:
                    row["quality_flag"] = "OK"
                merged.append(row)
                seen.add((
                    row.get("session_uid"),
                    row.get("car_index"),
                    row.get("lap_number"),
                    row.get("overall_frame_start"),
                ))
        elif self.output_path.exists():
            try:
                table = _ensure_quality_flag_column(pq.read_table(self.output_path))
                for row in table.to_pylist():
                    # Backward compat: old Parquet (no quality_flag) reads as
                    # null → treat as "OK".
                    if row.get("quality_flag") is None:
                        row["quality_flag"] = "OK"
                    merged.append(row)
                    seen.add((
                        row.get("session_uid"),
                        row.get("car_index"),
                        row.get("lap_number"),
                        row.get("overall_frame_start"),
                    ))
            except Exception:  # pragma: no cover - corrupt parquet
                log.warning("aggregator.parquet_read_failed", path=str(self.output_path))
        for row in self._rows:
            # Defensive: rows injected without quality_flag default to "OK".
            if row.get("quality_flag") is None:
                row["quality_flag"] = "OK"
            key = (
                row.get("session_uid"),
                row.get("car_index"),
                row.get("lap_number"),
                row.get("overall_frame_start"),
            )
            if key not in seen:
                merged.append(row)
        return merged

    def to_parquet_bytes(self) -> bytes:
        """Serialize all rows (disk + pending) to Parquet bytes (in-memory).

        Used by ``GET /api/samples/parquet`` download endpoint. Does not touch
        the on-disk file or the in-memory buffer.
        """
        import io

        rows = self.all_rows()
        if not rows:
            # Empty table with the correct schema.
            table = pa.table({f.name: [] for f in _SCHEMA}, schema=_SCHEMA)
        else:
            table = pa.Table.from_pylist(rows, schema=_SCHEMA)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return buf.getvalue()

    def flush(self) -> int:
        """Write all aggregated rows to the Parquet file. Returns count written.

        If the output file already exists, rows are appended. The in-memory row
        buffer is cleared after a successful write.

        Iter-120: uses ``_cached_table`` to avoid O(N) disk read on subsequent
        flushes. First flush reads the existing file (O(N)) and caches the
        merged Arrow table; subsequent flushes concat with the cache (O(batch))
        and write (O(N), unavoidable with Parquet single-file model). The disk
        read is eliminated, halving the per-flush cost.
        """
        if not self._rows:
            return 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        new_table = pa.Table.from_pylist(self._rows, schema=_SCHEMA)
        # Iter-120: use cached table if available; else read from disk (first flush).
        if self._cached_table is not None:
            merged = pa.concat_tables([self._cached_table, new_table])
        elif self.output_path.exists():
            existing = _ensure_quality_flag_column(pq.read_table(self.output_path))
            merged = pa.concat_tables([existing, new_table])
        else:
            merged = new_table
        pq.write_table(merged, self.output_path)
        self._cached_table = merged  # cache for next flush (avoids disk read)
        n = len(self._rows)
        self._rows.clear()
        return n

    def close(self) -> None:
        """Iter-120: release the in-memory cached table.

        Call on shutdown to free memory. After :meth:`close`, the next
        :meth:`flush` or :meth:`all_rows` will re-read from disk (O(N)).
        """
        self._cached_table = None

    def reset(self) -> None:
        """Clear all in-progress state and pending rows (does not delete files)."""
        self._laps.clear()
        self._rows.clear()
        self._cached_table = None
        self._frame_tracker.reset()
        self._last_session.clear()


__all__ = ["LapAggregator"]
