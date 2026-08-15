"""Multi-source alignment + unified 60Hz interpolated timeseries.

The :class:`TelemetryAligner` consumes parsed F1 25 packets (per ``packet_id``)
and produces a unified 60Hz timeseries that aligns Motion, CarTelemetry,
CarStatus, LapData and CarDamage onto a single regular time grid.

Iter-131 extension: the aligner is no longer restricted to the player car.
Per-car buffers are lazily instantiated for any car index observed via
:meth:`on_packet` (``car_index=...``) or :meth:`on_packet_all_cars`, and every
query method (``sample_60hz`` / ``latest_unified_frame``) accepts an optional
``car_index`` to retrieve a non-player car's unified timeseries. The default
behaviour is unchanged: when ``car_index`` is ``None`` the aligner uses the most
recent ``header.player_car_index``, preserving full backward compatibility with
single-car (player-only) callers.

Each source packet type contributes a bounded :class:`collections.deque` of
``(session_time, fields)`` samples per ``(packet_id, car_index)`` pair; deques
are created on first observation so a 22-car grid does not pay 22x memory up
front.

* :meth:`sample_60hz` linear-interpolates every field onto the requested grid
  (nearest-neighbour for integer fields); it is vectorised with numpy so that
  ~6000 frames x ~30 fields complete well under 50ms.
* :meth:`latest_unified_frame` returns the unified frame at the most recent
  available ``session_time`` across sources (real-time UI use).
* :meth:`available_car_indices` returns the set of car indices that have at
  least one ingested sample (Iter-131).
* :meth:`_interp_float` / :meth:`_nearest_int` are the canonical single-value
  helpers (bisect-based) used by :meth:`latest_unified_frame`.

Out-of-order arrival handling: buffers are kept **unsorted** (plain deque
appends, ``on_packet`` is O(1)) and **sorted on query** inside
:meth:`sample_60hz` / :meth:`latest_unified_frame`. This makes the ingest path
cheap and makes the aligner robust to late-arriving / reordered samples without
any insort cost on the hot recv path. The per-query sort is amortised over the
whole grid (or a single frame), so it is negligible relative to interpolation.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from typing import Any

import numpy as np

from .packets import NUM_CARS, PacketHeader

# Packet ids that carry per-frame player-car timeseries.
_MOTION = 0
_LAPDATA = 2
_CARTELEMETRY = 6
_CARSTATUS = 7
_CARDAMAGE = 10

#: Unified frame keys in canonical output order (``session_time`` first).
#:
#: Iter-172: 从 34 字段扩展到 47 字段, 覆盖 F1 25/26 全部关键遥测:
#: - 制动温度 (4 轮) / 离合器 / DRS 激活 / 转速灯
#: - ERS 当前圈回收/部署
#: - 进站状态 / 车手状态 / 结果状态 / 扇区时间 / 罚时
#: - 轮胎配方 (visual + actual) / 轮胎内温 (4 轮)
UNIFIED_KEYS: tuple[str, ...] = (
    "session_time",
    # CarTelemetry
    "speed", "throttle", "brake", "steer", "gear", "rpm",
    "clutch",                                    # Iter-172: m_clutch
    "drs_active",                                # Iter-172: m_drs (激活状态 vs 允许)
    "rev_lights",                                # Iter-172: m_revLightsPercent
    "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
    "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",  # Iter-172
    "tyre_inner_temp_fl", "tyre_inner_temp_fr",  # Iter-172: m_tyresInnerTemperature
    "tyre_inner_temp_rl", "tyre_inner_temp_rr",
    # Motion (g-force, world position/velocity, orientation)
    "g_lat", "g_long", "g_vert",
    "world_x", "world_y", "world_z",
    "velocity_x", "velocity_y", "velocity_z",
    "yaw", "pitch", "roll",
    # CarStatus (ERS / DRS / fuel / 轮胎配方)
    "ers_store", "ers_deploy_mode", "drs_allowed",
    "ers_harvested_this_lap",                    # Iter-172: m_ersHarvestedThisLapMGUK
    "ers_deployed_this_lap",                     # Iter-172: m_ersDeployedThisLap
    "fuel_in_tank", "fuel_remaining_laps",
    "tyre_compound",                             # Iter-172: m_visualTyreCompound
    "actual_tyre_compound",                      # Iter-172: m_actualTyreCompound
    "active_aero_x", "active_aero_z",            # Iter-191: F1 2026 主动空力 (X/Z)
    # LapData (圈速 / 扇区 / 状态 / 罚时)
    "lap_time", "lap_distance",
    "sector1_time", "sector2_time", "sector3_time",  # Iter-172: m_sector[123]TimeInMS
    "pit_status",                                # Iter-172: m_pitStatus
    "driver_status",                             # Iter-172: m_driverStatus
    "result_status",                             # Iter-172: m_resultStatus
    "penalties",                                 # Iter-172: m_penalties
    # CarDamage (tyre wear)
    "tyre_wear_fl", "tyre_wear_fr", "tyre_wear_rl", "tyre_wear_rr",
)

#: Integer (nearest-neighbour) fields — never linearly interpolated.
_INT_KEYS: frozenset[str] = frozenset(
    {
        "gear", "rpm", "ers_deploy_mode", "drs_allowed",
        # Iter-172: 新增整数字段
        "drs_active", "rev_lights",
        "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",
        "tyre_inner_temp_fl", "tyre_inner_temp_fr",
        "tyre_inner_temp_rl", "tyre_inner_temp_rr",
        "tyre_compound", "actual_tyre_compound",
        "pit_status", "driver_status", "result_status", "penalties",
    }
)

# A source field reference is either a plain key in the player-car dict, or a
# ``(array_key, index)`` tuple for array-valued fields (tyre temps / wear).
Source = str | tuple[str, int]
_FieldSpec = tuple[str, Source, bool]

#: Per-source field specs: packet_id -> tuple of (unified_key, source, is_int).
#:
#: Iter-172: 扩展全部 47 字段的 source mapping, 覆盖 F1 25/26 关键遥测:
#: - CarTelemetry: +clutch, +drs_active(m_drs), +rev_lights,
#:                 +brake_temp[4](m_brakesTemperature), +tyre_inner_temp[4]
#: - CarStatus: +ers_harvested_this_lap, +ers_deployed_this_lap,
#:              +tyre_compound, +actual_tyre_compound
#: - LapData: +sector[123]_time, +pit_status, +driver_status,
#:            +result_status, +penalties
_SOURCE_SPECS: dict[int, tuple[_FieldSpec, ...]] = {
    _MOTION: (
        ("g_lat", "m_gForceLateral", False),
        ("g_long", "m_gForceLongitudinal", False),
        ("g_vert", "m_gForceVertical", False),
        ("world_x", "m_worldPositionX", False),
        ("world_y", "m_worldPositionY", False),
        ("world_z", "m_worldPositionZ", False),
        ("velocity_x", "m_worldVelocityX", False),
        ("velocity_y", "m_worldVelocityY", False),
        ("velocity_z", "m_worldVelocityZ", False),
        ("yaw", "m_yaw", False),
        ("pitch", "m_pitch", False),
        ("roll", "m_roll", False),
    ),
    _LAPDATA: (
        ("lap_time", "m_currentLapTimeInMS", False),
        ("lap_distance", "m_lapDistance", False),
        # Iter-172: 扇区时间 (ms → s 由 _extract_value 调用方决定, 这里保留原值)
        ("sector1_time", "m_sector1TimeInMS", False),
        ("sector2_time", "m_sector2TimeInMS", False),
        ("sector3_time", "m_sector3TimeInMS", False),
        ("pit_status", "m_pitStatus", True),
        ("driver_status", "m_driverStatus", True),
        ("result_status", "m_resultStatus", True),
        ("penalties", "m_penalties", True),
    ),
    _CARTELEMETRY: (
        ("speed", "m_speed", False),
        ("throttle", "m_throttle", False),
        ("brake", "m_brake", False),
        ("steer", "m_steer", False),
        ("gear", "m_gear", True),
        ("rpm", "m_engineRPM", True),
        # Iter-172: 新增 CarTelemetry 字段
        ("clutch", "m_clutch", False),
        ("drs_active", "m_drs", True),
        ("rev_lights", "m_revLightsPercent", True),
        ("tyre_temp_fl", ("m_tyresSurfaceTemperature", 0), False),
        ("tyre_temp_fr", ("m_tyresSurfaceTemperature", 1), False),
        ("tyre_temp_rl", ("m_tyresSurfaceTemperature", 2), False),
        ("tyre_temp_rr", ("m_tyresSurfaceTemperature", 3), False),
        ("brake_temp_fl", ("m_brakesTemperature", 0), True),
        ("brake_temp_fr", ("m_brakesTemperature", 1), True),
        ("brake_temp_rl", ("m_brakesTemperature", 2), True),
        ("brake_temp_rr", ("m_brakesTemperature", 3), True),
        ("tyre_inner_temp_fl", ("m_tyresInnerTemperature", 0), True),
        ("tyre_inner_temp_fr", ("m_tyresInnerTemperature", 1), True),
        ("tyre_inner_temp_rl", ("m_tyresInnerTemperature", 2), True),
        ("tyre_inner_temp_rr", ("m_tyresInnerTemperature", 3), True),
    ),
    _CARSTATUS: (
        ("ers_store", "m_ersStoreEnergy", False),
        ("ers_deploy_mode", "m_ersDeployMode", True),
        ("drs_allowed", "m_drsAllowed", True),
        ("fuel_in_tank", "m_fuelInTank", False),
        ("fuel_remaining_laps", "m_fuelRemainingLaps", False),
        # Iter-172: 新增 CarStatus 字段
        ("ers_harvested_this_lap", "m_ersHarvestedThisLapMGUK", False),
        ("ers_deployed_this_lap", "m_ersDeployedThisLap", False),
        ("tyre_compound", "m_visualTyreCompound", True),
        ("actual_tyre_compound", "m_actualTyreCompound", True),
        # Iter-191: F1 2026 主动空力 (X=低阻/Z=高下压力 位置, 连续浮点)
        ("active_aero_x", "m_activeAeroX", False),
        ("active_aero_z", "m_activeAeroZ", False),
    ),
    _CARDAMAGE: (
        ("tyre_wear_fl", ("m_tyresWear", 0), False),
        ("tyre_wear_fr", ("m_tyresWear", 1), False),
        ("tyre_wear_rl", ("m_tyresWear", 2), False),
        ("tyre_wear_rr", ("m_tyresWear", 3), False),
    ),
}

#: Top-level key in each parsed packet holding the per-car list.
_CONTAINER_KEYS: dict[int, str] = {
    _MOTION: "m_carMotionData",
    _LAPDATA: "m_lapData",
    _CARTELEMETRY: "m_carTelemetryData",
    _CARSTATUS: "m_carStatusData",
    _CARDAMAGE: "m_carDamageData",
}


def _extract_value(car: dict[str, Any], source: Source) -> float:
    """Read a scalar value from the player-car dict (0.0 if missing)."""
    if isinstance(source, tuple):
        arr_key, idx = source
        arr = car.get(arr_key)
        if arr is None or idx >= len(arr):
            return 0.0
        return float(arr[idx])
    return float(car.get(source, 0.0))


class TelemetryAligner:
    """Align multi-source telemetry onto a unified 60Hz grid (per-car).

    Usage (player car — backward compatible)::

        aligner = TelemetryAligner()
        header, parsed = parse_packet(datagram)
        aligner.on_packet(header, parsed)
        frames = aligner.sample_60hz(t0, t1)

    Usage (multi-car — Iter-131)::

        aligner = TelemetryAligner()
        aligner.on_packet_all_cars(header, parsed)   # ingest all 22 cars
        car5 = aligner.sample_60hz(t0, t1, car_index=5)
        who = aligner.available_car_indices()        # e.g. {0, 1, 5, 7}
    """

    def __init__(self, buffer_size: int = 20000) -> None:
        # Iter-131: nested buffer dict — outer key is packet_id, inner key is
        # car_index. Inner deques are created lazily on first observation so
        # multi-car ingestion does not pay 22x memory up front.
        self._buffers: dict[int, dict[int, deque[tuple[float, dict[str, float]]]]] = {
            pid: {} for pid in _SOURCE_SPECS
        }
        self._buffer_size = buffer_size
        # Most recent player_car_index observed via on_packet; query methods
        # default to this when ``car_index`` is None.
        self._player_car_index: int = 0

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def on_packet(
        self,
        header: PacketHeader,
        parsed: dict[str, Any],
        *,
        car_index: int | None = None,
    ) -> None:
        """Append one car's slice of a relevant packet to its buffer.

        ``car_index`` selects which car in the per-car list to ingest; if
        ``None`` (default) the player car (``header.player_car_index``) is
        used. The most recently observed ``player_car_index`` is tracked so
        query methods default to it when ``car_index`` is omitted.

        Packets outside the alignment source set ({Motion, CarTelemetry,
        CarStatus, LapData, CarDamage}) are ignored.
        """
        pid = header.packet_id
        spec = _SOURCE_SPECS.get(pid)
        if spec is None:
            return
        pci = header.player_car_index
        if 0 <= pci < NUM_CARS:
            self._player_car_index = pci
        target = pci if car_index is None else car_index
        cars = parsed.get(_CONTAINER_KEYS[pid])
        if not cars or target < 0 or target >= len(cars):
            return
        car = cars[target]
        fields: dict[str, float] = {
            unified_key: _extract_value(car, source)
            for unified_key, source, _is_int in spec
        }
        inner = self._buffers[pid]
        buf = inner.get(target)
        if buf is None:
            buf = deque(maxlen=self._buffer_size)
            inner[target] = buf
        buf.append((float(header.session_time), fields))

    def on_packet_all_cars(
        self,
        header: PacketHeader,
        parsed: dict[str, Any],
    ) -> int:
        """Iter-131: ingest every car in the per-car list of one packet.

        Useful for broadcast / spectator views that need to query any of the
        22 cars. The player car (``header.player_car_index``) is still
        tracked for default-query behaviour. Returns the number of cars
        actually ingested (0 if the packet is outside the alignment source
        set or the per-car list is empty).
        """
        pid = header.packet_id
        spec = _SOURCE_SPECS.get(pid)
        if spec is None:
            return 0
        pci = header.player_car_index
        if 0 <= pci < NUM_CARS:
            self._player_car_index = pci
        cars = parsed.get(_CONTAINER_KEYS[pid])
        if not cars:
            return 0
        inner = self._buffers[pid]
        n = 0
        session_t = float(header.session_time)
        for target, car in enumerate(cars):
            if car is None:
                continue
            fields: dict[str, float] = {
                unified_key: _extract_value(car, source)
                for unified_key, source, _is_int in spec
            }
            buf = inner.get(target)
            if buf is None:
                buf = deque(maxlen=self._buffer_size)
                inner[target] = buf
            buf.append((session_t, fields))
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # Single-field interpolation helpers (bisect-based, reference path)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _interp_float(
        samples: list[tuple[float, float]], t: float
    ) -> float | None:
        """Linear-interpolate a float field at ``t``.

        ``samples`` is a list of ``(time, value)`` tuples sorted by time.
        Clamps to the endpoints outside the available range; returns ``None``
        when ``samples`` is empty.
        """
        if not samples:
            return None
        times = [s[0] for s in samples]
        idx = bisect.bisect_left(times, t)
        if idx == 0:
            return samples[0][1]
        if idx >= len(samples):
            return samples[-1][1]
        t0, v0 = samples[idx - 1]
        t1, v1 = samples[idx]
        if t1 == t0:
            return v0
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    @staticmethod
    def _nearest_int(
        samples: list[tuple[float, float]], t: float
    ) -> float | None:
        """Nearest-neighbour for an integer field at ``t``.

        ``samples`` is a list of ``(time, value)`` tuples sorted by time.
        Clamps to the endpoints outside the available range; returns ``None``
        when ``samples`` is empty. Ties pick the earlier sample.
        """
        if not samples:
            return None
        times = [s[0] for s in samples]
        idx = bisect.bisect_left(times, t)
        if idx == 0:
            return samples[0][1]
        if idx >= len(samples):
            return samples[-1][1]
        if abs(t - samples[idx - 1][0]) <= abs(t - samples[idx][0]):
            return samples[idx - 1][1]
        return samples[idx][1]

    # ------------------------------------------------------------------ #
    # Vectorised interpolation (fast path for sample_60hz)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nearest_vec(
        ts: np.ndarray, vs: np.ndarray, tq: np.ndarray
    ) -> np.ndarray:
        """Vectorised nearest-neighbour of ``vs`` (indexed by ``ts``) at ``tq``.

        ``ts`` must be sorted ascending. Returns clamped values outside range.
        """
        n = len(ts)
        idx = np.searchsorted(ts, tq)
        idx_left = np.clip(idx - 1, 0, n - 1)
        idx_right = np.clip(idx, 0, n - 1)
        choose_right = np.abs(tq - ts[idx_right]) < np.abs(tq - ts[idx_left])
        nearest_idx = np.where(choose_right, idx_right, idx_left)
        return vs[nearest_idx]

    def _sorted_items(
        self, pid: int, *, car_index: int | None = None
    ) -> list[tuple[float, dict[str, float]]]:
        """Return this source's buffer sorted by session_time (sort-on-query).

        Iter-131: ``car_index`` selects which car's buffer to sort; ``None``
        defaults to the most-recently-tracked player car.
        """
        target = self._player_car_index if car_index is None else car_index
        buf = self._buffers[pid].get(target)
        if not buf:
            return []
        return sorted(buf, key=lambda x: x[0])

    def available_car_indices(self) -> set[int]:
        """Iter-131: return the set of car indices that have >=1 ingested
        sample across any source.

        Useful for broadcast UIs that need to enumerate the field (e.g. for
        multi-car overlays or position-relative comparisons). Returns an
        empty set if no packets have been ingested.
        """
        seen: set[int] = set()
        for inner in self._buffers.values():
            seen.update(inner.keys())
        return seen

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def sample_60hz(
        self,
        t_start: float,
        t_end: float,
        dt: float = 1.0 / 60.0,
        *,
        car_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Produce unified frame dicts on the grid ``t_start, t_start+dt, ...``.

        Frames are emitted up to (not including) ``t_end``. Each frame carries
        every :data:`UNIFIED_KEYS` field; fields whose source has no samples
        yet are ``None``. Float fields are linearly interpolated (clamped
        outside range); integer fields use nearest-neighbour.

        Iter-131: ``car_index`` selects which car's timeseries to interpolate;
        ``None`` defaults to the player car.
        """
        span = t_end - t_start
        if span <= 0:
            return []
        n = int(math.ceil(span / dt - 1e-9))
        if n <= 0:
            return []
        # Grid points computed as t_start + i*dt (no drift accumulation).
        tq = t_start + np.arange(n, dtype=np.float64) * dt

        # Per-field interpolated columns (length-n numpy arrays).
        columns: dict[str, np.ndarray] = {}
        for pid in _SOURCE_SPECS:
            items = self._sorted_items(pid, car_index=car_index)
            if not items:
                continue
            ts = np.fromiter(
                (it[0] for it in items), dtype=np.float64, count=len(items)
            )
            for unified_key, _src, _is_int in _SOURCE_SPECS[pid]:
                vs = np.fromiter(
                    (it[1].get(unified_key, 0.0) for it in items),
                    dtype=np.float64,
                    count=len(items),
                )
                if unified_key in _INT_KEYS:
                    columns[unified_key] = self._nearest_vec(ts, vs, tq)
                else:
                    columns[unified_key] = np.interp(tq, ts, vs)

        # Assemble per-frame dicts (session_time + every unified key).
        keys = UNIFIED_KEYS[1:]
        frames: list[dict[str, Any]] = []
        for i in range(n):
            frame: dict[str, Any] = {"session_time": float(tq[i])}
            for key in keys:
                col = columns.get(key)
                if col is None:
                    frame[key] = None
                elif key in _INT_KEYS:
                    frame[key] = int(col[i])
                else:
                    frame[key] = float(col[i])
            frames.append(frame)
        return frames

    def latest_unified_frame(
        self, *, car_index: int | None = None
    ) -> dict[str, Any] | None:
        """Return the unified frame at the most recent available session_time.

        ``session_time`` is the max across all source buffers (the latest
        sample seen, regardless of source) for the requested car. Returns
        ``None`` when no packets have been observed for that car.

        Iter-131: ``car_index`` selects which car; ``None`` defaults to the
        player car.

        Iter-251: fast path — at the global max time the bisect interpolation
        clamps to each source's LAST sample, so we take each source's
        max-time sample directly instead of sorting every buffer and building
        per-field sample lists (``_frame_at``). This is O(S·F) instead of
        O(S·N·F) and keeps the 60Hz WS broadcast cheap as buffers grow.
        """
        target = self._player_car_index if car_index is None else car_index
        max_t: float | None = None
        latest: dict[int, tuple[float, dict[str, float]]] = {}
        for pid, inner in self._buffers.items():
            buf = inner.get(target)
            if not buf:
                continue
            it = max(buf, key=lambda x: x[0])  # O(N), once per source
            latest[pid] = it
            if max_t is None or it[0] > max_t:
                max_t = it[0]
        if max_t is None:
            return None
        frame: dict[str, Any] = {"session_time": float(max_t)}
        for key in UNIFIED_KEYS[1:]:
            frame[key] = None
        for pid, spec in _SOURCE_SPECS.items():
            entry = latest.get(pid)
            if entry is None:
                continue
            fields = entry[1]
            for unified_key, _src, is_int in spec:
                v = fields.get(unified_key, 0.0)
                frame[unified_key] = int(v) if is_int else float(v)
        return frame

    def _frame_at(
        self, t: float, *, car_index: int | None = None
    ) -> dict[str, Any]:
        """Build a single unified frame at ``t`` using the bisect helpers."""
        frame: dict[str, Any] = {"session_time": float(t)}
        for key in UNIFIED_KEYS[1:]:
            frame[key] = None
        for pid, spec in _SOURCE_SPECS.items():
            items = self._sorted_items(pid, car_index=car_index)
            if not items:
                continue
            for unified_key, _src, is_int in spec:
                samples = [
                    (it[0], it[1].get(unified_key, 0.0)) for it in items
                ]
                if is_int:
                    v = self._nearest_int(samples, t)
                    if v is not None:
                        frame[unified_key] = int(v)
                else:
                    v = self._interp_float(samples, t)
                    if v is not None:
                        frame[unified_key] = float(v)
        return frame


__all__ = ["TelemetryAligner", "UNIFIED_KEYS"]
