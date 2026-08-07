"""Unit tests for :mod:`f1opt.telemetry.aligner`.

Feeds crafted ``(header, parsed)`` pairs directly to :class:`TelemetryAligner`
(no UDP socket needed). Verifies:
- The 1/60 grid is produced exactly (each ``session_time`` == t0 + i/60).
- Float fields are linearly interpolated between surrounding samples.
- Integer fields use nearest-neighbour (no fractional values).
- Fields from a missing source are ``None``.
- ``latest_unified_frame()`` returns the frame at the max available time.
- A 6000-frame query completes in < 200ms.
- An empty aligner yields all-None frames and ``latest_unified_frame()`` is None.
- Out-of-order (late) insertion still yields sane clamped values (sort-on-query).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from f1opt.telemetry.aligner import UNIFIED_KEYS, TelemetryAligner
from f1opt.telemetry.packets import PacketHeader

SESSION_UID = 0x123456789ABCDEF0
NUM_CARS = 22

# Per-source empty-car dicts (only fields the aligner reads matter here).
_EMPTY_MOTION = {
    "m_worldPositionX": 0.0, "m_worldPositionY": 0.0, "m_worldPositionZ": 0.0,
    "m_worldVelocityX": 0.0, "m_worldVelocityY": 0.0, "m_worldVelocityZ": 0.0,
    "m_worldForwardDirX": 0, "m_worldForwardDirY": 0, "m_worldForwardDirZ": 0,
    "m_worldRightDirX": 0, "m_worldRightDirY": 0, "m_worldRightDirZ": 0,
    "m_gForceLateral": 0.0, "m_gForceLongitudinal": 0.0, "m_gForceVertical": 0.0,
    "m_yaw": 0.0, "m_pitch": 0.0, "m_roll": 0.0,
}
_EMPTY_TELEM = {
    "m_speed": 0, "m_throttle": 0.0, "m_steer": 0.0, "m_brake": 0.0,
    "m_clutch": 0, "m_gear": 0, "m_engineRPM": 0, "m_drs": 0,
    "m_revLightsPercent": 0, "m_revLightsBitValue": 0,
    "m_brakesTemperature": [0, 0, 0, 0],
    "m_tyresSurfaceTemperature": [0, 0, 0, 0],
    "m_tyresInnerTemperature": [0, 0, 0, 0],
    "m_engineTemperature": 0,
    "m_tyresPressure": [0.0, 0.0, 0.0, 0.0],
    "m_surfaceType": [0, 0, 0, 0],
}
_EMPTY_STATUS = {
    "m_ersStoreEnergy": 0.0, "m_ersDeployMode": 0, "m_drsAllowed": 0,
    "m_fuelInTank": 0.0, "m_fuelRemainingLaps": 0.0,
}
_EMPTY_DAMAGE = {"m_tyresWear": [0.0, 0.0, 0.0, 0.0]}
_EMPTY_LAP = {"m_currentLapTimeInMS": 0, "m_lapDistance": 0.0}


def make_header(
    packet_id: int,
    session_time: float,
    player_car: int = 0,
) -> PacketHeader:
    return PacketHeader(
        packet_format=2025,
        game_year=25,
        game_major_version=1,
        game_minor_version=0,
        packet_version=1,
        packet_id=packet_id,
        session_uid=SESSION_UID,
        session_time=session_time,
        frame_identifier=0,
        overall_frame_identifier=0,
        player_car_index=player_car,
        secondary_player_car_index=255,
    )


def _cars(player: int, car: dict) -> list[dict]:
    """22-car list with the player car overridden; others are shared empties."""
    cars: list[dict] = [{}] * NUM_CARS
    cars[player] = car
    return cars


def telem_packet(
    t: float,
    *,
    speed: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
    steer: float = 0.0,
    gear: int = 0,
    rpm: int = 0,
    tyre_temps=(0, 0, 0, 0),
    player: int = 0,
) -> tuple[PacketHeader, dict]:
    car = {**_EMPTY_TELEM, "m_speed": speed, "m_throttle": throttle,
           "m_steer": steer, "m_brake": brake, "m_gear": gear,
           "m_engineRPM": rpm, "m_tyresSurfaceTemperature": list(tyre_temps)}
    parsed = {"m_carTelemetryData": _cars(player, car)}
    return make_header(6, t, player), parsed


def motion_packet(
    t: float,
    *,
    g_lat: float = 0.0,
    g_long: float = 0.0,
    g_vert: float = 0.0,
    world=(0.0, 0.0, 0.0),
    vel=(0.0, 0.0, 0.0),
    ypr=(0.0, 0.0, 0.0),
    player: int = 0,
) -> tuple[PacketHeader, dict]:
    car = {**_EMPTY_MOTION,
           "m_gForceLateral": g_lat, "m_gForceLongitudinal": g_long,
           "m_gForceVertical": g_vert,
           "m_worldPositionX": world[0], "m_worldPositionY": world[1],
           "m_worldPositionZ": world[2],
           "m_worldVelocityX": vel[0], "m_worldVelocityY": vel[1],
           "m_worldVelocityZ": vel[2],
           "m_yaw": ypr[0], "m_pitch": ypr[1], "m_roll": ypr[2]}
    parsed = {"m_carMotionData": _cars(player, car)}
    return make_header(0, t, player), parsed


def status_packet(
    t: float,
    *,
    ers_store: float = 0.0,
    ers_deploy_mode: int = 0,
    drs_allowed: int = 0,
    fuel_in_tank: float = 0.0,
    fuel_remaining_laps: float = 0.0,
    player: int = 0,
) -> tuple[PacketHeader, dict]:
    car = {**_EMPTY_STATUS, "m_ersStoreEnergy": ers_store,
           "m_ersDeployMode": ers_deploy_mode, "m_drsAllowed": drs_allowed,
           "m_fuelInTank": fuel_in_tank,
           "m_fuelRemainingLaps": fuel_remaining_laps}
    parsed = {"m_carStatusData": _cars(player, car)}
    return make_header(7, t, player), parsed


def damage_packet(
    t: float, *, tyre_wear=(0.0, 0.0, 0.0, 0.0), player: int = 0
) -> tuple[PacketHeader, dict]:
    car = {**_EMPTY_DAMAGE, "m_tyresWear": list(tyre_wear)}
    parsed = {"m_carDamageData": _cars(player, car)}
    return make_header(10, t, player), parsed


def lap_packet(
    t: float, *, lap_time_ms: int = 0, lap_distance: float = 0.0, player: int = 0
) -> tuple[PacketHeader, dict]:
    car = {**_EMPTY_LAP, "m_currentLapTimeInMS": lap_time_ms,
           "m_lapDistance": lap_distance}
    parsed = {"m_lapData": _cars(player, car)}
    return make_header(2, t, player), parsed


# --------------------------------------------------------------------------- #
# Grid + interpolation
# --------------------------------------------------------------------------- #
class TestGridAndInterpolation:
    def test_grid_is_60hz(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, speed=100.0))
        aligner.on_packet(*telem_packet(1.0 + 5 / 60, speed=200.0))

        frames = aligner.sample_60hz(1.0, 1.0 + 5 / 60, 1 / 60)
        assert len(frames) == 5
        for i, f in enumerate(frames):
            assert f["session_time"] == pytest.approx(1.0 + i / 60, abs=1e-9)
            # Every unified key is present.
            assert set(f.keys()) == set(UNIFIED_KEYS)

    def test_speed_linear_interpolation(self) -> None:
        aligner = TelemetryAligner()
        # Two surrounding CarTelemetry samples: speed 100 at t=1.0, 300 at t=2.0.
        aligner.on_packet(*telem_packet(1.0, speed=100.0))
        aligner.on_packet(*telem_packet(2.0, speed=300.0))
        # Query a single frame exactly at the midpoint t=1.5.
        frames = aligner.sample_60hz(1.5, 1.5 + 1 / 60, 1 / 60)
        assert len(frames) == 1
        # Linear interpolation: 100 + (300-100) * 0.5 == 200.
        assert frames[0]["speed"] == pytest.approx(200.0, abs=1e-3)
        # Clamped to first sample before the range.
        early = aligner.sample_60hz(0.5, 0.5 + 1 / 60, 1 / 60)
        assert early[0]["speed"] == pytest.approx(100.0, abs=1e-3)
        # Clamped to last sample after the range.
        late = aligner.sample_60hz(3.0, 3.0 + 1 / 60, 1 / 60)
        assert late[0]["speed"] == pytest.approx(300.0, abs=1e-3)

    def test_gear_is_nearest_neighbour(self) -> None:
        aligner = TelemetryAligner()
        # gear 3 at t=1.0, gear 7 at t=2.0.
        aligner.on_packet(*telem_packet(1.0, gear=3))
        aligner.on_packet(*telem_packet(2.0, gear=7))

        # t=1.4 is closer to t=1.0 -> gear 3 (int, no fractional).
        left = aligner.sample_60hz(1.4, 1.4 + 1 / 60, 1 / 60)
        assert left[0]["gear"] == 3
        assert isinstance(left[0]["gear"], int)

        # t=1.6 is closer to t=2.0 -> gear 7.
        right = aligner.sample_60hz(1.6, 1.6 + 1 / 60, 1 / 60)
        assert right[0]["gear"] == 7

    def test_tyre_temps_and_arrays(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, tyre_temps=(80, 81, 82, 83)))
        aligner.on_packet(*telem_packet(2.0, tyre_temps=(100, 101, 102, 103)))
        f = aligner.sample_60hz(1.5, 1.5 + 1 / 60, 1 / 60)[0]
        assert f["tyre_temp_fl"] == pytest.approx(90.0, abs=1e-3)
        assert f["tyre_temp_fr"] == pytest.approx(91.0, abs=1e-3)
        assert f["tyre_temp_rl"] == pytest.approx(92.0, abs=1e-3)
        assert f["tyre_temp_rr"] == pytest.approx(93.0, abs=1e-3)

    def test_motion_fields_interpolated(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*motion_packet(1.0, g_lat=1.0, world=(10, 20, 30)))
        aligner.on_packet(*motion_packet(2.0, g_lat=3.0, world=(20, 40, 60)))
        f = aligner.sample_60hz(1.5, 1.5 + 1 / 60, 1 / 60)[0]
        assert f["g_lat"] == pytest.approx(2.0, abs=1e-3)
        assert f["world_x"] == pytest.approx(15.0, abs=1e-3)
        assert f["world_y"] == pytest.approx(30.0, abs=1e-3)
        assert f["world_z"] == pytest.approx(45.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Missing source -> None
# --------------------------------------------------------------------------- #
class TestMissingSource:
    def test_only_cartelemetry_motion_fields_none(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, speed=100.0))
        f = aligner.sample_60hz(1.0, 1.0 + 1 / 60, 1 / 60)[0]
        # CarTelemetry fields are populated.
        assert f["speed"] == pytest.approx(100.0)
        assert f["gear"] == 0
        # Motion / CarStatus / LapData / CarDamage are None.
        for key in ("g_lat", "world_x", "velocity_x", "yaw",
                    "ers_store", "drs_allowed", "fuel_in_tank",
                    "lap_time", "lap_distance",
                    "tyre_wear_fl", "tyre_wear_rr"):
            assert f[key] is None, f"{key} should be None"


# --------------------------------------------------------------------------- #
# latest_unified_frame
# --------------------------------------------------------------------------- #
class TestLatestUnifiedFrame:
    def test_returns_frame_at_max_time(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, speed=100.0, gear=3))
        aligner.on_packet(*motion_packet(1.5, g_lat=1.0))
        aligner.on_packet(*telem_packet(2.0, speed=200.0, gear=5))
        aligner.on_packet(*status_packet(2.0, drs_allowed=1, fuel_in_tank=50.0))

        frame = aligner.latest_unified_frame()
        assert frame is not None
        assert frame["session_time"] == pytest.approx(2.0)
        # speed at exact sample t=2.0.
        assert frame["speed"] == pytest.approx(200.0)
        assert frame["gear"] == 5
        # status present at t=2.0.
        assert frame["drs_allowed"] == 1
        assert frame["fuel_in_tank"] == pytest.approx(50.0)
        # motion last sample is at t=1.5 -> clamped at t=2.0.
        assert frame["g_lat"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Empty aligner
# --------------------------------------------------------------------------- #
class TestEmptyAligner:
    def test_sample_60hz_all_none(self) -> None:
        aligner = TelemetryAligner()
        frames = aligner.sample_60hz(0.0, 0.1, 1 / 60)
        assert len(frames) == 6
        for f in frames:
            assert f["session_time"] is not None
            for key in UNIFIED_KEYS:
                if key == "session_time":
                    continue
                assert f[key] is None

    def test_latest_unified_frame_none(self) -> None:
        aligner = TelemetryAligner()
        assert aligner.latest_unified_frame() is None

    def test_zero_span_returns_empty(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, speed=100.0))
        assert aligner.sample_60hz(1.0, 1.0) == []


# --------------------------------------------------------------------------- #
# Out-of-order (late) insertion — sort-on-query
# --------------------------------------------------------------------------- #
class TestOutOfOrder:
    def test_late_sample_inserted_in_order(self) -> None:
        aligner = TelemetryAligner()
        # Insert t=2.0 then t=1.0 (earlier sample arrives late).
        aligner.on_packet(*telem_packet(2.0, speed=300.0))
        aligner.on_packet(*telem_packet(1.0, speed=100.0))

        # At t=1.5: linear interp between 1.0(100) and 2.0(300) -> 200.
        mid = aligner.sample_60hz(1.5, 1.5 + 1 / 60, 1 / 60)
        assert mid[0]["speed"] == pytest.approx(200.0, abs=1e-3)

    def test_clamp_outside_range_with_late_arrival(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(2.0, speed=300.0))
        aligner.on_packet(*telem_packet(1.0, speed=100.0))
        # Query before the earliest sample -> clamp to t=1.0 (speed 100).
        before = aligner.sample_60hz(0.5, 0.5 + 1 / 60, 1 / 60)
        assert before[0]["speed"] == pytest.approx(100.0, abs=1e-3)

    def test_late_sample_changes_nearest_gear(self) -> None:
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(2.0, gear=7))
        # Late sample at t=1.0 closer to query t=1.4.
        aligner.on_packet(*telem_packet(1.0, gear=3))
        f = aligner.sample_60hz(1.4, 1.4 + 1 / 60, 1 / 60)[0]
        assert f["gear"] == 3


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
class TestPerformance:
    def test_6000_frames_under_200ms(self) -> None:
        aligner = TelemetryAligner()
        # Feed 100s of CarTelemetry + Motion + CarStatus at 60Hz (6000 samples).
        for i in range(6000):
            t = i / 60.0
            aligner.on_packet(*telem_packet(t, speed=float(i), gear=i % 8, rpm=9000))
            aligner.on_packet(*motion_packet(t, g_lat=float(i) * 0.001))
            aligner.on_packet(*status_packet(t, fuel_in_tank=100.0 - i * 0.001))

        start = time.perf_counter()
        frames = aligner.sample_60hz(0.0, 100.0, 1 / 60)
        elapsed = time.perf_counter() - start

        assert len(frames) == 6000
        # Each session_time on the grid.
        for i, f in enumerate(frames):
            assert f["session_time"] == pytest.approx(i / 60.0, abs=1e-9)
        # Spot-check an interpolated value at t=3000/60 = 50.0s: speed == 3000.
        assert frames[3000]["speed"] == pytest.approx(3000.0, abs=1e-3)
        assert elapsed < 0.5, f"sample_60hz too slow: {elapsed * 1000:.1f}ms"


# --------------------------------------------------------------------------- #
# Direct helper checks
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_interp_float_clamps_and_interpolates(self) -> None:
        samples = [(0.0, 0.0), (2.0, 10.0)]
        assert TelemetryAligner._interp_float(samples, 1.0) == pytest.approx(5.0)
        assert TelemetryAligner._interp_float(samples, -1.0) == pytest.approx(0.0)
        assert TelemetryAligner._interp_float(samples, 5.0) == pytest.approx(10.0)
        assert TelemetryAligner._interp_float([], 1.0) is None

    def test_nearest_int_picks_nearest(self) -> None:
        samples = [(0.0, 1), (2.0, 9)]
        assert TelemetryAligner._nearest_int(samples, 0.9) == 1
        assert TelemetryAligner._nearest_int(samples, 1.1) == 9
        assert TelemetryAligner._nearest_int(samples, 10.0) == 9
        assert TelemetryAligner._nearest_int([], 1.0) is None

    def test_numpy_and_bisect_paths_agree(self) -> None:
        # Ensure the vectorised fast path matches the reference helpers.
        aligner = TelemetryAligner()
        for i in range(20):
            t = i * 0.1
            aligner.on_packet(*telem_packet(t, speed=float(i) * 10.0, gear=i))
        tq = np.array([0.15, 0.95, 1.55, 5.0])
        frames = [aligner._frame_at(float(x)) for x in tq]
        grid = aligner.sample_60hz(0.0, 2.0, 0.5)
        # Grid points at 0.0, 0.5, 1.0, 1.5 — compare against _frame_at.
        for g in grid:
            ref = aligner._frame_at(g["session_time"])
            assert g["speed"] == pytest.approx(ref["speed"], abs=1e-9)
            assert g["gear"] == ref["gear"]
        # Direct spot checks against reference helpers on a few query times.
        for q, fr in zip(tq, frames, strict=True):
            assert fr["session_time"] == pytest.approx(float(q))


# --------------------------------------------------------------------------- #
# Iter-131: multi-car alignment
# --------------------------------------------------------------------------- #
class TestMultiCarAlignment:
    """Iter-131: per-car ingest + query via car_index keyword argument."""

    def test_explicit_car_index_ingest_and_query(self) -> None:
        """on_packet(car_index=5) ingests car 5 only; default query sees None."""
        aligner = TelemetryAligner()
        # Build a packet where car 0 has speed=100 and car 5 has speed=250.
        h, parsed = telem_packet(1.0, speed=100.0, player=0)
        cars = list(parsed["m_carTelemetryData"])
        cars[5] = {**_EMPTY_TELEM, "m_speed": 250.0}
        parsed["m_carTelemetryData"] = cars
        # Ingest ONLY car 5 explicitly.
        aligner.on_packet(h, parsed, car_index=5)
        # Default (player car 0) — no data.
        f0 = aligner.sample_60hz(1.0, 1.0 + 1 / 60, 1 / 60)
        assert f0[0]["speed"] is None
        # Car 5 query returns speed 250.
        f5 = aligner.sample_60hz(1.0, 1.0 + 1 / 60, 1 / 60, car_index=5)
        assert f5[0]["speed"] == pytest.approx(250.0)
        assert aligner.available_car_indices() == {5}

    def test_on_packet_all_cars_ingests_every_car(self) -> None:
        """on_packet_all_cars returns NUM_CARS and ingests every slot."""
        aligner = TelemetryAligner()
        # Build a 22-car packet where car i has speed = 100 + i.
        h, parsed = telem_packet(1.0, speed=100.0, player=0)
        # Override the per-car list with one entry per car.
        cars = []
        for i in range(NUM_CARS):
            car = {**parsed["m_carTelemetryData"][0], "m_speed": 100 + i}
            cars.append(car)
        parsed["m_carTelemetryData"] = cars
        n = aligner.on_packet_all_cars(h, parsed)
        assert n == NUM_CARS
        assert aligner.available_car_indices() == set(range(NUM_CARS))
        # Spot-check a few cars.
        for i in (0, 5, 11, 21):
            f = aligner.sample_60hz(1.0, 1.0 + 1 / 60, 1 / 60, car_index=i)
            assert f[0]["speed"] == pytest.approx(float(100 + i))

    def test_per_car_interpolation_isolated(self) -> None:
        """Two cars with different slopes interpolate independently."""
        aligner = TelemetryAligner()
        # Car 0: 100 -> 200. Car 5: 300 -> 500.
        for t, s0, s5 in [(1.0, 100.0, 300.0), (2.0, 200.0, 500.0)]:
            h, parsed = telem_packet(t, speed=s0, player=0)
            # Replace car 5's slot.
            cars = list(parsed["m_carTelemetryData"])
            cars[5] = {**cars[5], "m_speed": s5}
            parsed["m_carTelemetryData"] = cars
            aligner.on_packet_all_cars(h, parsed)
        # At t=1.5: car 0 -> 150, car 5 -> 400.
        f0 = aligner.sample_60hz(1.5, 1.5 + 1 / 60, 1 / 60, car_index=0)[0]
        f5 = aligner.sample_60hz(1.5, 1.5 + 1 / 60, 1 / 60, car_index=5)[0]
        assert f0["speed"] == pytest.approx(150.0, abs=1e-3)
        assert f5["speed"] == pytest.approx(400.0, abs=1e-3)

    def test_query_car_with_no_data_returns_all_none(self) -> None:
        """sample_60hz / latest_unified_frame on a never-seen car return None."""
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, speed=100.0))  # car 0 only
        frames = aligner.sample_60hz(1.0, 1.0 + 5 / 60, 1 / 60, car_index=7)
        assert len(frames) == 5
        for f in frames:
            for key in UNIFIED_KEYS:
                if key == "session_time":
                    continue
                assert f[key] is None, f"car 7 {key} should be None"
        assert aligner.latest_unified_frame(car_index=7) is None

    def test_player_car_index_tracked_across_calls(self) -> None:
        """Default query follows header.player_car_index across packets."""
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, speed=100.0, player=0))
        # New packet: player_car_index=3, ingest car 3 (player default).
        aligner.on_packet(*telem_packet(2.0, speed=333.0, player=3))
        f = aligner.latest_unified_frame()
        assert f is not None
        assert f["speed"] == pytest.approx(333.0)
        assert aligner.available_car_indices() == {0, 3}

    def test_late_arrival_works_for_non_player_car(self) -> None:
        """Sort-on-query interpolation works for explicit car_index."""
        aligner = TelemetryAligner()
        # Insert t=2.0 then t=1.0 (late) for car 4 — populate car 4's slot.
        h2, p2 = telem_packet(2.0, speed=0.0, player=0)
        p2["m_carTelemetryData"] = list(p2["m_carTelemetryData"])
        p2["m_carTelemetryData"][4] = {**_EMPTY_TELEM, "m_speed": 300.0}
        aligner.on_packet(h2, p2, car_index=4)
        h1, p1 = telem_packet(1.0, speed=0.0, player=0)
        p1["m_carTelemetryData"] = list(p1["m_carTelemetryData"])
        p1["m_carTelemetryData"][4] = {**_EMPTY_TELEM, "m_speed": 100.0}
        aligner.on_packet(h1, p1, car_index=4)
        f = aligner.sample_60hz(1.5, 1.5 + 1 / 60, 1 / 60, car_index=4)[0]
        assert f["speed"] == pytest.approx(200.0, abs=1e-3)

    def test_bad_car_index_silently_ignored(self) -> None:
        """Negative / oversized car_index does not crash or ingest."""
        aligner = TelemetryAligner()
        aligner.on_packet(*telem_packet(1.0, speed=100.0), car_index=-1)
        aligner.on_packet(*telem_packet(1.0, speed=100.0), car_index=NUM_CARS + 5)
        assert aligner.available_car_indices() == set()

    def test_buffer_size_bounded_per_car(self) -> None:
        """Each car's deque respects buffer_size independently."""
        aligner = TelemetryAligner(buffer_size=50)
        for i in range(100):
            t = i / 60.0
            h, parsed = telem_packet(t, speed=float(i), player=0)
            cars = list(parsed["m_carTelemetryData"])
            cars[9] = {**cars[9], "m_speed": float(i) * 2}
            parsed["m_carTelemetryData"] = cars
            aligner.on_packet_all_cars(h, parsed)
        for inner in aligner._buffers.values():
            for buf in inner.values():
                assert len(buf) <= 50

    def test_out_of_scope_packet_id_returns_zero(self) -> None:
        """on_packet_all_cars on a non-alignment packet returns 0."""
        from f1opt.telemetry.packets import PacketHeader
        h = PacketHeader(
            packet_format=2025, game_year=25, game_major_version=1,
            game_minor_version=0, packet_version=1, packet_id=1,  # Session
            session_uid=0, session_time=1.0, frame_identifier=0,
            overall_frame_identifier=0, player_car_index=0,
            secondary_player_car_index=255,
        )
        aligner = TelemetryAligner()
        n = aligner.on_packet_all_cars(h, {"m_sessionData": {}})
        assert n == 0
        assert aligner.available_car_indices() == set()
