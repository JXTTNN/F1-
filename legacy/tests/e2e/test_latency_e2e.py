"""End-to-end WebSocket latency measurement for Iter-08.

Measures synthetic 60Hz frame broadcast latency through the real FastAPI WS
endpoint (``/ws/telemetry``) using ``TestClient`` + :func:`push_frame`. Each
frame carries a ``_send_ts`` :func:`time.perf_counter` timestamp; the WS
client records ``recv_ts`` on receipt and the per-frame latency
(``recv_ts - send_ts``) is aggregated into min / p50 / p95 / max for the
iteration record.

Test layout (≥4 cases, all named ``test_latency_e2e_*``):

1. ``test_latency_e2e_single_client_60hz`` — 100 interleaved 60Hz frames through
   one WS connection; no drops, p95 ≤ 100ms. (Tasks 8.1.1 + 8.1.2)
2. ``test_latency_e2e_batch_send_50_no_drop`` — 50-frame batch send (within the
   per-client queue maxsize=64) then batch receive; no drops. (Task 8.1.2)
3. ``test_latency_e2e_multi_client`` — 2 WS clients both receive the same 50
   broadcast frames; p95 within 50ms of each other. (Task 8.1.3)
4. ``test_latency_e2e_high_volume_200_frames`` — 200 interleaved frames; no
   drops, p95 ≤ 100ms, broadcast stays non-blocking. (Reliability stress)

Threading note: ``TestClient`` runs the app's event loop in a portal thread.
``push_frame`` (called from the test thread) detects it is not on the app loop
and schedules ``broadcast`` via ``loop.call_soon_threadsafe``; the WS writer
task on the app loop drains the bounded per-client queue (maxsize=64,
drop-oldest) and ``send_json``s the frame through the ASGI transport to the
test-side ``ws.receive_json()``.

To avoid drop-oldest when bursting >64 frames, tests interleave
``push_frame`` / ``ws.receive_json`` one-for-one (the queue never holds >1
frame). This mirrors the proven pattern in
``tests/e2e/test_smoke_e2e.py::test_e2e_full_loop_udp_to_ui``. The batch test
(50 frames) stays under the 64-frame queue cap so batch send is also safe.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

from fastapi.testclient import TestClient

from f1opt.api.app import create_app, push_frame


def _make_frame(i: int) -> dict[str, Any]:
    """Build a synthetic 60Hz unified frame with a send-timestamp field.

    ``_send_ts`` is read back on the WS client to compute end-to-end latency.
    The leading underscore avoids colliding with real aligner frame keys.
    """
    t = i / 60.0
    return {
        "session_time": t,
        "speed": 200.0 + 5.0 * (i % 30),
        "throttle": 0.8,
        "brake": 0.0,
        "steer": 0.0,
        "gear": 6,
        "rpm": 9000,
        "g_lat": 1.5,
        "g_long": 0.0,
        "lap_time": 86.5 + t,
        "lap_distance": float(i),
        "ers_store": 1_000_000.0,
        "drs_allowed": 0,
        "tyre_wear_fl": 5.0,
        "tyre_wear_fr": 5.0,
        "tyre_wear_rl": 15.0,
        "tyre_wear_rr": 16.0,
        "fuel_in_tank": 30.0,
        "_send_ts": time.perf_counter(),
    }


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile of a pre-sorted list (p in [0, 100])."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
    return sorted_vals[idx]


def _stats(latencies_ms: list[float]) -> dict[str, float]:
    """Aggregate per-frame latencies (ms) into min/p50/p95/max."""
    s = sorted(latencies_ms)
    return {
        "n": len(s),
        "min_ms": s[0] if s else 0.0,
        "p50_ms": statistics.median(s) if s else 0.0,
        "p95_ms": _percentile(s, 95),
        "max_ms": s[-1] if s else 0.0,
    }


def _print_stats(label: str, stats: dict[str, float]) -> None:
    print(
        f"[latency-e2e] {label}: "
        f"n={stats['n']} "
        f"min={stats['min_ms']:.3f}ms "
        f"p50={stats['p50_ms']:.3f}ms "
        f"p95={stats['p95_ms']:.3f}ms "
        f"max={stats['max_ms']:.3f}ms"
    )


# --------------------------------------------------------------------------- #
# 8.1.1 + 8.1.2 — Single client, 100 interleaved 60Hz frames, no drops
# --------------------------------------------------------------------------- #
def test_latency_e2e_single_client_60hz() -> None:
    """100 synthetic 60Hz frames via push_frame -> WS; no drops, p95 ≤ 100ms."""
    app = create_app(start_listener=False)
    n = 100
    latencies_ms: list[float] = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry") as ws:
            # Interleave send/receive one-for-one: the per-client bounded queue
            # (maxsize=64, drop-oldest) never holds >1 frame, guaranteeing no
            # drops while still exercising the full push_frame -> broadcast ->
            # writer task -> ASGI -> receive_json path per frame.
            for i in range(n):
                frame = _make_frame(i)
                push_frame(app, frame)
                msg = ws.receive_json()
                recv_ts = time.perf_counter()
                assert msg["type"] == "frame"
                assert msg["speed"] == frame["speed"]
                latencies_ms.append((recv_ts - msg["_send_ts"]) * 1000.0)

    assert len(latencies_ms) == n, (
        f"expected {n} frames, received {len(latencies_ms)} (drop-oldest fired?)"
    )
    stats = _stats(latencies_ms)
    _print_stats("single_client_60hz", stats)
    assert stats["p95_ms"] <= 100.0, (
        f"p95 latency {stats['p95_ms']:.3f}ms > 100ms budget"
    )
    assert stats["min_ms"] >= 0.0


# --------------------------------------------------------------------------- #
# 8.1.2 — Batch send 50 (within queue maxsize=64), batch receive — no drops
# --------------------------------------------------------------------------- #
def test_latency_e2e_batch_send_50_no_drop() -> None:
    """50-frame batch send (≤ queue maxsize 64) then batch receive; no drops."""
    app = create_app(start_listener=False)
    n = 50
    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry") as ws:
            frames = [_make_frame(i) for i in range(n)]
            # Batch send all 50 first (queue maxsize=64 so no drop-oldest),
            # then batch receive. Closer to a real bursty-producer pattern.
            for f in frames:
                push_frame(app, f)
            received: list[tuple[dict[str, Any], float]] = []
            for _ in range(n):
                msg = ws.receive_json()
                received.append((msg, time.perf_counter()))

    assert len(received) == n, (
        f"expected {n} frames, received {len(received)} (drop-oldest fired?)"
    )
    latencies_ms: list[float] = []
    for (msg, recv_ts), f in zip(received, frames, strict=True):
        assert msg["type"] == "frame"
        assert msg["speed"] == f["speed"]
        latencies_ms.append((recv_ts - msg["_send_ts"]) * 1000.0)
    stats = _stats(latencies_ms)
    _print_stats("batch_send_50", stats)
    assert stats["p95_ms"] <= 100.0, (
        f"p95 latency {stats['p95_ms']:.3f}ms > 100ms budget"
    )


# --------------------------------------------------------------------------- #
# 8.1.3 — Multi-client: 2 WS connections both receive the same broadcast
# --------------------------------------------------------------------------- #
def test_latency_e2e_multi_client() -> None:
    """2 WS clients both receive the same 50 frames; p95 diff ≤ 50ms."""
    app = create_app(start_listener=False)
    n = 50
    latencies_a: list[float] = []
    latencies_b: list[float] = []
    with TestClient(app) as client:
        with (
            client.websocket_connect("/ws/telemetry") as ws_a,
            client.websocket_connect("/ws/telemetry") as ws_b,
        ):
            # Interleave: push 1 frame, then receive on both clients. Both
            # queues get the frame from a single broadcast; both clients should
            # see the same frame with similar end-to-end latency.
            for i in range(n):
                frame = _make_frame(i)
                push_frame(app, frame)
                msg_a = ws_a.receive_json()
                recv_a = time.perf_counter()
                msg_b = ws_b.receive_json()
                recv_b = time.perf_counter()
                assert msg_a["type"] == "frame"
                assert msg_b["type"] == "frame"
                assert msg_a["speed"] == frame["speed"]
                assert msg_b["speed"] == frame["speed"]
                latencies_a.append((recv_a - msg_a["_send_ts"]) * 1000.0)
                latencies_b.append((recv_b - msg_b["_send_ts"]) * 1000.0)

    assert len(latencies_a) == n
    assert len(latencies_b) == n
    stats_a = _stats(latencies_a)
    stats_b = _stats(latencies_b)
    _print_stats("multi_client_a", stats_a)
    _print_stats("multi_client_b", stats_b)
    p95_diff = abs(stats_a["p95_ms"] - stats_b["p95_ms"])
    print(f"[latency-e2e] multi_client p95_diff={p95_diff:.3f}ms")
    assert p95_diff <= 50.0, (
        f"p95 diff {p95_diff:.3f}ms > 50ms (clients diverged)"
    )
    assert stats_a["p95_ms"] <= 100.0
    assert stats_b["p95_ms"] <= 100.0


# --------------------------------------------------------------------------- #
# Reliability stress — 200 interleaved frames, no drops, broadcast non-blocking
# --------------------------------------------------------------------------- #
def test_latency_e2e_high_volume_200_frames() -> None:
    """200 interleaved frames; no drops, p95 ≤ 100ms, broadcast non-blocking."""
    app = create_app(start_listener=False)
    n = 200
    latencies_ms: list[float] = []
    wall_start = time.perf_counter()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry") as ws:
            for i in range(n):
                frame = _make_frame(i)
                push_frame(app, frame)
                msg = ws.receive_json()
                recv_ts = time.perf_counter()
                assert msg["type"] == "frame"
                latencies_ms.append((recv_ts - msg["_send_ts"]) * 1000.0)
    wall_s = time.perf_counter() - wall_start

    assert len(latencies_ms) == n, (
        f"expected {n} frames, received {len(latencies_ms)} (drop-oldest fired?)"
    )
    stats = _stats(latencies_ms)
    _print_stats("high_volume_200", stats)
    print(f"[latency-e2e] high_volume_200 wall={wall_s:.3f}s")
    assert stats["p95_ms"] <= 100.0, (
        f"p95 latency {stats['p95_ms']:.3f}ms > 100ms budget"
    )
    # Broadcast never blocked: 200 frames complete in well under 10s wall clock
    # (200 × 100ms worst-case per-frame = 20s; non-blocking should be << that).
    assert wall_s < 10.0, f"wall {wall_s:.3f}s suggests broadcast blocked"
