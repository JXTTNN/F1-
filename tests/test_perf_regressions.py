"""性能回归测试 —— 锁住"第一批性能优化"的关键性质。

覆盖：
1. WS 推送为**应用级单一任务**，每 tick 只 broadcast 一次（防止 N×N 扇出回潮）；
2. 录制 `on_raw_packet` 只入队、不阻塞接收线程，`stop()` 会先排空再关文件；
3. `index_json=False` 时跳过整包 JSON 序列化（`data_json` 为 NULL）；
4. `TelemetryStream.get_all_latest()` 对帧内一维 list 做值拷贝（改动不再污染缓存）。
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from setup_tuner.api import ws as ws_mod
from setup_tuner.telemetry.recorder import TelemetryRecorder
from setup_tuner.telemetry.stream import TelemetryStream

_N_CONNECTIONS = 8


class _FakeStream:
    """最小遥测流：只提供 CarTelemetry 最新帧。"""

    def __init__(self) -> None:
        self.snapshot_calls = 0

    def get_all_latest(self) -> dict[int, dict]:
        self.snapshot_calls += 1
        return {6: {"m_speed": 280, "m_gear": 7}}


class _FakeListener:
    is_running = True


class _CountingManager:
    """记录 broadcast 次数与总发送次数（模拟 N 个连接）。"""

    def __init__(self, connections: int) -> None:
        self.connections = connections
        self.broadcast_calls = 0
        self.sends = 0

    @property
    def connection_count(self) -> int:
        return self.connections

    async def broadcast(self, event: str, payload: dict) -> None:
        self.broadcast_calls += 1
        self.sends += self.connections


class _State:
    """最小 app.state 替身。"""


def test_pusher_is_single_task_and_broadcasts_once_per_tick() -> None:
    """WS 推送必须是单一任务，且每 tick 只 broadcast 一次（O(N) 而非 O(N²)）。"""

    async def _run() -> tuple[int, int, int, bool]:
        state = _State()
        stream = _FakeStream()
        state.telemetry_stream = stream
        state.telemetry_listener = _FakeListener()
        manager = _CountingManager(_N_CONNECTIONS)
        state.ws_manager = manager

        ws_mod._ensure_pusher(state)
        first_task = state.ws_pusher_task
        # 幂等：重复 ensure 不会起第二个推送任务
        ws_mod._ensure_pusher(state)
        assert state.ws_pusher_task is first_task, "重复 ensure 起了第二个推送任务"

        await asyncio.sleep(ws_mod._THROTTLE_INTERVAL_SEC * 4)

        broadcasts = manager.broadcast_calls
        sends = manager.sends
        snapshots = stream.snapshot_calls
        # 最后一个连接断开后应停掉推送任务
        manager.connections = 0
        await ws_mod._stop_pusher_if_idle(state, manager)
        return broadcasts, sends, snapshots, state.ws_pusher_task is None

    broadcasts, sends, snapshots, stopped = asyncio.run(_run())

    # 至少推了一帧
    assert snapshots >= 1, "推送任务没有取过快照"
    assert broadcasts >= 1, "推送任务没有产生任何 broadcast"
    # 旧实现：N 个连接 → 每 tick N 次 broadcast（每次又发给 N 个连接）= N² 次发送。
    # 新实现：每个推送 tick 至多 1 次 telemetry broadcast（+ 至多 1 次状态变化 broadcast），
    # 因此 broadcast 次数应与快照次数同量级，而不是它的 N 倍。
    assert broadcasts <= snapshots + 2, (
        f"每 tick broadcast 次数过高：broadcasts={broadcasts}, snapshots={snapshots}"
    )
    # 扇出 = broadcast 次数 × 连接数（说明扇出由 broadcast 一次性完成）
    assert sends == broadcasts * _N_CONNECTIONS
    assert stopped, "无连接时推送任务未停止"


def test_pusher_not_started_for_zero_connections() -> None:
    """没有客户端时推送任务应立即结束自己的空转（不产生 broadcast）。"""

    async def _run() -> int:
        state = _State()
        state.telemetry_stream = _FakeStream()
        state.telemetry_listener = _FakeListener()
        manager = _CountingManager(0)
        state.ws_manager = manager
        ws_mod._ensure_pusher(state)
        await asyncio.sleep(ws_mod._THROTTLE_INTERVAL_SEC * 3)
        await ws_mod._stop_pusher_if_idle(state, manager)
        return manager.broadcast_calls

    assert asyncio.run(_run()) == 0


def test_recorder_enqueue_does_not_block_and_stop_drains(tmp_path: Path) -> None:
    """on_raw_packet 只入队；stop 会等队列排空后才关闭文件与数据库。"""
    recorder = TelemetryRecorder(data_dir=str(tmp_path))
    session_id = recorder.start()

    packets = 60
    payload = b"\x01" * 1200
    parsed = {"packet_id": 6, "name": "CarTelemetry"}
    for _ in range(packets):
        recorder.on_raw_packet(payload, parsed)

    summary = recorder.stop()
    assert summary["packet_count"] == packets
    assert summary["dropped_count"] == 0

    conn = sqlite3.connect(str(tmp_path / f"{session_id}.db"))
    try:
        count = conn.execute("SELECT COUNT(*) FROM packets").fetchone()[0]
    finally:
        conn.close()
    assert count == packets, "stop() 未把队列中的包全部落库"

    # .f1rec 里应有 60 条记录（每条约 16 字节头 + 压缩数据）
    size = (tmp_path / f"{session_id}.f1rec").stat().st_size
    assert size > 69 + 60 * 16, f".f1rec 过小，可能漏写：{size}"


def test_recorder_index_json_off_skips_serialization(tmp_path: Path) -> None:
    """index_json=False 时 data_json 应为 NULL（省掉整包 JSON 序列化）。"""
    recorder = TelemetryRecorder(data_dir=str(tmp_path), index_json=False)
    session_id = recorder.start()
    recorder.on_raw_packet(b"\x02\x03", {"packet_id": 6, "name": "CarTelemetry"})
    recorder.stop()

    conn = sqlite3.connect(str(tmp_path / f"{session_id}.db"))
    try:
        row = conn.execute("SELECT data_json FROM packets").fetchone()
    finally:
        conn.close()
    assert row is not None and row[0] is None


def test_stream_snapshot_copies_nested_lists() -> None:
    """get_all_latest 必须拷贝帧内一维 list：调用方修改不得污染缓存。"""
    stream = TelemetryStream()
    stream.update(6, {"m_speed": 100, "m_tyresPressure": [23.0, 23.0, 21.0, 21.0]})

    snapshot = stream.get_all_latest()
    snapshot[6]["m_tyresPressure"][0] = 999.0

    again = stream.get_latest(6)
    assert again is not None
    assert again["m_tyresPressure"][0] == 23.0, "缓存被调用方的原地修改污染"
