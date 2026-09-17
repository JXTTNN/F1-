"""桌面遥测接收器测试（task-81）。

覆盖：
1. 包计数（含未知类型/畸形包）与 Recorder 转发
2. 监听/收集生命周期（真实 UDP 端到端）
3. TrainingExporter 重放导出（逐圈样本结构）
4. 负向验证：无圈号变化 → 0 样本（证明固化逻辑被测试覆盖）

测试全部使用临时数据目录（隔离约定），不触碰真实 ``data/``。
"""

from __future__ import annotations

import json
import socket
import struct
import time
from pathlib import Path

from setup_tuner.collector import CollectorApp, TrainingExporter
from setup_tuner.telemetry.packets import (
    _LAP_PER_STRUCT,
    _SETUP_PER_STRUCT,
    HEADER_FORMAT,
)
from setup_tuner.telemetry.recorder import TelemetryRecorder

_SESSION_UID = 0x1234_5678_9ABC_DEF0


# ===========================================================================
# 辅助：构造最小合法 UDP 包（玩家车 index=0，玩家段 + 零填充）
# ===========================================================================
def _header(packet_id: int, session_time: float = 1.0) -> bytes:
    return struct.pack(
        HEADER_FORMAT, 2026, 26, 1, 0, 1, packet_id,
        _SESSION_UID, session_time, 1, 1, 0, 255,
    )


def _session_packet(track_id: int = 3) -> bytes:
    """合法 Packet 1（Session），trackId 可指定（复用 test_recorder 的构造）。"""
    header = _header(1)
    prefix = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        0, 25, 22, 58, 5807, 0, track_id, 1, 1800, 3600, 60, 0, 0, 0, 0, 21,
    )
    marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
    return header + prefix + marshal_zones + struct.pack("<BBB", 0, 0, 0)


def _lap_packet(
    lap_num: int, *, last_lap_ms: int = 0, invalid: int = 0, sector: int = 0,
) -> bytes:
    """Packet 2（LapData）：玩家段 57 字节，可写圈号/上圈用时/无效标志。

    玩家段布局（``_LAP_PER_FMT``，size=57）：II(8) + HBHBHBHB(12) + fff(12)
    + 19×B + HHB(5) + fB(5)。按解析数组序：c[13]=carPosition → 19B 区块
    第 0 字节（段内偏移 32），c[14]=currentLapNum → 偏移 33，c[17]=sector
    → 偏移 36，c[18]=currentLapInvalid → 偏移 37。
    """
    assert _LAP_PER_STRUCT.size == 57
    body = bytearray(_LAP_PER_STRUCT.size)
    struct.pack_into("<I", body, 0, last_lap_ms)
    body[33] = lap_num
    body[36] = sector
    body[37] = invalid
    return _header(2) + bytes(body)


def _setup_packet(brake_bias: int = 55) -> bytes:
    """Packet 5（CarSetups）：玩家段 50 字节，brakeBias 写在偏移 27。"""
    assert _SETUP_PER_STRUCT.size == 50
    body = bytearray(_SETUP_PER_STRUCT.size)
    body[27] = brake_bias
    return _header(5) + bytes(body)


def _telemetry_packet(speed_kmh: int = 180) -> bytes:
    """Packet 6（CarTelemetry）：玩家段 59 字节，m_speed 写在前 2 字节。"""
    body = bytearray(59)
    struct.pack_into("<H", body, 0, speed_kmh)
    return _header(6) + bytes(body)


def _unknown_packet() -> bytes:
    """packet_id=9（LobbyInfo，解析器不支持）—— raw 应留存、计数应有名。"""
    return _header(9) + b"\x00" * 16


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# ===========================================================================
# 1. 计数与转发（单元，直接喂回调）
# ===========================================================================
class TestCounting:
    def test_counts_by_type(self, tmp_path: Path) -> None:
        app = CollectorApp(str(tmp_path / "recordings"))
        for _ in range(2):
            app._on_raw(_session_packet(track_id=3), None)
        app._on_raw(_lap_packet(1), None)
        app._on_raw(_unknown_packet(), None)
        st = app.status()
        assert st["counts"] == {"Session": 2, "LapData": 1, "LobbyInfo": 1}
        assert st["packets_total"] == 4

    def test_malformed_packet_counts_unknown(self, tmp_path: Path) -> None:
        app = CollectorApp(str(tmp_path / "recordings"))
        app._on_raw(_header(6)[:10], None)  # 短头
        st = app.status()
        assert st["counts"] == {"unknown": 1}

    def test_raw_forwarded_to_recorder(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "recordings"
        app = CollectorApp(str(data_dir))
        app.start_collect()
        for _ in range(3):
            app._on_raw(_session_packet(), None)
        summary = app.stop_collect()
        assert summary["packet_count"] == 3
        assert Path(summary["f1rec_path"]).exists()
        assert Path(summary["db_path"]).exists()
        app.shutdown()

    def test_parsed_updates_latest(self, tmp_path: Path) -> None:
        app = CollectorApp(str(tmp_path / "recordings"))
        parsed = {
            "packet_id": 1, "m_trackId": 7, "m_totalLaps": 5,
            "m_sessionType": 10,
        }
        app._on_parsed(parsed)
        latest = app.status()["latest"]
        assert latest["m_trackId"] == 7
        # 非摘要包不更新
        app._on_parsed({"packet_id": 9})
        assert app.status()["latest"]["m_trackId"] == 7


# ===========================================================================
# 2. 生命周期（真实 UDP 端到端）
# ===========================================================================
class TestLifecycle:
    def test_listen_collect_stop_cycle(self, tmp_path: Path) -> None:
        port = _free_udp_port()
        data_dir = tmp_path / "recordings"
        app = CollectorApp(str(data_dir), port=port)
        app.listen()
        assert app.status()["listening"] is True

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        target = ("127.0.0.1", port)
        for _ in range(5):
            sock.sendto(_session_packet(track_id=3), target)
        assert _wait_until(
            lambda: app.status()["counts"].get("Session", 0) >= 5,
        ), "监听线程未在超时内收到 5 个 Session 包"

        app.start_collect()
        for _ in range(3):
            sock.sendto(_lap_packet(lap_num=1), target)
        assert _wait_until(
            lambda: app.status()["recording"] is True,
        )
        assert _wait_until(
            lambda: app.status()["counts"].get("LapData", 0) >= 3,
        )
        summary = app.stop_collect()
        assert summary["packet_count"] == 3
        assert Path(summary["f1rec_path"]).exists()
        assert Path(summary["db_path"]).exists()

        app.stop_listen()
        assert app.status()["listening"] is False
        app.shutdown()
        sock.close()


# ===========================================================================
# 3. TrainingExporter（录制重放 → 逐圈 JSONL）
# ===========================================================================
class TestTrainingExporter:
    def _record_two_laps(self, tmp_path: Path) -> Path:
        """录制合成流：session + setup + 圈 1（若干帧）→ 圈 2 触发固化。"""
        recorder = TelemetryRecorder(str(tmp_path / "recordings"))
        recorder.start()
        raws = [
            _session_packet(track_id=3),
            _setup_packet(brake_bias=55),
            _lap_packet(lap_num=1, invalid=0),
            _telemetry_packet(speed_kmh=180),
            _telemetry_packet(speed_kmh=250),
            _telemetry_packet(speed_kmh=90),
            _lap_packet(lap_num=1),          # 同圈：不触发固化
            _lap_packet(lap_num=2, last_lap_ms=91234),  # 圈号变化：固化圈 1
            _telemetry_packet(speed_kmh=200),
            _lap_packet(lap_num=3, last_lap_ms=88_500),  # 固化圈 2
        ]
        for raw in raws:
            recorder.on_raw_packet(raw, None)
        return recorder.stop(), tmp_path

    def test_export_produces_lap_samples(self, tmp_path: Path) -> None:
        summary, _ = self._record_two_laps(tmp_path)
        out = tmp_path / "laps.jsonl"
        stats = TrainingExporter().export(summary["f1rec_path"], out)
        assert stats["laps"] == 2, f"应导出 2 个完整圈样本：{stats}"
        assert stats["packets"] == 10
        assert stats["parse_errors"] == 0

        samples = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert len(samples) == 2

        lap1, lap2 = samples
        assert lap1["track_id"] == 3
        assert lap1["lap_number"] == 1
        assert lap1["lap_time_ms"] == 91234
        assert lap1["lap_valid"] is True
        assert lap1["setup"]["m_brakeBias"] == 55
        assert isinstance(lap1["style"], list) and lap1["style"]
        assert lap1["frames"] == 3  # 3 帧 telemetry
        assert lap1["session_uid"] == str(_SESSION_UID)

        assert lap2["lap_number"] == 2
        assert lap2["lap_time_ms"] == 88_500
        assert lap2["frames"] == 1

    def test_export_skips_malformed_packets(self, tmp_path: Path) -> None:
        summary, _ = self._record_two_laps(tmp_path)
        # 追加坏包：头 29 字节但包体截断（PacketTooShortError 路径）
        recorder = TelemetryRecorder(str(tmp_path / "recordings2"))
        recorder.start()
        recorder.on_raw_packet(_header(6) + b"\x00", None)  # 59 字节只给了 1
        bad_summary = recorder.stop()

        stats = TrainingExporter().export(bad_summary["f1rec_path"], tmp_path / "b.jsonl")
        assert stats["parse_errors"] == 1
        assert stats["laps"] == 0
        assert (tmp_path / "b.jsonl").read_text(encoding="utf-8") == ""

    def test_export_no_completed_laps(self, tmp_path: Path) -> None:
        """负向验证：圈号恒定 → 无固化 → 0 样本（证明固化逻辑被覆盖）。"""
        recorder = TelemetryRecorder(str(tmp_path / "recordings3"))
        recorder.start()
        raws = [
            _session_packet(track_id=3),
            _lap_packet(lap_num=1),
            _telemetry_packet(speed_kmh=180),
            _lap_packet(lap_num=1),  # 圈号不变
            _telemetry_packet(speed_kmh=200),
        ]
        for raw in raws:
            recorder.on_raw_packet(raw, None)
        summary = recorder.stop()

        out = tmp_path / "laps.jsonl"
        stats = TrainingExporter().export(summary["f1rec_path"], out)
        assert stats["laps"] == 0
        assert stats["packets"] == 5
        assert out.read_text(encoding="utf-8") == ""

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        summary, _ = self._record_two_laps(tmp_path)
        out = tmp_path / "deep" / "nested" / "laps.jsonl"
        stats = TrainingExporter().export(summary["f1rec_path"], out)
        assert out.exists()
        assert stats["laps"] == 2


# ===========================================================================
# 4. 状态快照
# ===========================================================================
class TestStatus:
    def test_status_keys_complete(self, tmp_path: Path) -> None:
        app = CollectorApp(str(tmp_path / "recordings"))
        st = app.status()
        for key in (
            "listening", "recording", "session_id", "data_dir", "host",
            "port", "packets_total", "counts", "last_summary", "latest",
        ):
            assert key in st, f"status 缺少 {key}"
        assert st["listening"] is False
        assert st["recording"] is False
        # 回归防线：桌面接收器必须绑所有网卡——F1 游戏开「UDP 广播模式」时包
        # 发往 255.255.255.255，只绑 127.0.0.1 会 0 包（真实踩坑）。
        assert st["host"] == "0.0.0.0"

    def test_shutdown_idempotent(self, tmp_path: Path) -> None:
        app = CollectorApp(str(tmp_path / "recordings"))
        app.listen()
        app.start_collect()
        app.shutdown()
        app.shutdown()  # 二次调用不炸
        st = app.status()
        assert st["listening"] is False
        assert st["recording"] is False
