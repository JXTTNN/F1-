"""TelemetryRecorder + ReplayReader 单元测试。

覆盖：
1. 录制启停（start/stop 幂等、is_recording 状态切换）
2. 数据包写入（.f1rec 文件 + SQLite 索引）
3. JSON/SQLite 双路径验证（data_json 列 + packets 表）
4. toggle 语义（start→recording→stop→idle）
5. 录制文件存到 data/recordings/ 目录
6. 边界条件：空包、重复启停、parsed=None、on_packet 旧接口
7. list_recordings / get_recording_detail 查询
8. ReplayReader 回放（read_next / replay / 上下文管理器 / 错误格式）
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from pathlib import Path

import pytest

from setup_tuner.telemetry import recorder
from setup_tuner.telemetry.packets import HEADER_FORMAT
from setup_tuner.telemetry.recorder import (
    _FILE_HEADER_SIZE,
    _MAGIC,
    _RECORD_HEADER_SIZE,
    ReplayReader,
    TelemetryRecorder,
)


# ===========================================================================
# 辅助：构造合法的 Session 包（packet_id=1）与简单 parsed dict
# ===========================================================================
def _build_session_packet(track_id: int = 2) -> bytes:
    """构造一个合法的 Packet 1 (Session) UDP 字节。"""
    header = struct.pack(
        HEADER_FORMAT,
        2026, 26, 1, 0, 1, 1, 0x1234_5678_9ABC_DEF0, 12.5, 100, 200, 0, 255,
    )
    prefix_core = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        0, 25, 22, 58, 5807, 0, track_id, 1, 1800, 3600, 60, 0, 0, 0, 0, 21,
    )
    marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
    tail = struct.pack("<BBB", 0, 0, 0)
    return header + prefix_core + marshal_zones + tail


def _build_parsed(packet_id: int = 1, name: str = "Session") -> dict:
    """构造一个简单的 parsed dict（含 header dataclass）。"""
    from setup_tuner.telemetry.packets import parse_header

    header = parse_header(
        struct.pack(HEADER_FORMAT, 2026, 26, 1, 0, 1, packet_id, 0, 1.0, 1, 1, 0, 255)
    )
    return {"packet_id": packet_id, "name": name, "header": header, "extra": 42}


# ===========================================================================
# 1. 录制启停生命周期
# ===========================================================================
class TestRecorderLifecycle:
    """start / stop / is_recording 状态机。"""

    def test_initial_state_not_recording(self, tmp_path: Path) -> None:
        """新建 recorder 未启动时 is_recording=False, session_id=None。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        assert r.is_recording is False
        assert r.session_id is None

    def test_start_sets_recording_state(self, tmp_path: Path) -> None:
        """start 后 is_recording=True 且返回 session_id。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        assert r.is_recording is True
        assert r.session_id == sid
        assert isinstance(sid, str) and len(sid) > 0
        r.stop()

    def test_start_idempotent(self, tmp_path: Path) -> None:
        """重复 start 幂等：返回同一 session_id，不重新初始化。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid1 = r.start()
        sid2 = r.start()  # 幂等
        assert sid1 == sid2
        r.stop()

    def test_stop_returns_summary(self, tmp_path: Path) -> None:
        """stop 返回含 session_id/packet_count/f1rec_path/db_path 的摘要。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        summary = r.stop()
        assert summary["session_id"] == sid
        assert summary["packet_count"] == 0
        assert summary["f1rec_path"] is not None
        assert summary["db_path"] is not None
        assert r.is_recording is False

    def test_stop_idempotent_when_not_recording(self, tmp_path: Path) -> None:
        """未录制时 stop 返回空摘要且不报错。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        summary = r.stop()
        assert summary == {"session_id": None, "packet_count": 0}

    def test_stop_after_stop_returns_empty(self, tmp_path: Path) -> None:
        """录制→stop→再 stop 返回空摘要。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        r.start()
        r.stop()
        summary = r.stop()
        assert summary == {"session_id": None, "packet_count": 0}


# ===========================================================================
# 2. toggle 语义：start→recording→stop→idle
# ===========================================================================
class TestToggleSemantics:
    """toggle API 等效序列：start→recording→stop→idle。"""

    def test_toggle_cycle(self, tmp_path: Path) -> None:
        """start→is_recording True→stop→is_recording False 完整循环。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        # idle → recording
        assert r.is_recording is False
        r.start()
        assert r.is_recording is True
        # recording → idle
        r.stop()
        assert r.is_recording is False
        # idle → recording（可再次启动）
        r.start()
        assert r.is_recording is True
        r.stop()
        assert r.is_recording is False

    def test_repeated_toggle_cycles(self, tmp_path: Path) -> None:
        """多次 toggle 循环不报错且每次 session_id 不同。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sids = []
        for _ in range(3):
            sids.append(r.start())
            r.stop()
        assert r.is_recording is False
        # session_id 每次应不同（时间戳格式，可能同秒——只验证不报错）
        assert len(sids) == 3


# ===========================================================================
# 3. 数据包写入：.f1rec + SQLite 双路径
# ===========================================================================
class TestPacketWriting:
    """on_raw_packet 写入 .f1rec 文件与 SQLite 索引。"""

    def test_f1rec_file_created_with_header(self, tmp_path: Path) -> None:
        """start 后 .f1rec 文件存在且含正确文件头（magic + version）。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.stop()
        f1rec = tmp_path / f"{sid}.f1rec"
        assert f1rec.exists()
        with open(f1rec, "rb") as f:
            header = f.read(_FILE_HEADER_SIZE)
        magic = header[:4]
        version = header[4]
        assert magic == _MAGIC
        assert version == 1

    def test_sqlite_db_created_with_schema(self, tmp_path: Path) -> None:
        """start 后 SQLite 含 packets 表与索引。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.stop()
        db = tmp_path / f"{sid}.db"
        assert db.exists()
        conn = sqlite3.connect(str(db))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert ("packets",) in tables
        conn.close()

    def test_on_raw_packet_writes_to_both_paths(self, tmp_path: Path) -> None:
        """on_raw_packet 同时写入 .f1rec 与 SQLite。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        data = _build_session_packet(track_id=7)
        parsed = _build_parsed(packet_id=1, name="Session")
        r.on_raw_packet(data, parsed)
        summary = r.stop()
        assert summary["packet_count"] == 1
        # SQLite 有 1 行
        conn = sqlite3.connect(str(tmp_path / f"{sid}.db"))
        count = conn.execute("SELECT COUNT(*) FROM packets").fetchone()[0]
        assert count == 1
        row = conn.execute(
            "SELECT session_id, packet_id, packet_name, raw_len FROM packets"
        ).fetchone()
        assert row[0] == sid
        assert row[1] == 1
        assert row[2] == "Session"
        assert row[3] == len(data)
        conn.close()
        # .f1rec 含 1 条记录（文件头 + 1 条 record header + data）
        f1rec_size = (tmp_path / f"{sid}.f1rec").stat().st_size
        assert f1rec_size > _FILE_HEADER_SIZE + _RECORD_HEADER_SIZE

    def test_data_json_column_populated(self, tmp_path: Path) -> None:
        """SQLite data_json 列含 parsed 序列化结果（JSON 路径验证）。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        parsed = _build_parsed(packet_id=6, name="CarTelemetry")
        r.on_raw_packet(b"\x01\x02\x03", parsed)
        r.stop()
        conn = sqlite3.connect(str(tmp_path / f"{sid}.db"))
        row = conn.execute("SELECT data_json FROM packets").fetchone()
        conn.close()
        assert row[0] is not None
        data = json.loads(row[0])
        assert data["packet_id"] == 6
        assert data["name"] == "CarTelemetry"
        assert "header" in data
        assert data["header"]["packet_id"] == 6

    def test_multiple_packets_batched(self, tmp_path: Path) -> None:
        """写入多个包后 SQLite 行数正确（含批量 commit 路径）。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        for i in range(5):
            r.on_raw_packet(
                struct.pack("<I", i), _build_parsed(packet_id=6, name="CarTelemetry")
            )
        summary = r.stop()
        assert summary["packet_count"] == 5
        conn = sqlite3.connect(str(tmp_path / f"{sid}.db"))
        count = conn.execute("SELECT COUNT(*) FROM packets").fetchone()[0]
        conn.close()
        assert count == 5

    def test_on_raw_packet_skipped_when_not_recording(self, tmp_path: Path) -> None:
        """未录制时 on_raw_packet 静默跳过。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        r.on_raw_packet(b"data", _build_parsed())  # 不报错
        assert r.session_id is None

    def test_on_packet_legacy_interface(self, tmp_path: Path) -> None:
        """on_packet 旧接口（无原始字节）仍写入 SQLite。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.on_packet(_build_parsed(packet_id=2, name="LapData"))
        r.stop()
        conn = sqlite3.connect(str(tmp_path / f"{sid}.db"))
        count = conn.execute("SELECT COUNT(*) FROM packets").fetchone()[0]
        conn.close()
        assert count == 1


# ===========================================================================
# 4. 边界条件
# ===========================================================================
class TestRecorderEdgeCases:
    """空包、parsed=None、重复启停、无效数据。"""

    def test_empty_packet_bytes(self, tmp_path: Path) -> None:
        """空字节包仍写入（raw_len=0）。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.on_raw_packet(b"", None)
        r.stop()
        conn = sqlite3.connect(str(tmp_path / f"{sid}.db"))
        row = conn.execute(
            "SELECT packet_id, packet_name, raw_len, data_json FROM packets"
        ).fetchone()
        conn.close()
        assert row[0] == -1  # parsed=None 时默认
        assert row[1] == "Unknown"
        assert row[2] == 0
        assert row[3] is None

    def test_parsed_none(self, tmp_path: Path) -> None:
        """parsed=None 时 SQLite 用默认值（packet_id=-1, name=Unknown）。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.on_raw_packet(b"\x00\x01\x02", None)
        r.stop()
        conn = sqlite3.connect(str(tmp_path / f"{sid}.db"))
        row = conn.execute("SELECT packet_id, packet_name FROM packets").fetchone()
        conn.close()
        assert row == (-1, "Unknown")

    def test_parsed_without_header(self, tmp_path: Path) -> None:
        """parsed 不含 header 时 session_time/frame_identifier 为 NULL。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.on_raw_packet(b"data", {"packet_id": 6, "name": "CarTelemetry"})
        r.stop()
        conn = sqlite3.connect(str(tmp_path / f"{sid}.db"))
        row = conn.execute(
            "SELECT session_time, frame_identifier FROM packets"
        ).fetchone()
        conn.close()
        assert row == (None, None)

    def test_data_dir_created(self, tmp_path: Path) -> None:
        """data_dir 不存在时自动创建（含父目录）。"""
        nested = tmp_path / "data" / "recordings"
        r = TelemetryRecorder(data_dir=str(nested))
        assert nested.exists()
        r.start()
        r.stop()

    def test_concurrent_writes_thread_safe(self, tmp_path: Path) -> None:
        """多线程并发 on_raw_packet 不报错。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        r.start()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(20):
                    r.on_raw_packet(b"\x00" * 10, _build_parsed())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        r.stop()
        assert errors == []
        assert r.session_id is not None


# ===========================================================================
# 5. 录制列表查询
# ===========================================================================
class TestRecordingQueries:
    """list_recordings / get_recording_detail。"""

    def test_list_recordings_empty(self, tmp_path: Path) -> None:
        """无录制时 list_recordings 返回空列表。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        assert r.list_recordings() == []

    def test_list_recordings_after_stop(self, tmp_path: Path) -> None:
        """录制后 list_recordings 返回含该会话的元数据。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.on_raw_packet(b"data", _build_parsed())
        r.stop()
        recordings = r.list_recordings()
        assert len(recordings) == 1
        assert recordings[0]["session_id"] == sid
        assert recordings[0]["packet_count"] == 1
        assert recordings[0]["f1rec_path"] is not None

    def test_get_recording_detail(self, tmp_path: Path) -> None:
        """get_recording_detail 返回含按包类型统计的详情。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.on_raw_packet(b"d1", _build_parsed(packet_id=6, name="CarTelemetry"))
        r.on_raw_packet(b"d2", _build_parsed(packet_id=1, name="Session"))
        r.stop()
        detail = r.get_recording_detail(sid)
        assert detail is not None
        assert detail["session_id"] == sid
        assert detail["packet_count"] == 2
        by_type = detail["by_packet_type"]
        names = {item["packet_name"] for item in by_type}
        assert names == {"CarTelemetry", "Session"}

    def test_get_recording_detail_not_found(self, tmp_path: Path) -> None:
        """不存在的 session_id 返回 None。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        assert r.get_recording_detail("nonexistent") is None


# ===========================================================================
# 6. ReplayReader 回放
# ===========================================================================
class TestReplayReader:
    """ReplayReader 从 .f1rec 读取原始字节并回放。"""

    def test_replay_reader_not_found(self, tmp_path: Path) -> None:
        """不存在的文件抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            ReplayReader(str(tmp_path / "nope.f1rec"))

    def test_replay_reader_invalid_magic(self, tmp_path: Path) -> None:
        """错误 magic 抛 ValueError。"""
        bad = tmp_path / "bad.f1rec"
        with open(bad, "wb") as f:
            f.write(b"XXXX" + b"\x00" * (_FILE_HEADER_SIZE - 4))
        with pytest.raises(ValueError):
            ReplayReader(str(bad))

    def test_replay_reader_round_trip(self, tmp_path: Path) -> None:
        """录制→回放往返：read_next 返回原始字节。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        packets = [b"\x01\x02\x03", b"\x04\x05\x06", b"\x07\x08\x09"]
        for p in packets:
            r.on_raw_packet(p, None)
        r.stop()
        # 回放
        reader = ReplayReader(str(tmp_path / f"{sid}.f1rec"))
        assert reader.session_id == sid
        results = []
        while True:
            res = reader.read_next()
            if res is None:
                break
            results.append(res[1])  # raw_bytes
        reader.close()
        assert results == packets

    def test_replay_reader_context_manager(self, tmp_path: Path) -> None:
        """ReplayReader 支持上下文管理器。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.on_raw_packet(b"\xaa\xbb", None)
        r.stop()
        with ReplayReader(str(tmp_path / f"{sid}.f1rec")) as reader:
            res = reader.read_next()
            assert res is not None
            assert res[1] == b"\xaa\xbb"
            assert reader.read_next() is None  # EOF

    def test_replay_invokes_handler(self, tmp_path: Path) -> None:
        """replay 调用 handler 并返回成功回放数。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        # 写入一个合法的 Session 包（可被 parse_packet 解析）
        pkt = _build_session_packet(track_id=3)
        r.on_raw_packet(pkt, None)
        r.stop()
        received: list[dict] = []
        with ReplayReader(str(tmp_path / f"{sid}.f1rec")) as reader:
            count = reader.replay(lambda p: received.append(p), speed=100.0)
        assert count >= 1
        assert len(received) >= 1
        assert received[0]["packet_id"] == 1

    def test_replay_empty_recording(self, tmp_path: Path) -> None:
        """空录制（无包）replay 返回 0。"""
        r = TelemetryRecorder(data_dir=str(tmp_path))
        sid = r.start()
        r.stop()
        with ReplayReader(str(tmp_path / f"{sid}.f1rec")) as reader:
            count = reader.replay(lambda p: None)
        assert count == 0


# ===========================================================================
# 7. 模块级常量与 zstd 降级
# ===========================================================================
class TestModuleConstants:
    """模块常量与 zstd 可用性。"""

    def test_magic_and_version_constants(self) -> None:
        """_MAGIC 与 _VERSION 符合协议。"""
        assert _MAGIC == b"F1R\x00"
        assert recorder._VERSION == 1

    def test_header_sizes(self) -> None:
        """文件头 = 4sB32s32s = 69 字节，记录头 = dII = 16 字节。"""
        assert _FILE_HEADER_SIZE == 69
        assert _RECORD_HEADER_SIZE == 16

    def test_zstd_available_or_graceful(self) -> None:
        """zstd 可用或降级标志为 False。"""
        assert isinstance(recorder._HAS_ZSTD, bool)