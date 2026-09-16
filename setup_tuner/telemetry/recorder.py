"""遥测录制器 — 无损原始字节录制 + zstd 压缩 + replay 回放。

录制策略（参考 f126-race-engineer 最佳实践）：
- **无损录制**：保存原始 UDP 字节（zstd 压缩），而非解析后的 JSON dict。
  原始字节是协议的完整快照，可随时用更新版的解析器重新解析，
  不受录制时解析器版本/bug 影响。
- **SQLite 索引**：每包一行，存解析后的关键字段（packet_id, session_time,
  frame_identifier），用于查询统计。data_json 列保留 JSON 序列化结果
  供调试，但主数据源是 .f1rec 文件。
- **批量写入**：SQLite 每 100 包或 2 秒 commit 一次，避免 60Hz 下每包
  commit 的性能问题。
- **回放**：ReplayReader 从 .f1rec 文件读取原始字节，按时间戳回放，
  支持倍速、跳转。

文件格式 .f1rec：
    [File Header]
      magic: 4 bytes "F1R\\x00"
      version: uint8 (1)
      session_id: 32 bytes (UTF-8, null-padded)
      start_time: 32 bytes (ISO format, null-padded)
    [Packet Records (repeating)]
      timestamp: float64 (seconds since recording start)
      raw_len: uint32 (uncompressed byte length)
      compressed_len: uint32 (zstd compressed byte length)
      compressed_data: bytes[compressed_len]
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import struct
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
    zstd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
_MAGIC = b"F1R\x00"
_VERSION = 1
_FILE_HEADER_FMT = "<4sB32s32s"
_FILE_HEADER_SIZE = struct.calcsize(_FILE_HEADER_FMT)  # 73
_RECORD_HEADER_FMT = "<dII"
_RECORD_HEADER_SIZE = struct.calcsize(_RECORD_HEADER_FMT)  # 16

# SQLite 批量 commit 参数
_BATCH_SIZE = 100
_BATCH_TIMEOUT = 2.0  # 秒

# zstd 压缩级别（1-22，3 是速度/压缩比的平衡点）
_ZSTD_LEVEL = 3

# --------------------------------------------------------------------------- #
# 录制队列与工作线程
# --------------------------------------------------------------------------- #
# 性能：zstd 压缩 + 整包 JSON 序列化 + SQLite 入库原本全部在 UDP 接收线程上
# 串行执行（实测约 82 µs/包，425 包/秒负载下占单核约 3.5%），会拖慢收包甚至丢包。
# 现在改为：接收线程只做入队（约 1 µs），压缩/序列化/落库交给独立工作线程。
_QUEUE_MAX = 20000          # 队列上限 ≈ 20 秒 @ 1000 包/秒，超出即丢弃并计数
_WORKER_JOIN_TIMEOUT = 10.0  # stop 时等待工作线程排空的上限（秒）
_DROP_LOG_INTERVAL = 1000    # 每丢弃 N 个包记一次 warning


class TelemetryRecorder:
    """遥测录制器，线程安全。

    作为 TelemetryListener 的 raw_handler 注册，收到原始 UDP 字节时：
    1. 立即入队（接收线程只做这一步，约 1 µs/包，不阻塞收包）；
    2. 由独立工作线程做 zstd 压缩后追加写入 .f1rec 文件（无损主数据源）；
    3. 同一工作线程把解析结果批量插入 SQLite（索引/统计/调试用）。

    使用方式::

        recorder = TelemetryRecorder(data_dir="data/recordings")
        listener.add_raw_handler(recorder.on_raw_packet)
        recorder.start()
        ...
        recorder.stop()   # 会等待队列排空后再关闭文件

    Args:
        data_dir: 录制产物目录。
        index_json: 是否把整包解析结果序列化进 SQLite 的 ``data_json`` 列。
            默认 True（保持既有行为）。该列只服务调试，主数据源是 .f1rec；
            高负载场景可置 False 以省掉约三分之一的单包处理耗时。
        queue_max: 待写队列上限；溢出时丢弃并累计 ``dropped_count``。
    """

    def __init__(
        self,
        data_dir: str = "data/recordings",
        *,
        index_json: bool = True,
        queue_max: int = _QUEUE_MAX,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._recording = threading.Event()
        self._lock = threading.Lock()
        self._index_json = index_json
        self._queue_max = max(1, int(queue_max))
        self._queue: queue.Queue = queue.Queue(maxsize=self._queue_max)
        self._worker: threading.Thread | None = None
        self._dropped = 0

        # .f1rec 文件
        self._f1rec_path: Path | None = None
        self._f1rec_file: Any = None
        self._zstd_compressor: Any = None

        # SQLite（索引/统计）
        self._db_path: Path | None = None
        self._db_conn: sqlite3.Connection | None = None
        self._db_batch: list[tuple] = []
        self._db_last_commit: float = 0.0

        # 会话元数据
        self._session_id: str | None = None
        self._start_time_str: str | None = None
        self._start_time_mono: float = 0.0
        self._packet_count: int = 0

    # ------------------------------------------------------------------ #
    # 录制控制
    # ------------------------------------------------------------------ #
    def _init_recording_session(self) -> None:
        """初始化录制会话的 ID、时间戳与计数器。"""
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start_time_str = datetime.now().isoformat()
        self._start_time_mono = time.monotonic()
        self._packet_count = 0
        self._db_batch = []
        self._db_last_commit = time.monotonic()

    def _open_f1rec_file(self) -> None:
        """打开 .f1rec 文件并写入文件头。"""
        self._f1rec_path = self._data_dir / f"{self._session_id}.f1rec"
        self._f1rec_file = open(self._f1rec_path, "wb", buffering=0)
        session_bytes = self._session_id.encode("utf-8")[:32].ljust(32, b"\x00")
        start_bytes = self._start_time_str.encode("utf-8")[:32].ljust(32, b"\x00")
        header = struct.pack(
            _FILE_HEADER_FMT, _MAGIC, _VERSION, session_bytes, start_bytes,
        )
        self._f1rec_file.write(header)

    def _init_zstd_compressor(self) -> None:
        """初始化 zstd 压缩器（不可用时降级为无压缩）。"""
        if _HAS_ZSTD:
            self._zstd_compressor = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
        else:
            self._zstd_compressor = None
            logger.warning("zstandard not installed, recording uncompressed")

    def _init_sqlite_db(self) -> None:
        """初始化 SQLite 索引库（含 packets 表与索引）。"""
        self._db_path = self._data_dir / f"{self._session_id}.db"
        self._db_conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS packets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                packet_id INTEGER NOT NULL,
                packet_name TEXT,
                session_time REAL,
                frame_identifier INTEGER,
                timestamp REAL NOT NULL,
                raw_len INTEGER NOT NULL,
                data_json TEXT
            )
        """)
        self._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_packets_session ON packets(session_id)",
        )
        self._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_packets_packet_id ON packets(packet_id)",
        )
        self._db_conn.commit()

    def start(self) -> str:
        """开始录制，返回会话 ID（时间戳格式）。

        幂等：已录制时直接返回当前会话 ID。
        """
        if self._recording.is_set():
            assert self._session_id is not None
            return self._session_id

        self._init_recording_session()
        self._open_f1rec_file()
        self._init_zstd_compressor()
        self._init_sqlite_db()

        # 新会话使用全新队列，避免上一次会话的残留条目被写入
        self._queue = queue.Queue(maxsize=self._queue_max)
        self._dropped = 0
        worker = threading.Thread(
            target=self._worker_loop, name="f1opt-recorder", daemon=True,
        )
        self._worker = worker
        worker.start()

        self._recording.set()
        logger.info(
            "telemetry recording started: session=%s, zstd=%s, index_json=%s",
            self._session_id, _HAS_ZSTD, self._index_json,
        )
        return self._session_id

    def stop(self) -> dict[str, Any]:
        """停止录制，返回录制摘要。

        幂等：未录制时返回空摘要。
        会先等待工作线程把队列中剩余包写完，再关闭文件与数据库连接。
        """
        if not self._recording.is_set():
            return {"session_id": None, "packet_count": 0}

        self._recording.clear()
        # 工作线程会一直消费到队列为空且录制已停止，然后自行退出
        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            worker.join(timeout=_WORKER_JOIN_TIMEOUT)
            if worker.is_alive():
                logger.warning(
                    "recorder worker 未在 %.1fs 内退出，仍有 %d 条待写",
                    _WORKER_JOIN_TIMEOUT, self._queue.qsize(),
                )

        session_id = self._session_id
        packet_count = self._packet_count
        dropped = self._dropped
        f1rec_path = str(self._f1rec_path) if self._f1rec_path else None
        db_path = str(self._db_path) if self._db_path else None

        with self._lock:
            # flush 批量
            self._flush_batch()

            if self._f1rec_file is not None:
                self._f1rec_file.close()
                self._f1rec_file = None
            if self._db_conn is not None:
                self._db_conn.close()
                self._db_conn = None

        logger.info(
            "telemetry recording stopped: session=%s, packets=%d, dropped=%d",
            session_id, packet_count, dropped,
        )
        return {
            "session_id": session_id,
            "packet_count": packet_count,
            "dropped_count": dropped,
            "f1rec_path": f1rec_path,
            "db_path": db_path,
        }

    @property
    def is_recording(self) -> bool:
        """是否正在录制。"""
        return self._recording.is_set()

    @property
    def session_id(self) -> str | None:
        """当前录制会话 ID。"""
        return self._session_id

    # ------------------------------------------------------------------ #
    # 包处理（作为 TelemetryListener 的 raw_handler 注册）
    # ------------------------------------------------------------------ #
    def _write_f1rec_record(
        self, timestamp: float, data: bytes,
    ) -> tuple[int, bytes]:
        """写入 .f1rec 单条记录（含 zstd 压缩），返回 (raw_len, compressed)。"""
        raw_len = len(data)
        if self._zstd_compressor is not None:
            compressed = self._zstd_compressor.compress(data)
        else:
            compressed = data  # 无 zstd 时直接存原始字节
        compressed_len = len(compressed)
        record_header = struct.pack(
            _RECORD_HEADER_FMT, timestamp, raw_len, compressed_len,
        )
        if self._f1rec_file is not None:
            self._f1rec_file.write(record_header)
            self._f1rec_file.write(compressed)
        return raw_len, compressed

    def _build_db_record(
        self, timestamp: float, raw_len: int, parsed: dict[str, Any] | None,
    ) -> tuple:
        """从解析结果构造 SQLite 索引记录。

        ``data_json`` 由 :attr:`index_json` 控制：关闭时写 None，
        省掉整包 JSON 序列化（约占单包处理耗时的三分之一）。
        """
        packet_id = -1
        packet_name = "Unknown"
        session_time = None
        frame_identifier = None
        data_json = None
        if parsed is not None:
            packet_id = parsed.get("packet_id", -1)
            packet_name = parsed.get("name", "Unknown")
            header = parsed.get("header")
            if header is not None:
                session_time = getattr(header, "session_time", None)
                frame_identifier = getattr(header, "frame_identifier", None)
            if self._index_json:
                # 序列化解析结果（header 是 dataclass，需特殊处理）
                serializable = self._serialize_parsed(parsed)
                data_json = json.dumps(serializable, ensure_ascii=False, default=str)
        return (
            self._session_id, packet_id, packet_name,
            session_time, frame_identifier, timestamp, raw_len, data_json,
        )

    # ------------------------------------------------------------------ #
    # 工作线程（压缩 / 序列化 / 落库，从 UDP 接收线程移出）
    # ------------------------------------------------------------------ #
    def _worker_loop(self) -> None:
        """工作线程主循环：消费队列，直到队列空且录制已停止。"""
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if not self._recording.is_set():
                    return
                continue
            try:
                timestamp, data, parsed = item
                self._process_item(timestamp, data, parsed)
            except Exception:
                logger.exception("recorder worker 处理单包失败，继续")
            finally:
                self._queue.task_done()

    def _process_item(
        self, timestamp: float, data: bytes, parsed: dict[str, Any] | None,
    ) -> None:
        """写入一条包记录（.f1rec + SQLite 批量缓冲）。"""
        with self._lock:
            raw_len, _ = self._write_f1rec_record(timestamp, data)
            self._db_batch.append(self._build_db_record(timestamp, raw_len, parsed))
            # 批量 commit
            if (len(self._db_batch) >= _BATCH_SIZE or
                    time.monotonic() - self._db_last_commit >= _BATCH_TIMEOUT):
                self._flush_batch()

    def on_raw_packet(self, data: bytes, parsed: dict[str, Any] | None) -> None:
        """收到原始 UDP 字节 + 解析结果时入队（接收线程只做这一步）。

        非录制状态静默跳过；队列满时丢弃并累计 ``dropped_count``（不阻塞收包）。

        Args:
            data: 原始 UDP 字节（无损保存）。
            parsed: 解析后的 dict（可能为 None，如未知包类型）。
        """
        if not self._recording.is_set():
            return

        item = (time.monotonic() - self._start_time_mono, data, parsed)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1
            if self._dropped % _DROP_LOG_INTERVAL == 1:
                logger.warning(
                    "录制队列已满（上限 %d），已丢弃 %d 个包",
                    self._queue_max, self._dropped,
                )
            return
        with self._lock:
            self._packet_count += 1

    def on_packet(self, parsed: dict[str, Any]) -> None:
        """兼容旧接口：只收到解析结果（无原始字节）。

        降级模式：无法保存原始字节，只写入 SQLite。
        建议使用 on_raw_packet 以获得无损录制。
        """
        if not self._recording.is_set():
            return
        # 旧接口无法获取原始字节，跳过 .f1rec 写入
        # 只写入 SQLite 索引
        self.on_raw_packet(b"", parsed)

    def _serialize_parsed(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """将解析结果序列化为 JSON 兼容的 dict（header 是 frozen dataclass）。"""
        serializable: dict[str, Any] = {}
        for k, v in parsed.items():
            if k == "header":
                serializable[k] = {
                    "packet_format": getattr(v, "packet_format", None),
                    "game_year": getattr(v, "game_year", None),
                    "game_major_version": getattr(v, "game_major_version", None),
                    "game_minor_version": getattr(v, "game_minor_version", None),
                    "packet_version": getattr(v, "packet_version", None),
                    "packet_id": getattr(v, "packet_id", None),
                    "session_uid": getattr(v, "session_uid", None),
                    "session_time": getattr(v, "session_time", None),
                    "frame_identifier": getattr(v, "frame_identifier", None),
                    "overall_frame_identifier": getattr(v, "overall_frame_identifier", None),
                    "player_car_index": getattr(v, "player_car_index", None),
                    "secondary_player_car_index": getattr(v, "secondary_player_car_index", None),
                }
            else:
                serializable[k] = v
        return serializable

    def _flush_batch(self) -> None:
        """将批量缓冲区写入 SQLite 并 commit。"""
        if not self._db_batch or self._db_conn is None:
            return
        try:
            self._db_conn.executemany(
                "INSERT INTO packets (session_id, packet_id, packet_name, "
                "session_time, frame_identifier, timestamp, raw_len, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                self._db_batch,
            )
            self._db_conn.commit()
        except Exception:
            logger.exception("DB batch write failed, continuing")
        finally:
            self._db_batch = []
            self._db_last_commit = time.monotonic()

    # ------------------------------------------------------------------ #
    # 录制列表查询
    # ------------------------------------------------------------------ #
    def _read_f1rec_header(self, f1rec_file: Path) -> tuple[str | None, int]:
        """读取 .f1rec 文件头，返回 (start_time, est_packets)；失败返回 (None, 0)。"""
        try:
            with open(f1rec_file, "rb") as f:
                header_data = f.read(_FILE_HEADER_SIZE)
                if len(header_data) < _FILE_HEADER_SIZE:
                    return None, 0
                magic, version, sid_bytes, st_bytes = struct.unpack(
                    _FILE_HEADER_FMT, header_data,
                )
                if magic != _MAGIC:
                    return None, 0
                start_time = st_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
            file_size = f1rec_file.stat().st_size
            # 粗略估算：平均每包约 200 字节（含压缩）
            est_packets = max(0, (file_size - _FILE_HEADER_SIZE) // 200)
            return start_time, est_packets
        except Exception:
            # 文件不可读/非普通文件：回落到空元数据，但必须留痕，
            # 否则录制列表会出现"包数为 0、无起始时间"而无从追溯原因
            logger.warning("读取 .f1rec 文件头失败：%s", f1rec_file, exc_info=True)
            return None, 0

    @staticmethod
    def _query_db_packet_count(db_file: Path, est_packets: int) -> int:
        """查询 SQLite 获取精确包数，失败时回退估算值。"""
        if not db_file.exists():
            return est_packets
        try:
            conn = sqlite3.connect(str(db_file), check_same_thread=False)
            row = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM packets"
            ).fetchone()
            conn.close()
            if row and row[0] > 0:
                return row[0]
        except Exception:
            # 统计失败不致命：回落到估算值，但必须留痕，
            # 否则 UI 长期显示估算值而无人察觉
            logger.warning("读取录制包数失败，回落到估算值：%s", db_file, exc_info=True)
        return est_packets

    def list_recordings(self) -> list[dict[str, Any]]:
        """列出所有录制会话（按时间倒序）。

        扫描 data_dir 下的 .f1rec 文件，读取每个会话的元数据。
        """
        recordings: list[dict[str, Any]] = []
        for f1rec_file in sorted(self._data_dir.glob("*.f1rec"), reverse=True):
            session_id = f1rec_file.stem
            start_time, est_packets = self._read_f1rec_header(f1rec_file)
            db_file = self._data_dir / f"{session_id}.db"
            packet_count = self._query_db_packet_count(db_file, est_packets)
            recordings.append({
                "session_id": session_id,
                "packet_count": packet_count,
                "f1rec_path": str(f1rec_file),
                "db_path": str(db_file) if db_file.exists() else None,
                "start_time": start_time,
                "file_size": f1rec_file.stat().st_size,
            })
        return recordings

    def _read_recording_start_time(self, f1rec_file: Path) -> str | None:
        """从 .f1rec 文件头读取会话起始时间。"""
        if not f1rec_file.exists():
            return None
        try:
            with open(f1rec_file, "rb") as f:
                header_data = f.read(_FILE_HEADER_SIZE)
                if len(header_data) < _FILE_HEADER_SIZE:
                    return None
                magic, version, sid_bytes, st_bytes = struct.unpack(
                    _FILE_HEADER_FMT, header_data,
                )
                if magic == _MAGIC:
                    return st_bytes.rstrip(b"\x00").decode(
                        "utf-8", errors="replace",
                    )
        except Exception:
            # .f1rec 头部损坏：返回 None 触发调用方回落，但记录原因
            logger.warning("读取 .f1rec 会话头部失败：%s", f1rec_file, exc_info=True)
        return None

    def _query_db_recording_stats(self, db_file: Path) -> tuple[int, float | None, list[dict[str, Any]]]:
        """从 SQLite 查询录制统计，返回 (packet_count, end_time, by_type)。"""
        packet_count = 0
        by_type: list[dict[str, Any]] = []
        end_time = None
        if not db_file.exists():
            return packet_count, end_time, by_type
        try:
            conn = sqlite3.connect(str(db_file), check_same_thread=False)
            total = conn.execute("SELECT COUNT(*) FROM packets").fetchone()
            packet_count = total[0] if total else 0

            time_range = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM packets"
            ).fetchone()
            if time_range and time_range[1] is not None:
                end_time = time_range[1]

            by_type_rows = conn.execute(
                "SELECT packet_name, COUNT(*) FROM packets GROUP BY packet_name "
                "ORDER BY COUNT(*) DESC"
            ).fetchall()
            conn.close()
            by_type = [
                {"packet_name": name, "count": count}
                for name, count in by_type_rows
            ]
        except Exception:
            # 统计查询失败：返回零值，但记录以便定位（表缺失/库损坏等）
            logger.warning("查询录制统计失败：%s", db_file, exc_info=True)
        return packet_count, end_time, by_type

    def get_recording_detail(self, session_id: str) -> dict[str, Any] | None:
        """获取单个录制会话详情（含按包类型统计）。"""
        f1rec_file = self._data_dir / f"{session_id}.f1rec"
        db_file = self._data_dir / f"{session_id}.db"

        if not f1rec_file.exists() and not db_file.exists():
            return None

        start_time = self._read_recording_start_time(f1rec_file)
        packet_count, end_time, by_type = self._query_db_recording_stats(db_file)

        return {
            "session_id": session_id,
            "packet_count": packet_count,
            "start_time": start_time,
            "end_time": end_time,
            "by_packet_type": by_type,
            "f1rec_path": str(f1rec_file) if f1rec_file.exists() else None,
            "db_path": str(db_file) if db_file.exists() else None,
        }


# --------------------------------------------------------------------------- #
# ReplayReader — 从 .f1rec 文件回放遥测数据
# --------------------------------------------------------------------------- #
class ReplayReader:
    """从 .f1rec 文件读取原始字节并回放。

    使用方式::

        reader = ReplayReader("data/recordings/20260101_120000.f1rec")
        reader.replay(handler=my_handler, speed=2.0)  # 2倍速回放
        reader.close()

    或逐条读取::

        reader = ReplayReader("data/recordings/20260101_120000.f1rec")
        while True:
            result = reader.read_next()
            if result is None:
                break
            timestamp, raw_bytes = result
            parsed = parse_packet(raw_bytes)
            ...
        reader.close()
    """

    def __init__(self, f1rec_path: str) -> None:
        self._path = Path(f1rec_path)
        if not self._path.exists():
            raise FileNotFoundError(f"recording file not found: {f1rec_path}")

        self._file = open(self._path, "rb")
        self._zstd_decompressor: Any = None

        # 读取文件头
        header_data = self._file.read(_FILE_HEADER_SIZE)
        if len(header_data) < _FILE_HEADER_SIZE:
            raise ValueError("invalid .f1rec file: header too short")

        magic, version, sid_bytes, st_bytes = struct.unpack(
            _FILE_HEADER_FMT, header_data,
        )
        if magic != _MAGIC:
            raise ValueError(f"invalid .f1rec magic: {magic!r}")

        self._version = version
        self._session_id = sid_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        self._start_time = st_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")

        if _HAS_ZSTD:
            self._zstd_decompressor = zstd.ZstdDecompressor()

        self._packet_index = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def start_time(self) -> str:
        return self._start_time

    @property
    def packet_index(self) -> int:
        return self._packet_index

    def read_next(self) -> tuple[float, bytes] | None:
        """读取下一条记录，返回 (timestamp, raw_bytes)。

        返回 None 表示文件结束。
        """
        rec_header = self._file.read(_RECORD_HEADER_SIZE)
        if len(rec_header) < _RECORD_HEADER_SIZE:
            return None  # EOF

        timestamp, raw_len, compressed_len = struct.unpack(
            _RECORD_HEADER_FMT, rec_header,
        )

        compressed_data = self._file.read(compressed_len)
        if len(compressed_data) < compressed_len:
            return None  # truncated

        if self._zstd_decompressor is not None and compressed_len != raw_len:
            raw_bytes = self._zstd_decompressor.decompress(compressed_data)
        else:
            raw_bytes = compressed_data

        self._packet_index += 1
        return (timestamp, raw_bytes)

    def replay(
        self,
        handler: Callable[[dict[str, Any]], None],
        speed: float = 1.0,
        skip_unknown: bool = True,
    ) -> int:
        """按时间戳回放录制数据。

        Args:
            handler: 收到解析后的包 dict 时调用。
            speed: 回放倍速（1.0=原速，2.0=2倍速，0.5=半速）。
            skip_unknown: 跳过无法解析的包（True=跳过，False=抛异常）。

        Returns:
            成功回放的包数。
        """

        count = 0
        prev_timestamp: float | None = None

        while True:
            result = self.read_next()
            if result is None:
                break
            timestamp, raw_bytes = result
            prev_timestamp = _wait_for_replay_step(prev_timestamp, timestamp, speed)
            parsed = _safe_parse_packet(raw_bytes, skip_unknown)
            if parsed is None:
                continue
            if _safe_invoke_handler(handler, parsed):
                count += 1

        return count

    def close(self) -> None:
        """关闭文件。"""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> ReplayReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 回放辅助函数
# ---------------------------------------------------------------------------
def _wait_for_replay_step(
    prev_timestamp: float | None, timestamp: float, speed: float,
) -> float:
    """按时间戳间隔等待（模拟原始节奏），返回当前时间戳。"""
    if prev_timestamp is not None and speed > 0:
        delta = (timestamp - prev_timestamp) / speed
        if delta > 0:
            time.sleep(min(delta, 0.1))  # 上限100ms防卡死
    return timestamp


def _safe_parse_packet(
    raw_bytes: bytes, skip_unknown: bool,
) -> dict[str, Any] | None:
    """安全解析单个包；失败时根据 skip_unknown 决定跳过或抛异常。

    Returns:
        解析后的 dict；跳过时返回 None。
    """
    from .packets import PacketTooShortError, parse_packet

    try:
        return parse_packet(raw_bytes)
    except PacketTooShortError:
        if not skip_unknown:
            raise
        return None
    except Exception:
        if not skip_unknown:
            raise
        return None


def _safe_invoke_handler(
    handler: Callable[[dict[str, Any]], None], parsed: dict[str, Any],
) -> bool:
    """安全调用回放 handler；异常时记录日志并返回 True（仍计入计数）。"""
    try:
        handler(parsed)
    except Exception:
        logger.exception("replay handler raised, continuing")
    return True
