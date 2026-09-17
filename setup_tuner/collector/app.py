"""桌面遥测接收器 —— 编排核心（无 GUI，可独立测试）。

职责（task-81）：
1. 复用 :class:`~setup_tuner.telemetry.listener.TelemetryListener` 收 UDP
   （全类型 0-15 原始字节，含解析器暂不支持的业务包）；
2. 复用 :class:`~setup_tuner.telemetry.recorder.TelemetryRecorder` 无损留存
   （.f1rec 原始字节 + zstd 压缩 + SQLite 索引，批量写，供模型训练重放）；
3. 统计与实时摘要（按包类型计数、当前赛道/圈/速度），供桌面 GUI 轮询。

分层：本模块不含任何 tkinter —— GUI 在 :mod:`.gui`，入口 ``python -m
setup_tuner.collector``。监听（常驻收包）与收集（落盘录制）是两个独立
开关：只开监听可观察计数，开收集才写盘。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from setup_tuner.telemetry.listener import (
    DEFAULT_PORT,
    TelemetryListener,
)
from setup_tuner.telemetry.packets import packet_name, parse_header
from setup_tuner.telemetry.recorder import TelemetryRecorder

logger = logging.getLogger(__name__)

#: 桌面接收器绑定所有网卡（而非只绑 127.0.0.1）。
#: F1 游戏的「UDP 广播模式」开启时，包会被发到 255.255.255.255（广播）而非
#: 设置里填的 IP；只绑 127.0.0.1 会**一个包都收不到**（0 包的经典成因）。
#: 绑 0.0.0.0 可同时接收：单播到回环/局域网 IP + 广播。
#: 注意：``TelemetryListener`` 自身的 ``DEFAULT_HOST`` 仍是 127.0.0.1
#: （被 deep-slice 测试锁定），这里只改桌面接收器的绑定。
_BIND_ALL = "0.0.0.0"


class CollectorApp:
    """桌面遥测接收器编排：监听 + 留存 + 统计。

    线程模型：UDP 收包在监听线程，``_on_raw/_on_parsed`` 只做计数与
    轻量字段提取（锁保护）；落盘由 Recorder 的工作线程完成，不阻塞收包。
    ``status()`` 返回快照 dict，供 GUI 主循环轮询（每 500ms），无跨线程
    UI 调用。
    """

    def __init__(
        self,
        data_dir: str = "data/recordings",
        *,
        host: str = _BIND_ALL,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._host = host
        self._port = port
        self._listener = TelemetryListener(host, port)
        self._recorder = TelemetryRecorder(str(self._data_dir))
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._total = 0
        self._latest: dict[str, Any] = {}
        self._last_summary: dict[str, Any] = {}
        self._listener.add_raw_handler(self._on_raw)
        self._listener.add_handler(self._on_parsed)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    @property
    def data_dir(self) -> Path:
        """录制数据目录。"""
        return self._data_dir

    def listen(self) -> None:
        """启动 UDP 监听（常驻收包；不落盘，只计数/显示）。"""
        self._listener.start()

    def stop_listen(self) -> None:
        """停止 UDP 监听（幂等）。"""
        self._listener.stop()

    def start_collect(self, *, auto_listen: bool = False) -> dict[str, Any]:
        """开始收集（落盘录制），返回录制会话信息。

        Args:
            auto_listen: 为 True 时，若监听未启动则**先自动启动监听**再开录
                （GUI 用）——保证用户「点开始收集」就一定在收包并自动落盘。
                默认 False 保持库层语义纯粹：测试不会隐式占用默认端口。
        """
        if auto_listen and not self._listener.is_running:
            self.listen()
        return self._recorder.start()

    def stop_collect(self) -> dict[str, Any]:
        """停止收集，返回本次录制摘要（包数/时长/文件路径）。"""
        summary = self._recorder.stop()
        with self._lock:
            self._last_summary = dict(summary)
        return summary

    def shutdown(self) -> None:
        """退出前调用：停止收集 + 停止监听（均幂等）。"""
        if self._recorder.is_recording:
            self.stop_collect()
        self.stop_listen()

    # ------------------------------------------------------------------ #
    # 回调（监听线程）
    # ------------------------------------------------------------------ #
    def _on_raw(self, data: bytes, parsed: dict[str, Any] | None) -> None:
        """每个到达的 UDP 包：计数（含未知类型）+ 交给 Recorder。"""
        name = "unknown"
        if parsed is not None:
            pid = parsed.get("packet_id")
            if pid is not None:
                name = packet_name(pid)
        elif len(data) >= 29:
            try:
                name = packet_name(parse_header(data).packet_id)
            except Exception:
                name = "unknown"
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            self._total += 1
        # Recorder 非录制态自行忽略，这里无条件转发（转发的开销只有入队判断）
        self._recorder.on_raw_packet(data, parsed)

    def _on_parsed(self, parsed: dict[str, Any]) -> None:
        """解析成功的包：提取 GUI 摘要字段（赛道/圈/车速等）。"""
        pid = parsed.get("packet_id")
        updates: dict[str, Any] = {}
        if pid == 1:
            for key in ("m_trackId", "m_totalLaps", "m_sessionType"):
                if key in parsed:
                    updates[key] = parsed[key]
        elif pid == 2:
            for key in ("m_currentLapNum", "m_lapDistance", "m_currentLapTimeInMS"):
                if key in parsed:
                    updates[key] = parsed[key]
        elif pid == 6:
            for key in (
                "m_speed", "m_throttle", "m_brake", "m_gear",
                "m_engineRPM", "m_drs",
            ):
                if key in parsed:
                    updates[key] = parsed[key]
        if updates:
            with self._lock:
                self._latest.update(updates)

    # ------------------------------------------------------------------ #
    # 状态快照（GUI 轮询）
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """当前状态快照（线程安全，只读拷贝）。"""
        with self._lock:
            counts = dict(self._counts)
            latest = dict(self._latest)
            total = self._total
        return {
            "listening": self._listener.is_running,
            "recording": self._recorder.is_recording,
            "session_id": self._recorder.session_id,
            "data_dir": str(self._data_dir),
            "host": self._host,
            "port": self._port,
            "packets_total": total,
            "counts": counts,
            "last_summary": dict(self._last_summary),
            "latest": latest,
        }
