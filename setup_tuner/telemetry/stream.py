"""最新帧内存缓存（线程安全，供 WebSocket 推送）。

缓存各 packet_id 的最新解析数据，供 WebSocket 网关实时推送。
线程安全通过 :class:`threading.Lock` 保护；仅缓存最新一帧（覆盖写）。

使用方式::

    stream = TelemetryStream()
    stream.update(6, parsed_telemetry)   # 由 listener handler 调用
    latest = stream.get_latest(6)        # 由 WS 网关调用
    all_latest = stream.get_all_latest() # 一次性取全部最新帧
"""

from __future__ import annotations

import threading
from typing import Any

from .packets import SUPPORTED_PACKET_IDS


class TelemetryStream:
    """各 packet_id 最新帧的线程安全内存缓存。

    - ``update(packet_id, data)``：覆盖写入该 packet_id 的最新帧。
    - ``get_latest(packet_id)``：读取该 packet_id 的最新帧（无则 ``None``）。
    - ``get_all_latest()``：返回所有已缓存 packet_id 的最新帧快照（dict 副本）。

    只缓存 :data:`~setup_tuner.telemetry.packets.SUPPORTED_PACKET_IDS` 中的 6 类包；
    其他 packet_id 的 update 静默忽略（防御性）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # 仅为支持的 6 类包预留槽位；value 为最新解析 dict 或 None
        self._cache: dict[int, dict[str, Any] | None] = dict.fromkeys(SUPPORTED_PACKET_IDS)

    def update(self, packet_id: int, data: dict[str, Any]) -> None:
        """更新指定 packet_id 的最新帧（覆盖写，线程安全）。

        不在支持范围内的 packet_id 静默忽略。
        """
        with self._lock:
            if packet_id in self._cache:
                self._cache[packet_id] = data

    def get_latest(self, packet_id: int) -> dict[str, Any] | None:
        """获取指定 packet_id 的最新帧（无数据则 ``None``，线程安全）。"""
        with self._lock:
            val = self._cache.get(packet_id)
            return dict(val) if val is not None else None

    def get_all_latest(self) -> dict[int, dict[str, Any]]:
        """获取所有已缓存且有数据的 packet_id 的最新帧快照。

        返回的 dict 是深拷贝快照（各 value 也是 dict 副本），调用方可安全修改。
        未收到过的包不会出现在返回值中。
        """
        with self._lock:
            return {
                pid: dict(val)
                for pid, val in self._cache.items()
                if val is not None
            }

    def clear(self) -> None:
        """清空所有缓存（线程安全）。"""
        with self._lock:
            for pid in self._cache:
                self._cache[pid] = None