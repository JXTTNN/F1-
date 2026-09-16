"""F1 25 遥测层（纯 struct 解析，零 numpy/torch）。

导出：
- :class:`PacketHeader` / :func:`parse_header` / :func:`parse_packet` — 包解析
- :class:`PacketTooShortError` / :class:`UnknownPacketError` — 异常
- :class:`TelemetryListener` — UDP 监听
- :class:`TelemetryStream` — 最新帧缓存
- :class:`TelemetrySimulator` — 遥测模拟器（无真实 F1 游戏时回放模拟数据）
- :data:`SUPPORTED_PACKET_IDS` / :data:`NUM_CARS` — 常量
"""

from __future__ import annotations

from .lap_aggregator import LapAggregator
from .listener import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    TelemetryListener,
)
from .packets import (
    HEADER_SIZE,
    NUM_CARS,
    PACKET_NAMES,
    SUPPORTED_PACKET_IDS,
    PacketHeader,
    PacketTooShortError,
    UnknownPacketError,
    packet_name,
    parse_car_setups,
    parse_car_status,
    parse_car_telemetry,
    parse_header,
    parse_lap_data,
    parse_packet,
    parse_session,
)
from .simulator import TelemetrySimulator, generate_lap_snapshot
from .stream import TelemetryStream

__all__ = [
    # packets
    "PacketHeader",
    "PacketTooShortError",
    "UnknownPacketError",
    "parse_header",
    "parse_packet",
    "parse_session",
    "parse_lap_data",
    "parse_car_setups",
    "parse_car_telemetry",
    "parse_car_status",
    "packet_name",
    "HEADER_SIZE",
    "NUM_CARS",
    "PACKET_NAMES",
    "SUPPORTED_PACKET_IDS",
    # listener
    "TelemetryListener",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    # stream
    "TelemetryStream",
    # lap aggregator
    "LapAggregator",
    # simulator
    "TelemetrySimulator",
    "generate_lap_snapshot",
]
