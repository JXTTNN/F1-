"""F1 2026 遥测层（纯 struct 解析，零 numpy/torch）。

导出：
- :class:`PacketHeader` / :func:`parse_header` / :func:`parse_packet` — 包解析
- :class:`PacketTooShortError` / :class:`UnknownPacketError` — 异常
- :class:`TelemetryListener` — UDP 监听
- :class:`TelemetryStream` — 最新帧缓存
- :data:`SUPPORTED_PACKET_IDS` / :data:`NUM_CARS` — 常量
"""

from __future__ import annotations

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
    parse_car_setups,
    parse_car_status,
    parse_car_telemetry,
    parse_car_telemetry_2,
    parse_header,
    parse_lap_data,
    parse_packet,
    parse_session,
    packet_name,
)
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
    "parse_car_telemetry_2",
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
]
