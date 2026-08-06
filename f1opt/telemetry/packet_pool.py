from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
from collections import deque

import numpy as np

CAR_COUNT = 22


@dataclass(slots=True)
class PacketBuffer:
    header: dict = field(default_factory=dict)
    motion: dict = field(default_factory=dict)
    session: dict = field(default_factory=dict)
    lap_data: dict = field(default_factory=dict)
    event: dict = field(default_factory=dict)
    participants: list = field(default_factory=list)
    car_setups: dict = field(default_factory=dict)
    car_telemetry: list = field(default_factory=list)
    car_status: list = field(default_factory=list)
    final_classification: dict = field(default_factory=dict)
    lobby_info: dict = field(default_factory=dict)
    car_damage: list = field(default_factory=list)
    session_history: dict = field(default_factory=dict)
    tyre_sets: dict = field(default_factory=dict)
    motion_ex: dict = field(default_factory=dict)
    ts_ms: int = 0

    def reset(self) -> None:
        self.time = 0
        self.buffer_len = 0
        self.compute_health = 100.0
        self.compute_time_ms = 0.0


class PacketBufferPool:
    _pool: ClassVar[deque[PacketBuffer]] = deque()
    _max_size: ClassVar[int] = 128

    @classmethod
    def acquire(cls) -> PacketBuffer:
        if cls._pool:
            buf = cls._pool.popleft()
            buf.reset()
            return buf
        return PacketBuffer()

    @classmethod
    def release(cls, buf: PacketBuffer) -> None:
        if len(cls._pool) < cls._max_size:
            cls._pool.append(buf)

    @classmethod
    def pool_size(cls) -> int:
        return len(cls._pool)


class PacketBufferNamespace:
    __slots__ = ("_pool",)

    def __init__(self, max_size: int = 64) -> None:
        self._pool = deque(maxlen=max_size)

    async def acquire(self) -> PacketBuffer:
        if self._pool:
            buf = self._pool.popleft()
            buf.reset()
        else:
            buf = PacketBuffer()
        return buf

    async def release(self, buf: PacketBuffer) -> None:
        self._pool.append(buf)

    def __len__(self) -> int:
        return len(self._pool)
