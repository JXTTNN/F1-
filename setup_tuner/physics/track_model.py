"""赛道模型 — 弯道分布 → 每弯极限速度+分段.

物理原理:
    - 弯道半径: 从Corner的speed_kmh推算.
      radius = v² / (a_lat * g), v单位m/s
    - 弯道长度: radius * angle (弧度).
      slow→90°(π/2), medium→60°(π/3), fast→30°(π/6)
    - 弯道极限速度: 直接用Corner.speed_kmh / 3.6 (m/s)
    - 直道长度: 赛道总长 - 弯道总长, 平均分配到弯道之间的直道
    - 直道数量: max(1, len(corners) - 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from setup_tuner.domain.track import Track

G = 9.81  # m/s², 重力加速度

# 弯道类型 → 弯道角度 (弧度)
CORNER_ANGLE: dict[str, float] = {
    "slow": math.pi / 2,      # 90°
    "medium": math.pi / 3,    # 60°
    "fast": math.pi / 6,      # 30°
}

KMH_TO_MS = 3.6  # km/h → m/s 转换因子


@dataclass
class CornerSegment:
    """弯道分段定义."""
    number: int          # 弯道编号 (1-based)
    name: str            # 弯道名称
    corner_type: str     # slow / medium / fast
    radius: float        # m, 弯道半径
    length: float        # m, 弯道长度
    speed_max: float     # m/s, 极限速度


@dataclass
class StraightSegment:
    """直道分段定义."""
    length: float        # m, 直道长度
    speed_entry: float   # m/s, 入口速度（由simulator填充）
    speed_exit: float    # m/s, 出口速度（由simulator填充）


@dataclass
class TrackOutput:
    """赛道模型计算输出."""
    corners: list[CornerSegment]   # 弯道分段列表
    straights: list[StraightSegment]  # 直道分段列表
    total_length: float            # m, 赛道总长度
    corner_count: int              # 弯道数量
    straight_count: int            # 直道数量


class TrackModel:
    """赛道模型.

    根据Track对象初始化, 在给定最大侧向加速度条件下计算
    每个弯道的半径、长度和极限速度, 以及直道分段.
    """

    def __init__(self, track: Track) -> None:
        self.track = track

    def compute(self, max_lateral_accel: float = 2.5) -> TrackOutput:
        """计算赛道分段.

        Args:
            max_lateral_accel: 最大侧向加速度 (g), 默认2.5

        Returns:
            TrackOutput 包含弯道分段、直道分段和赛道统计信息
        """
        corners = self._build_corner_segments(max_lateral_accel)
        straights = self._build_straight_segments(corners)

        return TrackOutput(
            corners=corners,
            straights=straights,
            total_length=self.track.length_m,
            corner_count=len(corners),
            straight_count=len(straights),
        )

    def _build_corner_segments(self,
                               max_lateral_accel: float) -> list[CornerSegment]:
        """构建所有弯道分段."""
        segments: list[CornerSegment] = []
        for corner in self.track.corners:
            speed_ms = corner.speed_kmh / KMH_TO_MS
            radius = self._calc_radius(speed_ms, max_lateral_accel)
            angle = CORNER_ANGLE.get(corner.corner_type, CORNER_ANGLE["medium"])
            length = radius * angle
            segments.append(CornerSegment(
                number=corner.number,
                name=corner.name,
                corner_type=corner.corner_type,
                radius=radius,
                length=length,
                speed_max=speed_ms,
            ))
        return segments

    def _build_straight_segments(self,
                                 corners: list[CornerSegment]) -> list[StraightSegment]:
        """构建直道分段."""
        corner_total_length = sum(c.length for c in corners)
        straight_total = max(0.0, self.track.length_m - corner_total_length)
        n_straights = max(1, len(corners) - 1)
        straight_length = straight_total / n_straights if n_straights > 0 else 0.0

        return [StraightSegment(
            length=straight_length,
            speed_entry=0.0,
            speed_exit=0.0,
        ) for _ in range(n_straights)]

    def _calc_radius(self, speed_ms: float,
                     max_lateral_accel: float) -> float:
        """计算弯道半径: radius = v² / (a_lat * g)."""
        if max_lateral_accel <= 0:
            return 0.0
        return speed_ms * speed_ms / (max_lateral_accel * G)