"""赛道分段算法 — 基于 speed/throttle/brake/steer/lap_distance 识别赛道段。

G 力数据全部为 0，无法用于识别弯道阶段。本模块使用 corner_intensity 指标
（由 steer/throttle/brake 组合计算）来区分直道与弯道，并进一步将弯道
细分为入弯(entry)、弯中(apex)、出弯(exit)三个阶段。

段类型定义：
- STRAIGHT: 直道 — throttle 高、brake 低、steer 小
- ENTRY:    入弯 — brake 高、steer 增大、speed 下降
- APEX:     弯中 — steer 大、speed 低且稳定、throttle 低
- EXIT:     出弯 — throttle 增大、steer 减小、speed 上升
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# 段类型常量
# --------------------------------------------------------------------------- #
class SegmentType:
    """赛道段类型常量。"""

    STRAIGHT = "straight"
    ENTRY = "entry"
    APEX = "apex"
    EXIT = "exit"


# --------------------------------------------------------------------------- #
# Segment dataclass
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """单个赛道段的统计数据。

    Attributes:
        start_distance: 段起始距离（米）。
        end_distance: 段结束距离（米）。
        segment_type: 段类型（straight/entry/apex/exit）。
        avg_speed: 段内平均速度（km/h）。
        max_speed: 段内最大速度（km/h）。
        min_speed: 段内最小速度（km/h）。
        avg_throttle: 段内平均油门（0.0-1.0）。
        avg_brake: 段内平均刹车（0.0-1.0）。
        max_brake: 段内最大刹车（0.0-1.0）。
        avg_steer: 段内平均转向绝对值（0.0-1.0）。
        max_steer: 段内最大转向绝对值（0.0-1.0）。
        sample_count: 段内采样点数。
    """

    start_distance: float
    end_distance: float
    segment_type: str
    avg_speed: float
    max_speed: float
    min_speed: float
    avg_throttle: float
    avg_brake: float
    max_brake: float
    avg_steer: float
    max_steer: float
    sample_count: int


# --------------------------------------------------------------------------- #
# 阈值常量
# --------------------------------------------------------------------------- #
_CORNER_INTENSITY_THRESHOLD = 0.1   # corner_intensity >= 此值 → 弯道
_BRAKE_ENTRY_THRESHOLD = 0.3        # brake > 此值 → 入弯特征
_STEER_APEX_THRESHOLD = 0.15        # abs(steer) > 此值 → 弯中特征
_THROTTLE_EXIT_THRESHOLD = 0.5      # throttle > 此值 → 出弯特征
_SPEED_STABLE_THRESHOLD = 5.0       # |dspeed| < 此值 → speed 稳定
_MIN_SEGMENT_LENGTH = 50.0          # 段长度 < 此值 → 与相邻段合并


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def _compute_corner_intensity(sample: dict[str, Any]) -> float:
    """计算单个采样点的弯道强度指标。

    corner_intensity = abs(steer) * (1 - throttle) + brake * 0.5

    Args:
        sample: 包含 steer/throttle/brake 字段的采样点。

    Returns:
        弯道强度值（>= 0）。
    """
    steer = abs(float(sample.get("steer", 0.0)))
    throttle = float(sample.get("throttle", 0.0))
    brake = float(sample.get("brake", 0.0))
    return steer * (1.0 - throttle) + brake * 0.5


def _classify_corner_phase(
    sample: dict[str, Any],
    prev_sample: dict[str, Any] | None,
    next_sample: dict[str, Any] | None,
) -> str:
    """对弯道中的采样点进行阶段细分（entry/apex/exit）。

    Args:
        sample: 当前采样点。
        prev_sample: 前一个采样点（可能为 None）。
        next_sample: 后一个采样点（可能为 None）。

    Returns:
        段类型：ENTRY / APEX / EXIT。
    """
    brake = float(sample.get("brake", 0.0))
    throttle = float(sample.get("throttle", 0.0))
    steer_abs = abs(float(sample.get("steer", 0.0)))
    speed = float(sample.get("speed", 0.0))

    # 入弯：brake > 0.3 且 speed 正在下降
    if brake > _BRAKE_ENTRY_THRESHOLD:
        if prev_sample is not None:
            prev_speed = float(prev_sample.get("speed", 0.0))
            if speed < prev_speed:
                return SegmentType.ENTRY
        return SegmentType.ENTRY

    # 弯中：steer 绝对值 > 0.15 且 speed 变化率小
    if steer_abs > _STEER_APEX_THRESHOLD:
        speed_change = 0.0
        if prev_sample is not None and next_sample is not None:
            prev_speed = float(prev_sample.get("speed", 0.0))
            next_speed = float(next_sample.get("speed", 0.0))
            speed_change = abs(next_speed - prev_speed)
        if speed_change < _SPEED_STABLE_THRESHOLD:
            return SegmentType.APEX
        # speed 在变化但 steer 大 → 仍在弯中
        return SegmentType.APEX

    # 出弯：throttle > 0.5 且 steer 绝对值在减小
    if throttle > _THROTTLE_EXIT_THRESHOLD:
        if prev_sample is not None:
            prev_steer_abs = abs(float(prev_sample.get("steer", 0.0)))
            if steer_abs < prev_steer_abs:
                return SegmentType.EXIT
        return SegmentType.EXIT

    # 默认归类为弯中（在弯道范围内但无法明确分类）
    return SegmentType.APEX


def _build_segment(samples: list[dict[str, Any]], seg_type: str) -> Segment:
    """从采样点列表构建 Segment 对象。

    Args:
        samples: 该段内的所有采样点。
        seg_type: 段类型（straight/entry/apex/exit）。

    Returns:
        :class:`Segment` 对象。
    """
    if not samples:
        return Segment(
            start_distance=0.0, end_distance=0.0, segment_type=seg_type,
            avg_speed=0.0, max_speed=0.0, min_speed=0.0,
            avg_throttle=0.0, avg_brake=0.0, max_brake=0.0,
            avg_steer=0.0, max_steer=0.0, sample_count=0,
        )

    speeds = [float(s.get("speed", 0.0)) for s in samples]
    throttles = [float(s.get("throttle", 0.0)) for s in samples]
    brakes = [float(s.get("brake", 0.0)) for s in samples]
    steers_abs = [abs(float(s.get("steer", 0.0))) for s in samples]
    distances = [float(s.get("lap_distance", 0.0)) for s in samples]

    return Segment(
        start_distance=min(distances),
        end_distance=max(distances),
        segment_type=seg_type,
        avg_speed=statistics.fmean(speeds),
        max_speed=max(speeds),
        min_speed=min(speeds),
        avg_throttle=statistics.fmean(throttles),
        avg_brake=statistics.fmean(brakes),
        max_brake=max(brakes),
        avg_steer=statistics.fmean(steers_abs),
        max_steer=max(steers_abs),
        sample_count=len(samples),
    )


def _merge_short_segments(segments: list[Segment]) -> list[Segment]:
    """将过短的段（< 50 米）与相邻段合并。

    合并策略：短段优先与前一相邻段合并；若前一相邻段也是短段，
    则继续向前查找；若短段是第一个段，则与后一段合并。

    Args:
        segments: 待合并的段列表（按 lap_distance 排序）。

    Returns:
        合并后的段列表。
    """
    if len(segments) <= 1:
        return segments

    merged: list[Segment] = []
    for seg in segments:
        seg_length = seg.end_distance - seg.start_distance
        if seg_length < _MIN_SEGMENT_LENGTH and merged:
            # 与前一相邻段合并：重新计算合并后的统计值
            prev = merged[-1]
            combined_samples_count = prev.sample_count + seg.sample_count
            # 按采样数加权平均
            w_prev = prev.sample_count / combined_samples_count if combined_samples_count > 0 else 0
            w_seg = seg.sample_count / combined_samples_count if combined_samples_count > 0 else 0
            merged_seg = Segment(
                start_distance=prev.start_distance,
                end_distance=seg.end_distance,
                segment_type=prev.segment_type,  # 保留前一段的类型
                avg_speed=prev.avg_speed * w_prev + seg.avg_speed * w_seg,
                max_speed=max(prev.max_speed, seg.max_speed),
                min_speed=min(prev.min_speed, seg.min_speed),
                avg_throttle=prev.avg_throttle * w_prev + seg.avg_throttle * w_seg,
                avg_brake=prev.avg_brake * w_prev + seg.avg_brake * w_seg,
                max_brake=max(prev.max_brake, seg.max_brake),
                avg_steer=prev.avg_steer * w_prev + seg.avg_steer * w_seg,
                max_steer=max(prev.max_steer, seg.max_steer),
                sample_count=combined_samples_count,
            )
            merged[-1] = merged_seg
        else:
            merged.append(seg)

    # 递归处理：合并后可能产生新的短段
    if len(merged) < len(segments):
        return _merge_short_segments(merged)
    return merged


def _merge_adjacent_same_type(segments: list[Segment]) -> list[Segment]:
    """合并同类型的相邻段。

    短段合并后可能产生同类型相邻段（如两个 straight 段之间曾有一个短弯道
    被合并到前一段），此函数将它们重新合并为单个段。

    Args:
        segments: 待合并的段列表（按 lap_distance 排序）。

    Returns:
        合并后的段列表。
    """
    if len(segments) <= 1:
        return segments

    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.segment_type == prev.segment_type:
            combined_count = prev.sample_count + seg.sample_count
            w_prev = prev.sample_count / combined_count if combined_count > 0 else 0
            w_seg = seg.sample_count / combined_count if combined_count > 0 else 0
            merged[-1] = Segment(
                start_distance=prev.start_distance,
                end_distance=seg.end_distance,
                segment_type=prev.segment_type,
                avg_speed=prev.avg_speed * w_prev + seg.avg_speed * w_seg,
                max_speed=max(prev.max_speed, seg.max_speed),
                min_speed=min(prev.min_speed, seg.min_speed),
                avg_throttle=prev.avg_throttle * w_prev + seg.avg_throttle * w_seg,
                avg_brake=prev.avg_brake * w_prev + seg.avg_brake * w_seg,
                max_brake=max(prev.max_brake, seg.max_brake),
                avg_steer=prev.avg_steer * w_prev + seg.avg_steer * w_seg,
                max_steer=max(prev.max_steer, seg.max_steer),
                sample_count=combined_count,
            )
        else:
            merged.append(seg)
    return merged


# --------------------------------------------------------------------------- #
# 核心分段函数
# --------------------------------------------------------------------------- #
def segment_lap(samples: list[dict[str, Any]]) -> list[Segment]:
    """对单圈采样数据执行分段算法。

    算法步骤：
    1. 预处理：按 lap_distance 排序，计算每个采样点的 corner_intensity。
    2. 粗分类：corner_intensity < 0.1 → 直道，>= 0.1 → 弯道。
    3. 弯道阶段细分：入弯(entry) / 弯中(apex) / 出弯(exit)。
    4. 段合并：过短的段（< 50 米）与相邻段合并。

    Args:
        samples: 单圈采样数据列表，每个 sample 包含 lap_distance/speed/
            throttle/brake/steer 等字段。

    Returns:
        按 lap_distance 排序的 :class:`Segment` 列表。
    """
    if not samples:
        return []

    # 1. 预处理：按 lap_distance 排序
    sorted_samples = sorted(samples, key=lambda s: float(s.get("lap_distance", 0.0)))

    # 2. 逐采样点分类
    raw_labels: list[str] = []
    for i, s in enumerate(sorted_samples):
        intensity = _compute_corner_intensity(s)
        if intensity < _CORNER_INTENSITY_THRESHOLD:
            raw_labels.append(SegmentType.STRAIGHT)
        else:
            prev_s = sorted_samples[i - 1] if i > 0 else None
            next_s = sorted_samples[i + 1] if i < len(sorted_samples) - 1 else None
            raw_labels.append(_classify_corner_phase(s, prev_s, next_s))

    # 3. 连续相同标签 → 段
    segments: list[Segment] = []
    seg_start_idx = 0
    for i in range(1, len(sorted_samples)):
        if raw_labels[i] != raw_labels[seg_start_idx]:
            seg_samples = sorted_samples[seg_start_idx:i]
            segments.append(_build_segment(seg_samples, raw_labels[seg_start_idx]))
            seg_start_idx = i
    # 最后一段
    seg_samples = sorted_samples[seg_start_idx:]
    segments.append(_build_segment(seg_samples, raw_labels[seg_start_idx]))

    # 4. 合并过短的段，再合并同类型相邻段
    segments = _merge_short_segments(segments)
    segments = _merge_adjacent_same_type(segments)

    return segments


def segment_from_json(filepath: str) -> list[Segment]:
    """从 JSON 文件路径加载 samples 并执行分段。

    Args:
        filepath: JSON 文件绝对路径。

    Returns:
        按 lap_distance 排序的 :class:`Segment` 列表。

    Raises:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: JSON 格式错误。
    """
    path = Path(filepath)
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    samples: list[dict[str, Any]] = data.get("samples", [])
    return segment_lap(samples)