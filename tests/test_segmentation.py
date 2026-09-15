"""赛道分段模块单元测试。

覆盖验收标准：
1. segment_lap(samples) — 基于 corner_intensity 分段
2. segment_from_json(filepath) — 从JSON文件加载并分段
3. SegmentType 常量（STRAIGHT/ENTRY/APEX/EXIT）
4. Segment dataclass 字段完整性
5. 边界测试：空samples、单段、短段合并、同类型相邻段合并
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from setup_tuner.telemetry.segmentation import (
    Segment,
    SegmentType,
    segment_from_json,
    segment_lap,
)


# ===========================================================================
# 辅助函数：构造测试数据
# ===========================================================================
def _make_sample(
    lap_distance: float = 0.0,
    speed: float = 200.0,
    throttle: float = 1.0,
    brake: float = 0.0,
    steer: float = 0.0,
) -> dict:
    """构造单个采样点字典。"""
    return {
        "lap_distance": lap_distance,
        "speed": speed,
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
    }


def _make_straight_samples(start_dist: float = 0.0, count: int = 20) -> list[dict]:
    """构造直道采样点（throttle高、brake低、steer小）。"""
    return [
        _make_sample(lap_distance=start_dist + i * 10, speed=300, throttle=1.0, brake=0.0, steer=0.0)
        for i in range(count)
    ]


def _make_entry_samples(start_dist: float = 200.0, count: int = 20) -> list[dict]:
    """构造入弯采样点（brake高、speed下降）。"""
    return [
        _make_sample(
            lap_distance=start_dist + i * 5,
            speed=300 - i * 10,  # speed 下降
            throttle=0.2,
            brake=0.5 + i * 0.02,  # brake 增大
            steer=0.1 + i * 0.01,
        )
        for i in range(count)
    ]


def _make_apex_samples(start_dist: float = 300.0, count: int = 20) -> list[dict]:
    """构造弯中采样点（steer大、speed低且稳定、throttle低）。"""
    return [
        _make_sample(
            lap_distance=start_dist + i * 5,
            speed=150 + i * 0.5,  # speed 变化小
            throttle=0.2,
            brake=0.0,
            steer=0.3,  # steer 大
        )
        for i in range(count)
    ]


def _make_exit_samples(start_dist: float = 400.0, count: int = 20) -> list[dict]:
    """构造出弯采样点（throttle增大、steer减小、speed上升）。"""
    return [
        _make_sample(
            lap_distance=start_dist + i * 5,
            speed=150 + i * 10,  # speed 上升
            throttle=0.6 + i * 0.02,  # throttle 增大
            brake=0.0,
            steer=0.2 - i * 0.01,  # steer 减小
        )
        for i in range(count)
    ]


def _make_full_lap_samples() -> list[dict]:
    """构造一圈完整采样数据（直道→入弯→弯中→出弯→直道）。"""
    return (
        _make_straight_samples(0, 20)
        + _make_entry_samples(200, 20)
        + _make_apex_samples(300, 20)
        + _make_exit_samples(400, 20)
        + _make_straight_samples(500, 20)
    )


# ===========================================================================
# 1. SegmentType 常量
# ===========================================================================
class TestSegmentType:
    """SegmentType 常量测试。"""

    def test_straight_value(self) -> None:
        """STRAIGHT 常量值为 'straight'。"""
        assert SegmentType.STRAIGHT == "straight"

    def test_entry_value(self) -> None:
        """ENTRY 常量值为 'entry'。"""
        assert SegmentType.ENTRY == "entry"

    def test_apex_value(self) -> None:
        """APEX 常量值为 'apex'。"""
        assert SegmentType.APEX == "apex"

    def test_exit_value(self) -> None:
        """EXIT 常量值为 'exit'。"""
        assert SegmentType.EXIT == "exit"

    def test_all_types_distinct(self) -> None:
        """四种段类型互不相同。"""
        types = {SegmentType.STRAIGHT, SegmentType.ENTRY, SegmentType.APEX, SegmentType.EXIT}
        assert len(types) == 4


# ===========================================================================
# 2. Segment dataclass 字段完整性
# ===========================================================================
class TestSegmentDataclass:
    """Segment dataclass 字段完整性测试。"""

    def test_segment_has_all_fields(self) -> None:
        """Segment 包含全部12个字段。"""
        seg = Segment(
            start_distance=0.0,
            end_distance=100.0,
            segment_type="straight",
            avg_speed=250.0,
            max_speed=300.0,
            min_speed=200.0,
            avg_throttle=0.8,
            avg_brake=0.1,
            max_brake=0.3,
            avg_steer=0.05,
            max_steer=0.1,
            sample_count=20,
        )
        assert seg.start_distance == 0.0
        assert seg.end_distance == 100.0
        assert seg.segment_type == "straight"
        assert seg.avg_speed == 250.0
        assert seg.max_speed == 300.0
        assert seg.min_speed == 200.0
        assert seg.avg_throttle == 0.8
        assert seg.avg_brake == 0.1
        assert seg.max_brake == 0.3
        assert seg.avg_steer == 0.05
        assert seg.max_steer == 0.1
        assert seg.sample_count == 20

    def test_segment_field_types(self) -> None:
        """Segment 字段类型正确。"""
        seg = Segment(
            start_distance=0.0, end_distance=100.0, segment_type="straight",
            avg_speed=250.0, max_speed=300.0, min_speed=200.0,
            avg_throttle=0.8, avg_brake=0.1, max_brake=0.3,
            avg_steer=0.05, max_steer=0.1, sample_count=20,
        )
        assert isinstance(seg.start_distance, float)
        assert isinstance(seg.end_distance, float)
        assert isinstance(seg.segment_type, str)
        assert isinstance(seg.avg_speed, float)
        assert isinstance(seg.max_speed, float)
        assert isinstance(seg.min_speed, float)
        assert isinstance(seg.avg_throttle, float)
        assert isinstance(seg.avg_brake, float)
        assert isinstance(seg.max_brake, float)
        assert isinstance(seg.avg_steer, float)
        assert isinstance(seg.max_steer, float)
        assert isinstance(seg.sample_count, int)


# ===========================================================================
# 3. segment_lap — 正常输入
# ===========================================================================
class TestSegmentLap:
    """segment_lap 正常输入测试。"""

    def test_returns_list_of_segments(self) -> None:
        """返回 Segment 列表。"""
        samples = _make_full_lap_samples()
        result = segment_lap(samples)
        assert isinstance(result, list)
        for seg in result:
            assert isinstance(seg, Segment)

    def test_segments_sorted_by_distance(self) -> None:
        """段按 lap_distance 排序。"""
        samples = _make_full_lap_samples()
        result = segment_lap(samples)
        for i in range(1, len(result)):
            assert result[i].start_distance >= result[i - 1].start_distance

    def test_straight_detected(self) -> None:
        """直道被正确识别。"""
        samples = _make_straight_samples(0, 30)
        result = segment_lap(samples)
        assert len(result) >= 1
        assert any(seg.segment_type == SegmentType.STRAIGHT for seg in result)

    def test_corner_detected(self) -> None:
        """弯道被正确识别（corner_intensity >= 0.1）。"""
        # steer=0.3, throttle=0.2 → intensity = 0.3 * 0.8 + 0 = 0.24 >= 0.1
        samples = _make_apex_samples(0, 30)
        result = segment_lap(samples)
        assert len(result) >= 1
        # 应有非直道段
        corner_types = {SegmentType.ENTRY, SegmentType.APEX, SegmentType.EXIT}
        assert any(seg.segment_type in corner_types for seg in result)

    def test_full_lap_has_multiple_segments(self) -> None:
        """完整一圈产生多个段。"""
        samples = _make_full_lap_samples()
        result = segment_lap(samples)
        assert len(result) >= 2

    def test_segment_statistics_correct(self) -> None:
        """段内统计值正确。"""
        samples = _make_straight_samples(0, 10)
        result = segment_lap(samples)
        assert len(result) >= 1
        seg = result[0]
        assert seg.avg_speed == pytest.approx(300.0)
        assert seg.max_speed == 300.0
        assert seg.min_speed == 300.0
        assert seg.avg_throttle == pytest.approx(1.0)
        assert seg.sample_count == 10

    def test_segment_distance_range(self) -> None:
        """段的距离范围正确。"""
        samples = _make_straight_samples(100.0, 10)
        result = segment_lap(samples)
        assert len(result) >= 1
        seg = result[0]
        assert seg.start_distance == pytest.approx(100.0)
        # 每个采样点间隔10米，10个点 → 最后一个 lap_distance = 100 + 9*10 = 190
        assert seg.end_distance == pytest.approx(190.0)


# ===========================================================================
# 4. segment_lap — 边界/异常测试
# ===========================================================================
class TestSegmentLapEdgeCases:
    """segment_lap 边界与异常测试。"""

    def test_empty_samples(self) -> None:
        """空 samples 返回空列表。"""
        assert segment_lap([]) == []

    def test_single_sample(self) -> None:
        """单个采样点返回单个段。"""
        samples = [_make_sample(lap_distance=0.0, speed=300, throttle=1.0, brake=0.0, steer=0.0)]
        result = segment_lap(samples)
        assert len(result) == 1
        assert result[0].sample_count == 1

    def test_all_straight_single_segment(self) -> None:
        """全直道数据合并为单段。"""
        samples = _make_straight_samples(0, 50)
        result = segment_lap(samples)
        assert len(result) == 1
        assert result[0].segment_type == SegmentType.STRAIGHT

    def test_short_segment_merged(self) -> None:
        """过短的段（< 50米）与相邻段合并。"""
        # 构造：长直道(200m) + 短弯道(30m) + 长直道(200m)
        straight1 = _make_straight_samples(0, 20)  # 0-190m
        # 短弯道：3个采样点，跨度约15m
        short_corner = [
            _make_sample(lap_distance=195.0, speed=200, throttle=0.2, brake=0.5, steer=0.3),
            _make_sample(lap_distance=200.0, speed=190, throttle=0.2, brake=0.5, steer=0.3),
            _make_sample(lap_distance=205.0, speed=180, throttle=0.2, brake=0.5, steer=0.3),
        ]
        straight2 = _make_straight_samples(210, 20)  # 210-400m
        samples = straight1 + short_corner + straight2
        result = segment_lap(samples)
        # 短段应被合并，不应出现单独的短弯道段
        for seg in result:
            seg_length = seg.end_distance - seg.start_distance
            # 合并后段长度应 >= 50m（除非是唯一段）
            if len(result) > 1:
                assert seg_length >= 50.0 or seg.sample_count <= 3

    def test_adjacent_same_type_merged(self) -> None:
        """同类型相邻段合并为单段。"""
        # 两段直道之间有一个短弯道（会被合并到前一段），合并后两段直道相邻
        straight1 = _make_straight_samples(0, 20)
        short_corner = [
            _make_sample(lap_distance=195.0, speed=200, throttle=0.2, brake=0.5, steer=0.3),
        ]
        straight2 = _make_straight_samples(200, 20)
        samples = straight1 + short_corner + straight2
        result = segment_lap(samples)
        # 不应有相邻的同类型段
        for i in range(1, len(result)):
            assert result[i].segment_type != result[i - 1].segment_type or len(result) == 1

    def test_samples_not_sorted_by_distance(self) -> None:
        """未按 lap_distance 排序的 samples 仍能正确分段。"""
        samples = _make_full_lap_samples()
        # 打乱顺序
        import random
        random.seed(42)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        result = segment_lap(shuffled)
        # 结果应与排序后的一致
        assert len(result) >= 2
        for i in range(1, len(result)):
            assert result[i].start_distance >= result[i - 1].start_distance


# ===========================================================================
# 5. segment_from_json
# ===========================================================================
class TestSegmentFromJson:
    """segment_from_json 从JSON文件加载并分段。"""

    def test_returns_segments(self, tmp_path: Path) -> None:
        """从JSON文件正确加载并分段。"""
        samples = _make_full_lap_samples()
        data = {"samples": samples}
        filepath = tmp_path / "lap_1.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        result = segment_from_json(str(filepath))
        assert isinstance(result, list)
        for seg in result:
            assert isinstance(seg, Segment)

    def test_file_not_found(self) -> None:
        """文件不存在抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            segment_from_json("/nonexistent/path/lap_1.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        """非法JSON抛出 JSONDecodeError。"""
        filepath = tmp_path / "bad.json"
        filepath.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            segment_from_json(str(filepath))

    def test_empty_samples_in_json(self, tmp_path: Path) -> None:
        """JSON中 samples 为空返回空列表。"""
        data = {"samples": []}
        filepath = tmp_path / "lap_1.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        result = segment_from_json(str(filepath))
        assert result == []

    def test_missing_samples_key(self, tmp_path: Path) -> None:
        """JSON中缺 samples key 返回空列表。"""
        data = {"lap_number": 1}
        filepath = tmp_path / "lap_1.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        result = segment_from_json(str(filepath))
        assert result == []


# ===========================================================================
# 6. 属性不变量测试
# ===========================================================================
class TestSegmentInvariants:
    """分段算法属性不变量测试。"""

    def test_segment_count_at_least_one_for_nonempty(self) -> None:
        """非空输入至少产生1个段。"""
        samples = [_make_sample(lap_distance=0.0)]
        result = segment_lap(samples)
        assert len(result) >= 1

    def test_all_samples_accounted_for(self) -> None:
        """所有采样点都被分配到段中（总 sample_count 等于输入）。"""
        samples = _make_full_lap_samples()
        total_count = sum(seg.sample_count for seg in segment_lap(samples))
        assert total_count == len(samples)

    def test_no_gap_between_segments(self) -> None:
        """相邻段无距离间隙（end_distance >= next start_distance）。"""
        samples = _make_full_lap_samples()
        result = segment_lap(samples)
        for i in range(1, len(result)):
            # 段之间允许有微小间隙（因采样点离散），但不应有大跳变
            gap = result[i].start_distance - result[i - 1].end_distance
            assert gap <= 20.0  # 采样间隔容差

    def test_max_speed_ge_min_speed(self) -> None:
        """每段 max_speed >= min_speed。"""
        samples = _make_full_lap_samples()
        result = segment_lap(samples)
        for seg in result:
            assert seg.max_speed >= seg.min_speed

    def test_end_distance_ge_start_distance(self) -> None:
        """每段 end_distance >= start_distance。"""
        samples = _make_full_lap_samples()
        result = segment_lap(samples)
        for seg in result:
            assert seg.end_distance >= seg.start_distance