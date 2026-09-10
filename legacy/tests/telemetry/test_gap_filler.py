"""Tests for :mod:`f1opt.telemetry.gap_filler` (Iter-155)."""
from __future__ import annotations

from f1opt.telemetry.gap_filler import GapFiller, fill_frame_gaps


def _make_frame(fid: int, speed: float = 200.0, throttle: float = 0.5,
                gear: int = 3, drs: int = 0) -> dict:
    return {
        "frame_id": fid,
        "speed": speed,
        "throttle": throttle,
        "gear": gear,
        "drs": drs,
    }


class TestGapFiller:
    def test_no_gaps_unchanged(self) -> None:
        """连续帧序列不填充, 返回等价列表."""
        frames = [_make_frame(0), _make_frame(1), _make_frame(2)]
        result = GapFiller().fill_gaps(frames)
        assert len(result) == 3
        assert [f["frame_id"] for f in result] == [0, 1, 2]

    def test_single_frame_unchanged(self) -> None:
        """单帧序列直接返回."""
        frames = [_make_frame(0)]
        result = GapFiller().fill_gaps(frames)
        assert len(result) == 1

    def test_empty_list_unchanged(self) -> None:
        """空列表直接返回."""
        result = GapFiller().fill_gaps([])
        assert len(result) == 0

    def test_small_gap_filled(self) -> None:
        """帧 0 和帧 3 之间有 2 帧间隙 → 填充."""
        frames = [_make_frame(0, speed=200.0), _make_frame(3, speed=230.0)]
        result = GapFiller().fill_gaps(frames)
        assert len(result) == 4  # 0, 1, 2, 3
        assert [f["frame_id"] for f in result] == [0, 1, 2, 3]

    def test_linear_interpolation_correct(self) -> None:
        """线性插值: speed 从 200 到 230, 中间帧应为 210, 220."""
        frames = [_make_frame(0, speed=200.0), _make_frame(3, speed=230.0)]
        result = GapFiller().fill_gaps(frames)
        # gap=2, alpha for frame 1 = 1/3, frame 2 = 2/3
        assert abs(result[1]["speed"] - 210.0) < 0.01
        assert abs(result[2]["speed"] - 220.0) < 0.01

    def test_int_field_hold_last(self) -> None:
        """整数字段保持上一个值 (hold-last)."""
        frames = [_make_frame(0, gear=3), _make_frame(3, gear=5)]
        result = GapFiller().fill_gaps(frames)
        assert result[1]["gear"] == 3  # held
        assert result[2]["gear"] == 3  # held
        assert result[3]["gear"] == 5  # actual

    def test_large_gap_not_filled(self) -> None:
        """间隙 > max_gap 不填充."""
        frames = [_make_frame(0), _make_frame(20)]
        result = GapFiller(max_gap=10).fill_gaps(frames)
        assert len(result) == 2  # no fill

    def test_max_gap_boundary(self) -> None:
        """间隙 == max_gap 仍被填充 (边界条件)."""
        frames = [_make_frame(0), _make_frame(11)]
        result = GapFiller(max_gap=10).fill_gaps(frames)
        assert len(result) == 12  # 0..11

    def test_multiple_gaps(self) -> None:
        """多个间隙都被填充."""
        frames = [_make_frame(0), _make_frame(3), _make_frame(7)]
        result = GapFiller().fill_gaps(frames)
        assert len(result) == 8  # 0,1,2,3,4,5,6,7
        assert [f["frame_id"] for f in result] == list(range(8))

    def test_original_frames_not_modified(self) -> None:
        """原始帧不被修改."""
        frames = [_make_frame(0, speed=200.0), _make_frame(3, speed=230.0)]
        original_speed_0 = frames[0]["speed"]
        _ = GapFiller().fill_gaps(frames)
        assert frames[0]["speed"] == original_speed_0

    def test_custom_float_fields(self) -> None:
        """自定义 float 字段被插值."""
        frames = [
            {"frame_id": 0, "my_field": 10.0},
            {"frame_id": 2, "my_field": 20.0},
        ]
        result = GapFiller(float_fields=["my_field"]).fill_gaps(frames)
        assert abs(result[1]["my_field"] - 15.0) < 0.01

    def test_custom_int_fields(self) -> None:
        """自定义 int 字段保持上一个值."""
        frames = [
            {"frame_id": 0, "my_status": 1},
            {"frame_id": 2, "my_status": 2},
        ]
        result = GapFiller(int_fields=["my_status"]).fill_gaps(frames)
        assert result[1]["my_status"] == 1

    def test_none_values_hold_other(self) -> None:
        """字段为 None 时保持非 None 的值."""
        frames = [
            {"frame_id": 0, "speed": 200.0, "optional": None},
            {"frame_id": 2, "speed": 220.0, "optional": 42},
        ]
        result = GapFiller(float_fields=["speed"]).fill_gaps(frames)
        # optional is None in prev, so hold curr's value (42)
        assert result[1]["optional"] == 42

    def test_convenience_function(self) -> None:
        """fill_frame_gaps 便捷函数正常工作."""
        frames = [_make_frame(0), _make_frame(3)]
        result = fill_frame_gaps(frames, max_gap=5)
        assert len(result) == 4

    def test_duplicate_frame_ids(self) -> None:
        """重复 frame_id (gap <= 0) 不填充, 直接追加."""
        frames = [_make_frame(0), _make_frame(0), _make_frame(1)]
        result = GapFiller().fill_gaps(frames)
        assert len(result) == 3

    def test_out_of_order_frames(self) -> None:
        """帧序号倒退 (gap < 0) 不填充."""
        frames = [_make_frame(5), _make_frame(3)]
        result = GapFiller().fill_gaps(frames)
        assert len(result) == 2

    def test_unknown_field_hold_last(self) -> None:
        """未知字段默认 hold-last."""
        frames = [
            {"frame_id": 0, "custom": "hello"},
            {"frame_id": 2, "custom": "world"},
        ]
        result = GapFiller().fill_gaps(frames)
        assert result[1]["custom"] == "hello"

    def test_throttle_interpolation(self) -> None:
        """throttle 字段线性插值."""
        frames = [_make_frame(0, throttle=0.0), _make_frame(2, throttle=1.0)]
        result = GapFiller().fill_gaps(frames)
        assert abs(result[1]["throttle"] - 0.5) < 0.01
