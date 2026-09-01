"""真实 F1 26 遥测 fixture 回归 (R4 抽样集)。

fixture 由 ``scripts/sample_f1_26_fixtures.py`` 从真实采集 JSONL(112,036 帧)
等间隔抽样生成。本测试验证 ``f1opt.telemetry.packets.parse_packet`` 对真实数据:

- 全部样本可解析不崩溃, 头部字段与采集记录一致;
- CarTelemetry2 (pid 16) 的 2026 新字段存在;
- 关键边界: g-force -32768 饱和哨兵、空帧、截断帧。

fixture 路径: ``tests/data/real_f1_26_sample.jsonl``
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from f1opt.telemetry.packets import parse_packet

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "real_f1_26_sample.jsonl"

# 真实数据中出现的 14 种包型 (缺 8 FinalClassification / 9 LobbyInfo / 14 TimeTrial,
# 数据未出现, 不视为解析缺陷)
PRESENT_PACKET_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16}


def _load_frames() -> list[dict]:
    frames: list[dict] = []
    with open(FIXTURE, encoding="utf-8") as f:
        for line in f:
            frames.append(json.loads(line))
    return frames


_FRAMES = _load_frames()
_REAL = [f for f in _FRAMES if not f.get("synthetic")]
_SYNTHETIC = [f for f in _FRAMES if f.get("synthetic")]


def _real_frames(packet_id: int) -> list[dict]:
    return [f for f in _REAL if f.get("packetId") == packet_id]


# --------------------------------------------------------------------------- #
# 覆盖性
# --------------------------------------------------------------------------- #
def test_fixture_covers_all_present_packet_types() -> None:
    got = {f.get("packetId") for f in _REAL}
    assert PRESENT_PACKET_IDS <= got, f"missing packet types: {PRESENT_PACKET_IDS - got}"


def test_fixture_covers_event_codes() -> None:
    codes = set()
    for f in _REAL:
        if f.get("packetId") == 3:
            # 头 29B(58 hex 字符) 之后 4 字节即事件码
            codes.add(bytes.fromhex(f["hex"][58:66]).decode("ascii", "replace").rstrip("\x00"))
    assert len(codes) >= 10


# --------------------------------------------------------------------------- #
# 真实帧: 解析 + 头部字段一致
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fr", _REAL, ids=lambda fr: f"pid{fr['packetId']}@{fr.get('ts')}")
def test_real_frame_parses_and_header_matches(fr: dict) -> None:
    data = bytes.fromhex(fr["hex"])
    hdr, parsed = parse_packet(data)
    assert hdr.packet_id == fr["packetId"]
    assert hdr.session_uid == fr.get("sessionUID")
    assert hdr.frame_identifier == fr.get("frameIdentifier")
    assert hdr.player_car_index == fr.get("playerCarIndex")
    assert hdr.secondary_player_car_index == fr.get("secondaryPlayerCarIndex")
    assert isinstance(parsed, dict)


# --------------------------------------------------------------------------- #
# CarTelemetry2 (pid 16) — F1 2026 新字段
# --------------------------------------------------------------------------- #
def test_car_telemetry_2_has_2026_fields() -> None:
    frames = _real_frames(16)
    assert frames, "fixture 应含 CarTelemetry2 样本"
    _, parsed = parse_packet(bytes.fromhex(frames[0]["hex"]))
    cars = parsed["m_carTelemetryData2"]
    first = cars[0]
    for key in (
        "m_activeAeroMode",
        "m_overtakeAvailable",
        "m_overtakeActive",
        "m_2026Regulations",
    ):
        assert key in first, f"CarTelemetry2 缺少 2026 字段 {key}"


# --------------------------------------------------------------------------- #
# 边界: g-force 饱和哨兵 (-32768 ÷1000 = -32.768)
# --------------------------------------------------------------------------- #
def test_motion_gforce_saturation_sentinel_present() -> None:
    found = False
    for fr in _real_frames(0):
        _, parsed = parse_packet(bytes.fromhex(fr["hex"]))
        for c in parsed["m_carMotionData"]:
            if c.get("m_gForceLateral") == pytest.approx(-32.768, abs=1e-3):
                found = True
                break
        if found:
            break
    assert found, "fixture 应含 g-force raw -32768 饱和哨兵帧 (÷1000 = -32.768)"


# --------------------------------------------------------------------------- #
# 边界: 空帧 / 截断帧
# --------------------------------------------------------------------------- #
def test_empty_frame_raises_value_error() -> None:
    empty = [f for f in _SYNTHETIC if f.get("boundary") == "empty"]
    assert empty
    with pytest.raises(ValueError):
        parse_packet(bytes.fromhex(empty[0]["hex"]))


def test_truncated_header_only_does_not_crash() -> None:
    trunc = [f for f in _SYNTHETIC if f.get("boundary") == "truncated"]
    assert trunc
    hdr, parsed = parse_packet(bytes.fromhex(trunc[0]["hex"]))
    assert hdr.packet_id == 0
    # _unpack_body 对截断体零填充, 不抛异常
    assert isinstance(parsed, dict)
