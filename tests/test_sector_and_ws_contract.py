"""扇区基数（0 基 → 1 基）与 WS 字段契约回归测试。

背景（2026-09 审计，两处静默失效）：
1. UDP ``m_sector`` 是 **0 基**（0/1/2），但前端直接 ``'S' + sector`` 显示成
   "S0/S1/S2"（且 CSS ``sector-0`` 不存在），规则 5 判断 ``sector == 3``
   也永不成立。现统一由 ``telemetry.packets.to_sector_1based`` 转 1 基。
2. WS ``telemetry`` 事件只推 ``engine_rpm`` 等 7 个字段，而前端读
   ``t.rpm`` / ``t.lap_time_ms`` / ``t.sector`` → 实时面板的「转速」「圈速」
   恒为 "—"。现后端补齐字段、前端改用 ``engine_rpm``。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from setup_tuner.api.ws import _push_corner_highlight, _push_telemetry_frame
from setup_tuner.report.builder import extract_telemetry_summary
from setup_tuner.telemetry.packets import to_sector_1based

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_JS = _REPO_ROOT / "setup_tuner" / "ui" / "app.js"


class _FakeManager:
    """记录 broadcast 事件的假管理器。"""

    def __init__(self, connections: int = 1) -> None:
        self.connections = connections
        self.events: list[tuple[str, dict]] = []

    @property
    def connection_count(self) -> int:
        return self.connections

    async def broadcast(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))


class _FakeState:
    current_track_id = "suzuka"


# ===========================================================================
# 1. 扇区基数
# ===========================================================================
class TestSectorBase:
    """``to_sector_1based`` 与摘要中的 sector。"""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0, 1), (1, 2), (2, 3), (3, 3), (-1, 1), (None, None)],
    )
    def test_conversion(self, raw: int | None, expected: int | None) -> None:
        assert to_sector_1based(raw) == expected

    def test_non_numeric(self) -> None:
        assert to_sector_1based("abc") is None
        assert to_sector_1based(True) is None

    def test_summary_sector_is_one_based(self) -> None:
        """遥测摘要里的 sector 是 1 基（前端与规则 5 依赖该口径）。"""
        summary = extract_telemetry_summary({2: {"m_sector": 2, "m_lapDistance": 100.0}})
        assert summary["sector"] == 3
        summary0 = extract_telemetry_summary({2: {"m_sector": 0}})
        assert summary0["sector"] == 1


# ===========================================================================
# 2. WS 字段契约
# ===========================================================================
class TestWsPayloadContract:
    """后端推送字段必须覆盖前端读取字段。"""

    def _payload(self) -> dict:
        manager = _FakeManager()
        all_latest = {
            6: {"m_speed": 280, "m_throttle": 0.9, "m_brake": 0.0, "m_steer": 0.05,
                "m_gear": 7, "m_engineRPM": 11500, "m_drs": 1},
            2: {"m_lastLapTimeInMS": 91234, "m_sector": 1, "m_lapDistance": 1200.0},
        }
        asyncio.run(_push_telemetry_frame(manager, all_latest))
        assert manager.events and manager.events[0][0] == "telemetry"
        return manager.events[0][1]

    def test_payload_keys(self) -> None:
        """payload 含前端所需的全部字段，且 sector 为 1 基。"""
        payload = self._payload()
        for key in ("speed", "throttle", "brake", "steer", "gear",
                    "engine_rpm", "drs", "lap_time_ms", "sector"):
            assert key in payload, f"payload 缺少 {key}"
        assert payload["lap_time_ms"] == 91234
        assert payload["sector"] == 2  # UDP 1 → 1 基 2

    def test_frontend_reads_only_pushed_keys(self) -> None:
        """静态校验：前端 onTelemetry 读取的键必须都在 payload 里。

        例外：``last_lap_time_ms`` 是前端保留的兜底字段（后端暂不推送），
        属可接受的冗余分支。
        """
        source = _APP_JS.read_text(encoding="utf-8")
        match = re.search(r"function onTelemetry\(t\) \{(.*?)\n  \}", source, re.S)
        assert match is not None, "未找到 onTelemetry 函数"
        # 去掉注释行再提取，避免注释里的示例（如 t.rpm）被误判为读取
        body = "\n".join(
            line for line in match.group(1).splitlines() if "//" not in line
        )
        reads = set(re.findall(r"t\.(\w+)", body))
        payload = self._payload()
        allowed = set(payload) | {"last_lap_time_ms"}
        assert reads <= allowed, f"前端读取了后端不推送的字段：{sorted(reads - allowed)}"

    def test_frontend_sector_not_offset(self) -> None:
        """前端不得再对 sector 做二次 +1（转换已在后端完成）。"""
        source = _APP_JS.read_text(encoding="utf-8")
        assert "t.sector + 1" not in source
        assert "sector + 1" not in source


# ===========================================================================
# 3. 「当前弯」事件
# ===========================================================================
class TestCornerEvent:
    """corner 事件带 1 基 sector 与精确弯号。"""

    def test_corner_event_uses_arc_mapping(self) -> None:
        manager = _FakeManager()
        # suzuka 全长 5807m，26% ≈ 1510m：真实 GPS 几何下 T7 弯心在 ≈1500m
        # （S 弯群后段），该点恰在 T7 弯心旁，应判为 T7。
        # （2026-09-17 真实几何重标；更早的 T8 断言系示意表整体错位）
        all_latest = {2: {"m_lapDistance": 1510.0, "m_sector": 0}}
        corner = asyncio.run(
            _push_corner_highlight(manager, all_latest, _FakeState(), None),
        )
        assert corner == 7
        assert manager.events and manager.events[0][0] == "corner"
        payload = manager.events[0][1]
        assert payload["corner_number"] == 7
        assert payload["sector"] == 1  # UDP 0 → 1 基 1
        assert payload["track_id"] == "suzuka"
