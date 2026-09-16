"""WebSocket 网关 —— 单连接多事件，实时推送遥测/弯道/建议。

对齐 design.md 2.8.3 WebSocket 事件清单：

    | 事件名           | 方向       | 负载结构                                   | 触发               |
    |------------------|-----------|-------------------------------------------|--------------------|
    | telemetry        | 服务→前端 | {speed, throttle, gear, rpm, ...}          | 遥测新帧（≤60Hz）  |
    | corner           | 服务→前端 | {track_id, corner_number, sector}          | 落点映射变化        |
    | suggestion       | 服务→前端 | {suggestion_id, report_json}               | 建议生成完成        |
    | telemetry_status | 服务→前端 | {connected: bool}                          | 遥测连接/断开       |

设计要点：
    - 单连接多事件（一个 WS 通道，事件名区分）；
    - 服务端 ``TelemetryStream`` 最新帧缓存驱动推送；
    - 遥测不落库，仅推内存态；
    - 节流 ≤ 60Hz（每帧间隔 ≥ 16.67ms）；
    - 接收客户端消息（如选赛道、请求建议）。
"""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS
from setup_tuner.domain.corner_groups import (
    build_corner_groups,
    group_for_progress,
    nearest_member,
)
from setup_tuner.telemetry.packets import to_sector_1based

logger = logging.getLogger(__name__)

# WebSocket 路由
ws_router = APIRouter()

# 节流间隔（60Hz → 1/60 ≈ 0.01667 秒）
_THROTTLE_INTERVAL_SEC = 1.0 / 60.0


# =========================================================================== #
# WS 连接管理器（支持多连接广播）
# =========================================================================== #
class WSManager:
    """WebSocket 连接管理器 —— 管理活跃连接，支持广播。

    线程安全：使用 asyncio.Lock 保护连接列表的增删。
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        """接受并注册一个 WS 连接。"""
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WS connected, total=%d", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        """移除一个 WS 连接。"""
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info("WS disconnected, total=%d", len(self._connections))

    async def broadcast(self, event: str, payload: dict[str, Any]) -> None:
        """向所有活跃连接广播事件。

        Args:
            event: 事件名（telemetry/corner/suggestion/telemetry_status）。
            payload: 事件负载字典。
        """
        message = json.dumps(
            {"event": event, "payload": payload},
            ensure_ascii=False,
        )
        async with self._lock:
            connections = list(self._connections)
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                logger.debug("broadcast to a WS failed, removing", exc_info=True)
                async with self._lock:
                    if ws in self._connections:
                        self._connections.remove(ws)

    async def send_to(self, ws: WebSocket, event: str, payload: dict[str, Any]) -> None:
        """向单个连接发送事件。"""
        message = json.dumps(
            {"event": event, "payload": payload},
            ensure_ascii=False,
        )
        await ws.send_text(message)

    @property
    def connection_count(self) -> int:
        """当前活跃连接数。"""
        return len(self._connections)


# =========================================================================== #
# 弯道落点映射
# =========================================================================== #
# 弧长占比表：{track_id: ([排序后的占比], [对应弯道号])}，进程内构建一次
_ARC_TABLES: dict[str, tuple[list[float], list[int]]] | None = None


def _arc_tables() -> dict[str, tuple[list[float], list[int]]]:
    """构建「弯道弧长占比」二分查找表（进程内缓存，仅首次调用有开销）。"""
    global _ARC_TABLES
    if _ARC_TABLES is None:
        tables: dict[str, tuple[list[float], list[int]]] = {}
        for track_id, corners in TRACK_CORNER_ARCS.items():
            ordered = sorted(corners.items(), key=lambda kv: kv[1])
            tables[track_id] = (
                [fraction for _, fraction in ordered],
                [number for number, _ in ordered],
            )
        _ARC_TABLES = tables
    return _ARC_TABLES


def _map_corner(
    lap_distance: float,
    track_length: float,
    corners: list[Any],
    track_id: str | None = None,
) -> int | None:
    """根据圈内距离映射当前弯道编号。

    实现（2026-09 修正）：
        使用 ``domain/_track_arcs.TRACK_CORNER_ARCS`` —— 由
        ``scripts/gen_track_arcs.py`` 依据 ``ui/tracks/*.svg`` 的真实路径算出
        每个弯道锚点的累计弧长占比。把 ``lap_distance`` 归一化为圈内进度后，
        在该占比序列上做**循环最近邻**二分查找。

        早期实现按「弯道沿赛道均匀分布」（``idx = int(progress * total)``）近似，
        但弯道在真实赛道上并不等距：云端实测 24 赛道 × 200 采样点中判定错误
        3313/4800 = **69%**。修正后错误率 0%，单次查询约 0.2–0.3 µs。

        闭合回路归一化（2026-09 二次修正）：赛道 SVG 是闭合回路（折线首尾重合），
        当锚点恰好落在起跑线时，最近点在数值上会命中回路末端而返回 ≈1.0，
        使 T1 被误判为全圈最后一个弯，导致整条赛道弯号错位。
        ``gen_track_arcs.nearest_arc_fraction`` 现对「投影落在回路闭合点」做归一化，
        统一返回 0.0（起跑线）；受影响的 melbourne / mexico_city / monza / spielberg
        已修正，24 条赛道弧长表全部单调且 T1 ≈ 0。

    Args:
        lap_distance: 当前圈距离（米）。
        track_length: 赛道长度（米）。
        corners: 弯道列表（domain.Corner），仅回退路径使用。
        track_id: 赛道标识；提供且有弧长表时走精确路径。

    Returns:
        当前弯道编号（1-based）；无法映射时返回 None。
    """
    if track_length <= 0 or not corners:
        return None
    progress = (lap_distance % track_length) / track_length

    table = _arc_tables().get(track_id) if track_id else None
    if table is not None:
        fractions, numbers = table
        i = bisect.bisect_left(fractions, progress)
        prev_fraction = fractions[i - 1] if i > 0 else fractions[-1] - 1.0
        next_fraction = fractions[i] if i < len(fractions) else fractions[0] + 1.0
        # 循环比较（路径闭合，末段与首段相邻）
        if (progress - prev_fraction) <= (next_fraction - progress):
            return numbers[i - 1]
        return numbers[i % len(numbers)]

    # 回退：无弧长表的赛道仍用均匀分布近似（兼容未知赛道）
    total = len(corners)
    idx = int(progress * total)
    if idx >= total:
        idx = total - 1
    return corners[idx].number


# =========================================================================== #
# 遥测推送循环
# =========================================================================== #
async def _push_telemetry_status(
    ws_manager: Any, listener: Any, last_connected: bool | None,
) -> bool | None:
    """推送遥测连接状态（变化时），返回当前连接状态。"""
    connected = listener.is_running if listener else False
    if connected != last_connected:
        if ws_manager is not None:
            await ws_manager.broadcast(
                event="telemetry_status",
                payload={"connected": connected},
            )
    return connected


async def _push_telemetry_frame(ws_manager: Any, all_latest: dict[int, dict[str, Any]]) -> None:
    """推送遥测关键帧（Packet 6 CarTelemetry + Packet 2 LapData 的圈速/扇区）。

    字段契约（前端 ``ui/app.js::onTelemetry`` 读取）：
        ``speed / throttle / brake / steer / gear / engine_rpm / drs /
        lap_time_ms / sector``（sector 为 1 基）。

    修正记录：早期只推 ``engine_rpm``，而前端读 ``t.rpm``；且完全不推
    ``lap_time_ms`` / ``sector`` → 实时面板的「转速」「圈速」恒为 "—"。
    """
    telemetry_data = all_latest.get(6)
    if telemetry_data is None:
        return
    lap_data = all_latest.get(2) or {}
    payload = {
        "speed": telemetry_data.get("m_speed"),
        "throttle": telemetry_data.get("m_throttle"),
        "brake": telemetry_data.get("m_brake"),
        "steer": telemetry_data.get("m_steer"),
        "gear": telemetry_data.get("m_gear"),
        "engine_rpm": telemetry_data.get("m_engineRPM"),
        "drs": telemetry_data.get("m_drs"),
        "lap_time_ms": lap_data.get("m_lastLapTimeInMS"),
        "sector": to_sector_1based(lap_data.get("m_sector")),
    }
    if ws_manager is not None:
        await ws_manager.broadcast(event="telemetry", payload=payload)


_GROUPS_CACHE: dict[str, list[dict[str, Any]]] = {}


def _get_groups(track_id: str) -> list[dict[str, Any]]:
    """按赛道缓存弯道段（TRACK_CORNER_ARCS 静态，进程内缓存即可）。"""
    if track_id not in _GROUPS_CACHE:
        _GROUPS_CACHE[track_id] = build_corner_groups(
            TRACK_CORNER_ARCS.get(track_id, {}),
        )
    return _GROUPS_CACHE[track_id]


def _map_corner_segment(
    lap_distance: float, track_length: float, track_id: str,
    corners: list[Any],
) -> tuple[int | None, str | None, list[int] | None]:
    """task-63：段化定位 —— 进度 → 段（区间归属）→ 段内最近成员。

    连续弯（全赛道 39% 相邻弯对间距 < 4% 圈长）在单弯最近邻下会随微小
    偏移高频跳变；段级映射把跳变收敛到段边界。无弧长表的赛道回退
    :func:`_map_corner`。

    Returns:
        ``(corner_number, group_name, group_members)``；group 仅在
        多弯段时非 None（单弯段无需段表达）。
    """
    arcs = TRACK_CORNER_ARCS.get(track_id)
    if not arcs or track_length <= 0:
        return _map_corner(lap_distance, track_length, corners, track_id), None, None

    progress = (lap_distance / track_length) % 1.0
    group = group_for_progress(progress, _get_groups(track_id))
    if group is None:
        return _map_corner(lap_distance, track_length, corners, track_id), None, None
    corner = nearest_member(progress, group, arcs)
    if len(group["members"]) > 1:
        return corner, group["name"], list(group["members"])
    return corner, None, None


async def _push_corner_highlight(
    ws_manager: Any, all_latest: dict[int, dict[str, Any]], app_state: Any,
    last_corner: int | None,
) -> int | None:
    """推送当前弯道高亮（落点映射变化时），返回当前弯道编号。"""
    lap_data = all_latest.get(2)
    if lap_data is None:
        return last_corner
    lap_distance = lap_data.get("m_lapDistance")
    sector = to_sector_1based(lap_data.get("m_sector"))
    current_track_id = getattr(app_state, "current_track_id", None)
    if lap_distance is None or current_track_id is None:
        return last_corner
    from setup_tuner.domain.track import get_track_by_id

    track = get_track_by_id(current_track_id)
    if track is None:
        return last_corner
    corner_number, group_name, group_members = _map_corner_segment(
        float(lap_distance), track.length_m, current_track_id, track.corners,
    )
    if corner_number is not None and corner_number != last_corner:
        if ws_manager is not None:
            await ws_manager.broadcast(
                event="corner",
                payload={
                    "track_id": current_track_id,
                    "corner_number": corner_number,
                    "corner_group": group_name,
                    "corner_group_members": group_members,
                    "sector": sector,
                },
            )
    return corner_number if corner_number is not None else last_corner


async def _telemetry_push_loop(app_state: Any) -> None:
    """应用级**单一**推送循环（所有连接共享）。

    设计要点（性能）：
        此前是「每个连接各起一个推送循环」，而每个循环都调用
        ``ws_manager.broadcast()`` 向**所有**连接发送，于是 N 个客户端
        每 tick 产生 **N×N** 次快照读取 + 序列化 + 发送（实测 N=8 时
        每 tick 64 次、60Hz 下占单核 0.74%，并随 N² 增长）。
        现在改为单个应用级任务：每 tick 只取一次快照、只序列化一次，
        由 ``broadcast`` 完成 O(N) 扇出。

    推送事件：telemetry / corner / telemetry_status。
    节流 ≤ 60Hz（每帧间隔 ≥ 16.67ms）。
    """
    last_push_time = 0.0
    last_corner: int | None = None
    last_connected: bool | None = None

    stream = getattr(app_state, "telemetry_stream", None)
    listener = getattr(app_state, "telemetry_listener", None)
    ws_manager = getattr(app_state, "ws_manager", None)

    while True:
        if ws_manager is None or ws_manager.connection_count == 0:
            # 无客户端时不空转，等下一个 tick 再检查
            await asyncio.sleep(_THROTTLE_INTERVAL_SEC)
            continue

        await asyncio.sleep(_THROTTLE_INTERVAL_SEC)

        # 节流：距上次推送不足一个间隔则跳过
        now = time.monotonic()
        if now - last_push_time < _THROTTLE_INTERVAL_SEC:
            continue
        last_push_time = now

        if stream is None:
            continue

        # ① 遥测连接状态推送（变化时）
        last_connected = await _push_telemetry_status(ws_manager, listener, last_connected)

        # ② 遥测帧推送
        all_latest = stream.get_all_latest()
        if not all_latest:
            continue

        await _push_telemetry_frame(ws_manager, all_latest)

        # ③ 当前弯道高亮推送（落点映射变化时）
        last_corner = await _push_corner_highlight(ws_manager, all_latest, app_state, last_corner)


def _ensure_pusher(app_state: Any) -> None:
    """确保应用级推送任务存在（幂等；无客户端时由 _stop_pusher_if_idle 停掉）。"""
    task = getattr(app_state, "ws_pusher_task", None)
    if task is None or task.done():
        app_state.ws_pusher_task = asyncio.create_task(
            _telemetry_push_loop(app_state),
        )


async def _stop_pusher_if_idle(app_state: Any, ws_manager: Any) -> None:
    """最后一个连接断开后停掉推送任务，避免无人订阅时后台空转。"""
    if ws_manager is not None and ws_manager.connection_count > 0:
        return
    task = getattr(app_state, "ws_pusher_task", None)
    app_state.ws_pusher_task = None
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# =========================================================================== #
# WebSocket 端点
# =========================================================================== #
async def _handle_ws_message(
    ws: WebSocket, ws_manager: WSManager, msg: dict, app_state: Any,
) -> None:
    """处理单条客户端 WebSocket 消息。"""
    action = msg.get("action")
    if action == "select_track":
        track_id = msg.get("track_id")
        if track_id:
            app_state.current_track_id = track_id
            app_state.current_track_source = "manual"
            await ws_manager.send_to(
                ws,
                event="track_selected",
                payload={"track_id": track_id, "source": "manual"},
            )
    elif action == "request_suggestion":
        # 提示前端通过 REST POST /api/v1/suggest 触发
        await ws_manager.send_to(
            ws,
            event="info",
            payload={
                "message": "请通过 POST /api/v1/suggest 触发建议生成",
                "track_id": msg.get("track_id"),
            },
        )
    else:
        await ws_manager.send_to(
            ws,
            event="error",
            payload={"message": f"未知 action：{action}"},
        )


async def _ws_receive_loop(ws: WebSocket, ws_manager: WSManager, app_state: Any) -> None:
    """接收客户端消息循环。"""
    while True:
        try:
            text = await ws.receive_text()
        except WebSocketDisconnect:
            break
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            await ws_manager.send_to(
                ws,
                event="error",
                payload={"message": "无效的 JSON 消息"},
            )
            continue
        await _handle_ws_message(ws, ws_manager, msg, app_state)


@ws_router.websocket("/api/v1/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket 端点 —— 单连接多事件。

    服务→前端事件：telemetry / corner / suggestion / telemetry_status
    前端→服务消息（JSON）：
        - {"action": "select_track", "track_id": "suzuka"}
        - {"action": "request_suggestion", "track_id": "suzuka"}

    性能：遥测推送由**应用级单一任务**负责（首个连接接入时启动，
    最后一个连接断开时停止），本端点只处理连接注册与消息收发。
    """
    app_state = ws.app.state
    ws_manager = getattr(app_state, "ws_manager", None)

    # 如果 app.state 中没有 ws_manager，创建一个临时的（仅本连接用）
    if ws_manager is None:
        ws_manager = WSManager()
        app_state.ws_manager = ws_manager

    await ws_manager.connect(ws)
    _ensure_pusher(app_state)

    try:
        await _ws_receive_loop(ws, ws_manager, app_state)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WS endpoint error")
    finally:
        await ws_manager.disconnect(ws)
        await _stop_pusher_if_idle(app_state, ws_manager)