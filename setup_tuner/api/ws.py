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
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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
def _map_corner(
    lap_distance: float,
    track_length: float,
    corners: list[Any],
) -> int | None:
    """根据圈距离映射当前弯道编号。

    将 lap_distance 按赛道长度归一化为 0~1 的进度，再映射到弯道列表的编号。
    弯道锚点沿椭圆分布（见 domain/track.py _estimate_anchor），此处用
    弯道编号在总弯道数中的均匀分布近似定位。

    Args:
        lap_distance: 当前圈距离（米）。
        track_length: 赛道长度（米）。
        corners: 弯道列表（domain.Corner）。

    Returns:
        当前弯道编号（1-based）；无法映射时返回 None。
    """
    if track_length <= 0 or not corners:
        return None
    progress = (lap_distance % track_length) / track_length
    total = len(corners)
    # 均匀分布映射：progress * total → 弯道编号
    idx = int(progress * total)
    if idx >= total:
        idx = total - 1
    return corners[idx].number


# =========================================================================== #
# 遥测推送循环
# =========================================================================== #
async def _telemetry_push_loop(ws: WebSocket, app_state: Any) -> None:
    """遥测推送循环 —— 从 TelemetryStream 读取最新帧并推送。

    节流 ≤ 60Hz（每帧间隔 ≥ 16.67ms）。
    推送事件：telemetry / corner / telemetry_status。
    """
    last_push_time = 0.0
    last_corner: int | None = None
    last_connected: bool | None = None

    stream = getattr(app_state, "telemetry_stream", None)
    listener = getattr(app_state, "telemetry_listener", None)
    ws_manager = getattr(app_state, "ws_manager", None)

    while True:
        await asyncio.sleep(_THROTTLE_INTERVAL_SEC)

        # 节流：距上次推送不足一个间隔则跳过
        now = time.monotonic()
        if now - last_push_time < _THROTTLE_INTERVAL_SEC:
            continue
        last_push_time = now

        if stream is None:
            continue

        # ① 遥测连接状态推送（变化时）
        connected = listener.is_running if listener else False
        if connected != last_connected:
            last_connected = connected
            if ws_manager is not None:
                await ws_manager.broadcast(
                    event="telemetry_status",
                    payload={"connected": connected},
                )

        # ② 遥测帧推送
        all_latest = stream.get_all_latest()
        if not all_latest:
            continue

        # 提取遥测关键帧（Packet 6 CarTelemetry）
        telemetry_data = all_latest.get(6)
        if telemetry_data is not None:
            payload = {
                "speed": telemetry_data.get("m_speed"),
                "throttle": telemetry_data.get("m_throttle"),
                "brake": telemetry_data.get("m_brake"),
                "steer": telemetry_data.get("m_steer"),
                "gear": telemetry_data.get("m_gear"),
                "engine_rpm": telemetry_data.get("m_engineRPM"),
                "drs": telemetry_data.get("m_drs"),
            }
            if ws_manager is not None:
                await ws_manager.broadcast(event="telemetry", payload=payload)

        # ③ 当前弯道高亮推送（落点映射变化时）
        lap_data = all_latest.get(2)
        if lap_data is not None:
            lap_distance = lap_data.get("m_lapDistance")
            sector = lap_data.get("m_sector")
            current_track_id = getattr(app_state, "current_track_id", None)
            if lap_distance is not None and current_track_id is not None:
                from setup_tuner.domain.track import get_track_by_id

                track = get_track_by_id(current_track_id)
                if track is not None:
                    corner_number = _map_corner(
                        float(lap_distance), track.length_m, track.corners
                    )
                    if corner_number is not None and corner_number != last_corner:
                        last_corner = corner_number
                        if ws_manager is not None:
                            await ws_manager.broadcast(
                                event="corner",
                                payload={
                                    "track_id": current_track_id,
                                    "corner_number": corner_number,
                                    "sector": sector,
                                },
                            )


# =========================================================================== #
# WebSocket 端点
# =========================================================================== #
@ws_router.websocket("/api/v1/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket 端点 —— 单连接多事件。

    服务→前端事件：telemetry / corner / suggestion / telemetry_status
    前端→服务消息（JSON）：
        - {"action": "select_track", "track_id": "suzuka"}
        - {"action": "request_suggestion", "track_id": "suzuka"}
    """
    app_state = ws.app.state
    ws_manager = getattr(app_state, "ws_manager", None)

    # 如果 app.state 中没有 ws_manager，创建一个临时的（仅本连接用）
    if ws_manager is None:
        ws_manager = WSManager()
        app_state.ws_manager = ws_manager

    await ws_manager.connect(ws)

    # 启动遥测推送循环（后台任务）
    push_task = asyncio.create_task(_telemetry_push_loop(ws, app_state))

    try:
        # 接收客户端消息循环
        while True:
            try:
                text = await ws.receive_text()
            except WebSocketDisconnect:
                break

            # 解析客户端消息
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                await ws_manager.send_to(
                    ws,
                    event="error",
                    payload={"message": "无效的 JSON 消息"},
                )
                continue

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
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WS endpoint error")
    finally:
        push_task.cancel()
        try:
            await push_task
        except asyncio.CancelledError:
            pass
        await ws_manager.disconnect(ws)