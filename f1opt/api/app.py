"""FastAPI backend for F1OPT: REST endpoints + real-time telemetry WebSocket.

Exposes :func:`create_app` (factory) and a module-level :data:`app` instance for
``uvicorn f1opt.api.app:app``. The WebSocket ``/ws/telemetry`` streams unified
60Hz telemetry frames (produced by :class:`~f1opt.telemetry.aligner.TelemetryAligner`)
and completed clean laps (from :class:`~f1opt.telemetry.aggregator.LapAggregator`)
to all connected clients.

Broadcast design: each connected WS client owns a bounded ``asyncio.Queue``
(maxsize=64, drop-oldest on full). The UDP listener's async subscriber runs in
the listener dispatch task (a background task relative to the request handlers)
and enqueues unified frames / lap messages into every client queue. WS handlers
only read from their own queue and send — the hot recv path never blocks on a
slow client.
"""

from __future__ import annotations

import asyncio
import os as _os
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Windows: 兼容性处理。
# ProactorEventLoop 自 Python 3.8 起即为 Windows 默认事件循环，无需显式设置
# （显式设置 WindowsProactorEventLoopPolicy 在 Python 3.14+ 已弃用）；
# subprocess 默认 close_fds=True 已正确处理句柄继承。
if sys.platform == "win32":
    # Windows: 设置 multiprocessing 启动方法为 spawn (避免 fork 问题)
    import multiprocessing as _mp
    if hasattr(_mp, "set_start_method"):
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # 已设置, 忽略

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.staticfiles import StaticFiles

from f1opt import __version__
from f1opt.config import get_settings
from f1opt.data.setup_schema import DEFAULT_SETUP, CarSetup
from f1opt.data.tracks import ALL_TRACKS, TRACKS_BY_ID, get_track
from f1opt.model.online_correction import ObservationBuffer, add_observation
from f1opt.observability.audit import get_audit_logger
from f1opt.observability.metrics import MetricsRegistry
from f1opt.observability.tracing import is_tracing_enabled, span
from f1opt.telemetry.aggregator import LapAggregator
from f1opt.telemetry.aligner import TelemetryAligner
from f1opt.telemetry.listener import TelemetryListener
from f1opt.telemetry.packets import PacketHeader

# Fields exposed by the public Track dict (shared API contract).
_TRACK_FIELDS: tuple[str, ...] = (
    "track_id",
    "official_name",
    "circuit_name",
    "country",
    "round_number",
    "is_sprint",
    "length_m",
    "corners",
    "track_type",
)

#: Max items buffered per WS client before drop-oldest kicks in.
_CLIENT_QUEUE_MAX = 64

# 迭代历史记录位于 <project-root>/半成品/.trae/iterations/iter-NN.md.
# app.py is at f1opt/api/app.py → parents[2] = project root (workspace).
_ITERATIONS_DIR = Path(__file__).resolve().parents[2] / "半成品" / ".trae" / "iterations"
# Path-traversal guard for /api/iterations/{iter}: only iter-NN is allowed.
_ITER_ID_RE = re.compile(r"^iter-\d+$")
# Listing filter: primary iter-NN.md records (excludes verification/etc. files).
_ITER_FILE_RE = re.compile(r"^iter-(\d+)\.md$")
#: Max chars of the summary_preview returned by GET /api/iterations.
_ITER_SUMMARY_MAX = 300


def _track_dict(track: Any) -> dict[str, Any]:
    """Project a Track model onto the public Track dict (shared contract)."""
    return {k: getattr(track, k) for k in _TRACK_FIELDS}


def _frame_to_ws(frame: dict[str, Any]) -> dict[str, Any]:
    """Project a unified aligner frame onto the WS frame message (shared contract)."""
    return {
        "type": "frame",
        "t": frame.get("session_time"),
        "speed": frame.get("speed"),
        "throttle": frame.get("throttle"),
        "brake": frame.get("brake"),
        "steer": frame.get("steer"),
        "gear": frame.get("gear"),
        "rpm": frame.get("rpm"),
        "g_lat": frame.get("g_lat"),
        "g_long": frame.get("g_long"),
        "lap_time": frame.get("lap_time"),
        "lap_distance": frame.get("lap_distance"),
        "ers_store": frame.get("ers_store"),
        "drs_allowed": frame.get("drs_allowed"),
        "tyre_wear_fl": frame.get("tyre_wear_fl"),
        "tyre_wear_fr": frame.get("tyre_wear_fr"),
        "tyre_wear_rl": frame.get("tyre_wear_rl"),
        "tyre_wear_rr": frame.get("tyre_wear_rr"),
        "fuel_in_tank": frame.get("fuel_in_tank"),
        # F1 2026 主动空力 (X=低阻 / Z=高下压力) — 供 UI 指示器显示
        "active_aero_x": frame.get("active_aero_x"),
        "active_aero_z": frame.get("active_aero_z"),
    }


def _iter_meta(text: str) -> tuple[str, str]:
    """Parse an iteration markdown file into ``(title, summary_preview)``.

    ``title`` is the first H1 (``# …``) or, failing that, the first non-empty
    line. ``summary_preview`` is the first paragraph after the title (blockquote
    ``>`` markers stripped), truncated to :data:`_ITER_SUMMARY_MAX` chars.
    """
    lines = text.splitlines()
    title = ""
    title_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        title = s[2:].strip() if s.startswith("# ") else s
        title_idx = i
        break
    if title_idx < 0:
        return "", ""
    para: list[str] = []
    for line in lines[title_idx + 1 :]:
        s = line.strip()
        if not s:
            if para:
                break
            continue
        if s.startswith(">"):
            s = s.lstrip(">").strip()
        if s:
            para.append(s)
    summary = " ".join(para)
    if len(summary) > _ITER_SUMMARY_MAX:
        summary = summary[:_ITER_SUMMARY_MAX]
    return title, summary


class ConnectionManager:
    """Per-client bounded queues with drop-oldest broadcast.

    ``broadcast`` is non-blocking and safe to call from inside the event loop
    (it uses ``put_nowait``). For cross-thread pushes (e.g. from tests) use
    :func:`push_frame`, which schedules the broadcast via
    ``loop.call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()

    def register(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_CLIENT_QUEUE_MAX)
        self._queues.add(q)
        return q

    def unregister(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._queues.discard(q)

    def broadcast(self, msg: dict[str, Any]) -> None:
        """Enqueue ``msg`` to every client; drop-oldest on a full queue."""
        for q in list(self._queues):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass


class _TelemetryState:
    """Holds the runtime telemetry objects wired during lifespan startup."""

    def __init__(self) -> None:
        self.aligner = TelemetryAligner()
        settings = get_settings()
        self.lap_agg = LapAggregator(
            output_path=Path(settings.data_dir) / "laps.parquet"
        )
        self.manager = ConnectionManager()
        self.listener: TelemetryListener | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_row_count: int = 0
        # Last-known player-car sector times (ms) for lap-message assembly.
        self.last_sector1_ms: int = 0
        self.last_sector2_ms: int = 0
        # Iter-06: process-wide latency metrics for /api/metrics.
        self.metrics = MetricsRegistry()
        # Iter-117: 反馈闭环状态 — ObservationBuffer + player car setup 缓存 + 当前赛道.
        # player_setup_cache: 最新 packet 5 解析的 player car CarSetup (用于反馈闭环).
        # current_track_id: session metadata 设置的当前赛道 (aggregator row track_id=-1 时回退).
        self.obs_buffer = ObservationBuffer()
        self.player_setup_cache: CarSetup | None = None
        self.current_track_id: str | None = None


def _emit_lap(state: _TelemetryState, row: dict[str, Any]) -> None:
    """Broadcast a ``{"type":"lap",...}`` message for a completed clean lap."""
    lap_ms = int(row.get("lap_time_ms", 0))
    s1 = state.last_sector1_ms
    s2 = state.last_sector2_ms
    s3 = max(0, lap_ms - s1 - s2)
    # track_id: aggregator row 的 track_id 是 int8 (-1=未知). 若未知, 用 state.current_track_id.
    track_id_int = int(row.get("track_id", -1))
    track_id: str | None
    if 0 <= track_id_int < len(ALL_TRACKS):
        track_id = ALL_TRACKS[track_id_int].track_id
    else:
        track_id = state.current_track_id
    msg: dict[str, Any] = {
        "type": "lap",
        "lap_time": lap_ms / 1000.0,
        "clean": bool(row.get("clean", True)),
        "sector_times": [s1 / 1000.0, s2 / 1000.0, s3 / 1000.0],
        "track_id": track_id,
    }
    state.manager.broadcast(msg)


# Iter-117: packet 5 (CarSetups) m_carSetups 字段名 → CarSetup 构造参数映射.
# packet 5 有 22 字段 (4 胎压 + ballast), CarSetup 有 19 字段 (前/后胎压各 1 + fuel).
# 胎压取左右平均, ballast 忽略 (CarSetup 无此字段).
_PACKET5_TO_CARSETUP: dict[str, str] = {
    "m_frontWing": "front_wing",
    "m_rearWing": "rear_wing",
    "m_onThrottleDiff": "on_throttle_diff",
    "m_offThrottleDiff": "off_throttle_diff",
    "m_frontCamber": "front_camber",
    "m_rearCamber": "rear_camber",
    "m_frontToe": "front_toe",
    "m_rearToe": "rear_toe",
    "m_frontSuspension": "front_suspension",
    "m_rearSuspension": "rear_suspension",
    "m_frontAntiRollBar": "front_arb",
    "m_rearAntiRollBar": "rear_arb",
    "m_frontSuspensionHeight": "front_ride_height",
    "m_rearSuspensionHeight": "rear_ride_height",
    "m_brakePressure": "brake_pressure",
    "m_brakeBias": "front_brake_bias",
    "m_fuelLoad": "fuel_load",
}


def _packet5_setup_to_carsetup(pkt5_setup: dict[str, Any]) -> CarSetup | None:
    """把 packet 5 的单车 setup dict 映射为 CarSetup.

    胎压取左右平均 (front_left+front_right)/2, (rear_left+rear_right)/2.
    返回 None 表示字段缺失/无效 (无法构造合法 CarSetup).
    """
    try:
        kwargs: dict[str, Any] = {}
        for pkt_name, cs_name in _PACKET5_TO_CARSETUP.items():
            if pkt_name in pkt5_setup:
                kwargs[cs_name] = pkt5_setup[pkt_name]
        # 胎压平均 (packet 5 有 4 个胎压, CarSetup 有前/后各 1)
        fl = float(pkt5_setup.get("m_frontLeftTyrePressure", 0))
        fr = float(pkt5_setup.get("m_frontRightTyrePressure", 0))
        rl = float(pkt5_setup.get("m_rearLeftTyrePressure", 0))
        rr = float(pkt5_setup.get("m_rearRightTyrePressure", 0))
        if fl > 0 or fr > 0:
            kwargs["front_tyre_pressure"] = (fl + fr) / 2.0
        if rl > 0 or rr > 0:
            kwargs["rear_tyre_pressure"] = (rl + rr) / 2.0
        return CarSetup(**kwargs)
    except Exception:
        return None


def _feed_observation_buffer(state: _TelemetryState, row: dict[str, Any]) -> None:
    """Iter-117: 把 clean lap row 喂给 ObservationBuffer (反馈闭环桥接).

    只处理 player car (car_index == 0 默认) 的 clean lap. 需要:
    - state.player_setup_cache 中有该 session+car 的 setup (来自 packet 5)
    - row 的 track_id 已解析 (aggregator 当前写 -1=未知, 用 session metadata 补)
    - row 的 quality_flag 为 OK/SUSPECT (INVALID 跳过)

    观测质量映射: clean=True + quality_flag=OK → 1.0 (排位赛飞驰圈),
    clean=True + SUSPECT → 0.5 (练习赛), clean=False → 跳过.
    """
    if not state.player_setup_cache:
        return
    if not bool(row.get("clean", False)):
        return
    quality_flag = str(row.get("quality_flag", "OK"))
    if quality_flag == "INVALID":
        return

    # setup: 取缓存中最新 player car setup
    setup = state.player_setup_cache
    if setup is None:
        return

    # track_id: aggregator row 的 track_id 是 int8 (-1=未知). 若未知, 用 state.current_track_id.
    track_id_int = int(row.get("track_id", -1))
    if 0 <= track_id_int < len(ALL_TRACKS):
        track_id = ALL_TRACKS[track_id_int].track_id
    elif state.current_track_id is not None:
        track_id = state.current_track_id
    else:
        return  # 无法确定赛道, 跳过

    lap_time_s = float(row.get("lap_time_ms", 0)) / 1000.0
    if lap_time_s <= 0:
        return

    # 观测质量: OK=1.0 (高置信), SUSPECT=0.5 (中)
    quality = 1.0 if quality_flag == "OK" else 0.5

    try:
        add_observation(
            state.obs_buffer,
            setup,
            track_id,
            None,  # driver_profile 未知 (telemetry 不含 driver 画像), 用中性
            lap_time_s,
            quality=quality,
        )
    except Exception:
        # 反馈闭环是 best-effort: 不阻塞遥测主路径
        pass


def _make_subscriber(state: _TelemetryState) -> Any:
    """Build the async listener subscriber that drives aligner + aggregator + WS."""

    async def _on_packet(
        header: PacketHeader, parsed: dict[str, Any], raw: bytes
    ) -> None:
        # 1. Feed the aligner (unified 60Hz timeseries).
        state.aligner.on_packet(header, parsed)
        # 2. Feed the lap aggregator (clean-lap detection).
        await state.lap_agg(header, parsed, raw)
        # 3. Track player-car sector times for lap-message assembly.
        if header.packet_id == 2:  # LapData
            cars = parsed.get("m_lapData") or []
            pci = header.player_car_index
            if 0 <= pci < len(cars):
                car = cars[pci]
                state.last_sector1_ms = int(car.get("m_sector1TimeInMS", 0))
                state.last_sector2_ms = int(car.get("m_sector2TimeInMS", 0))
        # Iter-117: 3b. 缓存 player car setup (packet 5 CarSetups) 供反馈闭环.
        if header.packet_id == 5:  # CarSetups
            setups = parsed.get("m_carSetups") or []
            pci = header.player_car_index
            if 0 <= pci < len(setups):
                cs = _packet5_setup_to_carsetup(setups[pci])
                if cs is not None:
                    state.player_setup_cache = cs
        # 4. Emit lap messages for any newly-completed clean laps.
        rows = state.lap_agg.rows
        if len(rows) > state.last_row_count:
            for row in rows[state.last_row_count :]:
                _emit_lap(state, row)
                # Iter-117: 喂给 ObservationBuffer (反馈闭环核心).
                _feed_observation_buffer(state, row)
            state.last_row_count = len(rows)
        # 5. Broadcast the latest unified frame to all WS clients.
        frame = state.aligner.latest_unified_frame()
        if frame is not None:
            state.manager.broadcast(_frame_to_ws(frame))

    return _on_packet


# --------------------------------------------------------------------------- #
# Request body models
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    setup: dict[str, Any]
    track_id: str


class FeedbackRequest(BaseModel):
    """``POST /api/feedback`` body.

    ``driver_style`` 与 ``driver_profile`` 二选一: 若显式提供 ``driver_profile``
    字典则覆盖 ``driver_style`` (UI 使用 ``driver_style`` 即可). 解析逻辑与
    :class:`SearchRequest` 的 /api/search 端点一致 (Iter-05).

    Iter-07: ``session_id`` 启用多轮对话记忆. 当非 None 时, 反馈引擎会惰性
    创建/复用一个进程内 :class:`~f1opt.feedback.conversation.ConversationSession`
    并记录本轮问答, 使得后续带相同 session_id 的请求能解析 "刚才/那个/它"
    等指代词. 默认 None 时行为与 Iter-06 完全一致 (无记忆, 向后兼容).
    """

    frames: list[dict[str, Any]]
    setup: dict[str, Any]
    track_id: str
    question: str | None = None
    driver_profile: dict[str, Any] | None = None
    driver_style: Literal["default", "aggressive", "conservative"] = "default"
    session_id: str | None = None


class SearchRequest(BaseModel):
    """``POST /api/search`` body: 调教搜索入口.

    ``driver_style`` 与 ``driver_profile`` 二选一: 若显式提供 ``driver_profile``
    字典则覆盖 ``driver_style`` (UI 使用 ``driver_style`` 即可).
    """

    track_id: str
    driver_profile: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    iterations: int = 80
    seed: int | None = None
    driver_style: Literal["default", "aggressive", "conservative"] = "default"
    tire_wear_weight: float = 0.0


class WhatIfRequest(BaseModel):
    """``POST /api/whatif`` body: 单字段 what-if 分析 (Iter-119).

    对 ``setup`` 中的 ``field`` 改为 ``new_value``, 返回因果解释 + 圈速 delta +
    推荐联动调整. ``driver_style`` 与 ``driver_profile`` 二选一.
    """

    setup: dict[str, Any]
    track_id: str
    field: str
    new_value: float
    driver_profile: dict[str, Any] | None = None
    driver_style: Literal["default", "aggressive", "conservative"] = "default"


class WhatIfMultiRequest(BaseModel):
    """``POST /api/whatif/multi`` body: 多字段批量 what-if (Iter-119).

    ``changes`` 是 ``{field: new_value}`` 字典, 一次性应用多个 setup 改动,
    返回组合因果解释 + 组合圈速 delta.
    """

    setup: dict[str, Any]
    track_id: str
    changes: dict[str, float]
    driver_profile: dict[str, Any] | None = None
    driver_style: Literal["default", "aggressive", "conservative"] = "default"


class CausalExplainRequest(BaseModel):
    """``POST /api/causal/explain`` body: 纯因果解释 (无圈速预测, Iter-119).

    返回 ``field`` 从 ``current`` 改为 ``proposed`` 的因果链 + 风险评估,
    不调用 surrogate 模型 (轻量, 适合 UI 实时提示).
    """

    field: str
    current: float
    proposed: float
    track_type: str | None = None


class AnalyticsLapRequest(BaseModel):
    """``POST /api/analytics/lap`` body: 单圈完整分析 (Iter-130).

    接受一段连续遥测帧 (通常 = 一圈, 60Hz × ~90s ≈ 5400 帧), 返回:

    - ``analytics``: :class:`TelemetryAnalytics.compute_all` 子分析字典
      (speed/throttle/brake/steering/gforce/ers/drs/tire_load/smoothing/racing_line)
    - ``benchmark``: :class:`PerformanceBenchmark.benchmark` 评分卡 (含 grade)
    - ``anomalies``: :class:`AnomalyDetector.detect` 异常事件列表
    - ``severity_distribution``: ``{"low": N, "medium": N, "high": N}``

    ``track_id`` 用于 PerformanceBenchmark 选参考; 未知回退到 ``"medium"``.
    ``track_length_m`` 可选, 用于 racing_line_deviation 估计 (默认 5000m).
    """

    frames: list[dict[str, Any]]
    track_id: str = "monza"
    track_length_m: float = 5000.0


class AnalyticsAnomaliesRequest(BaseModel):
    """``POST /api/analytics/anomalies`` body: 仅异常检测 (Iter-130).

    轻量级端点 — 只跑 :class:`AnomalyDetector`, 不计算完整 analytics/benchmark.
    适合 UI 实时高亮异常帧 (无需等待完整圈).
    """

    frames: list[dict[str, Any]]


class CompareRequest(BaseModel):
    """``POST /api/feedback/compare`` body: 圈速对比 (Iter-183).

    对比当前圈与参考圈 (队友/最佳圈/目标圈), 返回 sector-by-sector 差异.
    """

    current_lap: dict[str, Any]
    reference_lap: dict[str, Any] | None = None


class TemplateSubmitRequest(BaseModel):
    """``POST /api/feedback/templates`` body: 自定义模板提交 (Iter-183)."""

    id: str
    granularity: str = "overall"
    category: str = "custom"
    text_zh: str
    text_en: str = ""


def create_app(start_listener: bool = True) -> FastAPI:
    """Build the FastAPI application.

    When ``start_listener`` is True (default) the UDP :class:`TelemetryListener`
    is started on lifespan startup using :func:`get_settings` host/port. Tests
    pass ``start_listener=False`` to avoid binding a real UDP port.
    """
    state = _TelemetryState()
    settings = get_settings()

    # Iter-170: rate limiter (slowapi) — per-IP, default 60/min global.
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["60/minute"],
        storage_uri="memory://",
    )
    audit = get_audit_logger()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> None:
        state.loop = asyncio.get_running_loop()
        if start_listener:
            listener = TelemetryListener(
                host=settings.udp_host, port=settings.udp_port
            )
            listener.subscribe(_make_subscriber(state))
            try:
                await listener.start()
                state.listener = listener
            except OSError:
                # Port binding failed — keep the API usable without live telemetry.
                state.listener = None
            except Exception as exc:
                # Windows: 捕获权限/防火墙/WinSock 等系统级错误
                if sys.platform == "win32":
                    import logging
                    logging.getLogger("f1opt.api").warning(
                        "遥测监听器启动失败 (Windows 常见原因: 防火墙/端口占用): %s", exc)
                state.listener = None
        yield
        if state.listener is not None:
            await state.listener.stop()
            state.listener = None

    app = FastAPI(title="F1OPT API", version=__version__, lifespan=lifespan)
    # Wire slowapi rate limiter into the app.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # Iter-204: Enhanced CORS with granular trusted origins, method restrictions,
    # and header allowlisting. Production deployments should override
    # F1OPT_CORS_ORIGINS with a comma-separated list.
    _cors_origins_raw = _os.environ.get("F1OPT_CORS_ORIGINS", "*")
    _cors_origins = [
        o.strip() for o in _cors_origins_raw.split(",") if o.strip()
    ] if _cors_origins_raw != "*" else ["*"]

    _cors_allowed_methods = ["GET", "POST", "OPTIONS", "HEAD"]
    _cors_allowed_headers = [
        "Content-Type", "Authorization", "X-Request-ID",
        "X-API-Key", "Accept", "Origin",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=_cors_allowed_methods,
        allow_headers=_cors_allowed_headers,
        expose_headers=["X-Request-ID", "X-RateLimit-Limit",
                        "X-RateLimit-Remaining", "X-RateLimit-Reset"],
        max_age=600,  # 10 minutes preflight cache
    )

    class _RequestLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Response:
            start = time.perf_counter()
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            audit.log(
                actor=get_remote_address(request),
                action="http_request",
                resource=f"{request.method} {request.url.path}",
                outcome="success" if response.status_code < 400 else "failure",
                ip=get_remote_address(request),
                user_agent=request.headers.get("user-agent"),
                metadata={
                    "status_code": response.status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            return response

    app.add_middleware(_RequestLoggingMiddleware)

    # Iter-205: Structured error response format with error codes.
    # Provides consistent error shapes across all endpoints, making it
    # easier for clients to handle errors programmatically.
    from fastapi.responses import JSONResponse

    @app.exception_handler(HTTPException)
    async def _structured_http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Wrap HTTPException in a structured error envelope (Iter-205)."""
        path = request.url.path
        method = request.method
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "path": path,
                    "method": method,
                },
                "request_id": request.headers.get("X-Request-ID", ""),
            },
        )

    @app.exception_handler(ValidationError)
    async def _structured_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Wrap Pydantic ValidationError in structured envelope (Iter-205)."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(exc),
                    "errors": exc.errors(),
                    "path": request.url.path,
                    "method": request.method,
                },
                "request_id": request.headers.get("X-Request-ID", ""),
            },
        )

    @app.exception_handler(Exception)
    async def _structured_generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all for unhandled exceptions (Iter-205)."""
        detail = "Internal server error"
        if _os.environ.get("F1OPT_DEBUG", "").lower() in ("1", "true", "yes"):
            detail = str(exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": detail,
                    "path": request.url.path,
                    "method": request.method,
                },
                "request_id": request.headers.get("X-Request-ID", ""),
            },
        )

    app.state.telemetry = state
    app.state.audit = audit

    # ----------------------------- REST ---------------------------------- #
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        listening = state.listener is not None and state.listener.is_running
        return {
            "status": "ok",
            "version": __version__,
            "udp_listening": listening,
            "validation_failures": (
                state.listener.validation_failures
                if state.listener is not None
                else 0
            ),
        }

    @app.get("/api/livez")
    async def livez() -> dict[str, Any]:
        """Liveness probe — process is alive (always 200 if serving)."""
        return {"status": "alive"}

    @app.get("/api/readyz")
    async def readyz() -> dict[str, Any]:
        """Readiness probe — can serve requests (listener optional)."""
        # Ready when the surrogate model can be imported; listener binding
        # is best-effort (may be unavailable in CI).
        try:
            from f1opt.model.surrogate import predict_lap_time  # noqa: F401
            model_ready = True
        except Exception:
            model_ready = False
        return {
            "ready": model_ready,
            "model_ready": model_ready,
            "udp_listening": (
                state.listener is not None and state.listener.is_running
            ),
            "tracing_enabled": is_tracing_enabled(),
        }

    @app.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        """Return runtime metrics: listener counters + endpoint latency
        histograms (predict/search/feedback) + process uptime."""
        return state.metrics.snapshot(state.listener)

    @app.get("/api/samples")
    async def list_samples(clean_only: bool = False) -> dict[str, Any]:
        """Return aggregated clean-lap samples (disk + in-memory, deduplicated).

        ``clean_only=true`` filters out laps flagged ``clean=False`` (field-level
        validation failure or flashback taint). Each row matches the Parquet
        schema (session_uid/car_index/lap_number/lap_time_ms/avg_*/clean/...).
        """
        rows = state.lap_agg.all_rows()
        if clean_only:
            rows = [r for r in rows if r.get("clean", True)]
        return {"count": len(rows), "samples": rows}

    @app.get("/api/telemetry/stats")
    async def telemetry_stats() -> dict[str, Any]:
        """Iter-183: Real-time telemetry statistics.

        Returns current telemetry state: last frame values, listener status,
        aggregator row count, and connection manager client count.
        """
        frame = state.aligner.latest_unified_frame()
        latest_frame: dict[str, Any] | None = None
        if frame is not None:
            latest_frame = _frame_to_ws(frame)
        return {
            "listener_running": (
                state.listener is not None and state.listener.is_running
            ),
            "listener_received": (
                state.listener.received if state.listener is not None else 0
            ),
            "aggregator_rows": len(state.lap_agg.rows),
            "ws_clients": len(state.manager._queues),
            "latest_frame": latest_frame,
            "player_setup_cached": state.player_setup_cache is not None,
            "current_track_id": state.current_track_id,
        }

    @app.get("/api/samples/parquet")
    async def download_samples_parquet() -> Response:
        """Download all aggregated lap samples as a Parquet file."""
        data = state.lap_agg.to_parquet_bytes()
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=laps.parquet"},
        )

    @app.get("/api/tracks")
    async def list_tracks() -> dict[str, Any]:
        return {"tracks": [_track_dict(t) for t in ALL_TRACKS]}

    @app.get("/api/tracks/{track_id}")
    async def get_track_endpoint(track_id: str) -> dict[str, Any]:
        track = TRACKS_BY_ID.get(track_id)
        if track is None:
            raise HTTPException(
                status_code=404, detail=f"unknown track_id: {track_id}"
            )
        return _track_dict(track)

    @app.get("/api/setup/default")
    async def default_setup() -> dict[str, Any]:
        return DEFAULT_SETUP.model_dump()

    @app.post("/api/predict")
    @limiter.limit("30/minute")
    async def predict(body: PredictRequest, request: Request) -> dict[str, Any]:
        start = time.perf_counter()
        client_ip = get_remote_address(request)
        try:
            with span("api.predict", track_id=body.track_id):
                # Validate setup (400 on validation error).
                try:
                    car_setup = CarSetup(**body.setup)
                except (ValidationError, ValueError) as exc:
                    audit.log(actor=client_ip, action="predict",
                              resource=f"setup/{body.track_id}",
                              outcome="failure", ip=client_ip,
                              user_agent=request.headers.get("user-agent"),
                              metadata={"error": "invalid_setup"})
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                # Validate track_id (400 on unknown, aliases resolved).
                from f1opt.data.ea_f1_2026_benchmark import canonical_track_id
                if TRACKS_BY_ID.get(canonical_track_id(body.track_id)) is None:
                    raise HTTPException(
                        status_code=400, detail=f"unknown track_id: {body.track_id}"
                    )
                # Lazy-import the surrogate model (503 if absent/broken).
                try:
                    from f1opt.model.surrogate import predict_lap_time

                    try:
                        from f1opt.model.surrogate import MODEL_VERSION as model_version
                    except ImportError:
                        model_version = "unknown"
                    result = predict_lap_time(car_setup, body.track_id)
                except ImportError:
                    audit.log(actor=client_ip, action="predict",
                              resource=f"setup/{body.track_id}",
                              outcome="error", ip=client_ip,
                              user_agent=request.headers.get("user-agent"),
                              metadata={"error": "model_unavailable"})
                    raise HTTPException(
                        status_code=503, detail="model not available"
                    ) from None
                except Exception as exc:  # model not ready / call-shape mismatch
                    audit.log(actor=client_ip, action="predict",
                              resource=f"setup/{body.track_id}",
                              outcome="error", ip=client_ip,
                              user_agent=request.headers.get("user-agent"),
                              metadata={"error": str(exc)[:200]})
                    raise HTTPException(
                        status_code=503, detail="model not available"
                    ) from exc
                if isinstance(result, dict):
                    predicted = float(
                        result.get("predicted_lap_time", result.get("lap_time", 0.0))
                    )
                    mv = str(result.get("model_version", model_version))
                else:
                    predicted = float(result)
                    mv = model_version
                audit.log(actor=client_ip, action="predict",
                          resource=f"setup/{body.track_id}",
                          outcome="success", ip=client_ip,
                          user_agent=request.headers.get("user-agent"),
                          metadata={"lap_time_s": predicted, "model_version": mv})
                return {"predicted_lap_time": predicted, "model_version": mv}
        finally:
            state.metrics.predict.record(time.perf_counter() - start)

    @app.post("/api/feedback")
    @limiter.limit("20/minute")
    async def feedback(body: FeedbackRequest, request: Request) -> Any:
        start = time.perf_counter()
        try:
            with span("api.feedback", track_id=body.track_id,
                      has_question=bool(body.question)):
                try:
                    from f1opt.feedback.engine import generate_feedback_async
                except ImportError:
                    raise HTTPException(
                        status_code=503, detail="feedback engine not available"
                    ) from None
                # Resolve driver profile: explicit dict wins over driver_style keyword
                # (mirrors /api/search). The engine normalises dict/list/DriverProfile.
                driver_profile: Any
                if body.driver_profile is not None:
                    driver_profile = body.driver_profile
                else:
                    try:
                        from f1opt.driver.profile import (
                            AGGRESSIVE_PROFILE,
                            CONSERVATIVE_PROFILE,
                            DEFAULT_PROFILE,
                        )

                        style_map = {
                            "default": DEFAULT_PROFILE,
                            "aggressive": AGGRESSIVE_PROFILE,
                            "conservative": CONSERVATIVE_PROFILE,
                        }
                        driver_profile = style_map[body.driver_style]
                    except ImportError:
                        # driver profile module absent — fall back to None (rule-based
                        # feedback still works without personalisation).
                        driver_profile = None
                try:
                    # Iter-122: use async version to avoid blocking the event loop
                    # during LLM HTTP calls (httpx.AsyncClient instead of httpx.Client).
                    result = await generate_feedback_async(
                        frames=body.frames,
                        setup=body.setup,
                        track_id=body.track_id,
                        question=body.question,
                        driver_profile=driver_profile,
                        session_id=body.session_id,
                    )
                except Exception as exc:  # engine not ready / call-shape mismatch
                    raise HTTPException(
                        status_code=503, detail="feedback engine not available"
                    ) from exc
                return result
        finally:
            state.metrics.feedback.record(time.perf_counter() - start)

    @app.post("/api/feedback/analyze")
    async def feedback_analyze(body: FeedbackRequest) -> Any:
        """Generate feedback with explicit LLM lifecycle: preload → analyze → unload.

        This is the recommended endpoint for post-gameplay feedback analysis.
        It preloads the LLM backend, generates feedback (with LLM enhancement
        if configured), then unloads the LLM to free memory. During gameplay
        the LLM stays unloaded via the standard ``/api/feedback`` endpoint,
        which only uses the rule-based path.
        """
        start = time.perf_counter()
        try:
            from f1opt.feedback.engine import (
                _default_engine,
                generate_feedback_async,
            )
            engine = _default_engine()
            # Step 1: preload LLM.
            preload_result = engine.preload_llm()
            # Step 2: generate feedback (LLM enhancement runs because
            # _llm_loaded is now True).
            driver_profile: Any
            if body.driver_profile is not None:
                driver_profile = body.driver_profile
            else:
                try:
                    from f1opt.driver.profile import (
                        AGGRESSIVE_PROFILE,
                        CONSERVATIVE_PROFILE,
                        DEFAULT_PROFILE,
                    )
                    style_map = {
                        "default": DEFAULT_PROFILE,
                        "aggressive": AGGRESSIVE_PROFILE,
                        "conservative": CONSERVATIVE_PROFILE,
                    }
                    driver_profile = style_map[body.driver_style]
                except ImportError:
                    driver_profile = None
            try:
                result = await generate_feedback_async(
                    frames=body.frames,
                    setup=body.setup,
                    track_id=body.track_id,
                    question=body.question,
                    driver_profile=driver_profile,
                    session_id=body.session_id,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="feedback engine not available"
                ) from exc
            # Step 3: unload LLM to free memory.
            unload_result = engine.unload_llm()
            result["_llm_lifecycle"] = {
                "preload": preload_result,
                "unload": unload_result,
            }
            return result
        finally:
            state.metrics.feedback.record(time.perf_counter() - start)

    # Iter-206: Batch feedback endpoint with concurrency handling.
    class BatchFeedbackRequest(BaseModel):
        """Request body for batch feedback processing (Iter-206)."""
        sessions: list[FeedbackRequest]
        max_concurrency: int = 4

    @app.post("/api/feedback/batch")
    @limiter.limit("10/minute")
    async def feedback_batch(
        body: BatchFeedbackRequest, request: Request
    ) -> dict[str, Any]:
        """Process multiple feedback sessions concurrently (Iter-206).

        Accepts a list of feedback sessions and processes them in parallel
        using an asyncio semaphore for controlled concurrency. Returns
        aggregated results including per-session feedback and overall
        statistics.
        """
        start = time.perf_counter()
        if not body.sessions:
            raise HTTPException(status_code=400, detail="no sessions provided")
        if len(body.sessions) > 50:
            raise HTTPException(
                status_code=400,
                detail="batch size exceeds maximum of 50 sessions",
            )

        try:
            from f1opt.feedback.engine import generate_feedback_async
        except ImportError:
            raise HTTPException(
                status_code=503, detail="feedback engine not available"
            ) from None

        sem = asyncio.Semaphore(max(1, body.max_concurrency))

        async def _process_one(
            idx: int, session: FeedbackRequest
        ) -> dict[str, Any]:
            async with sem:
                try:
                    result = await generate_feedback_async(
                        frames=session.frames,
                        setup=session.setup,
                        track_id=session.track_id,
                        question=session.question,
                        session_id=session.session_id,
                    )
                    return {"index": idx, "status": "success", "result": result}
                except Exception as exc:
                    return {
                        "index": idx,
                        "status": "error",
                        "error": str(exc)[:500],
                    }

        tasks = [
            _process_one(i, s) for i, s in enumerate(body.sessions)
        ]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = len(results) - success_count

        return {
            "total": len(results),
            "success": success_count,
            "errors": error_count,
            "results": results,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    @app.post("/api/llm/preload")
    async def llm_preload() -> dict[str, Any]:
        """Explicitly preload the LLM backend into memory.

        Call this after stopping telemetry / gameplay to prepare the LLM
        for subsequent ``/api/feedback`` calls. The LLM stays loaded until
        ``/api/llm/unload`` is called or the process exits.
        """
        from f1opt.feedback.engine import _default_engine
        engine = _default_engine()
        return engine.preload_llm()

    @app.post("/api/llm/unload")
    async def llm_unload() -> dict[str, Any]:
        """Explicitly unload the LLM backend and free memory.

        After this call, ``/api/feedback`` will return rule-based feedback
        only (no LLM enhancement). Call this before starting a new gameplay
        session to ensure the LLM doesn't consume memory during telemetry
        collection.
        """
        from f1opt.feedback.engine import _default_engine
        engine = _default_engine()
        return engine.unload_llm()

    @app.get("/api/llm/status")
    async def llm_status() -> dict[str, Any]:
        """Iter-183: Return LLM backend status (loaded, backend type, token usage).

        Returns:
            ``{"loaded": bool, "backend": str, "api_key_set": bool, "token_usage": {...}}``
        """
        from f1opt.config import get_settings
        from f1opt.feedback.engine import _default_engine
        settings = get_settings()
        engine = _default_engine()
        return {
            "loaded": engine._llm_loaded,
            "backend": settings.llm_backend,
            "api_key_set": bool(settings.llm_api_key),
            "token_usage": engine.token_usage(),
        }

    @app.post("/api/search")
    @limiter.limit("20/minute")
    async def search(body: SearchRequest, request: Request) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            with span("api.search", track_id=body.track_id,
                      iterations=body.iterations):
                # 1. Validate track_id (404 if missing).
                try:
                    get_track(body.track_id)
                except ValueError:
                    raise HTTPException(
                        status_code=404, detail=f"unknown track_id: {body.track_id}"
                    ) from None
                # 2. Validate baseline if provided (400 on validation error); else default.
                if body.baseline is not None:
                    try:
                        baseline_setup = CarSetup(**body.baseline)
                    except (ValidationError, ValueError) as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                else:
                    baseline_setup = DEFAULT_SETUP
                # 3. Resolve driver profile: explicit dict wins over driver_style keyword.
                driver_profile: Any
                if body.driver_profile is not None:
                    driver_profile = body.driver_profile
                else:
                    try:
                        from f1opt.driver.profile import (
                            AGGRESSIVE_PROFILE,
                            CONSERVATIVE_PROFILE,
                            DEFAULT_PROFILE,
                        )

                        style_map = {
                            "default": DEFAULT_PROFILE,
                            "aggressive": AGGRESSIVE_PROFILE,
                            "conservative": CONSERVATIVE_PROFILE,
                        }
                        driver_profile = style_map[body.driver_style]
                    except ImportError:
                        # driver profile module absent — fall back to None (surrogate
                        # still callable without a profile).
                        driver_profile = None
                # 4. Lazy-import the optimizer (503 if absent/broken).
                try:
                    from f1opt.model.optimizer import search_setup
                except ImportError:
                    raise HTTPException(
                        status_code=503, detail="optimizer not available"
                    ) from None
                try:
                    result = search_setup(
                        body.track_id,
                        driver_profile=driver_profile,
                        baseline=baseline_setup,
                        iterations=body.iterations,
                        seed=body.seed,
                        tire_wear_weight=body.tire_wear_weight,
                    )
                except Exception as exc:  # optimizer not ready / call-shape mismatch
                    raise HTTPException(
                        status_code=503, detail="optimizer not available"
                    ) from exc
                return result.model_dump()
        finally:
            state.metrics.search.record(time.perf_counter() - start)

    # ------------------------------------------------------------------ #
    # Iter-203: 策略规划 API 端点
    # ------------------------------------------------------------------ #
    @app.get("/api/strategy/plan")
    @limiter.limit("30/minute")
    async def strategy_plan(
        request: Request,
        track_id: str = "monza",
        total_laps: int = 50,
        fuel_load_kg: float = 100.0,
        available_compounds: str = "soft,medium,hard",
    ) -> dict[str, Any]:
        """Plan optimal race strategy with degradation analysis (Iter-203).

        Returns optimal pit-stop strategy, fuel-saving analysis, and
        compound degradation crossover data for the given track and
        parameters.
        """
        start = time.perf_counter()
        try:
            try:
                get_track(track_id)
            except ValueError:
                raise HTTPException(
                    status_code=404, detail=f"unknown track_id: {track_id}"
                ) from None

            try:
                from f1opt.model.strategy import RaceStrategyPlanner, StrategyComparator
            except ImportError:
                raise HTTPException(
                    status_code=503, detail="strategy planner not available"
                ) from None

            compounds = [c.strip() for c in available_compounds.split(",") if c.strip()]

            planner = RaceStrategyPlanner(
                track_id=track_id,
                total_laps=total_laps,
                fuel_load_kg=fuel_load_kg,
            )

            optimal = planner.optimal_strategy(compounds)
            fuel = planner.fuel_saving_analysis()
            crossover = planner.degradation_crossover(
                compound_a=compounds[0] if len(compounds) > 0 else "soft",
                compound_b=compounds[1] if len(compounds) > 1 else "medium",
            )

            return {
                "track_id": track_id,
                "total_laps": total_laps,
                "fuel_load_kg": fuel_load_kg,
                "optimal_strategy": optimal,
                "fuel_analysis": fuel,
                "degradation_crossover": {
                    "crossover_lap": crossover["crossover_lap"],
                    "recommendation": crossover["recommendation"],
                },
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        finally:
            state.metrics.search.record(time.perf_counter() - start)

    @app.get("/api/strategy/weather-impact")
    @limiter.limit("30/minute")
    async def strategy_weather_impact(
        request: Request,
        track_id: str = "monza",
        total_laps: int = 50,
        track_wetness: float = 0.0,
        rain_intensity: float = 0.0,
        track_temp_c: float = 35.0,
        wind_speed_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Weather-aware strategy adjustments (Iter-203).

        Evaluates how current weather conditions affect optimal race
        strategy, including compound shifts, pit stop recommendations,
        and pit loss modifications.
        """
        start = time.perf_counter()
        try:
            try:
                get_track(track_id)
            except ValueError:
                raise HTTPException(
                    status_code=404, detail=f"unknown track_id: {track_id}"
                ) from None

            try:
                from f1opt.model.strategy import RaceStrategyPlanner
            except ImportError:
                raise HTTPException(
                    status_code=503, detail="strategy planner not available"
                ) from None

            planner = RaceStrategyPlanner(
                track_id=track_id,
                total_laps=total_laps,
                fuel_load_kg=100.0,
            )

            weather_impact = planner.weather_impact_on_strategy(
                track_wetness=track_wetness,
                rain_intensity=rain_intensity,
                track_temp_c=track_temp_c,
                wind_speed_ms=wind_speed_ms,
            )

            return {
                "track_id": track_id,
                "total_laps": total_laps,
                "weather_impact": weather_impact,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        finally:
            state.metrics.search.record(time.perf_counter() - start)

    # ------------------------------------------------------------------ #
    # Iter-119: causal / WhatIf API 暴露
    # ------------------------------------------------------------------ #
    def _resolve_driver_profile(
        driver_profile: dict[str, Any] | None,
        driver_style: str,
    ) -> Any:
        """Resolve driver profile: explicit dict wins over driver_style keyword."""
        if driver_profile is not None:
            return driver_profile
        try:
            from f1opt.driver.profile import (
                AGGRESSIVE_PROFILE,
                CONSERVATIVE_PROFILE,
                DEFAULT_PROFILE,
            )

            style_map = {
                "default": DEFAULT_PROFILE,
                "aggressive": AGGRESSIVE_PROFILE,
                "conservative": CONSERVATIVE_PROFILE,
            }
            return style_map[driver_style]
        except ImportError:
            return None

    @app.post("/api/whatif")
    @limiter.limit("30/minute")
    async def whatif(body: WhatIfRequest, request: Request) -> dict[str, Any]:
        """Iter-119: 单字段 what-if 分析.

        对 ``setup`` 中的 ``field`` 改为 ``new_value``, 返回:
        - causal: 因果链 + 风险评估 (来自 CausalExplanationEngine)
        - lap_time_delta: 圈速变化 (秒, modified - baseline, 来自 surrogate)
        - confidence: 置信度 [0,1] (变化越大越低)
        - recommended_accompanying: 推荐联动调整
        """
        start = time.perf_counter()
        try:
            # 1. Validate track_id (404 if missing).
            try:
                get_track(body.track_id)
            except ValueError:
                raise HTTPException(
                    status_code=404, detail=f"unknown track_id: {body.track_id}"
                ) from None
            # 2. Validate setup (400 on validation error).
            try:
                car_setup = CarSetup(**body.setup)
            except (ValidationError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # 3. Lazy-import WhatIfAnalyzer (503 if absent).
            try:
                from f1opt.feedback.causal import WhatIfAnalyzer
            except ImportError:
                raise HTTPException(
                    status_code=503, detail="causal analyzer not available"
                ) from None
            # 4. Validate field exists in SETUP_FIELDS.
            from f1opt.data.setup_schema import SETUP_FIELDS
            if body.field not in SETUP_FIELDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown field: {body.field}. "
                           f"valid: {sorted(SETUP_FIELDS.keys())}",
                )
            # 5. Resolve driver profile.
            driver_profile = _resolve_driver_profile(
                body.driver_profile, body.driver_style,
            )
            # 6. Run analysis.
            try:
                analyzer = WhatIfAnalyzer(
                    car_setup, body.track_id, driver_profile,
                )
                result = analyzer.analyze_change(body.field, body.new_value)
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"what-if analysis failed: {exc}",
                ) from exc
            return result
        finally:
            state.metrics.predict.record(time.perf_counter() - start)

    @app.post("/api/whatif/multi")
    @limiter.limit("15/minute")
    async def whatif_multi(body: WhatIfMultiRequest, request: Request) -> dict[str, Any]:
        """Iter-119: 多字段批量 what-if 分析.

        一次性应用多个 setup 改动 (``changes`` dict), 返回组合因果解释 + 组合圈速 delta.
        """
        start = time.perf_counter()
        try:
            # 1. Validate track_id.
            try:
                get_track(body.track_id)
            except ValueError:
                raise HTTPException(
                    status_code=404, detail=f"unknown track_id: {body.track_id}"
                ) from None
            # 2. Validate setup.
            try:
                car_setup = CarSetup(**body.setup)
            except (ValidationError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # 3. Lazy-import.
            try:
                from f1opt.feedback.causal import WhatIfAnalyzer
            except ImportError:
                raise HTTPException(
                    status_code=503, detail="causal analyzer not available"
                ) from None
            # 4. Validate all fields in changes exist.
            from f1opt.data.setup_schema import SETUP_FIELDS
            invalid = [f for f in body.changes if f not in SETUP_FIELDS]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown fields: {invalid}. "
                           f"valid: {sorted(SETUP_FIELDS.keys())}",
                )
            if not body.changes:
                raise HTTPException(
                    status_code=400, detail="changes must not be empty",
                )
            # 5. Resolve driver profile.
            driver_profile = _resolve_driver_profile(
                body.driver_profile, body.driver_style,
            )
            # 6. Run multi-change analysis.
            try:
                analyzer = WhatIfAnalyzer(
                    car_setup, body.track_id, driver_profile,
                )
                result = analyzer.analyze_multi_change(body.changes)
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"what-if multi analysis failed: {exc}",
                ) from exc
            return result
        finally:
            state.metrics.predict.record(time.perf_counter() - start)

    @app.post("/api/causal/explain")
    async def causal_explain(body: CausalExplainRequest) -> dict[str, Any]:
        """Iter-119: 纯因果解释 (无圈速预测).

        返回 ``field`` 从 ``current`` 改为 ``proposed`` 的因果链 + 风险评估,
        不调用 surrogate 模型 (轻量, 适合 UI 实时提示).
        """
        # 1. Validate field exists.
        from f1opt.data.setup_schema import SETUP_FIELDS
        if body.field not in SETUP_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown field: {body.field}. "
                       f"valid: {sorted(SETUP_FIELDS.keys())}",
            )
        # 2. Lazy-import CausalExplanationEngine.
        try:
            from f1opt.feedback.causal import CausalExplanationEngine
        except ImportError:
            raise HTTPException(
                status_code=503, detail="causal engine not available"
            ) from None
        try:
            engine = CausalExplanationEngine(SETUP_FIELDS, track=None)
            result = engine.explain(
                body.field, body.current, body.proposed, {}, body.track_type,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"causal explain failed: {exc}",
            ) from exc
        return result

    @app.get("/api/causal/fields")
    async def causal_fields() -> dict[str, Any]:
        """Iter-119: 列出所有可解释的 setup 字段及其 spec.

        返回 ``{field: {min, max, step, kind, description}}`` 字典, 供 UI 渲染滑块和提示.
        """
        from f1opt.data.setup_schema import SETUP_FIELDS
        fields: dict[str, dict[str, Any]] = {}
        for name, spec in SETUP_FIELDS.items():
            fields[name] = {
                "min": spec.min,
                "max": spec.max,
                "step": spec.step,
                "kind": spec.kind,
                "unit": spec.unit,
                "description": spec.description,
            }
        return {"fields": fields, "count": len(fields)}

    # ------------------------------------------------------------------ #
    # Iter-183: Session management endpoint
    # ------------------------------------------------------------------ #
    @app.get("/api/session/{session_id}")
    async def get_session_info(session_id: str) -> dict[str, Any]:
        """Iter-183: Get session info including conversation history, lap trend,
        and setup change history.

        Returns:
            ``{"session_id": "...", "turn_count": N, "recent_history": [...],
            "focus_summary": "...", "lap_trend": {...}, "setup_changes": [...]}``
        """
        from f1opt.feedback.conversation import get_session
        session = get_session(session_id)
        return {
            "session_id": session_id,
            "turn_count": len(session.history),
            "recent_history": session.recent(10),
            "focus_summary": session.summarize_focus(),
            "lap_trend": session.get_lap_trend(),
            "setup_changes": session.get_setup_history(),
        }

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        """Iter-183: Delete a conversation session and reset its memory."""
        from f1opt.feedback.conversation import _sessions
        if session_id in _sessions:
            del _sessions[session_id]
            return {"session_id": session_id, "status": "deleted"}
        return {"session_id": session_id, "status": "not_found"}

    # ------------------------------------------------------------------ #
    # Iter-183: Lap comparison endpoint
    # ------------------------------------------------------------------ #
    @app.post("/api/feedback/compare")
    async def compare_laps(body: CompareRequest) -> dict[str, Any]:
        """Iter-183: Compare current lap against a reference lap (teammate/best/target).

        Returns sector-by-sector deltas, speed comparison, and a Chinese verdict.
        When ``reference_lap`` is None, returns a neutral comparison.
        """
        try:
            from f1opt.feedback.comparison import LapComparator
        except ImportError:
            raise HTTPException(
                status_code=503, detail="comparison module not available"
            ) from None
        comparator = LapComparator(reference_lap=body.reference_lap)
        result = comparator.compare(body.current_lap)
        return result

    # ------------------------------------------------------------------ #
    # Iter-183: Feedback history endpoint
    # ------------------------------------------------------------------ #
    @app.get("/api/feedback/history")
    async def feedback_history(
        session_id: str = "default",
        n: int = 20,
    ) -> dict[str, Any]:
        """Iter-183: Return recent feedback conversation history for a session.

        Args:
            session_id: Session identifier (default ``"default"``).
            n: Max number of recent turns to return (clamped to [1, 100]).

        Returns:
            ``{"session_id": "...", "turns": [...], "total_turns": N}``
            where each turn is ``{"role": "user"/"assistant", "content": "..."}``.
        """
        from f1opt.feedback.conversation import get_session
        n_clamped = max(1, min(100, n))
        session = get_session(session_id)
        recent = session.recent(n_clamped)
        return {
            "session_id": session_id,
            "turns": recent,
            "total_turns": len(session.history),
        }

    # ------------------------------------------------------------------ #
    # Iter-183: 车手反馈模板 API
    # ------------------------------------------------------------------ #
    @app.get("/api/templates")
    async def list_templates(
        group: str = "all",
        lang: str = "zh",
    ) -> dict[str, Any]:
        """Iter-183: 获取车手反馈模板列表.

        Args:
            group: 模板分组 (``corner`` / ``sector`` / ``overall`` / ``all``).
            lang: 语言 (``zh`` / ``en``).

        Returns:
            ``{"templates": [...], "group": "...", "count": N}``.
        """
        from f1opt.feedback.prompts import (
            DRIVER_FEEDBACK_TEMPLATES,
            FEEDBACK_TEMPLATE_GROUPS,
            render_feedback_template,
        )
        if group not in FEEDBACK_TEMPLATE_GROUPS:
            group = "all"
        template_ids = FEEDBACK_TEMPLATE_GROUPS[group]
        templates = []
        for tid in template_ids:
            t = DRIVER_FEEDBACK_TEMPLATES.get(tid)
            if t is None:
                continue
            templates.append({
                "id": t["id"],
                "granularity": t["granularity"],
                "category": t["category"],
                "text": render_feedback_template(tid, lang),
            })
        return {
            "templates": templates,
            "group": group,
            "count": len(templates),
        }

    @app.get("/api/templates/{template_id}")
    async def get_template(
        template_id: str,
        lang: str = "zh",
    ) -> dict[str, Any]:
        """Iter-183: 获取单个车手反馈模板.

        Args:
            template_id: 模板 ID (如 ``corner_understeer``).
            lang: 语言 (``zh`` / ``en``).

        Returns:
            ``{"id": "...", "granularity": "...", "category": "...", "text": "..."}``.
        """
        from f1opt.feedback.prompts import (
            DRIVER_FEEDBACK_TEMPLATES,
            render_feedback_template,
        )
        t = DRIVER_FEEDBACK_TEMPLATES.get(template_id)
        if t is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown template_id: {template_id}. "
                       f"Available: {', '.join(sorted(DRIVER_FEEDBACK_TEMPLATES.keys()))}"
            )
        return {
            "id": t["id"],
            "granularity": t["granularity"],
            "category": t["category"],
            "text": render_feedback_template(template_id, lang),
        }

    @app.get("/api/templates/categories")
    async def template_categories() -> dict[str, Any]:
        """Iter-183: List all template categories with counts.

        Returns:
            ``{"categories": [{"category": "...", "count": N, "templates": [...]}, ...]}``
        """
        from f1opt.feedback.prompts import (
            DRIVER_FEEDBACK_TEMPLATES,
            FEEDBACK_TEMPLATE_GROUPS,
        )
        categories: list[dict[str, Any]] = []
        for category, template_ids in FEEDBACK_TEMPLATE_GROUPS.items():
            entries = []
            for tid in template_ids:
                t = DRIVER_FEEDBACK_TEMPLATES.get(tid)
                if t:
                    entries.append({
                        "id": t["id"],
                        "granularity": t["granularity"],
                    })
            categories.append({
                "category": category,
                "count": len(entries),
                "templates": entries,
            })
        return {"categories": categories, "total_categories": len(categories)}

    @app.post("/api/feedback/templates")
    async def submit_template(body: TemplateSubmitRequest) -> dict[str, Any]:
        """Iter-183: Submit a custom feedback template.

        Registers a user-defined template that can be used for driver feedback
        generation. The template is stored in-memory for the process lifetime.
        """
        from f1opt.feedback.prompts import DRIVER_FEEDBACK_TEMPLATES, FEEDBACK_TEMPLATE_GROUPS

        if body.id in DRIVER_FEEDBACK_TEMPLATES:
            return {
                "status": "exists",
                "id": body.id,
                "message": f"Template '{body.id}' already exists and was not overwritten",
            }
        DRIVER_FEEDBACK_TEMPLATES[body.id] = {
            "id": body.id,
            "granularity": body.granularity,
            "category": body.category,
            "zh": body.text_zh,
            "en": body.text_en or body.text_zh,
        }
        if body.category not in FEEDBACK_TEMPLATE_GROUPS:
            FEEDBACK_TEMPLATE_GROUPS[body.category] = [body.id]
        else:
            FEEDBACK_TEMPLATE_GROUPS[body.category].append(body.id)
        return {
            "status": "created",
            "id": body.id,
            "granularity": body.granularity,
            "category": body.category,
        }

    # ------------------------------------------------------------------ #
    # Iter-130: Analytics API — 单圈完整分析 + 仅异常检测
    # ------------------------------------------------------------------ #
    @app.post("/api/analytics/lap")
    async def analytics_lap(body: AnalyticsLapRequest) -> dict[str, Any]:
        """Iter-130: 单圈完整分析.

        接受一段遥测帧列表 (通常 = 一圈), 返回:
        - ``analytics``: TelemetryAnalytics.compute_all() 子分析字典
        - ``benchmark``: PerformanceBenchmark.benchmark() 评分卡
        - ``anomalies``: AnomalyDetector.detect() 异常事件列表
        - ``severity_distribution``: ``{"low": N, "medium": N, "high": N}``
        - ``frame_count``: 输入帧数 (供客户端校验)
        - ``track_id``: 回显 track_id

        ``track_id`` 用于 PerformanceBenchmark 选参考; 未知回退到 ``"medium"``.
        空帧列表返回 200 + 全部为空的分析结构 (优雅降级).
        """
        try:
            from f1opt.telemetry.analytics import (
                AnomalyDetector,
                PerformanceBenchmark,
                TelemetryAnalytics,
            )
        except ImportError:
            raise HTTPException(
                status_code=503, detail="analytics module not available"
            ) from None
        # Defensive: a non-list frames payload would have failed pydantic
        # validation, but guard against None entries inside the list.
        frames = [f for f in body.frames if isinstance(f, dict)]
        ta = TelemetryAnalytics(frames, track_length_m=body.track_length_m)
        analytics = ta.compute_all()
        benchmark = PerformanceBenchmark(body.track_id).benchmark(analytics)
        detector = AnomalyDetector()
        anomalies = detector.detect(frames)
        severity = detector.severity_distribution(anomalies)
        return {
            "analytics": analytics,
            "benchmark": benchmark,
            "anomalies": anomalies,
            "severity_distribution": severity,
            "frame_count": len(frames),
            "track_id": body.track_id,
        }

    @app.post("/api/analytics/anomalies")
    async def analytics_anomalies(body: AnalyticsAnomaliesRequest) -> dict[str, Any]:
        """Iter-130: 仅异常检测 (轻量级, 不计算完整 analytics).

        返回:
        - ``anomalies``: AnomalyDetector.detect() 异常事件列表
        - ``severity_distribution``: ``{"low": N, "medium": N, "high": N}``
        - ``frame_count``: 输入帧数
        - ``anomaly_count``: 异常事件总数

        适合 UI 实时高亮异常帧 (无需等待完整圈或计算 benchmark).
        """
        try:
            from f1opt.telemetry.analytics import AnomalyDetector
        except ImportError:
            raise HTTPException(
                status_code=503, detail="analytics module not available"
            ) from None
        frames = [f for f in body.frames if isinstance(f, dict)]
        detector = AnomalyDetector()
        anomalies = detector.detect(frames)
        severity = detector.severity_distribution(anomalies)
        return {
            "anomalies": anomalies,
            "severity_distribution": severity,
            "frame_count": len(frames),
            "anomaly_count": len(anomalies),
        }

    @app.get("/api/feedback-loop")
    async def feedback_loop_status() -> dict[str, Any]:
        """Iter-117: 反馈闭环状态 — ObservationBuffer 统计 + player setup 缓存.

        返回:
            - observations: 缓冲中总观测数
            - tracks: 有观测的赛道列表 + 每赛道观测数
            - player_setup_cached: 是否已缓存 player car setup
            - current_track_id: 当前赛道 (session metadata 设置)
            - latest_residual: 最近一条观测的残差 (observed - predicted, 正=DNN 低估)
        """
        obs_buffer = state.obs_buffer
        tracks: dict[str, int] = {}
        for obs in obs_buffer._observations:
            tracks[obs.track_id] = tracks.get(obs.track_id, 0) + 1
        latest_residual: float | None = None
        if len(obs_buffer._observations) > 0:
            latest_residual = float(obs_buffer._observations[-1].residual)
        return {
            "observations": len(obs_buffer),
            "tracks": tracks,
            "player_setup_cached": state.player_setup_cache is not None,
            "current_track_id": state.current_track_id,
            "latest_residual_s": latest_residual,
        }

    @app.post("/api/feedback-loop/track")
    async def set_feedback_track(body: dict[str, Any]) -> dict[str, Any]:
        """Iter-117: 设置当前赛道 ID (session metadata 回退用).

        body: {"track_id": "monza"}
        """
        tid = body.get("track_id")
        if not isinstance(tid, str):
            raise HTTPException(status_code=400, detail="track_id required")
        try:
            get_track(tid)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"unknown track_id: {tid}") from None
        state.current_track_id = tid
        return {"track_id": tid, "status": "set"}

    @app.get("/api/iterations")
    async def list_iterations() -> dict[str, Any]:
        """List primary iteration records (``iter-NN.md``) sorted ascending.

        Returns ``{"iterations": [...]}``; an empty list (200) is returned when
        the iterations directory is absent so the UI degrades gracefully.
        """
        out: list[dict[str, Any]] = []
        if _ITERATIONS_DIR.is_dir():
            files: list[tuple[int, str]] = []
            for p in _ITERATIONS_DIR.iterdir():
                m = _ITER_FILE_RE.match(p.name)
                if m and p.is_file():
                    files.append((int(m.group(1)), p.name))
            files.sort(key=lambda t: t[0])
            for _, fname in files:
                path = _ITERATIONS_DIR / fname
                try:
                    text = path.read_text(encoding="utf-8")
                    mtime = datetime.fromtimestamp(
                        path.stat().st_mtime, tz=UTC
                    ).isoformat()
                except OSError:
                    continue
                title, summary = _iter_meta(text)
                out.append(
                    {
                        "iter": fname[:-3],  # strip ".md"
                        "title": title,
                        "file": fname,
                        "mtime": mtime,
                        "summary_preview": summary,
                    }
                )
        return {"iterations": out}

    @app.get("/api/iterations/{iter}")
    async def get_iteration(iter: str) -> dict[str, Any]:
        """Return the full markdown content of a single iteration record.

        ``iter`` must match ``^iter-\\d+$`` (400 otherwise). A missing file
        yields 404 ``{"detail": "iteration not found"}``.
        """
        if not _ITER_ID_RE.match(iter):
            raise HTTPException(status_code=400, detail="invalid iteration id")
        path = _ITERATIONS_DIR / f"{iter}.md"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="iteration not found")
        content = path.read_text(encoding="utf-8")
        return {"iter": iter, "content": content}

    @app.get("/api/audit")
    async def audit_tail(n: int = 100) -> dict[str, Any]:
        """Iter-170: return the last ``n`` audit log records (most-recent last).

        ``n`` is clamped to ``[0, 1000]``. Returns an empty list when the
        audit log file does not exist yet (no audited operations have
        occurred).
        """
        n_clamped = max(0, min(1000, n))
        records = audit.tail(n_clamped)
        return {
            "count": len(records),
            "n_requested": n_clamped,
            "records": records,
            "path": str(audit.path),
            "total_written": audit.count,
        }

    @app.get("/api/feedback/stream")
    async def feedback_stream(
        setup_json: str = "{}",
        track_id: str = "monza",
        question: str | None = None,
    ) -> StreamingResponse:
        """Iter-183: SSE (Server-Sent Events) for feedback streaming.

        Generates feedback incrementally and streams it as SSE events.
        Each event is ``data: {"type": "chunk", "text": "..."}`` followed
        by a final ``data: {"type": "done", "feedback": {...}}``.
        """
        import json as _json

        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback import generate_feedback_stream

        try:
            setup = _json.loads(setup_json)
        except Exception:
            setup = DEFAULT_SETUP.model_dump()

        async def event_stream():
            for event in generate_feedback_stream(
                [], setup, track_id, question=question
            ):
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --------------------------- WebSocket ------------------------------- #
    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = state.manager.register()
        send_lock = asyncio.Lock()

        async def reader() -> None:
            try:
                while True:
                    msg = await websocket.receive_json()
                    if isinstance(msg, dict) and msg.get("cmd") == "ping":
                        async with send_lock:
                            await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                return

        async def writer() -> None:
            try:
                while True:
                    msg = await queue.get()
                    async with send_lock:
                        await websocket.send_json(msg)
            except WebSocketDisconnect:
                return

        reader_task = asyncio.create_task(reader(), name="ws-reader")
        writer_task = asyncio.create_task(writer(), name="ws-writer")
        try:
            await asyncio.wait(
                {reader_task, writer_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            state.manager.unregister(queue)
            for task in (reader_task, writer_task):
                if not task.done():
                    task.cancel()

    # ----------------------- Static UI (optional) ------------------------ #
    # Mount LAST so it never shadows /api or /ws. Skipped if the dir is absent
    # so the API still boots before the UI task ships its bundle.
    static_dir = Path(__file__).resolve().parent.parent / "ui" / "static"
    if static_dir.is_dir():
        try:
            app.mount(
                "/",
                StaticFiles(directory=str(static_dir), html=True),
                name="static",
            )
        except Exception:  # static mount is best-effort
            pass

    return app


def push_frame(app: FastAPI, frame: dict[str, Any]) -> None:
    """Broadcast a synthetic frame to all connected WS clients.

    Test/debug helper. If ``frame`` already carries a ``type`` it is sent as-is;
    otherwise it is wrapped as ``{"type":"frame", **frame}``. The broadcast is
    scheduled on the app's event loop via ``call_soon_threadsafe`` so it is safe
    to call from any thread (e.g. a sync test driving the WS via TestClient).
    """
    state: _TelemetryState = app.state.telemetry
    msg = frame if "type" in frame else {"type": "frame", **frame}
    loop = state.loop
    if loop is not None and loop.is_running():
        try:
            if asyncio.get_running_loop() is loop:
                state.manager.broadcast(msg)
                return
        except RuntimeError:
            pass
        loop.call_soon_threadsafe(state.manager.broadcast, msg)
    else:
        state.manager.broadcast(msg)


app = create_app()


__all__ = ["app", "create_app", "push_frame"]
