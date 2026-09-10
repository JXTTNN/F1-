"""REST 端点 —— 统一前缀 /api/v1，响应统一 {code, message, data} 信封。

对齐 design.md 2.8.2 REST 端点清单与 spec FR-API-* 需求。

端点总览：
    | 方法 | 路径                          | 说明 |
    |------|-------------------------------|------|
    | GET  | /api/v1/health                | 健康检查 + 遥测连接状态 |
    | GET  | /api/v1/tracks                | 24 条赛道列表 |
    | GET  | /api/v1/tracks/{track_id}     | 单赛道 + 弯道锚点 |
    | POST | /api/v1/tracks/current        | 手动选赛道 |
    | POST | /api/v1/setup/import          | 一键导入 Car Setups 包 |
    | GET  | /api/v1/setup/current         | 读取导入的调教快照 |
    | POST | /api/v1/feedback              | 提交单条反馈 |
    | GET  | /api/v1/feedback              | 查询已提交反馈 |
    | POST | /api/v1/suggest               | 触发建议生成 |
    | GET  | /api/v1/suggest/latest        | 读取最新建议报告 |
    | GET  | /api/v1/iteration/history     | 历史反馈/建议对比 |

关键约定：
    - 所有入参出参用 pydantic BaseModel 强类型校验；
    - 响应统一 ``{code, message, data}`` 信封；
    - POST /suggest：先校验有反馈 → Dx → SetupDelta → 报告 → 落库 → WS 推送。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from setup_tuner.domain.setup import CarSetup
from setup_tuner.domain.symptoms import (
    DEFAULT_INTENSITY,
    INTENSITY_MAX,
    INTENSITY_MIN,
    Symptom,
)
from setup_tuner.domain.track import (
    get_all_tracks,
    get_track_by_id,
    get_track_by_udp_id,
)
from setup_tuner.engine.engine import generate_suggestion
from setup_tuner.report.builder import (
    build_report,
    extract_setup_from_packet5,
    extract_telemetry_summary,
    feedbacks_to_symptoms,
)

from .envelope import fail, ok

logger = logging.getLogger(__name__)

# REST 路由（统一前缀 /api/v1）
router = APIRouter(prefix="/api/v1", tags=["f1opt"])


# =========================================================================== #
# Pydantic 入参/出参模型（强类型校验，禁止 Any / 字符串 Map 传参）
# =========================================================================== #
class TrackRef(BaseModel):
    """赛道精简视图（列表项）。"""

    track_id: str
    official_name: str
    circuit_name: str
    city: str
    country: str
    track_type: str
    length_m: float
    corners: int
    udp_track_id: int
    svg_path: str


class CornerView(BaseModel):
    """弯道视图（含 SVG 热区锚点）。"""

    corner_number: int
    name: str
    corner_type: str
    speed_kmh: float
    anchor_x: float
    anchor_y: float


class TrackDetail(BaseModel):
    """赛道详情视图（含弯道锚点列表）。"""

    track: TrackRef
    corners: list[CornerView]


class SelectTrackRequest(BaseModel):
    """手动选赛道请求。"""

    track_id: str = Field(..., description="赛道标识，如 'suzuka'")


class SelectTrackResponse(BaseModel):
    """手动选赛道响应。"""

    current_track_id: str
    source: str = Field(..., description="来源：manual | telemetry")


class SetupSnapshot(BaseModel):
    """调教快照视图（23 参数 + 元数据）。"""

    setup_id: int | None = None
    track_id: str
    imported_at: str | None = None
    params: dict[str, float] = Field(..., description="23 项参数字典")


class FeedbackRequest(BaseModel):
    """提交单条反馈请求。"""

    track_id: str
    corner_number: int | None = Field(
        default=None, ge=1, le=50, description="弯道编号（1-based）；None=全局症状",
    )
    symptom: str = Field(..., description="12 症状标识之一")
    strength: int = Field(
        default=DEFAULT_INTENSITY,
        ge=INTENSITY_MIN,
        le=INTENSITY_MAX,
        description="强度 0-5，默认 3",
    )


class FeedbackRecord(BaseModel):
    """反馈记录视图。"""

    id: int
    track_id: str
    corner_number: int | None = None
    symptom: str
    category: str
    strength: int
    setup_id: int | None = None
    created_at: str | None = None


class SuggestRequest(BaseModel):
    """触发建议生成请求。"""

    track_id: str = Field(..., description="赛道标识")


class SuggestionView(BaseModel):
    """建议报告视图。"""

    suggestion_id: int | None = None
    track_id: str
    created_at: str | None = None
    report: dict[str, Any] = Field(..., description="报告 JSON（design 2.7.7 格式）")


class IterationView(BaseModel):
    """迭代记录视图。"""

    id: int
    track_id: str
    round_no: int
    before_setup_id: int | None = None
    after_setup_id: int | None = None
    suggestion_id: int | None = None
    created_at: str


class HealthView(BaseModel):
    """健康检查视图。"""

    status: str = Field(..., description="ok | degraded")
    telemetry_connected: bool
    udp_host: str
    udp_port: int
    current_track_id: str | None = None


# =========================================================================== #
# 辅助：从 app.state 获取全局服务实例
# =========================================================================== #
def _get_services(request: Request) -> dict[str, Any]:
    """从 app.state 获取全局服务实例（Store/FeedbackService/IterationService/...）。

    在 create_app 中初始化并挂载到 app.state。
    """
    state = request.app.state
    return {
        "store": getattr(state, "store", None),
        "feedback_service": getattr(state, "feedback_service", None),
        "iteration_service": getattr(state, "iteration_service", None),
        "telemetry_listener": getattr(state, "telemetry_listener", None),
        "telemetry_stream": getattr(state, "telemetry_stream", None),
        "current_track_id": getattr(state, "current_track_id", None),
        "current_track_source": getattr(state, "current_track_source", "manual"),
    }


def _set_current_track(request: Request, track_id: str, source: str) -> None:
    """设置当前赛道（存入 app.state）。"""
    request.app.state.current_track_id = track_id
    request.app.state.current_track_source = source


def _track_to_ref(track: Any) -> TrackRef:
    """将 domain.Track 转为 TrackRef 视图。"""
    return TrackRef(
        track_id=track.track_id,
        official_name=track.official_name,
        circuit_name=track.circuit_name,
        city=track.city,
        country=track.country,
        track_type=track.track_type,
        length_m=track.length_m,
        corners=len(track.corners),
        udp_track_id=track.udp_track_id,
        svg_path=track.svg_path,
    )


def _corner_to_view(corner: Any) -> CornerView:
    """将 domain.Corner 转为 CornerView 视图。"""
    return CornerView(
        corner_number=corner.number,
        name=corner.name,
        corner_type=corner.corner_type,
        speed_kmh=corner.speed_kmh,
        anchor_x=corner.anchor.anchor_x,
        anchor_y=corner.anchor.anchor_y,
    )


# =========================================================================== #
# 端点实现
# =========================================================================== #
@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """健康检查 + 遥测连接状态。

    返回 ``{status, telemetry_connected, udp_host, udp_port, current_track_id}``。
    """
    svc = _get_services(request)
    listener = svc["telemetry_listener"]
    config = getattr(request.app.state, "config", None)
    udp_host = getattr(config, "udp_host", "127.0.0.1") if config else "127.0.0.1"
    udp_port = getattr(config, "udp_port", 20777) if config else 20777

    telemetry_connected = listener.is_running if listener else False
    status = "ok" if telemetry_connected else "degraded"

    data = HealthView(
        status=status,
        telemetry_connected=telemetry_connected,
        udp_host=udp_host,
        udp_port=udp_port,
        current_track_id=svc["current_track_id"],
    )
    return ok(data=data.model_dump())


@router.get("/tracks")
async def list_tracks() -> dict[str, Any]:
    """24 条赛道列表。"""
    tracks = get_all_tracks()
    data = [_track_to_ref(t).model_dump() for t in tracks]
    return ok(data=data, message=f"共 {len(data)} 条赛道")


@router.get("/tracks/{track_id}")
async def get_track(track_id: str) -> dict[str, Any]:
    """单赛道 + 弯道锚点。"""
    track = get_track_by_id(track_id)
    if track is None:
        raise fail(
            message=f"未知赛道标识：{track_id}",
            code=4040,
            http_status=404,
        )
    data = TrackDetail(
        track=_track_to_ref(track),
        corners=[_corner_to_view(c) for c in track.corners],
    )
    return ok(data=data.model_dump())


@router.post("/tracks/current")
async def select_current_track(
    body: SelectTrackRequest, request: Request,
) -> dict[str, Any]:
    """手动选赛道（可被遥测覆盖）。"""
    track = get_track_by_id(body.track_id)
    if track is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )
    _set_current_track(request, body.track_id, "manual")
    data = SelectTrackResponse(
        current_track_id=body.track_id, source="manual",
    )
    return ok(data=data.model_dump(), message=f"已选择赛道：{track.official_name}")


@router.post("/setup/import")
async def import_setup(request: Request) -> dict[str, Any]:
    """一键导入 Car Setups 包（从遥测帧缓存读取）。

    前置：已收到 Car Setups 包（packet_id=5）；否则返回 409 提示「先连接遥测」。
    """
    svc = _get_services(request)
    stream = svc["telemetry_stream"]
    store = svc["store"]

    if stream is None or store is None:
        raise fail(
            message="服务未初始化（Store 或 TelemetryStream 缺失）",
            code=5001,
            http_status=500,
        )

    # 从遥测帧缓存读取 Packet 5（CarSetups）
    packet5 = stream.get_latest(5) if stream else None
    if packet5 is None:
        raise fail(
            message="未收到 Car Setups 包，请先连接遥测并进入车库",
            code=4091,
            http_status=409,
        )

    # 提取 23 参数快照
    params = extract_setup_from_packet5(packet5)

    # 确定赛道：优先用当前选定赛道，其次用遥测 Session 包的 m_trackId
    track_id = svc["current_track_id"]
    if track_id is None:
        session = stream.get_latest(1) if stream else None
        if session is not None:
            udp_track_id = session.get("m_trackId")
            if udp_track_id is not None:
                track = get_track_by_udp_id(int(udp_track_id))
                if track is not None:
                    track_id = track.track_id
                    _set_current_track(request, track_id, "telemetry")
    if track_id is None:
        raise fail(
            message="无法确定赛道，请先选择赛道或连接遥测",
            code=4092,
            http_status=409,
        )

    # 存入 DB
    setup_id = store.import_setup(track_id, params)
    data = SetupSnapshot(
        setup_id=setup_id,
        track_id=track_id,
        imported_at=None,
        params=params,
    )
    return ok(data=data.model_dump(), message=f"已导入调教快照（id={setup_id}）")


@router.get("/setup/current")
async def get_current_setup(
    request: Request,
    track_id: str | None = Query(default=None, description="赛道标识；缺省用当前赛道"),
) -> dict[str, Any]:
    """读取导入的调教快照。"""
    svc = _get_services(request)
    store = svc["store"]
    if store is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    tid = track_id or svc["current_track_id"]
    if tid is None:
        raise fail(
            message="未指定赛道，且无当前赛道",
            code=4093,
            http_status=409,
        )

    snapshot = store.get_latest_setup(tid)
    if snapshot is None:
        raise fail(
            message=f"赛道 {tid} 无导入的调教快照",
            code=4041,
            http_status=404,
        )
    data = SetupSnapshot(
        setup_id=snapshot["id"],
        track_id=snapshot["track_id"],
        imported_at=snapshot.get("imported_at"),
        params=snapshot["params"],
    )
    return ok(data=data.model_dump())


@router.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest, request: Request,
) -> dict[str, Any]:
    """提交单条反馈。"""
    svc = _get_services(request)
    feedback_service = svc["feedback_service"]
    if feedback_service is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    # 校验赛道存在
    track = get_track_by_id(body.track_id)
    if track is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )

    # 校验弯道编号（若提供）
    if body.corner_number is not None and body.corner_number > len(track.corners):
        raise fail(
            message=f"弯道编号 {body.corner_number} 超出赛道 {body.track_id} 的弯道数 {len(track.corners)}",
            code=4001,
        )

    # 校验症状标识
    valid_symptoms = {s.value for s in Symptom}
    if body.symptom not in valid_symptoms:
        raise fail(
            message=f"未知症状标识：{body.symptom}，合法值：{', '.join(sorted(valid_symptoms))}",
            code=4002,
        )

    # 获取关联的 setup_id（可选）
    setup_id = None
    store = svc["store"]
    if store is not None:
        latest = store.get_latest_setup(body.track_id)
        if latest is not None:
            setup_id = latest["id"]

    try:
        result = feedback_service.submit_feedback(
            track_id=body.track_id,
            corner_number=body.corner_number,
            symptom=body.symptom,
            strength=body.strength,
            setup_id=setup_id,
        )
    except ValueError as e:
        raise fail(message=str(e), code=4003) from e

    data = FeedbackRecord(
        id=result["id"],
        track_id=result["track_id"],
        corner_number=result["corner_number"],
        symptom=result["symptom"],
        category=result["category"],
        strength=result["strength"],
        setup_id=result.get("setup_id"),
    )
    return ok(data=data.model_dump(), message="反馈已提交")


@router.get("/feedback")
async def list_feedbacks(
    request: Request,
    track_id: str | None = Query(default=None, description="赛道标识；缺省用当前赛道"),
) -> dict[str, Any]:
    """查询已提交反馈。"""
    svc = _get_services(request)
    feedback_service = svc["feedback_service"]
    if feedback_service is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    tid = track_id or svc["current_track_id"]
    if tid is None:
        raise fail(
            message="未指定赛道，且无当前赛道",
            code=4093,
            http_status=409,
        )

    feedbacks = feedback_service.get_feedbacks(tid)
    data = [
        FeedbackRecord(
            id=fb["id"],
            track_id=fb["track_id"],
            corner_number=fb.get("corner_number"),
            symptom=fb["symptom"],
            category=fb["category"],
            strength=fb["strength"],
            setup_id=fb.get("setup_id"),
            created_at=fb.get("created_at"),
        ).model_dump()
        for fb in feedbacks
    ]
    return ok(data=data, message=f"共 {len(data)} 条反馈")


@router.post("/suggest")
async def suggest(
    body: SuggestRequest, request: Request,
) -> dict[str, Any]:
    """触发建议生成。

    流程：校验有反馈 → Dx → SetupDelta → 报告 → 落库 → WS 推送。
    无反馈时返回 400 引导消息。
    """
    svc = _get_services(request)
    store = svc["store"]
    feedback_service = svc["feedback_service"]
    iteration_service = svc["iteration_service"]
    stream = svc["telemetry_stream"]

    if store is None or feedback_service is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    # 校验赛道存在
    track = get_track_by_id(body.track_id)
    if track is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )

    # ① 校验有反馈
    can_suggest, hint = feedback_service.validate_before_suggest(body.track_id)
    if not can_suggest:
        raise fail(
            message=hint,
            code=4004,
            http_status=400,
        )

    # ② 收集症状
    feedbacks = feedback_service.get_feedbacks(body.track_id)
    symptoms = feedbacks_to_symptoms(feedbacks)

    # ③ 获取当前调教快照（无则用缺省）
    snapshot = store.get_latest_setup(body.track_id)
    if snapshot is not None:
        current_setup = snapshot["params"]
        setup_id = snapshot["id"]
    else:
        current_setup = CarSetup.default().to_dict()
        setup_id = None

    # ④ 提取遥测摘要
    telemetry_summary = None
    if stream is not None:
        all_latest = stream.get_all_latest()
        telemetry_summary = extract_telemetry_summary(all_latest)

    # ⑤ 生成建议
    try:
        suggestion_result = generate_suggestion(
            symptoms=symptoms,
            current_setup=current_setup,
            track_id=body.track_id,
            telemetry=telemetry_summary,
        )
    except Exception as e:
        logger.exception("generate_suggestion failed")
        raise fail(
            message=f"建议生成失败：{e}",
            code=5002,
            http_status=500,
        ) from e

    # ⑥ 组装报告
    report = build_report(
        suggestion_result=suggestion_result,
        track_id=body.track_id,
        setup_id=setup_id,
    )

    # ⑦ 落库
    report_json = json.dumps(report, ensure_ascii=False, sort_keys=True)
    suggestion_id = store.save_suggestion(
        track_id=body.track_id,
        report_json=report_json,
        setup_id=setup_id,
    )

    # ⑧ 迭代记录
    if iteration_service is not None:
        try:
            iteration_service.create_iteration(
                track_id=body.track_id,
                before_setup_id=setup_id,
                after_setup_id=None,
                suggestion_id=suggestion_id,
            )
        except Exception:
            logger.exception("create_iteration failed (non-fatal)")

    # ⑨ WS 推送（best-effort）
    await _push_suggestion_via_ws(request, suggestion_id, report)

    data = SuggestionView(
        suggestion_id=suggestion_id,
        track_id=body.track_id,
        created_at=None,
        report=report,
    )
    return ok(data=data.model_dump(), message="建议已生成")


@router.get("/suggest/latest")
async def get_latest_suggestion(
    request: Request,
    track_id: str | None = Query(default=None, description="赛道标识；缺省用当前赛道"),
) -> dict[str, Any]:
    """读取最新建议报告。"""
    svc = _get_services(request)
    store = svc["store"]
    if store is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    tid = track_id or svc["current_track_id"]
    if tid is None:
        raise fail(
            message="未指定赛道，且无当前赛道",
            code=4093,
            http_status=409,
        )

    row = store.get_latest_suggestion(tid)
    if row is None:
        raise fail(
            message=f"赛道 {tid} 无建议报告",
            code=4042,
            http_status=404,
        )

    try:
        report = json.loads(row["report_json"])
    except (json.JSONDecodeError, KeyError) as e:
        raise fail(
            message=f"报告解析失败：{e}",
            code=5003,
            http_status=500,
        ) from e

    data = SuggestionView(
        suggestion_id=row["id"],
        track_id=row["track_id"],
        created_at=row.get("created_at"),
        report=report,
    )
    return ok(data=data.model_dump())


@router.get("/iteration/history")
async def iteration_history(
    request: Request,
    track_id: str | None = Query(default=None, description="赛道标识；缺省用当前赛道"),
) -> dict[str, Any]:
    """历史反馈/建议对比。"""
    svc = _get_services(request)
    store = svc["store"]
    iteration_service = svc["iteration_service"]
    if store is None or iteration_service is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    tid = track_id or svc["current_track_id"]
    if tid is None:
        raise fail(
            message="未指定赛道，且无当前赛道",
            code=4093,
            http_status=409,
        )

    iterations = iteration_service.get_history(tid)
    data = [
        IterationView(
            id=it["id"],
            track_id=it["track_id"],
            round_no=it["round_no"],
            before_setup_id=it.get("before_setup_id"),
            after_setup_id=it.get("after_setup_id"),
            suggestion_id=it.get("suggestion_id"),
            created_at=it["created_at"],
        ).model_dump()
        for it in iterations
    ]
    return ok(data=data, message=f"共 {len(data)} 轮迭代")


# =========================================================================== #
# WS 推送辅助（best-effort，不阻塞 REST 响应）
# =========================================================================== #
async def _push_suggestion_via_ws(
    request: Request, suggestion_id: int, report: dict[str, Any],
) -> None:
    """通过 WebSocket 推送建议生成结果（best-effort）。

    从 app.state 获取 WS 连接管理器；无连接或推送失败时静默跳过。
    """
    ws_manager = getattr(request.app.state, "ws_manager", None)
    if ws_manager is None:
        return
    try:
        await ws_manager.broadcast(
            event="suggestion",
            payload={"suggestion_id": suggestion_id, "report_json": report},
        )
    except Exception:
        logger.exception("WS push suggestion failed (non-fatal)")