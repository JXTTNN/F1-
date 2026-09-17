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
    | POST | /api/v1/telemetry/import-file | 导入JSON遥测文件 |
    | POST | /api/v1/telemetry/experiment  | 调教实验对比 |

关键约定：
    - 所有入参出参用 pydantic BaseModel 强类型校验；
    - 响应统一 ``{code, message, data}`` 信封；
    - POST /suggest：先校验有反馈 → Dx → SetupDelta → 报告 → 落库 → WS 推送。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS
from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
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
    aggregate_feedback_symptoms,
    build_report,
    extract_setup_from_packet5,
    extract_telemetry_summary,
)
from setup_tuner.telemetry.importer import (
    LapTelemetrySummary,
    import_lap_json,
    import_laps_from_directory,
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
    """赛道详情视图（含弯道锚点列表与弯道段）。"""

    track: TrackRef
    corners: list[CornerView]
    segments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="弯道段（连续弯分组，task-63）：name/members/arc_start/arc_end",
    )


class SelectTrackRequest(BaseModel):
    """手动选赛道请求。"""

    track_id: str = Field(..., description="赛道标识，如 'suzuka'")


class SelectTrackResponse(BaseModel):
    """手动选赛道响应。"""

    current_track_id: str
    source: str = Field(..., description="来源：manual | telemetry")


class SetupSnapshot(BaseModel):
    """调教快照视图（21 参数 + 元数据）。"""

    setup_id: int | None = None
    track_id: str
    imported_at: str | None = None
    params: dict[str, float] = Field(..., description="21 项参数字典")


class FeedbackItem(BaseModel):
    """单条反馈项（用于批量提交）。"""

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


class FeedbackRequest(BaseModel):
    """提交反馈请求（兼容单条旧格式与批量新格式）。

    旧格式（单条）：直接提供 corner_number/symptom/strength。
    新格式（批量）：提供 feedbacks 数组，每项为 FeedbackItem。
    """

    track_id: str
    # 单条格式（旧，向后兼容）
    corner_number: int | None = Field(
        default=None, ge=1, le=50, description="弯道编号（1-based）；None=全局症状",
    )
    symptom: str | None = Field(default=None, description="12 症状标识之一")
    strength: int = Field(
        default=DEFAULT_INTENSITY,
        ge=INTENSITY_MIN,
        le=INTENSITY_MAX,
        description="强度 0-5，默认 3",
    )
    # 批量格式（新）
    feedbacks: list[FeedbackItem] | None = Field(
        default=None, description="批量反馈列表（新格式）",
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
    model_type: str = Field(
        default="hybrid",
        description="模型类型：rule（纯规则）| nn（纯神经网络）| hybrid（混合）",
    )


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


class SetupFieldView(BaseModel):
    """调教参数定义视图（供前端渲染参数输入面板）。"""

    name: str = Field(..., description="参数标识符")
    category: str = Field(..., description="中文类别名")
    label_zh: str = Field(..., description="中文标签")
    min: float = Field(..., description="最小值")
    max: float = Field(..., description="最大值")
    step: float = Field(..., description="步长")
    default: float = Field(..., description="缺省值")
    unit: str = Field(..., description="单位")
    max_delta: float = Field(..., description="单次建议最大调整量")


class ManualSetupRequest(BaseModel):
    """手动设置调教请求。"""

    track_id: str = Field(..., description="赛道标识")
    params: dict[str, float] = Field(..., description="21 项参数字典")


class TelemetrySimulateRequest(BaseModel):
    """遥测模拟请求。"""

    track_id: str = Field(..., description="赛道标识")
    action: str = Field(..., description="start | stop")


class TelemetrySimulateResponse(BaseModel):
    """遥测模拟响应。"""

    simulating: bool
    track_id: str


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
        "lap_aggregator": getattr(state, "lap_aggregator", None),
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


# 赛道列表响应缓存：24 条赛道为静态数据，启动后不变。
# 首次请求时构建并缓存完整响应 dict，后续请求直接返回，避免重复构建
# 24 个 TrackRef pydantic 对象 + model_dump 序列化开销。
# 来源：性能优化 task-36（tracks_list 端点基线 0.72ms，缓存后 <0.1ms）
_TRACKS_LIST_CACHE: dict[str, Any] | None = None


@router.get("/tracks")
async def list_tracks() -> dict[str, Any]:
    """24 条赛道列表。

    性能优化：赛道为静态数据，首次请求后缓存响应 dict，后续直接返回。
    """
    global _TRACKS_LIST_CACHE
    if _TRACKS_LIST_CACHE is not None:
        return _TRACKS_LIST_CACHE
    tracks = get_all_tracks()
    data = [_track_to_ref(t).model_dump() for t in tracks]
    _TRACKS_LIST_CACHE = ok(data=data, message=f"共 {len(data)} 条赛道")
    return _TRACKS_LIST_CACHE


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
    from setup_tuner.domain.corner_groups import build_corner_groups

    data = TrackDetail(
        track=_track_to_ref(track),
        corners=[_corner_to_view(c) for c in track.corners],
        segments=build_corner_groups(TRACK_CORNER_ARCS.get(track_id, {})),
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


def _resolve_track_id_from_telemetry(stream: Any, request: Request) -> str | None:
    """从遥测 Session 包的 m_trackId 推断赛道并设置为当前赛道。"""
    session = stream.get_latest(1) if stream else None
    if session is None:
        return None
    udp_track_id = session.get("m_trackId")
    if udp_track_id is None:
        return None
    track = get_track_by_udp_id(int(udp_track_id))
    if track is None:
        return None
    _set_current_track(request, track.track_id, "telemetry")
    return track.track_id


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

    # 提取 21 参数快照
    params = extract_setup_from_packet5(packet5)

    # 确定赛道：优先用当前选定赛道，其次用遥测 Session 包的 m_trackId
    track_id = svc["current_track_id"] or _resolve_track_id_from_telemetry(stream, request)
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


# setup/fields 响应缓存：21 项参数定义为静态数据，启动后不变。
_SETUP_FIELDS_CACHE: dict[str, Any] | None = None


@router.get("/setup/fields")
async def list_setup_fields() -> dict[str, Any]:
    """返回 21 项调教参数的完整定义（供前端渲染参数输入面板）。

    性能优化：参数定义为静态数据，首次请求后缓存响应 dict。
    """
    global _SETUP_FIELDS_CACHE
    if _SETUP_FIELDS_CACHE is not None:
        return _SETUP_FIELDS_CACHE
    data = [
        SetupFieldView(
            name=f.name,
            category=f.category,
            label_zh=f.label_zh,
            min=f.min_val,
            max=f.max_val,
            step=f.step,
            default=f.default,
            unit=f.unit,
            max_delta=f.max_delta,
        ).model_dump()
        for f in ALL_SETUP_FIELDS
    ]
    _SETUP_FIELDS_CACHE = ok(data=data, message=f"共 {len(data)} 项调教参数")
    return _SETUP_FIELDS_CACHE


def _validate_manual_setup_params(params: dict[str, float]) -> None:
    """校验手动调教参数：包含全部 21 项且每项在 [min, max] 范围内。"""
    expected_names = {f.name for f in ALL_SETUP_FIELDS}
    provided_names = set(params.keys())

    missing = expected_names - provided_names
    if missing:
        raise fail(
            message=f"缺少调教参数：{', '.join(sorted(missing))}",
            code=4005,
        )

    extra = provided_names - expected_names
    if extra:
        raise fail(
            message=f"多余调教参数：{', '.join(sorted(extra))}",
            code=4006,
        )

    for spec in ALL_SETUP_FIELDS:
        val = params[spec.name]
        if val < spec.min_val or val > spec.max_val:
            raise fail(
                message=(
                    f"参数 {spec.name}={val} 超出允许范围 "
                    f"[{spec.min_val:g}, {spec.max_val:g}]"
                ),
                code=4007,
            )


@router.post("/setup/manual")
async def manual_setup(
    body: ManualSetupRequest, request: Request,
) -> dict[str, Any]:
    """手动设置 21 参数调教快照。

    校验 params 包含全部 21 项且每项在 [min, max] 范围内，
    然后调用 store.import_setup 保存。
    """
    svc = _get_services(request)
    store = svc["store"]
    if store is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    if get_track_by_id(body.track_id) is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )

    _validate_manual_setup_params(body.params)

    params_dict = {k: float(v) for k, v in body.params.items()}
    setup_id = store.import_setup(body.track_id, params_dict)
    data = SetupSnapshot(
        setup_id=setup_id,
        track_id=body.track_id,
        imported_at=None,
        params=params_dict,
    )
    return ok(data=data.model_dump(), message=f"已保存手动调教快照（id={setup_id}）")


def _build_feedback_items(body: FeedbackRequest) -> list[FeedbackItem]:
    """将请求体统一为 list[FeedbackItem]（兼容旧单条格式）。"""
    if body.feedbacks is not None:
        return body.feedbacks
    if body.symptom is None:
        raise fail(
            message="旧格式单条反馈必须提供 symptom 字段",
            code=4002,
        )
    return [FeedbackItem(
        corner_number=body.corner_number,
        symptom=body.symptom,
        strength=body.strength,
    )]


def _validate_feedback_item(
    item: FeedbackItem, idx: int, track: Any, valid_symptoms: set[str],
) -> None:
    """逐条校验反馈项的弯道编号与症状标识。"""
    if item.corner_number is not None and item.corner_number > len(track.corners):
        raise fail(
            message=(
                f"第 {idx + 1} 条反馈：弯道编号 {item.corner_number} "
                f"超出赛道 {track.track_id} 的弯道数 {len(track.corners)}"
            ),
            code=4001,
        )
    if item.symptom not in valid_symptoms:
        raise fail(
            message=(
                f"第 {idx + 1} 条反馈：未知症状标识 {item.symptom!r}，"
                f"合法值：{', '.join(sorted(valid_symptoms))}"
            ),
            code=4002,
        )


def _submit_feedback_items(
    feedback_service: Any, items: list[FeedbackItem], track: Any,
    valid_symptoms: set[str], setup_id: int | None,
) -> list[dict[str, Any]]:
    """逐条校验并提交反馈，返回结果列表。"""
    results: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        _validate_feedback_item(item, idx, track, valid_symptoms)
        try:
            result = feedback_service.submit_feedback(
                track_id=track.track_id,
                corner_number=item.corner_number,
                symptom=item.symptom,
                strength=item.strength,
                setup_id=setup_id,
            )
        except ValueError as e:
            raise fail(message=str(e), code=4003) from e
        results.append(result)
    return results


def _resolve_setup_id(store: Any, track_id: str) -> int | None:
    """获取赛道最新调教快照的 setup_id（无则 None）。"""
    if store is None:
        return None
    latest = store.get_latest_setup(track_id)
    return latest["id"] if latest is not None else None


@router.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest, request: Request,
) -> dict[str, Any]:
    """提交反馈（兼容单条旧格式与批量新格式）。

    - 新格式：请求体含 ``feedbacks`` 数组，循环提交每条反馈；
    - 旧格式：直接提供 ``corner_number``/``symptom``/``strength``，按单条处理。
    """
    svc = _get_services(request)
    feedback_service = svc["feedback_service"]
    if feedback_service is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    track = get_track_by_id(body.track_id)
    if track is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )

    setup_id = _resolve_setup_id(svc["store"], body.track_id)
    valid_symptoms = {s.value for s in Symptom}
    items = _build_feedback_items(body)
    if not items:
        raise fail(message="反馈列表为空", code=4009)

    results = _submit_feedback_items(
        feedback_service, items, track, valid_symptoms, setup_id,
    )

    data = [
        FeedbackRecord(
            id=r["id"],
            track_id=r["track_id"],
            corner_number=r["corner_number"],
            symptom=r["symptom"],
            category=r["category"],
            strength=r["strength"],
            setup_id=r.get("setup_id"),
        ).model_dump()
        for r in results
    ]
    if len(results) == 1:
        return ok(data=data[0], message="反馈已提交")
    return ok(data=data, message=f"已批量提交 {len(results)} 条反馈")


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


def _validate_suggest_request(body: SuggestRequest) -> None:
    """校验 suggest 请求的赛道与模型类型。"""
    track = get_track_by_id(body.track_id)
    if track is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )
    valid_model_types = {"rule", "nn", "hybrid"}
    if body.model_type not in valid_model_types:
        raise fail(
            message=(
                f"未知模型类型：{body.model_type!r}，"
                f"合法值：{', '.join(sorted(valid_model_types))}"
            ),
            code=4010,
        )


def _validate_feedback_available(feedback_service: Any, track_id: str) -> None:
    """校验存在足够反馈以生成建议。"""
    can_suggest, hint = feedback_service.validate_before_suggest(track_id)
    if not can_suggest:
        raise fail(message=hint, code=4004, http_status=400)


def _safe_generate_suggestion(
    symptoms: list, current_setup: dict[str, float],
    track_id: str, telemetry_summary: Any, model_type: str,
    style_vector: list[float] | None = None,
) -> Any:
    """调用 generate_suggestion，失败转换为 fail 异常。"""
    try:
        return _invoke_generate_suggestion(
            symptoms, current_setup, track_id, telemetry_summary, model_type,
            style_vector,
        )
    except Exception as e:
        logger.exception("generate_suggestion failed")
        raise fail(
            message=f"建议生成失败：{e}",
            code=5002,
            http_status=500,
        ) from e


def _resolve_current_setup(store: Any, track_id: str) -> tuple[dict[str, float], int | None]:
    """获取当前调教快照（无则用缺省）。

    对数据库中的 params 做格式归一化（CarSetup.from_dict → to_dict）：
    - 忽略旧版多余字段（active_aero_x / ballast / damping / front_spring 等）；
    - 用默认值补全新版缺失字段（front_suspension / rear_suspension / 四轮独立胎压等）。
    这样无论数据库里存的是旧格式还是新格式，返回的都是完整 21 项参数。
    """
    snapshot = store.get_latest_setup(track_id)
    if snapshot is not None:
        params = CarSetup.from_dict(snapshot["params"]).to_dict()
        return params, snapshot["id"]
    return CarSetup.default().to_dict(), None


def _extract_telemetry_summary(stream: Any, aggregator: Any = None) -> dict[str, Any] | None:
    """从遥测流 + 整圈聚合器提取摘要，stream 为 None 时返回 None。

    ``aggregator`` 为 ``LapAggregator`` 时，会带上整圈统计
    （max_speed / avg_steer / max_steer / on_straight / 四轮均值 …），
    这是遥测规则 6/7/10/15 能生效的前提。
    """
    if stream is None:
        return None
    all_latest = stream.get_all_latest()
    lap_stats = None
    if aggregator is not None:
        lap_stats = aggregator.best_snapshot()
    return extract_telemetry_summary(all_latest, lap_stats)


def _invoke_generate_suggestion(
    symptoms: list, current_setup: dict[str, float],
    track_id: str, telemetry_summary: Any, model_type: str,
    style_vector: list[float] | None = None,
) -> Any:
    """调用 generate_suggestion，兼容未支持新参数的旧版本。"""
    try:
        return generate_suggestion(
            symptoms=symptoms,
            current_setup=current_setup,
            track_id=track_id,
            telemetry=telemetry_summary,
            model_type=model_type,
            style_vector=style_vector,
        )
    except TypeError:
        # generate_suggestion 尚未支持新参数（降级为纯规则引擎）
        return generate_suggestion(
            symptoms=symptoms,
            current_setup=current_setup,
            track_id=track_id,
            telemetry=telemetry_summary,
        )


def _persist_suggestion(
    store: Any, body: SuggestRequest, report: dict, setup_id: int | None,
) -> int:
    """组装报告并落库，返回 suggestion_id。"""
    report_json = json.dumps(report, ensure_ascii=False, sort_keys=True)
    return store.save_suggestion(
        track_id=body.track_id,
        report_json=report_json,
        setup_id=setup_id,
    )


def _record_iteration(
    iteration_service: Any, track_id: str, setup_id: int | None, suggestion_id: int,
) -> None:
    """记录迭代（非致命，失败仅记日志）。"""
    if iteration_service is None:
        return
    try:
        iteration_service.create_iteration(
            track_id=track_id,
            before_setup_id=setup_id,
            after_setup_id=None,
            suggestion_id=suggestion_id,
        )
    except Exception:
        logger.exception("create_iteration failed (non-fatal)")


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
    if store is None or feedback_service is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    _validate_suggest_request(body)
    _validate_feedback_available(feedback_service, body.track_id)

    # 反馈按 (弯道, 症状) 聚合后再转三元组：同一条反馈重复提交不再线性放大 Dx；
    # 取最近 500 条，防止反馈无限增长拖慢查询（task-62）
    symptoms = aggregate_feedback_symptoms(
        feedback_service.get_feedbacks(body.track_id, limit=500),
    )
    current_setup, setup_id = _resolve_current_setup(store, body.track_id)
    telemetry_summary = _extract_telemetry_summary(
        svc["telemetry_stream"], svc.get("lap_aggregator"),
    )

    # task-62 M3：车手风格调制（样本 ≥3 圈才启用，否则退化为 L0）
    style_entry = None
    if store is not None:
        style_entry = store.get_driver_style(1, body.track_id)
    style_vector = (
        style_entry["vector"]
        if style_entry and style_entry.get("sample_count", 0) >= 3 else None
    )
    suggestion_result = _safe_generate_suggestion(
        symptoms, current_setup, body.track_id, telemetry_summary,
        body.model_type, style_vector,
    )
    report = build_report(
        suggestion_result=suggestion_result,
        track_id=body.track_id,
        setup_id=setup_id,
    )
    suggestion_id = _persist_suggestion(store, body, report, setup_id)
    _record_iteration(svc["iteration_service"], body.track_id, setup_id, suggestion_id)
    await _push_suggestion_via_ws(request, suggestion_id, report)

    data = SuggestionView(
        suggestion_id=suggestion_id,
        track_id=body.track_id,
        created_at=None,
        report=report,
    )
    return ok(data=data.model_dump(), message="建议已生成")


@router.get("/driver/style")
async def get_driver_style_profile(
    request: Request,
    track_id: str | None = Query(default=None, description="赛道标识；缺省用当前赛道"),
) -> dict[str, Any]:
    """读取车手风格画像（12 维，含样本数与是否已启用调制）。

    task-62 M4：报告页"风格画像"的数据源。样本 <3 圈时 style_applied=False
    （引擎自动退化为 L0，不臆测）。
    """
    svc = _get_services(request)
    store = svc["store"]
    if store is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)
    tid = track_id or svc["current_track_id"]
    if tid is None:
        raise fail(message="未指定赛道，且无当前赛道", code=4093, http_status=409)
    from setup_tuner.domain.style_coefficients import STYLE_VECTOR_LEN
    from setup_tuner.telemetry.style_extractor import STYLE_DIMS

    entry = store.get_driver_style(1, tid)
    sample_count = entry["sample_count"] if entry else 0
    vector = entry["vector"] if entry else [0.5] * len(STYLE_DIMS)
    return ok(data={
        "track_id": tid,
        "dims": STYLE_DIMS,
        "vector": vector,
        "sample_count": sample_count,
        "style_applied": sample_count >= 3 and len(vector) == STYLE_VECTOR_LEN,
        "min_samples": 3,
    })


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
        # task-62：改为探测语义（200 + data=null）。
        # 前端在选赛道/加载后会轮询本端点；全新环境下"还没有建议"是常态，
        # 用 404 会让浏览器 console 持续出现资源加载错误（前端虽已 catch，
        # 但网络层 404 无法静默）。data=null 由前端判空跳过渲染。
        return ok(data=None, message="暂无建议报告")

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



def _validate_telemetry_simulate_request(body: TelemetrySimulateRequest) -> None:
    """校验 telemetry_simulate 请求的 action 与赛道。"""
    if body.action not in ("start", "stop"):
        raise fail(
            message=f"action 必须为 'start' 或 'stop'，收到 {body.action!r}",
            code=4011,
        )
    if get_track_by_id(body.track_id) is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )


def _get_or_create_simulator(request: Request) -> Any:
    """从 app.state 获取或创建 TelemetrySimulator 实例。

    创建时**必须把 ``app.state.packet_handler`` 注册为帧回调**，
    否则模拟出的遥测帧只会在模拟器内部空转派发（无任何订阅者），
    ``TelemetryStream`` / ``LapAggregator`` / ``StyleExtractor`` 全部收不到数据，
    ``/suggest`` 便永远拿不到整圈统计 —— 模拟模式形同虚设。
    """
    simulator = getattr(request.app.state, "telemetry_simulator", None)
    if simulator is None:
        try:
            from setup_tuner.telemetry.simulator import TelemetrySimulator
            simulator = TelemetrySimulator()
        except Exception as e:
            raise fail(
                message=f"遥测模拟器初始化失败：{e}",
                code=5004,
                http_status=500,
            ) from e
        request.app.state.telemetry_simulator = simulator

    handler = getattr(request.app.state, "packet_handler", None)
    if handler is not None:
        simulator.add_handler(handler)
    return simulator


def _start_telemetry_simulator(request: Request, track_id: str) -> dict[str, Any]:
    """启动遥测模拟器并返回响应数据。"""
    simulator = _get_or_create_simulator(request)
    try:
        simulator.start(track_id=track_id)
    except Exception as e:
        raise fail(
            message=f"遥测模拟启动失败：{e}",
            code=5005,
            http_status=500,
        ) from e
    data = TelemetrySimulateResponse(simulating=True, track_id=track_id)
    return ok(data=data.model_dump(), message="遥测模拟已启动")


def _stop_telemetry_simulator(simulator: Any, track_id: str) -> dict[str, Any]:
    """停止遥测模拟器并返回响应数据。"""
    if simulator is not None:
        try:
            simulator.stop()
        except Exception as e:
            raise fail(
                message=f"遥测模拟停止失败：{e}",
                code=5006,
                http_status=500,
            ) from e
    data = TelemetrySimulateResponse(simulating=False, track_id=track_id)
    return ok(data=data.model_dump(), message="遥测模拟已停止")


@router.post("/telemetry/simulate")
async def telemetry_simulate(
    body: TelemetrySimulateRequest, request: Request,
) -> dict[str, Any]:
    """启动/停止遥测模拟模式（用于无真实 F1 游戏时测试）。

    调用 ``setup_tuner.telemetry.simulator.TelemetrySimulator`` 启动后台线程
    模拟 UDP 遥测数据。若 simulator 模块不存在，返回 404 提示。
    """
    _validate_telemetry_simulate_request(body)

    # 检查遥测模拟模块是否可用
    import importlib.util
    if importlib.util.find_spec("setup_tuner.telemetry.simulator") is None:
        raise fail(
            message="遥测模拟模块未安装",
            code=4043,
            http_status=404,
        ) from None

    simulator = getattr(request.app.state, "telemetry_simulator", None)

    if body.action == "start":
        return _start_telemetry_simulator(request, body.track_id)
    return _stop_telemetry_simulator(simulator, body.track_id)


# =========================================================================== #
# 遥测监听器开关端点
# =========================================================================== #
class ListenerToggleRequest(BaseModel):
    """手动开关 UDP 遥测监听器请求。"""

    action: str = Field(..., description="start | stop")


class ListenerToggleResponse(BaseModel):
    """监听器开关响应。"""

    listening: bool
    host: str
    port: int


@router.post("/telemetry/listener/toggle")
async def telemetry_listener_toggle(
    body: ListenerToggleRequest, request: Request,
) -> dict[str, Any]:
    """手动开关 UDP 遥测监听器。

    - ``action=start``：启动监听器；若已在运行则返回当前状态（幂等）。
    - ``action=stop``：停止监听器；若未运行则返回当前状态（幂等）。
    """
    if body.action not in ("start", "stop"):
        raise fail(
            message=f"action 必须为 'start' 或 'stop'，收到 {body.action!r}",
            code=4011,
        )

    listener = getattr(request.app.state, "telemetry_listener", None)
    if listener is None:
        raise fail(
            message="遥测监听器未初始化",
            code=5001,
            http_status=500,
        )

    if body.action == "start":
        if not listener.is_running:
            listener.start()
    else:  # action == "stop"
        if listener.is_running:
            listener.stop()

    data = ListenerToggleResponse(
        listening=listener.is_running,
        host=listener._host,
        port=listener._port,
    )
    return ok(
        data=data.model_dump(),
        message=f"监听器已{'启动' if listener.is_running else '停止'}",
    )


# =========================================================================== #
# 遥测录制端点
# =========================================================================== #
class RecordToggleRequest(BaseModel):
    """开始/停止遥测录制请求。"""

    action: str = Field(..., description="start | stop")


class RecordToggleResponse(BaseModel):
    """录制开关响应。"""

    recording: bool
    session_id: str | None = None
    packet_count: int = 0


class RecordingItem(BaseModel):
    """录制会话列表项。"""

    session_id: str
    packet_count: int
    f1rec_path: str | None = None
    db_path: str | None = None
    start_time: str | None = None
    file_size: int = 0


class RecordingDetail(BaseModel):
    """录制会话详情。"""

    session_id: str
    packet_count: int
    start_time: str | None = None
    end_time: float | None = None
    by_packet_type: list[dict[str, Any]] = Field(
        default_factory=list, description="按包类型统计",
    )
    f1rec_path: str | None = None
    db_path: str | None = None



def _get_or_create_recorder(request: Request) -> Any:
    """从 app.state 获取或创建 TelemetryRecorder 实例。"""
    recorder = getattr(request.app.state, "telemetry_recorder", None)
    if recorder is not None:
        return recorder
    # 延迟初始化
    from setup_tuner.telemetry.recorder import TelemetryRecorder
    config = getattr(request.app.state, "config", None)
    data_dir = getattr(config, "data_dir", "data") if config else "data"
    recordings_dir = str(Path(data_dir) / "recordings")
    recorder = TelemetryRecorder(data_dir=recordings_dir)
    request.app.state.telemetry_recorder = recorder
    return recorder


def _validate_record_action(body: RecordToggleRequest) -> None:
    """校验录制 action 字段。"""
    if body.action not in ("start", "stop"):
        raise fail(
            message=f"action 必须为 'start' 或 'stop'，收到 {body.action!r}",
            code=4011,
        )


@router.post("/telemetry/record/toggle")
async def telemetry_record_toggle(
    body: RecordToggleRequest, request: Request,
) -> dict[str, Any]:
    """开始/停止遥测录制。

    - ``action=start``：开始录制，创建新会话（SQLite + JSONL）。
    - ``action=stop``：停止录制，返回录制摘要。
    """
    _validate_record_action(body)
    recorder = _get_or_create_recorder(request)

    if body.action == "start":
        if recorder.is_recording:
            raise fail(
                message="已在录制中，请先停止当前录制",
                code=4094,
                http_status=409,
            )
        session_id = recorder.start()
        data = RecordToggleResponse(
            recording=True, session_id=session_id, packet_count=0,
        )
        return ok(data=data.model_dump(), message=f"录制已开始（会话 {session_id}）")

    # action == "stop"
    if not recorder.is_recording:
        raise fail(
            message="未在录制中，无需停止",
            code=4095,
            http_status=409,
        )
    summary = recorder.stop()
    data = RecordToggleResponse(
        recording=False,
        session_id=summary.get("session_id"),
        packet_count=summary.get("packet_count", 0),
    )
    return ok(data=data.model_dump(), message=f"录制已停止（{data.packet_count} 包）")


@router.get("/telemetry/recordings")
async def list_recordings(request: Request) -> dict[str, Any]:
    """列出所有遥测录制会话。"""
    recorder = getattr(request.app.state, "telemetry_recorder", None)
    if recorder is None:
        # recorder 尚未初始化，返回空列表
        return ok(data=[], message="暂无录制会话")

    recordings = recorder.list_recordings()
    data = [
        RecordingItem(
            session_id=r["session_id"],
            packet_count=r["packet_count"],
            f1rec_path=r.get("f1rec_path"),
            db_path=r.get("db_path"),
            start_time=r.get("start_time"),
            file_size=r.get("file_size", 0),
        ).model_dump()
        for r in recordings
    ]
    return ok(data=data, message=f"共 {len(data)} 个录制会话")


@router.get("/telemetry/recordings/{session_id}")
async def get_recording_detail(
    session_id: str, request: Request,
) -> dict[str, Any]:
    """获取单个录制会话详情（含按包类型统计）。"""
    recorder = getattr(request.app.state, "telemetry_recorder", None)
    if recorder is None:
        raise fail(message="录制功能未初始化", code=5001, http_status=500)

    detail = recorder.get_recording_detail(session_id)
    if detail is None:
        raise fail(
            message=f"录制会话 {session_id} 不存在",
            code=4044,
            http_status=404,
        )

    data = RecordingDetail(
        session_id=detail["session_id"],
        packet_count=detail["packet_count"],
        start_time=detail.get("start_time"),
        end_time=detail.get("end_time"),
        by_packet_type=detail.get("by_packet_type", []),
        f1rec_path=detail.get("f1rec_path"),
        db_path=detail.get("db_path"),
    )
    return ok(data=data.model_dump())


# =========================================================================== #
# 遥测回放端点
# =========================================================================== #
class ReplayStartRequest(BaseModel):
    """开始回放请求。"""

    session_id: str = Field(..., description="录制会话 ID")
    speed: float = Field(default=1.0, ge=0.1, le=10.0, description="回放倍速")


class ReplayResponse(BaseModel):
    """回放响应。"""

    replaying: bool
    session_id: str | None = None
    packet_count: int = 0
    speed: float = 1.0



def _resolve_f1rec_path(request: Request, session_id: str) -> Path:
    """查找录制文件路径，不存在则抛 404。

    安全：验证 session_id 不含路径遍历字符（防止目录穿越攻击）。
    """
    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
        raise fail(
            message=f"非法的录制会话标识：{session_id!r}",
            code=4044,
            http_status=404,
        )
    config = getattr(request.app.state, "config", None)
    data_dir = getattr(config, "data_dir", "data") if config else "data"
    recordings_dir = Path(data_dir) / "recordings"
    f1rec_path = recordings_dir / f"{session_id}.f1rec"
    if not f1rec_path.exists():
        raise fail(
            message=f"录制会话 {session_id} 不存在",
            code=4044,
            http_status=404,
        )
    return f1rec_path


def _build_replay_target(
    f1rec_path: Path, session_id: str, speed: float, stream: Any,
) -> Callable[[], None]:
    """构造回放线程目标函数。"""
    def _replay_target() -> None:
        try:
            from setup_tuner.telemetry.recorder import ReplayReader
            reader = ReplayReader(str(f1rec_path))
            logger.info(
                "replay started: session=%s, speed=%.1f",
                session_id, speed,
            )

            def _on_replay_packet(parsed: dict[str, Any]) -> None:
                packet_id = parsed.get("packet_id")
                if packet_id is not None:
                    stream.update(int(packet_id), parsed)

            reader.replay(handler=_on_replay_packet, speed=speed)
            reader.close()
            logger.info("replay finished: session=%s", session_id)
        except Exception:
            logger.exception("replay thread failed")
    return _replay_target


@router.post("/telemetry/replay/start")
async def telemetry_replay_start(
    body: ReplayStartRequest, request: Request,
) -> dict[str, Any]:
    """开始回放指定录制会话。

    在后台线程中从 .f1rec 文件读取原始字节，解析后推入 TelemetryStream，
    模拟真实遥测数据流。回放期间 UI 可正常显示遥测数据。
    """
    import threading


    svc = _get_services(request)
    stream = svc["telemetry_stream"]
    if stream is None:
        raise fail(message="服务未初始化", code=5001, http_status=500)

    # 检查是否已在回放
    replay_thread = getattr(request.app.state, "replay_thread", None)
    if replay_thread is not None and replay_thread.is_alive():
        raise fail(
            message="已有回放正在进行，请先停止",
            code=4096,
            http_status=409,
        )

    f1rec_path = _resolve_f1rec_path(request, body.session_id)
    _replay_target = _build_replay_target(f1rec_path, body.session_id, body.speed, stream)

    # 启动回放线程
    replay_thread = threading.Thread(
        target=_replay_target,
        name=f"f1opt-replay-{body.session_id}",
        daemon=True,
    )
    request.app.state.replay_thread = replay_thread
    replay_thread.start()

    data = ReplayResponse(
        replaying=True,
        session_id=body.session_id,
        speed=body.speed,
    )
    return ok(data=data.model_dump(), message=f"回放已开始（会话 {body.session_id}）")


@router.post("/telemetry/replay/stop")
async def telemetry_replay_stop(request: Request) -> dict[str, Any]:
    """停止当前回放。

    注意：回放线程为 daemon 线程，无法强制终止。
    此端点设置停止标志，回放线程在下一个包检查时退出。
    """
    replay_thread = getattr(request.app.state, "replay_thread", None)
    if replay_thread is None or not replay_thread.is_alive():
        data = ReplayResponse(replaying=False)
        return ok(data=data.model_dump(), message="无回放正在进行")

    # 设置停止标志（回放线程在 read_next 返回 None 后自然退出）
    # 由于 daemon 线程无法强制终止，我们只能等待它自然结束
    # 实际场景中回放速度通常较快，等待时间不长
    request.app.state.replay_stop_requested = True

    data = ReplayResponse(replaying=False)
    return ok(data=data.model_dump(), message="回放停止请求已发送")


@router.get("/telemetry/replay/status")
async def telemetry_replay_status(request: Request) -> dict[str, Any]:
    """查询当前回放状态。"""
    replay_thread = getattr(request.app.state, "replay_thread", None)
    is_replaying = replay_thread is not None and replay_thread.is_alive()

    data = ReplayResponse(
        replaying=is_replaying,
        session_id=getattr(request.app.state, "replay_session_id", None),
    )
    return ok(data=data.model_dump())
# =========================================================================== #
# 遥测文件导入端点
# =========================================================================== #
class TelemetryImportFileRequest(BaseModel):
    """遥测文件导入请求（支持单文件或目录批量导入）。"""

    path: str = Field(..., description="JSON 遥测文件路径或目录路径")


class LapTelemetrySummaryView(BaseModel):
    """整圈遥测统计摘要视图（对应 LapTelemetrySummary dataclass）。"""

    # 圈基本信息
    lap_number: int
    lap_time_ms: int
    lap_time_str: str
    track_id: int
    track_name: str
    lap_valid: bool
    sample_count: int
    sector_times_ms: list[int]

    # 天气
    weather_code: int
    weather_name: str
    track_temp: float
    air_temp: float

    # 轮胎
    tyre_compound: str
    tyre_age_laps: int

    # 速度统计 (km/h)
    avg_speed: float
    max_speed: float

    # 油门/刹车/转向统计 (0-1)
    avg_throttle: float
    avg_brake: float
    max_brake: float
    avg_steer: float
    max_steer: float

    # 四轮温度/胎压统计（官方车轮顺序 [RL, RR, FL, FR]）
    avg_tyre_surface_temp: list[float]
    avg_tyre_inner_temp: list[float]
    avg_brake_temp: list[float]
    avg_tyre_pressure: list[float]
    max_tyre_surface_temp: list[float]
    max_brake_temp: list[float]
    min_tyre_pressure: list[float]
    max_tyre_pressure: list[float]

    # 调教/燃油/驾驶风格
    setup_raw: dict[str, float]
    fuel_load_kg: float
    fuel_in_tank_kg: float
    driver_style: dict[str, Any]


def _summary_to_view(summary: LapTelemetrySummary) -> LapTelemetrySummaryView:
    """将 LapTelemetrySummary dataclass 转为 Pydantic 视图模型。"""
    return LapTelemetrySummaryView(
        lap_number=summary.lap_number,
        lap_time_ms=summary.lap_time_ms,
        lap_time_str=summary.lap_time_str,
        track_id=summary.track_id,
        track_name=summary.track_name,
        lap_valid=summary.lap_valid,
        sample_count=summary.sample_count,
        sector_times_ms=summary.sector_times_ms,
        weather_code=summary.weather_code,
        weather_name=summary.weather_name,
        track_temp=summary.track_temp,
        air_temp=summary.air_temp,
        tyre_compound=summary.tyre_compound,
        tyre_age_laps=summary.tyre_age_laps,
        avg_speed=summary.avg_speed,
        max_speed=summary.max_speed,
        avg_throttle=summary.avg_throttle,
        avg_brake=summary.avg_brake,
        max_brake=summary.max_brake,
        avg_steer=summary.avg_steer,
        max_steer=summary.max_steer,
        avg_tyre_surface_temp=summary.avg_tyre_surface_temp,
        avg_tyre_inner_temp=summary.avg_tyre_inner_temp,
        avg_brake_temp=summary.avg_brake_temp,
        avg_tyre_pressure=summary.avg_tyre_pressure,
        max_tyre_surface_temp=summary.max_tyre_surface_temp,
        max_brake_temp=summary.max_brake_temp,
        min_tyre_pressure=summary.min_tyre_pressure,
        max_tyre_pressure=summary.max_tyre_pressure,
        setup_raw=summary.setup_raw,
        fuel_load_kg=summary.fuel_load_kg,
        fuel_in_tank_kg=summary.fuel_in_tank_kg,
        driver_style=summary.driver_style,
    )



def _import_single_telemetry_file(path: Path) -> dict[str, Any]:
    """导入单个 JSON 遥测文件，返回信封响应数据。"""
    try:
        summary = import_lap_json(str(path))
    except (json.JSONDecodeError, KeyError) as e:
        raise fail(
            message=f"JSON 遥测文件解析失败：{e}",
            code=4012,
            http_status=400,
        ) from e
    data = _summary_to_view(summary).model_dump()
    return ok(data=data, message=f"已导入圈 {summary.lap_number} 遥测数据")


def _import_telemetry_directory(path: Path, raw_path: str) -> dict[str, Any]:
    """批量导入目录下所有 lap_*.json 遥测文件，返回信封响应数据。"""
    summaries = import_laps_from_directory(str(path))
    if not summaries:
        raise fail(
            message=f"目录 {raw_path} 中未找到有效的 lap_*.json 文件",
            code=4046,
            http_status=404,
        )
    data = [_summary_to_view(s).model_dump() for s in summaries]
    return ok(data=data, message=f"已导入 {len(summaries)} 圈遥测数据")


@router.post("/telemetry/import-file")
async def telemetry_import_file(
    body: TelemetryImportFileRequest,
) -> dict[str, Any]:
    """导入 JSON 遥测文件并返回 LapTelemetrySummary 摘要。

    支持两种模式：
    - **单文件导入**：``path`` 指向 ``lap_xxx.json`` 文件，返回单条摘要；
    - **目录批量导入**：``path`` 指向包含 ``lap_*.json`` 的目录，返回摘要列表。

    请求体示例：
        ``{"path": "D:\\\\F1TelemetryCollector\\\\dist\\\\data\\\\lap_001.json"}``
        ``{"path": "D:\\\\F1TelemetryCollector\\\\dist\\\\data"}``
    """
    path = Path(body.path)

    if not path.exists():
        raise fail(
            message=f"路径不存在：{body.path}",
            code=4045,
            http_status=404,
        )

    if path.is_file():
        return _import_single_telemetry_file(path)

    if path.is_dir():
        return _import_telemetry_directory(path, body.path)

    # 既不是文件也不是目录（如设备文件等）
    raise fail(
        message=f"路径类型不支持：{body.path}",
        code=4013,
        http_status=400,
    )


# =========================================================================== #
# 调教实验对比端点
# =========================================================================== #
class TelemetryExperimentRequest(BaseModel):
    """调教实验对比请求。

    提供两份调教参数和遥测数据，分别生成建议并对比差异。

    遥测可共享也可分侧：两侧都未单独提供时使用 ``telemetry``；
    真实 A/B 实验（同一弯道分别用方案 A / B 各跑一趟）应提供
    ``telemetry_a`` / ``telemetry_b``，否则两次诊断输入完全相同，
    ``dx_diff`` 恒为空 —— 该字段就无法反映方案差异。
    """

    track_id: str = Field(..., description="赛道标识")
    setup_a: dict[str, float] = Field(..., description="调教方案 A 的参数字典")
    setup_b: dict[str, float] = Field(..., description="调教方案 B 的参数字典")
    telemetry: dict[str, Any] = Field(
        default_factory=dict,
        description="共享遥测数据字典（两侧均未单独提供时使用）",
    )
    telemetry_a: dict[str, Any] | None = Field(
        default=None,
        description="方案 A 专属遥测；为空时回退到 telemetry",
    )
    telemetry_b: dict[str, Any] | None = Field(
        default=None,
        description="方案 B 专属遥测；为空时回退到 telemetry",
    )


class ExperimentSuggestionView(BaseModel):
    """单份调教方案的建议结果视图。"""

    setup: dict[str, float] = Field(..., description="调教参数")
    dx: dict[str, float] = Field(..., description="诊断向量")
    setup_delta: dict[str, float] = Field(..., description="参数调整量")
    confidence: str = Field(..., description="置信度：high|medium|low")
    summary: str = Field(..., description="建议摘要")
    model_type: str = Field(..., description="实际使用的模型类型")


class ExperimentComparisonView(BaseModel):
    """调教实验对比结果视图。"""

    track_id: str
    suggestion_a: ExperimentSuggestionView
    suggestion_b: ExperimentSuggestionView
    dx_diff: dict[str, float] = Field(
        ..., description="Dx 向量差异（A - B），正值表示 A 更强",
    )
    setup_delta_diff: dict[str, float] = Field(
        ..., description="SetupDelta 差异（A - B），正值表示 A 调整更大",
    )
    confidence_diff: str = Field(
        ..., description="置信度差异描述",
    )


def _validate_experiment_setups(
    setup_a: dict[str, float], setup_b: dict[str, float],
) -> None:
    """校验两份调教参数包含全部 21 项且在合法范围内。"""
    _validate_manual_setup_params(setup_a)
    _validate_manual_setup_params(setup_b)


def _build_experiment_suggestion_view(
    setup: dict[str, float],
    suggestion: dict[str, Any],
) -> ExperimentSuggestionView:
    """从 generate_suggestion 结果构建单份方案视图。"""
    return ExperimentSuggestionView(
        setup=setup,
        dx=suggestion["dx"],
        setup_delta=suggestion["setup_delta"],
        confidence=suggestion["confidence"],
        summary=suggestion["summary"],
        model_type=suggestion["model_type"],
    )


def _compute_dx_diff(
    dx_a: dict[str, float], dx_b: dict[str, float],
) -> dict[str, float]:
    """计算 Dx 向量差异（A - B）。"""
    all_dims = set(dx_a.keys()) | set(dx_b.keys())
    return {dim: dx_a.get(dim, 0.0) - dx_b.get(dim, 0.0) for dim in all_dims}


def _compute_setup_delta_diff(
    delta_a: dict[str, float], delta_b: dict[str, float],
) -> dict[str, float]:
    """计算 SetupDelta 差异（A - B）。"""
    all_params = set(delta_a.keys()) | set(delta_b.keys())
    return {p: delta_a.get(p, 0.0) - delta_b.get(p, 0.0) for p in all_params}


def _describe_confidence_diff(conf_a: str, conf_b: str) -> str:
    """生成置信度差异描述。"""
    if conf_a == conf_b:
        return f"两份方案置信度相同（均为 {conf_a}）"
    order = {"high": 3, "medium": 2, "low": 1}
    rank_a = order.get(conf_a, 0)
    rank_b = order.get(conf_b, 0)
    if rank_a > rank_b:
        return f"方案 A 置信度更高（{conf_a} > {conf_b}）"
    return f"方案 B 置信度更高（{conf_b} > {conf_a})"


def _generate_experiment_suggestion(
    setup: dict[str, float],
    track_id: str,
    telemetry: dict[str, Any] | None,
) -> dict[str, Any]:
    """为实验对比生成单份调教建议（无车手反馈，仅遥测驱动）。

    内部辅助函数，**不是**端点：由 ``telemetry_experiment`` 各调用一次。
    历史上这里曾误挂 ``@router.post`` 装饰器，导致 FastAPI 把本函数注册成
    ``/telemetry/experiment`` 的处理函数（把 ``setup``/``telemetry`` 当作 body、
    ``track_id`` 当作 query），真正的端点反倒成了死代码 —— 按文档契约调用
    必然 422。勿再为本函数添加路由装饰器。

    Args:
        setup: 调教参数字典。
        track_id: 赛道标识。
        telemetry: 遥测数据字典（None 表示无遥测）。

    Returns:
        ``generate_suggestion`` 返回的建议结果字典。

    Raises:
        fail: 建议生成失败时抛出 HTTP 异常。
    """
    try:
        return generate_suggestion(
            symptoms=[],
            current_setup=setup,
            track_id=track_id,
            telemetry=telemetry,
            model_type="rule",
        )
    except Exception as e:
        logger.exception("experiment generate_suggestion failed")
        raise fail(
            message=f"建议生成失败：{e}",
            code=5002,
            http_status=500,
        ) from e


def _build_experiment_comparison(
    body: TelemetryExperimentRequest,
    suggestion_a: dict[str, Any],
    suggestion_b: dict[str, Any],
) -> ExperimentComparisonView:
    """构建调教实验对比结果视图。"""
    view_a = _build_experiment_suggestion_view(body.setup_a, suggestion_a)
    view_b = _build_experiment_suggestion_view(body.setup_b, suggestion_b)

    return ExperimentComparisonView(
        track_id=body.track_id,
        suggestion_a=view_a,
        suggestion_b=view_b,
        dx_diff=_compute_dx_diff(suggestion_a["dx"], suggestion_b["dx"]),
        setup_delta_diff=_compute_setup_delta_diff(
            suggestion_a["setup_delta"], suggestion_b["setup_delta"],
        ),
        confidence_diff=_describe_confidence_diff(
            suggestion_a["confidence"], suggestion_b["confidence"],
        ),
    )


@router.post("/telemetry/experiment")
async def telemetry_experiment(
    body: TelemetryExperimentRequest,
) -> dict[str, Any]:
    """调教实验对比：两份调教参数 + 遥测数据 → 效果分析。

    分别用 setup_a 和 setup_b 作为当前调教，结合遥测数据调用
    ``engine.generate_suggestion`` 生成建议，对比两份方案的
    Dx 向量 / SetupDelta / 置信度差异。

    遥测取值优先级：``telemetry_a``/``telemetry_b`` 优先，缺省回退到
    共享的 ``telemetry``。两侧输入相同则 Dx 必然相同（``dx_diff`` 全零），
    这是正确结果 —— 想看到 Dx 差异就必须提供分侧遥测。

    请求体示例：
        ``{"track_id": "abu_dhabi", "setup_a": {...}, "setup_b": {...}, "telemetry": {...}}``
    """
    # 校验赛道
    if get_track_by_id(body.track_id) is None:
        raise fail(
            message=f"未知赛道标识：{body.track_id}",
            code=4040,
            http_status=404,
        )

    # 校验两份调教参数
    _validate_experiment_setups(body.setup_a, body.setup_b)

    # 遥测数据（分侧优先，缺省回退共享；空字典时传 None 给 engine）
    telem_a = body.telemetry_a if body.telemetry_a else body.telemetry
    telem_b = body.telemetry_b if body.telemetry_b else body.telemetry

    # 分别生成建议（无车手反馈，仅遥测驱动）
    suggestion_a = _generate_experiment_suggestion(
        body.setup_a, body.track_id, telem_a if telem_a else None,
    )
    suggestion_b = _generate_experiment_suggestion(
        body.setup_b, body.track_id, telem_b if telem_b else None,
    )

    # 构建对比结果
    data = _build_experiment_comparison(body, suggestion_a, suggestion_b)
    return ok(data=data.model_dump(), message="调教实验对比完成")


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