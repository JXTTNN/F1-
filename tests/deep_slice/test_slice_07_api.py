"""切片7 深度测试 —— setup_tuner/api/routes.py + ws.py + envelope.py。

覆盖 REST API（11 个端点）+ WebSocket + 统一信封格式。
用 fastapi.testclient.TestClient 真实启动 API（含 lifespan），非 mock。

5 种测试方式（每个 class 对应一种，注释明确标注）：
    1. TestUnit     — 单元测试：每个端点的正常输入正确性
    2. TestBoundary — 边界/异常测试：非法 track_id、越界弯道、未知症状、缺参
    3. TestProperty — 属性不变量测试：信封格式一致性、幂等性、确定性
    4. TestStatic   — 静态分析：状态码正确性、信封 code 语义、路由前缀
    5. TestSmoke    — 实际运行冒烟：完整反馈→建议流程、WebSocket 连接
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from setup_tuner.api.envelope import ApiError, Envelope, error, fail, ok


# ===========================================================================
# 1. 单元测试 (unit) — 每个端点/函数的正常输入正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证各 API 端点与信封函数在正常输入下的正确行为。"""

    # ---------- 信封函数 ----------
    def test_ok_default(self) -> None:
        """ok() 默认返回 {code:0, message:'ok', data:None}。"""
        env = ok()
        assert env == {"code": 0, "message": "ok", "data": None}

    def test_ok_with_data(self) -> None:
        """ok(data=...) 携带数据载荷。"""
        env = ok(data={"a": 1}, message="成功")
        assert env["code"] == 0
        assert env["message"] == "成功"
        assert env["data"] == {"a": 1}

    def test_fail_returns_http_exception(self) -> None:
        """fail() 返回 HTTPException，detail 为信封 dict，4xx 状态码。"""
        exc = fail(message="参数错误", code=4001)
        assert exc.status_code == 400
        assert exc.detail["code"] == 4001
        assert exc.detail["message"] == "参数错误"
        assert exc.detail["data"] is None

    def test_error_returns_http_exception_5xx(self) -> None:
        """error() 返回 HTTPException，code 负数，5xx 状态码。"""
        exc = error(message="内部错误", code=-1)
        assert exc.status_code == 500
        assert exc.detail["code"] == -1

    def test_envelope_model_defaults(self) -> None:
        """Envelope pydantic 模型默认值正确。"""
        e = Envelope()
        assert e.code == 0
        assert e.message == "ok"
        assert e.data is None

    def test_api_error_attributes(self) -> None:
        """ApiError 业务异常携带 message/code/data/http_status 属性。"""
        err = ApiError("test", code=42, data={"k": "v"}, http_status=418)
        assert err.message == "test"
        assert err.code == 42
        assert err.data == {"k": "v"}
        assert err.http_status == 418

    # ---------- REST 端点 ----------
    def test_health(self, app_client: TestClient) -> None:
        """GET /api/v1/health 返回 200 + 信封 + 遥测状态。"""
        resp = app_client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert "status" in body["data"]
        assert "telemetry_connected" in body["data"]
        assert "udp_host" in body["data"]
        assert "udp_port" in body["data"]

    def test_list_tracks(self, app_client: TestClient) -> None:
        """GET /api/v1/tracks 返回 24 条赛道。"""
        resp = app_client.get("/api/v1/tracks")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert len(body["data"]) == 24
        # 每条含必要字段
        t = body["data"][0]
        for field in ("track_id", "official_name", "circuit_name", "track_type",
                      "length_m", "corners", "svg_path"):
            assert field in t

    def test_get_track_detail(self, app_client: TestClient) -> None:
        """GET /api/v1/tracks/suzuka 返回赛道详情 + 弯道锚点。"""
        resp = app_client.get("/api/v1/tracks/suzuka")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["track"]["track_id"] == "suzuka"
        assert len(body["data"]["corners"]) > 0

    def test_select_current_track(self, app_client: TestClient) -> None:
        """POST /api/v1/tracks/current 手动选赛道。"""
        resp = app_client.post("/api/v1/tracks/current", json={"track_id": "monza"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["current_track_id"] == "monza"
        assert body["data"]["source"] == "manual"

    def test_submit_feedback(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback 提交单条反馈。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka",
            "corner_number": 1,
            "symptom": "understeer",
            "strength": 4,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["symptom"] == "understeer"
        assert body["data"]["strength"] == 4
        assert body["data"]["category"] == "entry"

    def test_list_feedbacks(self, app_client: TestClient) -> None:
        """GET /api/v1/feedback 查询已提交反馈。"""
        app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "corner_number": 1,
            "symptom": "understeer", "strength": 3,
        })
        resp = app_client.get("/api/v1/feedback", params={"track_id": "suzuka"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert len(body["data"]) == 1

    def test_get_latest_suggestion_empty(self, app_client: TestClient) -> None:
        """GET /api/v1/suggest/latest 无建议时返回 404。"""
        resp = app_client.get("/api/v1/suggest/latest", params={"track_id": "suzuka"})
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0

    def test_iteration_history_empty(self, app_client: TestClient) -> None:
        """GET /api/v1/iteration/history 无迭代时返回空列表。"""
        resp = app_client.get("/api/v1/iteration/history", params={"track_id": "suzuka"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"] == []


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 非法 track_id、越界弯道、未知症状、缺参
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证 API 在非法/极端输入下的错误处理。"""

    def test_get_track_unknown(self, app_client: TestClient) -> None:
        """GET /api/v1/tracks/unknown 返回 404 + 非零 code。"""
        resp = app_client.get("/api/v1/tracks/unknown_track_xxx")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0
        assert "未知" in body["message"] or "不存在" in body["message"]

    def test_select_track_unknown(self, app_client: TestClient) -> None:
        """POST /api/v1/tracks/current 未知 track_id 返回 404。"""
        resp = app_client.post("/api/v1/tracks/current", json={"track_id": "ghost"})
        assert resp.status_code == 404
        assert resp.json()["code"] != 0

    def test_feedback_unknown_track(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback 未知 track_id 返回 404。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "ghost", "symptom": "understeer", "strength": 3,
        })
        assert resp.status_code == 404

    def test_feedback_unknown_symptom(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback 未知 symptom 返回 400。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "symptom": "ghost_symptom", "strength": 3,
        })
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] != 0

    def test_feedback_strength_out_of_range(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback strength=99 应被 pydantic 拒绝（422）。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "symptom": "understeer", "strength": 99,
        })
        assert resp.status_code == 422

    def test_feedback_strength_negative(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback strength=-1 应被 pydantic 拒绝。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "symptom": "understeer", "strength": -1,
        })
        assert resp.status_code == 422

    def test_feedback_corner_out_of_range(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback corner_number=30 超出赛道弯道数返回 400。

        注：corner_number 受 pydantic ``le=50`` 约束，>50 返回 422；
        此处用 30（<=50 但 > suzuka 的 18 弯）触发业务层 400。
        """
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "corner_number": 30,
            "symptom": "understeer", "strength": 3,
        })
        assert resp.status_code == 400

    def test_feedback_missing_track_id(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback 缺 track_id 应被 pydantic 拒绝。"""
        resp = app_client.post("/api/v1/feedback", json={
            "symptom": "understeer", "strength": 3,
        })
        assert resp.status_code == 422

    def test_feedback_missing_symptom(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback 缺 symptom 应被 pydantic 拒绝。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "strength": 3,
        })
        assert resp.status_code == 422

    def test_suggest_no_feedback(self, app_client: TestClient) -> None:
        """POST /api/v1/suggest 无反馈时返回 400 + 引导消息。"""
        resp = app_client.post("/api/v1/suggest", json={"track_id": "suzuka"})
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] != 0
        assert "反馈" in body["message"]

    def test_suggest_unknown_track(self, app_client: TestClient) -> None:
        """POST /api/v1/suggest 未知 track_id 返回 404。"""
        resp = app_client.post("/api/v1/suggest", json={"track_id": "ghost"})
        assert resp.status_code == 404

    def test_get_current_setup_no_track(self, app_client: TestClient) -> None:
        """GET /api/v1/setup/current 无当前赛道且未指定 track_id 返回 409。"""
        resp = app_client.get("/api/v1/setup/current")
        assert resp.status_code == 409

    def test_get_current_setup_no_snapshot(self, app_client: TestClient) -> None:
        """GET /api/v1/setup/current 指定无快照的赛道返回 404。"""
        resp = app_client.get("/api/v1/setup/current", params={"track_id": "suzuka"})
        assert resp.status_code == 404

    def test_import_setup_no_packet5(self, app_client: TestClient) -> None:
        """POST /api/v1/setup/import 无 Car Setups 包返回 409。"""
        resp = app_client.post("/api/v1/setup/import")
        assert resp.status_code == 409
        body = resp.json()
        assert body["code"] != 0

    def test_feedback_global_symptom_corner_none(self, app_client: TestClient) -> None:
        """POST /api/v1/feedback corner_number=None 全局症状应正常入库。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "symptom": "bottoming", "strength": 3,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["category"] == "global"
        assert body["data"]["corner_number"] is None

    @pytest.mark.parametrize("symptom", [
        "understeer", "oversteer", "turnin_unresponsive", "brake_long", "lockup",
        "midcorner_unstable", "midcorner_traction", "exit_wheelspin",
        "bottoming", "tyre_wear", "straight_slow", "lap_slow",
    ])
    def test_all_12_symptoms_accepted(
        self, app_client: TestClient, symptom: str,
    ) -> None:
        """全部 12 症状标识应被 API 接受。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "symptom": symptom, "strength": 3,
        })
        assert resp.status_code == 200, f"症状 {symptom} 应被接受"


# ===========================================================================
# 3. 属性不变量测试 (property) — 信封格式一致性、幂等性、确定性
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证信封格式一致性与 API 幂等性。"""

    def test_envelope_format_consistency(self, app_client: TestClient) -> None:
        """所有 200 响应必须含 code/message/data 三键，code=0。"""
        endpoints = [
            ("GET", "/api/v1/health", None),
            ("GET", "/api/v1/tracks", None),
            ("GET", "/api/v1/tracks/suzuka", None),
        ]
        for method, path, _ in endpoints:
            resp = app_client.get(path) if method == "GET" else app_client.post(path)
            body = resp.json()
            assert set(body.keys()) >= {"code", "message", "data"}, \
                f"{method} {path} 响应缺信封键"
            assert body["code"] == 0, f"{method} {path} code 应为 0"

    def test_error_envelope_format(self, app_client: TestClient) -> None:
        """所有错误响应也必须含 code/message/data 三键，code!=0。"""
        resp = app_client.get("/api/v1/tracks/ghost_track")
        body = resp.json()
        assert set(body.keys()) >= {"code", "message", "data"}
        assert body["code"] != 0

    def test_ok_function_idempotent(self) -> None:
        """ok() 是纯函数，相同输入相同输出（确定性）。"""
        e1 = ok(data={"x": 1}, message="m")
        e2 = ok(data={"x": 1}, message="m")
        assert e1 == e2

    def test_fail_function_idempotent(self) -> None:
        """fail() 是纯函数，相同输入产生等价 HTTPException。"""
        e1 = fail(message="err", code=42)
        e2 = fail(message="err", code=42)
        assert e1.status_code == e2.status_code
        assert e1.detail == e2.detail

    def test_list_tracks_deterministic(self, app_client: TestClient) -> None:
        """GET /api/v1/tracks 两次调用结果一致（确定性，无随机）。"""
        r1 = app_client.get("/api/v1/tracks").json()["data"]
        r2 = app_client.get("/api/v1/tracks").json()["data"]
        assert r1 == r2

    def test_track_detail_deterministic(self, app_client: TestClient) -> None:
        """GET /api/v1/tracks/suzuka 两次调用结果一致。"""
        r1 = app_client.get("/api/v1/tracks/suzuka").json()["data"]
        r2 = app_client.get("/api/v1/tracks/suzuka").json()["data"]
        # corners 列表应一致（不含随机锚点）
        assert r1["track"] == r2["track"]
        assert len(r1["corners"]) == len(r2["corners"])

    def test_feedback_then_list_consistent(self, app_client: TestClient) -> None:
        """提交反馈后查询列表应包含该反馈（读写一致）。"""
        resp = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "corner_number": 1,
            "symptom": "understeer", "strength": 4,
        })
        fb_id = resp.json()["data"]["id"]
        lst = app_client.get("/api/v1/feedback", params={"track_id": "suzuka"}).json()["data"]
        assert any(f["id"] == fb_id for f in lst)

    def test_envelope_code_semantics(self) -> None:
        """信封 code 语义：ok=0, fail>0, error<0。"""
        assert ok()["code"] == 0
        assert fail("x", code=1).detail["code"] > 0
        assert error("x", code=-1).detail["code"] < 0


# ===========================================================================
# 4. 静态分析 (static) — 状态码正确性、信封 code 语义、路由前缀
# ===========================================================================
class TestStatic:
    """静态分析：验证 HTTP 状态码、信封 code 语义与路由前缀约束。"""

    def test_health_status_200(self, app_client: TestClient) -> None:
        """GET /health 状态码 200。"""
        assert app_client.get("/api/v1/health").status_code == 200

    def test_tracks_status_200(self, app_client: TestClient) -> None:
        """GET /tracks 状态码 200。"""
        assert app_client.get("/api/v1/tracks").status_code == 200

    def test_track_detail_status_200_or_404(self, app_client: TestClient) -> None:
        """GET /tracks/{id} 存在则 200，不存在则 404。"""
        assert app_client.get("/api/v1/tracks/suzuka").status_code == 200
        assert app_client.get("/api/v1/tracks/ghost").status_code == 404

    def test_select_track_status_200_or_404(self, app_client: TestClient) -> None:
        """POST /tracks/current 存在则 200，不存在则 404。"""
        r1 = app_client.post("/api/v1/tracks/current", json={"track_id": "monza"})
        r2 = app_client.post("/api/v1/tracks/current", json={"track_id": "ghost"})
        assert r1.status_code == 200
        assert r2.status_code == 404

    def test_feedback_validation_status_422(self, app_client: TestClient) -> None:
        """POST /feedback 参数校验失败状态码 422。"""
        resp = app_client.post("/api/v1/feedback", json={"track_id": "suzuka"})
        assert resp.status_code == 422

    def test_suggest_no_feedback_status_400(self, app_client: TestClient) -> None:
        """POST /suggest 无反馈状态码 400（业务失败，非 500）。"""
        resp = app_client.post("/api/v1/suggest", json={"track_id": "suzuka"})
        assert resp.status_code == 400

    def test_import_setup_no_packet_status_409(self, app_client: TestClient) -> None:
        """POST /setup/import 无遥测包状态码 409（冲突，非 500）。"""
        resp = app_client.post("/api/v1/setup/import")
        assert resp.status_code == 409

    def test_unknown_route_status_404(self, app_client: TestClient) -> None:
        """不存在的路由状态码 404。"""
        assert app_client.get("/api/v1/ghost_route").status_code == 404

    def test_api_prefix_correct(self, app_client: TestClient) -> None:
        """所有 REST 端点应在 /api/v1 前缀下。"""
        # /api/v1/health 应 200，/health 应 404
        assert app_client.get("/api/v1/health").status_code == 200
        assert app_client.get("/health").status_code == 404

    def test_envelope_code_positive_for_business_error(
        self, app_client: TestClient,
    ) -> None:
        """业务失败（4xx）信封 code 应 > 0。"""
        resp = app_client.get("/api/v1/tracks/ghost")
        body = resp.json()
        assert body["code"] > 0, "4xx 业务失败 code 应为正数"

    def test_validation_error_envelope(self, app_client: TestClient) -> None:
        """pydantic 校验失败响应也是信封格式（code=4000）。"""
        resp = app_client.post("/api/v1/feedback", json={"track_id": "suzuka"})
        body = resp.json()
        assert "code" in body
        assert "message" in body
        assert body["code"] != 0

    def test_tracks_count_24(self, app_client: TestClient) -> None:
        """GET /tracks 返回恰好 24 条赛道（F1 2026 赛历）。"""
        data = app_client.get("/api/v1/tracks").json()["data"]
        assert len(data) == 24

    def test_track_anchor_in_unit_square(self, app_client: TestClient) -> None:
        """弯道锚点 anchor_x/anchor_y 应在 [0, 1] 区间。"""
        data = app_client.get("/api/v1/tracks/suzuka").json()["data"]
        for c in data["corners"]:
            assert 0.0 <= c["anchor_x"] <= 1.0, f"anchor_x 越界: {c['anchor_x']}"
            assert 0.0 <= c["anchor_y"] <= 1.0, f"anchor_y 越界: {c['anchor_y']}"


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 完整反馈→建议流程、WebSocket 连接
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：真实 API 启动，完整业务流程端到端验证。"""

    def test_full_feedback_suggest_flow(self, app_client: TestClient) -> None:
        """完整流程：选赛道 → 提交反馈 → 触发建议 → 查询建议。"""
        # 1. 选赛道
        r = app_client.post("/api/v1/tracks/current", json={"track_id": "suzuka"})
        assert r.status_code == 200

        # 2. 提交反馈
        r = app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "corner_number": 1,
            "symptom": "understeer", "strength": 4,
        })
        assert r.status_code == 200

        # 3. 触发建议
        r = app_client.post("/api/v1/suggest", json={"track_id": "suzuka"})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["suggestion_id"] >= 1
        assert "report" in body["data"]
        assert "parameters" in body["data"]["report"]

        # 4. 查询最新建议
        r = app_client.get("/api/v1/suggest/latest", params={"track_id": "suzuka"})
        assert r.status_code == 200
        assert r.json()["data"]["track_id"] == "suzuka"

    def test_multiple_feedbacks_then_suggest(self, app_client: TestClient) -> None:
        """多条反馈 → 建议生成（含不同弯道与症状）。"""
        app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "corner_number": 1,
            "symptom": "understeer", "strength": 4,
        })
        app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "corner_number": 2,
            "symptom": "oversteer", "strength": 3,
        })
        app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "symptom": "bottoming", "strength": 2,
        })

        r = app_client.post("/api/v1/suggest", json={"track_id": "suzuka"})
        assert r.status_code == 200
        report = r.json()["data"]["report"]
        assert "parameters" in report
        assert "summary" in report

    def test_iteration_history_after_suggest(self, app_client: TestClient) -> None:
        """触发建议后查询迭代历史应含 1 条记录。"""
        app_client.post("/api/v1/feedback", json={
            "track_id": "suzuka", "corner_number": 1,
            "symptom": "understeer", "strength": 3,
        })
        app_client.post("/api/v1/suggest", json={"track_id": "suzuka"})

        r = app_client.get("/api/v1/iteration/history", params={"track_id": "suzuka"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data) >= 1
        assert data[0]["track_id"] == "suzuka"

    def test_websocket_connect_and_select_track(
        self, app_client: TestClient,
    ) -> None:
        """WebSocket 连接 + 发送 select_track 消息 + 接收确认。"""
        with app_client.websocket_connect("/api/v1/ws") as ws:
            ws.send_text(json.dumps({"action": "select_track", "track_id": "suzuka"}))
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["event"] == "track_selected"
            assert data["payload"]["track_id"] == "suzuka"

    def test_websocket_invalid_json(self, app_client: TestClient) -> None:
        """WebSocket 发送非法 JSON 应收到 error 事件（不崩连接）。"""
        with app_client.websocket_connect("/api/v1/ws") as ws:
            ws.send_text("not a json string")
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["event"] == "error"

    def test_websocket_unknown_action(self, app_client: TestClient) -> None:
        """WebSocket 未知 action 应收到 error 事件。"""
        with app_client.websocket_connect("/api/v1/ws") as ws:
            ws.send_text(json.dumps({"action": "ghost_action"}))
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["event"] == "error"

    def test_websocket_request_suggestion_action(
        self, app_client: TestClient,
    ) -> None:
        """WebSocket request_suggestion action 应收到 info 事件。"""
        with app_client.websocket_connect("/api/v1/ws") as ws:
            ws.send_text(json.dumps({
                "action": "request_suggestion", "track_id": "suzuka",
            }))
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["event"] == "info"

    def test_root_endpoint(self, app_client: TestClient) -> None:
        """根路径 / 返回 API 信息。"""
        r = app_client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "name" in body
        assert "version" in body
        assert "docs" in body

    def test_openapi_schema_available(self, app_client: TestClient) -> None:
        """OpenAPI schema 端点可访问。"""
        r = app_client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert "/api/v1/health" in schema["paths"]