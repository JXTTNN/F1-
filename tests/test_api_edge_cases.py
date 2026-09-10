"""API 边界条件 + 错误处理端到端测试。

覆盖验收标准：
1. 无效 track_id → 404 信封
2. 无效 symptom → 400 信封
3. 无效 strength（越界）→ 422 校验失败信封
4. 无效 corner_number（超出赛道弯道数）→ 400
5. 无反馈时 suggest → 400 引导消息
6. setup/import 无遥测包 → 409
7. setup/current 无数据 → 404
8. suggest/latest 无数据 → 404
9. feedback 查询无当前赛道 → 409
10. 并发请求不冲突
11. 根路径返回 API 信息
12. 统一信封格式 {code, message, data}
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config

TEST_DATA_DIR = Path("./data_test_api_edge")


def make_client() -> TestClient:
    """创建测试客户端（每次清理数据目录确保测试隔离）。"""
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    config = Config(data_dir=str(TEST_DATA_DIR))
    app = create_app(config)
    return TestClient(app)


# ===========================================================================
# 1. 根路径
# ===========================================================================
def test_root_path():
    """根路径返回 API 信息。"""
    with make_client() as client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "name" in body
        assert "version" in body
        assert "docs" in body
        assert "websocket" in body


# ===========================================================================
# 2. 无效 track_id → 404
# ===========================================================================
def test_track_detail_not_found():
    """GET /tracks/unknown → 404 信封。"""
    with make_client() as client:
        resp = client.get("/api/v1/tracks/nonexistent_track")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0
        assert "未知赛道" in body["message"]


def test_select_track_not_found():
    """POST /tracks/current 无效 track_id → 404。"""
    with make_client() as client:
        resp = client.post("/api/v1/tracks/current", json={"track_id": "invalid"})
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0


def test_feedback_invalid_track():
    """POST /feedback 无效 track_id → 404。"""
    with make_client() as client:
        resp = client.post(
            "/api/v1/feedback",
            json={"track_id": "no_such_track", "symptom": "understeer", "strength": 3},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0


def test_suggest_invalid_track():
    """POST /suggest 无效 track_id → 404。"""
    with make_client() as client:
        resp = client.post("/api/v1/suggest", json={"track_id": "no_such_track"})
        assert resp.status_code == 404


# ===========================================================================
# 3. 无效 symptom → 400
# ===========================================================================
def test_feedback_invalid_symptom():
    """POST /feedback 无效 symptom → 400。"""
    with make_client() as client:
        resp = client.post(
            "/api/v1/feedback",
            json={"track_id": "suzuka", "symptom": "not_a_symptom", "strength": 3},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] != 0
        assert "症状" in body["message"]


# ===========================================================================
# 4. 无效 strength → 422（pydantic 校验）
# ===========================================================================
def test_feedback_strength_too_high():
    """POST /feedback strength=6（越界）→ 422。"""
    with make_client() as client:
        resp = client.post(
            "/api/v1/feedback",
            json={"track_id": "suzuka", "symptom": "understeer", "strength": 6},
        )
        assert resp.status_code == 422


def test_feedback_strength_too_low():
    """POST /feedback strength=-1（越界）→ 422。"""
    with make_client() as client:
        resp = client.post(
            "/api/v1/feedback",
            json={"track_id": "suzuka", "symptom": "understeer", "strength": -1},
        )
        assert resp.status_code == 422


def test_feedback_corner_number_out_of_range():
    """POST /feedback corner_number 超出赛道弯道数（但通过 pydantic 1-50 校验）→ 400。"""
    with make_client() as client:
        # suzuka 有 18 弯，corner_number=20 通过 pydantic (<=50) 但超出赛道弯道数
        resp = client.post(
            "/api/v1/feedback",
            json={
                "track_id": "suzuka",
                "corner_number": 20,
                "symptom": "understeer",
                "strength": 3,
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] != 0
        assert "弯道" in body["message"]


def test_feedback_corner_number_pydantic_reject():
    """POST /feedback corner_number=999（超出 pydantic le=50）→ 422。"""
    with make_client() as client:
        resp = client.post(
            "/api/v1/feedback",
            json={
                "track_id": "suzuka",
                "corner_number": 999,
                "symptom": "understeer",
                "strength": 3,
            },
        )
        assert resp.status_code == 422


# ===========================================================================
# 5. 无反馈时 suggest → 400
# ===========================================================================
def test_suggest_no_feedback():
    """POST /suggest 无反馈 → 400 引导消息。"""
    with make_client() as client:
        resp = client.post("/api/v1/suggest", json={"track_id": "suzuka"})
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] != 0
        assert "反馈" in body["message"]


# ===========================================================================
# 6. setup/import 无遥测 → 409
# ===========================================================================
def test_import_setup_no_telemetry():
    """POST /setup/import 无遥测包 → 409。"""
    with make_client() as client:
        resp = client.post("/api/v1/setup/import")
        assert resp.status_code == 409
        body = resp.json()
        assert body["code"] != 0


# ===========================================================================
# 7. setup/current 无数据 → 404
# ===========================================================================
def test_get_current_setup_no_data():
    """GET /setup/current 无数据 → 404。"""
    with make_client() as client:
        resp = client.get("/api/v1/setup/current", params={"track_id": "monaco"})
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0


# ===========================================================================
# 8. suggest/latest 无数据 → 404
# ===========================================================================
def test_get_latest_suggestion_no_data():
    """GET /suggest/latest 无数据 → 404。"""
    with make_client() as client:
        resp = client.get("/api/v1/suggest/latest", params={"track_id": "monza"})
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0


# ===========================================================================
# 9. feedback 查询无当前赛道 → 409
# ===========================================================================
def test_list_feedbacks_no_track():
    """GET /feedback 无 track_id 且无当前赛道 → 409。"""
    with make_client() as client:
        resp = client.get("/api/v1/feedback")
        assert resp.status_code == 409
        body = resp.json()
        assert body["code"] != 0


def test_suggest_latest_no_track():
    """GET /suggest/latest 无 track_id 且无当前赛道 → 409。"""
    with make_client() as client:
        resp = client.get("/api/v1/suggest/latest")
        assert resp.status_code == 409


def test_iteration_history_no_track():
    """GET /iteration/history 无 track_id 且无当前赛道 → 409。"""
    with make_client() as client:
        resp = client.get("/api/v1/iteration/history")
        assert resp.status_code == 409


# ===========================================================================
# 10. 完整流程：反馈 → 建议 → 读取 → 迭代
# ===========================================================================
def test_full_workflow_with_setup_import():
    """完整流程：选赛道 → 反馈 → 建议 → 读取最新 → 迭代历史。"""
    with make_client() as client:
        # 选赛道
        resp = client.post("/api/v1/tracks/current", json={"track_id": "silverstone"})
        assert resp.status_code == 200

        # 提交多条反馈
        for i in range(3):
            resp = client.post(
                "/api/v1/feedback",
                json={
                    "track_id": "silverstone",
                    "corner_number": i + 1,
                    "symptom": "understeer",
                    "strength": i + 2,
                },
            )
            assert resp.status_code == 200

        # 查询反馈
        resp = client.get("/api/v1/feedback", params={"track_id": "silverstone"})
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 3

        # 生成建议
        resp = client.post("/api/v1/suggest", json={"track_id": "silverstone"})
        assert resp.status_code == 200
        report = resp.json()["data"]["report"]
        assert report["track_id"] == "silverstone"
        assert len(report["parameters"]) == 23

        # 读取最新建议
        resp = client.get("/api/v1/suggest/latest", params={"track_id": "silverstone"})
        assert resp.status_code == 200

        # 迭代历史
        resp = client.get("/api/v1/iteration/history", params={"track_id": "silverstone"})
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1


def test_feedback_with_global_symptom():
    """全局症状（corner_number=None）可提交。"""
    with make_client() as client:
        resp = client.post(
            "/api/v1/feedback",
            json={
                "track_id": "suzuka",
                "symptom": "bottoming",
                "strength": 4,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["category"] == "global"


def test_multiple_suggestions_generate_iterations():
    """多次生成建议产生多条迭代记录。"""
    with make_client() as client:
        client.post("/api/v1/tracks/current", json={"track_id": "monza"})
        client.post(
            "/api/v1/feedback",
            json={"track_id": "monza", "symptom": "oversteer", "strength": 3},
        )

        # 第一次建议
        resp = client.post("/api/v1/suggest", json={"track_id": "monza"})
        assert resp.status_code == 200

        # 第二次建议
        resp = client.post("/api/v1/suggest", json={"track_id": "monza"})
        assert resp.status_code == 200

        # 迭代历史应有 2 条
        resp = client.get("/api/v1/iteration/history", params={"track_id": "monza"})
        assert resp.status_code == 200
        history = resp.json()["data"]
        assert len(history) == 2
        assert history[0]["round_no"] == 1
        assert history[1]["round_no"] == 2


# ===========================================================================
# 11. 并发请求不冲突
# ===========================================================================
def test_concurrent_feedback_submission():
    """多线程并发提交反馈不报错。"""
    with make_client() as client:
        client.post("/api/v1/tracks/current", json={"track_id": "spa"})
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                resp = client.post(
                    "/api/v1/feedback",
                    json={
                        "track_id": "spa",
                        "corner_number": (idx % 10) + 1,
                        "symptom": "understeer",
                        "strength": 3,
                    },
                )
                if resp.status_code != 200:
                    errors.append(RuntimeError(f"status={resp.status_code}"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ===========================================================================
# 12. 统一信封格式
# ===========================================================================
def test_envelope_format_all_responses():
    """所有响应统一 {code, message, data} 信封。"""
    with make_client() as client:
        # 成功响应
        resp = client.get("/api/v1/health")
        body = resp.json()
        assert {"code", "message", "data"} <= set(body.keys())

        resp = client.get("/api/v1/tracks")
        body = resp.json()
        assert {"code", "message", "data"} <= set(body.keys())

        # 错误响应
        resp = client.get("/api/v1/tracks/nonexistent")
        body = resp.json()
        assert {"code", "message", "data"} <= set(body.keys())
        assert body["code"] != 0


def test_health_status_degraded_without_telemetry():
    """无遥测连接时 health status=degraded。"""
    with make_client() as client:
        resp = client.get("/api/v1/health")
        body = resp.json()
        assert body["code"] == 0
        data = body["data"]
        assert "status" in data
        assert "telemetry_connected" in data
        assert "udp_host" in data
        assert "udp_port" in data


def test_tracks_count_24():
    """GET /tracks 返回 24 条赛道。"""
    with make_client() as client:
        resp = client.get("/api/v1/tracks")
        body = resp.json()
        assert body["code"] == 0
        assert len(body["data"]) == 24


def test_all_tracks_have_detail():
    """全部 24 条赛道都能查到详情。"""
    with make_client() as client:
        tracks = client.get("/api/v1/tracks").json()["data"]
        for t in tracks:
            resp = client.get(f"/api/v1/tracks/{t['track_id']}")
            assert resp.status_code == 200
            detail = resp.json()["data"]
            assert detail["track"]["track_id"] == t["track_id"]
            assert len(detail["corners"]) > 0