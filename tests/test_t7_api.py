"""T7 API 层端到端冒烟测试 —— 验证 REST 端点可访问 + 信封格式 + 核心流程。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config


def make_client() -> TestClient:
    """创建测试客户端（使用临时内存配置）。"""
    config = Config(data_dir="./data_test_t7")
    app = create_app(config)
    return TestClient(app)


def test_health():
    """GET /api/v1/health 返回信封格式。"""
    with make_client() as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "code" in body
        assert "message" in body
        assert "data" in body
        assert body["code"] == 0
        assert "status" in body["data"]
        assert "telemetry_connected" in body["data"]
        print("✅ health OK:", body)


def test_tracks():
    """GET /api/v1/tracks 返回 24 条赛道。"""
    with make_client() as client:
        resp = client.get("/api/v1/tracks")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert len(body["data"]) == 24
        print(f"✅ tracks OK: {len(body['data'])} 条赛道")


def test_track_detail():
    """GET /api/v1/tracks/suzuka 返回赛道详情 + 弯道锚点。"""
    with make_client() as client:
        resp = client.get("/api/v1/tracks/suzuka")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["track"]["track_id"] == "suzuka"
        assert len(body["data"]["corners"]) > 0
        # 弯道锚点在 0~1 区间
        for c in body["data"]["corners"]:
            assert 0.0 <= c["anchor_x"] <= 1.0
            assert 0.0 <= c["anchor_y"] <= 1.0
        print(f"✅ track detail OK: suzuka, {len(body['data']['corners'])} 弯")


def test_track_not_found():
    """GET /api/v1/tracks/unknown 返回 404 信封。"""
    with make_client() as client:
        resp = client.get("/api/v1/tracks/unknown_track")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0
        print("✅ track 404 OK:", body["message"])


def test_select_track():
    """POST /api/v1/tracks/current 手动选赛道。"""
    with make_client() as client:
        resp = client.post("/api/v1/tracks/current", json={"track_id": "monza"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["current_track_id"] == "monza"
        print("✅ select track OK:", body)


def test_feedback_and_suggest():
    """POST /feedback → POST /suggest 完整流程。"""
    with make_client() as client:
        # 选赛道
        client.post("/api/v1/tracks/current", json={"track_id": "suzuka"})

        # 无反馈时请求建议 → 400
        resp = client.post("/api/v1/suggest", json={"track_id": "suzuka"})
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] != 0
        assert "反馈" in body["message"]
        print("✅ suggest no feedback → 400 OK:", body["message"])

        # 提交反馈
        resp = client.post(
            "/api/v1/feedback",
            json={
                "track_id": "suzuka",
                "corner_number": 1,
                "symptom": "understeer",
                "strength": 4,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["symptom"] == "understeer"
        print("✅ feedback submit OK:", body["data"])

        # 查询反馈
        resp = client.get("/api/v1/feedback", params={"track_id": "suzuka"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 1
        print(f"✅ feedback list OK: {len(body['data'])} 条")

        # 生成建议
        resp = client.post("/api/v1/suggest", json={"track_id": "suzuka"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        report = body["data"]["report"]
        assert report["track_id"] == "suzuka"
        assert "generated_at" in report
        assert "parameters" in report
        assert "summary" in report
        assert len(report["parameters"]) == 23
        # 每参数含必要字段
        for p in report["parameters"]:
            assert "param" in p
            assert "current" in p
            assert "setup_delta" in p
            assert "linkages" in p
            assert "linked_notes" in p
            assert "source" in p
            assert "confidence" in p
        print(f"✅ suggest OK: {len(report['parameters'])} 参数, confidence={report['confidence']}")
        print(f"   summary: {report['summary']}")

        # 读取最新建议
        resp = client.get("/api/v1/suggest/latest", params={"track_id": "suzuka"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        print("✅ suggest latest OK")

        # 迭代历史
        resp = client.get("/api/v1/iteration/history", params={"track_id": "suzuka"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        print(f"✅ iteration history OK: {len(body['data'])} 轮")


def test_setup_current_no_data():
    """GET /setup/current 无数据时返回 404。"""
    with make_client() as client:
        resp = client.get("/api/v1/setup/current", params={"track_id": "monaco"})
        assert resp.status_code == 404
        print("✅ setup current 404 OK")


def test_invalid_symptom():
    """POST /feedback 无效症状 → 400。"""
    with make_client() as client:
        resp = client.post(
            "/api/v1/feedback",
            json={
                "track_id": "suzuka",
                "symptom": "invalid_symptom",
                "strength": 3,
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] != 0
        print("✅ invalid symptom 400 OK:", body["message"])


def test_envelope_format():
    """所有响应统一 {code, message, data} 信封。"""
    with make_client() as client:
        resp = client.get("/api/v1/health")
        body = resp.json()
        assert set(body.keys()) >= {"code", "message", "data"}
        print("✅ envelope format OK")


if __name__ == "__main__":
    test_health()
    test_tracks()
    test_track_detail()
    test_track_not_found()
    test_select_track()
    test_feedback_and_suggest()
    test_setup_current_no_data()
    test_invalid_symptom()
    test_envelope_format()
    print("\n🎉 所有 T7 端到端测试通过！")