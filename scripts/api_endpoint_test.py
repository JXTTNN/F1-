"""API端点完整性测试脚本（修正版）：验证所有REST API端点。

按照实际API入参格式发送请求，使用 with TestClient 触发 lifespan 初始化。
"""

import sys

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.domain.setup import ALL_SETUP_FIELDS

app = create_app()

passed = 0
failed = 0
errors = []

TRACK_ID = "suzuka"

print("=" * 80)
print("API端点完整性测试（修正版）")
print("=" * 80)

with TestClient(app) as client:

    # =========================================================================
    # 1. GET /api/v1/health
    # =========================================================================
    print("\n--- 1. GET /api/v1/health ---")
    try:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200, f"状态码: {resp.status_code}"
        body = resp.json()
        assert body["code"] == 0, f"code: {body['code']}"
        assert "status" in body["data"], "缺少status字段"
        print(f"  ✅ health 返回 status={body['data']['status']}")
        passed += 1
    except Exception as e:
        print(f"  ❌ health 失败: {e}")
        failed += 1
        errors.append(f"GET /health: {e}")

    # =========================================================================
    # 2. GET /api/v1/setup/fields — 验证返回20项参数（与 ALL_SETUP_FIELDS 一致）
    # =========================================================================
    print("\n--- 2. GET /api/v1/setup/fields ---")
    try:
        resp = client.get("/api/v1/setup/fields")
        assert resp.status_code == 200, f"状态码: {resp.status_code}"
        body = resp.json()
        assert body["code"] == 0, f"code: {body['code']}"
        data = body["data"]
        assert len(data) == 20, f"返回了{len(data)}项参数，应为20项"
        for item in data:
            assert "name" in item
            assert "min" in item
            assert "max" in item
            assert "step" in item
            assert "default" in item
        print(f"  ✅ setup/fields 返回 {len(data)} 项参数，字段完整")
        passed += 1
    except Exception as e:
        print(f"  ❌ setup/fields 失败: {e}")
        failed += 1
        errors.append(f"GET /setup/fields: {e}")

    # =========================================================================
    # 3. POST /api/v1/setup/manual — 验证接受20项参数
    # =========================================================================
    print("\n--- 3. POST /api/v1/setup/manual ---")
    try:
        params = {f.name: f.default for f in ALL_SETUP_FIELDS}
        resp = client.post("/api/v1/setup/manual", json={"track_id": TRACK_ID, "params": params})
        assert resp.status_code == 200, f"状态码: {resp.status_code}, body: {resp.text}"
        body = resp.json()
        assert body["code"] == 0, f"code: {body['code']}"
        assert body["data"]["track_id"] == TRACK_ID
        assert len(body["data"]["params"]) == 20
        print("  ✅ setup/manual 接受20项参数成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ setup/manual 失败: {e}")
        failed += 1
        errors.append(f"POST /setup/manual: {e}")

    # =========================================================================
    # 4. POST /api/v1/feedback — 验证多症状提交
    # =========================================================================
    print("\n--- 4. POST /api/v1/feedback ---")
    try:
        feedbacks = [
            {"corner_number": 1, "symptom": "understeer", "strength": 3},
            {"corner_number": 5, "symptom": "oversteer", "strength": 2},
            {"symptom": "tyre_wear", "strength": 3},
        ]
        resp = client.post("/api/v1/feedback", json={
            "track_id": TRACK_ID,
            "feedbacks": feedbacks,
        })
        assert resp.status_code == 200, f"状态码: {resp.status_code}, body: {resp.text}"
        body = resp.json()
        assert body["code"] == 0, f"code: {body['code']}"
        print(f"  ✅ feedback 批量提交{len(feedbacks)}条多症状反馈成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ feedback 失败: {e}")
        failed += 1
        errors.append(f"POST /feedback: {e}")

    try:
        resp = client.post("/api/v1/feedback", json={
            "track_id": TRACK_ID,
            "symptom": "understeer",
            "corner_number": 3,
            "strength": 2,
        })
        assert resp.status_code == 200, f"单条状态码: {resp.status_code}, body: {resp.text}"
        body = resp.json()
        assert body["code"] == 0
        print("  ✅ feedback 单条格式提交成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ feedback 单条格式失败: {e}")
        failed += 1
        errors.append(f"POST /feedback (single): {e}")

    # =========================================================================
    # 5. POST /api/v1/telemetry/simulate
    # =========================================================================
    print("\n--- 5. POST /api/v1/telemetry/simulate ---")
    try:
        resp = client.post("/api/v1/telemetry/simulate", json={
            "track_id": TRACK_ID, "action": "start",
        })
        assert resp.status_code == 200, f"状态码: {resp.status_code}, body: {resp.text}"
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["simulating"] is True
        print("  ✅ telemetry/simulate start 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ telemetry/simulate start 失败: {e}")
        failed += 1
        errors.append(f"POST /telemetry/simulate start: {e}")

    try:
        resp = client.post("/api/v1/telemetry/simulate", json={
            "track_id": TRACK_ID, "action": "stop",
        })
        assert resp.status_code == 200, f"状态码: {resp.status_code}, body: {resp.text}"
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["simulating"] is False
        print("  ✅ telemetry/simulate stop 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ telemetry/simulate stop 失败: {e}")
        failed += 1
        errors.append(f"POST /telemetry/simulate stop: {e}")

    # =========================================================================
    # 6. POST /api/v1/telemetry/record/toggle
    # =========================================================================
    print("\n--- 6. POST /api/v1/telemetry/record/toggle ---")
    try:
        resp = client.post("/api/v1/telemetry/record/toggle", json={"action": "start"})
        assert resp.status_code == 200, f"状态码: {resp.status_code}, body: {resp.text}"
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["recording"] is True
        session_id = body["data"]["session_id"]
        print(f"  ✅ telemetry/record/toggle start 成功 (session={session_id})")
        passed += 1
    except Exception as e:
        print(f"  ❌ telemetry/record/toggle start 失败: {e}")
        failed += 1
        errors.append(f"POST /telemetry/record/toggle start: {e}")

    try:
        resp = client.post("/api/v1/telemetry/record/toggle", json={"action": "stop"})
        assert resp.status_code == 200, f"状态码: {resp.status_code}, body: {resp.text}"
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["recording"] is False
        print("  ✅ telemetry/record/toggle stop 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ telemetry/record/toggle stop 失败: {e}")
        failed += 1
        errors.append(f"POST /telemetry/record/toggle stop: {e}")

    # =========================================================================
    # 7. GET /api/v1/telemetry/recordings
    # =========================================================================
    print("\n--- 7. GET /api/v1/telemetry/recordings ---")
    try:
        resp = client.get("/api/v1/telemetry/recordings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        recordings = body["data"]
        print(f"  ✅ telemetry/recordings 成功，返回 {len(recordings)} 个录制会话")
        passed += 1
    except Exception as e:
        print(f"  ❌ telemetry/recordings 失败: {e}")
        failed += 1
        errors.append(f"GET /telemetry/recordings: {e}")

    # =========================================================================
    # 8. POST /api/v1/suggest — 验证rule/nn/hybrid模式
    # =========================================================================
    print("\n--- 8. POST /api/v1/suggest (rule/nn/hybrid) ---")
    for mode in ["rule", "nn", "hybrid"]:
        try:
            # 显式关闭"生成后清除反馈"（2026-09-19 起的默认行为）：
            # 本检查点只验证三种模型类型都能出报告，需同一份反馈连续可用；
            # 清除行为本身由 tests/test_feedback_lifecycle.py 专项覆盖。
            resp = client.post("/api/v1/suggest", json={
                "track_id": TRACK_ID, "model_type": mode,
                "clear_feedback_after_suggest": False,
            })
            assert resp.status_code == 200, f"[{mode}] 状态码: {resp.status_code}, body: {resp.text}"
            body = resp.json()
            assert body["code"] == 0, f"[{mode}] code: {body['code']}"
            print(f"  ✅ suggest model_type={mode} 成功")
            passed += 1
        except Exception as e:
            print(f"  ❌ suggest model_type={mode} 失败: {e}")
            failed += 1
            errors.append(f"POST /suggest model_type={mode}: {e}")

    # =========================================================================
    # 9. 其他端点验证
    # =========================================================================
    print("\n--- 9. 其他端点验证 ---")

    # GET /api/v1/tracks
    try:
        resp = client.get("/api/v1/tracks")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        tracks = body["data"]
        assert len(tracks) == 24, f"赛道数={len(tracks)}, 应为24"
        print("  ✅ GET /tracks 返回24条赛道")
        passed += 1
    except Exception as e:
        print(f"  ❌ GET /tracks 失败: {e}")
        failed += 1
        errors.append(f"GET /tracks: {e}")

    # GET /api/v1/tracks/{track_id}
    try:
        resp = client.get(f"/api/v1/tracks/{TRACK_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["track"]["track_id"] == TRACK_ID
        print(f"  ✅ GET /tracks/{TRACK_ID} 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ GET /tracks/{TRACK_ID} 失败: {e}")
        failed += 1
        errors.append(f"GET /tracks/{TRACK_ID}: {e}")

    # POST /api/v1/tracks/current
    try:
        resp = client.post("/api/v1/tracks/current", json={"track_id": TRACK_ID})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["current_track_id"] == TRACK_ID
        print("  ✅ POST /tracks/current 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ POST /tracks/current 失败: {e}")
        failed += 1
        errors.append(f"POST /tracks/current: {e}")

    # GET /api/v1/setup/current
    try:
        resp = client.get(f"/api/v1/setup/current?track_id={TRACK_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert len(body["data"]["params"]) == 20
        print("  ✅ GET /setup/current 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ GET /setup/current 失败: {e}")
        failed += 1
        errors.append(f"GET /setup/current: {e}")

    # GET /api/v1/feedback
    try:
        resp = client.get(f"/api/v1/feedback?track_id={TRACK_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        print(f"  ✅ GET /feedback 成功，返回 {len(body['data'])} 条反馈")
        passed += 1
    except Exception as e:
        print(f"  ❌ GET /feedback 失败: {e}")
        failed += 1
        errors.append(f"GET /feedback: {e}")

    # GET /api/v1/suggest/latest
    try:
        resp = client.get(f"/api/v1/suggest/latest?track_id={TRACK_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        print("  ✅ GET /suggest/latest 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ GET /suggest/latest 失败: {e}")
        failed += 1
        errors.append(f"GET /suggest/latest: {e}")

    # GET /api/v1/iteration/history
    try:
        resp = client.get(f"/api/v1/iteration/history?track_id={TRACK_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        print("  ✅ GET /iteration/history 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ GET /iteration/history 失败: {e}")
        failed += 1
        errors.append(f"GET /iteration/history: {e}")

    # POST /api/v1/setup/import (无遥测时应409)
    try:
        resp = client.post("/api/v1/setup/import")
        assert resp.status_code in (200, 409), f"状态码: {resp.status_code}, body: {resp.text}"
        print(f"  ✅ POST /setup/import 返回 {resp.status_code}（预期200或409）")
        passed += 1
    except Exception as e:
        print(f"  ❌ POST /setup/import 失败: {e}")
        failed += 1
        errors.append(f"POST /setup/import: {e}")

    # GET /api/v1/telemetry/replay/status
    try:
        resp = client.get("/api/v1/telemetry/replay/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        print("  ✅ GET /telemetry/replay/status 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ GET /telemetry/replay/status 失败: {e}")
        failed += 1
        errors.append(f"GET /telemetry/replay/status: {e}")

    # POST /api/v1/telemetry/replay/stop
    try:
        resp = client.post("/api/v1/telemetry/replay/stop")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        print("  ✅ POST /telemetry/replay/stop 成功")
        passed += 1
    except Exception as e:
        print(f"  ❌ POST /telemetry/replay/stop 失败: {e}")
        failed += 1
        errors.append(f"POST /telemetry/replay/stop: {e}")

print("\n" + "=" * 80)
print(f"API端点完整性测试结果：{passed} passed, {failed} failed")
print("=" * 80)

if errors:
    print("\n失败详情：")
    for err in errors:
        print(f"  - {err}")

sys.exit(0 if failed == 0 else 1)
