"""方式 2：端到端业务闭环测试 —— 用 TestClient 真实启动 API 验证完整流程。

完整业务流程（6 步闭环）：
    1. 选赛道        → GET /api/v1/tracks → 选择一个赛道
    2. 导入调教      → POST /api/v1/setup/import → 导入 Car Setups 包
    3. 提交反馈      → POST /api/v1/feedback → 对多个弯道提交不同症状
    4. 生成建议      → POST /api/v1/suggest → 获取调教建议
    5. 查看迭代      → GET /api/v1/iteration/history → 验证迭代记录
    6. 多轮迭代      → 重复 3-4 验证建议收敛性

每一步验证：响应格式（统一 {code, message, data} 信封）、状态码、数据一致性。
使用 fastapi.testclient.TestClient 真实启动应用，不 mock 任何核心模块。
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config
from setup_tuner.telemetry.packets import HEADER_FORMAT

# ===========================================================================
# 辅助
# ===========================================================================
TEST_DATA_DIR = Path("./data_test_deep_e2e")


def make_client() -> TestClient:
    """创建测试客户端（每次清理数据目录确保测试隔离）。"""
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    config = Config(data_dir=str(TEST_DATA_DIR), udp_port=20799)  # 避开默认端口
    app = create_app(config)
    return TestClient(app)


def _build_header(packet_id: int, player_car_index: int = 0) -> bytes:
    """构造 29 字节包头。"""
    return struct.pack(
        HEADER_FORMAT,
        2026, 26, 1, 0, 1, packet_id,
        0x1234_5678_9ABC_DEF0, 12.5, 100, 200,
        player_car_index, 255,
    )


def _build_session_body(track_id: int = 2) -> bytes:
    """构造 Session 包体。"""
    prefix_core = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        0, 25, 22, 58, 5807, 0, track_id, 1, 1800, 3600, 60, 0, 0, 0, 0, 21,
    )
    marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
    tail = struct.pack("<BBB", 0, 0, 0)
    return prefix_core + marshal_zones + tail


def _build_car_setups_body() -> bytes:
    """构造 CarSetups 单车包体。"""
    return struct.pack(
        "<BBBBffffBBBBBBBBBffffBf",
        6, 4, 50, 50,
        -2.5, -2.5, 0.25, 0.25,
        1, 1, 1, 1, 1, 1, 80, 65, 50,
        25.5, 25.5, 25.5, 25.5,
        50, 100.0,
    )


def _inject_telemetry(client: TestClient, track_id_udp: int = 2) -> None:
    """向 TelemetryStream 注入遥测帧（模拟 UDP 收包）。

    通过 app.state.telemetry_stream 直接写入，绕过 UDP 监听
    （TestClient 不便真实收 UDP）。
    """
    import json

    # 构造 Packet 1 (Session) + Packet 5 (CarSetups) 解析结果
    session_data = _build_header(packet_id=1) + _build_session_body(track_id_udp)
    setups_data = _build_header(packet_id=5) + _build_car_setups_body()

    from setup_tuner.telemetry.packets import parse_packet

    session_parsed = parse_packet(session_data)
    setups_parsed = parse_packet(setups_data)
    assert session_parsed is not None
    assert setups_parsed is not None

    # 直接写入 TelemetryStream（与 listener handler 等价）
    stream = client.app.state.telemetry_stream  # type: ignore[attr-defined]
    stream.update(1, session_parsed)
    stream.update(5, setups_parsed)
    # 触发 json 导入以避免未使用警告
    _ = json


def _assert_envelope(body: dict, *, code: int = 0) -> dict:
    """断言响应符合统一信封格式并返回 data。"""
    assert "code" in body, f"信封缺 code 字段: {body}"
    assert "message" in body, f"信封缺 message 字段: {body}"
    assert "data" in body, f"信封缺 data 字段: {body}"
    assert body["code"] == code, (
        f"信封 code 期望 {code} 实际 {body['code']}: {body['message']}"
    )
    return body["data"]


# ===========================================================================
# 1. 选赛道
# ===========================================================================
class TestSelectTrack:
    """端到端：选赛道流程。"""

    def test_list_tracks_returns_24(self) -> None:
        """GET /api/v1/tracks 返回 24 条赛道，信封格式正确。"""
        with make_client() as client:
            resp = client.get("/api/v1/tracks")
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert len(data) == 24
            # 每条赛道含必要字段
            for t in data:
                assert "track_id" in t
                assert "official_name" in t
                assert "corners" in t
                assert t["corners"] > 0

    def test_select_track_success(self) -> None:
        """POST /api/v1/tracks/current 选择合法赛道 → 200。"""
        with make_client() as client:
            resp = client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert data["current_track_id"] == "suzuka"
            assert data["source"] == "manual"

    def test_select_unknown_track_404(self) -> None:
        """POST /api/v1/tracks/current 选择未知赛道 → 404 信封。"""
        with make_client() as client:
            resp = client.post(
                "/api/v1/tracks/current", json={"track_id": "nonexistent"},
            )
            assert resp.status_code == 404
            body = resp.json()
            assert body["code"] != 0

    def test_track_detail_with_corners(self) -> None:
        """GET /api/v1/tracks/{id} 返回赛道详情 + 弯道锚点。"""
        with make_client() as client:
            resp = client.get("/api/v1/tracks/monza")
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert data["track"]["track_id"] == "monza"
            assert len(data["corners"]) > 0
            for c in data["corners"]:
                assert 0.0 <= c["anchor_x"] <= 1.0
                assert 0.0 <= c["anchor_y"] <= 1.0


# ===========================================================================
# 2. 导入调教
# ===========================================================================
class TestImportSetup:
    """端到端：导入调教快照流程。"""

    def test_import_without_telemetry_409(self) -> None:
        """无遥测帧时 POST /setup/import → 409 引导消息。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            resp = client.post("/api/v1/setup/import")
            assert resp.status_code == 409
            body = resp.json()
            assert body["code"] != 0
            assert "Car Setups" in body["message"] or "遥测" in body["message"]

    def test_import_with_telemetry_success(self) -> None:
        """有遥测帧时 POST /setup/import → 200，含 23 参数。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            _inject_telemetry(client, track_id_udp=2)

            resp = client.post("/api/v1/setup/import")
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert data["track_id"] == "suzuka"
            assert data["setup_id"] > 0
            assert len(data["params"]) == 23

    def test_get_current_setup_after_import(self) -> None:
        """导入后 GET /setup/current 应返回刚导入的快照。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            _inject_telemetry(client)
            client.post("/api/v1/setup/import")

            resp = client.get(
                "/api/v1/setup/current", params={"track_id": "suzuka"},
            )
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert data["track_id"] == "suzuka"
            assert len(data["params"]) == 23


# ===========================================================================
# 3. 提交反馈
# ===========================================================================
class TestSubmitFeedback:
    """端到端：提交反馈流程。"""

    def test_submit_single_feedback(self) -> None:
        """POST /api/v1/feedback 提交单条反馈 → 200。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
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
            data = _assert_envelope(resp.json())
            assert data["symptom"] == "understeer"
            assert data["strength"] == 4
            assert data["corner_number"] == 1
            assert data["id"] > 0

    def test_submit_multiple_feedbacks_different_corners(self) -> None:
        """对多个弯道提交不同症状 → 全部 200 且可检索。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )

            feedbacks = [
                {"track_id": "suzuka", "corner_number": 1, "symptom": "understeer", "strength": 4},
                {"track_id": "suzuka", "corner_number": 5, "symptom": "brake_long", "strength": 3},
                {"track_id": "suzuka", "corner_number": 10, "symptom": "oversteer", "strength": 3},
                {"track_id": "suzuka", "corner_number": None, "symptom": "bottoming", "strength": 2},
            ]
            for fb in feedbacks:
                resp = client.post("/api/v1/feedback", json=fb)
                assert resp.status_code == 200, f"提交反馈失败: {fb}"

            # 检索全部反馈
            resp = client.get(
                "/api/v1/feedback", params={"track_id": "suzuka"},
            )
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert len(data) == 4

    def test_submit_invalid_symptom_400(self) -> None:
        """提交无效症状 → 400 信封。"""
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
            assert resp.json()["code"] != 0

    def test_submit_invalid_strength_422(self) -> None:
        """提交越界强度 → 422 校验失败信封。"""
        with make_client() as client:
            resp = client.post(
                "/api/v1/feedback",
                json={
                    "track_id": "suzuka",
                    "symptom": "understeer",
                    "strength": 99,
                },
            )
            assert resp.status_code == 422
            assert resp.json()["code"] != 0


# ===========================================================================
# 4. 生成建议
# ===========================================================================
class TestGenerateSuggestion:
    """端到端：生成建议流程。"""

    def test_suggest_without_feedback_400(self) -> None:
        """无反馈时 POST /suggest → 400 引导消息。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            resp = client.post("/api/v1/suggest", json={"track_id": "suzuka"})
            assert resp.status_code == 400
            body = resp.json()
            assert body["code"] != 0
            assert "反馈" in body["message"]

    def test_suggest_with_feedback_success(self) -> None:
        """有反馈时 POST /suggest → 200，报告含 23 参数。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": "suzuka", "corner_number": 1,
                    "symptom": "understeer", "strength": 4,
                },
            )

            resp = client.post("/api/v1/suggest", json={"track_id": "suzuka"})
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert data["track_id"] == "suzuka"
            assert data["suggestion_id"] > 0

            report = data["report"]
            assert report["track_id"] == "suzuka"
            assert "generated_at" in report
            assert len(report["parameters"]) == 23
            assert report["confidence"] in {"high", "medium", "low"}

            # 每参数含必要字段
            for p in report["parameters"]:
                assert "param" in p
                assert "current" in p
                assert "setup_delta" in p
                assert "linkages" in p
                assert "source" in p
                assert "confidence" in p

    def test_get_latest_suggestion(self) -> None:
        """生成建议后 GET /suggest/latest 应返回同一份报告。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": "suzuka", "corner_number": 1,
                    "symptom": "understeer", "strength": 3,
                },
            )
            resp1 = client.post("/api/v1/suggest", json={"track_id": "suzuka"})
            assert resp1.status_code == 200
            report1 = resp1.json()["data"]["report"]

            resp2 = client.get(
                "/api/v1/suggest/latest", params={"track_id": "suzuka"},
            )
            assert resp2.status_code == 200
            report2 = resp2.json()["data"]["report"]

            # 两份报告的 setup_delta 应一致（同一份建议）
            assert report1["setup_delta"] == report2["setup_delta"]


# ===========================================================================
# 5. 查看迭代历史
# ===========================================================================
class TestIterationHistory:
    """端到端：迭代历史流程。"""

    def test_iteration_history_after_suggest(self) -> None:
        """生成建议后 GET /iteration/history 应有 1 轮记录。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": "suzuka", "corner_number": 1,
                    "symptom": "understeer", "strength": 3,
                },
            )
            client.post("/api/v1/suggest", json={"track_id": "suzuka"})

            resp = client.get(
                "/api/v1/iteration/history", params={"track_id": "suzuka"},
            )
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert len(data) >= 1
            assert data[0]["track_id"] == "suzuka"
            assert data[0]["round_no"] == 1
            assert data[0]["suggestion_id"] is not None


# ===========================================================================
# 6. 多轮迭代收敛性
# ===========================================================================
class TestMultiRoundConvergence:
    """端到端：多轮迭代验证建议收敛性。

    收敛性定义：相同反馈连续生成建议，setup_delta 应保持稳定（确定性引擎）。
    """

    def test_repeated_suggest_is_deterministic(self) -> None:
        """相同反馈连续 3 轮生成建议，setup_delta 应完全一致。

        引擎为纯确定性函数，无反馈变化时建议应稳定。
        """
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "suzuka"},
            )
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": "suzuka", "corner_number": 1,
                    "symptom": "understeer", "strength": 3,
                },
            )

            deltas: list[dict] = []
            for _ in range(3):
                resp = client.post("/api/v1/suggest", json={"track_id": "suzuka"})
                assert resp.status_code == 200
                deltas.append(resp.json()["data"]["report"]["setup_delta"])

            # 3 轮 setup_delta 应完全一致（确定性）
            assert deltas[0] == deltas[1] == deltas[2]

    def test_multi_round_iteration_history_grows(self) -> None:
        """多轮生成建议后迭代历史应递增。"""
        with make_client() as client:
            client.post(
                "/api/v1/tracks/current", json={"track_id": "monza"},
            )
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": "monza", "corner_number": 1,
                    "symptom": "oversteer", "strength": 4,
                },
            )

            for _ in range(3):
                resp = client.post("/api/v1/suggest", json={"track_id": "monza"})
                assert resp.status_code == 200

            resp = client.get(
                "/api/v1/iteration/history", params={"track_id": "monza"},
            )
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert len(data) == 3
            # round_no 应为 1, 2, 3
            round_nos = [it["round_no"] for it in data]
            assert round_nos == [1, 2, 3]

    def test_increasing_strength_increases_delta_magnitude(self) -> None:
        """反馈强度增加时，建议的 |setup_delta| 应不减（单调性）。

        链路：strength=1 → suggest → strength=5 → suggest → 比较 |delta|
        """
        with make_client() as client:
            track_id = "silverstone"
            client.post(
                "/api/v1/tracks/current", json={"track_id": track_id},
            )

            # 第 1 轮：强度 1
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": track_id, "corner_number": 1,
                    "symptom": "understeer", "strength": 1,
                },
            )
            resp1 = client.post("/api/v1/suggest", json={"track_id": track_id})
            assert resp1.status_code == 200
            delta_weak = resp1.json()["data"]["report"]["setup_delta"]

            # 第 2 轮：追加强度 5 的反馈
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": track_id, "corner_number": 2,
                    "symptom": "understeer", "strength": 5,
                },
            )
            resp2 = client.post("/api/v1/suggest", json={"track_id": track_id})
            assert resp2.status_code == 200
            delta_strong = resp2.json()["data"]["report"]["setup_delta"]

            # 强度更大的反馈应使总调整幅度更大（至少有一个参数 |delta| 更大）
            weak_abs = sum(abs(v) for v in delta_weak.values())
            strong_abs = sum(abs(v) for v in delta_strong.values())
            assert strong_abs >= weak_abs - 1e-9


# ===========================================================================
# 7. 完整闭环冒烟（6 步串起来）
# ===========================================================================
class TestFullClosedLoop:
    """完整 6 步业务闭环冒烟测试。"""

    def test_full_6_step_loop(self) -> None:
        """完整跑一遍 6 步闭环：选道 → 导入 → 反馈 → 建议 → 迭代 → 收敛。"""
        with make_client() as client:
            # ① 选赛道
            resp = client.get("/api/v1/tracks")
            assert resp.status_code == 200
            tracks = _assert_envelope(resp.json())
            assert len(tracks) == 24
            track_id = "suzuka"

            resp = client.post(
                "/api/v1/tracks/current", json={"track_id": track_id},
            )
            assert resp.status_code == 200

            # ② 导入调教（注入遥测帧后导入）
            _inject_telemetry(client)
            resp = client.post("/api/v1/setup/import")
            assert resp.status_code == 200
            setup_data = _assert_envelope(resp.json())
            assert len(setup_data["params"]) == 23

            # ③ 提交多弯道反馈
            for corner, symptom, strength in [
                (1, "understeer", 4),
                (5, "brake_long", 3),
                (10, "oversteer", 3),
            ]:
                resp = client.post(
                    "/api/v1/feedback",
                    json={
                        "track_id": track_id,
                        "corner_number": corner,
                        "symptom": symptom,
                        "strength": strength,
                    },
                )
                assert resp.status_code == 200

            # ④ 生成建议
            resp = client.post("/api/v1/suggest", json={"track_id": track_id})
            assert resp.status_code == 200
            suggestion_data = _assert_envelope(resp.json())
            report = suggestion_data["report"]
            assert len(report["parameters"]) == 23
            # 应有非零调整（多症状非零 Dx）
            nonzero_count = sum(
                1 for v in report["setup_delta"].values() if abs(v) > 1e-9
            )
            assert nonzero_count > 0, "多症状反馈应产生非零调整"

            # ⑤ 查看迭代历史
            resp = client.get(
                "/api/v1/iteration/history", params={"track_id": track_id},
            )
            assert resp.status_code == 200
            iterations = _assert_envelope(resp.json())
            assert len(iterations) >= 1

            # ⑥ 再跑一轮验证收敛性
            resp2 = client.post("/api/v1/suggest", json={"track_id": track_id})
            assert resp2.status_code == 200
            report2 = resp2.json()["data"]["report"]
            assert report["setup_delta"] == report2["setup_delta"]

    def test_health_endpoint(self) -> None:
        """GET /api/v1/health 返回健康状态信封。"""
        with make_client() as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            data = _assert_envelope(resp.json())
            assert "status" in data
            assert "telemetry_connected" in data
            assert "udp_host" in data
            assert "udp_port" in data