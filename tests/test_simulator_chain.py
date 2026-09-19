"""模拟遥测链路端到端测试 —— 无真实 F1 游戏时的可验证性。

背景（本文件锁定的两个真实缺陷）：
    1. ``/api/v1/telemetry/simulate`` 启动 ``TelemetrySimulator`` 后，
       **从未把包处理函数注册为模拟器的帧回调**。模拟器内部 ``_dispatch``
       在无订阅者时静默丢弃每一帧，于是 ``TelemetryStream`` /
       ``LapAggregator`` / ``StyleExtractor`` 全收不到数据，
       ``/suggest`` 便永远返回「无遥测」。模拟模式形同虚设。
    2. 模拟器只生成 Packet 2 (LapData) / Packet 6 (CarTelemetry) 两类帧，
       不含 Packet 13 (MotionEx)，导致规则9 刮底检测在模拟模式下无输入。

修复：``app`` 把包处理函数存入 ``app.state.packet_handler``，
``_get_or_create_simulator`` 创建模拟器时注册该回调；
模拟器新增 ``_build_motion_frame`` 并派发。

本文件用真实 API + 真实模拟器（缩短帧间隔）验证整条链路。
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config

TEST_DATA_DIR = Path("./data_test_sim_chain")


def make_client() -> TestClient:
    """创建测试客户端（清理数据目录确保隔离）。"""
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    config = Config(data_dir=str(TEST_DATA_DIR))
    app = create_app(config)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    """进入 lifespan（app.state 才会被填充），退出时停止模拟器。"""
    c = make_client()
    with c:
        yield c
    simulator = getattr(c.app.state, "telemetry_simulator", None)
    if simulator is not None:
        try:
            simulator.stop()
        except Exception:
            pass


# ===========================================================================
# 1. app.state 暴露包处理函数
# ===========================================================================
class TestPacketHandlerInState:
    """包处理函数必须可从 app.state 取到（模拟器复用的前提）。"""

    def test_packet_handler_stored_in_state(self, client: TestClient) -> None:
        handler = getattr(client.app.state, "packet_handler", None)
        assert handler is not None, "app.state.packet_handler 未设置"
        assert callable(handler)


# ===========================================================================
# 2. 模拟器注册回调
# ===========================================================================
class TestSimulatorHandlerWiring:
    """创建模拟器时必须注册包处理函数，否则帧全部空转丢弃。"""

    def test_simulator_gets_handler_on_create(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/telemetry/simulate",
            json={"action": "start", "track_id": "monza"},
        )
        assert resp.status_code == 200, resp.text

        simulator = client.app.state.telemetry_simulator
        # 模拟器内部 handler 列表必须包含 app 的包处理函数
        handlers = getattr(simulator, "_handlers", None)
        assert handlers, "模拟器未注册任何 handler"
        assert client.app.state.packet_handler in handlers

    def test_simulator_feeds_aggregator(self, client: TestClient) -> None:
        """启动模拟后，LapAggregator 必须真的收到帧（核心断言）。"""
        resp = client.post(
            "/api/v1/telemetry/simulate",
            json={"action": "start", "track_id": "monza"},
        )
        assert resp.status_code == 200, resp.text

        aggregator = client.app.state.lap_aggregator
        # 模拟器默认 20fps，给足时间累积若干帧
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and aggregator.frames_total == 0:
            time.sleep(0.1)

        simulator = client.app.state.telemetry_simulator
        simulator.stop()

        assert aggregator.frames_total > 0, (
            "模拟器运行后 LapAggregator 仍收不到任何帧 —— "
            "handler 未接入模拟器"
        )

    def test_suggest_has_telemetry_after_simulate(self, client: TestClient) -> None:
        """模拟一段时间后，遥测统计必须真的进入流缓存供 /suggest 使用。"""
        client.post(
            "/api/v1/telemetry/simulate",
            json={"action": "start", "track_id": "monza"},
        )
        aggregator = client.app.state.lap_aggregator
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and aggregator.frames_total < 10:
            time.sleep(0.1)
        simulator = client.app.state.telemetry_simulator
        simulator.stop()

        # 遥测流必须缓存到 Packet 6 (CarTelemetry) 与 Packet 13 (MotionEx)
        stream = client.app.state.telemetry_stream
        latest = stream.get_all_latest()
        assert 6 in latest, "模拟器未把 CarTelemetry 写入遥测流"
        assert 13 in latest, "模拟器未把 MotionEx 写入遥测流"

        # 聚合器必须产出引擎所需字段
        snap = aggregator.snapshot()
        assert snap["lap_frames"] > 0
        assert "max_speed" in snap and snap["max_speed"] > 0
        # MotionEx 累积量存在（模拟已覆盖刮底输入）
        assert "motion_ex_frames" in snap


# ===========================================================================
# 3. 模拟器产出 MotionEx
# ===========================================================================
class TestSimulatorMotionEx:
    """模拟器必须产出 MotionEx 帧（规则9 在模拟模式下的输入）。"""

    def test_motion_frame_shape(self, client: TestClient) -> None:
        """``_build_motion_frame`` 产出的字典含规则9 所需字段。"""
        from setup_tuner.telemetry.simulator import TelemetrySimulator

        sim = TelemetrySimulator()
        telem = {
            "speed": 250.0,
            "brake": 0.8,
            "steer": 0.3,
            "in_corner": True,
            "lap_number": 1,
            "track_id": "monza",
        }
        import random

        motion = sim._build_motion_frame(telem, random.Random(42))  # noqa: SLF001

        assert motion["packet_id"] == 13
        assert motion["name"] == "MotionEx"
        assert "m_frontAeroHeight" in motion
        assert "m_rearAeroHeight" in motion
        assert len(motion["m_suspensionPosition"]) == 4
        assert 0.0 < motion["m_frontAeroHeight"] < 1.0
        assert 0.0 < motion["m_rearAeroHeight"] < 1.0

    def test_motion_frame_can_trigger_bottoming(self, client: TestClient) -> None:
        """重刹 + 高速时底板应下沉到能触发刮底判定的高度。"""
        from setup_tuner.telemetry.lap_aggregator import LapAggregator
        from setup_tuner.telemetry.simulator import TelemetrySimulator

        sim = TelemetrySimulator()
        agg = LapAggregator()
        import random

        # 多次采样，确认模拟数据能覆盖触底情形
        triggered = False
        for seed in range(40):
            motion = sim._build_motion_frame(  # noqa: SLF001
                {"speed": 340.0, "brake": 1.0, "steer": 0.0, "in_corner": False},
                random.Random(seed),
            )
            agg.reset()
            agg.on_motion_ex(motion)
            if agg.snapshot().get("plank_bottoming"):
                triggered = True
                break
        assert triggered, "重刹+高速的模拟数据未触发刮底 —— 规则9 在模拟模式下不可验证"

    def test_motion_frame_normal_driving_no_bottoming(self, client: TestClient) -> None:
        """巡航（不高不刹不弯）时不应刮底（避免模拟数据全体误报）。"""
        from setup_tuner.telemetry.lap_aggregator import LapAggregator
        from setup_tuner.telemetry.simulator import TelemetrySimulator

        sim = TelemetrySimulator()
        agg = LapAggregator()
        import random

        clean = 0
        for seed in range(40):
            motion = sim._build_motion_frame(  # noqa: SLF001
                {"speed": 120.0, "brake": 0.0, "steer": 0.0, "in_corner": False},
                random.Random(seed),
            )
            agg.reset()
            agg.on_motion_ex(motion)
            if not agg.snapshot().get("plank_bottoming"):
                clean += 1
        assert clean >= 30, f"巡航工况误报刮底过多（仅 {clean}/40 干净）"


# ===========================================================================
# 4. 停止模拟后不残留
# ===========================================================================
class TestSimulatorStop:
    """停止模拟后线程与帧计数应稳定。"""

    def test_stop_is_idempotent(self, client: TestClient) -> None:
        client.post(
            "/api/v1/telemetry/simulate",
            json={"action": "start", "track_id": "monza"},
        )
        time.sleep(0.5)
        r1 = client.post(
            "/api/v1/telemetry/simulate",
            json={"action": "stop", "track_id": "monza"},
        )
        r2 = client.post(
            "/api/v1/telemetry/simulate",
            json={"action": "stop", "track_id": "monza"},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["data"]["simulating"] is False
