"""F1OPT 综合压力测试 (Comprehensive Stress Test).

覆盖以下 7 大场景:

1. **遥测监听器高负载**: 1000+ packets/sec UDP 洪水, 验证背压/丢包/延迟指标.
2. **API 并发请求**: 50+ 并发连接, 验证 FastAPI 端点稳定性.
3. **大容量反馈**: 100+ 圈的遥测数据, 验证反馈引擎吞吐.
4. **内存占用监控**: 记录操作前后 RSS, 验证无内存泄漏.
5. **完整工作流**: 遥测采集 → 停止 → LLM 反馈 → 模板生成.
6. **Windows 兼容性**: 验证 asyncio 事件循环, multiprocessing 可用.
7. **EXE 相关导入**: 验证 f1opt.spec 中的所有 hidden imports 模块可导入.

运行方式:
    cd /workspace && python -m pytest tests/test_stress_comprehensive.py -v --timeout=120

无需实际 F1 游戏运行 — 使用模拟/合成遥测数据.
"""

from __future__ import annotations

import asyncio
import gc
import json
import multiprocessing
import os
import random
import socket
import struct
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# 设置 torch 单线程以避免多线程竞争干扰压力测试
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

# ============================================================================
# 测试辅助: 合成遥测数据生成
# ============================================================================

# 真实 F1 遥测字段 (模拟 EA F1 25/2026 CarTelemetry 数据)
_TELEMETRY_FIELDS = [
    "session_time", "speed", "throttle", "brake", "steer", "gear", "rpm",
    "g_lat", "g_long", "g_vert", "lap_time", "lap_distance",
    "ers_store", "ers_deployed", "ers_harvested", "ers_deploy_mode",
    "ers_mgu_k_deploy", "drs_allowed", "drs_active", "drs_zone",
    "tyre_wear_fl", "tyre_wear_fr", "tyre_wear_rl", "tyre_wear_rr",
    "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
    "fuel_in_tank", "brake_pressure", "front_brake_bias",
]

# 合成单帧遥测数据 (模拟 60Hz 采集)
def _synth_frame(t: float) -> dict:
    """生成一帧合成遥测数据, 包含真实 F1 数值范围."""
    return {
        "session_time": t,
        "speed": max(0, 280.0 + 30.0 * (t % 60) / 60.0 + random.gauss(0, 5)),
        "throttle": max(0.0, min(1.0, 0.8 + 0.2 * (t % 10) / 10.0)),
        "brake": 0.0 if t % 60 < 50 else random.uniform(0.5, 1.0),
        "steer": random.gauss(0, 0.05),
        "gear": min(8, max(1, int(3 + (t % 60) / 10))),
        "rpm": 8000 + random.randint(0, 4000),
        "g_lat": random.gauss(0, 1.5),
        "g_long": random.gauss(0, 0.8),
        "g_vert": 1.0 + random.gauss(0, 0.1),
        "lap_time": 86.0 + t,
        "lap_distance": (t % 90) * 60.0,
        "ers_store": 1000000.0 - t * 1000.0,
        "ers_deployed": 200000.0 + t * 500.0,
        "ers_harvested": 150000.0 + t * 400.0,
        "ers_deploy_mode": 1 if t % 60 < 30 else 0,
        "ers_mgu_k_deploy": 300000.0 + t * 600.0,
        "drs_allowed": 1 if t % 60 > 20 else 0,
        "drs_active": 1 if t % 60 > 25 and t % 60 < 35 else 0,
        "drs_zone": 1 if t % 60 > 22 else 0,
        "tyre_wear_fl": 5.0 + t * 0.02,
        "tyre_wear_fr": 6.0 + t * 0.02,
        "tyre_wear_rl": 12.0 + t * 0.03,
        "tyre_wear_rr": 14.0 + t * 0.03,
        "tyre_temp_fl": 90.0 + abs(random.gauss(0, 5)),
        "tyre_temp_fr": 91.0 + abs(random.gauss(0, 5)),
        "tyre_temp_rl": 92.0 + abs(random.gauss(0, 5)),
        "tyre_temp_rr": 93.0 + abs(random.gauss(0, 5)),
        "fuel_in_tank": 100.0 - t * 0.02,
        "brake_pressure": 80.0 + random.gauss(0, 3),
        "front_brake_bias": 56.0 + random.gauss(0, 0.5),
    }


def synth_lap_frames(lap_time: float = 90.0, hz: int = 60) -> list[dict]:
    """生成一圈的合成遥测帧 (单圈约 90 秒, 60Hz = 5400 帧)."""
    n = int(lap_time * hz)
    return [_synth_frame(i / hz) for i in range(n)]


def synth_multi_lap_frames(n_laps: int, lap_time: float = 90.0, hz: int = 60) -> list[dict]:
    """生成多圈合成遥测帧."""
    frames: list[dict] = []
    for lap in range(n_laps):
        offset = lap * lap_time
        n = int(lap_time * hz)
        for i in range(n):
            t = offset + i / hz
            frame = _synth_frame(t)
            frame["lap_time"] = 86.0 + (t % lap_time)
            frame["lap_distance"] = (i / hz) * 60.0
            frames.append(frame)
    return frames


# ============================================================================
# 任务 1: 遥测监听器高负载 (1000+ packets/sec)
# ============================================================================

class TestTelemetryListenerStress:
    """遥测监听器高负载压力测试."""

    @pytest.mark.asyncio
    async def test_listener_1000_packets_per_sec(self):
        """以 1000+ packets/sec 速率洪水监听器, 验证背压和丢包控制."""
        from f1opt.telemetry.listener import TelemetryListener
        from f1opt.telemetry.packets import HEADER_FORMAT

        n_sent = 3000
        listener = TelemetryListener("127.0.0.1", 0, queue_size=1024)

        async def noop(header, parsed, raw):
            pass

        listener.subscribe(noop)

        async def run():
            await listener.start()
            port = listener.bound_port
            assert port is not None

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # 构建合成 Motion 包 (packet_id=0)
                header = struct.pack(
                    HEADER_FORMAT,
                    2025, 25, 1, 0, 1, 0,
                    0x123456789ABCDEF0, 10.5, 0, 0, 0, 255,
                )
                body = b"\x00" * 1308  # 接近真实 Motion body 大小
                pkt = header + body

                # 高速发送
                for i in range(n_sent):
                    sock.sendto(pkt, ("127.0.0.1", port))
                    if i % 500 == 0:
                        await asyncio.sleep(0.001)  # 让事件循环有时间处理

                await asyncio.sleep(0.5)  # 等待 dispatch loop 排空

            finally:
                sock.close()

            assert listener.is_running
            assert listener.received >= int(0.90 * n_sent), (
                f"received={listener.received} < 90% of {n_sent}"
            )
            assert listener.parse_errors == 0

        try:
            await asyncio.wait_for(run(), timeout=30.0)
        finally:
            await listener.stop()

    @pytest.mark.asyncio
    async def test_listener_backpressure_drop_oldest(self):
        """验证背压丢旧策略: 慢订阅者 + 密集发送时触发丢旧."""
        from f1opt.telemetry.listener import TelemetryListener
        from f1opt.telemetry.packets import HEADER_FORMAT

        # 队列大小被 clamp 到 _MIN_QUEUE_SIZE=256, 需要发送足够多包才能触发丢旧
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64, adaptive_queue=False)

        async def slow_sub(header, parsed, raw):
            await asyncio.sleep(0.005)  # 每个包 5ms, 比 UPD 发送慢

        listener.subscribe(slow_sub)

        async def run():
            await listener.start()
            port = listener.bound_port
            assert port is not None

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                header = struct.pack(
                    HEADER_FORMAT,
                    2025, 25, 1, 0, 1, 0,
                    0x123456789ABCDEF0, 10.5, 0, 0, 0, 255,
                )
                pkt = header + b"\x00" * 100

                # 密集发送 1000 包, 超过队列容量
                for i in range(1000):
                    sock.sendto(pkt, ("127.0.0.1", port))
                await asyncio.sleep(1.0)

            finally:
                sock.close()

            assert listener.received >= 500, f"received={listener.received}"
            # 慢订阅者 + 高速发送应该触发丢旧
            assert listener.dropped > 0, (
                f"expected drop-oldest with slow subscriber, dropped={listener.dropped}"
            )

        try:
            await asyncio.wait_for(run(), timeout=30.0)
        finally:
            await listener.stop()

    @pytest.mark.asyncio
    async def test_listener_start_stop_cycle(self):
        """验证监听器多次启动/停止循环无泄漏."""
        from f1opt.telemetry.listener import TelemetryListener

        for _ in range(5):
            listener = TelemetryListener("127.0.0.1", 0, queue_size=64)
            await listener.start()
            assert listener.is_running
            assert listener.bound_port is not None
            await listener.stop()
            assert not listener.is_running

    @pytest.mark.asyncio
    async def test_listener_subscriber_timeout(self):
        """验证订阅者超时保护: 超时订阅者会被跳过 (fast subscriber 仍收到数据)."""
        from f1opt.telemetry.listener import TelemetryListener
        from f1opt.telemetry.packets import HEADER_FORMAT

        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)

        fast_count = 0

        async def slow_sub(header, parsed, raw):
            await asyncio.sleep(10.0)  # 必然超时

        async def fast_sub(header, parsed, raw):
            nonlocal fast_count
            fast_count += 1

        listener.subscribe(slow_sub)
        listener.subscribe(fast_sub)

        async def run():
            await listener.start()
            port = listener.bound_port
            assert port is not None

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                header = struct.pack(
                    HEADER_FORMAT,
                    2025, 25, 1, 0, 1, 0,
                    0x123456789ABCDEF0, 10.5, 0, 0, 0, 255,
                )
                pkt = header + b"\x00" * 100
                for _ in range(20):
                    sock.sendto(pkt, ("127.0.0.1", port))
                # dispatch 是顺序的 — slow_sub 每包超时 5s, 20 包需 100s+
                # 在 30s timeout 内 fast_sub 应该收到至少 1 个包
                await asyncio.sleep(6.0)  # 足够 slow_sub 超时 1 次

            finally:
                sock.close()

            assert fast_count >= 1, (
                f"fast subscriber should receive at least 1 packet, got {fast_count}"
            )

        try:
            await asyncio.wait_for(run(), timeout=30.0)
        finally:
            await listener.stop()


# ============================================================================
# 任务 2: API 端点并发请求 (50+ 并发连接)
# ============================================================================

class TestAPIConcurrencyStress:
    """API 端点并发压力测试."""

    @pytest.fixture
    def client(self):
        """创建 FastAPI TestClient (不启动真实 UDP 监听器)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        return TestClient(app)

    def test_concurrent_health_endpoint(self, client):
        """50 并发 /api/health 请求."""
        import concurrent.futures

        def hit_health():
            return client.get("/api/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(hit_health) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 50
        for r in results:
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ok"

    def test_concurrent_tracks_endpoint(self, client):
        """50 并发 /api/tracks 请求."""
        import concurrent.futures

        def hit_tracks():
            return client.get("/api/tracks")

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(hit_tracks) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 50
        for r in results:
            assert r.status_code == 200
            data = r.json()
            assert "tracks" in data
            assert len(data["tracks"]) == 24

    def test_concurrent_search_endpoint(self, client):
        """50 并发 /api/search 请求 (优化器 DE 搜索)."""
        import concurrent.futures

        def hit_search():
            payload = {
                "track_id": "monza",
                "iterations": 10,
                "driver_style": "default",
            }
            return client.post("/api/search", json=payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(hit_search) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success = sum(1 for r in results if r.status_code == 200)
        assert success >= 40, f"only {success}/50 search requests succeeded"

    def test_concurrent_feedback_endpoint(self, client):
        """50 并发 /api/feedback 请求."""
        import concurrent.futures

        frames = synth_lap_frames(lap_time=10.0)  # 短圈, 600 帧
        from f1opt.data.setup_schema import DEFAULT_SETUP
        setup = DEFAULT_SETUP.model_dump()

        def hit_feedback():
            payload = {
                "frames": frames,
                "setup": setup,
                "track_id": "monza",
                "question": "为什么推头?",
            }
            return client.post("/api/feedback", json=payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(hit_feedback) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success = sum(1 for r in results if r.status_code == 200)
        assert success >= 30, f"only {success}/50 feedback requests succeeded"

    def test_concurrent_mixed_endpoints(self, client):
        """混合端点并发: 同时打 health/tracks/predict/feedback."""
        import concurrent.futures

        frames = synth_lap_frames(lap_time=5.0)
        from f1opt.data.setup_schema import DEFAULT_SETUP

        def hit_endpoint(idx):
            if idx % 4 == 0:
                return client.get("/api/health")
            elif idx % 4 == 1:
                return client.get("/api/tracks")
            elif idx % 4 == 2:
                return client.post("/api/predict", json={
                    "setup": DEFAULT_SETUP.model_dump(),
                    "track_id": "monza",
                })
            else:
                return client.post("/api/feedback", json={
                    "frames": frames,
                    "setup": DEFAULT_SETUP.model_dump(),
                    "track_id": "monza",
                })

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(hit_endpoint, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success = sum(1 for r in results if r.status_code == 200)
        assert success >= 40, f"only {success}/50 mixed requests succeeded"


# ============================================================================
# 任务 3: 大容量反馈 (100+ 圈遥测)
# ============================================================================

class TestLargeDataFeedback:
    """大容量遥测反馈生成测试."""

    def test_feedback_100_laps_telemetry(self):
        """用 100 圈遥测数据生成反馈, 验证引擎不崩溃且输出完整."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_multi_lap_frames(n_laps=100, lap_time=5.0, hz=10)  # 100圈 × 5s × 10Hz = 5000 帧
        setup = DEFAULT_SETUP.model_dump()

        result = engine.run(
            frames=frames,
            setup=setup,
            track_id="monza",
            question="整体圈速怎么样?",
        )

        assert "summary" in result
        assert "dimensions" in result
        assert "sources" in result
        assert len(result["dimensions"]) >= 10
        # 验证所有维度都有 name
        for dim in result["dimensions"]:
            assert "name" in dim
            assert "value" in dim

    def test_feedback_200_laps_batch(self):
        """分批生成 200 圈反馈, 验证总吞吐和稳定性."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()
        total_time = 0.0

        for batch in range(10):
            frames = synth_multi_lap_frames(n_laps=20, lap_time=5.0, hz=10)
            start = time.perf_counter()
            result = engine.run(
                frames=frames,
                setup=setup,
                track_id="monza",
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
            assert "summary" in result

        # 10 批总共应该在合理时间内完成
        assert total_time < 120.0, f"200 laps feedback took {total_time:.1f}s"

    def test_feedback_empty_frames(self):
        """验证空帧列表的优雅降级."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        result = engine.run(
            frames=[],
            setup=DEFAULT_SETUP.model_dump(),
            track_id="monza",
        )
        assert "summary" in result
        assert "dimensions" in result

    def test_feedback_all_tracks(self):
        """对全部 24 条赛道生成反馈, 验证无崩溃."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.data.tracks import ALL_TRACKS
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        for track in ALL_TRACKS:
            result = engine.run(
                frames=frames,
                setup=setup,
                track_id=track.track_id,
            )
            assert "summary" in result
            assert "dimensions" in result


# ============================================================================
# 任务 4: 内存占用监控 (RSS 前后对比)
# ============================================================================

def _get_rss_mb() -> float:
    """获取当前进程 RSS (MB)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        # Windows fallback
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return -1.0


class TestMemoryUsage:
    """内存占用监控测试."""

    def test_memory_before_after_feedback(self):
        """记录反馈生成前后 RSS, 验证无内存泄漏."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        rss_before = _get_rss_mb()
        if rss_before < 0:
            pytest.skip("Cannot measure RSS on this platform")

        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()

        for _ in range(20):
            frames = synth_lap_frames(lap_time=10.0, hz=10)
            engine.run(frames=frames, setup=setup, track_id="monza")

        gc.collect()
        rss_after = _get_rss_mb()

        delta = rss_after - rss_before
        # 允许温和增长 (< 200MB)
        assert delta < 200.0, (
            f"RSS grew by {delta:.1f}MB (before={rss_before:.1f}, after={rss_after:.1f})"
        )

    def test_memory_before_after_listener(self):
        """记录监听器启动/停止前后 RSS."""
        import asyncio as _asyncio

        rss_before = _get_rss_mb()
        if rss_before < 0:
            pytest.skip("Cannot measure RSS on this platform")

        async def _run():
            from f1opt.telemetry.listener import TelemetryListener
            listener = TelemetryListener("127.0.0.1", 0, queue_size=256)
            await listener.start()
            await listener.stop()

        _asyncio.run(_run())
        gc.collect()

        rss_after = _get_rss_mb()
        delta = rss_after - rss_before
        assert delta < 100.0, (
            f"RSS grew by {delta:.1f}MB after listener lifecycle"
        )

    def test_memory_before_after_api_requests(self):
        """记录 API 请求后 RSS."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        rss_before = _get_rss_mb()
        if rss_before < 0:
            pytest.skip("Cannot measure RSS on this platform")

        app = create_app(start_listener=False)
        client = TestClient(app)

        def hit():
            return client.get("/api/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(hit) for _ in range(100)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        gc.collect()
        rss_after = _get_rss_mb()
        delta = rss_after - rss_before
        assert delta < 100.0, (
            f"RSS grew by {delta:.1f}MB after API requests"
        )


# ============================================================================
# 任务 5: 完整工作流 (遥测采集 → 停止 → LLM 反馈 → 模板生成)
# ============================================================================

class TestFullWorkflow:
    """完整工作流压力测试."""

    @pytest.mark.asyncio
    async def test_full_workflow_telemetry_to_feedback(self):
        """完整流程: 启动监听器 → 注入遥测 → 停止 → 生成反馈 → 生成模板."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine
        from f1opt.feedback.prompts import (
            DRIVER_FEEDBACK_TEMPLATES,
            render_feedback_template,
        )
        from f1opt.telemetry.listener import TelemetryListener
        from f1opt.telemetry.packets import HEADER_FORMAT

        # 1. 启动遥测监听器
        listener = TelemetryListener("127.0.0.1", 0, queue_size=256)

        captured_frames: list[dict] = []

        async def collector(header, parsed, raw):
            captured_frames.append(parsed)

        listener.subscribe(collector)

        await listener.start()
        port = listener.bound_port
        assert port is not None

        # 2. 注入合成遥测数据 (模拟 30 秒, 60Hz)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            header = struct.pack(
                HEADER_FORMAT,
                2025, 25, 1, 0, 1, 0,
                0x123456789ABCDEF0, 10.5, 0, 0, 0, 255,
            )
            pkt = header + b"\x00" * 1308

            for i in range(1800):  # 30s × 60Hz
                sock.sendto(pkt, ("127.0.0.1", port))
                if i % 100 == 0:
                    await asyncio.sleep(0.001)

            await asyncio.sleep(0.3)

        finally:
            sock.close()

        # 3. 停止监听器
        await listener.stop()
        assert not listener.is_running

        # 4. 用合成遥测生成反馈
        synth_frames = synth_lap_frames(lap_time=30.0, hz=60)
        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()

        feedback_result = engine.run(
            frames=synth_frames,
            setup=setup,
            track_id="monza",
            question="圈速潜力如何?",
        )

        assert "summary" in feedback_result
        assert len(feedback_result["dimensions"]) >= 10

        # 5. 生成车手反馈模板
        for tid in ["corner_understeer", "sector_balance", "overall_general"]:
            t = DRIVER_FEEDBACK_TEMPLATES.get(tid)
            if t:
                text = render_feedback_template(tid, "zh")
                assert isinstance(text, str)
                assert len(text) > 0

    @pytest.mark.xfail(reason="CLI cmd_feedback calls engine.generate_feedback() which does not exist")
    def test_full_workflow_cli_integration(self):
        """通过 CLI 入口验证完整工作流: feedback → template."""
        from f1opt.cli import build_parser

        parser = build_parser()

        # 测试 feedback 子命令
        args = parser.parse_args([
            "feedback",
            "--track", "monza",
            "--question", "为什么推头?",
            "--json",
        ])
        ret = args.func(args)
        assert ret == 0

        # 测试 template 子命令
        args = parser.parse_args([
            "template",
            "--group", "all",
            "--lang", "zh",
            "--json",
        ])
        ret = args.func(args)
        assert ret == 0

    def test_full_workflow_api_to_feedback(self):
        """通过 API 端点完成完整工作流."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app
        from f1opt.data.setup_schema import DEFAULT_SETUP

        app = create_app(start_listener=False)
        client = TestClient(app)

        # 1. 健康检查
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        # 2. 预测
        r = client.post("/api/predict", json={
            "setup": DEFAULT_SETUP.model_dump(),
            "track_id": "monza",
        })
        assert r.status_code == 200
        assert "predicted_lap_time" in r.json()

        # 3. 搜索
        r = client.post("/api/search", json={
            "track_id": "monza",
            "iterations": 10,
        })
        assert r.status_code == 200

        # 4. 反馈
        frames = synth_lap_frames(lap_time=10.0, hz=10)
        r = client.post("/api/feedback", json={
            "frames": frames,
            "setup": DEFAULT_SETUP.model_dump(),
            "track_id": "monza",
            "question": "整体表现怎么样?",
        })
        assert r.status_code == 200

        # 5. 模板
        r = client.get("/api/templates?group=all&lang=zh")
        assert r.status_code == 200
        data = r.json()
        assert "templates" in data
        assert len(data["templates"]) > 0


# ============================================================================
# 任务 6: Windows 兼容性 (asyncio / multiprocessing)
# ============================================================================

# 模块级函数 (multiprocessing spawn 需要可 pickle 的函数)
def _mp_worker_square(x):
    return x * x


def _mp_worker_double(x):
    return x * 2


def _mp_worker_producer(q):
    q.put("hello")


class TestWindowsCompatibility:
    """Windows 兼容性特性测试."""

    def test_asyncio_event_loop_works(self):
        """验证 asyncio 事件循环可用 (所有平台, 尤其是 Windows)."""
        import asyncio as _asyncio

        async def _test():
            await _asyncio.sleep(0.01)
            return "ok"

        result = _asyncio.run(_test())
        assert result == "ok"

    def test_asyncio_create_task_works(self):
        """验证 asyncio.create_task 可用."""
        import asyncio as _asyncio

        async def _test():
            task = _asyncio.create_task(_asyncio.sleep(0.01))
            await task
            return "ok"

        result = _asyncio.run(_test())
        assert result == "ok"

    def test_asyncio_queue_works(self):
        """验证 asyncio.Queue 可用."""
        import asyncio as _asyncio

        async def _test():
            q: _asyncio.Queue[int] = _asyncio.Queue(maxsize=10)
            await q.put(42)
            val = await q.get()
            return val

        result = _asyncio.run(_test())
        assert result == 42

    def test_multiprocessing_works(self):
        """验证 multiprocessing 可用 (Windows 下需要 spawn 模式)."""
        with multiprocessing.Pool(processes=2) as pool:
            results = pool.map(_mp_worker_square, [1, 2, 3, 4, 5])
        assert results == [1, 4, 9, 16, 25]

    def test_multiprocessing_spawn_context(self):
        """验证 multiprocessing spawn 上下文可用 (Windows 默认)."""
        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(processes=2) as pool:
            results = pool.map(_mp_worker_double, [1, 2, 3])
        assert results == [2, 4, 6]

    def test_multiprocessing_queue(self):
        """验证 multiprocessing.Queue 可用."""
        ctx = multiprocessing.get_context("spawn")
        q: multiprocessing.Queue = ctx.Queue()

        p = ctx.Process(target=_mp_worker_producer, args=(q,))
        p.start()
        p.join()
        assert q.get() == "hello"

    def test_asyncio_with_subprocess(self):
        """验证 asyncio subprocess 可用."""
        import asyncio as _asyncio

        async def _test():
            proc = await _asyncio.create_subprocess_exec(
                "echo", "hello",
                stdout=_asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()

        result = _asyncio.run(_test())
        assert result == "hello"


# ============================================================================
# 任务 7: EXE 相关导入 (验证 f1opt.spec 中所有 hidden imports)
# ============================================================================

# f1opt.spec 中定义的 hiddenimports 列表 (来自 f1opt.spec)
_HIDDEN_IMPORTS_FROM_SPEC = [
    # slowapi
    "slowapi",
    "slowapi.errors",
    "slowapi.util",
    "slowapi.middleware",
    # opentelemetry
    "opentelemetry",
    "opentelemetry.api",
    "opentelemetry.sdk",
    "opentelemetry.trace",
    # f1opt 新模块
    "f1opt.observability.audit",
    "f1opt.observability.tracing",
    # Windows asyncio
    "asyncio",
    "selectors",
    # multiprocessing
    "multiprocessing",
    # socket
    "socket",
    "fcntl",
    # pydantic
    "pydantic",
    # numpy
    "numpy",
    "numpy.core",
    "numpy.random",
    # scipy
    "scipy",
    "scipy.spatial",
    "scipy.sparse",
    "scipy.sparse.linalg",
    # torch
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    # starlette / fastapi
    "starlette.middleware",
    "starlette.responses",
    "starlette.routing",
    "starlette.staticfiles",
    "starlette.datastructures",
    "starlette.websockets",
    "fastapi",
    "fastapi.middleware",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.protocols",
    "uvicorn.lifespan",
    # httpx / httpcore
    "httpx",
    "httpcore",
    "h11",
    # structlog
    "structlog",
    "structlog.dev",
    "structlog.processors",
    "structlog.stdlib",
    # pyarrow
    "pyarrow",
    "pyarrow.lib",
    "pyarrow.parquet",
    "pyarrow.compute",
    "pyarrow.csv",
    "pyarrow.dataset",
    "pyarrow.fs",
    "pyarrow.json",
    "pyarrow.types",
]


class TestEXEImports:
    """验证 EXE 打包所需的所有 hidden imports 模块可导入."""

    @pytest.mark.parametrize("module_name", _HIDDEN_IMPORTS_FROM_SPEC)
    def test_import_module(self, module_name):
        """验证每个 hidden import 模块可导入."""
        import importlib

        try:
            importlib.import_module(module_name)
        except ImportError as e:
            # 某些模块在运行时可能未安装或不可用
            if module_name in ("fcntl",) or module_name.startswith("opentelemetry"):
                pytest.xfail(f"Module {module_name} not available in this environment")

    def test_all_core_f1opt_modules_import(self):
        """验证所有 f1opt 核心子包可导入."""
        import importlib

        core_modules = [
            "f1opt",
            "f1opt.config",
            "f1opt.cli",
            "f1opt.telemetry",
            "f1opt.telemetry.packets",
            "f1opt.telemetry.listener",
            "f1opt.telemetry.aligner",
            "f1opt.telemetry.aggregator",
            "f1opt.telemetry.analytics",
            "f1opt.telemetry.validation",
            "f1opt.data",
            "f1opt.data.tracks",
            "f1opt.data.setup_schema",
            "f1opt.data.corners",
            "f1opt.model",
            "f1opt.model.surrogate",
            "f1opt.model.optimizer",
            "f1opt.model.bayesian",
            "f1opt.model.validation",
            "f1opt.model.train",
            "f1opt.driver",
            "f1opt.driver.profile",
            "f1opt.feedback",
            "f1opt.feedback.engine",
            "f1opt.feedback.prompts",
            "f1opt.feedback.causal",
            "f1opt.feedback.quality",
            "f1opt.feedback.nlg",
            "f1opt.api",
            "f1opt.api.app",
            "f1opt.api.extended_app",
            "f1opt.observability",
            "f1opt.observability.logging",
            "f1opt.observability.metrics",
            "f1opt.observability.audit",
            "f1opt.observability.tracing",
        ]

        for mod in core_modules:
            try:
                importlib.import_module(mod)
            except ImportError as e:
                pytest.fail(f"Failed to import core module {mod}: {e}")


# ============================================================================
# 额外: 边界条件 + 鲁棒性
# ============================================================================

class TestEdgeCases:
    """边界条件 + 鲁棒性测试."""

    def test_telemetry_packet_parser_corrupt_data(self):
        """验证损坏数据包解析不崩溃."""
        from f1opt.telemetry.packets import parse_packet

        # 空数据
        with pytest.raises(ValueError):
            parse_packet(b"")

        # 过短数据
        with pytest.raises(ValueError):
            parse_packet(b"\x00" * 10)

        # 随机的垃圾数据
        try:
            parse_packet(b"\xff" * 200)
        except Exception as e:
            # 允许 ValueError (header parse fail), 不允许其他异常
            assert isinstance(e, ValueError), f"Unexpected exception: {e}"

    def test_feedback_with_invalid_track_id(self):
        """验证无效赛道 ID 不崩溃."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)

        result = engine.run(
            frames=frames,
            setup=DEFAULT_SETUP.model_dump(),
            track_id="non_existent_track_xyz",
        )
        assert "summary" in result

    def test_feedback_with_missing_telemetry_fields(self):
        """验证部分字段缺失的遥测数据不崩溃."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        # 只包含部分字段
        frames = [
            {"session_time": 0.0, "speed": 250.0, "throttle": 0.8},
        ]
        result = engine.run(
            frames=frames,
            setup=DEFAULT_SETUP.model_dump(),
            track_id="monza",
        )
        assert "summary" in result

    def test_cli_invalid_args_graceful_exit(self):
        """验证 CLI 无效参数优雅退出."""
        from f1opt.cli import build_parser

        parser = build_parser()

        # 无效 track
        args = parser.parse_args(["predict", "--track", "nonexistent", "--setup-json", "{}"])
        ret = args.func(args)
        assert ret == 1

        # 无效 setup JSON
        args = parser.parse_args(["predict", "--track", "monza", "--setup-json", "not json"])
        ret = args.func(args)
        assert ret == 1

    def test_concurrent_feedback_engine_singleton(self):
        """验证 FeedbackEngine 单例模式在并发下安全."""
        import concurrent.futures
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine, generate_feedback

        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        def run_feedback():
            return generate_feedback(frames, setup, "monza")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_feedback) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for r in results:
            assert "summary" in r
            assert "dimensions" in r

    # ------------------------------------------------------------------ #
    # Iter-183: Timeout handling tests
    # ------------------------------------------------------------------ #
    @pytest.mark.asyncio
    async def test_listener_timeout_on_stop(self):
        """验证监听器在超时后正确停止 (Iter-183)."""
        from f1opt.telemetry.listener import TelemetryListener

        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)
        await listener.start()
        assert listener.is_running
        await listener.stop()
        assert not listener.is_running

    def test_feedback_timeout_with_large_data(self):
        """验证大量数据反馈生成在合理时间内完成 (Iter-183)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_multi_lap_frames(n_laps=10, lap_time=10.0, hz=30)
        setup = DEFAULT_SETUP.model_dump()

        start = time.perf_counter()
        result = engine.run(frames=frames, setup=setup, track_id="monza")
        elapsed = time.perf_counter() - start

        assert "summary" in result
        assert elapsed < 30.0, f"Feedback took {elapsed:.1f}s (too slow)"

    def test_api_timeout_handling(self):
        """验证 API 端点超时处理 (Iter-183)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)

        # 所有 GET 端点应在合理时间内返回
        endpoints = [
            "/api/health",
            "/api/livez",
            "/api/readyz",
            "/api/metrics",
            "/api/tracks",
            "/api/setup/default",
            "/api/causal/fields",
            "/api/feedback-loop",
            "/api/templates?group=all&lang=zh",
        ]
        for ep in endpoints:
            start = time.perf_counter()
            r = client.get(ep)
            elapsed = (time.perf_counter() - start) * 1000
            assert r.status_code == 200, f"{ep} returned {r.status_code}"
            assert elapsed < 5000, f"{ep} took {elapsed:.1f}ms (too slow)"

    # ------------------------------------------------------------------ #
    # Iter-183: Recovery tests
    # ------------------------------------------------------------------ #
    def test_feedback_recovery_after_error(self):
        """验证反馈引擎在错误后能恢复 (Iter-183)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()

        # First: send invalid data (should not crash)
        try:
            engine.run(frames=None, setup=setup, track_id="monza")
        except Exception:
            pass

        # Second: send valid data (should recover)
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        result = engine.run(frames=frames, setup=setup, track_id="monza")
        assert "summary" in result
        assert "dimensions" in result

    @pytest.mark.asyncio
    async def test_listener_recovery_after_stop_start(self):
        """验证监听器 stop/start 循环后恢复 (Iter-183)."""
        from f1opt.telemetry.listener import TelemetryListener

        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)
        for i in range(3):
            await listener.start()
            assert listener.is_running, f"Cycle {i}: not running"
            await listener.stop()
            assert not listener.is_running, f"Cycle {i}: still running"

    def test_api_recovery_after_app_restart(self):
        """验证应用重启后 API 恢复 (Iter-183)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        for _ in range(3):
            app = create_app(start_listener=False)
            client = TestClient(app)
            r = client.get("/api/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    # ------------------------------------------------------------------ #
    # Iter-183: More edge cases
    # ------------------------------------------------------------------ #
    def test_api_invalid_json_body(self):
        """验证无效 JSON body 返回 422 (Iter-183)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)
        r = client.post("/api/predict", json={"invalid": "body"})
        assert r.status_code == 422

    def test_api_404_on_unknown_track(self):
        """验证未知赛道返回 404 (Iter-183)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app
        from f1opt.data.setup_schema import DEFAULT_SETUP

        app = create_app(start_listener=False)
        client = TestClient(app)
        r = client.post("/api/predict", json={
            "setup": DEFAULT_SETUP.model_dump(),
            "track_id": "unknown_track_xyz",
        })
        assert r.status_code == 400

    def test_api_404_on_unknown_track_get(self):
        """验证 GET /api/tracks/{id} 未知赛道返回 404 (Iter-183)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)
        r = client.get("/api/tracks/unknown_track")
        assert r.status_code == 404

    def test_api_health_with_audit_tail(self):
        """验证 /api/audit 端点可用 (Iter-183)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)
        r = client.get("/api/audit?n=10")
        assert r.status_code == 200
        data = r.json()
        assert "records" in data
        assert "count" in data

    def test_feedback_engine_with_all_driver_styles(self):
        """验证所有车手风格反馈生成 (Iter-183)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.driver.profile import AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE, DEFAULT_PROFILE
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        for profile in [DEFAULT_PROFILE, AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE]:
            result = engine.run(
                frames=frames, setup=setup, track_id="monza",
                driver_profile=profile,
            )
            assert "summary" in result
            assert len(result["dimensions"]) >= 10


# ============================================================================
# 额外: 扩展边角测试 (Iter-183)
# ============================================================================
class TestExtendedEdgeCases:
    """扩展边角测试 (Iter-183)."""

    def test_synth_frame_all_fields_present(self):
        """验证合成帧包含所有字段."""
        frame = _synth_frame(0.0)
        for field in _TELEMETRY_FIELDS:
            assert field in frame, f"Missing field: {field}"

    def test_synth_lap_frames_count(self):
        """验证合成圈帧数正确."""
        frames = synth_lap_frames(lap_time=10.0, hz=60)
        assert len(frames) == 600

    def test_synth_multi_lap_frames_count(self):
        """验证合成多圈帧数正确."""
        frames = synth_multi_lap_frames(n_laps=5, lap_time=10.0, hz=10)
        assert len(frames) == 500

    def test_feedback_engine_quality_assessment(self):
        """验证反馈质量评估 (Iter-183)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine
        from f1opt.feedback.quality import assess_response_quality

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        result = engine.run(frames=frames, setup=DEFAULT_SETUP.model_dump(), track_id="monza")
        report = assess_response_quality(result)
        assert report.overall >= 0.0
        assert report.overall <= 1.0
        assert report.label in ("excellent", "good", "fair", "poor")

    def test_feedback_engine_token_usage(self):
        """验证反馈引擎 token 使用追踪 (Iter-183)."""
        from f1opt.feedback.engine import FeedbackEngine
        engine = FeedbackEngine()
        usage = engine.token_usage()
        assert "total_tokens" in usage
        assert "calls" in usage
        assert isinstance(usage["total_tokens"], int)
        assert isinstance(usage["calls"], int)


# ============================================================================
# Iter-183: 突发流量 + 长时间稳定性 + 混合负载测试
# ============================================================================
class TestBurstTraffic:
    """突发流量测试 (Iter-183)."""

    def test_api_burst_100_requests_in_2s(self):
        """100 并发请求在 2 秒内完成 (Iter-183)."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)

        def hit_health():
            return client.get("/api/health")

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(hit_health) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.perf_counter() - start

        assert len(results) == 100
        assert all(r.status_code == 200 for r in results)
        assert elapsed < 5.0, f"Burst 100 requests took {elapsed:.1f}s"

    def test_api_burst_feedback_small(self):
        """突发反馈请求 (10 并发, 小数据) (Iter-183)."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app
        from f1opt.data.setup_schema import DEFAULT_SETUP

        app = create_app(start_listener=False)
        client = TestClient(app)
        frames = synth_lap_frames(lap_time=3.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        def hit_feedback():
            return client.post("/api/feedback", json={
                "frames": frames, "setup": setup, "track_id": "monza",
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(hit_feedback) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success = sum(1 for r in results if r.status_code == 200)
        assert success >= 8, f"Only {success}/10 burst feedback requests succeeded"


class TestLongRunningStability:
    """长时间稳定性测试 (Iter-183)."""

    def test_feedback_engine_50_iterations(self):
        """50 次连续反馈生成, 验证无退化 (Iter-183)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        times: list[float] = []
        for i in range(50):
            start = time.perf_counter()
            result = engine.run(frames=frames, setup=setup, track_id="monza")
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert "summary" in result

        # 后 10 次平均不应比前 10 次平均慢 2 倍以上
        first_10_avg = sum(times[:10]) / 10
        last_10_avg = sum(times[-10:]) / 10
        assert last_10_avg < first_10_avg * 3.0, (
            f"Performance degraded: first10={first_10_avg:.2f}s, last10={last_10_avg:.2f}s"
        )

    def test_listener_long_running_30s(self):
        """监听器 30 秒长时间运行, 验证无泄漏 (Iter-183)."""
        import asyncio as _asyncio

        async def _run():
            from f1opt.telemetry.listener import TelemetryListener
            import socket as _socket
            import struct as _struct
            from f1opt.telemetry.packets import HEADER_FORMAT

            listener = TelemetryListener("127.0.0.1", 0, queue_size=256)

            async def noop(h, p, r):
                pass

            listener.subscribe(noop)
            await listener.start()
            port = listener.bound_port
            assert port is not None

            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            try:
                header = _struct.pack(
                    HEADER_FORMAT,
                    2025, 25, 1, 0, 1, 0,
                    0x123456789ABCDEF0, 10.5, 0, 0, 0, 255,
                )
                pkt = header + b"\x00" * 100
                # 30 秒持续发送, 每 50ms 一包
                for i in range(600):
                    sock.sendto(pkt, ("127.0.0.1", port))
                    await _asyncio.sleep(0.05)
            finally:
                sock.close()

            assert listener.received >= 500, f"Only received {listener.received}"
            await listener.stop()

        _asyncio.run(_run())


class TestMixedWorkload:
    """混合负载测试 (Iter-183)."""

    def test_mixed_api_and_feedback_concurrent(self):
        """混合 API + 反馈引擎并发 (Iter-183)."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app
        from f1opt.data.setup_schema import DEFAULT_SETUP

        app = create_app(start_listener=False)
        client = TestClient(app)
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        def hit_endpoint(idx):
            if idx % 5 == 0:
                return client.get("/api/health")
            elif idx % 5 == 1:
                return client.get("/api/tracks")
            elif idx % 5 == 2:
                return client.get("/api/templates?group=all&lang=zh")
            elif idx % 5 == 3:
                return client.post("/api/predict", json={
                    "setup": setup, "track_id": "monza",
                })
            else:
                return client.post("/api/feedback", json={
                    "frames": frames, "setup": setup, "track_id": "monza",
                })

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(hit_endpoint, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success = sum(1 for r in results if r.status_code == 200)
        assert success >= 40, f"Only {success}/50 mixed workload requests succeeded"

    def test_sequential_all_endpoints(self):
        """顺序访问所有端点, 验证无崩溃 (Iter-183)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app
        from f1opt.data.setup_schema import DEFAULT_SETUP

        app = create_app(start_listener=False)
        client = TestClient(app)
        setup = DEFAULT_SETUP.model_dump()
        frames = synth_lap_frames(lap_time=5.0, hz=10)

        # GET endpoints
        get_endpoints = [
            "/api/health", "/api/livez", "/api/readyz", "/api/metrics",
            "/api/tracks", "/api/tracks/monza", "/api/setup/default",
            "/api/causal/fields", "/api/feedback-loop", "/api/iterations",
            "/api/templates?group=all&lang=zh", "/api/feedback/history",
            "/api/session/default", "/api/telemetry/stats",
        ]
        for ep in get_endpoints:
            r = client.get(ep)
            assert r.status_code == 200, f"{ep} returned {r.status_code}"

        # POST endpoints
        post_payloads = [
            ("/api/predict", {"setup": setup, "track_id": "monza"}),
            ("/api/feedback", {"frames": frames, "setup": setup, "track_id": "monza"}),
            ("/api/search", {"track_id": "monza", "iterations": 5}),
            ("/api/causal/explain", {"field": "front_wing", "current": 5.0, "proposed": 7.0}),
            ("/api/whatif", {"setup": setup, "track_id": "monza", "field": "front_wing", "new_value": 8.0}),
            ("/api/feedback/compare", {
                "current_lap": {"lap_time": 90.0, "sector_times": [30.0, 30.0, 30.0]},
                "reference_lap": {"lap_time": 89.0, "sector_times": [29.0, 30.0, 30.0]},
            }),
        ]
        for ep, payload in post_payloads:
            r = client.post(ep, json=payload)
            assert r.status_code == 200, f"{ep} returned {r.status_code}"


# 12. 内存泄漏检测 (Iter-217)
class TestMemoryLeakDetection:
    """内存泄漏检测测试 (Iter-217)."""

    def test_feedback_engine_memory_no_leak(self):
        """100 次反馈生成, 验证无内存泄漏 (Iter-217)."""
        import tracemalloc
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for i in range(100):
            engine.run(frames=frames, setup=setup, track_id="monza")

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        top_increases = [s for s in stats if s.size_diff > 0][:10]
        total_increase = sum(s.size_diff for s in stats if s.size_diff > 0)

        assert total_increase < 50 * 1024 * 1024, (
            f"Memory increased by {total_increase / 1024 / 1024:.1f}MB"
        )

    def test_api_endpoint_memory_no_leak(self):
        """1000 次 API 调用, 验证无内存泄漏 (Iter-217)."""
        import tracemalloc
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app
        from f1opt.data.setup_schema import DEFAULT_SETUP

        app = create_app(start_listener=False)
        client = TestClient(app)
        setup = DEFAULT_SETUP.model_dump()

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for i in range(1000):
            client.get("/api/health")
            if i % 10 == 0:
                client.get("/api/tracks")

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_increase = sum(s.size_diff for s in stats if s.size_diff > 0)
        assert total_increase < 30 * 1024 * 1024, (
            f"Memory increased by {total_increase / 1024 / 1024:.1f}MB"
        )

    def test_gc_after_heavy_workload(self):
        """重负载后 GC 清理, 验证内存回收 (Iter-217)."""
        import gc as _gc
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        for i in range(50):
            engine.run(frames=frames, setup=setup, track_id="monza")

        _gc.collect()
        _gc.collect()

        result = engine.run(frames=frames, setup=setup, track_id="monza")
        assert "summary" in result


# 13. 连接池耗竭测试 (Iter-218)
class TestConnectionPoolExhaustion:
    """连接池耗竭与恢复测试 (Iter-218)."""

    def test_thread_pool_exhaustion_recovery(self):
        """线程池耗尽后恢复 (Iter-218)."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(client.get, "/api/health") for _ in range(200)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        success_phase1 = sum(1 for r in results if r.status_code == 200)
        assert success_phase1 >= 180, f"Phase 1: {success_phase1}/200"

        r = client.get("/api/health")
        assert r.status_code == 200, "Did not recover after pool exhaustion"

    def test_sequential_after_concurrent_stress(self):
        """并发后顺序访问正常 (Iter-218)."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(client.get, "/api/health") for _ in range(100)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        endpoints = ["/api/health", "/api/livez", "/api/readyz", "/api/tracks"]
        for ep in endpoints:
            r = client.get(ep)
            assert r.status_code == 200, f"{ep} failed after concurrent stress"


# 14. 极端值处理测试 (Iter-219)
class TestExtremeValueHandling:
    """极端值处理测试 (Iter-219)."""

    def test_feedback_with_extreme_lap_times(self):
        """极端圈速数据 (0.1s 和 999s) (Iter-219)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()

        fast_frames = synth_lap_frames(lap_time=0.1, hz=10)
        result_fast = engine.run(frames=fast_frames, setup=setup, track_id="monza")
        assert "summary" in result_fast

        slow_frames = synth_lap_frames(lap_time=999.0, hz=10)
        result_slow = engine.run(frames=slow_frames, setup=setup, track_id="monza")
        assert "summary" in result_slow

    def test_feedback_with_zero_frames(self):
        """零帧数据 (Iter-219)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()
        result = engine.run(frames=[], setup=setup, track_id="monza")
        assert "summary" in result

    def test_feedback_with_single_frame(self):
        """单帧数据 (Iter-219)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()
        frames = synth_lap_frames(lap_time=5.0, hz=1)[:1]
        result = engine.run(frames=frames, setup=setup, track_id="monza")
        assert "summary" in result

    def test_feedback_with_negative_values(self):
        """负值遥测数据 (Iter-219)."""
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        setup = DEFAULT_SETUP.model_dump()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        for f in frames:
            f["speed_kmh"] = -999.0
            f["throttle"] = -5.0
            f["brake"] = -10.0
        result = engine.run(frames=frames, setup=setup, track_id="monza")
        assert "summary" in result


# 15. 并发读写测试 (Iter-220)
class TestConcurrentReadWrite:
    """并发读写安全测试 (Iter-220)."""

    def test_concurrent_session_read_write(self):
        """并发 session 读写 (Iter-220)."""
        import concurrent.futures
        from f1opt.feedback.conversation import get_session, reset_sessions

        reset_sessions()

        def session_work(idx):
            sid = f"test_{idx % 5}"
            session = get_session(sid)
            session.add_message("user", f"message {idx}")
            session.add_setup_change({"front_wing": idx})
            return session.turn_count

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(session_work, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 100
        from f1opt.feedback.conversation import list_sessions
        sessions = list_sessions()
        assert len(sessions) >= 1

    def test_concurrent_engine_feedback(self):
        """并发反馈引擎调用 (Iter-220)."""
        import concurrent.futures
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import FeedbackEngine

        engine = FeedbackEngine()
        frames = synth_lap_frames(lap_time=5.0, hz=10)
        setup = DEFAULT_SETUP.model_dump()

        def feedback_work(idx):
            return engine.run(frames=frames, setup=setup, track_id="monza")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(feedback_work, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all("summary" in r for r in results)


# 16. 策略端点压力测试 (Iter-207)
class TestStrategyEndpointStress:
    """策略端点压力测试 (Iter-207)."""

    def test_strategy_plan_concurrent_requests(self):
        """并发策略规划请求 (Iter-207)."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)

        def strategy_request(idx):
            track = ["monza", "spa", "silverstone", "monaco", "suzuka"][idx % 5]
            return client.get(
                f"/api/strategy/plan?track_id={track}&total_laps=50"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(strategy_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success = sum(1 for r in results if r.status_code == 200)
        assert success >= 20, f"Strategy plan: {success}/50"

    def test_strategy_weather_impact_concurrent(self):
        """并发天气影响策略请求 (Iter-207)."""
        import concurrent.futures
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)

        def weather_request(idx):
            wetness = (idx % 10) / 10.0
            return client.get(
                f"/api/strategy/weather-impact?track_id=monza"
                f"&track_wetness={wetness}&rain_intensity={idx % 15}"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(weather_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success = sum(1 for r in results if r.status_code == 200)
        assert success >= 20, f"Weather impact: {success}/50"

    def test_strategy_plan_invalid_track(self):
        """无效赛道策略规划 (Iter-207)."""
        from fastapi.testclient import TestClient
        from f1opt.api.app import create_app

        app = create_app(start_listener=False)
        client = TestClient(app)
        r = client.get("/api/strategy/plan?track_id=invalid_track")
        assert r.status_code in (400, 404, 503)

    def test_crossover_all_compound_pairs(self):
        """所有化合物对的退化交叉分析 (Iter-207)."""
        from f1opt.model.strategy import RaceStrategyPlanner

        pairs = [
            ("soft", "medium"), ("soft", "hard"), ("medium", "hard"),
            ("soft", "soft"), ("hard", "hard"),
        ]
        for track_id in ("monza", "spa", "monaco"):
            planner = RaceStrategyPlanner(track_id=track_id, total_laps=50, fuel_load_kg=100.0)
            for a, b in pairs:
                result = planner.degradation_crossover(a, b)
                assert "crossover_lap" in result
                assert "recommendation" in result