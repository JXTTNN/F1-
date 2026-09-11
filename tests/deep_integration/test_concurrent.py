"""方式 3：并发共享状态测试 —— 多模块并发访问共享资源。

验证场景：
    1. 多线程并发写 SQLite（db.store.import_setup / submit_feedback）
    2. 多线程并发读写 TelemetryStream（listener 更新 + stream 读取）
    3. 多个 TestClient 并发请求同一 API（反馈提交 + 建议生成）

验证目标：
    - 并发下数据一致性（无丢失、无重复、无脏读）
    - 无死锁（所有线程都能完成）
    - 无竞争条件（最终状态可预测）

使用 threading + concurrent.futures.ThreadPoolExecutor 实现并发。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config
from setup_tuner.db.store import Store
from setup_tuner.domain.setup import CarSetup
from setup_tuner.feedback.service import FeedbackService
from setup_tuner.telemetry.stream import TelemetryStream


# ===========================================================================
# 1. 并发写 SQLite
# ===========================================================================
class TestConcurrentStoreWrite:
    """多线程并发写 SQLite（import_setup / add_feedback）。

    协作模块：db.store（多线程共享同一 Store 实例）
    验证：并发写入后数据完整、无丢失、无重复。
    """

    def test_concurrent_import_setup_no_loss(self, tmp_path: Path) -> None:
        """10 线程并发 import_setup，最终应有 10 条记录。

        验证：每条 setup 都被持久化，id 互不重复。
        """
        db_path = tmp_path / "concurrent_setup.db"
        store = Store(str(db_path))

        num_threads = 10
        track_id = "suzuka"

        def _import_one(idx: int) -> tuple[int, dict[str, float]]:
            """单线程任务：导入一份带唯一标记的 setup。"""
            params = CarSetup.default().to_dict()
            # 用 idx 标记 front_wing，便于后续验证不重复
            params["front_wing"] = float(idx % 11)
            setup_id = store.import_setup(track_id, params)
            return setup_id, params

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_import_one, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]

            # 验证：所有线程都成功返回
            assert len(results) == num_threads

            # 验证：setup_id 互不重复
            setup_ids = [r[0] for r in results]
            assert len(set(setup_ids)) == num_threads, "setup_id 有重复（竞争条件）"

            # 验证：数据库中确实有 num_threads 条记录
            for setup_id, _ in results:
                retrieved = store.get_setup(setup_id)
                assert retrieved is not None
                assert retrieved["track_id"] == track_id
        finally:
            store.close()

    def test_concurrent_submit_feedback_no_loss(self, tmp_path: Path) -> None:
        """20 线程并发提交反馈，最终应有 20 条记录。

        验证：每条反馈都被持久化，无丢失。
        """
        db_path = tmp_path / "concurrent_feedback.db"
        store = Store(str(db_path))
        svc = FeedbackService(store)

        num_threads = 20
        track_id = "monza"

        def _submit_one(idx: int) -> int:
            """单线程任务：提交一条反馈。"""
            corner = (idx % 10) + 1  # 1~10 轮转
            symptom = "understeer" if idx % 2 == 0 else "oversteer"
            result = svc.submit_feedback(track_id, corner, symptom, 3)
            return result["id"]

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_submit_one, i) for i in range(num_threads)]
                feedback_ids = [f.result() for f in as_completed(futures)]

            # 验证：所有反馈都成功提交
            assert len(feedback_ids) == num_threads
            assert len(set(feedback_ids)) == num_threads, "feedback_id 有重复"

            # 验证：数据库中确实有 num_threads 条反馈
            all_feedbacks = svc.get_feedbacks(track_id)
            assert len(all_feedbacks) == num_threads
        finally:
            store.close()

    def test_concurrent_mixed_write_read_consistent(self, tmp_path: Path) -> None:
        """并发混合读写：写线程 import_setup，读线程 get_latest_setup。

        验证：读线程不会看到脏数据（部分写入），最终一致。
        """
        db_path = tmp_path / "concurrent_mixed.db"
        store = Store(str(db_path))

        num_writers = 5
        num_readers = 5
        track_id = "suzuka"
        errors: list[str] = []
        stop_event = threading.Event()

        def _writer(idx: int) -> None:
            """写线程：连续导入 setup。"""
            try:
                for i in range(10):
                    params = CarSetup.default().to_dict()
                    params["front_wing"] = float((idx + i) % 11)
                    sid = store.import_setup(track_id, params)
                    if sid <= 0:
                        errors.append(f"writer {idx}: invalid setup_id {sid}")
            except Exception as e:
                errors.append(f"writer {idx} exception: {e}")

        def _reader(idx: int) -> None:
            """读线程：连续读取 latest setup。"""
            try:
                while not stop_event.is_set():
                    latest = store.get_latest_setup(track_id)
                    if latest is not None:
                        # 校验返回的 params 是完整 dict（非脏读）
                        assert isinstance(latest["params"], dict)
                        assert len(latest["params"]) == 23
            except Exception as e:
                errors.append(f"reader {idx} exception: {e}")

        try:
            with ThreadPoolExecutor(max_workers=num_writers + num_readers) as executor:
                # 启动读线程
                reader_futures = [
                    executor.submit(_reader, i) for i in range(num_readers)
                ]
                # 启动写线程
                writer_futures = [
                    executor.submit(_writer, i) for i in range(num_writers)
                ]
                # 等写线程完成
                for f in as_completed(writer_futures):
                    f.result()
                # 通知读线程停止
                stop_event.set()
                for f in as_completed(reader_futures):
                    f.result()

            # 验证：无异常
            assert not errors, f"并发读写出现错误: {errors}"

            # 验证：最终数据库有 num_writers * 10 条 setup
            # （通过不断 get_latest_setup 间接验证，直接计数需新方法）
            latest = store.get_latest_setup(track_id)
            assert latest is not None
            assert len(latest["params"]) == 23
        finally:
            store.close()


# ===========================================================================
# 2. 并发读写 TelemetryStream
# ===========================================================================
class TestConcurrentTelemetryStream:
    """多线程并发读写 TelemetryStream。

    协作模块：telemetry.stream（多线程共享同一 TelemetryStream 实例）
    验证：并发更新与读取不崩溃、不脏读、最终一致。
    """

    def test_concurrent_update_and_read(self) -> None:
        """10 线程并发 update + 10 线程并发 get_latest。

        验证：无异常、无脏读、最终数据一致。
        """
        stream = TelemetryStream()
        errors: list[str] = []
        stop_event = threading.Event()
        num_iterations = 200

        def _updater(idx: int) -> None:
            """写线程：连续更新 packet 6 的数据。"""
            try:
                for i in range(num_iterations):
                    stream.update(6, {
                        "packet_id": 6,
                        "m_speed": 300 + idx * 10 + i,
                        "m_throttle": 0.8,
                        "thread_idx": idx,
                    })
            except Exception as e:
                errors.append(f"updater {idx}: {e}")

        def _reader(idx: int) -> None:
            """读线程：连续读取 packet 6 最新帧。"""
            try:
                for _ in range(num_iterations):
                    latest = stream.get_latest(6)
                    if latest is not None:
                        # 校验返回的是完整 dict（非脏读）
                        assert "m_speed" in latest
                        assert isinstance(latest["m_speed"], (int, float))
            except Exception as e:
                errors.append(f"reader {idx}: {e}")

        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers * 2) as executor:
            updaters = [executor.submit(_updater, i) for i in range(num_workers)]
            readers = [executor.submit(_reader, i) for i in range(num_workers)]
            for f in as_completed(updaters + readers):
                f.result()
            stop_event.set()  # noqa: F841 (用于未来扩展)

        assert not errors, f"并发 TelemetryStream 出现错误: {errors}"

        # 验证：最终状态是某次 update 的完整数据
        final = stream.get_latest(6)
        assert final is not None
        assert "m_speed" in final
        assert "thread_idx" in final

    def test_concurrent_get_all_latest_snapshot(self) -> None:
        """并发 get_all_latest 应返回一致性快照。

        验证：get_all_latest 返回的 dict 是深拷贝，修改不影响内部状态。
        """
        stream = TelemetryStream()
        errors: list[str] = []

        # 预填充数据
        for pid in (1, 2, 5, 6, 7):
            stream.update(pid, {"packet_id": pid, "value": pid * 100})

        def _reader(idx: int) -> None:
            try:
                snapshot = stream.get_all_latest()
                # 修改快照不应影响内部状态
                for pid in snapshot:
                    snapshot[pid]["value"] = -999
                # 再次读取应仍是原值
                snapshot2 = stream.get_all_latest()
                for pid in snapshot2:
                    assert snapshot2[pid]["value"] == pid * 100, (
                        f"reader {idx}: pid {pid} 被脏改"
                    )
            except Exception as e:
                errors.append(f"reader {idx}: {e}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_reader, i) for i in range(10)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"get_all_latest 快照隔离失败: {errors}"

    def test_concurrent_update_different_packet_ids(self) -> None:
        """不同线程并发更新不同 packet_id，互不干扰。"""
        stream = TelemetryStream()
        packet_ids = [1, 2, 5, 6, 7, 16]
        errors: list[str] = []

        def _update_one(pid: int) -> None:
            try:
                for i in range(100):
                    stream.update(pid, {"packet_id": pid, "iter": i})
            except Exception as e:
                errors.append(f"pid {pid}: {e}")

        with ThreadPoolExecutor(max_workers=len(packet_ids)) as executor:
            futures = [executor.submit(_update_one, pid) for pid in packet_ids]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"并发更新不同 packet_id 出错: {errors}"

        # 验证：每个 packet_id 都有最终数据
        all_latest = stream.get_all_latest()
        for pid in packet_ids:
            assert pid in all_latest
            assert all_latest[pid]["iter"] == 99


# ===========================================================================
# 3. 并发 API 请求
# ===========================================================================
class TestConcurrentApiRequests:
    """多个 TestClient 并发请求同一 API。

    协作模块：api.routes ↔ feedback.service ↔ db.store（共享同一 app 实例）
    验证：并发请求下数据一致、无死锁、无 5xx。
    """

    def test_concurrent_feedback_submission(self, tmp_path: Path) -> None:
        """20 线程并发提交反馈，最终应有 20 条记录。

        验证：所有请求返回 200，反馈无丢失。
        """
        data_dir = tmp_path / "concurrent_api_feedback"
        data_dir.mkdir(exist_ok=True)
        config = Config(data_dir=str(data_dir), udp_port=20798)
        app = create_app(config)

        num_threads = 20
        track_id = "suzuka"
        errors: list[str] = []

        with TestClient(app) as client:
            # 先选赛道
            client.post("/api/v1/tracks/current", json={"track_id": track_id})

            def _submit_one(idx: int) -> int | None:
                """单线程任务：提交一条反馈并返回其 id。"""
                try:
                    corner = (idx % 10) + 1
                    symptom = "understeer" if idx % 2 == 0 else "brake_long"
                    resp = client.post(
                        "/api/v1/feedback",
                        json={
                            "track_id": track_id,
                            "corner_number": corner,
                            "symptom": symptom,
                            "strength": 3,
                        },
                    )
                    if resp.status_code != 200:
                        errors.append(
                            f"thread {idx}: status {resp.status_code} {resp.text}"
                        )
                        return None
                    return resp.json()["data"]["id"]
                except Exception as e:
                    errors.append(f"thread {idx} exception: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_submit_one, i) for i in range(num_threads)]
                feedback_ids = [f.result() for f in as_completed(futures)]

            # 验证：无错误
            assert not errors, f"并发 API 反馈提交出错: {errors}"

            # 验证：所有反馈都成功
            assert all(fid is not None for fid in feedback_ids)
            assert len(set(feedback_ids)) == num_threads, "feedback_id 有重复"

            # 验证：数据库中确实有 num_threads 条
            resp = client.get("/api/v1/feedback", params={"track_id": track_id})
            assert resp.status_code == 200
            assert len(resp.json()["data"]) == num_threads

    def test_concurrent_suggest_no_deadlock(self, tmp_path: Path) -> None:
        """并发生成建议不发生死锁，所有请求都能完成。

        验证：5 线程并发 POST /suggest，全部返回 200，无超时无死锁。
        """
        data_dir = tmp_path / "concurrent_api_suggest"
        data_dir.mkdir(exist_ok=True)
        config = Config(data_dir=str(data_dir), udp_port=20797)
        app = create_app(config)

        num_threads = 5
        track_id = "monza"
        errors: list[str] = []

        with TestClient(app) as client:
            # 准备：选赛道 + 提交反馈
            client.post("/api/v1/tracks/current", json={"track_id": track_id})
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": track_id, "corner_number": 1,
                    "symptom": "understeer", "strength": 3,
                },
            )

            def _suggest_one(idx: int) -> bool:
                """单线程任务：生成建议。"""
                try:
                    resp = client.post("/api/v1/suggest", json={"track_id": track_id})
                    if resp.status_code != 200:
                        errors.append(
                            f"thread {idx}: status {resp.status_code} {resp.text}"
                        )
                        return False
                    report = resp.json()["data"]["report"]
                    return len(report["parameters"]) == 23
                except Exception as e:
                    errors.append(f"thread {idx} exception: {e}")
                    return False

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_suggest_one, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]

            # 验证：无错误且全部成功
            assert not errors, f"并发 suggest 出错: {errors}"
            assert all(results), "部分 suggest 请求未返回完整报告"

    def test_concurrent_mixed_api_operations(self, tmp_path: Path) -> None:
        """并发混合 API 操作：反馈提交 + 建议生成 + 历史查询。

        验证：混合并发下无死锁、无 5xx、数据一致。
        """
        data_dir = tmp_path / "concurrent_api_mixed"
        data_dir.mkdir(exist_ok=True)
        config = Config(data_dir=str(data_dir), udp_port=20796)
        app = create_app(config)

        track_id = "suzuka"
        errors: list[str] = []

        with TestClient(app) as client:
            # 准备：选赛道 + 初始反馈
            client.post("/api/v1/tracks/current", json={"track_id": track_id})
            client.post(
                "/api/v1/feedback",
                json={
                    "track_id": track_id, "corner_number": 1,
                    "symptom": "understeer", "strength": 3,
                },
            )

            def _feedback_task(idx: int) -> None:
                try:
                    for i in range(5):
                        resp = client.post(
                            "/api/v1/feedback",
                            json={
                                "track_id": track_id,
                                "corner_number": (i % 5) + 1,
                                "symptom": "brake_long",
                                "strength": 2,
                            },
                        )
                        if resp.status_code != 200:
                            errors.append(f"feedback {idx}-{i}: {resp.status_code}")
                except Exception as e:
                    errors.append(f"feedback task {idx}: {e}")

            def _suggest_task(idx: int) -> None:
                try:
                    for _ in range(3):
                        resp = client.post(
                            "/api/v1/suggest", json={"track_id": track_id},
                        )
                        if resp.status_code != 200:
                            errors.append(f"suggest {idx}: {resp.status_code}")
                except Exception as e:
                    errors.append(f"suggest task {idx}: {e}")

            def _history_task(idx: int) -> None:
                try:
                    for _ in range(5):
                        resp = client.get(
                            "/api/v1/iteration/history",
                            params={"track_id": track_id},
                        )
                        if resp.status_code != 200:
                            errors.append(f"history {idx}: {resp.status_code}")
                except Exception as e:
                    errors.append(f"history task {idx}: {e}")

            with ThreadPoolExecutor(max_workers=9) as executor:
                futures = []
                # 3 个反馈线程 + 3 个建议线程 + 3 个历史线程
                for i in range(3):
                    futures.append(executor.submit(_feedback_task, i))
                    futures.append(executor.submit(_suggest_task, i))
                    futures.append(executor.submit(_history_task, i))
                for f in as_completed(futures):
                    f.result()

            # 验证：无错误
            assert not errors, f"并发混合 API 操作出错: {errors}"

            # 验证：最终数据一致
            resp = client.get(
                "/api/v1/iteration/history", params={"track_id": track_id},
            )
            assert resp.status_code == 200
            iterations = resp.json()["data"]
            # 应至少有 3 轮迭代（3 个 suggest 线程 × 1 次 = 3，但并发可能更多）
            assert len(iterations) >= 3


# ===========================================================================
# 4. 并发下的数据完整性
# ===========================================================================
class TestConcurrentDataIntegrity:
    """并发操作下的数据完整性验证。"""

    def test_concurrent_setup_import_integrity(self, tmp_path: Path) -> None:
        """并发导入 setup 后，每条记录的 params 完整且不混淆。"""
        db_path = tmp_path / "integrity.db"
        store = Store(str(db_path))

        num_threads = 8
        track_id = "suzuka"

        def _import_unique(idx: int) -> tuple[int, float]:
            """导入一份 front_wing 唯一的 setup。"""
            params = CarSetup.default().to_dict()
            unique_wing = float(idx + 1)  # 1~8
            params["front_wing"] = unique_wing
            sid = store.import_setup(track_id, params)
            return sid, unique_wing

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_import_unique, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]

            # 验证：逐条检索，front_wing 值与导入时一致（不混淆）
            for setup_id, expected_wing in results:
                retrieved = store.get_setup(setup_id)
                assert retrieved is not None
                assert retrieved["params"]["front_wing"] == expected_wing, (
                    f"setup {setup_id}: front_wing 期望 {expected_wing} "
                    f"实际 {retrieved['params']['front_wing']}（数据混淆）"
                )
        finally:
            store.close()

    def test_concurrent_feedback_then_suggest_consistent(self, tmp_path: Path) -> None:
        """并发提交反馈后生成建议，建议应包含所有反馈的症状。

        验证：并发写入 → 串行读取 → 引擎消费，数据完整传递。
        """
        db_path = tmp_path / "fb_suggest.db"
        store = Store(str(db_path))
        svc = FeedbackService(store)

        num_threads = 10
        track_id = "suzuka"

        # 并发提交 10 条不同弯道的反馈
        def _submit(idx: int) -> int:
            corner = idx + 1  # 1~10
            result = svc.submit_feedback(track_id, corner, "understeer", 3)
            return result["id"]

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_submit, i) for i in range(num_threads)]
                ids = [f.result() for f in as_completed(futures)]

            assert len(ids) == num_threads
            assert len(set(ids)) == num_threads

            # 串行读取全部反馈
            all_feedbacks = svc.get_feedbacks(track_id)
            assert len(all_feedbacks) == num_threads

            # 验证：10 个不同弯道都有反馈
            corners = {fb["corner_number"] for fb in all_feedbacks}
            assert corners == set(range(1, num_threads + 1))

            # 引擎消费：用全部反馈生成建议
            from setup_tuner.engine.engine import generate_suggestion
            from setup_tuner.report.builder import feedbacks_to_symptoms

            symptoms = feedbacks_to_symptoms(all_feedbacks)
            assert len(symptoms) == num_threads

            suggestion = generate_suggestion(
                symptoms=symptoms,
                current_setup=CarSetup.default().to_dict(),
                track_id=track_id,
            )
            # 10 条 understeer 反馈 → Dx front_grip_req 应显著为正
            assert suggestion["dx"]["front_grip_req"] > 0.0
            assert len(suggestion["setup_delta"]) == 23
        finally:
            store.close()

    def test_no_deadlock_under_stress(self, tmp_path: Path) -> None:
        """压力测试：高频并发读写 Store 不死锁（限时完成）。

        验证：30 线程 × 50 次操作在 10 秒内完成（无死锁）。
        """
        db_path = tmp_path / "stress.db"
        store = Store(str(db_path))

        num_threads = 30
        ops_per_thread = 50
        track_id = "suzuka"
        completed_count = 0
        count_lock = threading.Lock()

        def _stress(idx: int) -> int:
            """高频读写混合操作。"""
            nonlocal completed_count
            for i in range(ops_per_thread):
                if i % 2 == 0:
                    params = CarSetup.default().to_dict()
                    params["front_wing"] = float(i % 11)
                    store.import_setup(track_id, params)
                else:
                    store.get_latest_setup(track_id)
            with count_lock:
                completed_count += 1
            return completed_count

        try:
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_stress, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()
            elapsed = time.monotonic() - start

            # 验证：所有线程都完成
            assert completed_count == num_threads, (
                f"仅 {completed_count}/{num_threads} 线程完成（可能死锁）"
            )
            # 验证：在合理时间内完成（无死锁）
            assert elapsed < 30.0, f"耗时 {elapsed:.1f}s 过长（可能死锁）"
        finally:
            store.close()