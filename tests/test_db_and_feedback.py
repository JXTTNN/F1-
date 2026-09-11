"""T5 数据层 + 反馈模块 单元测试。

覆盖验收标准：
1. schema.sql 6 张表 DDL 正确
2. Store CRUD 全部可用
3. 反馈可写入/查询
4. 未点击弯道不生成记录（get_normal_corners 正确返回无反馈的弯道）
5. 无反馈时请求建议提示正确（validate_before_suggest 返回 (False, 引导消息)）
6. 迭代历史可记录与查询
"""

from __future__ import annotations

import sqlite3

import pytest

from setup_tuner.db.store import Store
from setup_tuner.feedback.iteration import IterationService
from setup_tuner.feedback.service import FeedbackService


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store() -> Store:
    """每个测试用独立的内存 SQLite 库。"""
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def feedback_service(store: Store) -> FeedbackService:
    return FeedbackService(store)


@pytest.fixture
def iteration_service(store: Store) -> IterationService:
    return IterationService(store)


# 样例调教参数（23 项子集即可，store 不校验参数完整性）
SAMPLE_PARAMS = {
    "front_wing": 5.0,
    "rear_wing": 5.0,
    "brake_pressure": 75.0,
    "front_tyre_pressure": 25.5,
}


# ===========================================================================
# 1. schema.sql 6 张表 DDL 正确
# ===========================================================================
class TestSchema:
    def test_six_tables_created(self, store: Store) -> None:
        """建表后应存在 6 张核心表。"""
        rows = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in rows}
        expected = {"track", "corner", "setup", "feedback", "suggestion", "iteration"}
        assert expected.issubset(table_names), f"缺失表: {expected - table_names}"

    def test_feedback_strength_check_constraint(self, store: Store) -> None:
        """feedback.strength 应有 CHECK 约束 0-5。"""
        store._conn.execute(
            "INSERT INTO feedback (track_id, symptom, category, strength, created_at) "
            "VALUES ('t1', 'understeer', 'entry', 3, '2026-01-01T00:00:00Z')"
        )
        store._conn.commit()
        # 越界应被拒绝
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO feedback (track_id, symptom, category, strength, created_at) "
                "VALUES ('t1', 'understeer', 'entry', 6, '2026-01-01T00:00:00Z')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO feedback (track_id, symptom, category, strength, created_at) "
                "VALUES ('t1', 'understeer', 'entry', -1, '2026-01-01T00:00:00Z')"
            )

    def test_corner_primary_key_composite(self, store: Store) -> None:
        """corner 表主键为 (track_id, corner_number) 复合键。"""
        store.upsert_track("suzuka", "铃鹿", "Suzuka", "mixed", 5807.0, 18, "tracks/suzuka.svg")
        store.upsert_corner("suzuka", 1, "slow", 0.5, 0.5)
        # 同 track_id 同 corner_number 应冲突
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO corner (track_id, corner_number, corner_type, anchor_x, anchor_y) "
                "VALUES ('suzuka', 1, 'fast', 0.6, 0.6)"
            )


# ===========================================================================
# 2. Store CRUD 全部可用
# ===========================================================================
class TestStoreCrud:
    def test_setup_import_and_get(self, store: Store) -> None:
        """调教快照导入与读取。"""
        sid = store.import_setup("suzuka", SAMPLE_PARAMS)
        assert sid > 0
        got = store.get_setup(sid)
        assert got is not None
        assert got["track_id"] == "suzuka"
        assert got["params"] == SAMPLE_PARAMS

    def test_get_setup_not_found(self, store: Store) -> None:
        """不存在的 setup_id 返回 None。"""
        assert store.get_setup(99999) is None

    def test_get_latest_setup(self, store: Store) -> None:
        """获取赛道最新调教（按 imported_at 降序）。"""
        _sid1 = store.import_setup("suzuka", {"front_wing": 1.0})
        sid2 = store.import_setup("suzuka", {"front_wing": 2.0})
        latest = store.get_latest_setup("suzuka")
        assert latest is not None
        assert latest["id"] == sid2
        # 另一赛道不影响
        store.import_setup("monza", {"front_wing": 9.0})
        assert store.get_latest_setup("suzuka")["id"] == sid2

    def test_get_latest_setup_empty(self, store: Store) -> None:
        """无调教时返回 None。"""
        assert store.get_latest_setup("nope") is None

    def test_track_upsert_idempotent(self, store: Store) -> None:
        """赛道 upsert 幂等。"""
        store.upsert_track("suzuka", "铃鹿", "Suzuka", "mixed", 5807.0, 18, "tracks/suzuka.svg")
        store.upsert_track("suzuka", "铃鹿国际", "Suzuka", "mixed", 5807.0, 18, "tracks/suzuka.svg", 7)
        track = store.get_track("suzuka")
        assert track is not None
        assert track["official_name"] == "铃鹿国际"
        assert track["udp_track_id"] == 7

    def test_corner_upsert_and_query(self, store: Store) -> None:
        """弯道 upsert 与查询。"""
        store.upsert_track("suzuka", "铃鹿", "Suzuka", "mixed", 5807.0, 18, "tracks/suzuka.svg")
        store.upsert_corner("suzuka", 1, "slow", 0.1, 0.2, name="S1", speed_kmh=80.0)
        store.upsert_corner("suzuka", 2, "fast", 0.3, 0.4, name="S2", speed_kmh=200.0)
        corners = store.get_corners("suzuka")
        assert len(corners) == 2
        assert corners[0]["corner_number"] == 1
        assert corners[1]["corner_number"] == 2

    def test_suggestion_save_and_get_latest(self, store: Store) -> None:
        """建议保存与最新查询。"""
        sid1 = store.save_suggestion("suzuka", '{"delta": 1}')
        sid2 = store.save_suggestion("suzuka", '{"delta": 2}')
        assert sid1 > 0 and sid2 > 0
        latest = store.get_latest_suggestion("suzuka")
        assert latest is not None
        assert latest["report_json"] == '{"delta": 2}'

    def test_get_latest_suggestion_empty(self, store: Store) -> None:
        assert store.get_latest_suggestion("nope") is None


# ===========================================================================
# 3. 反馈可写入/查询
# ===========================================================================
class TestFeedbackWrite:
    def test_submit_and_get_feedback(self, feedback_service: FeedbackService) -> None:
        """反馈写入后可查询。"""
        rec = feedback_service.submit_feedback("suzuka", 3, "understeer", strength=4)
        assert rec["symptom"] == "understeer"
        assert rec["category"] == "entry"
        assert rec["strength"] == 4
        assert rec["corner_number"] == 3

        feedbacks = feedback_service.get_feedbacks("suzuka")
        assert len(feedbacks) == 1
        assert feedbacks[0]["symptom"] == "understeer"

    def test_submit_global_symptom(self, feedback_service: FeedbackService) -> None:
        """全局症状 corner_number=None。"""
        rec = feedback_service.submit_feedback("suzuka", None, "bottoming", strength=5)
        assert rec["corner_number"] is None
        assert rec["category"] == "global"

    def test_invalid_symptom_rejected(self, feedback_service: FeedbackService) -> None:
        """非法症状标识被拒绝。"""
        with pytest.raises(ValueError, match="未知症状标识"):
            feedback_service.submit_feedback("suzuka", 1, "not_a_symptom")

    def test_invalid_strength_rejected(self, feedback_service: FeedbackService) -> None:
        """越界强度被拒绝。"""
        with pytest.raises(ValueError, match="越界"):
            feedback_service.submit_feedback("suzuka", 1, "understeer", strength=6)
        with pytest.raises(ValueError, match="越界"):
            feedback_service.submit_feedback("suzuka", 1, "understeer", strength=-1)

    def test_default_strength_is_three(self, feedback_service: FeedbackService) -> None:
        """默认强度为 3。"""
        rec = feedback_service.submit_feedback("suzuka", 1, "understeer")
        assert rec["strength"] == 3

    def test_get_corner_feedbacks_grouped(self, feedback_service: FeedbackService) -> None:
        """按弯道分组查询。"""
        feedback_service.submit_feedback("suzuka", 1, "understeer")
        feedback_service.submit_feedback("suzuka", 1, "oversteer")
        feedback_service.submit_feedback("suzuka", 3, "lockup")
        feedback_service.submit_feedback("suzuka", None, "bottoming")

        grouped = feedback_service.get_corner_feedbacks("suzuka")
        assert len(grouped[1]) == 2
        assert len(grouped[3]) == 1
        assert len(grouped[None]) == 1


# ===========================================================================
# 4. 未点击弯道不生成记录（get_normal_corners 正确返回无反馈的弯道）
# ===========================================================================
class TestNormalCorners:
    def test_all_normal_when_no_feedback(self, feedback_service: FeedbackService) -> None:
        """无反馈时全部弯道正常。"""
        normal = feedback_service.get_normal_corners("suzuka", total_corners=5)
        assert normal == [1, 2, 3, 4, 5]

    def test_clicked_corners_excluded(self, feedback_service: FeedbackService) -> None:
        """有反馈的弯道从正常列表中排除。"""
        feedback_service.submit_feedback("suzuka", 2, "understeer")
        feedback_service.submit_feedback("suzuka", 4, "oversteer")
        normal = feedback_service.get_normal_corners("suzuka", total_corners=5)
        # 弯道 2、4 有反馈，应排除
        assert normal == [1, 3, 5]

    def test_global_symptom_does_not_affect_normal(self, feedback_service: FeedbackService) -> None:
        """全局症状（corner_number=None）不影响某弯道是否正常。"""
        feedback_service.submit_feedback("suzuka", None, "bottoming")
        normal = feedback_service.get_normal_corners("suzuka", total_corners=5)
        # 全局症状不绑定具体弯道，所有弯道仍正常
        assert normal == [1, 2, 3, 4, 5]

    def test_no_record_for_unclicked(self, feedback_service: FeedbackService, store: Store) -> None:
        """未点击弯道在 feedback 表中无记录。"""
        feedback_service.submit_feedback("suzuka", 2, "understeer")
        all_feedbacks = store.get_feedbacks("suzuka")
        clicked_corners = {fb["corner_number"] for fb in all_feedbacks}
        # 只有弯道 2 有记录，弯道 1/3/4/5 无记录
        assert clicked_corners == {2}


# ===========================================================================
# 5. 无反馈时请求建议提示正确
# ===========================================================================
class TestValidateBeforeSuggest:
    def test_no_feedback_returns_false_with_hint(self, feedback_service: FeedbackService) -> None:
        """无反馈时返回 (False, 引导消息)。"""
        ok, msg = feedback_service.validate_before_suggest("suzuka")
        assert ok is False
        assert "未发现反馈" in msg
        assert "点击赛道图" in msg

    def test_with_feedback_returns_true(self, feedback_service: FeedbackService) -> None:
        """有反馈时返回 (True, "")。"""
        feedback_service.submit_feedback("suzuka", 1, "understeer")
        ok, msg = feedback_service.validate_before_suggest("suzuka")
        assert ok is True
        assert msg == ""

    def test_global_symptom_satisfies_validation(self, feedback_service: FeedbackService) -> None:
        """全局症状也满足校验。"""
        feedback_service.submit_feedback("suzuka", None, "lap_slow")
        ok, msg = feedback_service.validate_before_suggest("suzuka")
        assert ok is True

    def test_has_feedback_store_method(self, store: Store) -> None:
        """Store.has_feedback 正确反映反馈存在性。"""
        assert store.has_feedback("suzuka") is False
        store.add_feedback("suzuka", 1, "understeer", "entry", 3)
        assert store.has_feedback("suzuka") is True


# ===========================================================================
# 6. 迭代历史可记录与查询
# ===========================================================================
class TestIteration:
    def test_create_and_get_history(self, iteration_service: IterationService, store: Store) -> None:
        """迭代记录创建与查询。"""
        before_id = store.import_setup("suzuka", {"front_wing": 1.0})
        after_id = store.import_setup("suzuka", {"front_wing": 2.0})
        sug_id = store.save_suggestion("suzuka", '{"delta": 1}')

        it_id = iteration_service.create_iteration(
            "suzuka", before_setup_id=before_id, after_setup_id=after_id, suggestion_id=sug_id
        )
        assert it_id > 0

        history = iteration_service.get_history("suzuka")
        assert len(history) == 1
        assert history[0]["round_no"] == 1
        assert history[0]["before_setup_id"] == before_id
        assert history[0]["after_setup_id"] == after_id

    def test_round_auto_increment(self, iteration_service: IterationService) -> None:
        """轮次号自动递增。"""
        iteration_service.create_iteration("suzuka", None, None, None)
        iteration_service.create_iteration("suzuka", None, None, None)
        iteration_service.create_iteration("suzuka", None, None, None)
        history = iteration_service.get_history("suzuka")
        assert [it["round_no"] for it in history] == [1, 2, 3]

    def test_get_latest_round_empty(self, iteration_service: IterationService) -> None:
        """无历史时最新轮次为 0。"""
        assert iteration_service.get_latest_round("suzuka") == 0

    def test_get_latest_round_with_history(self, iteration_service: IterationService) -> None:
        """有历史时返回最大轮次号。"""
        iteration_service.create_iteration("suzuka", None, None, None)
        iteration_service.create_iteration("suzuka", None, None, None)
        assert iteration_service.get_latest_round("suzuka") == 2

    def test_compare_setups(self, iteration_service: IterationService) -> None:
        """前后调教对比。"""
        before = {"front_wing": 5.0, "rear_wing": 5.0, "brake_pressure": 75.0}
        after = {"front_wing": 7.0, "rear_wing": 5.0, "brake_pressure": 70.0}
        result = IterationService.compare_setups(before, after)
        assert result["changed_count"] == 2
        assert result["total_params"] == 3
        assert result["unchanged_count"] == 1
        change_names = {c["name"] for c in result["changes"]}
        assert change_names == {"front_wing", "brake_pressure"}
        # 检查 delta 计算
        for c in result["changes"]:
            if c["name"] == "front_wing":
                assert c["delta"] == 2.0
            if c["name"] == "brake_pressure":
                assert c["delta"] == -5.0

    def test_compare_setups_no_change(self, iteration_service: IterationService) -> None:
        """相同调教对比无变化。"""
        params = {"front_wing": 5.0, "rear_wing": 5.0}
        result = IterationService.compare_setups(params, params)
        assert result["changed_count"] == 0
        assert result["unchanged_count"] == 2


# ===========================================================================
# 线程安全
# ===========================================================================
class TestThreadSafety:
    def test_concurrent_writes(self) -> None:
        """多线程并发写入不报错（Lock 串行化）。"""
        import threading

        s = Store(":memory:")
        errors: list[Exception] = []

        def worker():
            try:
                for i in range(20):
                    s.import_setup("suzuka", {"front_wing": float(i)})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        s.close()
        assert errors == []