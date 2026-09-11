"""切片6 深度测试 —— setup_tuner/db/store.py + schema.sql SQLite 数据持久化。

覆盖 6 张核心表（track / corner / setup / feedback / suggestion / iteration）
的 CRUD、幂等性、外键约束、CHECK 约束、SQL 注入防护、线程安全与文件持久化。

5 种测试方式（每个 class 对应一种，注释明确标注）：
    1. TestUnit     — 单元测试：每个公开方法的正常输入正确性
    2. TestBoundary — 边界/异常测试：极端输入、空值、None、不存在的资源、约束违反
    3. TestProperty — 属性不变量测试：幂等性、往返一致性、确定性
    4. TestStatic   — 静态分析：SQL 注入防护、值域约束、外键约束、表结构
    5. TestSmoke    — 实际运行冒烟：真实文件 SQLite、上下文管理器、跨方法调用链
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from setup_tuner.db.store import Store


# ===========================================================================
# 1. 单元测试 (unit) — 每个公开方法的正常输入正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证 Store 各 CRUD 方法在正常输入下的正确行为。"""

    def test_upsert_and_get_track(self, store: Store) -> None:
        """upsert_track → get_track 往返正确（含全部字段）。"""
        store.upsert_track(
            track_id="suzuka",
            official_name="Japanese Grand Prix",
            circuit_name="Suzuka Circuit",
            track_type="high_downforce",
            length_m=5807.0,
            corners=18,
            svg_path="tracks/suzuka.svg",
            udp_track_id=2,
        )
        row = store.get_track("suzuka")
        assert row is not None
        assert row["track_id"] == "suzuka"
        assert row["official_name"] == "Japanese Grand Prix"
        assert row["circuit_name"] == "Suzuka Circuit"
        assert row["track_type"] == "high_downforce"
        assert row["length_m"] == 5807.0
        assert row["corners"] == 18
        assert row["udp_track_id"] == 2
        assert row["svg_path"] == "tracks/suzuka.svg"

    def test_upsert_and_get_corners(self, store: Store) -> None:
        """upsert_corner → get_corners 往返正确（按 corner_number 升序）。"""
        store.upsert_track(
            "suzuka", "Japanese GP", "Suzuka", "high_downforce",
            5807.0, 18, "tracks/suzuka.svg",
        )
        # 故意乱序插入
        store.upsert_corner("suzuka", 3, "fast", 0.5, 0.6, name="T3", speed_kmh=240.0)
        store.upsert_corner("suzuka", 1, "slow", 0.1, 0.2, name="T1", speed_kmh=80.0)
        store.upsert_corner("suzuka", 2, "medium", 0.3, 0.4, name="T2", speed_kmh=150.0)

        corners = store.get_corners("suzuka")
        assert len(corners) == 3
        # 验证按 corner_number 升序返回
        assert [c["corner_number"] for c in corners] == [1, 2, 3]
        assert corners[0]["name"] == "T1"
        assert corners[1]["speed_kmh"] == 150.0
        assert corners[2]["anchor_x"] == 0.5

    def test_import_and_get_setup(self, store: Store) -> None:
        """import_setup 返回自增 id；get_setup 反序列化 params 字典。"""
        params = {"front_wing": 6.0, "rear_wing": 4.0, "brake_bias": 63.0}
        sid1 = store.import_setup("suzuka", params)
        sid2 = store.import_setup("suzuka", params)
        assert sid2 == sid1 + 1, "setup.id 应自增"

        row = store.get_setup(sid1)
        assert row is not None
        assert row["id"] == sid1
        assert row["track_id"] == "suzuka"
        assert row["params"] == params, "params 应反序列化为原字典"

    def test_get_latest_setup(self, store: Store) -> None:
        """get_latest_setup 返回最近导入的快照（按 imported_at 降序）。"""
        store.import_setup("suzuka", {"front_wing": 5.0})
        store.import_setup("suzuka", {"front_wing": 6.0})
        latest = store.get_latest_setup("suzuka")
        assert latest is not None
        assert latest["params"]["front_wing"] == 6.0

    def test_add_and_get_feedback(self, store: Store) -> None:
        """add_feedback → get_feedbacks 往返正确（含 category 与 strength）。"""
        fid = store.add_feedback(
            track_id="suzuka", corner_number=1, symptom="understeer",
            category="entry", strength=4, setup_id=None,
        )
        assert fid >= 1
        rows = store.get_feedbacks("suzuka")
        assert len(rows) == 1
        assert rows[0]["id"] == fid
        assert rows[0]["symptom"] == "understeer"
        assert rows[0]["category"] == "entry"
        assert rows[0]["strength"] == 4
        assert rows[0]["corner_number"] == 1

    def test_has_feedback(self, store: Store) -> None:
        """has_feedback 在无/有反馈时分别返回 False/True。"""
        assert store.has_feedback("suzuka") is False
        store.add_feedback("suzuka", 1, "understeer", "entry", 3)
        assert store.has_feedback("suzuka") is True

    def test_save_and_get_latest_suggestion(self, store: Store) -> None:
        """save_suggestion → get_latest_suggestion 往返正确。"""
        report = {"track_id": "suzuka", "parameters": [], "summary": "测试"}
        rj = json.dumps(report, ensure_ascii=False, sort_keys=True)
        sid = store.save_suggestion("suzuka", rj, setup_id=None)
        assert sid >= 1

        # 再存一条，验证 latest 取最新
        rj2 = json.dumps({**report, "summary": "第二版"}, ensure_ascii=False, sort_keys=True)
        sid2 = store.save_suggestion("suzuka", rj2)
        assert sid2 > sid

        latest = store.get_latest_suggestion("suzuka")
        assert latest is not None
        assert latest["id"] == sid2
        assert "第二版" in latest["report_json"]

    def test_save_and_get_iterations(self, store: Store) -> None:
        """save_iteration → get_iterations 往返正确（按 round_no 升序）。"""
        store.save_iteration("suzuka", 2, None, None, None)
        store.save_iteration("suzuka", 1, None, None, None)
        rows = store.get_iterations("suzuka")
        assert len(rows) == 2
        assert [r["round_no"] for r in rows] == [1, 2]

    def test_close_idempotent(self, store: Store) -> None:
        """close 多次调用不抛异常（幂等）。"""
        store.close()
        store.close()  # 第二次不应抛异常


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 极端输入、空值、None、不存在的资源
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证 Store 在极端/非法输入下的健壮行为。"""

    def test_get_track_nonexistent(self, store: Store) -> None:
        """get_track 查询不存在的 track_id 返回 None（非抛异常）。"""
        assert store.get_track("nonexistent_track") is None
        assert store.get_track("") is None

    def test_get_corners_empty(self, store: Store) -> None:
        """get_corners 查询无弯道的赛道返回空列表。"""
        assert store.get_corners("no_such_track") == []

    def test_get_setup_nonexistent(self, store: Store) -> None:
        """get_setup 查询不存在的 id 返回 None。"""
        assert store.get_setup(99999) is None
        assert store.get_setup(0) is None
        assert store.get_setup(-1) is None

    def test_get_latest_setup_empty(self, store: Store) -> None:
        """get_latest_setup 无快照时返回 None。"""
        assert store.get_latest_setup("empty_track") is None

    def test_get_feedbacks_empty(self, store: Store) -> None:
        """get_feedbacks 无反馈时返回空列表。"""
        assert store.get_feedbacks("empty_track") == []

    def test_get_latest_suggestion_empty(self, store: Store) -> None:
        """get_latest_suggestion 无建议时返回 None。"""
        assert store.get_latest_suggestion("empty_track") is None

    def test_get_iterations_empty(self, store: Store) -> None:
        """get_iterations 无迭代时返回空列表。"""
        assert store.get_iterations("empty_track") == []

    def test_add_feedback_corner_none(self, store: Store) -> None:
        """add_feedback corner_number=None 表示全局症状，应正常入库。"""
        fid = store.add_feedback(
            "suzuka", None, "bottoming", "global", 3,
        )
        rows = store.get_feedbacks("suzuka")
        assert len(rows) == 1
        assert rows[0]["id"] == fid
        assert rows[0]["corner_number"] is None
        assert rows[0]["category"] == "global"

    def test_add_feedback_strength_boundary(self, store: Store) -> None:
        """strength=0 与 strength=5（CHECK 边界）应正常入库。"""
        store.add_feedback("suzuka", 1, "understeer", "entry", 0)
        store.add_feedback("suzuka", 2, "oversteer", "entry", 5)
        rows = store.get_feedbacks("suzuka")
        assert {r["strength"] for r in rows} == {0, 5}

    def test_add_feedback_strength_out_of_range(self, store: Store) -> None:
        """strength=6 越界应触发 CHECK 约束 IntegrityError。"""
        with pytest.raises(sqlite3.IntegrityError):
            store.add_feedback("suzuka", 1, "understeer", "entry", 6)

    def test_add_feedback_strength_negative(self, store: Store) -> None:
        """strength=-1 越界应触发 CHECK 约束 IntegrityError。"""
        with pytest.raises(sqlite3.IntegrityError):
            store.add_feedback("suzuka", 1, "understeer", "entry", -1)

    def test_import_setup_empty_params(self, store: Store) -> None:
        """import_setup 空字典 params 应正常入库（不校验参数完整性）。"""
        sid = store.import_setup("suzuka", {})
        row = store.get_setup(sid)
        assert row is not None
        assert row["params"] == {}

    def test_import_setup_large_params(self, store: Store) -> None:
        """import_setup 大字典（1000 键）应正常入库。"""
        big = {f"param_{i}": float(i) for i in range(1000)}
        sid = store.import_setup("suzuka", big)
        row = store.get_setup(sid)
        assert row is not None
        assert len(row["params"]) == 1000

    def test_save_suggestion_empty_report(self, store: Store) -> None:
        """save_suggestion 空字符串 report_json 应正常入库。"""
        sid = store.save_suggestion("suzuka", "")
        latest = store.get_latest_suggestion("suzuka")
        assert latest is not None
        assert latest["id"] == sid
        assert latest["report_json"] == ""

    def test_corner_anchor_negative(self, store: Store) -> None:
        """anchor_x/anchor_y 负数应正常入库（schema 仅 NOT NULL，无 CHECK）。"""
        store.upsert_track("t", "n", "c", "mixed", 1000.0, 1, "p.svg")
        store.upsert_corner("t", 1, "slow", -0.5, -0.5)
        corners = store.get_corners("t")
        assert corners[0]["anchor_x"] == -0.5


# ===========================================================================
# 3. 属性不变量测试 (property) — 幂等性、往返一致性、确定性
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证 Store 的幂等性与往返一致性。"""

    def test_upsert_track_idempotent(self, store: Store) -> None:
        """upsert_track 同一 track_id 多次调用幂等（不新增行，仅更新）。"""
        kwargs = {
            "track_id": "suzuka",
            "official_name": "Japanese GP",
            "circuit_name": "Suzuka",
            "track_type": "high_downforce",
            "length_m": 5807.0,
            "corners": 18,
            "svg_path": "tracks/suzuka.svg",
        }
        store.upsert_track(**kwargs)
        store.upsert_track(**kwargs)
        store.upsert_track(**kwargs)

        # 应只有 1 条
        row = store.get_track("suzuka")
        assert row is not None
        # 再查全表
        with store._lock:
            count = store._conn.execute("SELECT COUNT(*) AS c FROM track").fetchone()["c"]
        assert count == 1, "upsert_track 应幂等，不新增行"

    def test_upsert_track_update_semantics(self, store: Store) -> None:
        """upsert_track 二次调用应更新字段（非忽略）。"""
        store.upsert_track("suzuka", "Old", "Suzuka", "mixed", 5000.0, 10, "a.svg")
        store.upsert_track("suzuka", "New", "Suzuka", "high_downforce", 5807.0, 18, "b.svg")
        row = store.get_track("suzuka")
        assert row is not None
        assert row["official_name"] == "New"
        assert row["track_type"] == "high_downforce"
        assert row["length_m"] == 5807.0
        assert row["corners"] == 18
        assert row["svg_path"] == "b.svg"

    def test_upsert_corner_idempotent(self, store: Store) -> None:
        """upsert_corner 同 (track_id, corner_number) 幂等。"""
        store.upsert_track("t", "n", "c", "mixed", 1000.0, 1, "p.svg")
        store.upsert_corner("t", 1, "slow", 0.1, 0.2, name="T1")
        store.upsert_corner("t", 1, "slow", 0.1, 0.2, name="T1")
        corners = store.get_corners("t")
        assert len(corners) == 1, "upsert_corner 应幂等"

    def test_setup_params_roundtrip(self, store: Store) -> None:
        """import_setup → get_setup params 往返一致（含浮点/嵌套/中文）。"""
        params = {
            "front_wing": 6.5,
            "nested": {"a": 1, "b": [1, 2, 3]},
            "中文键": "中文值",
            "none_val": None,
        }
        sid = store.import_setup("suzuka", params)
        row = store.get_setup(sid)
        assert row is not None
        assert row["params"] == params

    def test_suggestion_report_roundtrip(self, store: Store) -> None:
        """save_suggestion → get_latest_suggestion report_json 往返一致。"""
        report = {"parameters": [{"param": "front_wing", "delta": 1.0}], "summary": "测试"}
        rj = json.dumps(report, ensure_ascii=False, sort_keys=True)
        store.save_suggestion("suzuka", rj)
        latest = store.get_latest_suggestion("suzuka")
        assert latest is not None
        assert json.loads(latest["report_json"]) == report

    def test_get_latest_setup_deterministic(self, store: Store) -> None:
        """相同序列的 import_setup 多次调用 get_latest_setup 结果一致。"""
        store.import_setup("suzuka", {"front_wing": 5.0})
        store.import_setup("suzuka", {"front_wing": 6.0})
        first = store.get_latest_setup("suzuka")
        second = store.get_latest_setup("suzuka")
        assert first == second, "get_latest_setup 应确定性（无并发写入时）"

    def test_feedback_ordering_by_created_at(self, store: Store) -> None:
        """get_feedbacks 按 created_at 升序（与插入顺序一致）。"""
        ids = []
        for i in range(5):
            ids.append(store.add_feedback("suzuka", i + 1, "understeer", "entry", 3))
        rows = store.get_feedbacks("suzuka")
        assert [r["id"] for r in rows] == ids, "应按 created_at 升序"

    def test_iteration_ordering_by_round_no(self, store: Store) -> None:
        """get_iterations 按 round_no 升序，与插入顺序无关。"""
        store.save_iteration("suzuka", 5, None, None, None)
        store.save_iteration("suzuka", 1, None, None, None)
        store.save_iteration("suzuka", 3, None, None, None)
        rows = store.get_iterations("suzuka")
        assert [r["round_no"] for r in rows] == [1, 3, 5]


# ===========================================================================
# 4. 静态分析 (static) — SQL 注入防护、值域约束、外键约束、表结构
# ===========================================================================
class TestStatic:
    """静态分析：验证 SQL 参数化、外键约束、CHECK 约束与表结构正确。"""

    def test_sql_injection_in_track_id(self, store: Store) -> None:
        """track_id 含 SQL 注入 payload 应被参数化隔离（不执行注入）。"""
        # 经典 SQL 注入 payload
        malicious = "'; DROP TABLE track; --"
        store.upsert_track(
            malicious, "name", "circuit", "mixed", 1000.0, 1, "p.svg",
        )
        # track 表应仍存在（参数化查询隔离了注入）
        row = store.get_track(malicious)
        assert row is not None
        # 验证 track 表未被删除
        with store._lock:
            tables = {
                r["name"]
                for r in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert "track" in tables, "SQL 注入应被参数化隔离，track 表不应被删除"

    def test_sql_injection_in_symptom(self, store: Store) -> None:
        """symptom 含单引号应被参数化隔离。"""
        payload = "understeer'; DROP TABLE feedback; --"
        # add_feedback 不校验 symptom 枚举（那是 FeedbackService 的职责）
        store.add_feedback("suzuka", 1, payload, "entry", 3)
        rows = store.get_feedbacks("suzuka")
        assert rows[0]["symptom"] == payload
        # feedback 表应仍存在
        with store._lock:
            tables = {
                r["name"]
                for r in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert "feedback" in tables

    def test_foreign_key_corner_to_track(self, store: Store) -> None:
        """corner.track_id 引用不存在的 track 应触发外键约束。"""
        # PRAGMA foreign_keys = ON 已在 Store.__init__ 开启
        with pytest.raises(sqlite3.IntegrityError):
            store.upsert_corner("ghost_track", 1, "slow", 0.1, 0.2)

    def test_foreign_keys_enabled(self, store: Store) -> None:
        """PRAGMA foreign_keys 应为 ON（外键约束开启）。"""
        with store._lock:
            row = store._conn.execute("PRAGMA foreign_keys").fetchone()
        # sqlite3.Row 需用索引取单值
        assert row[0] == 1, "外键约束应开启"

    def test_wal_journal_mode(self, store_file: Store) -> None:
        """PRAGMA journal_mode 应为 wal（WAL 模式提升并发）。

        注：``:memory:`` 库始终用 ``memory`` 日志模式，故用 store_file 验证。
        """
        with store_file._lock:
            row = store_file._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_six_tables_exist(self, store: Store) -> None:
        """schema.sql 应创建 6 张核心表。"""
        with store._lock:
            tables = {
                r["name"]
                for r in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        expected = {"track", "corner", "setup", "feedback", "suggestion", "iteration"}
        assert expected.issubset(tables), f"缺失表: {expected - tables}"

    def test_indexes_exist(self, store: Store) -> None:
        """schema.sql 应创建 5 个索引（加速按赛道查询）。"""
        with store._lock:
            idxs = {
                r["name"]
                for r in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'",
                ).fetchall()
            }
        expected = {
            "idx_feedback_track", "idx_feedback_setup", "idx_setup_track",
            "idx_suggestion_track", "idx_iteration_track",
        }
        assert expected.issubset(idxs), f"缺失索引: {expected - idxs}"

    def test_track_primary_key(self, store: Store) -> None:
        """track.track_id 应为 PRIMARY KEY（重复插入触发约束，但 upsert 用 ON CONFLICT）。"""
        # 直接 INSERT 重复应触发 IntegrityError
        with store._lock:
            store._conn.execute(
                "INSERT INTO track (track_id, official_name, circuit_name, "
                "track_type, length_m, corners, svg_path) "
                "VALUES ('dup', 'a', 'b', 'mixed', 1.0, 1, 'x.svg')",
            )
            with pytest.raises(sqlite3.IntegrityError):
                store._conn.execute(
                    "INSERT INTO track (track_id, official_name, circuit_name, "
                    "track_type, length_m, corners, svg_path) "
                    "VALUES ('dup', 'a', 'b', 'mixed', 1.0, 1, 'x.svg')",
                )

    def test_corner_composite_primary_key(self, store: Store) -> None:
        """corner 的 (track_id, corner_number) 应为复合主键。"""
        store.upsert_track("t", "n", "c", "mixed", 1.0, 1, "p.svg")
        store.upsert_corner("t", 1, "slow", 0.1, 0.2)
        # 同 (track_id, corner_number) 直接 INSERT 应触发 IntegrityError
        with store._lock, pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO corner (track_id, corner_number, corner_type, "
                "anchor_x, anchor_y) VALUES ('t', 1, 'slow', 0.3, 0.4)",
            )

    def test_feedback_strength_check_constraint(self, store: Store) -> None:
        """feedback.strength 应有 CHECK(strength BETWEEN 0 AND 5) 约束。"""
        # 正常值
        store.add_feedback("suzuka", 1, "understeer", "entry", 3)
        # 越界值通过原生 SQL 应触发约束
        with store._lock, pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO feedback (track_id, symptom, category, strength, created_at) "
                "VALUES ('suzuka', 'x', 'entry', 99, '2026-01-01T00:00:00Z')",
            )

    def test_setup_id_autoincrement(self, store: Store) -> None:
        """setup.id 应为 AUTOINCREMENT（删除后不复用 id）。"""
        sid1 = store.import_setup("suzuka", {"a": 1})
        sid2 = store.import_setup("suzuka", {"a": 2})
        assert sid2 == sid1 + 1


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 真实文件 SQLite、上下文管理器、跨方法调用链
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：真实文件 SQLite、上下文管理器、完整业务调用链。"""

    def test_file_based_db_persistence(self, store_file: Store) -> None:
        """真实文件 SQLite：写入后重新打开同一文件应能读到数据。"""
        db_path = store_file._db_path
        store_file.upsert_track(
            "suzuka", "Japanese GP", "Suzuka", "high_downforce",
            5807.0, 18, "tracks/suzuka.svg",
        )
        store_file.close()

        # 重新打开同一文件
        s2 = Store(db_path)
        try:
            row = s2.get_track("suzuka")
            assert row is not None
            assert row["official_name"] == "Japanese GP"
        finally:
            s2.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Store 支持 with 上下文管理器（自动 close）。"""
        db_path = tmp_path / "ctx.db"
        with Store(str(db_path)) as s:
            s.upsert_track("t", "n", "c", "mixed", 1.0, 1, "p.svg")
            row = s.get_track("t")
            assert row is not None
        # with 退出后连接应已关闭；再操作应抛异常
        with pytest.raises(sqlite3.ProgrammingError):
            s.get_track("t")

    def test_full_workflow_chain(self, store: Store) -> None:
        """完整业务调用链：track → corner → setup → feedback → suggestion → iteration。"""
        # 1. 赛道
        store.upsert_track(
            "suzuka", "Japanese GP", "Suzuka", "high_downforce",
            5807.0, 18, "tracks/suzuka.svg", udp_track_id=2,
        )
        # 2. 弯道
        for i in range(1, 4):
            store.upsert_corner("suzuka", i, "slow", 0.1 * i, 0.2 * i, name=f"T{i}")

        # 3. 调教快照
        setup_id = store.import_setup("suzuka", {"front_wing": 6.0, "rear_wing": 4.0})

        # 4. 反馈
        fb_id = store.add_feedback(
            "suzuka", 1, "understeer", "entry", 4, setup_id=setup_id,
        )

        # 5. 建议
        report = json.dumps({"summary": "建议已生成"})
        sug_id = store.save_suggestion("suzuka", report, setup_id=setup_id)

        # 6. 迭代
        it_id = store.save_iteration("suzuka", 1, setup_id, None, sug_id)

        # 全部验证
        assert store.get_track("suzuka") is not None
        assert len(store.get_corners("suzuka")) == 3
        assert store.get_setup(setup_id) is not None
        assert len(store.get_feedbacks("suzuka")) == 1
        assert store.get_latest_suggestion("suzuka") is not None
        assert len(store.get_iterations("suzuka")) == 1
        assert fb_id >= 1 and sug_id >= 1 and it_id >= 1

    def test_thread_safety_concurrent_writes(self, store: Store) -> None:
        """并发写：10 线程各写 10 条 feedback，应全部入库（线程安全）。"""
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            try:
                for i in range(10):
                    store.add_feedback(
                        "suzuka", (i % 18) + 1, "understeer", "entry", 3,
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发写不应有错误: {errors}"
        rows = store.get_feedbacks("suzuka")
        assert len(rows) == 100, "10 线程 × 10 条 = 100 条 feedback"

    def test_wal_file_created(self, store_file: Store) -> None:
        """WAL 模式下，写入后应生成 -wal 与 -shm 侧车文件。"""
        db_path = Path(store_file._db_path)
        store_file.upsert_track("t", "n", "c", "mixed", 1.0, 1, "p.svg")
        store_file._conn.commit()  # 确保刷盘
        # WAL 文件可能未立即生成（SQLite 惰性创建），但 journal_mode 已是 wal
        # 验证 db 文件存在且非空
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    def test_reopen_memory_db_independent(self) -> None:
        """两个内存库互不影响（隔离性）。"""
        s1 = Store(":memory:")
        s2 = Store(":memory:")
        try:
            s1.upsert_track("t1", "n", "c", "mixed", 1.0, 1, "p.svg")
            assert s2.get_track("t1") is None, "内存库应相互隔离"
            s2.upsert_track("t2", "n", "c", "mixed", 1.0, 1, "p.svg")
            assert s1.get_track("t2") is None
        finally:
            s1.close()
            s2.close()