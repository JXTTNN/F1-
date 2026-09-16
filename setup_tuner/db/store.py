"""SQLite 持久化层 —— Store 类封装连接与 CRUD。

仅依赖标准库 sqlite3（不引入 SQLAlchemy），线程安全
（check_same_thread=False + threading.Lock 串行化写操作）。

覆盖表：track / corner / setup / feedback / suggestion / iteration。
对照 design.md 2.3.2 节表结构。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# schema.sql 路径定位 —— 多候选探测，兼容开发模式与 Nuitka 打包模式
def _find_schema_path() -> Path:
    """查找 schema.sql，兼容开发模式和 Nuitka onefile/standalone 模式。"""
    candidates = [
        # 1. 基于 __file__（开发模式：setup_tuner/db/store.py → setup_tuner/db/schema.sql）
        Path(__file__).resolve().parent / "schema.sql",
        # 2. 基于 sys.executable + setup_tuner/db/schema.sql（Nuitka onefile/standalone）
        Path(sys.executable).resolve().parent / "setup_tuner" / "db" / "schema.sql",
        # 3. 基于 sys.executable + db/schema.sql（Nuitka onefile 根目录极端情况）
        Path(sys.executable).resolve().parent / "db" / "schema.sql",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_SCHEMA_PATH = _find_schema_path()

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串（带时区标记）。"""
    return datetime.now(UTC).isoformat()


class Store:
    """SQLite 连接与 CRUD 封装。

    线程安全：底层连接 check_same_thread=False，所有写操作经 ``_lock`` 串行化。
    读操作同样加锁以保证一致性视图（SQLite 单写多读，锁开销极小）。

    Args:
        db_path: SQLite 数据库文件路径。使用 ``":memory:"`` 可创建内存库（测试用）。
    """

    def __init__(self, db_path: str, *, seed: bool = True) -> None:
        self._db_path = db_path
        # check_same_thread=False 允许跨线程使用同一连接；
        # 线程安全由 self._lock 保证。
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # 性能 PRAGMA：连接创建后立即执行，提升写入并发与读吞吐
        # （来源：2026-09-11-sqlite-pragma-tuning）
        # WAL：Write-Ahead Logging，读不阻塞写、写不阻塞读
        self._conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL：每个事务提交时同步一次（而非 FULL 的每次写入都 fsync），
        # 在 WAL 模式下仅在最坏情况丢失最后一个事务
        self._conn.execute("PRAGMA synchronous = NORMAL")
        # 8MB 页缓存（负值表示 KB 单位：-8000 ≈ 8MB）
        self._conn.execute("PRAGMA cache_size = -8000")
        # 临时表与中间结果存内存，避免磁盘 I/O
        self._conn.execute("PRAGMA temp_store = MEMORY")
        # 外键约束开启（corner/setup 等引用 track/setup）
        self._conn.execute("PRAGMA foreign_keys = ON")
        # row_factory 让查询结果可按列名访问
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()
        if seed:
            self._seed_track_data()
        self._migrate_feedback_strength()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        """读取 schema.sql 并执行建表 DDL（幂等）。"""
        ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._lock:
            self._conn.executescript(ddl)
            self._conn.commit()

    def _migrate_feedback_strength(self) -> None:
        """一次性迁移：强度档位 0–5 → 1–3（task-61）。

        归并规则：{1}→1、{2,3}→2、{4,5}→3；0（=未反馈）保持不变。
        幂等：已迁移的数据再跑一遍不会有任何变化。
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE strength >= 4"
            )
            if cur.fetchone()[0] == 0:
                return
            self._conn.execute(
                "UPDATE feedback SET strength = CASE "
                "WHEN strength >= 4 THEN 3 ELSE 2 END WHERE strength >= 2"
            )
            self._conn.commit()
            logger.info("feedback strength 迁移完成（0-5 → 1-3）")

    def _seed_track_data(self) -> None:
        """将 ALL_TRACKS 的 24 条赛道 + 弯道数据同步到数据库（幂等）。

        应用启动时调用，确保 track / corner 表有静态数据。
        使用 upsert 语义（ON CONFLICT DO UPDATE），重复启动不会产生重复记录。
        """
        from setup_tuner.domain.track import get_all_tracks

        tracks = get_all_tracks()
        # 性能：所有 track/corner 的 upsert 共用一个事务，最后统一 commit。
        # 原实现每条记录 commit 一次（24 + 404 = 428 次），WAL 下每次 commit
        # 都要走一次事务边界；合并为单事务后启动路径只提交一次。
        for t in tracks:
            self.upsert_track(
                track_id=t.track_id,
                official_name=t.official_name,
                circuit_name=t.circuit_name,
                track_type=t.track_type,
                length_m=t.length_m,
                corners=len(t.corners),
                svg_path=t.svg_path,
                udp_track_id=t.udp_track_id,
                commit=False,
            )
            for c in t.corners:
                self.upsert_corner(
                    track_id=t.track_id,
                    corner_number=c.number,
                    corner_type=c.corner_type,
                    anchor_x=c.anchor.anchor_x,
                    anchor_y=c.anchor.anchor_y,
                    name=c.name,
                    speed_kmh=c.speed_kmh,
                    commit=False,
                )
        with self._lock:
            self._conn.commit()
        logger.debug("seeded %d tracks with corners into database", len(tracks))

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # 上下文管理器支持
    # ------------------------------------------------------------------
    def __enter__(self) -> Store:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # track / corner（赛道与弯道静态数据）
    # ------------------------------------------------------------------
    def upsert_track(
        self,
        track_id: str,
        official_name: str,
        circuit_name: str,
        track_type: str,
        length_m: float,
        corners: int,
        svg_path: str,
        udp_track_id: int | None = None,
        commit: bool = True,
    ) -> None:
        """插入或更新赛道主表记录（按 track_id 幂等）。

        Args:
            commit: 是否立即提交事务。批量 seed 时传 False，由调用方统一提交。
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO track
                    (track_id, official_name, circuit_name, track_type,
                     length_m, corners, udp_track_id, svg_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                    official_name = excluded.official_name,
                    circuit_name  = excluded.circuit_name,
                    track_type    = excluded.track_type,
                    length_m      = excluded.length_m,
                    corners       = excluded.corners,
                    udp_track_id  = excluded.udp_track_id,
                    svg_path      = excluded.svg_path
                """,
                (
                    track_id, official_name, circuit_name, track_type,
                    length_m, corners, udp_track_id, svg_path,
                ),
            )
            if commit:
                self._conn.commit()

    def get_track(self, track_id: str) -> dict[str, Any] | None:
        """按 track_id 查询赛道主表。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM track WHERE track_id = ?", (track_id,),
            ).fetchone()
        return dict(row) if row else None

    def upsert_corner(
        self,
        track_id: str,
        corner_number: int,
        corner_type: str,
        anchor_x: float,
        anchor_y: float,
        name: str | None = None,
        speed_kmh: float | None = None,
        commit: bool = True,
    ) -> None:
        """插入或更新弯道记录（按 (track_id, corner_number) 幂等）。

        Args:
            commit: 是否立即提交事务。批量 seed 时传 False，由调用方统一提交。
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO corner
                    (track_id, corner_number, name, corner_type,
                     speed_kmh, anchor_x, anchor_y)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_id, corner_number) DO UPDATE SET
                    name        = excluded.name,
                    corner_type = excluded.corner_type,
                    speed_kmh   = excluded.speed_kmh,
                    anchor_x    = excluded.anchor_x,
                    anchor_y    = excluded.anchor_y
                """,
                (track_id, corner_number, name, corner_type, speed_kmh, anchor_x, anchor_y),
            )
            if commit:
                self._conn.commit()

    def get_corners(self, track_id: str) -> list[dict[str, Any]]:
        """查询某赛道的全部弯道（按 corner_number 升序）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM corner WHERE track_id = ? ORDER BY corner_number",
                (track_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # setup（调教快照）
    # ------------------------------------------------------------------
    def import_setup(self, track_id: str, params: dict[str, Any]) -> int:
        """导入一份调教快照。

        Args:
            track_id: 所属赛道标识。
            params: 21 项参数字典（序列化为 JSON 存储）。

        Returns:
            新插入的 setup.id。
        """
        params_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
        imported_at = _now_iso()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO setup (track_id, imported_at, params_json) VALUES (?, ?, ?)",
                (track_id, imported_at, params_json),
            )
            self._conn.commit()
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def get_setup(self, setup_id: int) -> dict[str, Any] | None:
        """按 setup_id 读取调教快照。

        返回字典含 ``id / track_id / imported_at / params``（params 为反序列化后的 dict）。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM setup WHERE id = ?", (setup_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["params"] = json.loads(result.pop("params_json"))
        return result

    def get_latest_setup(self, track_id: str) -> dict[str, Any] | None:
        """获取某赛道最近一次导入的调教快照（按 imported_at 降序取首条）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM setup WHERE track_id = ? ORDER BY imported_at DESC LIMIT 1",
                (track_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["params"] = json.loads(result.pop("params_json"))
        return result

    # ------------------------------------------------------------------
    # feedback（玩家弯道反馈）
    # ------------------------------------------------------------------
    def add_feedback(
        self,
        track_id: str,
        corner_number: int | None,
        symptom: str,
        category: str,
        strength: int = 2,
        setup_id: int | None = None,
    ) -> int:
        """录入一条玩家反馈。

        Args:
            track_id: 赛道标识。
            corner_number: 弯道编号（1-based）；None 表示全局症状。
            symptom: 12 症状标识之一。
            category: entry|apex|exit|global。
            strength: 强度 1–3（1 轻微 / 2 明显 / 3 严重），默认 2。
            setup_id: 关联的调教快照 id（可选）。

        Returns:
            新插入的 feedback.id。
        """
        created_at = _now_iso()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO feedback
                    (setup_id, track_id, corner_number, symptom, category, strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (setup_id, track_id, corner_number, symptom, category, strength, created_at),
            )
            self._conn.commit()
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def get_feedbacks(self, track_id: str) -> list[dict[str, Any]]:
        """查询某赛道的全部反馈（按 created_at 升序）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM feedback WHERE track_id = ? ORDER BY created_at",
                (track_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def has_feedback(self, track_id: str) -> bool:
        """判断某赛道是否至少有 1 条反馈记录。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM feedback WHERE track_id = ? LIMIT 1", (track_id,),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # suggestion（调教建议）
    # ------------------------------------------------------------------
    def save_suggestion(
        self,
        track_id: str,
        report_json: str,
        setup_id: int | None = None,
    ) -> int:
        """保存一条调教建议报告。

        Args:
            track_id: 赛道标识。
            report_json: 报告 JSON 字符串（含 setupDelta/联动/出处/置信度/tradeoff）。
            setup_id: 关联的调教快照 id（可选）。

        Returns:
            新插入的 suggestion.id。
        """
        created_at = _now_iso()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO suggestion (setup_id, track_id, created_at, report_json)
                VALUES (?, ?, ?, ?)
                """,
                (setup_id, track_id, created_at, report_json),
            )
            self._conn.commit()
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def get_latest_suggestion(self, track_id: str) -> dict[str, Any] | None:
        """获取某赛道最新一条调教建议（按 created_at 降序取首条）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM suggestion WHERE track_id = ? ORDER BY created_at DESC LIMIT 1",
                (track_id,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # iteration（迭代闭环记录）
    # ------------------------------------------------------------------
    def save_iteration(
        self,
        track_id: str,
        round_no: int,
        before_setup_id: int | None,
        after_setup_id: int | None,
        suggestion_id: int | None,
    ) -> int:
        """保存一条迭代闭环记录。

        Args:
            track_id: 赛道标识。
            round_no: 第几轮闭环。
            before_setup_id: 调整前调教快照 id。
            after_setup_id: 调整后调教快照 id。
            suggestion_id: 对应建议 id。

        Returns:
            新插入的 iteration.id。
        """
        created_at = _now_iso()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO iteration
                    (track_id, round_no, before_setup_id, after_setup_id,
                     suggestion_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (track_id, round_no, before_setup_id, after_setup_id, suggestion_id, created_at),
            )
            self._conn.commit()
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def get_iterations(self, track_id: str) -> list[dict[str, Any]]:
        """查询某赛道的迭代历史（按 round_no 升序）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM iteration WHERE track_id = ? ORDER BY round_no",
                (track_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_round(self, track_id: str) -> int:
        """获取某赛道最新轮次号（用 SQL MAX，避免拉全部记录）。

        性能优化（task-36）：替代 IterationService.get_latest_round 拉全部
        iterations 再取 max 的实现，单次聚合查询 O(1)。

        Returns:
            最新轮次号；无历史记录时返回 0。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(round_no) AS m FROM iteration WHERE track_id = ?",
                (track_id,),
            ).fetchone()
        if row is None or row["m"] is None:
            return 0
        return int(row["m"])