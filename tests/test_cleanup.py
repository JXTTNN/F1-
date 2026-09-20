"""关闭时遥测清理的测试（**录制必须保留**）。

对应约定（2026-09-20）：遥测数据除录制外，应用关闭后删除。
本文件锁定三件事：
1. 该删的删（training / sim_telemetry / training_dataset.json）；
2. **录制一个字节都不动**（含子目录、嵌套文件）；
3. 安全护栏：危险 data_dir（文件系统根 / 用户主目录）与路径含 recordings 时拒绝执行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setup_tuner.telemetry.cleanup import (
    KEEP_ENV_VAR,
    cleanup_telemetry,
    keep_telemetry_enabled,
)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """构造一个典型的 data 目录：录制 + 派生遥测 + 模型 + 数据库。"""
    root = tmp_path / "data"
    (root / "recordings").mkdir(parents=True)
    (root / "recordings" / "session1.f1rec").write_bytes(b"REC" * 100)
    (root / "recordings" / "session1_laps.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "recordings" / "nested").mkdir()
    (root / "recordings" / "nested" / "keep.txt").write_text("keep", encoding="utf-8")

    (root / "training").mkdir()
    (root / "training" / "corner_dataset.jsonl").write_text("x" * 500, encoding="utf-8")
    (root / "sim_telemetry").mkdir()
    (root / "sim_telemetry" / "sim_packets_monza.jsonl").write_text("y" * 300, encoding="utf-8")
    (root / "training_dataset.json").write_text("z" * 100, encoding="utf-8")

    (root / "models").mkdir()
    (root / "models" / "setup_sim_nn.json").write_text("{}", encoding="utf-8")
    # 真实的空 SQLite 库（应用启动时会打开它，假的文件头会报 "not a database"）
    import sqlite3

    with sqlite3.connect(root / "f1opt.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS marker (x INTEGER)")
    return root


class TestCleanupTelemetry:
    def test_deletes_derived_telemetry(self, data_dir: Path) -> None:
        report = cleanup_telemetry(data_dir)
        assert report.ok
        assert not (data_dir / "training").exists()
        assert not (data_dir / "sim_telemetry").exists()
        assert not (data_dir / "training_dataset.json").exists()
        assert report.freed_bytes > 0
        assert report.removed

    def test_keeps_recordings_untouched(self, data_dir: Path) -> None:
        """录制（含嵌套文件）必须原样保留 —— 本功能最重要的不变量。"""
        before = sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob("*"))
        cleanup_telemetry(data_dir)
        after = sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob("*"))
        for name in before:
            if name.startswith("training") or name.startswith("sim_telemetry"):
                continue
            assert name in after, f"录制/模型/数据库被误删：{name}"
        assert (data_dir / "recordings" / "session1.f1rec").read_bytes() == b"REC" * 100
        assert (data_dir / "recordings" / "nested" / "keep.txt").exists()

    def test_keeps_models_and_db(self, data_dir: Path) -> None:
        cleanup_telemetry(data_dir)
        assert (data_dir / "models" / "setup_sim_nn.json").exists()
        assert (data_dir / "f1opt.db").exists()

    def test_dry_run_removes_nothing(self, data_dir: Path) -> None:
        report = cleanup_telemetry(data_dir, dry_run=True)
        assert report.dry_run is True
        assert (data_dir / "training").exists()
        assert (data_dir / "sim_telemetry").exists()
        assert (data_dir / "training_dataset.json").exists()
        assert report.removed and "模拟" in report.describe()

    def test_missing_dir_is_not_an_error(self, tmp_path: Path) -> None:
        report = cleanup_telemetry(tmp_path / "nope")
        assert not report.ok
        assert "不存在" in report.aborted_reason
        assert report.removed == []

    def test_idempotent(self, data_dir: Path) -> None:
        cleanup_telemetry(data_dir)
        second = cleanup_telemetry(data_dir)
        assert second.ok
        assert second.removed == []
        assert "没有需要删除" in second.describe()

    @pytest.mark.parametrize("bad", ["/", str(Path.home())])
    def test_refuses_dangerous_data_dir(self, bad: str) -> None:
        """文件系统根 / 用户主目录 → 整体拒绝（不删任何东西）。"""
        report = cleanup_telemetry(bad)
        assert not report.ok
        assert report.removed == []
        assert "拒绝清理" in report.aborted_reason

    def test_refuses_path_containing_recordings(self, tmp_path: Path) -> None:
        """路径里出现 recordings → 拒绝（防止配置指向录制目录）。"""
        weird = tmp_path / "recordings"
        (weird / "training").mkdir(parents=True)
        report = cleanup_telemetry(weird)
        assert not report.ok
        assert (weird / "training").exists()


class TestKeepTelemetryEnv:
    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_enabled_values(self, value: str) -> None:
        assert keep_telemetry_enabled({KEEP_ENV_VAR: value}) is True

    @pytest.mark.parametrize("value", ["", "0", "no", "off"])
    def test_disabled_values(self, value: str) -> None:
        assert keep_telemetry_enabled({KEEP_ENV_VAR: value}) is False


class TestStartupCleanup:
    """应用**启动时**也要清理上次残留（强杀/崩溃时关闭钩子不会执行）。"""

    def test_lifespan_startup_cleans_leftovers(self, data_dir: Path) -> None:
        from fastapi.testclient import TestClient

        from setup_tuner.app import create_app
        from setup_tuner.config import Config

        app = create_app(Config(data_dir=str(data_dir)))
        with TestClient(app):
            # 进入上下文 = 触发 lifespan 启动
            assert not (data_dir / "training").exists(), "启动时未清理派生训练数据"
            assert not (data_dir / "sim_telemetry").exists(), "启动时未清理模拟包流"
            assert not (data_dir / "training_dataset.json").exists()
            # 录制必须完好
            assert (data_dir / "recordings" / "session1.f1rec").exists()
            assert (data_dir / "recordings" / "nested" / "keep.txt").exists()

    def test_keep_telemetry_skips_startup_cleanup(self, data_dir: Path) -> None:
        from fastapi.testclient import TestClient

        from setup_tuner.app import create_app
        from setup_tuner.config import Config

        app = create_app(
            Config(data_dir=str(data_dir), keep_telemetry=True),
        )
        with TestClient(app):
            assert (data_dir / "training" / "corner_dataset.jsonl").exists(), (
                "keep_telemetry=True 时不得清理"
            )
