"""CLI 日志通道测试：便携版双击失败时，必须留下可查的痕迹。

背景：exe 曾经是 GUI 子系统（`--windows-console-mode=disable`），双击后既
没有窗口也没有输出，`logging.basicConfig` 的内容全部丢失。用户遇到「打不开」
时只能靠猜；日志文件是唯一能把现场保留下来的东西。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from setup_tuner import cli
from setup_tuner.config import Config


@pytest.fixture
def preserve_logging() -> Iterator[None]:
    """备份/恢复 root logger，避免 basicConfig(force=True) 污染其它测试。"""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        for h in list(root.handlers):
            if h not in handlers:
                h.close()
                root.removeHandler(h)
        for h in handlers:
            if h not in root.handlers:
                root.addHandler(h)
        root.setLevel(level)


class TestFileLogging:
    def test_log_file_is_created_with_records(
        self, tmp_path: Path, preserve_logging: None,
    ) -> None:
        config = Config(data_dir=str(tmp_path), udp_port=0)

        path = cli._setup_logging(config)

        assert path == tmp_path / "logs" / "f1opt.log"
        assert path is not None and path.exists()
        logging.getLogger("setup_tuner.test").warning("写入测试记录")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "写入测试记录" in path.read_text(encoding="utf-8")

    def test_returns_none_when_directory_is_not_writable(
        self, tmp_path: Path, preserve_logging: None,
    ) -> None:
        """数据目录不可写时退回仅控制台，不能因此启动失败。"""
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        config = Config(data_dir=str(blocker / "data"), udp_port=0)

        assert cli._setup_logging(config) is None
        logging.getLogger("setup_tuner.test").warning("仍然能打印")
        assert len(logging.getLogger().handlers) >= 1

    def test_log_file_lives_next_to_recordings(self, tmp_path: Path) -> None:
        """日志必须落在数据目录下，用户导出 data\\ 时能一起带走。"""
        config = Config(data_dir=str(tmp_path / "data"), udp_port=0)
        assert config.resolved_data_dir() == str(tmp_path / "data")
        assert Path(config.resolved_data_dir()) / "logs" == tmp_path / "data" / "logs"
