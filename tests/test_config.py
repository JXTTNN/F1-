"""配置模块单元测试。

覆盖验收标准：
1. Config 数据类缺省值
2. _parse_env_file 解析 .env 文件（注释/空行/引号/无等号）
3. load_config 优先级：环境变量 > .env 文件 > 缺省值
4. load_config 不存在的 .env 文件使用缺省
5. 边界条件：空文件、只有注释、引号去除
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from setup_tuner.config import Config, _parse_env_file, load_config


# ===========================================================================
# 1. Config 数据类
# ===========================================================================
class TestConfigDataclass:
    """Config 缺省值与不可变性。"""

    def test_default_values(self) -> None:
        """Config 缺省值正确。"""
        c = Config()
        assert c.udp_host == "127.0.0.1"
        assert c.udp_port == 20777
        assert c.api_host == "127.0.0.1"
        assert c.api_port == 8000
        assert c.data_dir == "./data"
        assert c.log_level == "INFO"

    def test_custom_values(self) -> None:
        """自定义值。"""
        c = Config(udp_host="0.0.0.0", udp_port=12345, api_port=9000)
        assert c.udp_host == "0.0.0.0"
        assert c.udp_port == 12345
        assert c.api_port == 9000

    def test_frozen(self) -> None:
        """Config 为 frozen dataclass，不可修改。"""
        c = Config()
        with pytest.raises(AttributeError):
            c.udp_host = "other"  # type: ignore[misc]


# ===========================================================================
# 2. _parse_env_file
# ===========================================================================
class TestParseEnvFile:
    """_parse_env_file 解析逻辑。"""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """不存在的文件返回空字典。"""
        result = _parse_env_file(tmp_path / "nonexistent.env")
        assert result == {}

    def test_empty_file(self, tmp_path: Path) -> None:
        """空文件返回空字典。"""
        env = tmp_path / ".env"
        env.write_text("", encoding="utf-8")
        assert _parse_env_file(env) == {}

    def test_comments_and_empty_lines(self, tmp_path: Path) -> None:
        """注释行和空行被忽略。"""
        env = tmp_path / ".env"
        env.write_text(
            "# This is a comment\n"
            "\n"
            "   # Indented comment\n"
            "UDP_HOST=0.0.0.0\n",
            encoding="utf-8",
        )
        result = _parse_env_file(env)
        assert result == {"UDP_HOST": "0.0.0.0"}

    def test_key_value_parsing(self, tmp_path: Path) -> None:
        """key=value 解析（含空格）。"""
        env = tmp_path / ".env"
        env.write_text(
            "UDP_HOST = 0.0.0.0 \n"
            "UDP_PORT=20778\n"
            "API_HOST=localhost\n",
            encoding="utf-8",
        )
        result = _parse_env_file(env)
        assert result["UDP_HOST"] == "0.0.0.0"
        assert result["UDP_PORT"] == "20778"
        assert result["API_HOST"] == "localhost"

    def test_quote_removal(self, tmp_path: Path) -> None:
        """value 两侧引号被去除。"""
        env = tmp_path / ".env"
        env.write_text(
            'DATA_DIR="./my data"\n'
            "LOG_LEVEL='DEBUG'\n",
            encoding="utf-8",
        )
        result = _parse_env_file(env)
        assert result["DATA_DIR"] == "./my data"
        assert result["LOG_LEVEL"] == "DEBUG"

    def test_no_equals_sign_skipped(self, tmp_path: Path) -> None:
        """无等号的行被跳过。"""
        env = tmp_path / ".env"
        env.write_text(
            "INVALID_LINE\n"
            "UDP_HOST=0.0.0.0\n",
            encoding="utf-8",
        )
        result = _parse_env_file(env)
        assert result == {"UDP_HOST": "0.0.0.0"}

    def test_single_char_value(self, tmp_path: Path) -> None:
        """单字符 value 不被当引号处理。"""
        env = tmp_path / ".env"
        env.write_text("LOG_LEVEL=A\n", encoding="utf-8")
        result = _parse_env_file(env)
        assert result["LOG_LEVEL"] == "A"


# ===========================================================================
# 3. load_config
# ===========================================================================
class TestLoadConfig:
    """load_config 优先级与缺省。"""

    def test_defaults_when_no_env(self, tmp_path: Path) -> None:
        """无 .env 文件且无环境变量时使用缺省。"""
        env = tmp_path / ".env"
        with patch.dict(os.environ, {}, clear=False):
            for key in ("UDP_HOST", "UDP_PORT", "API_HOST", "API_PORT", "DATA_DIR", "LOG_LEVEL"):
                os.environ.pop(key, None)
            c = load_config(env_path=env)
        assert c.udp_host == "127.0.0.1"
        assert c.udp_port == 20777
        assert c.api_host == "127.0.0.1"
        assert c.api_port == 8000
        assert c.data_dir == "./data"
        assert c.log_level == "INFO"

    def test_load_from_env_file(self, tmp_path: Path) -> None:
        """从 .env 文件加载配置。"""
        env = tmp_path / ".env"
        env.write_text(
            "UDP_HOST=0.0.0.0\n"
            "UDP_PORT=20778\n"
            "API_HOST=localhost\n"
            "API_PORT=9000\n"
            "DATA_DIR=/tmp/data\n"
            "LOG_LEVEL=DEBUG\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {}, clear=False):
            for key in ("UDP_HOST", "UDP_PORT", "API_HOST", "API_PORT", "DATA_DIR", "LOG_LEVEL"):
                os.environ.pop(key, None)
            c = load_config(env_path=env)
        assert c.udp_host == "0.0.0.0"
        assert c.udp_port == 20778
        assert c.api_host == "localhost"
        assert c.api_port == 9000
        assert c.data_dir == "/tmp/data"
        assert c.log_level == "DEBUG"

    def test_env_var_overrides_file(self, tmp_path: Path) -> None:
        """环境变量优先于 .env 文件。"""
        env = tmp_path / ".env"
        env.write_text("UDP_HOST=from_file\n", encoding="utf-8")
        with patch.dict(os.environ, {"UDP_HOST": "from_env"}, clear=False):
            c = load_config(env_path=env)
        assert c.udp_host == "from_env"

    def test_env_var_overrides_default(self, tmp_path: Path) -> None:
        """环境变量优先于缺省值。"""
        env = tmp_path / ".env"  # 不存在
        with patch.dict(os.environ, {"UDP_PORT": "30000", "API_PORT": "5000"}, clear=False):
            c = load_config(env_path=env)
        assert c.udp_port == 30000
        assert c.api_port == 5000

    def test_default_env_path(self, tmp_path: Path) -> None:
        """env_path=None 时默认查找当前目录 .env。"""
        env = tmp_path / ".env"
        env.write_text("UDP_PORT=11111\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UDP_PORT", None)
            original = Path.cwd
            try:
                Path.cwd = lambda: tmp_path  # type: ignore[assignment]
                c = load_config()
            finally:
                Path.cwd = original  # type: ignore[assignment]
        assert c.udp_port == 11111