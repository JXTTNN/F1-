"""烟雾测试: 验证包可导入且默认配置正确。

Task 0 (bootstrap) 的最低可执行验证。
"""

from __future__ import annotations

import f1opt
from f1opt.config import Settings


def test_package_version() -> None:
    """顶层包应导出 ``__version__`` 字符串。"""
    assert isinstance(f1opt.__version__, str)
    assert f1opt.__version__ == "0.1.0"


def test_default_udp_port() -> None:
    """默认 UDP 端口必须为 EA F1 25/2026 官方端口 20777。"""
    settings = Settings()
    assert settings.udp_port == 20777


def test_default_api_settings() -> None:
    """默认 API 监听 127.0.0.1:8000。"""
    settings = Settings()
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000


def test_get_settings_is_cached() -> None:
    """:func:`get_settings` 返回同一单例对象。"""
    from f1opt.config import get_settings

    assert get_settings() is get_settings()
