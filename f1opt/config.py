"""应用配置。

通过环境变量（前缀 ``F1OPT_``）注入运行期配置，例如::

    F1OPT_UDP_PORT=20777 F1OPT_API_PORT=8000 python -m f1opt

使用 :func:`get_settings` 获取单例配置。
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

_ENV_PREFIX = "F1OPT_"


def _env(key: str, default: str) -> str:
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


class Settings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore", frozen=True)

    udp_host: str = Field(
        default_factory=lambda: _env("UDP_HOST", "0.0.0.0"),
        description="UDP 监听地址",
    )
    udp_port: int = Field(
        default_factory=lambda: _env("UDP_PORT", "20777"),
        description="F1 25/2026 默认遥测端口",
    )
    api_host: str = Field(
        default_factory=lambda: _env("API_HOST", "127.0.0.1"),
        description="FastAPI 监听地址",
    )
    api_port: int = Field(
        default_factory=lambda: _env("API_PORT", "8000"),
        description="FastAPI 监听端口",
    )
    data_dir: str = Field(
        default_factory=lambda: _env("DATA_DIR", "data_store"),
        description="运行期数据目录",
    )
    llm_backend: str = Field(
        default_factory=lambda: _env("LLM_BACKEND", "none"),
        description="LLM 后端: none|openai|ollama|local",
    )
    llm_api_key: str = Field(
        default_factory=lambda: _env("LLM_API_KEY", ""),
        description="LLM API Key",
    )
    llm_model: str = Field(
        default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"),
        description="LLM 模型名称",
    )
    log_level: str = Field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO"),
        description="structlog 日志等级",
    )

    @property
    def is_windows(self) -> bool:
        return sys.platform == "win32"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
