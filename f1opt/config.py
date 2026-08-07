"""应用配置。

通过环境变量（前缀 ``F1OPT_``）注入运行期配置，例如::

    F1OPT_UDP_PORT=20777 F1OPT_API_PORT=8000 python -m f1opt

使用 :func:`get_settings` 获取单例配置，避免重复解析环境变量。

实现说明：仅依赖 ``pydantic``（v2）。env 前缀注入通过 :func:`_env`
辅助函数读取 ``F1OPT_<KEY>`` 环境变量并作为字段默认值；配合
``validate_default=True`` 让 pydantic 对 env 字符串做类型强转与校验，
行为与官方 ``pydantic-settings`` 的 env 注入一致，但无需额外依赖。
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

_ENV_PREFIX = "F1OPT_"


def _env(key: str, default: str) -> str:
    """读取 ``F1OPT_<key>`` 环境变量，未设置时返回 ``default``。"""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


class Settings(BaseModel):
    """运行期配置项。

    所有字段均带合理默认值，对应 F1 25 / 2026 默认 UDP 端口 20777
    与本地 API 监听地址 127.0.0.1:8000。
    """

    model_config = ConfigDict(validate_default=True, extra="ignore", frozen=True)

    # --- 遥测采集 ---
    udp_host: str = Field(
        default_factory=lambda: _env("UDP_HOST", "0.0.0.0"),
        description="UDP 监听地址",
    )
    udp_port: int = Field(
        default_factory=lambda: _env("UDP_PORT", "20777"),
        description="F1 25/2026 默认遥测端口",
    )

    # --- API 后端 ---
    api_host: str = Field(
        default_factory=lambda: _env("API_HOST", "127.0.0.1"),
        description="FastAPI 监听地址",
    )
    api_port: int = Field(
        default_factory=lambda: _env("API_PORT", "8000"),
        description="FastAPI 监听端口",
    )

    # --- 数据目录 ---
    data_dir: str = Field(
        default_factory=lambda: _env("DATA_DIR", "data_store"),
        description="运行期数据目录 (parquet/sqlite)",
    )

    # --- LLM 反馈引擎 ---
    llm_backend: str = Field(
        default_factory=lambda: _env("LLM_BACKEND", "none"),
        description="LLM 后端: none|openai|local",
    )
    llm_api_key: str = Field(
        default_factory=lambda: _env("LLM_API_KEY", ""),
        description="LLM API Key (如启用云后端)",
    )
    llm_model: str = Field(
        default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"),
        description="LLM 模型名称",
    )

    # --- 日志 ---
    log_level: str = Field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO"),
        description="structlog 日志等级",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回单例 :class:`Settings`，结果被 lru_cache 缓存。"""
    return Settings()
