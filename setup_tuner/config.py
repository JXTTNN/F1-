"""本地配置模块 - 从 .env 文件读取运行时配置。

不引入 python-dotenv 依赖，手动解析 .env 文件。
所有配置项均有缺省值，无 .env 文件时使用缺省。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """运行时配置。"""

    # UDP 遥测监听
    udp_host: str = "127.0.0.1"
    udp_port: int = 20777

    # API 服务
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # 数据目录（SQLite 文件存放）
    data_dir: str = "./data"

    # 日志级别
    log_level: str = "INFO"


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """手动解析 .env 文件，返回 key=value 字典。

    支持：
    - 忽略空行和 # 注释行
    - key=value 格式（= 两侧空格可选）
    - value 两侧的引号会被去除
    """
    result: dict[str, str] = {}
    if not env_path.exists():
        return result

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # 去除两侧引号
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value

    return result


def load_config(env_path: str | Path | None = None) -> Config:
    """加载配置。

    优先级：环境变量 > .env 文件 > 缺省值。

    Args:
        env_path: .env 文件路径。None 时默认查找当前目录下的 .env。

    Returns:
        Config 实例。
    """
    env_path = Path.cwd() / ".env" if env_path is None else Path(env_path)

    env_vars = _parse_env_file(env_path)

    def _get(key: str, default: str) -> str:
        """环境变量 > .env 文件 > 缺省值。"""
        return os.environ.get(key, env_vars.get(key, default))

    return Config(
        udp_host=_get("UDP_HOST", "127.0.0.1"),
        udp_port=int(_get("UDP_PORT", "20777")),
        api_host=_get("API_HOST", "127.0.0.1"),
        api_port=int(_get("API_PORT", "8000")),
        data_dir=_get("DATA_DIR", "./data"),
        log_level=_get("LOG_LEVEL", "INFO"),
    )