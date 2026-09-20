"""本地配置模块 - 从 .env 文件读取运行时配置。

不引入 python-dotenv 依赖，手动解析 .env 文件。
所有配置项均有缺省值，无 .env 文件时使用缺省。

**数据目录口径（2026-09-20 起）**：默认数据目录固定在**安装文件夹内**
（`<安装根>/data`），不再跟随当前工作目录 —— 用户要求"所有数据只能在安装
文件夹中"，且"从 GitHub 下载安装的就是完整无误的系统，和本地一样"。
安装根 = 含 `setup_tuner/` 包的目录：

- 便携包 / 仓库开发：仓库根（`.bat`、`README.md` 所在处）
- `pip install`：虚拟环境的 `Lib/site-packages/`

这样无论从哪里启动（双击、IDE、任意 cwd），录制 / 数据库 / 派生数据都落在
安装文件夹内，不会散落到用户主目录或其它位置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _write_frozen_diag() -> None:
    """冻结形态下把"安装根判定依据"写到系统临时目录（排障用，失败静默）。

    背景：便携版的安装根判定依赖冻结框架的内部变量（argv[0] / __file__ /
    __compiled__.containing_dir），不同 Nuitka 版本行为有差异 —— 一旦判错，
    数据会落到解包临时目录（退出即丢）或构建期目录。留一份现场记录，
    出问题时不靠猜。
    """
    if not _is_frozen():
        return          # 源码/普通安装不写（避免污染临时目录）
    try:
        import json
        import sys
        import tempfile

        compiled = globals().get("__compiled__")
        payload = {
            "frozen": _is_frozen(),
            "argv0": sys.argv[0] if sys.argv else None,
            "file": __file__,
            "executable": sys.executable,
            "containing_dir": getattr(compiled, "containing_dir", None),
            "install_root": str(install_root()),
            "resolved_data_dir": default_data_dir(),
        }
        target = Path(tempfile.gettempdir()) / "f1opt_frozen_diag.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - 诊断绝不能影响主流程
        pass


def _is_frozen() -> bool:
    """是否运行在**冻结打包**产物里（PyInstaller / Nuitka）。

    官方口径（核实自 Nuitka 文档）：
    - Nuitka **不设置** ``sys.frozen``，而是在每个编译模块注入 ``__compiled__``；
    - PyInstaller 设置 ``sys.frozen``。
    两者都认，避免"冻结形态没识别出来 → 数据写进解包临时目录"。
    """
    import sys

    if getattr(sys, "frozen", False):           # PyInstaller
        return True
    if globals().get("__compiled__") is not None:  # Nuitka（本模块被编译时）
        return True
    if hasattr(sys, "nuitka_version"):
        return True
    main = sys.modules.get("__main__")
    return main is not None and hasattr(main, "__compiled__")


def _frozen_install_root() -> Path | None:
    """冻结打包时的安装根目录；非冻结返回 None。

    解析顺序（**实测校正**，2026-09-20 CI）：
    1. ``sys.argv[0]`` 所在目录 —— 官方明确：onefile 下 argv[0] 就是**原始可执行
       文件**路径，最可靠（便携版解压到哪就以哪为安装根）；
    2. ``__compiled__.containing_dir`` —— 官方推荐用于 standalone/App Bundle，
       但 onefile 下它是**构建期**的 dist 目录（CI 机器上的路径），用户机上不存在
       → 只作为回退，且需通过"该目录里确实有同名可执行文件"的校验；
    3. ``sys.executable`` 所在目录 —— 兜底（Nuitka 下可能指向解释器，不可靠）。
    """
    import sys

    if not _is_frozen():
        return None

    argv0 = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if argv0 is not None and argv0.suffix.lower() not in (".py", ".pyc", ".pyo"):
        # argv[0] 就是当前 exe 本身 → 其目录即安装根
        return argv0.parent

    candidates: list[Path] = []
    containing = getattr(globals().get("__compiled__"), "containing_dir", None)
    if containing:
        candidates.append(Path(containing).resolve())
    if sys.executable:
        candidates.append(Path(sys.executable).resolve().parent)
    # 回退候选必须"确实包含当前可执行文件"，否则很可能指向构建期目录
    exe_name = argv0.name if argv0 is not None else Path(sys.executable).name
    for path in candidates:
        if exe_name and (path / exe_name).exists():
            return path
    return candidates[0] if candidates else None


def install_root() -> Path:
    """安装根目录（含 `setup_tuner/` 包、或可执行文件所在的那一层）。

    - 常规安装/源码运行：本文件位于 `<安装根>/setup_tuner/config.py` → 上一级
    - **冻结打包**（便携版单文件）：`__file__` 指向运行时解包出来的临时目录
      （退出即删），不能用来放数据 → 改用可执行文件所在目录（用户解压的地方）
    """
    frozen = _frozen_install_root()
    if frozen is not None:
        return frozen
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> str:
    """默认数据目录：**安装文件夹内**的 `data/`（与 cwd 无关）。"""
    return str(install_root() / "data")


@dataclass(frozen=True)
class Config:
    """运行时配置。"""

    # UDP 遥测监听
    # 默认绑所有网卡（而非仅回环）：F1 游戏「UDP 广播模式」开启时，包会发往
    # 255.255.255.255 广播地址，只绑 127.0.0.1 会**一个包都收不到**（真实踩坑）。
    # 绑 0.0.0.0 可同时接收：单播到回环/局域网 IP + 广播。
    udp_host: str = "0.0.0.0"
    udp_port: int = 20777

    # API 服务
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # 数据目录（录制 / SQLite / 派生数据）。
    # **默认 = 安装文件夹内的 data/**（见 default_data_dir）；显式设置
    # `F1OPT_DATA_DIR` 才会改到别处（高级用法，不推荐）。
    data_dir: str = ""    # 空串 → 由 resolve_data_dir() 解析为安装文件夹内 data/

    # 关闭时是否保留遥测数据（默认 False = 按用户约定清理，录制永久保留）。
    # 对应环境变量 F1OPT_KEEP_TELEMETRY=1（或 KEEP_TELEMETRY=1）。
    keep_telemetry: bool = False

    # 日志级别
    log_level: str = "INFO"

    def resolved_data_dir(self) -> str:
        """实际使用的数据目录（空 `data_dir` → 安装文件夹内的 `data/`）。"""
        return self.data_dir or default_data_dir()


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
        """环境变量 > .env 文件 > 缺省值。

        **双键查找**（2026-09-20 修正）：README 一直文档化的是 ``F1OPT_*`` 前缀，
        而代码只读无前缀键（``UDP_PORT`` 等）—— 文档化的接口实际不生效，
        用户按 README 设 ``F1OPT_DATA_DIR`` 会被静默忽略。现两者都认，
        带前缀优先（避免与系统同名变量冲突）。
        """
        for name in (f"F1OPT_{key}", key):
            if name in os.environ:
                return os.environ[name]
        for name in (f"F1OPT_{key}", key):
            if name in env_vars:
                return env_vars[name]
        return default

    return Config(
        udp_host=_get("UDP_HOST", "0.0.0.0"),
        udp_port=int(_get("UDP_PORT", "20777")),
        api_host=_get("API_HOST", "127.0.0.1"),
        api_port=int(_get("API_PORT", "8000")),
        # 默认空串 = 安装文件夹内的 data/（见 Config.resolved_data_dir）
        data_dir=_get("DATA_DIR", ""),
        keep_telemetry=_get("KEEP_TELEMETRY", "").strip().lower()
        in ("1", "true", "yes", "on"),
        log_level=_get("LOG_LEVEL", "INFO"),
    )