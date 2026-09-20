"""关闭时清理遥测数据（**录制永久保留**）。

用户约定（2026-09-20）
--------------------
> 「遥测数据，除了录制的，其他的关闭后都要删除。」

| 类别 | 路径 | 关闭时 |
|------|------|--------|
| **录制** | `<data_dir>/recordings/`（.f1rec + 逐圈样本） | **保留**（用户要留的原始数据） |
| 模型产物 | `<data_dir>/models/` 与包内 `setup_tuner/resources/models/` | 保留（运行时必需，可复算） |
| 应用数据 | `<data_dir>/f1opt.db`（反馈 / 建议 / 迭代历史） | 保留（用户数据，非遥测） |
| 派生训练数据 | `<data_dir>/training/` | **删除** |
| 模拟包流 | `<data_dir>/sim_telemetry/` | **删除** |
| 旧训练集 | `<data_dir>/training_dataset.json` | **删除** |

安全约束（**不可放宽**）
----------------------
1. **白名单**：只删上表"删除"列的条目；绝不递归删 `data_dir` 本身；
2. **录制保护**：任何路径片段命中 ``recordings`` 一律跳过（双保险）；
3. **数据目录校验**：`data_dir` 解析后不得是文件系统根、用户主目录或桌面
   —— 否则整体跳过（防止配置写错把家目录删了）；
4. **失败不致命**：任何异常只记日志，绝不影响应用退出流程；
5. 支持 ``dry_run`` 先看清单。

用法::

    from setup_tuner.telemetry.cleanup import cleanup_telemetry

    report = cleanup_telemetry("data")          # 清理
    report = cleanup_telemetry("data", dry_run=True)   # 只看会删什么
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: 关闭时必须删除的**子目录**（相对 data_dir；白名单，按名精确匹配）
DELETE_DIRS: tuple[str, ...] = ("training", "sim_telemetry", "replays")
#: 关闭时必须删除的**文件**（相对 data_dir）
DELETE_FILES: tuple[str, ...] = ("training_dataset.json",)
#: 永久保留的目录片段（双保险：路径里出现即跳过）
_PROTECTED_PARTS: frozenset[str] = frozenset({"recordings"})
#: 不允许作为 data_dir 的路径（解析后的绝对路径）
_FORBIDDEN_DATA_DIRS: frozenset[str] = frozenset({"/", "\\"})

#: 环境变量：设为 1 时保留遥测数据（调试 / 想复用派生数据的场景）
KEEP_ENV_VAR = "F1OPT_KEEP_TELEMETRY"


@dataclass(slots=True)
class CleanupReport:
    """清理结果（用于日志与自检）。"""

    data_dir: str = ""
    removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    dry_run: bool = False
    aborted_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.aborted_reason

    def describe(self) -> str:
        if self.aborted_reason:
            return f"清理跳过：{self.aborted_reason}"
        if not self.removed:
            return "清理完成：没有需要删除的遥测数据"
        head = "[模拟] " if self.dry_run else ""
        return (
            f"{head}清理完成：删除 {len(self.removed)} 项（"
            f"{self.freed_bytes / 1024:.0f} KB）：{', '.join(self.removed)}"
        )


def _is_safe_data_dir(path: Path) -> str:
    """校验 data_dir 是否可安全清理；返回空串表示安全，否则返回拒绝原因。"""
    try:
        resolved = path.resolve()
    except OSError:
        return "路径无法解析"
    if str(resolved) in _FORBIDDEN_DATA_DIRS or resolved.parent == resolved:
        return f"拒绝清理文件系统根目录（{resolved}）"
    home = Path.home().resolve()
    if resolved == home:
        return f"拒绝清理用户主目录（{resolved}）"
    for name in ("Desktop", "Documents", "Downloads"):
        if resolved == (home / name):
            return f"拒绝清理个人目录（{resolved}）"
    if any(part.lower() in _PROTECTED_PARTS for part in resolved.parts):
        return f"路径命中受保护片段 recordings（{resolved}）"
    return ""


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total


def cleanup_telemetry(data_dir: str | Path, *, dry_run: bool = False) -> CleanupReport:
    """删除 `data_dir` 下的遥测数据（**录制永久保留**）。

    Args:
        data_dir: 数据目录（通常是 ``Config.data_dir``，默认 ``./data``）。
        dry_run: 只统计与列清单，不真正删除。

    Returns:
        :class:`CleanupReport`；被安全校验拒绝时 ``aborted_reason`` 非空且未删任何东西。
    """
    root = Path(data_dir)
    report = CleanupReport(data_dir=str(root), dry_run=dry_run)

    reason = _is_safe_data_dir(root)
    if reason:
        report.aborted_reason = reason
        logger.warning("遥测清理被拒绝：%s", reason)
        return report
    if not root.exists():
        report.aborted_reason = f"数据目录不存在（{root}）"
        return report

    # 目录（白名单精确匹配 + 录制保护）
    for name in DELETE_DIRS:
        target = root / name
        if not target.is_dir():
            continue
        if any(part.lower() in _PROTECTED_PARTS for part in target.parts):
            report.skipped.append(str(target))
            continue
        # 双保险：目标必须是 data_dir 的直接子目录
        if target.parent.resolve() != root.resolve():
            report.skipped.append(str(target))
            continue
        size = _dir_size(target)
        if dry_run:
            report.removed.append(f"{name}/（{size / 1024:.0f} KB）")
            report.freed_bytes += size
            continue
        try:
            shutil.rmtree(target)
            report.removed.append(f"{name}/")
            report.freed_bytes += size
        except OSError:
            logger.warning("删除遥测目录失败：%s", target, exc_info=True)
            report.skipped.append(str(target))

    # 文件（白名单精确匹配）
    for name in DELETE_FILES:
        target = root / name
        if not target.is_file():
            continue
        size = target.stat().st_size
        if dry_run:
            report.removed.append(f"{name}（{size / 1024:.0f} KB）")
            report.freed_bytes += size
            continue
        try:
            target.unlink()
            report.removed.append(name)
            report.freed_bytes += size
        except OSError:
            logger.warning("删除遥测文件失败：%s", target, exc_info=True)
            report.skipped.append(str(target))

    logger.info("遥测清理：%s", report.describe())
    return report


def keep_telemetry_enabled(env: dict[str, str] | None = None) -> bool:
    """是否按环境变量要求保留遥测数据（``F1OPT_KEEP_TELEMETRY=1``）。"""
    import os

    source = env if env is not None else os.environ
    return str(source.get(KEEP_ENV_VAR, "")).strip().lower() in ("1", "true", "yes", "on")
