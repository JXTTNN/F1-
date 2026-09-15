"""数据质量检测模块 — 对遥测摘要执行多项质量检查。

检测项：
1. G力全零检测 — G力数据（lat_g/long_g/vert_g）全为零或不存在
2. 胎压异常检测 — setup_raw中胎压值为科学记数法（绝对值<1e-10或>1e10）
3. 调教值域不匹配检测 — setup_raw中参数值超出游戏内合法范围
4. 采样率检测 — sample_count不足（<100警告，<10错误）
5. 数据缺失检测 — 关键字段不存在或全零
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.telemetry.importer import LapTelemetrySummary

# 胎压字段名
_TYRE_PRESSURE_FIELDS: list[str] = [
    "front_left_tyre_pressure",
    "front_right_tyre_pressure",
    "rear_left_tyre_pressure",
    "rear_right_tyre_pressure",
]

# 科学记数法异常阈值
_SCIENTIFIC_MIN = 1e-10
_SCIENTIFIC_MAX = 1e10

# G力字段名
_G_FORCE_FIELDS: list[str] = ["lat_g", "long_g", "vert_g"]


@dataclass
class QualityIssue:
    """单个质量问题。

    Attributes:
        severity: 严重程度 — "error" / "warning" / "info"
        category: 问题类别 — "g_force" / "tyre_pressure" / "setup_range" /
            "sample_rate" / "data_missing"
        message: 问题描述
        detail: 详细信息（如哪些字段有问题、具体数值等）
    """

    severity: str
    category: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """数据质量报告。

    Attributes:
        issues: 所有检测到的问题
        overall_status: 整体状态 — "pass" / "warning" / "error"
        summary: 中文摘要
    """

    issues: list[QualityIssue] = field(default_factory=list)
    overall_status: str = "pass"
    summary: str = ""


# --------------------------------------------------------------------------- #
# 各检测项辅助函数
# --------------------------------------------------------------------------- #
def _check_g_force(issues: list[QualityIssue], summary: LapTelemetrySummary) -> None:
    """G力全零检测 — 检查G力数据是否存在且非全零。"""
    all_zero_or_missing = True
    for f in _G_FORCE_FIELDS:
        val = getattr(summary, f, None)
        if val is not None and val != 0.0:
            all_zero_or_missing = False
            break
    if all_zero_or_missing:
        issues.append(QualityIssue(
            severity="info",
            category="g_force",
            message="G力数据（lat_g/long_g/vert_g）全为零或不存在",
            detail={"fields": _G_FORCE_FIELDS},
        ))


def _check_tyre_pressure(issues: list[QualityIssue], setup_raw: dict[str, float]) -> None:
    """胎压异常检测 — setup_raw中胎压值为科学记数法。"""
    abnormal: dict[str, float] = {}
    for name in _TYRE_PRESSURE_FIELDS:
        val = setup_raw.get(name)
        if val is not None and (abs(val) < _SCIENTIFIC_MIN or abs(val) > _SCIENTIFIC_MAX):
            abnormal[name] = val
    if abnormal:
        issues.append(QualityIssue(
            severity="error",
            category="tyre_pressure",
            message="setup_raw中胎压值为科学记数法异常值",
            detail={"abnormal_fields": abnormal},
        ))


def _check_setup_range(issues: list[QualityIssue], setup_raw: dict[str, float]) -> None:
    """调教值域不匹配检测 — setup_raw中参数值超出游戏内合法范围。"""
    out_of_range: dict[str, dict[str, Any]] = {}
    for spec in ALL_SETUP_FIELDS:
        val = setup_raw.get(spec.name)
        if val is not None and (val < spec.min_val or val > spec.max_val):
            out_of_range[spec.name] = {
                "value": val,
                "min_val": spec.min_val,
                "max_val": spec.max_val,
                "unit": spec.unit,
            }
    if out_of_range:
        issues.append(QualityIssue(
            severity="warning",
            category="setup_range",
            message="setup_raw中调教参数值超出游戏内合法范围（UDP原始值未转换）",
            detail={"out_of_range_fields": out_of_range},
        ))


def _check_sample_rate(issues: list[QualityIssue], sample_count: int) -> None:
    """采样率检测 — sample_count不足。"""
    if sample_count < 10:
        issues.append(QualityIssue(
            severity="error",
            category="sample_rate",
            message=f"采样点数量严重不足（{sample_count} < 10）",
            detail={"sample_count": sample_count},
        ))
    elif sample_count < 100:
        issues.append(QualityIssue(
            severity="warning",
            category="sample_rate",
            message=f"采样点数量不足（{sample_count} < 100）",
            detail={"sample_count": sample_count},
        ))


def _check_data_missing(issues: list[QualityIssue], summary: LapTelemetrySummary) -> None:
    """数据缺失检测 — 关键字段不存在或全零。"""
    missing: list[str] = []
    if summary.avg_speed == 0.0 and summary.max_speed == 0.0:
        missing.append("speed")
    if summary.avg_throttle == 0.0:
        missing.append("throttle")
    if summary.avg_brake == 0.0 and summary.max_brake == 0.0:
        missing.append("brake")
    if summary.avg_steer == 0.0 and summary.max_steer == 0.0:
        missing.append("steer")
    if all(v == 0.0 for v in summary.avg_tyre_surface_temp):
        missing.append("tyre_surface_temp")
    if all(v == 0.0 for v in summary.avg_tyre_pressure):
        missing.append("tyre_pressure")
    if missing:
        issues.append(QualityIssue(
            severity="warning",
            category="data_missing",
            message="关键字段数据缺失或全零",
            detail={"missing_fields": missing},
        ))


# --------------------------------------------------------------------------- #
# 报告生成辅助函数
# --------------------------------------------------------------------------- #
def _compute_status(issues: list[QualityIssue]) -> str:
    """根据issues列表计算整体状态。"""
    if any(i.severity == "error" for i in issues):
        return "error"
    if any(i.severity == "warning" for i in issues):
        return "warning"
    return "pass"


def _build_summary(issues: list[QualityIssue]) -> str:
    """根据issues列表生成中文摘要。"""
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    info_count = sum(1 for i in issues if i.severity == "info")
    parts: list[str] = []
    if error_count:
        parts.append(f"{error_count}个错误")
    if warning_count:
        parts.append(f"{warning_count}个警告")
    if info_count:
        parts.append(f"{info_count}个提示")
    return "、".join(parts) if parts else "无质量问题"


# --------------------------------------------------------------------------- #
# 公共 API
# --------------------------------------------------------------------------- #
def check_quality(summary: LapTelemetrySummary) -> QualityReport:
    """对单圈遥测摘要执行质量检测，返回报告。

    检测项：
    1. G力全零检测
    2. 胎压异常检测（科学记数法）
    3. 调教值域不匹配检测
    4. 采样率检测
    5. 数据缺失检测

    Args:
        summary: 整圈遥测统计摘要。

    Returns:
        :class:`QualityReport` 数据质量报告。
    """
    issues: list[QualityIssue] = []
    _check_g_force(issues, summary)
    _check_tyre_pressure(issues, summary.setup_raw)
    _check_setup_range(issues, summary.setup_raw)
    _check_sample_rate(issues, summary.sample_count)
    _check_data_missing(issues, summary)
    return QualityReport(
        issues=issues,
        overall_status=_compute_status(issues),
        summary=_build_summary(issues),
    )


def check_quality_batch(summaries: list[LapTelemetrySummary]) -> list[QualityReport]:
    """批量检测多圈数据质量。

    Args:
        summaries: 多圈遥测统计摘要列表。

    Returns:
        每圈对应的 :class:`QualityReport` 列表。
    """
    return [check_quality(s) for s in summaries]