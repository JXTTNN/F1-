"""报告组装层 - 建议报告生成。

导出：
    - :func:`build_report` — 组装 design 2.7.7 格式的报告 JSON
    - :func:`format_linkages` — 格式化联动说明
    - :func:`build_summary` — 生成摘要文本
    - :func:`extract_setup_from_packet5` — 从遥测 Packet 5 提取 23 参数快照
    - :func:`extract_telemetry_summary` — 从遥测帧缓存提取遥测摘要
    - :func:`feedbacks_to_symptoms` — 反馈记录转症状列表
"""

from __future__ import annotations

from .builder import (
    build_report,
    build_summary,
    extract_setup_from_packet5,
    extract_telemetry_summary,
    feedbacks_to_symptoms,
    format_linkages,
)

__all__ = [
    "build_report",
    "build_summary",
    "format_linkages",
    "extract_setup_from_packet5",
    "extract_telemetry_summary",
    "feedbacks_to_symptoms",
]
