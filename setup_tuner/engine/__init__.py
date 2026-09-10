"""规则引擎层 - 整体性调教建议（Dx×C→SetupDelta）。

本包实现 design 2.7 的分析模型，纯确定性、零模型、零训练：

    症状集合 → Dx(1×9) → × C(9×23) → 原始增量(1×23)
            → 遥测校准 → clamp(区间 + 上限) → 档位对齐 → SetupDelta

模块导出（对齐 T6 任务接口）：

    - **诊断向量 Dx**（``diagnostic``）：
        - :data:`DIAG_DIMS`、:data:`DIAG_DIMS_ZH`、:data:`SYMPTOM_TO_DX`
        - :func:`compute_dx`
    - **耦合矩阵 C**（``coupling``）：
        - :class:`CouplingCell`、:data:`COUPLING_MATRIX`
        - :func:`get_coupling`、:func:`validate_matrix`
    - **规则库**（``rules``）：
        - :func:`load_rules`、:func:`get_rule`、:func:`get_all_rules`
    - **SetupDelta 引擎**（``engine``）：
        - :func:`compute_setup_delta`、:func:`generate_suggestion`
    - **置信度**（``confidence``）：
        - :func:`assess_confidence`

所有函数为纯函数，零 IO、零随机、零时间依赖，满足 FR-ENG-05 / FR-NFR-R1。
"""

from __future__ import annotations

from .confidence import assess_confidence
from .coupling import (
    COUPLING_MATRIX,
    CouplingCell,
    get_coupling,
    matrix_stats,
    validate_matrix,
)
from .diagnostic import (
    DIAG_DIMS,
    DIAG_DIMS_POSITIVE_SEMANTICS,
    DIAG_DIMS_ZH,
    SYMPTOM_TO_DX,
    compute_dx,
    dx_to_vector,
    empty_dx,
    is_zero_dx,
)
from .engine import compute_setup_delta, generate_suggestion, validate_engine
from .rules import get_all_rules, get_rule, load_rules, validate_rules

__all__ = [
    # 诊断向量
    "DIAG_DIMS",
    "DIAG_DIMS_ZH",
    "DIAG_DIMS_POSITIVE_SEMANTICS",
    "SYMPTOM_TO_DX",
    "compute_dx",
    "empty_dx",
    "is_zero_dx",
    "dx_to_vector",
    # 耦合矩阵
    "CouplingCell",
    "COUPLING_MATRIX",
    "get_coupling",
    "validate_matrix",
    "matrix_stats",
    # 规则库
    "load_rules",
    "get_rule",
    "get_all_rules",
    "validate_rules",
    # 引擎
    "compute_setup_delta",
    "generate_suggestion",
    "validate_engine",
    # 置信度
    "assess_confidence",
]
