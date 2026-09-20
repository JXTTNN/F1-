"""规则引擎层 - 整体性调教建议（Dx×C→SetupDelta + 神经网络模拟优化）。

本包实现 design 2.7 的分析模型：

    症状集合 → Dx(1×9) → × C(9×20) → 原始增量(1×20)
            → 遥测校准 → clamp(区间 + 上限) → 档位对齐 → SetupDelta

神经网络模拟优化（2026-09-19 大改，取代旧的 torch 混合分支）：
    - 纯规则路径（``model_type="rule"``）：确定性 6 步流水线
    - 神经网络模拟优化（``model_type="nn"`` / ``"hybrid"``）：
      规则路径给出方向后，由**遥测锚定仿真训练的调教性能 NN**
      （``setup_sim``，纯标准库推理，无 PyTorch）在坐标上升循环里
      不断模拟候选调教并保留更优解（``sim_optimizer``）
    - 自动降级：模型不可用/赛道未覆盖时回退纯规则路径（逐位一致）

模块导出（对齐 T6 任务接口 + 神经网络模拟优化扩展）：

    - **诊断向量 Dx**（``diagnostic``）：
        - :data:`DIAG_DIMS`、:data:`DIAG_DIMS_ZH`、:data:`SYMPTOM_TO_DX`
        - :func:`compute_dx`
    - **耦合矩阵 C**（``coupling``）：
        - :class:`CouplingCell`、:data:`COUPLING_MATRIX`
        - :func:`get_coupling`、:func:`validate_matrix`
    - **SetupDelta 引擎**（``engine``）：
        - :func:`compute_setup_delta`、:func:`generate_suggestion`
    - **置信度**（``confidence``）：
        - :func:`assess_confidence`
    - **调教性能 NN**（``setup_sim`` + ``sim_optimizer``）：
        - :class:`SetupSimModel`、:func:`get_setup_sim`、:func:`reset_setup_sim`
        - :func:`sim_refine`、:class:`SimRefineResult`

所有函数为纯函数，零 IO、零随机、零时间依赖，满足 FR-ENG-05 / FR-NFR-R1。
神经网络为零第三方依赖（纯标准库），始终可用。
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
    SYMPTOM_DEFAULT_STAGE,
    SYMPTOM_STAGE_TO_DX,
    SYMPTOM_TO_DX,
    compute_dx,
    dx_to_vector,
    empty_dx,
    is_zero_dx,
)
from .engine import compute_setup_delta, generate_suggestion, validate_engine
from .setup_sim import SetupSimModel, get_setup_sim, reset_setup_sim
from .sim_optimizer import SimRefineResult, sim_refine

__all__ = [
    # 诊断向量
    "DIAG_DIMS",
    "DIAG_DIMS_ZH",
    "DIAG_DIMS_POSITIVE_SEMANTICS",
    "SYMPTOM_TO_DX",
    "SYMPTOM_STAGE_TO_DX",
    "SYMPTOM_DEFAULT_STAGE",
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
    # 引擎
    "compute_setup_delta",
    "generate_suggestion",
    "validate_engine",
    # 置信度
    "assess_confidence",
    # 调教性能 NN（模拟优化）
    "SetupSimModel",
    "get_setup_sim",
    "reset_setup_sim",
    "SimRefineResult",
    "sim_refine",
]
