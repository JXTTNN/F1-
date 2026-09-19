"""规则引擎层 - 整体性调教建议（Dx×C→SetupDelta + 混合神经网络）。

本包实现 design 2.7 的分析模型：

    症状集合 → Dx(1×9) → × C(9×23) → 原始增量(1×23)
            → 遥测校准 → clamp(区间 + 上限) → 档位对齐 → SetupDelta

混合模型扩展（task-43）：
    - 纯规则引擎（``model_type="rule"``）：确定性 6 步流水线
    - 纯神经网络（``model_type="nn"``）：PyTorch 神经网络
    - 混合模型（``model_type="hybrid"``）：规则 60% + 神经网络 40%
    - 自动降级：PyTorch 不可用时回退为纯规则引擎

模块导出（对齐 T6 任务接口 + task-43 神经网络扩展）：

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
    - **神经网络模型**（``nn_model``，task-43）：
        - :class:`F1SetupNet`、:class:`NNModelManager`
        - :func:`get_nn_manager`、:func:`is_torch_available`

所有函数为纯函数，零 IO、零随机、零时间依赖，满足 FR-ENG-05 / FR-NFR-R1。
神经网络分支为可选依赖（PyTorch），不可用时自动降级。
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
from .nn_model import (
    F1SetupNet,
    NNModelManager,
    get_nn_manager,
    is_torch_available,
    reset_nn_manager,
)

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
    # 神经网络模型（task-43）
    "F1SetupNet",
    "NNModelManager",
    "get_nn_manager",
    "is_torch_available",
    "reset_nn_manager",
]
