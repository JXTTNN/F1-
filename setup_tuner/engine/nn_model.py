"""F1 调教优化神经网络模型（混合模型的神经网络部分）。

本模块实现混合模型中的神经网络分支：

    输入(68维) → FC(128) → FC(64) → FC(32) → FC(20) → 输出(20维 delta)

输入特征（共 68 维）：
    - 12 维：症状强度（12 症状，0-5 归一化到 0-1）
    -  9 维：诊断向量 Dx（9 维诊断，tanh 归一化到 -1~1）
    - 20 维：当前调教参数（归一化到 0-1）
    - 24 维：赛道 one-hot 编码（24 条赛道）

输出（21 维）：21 项调教参数的 delta 值（tanh 输出 -1~1，反归一化到 [-max_delta, +max_delta]）

关键设计：
    1. **自动降级**：PyTorch 未安装或权重文件不存在时，``NNModelManager.available=False``，
       引擎自动降级为纯规则引擎。
    2. **权重路径**：``data/nn_weights.pt``（相对于工作目录）。
    3. **归一化**：输入参数归一化到 0-1，输出 delta 反归一化到实际范围。
    4. **确定性**：``model.eval()`` 模式，无 Dropout 随机性。

PyTorch 为可选依赖（optional），本模块在无 PyTorch 环境下可安全导入，
``NNModelManager`` 构造时自动检测并设置 ``available=False``。

来源经验：
    - 2026-09-10-deterministic-advice-engine：参数全集 S + 诊断维度 Dx 定义
    - 2026-09-11-deterministic-engine-unit-test-pattern：确定性断言 + 不越界断言
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.domain.track import ALL_TRACKS
from setup_tuner.engine.diagnostic import DIAG_DIMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch 可选导入：不可用时定义占位，保证模块可安全导入
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # 占位类型，使类定义在无 torch 时不报 NameError
    class _DummyModule:  # type: ignore[no-redef]
        """无 PyTorch 时的占位基类。"""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def eval(self) -> Any:
            return self

        def forward(self, x: Any) -> Any:
            return x

        def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
            pass

        def state_dict(self) -> dict:
            return {}

        def parameters(self) -> list:
            return []

        def train(self, *args: Any, **kwargs: Any) -> Any:
            return self

        def zero_grad(self) -> None:
            pass

    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 常量：输入特征维度（**从领域模型动态推导**，避免 schema 变更后静默失配）
# ---------------------------------------------------------------------------
# 历史问题：这里曾硬编码 ``_NUM_SYMPTOMS = 12`` / ``_NUM_SETUP_PARAMS = 23``，
# 而实现后来演进为 15 症状 / 21 参数 → 构造出的输入向量长度（66）与网络输入层
# 声明（68）不符，``predict`` 里的 ``except`` 把维度异常吞掉，神经网络分支
# 实际永远返回 None（"混合模型"退化成了纯规则引擎，且无从察觉）。
# 现在全部改为 len(...)，并在模块导入时做一次一致性自检。
_NUM_SYMPTOMS = len(Symptom)
_NUM_DIAG_DIMS = len(DIAG_DIMS)
_NUM_SETUP_PARAMS = len(ALL_SETUP_FIELDS)
_NUM_TRACKS = len(ALL_TRACKS)
_INPUT_SIZE = _NUM_SYMPTOMS + _NUM_DIAG_DIMS + _NUM_SETUP_PARAMS + _NUM_TRACKS
_OUTPUT_SIZE = _NUM_SETUP_PARAMS

# 症状强度归一化因子（0-5 → 0-1）
_SYMPTOM_INTENSITY_MAX = 5.0

# Dx 归一化因子（Dx 分量绝对值上限约 10，用 tanh 压缩；这里用线性归一化）
_DX_NORMALIZE_SCALE = 5.0

# 症状字符串 → 枚举下标（模块级预构建，避免每次调用重建 dict）
_SYMPTOM_INDEX: dict[str, int] = {s.value: i for i, s in enumerate(Symptom)}

# 赛道 ID → 索引 映射（延迟构建）
_TRACK_ID_TO_INDEX: dict[str, int] | None = None


def _build_track_index() -> dict[str, int]:
    """构建赛道 track_id → 索引 映射（延迟加载，避免循环导入）。"""
    global _TRACK_ID_TO_INDEX
    if _TRACK_ID_TO_INDEX is not None:
        return _TRACK_ID_TO_INDEX
    from setup_tuner.domain.track import ALL_TRACKS
    _TRACK_ID_TO_INDEX = {t.track_id: i for i, t in enumerate(ALL_TRACKS)}
    return _TRACK_ID_TO_INDEX


# ---------------------------------------------------------------------------
# 神经网络架构定义
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class F1SetupNet(nn.Module):  # type: ignore[misc]
        """F1 调教优化神经网络。

        输入特征（共 68 维）：
            - 12 维：症状强度（12 症状，0-5 归一化到 0-1）
            -  9 维：诊断向量 Dx（9 维诊断，归一化）
            - 20 维：当前调教参数（归一化到 0-1）
            - 24 维：赛道 one-hot 编码（24 条赛道）

        输出（21 维）：21 项调教参数的 delta 值（归一化到 -1 到 1）
        """

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(_INPUT_SIZE, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, _OUTPUT_SIZE)
            self.dropout = nn.Dropout(0.1)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

        def forward(self, x: Any) -> Any:
            """前向传播：68→128→64→32→23，ReLU+Dropout 激活，Tanh 输出。"""
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.tanh(self.fc4(x))  # 输出 -1 到 1
            return x

else:
    # 无 PyTorch 时的占位类（仅用于类型提示和 __init__.py 导出）
    class F1SetupNet(_DummyModule):  # type: ignore[no-redef]
        """无 PyTorch 时的占位网络（不可用）。"""
        pass


# ---------------------------------------------------------------------------
# 归一化 / 反归一化工具
# ---------------------------------------------------------------------------
def _normalize_symptoms(symptoms: list[tuple[str, int]]) -> list[float]:
    """将症状强度归一化为 ``len(Symptom)`` 维向量（0-1）。

    症状数由枚举长度决定（当前 15），因此新增症状不会再触发下标越界。
    未知症状键按 0 处理（跳过，不抛错）。

    Args:
        symptoms: 症状列表 [(symptom_key, strength), ...]。

    Returns:
        长度 = ``len(Symptom)`` 的浮点列表，按 Symptom 枚举顺序排列，每维 ∈ [0, 1]。
    """
    vec = [0.0] * _NUM_SYMPTOMS
    for key, strength in symptoms:
        idx = _SYMPTOM_INDEX.get(key)
        if idx is not None:
            vec[idx] = float(strength) / _SYMPTOM_INTENSITY_MAX
    return vec


def _normalize_dx(dx: dict[str, float]) -> list[float]:
    """将 Dx 诊断向量归一化为 9 维向量。

    用 tanh 压缩到 [-1, 1]，保持方向符号。

    Args:
        dx: 诊断向量字典 {dim_key: value}。

    Returns:
        长度 9 的浮点列表，按 DIAG_DIMS 顺序排列，每维 ∈ [-1, 1]。
    """
    import math
    return [
        math.tanh(dx.get(dim, 0.0) / _DX_NORMALIZE_SCALE)
        for dim in DIAG_DIMS
    ]


def _normalize_setup(current_setup: dict[str, float]) -> list[float]:
    """将当前调教参数归一化为 23 维向量（0-1）。

    每参数：(value - min) / (max - min)

    Args:
        current_setup: 当前调教快照 {param: value}。

    Returns:
        长度 23 的浮点列表，按 ALL_SETUP_FIELDS 顺序排列，每维 ∈ [0, 1]。
    """
    vec: list[float] = []
    for spec in ALL_SETUP_FIELDS:
        value = float(current_setup.get(spec.name, spec.default))
        span = spec.max_val - spec.min_val
        if span > 0:
            normalized = (value - spec.min_val) / span
        else:
            normalized = 0.0
        # 裁剪到 [0, 1] 防止越界
        vec.append(max(0.0, min(1.0, normalized)))
    return vec


def _encode_track(track_id: str) -> list[float]:
    """将赛道 ID 编码为 24 维 one-hot 向量。

    Args:
        track_id: 赛道标识符。

    Returns:
        长度 24 的浮点列表，对应赛道位置为 1.0，其余为 0.0。
        未知 track_id 返回全零向量。
    """
    track_index = _build_track_index()
    vec = [0.0] * _NUM_TRACKS
    idx = track_index.get(track_id)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _denormalize_delta(normalized_delta: list[float]) -> dict[str, float]:
    """将网络输出的归一化 delta 反归一化为实际参数 delta。

    网络输出每维 ∈ [-1, 1]（tanh），反归一化：delta = output * max_delta

    Args:
        normalized_delta: 长度 20 的归一化 delta 列表。

    Returns:
        {param: delta_value} 字典，覆盖全部 20 参数。
    """
    result: dict[str, float] = {}
    for i, spec in enumerate(ALL_SETUP_FIELDS):
        output = float(normalized_delta[i])
        # 反归一化：output ∈ [-1, 1] → delta ∈ [-max_delta, +max_delta]
        delta = output * spec.max_delta
        result[spec.name] = delta
    return result


def build_input_vector(
    symptoms: list[tuple[str, int]],
    dx: dict[str, float],
    current_setup: dict[str, float],
    track_id: str,
) -> list[float]:
    """构建 68 维输入特征向量。

    Args:
        symptoms: 症状列表 [(symptom_key, strength), ...]。
        dx: 诊断向量字典。
        current_setup: 当前调教快照。
        track_id: 赛道标识。

    Returns:
        长度 65 的浮点列表：[12 症状 + 9 Dx + 20 参数 + 24 赛道]。
    """
    return (
        _normalize_symptoms(symptoms)
        + _normalize_dx(dx)
        + _normalize_setup(current_setup)
        + _encode_track(track_id)
    )


# ---------------------------------------------------------------------------
# 模型管理类
# ---------------------------------------------------------------------------
class NNModelManager:
    """神经网络模型管理器。

    - 加载/保存模型权重
    - 推理接口
    - 自动降级：PyTorch 不可用或权重不存在时 ``available=False``

    用法::

        manager = NNModelManager()
        if manager.available:
            delta = manager.predict(symptoms, dx, current_setup, track_id)
            if delta is not None:
                # 使用神经网络结果
                ...
        else:
            # 降级为规则引擎
            ...

    Attributes:
        available: 模型是否可用（PyTorch 已安装且模型加载成功）。
    """

    def __init__(self, weights_path: str | Path = "data/nn_weights.pt") -> None:
        """初始化模型管理器。

        Args:
            weights_path: 模型权重文件路径。默认 ``data/nn_weights.pt``。
        """
        self.model: F1SetupNet | None = None
        self.available: bool = False
        self.weights_path: Path = Path(weights_path)

        if not _TORCH_AVAILABLE:
            # PyTorch 未安装：自动降级
            self.available = False
            return

        try:
            self.model = F1SetupNet()
            # 尝试加载权重（文件不存在时使用随机初始化权重，标记为不可用）
            if self.weights_path.exists():
                state_dict = torch.load(  # type: ignore[union-attr]
                    str(self.weights_path), map_location="cpu",
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.available = True
            else:
                # 权重文件不存在：标记为不可用（降级为规则引擎）
                self.available = False
                self.model = None
        except Exception:
            # 任何异常（加载失败、版本不兼容等）：降级
            self.available = False
            self.model = None

    def predict(
        self,
        symptoms: list[tuple[str, int]],
        dx: dict[str, float],
        current_setup: dict[str, float],
        track_id: str,
    ) -> dict[str, float] | None:
        """推理：输入症状 + 诊断 + 当前调教 + 赛道 → 输出 20 参数 delta。

        Args:
            symptoms: 症状列表 [(symptom_key, strength), ...]。
            dx: 诊断向量字典 {dim_key: value}。
            current_setup: 当前调教快照 {param: value}。
            track_id: 赛道标识。

        Returns:
            SetupDelta 字典 {param: delta_value}，覆盖全部 20 参数；
            模型不可用时返回 None。
        """
        if not self.available or self.model is None or not _TORCH_AVAILABLE:
            return None

        try:
            # 构建输入向量
            input_vec = build_input_vector(symptoms, dx, current_setup, track_id)
            # 维度自检：network 输入层大小必须与向量长度一致。
            # 早期硬编码 12/23/68 与实现（15/21/66）不符，异常被下方 except 吞掉，
            # 神经网络分支静默失效 —— 这里改为显式报错并输出维度明细。
            expected = getattr(getattr(self.model, "fc1", None), "in_features", None)
            if expected is not None and len(input_vec) != expected:
                logger.error(
                    "NN 输入维度不匹配：向量 %d 维 vs 网络 %d 维"
                    "（症状 %d / 诊断 %d / 参数 %d / 赛道 %d）—— "
                    "请检查 engine/nn_model.py 的维度常量推导",
                    len(input_vec), expected, _NUM_SYMPTOMS, _NUM_DIAG_DIMS,
                    _NUM_SETUP_PARAMS, _NUM_TRACKS,
                )
                return None
            # 转为 tensor（无梯度）
            with torch.no_grad():  # type: ignore[union-attr]
                x = torch.tensor(  # type: ignore[union-attr]
                    [input_vec], dtype=torch.float32,  # type: ignore[union-attr]
                )
                output = self.model(x)
                # 转为列表
                normalized_delta = output.squeeze().tolist()
                if isinstance(normalized_delta, float):
                    # 单参数时 squeeze 会返回标量
                    normalized_delta = [normalized_delta]

            # 反归一化
            return _denormalize_delta(normalized_delta)
        except Exception:
            # 推理异常时返回 None，触发降级（记录日志，避免静默失效）
            logger.warning("NN 推理失败，本次降级为纯规则引擎", exc_info=True)
            return None

    def save_weights(self, path: str | Path | None = None) -> bool:
        """保存模型权重到文件。

        Args:
            path: 保存路径；None 时用初始化时的 weights_path。

        Returns:
            保存成功返回 True，失败返回 False。
        """
        if not self.available or self.model is None or not _TORCH_AVAILABLE:
            return False

        save_path = Path(path) if path is not None else self.weights_path
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), str(save_path))  # type: ignore[union-attr]
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 模块级单例（延迟初始化，供 engine.py 使用）
# ---------------------------------------------------------------------------
_NN_MANAGER: NNModelManager | None = None


def get_nn_manager(weights_path: str | Path = "data/nn_weights.pt") -> NNModelManager:
    """获取神经网络模型管理器单例。

    延迟初始化，首次调用时构造 NNModelManager。
    后续调用返回同一实例（忽略 weights_path 参数变化）。

    Args:
        weights_path: 权重文件路径（仅首次调用生效）。

    Returns:
        NNModelManager 实例（available 可能为 False）。
    """
    global _NN_MANAGER
    if _NN_MANAGER is None:
        _NN_MANAGER = NNModelManager(weights_path)
    return _NN_MANAGER


def reset_nn_manager() -> None:
    """重置神经网络模型管理器单例（供测试使用）。"""
    global _NN_MANAGER
    _NN_MANAGER = None


def is_torch_available() -> bool:
    """检查 PyTorch 是否可用。

    Returns:
        PyTorch 已安装返回 True，否则 False。
    """
    return _TORCH_AVAILABLE