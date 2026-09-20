"""调教性能神经网络（setup_sim）—— 纯标准库，无 PyTorch 依赖。

角色（用户明确的优化方式）
------------------------
    「参数矩阵给神经网络模型方向，由神经网络不断模拟优化」

    - 方向：``engine.compute_setup_delta``（Dx × C 矩阵的 6 步流水线）；
    - 模拟优化：本模块提供的**调教→圈速模型**在
      :mod:`setup_tuner.engine.sim_optimizer` 的循环里对候选调教逐一
      「模拟」出预测圈速，选出更优解。

模型来源
--------
``scripts/build_setup_sim_dataset.py``（遥测锚定仿真数据集，80k+ 真实逐弯
样本做锚点）→ ``scripts/train_setup_sim_nn.py``（纯标准库 MLP 训练）→
``data/models/setup_sim_nn.json``。

关键性质
--------
- **零第三方依赖**：推理只用 ``pure_nn.MLP``（标准库 + 算术），
  venv 里没有 torch / numpy 也完全可用 —— 终结「PyTorch 未安装 → 静默降级」。
- **无模型优雅降级**：文件缺失/损坏 → ``available=False``，优化器跳过
  模拟环节，行为与纯规则引擎一致（与 ``track_demand`` 的中性降级同约定）。
- **未知赛道中性降级**：模型只覆盖训练过的赛道；未见过的 ``track_id``
  返回 ``None``，调用方跳过（合成 id 的既有测试不受影响）。
- 单例懒加载 + 线程安全；模型文件是静态资源，加载一次后缓存。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field

from .model_io import resolve_model_path
from .pure_nn import MLP

logger = logging.getLogger(__name__)

#: 模型文件名（解析顺序见 :func:`.model_io.resolve_model_path`）
MODEL_NAME = "setup_sim_nn.json"
DEFAULT_PATH = Path("data") / "models" / MODEL_NAME

#: 赛道聚合特征（顺序即特征向量顺序，训练/推理必须一致）
TRACK_FEATURE_KEYS: tuple[str, ...] = (
    "traction_index", "aero_index", "braking_index", "slow_share", "fast_share",
)
#: 工况特征（干湿/胎温/气温；胎软度来自遥测或默认）
CONDITION_KEYS: tuple[str, ...] = (
    "tyre_softness", "wet", "track_temp", "air_temp",
)


def _resolve_path(path: Path) -> Path:
    """解析权重路径：显式 → 工作目录 data/models/ → 仓库根 → **包内资源**。

    用户可能从任意工作目录启动服务（快捷方式/IDE/其它 cwd），也可能
    ``pip install`` 后根本没有仓库根的 data/ —— 两种情况都必须能找到模型，
    否则表现为"模型训练好了却没生效"。集中实现见 :mod:`.model_io`。
    """
    if path.is_absolute():
        return path
    return resolve_model_path(path.name, path)


# --------------------------------------------------------------------------- #
# 调教归一化（与训练数据构建脚本同一口径，必须一致）
# --------------------------------------------------------------------------- #
def half_range(spec: Any) -> float:
    """参数归一化的半幅（到较远一端），保证 n ∈ [-1, +1]。"""
    return max(abs(spec.max_val - spec.default), abs(spec.default - spec.min_val)) or 1.0


def normalize_setup(real_setup: dict[str, float]) -> dict[str, float]:
    """真实调教值 → 归一化 n ∈ [-1, +1]（相对默认值，与训练口径一致）。"""
    out: dict[str, float] = {}
    for spec in ALL_SETUP_FIELDS:
        value = float(real_setup.get(spec.name, spec.default))
        out[spec.name] = (value - spec.default) / half_range(spec)
    return out


def denormalize_param(name: str, n: float) -> float:
    """归一化 n → 真实参数值（用于从模型视角解释），夹进合法区间。"""
    spec = get_field(name)
    value = spec.default + float(n) * half_range(spec)
    return max(spec.min_val, min(spec.max_val, value))


def build_feature_row(
    track_ids: list[str],
    track_features: dict[str, dict[str, float]],
    track_id: str,
    setup_norm: dict[str, float],
) -> list[float] | None:
    """(赛道, 调教) → 特征向量（训练与推理共用**同一实现**，防止口径漂移）。

    结构：``[赛道 one-hot | 赛道聚合特征 5 | 调教 20 | 工况 4]``。
    未知赛道返回 ``None``（中性降级）。
    """
    tf = track_features.get(track_id)
    if tf is None:
        return None
    onehot = [1.0 if tid == track_id else 0.0 for tid in track_ids]
    agg = [float(tf.get(k, 0.0)) for k in TRACK_FEATURE_KEYS]
    setup = [float(setup_norm.get(spec.name, 0.0)) for spec in ALL_SETUP_FIELDS]
    cond = [
        float(tf.get("tyre_softness", 0.6)),
        float(tf.get("wet", 0.0)),
        float(tf.get("track_temp", 35.0)) / 50.0,
        float(tf.get("air_temp", 24.0)) / 40.0,
    ]
    return onehot + agg + setup + cond


def feature_names(track_ids: list[str]) -> list[str]:
    """特征名（与 :func:`build_feature_row` 顺序一致，供落盘自描述）。"""
    return (
        [f"track={t}" for t in track_ids]
        + list(TRACK_FEATURE_KEYS)
        + [f.name for f in ALL_SETUP_FIELDS]
        + list(CONDITION_KEYS)
    )


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class SetupSimModel:
    """调教性能模型（遥测锚定仿真的神经网络代理）。

    线程安全的懒加载单例由 :func:`get_setup_sim` 管理；本类只管加载与推理。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = _resolve_path(
            Path(path) if path is not None else DEFAULT_PATH,
        )
        self.available = False
        self.reason = "模型未加载"
        self.track_ids: list[str] = []
        self.track_features: dict[str, dict[str, float]] = {}
        self.feature_keys: list[str] = []
        self.metrics: dict[str, Any] = {}
        self.n_samples = 0
        self.created = ""
        self._mlp: MLP | None = None
        self._load()

    # ---------------- 加载 ---------------- #
    def _load(self) -> None:
        if not self.path.exists():
            self.reason = f"模型文件不存在（{self.path}）"
            logger.info("调教性能模型不存在（%s），优化按纯规则运行", self.path)
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.reason = "模型文件读取/解析失败"
            logger.warning("调教性能模型读取失败：%s", self.path, exc_info=True)
            return
        mlp_block = data.get("mlp") or {}
        if not mlp_block.get("model"):
            self.reason = "模型文件缺少 mlp 权重"
            return
        try:
            self._mlp = MLP.from_dict(mlp_block["model"])
        except (KeyError, TypeError, ValueError):
            self.reason = "MLP 权重还原失败"
            logger.warning("调教性能模型权重还原失败", exc_info=True)
            return
        self.track_ids = list(data.get("track_ids") or [])
        self.track_features = data.get("track_features") or {}
        self.feature_keys = list(data.get("feature_keys") or [])
        self.metrics = data.get("metrics") or {}
        self.n_samples = int(data.get("n_samples") or 0)
        self.created = str(data.get("created") or "")
        # 输出缩放（训练端把目标缩到 ~1 量级；推理必须乘回）
        self.y_mean = float(data.get("y_mean") or 0.0)
        self.y_scale = float(data.get("y_scale") or 1.0)
        self.available = bool(self.track_ids) and self._mlp is not None
        self.reason = "ok" if self.available else "模型不完整"

    # ---------------- 特征 ---------------- #
    def build_features(
        self, track_id: str, setup_norm: dict[str, float],
    ) -> list[float] | None:
        """(赛道, 调教) → 特征向量；未知赛道返回 None（中性降级）。"""
        return build_feature_row(self.track_ids, self.track_features, track_id, setup_norm)

    # ---------------- 推理 ---------------- #
    def predict_delta_s(
        self, track_id: str, setup_norm: dict[str, float],
    ) -> float | None:
        """预测该调教相对**默认调教**的圈速增量（秒；正 = 更慢）。

        模型不可用或赛道未知 → ``None``。
        """
        if not self.available or self._mlp is None:
            return None
        row = self.build_features(track_id, setup_norm)
        if row is None:
            return None
        try:
            raw = float(self._mlp.predict_one(row)[0])
            return raw * self.y_scale + self.y_mean
        except (IndexError, KeyError, ValueError):
            logger.warning("调教性能推理失败（track=%s）", track_id, exc_info=True)
            return None

    def predict_for_setup(
        self, track_id: str, real_setup: dict[str, float],
    ) -> float | None:
        """便捷入口：真实调教（含默认值填充）→ 预测圈速增量（秒）。"""
        return self.predict_delta_s(track_id, normalize_setup(real_setup))

    def covers(self, track_id: str) -> bool:
        """该赛道是否在模型覆盖范围（决定是否参与模拟优化）。"""
        return self.available and track_id in self.track_features

    def describe(self) -> str:
        """可读摘要（报告/调试用）。"""
        if not self.available:
            return f"调教性能模型不可用（{self.reason}）"
        m = self.metrics
        mae = m.get("val_mae_ms")
        r2 = m.get("val_r2")
        return (
            f"调教性能 NN（纯标准库，{self.n_samples} 仿真样本 / "
            f"{len(self.track_ids)} 赛道，验证 MAE {mae} ms / R² {r2}）"
        )


# --------------------------------------------------------------------------- #
# 单例（线程安全懒加载）
# --------------------------------------------------------------------------- #
_SIM: SetupSimModel | None = None
_LOCK = threading.Lock()
#: 哨兵：表示"已尝试加载且失败"，避免每次请求重复读盘解析
_LOAD_FAILED: Any = object()


def get_setup_sim(path: str | Path | None = None) -> SetupSimModel:
    """获取调教性能模型单例（懒加载，线程安全）。

    加载失败（异常）时缓存失败哨兵，不反复重试。
    """
    global _SIM
    if _SIM is None:
        with _LOCK:
            if _SIM is None:
                try:
                    _SIM = SetupSimModel(path)
                except Exception:
                    logger.warning("调教性能模型加载异常，禁用模拟优化", exc_info=True)
                    _SIM = _LOAD_FAILED
    if _SIM is _LOAD_FAILED:
        # 失败哨兵对外统一表现为"不可用模型"（属性齐全，调用方无感）
        return _FAILED_MODEL
    return _SIM


def reset_setup_sim() -> None:
    """重置单例（供测试使用）。"""
    global _SIM
    with _LOCK:
        _SIM = None


#: 失败兜底实例：available=False，所有方法中性返回
_FAILED_MODEL = SetupSimModel.__new__(SetupSimModel)
_FAILED_MODEL.path = DEFAULT_PATH
_FAILED_MODEL.available = False
_FAILED_MODEL.reason = "加载失败（哨兵）"
_FAILED_MODEL.track_ids = []
_FAILED_MODEL.track_features = {}
_FAILED_MODEL.feature_keys = []
_FAILED_MODEL.metrics = {}
_FAILED_MODEL.n_samples = 0
_FAILED_MODEL.created = ""
_FAILED_MODEL.y_mean = 0.0
_FAILED_MODEL.y_scale = 1.0
_FAILED_MODEL._mlp = None
