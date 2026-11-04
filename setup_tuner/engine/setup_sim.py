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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field

from .model_io import resolve_model_path
from .pure_nn import MLP

logger = logging.getLogger(__name__)

#: 模型文件名（解析顺序见 :func:`.model_io.resolve_model_path`）
MODEL_NAME = "setup_sim_nn.json"
DEFAULT_PATH = Path("data") / "models" / MODEL_NAME
#: 真实调教-圈速校准文件（由 ``scripts/train_real_lap_model.py`` 产出）。
#: 缺失时纯仿真运行，不报错 —— 旧版本部署/测试自建模型都走这条路径。
CALIBRATION_NAME = "real_lap_calibration.json"

#: 赛道聚合特征（顺序即特征向量顺序，训练/推理必须一致）
TRACK_FEATURE_KEYS: tuple[str, ...] = (
    "traction_index", "aero_index", "braking_index", "slow_share", "fast_share",
)
#: 工况特征（干湿/胎温/气温；胎软度来自遥测或默认）
CONDITION_KEYS: tuple[str, ...] = (
    "tyre_softness", "wet", "track_temp", "air_temp",
)
#: 默认调教的归一化向量（全 0）—— 差分校准的锚点
_ZERO_SETUP: dict[str, float] = {spec.name: 0.0 for spec in ALL_SETUP_FIELDS}


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


def build_norm_vector(real_setup: dict[str, float]) -> list[float]:
    """真实调教 → 归一化向量（顺序 = :data:`ALL_SETUP_FIELDS`）。

    训练端（真实圈速校准）与推理端（``predict_real_delta_s``）共用这一个
    实现，避免「训练时列顺序与推理时不一致」这类静默错位。
    """
    norm = normalize_setup(real_setup)
    return [float(norm[spec.name]) for spec in ALL_SETUP_FIELDS]


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

    def __init__(
        self,
        path: str | Path | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.path = _resolve_path(
            Path(path) if path is not None else DEFAULT_PATH,
        )
        #: 真实校准文件的显式路径（测试用）；None = 按 model_io 搜索规则解析
        self.calibration_path = (
            Path(calibration_path) if calibration_path is not None else None
        )
        self.available = False
        self.reason = "模型未加载"
        self.track_ids: list[str] = []
        self.track_features: dict[str, dict[str, float]] = {}
        self.feature_keys: list[str] = []
        self.metrics: dict[str, Any] = {}
        #: 逐站锚点来源：telemetry=真实遥测；authored=赛道几何+全局常数推定
        self.anchor_sources: dict[str, str] = {}
        self.n_samples = 0
        self.created = ""
        self._mlp: MLP | None = None
        #: 输出缩放：全局兜底（旧模型只有这一对）
        self.y_mean = 0.0
        self.y_scale = 1.0
        #: 输出缩放：逐赛道口径（新模型才有）。同一套 1 s 的标签在 Monaco 与
        #: Monza 不是同一个量级的概念，全局尺度会被「站间差异」吃掉动态范围。
        self.y_mean_by_track: dict[str, float] = {}
        self.y_scale_by_track: dict[str, float] = {}
        #: 每赛道「默认调教」的原始预测（差分校准基准），懒算并缓存
        self._zero_pred: dict[str, float] = {}
        #: 真实圈速校准（可选）：逐站线性系数 + 与仿真的融合权重。
        #: 缺失时保持空，`predict_delta_s` 退化为纯仿真（向后兼容）。
        self.real_track_coef: dict[str, list[float]] = {}
        #: 每站对**纯仿真预测**的系数（堆叠校准的第一项；缺失视为 0）
        self.real_sim_coef: dict[str, float] = {}
        self.real_weight = 0.0
        self.real_weight_by_track: dict[str, float] = {}
        #: 逐站留出集配对准确率（0~1）。Panel 用它告诉用户「这一站有多可信」，
        #: 而不是只显示融合权重（0.9 很容易被读成「准确率 90%」）。
        self.real_accuracy_by_track: dict[str, float] = {}
        self.real_metrics: dict[str, Any] = {}
        self.real_n_samples = 0
        self.real_created = ""
        self.real_reason = "未加载"
        self._load()
        self._load_real_calibration()

    # ---------------- 真实圈速校准（可选层） ---------------- #
    def _load_real_calibration(self) -> None:
        """加载 ``real_lap_calibration.json``；缺失/损坏一律静默降级。

        为什么单独一层而不是重训仿真：仿真标签是自洽的，重训只会更贴近仿真。
        真实校准是**外部真值**，它修正的是仿真里「一组全局系数无法同时适配
        Monza 与 Monaco」的结构性偏差。
        """
        path = (
            self.calibration_path
            if self.calibration_path is not None
            else resolve_model_path(CALIBRATION_NAME)
        )
        if not path.exists():
            self.real_reason = f"校准文件不存在（{path.name}），按纯仿真运行"
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.real_reason = "校准文件读取/解析失败"
            logger.warning("真实圈速校准读取失败：%s", path, exc_info=True)
            return

        names = [spec.name for spec in ALL_SETUP_FIELDS]
        # 按**名字**而不是位置取系数：模型迭代时参数顺序可能变，
        # 名字对齐能避免「静默错位」这类最难查的故障。
        coef: dict[str, list[float]] = {}
        for tid, row in (data.get("track_coef") or {}).items():
            if not isinstance(row, dict):
                continue
            vec: list[float] = []
            for name in names:
                try:
                    vec.append(float(row.get(name, 0.0)))
                except (TypeError, ValueError):
                    vec.append(0.0)
            coef[str(tid)] = vec
        if not coef:
            self.real_reason = "校准文件没有可用的逐站系数"
            return

        def _weight(raw: Any) -> float:
            try:
                w = float(raw)
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, min(1.0, w))

        raw_sim = data.get("track_sim_coef") or {}
        if isinstance(raw_sim, dict):
            for tid, value in raw_sim.items():
                try:
                    self.real_sim_coef[str(tid)] = float(value)
                except (TypeError, ValueError):
                    continue
        self.real_track_coef = coef
        self.real_weight = _weight(data.get("blend_weight"))
        raw_w = data.get("blend_weight_by_track") or {}
        if isinstance(raw_w, dict):
            self.real_weight_by_track = {
                str(k): _weight(v) for k, v in raw_w.items()
            }
        raw_acc = data.get("track_accuracy") or {}
        if isinstance(raw_acc, dict):
            for tid, value in raw_acc.items():
                try:
                    acc = float(value)
                except (TypeError, ValueError):
                    continue
                # 0.5 以下等于随机，当成「没有可用精度」处理，别在面板上误导用户
                if 0.5 < acc <= 1.0:
                    self.real_accuracy_by_track[str(tid)] = acc
        self.real_metrics = data.get("metrics") or {}
        self.real_n_samples = int(data.get("n_samples") or 0)
        self.real_created = str(data.get("created") or "")
        self.real_reason = "ok"

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
        raw_anchors = data.get("anchor_sources") or {}
        self.anchor_sources = (
            {str(k): str(v) for k, v in raw_anchors.items()}
            if isinstance(raw_anchors, dict) else {}
        )
        self.n_samples = int(data.get("n_samples") or 0)
        self.created = str(data.get("created") or "")
        # 输出缩放（训练端把目标缩到 ~1 量级；推理必须乘回）
        self.y_mean = float(data.get("y_mean") or 0.0)
        self.y_scale = float(data.get("y_scale") or 1.0)
        raw_mean = data.get("y_mean_by_track") or {}
        raw_scale = data.get("y_scale_by_track") or {}
        if isinstance(raw_mean, dict) and isinstance(raw_scale, dict):
            self.y_mean_by_track = {
                str(k): float(v) for k, v in raw_mean.items() if v is not None
            }
            self.y_scale_by_track = {
                str(k): float(v) for k, v in raw_scale.items() if v is not None
            }
        self.available = bool(self.track_ids) and self._mlp is not None
        self.reason = "ok" if self.available else "模型不完整"

    # ---------------- 特征 ---------------- #
    def build_features(
        self, track_id: str, setup_norm: dict[str, float],
    ) -> list[float] | None:
        """(赛道, 调教) → 特征向量；未知赛道返回 None（中性降级）。"""
        return build_feature_row(self.track_ids, self.track_features, track_id, setup_norm)

    # ---------------- 推理 ---------------- #
    def output_scale(self, track_id: str) -> tuple[float, float]:
        """该赛道的输出缩放 ``(mean, scale)``：解码 ``y = z * scale + mean``。

        逐赛道口径缺失时退回全局口径 —— 旧模型（只有 ``y_mean``/``y_scale``）
        与测试里手搓的模型都靠这条兜底继续可用。
        """
        mean = self.y_mean_by_track.get(track_id)
        scale = self.y_scale_by_track.get(track_id)
        if mean is None or scale is None or not scale:
            return self.y_mean, self.y_scale
        return mean, scale

    def predict_raw_delta_s(
        self, track_id: str, setup_norm: dict[str, float],
    ) -> float | None:
        """**未校准**的原始预测（秒）。仅供评测/排查，不要直接用于决策。"""
        if not self.available or self._mlp is None:
            return None
        row = self.build_features(track_id, setup_norm)
        if row is None:
            return None
        try:
            raw = float(self._mlp.predict_one(row)[0])
            mean, scale = self.output_scale(track_id)
            return raw * scale + mean
        except (IndexError, KeyError, ValueError):
            logger.warning("调教性能推理失败（track=%s）", track_id, exc_info=True)
            return None

    def sim_delta_s(
        self, track_id: str, setup_norm: dict[str, float],
    ) -> float | None:
        """**纯仿真**的差分预测（秒；正 = 更慢），不含真实校准层。

        单独拆出来有两个用途：一是训练真实校准时必须用不含校准的仿真分量
        （否则会把上一次的模型当标签，形成自我强化）；二是报告要能分别
        展示「仿真怎么说」与「真实数据怎么修正」。
        """
        raw = self.predict_raw_delta_s(track_id, setup_norm)
        if raw is None:
            return None
        base = self._zero_pred.get(track_id)
        if base is None:
            base = self.predict_raw_delta_s(track_id, _ZERO_SETUP)
            if base is None:
                return None
            self._zero_pred[track_id] = base
        return raw - base

    def predict_real_delta_s(
        self,
        track_id: str,
        setup_norm: dict[str, float] | Sequence[float],
        sim_delta: float | None = None,
    ) -> float | None:
        """真实数据校准的预测（秒；正 = 更慢）。无该校准或该站无数据 → None。

        结构是**堆叠**的：``coef·n + sim_coef·仿真预测``。``sim_coef≈1`` 且
        其余系数≈0 时退化为「完全信任仿真」，所以真实数据只能修正它。
        线性且无截距，因此默认调教恒为 0。

        ``setup_norm`` 同时接受 dict 与 :func:`build_norm_vector` 的有序向量
        （顺序 = :data:`ALL_SETUP_FIELDS`）。训练端用的是向量，推理端用的是
        dict，两边都走这一个入口，才能保证系数顺序不会静默错位。
        """
        coef = self.real_track_coef.get(track_id)
        if not coef:
            return None
        if not isinstance(setup_norm, dict):
            setup_norm = {
                spec.name: float(v)
                for spec, v in zip(ALL_SETUP_FIELDS, setup_norm, strict=True)
            }
        total = sum(
            c * float(setup_norm.get(spec.name, 0.0))
            for c, spec in zip(coef, ALL_SETUP_FIELDS, strict=True)
        )
        k = self.real_sim_coef.get(track_id, 0.0)
        if k:
            base = self.sim_delta_s(track_id, setup_norm) if sim_delta is None else sim_delta
            if base is None:
                return None
            total += k * base
        return total

    def predict_delta_s(
        self, track_id: str, setup_norm: dict[str, float],
    ) -> float | None:
        """预测该调教相对**默认调教**的圈速增量（秒；正 = 更慢）。

        做了**差分校准**：``raw(setup) − raw(默认)``。

        为什么必须校准：仿真的定义里「默认调教 = 0」，所以同赛道默认点的
        原始预测就是纯误差，表现为一个每赛道的常数偏置（实测 36~67 ms）。
        模拟优化只比较**同一赛道内两套调教**谁更快，常数偏置对决策没有信息，
        却会污染报告里的绝对数字、也会让「默认调教应预测 0」这条契约失效。
        减去同赛道默认点后：偏置被消掉、``predict(默认) ≡ 0`` 成为可断言性质，
        而任意两套调教的差值保持不变。

        若加载了真实圈速校准，则按该站权重融合：
        ``(1−w)·仿真 + w·真实校准``。两路预测在默认点都为 0，
        所以融合后 ``predict(默认) ≡ 0`` 依然成立。

        模型不可用或赛道未知 → ``None``。
        """
        sim_delta = self.sim_delta_s(track_id, setup_norm)
        if sim_delta is None:
            return None

        weight = self.real_weight_by_track.get(track_id, self.real_weight)
        if weight <= 0.0:
            return sim_delta
        real_delta = self.predict_real_delta_s(track_id, setup_norm, sim_delta)
        if real_delta is None:
            return sim_delta
        return (1.0 - weight) * sim_delta + weight * real_delta

    def predict_for_setup(
        self, track_id: str, real_setup: dict[str, float],
    ) -> float | None:
        """便捷入口：真实调教（含默认值填充）→ 预测圈速增量（秒）。"""
        return self.predict_delta_s(track_id, normalize_setup(real_setup))

    def real_weight_for(self, track_id: str) -> float:
        """该站实际使用的真实校准权重（0 = 纯仿真）。"""
        if track_id not in self.real_track_coef:
            return 0.0
        return self.real_weight_by_track.get(track_id, self.real_weight)

    def real_accuracy_for(self, track_id: str) -> float | None:
        """该站真实校准的留出集配对准确率（0~1）；无数据返回 ``None``。

        这是「这一站的真实校准有多可信」的**唯一**可对外说法。融合权重
        0.9 说的是「9 成听真实数据」，不是「9 成准确」—— 两者差得远，
        面板必须显示前者还是后者要分清楚。
        """
        if track_id not in self.real_track_coef:
            return None
        acc = self.real_accuracy_by_track.get(track_id)
        if acc is None or not 0.0 < acc <= 1.0:
            return None
        return acc

    def covers(self, track_id: str) -> bool:
        """该赛道是否在模型覆盖范围（决定是否参与模拟优化）。"""
        return self.available and track_id in self.track_features

    def anchor_of(self, track_id: str) -> str:
        """该赛道的锚点来源：``telemetry`` / ``authored``（未知赛道返回空串）。

        报告据此如实标注「这条赛道是实测锚定还是几何推定」，避免把推定当实测。
        """
        return self.anchor_sources.get(track_id, "")

    def coverage_counts(self) -> tuple[int, int]:
        """(遥测锚定赛道数, 几何推定赛道数)。"""
        n_auth = sum(1 for v in self.anchor_sources.values() if v != "telemetry")
        return len(self.track_ids) - n_auth, n_auth

    def describe(self) -> str:
        """可读摘要（报告/调试用）。"""
        if not self.available:
            return f"调教性能模型不可用（{self.reason}）"
        m = self.metrics
        mae = m.get("val_mae_ms")
        r2 = m.get("val_r2")
        # 局部 MAE（距该赛道最优 <300 ms 的样本）才是调教优化的决策尺度：
        # 全量程 MAE 被「换掉整套调教」的巨大差异主导，单看它会低估/高估。
        local = m.get("val_mae_ms_local")
        detail = f"验证 MAE {mae} ms"
        if local is not None:
            detail += f" / 局部 {local} ms"
        if r2 is not None:
            detail += f" / R² {r2}"
        if self.y_scale_by_track:
            detail += " / 逐赛道标准化"
        if self.real_track_coef and self.real_weight > 0.0:
            detail += (
                f" / 真实校准 {len(self.real_track_coef)} 站"
                f"（{self.real_n_samples} 条配对，权重 {self.real_weight:.2f}）"
            )
        tele, auth = self.coverage_counts()
        composition = (
            f"{len(self.track_ids)} 赛道" if auth == 0
            else f"{len(self.track_ids)} 赛道（遥测 {tele} + 几何推定 {auth}）"
        )
        return (
            f"调教性能 NN（纯标准库，{self.n_samples} 仿真样本 / "
            f"{composition}，{detail}）"
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
_FAILED_MODEL.anchor_sources = {}
_FAILED_MODEL.n_samples = 0
_FAILED_MODEL.created = ""
_FAILED_MODEL.y_mean = 0.0
_FAILED_MODEL.y_scale = 1.0
_FAILED_MODEL._mlp = None
_FAILED_MODEL._zero_pred = {}
_FAILED_MODEL.real_track_coef = {}
_FAILED_MODEL.real_sim_coef = {}
_FAILED_MODEL.real_sim_coef = {}
_FAILED_MODEL.real_weight = 0.0
_FAILED_MODEL.real_weight_by_track = {}
_FAILED_MODEL.real_accuracy_by_track = {}
_FAILED_MODEL.real_metrics = {}
_FAILED_MODEL.real_n_samples = 0
_FAILED_MODEL.real_created = ""
_FAILED_MODEL.real_reason = "加载失败（哨兵）"
