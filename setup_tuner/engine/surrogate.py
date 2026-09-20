"""遥测代理模型（surrogate）—— 把「训练出来的模型」真正接进调教优化。

加载 :mod:`scripts.train_telemetry_surrogate` 产出的权重，供引擎使用：

1. **逐弯重要度（数据驱动）** —— 取代 ``lap_model`` 里
   ``importance = 120 / 该弯参考速度`` 这条纯经验启发式。权重来自
   79k+ 个真实 2026 逐弯样本拟合出的"该弯占整圈时间的比例"。
2. **逐弯期望时间与残差** —— 给定赛道/弯号/工况，模型给出该弯的期望
   占比与期望秒数；实际遥测若明显更慢，即为**车手未反馈的问题**，
   由 :mod:`setup_tuner.engine.telemetry_diagnosis` 转成隐式症状。
3. **整圈配速头（含 setup 与车手风格）** —— 只有在留一交叉验证中真的
   优于"同赛道均值"基线时 ``skill > 0``；否则引擎不使用它，
   避免用没学到东西的模型去覆盖物理方向。

设计约束
--------
- 纯函数式读取：加载一次后缓存（模型是静态文件）。
- **无模型时全部优雅降级**（返回 ``None`` / 中性值），不抛错 ——
  与 ``track_demand`` 对未知赛道的中性降级同一约定。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_io import resolve_model_path
from .pure_nn import MLP

logger = logging.getLogger(__name__)

#: 模型文件名（解析顺序见 :func:`.model_io.resolve_model_path`）
MODEL_NAME = "telemetry_surrogate.json"
DEFAULT_PATH = Path("data") / "models" / MODEL_NAME


def _resolve_path(path: Path) -> Path:
    """解析权重路径：显式 → 工作目录 data/models/ → 仓库根 → **包内资源**。

    为什么需要多级回退：用户可能从任意工作目录启动服务（快捷方式/IDE/其它
    cwd），也可能 ``pip install`` 后没有仓库根的 data/ —— 两种情况都必须能
    找到模型。集中实现见 :mod:`.model_io`。
    """
    if path.is_absolute():
        return path
    return resolve_model_path(path.name, path)


@dataclass(frozen=True, slots=True)
class CornerExpectation:
    """某个弯在给定工况下的期望表现。"""

    track_id: str
    corner: int
    share: float            # 该弯占整圈时间的比例（0..1）
    expected_s: float       # 期望通过时间（秒，= share × 该赛道中位圈速）
    source: str = "telemetry-2026"


class SurrogateModel:
    """遥测代理模型的运行时封装（线程安全的懒加载单例）。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = _resolve_path(
            Path(path) if path is not None else DEFAULT_PATH,
        )
        self.available = False
        self.corner_active: str = "none"
        self.pair_keys: list[str] = []
        self.corner_feature_keys: list[str] = []
        self.importance: dict[str, dict[str, float]] = {}
        self.expected_s: dict[str, dict[str, float]] = {}
        self.lap_skill: float = 0.0
        self.lap_metrics: dict[str, Any] = {}
        self.corner_metrics: dict[str, Any] = {}
        self.trained_samples: int = 0
        self._ridge: dict[str, Any] | None = None
        self._mlp: MLP | None = None
        self._lap_ridge: dict[str, Any] | None = None
        self._lap_setup_names: list[str] = []
        self._lap_feature_keys: list[str] = []
        self._load()

    # ---------------- 加载 ---------------- #
    def _load(self) -> None:
        if not self.path.exists():
            logger.info("遥测代理模型不存在（%s），调教优化按纯物理规则运行", self.path)
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("遥测代理模型读取失败：%s", self.path, exc_info=True)
            return
        head = data.get("corner_head") or {}
        if head:
            self.pair_keys = list(head.get("pair_keys") or [])
            self.corner_feature_keys = list(head.get("feature_keys") or [])
            self.importance = head.get("corner_importance") or {}
            self.expected_s = head.get("expected_corner_time_s") or {}
            self.trained_samples = int(head.get("n_samples") or 0)
            active = head.get("active") or {}
            self.corner_active = str(active.get("kind") or "ridge")
            self.corner_metrics = active.get("metrics") or {}
            self._ridge = (((head.get("family") or {}).get("ridge") or {}).get("model"))
            mlp_block = head.get("mlp") or {}
            if self.corner_active == "mlp" and mlp_block.get("model"):
                try:
                    self._mlp = MLP.from_dict(mlp_block["model"])
                except (KeyError, TypeError, ValueError):
                    logger.warning("MLP 权重还原失败，回退到岭回归", exc_info=True)
                    self.corner_active = "ridge"
                    self._mlp = None
        lap = data.get("lap_head") or {}
        if lap.get("available"):
            self.lap_skill = float((lap.get("metrics") or {}).get("skill") or 0.0)
            self.lap_metrics = lap.get("metrics") or {}
            self._lap_ridge = lap.get("model")
            self._lap_setup_names = list(lap.get("setup_names") or [])
            self._lap_feature_keys = list(lap.get("feature_keys") or [])
        self.available = bool(self.importance) and (
            self._ridge is not None or self._mlp is not None
        )

    # ---------------- 逐弯重要度 ---------------- #
    def corner_importance(self, track_id: str) -> dict[int, float] | None:
        """该赛道的逐弯重要度（弯号 → 权重，归一化到和为 1）。

        数据来源：79k+ 个 2026 真实逐弯样本拟合出的弯时占比。
        无该赛道数据时返回 ``None``（调用方回退到启发式权重）。
        """
        table = self.importance.get(track_id)
        if not table:
            return None
        return {int(k): float(v) for k, v in table.items()}

    def expected_corner_s(self, track_id: str) -> dict[int, float] | None:
        """该赛道各弯的期望通过时间（秒）。"""
        table = self.expected_s.get(track_id)
        if not table:
            return None
        return {int(k): float(v) for k, v in table.items()}

    # ---------------- 弯级推理 ---------------- #
    def _corner_row(
        self, track_id: str, corner: int, conditions: dict[str, Any],
    ) -> list[float]:
        """构造与训练时同口径的特征向量。"""
        onehot = [0.0] * len(self.pair_keys)
        key = f"{track_id}#{corner}"
        try:
            idx = self.pair_keys.index(key)
            onehot[idx] = 1.0
        except ValueError:
            # 未见过该弯：整行留零 → 预测回落到条件项的全局均值，
            # 不抛错（未知赛道/弯号必须中性降级）
            pass
        cls = str(conditions.get("corner_class") or "medium").lower()
        tyre = str(conditions.get("tyre_class") or "medium").lower()
        return onehot + [
            1.0 if cls == "slow" else 0.0,
            1.0 if cls == "medium" else 0.0,
            1.0 if cls == "fast" else 0.0,
            float(conditions.get("tyre_softness") or 0.6),
            1.0 if tyre == "medium" else 0.0,
            1.0 if tyre in ("wet", "inter") else 0.0,
            float(conditions.get("wet") or 0.0),
            float(conditions.get("track_temp") or 35.0),
            float(conditions.get("air_temp") or 24.0),
            float(conditions.get("traction_index") or 0.0),
            float(conditions.get("aero_index") or 0.0),
            float(conditions.get("braking_index") or 0.0),
            float(conditions.get("slow_share") or 0.0),
            float(conditions.get("fast_share") or 0.0),
        ]

    def predict_corner_share(
        self, track_id: str, corner: int, conditions: dict[str, Any],
    ) -> float | None:
        """预测该弯占整圈时间的比例。模型不可用时返回 None。"""
        if not self.available:
            return None
        row = self._corner_row(track_id, corner, conditions)
        try:
            if self.corner_active == "mlp" and self._mlp is not None:
                return float(self._mlp.predict_one(row)[0])
            if self._ridge is not None:
                coef = self._ridge["coef"]
                mean = self._ridge["mean"]
                std = self._ridge["std"]
                s = float(self._ridge["intercept"])
                for j, c in enumerate(coef):
                    s += c * (row[j] - mean[j]) / std[j]
                return float(s)
        except (IndexError, KeyError, ValueError):
            logger.warning("弯级推理失败（track=%s corner=%s）", track_id, corner,
                           exc_info=True)
        return None

    # ---------------- 整圈配速头 ---------------- #
    @property
    def lap_head_usable(self) -> bool:
        """整圈配速头是否值得使用（必须真正优于同赛道均值基线）。"""
        return self.lap_skill > 0.0 and self._lap_ridge is not None

    def predict_lap_ms(self, feature_row: list[float]) -> float | None:
        """整圈圈速预测（ms）。仅在 ``lap_head_usable`` 为真时有意义。"""
        if not self.lap_head_usable or self._lap_ridge is None:
            return None
        try:
            coef = self._lap_ridge["coef"]
            mean = self._lap_ridge["mean"]
            std = self._lap_ridge["std"]
            s = float(self._lap_ridge["intercept"])
            for j, c in enumerate(coef):
                s += c * (feature_row[j] - mean[j]) / std[j]
            return float(s)
        except (IndexError, KeyError, ValueError):
            return None

    def describe(self) -> str:
        """可读摘要（用于报告与调试）。"""
        if not self.available:
            return "遥测代理模型不可用（按纯物理规则运行）"
        metrics = self.corner_metrics
        mae = metrics.get("val_mae")
        r2 = metrics.get("val_r2")
        parts = [
            f"弯速模型 {self.corner_active}（{self.trained_samples} 样本，"
            f"MAE {mae} 占比 / R² {r2}）"
        ]
        if self.lap_head_usable:
            parts.append(f"整圈配速头 skill={self.lap_skill:.3f}")
        else:
            parts.append("整圈配速头未达标（skill=0，不参与）")
        return "；".join(parts)


_SURROGATE: SurrogateModel | None = None
_LOCK = threading.Lock()


def get_surrogate(path: str | Path | None = None) -> SurrogateModel:
    """获取代理模型单例（懒加载，线程安全）。"""
    global _SURROGATE
    if _SURROGATE is None:
        with _LOCK:
            if _SURROGATE is None:
                _SURROGATE = SurrogateModel(path)
    return _SURROGATE


def reset_surrogate() -> None:
    """重置单例（供测试使用）。"""
    global _SURROGATE
    with _LOCK:
        _SURROGATE = None
