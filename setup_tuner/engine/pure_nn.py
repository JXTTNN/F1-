"""纯标准库多层感知机（MLP）—— 不依赖 PyTorch / numpy 的可训练模型。

为什么需要它
------------
引擎原本的神经网络分支（:mod:`setup_tuner.engine.nn_model`）依赖 PyTorch，
而本项目的运行环境（venv）里没有 torch，也没有 numpy：
``NNModelManager.available`` 恒为 ``False``，``model_type="hybrid"`` / ``"nn"``
**静默降级为纯规则引擎** —— 所谓"神经网络参与调教优化"从未真正发生。

本模块提供一个**零第三方依赖**的 MLP，训练与推理全部用标准库实现，
使得"用遥测数据训练模型并应用到调教优化"成为可运行的事实，而不是声明。

实现要点
--------
- 结构：``Dense(relu) × n_hidden`` → ``Dense(linear)``，支持多输出（多目标头）。
- 损失：MSE；优化器：Adam（含偏置修正）；小批量 + 确定性洗牌（固定种子）。
- 标准化：每特征 ``(x - mean) / std``，随权重一起持久化（推理必须同源口径）。
- 数值：纯 ``float`` 运算；梯度手工反传，不依赖自动微分。
- 确定性：给定 ``seed`` 时训练过程完全可复现（无时间/随机源依赖）。

用法::

    from setup_tuner.engine.pure_nn import MLP

    model = MLP([n_in, 32, 16, n_out], seed=42)
    model.fit(X, y, epochs=300, lr=0.01, batch_size=32)
    preds = model.predict(X)
    model.save("data/models/x.json")
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

__all__ = ["MLP", "standardize", "r2_score", "mean_abs_error"]


# --------------------------------------------------------------------------- #
# 数值工具
# --------------------------------------------------------------------------- #
def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _mat(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _relu(v: float) -> float:
    return v if v > 0.0 else 0.0


def _relu_grad(v: float) -> float:
    return 1.0 if v > 0.0 else 0.0


def standardize(
    x: list[list[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    """按列标准化到零均值单位方差；返回 (标准化数据, mean, std)。

    std 为 0 的列（常量列）取 1.0，避免除零。
    """
    if not x:
        return [], [], []
    n_cols = len(x[0])
    mean = [0.0] * n_cols
    for row in x:
        for j in range(n_cols):
            mean[j] += row[j]
    for j in range(n_cols):
        mean[j] /= len(x)
    std = [0.0] * n_cols
    for row in x:
        for j in range(n_cols):
            d = row[j] - mean[j]
            std[j] += d * d
    for j in range(n_cols):
        std[j] = math.sqrt(std[j] / len(x))
        if std[j] < 1e-9:
            std[j] = 1.0
    out = [[(row[j] - mean[j]) / std[j] for j in range(n_cols)] for row in x]
    return out, mean, std


def mean_abs_error(y_true: list[float], y_pred: list[float]) -> float:
    """平均绝对误差（MAE）。"""
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    """决定系数 R²（1.0 = 完美，0.0 = 等于均值基线，负 = 比基线差）。"""
    if not y_true:
        return 0.0
    mean = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True))
    ss_tot = sum((a - mean) ** 2 for a in y_true)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class MLP:
    """多层感知机（纯标准库实现，支持多输出）。"""

    def __init__(self, sizes: list[int], seed: int = 42) -> None:
        """构造网络。

        Args:
            sizes: 各层神经元数，形如 ``[n_in, h1, h2, n_out]``；至少 3 层。
            seed: 权重初始化种子（保证确定性）。
        """
        if len(sizes) < 3:
            raise ValueError(f"sizes 至少需要 3 层（输入/隐藏/输出），收到 {sizes!r}")
        self.sizes = [int(s) for s in sizes]
        self.seed = int(seed)
        rng = random.Random(seed)
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []
        for i in range(len(self.sizes) - 1):
            fan_in = self.sizes[i]
            fan_out = self.sizes[i + 1]
            # He 初始化（适配 ReLU）
            scale = math.sqrt(2.0 / max(1, fan_in))
            w = [
                [rng.gauss(0.0, scale) for _ in range(fan_in)]
                for _ in range(fan_out)
            ]
            self.weights.append(w)
            self.biases.append(_zeros(fan_out))
        # 标准化口径（训练时写入；推理必须使用同一口径）
        self.x_mean: list[float] = _zeros(self.sizes[0])
        self.x_std: list[float] = [1.0] * self.sizes[0]
        self.n_features = self.sizes[0]
        self.n_outputs = self.sizes[-1]
        self.fitted = False

    # ---------------- 前向 ---------------- #
    def _forward(
        self, row: list[float],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """单样本前向，返回 (各层激活值, 各层加权和)。"""
        acts: list[list[float]] = [row]
        zs: list[list[float]] = []
        cur = row
        last = len(self.weights) - 1
        for li, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = [sum(w[o][i] * cur[i] for i in range(len(cur))) + b[o]
                 for o in range(len(w))]
            zs.append(z)
            if li == last:
                a = list(z)          # 输出层线性
            else:
                a = [_relu(v) for v in z]
            acts.append(a)
            cur = a
        return acts, zs

    def _predict_std(self, row: list[float]) -> list[float]:
        acts, _ = self._forward(row)
        return acts[-1]

    # ---------------- 训练 ---------------- #
    def fit(
        self,
        x: list[list[float]],
        y: list[list[float]],
        epochs: int = 300,
        lr: float = 0.01,
        batch_size: int = 32,
        l2: float = 1e-4,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """训练模型（Adam + MSE，确定性洗牌）。

        Args:
            x: 输入，形状 ``[n_samples][n_features]``。
            y: 目标，形状 ``[n_samples][n_outputs]``；单输出可传 ``[[v], ...]``。
            epochs: 训练轮数。
            lr: 学习率。
            batch_size: 小批量大小。
            l2: L2 正则系数（作用在权重上）。
            verbose: 是否打印训练日志。

        Returns:
            ``{"final_loss": float, "epochs": int, "n_samples": int}``。
        """
        if not x or not y or len(x) != len(y):
            raise ValueError("x/y 不可为空且样本数必须一致")
        n_features = len(x[0])
        if n_features != self.sizes[0]:
            raise ValueError(
                f"输入维度 {n_features} 与网络输入层 {self.sizes[0]} 不符"
            )
        n_out = len(y[0])
        if n_out != self.sizes[-1]:
            raise ValueError(
                f"输出维度 {n_out} 与网络输出层 {self.sizes[-1]} 不符"
            )

        # 标准化（口径持久化）
        xs, self.x_mean, self.x_std = standardize(x)

        # Adam 状态
        m_w = [[_zeros(len(w[0])) for w in self.weights] for _ in range(1)]
        v_w = [[_zeros(len(w[0])) for w in self.weights] for _ in range(1)]
        m_w = [[_zeros(len(row)) for row in layer] for layer in self.weights]
        v_w = [[_zeros(len(row)) for row in layer] for layer in self.weights]
        m_b = [_zeros(len(b)) for b in self.biases]
        v_b = [_zeros(len(b)) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        idx = list(range(len(xs)))
        rng = random.Random(self.seed)
        loss = 0.0

        for epoch in range(1, epochs + 1):
            rng.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                batch = idx[start:start + batch_size]
                # 梯度累加
                g_w = [[_zeros(len(row)) for row in layer] for layer in self.weights]
                g_b = [_zeros(len(b)) for b in self.biases]
                loss = 0.0
                for si in batch:
                    acts, zs = self._forward(xs[si])
                    target = y[si]
                    # 输出层误差（线性输出 + MSE → dL/dz = (pred - target)/N）
                    delta = [
                        (acts[-1][o] - target[o]) / len(batch)
                        for o in range(len(target))
                    ]
                    loss += sum(
                        (acts[-1][o] - target[o]) ** 2 for o in range(len(target))
                    ) / len(batch)
                    for li in range(len(self.weights) - 1, -1, -1):
                        a_prev = acts[li]
                        for o in range(len(self.weights[li])):
                            d = delta[o]
                            if d == 0.0:
                                continue
                            g_row = g_w[li][o]
                            for i in range(len(a_prev)):
                                g_row[i] += d * a_prev[i]
                            g_b[li][o] += d
                        if li > 0:
                            prev_delta = _zeros(len(a_prev))
                            for o in range(len(self.weights[li])):
                                d = delta[o]
                                if d == 0.0:
                                    continue
                                w_row = self.weights[li][o]
                                for i in range(len(a_prev)):
                                    prev_delta[i] += d * w_row[i]
                            delta = [
                                prev_delta[i] * _relu_grad(zs[li - 1][i])
                                for i in range(len(prev_delta))
                            ]
                # 参数更新
                step += 1
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                for li in range(len(self.weights)):
                    for o in range(len(self.weights[li])):
                        for i in range(len(self.weights[li][o])):
                            g = g_w[li][o][i] + l2 * self.weights[li][o][i]
                            m_w[li][o][i] = beta1 * m_w[li][o][i] + (1 - beta1) * g
                            v_w[li][o][i] = beta2 * v_w[li][o][i] + (1 - beta2) * g * g
                            mh = m_w[li][o][i] / bc1
                            vh = v_w[li][o][i] / bc2
                            self.weights[li][o][i] -= lr * mh / (math.sqrt(vh) + eps)
                        g = g_b[li][o]
                        m_b[li][o] = beta1 * m_b[li][o] + (1 - beta1) * g
                        v_b[li][o] = beta2 * v_b[li][o] + (1 - beta2) * g * g
                        mh = m_b[li][o] / bc1
                        vh = v_b[li][o] / bc2
                        self.biases[li][o] -= lr * mh / (math.sqrt(vh) + eps)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"  epoch {epoch:4d}  loss={loss:.6f}")
        self.fitted = True
        return {"final_loss": round(loss, 8), "epochs": epochs, "n_samples": len(x)}

    # ---------------- 推理 ---------------- #
    def predict(self, x: list[list[float]]) -> list[list[float]]:
        """批量推理（内部按训练时的标准化口径换算）。"""
        if not self.fitted:
            raise RuntimeError("模型尚未训练（fit 未调用）")
        out: list[list[float]] = []
        for row in x:
            if len(row) != len(self.x_mean):
                raise ValueError(
                    f"推理输入维度 {len(row)} 与训练维度 {len(self.x_mean)} 不符"
                )
            std_row = [
                (row[j] - self.x_mean[j]) / self.x_std[j]
                for j in range(len(row))
            ]
            out.append(self._predict_std(std_row))
        return out

    def predict_one(self, row: list[float]) -> list[float]:
        """单样本推理。"""
        return self.predict([row])[0]

    # ---------------- 持久化 ---------------- #
    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 存储的字典。"""
        return {
            "schema": "f1opt-pure-mlp/1",
            "sizes": self.sizes,
            "seed": self.seed,
            "weights": self.weights,
            "biases": self.biases,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "fitted": self.fitted,
        }

    def save(self, path: str | Path) -> Path:
        """保存权重到 JSON 文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8",
        )
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLP:
        """从字典还原模型。"""
        model = cls(list(data["sizes"]), seed=int(data.get("seed", 42)))
        model.weights = data["weights"]
        model.biases = data["biases"]
        model.x_mean = data["x_mean"]
        model.x_std = data["x_std"]
        model.fitted = bool(data.get("fitted", True))
        return model

    @classmethod
    def load(cls, path: str | Path) -> MLP:
        """从 JSON 文件加载模型。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
