"""在线残差修正层 (Iter-70 反馈闭环: 遥测反哺 surrogate).

EA F1 2026 专业车队工作流: 仿真模型 (DNN surrogate) 给出先验预测, 练习赛/
排位赛的真实遥测观测 (setup, track, driver -> 实测圈速) 局部修正先验. 本模块
实现这个 *反馈闭环* 而不重训 DNN (在线重训太贵且不稳定):

- :class:`ObservationBuffer` — 存储近期真实观测 (setup 向量 + track + driver +
  实测圈速 + DNN 预测圈速 + 残差).
- :func:`add_observation` — 记录一条观测 (setup, track, driver, observed_lap).
- :func:`corrected_lap_time` — DNN 预测 + 核加权残差修正 (Gaussian kernel on
  setup 距离, track 精确匹配, driver 软匹配).

物理动机: DNN 在训练分布外 (out-of-distribution) 的 setup 上会有系统偏差
(先验近似误差 + 训练样本稀疏区域). 真实观测在 *附近* setup 上的残差能指示
该区域的偏差方向, 用核加权平均修正新预测. 这是 *局部加权回归* (LOESS) 的
思想, 在 DNN 先验之上做局部校正.

带宽选择: setup 距离带宽 σ_setup=0.10 (归一化空间, ≈10 档) — 太小 (<0.05)
会过拟合单条观测, 太大 (>0.2) 退化为全局常数修正. driver 带宽 σ_driver=0.15
(8 维 [0,1] 空间). track_id 必须精确匹配 (不同赛道物理完全不同).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from f1opt.data.setup_schema import CarSetup
from f1opt.model.surrogate import predict_lap_time

# 在线修正的默认带宽 (归一化 setup 空间).
_DEFAULT_SIGMA_SETUP = 0.10
_DEFAULT_SIGMA_DRIVER = 0.15
# 观测缓冲默认容量 (近期 N 条观测; FIFO 淘汰过旧观测, 因为 setup/track 漂移).
_DEFAULT_BUFFER_CAPACITY = 200
# 最小有效权重阈值 (权重总和低于此值时不修正, 退回纯 DNN 预测).
_MIN_TOTAL_WEIGHT = 0.05
# Iter-80: 异常观测检测阈值. 残差绝对值超过此值的观测被视为 outlier (被交通/
# 黄旗/失误污染的圈速), 不参与核加权修正. 来源: EA F1 2026 专业车队经验 —
# DNN 预测误差通常 < 1.5s (held-out lap MAE 0.29s + 3σ), 残差 > 5.0s 几乎必然
# 是观测污染 (交通损失 ~2-4s, 黄旗减速 ~3-5s, 大失误 ~2-5s).
# Iter-164.21: 阈值从 3.0 提高到 5.0 — 3.0s 过严, 真实 F1 圈速差异 (燃油/胎温/
# 车手节奏) 可达 4-5s, 3.0s 会误滤合法观测导致 corrected_lap_time 不收敛到观测.
_OUTLIER_RESIDUAL_THRESHOLD_S = 5.0


@dataclass
class _Observation:
    """单条真实观测: (setup_vec, track_id, driver_vec, observed_lap, predicted_lap).

    quality (Iter-79): 观测质量权重 [0,1], 1=高置信 (排位赛飞驰圈), 0.5=中 (练习赛
    推进圈), 0.3=低 (练习赛安装圈). 影响核加权修正的权重 (quality * distance_weight).
    """

    setup_vec: np.ndarray  # 归一化 [0,1]^21
    track_id: str
    driver_vec: np.ndarray  # 8 维
    observed_lap: float  # 实测圈速 (秒)
    predicted_lap: float  # DNN 预测圈速 (秒, 加入缓冲时计算)
    quality: float = 1.0  # 观测质量 [0,1] (Iter-79)
    residual: float = 0.0  # observed - predicted (正 = DNN 低估)

    def __post_init__(self) -> None:
        self.residual = self.observed_lap - self.predicted_lap


@dataclass
class ObservationBuffer:
    """真实遥测观测缓冲 (FIFO, 容量受限).

    存储近期 (setup, track, driver, observed_lap) 观测, 供
    :func:`corrected_lap_time` 做核加权残差修正. 容量满后 FIFO 淘汰最旧观测
    (因为 setup/track/轮胎会随 session 演化, 过旧观测不再代表当前条件).
    """

    capacity: int = _DEFAULT_BUFFER_CAPACITY
    _observations: deque[_Observation] = field(
        default_factory=lambda: deque(maxlen=_DEFAULT_BUFFER_CAPACITY)
    )

    def __post_init__(self) -> None:
        # 重建 deque 以应用自定义 capacity.
        self._observations = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._observations)

    def add(self, obs: _Observation) -> None:
        """追加一条观测 (容量满时 FIFO 淘汰最旧)."""
        self._observations.append(obs)

    def observations_for_track(self, track_id: str) -> list[_Observation]:
        """返回指定赛道的所有观测 (track 精确匹配)."""
        return [o for o in self._observations if o.track_id == track_id]


def add_observation(
    buffer: ObservationBuffer,
    setup: CarSetup,
    track_id: str,
    driver_profile: object | None,
    observed_lap: float,
    quality: float = 1.0,
) -> None:
    """记录一条真实观测到缓冲.

    Args:
        buffer: 观测缓冲.
        setup: 真实使用的调教.
        track_id: 赛道 ID.
        driver_profile: 车手画像 (任何 surrogate 接受的形态).
        observed_lap: 实测圈速 (秒, 来自遥测 lap_time 字段).
        quality: 观测质量 [0,1] (Iter-79). 1.0=排位赛飞驰圈 (高置信), 0.5=练习赛
            推进圈, 0.3=练习赛安装圈 (低置信). 影响核加权修正权重.

    同时计算 DNN 预测圈速 (加入缓冲时), 避免修正时重复 forward.
    """
    from f1opt.model.surrogate import driver_vector

    predicted = float(predict_lap_time(setup, track_id, driver_profile))
    obs = _Observation(
        setup_vec=np.asarray(setup.to_vector(), dtype=np.float64),
        track_id=track_id,
        driver_vec=np.asarray(driver_vector(driver_profile), dtype=np.float64),
        observed_lap=float(observed_lap),
        predicted_lap=predicted,
        quality=float(max(0.0, min(1.0, quality))),
    )
    buffer.add(obs)


def corrected_lap_time(
    setup: CarSetup,
    track_id: str,
    driver_profile: object | None,
    buffer: ObservationBuffer,
    sigma_setup: float = _DEFAULT_SIGMA_SETUP,
    sigma_driver: float = _DEFAULT_SIGMA_DRIVER,
) -> float:
    """DNN 预测 + 核加权残差修正 (反馈闭环核心).

    修正公式::

        corrected = dnn_predict + Σ w_i * residual_i / Σ w_i

    其中 ``w_i = exp(-0.5 * (d_setup_i²/σ_setup² + d_driver_i²/σ_driver²))``,
    ``d_setup_i`` = 归一化 setup 欧氏距离, ``d_driver_i`` = driver 向量欧氏距离.
    仅对 *同赛道* 观测求和 (track 精确匹配). 权重总和 < ``_MIN_TOTAL_WEIGHT``
    时不修正 (退回纯 DNN 预测, 避免远距离观测过拟合).

    Args:
        setup: 待预测的调教.
        track_id: 赛道 ID.
        driver_profile: 车手画像.
        buffer: 观测缓冲 (含历史真实观测).
        sigma_setup: setup 距离带宽 (归一化空间).
        sigma_driver: driver 距离带宽.

    Returns:
        修正后圈速 (秒). 无有效观测时 == DNN 预测.
    """
    from f1opt.model.surrogate import driver_vector

    dnn_pred = float(predict_lap_time(setup, track_id, driver_profile))
    obs_list = buffer.observations_for_track(track_id)
    if not obs_list:
        return dnn_pred

    setup_vec = np.asarray(setup.to_vector(), dtype=np.float64)
    drv_vec = np.asarray(driver_vector(driver_profile), dtype=np.float64)

    total_weight = 0.0
    weighted_residual = 0.0
    inv_2sig_setup_sq = 0.5 / (sigma_setup * sigma_setup)
    inv_2sig_driver_sq = 0.5 / (sigma_driver * sigma_driver)

    for obs in obs_list:
        # Iter-80/164.21: 异常观测过滤 — 残差绝对值 > 5.0s 的观测视为 outlier
        # (交通/黄旗/失误污染), 不参与修正. 防止一个污染圈把修正拉偏.
        if abs(obs.residual) > _OUTLIER_RESIDUAL_THRESHOLD_S:
            continue
        d_setup_sq = float(np.sum((obs.setup_vec - setup_vec) ** 2))
        d_driver_sq = float(np.sum((obs.driver_vec - drv_vec) ** 2))
        # Iter-79: 距离权重 × 观测质量权重 (高质量观测影响更大).
        dist_weight = np.exp(
            -inv_2sig_setup_sq * d_setup_sq - inv_2sig_driver_sq * d_driver_sq
        )
        w = dist_weight * obs.quality
        total_weight += w
        weighted_residual += w * obs.residual

    if total_weight < _MIN_TOTAL_WEIGHT:
        return dnn_pred

    correction = weighted_residual / total_weight
    return float(dnn_pred + correction)
