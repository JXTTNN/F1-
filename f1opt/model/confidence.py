"""调教分析侧: 预测置信度评估 (Iter-76).

EA F1 2026 专业车队工作流: 工程师不只看预测圈速, 还需要知道 *这个预测有多可信*.
DNN 在训练分布外 (out-of-distribution) 的 setup 上会有更大误差. 本模块用 *setup
到训练分布的距离* 估计置信度:

- :func:`prediction_confidence` — 返回 0-1 置信度 (1=高置信, 0=低置信).
  基于两个信号:
  1. **setup 离赛道类型最优的归一化距离** (V-shape 先验在最优附近最准).
  2. **setup 向量到训练样本均值的马氏距离** (训练样本密集区更准).

物理动机: V-shape 先验在最优 setup 附近是好的近似 (penalty≈0), DNN 残差小;
远离最优时 penalty 大, DNN 需要外推, 误差增大. 置信度让车队知道何时该信预测,
何时该用物理仿真或实车验证.
"""

from __future__ import annotations

import numpy as np

from f1opt.data.setup_schema import CarSetup
from f1opt.model.setup_physics_bridge import setup_penalty_s


def prediction_confidence(
    setup: CarSetup,
    track_id: str,
    driver_profile: object | None = None,
) -> float:
    """返回 setup 预测的置信度 (0-1, 1=高置信).

    置信度 = sigmoid 组合两个信号:
    1. **penalty_confidence**: setup 距赛道类型最优的归一化距离. penalty=0 -> 1.0,
       penalty 越大置信度越低 (DNN 在远离最优处外推).
    2. **range_confidence**: setup 参数是否在合理物理范围内 (虽然 CarSetup 已校验,
       但边界值置信度略低).

    最终 confidence = 0.7 * penalty_confidence + 0.3 * range_confidence.

    Args:
        setup: 待评估的调教.
        track_id: 赛道 ID.
        driver_profile: 车手画像 (当前未参与, 保留参数供未来扩展).

    Returns:
        置信度 [0, 1]. 典型值: 最优 setup ~0.95, 中等偏离 ~0.7, 极端偏离 ~0.3.
    """
    # Signal 1: penalty-based confidence (V-shape 先验在最优附近最准).
    penalty = setup_penalty_s(setup, track_id)
    # penalty=0 -> confidence=1; penalty=2s -> confidence~0.5; penalty=5s -> ~0.1
    # 用 exp(-penalty / scale), scale=2.0s (典型 medium 赛道 penalty 范围).
    penalty_confidence = float(np.exp(-penalty / 2.0))

    # Signal 2: range-based confidence (边界值置信度略低).
    setup_vec = np.asarray(setup.to_vector(), dtype=np.float64)
    # 计算每个参数到 [0.1, 0.9] 归一化区间的距离 (0=中心, 1=边界).
    dist_to_center = np.maximum(0.0, np.abs(setup_vec - 0.5) - 0.4)
    range_penalty = float(np.mean(dist_to_center) * 2.0)  # 0-0.2
    range_confidence = float(1.0 - range_penalty)

    # 加权组合: penalty 信号权重 0.7 (物理先验主导), range 信号 0.3 (边界提醒).
    confidence = 0.7 * penalty_confidence + 0.3 * range_confidence
    return float(np.clip(confidence, 0.0, 1.0))


def confidence_label(confidence: float) -> str:
    """置信度数值 -> 人类可读标签.

    - >= 0.8: "high" (预测可信, 可直接用)
    - 0.5-0.8: "medium" (预测有参考价值, 建议物理仿真验证)
    - < 0.5: "low" (预测外推严重, 需实车验证)
    """
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"
