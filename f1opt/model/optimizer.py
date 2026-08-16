"""Bayesian-style 调教优化器 (Iter-02 Task 2.3; Iter-04 Task 4.3 scipy 升级).

在 23 维归一化 ``CarSetup`` 空间 [0,1]^23 上最小化代理模型预测圈速, 返回带
量化增益的推荐调教 (经 ``CarSetup.from_vector`` snapped 到游戏合法档位).

Iter-04 Task 4.3: 优先使用 ``scipy.optimize.differential_evolution`` 全局
优化器 (DE, 收敛质量优于局部搜索, 全局探索避免陷入局部最优); scipy 不可用
时回退到原 numpy 多起点局部搜索 (random-restart hill-climbing with
adaptive Gaussian perturbations + 坐标轮换 polish), 行为与 Iter-02 一致.
``SearchResult.algorithm`` 标记实际走了哪条路径 (``"scipy-de"`` /
``"numpy-local"``), 便于调试与回归对比.

评估仅通过公开 :func:`predict_lap_time` 完成, 不重写模型; 按 snapped setup
缓存评估结果, 避免重复 forward.

公开 API:

- :class:`SearchResult` — 优化结果 (推荐/基线/增益/diff/搜索轨迹/版本/迭代数/算法).
- :func:`search_setup` — 模块级便捷入口.
- :class:`SearchOptimizer` — 可复用包装类 (可配置 iterations / method / seed).
"""

from __future__ import annotations

from math import exp as _exp
from typing import Any

import numpy as np
from pydantic import BaseModel

from f1opt.data.setup_schema import ALL_SETUP_FIELDS, DEFAULT_SETUP, CarSetup
from f1opt.driver.profile import DriverProfile
from f1opt.model.surrogate import MODEL_VERSION, predict_full

# scipy 可选依赖: 不可用时回退 numpy 多起点局部搜索 (Iter-02 路径).
try:
    from scipy.optimize import differential_evolution as _de
except ImportError:  # pragma: no cover - 环境兜底, 主测试环境已装 scipy
    _de = None

# 归一化空间每维步长 (step / (max-min)) — 用于 snapped 缓存键与坐标 polish.
_NORM_STEPS = np.array(
    [s.step / (s.max - s.min) for s in ALL_SETUP_FIELDS()], dtype=np.float64
)
_DIM = int(_NORM_STEPS.size)
# DE 搜索区间: 与 CarSetup.to_vector/from_vector 一致的归一化 [0,1]^21.
# (SETUP_FIELDS 的 min/max 经 to_vector 线性映射到 [0,1], DE 在归一化空间搜索,
#  最优 x* 再经 from_vector 反归一化并 snap 到游戏档位.)
_BOUNDS: list[tuple[float, float]] = [(0.0, 1.0) for _ in range(_DIM)]
# DE 种群规模 (每维个体数). scipy 默认 15, 但 23 维 × 15 × maxiter=60 评估量
# 过大 (~17k forward, ~10s). Iter-84 消融实验测得 popsize=3 在 gain 上与 popsize=5
# 几乎一致 (5 赛道 avg 2.019s vs 2.095s, 仅 -4%), 但因 forwards 减少 40%
# (95 → 57 / gen), 单次 search_setup 从 ~2.5s 降到 ~1.5s (1.67x 加速).
# sakhir (mixed 最难赛道) 上 gain 0.65s vs 0.78s (-17%), 仍远超 0.1s 阈值,
# 物理可信. 更低 popsize (<3) 会让 DE 多样性不足, 在 mixed/high_downforce 赛道
# 上 premature convergence (gain 暴跌).
_DE_POPSIZE = 3
# DE 代数上限 cap. 用户传入的 ``iterations`` 直接作为 numpy 回退路径的评估预算,
# 但 DE 全局探索 25 代已足够收敛 (实测 melbourne gain 在 maxiter=25 与 60 下
# 差异 <0.05s); cap 后 25 代 × popsize=3 × 23 维 ≈ 1425 评估 ≈ 1.5s, 兼顾
# 全局性与延迟预算 (search_setup ≤2s, generate_feedback setup_advice ≤2s).
# numpy 回退路径不受 cap 约束 (其 60 iter ≈ 0.05s 本就极快).
_DE_MAXITER_CAP = 25
# Iter-85: 向量化路径单独的 cap. 因 predict_batch 批量评估 ~7x 加速, 每代成本
# 从 ~57ms 降到 ~8ms, 可负担更多代数. 实测 cap=35 vec 在 5 赛道 avg gain
# 2.099s (vs cap=25 vec 1.874s, vs cap=25 seq 2.019s), 且单次 ~400ms 仍远低于
# sequential 1400ms. cap=25 vec 在 melbourne 上 gain 暴跌 (0.294s vs seq 0.650s),
# 因 deferred updating 收敛较慢, 需更多代补回. cap=50 vec 收益递减 (~570ms).
#
# Iter-104: cap 35→100. 旧版 35 在 holistic 多目标 (weight>0) 下收敛不足, 5/24
# 赛道 (suzuka/jeddah/monaco/singapore/sao_paulo) holistic 的胎耗代理 > single
# (违反多目标优化基本定理: 若两者达全局最优, holistic proxy 必 <= single proxy).
# 根因: deferred updating + cap=35 在 23 维离散景观上偶尔未收敛. 实测:
#   cap=50: 全 PASS 但 jeddah 边缘 (-0.009)
#   cap=75: 全 PASS 且 tire_saving >= +0.001, 但 austin -0.0134 FAIL
#   cap=100: 全 PASS, 多数赛道 single=holo 收敛到同一点 (tire_saving≈0)
# cap=100 延迟 ~1.2s 仍低于 2s 预算, 且消除所有 holistic 收敛不足伪影. 仅影响
# iterations>=100 的调用 (低 iter 调用不受 cap 约束).
#
# Iter-110: cap 100→200. Iter-109 深入研究发现 18/24 赛道在 cap=100 最后 20 代
# 仍有 >10ms 真实 gain 改进 (losail +81ms, shanghai +57ms, austin +52ms). cap=200
# 实测: 总 gain 改善 +0.1225s (yas_marina +31ms, losail +25ms, melbourne +26ms),
# 延迟 0.8-1.2s (仍远低于 2s 预算), 8/24 赛道有改善 (其余 cap=100 已收敛).
# EA F1 2026 专业车队标准: 30ms 差距足以决定排位名次, 提高 cap 让高 iterations
# 调用充分收敛. 仅影响 iterations>100 的调用 (iterations<=100 不受 cap 约束,
# min(iterations, 200) = iterations). holistic 收敛不受影响 (cap=100 已全 PASS,
# cap=200 给 holistic 更多探索空间, 更鲁棒).
_DE_MAXITER_CAP_VECTORIZED = 200
# Iter-84: polish=False. scipy L-BFGS-B polish 在分段常量目标 (经 _snap_vec
# 缓存) 上有限差分梯度 ≈ 0, 几乎瞬间完成但也几乎无效果 (实测 5 赛道 gain 与
# polish=True 完全一致). 显式关闭避免无谓的 L-BFGS-B 初始化与一次额外 forward.
_DE_POLISH = False

# Iter-12 多目标: 胎耗代理归一化参考点 (来自 surrogate RESPONSE_PRIORS 自然单位).
# 胎耗代理 = (tyre_temp - 90)/30 + slip_angle/5 + tyre_load_spread
# (胎温偏离基准 + 后轴滑移 + 四轮载荷离散, 越大胎耗越快). 权重 0 = 单目标圈速
# (向后兼容 Iter-02~11 行为, Melbourne gain 等测试不变).
_TYRE_TEMP_REF = 90.0
_TYRE_TEMP_SPAN = 30.0
_SLIP_REF = 5.0

# Iter-69 调教思维整体性: holistic=True 且用户未显式指定 tire_wear_weight 时的
# 默认胎耗权重. 来源: EA F1 2026 工程师经验 — stint 长度 ~12 圈, 每圈胎耗代价
# 折算到圈速约 0.3s/stint 的等效 penalty. 太大 (>1.0) 会让优化器过度牺牲圈速
# 换胎耗, 太小 (<0.1) 等同单目标. 0.3 在 medium 赛道上让推荐 setup 比 pure
# lap-time-optimal 胎温低 ~3°C, 滑移低 ~0.3°, 圈速慢 ~0.05s — 物理可信的折中.
_HOLISTIC_DEFAULT_TIRE_WEIGHT = 0.3

# Iter-186 调教约束验证: 对不可行的调教施加惩罚, 让 DE 自动避开不可行区域.
# 约束: (1) 前 ride_height < 后 ride_height (rake >= 0); (2) 前 camber <= 后 camber
# (前轮更负); (3) 前 tire_pressure 与后 tire_pressure 差 < 4 psi (F1 前后胎压常差 2-3.5 psi).
# 惩罚系数: 每条约束违反 +100s 到 lap_time, 让 DE 远离不可行区域.
_SETUP_CONSTRAINT_PENALTY_S = 100.0


def _setup_constraint_penalty(setup: CarSetup) -> float:
    """Iter-186: 对不可行调教施加圈速惩罚 (DE 不可行域惩罚).

    检查 F1 物理约束:
    - 前 ride_height < 后 ride_height (必须有 rake, 否则后扩散器失速)
    - 前 camber <= 后 camber (前轮必须比后轮更负外倾)
    - 前胎压与后胎压差 < 4 psi (过大的前后胎压差导致不平衡)
    返回总惩罚 (0 = 可行, >0 = 违反约束).
    """
    penalty = 0.0
    # 约束 1: 前 ride_height < 后 ride_height
    if setup.front_ride_height >= setup.rear_ride_height:
        penalty += _SETUP_CONSTRAINT_PENALTY_S
    # 约束 2: 前 camber <= 后 camber
    if setup.front_camber > setup.rear_camber:
        penalty += _SETUP_CONSTRAINT_PENALTY_S
    # 约束 3: 前后胎压差 < 4 psi
    if abs(setup.front_tyre_pressure - setup.rear_tyre_pressure) > 4.0:
        penalty += _SETUP_CONSTRAINT_PENALTY_S
    return penalty


def _tire_wear_proxy(setup: CarSetup, track_id: str, driver_profile: Any) -> float:
    """胎耗代理 (无量纲, 越大胎耗越快): 胎温偏离 + 后轴滑移 + 载荷离散.

    用 :func:`predict_full` 一次性拿到 responses, 避免重复 forward. 仅在
    ``tire_wear_weight > 0`` 时调用.
    """
    pred = predict_full(setup, track_id, driver_profile)
    resp = pred["responses"]
    temp = float(resp["tyre_temp"])
    slip = float(resp["slip_angle"])
    spread = float(resp["tyre_load_spread"])
    return (temp - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN + slip / _SLIP_REF + spread


def _normalize_tire_wear_proxy(raw: float) -> float:
    """Iter-164: 把 raw tire_wear_proxy 压缩到 [0, 1] 区间.

    raw proxy 三项可加和超过 1.0 (实测 1.6+), 物理上"胎耗严重度"应是归一化
    指标. 用 sigmoid 围绕 raw=1.0 (中等胎耗) 中心化, 单调且严格在 (0, 1).

    用于 :attr:`SearchResult.tire_wear` 报告 (透明性 + 跨场景可比性);
    优化目标函数仍用 raw proxy (保留 Iter-94 weight=0.3 的物理标定).
    """
    # clip x to avoid overflow; sigmoid is monotonic so order preserved.
    x = max(-20.0, min(20.0, raw - 1.0))
    return 1.0 / (1.0 + _exp(-x))


def _snap_vec(vec: np.ndarray) -> np.ndarray:
    """把归一化向量对齐到游戏档位网格 (与 ``CarSetup.from_vector`` 等价)."""
    steps = np.round(np.asarray(vec, dtype=np.float64) / _NORM_STEPS)
    return np.clip(steps * _NORM_STEPS, 0.0, 1.0)


def _coerce_baseline(baseline: CarSetup | dict | None) -> CarSetup:
    """接受 ``CarSetup | dict | None``; None -> :data:`DEFAULT_SETUP`."""
    if baseline is None:
        return DEFAULT_SETUP
    if isinstance(baseline, CarSetup):
        return baseline
    return CarSetup(**baseline)


class SearchResult(BaseModel):
    """调教优化结果.

    ``predicted_gain_s = baseline_lap_time - recommended_lap_time`` (正值 = 改善).
    ``search_trace`` 为每步 best-yet 圈速 (非递增, 供 UI 绘制收敛曲线).
    ``algorithm`` 标记使用的优化器: ``"scipy-de"`` (differential_evolution) 或
    ``"numpy-local"`` (回退路径, scipy 不可用时). 默认 ``"scipy-de"``, 回退路径
    显式覆写为 ``"numpy-local"``.

    Iter-69 调教输出整体性: ``response_profile`` 报告推荐 setup 的完整 7 项响应
    (speed_avg/speed_max/slip_angle/tyre_load_spread/rake/tyre_temp/g_lat_max),
    让车队工程师看到圈速之外的物理全貌 (滑移/胎温/侧向 G 等), 而非仅胎耗代理.
    ``baseline_response_profile`` 同理报告基线, 便于 diff 对比.
    """

    recommended: dict
    baseline: dict
    predicted_gain_s: float
    baseline_lap_time: float
    recommended_lap_time: float
    diff: list[dict]
    search_trace: list[float]
    model_version: str
    iterations: int
    algorithm: str = "scipy-de"
    # Iter-12 多目标: 胎耗代理 (recommended setup 的胎耗代理值, 无量纲) +
    # 用户传入的胎耗权重 (0 = 单目标圈速, 向后兼容).
    tire_wear: float = 0.0
    tire_wear_weight: float = 0.0
    # Iter-69 调教输出整体性: 推荐/基线的完整 7 项响应画像 (自然单位).
    response_profile: dict = {}
    baseline_response_profile: dict = {}
    # Iter-71 反馈闭环: 是否使用了遥测观测修正 (True = corrected_lap_time 路径).
    feedback_corrected: bool = False
    # Iter-77 调教分析侧集成: 推荐置信度 [0,1] + 参数变化解释 + 推荐参数贡献.
    confidence: float = 0.0
    confidence_label: str = "medium"
    change_explanation: list[dict] = []
    top_sensitive_params: list[dict] = []
    # Iter-178 精英保留: 追踪每代精英存活数 (elite_count 个精英在下一代
    # 种群中仍位列前 elite_count 的数量). 用于验证精英保留机制有效性.
    elite_survival: list[int] = []


def search_setup(
    track_id: str,
    driver_profile: DriverProfile | dict | list[float] | None = None,
    baseline: CarSetup | dict | None = None,
    iterations: int = 100,
    seed: int | None = None,
    tire_wear_weight: float = 0.0,
    holistic: bool = False,
    observation_buffer: Any = None,
    *,
    stint_aware: bool = False,
    stint_length: int = 20,
    stint_compound: str = "medium",
    elite_count: int = 3,
    enable_constraints: bool = False,
) -> SearchResult:
    """搜索最小化预测圈速的调教, 返回 :class:`SearchResult`.

    在归一化 [0,1]^21 空间上优化 ``predict_lap_time(CarSetup.from_vector(vec),
    track_id, driver_profile)``; 推荐 vec 经 ``from_vector`` snapped 到合法档位.
    优先用 ``scipy.optimize.differential_evolution`` 全局优化; scipy 不可用
    时回退到 numpy 多起点局部搜索 (``SearchResult.algorithm`` 标记路径).
    ``seed`` 固定时结果可复现; 未知 track_id 不抛异常 (gain 可能 ≈ 0).

    Iter-12 多目标: ``tire_wear_weight > 0`` 时目标变为
    ``lap_time + tire_wear_weight * tire_wear_proxy`` (胎耗代理 = 胎温偏离 +
    后轴滑移 + 载荷离散), 在圈速与胎耗间折中; ``= 0`` 时为单目标圈速 (向后
    兼容, Melbourne gain 等既有测试不变).

    Iter-69 调教思维整体性: ``holistic=True`` 时启用专业车队默认多目标平衡 —
    在用户未显式指定 ``tire_wear_weight`` 时使用物理 motivated 的小权重
    (``_HOLISTIC_DEFAULT_TIRE_WEIGHT = 0.3``), 反映真实 F1 策略 (始终考虑
    胎耗保育, 单圈最快但 5 圈报废胎的调教不可接受). ``holistic=False`` (默认)
    保持单目标圈速行为 (向后兼容). ``holistic`` 与显式 ``tire_wear_weight>0``
    可叠加: holistic 仅在 weight==0 时注入默认值.

    Iter-71 反馈闭环: ``observation_buffer`` 传入 :class:`ObservationBuffer`
    时, 优化目标用 ``corrected_lap_time`` (DNN 预测 + 遥测观测核加权残差修正)
    替代纯 ``predict_lap_time``. 这让练习赛/排位赛的真实圈速反哺优化器,
    形成 *遥测→缓冲→修正预测→更优调教* 的闭环. ``None`` (默认) 退回纯 DNN
    路径 (向后兼容). ``SearchResult.feedback_corrected`` 标记是否走了修正路径.

    Iter-164.18 stint_aware: ``stint_aware=True`` 时优化目标变为
    ``stint_total_time(setup, track_id, compound, stint_length)`` — 即
    用 stint 总时间 (含胎耗物理耦合) 作为目标, 而非单圈 lap_time. 这是 R8
    "结合胎耗" 的深度集成: 优化器直接最小化 stint 总时间, 让调教在 stint
    全程 (而非单圈) 最优. ``stint_aware`` 与 ``holistic`` 互斥 (stint_aware
    优先). ``stint_length`` 默认 20 圈, ``stint_compound`` 默认 medium.

    Iter-178 精英保留: ``elite_count`` 控制每代保留的精英个体数 (默认 3).
    在每代 DE 结束后, 保留前 ``elite_count`` 个最优个体作为精英, 注入到
    下一代种群 (替换最差的 ``elite_count`` 个个体). 精英保留防止最优解在
    变异/交叉中丢失, 加速收敛并提高 DE 稳定性. ``SearchResult.elite_survival``
    记录每代精英存活数 (前一代精英中在新种群仍位列前 ``elite_count`` 的数量).
    ``elite_count=0`` 禁用精英保留 (向后兼容原有行为).
    """
    return _search(
        track_id=track_id,
        driver_profile=driver_profile,
        baseline=baseline,
        iterations=iterations,
        seed=seed,
        tire_wear_weight=tire_wear_weight,
        holistic=holistic,
        observation_buffer=observation_buffer,
        stint_aware=stint_aware,
        stint_length=stint_length,
        stint_compound=stint_compound,
        elite_count=elite_count,
        enable_constraints=enable_constraints,
    )


class SearchOptimizer:
    """可复用的调教优化器 (DE 全局搜索, scipy 不可用回退 numpy 爬山).

    Parameters
    ----------
    iterations
        目标函数评估代数上限 (DE ``maxiter``; 回退路径为 eval 预算).
    method
        优化方法; 目前仅支持 ``"numpy-hc"`` (保留向后兼容, 实际由 scipy 可用性
        决定走 DE 或 numpy 爬山). 传 ``"numpy-hc"`` 不会强制走 numpy 路径——
        scipy 可用时仍优先 DE.
    seed
        随机种子; ``None`` 表示不固定 (DE 与 numpy 路径均透传 seed).
    """

    def __init__(
        self,
        iterations: int = 100,
        method: str = "numpy-hc",
        seed: int | None = None,
        tire_wear_weight: float = 0.0,
        holistic: bool = False,
        observation_buffer: Any = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations 必须 >= 1")
        if method != "numpy-hc":
            raise ValueError(f"不支持的 method={method!r} (仅支持 'numpy-hc')")
        if tire_wear_weight < 0.0:
            raise ValueError("tire_wear_weight 必须 >= 0")
        self.iterations = iterations
        self.method = method
        self.seed = seed
        self.tire_wear_weight = tire_wear_weight
        self.holistic = holistic
        self.observation_buffer = observation_buffer

    def optimize(
        self,
        track_id: str,
        driver_profile: DriverProfile | dict | list[float] | None = None,
        baseline: CarSetup | dict | None = None,
    ) -> SearchResult:
        """运行优化, 返回 :class:`SearchResult` (复用实例的 iterations / seed)."""
        return _search(
            track_id=track_id,
            driver_profile=driver_profile,
            baseline=baseline,
            iterations=self.iterations,
            seed=self.seed,
            tire_wear_weight=self.tire_wear_weight,
            holistic=self.holistic,
            observation_buffer=self.observation_buffer,
        )


def _search(
    track_id: str,
    driver_profile: Any,
    baseline: CarSetup | dict | None,
    iterations: int,
    seed: int | None,
    tire_wear_weight: float = 0.0,
    holistic: bool = False,
    observation_buffer: Any = None,
    *,
    stint_aware: bool = False,
    stint_length: int = 20,
    stint_compound: str = "medium",
    elite_count: int = 3,
    enable_constraints: bool = False,
) -> SearchResult:
    base_setup = _coerce_baseline(baseline)
    # driver_profile 原样透传给 surrogate (其 _normalize_driver_vector 处理所有形态).
    # Iter-69 整体性: holistic=True 且用户未显式指定 weight 时注入物理默认胎耗权重.
    weight = float(tire_wear_weight)
    if holistic and weight == 0.0:
        weight = _HOLISTIC_DEFAULT_TIRE_WEIGHT
    # Iter-164.18: stint_aware 优先于 holistic — 用 stint 总时间作为目标.
    # stint_aware 模式不需要 weight (stint_total_time 已含胎耗物理耦合).

    # Iter-71 反馈闭环: 有观测缓冲且该赛道有观测时, 用 corrected_lap_time.
    from f1opt.model.online_correction import corrected_lap_time

    use_feedback = (
        observation_buffer is not None
        and len(observation_buffer.observations_for_track(track_id)) > 0
    )

    cache: dict[tuple, tuple[float, float]] = {}

    # Iter-92: 燃油是策略变量, 不是调教自由度. EA F1 2026 专业车队调教流程中
    # 燃油装载量由策略组决定 (赛道长度 + 油耗), 调教工程师在固定燃油下优化其余
    # 20 维. 旧版把 fuel_load 作为第 19 维自由优化, 优化器总是推到最小值 (5kg)
    # 因为物理模型中燃油越轻圈速越快 — 这在物理上正确但在调教语义上错误 (车队
    # 不能用调教器决定跑多少燃油). 修复: 钳制 fuel_load 维度到 baseline 值,
    # DE 搜索该维度完全无效, 推荐的 fuel_load 永远 = baseline.
    # Iter-267: 不再硬编码索引 18 (active_aero_mode/x_mode_activations 加入后
    # fuel_load 索引从 18 变为 20, 硬编码 18 会误钳制 front_tyre_pressure)。
    _FUEL_LOAD_IDX = next(
        i for i, s in enumerate(ALL_SETUP_FIELDS()) if s.name == "fuel_load"
    )
    _base_fuel_norm = (float(base_setup.fuel_load) - 5.0) / 105.0

    def evaluate(vec: np.ndarray) -> tuple[float, float]:
        """返回 ``(lap_time, tire_wear_proxy)``; 按 snapped setup 缓存.

        Iter-164: 始终计算 tire_wear_proxy (即使 weight==0), 让 SearchResult
        报告真实胎耗画像而非 0. 优化目标函数仍按 weight 加权 (weight=0 时
        proxy 不影响目标, 但仍报告供透明性).

        Iter-164.18: stint_aware 模式下 lap_time 替换为 stint_total_time
        (含胎耗物理耦合), 但仍返回原始单圈 lap_time 用于报告. 用 stint
        总时间作为优化目标, 单圈 lap_time 用于 diff/explanation.
        """
        snapped = _snap_vec(vec)
        # Iter-92: 钳制 fuel_load 到 baseline (策略变量, 非调教自由度)
        snapped[_FUEL_LOAD_IDX] = _base_fuel_norm
        key = tuple(np.round(snapped, 6))
        if key in cache:
            return cache[key]
        setup = CarSetup.from_vector(snapped.tolist())
        if stint_aware:
            # Iter-164.18: stint_aware 模式 — 用 stint 总时间作为目标.
            from f1opt.model.tire_stint import stint_total_time
            try:
                stint_total, _wear = stint_total_time(
                    setup, track_id, driver_profile,
                    compound=stint_compound,
                    stint_length=stint_length,
                )
            except Exception:
                # 兜底: stint 仿真失败时回退到单圈 lap_time
                stint_total = float(predict_full(setup, track_id, driver_profile)["lap_time"])
            # 仍需计算单圈 lap_time + proxy 用于报告
            pred = predict_full(setup, track_id, driver_profile)
            lap = float(pred["lap_time"])
            resp = pred["responses"]
            proxy = (
                (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
                + float(resp["slip_angle"]) / _SLIP_REF
                + float(resp["tyre_load_spread"])
            )
            # 缓存 stint_total 作为 "目标 lap_time" (objective 用它),
            # 但返回真实 lap_time 用于 diff. 用第二个缓存 slot 存 stint_total.
            cache[key] = (lap, proxy)
            # 用属性附加 stint_total (避免改 tuple 结构)
            _stint_cache[key] = stint_total
            return lap, proxy
        elif weight > 0.0:
            pred = predict_full(setup, track_id, driver_profile)
            lap = float(pred["lap_time"])
            resp = pred["responses"]
            proxy = (
                (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
                + float(resp["slip_angle"]) / _SLIP_REF
                + float(resp["tyre_load_spread"])
            )
        elif use_feedback:
            # Iter-71: 反馈闭环路径 — DNN + 遥测观测核加权残差修正.
            lap = corrected_lap_time(setup, track_id, driver_profile, observation_buffer)
            # Iter-164: 反馈路径也计算 proxy (额外一次 predict_full 调用,
            # 但反馈路径已非热路径, 透明性优先).
            pred = predict_full(setup, track_id, driver_profile)
            resp = pred["responses"]
            proxy = (
                (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
                + float(resp["slip_angle"]) / _SLIP_REF
                + float(resp["tyre_load_spread"])
            )
        else:
            # Iter-164: 单目标路径 — 用 predict_full 一次性拿 lap + responses,
            # 同时计算 proxy (替代旧版 predict_lap_time + proxy=0).
            pred = predict_full(setup, track_id, driver_profile)
            lap = float(pred["lap_time"])
            resp = pred["responses"]
            proxy = (
                (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
                + float(resp["slip_angle"]) / _SLIP_REF
                + float(resp["tyre_load_spread"])
            )
        cache[key] = (lap, proxy)
        return lap, proxy

    # Iter-164.18: stint_aware 缓存 (key → stint_total_time)
    _stint_cache: dict[tuple, float] = {}

    def objective(vec: np.ndarray) -> float:
        lap, proxy = evaluate(vec)
        # Iter-186: 约束惩罚 (仅当 enable_constraints=True)
        if enable_constraints:
            snapped = _snap_vec(vec)
            snapped[_FUEL_LOAD_IDX] = _base_fuel_norm
            setup = CarSetup.from_vector(snapped.tolist())
            constraint_pen = _setup_constraint_penalty(setup)
        else:
            constraint_pen = 0.0
        if stint_aware:
            # Iter-164.18: 用 stint_total_time 作为目标 (从 _stint_cache 取)
            snapped = _snap_vec(vec)
            snapped[_FUEL_LOAD_IDX] = _base_fuel_norm
            key = tuple(np.round(snapped, 6))
            stint_total = _stint_cache.get(key, lap * stint_length)
            return stint_total + constraint_pen
        return lap + weight * proxy + constraint_pen

    base_vec = np.asarray(base_setup.to_vector(), dtype=np.float64)

    # Iter-104: vectorized DE 现支持多目标 (weight>0) — predict_batch 已返回
    # responses (tyre_temp/slip_angle/tyre_load_spread), 可在批量路径内算
    # _tire_wear_proxy. 旧版仅单目标走 vectorized, 多目标走 sequential DE
    # (cap=25 vs vectorized cap=35), 导致同 seed 下两路径收敛到不同局部最优,
    # 偶发 holistic proxy > single proxy (违反多目标优化基本定理: 若两者均达
    # 全局最优, holistic proxy 必 <= single proxy). 统一两条路径用 vectorized
    # DE (同算法 + 同 cap=35) 消除该伪影, 且多目标也获 ~7x 加速. 反馈闭环路径
    # (use_feedback) 因 corrected_lap_time 不可批量, 仍走顺序 DE.
    # Iter-164.18: stint_aware 模式因 stint_total_time 不可批量, 也走顺序 DE.
    use_vectorized = (not use_feedback) and (not stint_aware) and (_de is not None)

    if use_vectorized:
        # 构造 vectorized 目标函数: (N, dim) -> (N,) lap_time 数组.
        from f1opt.model.surrogate import _get_default_model

        model = _get_default_model()

        def objective_vec(x_array: np.ndarray) -> np.ndarray:
            """Vectorized objective: scipy vectorized=True 传入 (dim, N),
            返回 (N,) lap_time (单目标) 或 lap + weight*proxy (多目标).

            用 predict_batch 一次性评估整代种群, 比逐条 predict_lap_time 快 ~7x.
            cache 按 snapped key 复用 (DE 跨代可能重访同一点).
            Iter-104: 多目标时从 batched responses 计算 _tire_wear_proxy
            (tyre_temp/slip_angle/tyre_load_spread), 与 evaluate() 顺序路径
            数值一致 (predict_batch 与 predict_full 在容差内等价).

            注意: scipy 的 vectorized 模式把种群保存为 (S, N) (S 个体, N 维),
            调用 func(x.T) 即 (N, S), 列是个体. 这里转回 (S, N) 处理.
            """
            arr = np.asarray(x_array, dtype=np.float64)
            if arr.ndim == 1:
                # scipy 偶尔传 1D (e.g. 兜底评估, 单个体 (dim,))
                # Iter-104: 多目标时返回 lap + weight*proxy (与 objective() 一致).
                lap, proxy = evaluate(arr)
                return np.array([lap + weight * proxy], dtype=np.float64)
            # arr shape: (dim, N) — scipy vectorized 约定 (列是个体)
            # 转回 (N, dim) 处理
            arr_t = arr.T  # (N, dim)
            # Iter-92: 钳制 fuel_load 维度到 baseline (与 evaluate 一致)
            arr_t[:, _FUEL_LOAD_IDX] = _base_fuel_norm
            N = arr_t.shape[0]
            snapped = _snap_vec_batch(arr_t)
            # 逐行查 cache, 收集未缓存项. cache 存 (lap, proxy) 二元组.
            results = np.empty(N, dtype=np.float64)
            uncached_idxs: list[int] = []
            uncached_setups: list[CarSetup] = []
            for i in range(N):
                key = tuple(np.round(snapped[i], 6))
                cached = cache.get(key)
                setup = CarSetup.from_vector(snapped[i].tolist())
                if cached is not None:
                    lap_c, proxy_c = cached
                    # Iter-186: 约束惩罚 (仅当 enable_constraints=True)
                    constraint_pen = _setup_constraint_penalty(setup) if enable_constraints else 0.0
                    results[i] = lap_c + weight * proxy_c + constraint_pen
                else:
                    uncached_idxs.append(i)
                    uncached_setups.append(setup)
            # 批量预测未缓存项
            if uncached_setups:
                items = [(s, track_id, driver_profile) for s in uncached_setups]
                preds = model.predict_batch(items)
                for idx, setup, pred in zip(
                    uncached_idxs, uncached_setups, preds, strict=True,
                ):
                    lap = float(pred["lap_time"])
                    # Iter-164.03: 始终从 batched responses 计算 proxy
                    # (即使 weight==0), 让 SearchResult 报告真实胎耗画像.
                    # 与 evaluate() 顺序路径数值一致.
                    resp = pred["responses"]
                    proxy = (
                        (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
                        + float(resp["slip_angle"]) / _SLIP_REF
                        + float(resp["tyre_load_spread"])
                    )
                    # Iter-186: 约束惩罚 (仅当 enable_constraints=True)
                    constraint_pen = _setup_constraint_penalty(setup) if enable_constraints else 0.0
                    results[idx] = lap + weight * proxy + constraint_pen
                    # cache key 与 evaluate() 一致: tuple(round(snapped, 6))
                    sv_key = tuple(np.round(snapped[idx], 6))
                    cache[sv_key] = (lap, proxy)
            return results

        best_vec, trace, algorithm, elite_survival = _vectorized_differential_evolution(
            objective_vec=objective_vec,
            iterations=iterations,
            seed=seed,
            elite_count=elite_count,
        )
    elif _de is not None:
        best_vec, trace, algorithm, elite_survival = _differential_evolution(
            objective=objective,
            iterations=iterations,
            seed=seed,
            elite_count=elite_count,
        )
    else:
        best_vec, trace = _hill_climb(
            objective=objective,
            baseline_vec=base_vec,
            iterations=iterations,
            seed=seed,
        )
        algorithm = "numpy-local"
        elite_survival = []

    # Iter-92: 钳制 best_vec 的 fuel_load 维度到 baseline (与 evaluate/objective_vec
    # 内部钳制一致). DE 搜索时 fuel_load 被钳制评估, 但 best_vec 本身是 DE 原始
    # 输出, fuel_load 维度可能任意. 若不钳制, recommended_setup.fuel_load 与
    # recommended_lap (用 baseline fuel 评估) 不一致, 导致 "永不推荐比基线更差"
    # 保障失效 (lap 对但 setup 错).
    best_vec[_FUEL_LOAD_IDX] = _base_fuel_norm
    recommended_setup = CarSetup.from_vector(best_vec.tolist())
    recommended_lap, recommended_proxy = evaluate(best_vec)
    baseline_lap, baseline_proxy = evaluate(base_vec)

    # Iter-67 调教输出质量保障: 永不推荐比基线更差的调教 (EA F1 2026 专业车队标准).
    # DE 在低迭代预算 (iterations<10) 下可能找不到比基线更优的解, 尤其当基线已
    # 接近赛道类型最优时 (V-shape 先验下 gain≈0 是物理正确的, DNN 残差噪声可能
    # 让 DE 跑到略差的点). 此时回退到基线, gain=0, 保证调教输出可验证地不劣于
    # 输入. 多目标时比较组合目标 (lap + weight*proxy); 单目标退化为 lap 比较.
    recommended_obj = recommended_lap + weight * recommended_proxy
    baseline_obj = baseline_lap + weight * baseline_proxy
    if recommended_obj > baseline_obj + 1e-9:
        recommended_setup = base_setup
        recommended_lap = baseline_lap
        recommended_proxy = baseline_proxy

    gain = baseline_lap - recommended_lap
    diff = recommended_setup.diff(base_setup)

    # Iter-69 调教输出整体性: 报告推荐/基线的完整 7 项响应画像 (自然单位).
    # predict_full 一次性返回圈速 + 分段 + 7 项响应; 缓存已在 evaluate 中命中.
    rec_profile = _response_profile(recommended_setup, track_id, driver_profile)
    base_profile = _response_profile(base_setup, track_id, driver_profile)

    # Iter-77 调教分析侧集成: 置信度 + 变化解释 + 敏感参数.
    from f1opt.model.confidence import confidence_label, prediction_confidence
    from f1opt.model.setup_analysis import (
        analyze_setup_contributions,
        explain_setup_change,
    )

    conf = prediction_confidence(recommended_setup, track_id, driver_profile)
    conf_label = confidence_label(conf)
    # 变化解释 (推荐 vs 基线), 按 |contribution| 降序.
    change_expl = explain_setup_change(
        base_setup, recommended_setup, track_id, driver_profile,
    )
    # 推荐参数 top-5 敏感度 (供工程师理解推荐 setup 在哪些参数上最敏感).
    contribs = analyze_setup_contributions(
        recommended_setup, track_id, driver_profile,
    )
    top_sensitive = [
        {
            "field": c.field_name,
            "sensitivity": c.sensitivity,
            "optimal_direction": c.optimal_direction,
        }
        for c in contribs[:5]
    ]

    return SearchResult(
        recommended=recommended_setup.model_dump(),
        baseline=base_setup.model_dump(),
        predicted_gain_s=float(gain),
        baseline_lap_time=float(baseline_lap),
        recommended_lap_time=float(recommended_lap),
        diff=diff,
        search_trace=[float(t) for t in trace],
        model_version=MODEL_VERSION,
        iterations=int(iterations),
        algorithm=algorithm,
        # Iter-164.05: 报告归一化胎耗代理 [0,1] (raw proxy 可超 1.0, 不便跨场景对比).
        # 优化目标函数仍用 raw proxy (保留 Iter-94 weight=0.3 的物理标定).
        tire_wear=_normalize_tire_wear_proxy(float(recommended_proxy)),
        tire_wear_weight=weight,
        response_profile=rec_profile,
        baseline_response_profile=base_profile,
        feedback_corrected=use_feedback,
        confidence=float(conf),
        confidence_label=conf_label,
        change_explanation=change_expl,
        top_sensitive_params=top_sensitive,
        elite_survival=elite_survival,
    )


def _response_profile(
    setup: CarSetup, track_id: str, driver_profile: Any,
) -> dict[str, float]:
    """Iter-69: 返回 setup 的完整 7 项响应画像 (自然单位).

    用 :func:`predict_full` 一次性拿到 responses, 转成 ``{name: value}`` 字典.
    用于 :class:`SearchResult.response_profile` / ``baseline_response_profile``,
    让车队工程师看到圈速之外的物理全貌 (滑移/胎温/侧向 G/载荷离散等).
    """
    pred = predict_full(setup, track_id, driver_profile)
    resp = pred["responses"]
    return {name: float(val) for name, val in resp.items()}


# --- Iter-178 精英保留辅助函数 ---------------------------------------------
def _extract_population(result: Any) -> np.ndarray | None:
    """从 scipy DE 结果中提取种群矩阵 (N, dim).

    仅在 ``polish=False`` 时 scipy 会在结果中存储 ``population`` 和
    ``population_energies``. 若不可用返回 None, 调用方应跳过精英保留.
    """
    if hasattr(result, "population") and result.population is not None:
        return np.asarray(result.population, dtype=np.float64)
    return None


def _extract_energies(
    result: Any, pop: np.ndarray, objective: Any,
) -> np.ndarray:
    """提取种群能量; 优先用 ``result.population_energies``, 不可用时逐条评估."""
    if (
        hasattr(result, "population_energies")
        and result.population_energies is not None
    ):
        return np.asarray(result.population_energies, dtype=np.float64)
    return np.array([float(objective(p)) for p in pop], dtype=np.float64)


def _count_elite_survival(
    pop: np.ndarray,
    sorted_idx: np.ndarray,
    prev_elites: list[np.ndarray],
    elite_count: int,
) -> int:
    """统计上一代精英在当前种群 top-elite_count 中的存活数.

    用 rounded 向量作为 key (与 ``evaluate`` 中的缓存键一致) 做集合成员检查.
    """
    elite_keys = {tuple(np.round(e, 6)) for e in prev_elites}
    survived = 0
    for i in range(min(elite_count, len(sorted_idx))):
        key = tuple(np.round(pop[sorted_idx[i]], 6))
        if key in elite_keys:
            survived += 1
    return survived


def _inject_elites(
    pop: np.ndarray,
    sorted_idx: np.ndarray,
    elite_vectors: list[np.ndarray],
    elite_count: int,
) -> np.ndarray:
    """构建下一代种群: 用精英替换最差的 ``elite_count`` 个个体.

    返回新种群 (copy), 不修改传入的 ``pop``.
    """
    n = min(elite_count, len(sorted_idx))
    new_pop = pop.copy()
    worst_idx = sorted_idx[-n:]
    for i in range(min(n, len(elite_vectors))):
        new_pop[worst_idx[i]] = elite_vectors[i]
    return new_pop


def _differential_evolution(
    objective: Any,
    iterations: int,
    seed: int | None,
    elite_count: int = 3,
) -> tuple[np.ndarray, list[float], str, list[int]]:
    """``scipy.optimize.differential_evolution`` 全局优化.

    在归一化 [0,1]^21 空间上最小化 ``objective``; 用 callback 收集每代最优
    圈速 (非递增) 作为 ``search_trace``. 返回 ``(best_vec (snapped), trace,
    "scipy-de", elite_survival)``.

    ``polish=True`` 让 scipy 在 DE 收敛后跑一次 L-BFGS-B 局部精修; 目标函数经
    ``_snap_vec`` 缓存后基本分段常量, L-BFGS-B 有限差分梯度 ≈ 0, 几乎瞬间完成
    但也无效果. Iter-84 实测 5 赛道 gain 在 polish=True/False 完全一致, 故显式
    关闭 (``_DE_POLISH = False``) 避免无谓初始化开销.
    ``maxiter=min(iterations, _DE_MAXITER_CAP)``: DE 全局探索 25 代已足够收敛,
    cap 后延迟 ≤2s; numpy 回退路径不受 cap 约束 (其 ``iterations`` 即评估预算).
    ``popsize`` 取 3 (见 :data:`_DE_POPSIZE` 注释, Iter-84 消融实验) 兼顾全局性
    与时延 (1.67x 加速, gain 仅 -4%).

    Iter-178 精英保留: ``elite_count > 0`` 时每代 DE 结束后保留前 ``elite_count``
    个最优个体作为精英, 注入下一代种群 (替换最差个体). 精英保留防止最优解在
    变异/交叉中丢失, 加速收敛. ``elite_count=0`` 时回退到原始行为 (单次 DE 调用,
    无精英保留, ``elite_survival=[]``).
    """
    trace: list[float] = []
    elite_survival: list[int] = []

    maxiter = min(iterations, _DE_MAXITER_CAP)

    if elite_count <= 0:
        def callback(intermediate_result: Any) -> None:
            try:
                trace.append(float(intermediate_result.fun))
            except (AttributeError, TypeError):
                pass

        result = _de(
            objective,
            _BOUNDS,
            maxiter=maxiter,
            seed=seed,
            polish=_DE_POLISH,
            tol=1e-6,
            popsize=_DE_POPSIZE,
            callback=callback,
        )
        best_vec = _snap_vec(np.asarray(result.x, dtype=np.float64))
        if not trace:
            trace.append(float(objective(best_vec)))
        return best_vec, trace, "scipy-de", elite_survival

    # Iter-178 精英保留路径: 逐代运行 DE, 代间注入精英
    elite_vectors: list[np.ndarray] = []
    population: np.ndarray | None = None

    for gen in range(maxiter):
        # 每代用确定性派生 seed (seed+gen)，避免 gen>0 时 seed=None 导致
        # 变异/交叉非确定 (同 seed 下 search_setup 结果漂移 ~0.6%)。
        gen_seed = (seed + gen) if seed is not None else None

        kwargs: dict[str, Any] = {
            "maxiter": 1,
            "seed": gen_seed,
            "polish": False,
            "tol": 1e-6,
        }
        if population is not None:
            kwargs["init"] = population
        else:
            kwargs["popsize"] = _DE_POPSIZE

        result = _de(objective, _BOUNDS, **kwargs)

        trace.append(float(result.fun))

        pop = _extract_population(result)
        if pop is not None:
            energies = _extract_energies(result, pop, objective)
            sorted_idx = np.argsort(energies)

            if elite_vectors and gen > 0:
                survived = _count_elite_survival(
                    pop, sorted_idx, elite_vectors, elite_count,
                )
                elite_survival.append(survived)

            elite_vectors = [
                pop[sorted_idx[i]].copy()
                for i in range(min(elite_count, len(sorted_idx)))
            ]

            if gen < maxiter - 1:
                population = _inject_elites(
                    pop, sorted_idx, elite_vectors, elite_count,
                )
        else:
            elite_vectors = []
            population = None

    best_vec = _snap_vec(np.asarray(result.x, dtype=np.float64))
    if not trace:
        trace.append(float(objective(best_vec)))

    return best_vec, trace, "scipy-de", elite_survival


# --- Iter-85 向量化 DE (predict_batch 批量评估) -----------------------------
def _snap_vec_batch(x_array: np.ndarray) -> np.ndarray:
    """``_snap_vec`` 的批量版本: 对 (N, dim) 数组每行 snap 到游戏档位网格.

    与 :func:`_snap_vec` 在单行上等价 (逐行 round → clip), 但用 numpy 向量化
    避免 Python 循环. 用于 :func:`_vectorized_differential_evolution` 的目标
    函数, 把整代种群一次性 snap.
    """
    arr = np.asarray(x_array, dtype=np.float64)
    if arr.ndim == 1:
        return _snap_vec(arr)
    steps = np.round(arr / _NORM_STEPS)
    return np.clip(steps * _NORM_STEPS, 0.0, 1.0)


def _vectorized_differential_evolution(
    objective_vec: Any,
    iterations: int,
    seed: int | None,
    elite_count: int = 3,
) -> tuple[np.ndarray, list[float], str, list[int]]:
    """``scipy.optimize.differential_evolution`` 配 ``vectorized=True``.

    Iter-85: scipy 的 ``vectorized=True`` 模式把整代种群 (N×dim) 一次性传给
    ``objective_vec``, 后者用 :meth:`SurrogateModel.predict_batch` 做一次批量
    DNN forward. 实测 predict_batch 比逐条 predict_lap_time 快 ~7x
    (0.10 ms/each vs 0.72 ms/each @ N=57). DE cap=25 popsize=3 总 forwards
    从 ~1.5s 降到 ~0.2s, search_setup 总耗时从 ~1.5s 降到 ~0.3s.

    返回 ``(best_vec (snapped), trace, "scipy-de-vec", elite_survival)``.

    算法与 :func:`_differential_evolution` 一致 (相同 seed 下结果近似, 仅批次
    评估顺序不同), 故 gain 质量保持 Iter-84 水平 (avg 2.019s, sakhir 0.65s).

    Iter-178 精英保留: ``elite_count > 0`` 时逐代运行 vectorized DE, 代间注入
    精英 (与 :func:`_differential_evolution` 相同的精英保留逻辑).
    ``elite_count=0`` 时回退到原始行为 (单次 DE 调用).
    """
    trace: list[float] = []
    elite_survival: list[int] = []

    maxiter = min(iterations, _DE_MAXITER_CAP_VECTORIZED)

    if elite_count <= 0:
        def callback(intermediate_result: Any) -> None:
            try:
                trace.append(float(intermediate_result.fun))
            except (AttributeError, TypeError):
                pass

        result = _de(
            objective_vec,
            _BOUNDS,
            maxiter=maxiter,
            seed=seed,
            polish=False,
            tol=1e-6,
            popsize=_DE_POPSIZE,
            callback=callback,
            vectorized=True,
            updating="deferred",
        )
        best_vec = _snap_vec(np.asarray(result.x, dtype=np.float64))
        if not trace:
            trace.append(float(objective_vec(best_vec)))
        return best_vec, trace, "scipy-de-vec", elite_survival

    # Iter-178 精英保留路径: 逐代运行 vectorized DE, 代间注入精英
    elite_vectors: list[np.ndarray] = []
    population: np.ndarray | None = None

    for gen in range(maxiter):
        # 每代用确定性派生 seed (seed+gen)，避免 gen>0 时 seed=None 导致
        # 变异/交叉非确定 (同 seed 下 search_setup 结果漂移 ~0.6%)。
        gen_seed = (seed + gen) if seed is not None else None

        kwargs: dict[str, Any] = {
            "maxiter": 1,
            "seed": gen_seed,
            "polish": False,
            "tol": 1e-6,
            "vectorized": True,
            "updating": "deferred",
        }
        if population is not None:
            kwargs["init"] = population
        else:
            kwargs["popsize"] = _DE_POPSIZE

        result = _de(objective_vec, _BOUNDS, **kwargs)

        trace.append(float(result.fun))

        pop = _extract_population(result)
        if pop is not None:
            energies = _extract_energies(result, pop, objective_vec)
            sorted_idx = np.argsort(energies)

            if elite_vectors and gen > 0:
                survived = _count_elite_survival(
                    pop, sorted_idx, elite_vectors, elite_count,
                )
                elite_survival.append(survived)

            elite_vectors = [
                pop[sorted_idx[i]].copy()
                for i in range(min(elite_count, len(sorted_idx)))
            ]

            if gen < maxiter - 1:
                population = _inject_elites(
                    pop, sorted_idx, elite_vectors, elite_count,
                )
        else:
            elite_vectors = []
            population = None

    best_vec = _snap_vec(np.asarray(result.x, dtype=np.float64))
    if not trace:
        trace.append(float(objective_vec(best_vec)))

    return best_vec, trace, "scipy-de-vec", elite_survival


def _hill_climb(
    objective: Any,
    baseline_vec: np.ndarray,
    iterations: int,
    seed: int | None,
) -> tuple[np.ndarray, list[float]]:
    """多起点爬山: 基线起点 + 随机重启 + 自适应高斯扰动 + 坐标轮换 polish.

    返回 ``(best_vec (snapped), search_trace)``; ``search_trace`` 长度 ==
    ``iterations``, 每项为该步结束时的 best-yet 圈速 (非递增).
    """
    rng = np.random.default_rng(seed)
    trace: list[float] = []

    best_vec = _snap_vec(baseline_vec).copy()
    best_yet = objective(best_vec)

    if iterations <= 0:
        return best_vec, trace

    evals = 0

    def record(vec: np.ndarray) -> float:
        nonlocal best_yet, best_vec, evals
        lap = objective(vec)
        if lap < best_yet - 1e-12:
            best_yet = lap
            best_vec = vec.copy()
        trace.append(best_yet)
        evals += 1
        return lap

    # 预算分配: iterations 切给若干重启 (第一段从基线出发), 每段有上限以保证
    # 能跑出多段重启; 停滞提前结束本段, 剩余预算给后续重启 / 坐标 polish.
    n_restarts = max(1, min(8, iterations // 20))
    max_per_restart = max(_DIM, iterations // n_restarts)

    r = 0
    while evals < iterations:
        # 起点: 第一段从 (snapped) 基线出发; 后续段随机重启.
        if r == 0:
            cur = best_vec.copy()
            cur_lap = best_yet
        else:
            cur = _snap_vec(rng.random(_DIM))
            cur_lap = record(cur)
            if evals >= iterations:
                break
        r += 1

        sigma = 0.12
        stall = 0
        steps_here = 0
        while evals < iterations and steps_here < max_per_restart:
            steps_here += 1
            # 30% 概率只扰动少数维 (精细搜索), 70% 全维高斯 (广度探索).
            if rng.random() < 0.3:
                k = max(1, int(rng.integers(1, _DIM + 1)))
                idx = rng.choice(_DIM, size=k, replace=False)
                cand = cur.copy()
                cand[idx] = cand[idx] + rng.normal(0.0, sigma, size=k)
            else:
                cand = cur + rng.normal(0.0, sigma, size=_DIM)
            np.clip(cand, 0.0, 1.0, out=cand)
            cand = _snap_vec(cand)
            lap = record(cand)
            if lap < cur_lap - 1e-12:
                cur = cand
                cur_lap = lap
                stall = 0
                sigma = min(0.25, sigma * 1.2)  # 成功 -> 略放大步长加速
            else:
                stall += 1
                sigma = max(0.01, sigma * 0.85)  # 停滞 -> 缩小步长精细逼近
            if stall >= _DIM * 2:
                break

    # 坐标轮换 polish: 沿各维 ±1 step 试探 (用剩余预算), 收敛到局部最优档位.
    for i in rng.permutation(_DIM):
        if evals >= iterations:
            break
        for sign in (1.0, -1.0):
            if evals >= iterations:
                break
            cand = best_vec.copy()
            cand[i] = cand[i] + sign * _NORM_STEPS[i]
            cand = _snap_vec(np.clip(cand, 0.0, 1.0))
            if np.array_equal(cand, best_vec):
                continue
            record(cand)

    return best_vec, trace
