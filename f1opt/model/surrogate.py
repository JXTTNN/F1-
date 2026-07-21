"""分段调教条件代理模型 (segment-level setup-conditioned surrogate).

Iter-02 把 Iter-01 的单标量 MLP 升级为 *分段* (sector-level) 多任务 DNN:

- 输入向量 = concat(``CarSetup.to_vector()`` (19), 赛道上下文 (10), 车手画像 (8)) = 37 维.
- 主干 MLP ``37 -> 128 -> 128 -> 64`` (ReLU) 之后接两个零初始化的输出头:
  * sector head (3): 三段计时段 (秒), 求和 = 圈速; 残差加到分段先验上.
  * response head (7): 速度/滑移/载荷/侧倾/胎温/侧向 G 等响应指标.
- 零初始化输出层使 *未训练* 模型恰好返回先验, 保证圈速落在合理 F1 区间.
- 训练后模型对 setup / driver 真正敏感 (见 ``tests/model/test_surrogate.py``).

公开 API (供 api / feedback / 训练脚本使用):

- :class:`SurrogateModel` — 训练 / 推理 / 存取.
- :func:`predict_lap_time` — 模块级便捷函数 (向后兼容, 返回 float).
- :func:`predict_full` — 模块级便捷函数, 返回富字典 (圈速/分段/响应/版本).
- :data:`MODEL_VERSION` — *公开* 版本常量 (修复 api 导入 bug).
- :func:`_get_default_model` / :func:`reset_default_model_cache` — 缓存默认模型.

Response head 输出单位 (自然单位, 残差经 ``RESPONSE_SCALES`` 反归一化):

- ``speed_avg``        : m/s        (整圈平均速度)
- ``speed_max``        : m/s        (圈中最高速度)
- ``slip_angle``       : degrees    (后轴平均滑移角)
- ``tyre_load_spread`` : 归一化 [0,1] (四轮载荷离散度)
- ``rake``             : degrees    (后-前离地高度差映射)
- ``tyre_temp``        : celsius    (四轮平均胎温)
- ``g_lat_max``        : G          (圈中最大横向加速度)
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from f1opt.config import get_settings
from f1opt.data.ea_f1_2026_benchmark import EA_F1_2026_LAP_TIME_BENCHMARK
from f1opt.data.sector_times import sector_times_for
from f1opt.data.setup_schema import CarSetup
from f1opt.data.tracks import TRACKS_BY_ID, Track

# --- 维度常量 ---------------------------------------------------------------
SETUP_DIM = 19
# [length, corners, is_sprint, one-hot track_type (5), elevation, unknown_flag]
TRACK_CONTEXT_DIM = 1 + 1 + 1 + 5 + 1 + 1
DRIVER_DIM = 8
INPUT_DIM = SETUP_DIM + TRACK_CONTEXT_DIM + DRIVER_DIM  # 37

N_SECTORS = 3
N_RESPONSES = 7
RESPONSE_NAMES: tuple[str, ...] = (
    "speed_avg",
    "speed_max",
    "slip_angle",
    "tyre_load_spread",
    "rake",
    "tyre_temp",
    "g_lat_max",
)

# --- 公开版本常量 (修复 api/app.py 的 ``import MODEL_VERSION`` bug) -----------
MODEL_VERSION = "seg-dnn-torch-v0.3"

# --- 赛道类型 one-hot 顺序 (与 tracks.TrackType 定义一致) -------------------
TRACK_TYPES: list[str] = [
    "high_speed_low_downforce",
    "street",
    "high_downforce",
    "medium",
    "mixed",
]
_TRACK_TYPE_IDX: dict[str, int] = {t: i for i, t in enumerate(TRACK_TYPES)}

# --- 物理先验参数 -----------------------------------------------------------
# 按赛道类型估算的平均速度 (m/s), 用于 length/avg_speed 圈速基线.
AVG_SPEED: dict[str, float] = {
    "high_speed_low_downforce": 80.0,
    "street": 50.0,
    "high_downforce": 65.0,
    "medium": 70.0,
    "mixed": 72.0,
}
_DEFAULT_AVG_SPEED = 70.0
_DEFAULT_LENGTH_M = 5000.0  # 未知赛道回退长度
_FUEL_PENALTY_PER_KG = 0.03
_MIN_LAP_TIME = 60.0
_MAX_LAP_TIME = 200.0

# EA F1 2026 系统物理增益 (Iter-67): lap_simulator 在 reference 条件下相对 raw
# benchmark 的净圈速增益 = ERS 部署 + 主动空动 X-mode + DRS, 约 -1.70s (24 赛道
# 一致, std<0.02s; 高速赛道略多 -1.748s). 把它纳入先验让 DNN 残差从 ~1.7s 降到
# ~0.05s, 防止 DNN 为拟合大常数偏移而过拟合翻转单个 setup 参数的方向.
# 来源: setup_lap_time(optimal_setup, track, driver_offset=0) - (benchmark+fuel+penalty)
#
# Iter-108: -1.70 → -0.90. Iter-108 修复 lap_simulator PU delta 用 per-track
# reference (而非全局 _REF_PU_GAIN_S 假设 harvest_factor=1.0), 消除 5 道
# low-harvest 赛道的 -0.048s 偏差. 修复后 24 道 sim_lap = benchmark (0.0000s
# 误差, 0%), 新 offset = sim_lap - benchmark - fuel_pen = 0 - 0.9 = -0.9 (std=0,
# 24 道完全一致). DNN 残差从 ~0.9s (旧 offset 多估 0.8s) 降到 ~0.05s.
_PHYSICS_SYSTEM_OFFSET_S = -0.90

# 分段先验权重 (三段大致等分, 和为 1).
_SECTOR_PRIOR_WEIGHTS: tuple[float, float, float] = (0.34, 0.33, 0.33)

# Response 先验 *偏移* 与 *归一化尺度* (训练时把目标归一化到 O(1), 推理时反归一化).
# index 顺序与 RESPONSE_NAMES 一致.
RESPONSE_PRIORS: tuple[float, ...] = (
    70.0,   # speed_avg (m/s) — 被赛道先验覆盖
    98.0,   # speed_max (m/s)
    2.0,    # slip_angle (deg)
    0.3,    # tyre_load_spread
    0.5,    # rake (deg)
    90.0,   # tyre_temp (celsius)
    2.5,    # g_lat_max (G)
)
RESPONSE_SCALES: tuple[float, ...] = (
    10.0,   # speed_avg
    15.0,   # speed_max
    1.0,    # slip_angle
    0.2,    # tyre_load_spread
    0.5,    # rake
    20.0,   # tyre_temp
    1.0,    # g_lat_max
)
SECTOR_SCALE = 1.0  # 秒


# --- Driver × setup × track 物理修正 (Iter-11) --------------------------------
# 即使 DNN 输出头未训练 (残差=0), 此修正让预测对 (车手画像 × 调教 × 赛道类型)
# 交叉敏感, 使优化器能为不同驾驶风格推荐不同 setup — EA F1 工程标准
# "driver-style setup".
#
# Setup 向量索引 (与 CarSetup.to_vector() 一致, 已归一化 [0,1]):
#   0 front_wing  1 rear_wing  2 on_throttle_diff  3 off_throttle_diff
#   4 front_camber 5 rear_camber 6 front_toe 7 rear_toe
#   8 front_suspension 9 rear_suspension 10 front_arb 11 rear_arb
#   12 front_ride_height 13 rear_ride_height 14 brake_pressure 15 front_brake_bias
#   16 front_tyre_pressure 17 rear_tyre_pressure 18 fuel_load
#
# Driver 向量索引 (DRIVER_DIM=8):
#   0 brake_point_norm (低=晚刹激进)  1 throttle_smoothness  2 steer_smoothness
#   3 corner_balance_pref (高=偏好转向过度)  4 aggression_score
#   5 consistency_score  6 ers_usage_intensity  7 drs_usage_efficiency
#
# 交叉项: (driver_idx, setup_idx, sector, gain, amp_key)
#   delta = gain * (driver[di]-0.5) * (setup[si]-0.5) * track_amp
#   gain<0: 同向 (都高或都低) → 更快; gain>0: 反向 → 更快
_DRIVER_SETUP_CROSS_TERMS: tuple[tuple[int, int, int, float, str], ...] = (
    # --- 原始 8 项 (Iter-11) ---
    (0, 15, 0, 0.60, "brake"),    # 晚刹车手需更多前制动分配 (sector 1 重刹区)
    (1, 2, 2, -0.50, "traction"), # 油门平顺者配高锁止差速器 (sector 3 出弯)
    (2, 10, 1, -0.40, "corner"),  # 转向平顺者配硬前防倾杆 (sector 2 技术段)
    (4, 1, 2, -0.40, "aero"),     # 激进车手利用高后翼下压力 (sector 3)
    (3, 0, 1, 0.35, "corner"),    # 转向过度偏好者配低前翼 (sector 2)
    (6, 14, 0, -0.30, "brake"),   # ERS 攻击模式配高制动压力 (sector 1 回收)
    (5, 6, 1, -0.25, "corner"),   # 一致性高者配大前束 (sector 2 指向性)
    (7, 13, 2, -0.30, "aero"),    # DRS 高效者配低后离地 (sector 3 直道)
    # --- Iter-90: 补充 10 项覆盖剩余调教维 (8→18 项, 全 18 调教维覆盖) ---
    (4, 3, 0, -0.35, "brake"),    # 激进+多收油锁止→进弯旋转 (同向→快)
    (3, 4, 1, 0.30, "corner"),    # 过度转向偏好+更负前外倾→前抓地 (反向→快)
    # Iter-90 修正: oversteer-pref driver 想要 *减少* 后抓地 (高 setup[5]=less
    # negative rear camber=less rear grip=更多 oversteer). 同向→快, gain<0.
    # 旧版 gain>0 方向反, 导致 held-out lap MAE 0.296→0.339 回归.
    (3, 5, 1, -0.25, "corner"),   # 过度转向偏好+更不负后外倾→后抓地少 (同向→快)
    (5, 7, 1, -0.20, "corner"),   # 一致性好+大后束→稳定 (同向→快)
    (2, 8, 1, 0.30, "corner"),    # 平顺转向+软前悬→路感 (反向→快)
    (1, 9, 2, 0.30, "traction"),  # 平顺油门+软后悬→牵引 (反向→快)
    (4, 11, 1, -0.30, "corner"),  # 激进+硬后防倾杆→旋转 (同向→快)
    # Iter-90 修正: 晚刹车手 (low driver[0]) 需要低前离地 (low setup[12]) 减小
    # 制动俯冲, 稳定刹车点. 同向→快, gain<0. 旧版 gain>0 方向反.
    (0, 12, 0, -0.30, "brake"),   # 晚刹+低前离地→制动稳定 (同向→快)
    (5, 16, 1, -0.20, "corner"),  # 一致+高前压→精准 (同向→快)
    (6, 17, 2, 0.25, "traction"), # ERS重+低后压→牵引 (反向→快)
)

# 赛道类型对各 amp_key 的放大系数 (EA F1: 赛道特性决定 driver-setup 匹配收益)
_TRACK_TYPE_AMP: dict[str, dict[str, float]] = {
    "high_speed_low_downforce": {"brake": 1.3, "traction": 0.9, "corner": 0.8, "aero": 1.2},
    "street": {"brake": 0.9, "traction": 1.3, "corner": 1.4, "aero": 0.8},
    "high_downforce": {"brake": 1.0, "traction": 1.1, "corner": 1.2, "aero": 1.2},
    "medium": {"brake": 1.0, "traction": 1.0, "corner": 1.0, "aero": 1.0},
    "mixed": {"brake": 1.0, "traction": 1.0, "corner": 1.0, "aero": 1.0},
}
_DEFAULT_AMP: dict[str, float] = {"brake": 1.0, "traction": 1.0, "corner": 1.0, "aero": 1.0}

# Setup 无关的车手基线偏移 (秒, 负=更快). (driver_idx, per-sector gains)
# 使激进/保守车手整体圈速差异落到真实 F1 区间 (~0.4-0.9 s).
_DRIVER_BASELINE_SHIFTS: tuple[tuple[int, tuple[float, float, float]], ...] = (
    (4, (-0.20, -0.20, -0.20)),  # aggression: 激进整体快 ~0.5s 满量程
    (5, (-0.04, -0.05, -0.04)),  # consistency: 一致性高略快
    (6, (-0.12, 0.0, -0.12)),    # ers_usage: 直道快
    (7, (0.0, 0.0, -0.08)),      # drs_eff: 最长直道快
)


def _driver_sector_correction(
    driver_vec: np.ndarray, setup_vec: np.ndarray, track_id: str
) -> np.ndarray:
    """Driver × setup × track 物理修正: 返回三段秒级 delta (负=更快).

    即使 DNN 输出头零初始化 (残差=0), 此修正保证预测对车手画像 + 调教
    交叉敏感, 使 :func:`search_setup` 能为不同驾驶风格推荐不同 setup.
    """
    corr = np.zeros(N_SECTORS, dtype=np.float32)
    track = _resolve_track(track_id)
    amp = _TRACK_TYPE_AMP.get(track.track_type if track is not None else "", _DEFAULT_AMP)
    for di, si, sec, gain, key in _DRIVER_SETUP_CROSS_TERMS:
        d_dev = float(driver_vec[di]) - 0.5
        s_dev = float(setup_vec[si]) - 0.5
        corr[sec] += gain * d_dev * s_dev * amp.get(key, 1.0)
    for di, sec_gains in _DRIVER_BASELINE_SHIFTS:
        d_dev = float(driver_vec[di]) - 0.5
        corr[0] += d_dev * sec_gains[0]
        corr[1] += d_dev * sec_gains[1]
        corr[2] += d_dev * sec_gains[2]
    return corr


# --- 特征工程 ---------------------------------------------------------------
def _resolve_track(track_id: str) -> Track | None:
    """按 ``track_id`` 查询赛道; 未知返回 None (不抛异常).

    Iter-72: 走 ``canonical_track_id`` 把 benchmark 规范名 (bahrain) 反向映射
    到 TRACKS_BY_ID 城市名 (sakhir), 保证别名/规范名行为一致.
    """
    from f1opt.data.ea_f1_2026_benchmark import canonical_track_id

    return TRACKS_BY_ID.get(canonical_track_id(track_id))


def track_context(track_id: str) -> np.ndarray:
    """返回归一化赛道上下文向量 (长度 = TRACK_CONTEXT_DIM).

    未知 track_id 返回全零向量并把最后一位置 1 (unknown_flag).
    """
    vec = np.zeros(TRACK_CONTEXT_DIM, dtype=np.float32)
    track = _resolve_track(track_id)
    if track is None:
        vec[-1] = 1.0  # unknown flag
        return vec
    vec[0] = track.length_m / 8000.0
    vec[1] = track.corners / 30.0
    vec[2] = float(track.is_sprint)
    idx = _TRACK_TYPE_IDX.get(track.track_type, -1)
    if idx >= 0:
        vec[3 + idx] = 1.0
    vec[8] = track.elevation_change_m / 120.0
    # vec[9] (unknown flag) 保持 0.0
    return vec


def _driver_vec_from_iterable(values: Any) -> np.ndarray:
    out = np.zeros(DRIVER_DIM, dtype=np.float32)
    vals: list[float] = []
    for v in values:
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float | np.floating | np.integer):
            vals.append(float(v))
        if len(vals) >= DRIVER_DIM:
            break
    out[: len(vals)] = vals[:DRIVER_DIM]
    return out


def _normalize_driver_vector(driver_profile: Any) -> np.ndarray:
    """把多种车手画像表示归一为长度 ``DRIVER_DIM`` 的 float 向量.

    接受 ``DriverProfile | dict | list[float] | None``; 任何无法识别的输入
    回退到中性向量 (保证推理永不抛异常). ``DriverProfile`` 通过 *惰性* 导入
    引用, 避免 ``f1opt.driver`` 未就绪时的硬循环依赖.

    Iter-93 修复: ``None`` 旧版返回全零 [0]*8, 但全零向量对应 default profile
    (driver_physical_offset = +0.75s, 最慢), 导致 driver=None 时 DNN 预测偏高
    ~0.75s. 新版返回 [0.5]*8 (真正的中性车手, offset≈0), DNN 残差从 +1.5s 降到
    +0.14s (接近期望 +0.2s).
    """
    if driver_profile is None:
        # Iter-93: 中性车手 = [0.5]*8 (非全零, 全零对应最慢 default profile)
        return np.full(DRIVER_DIM, 0.5, dtype=np.float32)
    if isinstance(driver_profile, list | tuple | np.ndarray):
        return _driver_vec_from_iterable(driver_profile)
    if isinstance(driver_profile, dict):
        return _driver_vec_from_iterable(
            driver_profile[k] for k in sorted(driver_profile.keys())
        )
    # 惰性导入 DriverProfile (driver 子任务可能尚未发布).
    try:
        from f1opt.driver.profile import DriverProfile  # type: ignore[import-not-found]
        if isinstance(driver_profile, DriverProfile) and hasattr(driver_profile, "to_vector"):
            return _driver_vec_from_iterable(driver_profile.to_vector())
    except Exception:
        pass
    # 鸭子类型: 任何带 to_vector() 的对象.
    if hasattr(driver_profile, "to_vector"):
        try:
            return _driver_vec_from_iterable(driver_profile.to_vector())
        except Exception:
            pass
    # Iter-93: 无法识别的输入回退到中性 [0.5]*8 (非全零)
    return np.full(DRIVER_DIM, 0.5, dtype=np.float32)


def driver_vector(driver_profile: Any) -> np.ndarray:
    """返回长度 DRIVER_DIM 的车手画像向量; None -> 全零 (占位).

    向后兼容包装: 旧版仅接受 ``dict | None``, 现接受 ``DriverProfile |
    dict | list[float] | None``.
    """
    return _normalize_driver_vector(driver_profile)


def build_input_vector(
    setup: CarSetup,
    track_id: str,
    driver_profile: Any = None,
) -> np.ndarray:
    """拼接完整输入向量 (长度 = INPUT_DIM)."""
    sv = np.asarray(setup.to_vector(), dtype=np.float32)
    tv = track_context(track_id)
    dv = _normalize_driver_vector(driver_profile)
    return np.concatenate([sv, tv, dv]).astype(np.float32)


def track_prior(track_id: str, setup: CarSetup) -> float:
    """物理先验圈速 (Iter-67: setup-aware, EA F1 2026 benchmark + setup 物理惩罚).

    24 赛道用 EA F1 2026 精确基准 (0.01% 误差), 未知赛道回退到 length/speed 启发式.
    燃油惩罚叠加 (fuel_load × 0.03 s/kg).

    Iter-67: 叠加 :func:`setup_penalty_s` (来自 setup_physics_bridge) — setup 偏离
    该赛道类型最优的物理代价. 这让先验 *setup-aware*, DNN 只需学习小残差
    (车手交互 + 非线性 + 噪声), 物理真值标签 (Iter-67 generate_physics_dataset)
    的 held-out MAE 从 2.5s 降到 < 0.3s.
    """
    # 延迟导入避免循环依赖 (setup_physics_bridge -> lap_simulator_2026 -> ...)
    from f1opt.data.ea_f1_2026_benchmark import resolve_track_id
    from f1opt.model.setup_physics_bridge import setup_penalty_s

    bench = EA_F1_2026_LAP_TIME_BENCHMARK.get(resolve_track_id(track_id))
    if bench is not None:
        base = bench + setup.fuel_load * _FUEL_PENALTY_PER_KG
    else:
        track = _resolve_track(track_id)
        if track is not None:
            speed = AVG_SPEED.get(track.track_type, _DEFAULT_AVG_SPEED)
            base = track.length_m / speed + setup.fuel_load * _FUEL_PENALTY_PER_KG
        else:
            base = _DEFAULT_LENGTH_M / _DEFAULT_AVG_SPEED + setup.fuel_load * _FUEL_PENALTY_PER_KG
    # Iter-67: + 系统物理增益 (ERS/主动空动/DRS) + setup 偏离最优的物理代价
    return float(base + _PHYSICS_SYSTEM_OFFSET_S + setup_penalty_s(setup, track_id))


def track_avg_speed(track_id: str) -> float:
    """赛道平均速度先验 (m/s), 未知赛道返回默认值."""
    track = _resolve_track(track_id)
    if track is None:
        return _DEFAULT_AVG_SPEED
    return AVG_SPEED.get(track.track_type, _DEFAULT_AVG_SPEED)


def sector_priors(track_id: str, setup: CarSetup) -> list[float]:
    """把 :func:`track_prior` 拆成三段, 优先用真实 sector 比例 (sector_times).

    EA F1 2026: 24 赛道有真实 S1/S2/S3 比例; 未知赛道回退到 34/33/33 等分.
    """
    lap = track_prior(track_id, setup)
    try:
        sd = sector_times_for(track_id)
        total = sd.total_lap_time_s
        if total > 0.0:
            return [
                lap * sd.s1_s / total,
                lap * sd.s2_s / total,
                lap * sd.s3_s / total,
            ]
    except ValueError:
        pass
    return [lap * w for w in _SECTOR_PRIOR_WEIGHTS]


def response_priors(track_id: str, setup: CarSetup) -> list[float]:
    """Response head 先验 (自然单位, 7 项).

    速度项与赛道先验挂钩, 其余取全局合理常量.
    """
    avg_spd = track_avg_speed(track_id)
    return [
        avg_spd,                 # speed_avg
        avg_spd * 1.4,           # speed_max
        RESPONSE_PRIORS[2],      # slip_angle
        RESPONSE_PRIORS[3],      # tyre_load_spread
        RESPONSE_PRIORS[4],      # rake
        RESPONSE_PRIORS[5],      # tyre_temp
        RESPONSE_PRIORS[6],      # g_lat_max
    ]


def _clamp_lap_time(value: float) -> float:
    return float(min(max(value, _MIN_LAP_TIME), _MAX_LAP_TIME))


# --- 模型 -------------------------------------------------------------------
class SurrogateModel(nn.Module):
    """分段多任务代理模型: (setup, track, driver) -> (3 sectors, 7 responses).

    Iter-3v3 升级: 主干 ``37 -> 256 -> 256 -> 128 -> 64`` (GELU + BatchNorm),
    参数量 ~120K (v0.2 为 ~30K). 零初始化输出头保留, 使未训练模型预测 == 先验.
    BatchNorm 在 eval 模式下使用 running stats (未训练时 mean=0/var=1, 不影响零头).
    """

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.sector_head = nn.Linear(64, N_SECTORS)
        self.response_head = nn.Linear(64, N_RESPONSES)
        self._zero_init_heads()

    def _zero_init_heads(self) -> None:
        """零初始化两个输出层 -> 初始残差 == 0 -> 预测 == 先验."""
        for head in (self.sector_head, self.response_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    # --- 前向 / 训练侧 ------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向计算, 返回 (sector_residual (N,3), response_residual (N,7)).

        残差均为 *归一化* 空间: sector 残差单位 = 秒 (scale 1.0),
        response 残差需乘 ``RESPONSE_SCALES`` 才回到自然单位.
        """
        h = self.trunk(x)
        return self.sector_head(h), self.response_head(h)

    def residuals(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """训练侧别名, 与 :meth:`forward` 一致 (供 train 脚本使用)."""
        return self.forward(x)

    # --- 推理侧 -------------------------------------------------------------
    def predict(
        self,
        setup: CarSetup,
        track_id: str,
        driver_profile: Any = None,
    ) -> dict[str, Any]:
        """富预测: 返回圈速 / 三段 / 响应指标 / 模型版本."""
        x = torch.from_numpy(
            build_input_vector(setup, track_id, driver_profile)
        ).unsqueeze(0)
        sec_prior = np.asarray(sector_priors(track_id, setup), dtype=np.float32)
        resp_prior = np.asarray(response_priors(track_id, setup), dtype=np.float32)
        scales = np.asarray(RESPONSE_SCALES, dtype=np.float32)
        # Driver × setup × track 物理修正 (Iter-11): 即使 DNN 未训练也生效.
        dv = _normalize_driver_vector(driver_profile)
        sv = np.asarray(setup.to_vector(), dtype=np.float32)
        driver_corr = _driver_sector_correction(dv, sv, track_id)
        self.eval()
        with torch.no_grad():
            sec_res, resp_res = self.forward(x)
        sec_res = sec_res.squeeze(0).numpy() * SECTOR_SCALE + sec_prior + driver_corr
        resp_res = resp_res.squeeze(0).numpy() * scales + resp_prior
        sectors = [max(0.01, float(s)) for s in sec_res]
        lap_time = float(sum(sectors))
        responses = {
            name: float(v) for name, v in zip(RESPONSE_NAMES, resp_res, strict=True)
        }
        return {
            "lap_time": lap_time,
            "sectors": sectors,
            "responses": responses,
            "model_version": MODEL_VERSION,
        }

    def predict_lap_time(
        self,
        setup: CarSetup,
        track_id: str,
        driver_profile: Any = None,
    ) -> float:
        """预测单圈圈速 (秒) = ``predict(...)["lap_time"]`` (向后兼容)."""
        return float(self.predict(setup, track_id, driver_profile)["lap_time"])

    # ------------------------------------------------------------------ #
    # Iter-133: Inference confidence estimation
    # ------------------------------------------------------------------ #
    def predict_with_confidence(
        self,
        setup: CarSetup,
        track_id: str,
        driver_profile: Any = None,
    ) -> dict[str, Any]:
        """Iter-133: 富预测 + 置信度估计.

        返回 :meth:`predict` 的全部字段, 外加:

        - ``confidence``: float in [0, 1] (1.0 = 高置信, 0.0 = 低置信).
        - ``confidence_factors``: dict 分解置信度来源::

            {"ood_input_dims": int,        # 输入向量超出 [0,1] 的维度数
             "max_residual_ratio": float,  # max(|sec_residual| / sec_prior)
             "input_penalty": float,       # OOD 输入惩罚 (0.0 - 0.5)
             "residual_penalty": float,    # 大残差惩罚 (0.0 - 0.5)
             "label": str}                 # "high" / "medium" / "low"

        置信度公式::

            confidence = 1.0
                       - input_penalty     # 0.05 * ood_input_dims (capped 0.5)
                       - residual_penalty  # max(0, (ratio - 0.15) / 0.85) * 0.5

        逻辑:
        - **输入空间 OOD**: setup/driver/track 输入归一化到 [0,1]; 任何维度
          超出 [0,1] 表示外推, 每个维度惩罚 0.05 (上限 0.5). 10 个 OOD 维度
          即达到最大惩罚.
        - **残差幅度**: DNN sector 残差 / sector 先验 > 15% 表示模型在做大幅
          修正 (可能在 OOD 区域), 按比例惩罚 (上限 0.5). 残差 < 15% 不惩罚
          (模型在训练分布内, 残差为正常修正).

        ``label`` 阈值: confidence >= 0.8 -> "high", >= 0.5 -> "medium", < 0.5 -> "low".
        """
        result = self.predict(setup, track_id, driver_profile)
        x = build_input_vector(setup, track_id, driver_profile)

        # Factor 1: input-space OOD dims (count dims outside [0, 1]).
        ood_dims = int(np.sum((x < 0.0) | (x > 1.0)))
        input_penalty = min(0.5, 0.05 * ood_dims)

        # Factor 2: residual magnitude relative to sector prior.
        # Reconstruct the DNN residual (sec_res before prior addition).
        sec_prior = np.asarray(sector_priors(track_id, setup), dtype=np.float32)
        sectors = np.asarray(result["sectors"], dtype=np.float32)
        # residual = predicted_sector - prior (the DNN's correction).
        sec_residual = sectors - sec_prior
        # Ratio = |residual| / prior (per sector); take the max.
        safe_prior = np.maximum(sec_prior, 0.5)  # avoid div-by-zero
        ratios = np.abs(sec_residual) / safe_prior
        max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 0.0
        # Penalize only if ratio > 0.15 (15%); scale linearly to 0.5 at ratio=1.0.
        residual_penalty = max(0.0, (max_ratio - 0.15) / 0.85) * 0.5
        residual_penalty = min(0.5, residual_penalty)

        confidence = max(0.0, 1.0 - input_penalty - residual_penalty)
        if confidence >= 0.8:
            label = "high"
        elif confidence >= 0.5:
            label = "medium"
        else:
            label = "low"

        result["confidence"] = confidence
        result["confidence_factors"] = {
            "ood_input_dims": ood_dims,
            "max_residual_ratio": max_ratio,
            "input_penalty": input_penalty,
            "residual_penalty": residual_penalty,
            "label": label,
        }
        return result

    def predict_batch(
        self,
        items: list[tuple[CarSetup, str]] | list[tuple[CarSetup, str, Any]],
    ) -> list[dict[str, Any]]:
        """批量富预测; 与逐条 :meth:`predict` 在容差内一致."""
        if not items:
            return []
        xs: list[np.ndarray] = []
        sec_priors: list[np.ndarray] = []
        resp_priors: list[np.ndarray] = []
        driver_corrs: list[np.ndarray] = []
        for item in items:
            if len(item) == 3:
                setup, track_id, drv = item  # type: ignore[misc]
            else:
                setup, track_id = item  # type: ignore[misc]
                drv = None
            xs.append(build_input_vector(setup, track_id, drv))
            sec_priors.append(np.asarray(sector_priors(track_id, setup), dtype=np.float32))
            resp_priors.append(np.asarray(response_priors(track_id, setup), dtype=np.float32))
            dv = _normalize_driver_vector(drv)
            sv = np.asarray(setup.to_vector(), dtype=np.float32)
            driver_corrs.append(_driver_sector_correction(dv, sv, track_id))
        x_t = torch.from_numpy(np.stack(xs))
        sec_prior_arr = np.stack(sec_priors)
        resp_prior_arr = np.stack(resp_priors)
        driver_corr_arr = np.stack(driver_corrs)
        scales = np.asarray(RESPONSE_SCALES, dtype=np.float32)
        self.eval()
        with torch.no_grad():
            sec_res, resp_res = self.forward(x_t)
        sec_res = sec_res.numpy() * SECTOR_SCALE + sec_prior_arr + driver_corr_arr
        resp_res = resp_res.numpy() * scales + resp_prior_arr
        out: list[dict[str, Any]] = []
        for s_row, r_row in zip(sec_res, resp_res, strict=True):
            sectors = [max(0.01, float(s)) for s in s_row]
            lap_time = float(sum(sectors))
            responses = {
                name: float(v) for name, v in zip(RESPONSE_NAMES, r_row, strict=True)
            }
            out.append(
                {
                    "lap_time": lap_time,
                    "sectors": sectors,
                    "responses": responses,
                    "model_version": MODEL_VERSION,
                }
            )
        return out

    # --- 存取 ---------------------------------------------------------------
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """返回可序列化的模型状态 (含版本号与 torch state_dict)."""
        return {
            "model_version": MODEL_VERSION,
            "input_dim": INPUT_DIM,
            "state_dict": super().state_dict(*args, **kwargs),
        }

    def load_state_dict(
        self, d: dict[str, Any], strict: bool = True
    ) -> None:
        """从 :meth:`state_dict` 返回值恢复权重."""
        sd = d.get("state_dict", d)
        super().load_state_dict(sd, strict=strict)

    def save(self, path: str | Path) -> None:
        """保存到 ``.pt`` 文件 (创建父目录)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    @classmethod
    def load(cls, path: str | Path) -> SurrogateModel:
        """从 ``.pt`` 文件加载; 文件损坏抛出对应异常."""
        model = cls()
        d = torch.load(path, weights_only=True)
        model.load_state_dict(d)
        return model


# --- 默认模型 (lru_cache) ---------------------------------------------------
def default_model_path() -> Path:
    """返回默认权重路径: ``{data_dir}/models/segment_surrogate.pt``."""
    return Path(get_settings().data_dir) / "models" / "segment_surrogate.pt"


@lru_cache(maxsize=1)
def _get_default_model() -> SurrogateModel:
    """返回缓存的默认模型.

    若磁盘存在权重文件则加载; 否则返回未训练模型 (返回先验).
    加载失败时回退到未训练模型, 保证 API 永远可用.
    """
    model = SurrogateModel()
    path = default_model_path()
    if path.exists():
        try:
            d = torch.load(path, weights_only=True)
            model.load_state_dict(d)
        except (OSError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError):
            # 权限/格式/键不匹配, 或 weights_only 拒绝不受信任的 pickle
            # -> 回退到未训练先验, 不阻断推理
            model = SurrogateModel()
    return model


def reset_default_model_cache() -> None:
    """清空默认模型缓存 (训练保存后调用, 使后续推理加载新权重)."""
    _get_default_model.cache_clear()


def predict_lap_time(
    setup: CarSetup,
    track_id: str,
    driver_profile: Any = None,
) -> float:
    """模块级便捷函数: 使用缓存默认模型预测单圈圈速 (秒).

    ``from f1opt.model.surrogate import predict_lap_time`` 即可直接调用,
    即使未训练也返回合理先验. 签名与 Iter-01 保持向后兼容.
    """
    return float(_get_default_model().predict_lap_time(setup, track_id, driver_profile))


def predict_full(
    setup: CarSetup,
    track_id: str,
    driver_profile: Any = None,
) -> dict[str, Any]:
    """模块级便捷函数: 使用缓存默认模型返回富预测字典.

    返回结构与 :meth:`SurrogateModel.predict` 一致 (圈速 / 分段 / 响应 / 版本).
    """
    return _get_default_model().predict(setup, track_id, driver_profile)


# --- predict_lap_time 缓存层 (perf) ----------------------------------------
# 手动 LRU 缓存 (OrderedDict) + 命中/未命中计数, 避免 ``functools.lru_cache``
# 在带计数包装时的内省/装饰器嵌套问题. 键 = (setup_vector_tuple, track_id,
# driver_vector_tuple), 全部 hashable (setup 向量需转 tuple). 不改变
# :func:`predict_lap_time` / :func:`predict_full` / :class:`SurrogateModel` 签名.
_PREDICT_CACHE_MAXSIZE = 512
_predict_cache: OrderedDict[
    tuple[tuple[float, ...], str, tuple[float, ...]], float
] = OrderedDict()
_PREDICT_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}


def predict_lap_time_cached(
    setup_vector_tuple: tuple[float, ...],
    track_id: str,
    driver_vector_tuple: tuple[float, ...],
) -> float:
    """LRU 缓存版 :func:`predict_lap_time`, 键为 hashable 元组.

    调用方需把 ``CarSetup.to_vector()`` 与 ``driver_vector(...)`` 转成 tuple
    (setup 向量必须转 tuple 才可哈希; numpy 标量亦会被归一为纯 Python float
    以保证稳定哈希). 内部把 setup tuple 经 ``CarSetup.from_vector`` 反归一化回
    ``CarSetup`` 后调用 :func:`predict_lap_time`; driver tuple 作为 list 透传
    (空 tuple 等价于 None -> 全零车手向量). 命中时更新 LRU 顺序并累加
    ``hits``; 未命中时计算、写入, 并在超限 (``_PREDICT_CACHE_MAXSIZE``) 时按
    FIFO 淘汰最旧条目并累加 ``misses``.
    """
    # 归一为纯 Python float 以保证稳定哈希 (调用方可能传入 numpy 标量).
    sv = tuple(float(v) for v in setup_vector_tuple)
    dv = tuple(float(v) for v in driver_vector_tuple)
    key: tuple[tuple[float, ...], str, tuple[float, ...]] = (sv, track_id, dv)
    cached = _predict_cache.get(key)
    if cached is not None:
        _predict_cache.move_to_end(key)
        _PREDICT_CACHE_STATS["hits"] += 1
        return cached
    setup = CarSetup.from_vector(list(sv))
    driver: Any = list(dv) if dv else None
    value = float(predict_lap_time(setup, track_id, driver))
    _predict_cache[key] = value
    if len(_predict_cache) > _PREDICT_CACHE_MAXSIZE:
        _predict_cache.popitem(last=False)  # LRU 淘汰最旧
    _PREDICT_CACHE_STATS["misses"] += 1
    return value


def clear_predict_cache() -> None:
    """清空 :func:`predict_lap_time_cached` 的 LRU 缓存并重置命中/未命中计数."""
    _predict_cache.clear()
    _PREDICT_CACHE_STATS["hits"] = 0
    _PREDICT_CACHE_STATS["misses"] = 0


# ---------------------------------------------------------------------------
# Iter-127: Ensemble surrogate model (multi-seed averaging for variance reduction)
# ---------------------------------------------------------------------------
class EnsembleSurrogateModel(nn.Module):
    """Ensemble of N :class:`SurrogateModel` instances (Iter-127).

    Averages sector + response residuals across members, reducing prediction
    variance — especially in OOD / extrapolation regions where individual
    models may disagree. Drop-in replacement for :class:`SurrogateModel` in
    ``predict`` / ``predict_lap_time`` / ``predict_batch``: the optimizer,
    validation, and feedback modules can use an ensemble without code changes.

    Members are typically trained with different seeds (``seed=0,1,...,N-1``)
    so their weight initialisation and mini-batch shuffling differ, producing
    decorrelated residual errors that average out.

    Usage::

        from f1opt.model.train import train_ensemble
        ens = train_ensemble(n_members=3, base_seed=0, iterations=500,
                             n_samples=2000, save=False, log=False)
        lap = ens.predict_lap_time(setup, "silverstone")
    """

    def __init__(self, models: list[SurrogateModel]) -> None:
        super().__init__()
        if not models:
            raise ValueError("EnsembleSurrogateModel requires at least 1 model")
        self.models = nn.ModuleList(models)
        self.n_members = len(models)

    # ------------------------------------------------------------------ #
    # Forward (averaged residuals — training-side compatible)
    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average (sector_residual, response_residual) across all members."""
        sec_sum: torch.Tensor | None = None
        resp_sum: torch.Tensor | None = None
        for m in self.models:
            m.eval()
            sec, resp = m(x)
            sec_sum = sec if sec_sum is None else sec_sum + sec
            resp_sum = resp if resp_sum is None else resp_sum + resp
        assert sec_sum is not None and resp_sum is not None
        return sec_sum / self.n_members, resp_sum / self.n_members

    def residuals(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-side alias (same as :meth:`forward`)."""
        return self.forward(x)

    # ------------------------------------------------------------------ #
    # Inference (averaged final predictions)
    # ------------------------------------------------------------------ #
    def predict(
        self,
        setup: CarSetup,
        track_id: str,
        driver_profile: Any = None,
    ) -> dict[str, Any]:
        """Average :meth:`SurrogateModel.predict` across all members."""
        results = [m.predict(setup, track_id, driver_profile) for m in self.models]
        avg_lap = float(np.mean([r["lap_time"] for r in results]))
        avg_sectors = [
            float(np.mean([r["sectors"][i] for r in results]))
            for i in range(N_SECTORS)
        ]
        avg_responses = {
            name: float(np.mean([r["responses"][name] for r in results]))
            for name in RESPONSE_NAMES
        }
        return {
            "lap_time": avg_lap,
            "sectors": avg_sectors,
            "responses": avg_responses,
            "model_version": f"{MODEL_VERSION}-ensemble-{self.n_members}",
            "n_members": self.n_members,
        }

    def predict_lap_time(
        self,
        setup: CarSetup,
        track_id: str,
        driver_profile: Any = None,
    ) -> float:
        """Average lap time across all members."""
        return float(self.predict(setup, track_id, driver_profile)["lap_time"])

    # ------------------------------------------------------------------ #
    # Iter-133: Inference confidence estimation (ensemble variant)
    # ------------------------------------------------------------------ #
    def predict_with_confidence(
        self,
        setup: CarSetup,
        track_id: str,
        driver_profile: Any = None,
    ) -> dict[str, Any]:
        """Iter-133: 富预测 + 置信度估计 (ensemble 版, 含成员分歧因子).

        比 :meth:`SurrogateModel.predict_with_confidence` 多一个因子:

        - ``disagreement_penalty``: 成员间 lap_time 标准差 (秒) 映射到 [0, 0.3]
          惩罚. 成员分歧大 = OOD / 外推, 置信度降低. 0.3s std 对应最大惩罚.

        置信度公式::

            confidence = 1.0
                       - input_penalty
                       - residual_penalty
                       - disagreement_penalty   # ensemble-only

        单成员 ensemble (n_members=1) 的 disagreement_penalty == 0.0 (无分歧),
        退化为 :meth:`SurrogateModel.predict_with_confidence`.
        """
        result = self.predict(setup, track_id, driver_profile)
        x = build_input_vector(setup, track_id, driver_profile)

        # Factor 1: input-space OOD dims.
        ood_dims = int(np.sum((x < 0.0) | (x > 1.0)))
        input_penalty = min(0.5, 0.05 * ood_dims)

        # Factor 2: residual magnitude relative to sector prior.
        sec_prior = np.asarray(sector_priors(track_id, setup), dtype=np.float32)
        sectors = np.asarray(result["sectors"], dtype=np.float32)
        sec_residual = sectors - sec_prior
        safe_prior = np.maximum(sec_prior, 0.5)
        ratios = np.abs(sec_residual) / safe_prior
        max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 0.0
        residual_penalty = min(0.5, max(0.0, (max_ratio - 0.15) / 0.85) * 0.5)

        # Factor 3 (ensemble-only): member disagreement on lap_time.
        member_laps = [m.predict_lap_time(setup, track_id, driver_profile)
                       for m in self.models]
        lap_std = float(np.std(member_laps)) if len(member_laps) > 1 else 0.0
        # 0.3s std -> max penalty 0.3; scale linearly.
        disagreement_penalty = min(0.3, lap_std / 0.3 * 0.3)

        confidence = max(0.0, 1.0 - input_penalty - residual_penalty
                         - disagreement_penalty)
        if confidence >= 0.8:
            label = "high"
        elif confidence >= 0.5:
            label = "medium"
        else:
            label = "low"

        result["confidence"] = confidence
        result["confidence_factors"] = {
            "ood_input_dims": ood_dims,
            "max_residual_ratio": max_ratio,
            "input_penalty": input_penalty,
            "residual_penalty": residual_penalty,
            "disagreement_penalty": disagreement_penalty,
            "member_lap_std_s": lap_std,
            "label": label,
        }
        return result

    def predict_batch(
        self,
        items: list[tuple[CarSetup, str]] | list[tuple[CarSetup, str, Any]],
    ) -> list[dict[str, Any]]:
        """Average :meth:`SurrogateModel.predict_batch` across all members."""
        if not items:
            return []
        all_results = [m.predict_batch(items) for m in self.models]
        out: list[dict[str, Any]] = []
        for i in range(len(items)):
            r_list = [all_results[m][i] for m in range(self.n_members)]
            avg_lap = float(np.mean([r["lap_time"] for r in r_list]))
            avg_sectors = [
                float(np.mean([r["sectors"][j] for r in r_list]))
                for j in range(N_SECTORS)
            ]
            avg_responses = {
                name: float(np.mean([r["responses"][name] for r in r_list]))
                for name in RESPONSE_NAMES
            }
            out.append({
                "lap_time": avg_lap,
                "sectors": avg_sectors,
                "responses": avg_responses,
                "model_version": f"{MODEL_VERSION}-ensemble-{self.n_members}",
                "n_members": self.n_members,
            })
        return out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize all member models + ensemble metadata."""
        return {
            "model_version": f"{MODEL_VERSION}-ensemble-{self.n_members}",
            "n_members": self.n_members,
            "input_dim": INPUT_DIM,
            "members": [m.state_dict(*args, **kwargs) for m in self.models],
        }

    def load_state_dict(
        self, d: dict[str, Any], strict: bool = True
    ) -> None:
        """Restore member weights from :meth:`state_dict` output.

        The number of members in ``d`` must match ``self.n_members``. Each
        member's weights are loaded individually.
        """
        members = d.get("members", [])
        if len(members) != self.n_members:
            raise ValueError(
                f"Ensemble state_dict has {len(members)} members but "
                f"model has {self.n_members}"
            )
        for m, md in zip(self.models, members, strict=True):
            m.load_state_dict(md, strict=strict)

    def save(self, path: str | Path) -> None:
        """Save ensemble to ``.pt`` file (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    @classmethod
    def load(cls, path: str | Path) -> EnsembleSurrogateModel:
        """Load ensemble from ``.pt`` file."""
        d = torch.load(path, weights_only=True)
        n = d.get("n_members", 1)
        models = [SurrogateModel() for _ in range(n)]
        ens = cls(models)
        ens.load_state_dict(d)
        return ens
