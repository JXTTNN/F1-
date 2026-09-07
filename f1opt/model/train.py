"""分段代理模型训练脚本 (Iter-02 Task 2.2).

生成 *分段* 合成训练数据 (sector targets + response targets), 用手写的物理
启发式目标函数提供 *依赖 (setup, track, driver)* 的可学习映射, 全批梯度下降
训练 :class:`~f1opt.model.surrogate.SurrogateModel`, 并把权重保存到
``{data_dir}/models/segment_surrogate.pt``.

合成目标 (每段计时段, 秒)::

    sector_time[s] = base_sector(track)
        + aero_penalty(setup, track, s)
        + on_throttle_diff_penalty(setup, track, s)
        + off_throttle_diff_penalty(setup, track, s)
        + suspension_penalty(setup, track, s)
        + tyre_pressure_penalty(setup, track, s)
        + camber_penalty(setup, track, s)
        + driver_penalty(driver, track, s)
        + fuel_penalty / 3
        + noise

三段求和 = 圈速. response 目标 (7 项) 由 setup/track/sector 推得.
启发式刻意让每段的惩罚随 setup 变化, 使 DNN 必须学到 *分段 setup 敏感性*,
而不仅仅回归到 track_prior (Iter-01 的失效模式).

运行::

    python -m f1opt.model.train
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f1opt.config import get_settings
from f1opt.data.ea_f1_2026_benchmark import EA_F1_2026_LAP_TIME_BENCHMARK, resolve_track_id
from f1opt.data.sector_times import sector_times_for
from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.data.tracks import ALL_TRACKS, Track
from f1opt.model.surrogate import (
    AVG_SPEED,
    INPUT_DIM,
    MODEL_VERSION,
    RESPONSE_SCALES,
    SETUP_DIM,
    TRACK_CONTEXT_DIM,
    EnsembleSurrogateModel,
    SurrogateModel,
    build_input_vector,
    default_model_path,
    driver_vector,
    reset_default_model_cache,
    response_priors,
    sector_priors,
)

# --- 分段结构 ---------------------------------------------------------------
# 每段弯角数占比 (中段最多), 用于区分各段性质.
_SECTOR_CORNER_WEIGHTS: tuple[float, float, float] = (0.3, 0.4, 0.3)

# 车手 "激进度" 投影: 把 8 维车手画像投影成单个标量, 作为分段惩罚的强度因子.
#
# Iter-96 修复: 旧版权重 [0.5, 0.5, 0.4, -0.4, -0.5, 0.3, -0.4, -0.3] 有 5/8 维
# 符号与 DriverProfile 字段语义相反, 导致 AGGR 投影 (-0.97) < CONS (+0.99) —
# 即 "保守车手更激进", 与物理完全相反. 训练标签因此让 CONS 比 AGGR 更快
# (phys_offset CONS=-0.075s vs AGGR=-0.038s), DNN 学到错误方向.
#
# DriverProfile 8 维语义 (见 driver/profile.py):
#   0 brake_point_norm:  0=晚制动(激进), 1=早制动(保守) → 负权 (低=激进=快)
#   1 throttle_smoothness: 高=平顺(保守) → 负权
#   2 steer_smoothness:   高=平顺(保守) → 负权
#   3 corner_balance_pref: 中性 (过弯平衡偏好, 非激进度) → 0
#   4 aggression_score:    高=激进 → 正权
#   5 consistency_score:   高=一致=少失误=快 → 正权
#   6 ers_usage_intensity: 高=激进部署=快 → 正权
#   7 drs_usage_efficiency: 高=高效=快 → 正权
#
# 校验: AGGR [0.15,0.2,0.25,0.85,0.9,0.3,0.8,0.75] -> +0.67 (激进=快)
#        CONS [0.85,0.85,0.8,0.3,0.15,0.8,0.3,0.35]  -> -0.61 (保守=慢)
#        neutral [0.5]*8                              -> 0.0
_DRIVER_PROJ = np.array(
    [-0.5, -0.4, -0.3, 0.0, 0.5, 0.2, 0.3, 0.2], dtype=np.float32
)


def _driver_aggression(driver_vec: np.ndarray | None) -> float:
    """车手激进度标量 = 车手画像与固定投影的内积; None -> 0."""
    if driver_vec is None:
        return 0.0
    return float(np.dot(np.asarray(driver_vec, dtype=np.float32), _DRIVER_PROJ))


def _sector_corners(track: Track) -> list[float]:
    return [track.corners * w for w in _SECTOR_CORNER_WEIGHTS]


# --- 归一化辅助 -------------------------------------------------------------
def _setup_norms(setup: CarSetup) -> dict[str, float]:
    return {
        "rw": setup.rear_wing / 50.0,
        "fw": setup.front_wing / 50.0,
        "otd": (setup.on_throttle_diff - 50.0) / 50.0,
        "offtd": (setup.off_throttle_diff - 10.0) / 90.0,
        "fs": (setup.front_suspension - 1.0) / 49.0,
        "rs": (setup.rear_suspension - 1.0) / 49.0,
        "farb": (setup.front_arb - 1.0) / 49.0,
        "rarb": (setup.rear_arb - 1.0) / 49.0,
        "ftp": (setup.front_tyre_pressure - 21.0) / 7.0,
        "rtp": (setup.rear_tyre_pressure - 19.0) / 6.0,
        # camber: 0 = 最负 (最激进), 1 = 最不负 (最弱)
        "fcamb": (setup.front_camber - (-3.5)) / 1.0,
        "rcamb": (setup.rear_camber - (-2.0)) / 1.0,
        "frh": (setup.front_ride_height - 1.0) / 49.0,
        "rrh": (setup.rear_ride_height - 1.0) / 49.0,
        # toe: 0 = 最小前束, 1 = 最大 (Iter-88: 补全, 供 tyre_temp 物理标签使用)
        "ftoe": (setup.front_toe - 0.00) / 0.10,
        "rtoe": (setup.rear_toe - 0.10) / 0.20,
        # brake_bias: 0 = 最前制动 (45%), 1 = 最后制动 (55%), 0.5 = 最优
        "bb": (setup.front_brake_bias - 45.0) / 10.0,
        # brake_pressure: 0 = 最低 (80%), 1 = 最高 (100%)
        "bp": (setup.brake_pressure - 80.0) / 20.0,
        # fuel_load: 0 = 最轻 (5kg), 1 = 最重 (110kg)
        "fl": (setup.fuel_load - 5.0) / 105.0,
    }


def _base_sectors(track: Track) -> list[float]:
    """三段基线圈速 (秒), 对齐 surrogate.sector_priors 的 EA F1 2026 物理.

    24 赛道用 benchmark × 真实 sector 比例 (与模型先验一致, 残差只反映 setup 偏离);
    未知赛道回退到 length/speed 等分.
    """
    bench = EA_F1_2026_LAP_TIME_BENCHMARK.get(resolve_track_id(track.track_id))
    if bench is not None:
        try:
            sd = sector_times_for(track.track_id)
            total = sd.total_lap_time_s
            if total > 0.0:
                return [
                    bench * sd.s1_s / total,
                    bench * sd.s2_s / total,
                    bench * sd.s3_s / total,
                ]
        except ValueError:
            pass
        return [bench / 3.0, bench / 3.0, bench / 3.0]
    speed = AVG_SPEED.get(track.track_type, 70.0)
    b = track.length_m / speed
    return [b / 3.0, b / 3.0, b / 3.0]


# --- 分段启发式 -------------------------------------------------------------
def heuristic_sectors(
    setup: CarSetup, track: Track, driver_vec: np.ndarray | None = None
) -> list[float]:
    """分段计时段 (秒, 无噪声): base + 每段 setup/driver 惩罚.

    基线对齐 EA F1 2026 benchmark × 真实 sector 比例 (与 surrogate.sector_priors 一致),
    使残差 = setup 偏离基准, 不含基线误差. 每段惩罚均依赖 setup, 且强度随该段
    弯角密度变化, 使模型必须学到 *分段 setup 敏感性*.
    """
    base_secs = _base_sectors(track)
    sec_corners = _sector_corners(track)
    elev_norm = track.elevation_change_m / 120.0
    track_len_norm = track.length_m / 8000.0  # 归一化赛道长度
    n = _setup_norms(setup)
    rw, fw = n["rw"], n["fw"]
    otd, offtd = n["otd"], n["offtd"]
    stiffness = (n["fs"] + n["rs"]) / 2.0
    tp = (n["ftp"] + n["rtp"]) / 2.0
    camber = (n["fcamb"] + n["rcamb"]) / 2.0  # 0 = 最负
    bb = n["bb"]  # brake_bias norm, 0.5 = 最优
    fl = n["fl"]  # fuel_load norm
    aggression = _driver_aggression(driver_vec)
    # 燃油质量惩罚: 重燃油 × (1 + 赛道长度) — 长赛道每圈消耗更多时间.
    fuel_per_sec = setup.fuel_load * 0.035 / 3.0 * (1.0 + 0.3 * track_len_norm)

    sectors: list[float] = []
    for s in range(3):
        base_sec = base_secs[s]
        cf = sec_corners[s] / 10.0  # 弯角密度因子
        straight_f = 1.0 - cf  # 直道密度因子 (低 cf = 长直道)
        # Aero: 高尾翼增加计时, 在多弯段更显著 (基础阻力 + 弯角放大).
        aero = (rw * (0.08 + cf * 0.10) + fw * (0.03 + cf * 0.04)) * 1.2
        # on_throttle_diff: 最优 ~0.8 (90%), V 形惩罚, 牵引段更强.
        otd_pen = abs(otd - 0.8) * (0.3 + cf * 1.2) * 1.4
        # off_throttle_diff: 低压好 (进弯旋转), 制动段更强.
        offtd_pen = offtd * (0.2 + cf * 0.9) * 0.8
        # 悬挂硬度: 颠簸段 (高 elev) 硬悬受罚, 平滑段反而受益.
        susp_pen = stiffness * (elev_norm * 2.0 - 0.5) * (0.4 + cf * 0.6) * 1.0
        # 胎压: 高压伤牵引段.
        tp_pen = tp * (0.2 + cf * 1.0) * 0.6
        # 外倾: 更负 (低 camber norm) 利于过弯, 最优 ~0.3, 偏离 V 形惩罚.
        camb_pen = (camber - 0.3) ** 2 * (0.3 + cf * 1.5) * 1.2
        # 车手: 激进车手更快 (晚制动 + 带速入弯 + 极限横向 g), 且在多弯段
        # (高 cf) 收益更大 — 真实 F1 激进车手在弯角密集的中段 (sector 2) 圈速
        # 优势最显著. Iter-96 修复: 旧版 ``aggression * (cf*0.6 - 0.4) * 1.2``
        # 让激进车手在低 cf (直道) 段更快, 高 cf 段反而更慢 — 物理相反. 现改为
        # 高 cf 段收益更大, 与真实 F1 一致.
        #   aggression > 0 (激进) → drv_pen < 0 (快), 高 cf → 更负 (更快).
        drv_pen = -aggression * (0.2 + cf * 1.5) * 0.3
        # brake_bias: V 形惩罚偏离 0.5 (50%); 多弯段 (高 cf, 重制动) 更显著.
        bb_pen = abs(bb - 0.5) * (0.15 + cf * 1.0) * 0.6
        # ERS 部署 (用 fuel_load 近似, CarSetup 无 ers 字段): 长直道段 (低 cf)
        # ers 部署降圈速; 长赛道收益更大.
        ers_benefit = -fl * straight_f * 0.4 * (1.0 + 0.5 * track_len_norm)
        # 交互: aero×suspension — 高下压力 + 硬悬挂在多弯段过度侧倾.
        aero_susp_int = (rw + fw) * stiffness * cf * 0.2
        # 交互: tyre_pressure×camber — 高压 + 弱外倾 (高 camber norm) → 胎温升 + 抓地降.
        tp_camb_int = tp * camber * (0.15 + cf * 0.6) * 0.6
        sec_time = (
            base_sec
            + aero
            + otd_pen
            + offtd_pen
            + susp_pen
            + tp_pen
            + camb_pen
            + drv_pen
            + fuel_per_sec
            + bb_pen
            + ers_benefit
            + aero_susp_int
            + tp_camb_int
        )
        sectors.append(max(0.5, float(sec_time)))
    return sectors


def heuristic_responses(
    setup: CarSetup, track: Track, sectors: list[float]
) -> list[float]:
    """7 项 response 目标 (自然单位), 由 setup/track/sectors 推得."""
    n = _setup_norms(setup)
    lap = sum(sectors)
    speed_avg = track.length_m / lap  # m/s
    base_spd = AVG_SPEED.get(track.track_type, 70.0)
    # Iter-89: speed_max 补前翼 + 离地间隙依赖 (旧版仅 rear_wing, front_wing/
    # ride_height 方向 WRONG). 翼面↑ → 阻力 → 最高速↓; 离地↑ → 阻力 → ↓.
    speed_max = (
        base_spd * 1.5 + 5.0
        - n["rw"] * 12.0              # 后翼阻力主导
        - n["fw"] * 8.0               # 前翼阻力 (略弱于后翼)
        - ((n["frh"] + n["rrh"]) / 2.0) * 4.0  # 离地间隙 → 阻力
        + ((n["ftp"] + n["rtp"]) / 2.0) * 3.0  # 胎压 → 滚阻降 → 最高速↑
    )
    # Iter-89: slip_angle 完整物理模型 (旧版仅 otd+susp 2 项且 susp 符号错,
    # 13/18 方向 WRONG). 现补全 18 维 setup 物理依赖, 全部符号校正.
    # slip_angle = 弯中轮胎滑移角 (deg, ↑=更多滑移=更少抓地).
    slip_angle = (
        2.0  # baseline
        + ((n["ftp"] + n["rtp"]) / 2.0) * 0.8     # 胎压↑ → 接触面小 → 滑移↑
        + ((n["fcamb"] + n["rcamb"]) / 2.0) * 1.0  # 外倾值↑(更不负) → 弯中接触差 → 滑移↑
        - ((n["fw"] + n["rw"]) / 2.0) * 1.2        # 翼面↑ → 下压力 → 滑移↓
        - n["otd"] * 0.8                            # 油门锁止↑ → wheelspin少 → 滑移↓
        + n["offtd"] * 0.6                          # 收油锁止↑ → 进弯旋转 → 滑移↑
        + ((n["ftoe"] + n["rtoe"]) / 2.0) * 0.5    # 前束↑ → 滚阻 → 滑移微↑
        + ((n["fs"] + n["rs"]) / 2.0) * 1.0         # 悬挂硬↑ → 机械抓地降 → 滑移↑
        + ((n["farb"] + n["rarb"]) / 2.0) * 0.8    # 防倾杆硬↑ → 内侧轮侧滑 → 滑移↑
        + ((n["frh"] + n["rrh"]) / 2.0) * 0.6      # 离地间隙↑ → 载荷转移 → 滑移↑
        + n["bp"] * 0.5                             # 制动压力↑ → 锁死风险 → 滑移↑
        + (n["bb"] - 0.5) * 0.4                     # 前制动↑ → 前轮锁死 → 滑移↑
        + n["fl"] * 0.6                             # 燃油↑ → 惯性 → 滑移↑
    )
    # Iter-89: tyre_load_spread 补胎压依赖 (旧版仅 susp+arb, 胎压方向 WRONG).
    # 胎压↑ → 胎体更硬 → 载荷转移更直接 → spread↑.
    tyre_load_spread = (
        0.3
        + ((n["fs"] + n["rs"]) / 2.0) * 0.2
        + ((n["farb"] + n["rarb"]) / 2.0) * 0.15
        + ((n["ftp"] + n["rtp"]) / 2.0) * 0.1
    )
    rake = 0.3 + (n["rrh"] - n["frh"]) * 0.8
    # Iter-88: 完整物理 tyre_temp 模型 (EA F1 2026 garage 工程师经验).
    # 旧版仅含 tyre_pressure + front_camber 2 项, 导致 DNN 对其余 16 维学得
    # 错误方向 (Iter-87 审计: 10/18 WRONG). 现补全所有 18 维 setup 物理依赖:
    #   - 压力 ↑ → 接触面减小, 滑移摩擦生热多, 胎体硬散热差 → temp ↑
    #   - 外倾更负 (fcamb/rcamb ↓) → 弯中接触均匀, 生热多 → temp ↑
    #   - 翼面 ↑ → 下压力增加, 弯中滑移减小 → temp ↓
    #   - on_throttle_diff ↑ → 锁止多, 出弯 wheelspin 少 → temp ↓
    #   - off_throttle_diff ↑ → 进弯旋转, 后轮滑移 → temp ↑
    #   - 前束 ↑ → 滚动阻力增加 → temp ↑
    #   - 悬挂 ↑ (硬) → 机械抓地减小, 滑移多 → temp ↑
    #   - 防倾杆 ↑ (硬) → 内侧轮载荷增加, 侧滑 → temp ↑
    #   - 离地间隙 ↑ → 重心高, 载荷转移大, 滑移多 → temp ↑
    #   - 制动压力 ↑ → 制动热传胎 → temp ↑
    #   - 前制动分配 ↑ (向前) → 前轮热多, 后轮热少, 四轮均值略 ↑
    #   - 燃油 ↑ → 载荷大, 滚阻 + 滑移 → temp ↑
    # 每项系数 (°C / norm 满量程) 来自 EA F1 2026 garage 遥测幅度经验.
    tyre_temp = (
        90.0  # baseline 操作温度 (与 response_prior 一致)
        + ((n["ftp"] + n["rtp"]) / 2.0) * 8.0      # 胎压
        + (1.0 - n["fcamb"]) * 4.0                  # 前外倾 (更负 → temp ↑)
        + (1.0 - n["rcamb"]) * 3.0                  # 后外倾 (同物理, 略弱)
        - ((n["fw"] + n["rw"]) / 2.0) * 6.0         # 翼面 (下压力 → temp ↓)
        - n["otd"] * 3.0                            # 油门锁止 (出弯滑移少 → ↓)
        + n["offtd"] * 2.0                          # 收油锁止 (进弯旋转 → ↑)
        + ((n["ftoe"] + n["rtoe"]) / 2.0) * 4.0     # 前束 (滚阻 → ↑)
        + ((n["fs"] + n["rs"]) / 2.0) * 3.0         # 悬挂硬 (滑移 → ↑)
        + ((n["farb"] + n["rarb"]) / 2.0) * 2.0     # 防倾杆硬 (载荷转移 → ↑)
        + ((n["frh"] + n["rrh"]) / 2.0) * 2.0       # 离地间隙高 (载荷转移 → ↑)
        + n["bp"] * 4.0                             # 制动压力 (制动热 → ↑)
        + (n["bb"] - 0.5) * 1.5                     # 前制动多 (前热↑ 后热↓ 均值微↑)
        + n["fl"] * 5.0                             # 燃油 (载荷 → ↑)
    )
    # Iter-89: g_lat_max 补外倾依赖 (旧版仅 wings, rear_camber 方向 WRONG).
    # 外倾值↑ (更不负, fcamb/rcamb↑) → 弯中抓地降 → g_lat↓.
    g_lat_max = (
        2.5
        + (n["rw"] + n["fw"]) * 0.6
        - ((n["fcamb"] + n["rcamb"]) / 2.0) * 0.5
        + track.corners / 30.0 * 1.2
    )
    return [speed_avg, speed_max, slip_angle, tyre_load_spread, rake, tyre_temp, g_lat_max]


def heuristic_lap_time(setup: CarSetup, track: Track) -> float:
    """手写物理启发式圈速 (无噪声, 无车手), = 三段之和. 向后兼容入口."""
    return float(sum(heuristic_sectors(setup, track, None)))


# --- 车手画像 exemplars -----------------------------------------------------
def _driver_exemplars() -> list[np.ndarray]:
    """返回 3 个车手画像向量 (激进/保守/默认), 优先用 driver 子模块.

    若 ``f1opt.driver.profile`` 尚未发布, 回退到内置 exemplar, 保证训练闭环
    不依赖并行任务.
    """
    try:
        from f1opt.driver.profile import (  # type: ignore[import-not-found]
            AGGRESSIVE_PROFILE,
            CONSERVATIVE_PROFILE,
            DEFAULT_PROFILE,
        )
        exemplars = [
            driver_vector(AGGRESSIVE_PROFILE),
            driver_vector(CONSERVATIVE_PROFILE),
            driver_vector(DEFAULT_PROFILE),
        ]
        if any(np.any(v != 0) for v in exemplars):
            return exemplars
    except Exception:
        pass
    return [
        np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.5, 0.7, 0.8], dtype=np.float32),
        np.array([0.3, 0.4, 0.3, 0.5, 0.2, 0.6, 0.3, 0.3], dtype=np.float32),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    ]


def _random_driver(rng: np.random.Generator, exemplars: list[np.ndarray]) -> np.ndarray:
    """60% 取随机 exemplar, 40% 全随机 [0,1]^8, 保证分布多样."""
    if rng.random() < 0.6:
        return exemplars[int(rng.integers(0, len(exemplars)))].copy()
    return rng.random(8).astype(np.float32)


def _random_setup(rng: np.random.Generator) -> CarSetup:
    """在归一化空间均匀采样, 经 from_vector 对齐到合法档位."""
    vec = rng.random(SETUP_DIM).tolist()
    return CarSetup.from_vector(vec)


def _latin_hypercube_sample(
    rng: np.random.Generator,
    n: int,
    dim: int,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Latin Hypercube Sampling (LHS) — Iter-121.

    在 ``[lower, upper]^dim`` 空间生成 ``n`` 个样本, 保证每一维都被分层覆盖
    (每维分成 ``n`` 个等概率 stratum, 每个 stratum 恰好一个样本点). 相比独立
    均匀采样, LHS 消除了聚类空洞, 在高维空间 (23 维 setup) 显著改善覆盖率.

    算法:
    1. 每维生成 ``n`` 个 stratum 中心: ``[0.5/n, 1.5/n, ..., (n-0.5)/n]``
    2. 每维内加 ``Uniform(-0.5/n, +0.5/n)`` 微扰 (centered LHS)
    3. 每维独立 shuffle, 列拼接成 ``(n, dim)`` 矩阵
    4. 线性映射到 ``[lower, upper]``

    Args:
        rng: 随机数生成器.
        n: 样本数.
        dim: 维度.
        lower: 每维下界 (归一化空间).
        upper: 每维上界 (归一化空间).

    Returns:
        ``(n, dim)`` float64 数组, 值域 ``[lower, upper]``.
    """
    if n <= 0:
        return np.empty((0, dim), dtype=np.float64)
    # 每维的 stratum 中心 + 微扰
    centers = (np.arange(n, dtype=np.float64) + 0.5) / n  # [0.5/n, 1.5/n, ...]
    jitter = rng.uniform(-0.5 / n, 0.5 / n, size=(n, dim))
    samples = np.empty((n, dim), dtype=np.float64)
    for d in range(dim):
        col = centers + jitter[:, d]
        rng.shuffle(col)  # in-place shuffle per dimension
        samples[:, d] = col
    # 线性映射 [0,1] -> [lower, upper]
    if lower != 0.0 or upper != 1.0:
        samples = lower + samples * (upper - lower)
    return samples


def _lhs_setup_table(
    rng: np.random.Generator,
    n_uniform: int,
    n_tight: int,
    n_practice: int,
) -> list[tuple[str, np.ndarray]]:
    """Iter-121: 用 LHS 预生成 setup 采样表 (替代逐样本独立采样).

    生成 ``n_uniform + n_tight + n_practice`` 个采样计划, 每个计划是
    ``(stratum, vec)`` 对:
    - ``stratum="uniform"``: vec 是 ``[0,1]^21`` 的 LHS 样本 (全局覆盖)
    - ``stratum="tight"``: vec 是 ``[-1,1]^21`` 的 LHS 扰动 (正赛 ±3 档)
    - ``stratum="practice"``: vec 是 ``[-1,1]^21`` 的 LHS 扰动 (练习赛 ±8 档)

    扰动 vec 在 ``_realistic_setup_from_plan`` 中按 sigma 缩放后叠加到 per-track
    baseline 上, 保证 LHS 分层覆盖 perturbation 空间.

    Args:
        rng: 随机数生成器.
        n_uniform: 均匀采样数 (20%).
        n_tight: tight 扰动数 (30%).
        n_practice: practice 扰动数 (50%).

    Returns:
        采样计划列表, 每项 ``(stratum_name, vec)``; vec 形状 ``(19,)``.
    """
    plans: list[tuple[str, np.ndarray]] = []
    if n_uniform > 0:
        uniform = _latin_hypercube_sample(rng, n_uniform, SETUP_DIM, 0.0, 1.0)
        for i in range(n_uniform):
            plans.append(("uniform", uniform[i]))
    # tight / practice 扰动在 [-1, 1]^21 空间 LHS, 后续按 sigma 缩放
    if n_tight > 0:
        tight = _latin_hypercube_sample(rng, n_tight, SETUP_DIM, -1.0, 1.0)
        for i in range(n_tight):
            plans.append(("tight", tight[i]))
    if n_practice > 0:
        practice = _latin_hypercube_sample(rng, n_practice, SETUP_DIM, -1.0, 1.0)
        for i in range(n_practice):
            plans.append(("practice", practice[i]))
    # 打乱顺序, 使 uniform/tight/practice 在 track 循环中均匀混合
    rng.shuffle(plans)
    return plans


def _realistic_setup_from_plan(
    plan: tuple[str, np.ndarray],
    track_id: str,
) -> CarSetup:
    """Iter-121: 从 LHS 采样计划构造 ``CarSetup`` (替代 ``_realistic_random_setup``).

    Args:
        plan: ``(stratum, vec)`` 对. uniform: vec 直接是 [0,1]^21; tight/practice:
            vec 是 [-1,1]^21 扰动, 按 sigma 缩放后叠加到 track-type baseline.
        track_id: 赛道 ID.

    Returns:
        合法 ``CarSetup`` (经 from_vector 对齐档位).
    """
    from f1opt.data.tracks import TRACKS_BY_ID
    from f1opt.model.setup_physics_bridge import optimal_setup_for_track_type

    stratum, vec = plan
    if stratum == "uniform":
        return CarSetup.from_vector(vec.tolist())

    # tight / practice: 叠加到 per-track baseline
    track = TRACKS_BY_ID.get(track_id)
    if track is None:
        # 未知赛道: 把 [-1,1] 扰动映射回 [0,1] 作为回退 (避免负值)
        vec_norm = (np.asarray(vec) + 1.0) * 0.5
        return CarSetup.from_vector(np.clip(vec_norm, 0.0, 1.0).tolist())
    base = optimal_setup_for_track_type(track.track_type)
    base_vec = np.array(base.to_vector(), dtype=np.float64)
    sigma = 0.03 if stratum == "tight" else 0.08
    perturbed = np.clip(base_vec + vec * (3.0 * sigma), 0.0, 1.0)
    return CarSetup.from_vector(perturbed.tolist())


def _realistic_random_setup(rng: np.random.Generator, track_id: str) -> CarSetup:
    """在 *该赛道类型最优* 附近采样 (Iter-67), 模拟真实车队调教流程.

    真实 F1 车队不会在 garage 里均匀随机试 setup — 他们从 baseline (track-type
    最优) 出发, 每次练习只微调 ±5~15 档. 均匀随机会产生极端 setup (如 Monza
    上高下压力 = +36s 惩罚), 让 DNN 学习残差范围过大 (+0.6~+36s) 而难以收敛.

    Iter-93 修复: 旧版 sigma=0.15 (±15 档) 导致训练数据 setup_penalty 平均
    3.4s, baseline (penalty=0) 成为 OOD 点, DNN 在 baseline 上残差 +1.5s
    (期望 +0.2s). 新版分层采样:
    - 20% 均匀采样 (全局覆盖, DNN 学单调性)
    - 30% tight (sigma=0.03, ±3 档, 正赛范围, baseline 高密度覆盖)
    - 50% practice (sigma=0.08, ±8 档, 练习赛探索范围)
    baseline 附近 80% 样本, DNN 在 baseline 上不再 OOD.

    Args:
        rng: 随机数生成器.
        track_id: 赛道 ID (用于解析 track_type 找最优 setup).

    Returns:
        合法 ``CarSetup`` (经 from_vector 对齐档位).
    """
    # 延迟导入避免循环依赖
    from f1opt.data.tracks import TRACKS_BY_ID
    from f1opt.model.setup_physics_bridge import optimal_setup_for_track_type

    r = rng.random()
    if r < 0.2:
        # 20% 均匀采样 (全局覆盖)
        return _random_setup(rng)

    # 80% 围绕 track-type 最优分层高斯扰动
    track = TRACKS_BY_ID.get(track_id)
    if track is None:
        return _random_setup(rng)  # 未知赛道回退均匀
    base = optimal_setup_for_track_type(track.track_type)
    base_vec = np.array(base.to_vector(), dtype=np.float64)  # 归一化 [0,1]^21
    # Iter-93: 分层 sigma — 30% tight (正赛 ±3 档), 50% practice (练习赛 ±8 档)
    sigma = 0.03 if r < 0.5 else 0.08
    perturbed = np.clip(
        base_vec + rng.normal(0.0, sigma, size=SETUP_DIM),
        0.0, 1.0,
    )
    return CarSetup.from_vector(perturbed.tolist())


def generate_synthetic_dataset(
    n_samples: int = 5000,
    seed: int = 0,
    noise_std: float = 0.1,
) -> dict[str, Any]:
    """生成合成训练集: 随机 setup + 随机赛道 + 车手画像 + 分段/response 目标.

    返回 dict 含: setups, track_ids, driver_vecs (N,8), sector_targets (N,3),
    response_targets (N,7), lap_targets (N,).
    """
    rng = np.random.default_rng(seed)
    tracks = ALL_TRACKS
    exemplars = _driver_exemplars()
    setups: list[CarSetup] = []
    track_ids: list[str] = []
    driver_vecs = np.zeros((n_samples, 8), dtype=np.float32)
    sector_targets = np.zeros((n_samples, 3), dtype=np.float32)
    response_targets = np.zeros((n_samples, 7), dtype=np.float32)
    lap_targets = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        track = tracks[int(rng.integers(0, len(tracks)))]
        setup = _random_setup(rng)
        drv = _random_driver(rng, exemplars)
        sec = heuristic_sectors(setup, track, drv)
        resp = heuristic_responses(setup, track, sec)
        noise = rng.normal(0.0, noise_std, size=3).astype(np.float32)
        sec_noisy = [max(0.5, float(s + n)) for s, n in zip(sec, noise, strict=True)]
        setups.append(setup)
        track_ids.append(track.track_id)
        driver_vecs[i] = drv
        sector_targets[i] = sec_noisy
        response_targets[i] = resp
        lap_targets[i] = sum(sec_noisy)
    return {
        "setups": setups,
        "track_ids": track_ids,
        "driver_vecs": driver_vecs,
        "sector_targets": sector_targets,
        "response_targets": response_targets,
        "lap_targets": lap_targets,
    }


# --- 物理真值标签生成 (Iter-67) ----------------------------------------------
def _driver_physical_offset_s(driver_vec: np.ndarray | None) -> float:
    """把 8 维车手画像映射成物理圈速偏移 (秒, 正=慢).

    Iter-96 修复: 旧版用 ``mean(driver_vec)`` 作激进度代理, 但 DriverProfile 8 维
    语义不一致 (smoothness/consistency 高=保守但 "好", aggression 高=激进且 "快"),
    取均值让 CONS (高 smoothness/consistency) 均值 > AGGR, 反而 CONS 更快 — 物理
    完全相反. 现改用 :func:`_driver_aggression` 的同一投影 (语义校正后的 _DRIVER_PROJ),
    保证 per-sector 启发式与物理总时间的 driver 方向一致.

    量级 ±0.27s (AGGR=-0.27s 快, CONS=+0.24s 慢, delta=0.51s), 对应真实 F1 车手
    圈速差 (Hamilton vs 后段车手 ~0.5s).
    """
    if driver_vec is None:
        return 0.0
    # aggression > 0 = 激进 = 快 = 负偏移; < 0 = 保守 = 慢 = 正偏移.
    # 系数 0.4: 投影范围 ~[-1.5,+1.5] → ±0.6s, AGGR(+0.67)→-0.27s, CONS(-0.61)→+0.24s.
    return -_driver_aggression(driver_vec) * 0.4


def generate_physics_dataset(
    n_samples: int = 5000,
    seed: int = 0,
    noise_std: float = 0.1,
    use_lhs: bool = True,
) -> dict[str, Any]:
    """生成 *物理真值* 训练集 (Iter-67): 用 EA F1 2026 lap_simulator 评估任意 setup.

    与 :func:`generate_synthetic_dataset` 的区别 (用户 Iter-65 后方向 — 训练样本
    真实度/物理模型准确性):

    - **圈速总时间**: ``setup_lap_time(setup, track_id, driver_offset)`` — 来自
      EA F1 2026 物理引擎 (24 赛道 0.01% 精度), 含 benchmark + setup 物理惩罚
      + 燃油模型 + PU + 轮胎温度, 而非 ``heuristic_sectors`` 的纯算术和.
    - **分段比例**: 仍用 ``heuristic_sectors`` 的 *相对形状* (per-sector setup
      敏感性), 但缩放到物理总时间, 保留 DNN 学到 *分段 setup 敏感性* 的能力,
      同时保证 ``sum(sectors) == physics_lap_time``.
    - **车手画像**: 通过 ``driver_skill_offset_s`` 物理作用于圈速 (而非仅影响
      启发式分段惩罚), 与真实 F1 遥测一致 (AGGR 比 CONS 快 ~0.5s).
    - **response 目标**: 仍用 :func:`heuristic_responses` (bridge 不计算
      speed_max/slip_angle/...), 由 setup + track + sectors 推得.

    Iter-121 增强: ``use_lhs=True`` (默认) 用 Latin Hypercube Sampling 替代
    独立均匀/高斯采样, 在 23 维 setup 空间消除聚类空洞, 改善 OOD 泛化. 采样
    分层比例不变 (20% uniform / 30% tight / 50% practice), 但每层内部用 LHS
    分层覆盖. ``use_lhs=False`` 回退到 Iter-67 的逐样本独立采样 (向后兼容).

    返回结构与 :func:`generate_synthetic_dataset` 完全一致, 可直接喂给
    :func:`_build_tensors` 与 :func:`train`.

    性能: 5000 样本 × 27us/样本 (bridge + physics) ≈ 135ms.
    """
    # 延迟导入避免循环依赖 (setup_physics_bridge -> lap_simulator_2026 ->
    # tire_temperature/.../fuel_model; train.py 已导入 surrogate/tracks)
    from f1opt.model.setup_physics_bridge import setup_lap_time

    rng = np.random.default_rng(seed)
    tracks = ALL_TRACKS
    exemplars = _driver_exemplars()
    setups: list[CarSetup] = []
    track_ids: list[str] = []
    driver_vecs = np.zeros((n_samples, 8), dtype=np.float32)
    sector_targets = np.zeros((n_samples, 3), dtype=np.float32)
    response_targets = np.zeros((n_samples, 7), dtype=np.float32)
    lap_targets = np.zeros(n_samples, dtype=np.float32)

    # Iter-121: 预生成 LHS 采样表 (20% uniform / 30% tight / 50% practice).
    if use_lhs:
        n_uniform = int(round(n_samples * 0.20))
        n_tight = int(round(n_samples * 0.30))
        n_practice = n_samples - n_uniform - n_tight  # 余数给 practice
        setup_plans = _lhs_setup_table(rng, n_uniform, n_tight, n_practice)
        # 防御: 若 n_samples 极小导致 plans 不足, 补齐
        while len(setup_plans) < n_samples:
            setup_plans.append(("uniform", rng.random(SETUP_DIM)))
    else:
        setup_plans = None

    for i in range(n_samples):
        track = tracks[int(rng.integers(0, len(tracks)))]
        if use_lhs and setup_plans is not None:
            setup = _realistic_setup_from_plan(setup_plans[i], track.track_id)
        else:
            setup = _realistic_random_setup(rng, track.track_id)  # Iter-67: 现实采样
        drv = _random_driver(rng, exemplars)
        # 1. 物理真值圈速 (EA F1 2026 引擎, 含 setup 惩罚 + 燃油 + 车手偏移)
        drv_offset = _driver_physical_offset_s(drv)
        lap_physics = setup_lap_time(setup, track.track_id,
                                     driver_skill_offset_s=drv_offset)

        # 2. 启发式分段 (per-sector setup 敏感性形状)
        heur_secs = heuristic_sectors(setup, track, drv)
        heur_total = sum(heur_secs)
        if heur_total <= 0.0:
            heur_total = 1.0  # 兜底, 避免除零

        # 3. 缩放启发式分段到物理总时间 (保留 per-sector 形状, 总和 = 物理)
        sec_scaled = [lap_physics * (hs / heur_total) for hs in heur_secs]

        # 4. response 目标 (与 heuristic 一致, 由 setup + track + secs 推得)
        resp = heuristic_responses(setup, track, sec_scaled)

        # 5. 加噪声 (与 generate_synthetic_dataset 一致)
        noise = rng.normal(0.0, noise_std, size=3).astype(np.float32)
        sec_noisy = [max(0.5, float(s + n))
                     for s, n in zip(sec_scaled, noise, strict=True)]

        setups.append(setup)
        track_ids.append(track.track_id)
        driver_vecs[i] = drv
        sector_targets[i] = sec_noisy
        response_targets[i] = resp
        lap_targets[i] = sum(sec_noisy)

    return {
        "setups": setups,
        "track_ids": track_ids,
        "driver_vecs": driver_vecs,
        "sector_targets": sector_targets,
        "response_targets": response_targets,
        "lap_targets": lap_targets,
    }


def generate_dataset(
    n_samples: int = 5000,
    seed: int = 0,
    noise_std: float = 0.1,
    label_source: str = "physics",
    use_lhs: bool = True,
) -> dict[str, Any]:
    """统一入口: 按 ``label_source`` 选择物理真值或纯启发式标签.

    Args:
        label_source: ``"physics"`` (默认, Iter-67) 用 EA F1 2026 物理引擎
            生成圈速真值; ``"heuristic"`` 用纯启发式 (向后兼容 Iter-02~65).
        use_lhs: Iter-121 — 仅对 ``label_source="physics"`` 生效. ``True`` (默认)
            用 Latin Hypercube Sampling; ``False`` 回退逐样本独立采样.
    """
    if label_source == "physics":
        return generate_physics_dataset(n_samples, seed, noise_std, use_lhs=use_lhs)
    if label_source == "heuristic":
        return generate_synthetic_dataset(n_samples, seed, noise_std)
    raise ValueError(f"未知 label_source={label_source!r} (可选: 'physics' / 'heuristic')")


# --- 训练 -------------------------------------------------------------------

# Iter-123: 物理一致性 loss 的 driver 向量索引 (driver 在输入向量的最后 8 维).
# Iter-271: 不再硬编码 29/37 (SETUP_DIM 19->21 后 driver 段实际在 [31:39]);
# 硬编码 29 会覆盖末尾 2 维 track-context 且漏掉末尾 2 维 driver。
_DRIVER_VEC_START = SETUP_DIM + TRACK_CONTEXT_DIM  # 23 + 10 = 33
_DRIVER_VEC_END = INPUT_DIM  # 41


def _add_gradient_noise(model: SurrogateModel, std: float) -> None:
    """Iter-152: add Gaussian noise to gradients for improved generalization."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.add_(torch.randn_like(param.grad) * std)


class _LabelSmoothedMSELoss(torch.nn.Module):
    """Iter-160: MSE loss with regression label smoothing.

    Blends the standard MSE target with the batch mean, reducing the model's
    tendency to overfit to individual noisy labels. For a smoothing factor
    ``alpha``::

        smoothed_target = (1 - alpha) * target + alpha * batch_mean

    This is the regression analogue of classification label smoothing: it
    prevents the model from becoming overconfident on any single training
    example, which improves generalization — especially when labels contain
    observation noise (e.g. lap-time variability from non-setup factors).

    Args:
        alpha: Smoothing factor in [0, 1). 0.0 = pure MSE (no smoothing).
            Typical values: 0.05–0.15.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self.alpha = float(alpha)
        self._mse = torch.nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.alpha <= 0.0:
            return self._mse(pred, target)
        batch_mean = target.mean(dim=0, keepdim=True)
        smoothed = (1.0 - self.alpha) * target + self.alpha * batch_mean
        return self._mse(pred, smoothed)


def _physics_consistency_loss(
    model: SurrogateModel,
    x: torch.Tensor,
    aggr_vec: torch.Tensor,
    cons_vec: torch.Tensor,
    margin: float = 0.02,
) -> torch.Tensor:
    """Iter-123: 物理一致性 loss — driver×setup 方向约束.

    强制模型预测满足物理方向: 激进车手 (AGGR) 应比保守车手 (CONS) 更快
    (圈速更低), 对同一 (setup, track) 保持一致. 这通过 hinge loss 实现::

        loss = mean( max(0, lap_aggr - lap_cons + margin) )

    当 ``lap_aggr < lap_cons - margin`` (AGGR 明显更快) 时 loss=0; 否则
    penalize. margin 是容差 (0.02s 残差空间), 避免过强约束导致 DNN 收敛困难.

    实现方式: 对 batch 中每个样本, 替换 driver 向量为 AGGR / CONS exemplar,
    前向计算 sector 残差, 取三段之和作为 lap 残差, 比较 AGGR vs CONS 方向.

    Args:
        model: SurrogateModel (用于前向计算).
        x: 输入张量 ``(N, INPUT_DIM)``.
        aggr_vec: 激进车手 driver 向量 ``(8,)``.
        cons_vec: 保守车手 driver 向量 ``(8,)``.
        margin: hinge 容差 (秒, 残差空间).

    Returns:
        标量 loss 张量 (可反向传播).
    """
    # 创建 AGGR / CONS 变体: 替换 driver 部分 (indices 31:39)
    x_aggr = x.clone()
    x_cons = x.clone()
    x_aggr[:, _DRIVER_VEC_START:_DRIVER_VEC_END] = aggr_vec
    x_cons[:, _DRIVER_VEC_START:_DRIVER_VEC_END] = cons_vec

    # 前向计算 sector 残差 (N, 3)
    sec_aggr, _ = model(x_aggr)
    sec_cons, _ = model(x_cons)

    # lap 残差 = 三段之和 (N,)
    lap_aggr = sec_aggr.sum(dim=1)
    lap_cons = sec_cons.sum(dim=1)

    # Hinge loss: AGGR 应比 CONS 快 (lap_aggr < lap_cons)
    # 当 lap_aggr - lap_cons > -margin 时 penalize
    loss = torch.clamp(lap_aggr - lap_cons + margin, min=0.0).mean()
    return loss


def _build_tensors(
    data: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造训练张量: x, sector_target_norm, response_target_norm, lap_target, sec_priors."""
    setups = data["setups"]
    track_ids = data["track_ids"]
    driver_vecs = data["driver_vecs"]
    x = np.stack(
        [
            build_input_vector(s, t, driver_vecs[i])
            for i, (s, t) in enumerate(zip(setups, track_ids, strict=True))
        ]
    )
    sec_priors = np.stack(
        [sector_priors(t, s) for s, t in zip(setups, track_ids, strict=True)]
    )
    resp_priors = np.stack(
        [response_priors(t, s) for s, t in zip(setups, track_ids, strict=True)]
    )
    scales = np.asarray(RESPONSE_SCALES, dtype=np.float32)
    sector_targets = data["sector_targets"]
    response_targets = data["response_targets"]
    sec_norm = (sector_targets - sec_priors).astype(np.float32)  # 秒
    resp_norm = ((response_targets - resp_priors) / scales).astype(np.float32)
    return (
        torch.from_numpy(x),
        torch.from_numpy(sec_norm),
        torch.from_numpy(resp_norm),
        torch.from_numpy(sec_priors),
        torch.from_numpy(resp_priors),
    )


def _held_out_mae(
    model: SurrogateModel | EnsembleSurrogateModel, seed: int, n: int,
    noise_std: float = 0.0, label_source: str = "physics",
) -> tuple[float, float]:
    """在无噪声 held-out 集上计算 sector MAE 与 lap-time MAE.

    ``label_source`` 透传到 :func:`generate_dataset`, 默认 ``"physics"`` (Iter-67)
    以保证 held-out 评估与训练标签同源.
    """
    data = generate_dataset(n_samples=n, seed=seed, noise_std=noise_std,
                            label_source=label_source)
    sector_mae = 0.0
    lap_mae = 0.0
    for i, (s, t) in enumerate(zip(data["setups"], data["track_ids"], strict=True)):
        drv = data["driver_vecs"][i]
        pred = model.predict(s, t, drv)
        true_sec = data["sector_targets"][i]
        sector_mae += float(np.mean(np.abs(np.array(pred["sectors"]) - true_sec)))
        lap_mae += abs(pred["lap_time"] - data["lap_targets"][i])
    return sector_mae / n, lap_mae / n


# --- Iter-118: 详细 MAE 分解 (per-track / per-sector / per-driver) + OOD + 持久化 ---
# Driver exemplar 标签顺序与 _driver_exemplars() 一致 (AGGR/CONS/NEUT).
_DRIVER_EXEMPLAR_LABELS: tuple[str, ...] = ("AGGR", "CONS", "NEUT")


def _classify_driver(drv_vec: np.ndarray, exemplars: list[np.ndarray]) -> str:
    """把 driver 向量分类为最近的 exemplar (AGGR / CONS / NEUT).

    用 8 维欧氏距离最近的 exemplar 标签, 与训练集 driver 采样分布对齐.
    """
    if not exemplars:
        return "NEUT"
    dists = [float(np.sum((drv_vec - ex) ** 2)) for ex in exemplars]
    idx = int(np.argmin(dists))
    if 0 <= idx < len(_DRIVER_EXEMPLAR_LABELS):
        return _DRIVER_EXEMPLAR_LABELS[idx]
    return f"D{idx}"


def _detailed_mae_breakdown(
    model: SurrogateModel,
    seed: int,
    n: int = 200,
    noise_std: float = 0.0,
    label_source: str = "physics",
) -> dict[str, Any]:
    """在无噪声 held-out 集上计算 MAE 分解 (per-track / per-sector / per-driver).

    Iter-118: 替代 _held_out_mae 的简单聚合, 提供更细粒度的诊断信息.
    - per_track: 每赛道 (24) 的 sector MAE + lap MAE + 样本数.
    - per_sector: S1/S2/S3 各段 MAE (3 行).
    - per_driver: AGGR/CONS/NEUT 各类车手 MAE (3 行).

    Args:
        model: 已训练模型.
        seed: 评估集随机种子 (建议与训练种子不同, 避免数据泄露).
        n: 评估样本数 (默认 200, 比旧版 100 更稳定).
        noise_std: 0.0 (无噪声真值, 评估纯模型误差).
        label_source: 透传到 generate_dataset.

    Returns:
        dict 含: sector_mae, lap_mae, per_track, per_sector, per_driver.
    """
    data = generate_dataset(n_samples=n, seed=seed, noise_std=noise_std,
                            label_source=label_source)
    exemplars = _driver_exemplars()

    # per-track 累计
    per_track_sec: dict[str, float] = {}
    per_track_lap: dict[str, float] = {}
    per_track_n: dict[str, int] = {}

    # per-sector 累计
    per_sec_sum = [0.0, 0.0, 0.0]
    per_sec_n = [0, 0, 0]

    # per-driver 累计
    per_drv_sec: dict[str, float] = {}
    per_drv_lap: dict[str, float] = {}
    per_drv_n: dict[str, int] = {}

    sector_mae = 0.0
    lap_mae = 0.0

    for i, (s, t) in enumerate(zip(data["setups"], data["track_ids"], strict=True)):
        drv = data["driver_vecs"][i]
        pred = model.predict(s, t, drv)
        true_sec = np.asarray(data["sector_targets"][i], dtype=np.float64)
        sec_errs = np.abs(np.asarray(pred["sectors"], dtype=np.float64) - true_sec)
        sec_mae_i = float(np.mean(sec_errs))
        lap_err_i = abs(pred["lap_time"] - data["lap_targets"][i])

        sector_mae += sec_mae_i
        lap_mae += lap_err_i

        # per-track 累计
        per_track_sec[t] = per_track_sec.get(t, 0.0) + sec_mae_i
        per_track_lap[t] = per_track_lap.get(t, 0.0) + lap_err_i
        per_track_n[t] = per_track_n.get(t, 0) + 1

        # per-sector 累计
        for j in range(3):
            per_sec_sum[j] += float(sec_errs[j])
            per_sec_n[j] += 1

        # per-driver 累计
        drv_cat = _classify_driver(drv, exemplars)
        per_drv_sec[drv_cat] = per_drv_sec.get(drv_cat, 0.0) + sec_mae_i
        per_drv_lap[drv_cat] = per_drv_lap.get(drv_cat, 0.0) + lap_err_i
        per_drv_n[drv_cat] = per_drv_n.get(drv_cat, 0) + 1

    # 归一化 per-track
    per_track: dict[str, dict[str, float]] = {}
    for t in per_track_n:
        n_t = max(1, per_track_n[t])
        per_track[t] = {
            "sector_mae": per_track_sec[t] / n_t,
            "lap_mae": per_track_lap[t] / n_t,
            "n": per_track_n[t],
        }

    # 归一化 per-driver
    per_driver: dict[str, dict[str, float]] = {}
    for d in per_drv_n:
        n_d = max(1, per_drv_n[d])
        per_driver[d] = {
            "sector_mae": per_drv_sec[d] / n_d,
            "lap_mae": per_drv_lap[d] / n_d,
            "n": per_drv_n[d],
        }

    per_sector = {
        "s1": per_sec_sum[0] / max(1, per_sec_n[0]),
        "s2": per_sec_sum[1] / max(1, per_sec_n[1]),
        "s3": per_sec_sum[2] / max(1, per_sec_n[2]),
    }

    return {
        "n": n,
        "sector_mae": sector_mae / n,
        "lap_mae": lap_mae / n,
        "per_track": per_track,
        "per_sector": per_sector,
        "per_driver": per_driver,
    }


def _evaluate_ood(
    model: SurrogateModel,
) -> dict[str, Any]:
    """Iter-118: 在 OOD (out-of-distribution) 极端样本上评估 MAE.

    OOD 评估集构造: 24 赛道 × 4 极端 setup × 3 exemplar driver = 288 样本.
    极端 setup 取归一化 [0,1]^21 的角点 (全 0/全 1/全 0.25/全 0.75),
    距离训练分布中心 (track-type 最优 ±0.03/0.08) 远, 用于检测模型外推稳定性.

    圈速真值: 用 setup_lap_time (EA F1 2026 物理引擎) 与训练集同源.
    分段真值: heuristic_sectors 缩放到物理总时间 (与训练集一致).

    Returns:
        dict 含: n, sector_mae, lap_mae, per_setup_type (4 类).
    """
    from f1opt.model.setup_physics_bridge import setup_lap_time

    extreme_setups: list[tuple[str, np.ndarray]] = [
        ("all_min", np.zeros(SETUP_DIM, dtype=np.float32)),
        ("all_max", np.ones(SETUP_DIM, dtype=np.float32)),
        ("all_low", np.full(SETUP_DIM, 0.25, dtype=np.float32)),
        ("all_high", np.full(SETUP_DIM, 0.75, dtype=np.float32)),
    ]
    exemplars = _driver_exemplars()

    sector_mae = 0.0
    lap_mae = 0.0
    n = 0
    per_setup_sec: dict[str, float] = {name: 0.0 for name, _ in extreme_setups}
    per_setup_lap: dict[str, float] = {name: 0.0 for name, _ in extreme_setups}
    per_setup_n: dict[str, int] = {name: 0 for name, _ in extreme_setups}

    for setup_name, vec in extreme_setups:
        setup = CarSetup.from_vector(vec.tolist())
        for track in ALL_TRACKS:
            for drv in exemplars:
                # 物理真值 (与训练集一致)
                drv_offset = _driver_physical_offset_s(drv)
                lap_true = setup_lap_time(setup, track.track_id,
                                          driver_skill_offset_s=drv_offset)
                # 启发式分段缩放 (与训练集一致)
                heur_secs = heuristic_sectors(setup, track, drv)
                heur_total = sum(heur_secs)
                if heur_total <= 0.0:
                    heur_total = 1.0
                sec_true = [lap_true * (hs / heur_total) for hs in heur_secs]

                pred = model.predict(setup, track.track_id, drv)
                sec_errs = np.abs(np.asarray(pred["sectors"], dtype=np.float64)
                                  - np.asarray(sec_true, dtype=np.float64))
                sec_mae_i = float(np.mean(sec_errs))
                lap_err_i = abs(pred["lap_time"] - lap_true)

                sector_mae += sec_mae_i
                lap_mae += lap_err_i
                per_setup_sec[setup_name] += sec_mae_i
                per_setup_lap[setup_name] += lap_err_i
                per_setup_n[setup_name] += 1
                n += 1

    per_setup_type: dict[str, dict[str, float]] = {}
    for name, _ in extreme_setups:
        n_s = max(1, per_setup_n[name])
        per_setup_type[name] = {
            "sector_mae": per_setup_sec[name] / n_s,
            "lap_mae": per_setup_lap[name] / n_s,
            "n": per_setup_n[name],
        }

    return {
        "n": n,
        "sector_mae": sector_mae / max(1, n),
        "lap_mae": lap_mae / max(1, n),
        "per_setup_type": per_setup_type,
    }


def _persist_training_metrics(
    metrics: dict[str, Any],
    log_path: Path | None = None,
) -> Path:
    """Iter-118: 把训练指标追加到 JSONL 文件 (每行一条 JSON 记录).

    路径默认 ``{data_dir}/models/training_log.jsonl``. 追加模式, 每次训练写一行,
    便于后续分析 (如绘图 / 对比不同版本 / 监控回归).

    Args:
        metrics: 训练指标 dict (含 sector_mae, lap_mae, per_track, ...).
        log_path: 自定义路径; None 用默认路径.

    Returns:
        实际写入的路径.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    def _to_native(obj: Any) -> Any:
        """递归把 numpy 标量/数组转 Python 原生类型 (JSON 可序列化)."""
        if isinstance(obj, dict):
            return {str(k): _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    if log_path is None:
        log_path = Path(get_settings().data_dir) / "models" / "training_log.jsonl"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_version": MODEL_VERSION,
        **_to_native(metrics),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path


class _EMAWeights:
    """Iter-132: Exponential Moving Average of model parameters + buffers.

    Maintains detached shadow copies of every trainable parameter and every
    buffer (BatchNorm running_mean / running_var). After each
    ``optimizer.step()``, call :meth:`update` to advance the shadow:

        shadow[k] <- decay * shadow[k] + (1 - decay) * current[k]

    After training, call :meth:`apply_to` to load the EMA weights into the
    model (for held-out evaluation and saving). The original weights are
    backed up so :meth:`restore` can undo the swap if needed.

    EMA smooths the training trajectory and typically improves generalization
    on small datasets (Polyak averaging, used in most modern training
    pipelines). ``decay`` close to 1.0 (e.g. 0.999) gives a long-memory
    average; smaller values (e.g. 0.99) track the current weights more
    closely. BatchNorm running stats are also EMA-averaged, which is
    equivalent to a smoothed BN-stat estimate and is the standard practice.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self._shadow_params: dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }
        # Iter-132: only track *floating-point* buffers — BatchNorm's
        # ``num_batches_tracked`` is int64 and EMA (float math) on it would
        # raise "result type Float can't be cast to the desired output type
        # Long". running_mean / running_var are float and ARE tracked.
        self._shadow_buffers: dict[str, torch.Tensor] = {
            n: b.detach().clone()
            for n, b in model.named_buffers()
            if b.is_floating_point()
        }
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Advance shadow with current model weights (call after step)."""
        d = self.decay
        for name, param in model.named_parameters():
            shadow = self._shadow_params.get(name)
            if shadow is not None:
                shadow.mul_(d).add_(param.detach(), alpha=1.0 - d)
        for name, buf in model.named_buffers():
            if not buf.is_floating_point():
                continue
            shadow = self._shadow_buffers.get(name)
            if shadow is not None:
                shadow.mul_(d).add_(buf.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None: