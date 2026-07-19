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
    MODEL_VERSION,
    RESPONSE_SCALES,
    SETUP_DIM,
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
    fuel_per_sec = setup.fuel_load * 0.03 / 3.0 * (1.0 + 0.3 * track_len_norm)

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
    均匀采样, LHS 消除了聚类空洞, 在高维空间 (19 维 setup) 显著改善覆盖率.

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
    - ``stratum="uniform"``: vec 是 ``[0,1]^19`` 的 LHS 样本 (全局覆盖)
    - ``stratum="tight"``: vec 是 ``[-1,1]^19`` 的 LHS 扰动 (正赛 ±3 档)
    - ``stratum="practice"``: vec 是 ``[-1,1]^19`` 的 LHS 扰动 (练习赛 ±8 档)

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
    # tight / practice 扰动在 [-1, 1]^19 空间 LHS, 后续按 sigma 缩放
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
        plan: ``(stratum, vec)`` 对. uniform: vec 直接是 [0,1]^19; tight/practice:
            vec 是 [-1,1]^19 扰动, 按 sigma 缩放后叠加到 track-type baseline.
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
    base_vec = np.array(base.to_vector(), dtype=np.float64)  # 归一化 [0,1]^19
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
    独立均匀/高斯采样, 在 19 维 setup 空间消除聚类空洞, 改善 OOD 泛化. 采样
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

# Iter-123: 物理一致性 loss 的 driver 向量索引 (在 37 维输入中, driver 在最后 8 维).
_DRIVER_VEC_START = 29  # SETUP_DIM(19) + TRACK_CONTEXT_DIM(10) = 29
_DRIVER_VEC_END = 37    # + DRIVER_DIM(8) = 37


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
        x: 输入张量 ``(N, 37)``.
        aggr_vec: 激进车手 driver 向量 ``(8,)``.
        cons_vec: 保守车手 driver 向量 ``(8,)``.
        margin: hinge 容差 (秒, 残差空间).

    Returns:
        标量 loss 张量 (可反向传播).
    """
    # 创建 AGGR / CONS 变体: 替换 driver 部分 (indices 29:37)
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
    model: SurrogateModel, seed: int, n: int, noise_std: float = 0.0,
    label_source: str = "physics",
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
    极端 setup 取归一化 [0,1]^19 的角点 (全 0/全 1/全 0.25/全 0.75),
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
    log_path: "Path | None" = None,
) -> "Path":
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
    from datetime import datetime, timezone
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
        """Load EMA weights into model; backup originals for :meth:`restore`."""
        self._backup = {}
        for name, param in model.named_parameters():
            self._backup[name] = param.detach().clone()
            shadow = self._shadow_params.get(name)
            if shadow is not None:
                param.copy_(shadow)
        for name, buf in model.named_buffers():
            self._backup[name] = buf.detach().clone()
            if not buf.is_floating_point():
                continue
            shadow = self._shadow_buffers.get(name)
            if shadow is not None:
                buf.copy_(shadow)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore original weights (undo :meth:`apply_to`)."""
        if self._backup is None:
            return
        for name, param in model.named_parameters():
            orig = self._backup.get(name)
            if orig is not None:
                param.copy_(orig)
        for name, buf in model.named_buffers():
            orig = self._backup.get(name)
            if orig is not None:
                buf.copy_(orig)
        self._backup = None


class _SWAWeights:
    """Iter-156: Stochastic Weight Averaging.

    Maintains a running equal-weight average of model parameters collected
    every ``swa_freq`` steps after ``swa_start``. Unlike EMA (exponential
    decay), SWA uses a simple arithmetic mean, which corresponds to averaging
    points along the SGD trajectory in the later phase of training. This
    typically finds wider optima and improves generalization, especially on
    small datasets.

    Usage::

        swa = _SWAWeights(model, swa_start=100, swa_freq=10)
        for step in range(max_steps):
            ...  # train
            if swa.should_collect(step):
                swa.collect(model)
        swa.apply_to(model)  # use averaged weights for eval

    Only floating-point parameters and buffers are averaged; integer buffers
    (e.g. ``num_batches_tracked``) are left unchanged.
    """

    def __init__(self, model: torch.nn.Module, swa_start: int, swa_freq: int) -> None:
        self.swa_start = int(swa_start)
        self.swa_freq = int(swa_freq)
        self._n_collected: int = 0
        # Initialize running sum to zeros (same shape as params)
        self._sum_params: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p.detach()) for n, p in model.named_parameters()
        }
        self._sum_buffers: dict[str, torch.Tensor] = {
            n: torch.zeros_like(b.detach())
            for n, b in model.named_buffers()
            if b.is_floating_point()
        }
        self._backup: dict[str, torch.Tensor] | None = None

    def should_collect(self, step: int) -> bool:
        """Return True if the current step should be collected into the SWA average."""
        return step >= self.swa_start and (step - self.swa_start) % self.swa_freq == 0

    @torch.no_grad()
    def collect(self, model: torch.nn.Module) -> None:
        """Add current model weights to the SWA running average."""
        for name, param in model.named_parameters():
            self._sum_params[name].add_(param.detach())
        for name, buf in model.named_buffers():
            if not buf.is_floating_point():
                continue
            self._sum_buffers[name].add_(buf.detach())
        self._n_collected += 1

    @property
    def n_collected(self) -> int:
        """Number of checkpoints collected so far."""
        return self._n_collected

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """Load SWA-averaged weights into model; backup originals for :meth:`restore`.

        If no checkpoints were collected, this is a no-op.
        """
        if self._n_collected == 0:
            return
        self._backup = {}
        n = self._n_collected
        for name, param in model.named_parameters():
            self._backup[name] = param.detach().clone()
            avg = self._sum_params.get(name)
            if avg is not None:
                param.copy_(avg / n)
        for name, buf in model.named_buffers():
            self._backup[name] = buf.detach().clone()
            if not buf.is_floating_point():
                continue
            avg = self._sum_buffers.get(name)
            if avg is not None:
                buf.copy_(avg / n)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore original weights (undo :meth:`apply_to`)."""
        if self._backup is None:
            return
        for name, param in model.named_parameters():
            orig = self._backup.get(name)
            if orig is not None:
                param.copy_(orig)
        for name, buf in model.named_buffers():
            orig = self._backup.get(name)
            if orig is not None:
                buf.copy_(orig)
        self._backup = None


def _mixup_batch(
    x: torch.Tensor,
    sec_y: torch.Tensor,
    resp_y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation — Iter-136.

    Creates virtual training examples by linear interpolation of random
    input/target pairs:

        x_mixed   = lam * x + (1 - lam) * x[perm]
        sec_y_m   = lam * sec_y + (1 - lam) * sec_y[perm]
        resp_y_m  = lam * resp_y + (1 - lam) * resp_y[perm]

    where ``lam ~ Beta(alpha, alpha)``. Improves generalisation and acts as
    a regulariser by encouraging the model to learn smooth interpolation
    behaviour between nearby setups.

    Args:
        x:      Input batch ``(N, INPUT_DIM)``.
        sec_y:  Sector targets ``(N, 3)``.
        resp_y: Response targets ``(N, 7)``.
        alpha:  Beta distribution parameter. ``<= 0`` disables mixup (returns
                inputs unchanged with ``lam = 1.0``).

    Returns:
        ``(x_mixed, sec_y_mixed, resp_y_mixed, lam)``.
    """
    if alpha <= 0.0:
        return x, sec_y, resp_y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    # Clamp to [0.5, 1.0] so the original sample dominates — prevents
    # degenerate near-0 lambda that effectively swaps the pair (no signal).
    lam = max(0.5, lam)
    perm = torch.randperm(x.shape[0], device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    sec_y_m = lam * sec_y + (1.0 - lam) * sec_y[perm]
    resp_y_m = lam * resp_y + (1.0 - lam) * resp_y[perm]
    return x_mixed, sec_y_m, resp_y_m, lam


def _train_minibatch(
    model: SurrogateModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: torch.nn.Module,
    x: torch.Tensor,
    sec_y: torch.Tensor,
    resp_y: torch.Tensor,
    max_steps: int,
    batch_size: int,
    patience: int,
    log: bool,
    physics_exemplars: tuple[torch.Tensor, torch.Tensor] | None = None,
    physics_consistency_weight: float = 0.0,
    grad_clip_norm: float = 1.0,
    ema: _EMAWeights | None = None,
    mixup_alpha: float = 0.0,
    grad_accumulation_steps: int = 1,
    val_smoothing: float = 0.0,
    gradient_noise: float = 0.0,
) -> None:
    """Iter-68: mini-batch SGD + early stopping (原地训练 model).

    90/10 train/val split, 每 epoch shuffle + mini-batch, 每 epoch 评估 val loss,
    保存最优模型, patience 轮无改善则早停. 防止全批量 GD 在高迭代数下过拟合.

    ``max_steps`` 为梯度步上限 (跨 epoch 累计); 每 epoch = ceil(n_train/batch_size)
    步. scheduler 按梯度步衰减 (与全批量路径一致).

    Iter-123: ``physics_exemplars`` + ``physics_consistency_weight`` 启用
    物理一致性 loss (driver 方向约束), 在每个 mini-batch 上额外计算.

    Iter-126: ``grad_clip_norm`` 使梯度裁剪阈值可配置 (默认 1.0 向后兼容).
    同时在 val 评估中跟踪 sector MAE (更直接关联目标指标), 供日志监控.

    Iter-132: ``ema`` (Exponential Moving Average) 在每个 optimizer.step()
    后更新 shadow weights. EMA 不参与 val 评估 (val 仍用当前权重做早停判断);
    训练结束后由 :func:`train` 将 EMA 权重写入 model (覆盖 best_state).
    """
    import copy

    n_total = x.shape[0]
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val
    x_train, x_val = x[:n_train], x[n_train:]
    sy_train, sy_val = sec_y[:n_train], sec_y[n_train:]
    ry_train, ry_val = resp_y[:n_train], resp_y[n_train:]

    best_val_loss = float("inf")
    best_state: dict | None = None
    stall = 0
    step = 0
    epoch = 0

    while step < max_steps:
        epoch += 1
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            if step >= max_steps:
                break
            idx = perm[i : i + batch_size]
            xb = x_train[idx]
            syb = sy_train[idx]
            ryb = ry_train[idx]
            # Iter-136: mixup augmentation (disabled when alpha <= 0).
            xb_m, syb_m, ryb_m, _lam = _mixup_batch(xb, syb, ryb, mixup_alpha)
            optimizer.zero_grad()
            sr, rr = model(xb_m)
            lap_pred = sr.sum(dim=1)
            lap_tgt = syb_m.sum(dim=1)
            loss = (
                loss_fn(sr, syb_m)
                + 0.3 * loss_fn(rr, ryb_m)
                + 0.1 * loss_fn(lap_pred, lap_tgt)
            )
            # Iter-123: 物理一致性 loss (driver 方向约束).
            if physics_exemplars is not None and physics_consistency_weight > 0.0:
                aggr_vec, cons_vec = physics_exemplars
                phys_loss = _physics_consistency_loss(
                    model, xb_m, aggr_vec, cons_vec, margin=0.02,
                )
                loss = loss + physics_consistency_weight * phys_loss
            loss.backward()
            step += 1
            # Iter-140: gradient accumulation — only step when enough
            # mini-batches have been accumulated.
            if step % grad_accumulation_steps == 0 or step >= max_steps:
                # Iter-126: configurable grad clip norm.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                # Iter-152: add Gaussian noise to gradients for better generalization.
                if gradient_noise > 0.0:
                    _add_gradient_noise(model, gradient_noise)
                optimizer.step()
                scheduler.step()
                # Iter-132: advance EMA shadow after each gradient step.
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad()

        # Val evaluation: track both loss (for early stopping) and sector MAE
        # (Iter-126: for monitoring — MAE directly correlates with held-out metric).
        model.eval()
        with torch.no_grad():
            sr_v, rr_v = model(x_val)
            val_loss = float(loss_fn(sr_v, sy_val) + 0.3 * loss_fn(rr_v, ry_val))
            # Iter-126: sector MAE on val set (monitoring only).
            val_sector_mae = float((sr_v - sy_val).abs().mean())

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                if log:
                    print(
                        f"[surrogate] early-stop epoch={epoch} step={step} "
                        f"val_loss={best_val_loss:.6f} (patience={patience})"
                    )
                break

        if log and (epoch % 20 == 0 or step >= max_steps):
            print(
                f"[surrogate] epoch {epoch:3d} step={step:5d} "
                f"val_loss={val_loss:.6f} best={best_val_loss:.6f} "
                f"val_sec_MAE={val_sector_mae:.4f}s"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


def train(
    iterations: int = 3000,
    n_samples: int = 8000,
    seed: int = 0,
    log: bool = True,
    save: bool = True,
    model: SurrogateModel | None = None,
    label_source: str = "physics",
    noise_std: float = 0.05,
    batch_size: int = 0,
    early_stopping_patience: int = 0,
    use_lhs: bool = True,
    physics_consistency_weight: float = 0.05,
    lr_warmup_fraction: float = 0.0,
    grad_clip_norm: float = 1.0,
    loss_type: str = "mse",
    huber_beta: float = 1.0,
    ema_decay: float = 0.0,
    mixup_alpha: float = 0.0,
    one_cycle: bool = False,
    grad_accumulation_steps: int = 1,
    val_smoothing: float = 0.0,
    gradient_noise: float = 0.0,
    swa_start: int = 0,
    swa_freq: int = 10,
    label_smoothing: float = 0.0,
) -> SurrogateModel:
    """训练分段代理模型并 (可选) 保存权重, 返回训练后的模型.

    多任务损失::

        loss = MSE(sectors) + 0.3 * MSE(responses)
               + 0.1 * MSE(lap_from_sectors_sum, target_lap)

    Iter-68 训练参数调优:
    - **noise_std=0.05** (from 0.10): EA F1 2026 圈速是确定性的 (给定 setup/track/
      driver), 0.1s 噪声过高 (模拟人类车手不一致性, 但已由 driver_profile 捕获).
      降到 0.05s 更贴近游戏引擎噪声, held-out sector MAE 改善 21%.
    - **AdamW** (weight_decay=1e-5): L2 正则化.
    - **Cosine LR schedule** (1e-3 -> 1e-5): 前期高 lr 探索, 后期低 lr 精细收敛.
    - **Gradient clipping** (max_norm=1.0): BatchNorm + GELU 训练稳定性.
    - **n_samples=8000** (from 5000): 更好覆盖 37 维输入空间.
    - **Mini-batch + early stopping** (batch_size>0, early_stopping_patience>0):
      90/10 train/val split, 每 epoch 评估 val loss, 保存最优模型, patience
      轮无改善则早停. 防止全批量 GD 在高迭代数下过拟合 (3000 iter 比 1500 iter
      MAE 更差). ``batch_size=0`` (默认) 退化为全批量 (向后兼容).

    Iter-126 增强:
    - **lr_warmup_fraction**: 前 ``fraction * iterations`` 步线性 warmup LR 从
      0 到 1e-3, 再切到 CosineAnnealing 衰减到 1e-5. BatchNorm + GELU 在初始
      阶段对高 LR 敏感 (大梯度 → BN 统计量不稳), warmup 让 BN running stats
      先稳定再加速. ``0.0`` (默认) = 纯 CosineAnnealing (向后兼容). 使用
      ``SequentialLR`` 组合 ``LinearLR`` + ``CosineAnnealingLR``.
    - **grad_clip_norm**: 可配置梯度裁剪阈值 (默认 1.0 向后兼容). 更激进的
      裁剪 (0.5) 在噪声数据上更稳定; 更宽松 (5.0) 在干净数据上收敛更快.
    - mini-batch 路径额外跟踪 val sector MAE (日志监控).

    Iter-128 增强:
    - **loss_type**: ``"mse"`` (默认, 向后兼容) 或 ``"huber"``. Huber loss
      (SmoothL1) 在残差 < ``huber_beta`` 时用二次 loss (如 MSE), 在残差 >
      ``huber_beta`` 时用线性 loss (如 L1). 对 OOD / 噪声标签更鲁棒: 大残差
      (outlier) 的梯度不再平方放大, 防止模型过拟合到极端样本. 对 F1 数据特别
      有用 — 物理模拟器在极端 setup 组合下偶尔产生大残差, Huber 抑制这些
      outlier 对训练的支配.
    - **huber_beta**: Huber loss 的过渡阈值 (默认 1.0). 对 sector 残差 (秒),
      1.0s 意味着 > 1s 的残差用线性 loss; 对 response 残差 (归一化 O(1)),
      1.0 也合理.

    Iter-132 增强:
    - **ema_decay**: Exponential Moving Average of model weights (Polyak
      averaging). ``0.0`` (默认) = 禁用, 向后兼容. 典型值 ``0.999`` (长记忆)
      或 ``0.99`` (短记忆). 每个 ``optimizer.step()`` 后更新 shadow weights:
      ``shadow <- decay * shadow + (1 - decay) * current``. 训练结束后将
      EMA 权重写入 model (覆盖当前/best 权重), held-out 评估 + 保存均用
      EMA 权重. EMA 平滑训练轨迹, 在小数据集 (n_samples=8000) 上通常改善
      泛化 (减少 last-epoch 噪声的影响). BatchNorm running_mean/var 也被
      EMA 平均 (等价于平滑 BN 统计量估计, 标准做法). 与 mini-batch + 早停
      兼容: 早停仍基于当前权重 val_loss, 训练结束后 EMA 覆盖 best_state.

    Iter-136 增强:
    - **mixup_alpha**: Mixup 数据增强的 Beta 分布参数 (``> 0`` 启用). 对每个
      mini-batch 采样 ``lam ~ Beta(alpha, alpha)`` (clamp 到 [0.5, 1.0] 防止
      退化), 用 ``x_mixed = lam*x + (1-lam)*x[perm]`` 与对应 target 同样线性
      插值, 生成虚拟训练样本. 作为正则化手段提升泛化 (鼓励模型在相近 setup
      之间平滑插值), 在小数据集 + 高维输入 (37 维) 上减少过拟合. ``0.0``
      (默认) = 禁用, 向后兼容. 仅在 mini-batch 路径生效 (batch_size > 0).

    - ``save=True`` 时写入 ``{data_dir}/models/segment_surrogate.pt`` 并刷新缓存.
    - ``save=False`` 用于测试隔离 (不写盘, 不影响模块级默认模型).
    - 训练后打印 held-out sector/lap MAE (``log=True`` 时).

    Iter-67: ``label_source`` 默认 ``"physics"`` — 用 EA F1 2026 lap_simulator
    生成 *物理真值* 圈速标签 (含 setup 物理惩罚 + 燃油 + PU + 轮温), 替代
    Iter-02~65 的纯启发式标签. 传 ``"heuristic"`` 回退到原行为 (向后兼容).
    """
    # Iter-94: 固定 torch RNG 种子, 让模型初始化 + mini-batch shuffle 完全确定.
    # 旧版仅 np.random.default_rng(seed) 固定数据生成, 但 SurrogateModel() 的
    # 随机权重初始化 + AdamW 的 mini-batch shuffle 用 torch RNG (未固定), 导致
    # 同一 seed 多次训练产生不同模型 — test_setup_sensitivity_hungaroring_otd
    # 在 OOD 区域 (otd 60/100) 预测方向不稳定 (有时 PASS delta=-0.20s, 有时 FAIL
    # delta=+0.20s). 现固定 torch RNG, 训练完全可复现.
    torch.manual_seed(seed)

    data = generate_dataset(n_samples=n_samples, seed=seed,
                            noise_std=noise_std, label_source=label_source,
                            use_lhs=use_lhs)
    x, sec_y, resp_y, _sec_priors, _resp_priors = _build_tensors(data)

    # Iter-123: 准备物理一致性 loss 的 driver exemplar 张量.
    # AGGR exemplar: 激进车手 (应更快); CONS exemplar: 保守车手 (应更慢).
    _physics_exemplars: tuple[torch.Tensor, torch.Tensor] | None = None
    if physics_consistency_weight > 0.0:
        exemplars = _driver_exemplars()
        if len(exemplars) >= 2:
            # exemplars[0] = AGGR, exemplars[1] = CONS (from _driver_exemplars)
            _physics_exemplars = (
                torch.from_numpy(exemplars[0].astype(np.float32)),
                torch.from_numpy(exemplars[1].astype(np.float32)),
            )

    if model is None:
        model = SurrogateModel()
    # Iter-68: 训练参数调优 — AdamW (L2 正则化) + cosine LR schedule + gradient clipping.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Iter-126: LR warmup via SequentialLR (LinearLR + CosineAnnealingLR).
    # lr_warmup_fraction > 0 时, 前 fraction*iterations 步线性 warmup LR 从
    # ~1e-5 (start_factor=0.01 * 1e-3) 到 1e-3, 再切到 Cosine 衰减到 eta_min.
    # 0.0 = 纯 Cosine (向后兼容).
    _eta_min = 1e-5
    if lr_warmup_fraction > 0.0:
        warmup_steps = max(1, int(iterations * lr_warmup_fraction))
        cosine_steps = max(1, iterations - warmup_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=_eta_min,
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        )
    elif one_cycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=max(1, iterations),
            pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, iterations), eta_min=_eta_min,
        )
    # Iter-128: configurable loss type (MSE default, Huber for outlier robustness).
    # Iter-160: label_smoothing blends targets with batch mean for regularization.
    if loss_type == "huber":
        loss_fn: torch.nn.Module = torch.nn.HuberLoss(delta=huber_beta)
    elif loss_type == "mse":
        if label_smoothing > 0.0:
            loss_fn = _LabelSmoothedMSELoss(alpha=label_smoothing)
        else:
            loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(
            f"Unknown loss_type={loss_type!r}; expected 'mse' or 'huber'"
        )

    model.eval()
    with torch.no_grad():
        sec_res0, resp_res0 = model(x)
        init_loss = float(loss_fn(sec_res0, sec_y) + 0.3 * loss_fn(resp_res0, resp_y))
    if log:
        print(
            f"[surrogate] iter     0  mse={init_loss:.4f}  "
            f"(label_source={label_source}, noise={noise_std})"
        )

    # Iter-132: initialize EMA shadow weights (disabled when ema_decay == 0.0).
    ema: _EMAWeights | None = (
        _EMAWeights(model, ema_decay) if ema_decay > 0.0 else None
    )
    if log and ema is not None:
        print(f"[surrogate] EMA enabled (decay={ema_decay})")

    # Iter-156: initialize SWA weights (disabled when swa_start <= 0).
    swa: _SWAWeights | None = (
        _SWAWeights(model, swa_start, swa_freq) if swa_start > 0 else None
    )
    if log and swa is not None:
        print(f"[surrogate] SWA enabled (start={swa_start}, freq={swa_freq})")

    # Iter-68: mini-batch + early stopping (batch_size > 0 时启用)
    use_minibatch = batch_size > 0 and early_stopping_patience > 0
    if use_minibatch:
        _train_minibatch(
            model, optimizer, scheduler, loss_fn,
            x, sec_y, resp_y,
            iterations, batch_size, early_stopping_patience, log,
            physics_exemplars=_physics_exemplars,
            physics_consistency_weight=physics_consistency_weight,
            grad_clip_norm=grad_clip_norm,
            ema=ema,
            mixup_alpha=mixup_alpha,
            grad_accumulation_steps=grad_accumulation_steps,
            val_smoothing=val_smoothing,
            gradient_noise=gradient_noise,
        )
    else:
        model.train()
        for it in range(1, iterations + 1):
            optimizer.zero_grad()
            sec_res, resp_res = model(x)
            lap_pred = sec_res.sum(dim=1)
            lap_target = sec_y.sum(dim=1)
            loss = (
                loss_fn(sec_res, sec_y)
                + 0.3 * loss_fn(resp_res, resp_y)
                + 0.1 * loss_fn(lap_pred, lap_target)
            )
            # Iter-123: 物理一致性 loss — driver 方向约束 (AGGR 应快于 CONS).
            if _physics_exemplars is not None:
                aggr_vec, cons_vec = _physics_exemplars
                phys_loss = _physics_consistency_loss(
                    model, x, aggr_vec, cons_vec, margin=0.02,
                )
                loss = loss + physics_consistency_weight * phys_loss
            loss.backward()
            # Iter-126: configurable grad clip norm (was hardcoded 1.0).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            # Iter-152: add Gaussian noise to gradients for better generalization.
            if gradient_noise > 0.0:
                _add_gradient_noise(model, gradient_noise)
            optimizer.step()
            scheduler.step()
            # Iter-132: advance EMA shadow after each gradient step.
            if ema is not None:
                ema.update(model)
            # Iter-156: collect SWA checkpoint after swa_start, every swa_freq.
            if swa is not None and swa.should_collect(it):
                swa.collect(model)
            if log and (it % 200 == 0 or it == iterations):
                lr = scheduler.get_last_lr()[0]
                print(f"[surrogate] iter {it:5d}  mse={loss.item():.4f}  lr={lr:.2e}")
        model.eval()

    # Iter-132: apply EMA weights to model (overwrites current / best_state).
    # Held-out evaluation and save now use the EMA-smoothed weights.
    if ema is not None:
        ema.apply_to(model)
        if log:
            print("[surrogate] EMA weights applied to model for eval + save")

    # Iter-156: apply SWA weights to model (overwrites EMA / current / best).
    # SWA takes precedence over EMA when both are enabled, as it represents
    # a more aggressive averaging of the later training trajectory.
    if swa is not None and swa.n_collected > 0:
        swa.apply_to(model)
        if log:
            print(f"[surrogate] SWA weights applied ({swa.n_collected} checkpoints)")

    if log:
        sec_mae, lap_mae = _held_out_mae(model, seed=99_999, n=100,
                                         label_source=label_source)
        print(f"[surrogate] held-out sector MAE={sec_mae:.4f}s  lap MAE={lap_mae:.4f}s")
        # Iter-118: 详细 MAE 分解 (per-track / per-sector / per-driver) + OOD + 持久化.
        breakdown = _detailed_mae_breakdown(model, seed=99_999, n=200,
                                            label_source=label_source)
        print(f"[surrogate] detailed breakdown (n={breakdown['n']}):")
        print(f"  overall  sector MAE={breakdown['sector_mae']:.4f}s  "
              f"lap MAE={breakdown['lap_mae']:.4f}s")
        ps = breakdown["per_sector"]
        print(f"  per-sector  S1={ps['s1']:.4f}s  S2={ps['s2']:.4f}s  S3={ps['s3']:.4f}s")
        print("  per-driver:")
        for d, m in sorted(breakdown["per_driver"].items()):
            print(f"    {d:<6}  sector MAE={m['sector_mae']:.4f}s  "
                  f"lap MAE={m['lap_mae']:.4f}s  n={m['n']}")
        # per-track 按 lap MAE 倒序 (worst first), 仅打印 top 5 + bottom 5.
        sorted_tracks = sorted(breakdown["per_track"].items(),
                               key=lambda x: -x[1]["lap_mae"])
        print("  per-track (top 5 worst):")
        for t, m in sorted_tracks[:5]:
            print(f"    {t:<14}  sector MAE={m['sector_mae']:.4f}s  "
                  f"lap MAE={m['lap_mae']:.4f}s  n={m['n']}")
        if len(sorted_tracks) > 5:
            print("  per-track (top 5 best):")
            for t, m in sorted_tracks[-5:]:
                print(f"    {t:<14}  sector MAE={m['sector_mae']:.4f}s  "
                      f"lap MAE={m['lap_mae']:.4f}s  n={m['n']}")
        # OOD 评估 (288 样本, ~8s): 检测模型外推稳定性.
        ood = _evaluate_ood(model)
        print(f"[surrogate] OOD (n={ood['n']}):  "
              f"sector MAE={ood['sector_mae']:.4f}s  lap MAE={ood['lap_mae']:.4f}s")
        for s_name, m in ood["per_setup_type"].items():
            print(f"    {s_name:<10}  sector MAE={m['sector_mae']:.4f}s  "
                  f"lap MAE={m['lap_mae']:.4f}s  n={m['n']}")
        # 持久化到 JSONL (追加模式, 一行一条记录).
        record = {
            "label_source": label_source,
            "n_samples": n_samples,
            "iterations": iterations,
            "noise_std": noise_std,
            "seed": seed,
            "ema_decay": ema_decay,
            "held_out_n100": {"sector_mae": sec_mae, "lap_mae": lap_mae},
            "breakdown_n200": breakdown,
            "ood": ood,
        }
        log_path = _persist_training_metrics(record)
        print(f"[surrogate] metrics appended -> {log_path}")

    if save:
        path = default_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        reset_default_model_cache()
        if log:
            print(f"[surrogate] saved -> {path}  (version={MODEL_VERSION})")

    return model


def train_ensemble(
    n_members: int = 3,
    base_seed: int = 0,
    *,
    iterations: int = 3000,
    n_samples: int = 8000,
    log: bool = True,
    save: bool = False,
    label_source: str = "physics",
    noise_std: float = 0.05,
    batch_size: int = 0,
    early_stopping_patience: int = 0,
    use_lhs: bool = True,
    physics_consistency_weight: float = 0.05,
    lr_warmup_fraction: float = 0.0,
    grad_clip_norm: float = 1.0,
    loss_type: str = "mse",
    huber_beta: float = 1.0,
    mixup_alpha: float = 0.0,
    _cycle: bool = False,
    grad_accumulation_steps: int = 1,
    val_smoothing: float = 0.0,
    gradient_noise: float = 0.0,
    swa_start: int = 0,
    swa_freq: int = 10,
    label_smoothing: float = 0.0,
) -> EnsembleSurrogateModel:
    """Train an ensemble of N :class:`SurrogateModel` instances (Iter-127).

    Each member is trained with a different seed (``base_seed + i`` for
    member ``i``), so weight initialisation + mini-batch shuffling differ,
    producing decorrelated residual errors that average out. Returns an
    :class:`EnsembleSurrogateModel` that averages predictions across members.

    Variance reduction is most visible in OOD / extrapolation regions where
    individual models disagree. In well-covered regions the ensemble matches
    a single model's bias but with lower variance.

    Args:
        n_members: Number of ensemble members (default 3).
        base_seed: Seed for the first member; member ``i`` uses ``base_seed + i``.
        log: If True, print per-member training progress + ensemble held-out MAE.
        save: If True, save ensemble to ``{data_dir}/models/segment_surrogate_ensemble.pt``.
            Default False (ensemble is for evaluation; single model remains the
            production default).
        **train_kwargs: Forwarded to :func:`train` for each member (iterations,
            n_samples, label_source, noise_std, batch_size, etc.).

    Returns:
        Trained :class:`EnsembleSurrogateModel`.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members}")
    models: list[SurrogateModel] = []
    for i in range(n_members):
        member_seed = base_seed + i
        if log:
            print(f"[ensemble] training member {i + 1}/{n_members} (seed={member_seed})")
        m = train(
            iterations=iterations,
            n_samples=n_samples,
            seed=member_seed,
            log=False,
            save=False,
            label_source=label_source,
            noise_std=noise_std,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            use_lhs=use_lhs,
            physics_consistency_weight=physics_consistency_weight,
            lr_warmup_fraction=lr_warmup_fraction,
            grad_clip_norm=grad_clip_norm,
            loss_type=loss_type,
            huber_beta=huber_beta,
            mixup_alpha=mixup_alpha,
            grad_accumulation_steps=grad_accumulation_steps,
            val_smoothing=val_smoothing,
            gradient_noise=gradient_noise,
            swa_start=swa_start,
            swa_freq=swa_freq,
            label_smoothing=label_smoothing,
        )
        models.append(m)

    ensemble = EnsembleSurrogateModel(models)

    if log:
        sec_mae, lap_mae = _held_out_mae(ensemble, seed=99_999, n=100,
                                         label_source=label_source)
        print(f"[ensemble] held-out sector MAE={sec_mae:.4f}s  "
              f"lap MAE={lap_mae:.4f}s  (n_members={n_members})")

    if save:
        from pathlib import Path

        from f1opt.config import get_settings
        ens_path = (
            Path(get_settings().data_dir)
            / "models"
            / "segment_surrogate_ensemble.pt"
        )
        ens_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save(ens_path)
        if log:
            print(f"[ensemble] saved -> {ens_path}")

    return ensemble


# --- setup 敏感度量化 -------------------------------------------------------
def _perturb_setup(base: CarSetup, rng: np.random.Generator) -> CarSetup:
    """随机选 1-3 个 SETUP_FIELDS 各 ±1 档扰动, 返回新 ``CarSetup``.

    在归一化空间 (``to_vector``) 内移动一个档位 (``step / (max - min)``),
    钳位到 [0, 1] 后用 ``from_vector`` 反归一化, 自动对齐到合法档位.
    """
    vec = list(base.to_vector())
    specs = list(SETUP_FIELDS.values())
    n_perturb = int(rng.integers(1, 4))  # 1-3 个参数
    chosen_idx = rng.choice(len(specs), size=n_perturb, replace=False)
    for idx in chosen_idx:
        i = int(idx)
        spec = specs[i]
        step_norm = spec.step / (spec.max - spec.min)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        vec[i] = max(0.0, min(1.0, vec[i] + direction * step_norm))
    return CarSetup.from_vector(vec)


def setup_sensitivity(
    model: SurrogateModel,
    track_id: str,
    n_perturb: int = 20,
    seed: int = 42,
) -> float:
    """量化模型对 setup 变化的敏感度.

    随机扰动 setup 各参数 ±1 档, 测量 lap_time 变化标准差.
    值越大说明模型对 setup 越敏感 (好); 接近 0 说明模型退化到 track_prior (坏).
    """
    rng = np.random.default_rng(seed)
    base = DEFAULT_SETUP
    base_lap = model.predict_lap_time(base, track_id)
    deltas: list[float] = []
    for _ in range(n_perturb):
        perturbed = _perturb_setup(base, rng)
        lap = model.predict_lap_time(perturbed, track_id)
        deltas.append(lap - base_lap)
    return float(np.std(deltas))


if __name__ == "__main__":
    train()
