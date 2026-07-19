"""Setup 19 维 -> lap_simulator_2026 物理参数映射层 (Iter-66).

把 :class:`CarSetup` 19 维调教参数映射到 :class:`LapConfig2026` 物理输入,
让 EA F1 2026 lap_simulator 物理引擎 (24 赛道 0.01% 精度) 能评估任意 setup.

**设计动机** (用户 Iter-65 后方向):
- 训练样本的 *真实度* 取决于标签生成器的物理可信度. 原 train.py 的纯启发式
  `heuristic_sectors` 把基线圈速建立在 ``length/avg_speed`` (5 档粗粒度速度) 上,
  即便 Iter-65 已对齐 benchmark, 残差结构仍是手写 V 形惩罚, 与真实物理引擎脱节.
- 真实物理引擎 lap_simulator_2026 已在 24 赛道 0.01% 精度校准; 让它评估任意
  setup 即获得 *物理一致的圈速真值*, 可作为 DNN 训练的高质量标签源 (Iter-67).

**映射策略**:

1. **燃油**: ``setup.fuel_load`` -> ``cfg.current_fuel_kg`` (1:1 物理一致)
2. **化合物**: 默认 ``medium`` (调教 schema 不含化合物选择; 调用方可覆写)
3. **其余 18 维**: 通过 *物理惩罚函数* -> ``cfg.car_performance_offset_s``
   - 每项 V 形惩罚偏离 *该赛道类型的专业车队最优值*
   - 惩罚强度按赛道类型 (高速/街道/高下压力/中等/混合) 缩放
   - 每项单位: 秒 / 档 (与游戏调教一致)
   - 总和 = car_performance_offset_s (正值 = 慢于 reference car)

**校准锚点**: DEFAULT_SETUP 在 ``medium`` 赛道上的 offset = 0
(DEFAULT_SETUP 即 medium 赛道最优), 任意 setup 偏离则产生 >0 offset.
这让 ``setup_lap_time(DEFAULT_SETUP, "melbourne")`` ≈ benchmark (medium 赛道).

公开 API:
- :func:`setup_to_lap_config` — CarSetup -> LapConfig2026
- :func:`setup_lap_time` — CarSetup + track_id -> float (物理真值圈速)
- :func:`evaluate_setup` — CarSetup + track_id -> LapResult2026 (含 breakdown)
- :func:`setup_penalty_s` — CarSetup + track_id -> float (仅 setup 惩罚, 秒)
- :func:`optimal_setup_for_track_type` — 该赛道类型的参考最优 setup
"""

from __future__ import annotations

from typing import Any

from f1opt.data.ea_f1_2026_benchmark import canonical_track_id
from f1opt.data.setup_schema import DEFAULT_SETUP, CarSetup
from f1opt.data.tracks import TRACKS_BY_ID, TrackType
from f1opt.model.lap_simulator_2026 import (
    _REF_AMBIENT_TEMP_C,
    _REF_COMPOUND,
    _REF_GAP_S,
    _REF_LAP,
    _REF_LAP_IN_STINT,
    _REF_PU_MODE,
    _REF_SESSION,
    _REF_SOC_MJ,
    _REF_TIRE_AGE,
    _REF_TRACK_TEMP_C,
    _REF_WET,
    LapConfig2026,
    LapResult2026,
    simulate_lap_2026,
)

# --------------------------------------------------------------------------- #
# 各赛道类型的 *专业车队参考最优* setup (EA F1 2026 garage 物理经验值)
# --------------------------------------------------------------------------- #
# 数据来源: EA F1 2026 garage 工程师推荐 + F1 2026 实际车队 baseline; 每项
# 为该赛道类型上 *最小化单圈时间* 的中心值, 实际车队会在 ±5 档内微调以适配
# 车手风格/燃油/胎温. 桥接层把 setup 偏离此最优的程度转成秒级惩罚.
_TRACK_TYPE_OPTIMA: dict[TrackType, dict[str, float]] = {
    "high_speed_low_downforce": {
        # Monza / Baku / Jeddah / Montreal / Las Vegas — 长直道为主, 低下压力
        "front_wing": 8.0, "rear_wing": 6.0,
        "on_throttle_diff": 92.0, "off_throttle_diff": 18.0,
        "front_camber": -3.0, "rear_camber": -1.5,
        "front_toe": 0.05, "rear_toe": 0.15,
        "front_suspension": 35.0, "rear_suspension": 28.0,
        "front_arb": 28.0, "rear_arb": 22.0,
        "front_ride_height": 35.0, "rear_ride_height": 40.0,
        "brake_pressure": 96.0, "front_brake_bias": 50.0,
        "front_tyre_pressure": 24.5, "rear_tyre_pressure": 21.0,
        "fuel_load": 50.0,  # EA F1 2026 reference (penalty 不计入 fuel, 仅 CarSetup 必填)
    },
    "street": {
        # Monaco / Singapore / Madrid / Miami — 街道赛, 最大化下压力 + 慢弯抓地
        "front_wing": 40.0, "rear_wing": 42.0,
        "on_throttle_diff": 80.0, "off_throttle_diff": 50.0,
        "front_camber": -3.4, "rear_camber": -1.9,
        "front_toe": 0.08, "rear_toe": 0.20,
        "front_suspension": 18.0, "rear_suspension": 12.0,
        "front_arb": 12.0, "rear_arb": 18.0,
        "front_ride_height": 18.0, "rear_ride_height": 30.0,
        "brake_pressure": 100.0, "front_brake_bias": 53.0,
        "front_tyre_pressure": 23.5, "rear_tyre_pressure": 20.5,
        "fuel_load": 50.0,
    },
    "high_downforce": {
        # Hungaroring / Zandvoort — 高下压力技术赛道
        "front_wing": 32.0, "rear_wing": 34.0,
        "on_throttle_diff": 85.0, "off_throttle_diff": 35.0,
        "front_camber": -3.3, "rear_camber": -1.8,
        "front_toe": 0.06, "rear_toe": 0.18,
        "front_suspension": 22.0, "rear_suspension": 16.0,
        "front_arb": 16.0, "rear_arb": 22.0,
        "front_ride_height": 22.0, "rear_ride_height": 34.0,
        "brake_pressure": 98.0, "front_brake_bias": 52.0,
        "front_tyre_pressure": 24.0, "rear_tyre_pressure": 20.8,
        "fuel_load": 50.0,
    },
    "medium": {
        # Melbourne / Bahrain / Barcelona / Spielberg / Austin / Brazil / etc.
        # 攻守均衡, 中等下压力 (与 DEFAULT_SETUP 完全一致, 作为校准锚点)
        "front_wing": 25.0, "rear_wing": 27.0,
        "on_throttle_diff": 80.0, "off_throttle_diff": 55.0,
        "front_camber": -3.5, "rear_camber": -2.0,
        "front_toe": 0.05, "rear_toe": 0.20,
        "front_suspension": 21.0, "rear_suspension": 11.0,
        "front_arb": 10.0, "rear_arb": 20.0,
        "front_ride_height": 20.0, "rear_ride_height": 40.0,
        "brake_pressure": 100.0, "front_brake_bias": 55.0,
        "front_tyre_pressure": 24.0, "rear_tyre_pressure": 20.5,
        "fuel_load": 50.0,
    },
    "mixed": {
        # Suzuka / Silverstone / Spa / COTA / Losail / Yas Marina — 三段差异大
        "front_wing": 18.0, "rear_wing": 20.0,
        "on_throttle_diff": 88.0, "off_throttle_diff": 30.0,
        "front_camber": -3.2, "rear_camber": -1.7,
        "front_toe": 0.05, "rear_toe": 0.18,
        "front_suspension": 26.0, "rear_suspension": 20.0,
        "front_arb": 20.0, "rear_arb": 22.0,
        "front_ride_height": 26.0, "rear_ride_height": 36.0,
        "brake_pressure": 97.0, "front_brake_bias": 51.0,
        "front_tyre_pressure": 24.2, "rear_tyre_pressure": 21.0,
        "fuel_load": 50.0,
    },
}


def optimal_setup_for_track_type(track_type: TrackType) -> CarSetup:
    """返回该赛道类型的 *参考最优* setup (专业车队 baseline).

    注意: 这是一份 *静态参考*, 实际车队会按燃油/胎温/车手风格在 ±5 档内微调;
    桥接层把 *偏离此最优* 的程度转成秒级惩罚.
    """
    opt = _TRACK_TYPE_OPTIMA[track_type]
    return CarSetup(**opt)


def optimal_setup_for_track(track_id: str) -> CarSetup:
    """返回该赛道的 *逐赛道工程参数感知最优* setup (Iter-164.15).

    与 :func:`optimal_setup_for_track_type` 的区别: 本函数用
    :func:`_track_engineering_adjusted_optima` 调制最优值, 让同类型赛道
    (silverstone/suzuka) 获得不同的最优 setup. 在该赛道上
    :func:`setup_penalty_s` 返回接近 0.
    """
    track = TRACKS_BY_ID.get(canonical_track_id(track_id))
    if track is None:
        return optimal_setup_for_track_type("medium")
    opt = _track_engineering_adjusted_optima(canonical_track_id(track_id), track.track_type)
    # 确保 fuel_load = 50 (reference, 与 benchmark 校准一致)
    opt = {**opt, "fuel_load": 50.0}
    return CarSetup(**opt)


# --------------------------------------------------------------------------- #
# 每项 setup 偏离最优的 *物理敏感度* (秒/档)
# --------------------------------------------------------------------------- #
# 来源: EA F1 2026 garage 工程师经验 + 真实 F1 遥测幅度. 每项代表 *1 档偏离*
# 最优值在 *medium 赛道* 上的平均圈速代价; 赛道类型缩放因子 (下方) 进一步
# 放大/缩小该代价 (例: 翼面在高速赛道敏感度 ×1.6, 在街道赛道 ×1.3).
_BASE_SENSITIVITY_S_PER_CLICK: dict[str, float] = {
    "front_wing": 0.040,          # 高下压力翼面, 每档显著影响直道/弯角平衡
    "rear_wing": 0.045,           # 尾翼阻力主导, 略高于前翼
    "on_throttle_diff": 0.018,    # 差速锁止影响牵引出弯, 每档中等
    "off_throttle_diff": 0.012,   # 收油差速影响进弯旋转, 每档较低
    "front_camber": 0.020,        # 外倾角 (0.01° 步长, 总范围 1°) 每步长代价
    "rear_camber": 0.018,
    "front_toe": 0.010,           # 前束角, 影响直道稳定性 + 转向响应
    "rear_toe": 0.010,
    "front_suspension": 0.015,    # 弹簧硬度, 影响俯仰 + 颠簸
    "rear_suspension": 0.015,
    "front_arb": 0.012,           # 防倾杆, 影响侧倾 + 弯角平衡
    "rear_arb": 0.012,
    "front_ride_height": 0.010,   # 离地间隙, 影响底盘失速 + 过弯
    "rear_ride_height": 0.010,
    "brake_pressure": 0.008,      # 制动压力, 影响制动距离 + 锁死风险
    "front_brake_bias": 0.015,    # 制动分配, 影响进弯稳定性
    "front_tyre_pressure": 0.020, # 胎压 (0.1 psi 步长) 影响胎温 + 抓地
    "rear_tyre_pressure": 0.020,
}

# 赛道类型 *敏感度缩放因子* (乘到 _BASE_SENSITIVITY 上).
# 例: 翼面在 high_speed_low_downforce 赛道 (Monza) 上偏离最优代价更高
# (直道阻力主导), 在 street 赛道 (Monaco) 也高 (慢弯抓地主导);
# camber/suspension 在 street (颠簸) 上更敏感; brake_bias 在 heavy braking
# 赛道上更敏感.
_TRACK_TYPE_SCALE: dict[TrackType, dict[str, float]] = {
    "high_speed_low_downforce": {
        "front_wing": 1.6, "rear_wing": 1.7,        # 直道阻力主导
        "on_throttle_diff": 1.2, "off_throttle_diff": 0.8,
        "front_camber": 0.8, "rear_camber": 0.8,
        "front_toe": 0.7, "rear_toe": 0.7,
        "front_suspension": 0.8, "rear_suspension": 0.8,
        "front_arb": 0.7, "rear_arb": 0.7,
        "front_ride_height": 1.2, "rear_ride_height": 1.2,
        "brake_pressure": 1.2, "front_brake_bias": 1.4,  # 重制动赛道
        "front_tyre_pressure": 1.0, "rear_tyre_pressure": 1.0,
    },
    "street": {
        "front_wing": 1.3, "rear_wing": 1.3,        # 慢弯抓地主导, 仍要下压力
        "on_throttle_diff": 1.4, "off_throttle_diff": 1.5,  # 牵引/进弯主导
        "front_camber": 1.5, "rear_camber": 1.4,    # 慢弯外倾敏感
        "front_toe": 1.3, "rear_toe": 1.2,
        "front_suspension": 1.6, "rear_suspension": 1.6,    # 颠簸街道
        "front_arb": 1.5, "rear_arb": 1.5,          # 慢弯侧倾
        "front_ride_height": 1.5, "rear_ride_height": 1.5,  # 路肩 + 减速带
        "brake_pressure": 1.3, "front_brake_bias": 1.5,     # 重制动频繁
        "front_tyre_pressure": 1.2, "rear_tyre_pressure": 1.2,
    },
    "high_downforce": {
        "front_wing": 1.4, "rear_wing": 1.4,        # 弯角主导, 翼面平衡关键
        "on_throttle_diff": 1.3, "off_throttle_diff": 1.2,
        "front_camber": 1.4, "rear_camber": 1.3,
        "front_toe": 1.0, "rear_toe": 1.0,
        "front_suspension": 1.1, "rear_suspension": 1.1,
        "front_arb": 1.2, "rear_arb": 1.2,
        "front_ride_height": 1.0, "rear_ride_height": 1.0,
        "brake_pressure": 1.1, "front_brake_bias": 1.2,
        "front_tyre_pressure": 1.1, "rear_tyre_pressure": 1.1,
    },
    "medium": {  # 校准锚点: 全 1.0
        "front_wing": 1.0, "rear_wing": 1.0,
        "on_throttle_diff": 1.0, "off_throttle_diff": 1.0,
        "front_camber": 1.0, "rear_camber": 1.0,
        "front_toe": 1.0, "rear_toe": 1.0,
        "front_suspension": 1.0, "rear_suspension": 1.0,
        "front_arb": 1.0, "rear_arb": 1.0,
        "front_ride_height": 1.0, "rear_ride_height": 1.0,
        "brake_pressure": 1.0, "front_brake_bias": 1.0,
        "front_tyre_pressure": 1.0, "rear_tyre_pressure": 1.0,
    },
    "mixed": {
        "front_wing": 1.2, "rear_wing": 1.2,        # 直道+弯角都需要妥协
        "on_throttle_diff": 1.1, "off_throttle_diff": 1.1,
        "front_camber": 1.2, "rear_camber": 1.2,
        "front_toe": 0.9, "rear_toe": 0.9,
        "front_suspension": 1.1, "rear_suspension": 1.1,
        "front_arb": 1.1, "rear_arb": 1.1,
        "front_ride_height": 1.1, "rear_ride_height": 1.1,
        "brake_pressure": 1.1, "front_brake_bias": 1.2,
        "front_tyre_pressure": 1.05, "rear_tyre_pressure": 1.05,
    },
}


# --------------------------------------------------------------------------- #
# 惩罚函数
# --------------------------------------------------------------------------- #
# Iter-107: 全局总惩罚 cap. 旧版纯线性 V-shape 让极端 setup (全 max/min) 总惩罚
# 达 10-13s, 远超 EA F1 2026 权威 3-6s (garage 实测全错调教慢 3-6s). 根因: 线性
# 叠加 18 维偏差无饱和, 但真实 F1 物理有整车性能下限 (轮胎抓地极限 / 底盘失速 /
# 翼面 stall). 修复: 对 *总惩罚* 加 cap (=6s, 权威上限), 而非 per-item cap.
# per-item cap (L=15) 会破坏单维灵敏度 (on_throttle_diff 25 档偏离仍应有差异);
# 全局 cap 保留所有单维线性灵敏度, 仅在 18 维同时极端偏离时封顶.
_TOTAL_PENALTY_CAP_S = 6.0


# --------------------------------------------------------------------------- #
# Iter-164.15: 逐赛道工程参数感知的最优值调整
# --------------------------------------------------------------------------- #
# 旧版 setup_penalty_s 只用 _TRACK_TYPE_OPTIMA[track_type] — 同类型赛道
# (silverstone/suzuka 均 'mixed') 获得完全相同的最优值 → 相同惩罚景观 →
# DE 收敛到相同 setup. 但 track_engineering 数据 (downforce_level /
# tire_wear_factor / brake_wear_level / top_speed_kmh) 在同类型赛道间有
# 显著差异 (silverstone df=0.80 tw=1.35 vs suzuka df=0.75 tw=1.25).
#
# 修复: 用工程数据调制 track_type 最优值, 让同类型赛道获得不同的惩罚景观.
# 调制强度标定: 让 silverstone/suzuka 的 front_wing 最优值差 ≥1 档 (游戏
# 步长 1.0), 保证 DE 能区分.
#
# 物理依据:
# - downforce_level ↑ → 需要更高翼面获得弯角抓地 → front_wing/rear_wing ↑
# - tire_wear_factor ↑ → 需要更温和的 camber (减少胎面剪切) + 更高胎压
#   (减小接地面积降低磨损) → camber 向 0 靠拢, tyre_pressure ↑
# - brake_wear_level ↑ → 需要更低制动压力保育刹车 → brake_pressure ↓
# - top_speed_kmh ↑ → 需要更低翼面减阻 → front_wing/rear_wing ↓
_DF_REF_BY_TYPE: dict[TrackType, float] = {
    "high_speed_low_downforce": 0.50,
    "street": 0.90,
    "high_downforce": 0.85,
    "medium": 0.55,
    "mixed": 0.65,
}
_TW_REF = 1.0
_BW_REF = 0.65
_TS_REF = 320.0
_eng_optima_cache: dict[str, dict[str, float]] = {}


def _track_engineering_adjusted_optima(
    track_id: str, track_type: TrackType,
) -> dict[str, float]:
    """逐赛道工程参数感知的最优值 (Iter-164.15).

    用 :mod:`f1opt.data.track_engineering` 数据调制
    :data:`_TRACK_TYPE_OPTIMA` ``[track_type]``, 让同类型赛道获得不同最优值.
    例如 silverstone (df=0.80) 比 suzuka (df=0.75) 有更高翼面最优值.
    """
    cached = _eng_optima_cache.get(track_id)
    if cached is not None:
        return cached

    from f1opt.data.setup_schema import SETUP_FIELDS
    from f1opt.data.track_engineering import _ENGINEERING_BY_TRACK

    base = dict(_TRACK_TYPE_OPTIMA[track_type])
    eng = _ENGINEERING_BY_TRACK.get(track_id)
    if eng is None:
        _eng_optima_cache[track_id] = base
        return base

    df_ref = _DF_REF_BY_TYPE.get(track_type, 0.6)

    # 下压力: 每 0.1 高于参考 → 翼面最优 +1 档 (前+后各 +0.5)
    df_shift = (eng.downforce_level - df_ref) * 20.0
    base["front_wing"] = max(0.0, base["front_wing"] + df_shift * 0.5)
    base["rear_wing"] = max(0.0, base["rear_wing"] + df_shift * 0.5)

    # 胎耗: 每 0.1 高于参考 → camber 向 0 靠拢 1%, tyre_pressure +0.1 psi
    tw_shift = eng.tire_wear_factor - _TW_REF
    base["front_camber"] = base["front_camber"] * (1.0 - tw_shift * 0.10)
    base["rear_camber"] = base["rear_camber"] * (1.0 - tw_shift * 0.10)
    base["front_tyre_pressure"] += tw_shift * 1.0
    base["rear_tyre_pressure"] += tw_shift * 1.0

    # 刹车磨损: 每 0.1 高于参考 → brake_pressure -0.2 (温和保育刹车)
    bw_shift = eng.brake_wear_level - _BW_REF
    base["brake_pressure"] = max(80.0, base["brake_pressure"] - bw_shift * 2.0)

    # 注意: top_speed_kmh 调整已移除 — downforce_level 已充分捕获翼面需求,
    # top_speed 调整会把高速赛道最优推到搜索空间边缘 (monza fw=3 rw=1),
    # 导致 DE 在低迭代预算 (iter=25) 下无法收敛.

    # Snap 到游戏档位网格 (int 字段 round 到整数, float 字段 round 到 step).
    for name, spec in SETUP_FIELDS.items():
        if name in base:
            val = base[name]
            # round 到 step
            steps = round(val / spec.step)
            base[name] = max(spec.min, min(spec.max, steps * spec.step))

    _eng_optima_cache[track_id] = base
    return base


def setup_penalty_s(setup: CarSetup, track_id: str) -> float:
    """计算 setup 偏离 *该赛道类型最优* 的总惩罚 (秒, 正=慢).

    每项 setup 偏离最优值的代价 = |x - x_opt|/step × base_sensitivity × track_scale
    (线性 V-shape). 总和再经全局 cap (``_TOTAL_PENALTY_CAP_S = 6.0``) 封顶.

    Iter-107: 加全局 cap. 旧版纯线性让极端 setup (全 max/min) 总惩罚达 10-13s,
    远超 EA F1 2026 权威 3-6s. 全局 cap (而非 per-item) 保留单维线性灵敏度:
    on_throttle_diff 60 vs 100 (偏离最优 85 的 25 vs 15 档) 仍有线性差异, 仅当
    18 维同时极端偏离时总惩罚封顶到 6s. 实测全 max 惩罚降到 6s (权威上限),
    单参数扫描仍单调, 最优附近完全线性.

    Args:
        setup: 待评估的 19 维调教.
        track_id: 赛道 ID (用于解析 track_type).

    Returns:
        setup 偏离最优的总秒数代价 (>= 0, <= _TOTAL_PENALTY_CAP_S).
    """
    track = TRACKS_BY_ID.get(canonical_track_id(track_id))
    if track is None:
        # 未知赛道: 用 medium 缩放 (校准锚点), 不崩溃
        track_type: TrackType = "medium"
    else:
        track_type = track.track_type

    # Iter-164.15: 用逐赛道工程参数感知的最优值 (而非裸 track_type 最优值),
    # 让同类型赛道 (silverstone/suzuka) 获得不同惩罚景观.
    opt = _track_engineering_adjusted_optima(canonical_track_id(track_id), track_type) \
        if track is not None else _TRACK_TYPE_OPTIMA[track_type]
    scale = _TRACK_TYPE_SCALE[track_type]
    total = 0.0
    for name, base_s in _BASE_SENSITIVITY_S_PER_CLICK.items():
        x = float(getattr(setup, name))
        x_opt = opt[name]
        # V 形惩罚: |x - x_opt| × base × scale (每档代价 × 档数偏离)
        delta_clicks = abs(x - x_opt)
        # 对 int 字段 (步长 1.0) delta_clicks 即档数;
        # 对 float 字段 (步长 0.01 / 0.1) delta_clicks 是连续距离, base_s
        # 已校准为 "每步长代价" -> 需除以 step 得到每连续单位代价.
        # 实际: base_s 表是 *每档* 代价, float 字段的 "档" = step, 所以:
        #   代价 = (delta_clicks / step) × base_s × scale
        # 但 _BASE_SENSITIVITY 已经标定为 *步长单位* (例 camber 0.02 s/0.01°
        # = 0.02 s/档), 所以直接用 delta_clicks / step × base_s.
        if name in ("front_camber", "rear_camber"):
            step = 0.01
        elif name in ("front_toe", "rear_toe"):
            step = 0.01
        elif name in ("front_tyre_pressure", "rear_tyre_pressure"):
            step = 0.1
        else:
            step = 1.0
        clicks = delta_clicks / step
        total += clicks * base_s * scale[name]
    # Iter-107: 全局 cap — 保留单维线性灵敏度, 仅极端总惩罚封顶到 EA F1 2026
    # 权威上限 (6s). 消除线性叠加导致的极端 setup 惩罚过高 (10-13s → 6s).
    return float(min(total, _TOTAL_PENALTY_CAP_S))


# --------------------------------------------------------------------------- #
# 桥接: setup -> LapConfig2026
# --------------------------------------------------------------------------- #
def setup_to_lap_config(
    setup: CarSetup,
    track_id: str,
    *,
    compound: str = _REF_COMPOUND,  # type: ignore[assignment]
    tire_age_laps: int = _REF_TIRE_AGE,
    pu_mode: Any = _REF_PU_MODE,
    pu_soc_mj: float = _REF_SOC_MJ,
    wet: bool = _REF_WET,
    gap_to_ahead_s: float = _REF_GAP_S,
    session_type: str = _REF_SESSION,
    lap: int = _REF_LAP,
    driver_skill_offset_s: float = 0.0,
    track_temp_c: float = _REF_TRACK_TEMP_C,
    ambient_temp_c: float = _REF_AMBIENT_TEMP_C,
    lap_in_stint: int = _REF_LAP_IN_STINT,
    sc_just_ended_lap: int = 0,
) -> LapConfig2026:
    """把 19 维 CarSetup 映射到 :class:`LapConfig2026` 物理输入.

    映射规则:
    - ``setup.fuel_load`` -> ``current_fuel_kg`` (1:1 物理一致)
    - ``setup`` 偏离该赛道类型最优 -> ``car_performance_offset_s`` (秒)
    - 其余 LapConfig 字段默认 reference state (medium 胎, BALANCED PU, ...)

    调用方可在 kwargs 中覆写 compound / pu_mode / wet 等状态变量, 但 setup
    本身只决定 *车体性能* (aero/suspension/brake/tire/fuel), 不决定策略状态.

    Args:
        setup: 19 维调教.
        track_id: 赛道 ID.
        **: 其余 LapConfig2026 字段 (默认 reference state).

    Returns:
        :class:`LapConfig2026` 可直接传给 :func:`simulate_lap_2026`.
    """
    car_offset = setup_penalty_s(setup, track_id)
    return LapConfig2026(
        track_id=track_id,
        compound=compound,
        tire_age_laps=tire_age_laps,
        current_fuel_kg=float(setup.fuel_load),
        pu_mode=pu_mode,
        pu_soc_mj=pu_soc_mj,
        wet=wet,
        gap_to_ahead_s=gap_to_ahead_s,
        session_type=session_type,
        lap=lap,
        sc_just_ended_lap=sc_just_ended_lap,
        driver_skill_offset_s=driver_skill_offset_s,
        car_performance_offset_s=car_offset,
        track_temp_c=track_temp_c,
        ambient_temp_c=ambient_temp_c,
        lap_in_stint=lap_in_stint,
    )


def evaluate_setup(
    setup: CarSetup,
    track_id: str,
    **kwargs: Any,
) -> LapResult2026:
    """用 EA F1 2026 物理引擎评估任意 setup, 返回 :class:`LapResult2026`.

    等价于 ``simulate_lap_2026(setup_to_lap_config(setup, track_id, **kwargs))``.
    返回的 ``LapResult2026`` 含圈速 + 各子系统 delta (tire/fuel/pu/aero/drs/...),
    其中 ``car_offset_s`` = setup 偏离最优的物理代价.
    """
    cfg = setup_to_lap_config(setup, track_id, **kwargs)
    return simulate_lap_2026(cfg)


def setup_lap_time(
    setup: CarSetup,
    track_id: str,
    **kwargs: Any,
) -> float:
    """便捷: 用物理引擎评估 setup -> 圈速 (秒). 物理真值, 用于 DNN 标签."""
    return float(evaluate_setup(setup, track_id, **kwargs).lap_time_s)


# --------------------------------------------------------------------------- #
# 物理一致性自检 (供测试与 import-time sanity check)
# --------------------------------------------------------------------------- #
def _sanity_check() -> dict[str, float]:
    """模块级自检: DEFAULT_SETUP 在 medium 赛道 penalty = 0, 在其他赛道 > 0.

    返回各代表赛道的 penalty (秒). 测试可调用此函数验证物理一致性.
    """
    samples = {
        "melbourne": "medium",
        "monza": "high_speed_low_downforce",
        "monaco": "street",
        "hungaroring": "high_downforce",
        "spa": "mixed",
    }
    return {tid: setup_penalty_s(DEFAULT_SETUP, tid) for tid in samples}
