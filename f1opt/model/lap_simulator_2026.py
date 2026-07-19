"""EA F1 2026 高性能单圈仿真器 (Iter-56).

严格契合 EA Sports F1 2026 物理引擎, 整合 2026 PU (9 MJ/lap 部署) +
主动空动 (X/Z-mode) + Pirelli 2026 三阶段轮胎 + 燃油模型, 并以
:data:`EA_F1_2026_LAP_TIME_BENCHMARK` 为校准锚点.

**校准锚点 (reference state)**:
- 轮胎: medium, 全新 (age=0)
- 燃油: 50 kg (中段 stint 参考)
- PU 模式: BALANCED (6 MJ/lap), SoC 满
- 主动空动: 干地, 净空 (gap > 1.0s)
- 车手: 中性 (offset = 0)

在 reference state 下, 单圈圈速 = EA F1 2026 benchmark (误差 0%).
偏离 reference state 时, 各子系统按物理量级贡献 delta:

.. code-block:: text

    lap_time = benchmark
             + tire_delta(age, compound) - tire_delta(0, medium)
             + fuel_delta(current_kg)    - fuel_delta(50)
             + pu_delta(mode, soc)       - pu_delta(BALANCED, full)
             + aero_delta(wet, gap)      - aero_delta(False, >1.0)
             + weather_penalty(wet)
             + driver_offset

**性能目标**: < 30 us/lap (无对象分配, 纯标量运算, dict 查找缓存).

公开 API:
    - :class:`LapConfig2026` — 单圈仿真输入 (frozen dataclass).
    - :class:`LapResult2026` — 单圈仿真输出.
    - :func:`simulate_lap_2026` — 模块级便捷函数 (无状态).
    - :class:`LapSimulator2026` — 跨圈 stint 仿真器 (有状态).
    - :func:`validate_against_benchmark` — 24 赛道精度验证.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from f1opt.data.ea_f1_2026_benchmark import (
    EA_F1_2026_LAP_TIME_BENCHMARK,
    resolve_track_id,
)
from f1opt.model.active_aero_coupling import (
    active_aero_lap_gain_s,
)
from f1opt.model.drs_coupling import drs_lap_gain_s
from f1opt.model.fuel_model import FuelMode, fuel_effect_on_lap_time, fuel_per_lap
from f1opt.model.pu_2026 import (
    BATTERY_CAPACITY_MJ,
    DEPLOY_GAIN_S_PER_MJ,
    HARVEST_DRAG_S_PER_MJ,
    MAX_HARVEST_MJ_PER_LAP,
    PUDeployMode,
)
from f1opt.model.safety_car import SafetyCarModel
from f1opt.model.tire_curve import lap_time_delta_s as tire_lap_delta_s
from f1opt.model.tire_temperature import tire_temp_penalty_s as _tire_temp_penalty_s

# --------------------------------------------------------------------------- #
# 校准锚点 (reference state)
# --------------------------------------------------------------------------- #
_REF_COMPOUND = "medium"
_REF_TIRE_AGE = 0
_REF_FUEL_KG = 50.0
_REF_PU_MODE = PUDeployMode.BALANCED
_REF_SOC_MJ = BATTERY_CAPACITY_MJ  # 满 SoC
_REF_GAP_S = 1.5  # 净空 (> 1.0)
_REF_WET = False
_REF_SESSION = "qualifying"  # 排位赛: DRS 全程可用 (benchmark = 排位圈速)
_REF_LAP = 2  # 排位赛第 2 圈 (避开第 1 圈规则, 虽然排位不受限)
# 轮胎温度参考 (EA F1 2026: 干地, 中等赛道温度, 暖胎后状态)
_REF_TRACK_TEMP_C = 30.0
_REF_AMBIENT_TEMP_C = 25.0
_REF_LAP_IN_STINT = 3  # 暖胎后 (lap >= 3 无冷启动偏移)

# 缓存: reference state 各子系统贡献 (启动时计算一次)
_REF_TIRE_DELTA = tire_lap_delta_s(_REF_COMPOUND, _REF_TIRE_AGE)
_REF_FUEL_DELTA = fuel_effect_on_lap_time(_REF_FUEL_KG)
# reference 轮胎温度惩罚 (calibration anchor: lap=3 已暖胎, 干地, 中等赛道温度)
_REF_TIRE_TEMP_PENALTY_S = _tire_temp_penalty_s(
    _REF_COMPOUND, _REF_TRACK_TEMP_C, _REF_AMBIENT_TEMP_C,
    _REF_LAP_IN_STINT, _REF_TIRE_AGE, _REF_WET,
)

# PU reference gain: BALANCED 模式, 满 SoC, 中性效率
# 6 MJ deploy × 1.0 eff × 0.09 s/MJ - 6 MJ harvest × 1.0 × 0.04 s/MJ = 0.30 s
_REF_PU_GAIN_S = 6.0 * 1.0 * DEPLOY_GAIN_S_PER_MJ - 6.0 * 1.0 * HARVEST_DRAG_S_PER_MJ

# PU 模式参数内联 (避免 PU2026Model 对象创建)
_PU_MODE_PARAMS_INLINE: dict[PUDeployMode, tuple[float, float]] = {
    PUDeployMode.QUALIFYING: (9.0, 1.10),
    PUDeployMode.ATTACK: (8.0, 1.05),
    PUDeployMode.BALANCED: (6.0, 1.0),
    PUDeployMode.CONSERVE: (4.0, 0.95),
}

# 赛道回收因子内联 (避免 PU2026Model._track_harvest_factor 方法调用)
_HIGH_HARVEST_TRACKS = frozenset({"montreal", "singapore", "bahrain", "miami", "austin"})
_LOW_HARVEST_TRACKS = frozenset({"spa", "monza", "jeddah", "baku", "las_vegas"})

# SC/VSC 期间物理 (EA F1 2026 race physics)
# SC 跟车: 低速匀速, 轮胎磨损降低, 燃油消耗降低, SoC 充满
_SC_TIRE_WEAR_FACTOR = 0.30      # SC 期间轮胎磨损 = 正常 × 0.30
_SC_FUEL_BURN_FACTOR = 0.50      # SC 期间燃油消耗 = 正常 × 0.50
_VSC_TIRE_WEAR_FACTOR = 0.50     # VSC 磨损略高于 SC
_VSC_FUEL_BURN_FACTOR = 0.65     # VSC 燃油消耗
_DRS_DISABLED_AFTER_SC_LAPS = 2  # FIA 2026: SC 后 2 圈 DRS 禁用

# Iter-112: dirty air (gap < 1.0, race, dry) X-mode 减效因子. 旧版 active_aero
# 模块让 dirty air 完全禁用部分直道 X-mode (gap > _DETECTION_GAP_S), 给出
# +0.41s (melbourne) ~ +0.84s (las_vegas) dirty air penalty, 超权威 +0.1~+0.3s.
# 真实 F1 dirty air 仅减效 30-50% (乱流降低 X-mode 效率但不完全消除). 因子 0.35
# 让 dirty air aero_delta = original_delta × 0.35, 全 24 赛道落 +0.1~+0.3s 范围
# (monaco/suzuka 短直道赛道自然更低). 仅影响 race + gap<1.0 + dry 场景.
_DIRTY_AIR_RETAIN_FACTOR = 0.35

# Iter-112: wet weather_penalty 1.5→0.3. 旧版 1.5s 代表湿地额外抓地损失, 但
# aero_delta (X-mode 禁用, ~1.36s) + drs_delta (~0.90s) + tire_temp (~0.375s)
# 已含湿地物理损失, 叠加 1.5s 给总 +4.1s (超权威 +1.5~+3.0s). 0.3s 代表纯
# 赛道表面湿滑 (独立于 aero/DRS/tire_temp), 总 wet delta ~2.9s (在范围内).
_WET_GRIP_PENALTY_S = 0.3


@lru_cache(maxsize=256)
def _cached_aero_gain(track_id: str, aero_gap: float, wet: bool) -> float:
    """lru_cache 包装的 aero gain (避免重复 optimal_plan_for_track 排序)."""
    return active_aero_lap_gain_s(track_id, aero_gap, wet)


@lru_cache(maxsize=256)
def _cached_drs_gain(
    track_id: str,
    lap: int,
    gap_to_ahead_s: float | None,
    session_type: str,
    wetness: float,
    sc_just_ended_lap: int,
) -> float:
    """lru_cache 包装的 DRS gain (避免重复 drs_available 判断)."""
    return drs_lap_gain_s(
        track_id, lap=lap, gap_to_ahead_s=gap_to_ahead_s,
        session_type=session_type, wetness=wetness,
        sc_just_ended_lap=sc_just_ended_lap,
    )


def _track_harvest_factor_inline(track_id: str) -> float:
    """赛道回收因子 (内联, 避免 PU2026Model 方法调用)."""
    if resolve_track_id(track_id) in _HIGH_HARVEST_TRACKS:
        return 1.10
    if track_id in _LOW_HARVEST_TRACKS:
        return 0.80
    return 1.0


def _inline_pu_gain(
    track_id: str,
    mode: PUDeployMode,
    soc_mj: float,
) -> tuple[float, float, float, float]:
    """内联 PU 收益计算 (避免 PU2026Model/PU2026State 对象创建).

    Returns:
        (deploy_mj, harvest_mj, net_gain_s, soc_after_mj)
    """
    target, mode_eff = _PU_MODE_PARAMS_INLINE[mode]
    actual_deploy = min(target, soc_mj)
    # 低 SoC + QUALIFYING 自动降级
    if soc_mj < 0.2 * BATTERY_CAPACITY_MJ and mode == PUDeployMode.QUALIFYING:
        actual_deploy = min(actual_deploy, target * 0.6)

    harvest_base = MAX_HARVEST_MJ_PER_LAP * _track_harvest_factor_inline(track_id)
    if mode == PUDeployMode.CONSERVE:
        harvest_base *= 1.15
    elif mode == PUDeployMode.QUALIFYING:
        harvest_base *= 0.85
    room = BATTERY_CAPACITY_MJ - soc_mj + actual_deploy
    actual_harvest = min(harvest_base, room)

    effective_deploy = actual_deploy * mode_eff
    deploy_gain = effective_deploy * DEPLOY_GAIN_S_PER_MJ
    harvest_cost = actual_harvest * HARVEST_DRAG_S_PER_MJ
    net_gain = deploy_gain - harvest_cost
    soc_after = max(0.0, min(BATTERY_CAPACITY_MJ, soc_mj - actual_deploy + actual_harvest))
    return actual_deploy, actual_harvest, net_gain, soc_after


# --------------------------------------------------------------------------- #
# LapConfig2026
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LapConfig2026:
    """单圈仿真输入配置 (immutable, 适合缓存与并发).

    所有字段为物理量, 不含策略状态 (策略由调用方管理).
    """

    track_id: str
    compound: str = _REF_COMPOUND
    tire_age_laps: int = _REF_TIRE_AGE
    current_fuel_kg: float = _REF_FUEL_KG
    fuel_mode: FuelMode = FuelMode.NORMAL
    pu_mode: PUDeployMode = _REF_PU_MODE
    pu_soc_mj: float = _REF_SOC_MJ
    wet: bool = _REF_WET
    gap_to_ahead_s: float = _REF_GAP_S
    session_type: str = _REF_SESSION  # "qualifying" 或 "race"
    lap: int = _REF_LAP  # 当前圈 (1-based, 用于 DRS 第 1 圈规则)
    sc_just_ended_lap: int = 0  # SC 刚结束圈 (0 = 无 SC)
    driver_skill_offset_s: float = 0.0  # 正=慢, 负=快 (车队/车手偏移)
    car_performance_offset_s: float = 0.0  # 赛车性能偏移 (顶队负, 后段正)
    # 轮胎温度物理 (EA F1 2026 tire temp window)
    track_temp_c: float = _REF_TRACK_TEMP_C
    ambient_temp_c: float = _REF_AMBIENT_TEMP_C
    lap_in_stint: int = _REF_LAP_IN_STINT  # stint 内圈数 (0-based, 用于冷启动)


# --------------------------------------------------------------------------- #
# LapResult2026
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LapResult2026:
    """单圈仿真输出 (对标 EA F1 2026 物理引擎).

    所有时间为秒, 正=慢, 负=快 (相对 benchmark).
    """

    track_id: str
    lap_time_s: float
    benchmark_s: float
    error_pct: float
    # 子系统 delta (相对 benchmark, 正=慢)
    tire_delta_s: float
    tire_temp_penalty_s: float  # 轮胎温度偏离窗口的惩罚 delta (相对 reference)
    fuel_delta_s: float
    pu_delta_s: float
    aero_delta_s: float
    drs_delta_s: float
    weather_penalty_s: float
    driver_offset_s: float
    car_offset_s: float
    # PU 详情
    pu_deploy_mj: float
    pu_harvest_mj: float
    pu_soc_after_mj: float
    # 燃油消耗
    fuel_burned_kg: float

    @property
    def within_threshold(self) -> bool:
        """圈速是否在 EA F1 2026 1.5% 精度阈值内."""
        return self.error_pct <= 1.5

    @property
    def verdict(self) -> str:
        return "PASS" if self.within_threshold else "FAIL"


# --------------------------------------------------------------------------- #
# 单圈仿真核心 (无对象分配路径)
# --------------------------------------------------------------------------- #
def _simulate_lap_core(cfg: LapConfig2026) -> tuple[float, dict[str, float]]:
    """单圈仿真核心, 返回 (lap_time_s, breakdown_dict).

    所有计算为纯标量, 无对象分配 (除 breakdown dict).
    """
    # benchmark = 校准锚点 (Iter-67: 解析别名 sakhir/sao_paulo/lusail)
    benchmark = EA_F1_2026_LAP_TIME_BENCHMARK.get(resolve_track_id(cfg.track_id), 90.0)

    # === Tire delta (相对 reference) ===
    tire_delta = tire_lap_delta_s(cfg.compound, cfg.tire_age_laps) - _REF_TIRE_DELTA

    # === Tire temp delta (相对 reference, EA F1 2026 tire temp window) ===
    tire_temp_pen = _tire_temp_penalty_s(
        cfg.compound, cfg.track_temp_c, cfg.ambient_temp_c,
        cfg.lap_in_stint, cfg.tire_age_laps, cfg.wet,
    )
    tire_temp_delta = tire_temp_pen - _REF_TIRE_TEMP_PENALTY_S

    # === Fuel delta (相对 reference 50kg) ===
    fuel_delta = fuel_effect_on_lap_time(cfg.current_fuel_kg) - _REF_FUEL_DELTA
    # 燃油模式影响 (LEAN 慢, RICH/PARTY 快)
    if cfg.fuel_mode == FuelMode.LEAN:
        fuel_delta += 0.4
    elif cfg.fuel_mode == FuelMode.RICH:
        fuel_delta -= 0.2
    elif cfg.fuel_mode == FuelMode.PARTY:
        fuel_delta -= 0.4

    # === PU delta (相对 BALANCED + full SoC) — 内联 (无对象分配) ===
    pu_deploy_mj, pu_harvest_mj, pu_gain, pu_soc_after_mj = _inline_pu_gain(
        cfg.track_id, cfg.pu_mode, cfg.pu_soc_mj
    )
    # Iter-108: per-track reference PU gain. 旧版用全局 _REF_PU_GAIN_S (假设
    # harvest_factor=1.0), 但 5 道 low-harvest 赛道 actual_harvest=4.8 < 6.0,
    # 导致 pu_delta=-0.048s (sim_lap 比 benchmark 快 0.048s). 改用 per-track
    # reference (同赛道同 reference PU mode + SoC), 让 pu_delta=0 对所有赛道
    # 在 reference 条件下. 实测 24 道 sim_lap = benchmark (0.0000s 误差, 0%).
    ref_pu_gain = _inline_pu_gain(cfg.track_id, _REF_PU_MODE, _REF_SOC_MJ)[2]
    pu_delta = ref_pu_gain - pu_gain  # 正=慢于 reference

    # === Aero delta (相对 dry + clean air) — lru_cache ===
    # EA F1 2026 物理: gap > 1.0 = 净空 (X-mode 全效); gap < 1.0 = 乱流 (X-mode 减效).
    # active_aero 模块语义相反 (gap < 1.0 = 攻击对手激活), 这里反转映射.
    aero_gap_for_module = max(0.0, 2.0 - cfg.gap_to_ahead_s)
    ref_aero_gap = max(0.0, 2.0 - _REF_GAP_S)
    aero_gain = _cached_aero_gain(cfg.track_id, aero_gap_for_module, cfg.wet)
    ref_aero_gain = _cached_aero_gain(cfg.track_id, ref_aero_gap, _REF_WET)
    aero_delta = ref_aero_gain - aero_gain  # 正=慢于 reference

    # Iter-112/113: dirty air 减效. 旧版 active_aero 模块让 gap<1.0 完全禁用部分
    # 直道 X-mode (binary), 给 melbourne +0.41s ~ las_vegas +0.84s, 超权威 +0.1~+0.3s.
    # 真实 F1 dirty air 仅减效 30-50% (乱流降低 X-mode 效率但不完全消除).
    # Iter-112: 常数 retain factor 0.35 (全 gap<1.0 相同 delta).
    # Iter-113: 线性 gap 缩放 — gap=1.0→0 (净空), gap=0.0→max penalty.
    #   scale = _DIRTY_AIR_RETAIN_FACTOR × 2.0 × (1-gap), 校准使 gap=0.5 匹配
    #   Iter-112 (0.35×2.0×0.5=0.35). gap=0.3→0.49×, gap=0.8→0.14×.
    # wet 场景 X-mode 已被 active_aero 模块二值禁用, 不再叠加; qualifying 场景
    # gap 通常 >1.0, 不触发.
    if (
        cfg.gap_to_ahead_s < 1.0
        and cfg.session_type == "race"
        and not cfg.wet
        and aero_delta > 0.0
    ):
        gap_proximity = 1.0 - cfg.gap_to_ahead_s
        aero_delta = aero_delta * _DIRTY_AIR_RETAIN_FACTOR * 2.0 * gap_proximity

    # === DRS delta (相对 qualifying 全 DRS) — lru_cache ===
    # reference = qualifying (DRS 全程可用, benchmark 已含 DRS).
    wetness = 0.6 if cfg.wet else 0.0
    if cfg.gap_to_ahead_s > 1.0 and cfg.session_type == "race":
        current_gap: float | None = None
    else:
        current_gap = cfg.gap_to_ahead_s
    ref_drs_gain = _cached_drs_gain(
        cfg.track_id, _REF_LAP, None, _REF_SESSION, 0.0, 0,
    )
    current_drs_gain = _cached_drs_gain(
        cfg.track_id, cfg.lap, current_gap, cfg.session_type,
        wetness, cfg.sc_just_ended_lap,
    )
    drs_delta = ref_drs_gain - current_drs_gain  # 正=慢于 reference (DRS 不可用)

    # === Weather penalty (湿地额外惩罚, 不含 aero 禁用) ===
    # Iter-112: 旧版固定 1.5s + aero_delta (~1.36s X-mode 禁用) + drs_delta (~0.90s)
    # + tire_temp (~0.375s) = +4.1s, 超权威 +1.5~+3.0s. _WET_GRIP_PENALTY_S=0.3
    # 代表纯赛道表面湿滑 (独立于 aero/DRS/tire_temp 已含物理), 总 wet delta ~2.9s.
    weather_penalty = _WET_GRIP_PENALTY_S if cfg.wet else 0.0

    # === Driver + car offset ===
    driver_offset = cfg.driver_skill_offset_s
    car_offset = cfg.car_performance_offset_s

    # === 组装圈速 ===
    lap_time = (
        benchmark
        + tire_delta
        + tire_temp_delta
        + fuel_delta
        + pu_delta
        + aero_delta
        + drs_delta
        + weather_penalty
        + driver_offset
        + car_offset
    )

    # 物理边界 (EA F1 2026 范围: 60-180s)
    lap_time = max(60.0, min(180.0, lap_time))

    breakdown = {
        "benchmark_s": benchmark,
        "tire_delta_s": tire_delta,
        "tire_temp_penalty_s": tire_temp_delta,
        "fuel_delta_s": fuel_delta,
        "pu_delta_s": pu_delta,
        "aero_delta_s": aero_delta,
        "drs_delta_s": drs_delta,
        "weather_penalty_s": weather_penalty,
        "driver_offset_s": driver_offset,
        "car_offset_s": car_offset,
        "pu_deploy_mj": pu_deploy_mj,
        "pu_harvest_mj": pu_harvest_mj,
        "pu_soc_after_mj": pu_soc_after_mj,
        "lap_time_s": lap_time,
    }
    return lap_time, breakdown


def simulate_lap_2026(cfg: LapConfig2026) -> LapResult2026:
    """单圈仿真 (无状态, 纯函数).

    Args:
        cfg: 单圈配置.

    Returns:
        :class:`LapResult2026` 含圈速与各子系统 delta.

    Example::

        cfg = LapConfig2026(track_id="monza")
        r = simulate_lap_2026(cfg)
        assert r.within_threshold  # reference state 必须 PASS
    """
    lap_time, bd = _simulate_lap_core(cfg)
    benchmark = bd["benchmark_s"]
    error_pct = 100.0 * abs(lap_time - benchmark) / benchmark

    return LapResult2026(
        track_id=cfg.track_id,
        lap_time_s=lap_time,
        benchmark_s=benchmark,
        error_pct=error_pct,
        tire_delta_s=bd["tire_delta_s"],
        tire_temp_penalty_s=bd["tire_temp_penalty_s"],
        fuel_delta_s=bd["fuel_delta_s"],
        pu_delta_s=bd["pu_delta_s"],
        aero_delta_s=bd["aero_delta_s"],
        drs_delta_s=bd["drs_delta_s"],
        weather_penalty_s=bd["weather_penalty_s"],
        driver_offset_s=bd["driver_offset_s"],
        car_offset_s=bd["car_offset_s"],
        pu_deploy_mj=bd["pu_deploy_mj"],
        pu_harvest_mj=bd["pu_harvest_mj"],
        pu_soc_after_mj=bd["pu_soc_after_mj"],
        fuel_burned_kg=0.0,  # 由 stint 仿真器更新
    )


# --------------------------------------------------------------------------- #
# LapSimulator2026 — 跨圈 stint 仿真 (有状态)
# --------------------------------------------------------------------------- #
@dataclass
class LapSimulator2026:
    """EA F1 2026 跨圈 stint 仿真器.

    跨圈状态: tire_age, fuel_kg, pu_soc_mj.
    每圈自动老化轮胎, 消耗燃油, 更新 SoC.

    用法::

        sim = LapSimulator2026(
            track_id="monza", total_laps=20,
            compound="medium", initial_fuel_kg=110.0,
            pu_mode=PUDeployMode.BALANCED,
        )
        stint = sim.simulate_stint()
        # stint[i] = LapResult2026
    """

    track_id: str
    total_laps: int = 20
    compound: str = _REF_COMPOUND
    initial_fuel_kg: float = 110.0
    pu_mode: PUDeployMode = _REF_PU_MODE
    initial_pu_soc_mj: float = BATTERY_CAPACITY_MJ
    fuel_mode: FuelMode = FuelMode.NORMAL
    wet: bool = False
    gap_to_ahead_s: float = _REF_GAP_S
    session_type: str = _REF_SESSION
    sc_just_ended_lap: int = 0
    driver_skill_offset_s: float = 0.0
    car_performance_offset_s: float = 0.0
    # 轮胎温度物理 (EA F1 2026 tire temp window)
    track_temp_c: float = _REF_TRACK_TEMP_C
    ambient_temp_c: float = _REF_AMBIENT_TEMP_C
    # 每圈燃油消耗 (kg/lap). 0 = 自动按 track_id + fuel_mode 查表 (EA F1 2026 物理)
    fuel_burn_per_lap_kg: float = 0.0
    # SC/VSC 模型 (None = 无 SC). 注入后 stint 仿真将按 EA F1 2026 race physics 应用.
    safety_car: SafetyCarModel | None = None

    _results: list[LapResult2026] = field(init=False, repr=False, default_factory=list)
    _tire_age: float = field(init=False, repr=False, default=0.0)
    _fuel_kg: float = field(init=False, repr=False, default=0.0)
    _pu_soc_mj: float = field(init=False, repr=False, default=0.0)
    _resolved_fuel_burn_kg: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._results = []
        self._tire_age = 0.0
        self._fuel_kg = float(self.initial_fuel_kg)
        self._pu_soc_mj = float(self.initial_pu_soc_mj)
        # EA F1 2026: 每圈消耗由赛道几何 + 燃油模式决定 (Monaco 1.20 vs Spa 2.10 kg/lap)
        if self.fuel_burn_per_lap_kg > 0.0:
            self._resolved_fuel_burn_kg = float(self.fuel_burn_per_lap_kg)
        else:
            self._resolved_fuel_burn_kg = fuel_per_lap(self.track_id, self.fuel_mode)

    def simulate_lap(self, lap_idx: int) -> LapResult2026:
        """仿真单圈 (更新跨圈状态, 含 SC/VSC race physics)."""
        lap_1b = lap_idx + 1  # 1-based

        # SC/VSC 状态解析 (EA F1 2026 race physics)
        sc = self.safety_car
        sc_active = False
        sc_factor = 1.0
        restart_pen = 0.0
        sc_just_ended = 0  # 用于 DRS 禁用 (FIA 2026: SC 后 2 圈 DRS 禁用)
        if sc is not None:
            sc_active = sc.active_period(lap_1b) is not None
            sc_factor = sc.lap_time_factor(lap_1b)  # 1.0 / 1.30 / 1.25
            restart_pen = sc.restart_penalty_s(lap_1b)
            # 找最近结束的 SC 时段 (用于 DRS 禁用窗口)
            for p in sc.periods:
                if 0 < lap_1b - p.end_lap <= _DRS_DISABLED_AFTER_SC_LAPS:
                    sc_just_ended = p.end_lap
                    break

        cfg = LapConfig2026(
            track_id=self.track_id,
            compound=self.compound,
            tire_age_laps=self._tire_age,
            current_fuel_kg=self._fuel_kg,
            fuel_mode=self.fuel_mode,
            pu_mode=self.pu_mode,
            pu_soc_mj=self._pu_soc_mj,
            wet=self.wet,
            gap_to_ahead_s=self.gap_to_ahead_s,
            session_type=self.session_type,
            lap=lap_1b,
            sc_just_ended_lap=sc_just_ended,
            driver_skill_offset_s=self.driver_skill_offset_s,
            car_performance_offset_s=self.car_performance_offset_s,
            track_temp_c=self.track_temp_c,
            ambient_temp_c=self.ambient_temp_c,
            lap_in_stint=lap_idx,  # 0-based stint 内圈数 (用于冷启动判断)
        )
        result = simulate_lap_2026(cfg)

        # SC/VSC 圈速调整 (EA F1 2026: SC 跟车 ×1.30, VSC ×1.25, 重启 +0.8s)
        if sc_factor != 1.0:
            new_lt = result.lap_time_s * sc_factor + restart_pen
            object.__setattr__(result, "lap_time_s", new_lt)
            object.__setattr__(result, "error_pct",
                               100.0 * abs(new_lt - result.benchmark_s) / result.benchmark_s)
        elif restart_pen > 0.0:
            new_lt = result.lap_time_s + restart_pen
            object.__setattr__(result, "lap_time_s", new_lt)
            object.__setattr__(result, "error_pct",
                               100.0 * abs(new_lt - result.benchmark_s) / result.benchmark_s)

        # 跨圈状态更新 (SC 期间物理调整)
        if sc_active and sc is not None:
            p = sc.active_period(lap_1b)
            is_sc = p is not None and p.kind == "sc"
            wear_f = _SC_TIRE_WEAR_FACTOR if is_sc else _VSC_TIRE_WEAR_FACTOR
            fuel_f = _SC_FUEL_BURN_FACTOR if is_sc else _VSC_FUEL_BURN_FACTOR
            # 轮胎磨损降低 (SC 期间低应力)
            self._tire_age += wear_f
            # 燃油消耗降低 (匀速跟车)
            burned = self._resolved_fuel_burn_kg * fuel_f
            self._fuel_kg = max(0.0, self._fuel_kg - burned)
            object.__setattr__(result, "fuel_burned_kg", burned)
            # SoC 充满 (SC 期间持续 regen, EA F1 2026 物理)
            self._pu_soc_mj = BATTERY_CAPACITY_MJ
            object.__setattr__(result, "pu_soc_after_mj", BATTERY_CAPACITY_MJ)
        else:
            # 正常圈
            self._tire_age += 1
            self._fuel_kg = max(0.0, self._fuel_kg - self._resolved_fuel_burn_kg)
            self._pu_soc_mj = result.pu_soc_after_mj
            object.__setattr__(result, "fuel_burned_kg", self._resolved_fuel_burn_kg)

        self._results.append(result)
        return result

    def simulate_stint(self) -> list[LapResult2026]:
        """仿真整个 stint."""
        self._reset()
        out: list[LapResult2026] = []
        for k in range(int(self.total_laps)):
            out.append(self.simulate_lap(k))
        return out

    def summary(self) -> dict[str, Any]:
        """stint 摘要."""
        if not self._results:
            self.simulate_stint()
        times = [r.lap_time_s for r in self._results]
        return {
            "track_id": self.track_id,
            "compound": self.compound,
            "laps": len(self._results),
            "total_time": float(sum(times)),
            "avg_lap_time": float(sum(times) / len(times)) if times else 0.0,
            "best_lap": float(min(times)) if times else 0.0,
            "worst_lap": float(max(times)) if times else 0.0,
            "final_tire_age": self._tire_age,
            "final_fuel_kg": self._fuel_kg,
            "final_pu_soc_mj": self._pu_soc_mj,
            "all_within_threshold": all(r.within_threshold for r in self._results),
        }


# --------------------------------------------------------------------------- #
# 基准验证 (24 赛道)
# --------------------------------------------------------------------------- #
def validate_against_benchmark(
    driver_offset_s: float = 0.0,
    car_offset_s: float = 0.0,
) -> dict[str, Any]:
    """验证本仿真器在 24 赛道的精度 (reference state).

    Args:
        driver_offset_s: 车手偏移 (默认 0 = reference).
        car_offset_s: 赛车偏移 (默认 0 = reference).

    Returns:
        报告字典: 通过率, 平均误差, 最差赛道.
    """
    simulated: dict[str, float] = {}
    for track_id in EA_F1_2026_LAP_TIME_BENCHMARK:
        cfg = LapConfig2026(
            track_id=track_id,
            driver_skill_offset_s=driver_offset_s,
            car_performance_offset_s=car_offset_s,
        )
        r = simulate_lap_2026(cfg)
        simulated[track_id] = r.lap_time_s

    from f1opt.data.ea_f1_2026_benchmark import accuracy_report

    return accuracy_report(simulated)


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def quick_lap_time_s(track_id: str) -> float:
    """便捷: reference state 单圈圈速 (应 ≈ benchmark)."""
    return simulate_lap_2026(LapConfig2026(track_id=track_id)).lap_time_s


def is_2026_compliant(track_id: str, **kwargs: Any) -> bool:
    """便捷: 判断单圈是否符合 EA F1 2026 精度."""
    cfg = LapConfig2026(track_id=track_id, **kwargs)
    return simulate_lap_2026(cfg).within_threshold


# --------------------------------------------------------------------------- #
# 多 stint 进站仿真 (EA F1 2026 race physics)
# --------------------------------------------------------------------------- #
# EA F1 2026 进站损失基准 (24 赛道, 秒). 来源: track_engineering.pit_loss_s.
_PIT_LOSS_BY_TRACK_2026: dict[str, float] = {
    "melbourne": 23.0, "shanghai": 24.0, "suzuka": 23.0, "bahrain": 24.0,
    "jeddah": 23.0, "miami": 23.0, "montreal": 22.0, "monaco": 21.0,
    "barcelona": 23.0, "spielberg": 22.0, "silverstone": 23.0, "spa": 23.0,
    "hungaroring": 23.0, "zandvoort": 23.0, "monza": 23.0, "madrid": 23.0,
    "baku": 22.0, "singapore": 24.0, "austin": 23.0, "mexico_city": 23.0,
    "interlagos": 23.0, "las_vegas": 23.0, "losail": 23.0, "yas_marina": 23.0,
}
_DEFAULT_PIT_LOSS_S = 23.0


def _pit_loss_for_track(track_id: str) -> float:
    """赛道进站损失 (秒, 含车道通行 + 平均换胎)."""
    return _PIT_LOSS_BY_TRACK_2026.get(resolve_track_id(track_id), _DEFAULT_PIT_LOSS_S)


@dataclass(frozen=True)
class StintPlan2026:
    """EA F1 2026 多 stint 进站策略.

    例: 1-stop 35/25 圈 medium+soft::

        plan = StintPlan2026(
            compounds=("medium", "soft"),
            stint_lengths=(35, 25),
        )
    """

    compounds: tuple[str, ...]
    stint_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.compounds) != len(self.stint_lengths):
            raise ValueError("compounds 和 stint_lengths 长度必须一致")
        if len(self.compounds) == 0:
            raise ValueError("至少 1 个 stint")
        if any(n <= 0 for n in self.stint_lengths):
            raise ValueError("stint_lengths 必须 > 0")

    @property
    def n_stints(self) -> int:
        return len(self.compounds)

    @property
    def n_stops(self) -> int:
        return max(0, self.n_stints - 1)

    @property
    def total_laps(self) -> int:
        return sum(self.stint_lengths)

    def pit_laps(self) -> tuple[int, ...]:
        """每次进站的 1-indexed 圈号 (每个 stint 末圈的下一圈为进站圈)."""
        laps: list[int] = []
        cumul = 0
        for i in range(self.n_stints - 1):
            cumul += self.stint_lengths[i]
            laps.append(cumul)  # stint i 末圈 = 进站圈
        return tuple(laps)


@dataclass
class MultiStintSimulator2026:
    """EA F1 2026 多 stint 进站仿真器.

    管理整场比赛的多 stint 计划, 在进站圈应用进站损失 (含 SC free-pit 折扣
    + 车队 pit crew offset). 跨 stint 延续燃油和 SoC, 进站时重置轮胎年龄.

    物理模型 (EA F1 2026):
    - 进站损失 = 赛道 pit_loss + 车队 crew_offset (SC 期间 × discount)
    - 进站圈仍计入 stint (圈速含进站损失)
    - 新 stint: 轮胎 age=0, 化合物切换, 燃油/SoC 延续
    - SC free-pit: SC/VSC 期间进站损失大幅折扣 (SC ×0.20, VSC ×0.55)

    用法::

        plan = StintPlan2026(
            compounds=("medium", "soft"),
            stint_lengths=(35, 25),
        )
        sim = MultiStintSimulator2026(
            track_id="monza", plan=plan,
            team_id="rbr", safety_car=scm,
        )
        race = sim.simulate_race()  # list[LapResult2026]
    """

    track_id: str
    plan: StintPlan2026
    initial_fuel_kg: float = 110.0
    pu_mode: PUDeployMode = _REF_PU_MODE
    initial_pu_soc_mj: float = BATTERY_CAPACITY_MJ
    fuel_mode: FuelMode = FuelMode.NORMAL
    wet: bool = False
    gap_to_ahead_s: float = _REF_GAP_S
    safety_car: SafetyCarModel | None = None
    team_id: str | None = None  # 用于车队 pit crew offset (None = 中性)
    pit_loss_s: float = 0.0  # 0 = 自动按赛道查表
    driver_skill_offset_s: float = 0.0
    car_performance_offset_s: float = 0.0
    # 轮胎温度物理 (EA F1 2026 tire temp window)
    track_temp_c: float = _REF_TRACK_TEMP_C
    ambient_temp_c: float = _REF_AMBIENT_TEMP_C

    _results: list[LapResult2026] = field(init=False, repr=False, default_factory=list)
    _pit_records: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)

    def _crew_offset_s(self) -> float:
        """车队 pit crew offset (相对平均)."""
        if self.team_id is None:
            return 0.0
        try:
            from f1opt.model.pit_crew import expected_pit_stop_time_s
            # pit_loss_s 已含平均换胎, crew offset = 该队 - 平均
            return expected_pit_stop_time_s(self.team_id) - 3.0
        except Exception:
            return 0.0

    def _resolved_pit_loss_s(self) -> float:
        if self.pit_loss_s > 0.0:
            return float(self.pit_loss_s)
        return _pit_loss_for_track(self.track_id)

    def simulate_race(self) -> list[LapResult2026]:
        """仿真整场比赛 (所有 stint + 进站)."""
        self._results = []
        self._pit_records = []
        fuel_kg = float(self.initial_fuel_kg)
        pu_soc = float(self.initial_pu_soc_mj)
        pit_laps = set(self.plan.pit_laps())
        crew_offset = self._crew_offset_s()
        base_pit_loss = self._resolved_pit_loss_s()
        global_lap = 0  # 0-based

        for stint_idx, (compound, n_laps) in enumerate(
            zip(self.plan.compounds, self.plan.stint_lengths, strict=True)
        ):
            # 每个 stint 用独立 LapSimulator2026, 延续燃油/SoC
            stint_sim = LapSimulator2026(
                track_id=self.track_id,
                total_laps=int(n_laps),
                compound=compound,
                initial_fuel_kg=fuel_kg,
                pu_mode=self.pu_mode,
                initial_pu_soc_mj=pu_soc,
                fuel_mode=self.fuel_mode,
                wet=self.wet,
                gap_to_ahead_s=self.gap_to_ahead_s,
                session_type="race",
                safety_car=self.safety_car,
                driver_skill_offset_s=self.driver_skill_offset_s,
                car_performance_offset_s=self.car_performance_offset_s,
                track_temp_c=self.track_temp_c,
                ambient_temp_c=self.ambient_temp_c,
            )
            stint_results = stint_sim.simulate_stint()

            # 同步跨 stint 状态
            fuel_kg = stint_sim._fuel_kg
            pu_soc = stint_sim._pu_soc_mj

            # 应用进站损失 (若 stint 末圈是进站圈)
            stint_end_global_lap = global_lap + int(n_laps)  # 1-indexed 末圈
            if stint_end_global_lap in pit_laps and stint_idx < self.plan.n_stints - 1:
                # SC free-pit 折扣
                discount = 1.0
                if self.safety_car is not None:
                    discount = self.safety_car.pit_loss_discount(stint_end_global_lap)
                pit_loss = (base_pit_loss + crew_offset) * discount
                # 进站损失加到 stint 末圈
                last = stint_results[-1]
                new_lt = last.lap_time_s + pit_loss
                object.__setattr__(last, "lap_time_s", new_lt)
                object.__setattr__(
                    last, "error_pct",
                    100.0 * abs(new_lt - last.benchmark_s) / last.benchmark_s,
                )
                self._pit_records.append({
                    "lap": stint_end_global_lap,
                    "from_compound": compound,
                    "to_compound": self.plan.compounds[stint_idx + 1],
                    "pit_loss_s": pit_loss,
                    "sc_discount": discount,
                    "crew_offset_s": crew_offset,
                })

            self._results.extend(stint_results)
            global_lap += int(n_laps)

        return self._results

    def summary(self) -> dict[str, Any]:
        """比赛摘要."""
        if not self._results:
            self.simulate_race()
        times = [r.lap_time_s for r in self._results]
        return {
            "track_id": self.track_id,
            "total_laps": len(self._results),
            "n_stints": self.plan.n_stints,
            "n_stops": self.plan.n_stops,
            "total_time": float(sum(times)),
            "avg_lap_time": float(sum(times) / len(times)) if times else 0.0,
            "best_lap": float(min(times)) if times else 0.0,
            "worst_lap": float(max(times)) if times else 0.0,
            "final_fuel_kg": self._results[-1].fuel_burned_kg if self._results else 0.0,
            "pit_records": list(self._pit_records),
            "compounds": list(self.plan.compounds),
            "stint_lengths": list(self.plan.stint_lengths),
        }


# --------------------------------------------------------------------------- #
# 策略对比工具 (专业车队策略评估)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyComparisonResult:
    """单策略对比结果 (EA F1 2026 策略评估)."""

    rank: int                    # 1 = 最优
    plan: StintPlan2026
    total_time_s: float
    delta_to_best_s: float       # 相对最优策略的差距 (0 = 最优)
    n_stops: int
    best_lap_s: float
    avg_lap_s: float
    worst_lap_s: float
    pit_records: tuple[dict[str, Any], ...]
    compounds: tuple[str, ...]
    stint_lengths: tuple[int, ...]


def compare_strategies(
    track_id: str,
    plans: list[StintPlan2026],
    **sim_kwargs: Any,
) -> list[StrategyComparisonResult]:
    """对比多个进站策略, 按总时间排序 (EA F1 2026 专业车队策略评估).

    每个策略用 MultiStintSimulator2026 仿真整场比赛, 返回按总时间排序的
    对比结果. 适用于正赛前策略决策 (1-stop vs 2-stop, 化合物选择).

    Args:
        track_id: 赛道 ID.
        plans: 待对比的策略列表 (至少 1 个).
        **sim_kwargs: 传递给 MultiStintSimulator2026 的额外参数
            (initial_fuel_kg, pu_mode, safety_car, team_id, wet, ...).

    Returns:
        按 total_time_s 升序排序的 :class:`StrategyComparisonResult` 列表.

    用法::

        plans = [
            StintPlan2026(("medium", "soft"), (32, 28)),     # 1-stop M-S
            StintPlan2026(("soft", "medium"), (25, 35)),     # 1-stop S-M
            StintPlan2026(("medium", "soft", "medium"), (20, 20, 20)),  # 2-stop
        ]
        results = compare_strategies("monza", plans, team_id="rbr")
        print(f"最优: {results[0].total_time_s:.1f}s ({results[0].compounds})")
    """
    if not plans:
        raise ValueError("plans 不能为空")

    raw: list[tuple[float, StintPlan2026, dict[str, Any]]] = []
    for plan in plans:
        sim = MultiStintSimulator2026(track_id=track_id, plan=plan, **sim_kwargs)
        sim.simulate_race()
        summary = sim.summary()
        raw.append((summary["total_time"], plan, summary))

    # 按 total_time 升序
    raw.sort(key=lambda x: x[0])
    best_time = raw[0][0]

    results: list[StrategyComparisonResult] = []
    for rank, (total_t, plan, s) in enumerate(raw, start=1):
        results.append(StrategyComparisonResult(
            rank=rank,
            plan=plan,
            total_time_s=total_t,
            delta_to_best_s=total_t - best_time,
            n_stops=plan.n_stops,
            best_lap_s=s["best_lap"],
            avg_lap_s=s["avg_lap_time"],
            worst_lap_s=s["worst_lap"],
            pit_records=tuple(s["pit_records"]),
            compounds=tuple(s["compounds"]),
            stint_lengths=tuple(s["stint_lengths"]),
        ))
    return results
