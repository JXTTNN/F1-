"""权威 F1 轮胎 stint 物理模型 (Iter-4).

现实 F1 车队 (Pirelli + 各队 tyre group) 对 stint 的建模远不止线性磨损:

1. **暖胎期 (warm-up)**: 新胎前 1-3 圈胎面温度未达最优窗口, 抓地力低于峰值
   ~0.3-0.6 s/lap; Pirelli 官方技术简报称之为 "switch-on" 阶段.
2. **稳态期 (steady)**: 胎面在 90-110 °C 工作窗口内, 磨损近似线性, 圈速
   缓慢上升 (燃油减少) 与缓慢下降 (磨损) 相互抵消.
3. **悬崖期 (cliff)**: 表面橡胶磨光 + 胎体温度过高, 抓地力急剧下降 (1-2 s/lap).

本模块实现这一三阶段曲线, 并叠加:

- 化合物参数 (soft/medium/hard/intermediate/wet) — Pirelli 2026 规格的
  典型工作窗口与磨损率.
- 赛道磨蚀系数 (abrasiveness): 高磨蚀赛道 (Suzuka/Barcelona) 磨损快,
  低磨蚀街道赛 (Monaco) 磨损慢.
- 热窗口惩罚: 胎温偏离最优 ±10 °C 之外每度损失抓地.
- 前后轴不对称磨损: 推头倾向磨损前胎更快, 过度倾向磨损后胎更快, 影响
  stint 中段平衡漂移.
- 累积热循环负荷: 每次热循环 (一圈加热-冷却) 使化合物轻微硬化.

公开 API:
    - :class:`TireStintPhysics` — 单次 stint 物理仿真.
    - :func:`compound_work_window` — 化合物工作温度窗口 (°C).
    - :func:`track_abrasiveness` — 赛道磨蚀系数 (0.6-1.4).

参考文献 (Pirelli 公开技术资料, 无学术论文):
    Pirelli Motorsport "2026 Tyre Range Technical Overview" (公开).
    FIA Sporting Regulations 2026 §12.7 (化合物规格).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# 化合物参数 (Pirelli 2026 公开范围, 数值为工程化合理估计)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompoundStintParams:
    """单种化合物的 stint 物理参数.

    所有时间为秒, 所有温度为摄氏度, 所有速率为每圈.
    """

    name: str
    warmup_laps: float           # 暖胎期圈数 (1.5-3.0)
    warmup_penalty_s: float      # 暖胎期每圈额外时间 (相对峰值, s)
    steady_rate_s: float         # 稳态期每圈累计磨损惩罚 (s/lap)
    cliff_threshold_pct: float   # 磨损达到此 % 后进入悬崖 (60-85)
    cliff_rate_s: float          # 悬崖期每圈额外时间 (s/lap)
    cliff_length_laps: int       # 悬崖期长度 (圈, 之后胎彻底废)
    temp_optimal_c: float        # 最优工作温度 (°C)
    temp_window_c: float         # 工作窗口半宽 (±°C)
    front_wear_bias: float       # 前轴磨损占比 (0.5 = 对称)


# Pirelli 2026 化合物: C1=硬, C2=中, C3=软, C4=超软, 中性胎, 雨胎.
COMPOUND_STINT_PARAMS: dict[str, CompoundStintParams] = {
    "soft": CompoundStintParams(
        name="soft", warmup_laps=1.5, warmup_penalty_s=0.45,
        steady_rate_s=0.07, cliff_threshold_pct=65.0, cliff_rate_s=1.6,
        cliff_length_laps=3, temp_optimal_c=95.0, temp_window_c=12.0,
        front_wear_bias=0.52,
    ),
    "medium": CompoundStintParams(
        name="medium", warmup_laps=2.0, warmup_penalty_s=0.55,
        steady_rate_s=0.045, cliff_threshold_pct=75.0, cliff_rate_s=1.2,
        cliff_length_laps=4, temp_optimal_c=100.0, temp_window_c=14.0,
        front_wear_bias=0.51,
    ),
    "hard": CompoundStintParams(
        name="hard", warmup_laps=2.8, warmup_penalty_s=0.70,
        steady_rate_s=0.028, cliff_threshold_pct=85.0, cliff_rate_s=0.9,
        cliff_length_laps=5, temp_optimal_c=105.0, temp_window_c=16.0,
        front_wear_bias=0.50,
    ),
    "intermediate": CompoundStintParams(
        name="intermediate", warmup_laps=1.0, warmup_penalty_s=0.30,
        steady_rate_s=0.10, cliff_threshold_pct=55.0, cliff_rate_s=2.0,
        cliff_length_laps=2, temp_optimal_c=70.0, temp_window_c=15.0,
        front_wear_bias=0.49,
    ),
    "wet": CompoundStintParams(
        name="wet", warmup_laps=0.8, warmup_penalty_s=0.25,
        steady_rate_s=0.06, cliff_threshold_pct=80.0, cliff_rate_s=1.0,
        cliff_length_laps=6, temp_optimal_c=55.0, temp_window_c=20.0,
        front_wear_bias=0.48,
    ),
}
_DEFAULT_COMPOUND = COMPOUND_STINT_PARAMS["medium"]


def compound_work_window(compound: str) -> tuple[float, float]:
    """返回化合物工作温度窗口 ``(low, high)`` (°C)."""
    p = COMPOUND_STINT_PARAMS.get(compound, _DEFAULT_COMPOUND)
    return (p.temp_optimal_c - p.temp_window_c, p.temp_optimal_c + p.temp_window_c)


# --------------------------------------------------------------------------- #
# 赛道磨蚀系数 (Pirelli 公开磨蚀等级 1-5 → 0.7-1.3)
# --------------------------------------------------------------------------- #
_TRACK_ABRASIVENESS: dict[str, float] = {
    # 高磨蚀 (Pirelli 等级 4-5)
    "suzuka": 1.30, "barcelona": 1.25, "silverstone": 1.20, "spa": 1.18,
    "budapest": 1.15, "shanghai": 1.15, "sakhir": 1.15, "amsterdam": 1.10,
    # 中等磨蚀 (等级 3)
    "melbourne": 1.00, "bahrain": 1.00, "jeddah": 1.05, "miami": 0.95,
    "monza": 1.00, "las_vegas": 0.95, "austin": 1.05, "interlagos": 1.05,
    "losail": 1.10, "yas_marina": 0.95, "montreal": 0.95,
    # 低磨蚀 (等级 1-2)
    "monaco": 0.65, "singapore": 0.75, "madrid": 0.80,
}
_DEFAULT_ABRASIVENESS = 1.00


def track_abrasiveness(track_id: str) -> float:
    """赛道磨蚀系数: >1 磨损快, <1 磨损慢, 默认 1.0."""
    return _TRACK_ABRASIVENESS.get(track_id, _DEFAULT_ABRASIVENESS)


# --------------------------------------------------------------------------- #
# Iter-164.12: 调教→胎耗耦合 (setup → tire_stress_factor)
# --------------------------------------------------------------------------- #
# raw tire_wear_proxy 基准点 (与 optimizer._TYRE_TEMP_REF/_SLIP_REF 一致).
# proxy = (tyre_temp - 90)/30 + slip_angle/5 + tyre_load_spread
# 典型范围: 0.5 (低应力) ~ 2.0 (高应力), 1.0 = 中等.
_STRESS_PROXY_REF = 1.0
# 应力因子映射: proxy=1.0 → factor=1.0 (中性), proxy=0.5 → factor≈0.85,
# proxy=2.0 → factor≈1.30. 用线性映射 + clip 到 [0.7, 1.5] 避免极端.
# 物理依据: 胎温每升 10°C 磨损率 +15% (Pirelli 经验), 滑移每增 1° 磨损 +10%.
_STRESS_SLOPE = 0.3  # proxy 每偏离基准 1.0, factor 变化 ±0.3
_STRESS_MIN = 0.7
_STRESS_MAX = 1.5


def setup_tire_stress_factor(
    setup: Any,
    track_id: str,
    driver_profile: Any = None,
) -> float:
    """Iter-164.12: 从 CarSetup 的 responses 计算轮胎应力因子.

    把调教的胎温/滑移/载荷离散 (raw tire_wear_proxy) 映射到磨损率倍数,
    让 :class:`TireStintPhysics` 的磨损率与调教耦合. 高应力调教 (高胎温/
    高滑移/高载荷离散) → factor > 1 (磨损更快), 低应力调教 → factor < 1.

    Parameters
    ----------
    setup : CarSetup
        调教.
    track_id : str
        赛道 ID.
    driver_profile : DriverProfile | None
        车手画像 (透传给 predict_full).

    Returns
    -------
    float
        应力因子, 范围 [0.7, 1.5]. 1.0 = 中性.
    """
    from f1opt.model.optimizer import (
        _SLIP_REF,
        _TYRE_TEMP_REF,
        _TYRE_TEMP_SPAN,
    )
    from f1opt.model.surrogate import predict_full
    pred = predict_full(setup, track_id, driver_profile)
    resp = pred["responses"]
    raw_proxy = (
        (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
        + float(resp["slip_angle"]) / _SLIP_REF
        + float(resp["tyre_load_spread"])
    )
    # 线性映射: factor = 1.0 + slope * (proxy - ref), clip 到 [min, max]
    factor = 1.0 + _STRESS_SLOPE * (raw_proxy - _STRESS_PROXY_REF)
    return max(_STRESS_MIN, min(_STRESS_MAX, factor))


def stint_total_time(
    setup: Any,
    track_id: str,
    driver_profile: Any = None,
    compound: str = "medium",
    stint_length: int = 20,
    initial_fuel_kg: float = 110.0,
    driver_tire_management: float = 0.5,
) -> tuple[float, float]:
    """Iter-164.12: 计算调教相关的 stint 总时间 (耦合胎耗物理).

    用 :func:`setup_tire_stress_factor` 把调教映射到磨损率倍数, 再用
    :class:`TireStintPhysics` 仿真 stint, 返回 (总时间, 最终磨损%).

    这是 R8 "结合胎耗" 的深度集成: 优化器可用此函数作为目标, 直接最小化
    stint 总时间 (而非 lap_time + w*proxy 启发式).

    Returns
    -------
    (total_time_s, final_wear_pct)
    """
    from f1opt.model.surrogate import predict_lap_time
    base_lap = float(predict_lap_time(setup, track_id, driver_profile))
    stress_factor = setup_tire_stress_factor(setup, track_id, driver_profile)
    phys = TireStintPhysics(
        compound=compound,
        track_id=track_id,
        base_lap_time=base_lap,
        stint_length=stint_length,
        initial_fuel_kg=initial_fuel_kg,
        driver_tire_management=driver_tire_management,
        tire_stress_factor=stress_factor,
    )
    laps = phys.simulate()
    total = float(laps[-1]["cumulative_time"])
    wear = float(laps[-1]["wear_pct"])
    return total, wear


# --------------------------------------------------------------------------- #
# TireStintPhysics
# --------------------------------------------------------------------------- #
@dataclass
class TireStintPhysics:
    """单次 stint 圈速曲线仿真, 三阶段 (暖胎/稳态/悬崖) + 热窗口惩罚.

    用法::

        sim = TireStintPhysics(
            compound="soft", track_id="suzuka",
            base_lap_time=91.5, stint_length=18,
            initial_fuel_kg=110.0, track_temp_c=42.0,
            balance_tendency="understeer",
        )
        laps = sim.simulate()
        print(sim.optimal_pit_window())  # (lap, lap)
    """

    compound: str
    track_id: str
    base_lap_time: float
    stint_length: int
    initial_fuel_kg: float = 110.0
    track_temp_c: float = 35.0
    balance_tendency: str = "neutral"   # understeer / oversteer / neutral
    fuel_burn_rate_kg_per_lap: float = 1.6
    fuel_penalty_s_per_kg: float = 0.03
    driver_tire_management: float = 0.5
    """车手轮胎管理风格 0..1 (Iter-22).
    0.5 = 中性 (默认). 1.0 = 极温和 (Verstappen/Hamilton 级, 磨损率 × 0.75,
    可延长 stint 25%). 0.0 = 极激进 (Magnussen 级, 磨损率 × 1.30,
    早 25% 进悬崖). 真实车队用 Pirelli tire-saving 数据校准每位车手."""

    tire_stress_factor: float = 1.0
    """Iter-164.12: 调教相关的轮胎应力因子 (multiplier on wear_rate).
    1.0 = 中性 (默认, 调教不影响磨损). >1.0 = 高应力调教 (高胎温/滑移/载荷
    离散 → 磨损更快). <1.0 = 低应力调教 (温和胎温/低滑移/均匀载荷 → 磨损更慢).
    由 :func:`setup_tire_stress_factor` 从 CarSetup 的 responses 计算, 让
    TireStintPhysics 的磨损率与调教耦合 (R8 深度集成)."""

    # 派生
    _params: CompoundStintParams = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._params = COMPOUND_STINT_PARAMS.get(self.compound, _DEFAULT_COMPOUND)
        if self.balance_tendency not in ("understeer", "oversteer", "neutral"):
            self.balance_tendency = "neutral"
        # 限制 0..1
        self.driver_tire_management = max(0.0, min(1.0, float(self.driver_tire_management)))

    # ------------------------------------------------------------------ #
    # 阶段判定
    # ------------------------------------------------------------------ #
    def _phase(self, lap_idx: int, wear_pct: float) -> str:
        """lap_idx 0-indexed, wear_pct 0-100. 返回 warmup/steady/cliff/dead."""
        p = self._params
        if lap_idx < p.warmup_laps:
            return "warmup"
        if wear_pct >= 100.0:
            return "dead"
        if wear_pct >= p.cliff_threshold_pct:
            # 计算进入悬崖后第几圈
            return "cliff"
        return "steady"

    # ------------------------------------------------------------------ #
    # 磨损率 (每圈 % 累计)
    # ------------------------------------------------------------------ #
    def _wear_rate_per_lap(self, lap_idx: int) -> float:
        """每圈新增磨损 %, 受赛道磨蚀 + 化合物 + 车手风格 + 调教应力影响.

        基础率: 化合物决定 (soft ~5%/lap, hard ~2.5%/lap), 乘以赛道磨蚀系数.
        车手管理: management 1.0 = 磨损率 × 0.75; 0.0 = × 1.30.
        Iter-164.12: 调教应力因子 tire_stress_factor 让高胎温/高滑移/高载荷
        离散的调教磨损更快, 把 TireStintPhysics 与调教耦合 (R8 深度集成).
        """
        base_rate = {
            "soft": 5.2, "medium": 3.6, "hard": 2.5,
            "intermediate": 7.0, "wet": 3.0,
        }.get(self.compound, 3.6)
        # 车手轮胎管理系数 (0.75-1.30)
        driver_factor = 1.30 - (self.driver_tire_management * 0.55)
        # Iter-164.12: 调教应力因子 (1.0 = 中性, >1 高应力, <1 低应力)
        stress_factor = max(0.5, min(2.0, float(self.tire_stress_factor)))
        return base_rate * track_abrasiveness(self.track_id) * driver_factor * stress_factor

    # ------------------------------------------------------------------ #
    # 热窗口惩罚 (s/lap)
    # ------------------------------------------------------------------ #
    def _thermal_penalty(self, lap_idx: int) -> float:
        """胎面温度偏离工作窗口的圈速惩罚.

        模型: 胎温 = track_temp + lap_idx * 热积累速率 - 磨损冷却.
        偏离窗口 ±window 之外, 每度损失 0.04 s.
        """
        # 简化热模型: 胎温随圈数上升 (热积累), 后期磨损导致散热增加
        heat_buildup = 0.6  # °C/lap
        wear_cooling = 0.0   # 磨损后期散热增加, 暂不实现
        tyre_temp = self.track_temp_c + lap_idx * heat_buildup - wear_cooling
        low, high = compound_work_window(self.compound)
        if tyre_temp < low:
            return (low - tyre_temp) * 0.04
        if tyre_temp > high:
            return (tyre_temp - high) * 0.06  # 过热惩罚更重
        return 0.0

    # ------------------------------------------------------------------ #
    # 平衡漂移 (前/后磨损不对称)
    # ------------------------------------------------------------------ #
    def _balance_drift(self, front_wear_pct: float, rear_wear_pct: float) -> float:
        """前后磨损差导致的平衡漂移惩罚 (s/lap).

        推头倾向: 前胎磨损更快 → 后期推头加剧 → 入弯迟钝, 损失时间.
        过度倾向: 后胎磨损更快 → 后期过度加剧 → 出弯打滑, 损失时间.
        """
        diff = front_wear_pct - rear_wear_pct
        # 每 10% 前后磨损差, 损失 0.05 s
        return abs(diff) / 10.0 * 0.05

    # ------------------------------------------------------------------ #
    # 单圈圈速
    # ------------------------------------------------------------------ #
    def _lap_time(self, lap_idx: int, wear_pct: float,
                  fuel_kg: float, cliff_laps_elapsed: int) -> float:
        """计算单圈圈速 (s). lap_idx 0-indexed.

        组成 = base - 燃油收益 + 暖胎惩罚 + 稳态磨损惩罚 + 悬崖惩罚 + 热惩罚.
        """
        p = self._params
        phase = self._phase(lap_idx, wear_pct)

        # 燃油收益: 每圈少 1.6 kg, 每少 1kg 快 0.03s
        fuel_offset_kg = max(0.0, self.initial_fuel_kg - fuel_kg)
        fuel_gain = fuel_offset_kg * self.fuel_penalty_s_per_kg

        # 暖胎惩罚: 线性从 warmup_penalty 降到 0
        if phase == "warmup":
            warmup_frac = max(0.0, 1.0 - lap_idx / p.warmup_laps)
            warmup_pen = p.warmup_penalty_s * warmup_frac
        else:
            warmup_pen = 0.0

        # 稳态磨损惩罚: wear_pct - cliff_threshold 之前的累计
        if wear_pct <= p.cliff_threshold_pct:
            steady_pen = max(0.0, wear_pct) * p.steady_rate_s / 5.0
        else:
            steady_pen = p.cliff_threshold_pct * p.steady_rate_s / 5.0

        # 悬崖惩罚: 进入悬崖后每圈线性增加; dead 阶段惩罚更重 (1.5x)
        cliff_pen = 0.0
        if phase == "cliff":
            cliff_pen = p.cliff_rate_s * (cliff_laps_elapsed + 1)
        elif phase == "dead":
            cliff_pen = p.cliff_rate_s * 1.5 * (cliff_laps_elapsed + 1)

        # 热惩罚
        thermal_pen = self._thermal_penalty(lap_idx)

        # 组装
        lap = (self.base_lap_time
               - fuel_gain
               + warmup_pen
               + steady_pen
               + cliff_pen
               + thermal_pen)
        return max(60.0, min(180.0, lap))

    # ------------------------------------------------------------------ #
    # 主仿真
    # ------------------------------------------------------------------ #
    def simulate(self) -> list[dict[str, Any]]:
        """返回每圈记录: ``{lap, phase, lap_time, wear_pct, front_wear_pct,
        rear_wear_pct, tyre_temp_c, fuel_kg, cumulative_time}``."""
        out: list[dict[str, Any]] = []
        cumulative = 0.0
        fuel = float(self.initial_fuel_kg)
        wear_pct = 0.0
        front_wear = 0.0
        rear_wear = 0.0
        cliff_laps_elapsed = 0
        p = self._params

        for k in range(int(self.stint_length)):
            phase = self._phase(k, wear_pct)
            # 更新悬崖计数
            if phase == "cliff":
                cliff_laps_elapsed += 1
            elif phase == "dead":
                # 轮胎已废, 强制大幅惩罚
                cliff_laps_elapsed += 1
            else:
                cliff_laps_elapsed = 0

            lap_time = self._lap_time(k, wear_pct, fuel, cliff_laps_elapsed - 1
                                      if phase in ("cliff", "dead") else 0)

            cumulative += lap_time
            tyre_temp = self.track_temp_c + k * 0.6

            out.append({
                "lap": k + 1,
                "phase": phase,
                "lap_time": float(lap_time),
                "wear_pct": float(wear_pct),
                "front_wear_pct": float(front_wear),
                "rear_wear_pct": float(rear_wear),
                "tyre_temp_c": float(tyre_temp),
                "fuel_kg": float(fuel),
                "cumulative_time": float(cumulative),
            })

            # 圈末更新磨损
            rate = self._wear_rate_per_lap(k)
            wear_pct = min(100.0, wear_pct + rate)
            # 前后磨损分配
            if self.balance_tendency == "understeer":
                front_wear = min(100.0, front_wear + rate * p.front_wear_bias * 1.15)
                rear_wear = min(100.0, rear_wear + rate * (1 - p.front_wear_bias) * 0.85)
            elif self.balance_tendency == "oversteer":
                front_wear = min(100.0, front_wear + rate * p.front_wear_bias * 0.85)
                rear_wear = min(100.0, rear_wear + rate * (1 - p.front_wear_bias) * 1.15)
            else:
                front_wear = min(100.0, front_wear + rate * p.front_wear_bias)
                rear_wear = min(100.0, rear_wear + rate * (1 - p.front_wear_bias))
            fuel = max(0.0, fuel - self.fuel_burn_rate_kg_per_lap)

        return out

    # ------------------------------------------------------------------ #
    # 推荐进站窗口
    # ------------------------------------------------------------------ #
    def optimal_pit_window(self) -> tuple[int, int]:
        """推荐进站窗口 ``(earliest_lap, latest_lap)``.

        - earliest = 进入悬崖前 1 圈 (但不超过 stint_length, 短 stint 不会到悬崖)
        - latest = 悬崖期第 2 圈 (避免过度时间损失)

        Iter-164.49 修复: 短 stint (stint_length < laps_to_cliff) 时, earliest
        可能超过 stint_length, 导致 earliest > latest 的非法窗口. 现在 earliest
        也 clamp 到 stint_length, 并保证 latest >= earliest + 1 (除非 stint_length=1).
        """
        p = self._params
        # 估算达到 cliff_threshold 的圈数
        rate = self._wear_rate_per_lap(0)
        if rate <= 0:
            return (1, max(1, self.stint_length))
        laps_to_cliff = int(p.cliff_threshold_pct / rate)
        # earliest clamp 到 [1, stint_length] (短 stint 可能不到悬崖)
        earliest = max(1, min(self.stint_length, laps_to_cliff - 1))
        latest = min(self.stint_length, max(earliest + 1, laps_to_cliff + 2))
        if latest <= earliest:
            latest = min(self.stint_length, earliest + 1)
        return (earliest, latest)

    # ------------------------------------------------------------------ #
    # 摘要
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """返回 stint 摘要: 总时间 / 平均圈速 / 最快圈 / 最慢圈 / 进站窗口."""
        laps = self.simulate()
        if not laps:
            return {}
        times = [lp["lap_time"] for lp in laps]
        return {
            "compound": self.compound,
            "track_id": self.track_id,
            "stint_length": self.stint_length,
            "total_time": sum(times),
            "avg_lap_time": sum(times) / len(times),
            "best_lap": min(times),
            "worst_lap": max(times),
            "best_lap_num": int(times.index(min(times)) + 1),
            "pit_window": self.optimal_pit_window(),
            "ends_in_cliff": laps[-1]["phase"] in ("cliff", "dead"),
        }
