"""F1 单圈综合仿真器 (Iter-8): 整合所有子物理模型.

真实 F1 车队 (Mercedes AMG Petronas / Red Bull Racing 等) 的 "lap time
simulator" 把所有子系统模型 (aero / tyre / ERS / brake / suspension /
powertrain / driver) 耦合到一起, 单圈输出考虑所有相互影响.

本模块整合以下已有模型到统一仿真:

- :class:`f1opt.model.aerodynamics.AerodynamicsModel` — 下压力/阻力/DRS.
- :class:`f1opt.model.tire_stint.TireStintPhysics` — 三阶段轮胎磨损.
- :class:`f1opt.model.ers_model.ERSDeploymentModel` — ERS 部署/回收.
- :class:`f1opt.model.brake_model.BrakeModel` — 刹车偏置/温度/磨损.
- :class:`f1opt.model.surrogate.track_prior` — 赛道圈速先验.

单圈圈速公式::

    lap_time = base_prior
             + aero_drag_loss - aero_corner_gain - drs_gain
             - ers_net_gain
             + brake_penalty
             + tire_wear_penalty (来自 TireStintPhysics)
             + suspension_penalty
             + driver_penalty (基于车手画像)
             - fuel_gain (燃油消耗)

公开 API:
    - :class:`LapTimeSimulator` — 单圈仿真 (跨圈状态传递).
    - :func:`simulate_lap` — 模块级便捷函数.
    - :func:`simulate_stint` — 多圈仿真 (整合所有模型).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from f1opt.data.setup_schema import CarSetup
from f1opt.model.aerodynamics import AerodynamicsModel
from f1opt.model.brake_model import BrakeModel
from f1opt.model.ers_model import ERSDeploymentModel
from f1opt.model.surrogate import (
    track_avg_speed,
    track_prior,
)
from f1opt.model.tire_stint import TireStintPhysics
from f1opt.model.weather import WeatherModel

# --------------------------------------------------------------------------- #
# 默认参数
# --------------------------------------------------------------------------- #
_DEFAULT_FUEL_BURN_KG_PER_LAP = 1.6
_DEFAULT_FUEL_PENALTY_S_PER_KG = 0.03
_DRIVER_NEUTRAL_AGGRESSION = 0.5


# --------------------------------------------------------------------------- #
# LapTimeSimulator
# --------------------------------------------------------------------------- #
@dataclass
class LapTimeSimulator:
    """综合单圈仿真, 整合 aero/tyre/ERS/brake 各子系统.

    跨圈状态: 轮胎磨损 %, ERS SoC, 刹车温度, 燃油量.

    用法::

        sim = LapTimeSimulator(
            setup=DEFAULT_SETUP, track_id="monaco",
            compound="soft", driver_aggression=0.7,
            initial_fuel_kg=110.0, brake_temp_c=450.0,
        )
        lap = sim.simulate_lap(lap_idx=0)        # 第 1 圈
        stint = sim.simulate_stint(laps=15)       # 15 圈 stint
    """

    setup: CarSetup
    track_id: str
    compound: str = "medium"
    driver_aggression: float = _DRIVER_NEUTRAL_AGGRESSION
    driver_smoothness: float = 0.5
    driver_consistency: float = 0.5
    initial_fuel_kg: float = 110.0
    brake_temp_c: float = 450.0
    brake_front_fraction: float = 0.56
    brake_cooling_duct_area_mm2: float = 100.0
    brake_energy_kj_per_lap: float = 300.0
    ers_mode: str = "balanced"
    ers_initial_soc: float = 0.5
    track_temp_c: float = 35.0
    balance_tendency: str = "neutral"
    fuel_burn_rate_kg_per_lap: float = _DEFAULT_FUEL_BURN_KG_PER_LAP
    weather: WeatherModel | None = None
    driver_tire_management: float = 0.5
    """车手轮胎管理风格 0..1 (Iter-22). 传递给 TireStintPhysics."""
    car_performance_offset_s: float = 0.0
    """车队赛车性能偏移 s/lap (Iter-36).
    负 = 快于基准 (顶队 RBR/MCL ~ -0.6), 正 = 慢于基准 (后段 +0.8).
    来自 :func:`f1opt.data.teams_2026.pace_offset_for_team`.
    """
    driver_track_affinity_s: float = 0.0
    """车手-赛道亲和度 s/lap (Iter-38).
    正 = 车手擅长该赛道 (圈速更快), 负 = 不擅长.
    来自 :func:`f1opt.data.driver_track_affinity.driver_track_affinity`.
    """

    # 子模型 (惰性构建, 用 _build_models)
    _aero: AerodynamicsModel = field(init=False, repr=False)
    _tire: TireStintPhysics = field(init=False, repr=False)
    _ers: ERSDeploymentModel = field(init=False, repr=False)
    _brake: BrakeModel = field(init=False, repr=False)
    _state: dict[str, Any] = field(init=False, repr=False)
    _tire_cache: list | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.driver_aggression = max(0.0, min(1.0, float(self.driver_aggression)))
        self.driver_smoothness = max(0.0, min(1.0, float(self.driver_smoothness)))
        self.driver_consistency = max(0.0, min(1.0, float(self.driver_consistency)))
        self._build_models()
        self._tire_cache = None
        self._state = {
            "tire_wear_pct": 0.0,
            "front_wear_pct": 0.0,
            "rear_wear_pct": 0.0,
            "fuel_kg": float(self.initial_fuel_kg),
            "brake_temp_c": float(self.brake_temp_c),
            "ers_soc": float(self.ers_initial_soc),
            "lap_idx": 0,
        }

    def _build_models(self) -> None:
        """从 setup 构建子模型."""
        sv = self.setup.to_vector()
        # CarSetup.to_vector() 顺序见 setup_schema, 假设前几项:
        # front_wing, rear_wing, ride_height_front, ride_height_rear, ...
        # 用安全 fallback.
        front_wing = float(sv[0]) if len(sv) > 0 else 0.5
        rear_wing = float(sv[1]) if len(sv) > 1 else 0.5
        # ride height 不一定在 setup 中, 默认值
        rhf = 22.0
        rhr = 32.0

        self._aero = AerodynamicsModel(
            track_id=self.track_id,
            front_wing=front_wing,
            rear_wing=rear_wing,
            ride_height_front_mm=rhf,
            ride_height_rear_mm=rhr,
        )
        self._tire = TireStintPhysics(
            compound=self.compound,
            track_id=self.track_id,
            base_lap_time=track_prior(self.track_id, self.setup),
            stint_length=999,   # 单圈接单圈, 不限制
            initial_fuel_kg=self.initial_fuel_kg,
            track_temp_c=self.track_temp_c,
            balance_tendency=self.balance_tendency,
            fuel_burn_rate_kg_per_lap=self.fuel_burn_rate_kg_per_lap,
            driver_tire_management=self.driver_tire_management,
        )
        self._ers = ERSDeploymentModel(
            track_id=self.track_id,
            mode=self.ers_mode,
            initial_soc=self.ers_initial_soc,
        )
        self._brake = BrakeModel(
            track_id=self.track_id,
            front_fraction=self.brake_front_fraction,
            cooling_duct_area_mm2=self.brake_cooling_duct_area_mm2,
            brake_energy_kj_per_lap=self.brake_energy_kj_per_lap,
        )

    # ------------------------------------------------------------------ #
    # 单圈仿真
    # ------------------------------------------------------------------ #
    def simulate_lap(self, lap_idx: int | None = None) -> dict[str, Any]:
        """仿真单圈, 返回综合圈速与各子系统贡献."""
        if lap_idx is None:
            lap_idx = self._state["lap_idx"]

        # 基础圈速 (赛道先验)
        base_lap = track_prior(self.track_id, self.setup)
        avg_speed = track_avg_speed(self.track_id)

        # === Aero ===
        # max_speed = avg * 1.4 (估算)
        max_speed = avg_speed * 1.4
        aero_r = self._aero.compute_lap_aero(
            avg_speed_ms=float(avg_speed), max_speed_ms=float(max_speed)
        )
        aero_net_gain = aero_r["net_lap_gain_s"]   # 正 = 圈速更快

        # === Tire (stint 进度) ===
        # 计算这一圈在 stint 中的位置 (基于累计 wear)
        # 缓存 tire_laps 避免每圈重算 (Iter-16 性能优化)
        if self._tire_cache is None:
            self._tire_cache = self._tire.simulate()
        tire_laps = self._tire_cache
        # 用 lap_idx 索引 (若超过 tire_stint_length, 取最后一圈)
        if lap_idx < len(tire_laps):
            tire_lap = tire_laps[lap_idx]
        else:
            tire_lap = tire_laps[-1] if tire_laps else {}
        # TireStintPhysics 已经包含 base_lap + tire_penalty + fuel_gain
        tire_lap_time = tire_lap.get("lap_time", base_lap)

        # === ERS ===
        ers_r = self._ers.simulate_lap()
        ers_net_gain = ers_r["net_lap_gain_s"]   # 正 = 圈速更快

        # === Brake ===
        brake_r = self._brake.simulate_lap(current_temp_c=self._state["brake_temp_c"])
        brake_penalty = brake_r["total_lap_penalty_s"]

        # === Driver (基于 aggression/smoothness/consistency) ===
        # aggression 0.5 = 中性, 高于 0.5 = 更激进 (圈速快但风险高)
        # 模型: 0.5 = 0 偏移, > 0.5 = 圈速减少 (越激进越快), < 0.5 = 圈速增加
        driver_gain = (0.5 - self.driver_aggression) * 1.5  # ±0.75 s
        # smoothness 0.5 = 中性, < 0.5 = 圈速损失 (操作粗糙)
        smoothness_penalty = max(0.0, (0.5 - self.driver_smoothness)) * 0.6
        # consistency 0.5 = 中性, < 0.5 = 圈速波动 (这里取惩罚)
        consistency_penalty = max(0.0, (0.5 - self.driver_consistency)) * 0.4

        # === 组装最终圈速 ===
        # 起点: tire_lap_time (含 base + tire + fuel)
        # 调整: + (base_lap - tire_lap_time 内的 base) — 简化, 直接以 tire 为基础
        # 实际: 我们要 = base - aero_gain - ers_gain + brake_pen + driver_pen
        # 但 tire_lap_time 已经包含 base + tire_deg - fuel_gain
        # 所以: lap_time = tire_lap_time - aero_net_gain - ers_net_gain
        #                + brake_penalty + smoothness_penalty
        #                + consistency_penalty + driver_gain (negative when aggressive)

        # === Weather (Iter-13) ===
        # 雨地圈速惩罚: 错误轮胎在错误湿润度下损失巨大
        weather_penalty = 0.0
        track_wetness = 0.0
        weather_rec_compound: str | None = None
        follow_loss_factor = 1.0
        if self.weather is not None:
            weather_penalty = self.weather.lap_time_penalty(self.compound, base_lap)
            track_wetness = float(self.weather.state.track_wetness)
            weather_rec_compound = self.weather.recommend_compound()
            follow_loss_factor = self.weather.follow_loss_factor()

        lap_time = (
            tire_lap_time
            - aero_net_gain      # aero 增益
            - ers_net_gain       # ERS 增益
            + brake_penalty      # 刹车惩罚
            + smoothness_penalty  # 平顺度惩罚
            + consistency_penalty  # 一致性惩罚
            + driver_gain         # 车手激进/保守偏移 (可正可负)
            + weather_penalty     # 雨地惩罚 (Iter-13)
            + self.car_performance_offset_s  # 车队赛车性能偏移 (Iter-36)
            - self.driver_track_affinity_s   # 车手-赛道亲和度 (Iter-38, 正=快)
        )

        # 物理边界
        lap_time = max(60.0, min(180.0, float(lap_time)))

        # === 更新跨圈状态 ===
        # 轮胎磨损更新 (使用 tire 模型自身的累计值)
        if tire_lap:
            self._state["tire_wear_pct"] = tire_lap.get("wear_pct", 0.0)
            self._state["front_wear_pct"] = tire_lap.get("front_wear_pct", 0.0)
            self._state["rear_wear_pct"] = tire_lap.get("rear_wear_pct", 0.0)
            self._state["fuel_kg"] = tire_lap.get("fuel_kg", self._state["fuel_kg"])
        # 刹车温度更新
        self._state["brake_temp_c"] = brake_r["temp_after_c"]
        # ERS SoC 更新
        self._state["ers_soc"] = self._ers.soc
        # 圈数计数
        self._state["lap_idx"] = lap_idx + 1

        return {
            "lap": lap_idx + 1,
            "lap_time": float(lap_time),
            "base_prior": float(base_lap),
            "tire_lap_time": float(tire_lap_time),
            "tire_wear_pct": float(self._state["tire_wear_pct"]),
            "tire_phase": tire_lap.get("phase", "unknown"),
            "aero_net_gain_s": float(aero_net_gain),
            "aero_cl": float(aero_r["cl"]),
            "aero_cd": float(aero_r["cd"]),
            "ers_net_gain_s": float(ers_net_gain),
            "ers_soc_after": float(self._state["ers_soc"]),
            "ers_deploy_mj": float(ers_r["deploy_mj"]),
            "brake_penalty_s": float(brake_penalty),
            "brake_temp_after_c": float(self._state["brake_temp_c"]),
            "brake_in_window": bool(brake_r["in_window"]),
            "driver_gain_s": float(driver_gain),
            "smoothness_penalty_s": float(smoothness_penalty),
            "consistency_penalty_s": float(consistency_penalty),
            "fuel_kg": float(self._state["fuel_kg"]),
            "track_id": self.track_id,
            "compound": self.compound,
            "weather_penalty_s": float(weather_penalty),
            "track_wetness": float(track_wetness),
            "weather_recommended_compound": weather_rec_compound,
            "follow_loss_factor": float(follow_loss_factor),
            "car_performance_offset_s": float(self.car_performance_offset_s),
            "driver_track_affinity_s": float(self.driver_track_affinity_s),
        }

    # ------------------------------------------------------------------ #
    # 多圈 stint
    # ------------------------------------------------------------------ #
    def simulate_stint(self, laps: int) -> list[dict[str, Any]]:
        """仿真多圈, 跨圈状态传递."""
        # 重置状态
        self._state = {
            "tire_wear_pct": 0.0,
            "front_wear_pct": 0.0,
            "rear_wear_pct": 0.0,
            "fuel_kg": float(self.initial_fuel_kg),
            "brake_temp_c": float(self.brake_temp_c),
            "ers_soc": float(self.ers_initial_soc),
            "lap_idx": 0,
        }
        # 重建子模型 (重置内部状态)
        self._build_models()
        self._tire_cache = None
        out: list[dict[str, Any]] = []
        for k in range(int(laps)):
            out.append(self.simulate_lap(lap_idx=k))
        return out

    # ------------------------------------------------------------------ #
    # 摘要
    # ------------------------------------------------------------------ #
    def summary(self, laps: int = 20) -> dict[str, Any]:
        """仿真 N 圈并返回摘要."""
        stint = self.simulate_stint(laps)
        if not stint:
            return {}
        times = [lp["lap_time"] for lp in stint]
        return {
            "track_id": self.track_id,
            "compound": self.compound,
            "laps": laps,
            "total_time": float(sum(times)),
            "avg_lap_time": float(sum(times) / len(times)),
            "best_lap": float(min(times)),
            "best_lap_num": int(times.index(min(times)) + 1),
            "worst_lap": float(max(times)),
            "final_tire_wear_pct": float(stint[-1]["tire_wear_pct"]),
            "final_ers_soc": float(stint[-1]["ers_soc_after"]),
            "final_brake_temp_c": float(stint[-1]["brake_temp_after_c"]),
            "final_fuel_kg": float(stint[-1]["fuel_kg"]),
            "lap_times": times,
        }


# --------------------------------------------------------------------------- #
# 模块级便捷函数
# --------------------------------------------------------------------------- #
def simulate_lap(
    setup: CarSetup, track_id: str, compound: str = "medium", **kwargs: Any
) -> dict[str, Any]:
    """单圈综合仿真便捷函数."""
    sim = LapTimeSimulator(setup=setup, track_id=track_id, compound=compound, **kwargs)
    return sim.simulate_lap(lap_idx=0)


def simulate_stint(
    setup: CarSetup, track_id: str, laps: int, compound: str = "medium", **kwargs: Any
) -> list[dict[str, Any]]:
    """多圈 stint 综合仿真便捷函数."""
    sim = LapTimeSimulator(setup=setup, track_id=track_id, compound=compound, **kwargs)
    return sim.simulate_stint(laps)
