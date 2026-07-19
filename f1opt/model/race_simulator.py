"""F1 正赛仿真 (Iter-10): 完整 50-圈赛事.

真实 F1 正赛是各种相互作用的复杂系统: 起步 + 进站策略 + 交通 + 安全车
+ DRS 列车 + 退赛. 本模块仿真除安全车/退赛外的核心逻辑:

- **起步**: P1 起步无影响, 后方车手有 0.05-0.3 s 起步延迟 (取决于发车位).
- **每圈**: 用 :class:`LapTimeSimulator` 计算圈速, 受交通影响 (前车在 1 秒
  内会拖慢).
- **进站**: 按策略在某圈进站, 损失 ~22-25 s (赛道相关), 换胎并重置 stint.
- **DRS 列车**: 前车在 1 秒内时, 后车有 0.2-0.4 s DRS 收益.
- **位置追踪**: 每圈后按累计时间排序, 同圈车手按本圈圈速.

公开 API:
    - :class:`RaceStrategy` — 单车手的进站策略.
    - :class:`RaceCar` — 单车手状态.
    - :class:`RaceSimulation` — 完整赛事仿真.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from f1opt.data.setup_schema import CarSetup
from f1opt.model.lap_simulator import LapTimeSimulator
from f1opt.model.safety_car import SafetyCarModel
from f1opt.model.weather import WeatherModel

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
_DEFAULT_PIT_LOSS_S = 23.0
_DEFAULT_FUEL_KG = 110.0
_DEFAULT_FUEL_BURN_KG_PER_LAP = 1.6

# 赛道特定 pit loss (秒)
_PIT_LOSS_BY_TRACK: dict[str, float] = {
    "monaco": 21.0, "monza": 23.0, "spa": 22.0, "silverstone": 23.0,
    "suzuka": 22.0, "bahrain": 24.0, "jeddah": 22.0, "melbourne": 23.0,
    "singapore": 22.0, "yas_marina": 23.0, "baku": 22.0, "miami": 23.0,
    "barcelona": 23.0, "austin": 23.0, "interlagos": 22.0, "losail": 23.0,
    "shanghai": 24.0, "budapest": 22.0, "amsterdam": 23.0, "montreal": 23.0,
    "las_vegas": 23.0, "madrid": 23.0,
}


def _pit_loss_for(track_id: str) -> float:
    return _PIT_LOSS_BY_TRACK.get(track_id, _DEFAULT_PIT_LOSS_S)


# --------------------------------------------------------------------------- #
# RaceStrategy
# --------------------------------------------------------------------------- #
@dataclass
class RaceStrategy:
    """单车手正赛进站策略.

    - ``pit_laps``: 进站圈列表 (1-indexed), 例如 [18, 38] = 两次进站.
    - ``compounds``: 每个 stint 的化合物, 长度 = len(pit_laps) + 1.
        例如 ['medium', 'medium', 'soft'] = 3 stint, 第 1 stint medium.
    """

    pit_laps: tuple[int, ...]
    compounds: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.compounds) != len(self.pit_laps) + 1:
            raise ValueError(
                f"compounds ({len(self.compounds)}) must be "
                f"len(pit_laps)+1 ({len(self.pit_laps)+1})"
            )

    @property
    def n_stops(self) -> int:
        return len(self.pit_laps)

    def compound_for_lap(self, lap: int) -> str:
        """返回该圈所用化合物 (1-indexed)."""
        stint_idx = 0
        for pl in self.pit_laps:
            if lap >= pl + 1:  # 进站圈之后用下一个 compound
                stint_idx += 1
            else:
                break
        return self.compounds[stint_idx]

    def laps_until_pit(self, lap: int) -> int | None:
        """距离下一次进站还有几圈; 无进站返回 None."""
        for pl in self.pit_laps:
            if pl >= lap:
                return pl - lap
        return None


# --------------------------------------------------------------------------- #
# RaceCar
# --------------------------------------------------------------------------- #
@dataclass
class RaceCar:
    """单车手正赛状态."""

    driver_id: str
    driver_name: str
    setup: CarSetup
    grid_position: int
    strategy: RaceStrategy
    driver_aggression: float = 0.7
    driver_smoothness: float = 0.7
    driver_consistency: float = 0.7
    driver_tire_management: float = 0.5
    """车手轮胎管理风格 0..1 (Iter-22)."""
    car_performance_offset_s: float = 0.0
    """车队赛车性能偏移 s/lap (Iter-36). 来自 teams_2026."""
    driver_track_affinity_s: float = 0.0
    """车手-赛道亲和度 s/lap (Iter-38). 正=快, 负=慢."""
    team_id: str = ""
    """车队 ID (Iter-40). 用于 pit crew 性能查询."""
    pit_crew_offset_s: float = 0.0
    """车队 pit crew 偏移 s (Iter-40).
    正 = 慢于基准 (后段车队), 负 = 快于基准 (顶队).
    pit_loss = track_pit_loss + pit_crew_offset_s (含随机慢停)."""

    # 仿真状态 (动态)
    current_lap: int = 0
    current_compound: str = "medium"
    cumulative_time: float = 0.0
    laps_completed: int = 0
    pit_stops_done: int = 0
    last_lap_time: float = 0.0
    position: int = 0
    retired: bool = False
    retirement_lap: int | None = None

    # 子模型 (仿真时构建)
    _sim: LapTimeSimulator | None = field(init=False, repr=False, default=None)
    _stint_lap_idx: int = field(init=False, repr=False, default=0)


# --------------------------------------------------------------------------- #
# RaceSimulation
# --------------------------------------------------------------------------- #
class RaceSimulation:
    """完整 F1 正赛仿真.

    用法::

        cars = [RaceCar(driver_id=f"d{i:02d}", driver_name=f"D{i+1}",
                        setup=DEFAULT_SETUP, grid_position=i+1,
                        strategy=RaceStrategy(pit_laps=(20, 40),
                                              compounds=('medium','medium','soft')))
                for i in range(20)]
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=53, seed=42)
        results = sim.run()
        # results = list[(position, RaceCar)]
    """

    def __init__(
        self,
        track_id: str,
        cars: list[RaceCar],
        total_laps: int,
        seed: int | None = None,
        track_temp_c: float = 35.0,
        brake_temp_c: float = 500.0,
        retirement_prob_per_lap: float = 0.001,  # 0.1% per lap per car
        weather: WeatherModel | None = None,
        safety_car: SafetyCarModel | None = None,
        weather_rain_mmh: float = 0.0,
        lap_duration_min: float = 1.5,
    ) -> None:
        if not cars:
            raise ValueError("Need at least 1 car")
        self.track_id = track_id
        self.cars = list(cars)
        self.total_laps = int(total_laps)
        self.rng = random.Random(seed)
        self.track_temp_c = float(track_temp_c)
        self.brake_temp_c = float(brake_temp_c)
        self.retirement_prob_per_lap = float(retirement_prob_per_lap)
        self.pit_loss_s = _pit_loss_for(track_id)
        self._initial_seed = seed
        # Weather + Safety Car (Iter-15)
        self.weather = weather
        self._weather_rain_mmh = float(weather_rain_mmh)
        self._lap_duration_min = float(lap_duration_min)
        self.safety_car = safety_car if safety_car is not None else SafetyCarModel(seed=seed)
        self._sc_generated = False

    # ------------------------------------------------------------------ #
    def _init_car_sim(self, car: RaceCar) -> None:
        """为车手构建 LapTimeSimulator, 设置当前 stint 起始状态."""
        car._sim = LapTimeSimulator(
            setup=car.setup,
            track_id=self.track_id,
            compound=car.current_compound,
            driver_aggression=car.driver_aggression,
            driver_smoothness=car.driver_smoothness,
            driver_consistency=car.driver_consistency,
            driver_tire_management=car.driver_tire_management,
            initial_fuel_kg=_DEFAULT_FUEL_KG,
            brake_temp_c=self.brake_temp_c,
            track_temp_c=self.track_temp_c,
            weather=self.weather,
            car_performance_offset_s=car.car_performance_offset_s,
            driver_track_affinity_s=car.driver_track_affinity_s,
        )
        car._stint_lap_idx = 0

    # ------------------------------------------------------------------ #
    def run(self) -> list[tuple[int, RaceCar]]:
        """运行完整赛事, 返回 [(position, RaceCar)] 排序结果."""
        # 重置 RNG 以保证幂等
        self.rng = random.Random(self._initial_seed)
        # 重置 Weather (Iter-15)
        if self.weather is not None:
            self.weather.reset()
        # 重置 Safety Car (Iter-15)
        self.safety_car.reset()
        self._sc_generated = False
        # 重置所有车手状态
        for car in self.cars:
            car.current_lap = 0
            car.cumulative_time = 0.0
            car.laps_completed = 0
            car.pit_stops_done = 0
            car.last_lap_time = 0.0
            car.retired = False
            car.retirement_lap = None
            car.position = car.grid_position
            # 起步用第一个 stint 的 compound
            car.current_compound = car.strategy.compounds[0]
            self._init_car_sim(car)

        # 按发车位排序
        active_cars = sorted(self.cars, key=lambda c: c.grid_position)

        # 起步延迟 (后方车慢一点)
        for car in active_cars:
            start_delay = max(0.0, (car.grid_position - 1) * 0.05)
            car.cumulative_time = start_delay

        # 圈循环
        retirements_this_race = 0
        for lap in range(1, self.total_laps + 1):
            # === 预生成 SC 时段 (基于预估退赛 + 天气) ===
            if not self._sc_generated:
                wetness = (self.weather.state.track_wetness
                           if self.weather is not None else 0.0)
                est_retirements = max(0, int(
                    self.retirement_prob_per_lap * len(active_cars) * self.total_laps
                ))
                self.safety_car.generate_periods(
                    total_laps=self.total_laps,
                    n_retirements=est_retirements,
                    weather_wetness=wetness,
                    rng=self.rng,
                )
                self._sc_generated = True

            under_sc = self.safety_car.is_under_sc(lap)
            under_vsc = self.safety_car.is_under_vsc(lap)
            sc_active = under_sc or under_vsc

            for car in active_cars:
                if car.retired:
                    continue
                car.current_lap = lap

                # 检查是否本圈进站
                is_pit_lap = (car.pit_stops_done < car.strategy.n_stops and
                              car.strategy.pit_laps[car.pit_stops_done] == lap)

                # 计算圈速
                lap_time = self._compute_lap_time(car, active_cars)

                # === Safety Car 圈速因子 (Iter-15) ===
                # SC/VSC 期间全场慢, DRS 禁用
                if sc_active:
                    lap_time *= self.safety_car.lap_time_factor(lap)
                else:
                    # 非 SC: 应用 DRS 收益 + 交通损失 (SC 期间无超车)
                    drs_gain = self._drs_gain(car, active_cars)
                    lap_time -= drs_gain
                    traffic_loss = self._traffic_loss(car, active_cars)
                    lap_time += traffic_loss
                    # 重启圈惩罚
                    lap_time += self.safety_car.restart_penalty_s(lap)

                # === 进站损失 (SC 期间折扣, "free pit") + 团队 pit crew 偏移 ===
                if is_pit_lap:
                    discount = self.safety_car.pit_loss_discount(lap)
                    # 团队 pit crew 偏移 (Iter-40): 顶队快, 后段慢
                    # 含随机慢停 (基于 team_id + 当前 rng)
                    crew_offset = self._pit_crew_offset_for(car, lap)
                    effective_pit_loss = self.pit_loss_s + crew_offset
                    lap_time += effective_pit_loss * discount

                # 更新累计时间
                car.cumulative_time += lap_time
                car.last_lap_time = lap_time
                car.laps_completed = lap
                car._stint_lap_idx += 1

                # 进站后切换 compound, 重置 stint
                if is_pit_lap:
                    car.pit_stops_done += 1
                    car.current_compound = car.strategy.compounds[car.pit_stops_done]
                    self._init_car_sim(car)

                # 退赛概率检查 (SC 期间退赛概率降低 — 无高速事故)
                ret_prob = self.retirement_prob_per_lap
                if sc_active:
                    ret_prob *= 0.1
                if self.rng.random() < ret_prob:
                    car.retired = True
                    car.retirement_lap = lap
                    retirements_this_race += 1

            # === 天气演化 (每圈推进 lap_duration_min 分钟) ===
            if self.weather is not None:
                self.weather.step(rain_mmh=self._weather_rain_mmh,
                                  minutes=self._lap_duration_min)
                # 同步赛道温度到 LapTimeSimulator (通过 weather.state)
                for car in active_cars:
                    if car._sim is not None and not car.retired:
                        car._sim.track_temp_c = self.weather.state.track_temp_c

            # 更新位置 (按累计时间排序, 退赛车放最后)
            active_cars.sort(key=lambda c: (c.retired, c.cumulative_time))
            for i, car in enumerate(active_cars):
                car.position = i + 1

        # 最终排序
        active_cars.sort(key=lambda c: (
            c.retired,  # 退赛车在后
            -c.laps_completed,  # 完赛圈数少的在后
            c.cumulative_time  # 时间少的在前
        ))
        return [(i + 1, car) for i, car in enumerate(active_cars)]

    # ------------------------------------------------------------------ #
    def _compute_lap_time(self, car: RaceCar, all_cars: list[RaceCar]) -> float:
        """用 LapTimeSimulator 计算车手当前圈圈速."""
        if car._sim is None:
            self._init_car_sim(car)
        # 一致性噪声
        noise = self.rng.gauss(0.0, (1.0 - car.driver_consistency) * 0.4)
        lap_r = car._sim.simulate_lap(lap_idx=car._stint_lap_idx)
        return lap_r["lap_time"] + noise

    # ------------------------------------------------------------------ #
    def _drs_gain(self, car: RaceCar, all_cars: list[RaceCar]) -> float:
        """如果前方有车在 1.5 秒内, 给予 DRS 收益 (0.2-0.4 s)."""
        ahead_cars = [c for c in all_cars
                      if c.position < car.position and not c.retired]
        if not ahead_cars:
            return 0.0
        # 找最近的前车
        ahead = min(ahead_cars, key=lambda c: abs(c.cumulative_time - car.cumulative_time))
        gap = abs(ahead.cumulative_time - car.cumulative_time)
        if gap < 1.5:
            # DRS 收益: 越近收益越大
            return max(0.0, 0.4 - gap * 0.13)
        return 0.0

    # ------------------------------------------------------------------ #
    def _traffic_loss(self, car: RaceCar, all_cars: list[RaceCar]) -> float:
        """如果前方有被套圈的车 (落后 1+ 圈), 给予交通损失 (0.1-0.3 s)."""
        ahead_lapped = [c for c in all_cars
                        if c.laps_completed < car.laps_completed - 0
                        and c.cumulative_time < car.cumulative_time
                        and not c.retired and c is not car]
        if not ahead_lapped:
            return 0.0
        # 10% 概率遇到交通
        if self.rng.random() < 0.3:
            return self.rng.uniform(0.1, 0.3)
        return 0.0

    # ------------------------------------------------------------------ #
    def _pit_crew_offset_for(self, car: RaceCar, lap: int) -> float:
        """返回该车手本圈的 pit crew 偏移 (Iter-40).

        若 car.pit_crew_offset_s 已显式设置 (>0 或 <0), 直接使用 (确定性).
        否则若有 team_id, 用 :func:`pit_stop_time_s` 动态查询 (含随机慢停).
        否则返回 0 (基准).
        """
        # 显式偏移优先 (来自 SeasonSimulator 预计算)
        if car.pit_crew_offset_s != 0.0:
            return car.pit_crew_offset_s
        # 有 team_id → 动态查询 pit crew 性能 (含随机性)
        if car.team_id:
            try:
                from f1opt.model.pit_crew import expected_pit_stop_time_s
                # 用期望值 (确定性, 避免每圈不同)
                # 减去基准 3.0s (track pit_loss_s 已含平均停车时间)
                return expected_pit_stop_time_s(car.team_id) - 3.0
            except (ValueError, ImportError):
                pass
        return 0.0

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """运行后返回摘要."""
        results = self.run()
        winner = results[0][1]
        return {
            "track_id": self.track_id,
            "total_laps": self.total_laps,
            "winner": winner.driver_name,
            "winner_time": winner.cumulative_time,
            "n_finishers": sum(1 for _, c in results if not c.retired),
            "n_retirements": sum(1 for _, c in results if c.retired),
            "classification": [
                {
                    "position": pos,
                    "driver": car.driver_name,
                    "grid": car.grid_position,
                    "laps": car.laps_completed,
                    "total_time": car.cumulative_time,
                    "gap_to_leader": car.cumulative_time - winner.cumulative_time
                                     if not car.retired else None,
                    "pit_stops": car.pit_stops_done,
                    "retired": car.retired,
                    "retirement_lap": car.retirement_lap,
                    "final_compound": car.current_compound,
                }
                for pos, car in results
            ],
        }
