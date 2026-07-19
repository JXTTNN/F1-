"""F1 2026 DRS Train 模型 (Iter-41).

DRS 列车 (DRS Train) 是 F1 经典复杂场景: 3-5 车在 DRS 区互相跟随,
均获 DRS 但无人能超, 形成"列车". 典型案例:
- 巴库 2018: 5 车 DRS 列车持续 10 圈.
- 阿尔伯特公园 2023: 中段 6 车列车.
- 斯帕 2024: Eau Rouge 后直道 4 车列车.

物理机制:
- 前车有 DRS (因为前前车在 1s 内) → 减阻 +18 km/h
- 后车也有 DRS → 同样 +18 km/h
- 净效应: 相对速度差 = 0, 无超车
- 但列车内圈速比自由圈慢 0.2-0.5s (相互拖累)

影响策略:
- 列车内车手应早进站 undercut 脱离列车.
- 列车前车 (领头) 可正常跑.
- 列车后车被严重拖累, 应考虑换胎.

EA Sports F1 2026 游戏官方 DRS Train 物理:
- 3+ 车在 1.0s 间隔内形成 DRS 列车.
- 列车内每车圈速损失 0.15-0.40s (取决于位置).
- 列车领头损失最少 (0.15s), 末尾损失最多 (0.40s).
- 列车持续到 SC / 进站 / 失误打破.

参考文献:
- FIA Sporting Regulations 2026 §18.6 (DRS activation)
- 公开 F1 DRS 列车分析 (The Race 2024)

公开 API:
    - :class:`DRSTrainModel` — DRS 列车检测与圈速影响.
    - :func:`detect_drs_train` — 检测 DRS 列车.
    - :func:`drs_train_lap_penalty` — 列车内车手圈速损失.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
_DRS_GAP_THRESHOLD_S = 1.0
"""DRS 激活阈值 (前车间隔 ≤ 1.0s)."""

_DRS_TRAIN_MIN_SIZE = 3
"""DRS 列车最小车数 (3+ 车在 1.0s 内)."""

_DRS_TRAIN_PENALTY_LEADER_S = 0.15
"""列车领头圈速损失 s."""

_DRS_TRAIN_PENALTY_PER_CAR_S = 0.08
"""列车内每增加一辆车, 后车额外损失 s."""

_DRS_TRAIN_PENALTY_MAX_S = 0.50
"""列车内单圈最大损失 s."""

_DRS_TRAIN_BREAK_PROB_PER_LAP = 0.05
"""每圈列车自然解散概率 (失误/进站/超车)."""


@dataclass
class DRSTrainCar:
    """DRS 列车内单车状态."""

    driver_id: str
    position_in_train: int
    """列车内位置 (1 = 领头)."""
    gap_ahead_s: float
    """与前车间隔 s."""
    lap_penalty_s: float
    """本圈列车损失 s."""


@dataclass
class DRSTrain:
    """一个 DRS 列车."""

    cars: list[DRSTrainCar]
    """列车内车手 (按位置排序, 第 0 = 领头)."""
    track_id: str = ""
    formed_lap: int = 0
    """列车形成的圈数."""

    @property
    def size(self) -> int:
        return len(self.cars)

    @property
    def leader_id(self) -> str | None:
        return self.cars[0].driver_id if self.cars else None

    @property
    def tail_id(self) -> str | None:
        return self.cars[-1].driver_id if self.cars else None

    def penalty_for(self, driver_id: str) -> float:
        """返回该车手在列车内的圈速损失 s (0 = 不在列车)."""
        for c in self.cars:
            if c.driver_id == driver_id:
                return c.lap_penalty_s
        return 0.0


# --------------------------------------------------------------------------- #
# DRSTrainModel
# --------------------------------------------------------------------------- #
@dataclass
class DRSTrainModel:
    """DRS 列车检测与圈速影响模型.

    用法::

        model = DRSTrainModel(track_id="baku")
        # 输入: 每圈各车手间隔 (sorted by position)
        gaps = [("d1", 0.0), ("d2", 0.8), ("d3", 0.7), ("d4", 0.9)]
        trains = model.detect_trains(gaps, current_lap=10)
        for train in trains:
            print(train.size, train.leader_id)
            penalty = train.penalty_for("d3")  # d3 在列车内的损失
    """

    track_id: str = ""
    drs_gap_threshold_s: float = _DRS_GAP_THRESHOLD_S
    train_min_size: int = _DRS_TRAIN_MIN_SIZE
    penalty_leader_s: float = _DRS_TRAIN_PENALTY_LEADER_S
    penalty_per_car_s: float = _DRS_TRAIN_PENALTY_PER_CAR_S
    penalty_max_s: float = _DRS_TRAIN_PENALTY_MAX_S
    break_prob_per_lap: float = _DRS_TRAIN_BREAK_PROB_PER_LAP

    # 状态: 当前活跃列车
    _active_trains: list[DRSTrain] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ #
    def detect_trains(
        self,
        gaps: list[tuple[str, float]],
        current_lap: int = 0,
    ) -> list[DRSTrain]:
        """检测 DRS 列车.

        Args:
            gaps: 各车手 (driver_id, gap_ahead_s), 按位置排序 (前到后).
                gap_ahead_s = 与前车的间隔 (领头为 0.0).
            current_lap: 当前圈数.

        Returns:
            检测到的 DRS 列车列表.
        """
        if len(gaps) < self.train_min_size:
            self._active_trains = []
            return []

        trains: list[DRSTrain] = []
        i = 0
        while i < len(gaps):
            # 跳过领头 (gap=0)
            if i == 0:
                i += 1
                continue
            # 检查从 i 开始是否形成列车
            train_cars: list[DRSTrainCar] = []
            # 列车领头是 gaps[i-1] (前车)
            if i - 1 < 0:
                i += 1
                continue
            leader_id = gaps[i - 1][0]
            # 加入领头
            train_cars.append(DRSTrainCar(
                driver_id=leader_id,
                position_in_train=1,
                gap_ahead_s=0.0,
                lap_penalty_s=self._penalty_for_position(1),
            ))
            # 向后扫描, 只要间隔 <= threshold
            j = i
            while j < len(gaps) and gaps[j][1] <= self.drs_gap_threshold_s:
                pos = len(train_cars) + 1
                train_cars.append(DRSTrainCar(
                    driver_id=gaps[j][0],
                    position_in_train=pos,
                    gap_ahead_s=gaps[j][1],
                    lap_penalty_s=self._penalty_for_position(pos),
                ))
                j += 1
            # 检查列车大小
            if len(train_cars) >= self.train_min_size:
                trains.append(DRSTrain(
                    cars=train_cars,
                    track_id=self.track_id,
                    formed_lap=current_lap,
                ))
                i = j  # 跳过已加入列车的车
            else:
                i += 1

        self._active_trains = trains
        return trains

    def _penalty_for_position(self, position: int) -> float:
        """列车内位置 → 圈速损失 s.

        领头 (pos=1) 损失最少, 末尾损失最多.
        """
        if position <= 1:
            return self.penalty_leader_s
        penalty = self.penalty_leader_s + (position - 1) * self.penalty_per_car_s
        return min(penalty, self.penalty_max_s)

    # ------------------------------------------------------------------ #
    def get_active_train_for(self, driver_id: str) -> DRSTrain | None:
        """返回该车手当前所在的 DRS 列车 (无则 None)."""
        for train in self._active_trains:
            if train.penalty_for(driver_id) > 0:
                return train
        return None

    def penalty_for_driver(self, driver_id: str) -> float:
        """返回该车手当前 DRS 列车圈速损失 s (0 = 不在列车)."""
        for train in self._active_trains:
            p = train.penalty_for(driver_id)
            if p > 0:
                return p
        return 0.0

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, object]:
        """返回当前 DRS 列车状态摘要."""
        return {
            "track_id": self.track_id,
            "n_active_trains": len(self._active_trains),
            "trains": [
                {
                    "size": t.size,
                    "leader": t.leader_id,
                    "tail": t.tail_id,
                    "formed_lap": t.formed_lap,
                }
                for t in self._active_trains
            ],
        }


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #
def detect_drs_train(
    gaps: list[tuple[str, float]],
    track_id: str = "",
    current_lap: int = 0,
) -> list[DRSTrain]:
    """便捷函数: 检测 DRS 列车."""
    return DRSTrainModel(track_id=track_id).detect_trains(gaps, current_lap)


def drs_train_lap_penalty(
    gaps: list[tuple[str, float]],
    driver_id: str,
) -> float:
    """便捷函数: 返回该车手在 DRS 列车内的圈速损失 s."""
    trains = detect_drs_train(gaps)
    for t in trains:
        p = t.penalty_for(driver_id)
        if p > 0:
            return p
    return 0.0
