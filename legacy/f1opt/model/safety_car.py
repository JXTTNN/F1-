"""F1 安全车 / 虚拟安全车模型 (Iter-14).

真实 F1 正赛中安全车 (Safety Car, SC) 与虚拟安全车 (VSC) 是重大策略事件:
车队利用 SC 期间进站 ("free pit") 节省 ~20 s. FIA 2026 体育规则 §39 (SC) +
§48 (VSC):

- **SC**: 全场黄旗, 车手列队跟随安全车, 禁止超车. 圈速 ~30% 慢 (SC pace).
  持续 3-5 圈 (碎片清理/事故救援).
- **VSC**: 分段黄旗, 圈速 ~25% 慢, 不列队, 禁止超车. 持续 1-3 圈.
- **触发**: 退赛 (机械故障撞车) / 碎片 / 极端天气 / 首圈事故.
- **SC 期间进站 ("free pit")**: 因全场已慢, 进站损失大幅减少 (~80% 折扣).
- **重启**: SC 进站后第 1 圈圈速略高 (冷胎 + 重启混乱).

公开 API:
    - :class:`SafetyCarPeriod` — 单次 SC/VSC 时段.
    - :class:`SafetyCarModel` — 管理多个 SC 时段 + 圈速因子 + 进站折扣.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# 物理常量 (F1 工程估算 + FIA 规则)
# --------------------------------------------------------------------------- #
# SC/VSC 圈速因子 (1.0 = 正常, >1 = 慢)
_SC_LAP_TIME_FACTOR = 1.30      # SC 跟车圈速 = 正常 × 1.30
_VSC_LAP_TIME_FACTOR = 1.25     # VSC 圈速 = 正常 × 1.25

# SC 持续圈数范围
_SC_MIN_LAPS = 3
_SC_MAX_LAPS = 6
_VSC_MIN_LAPS = 1
_VSC_MAX_LAPS = 3

# SC 期间进站损失折扣 (free pit)
_SC_PIT_LOSS_DISCOUNT = 0.20    # SC 期间进站只损失 20% 原进站损失
_VSC_PIT_LOSS_DISCOUNT = 0.55   # VSC 期间进站损失 55%

# 重启圈惩罚 (SC 结束后第 1 圈)
_RESTART_LAP_PENALTY_S = 0.8    # 冷胎 + 重启混乱

# 触发概率 (每圈每位车手)
_RETRIGGER_PROB_PER_LAP = 0.0015  # 退赛触发 SC
_DEBRIS_PROB_PER_LAP = 0.0008     # 碎片触发 VSC
_WEATHER_SC_PROB_PER_LAP = 0.005  # 极端天气下触发概率放大

# 首圈事故触发概率
_LAP1_INCIDENT_PROB = 0.08


# --------------------------------------------------------------------------- #
# SafetyCarPeriod
# --------------------------------------------------------------------------- #
@dataclass
class SafetyCarPeriod:
    """单次 SC/VSC 时段 (1-indexed 圈数)."""

    start_lap: int          # SC 开始圈
    end_lap: int            # SC 结束圈 (含)
    kind: str = "sc"        # "sc" | "vsc"
    reason: str = "incident"

    @property
    def duration_laps(self) -> int:
        return max(0, self.end_lap - self.start_lap + 1)

    def active_during(self, lap: int) -> bool:
        return self.start_lap <= lap <= self.end_lap

    def is_restart_lap(self, lap: int) -> bool:
        """SC/VSC 结束后下一圈 = 重启圈."""
        return lap == self.end_lap + 1


# --------------------------------------------------------------------------- #
# SafetyCarModel
# --------------------------------------------------------------------------- #
@dataclass
class SafetyCarModel:
    """管理 SC/VSC 时段 + 圈速因子 + 进站折扣.

    用法::

        scm = SafetyCarModel(seed=42)
        scm.generate_periods(total_laps=58, n_retirements=2,
                              weather_wetness=0.0, rng=...)
        # 查询某圈
        factor = scm.lap_time_factor(lap=20)        # 1.0 或 1.30 (SC) 或 1.25 (VSC)
        discount = scm.pit_loss_discount(lap=20)    # 1.0 (无 SC) 或 0.20 (SC)
    """

    periods: list[SafetyCarPeriod] = field(default_factory=list)
    seed: int | None = None

    # ------------------------------------------------------------------ #
    def generate_periods(
        self,
        total_laps: int,
        n_retirements: int = 0,
        weather_wetness: float = 0.0,
        rng: random.Random | None = None,
    ) -> list[SafetyCarPeriod]:
        """根据退赛数/天气生成 SC/VSC 时段 (确定性 by rng)."""
        if rng is None:
            rng = random.Random(self.seed)
        self.periods = []
        if total_laps <= 2:
            return self.periods

        # 首圈事故 (lap 1) 触发 SC 概率
        if rng.random() < _LAP1_INCIDENT_PROB:
            dur = rng.randint(_SC_MIN_LAPS, _SC_MAX_LAPS)
            self._add_period(1, min(1 + dur - 1, total_laps), "sc", "lap1_incident")

        # 退赛触发: 每次退赛 ~30% 概率触发 SC, ~20% 概率触发 VSC
        for _ in range(n_retirements):
            r = rng.random()
            if r < 0.30:
                trigger_lap = rng.randint(3, max(3, total_laps - 2))
                dur = rng.randint(_SC_MIN_LAPS, _SC_MAX_LAPS)
                self._add_period(trigger_lap,
                                 min(trigger_lap + dur - 1, total_laps),
                                 "sc", "retirement")
            elif r < 0.50:
                trigger_lap = rng.randint(3, max(3, total_laps - 2))
                dur = rng.randint(_VSC_MIN_LAPS, _VSC_MAX_LAPS)
                self._add_period(trigger_lap,
                                 min(trigger_lap + dur - 1, total_laps),
                                 "vsc", "retirement")

        # 碎片触发 VSC (随机扫描)
        for lap in range(2, total_laps - 1):
            base_prob = _DEBRIS_PROB_PER_LAP
            if weather_wetness > 0.5:
                base_prob += _WEATHER_SC_PROB_PER_LAP
            if rng.random() < base_prob and not self._lap_covered(lap):
                dur = rng.randint(_VSC_MIN_LAPS, _VSC_MAX_LAPS)
                self._add_period(lap, min(lap + dur - 1, total_laps),
                                 "vsc", "debris")

        # 合并重叠时段
        self.periods = self._merge_overlapping()
        return self.periods

    def _add_period(self, start: int, end: int, kind: str, reason: str) -> None:
        if end >= start:
            self.periods.append(
                SafetyCarPeriod(start_lap=start, end_lap=end, kind=kind, reason=reason)
            )

    def _lap_covered(self, lap: int) -> bool:
        return any(p.active_during(lap) for p in self.periods)

    def _merge_overlapping(self) -> list[SafetyCarPeriod]:
        if not self.periods:
            return []
        # 按 start_lap 排序
        sorted_p = sorted(self.periods, key=lambda p: p.start_lap)
        merged = [sorted_p[0]]
        for p in sorted_p[1:]:
            last = merged[-1]
            if p.start_lap <= last.end_lap + 1:
                # 重叠/相邻: 合并, SC 优先于 VSC
                kind = "sc" if "sc" in (last.kind, p.kind) else "vsc"
                merged[-1] = SafetyCarPeriod(
                    start_lap=last.start_lap,
                    end_lap=max(last.end_lap, p.end_lap),
                    kind=kind,
                    reason=f"{last.reason}+{p.reason}",
                )
            else:
                merged.append(p)
        return merged

    # ------------------------------------------------------------------ #
    def active_period(self, lap: int) -> SafetyCarPeriod | None:
        """返回覆盖该圈的 SC/VSC 时段, 无则 None."""
        for p in self.periods:
            if p.active_during(lap):
                return p
        return None

    def is_under_sc(self, lap: int) -> bool:
        p = self.active_period(lap)
        return p is not None and p.kind == "sc"

    def is_under_vsc(self, lap: int) -> bool:
        p = self.active_period(lap)
        return p is not None and p.kind == "vsc"

    def is_restart_lap(self, lap: int) -> bool:
        return any(p.is_restart_lap(lap) for p in self.periods)

    # ------------------------------------------------------------------ #
    def lap_time_factor(self, lap: int) -> float:
        """返回该圈圈速因子 (乘以正常圈速). SC=1.30, VSC=1.25, 正常=1.0."""
        p = self.active_period(lap)
        if p is None:
            return 1.0
        return _SC_LAP_TIME_FACTOR if p.kind == "sc" else _VSC_LAP_TIME_FACTOR

    def pit_loss_discount(self, lap: int) -> float:
        """返回该圈进站损失折扣因子 (1.0=全额, 0.20=SC期间只付20%)."""
        p = self.active_period(lap)
        if p is None:
            return 1.0
        return _SC_PIT_LOSS_DISCOUNT if p.kind == "sc" else _VSC_PIT_LOSS_DISCOUNT

    def restart_penalty_s(self, lap: int) -> float:
        """重启圈惩罚 (秒). 非重启圈返回 0."""
        return _RESTART_LAP_PENALTY_S if self.is_restart_lap(lap) else 0.0

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        return {
            "n_periods": len(self.periods),
            "n_sc": sum(1 for p in self.periods if p.kind == "sc"),
            "n_vsc": sum(1 for p in self.periods if p.kind == "vsc"),
            "total_sc_laps": sum(p.duration_laps for p in self.periods),
            "periods": [
                {
                    "start_lap": p.start_lap,
                    "end_lap": p.end_lap,
                    "duration_laps": p.duration_laps,
                    "kind": p.kind,
                    "reason": p.reason,
                }
                for p in self.periods
            ],
        }

    def reset(self) -> None:
        self.periods = []
