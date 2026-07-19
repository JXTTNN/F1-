"""F1 ERS (Energy Recovery System) 部署模型 (Iter-5).

F1 2026 规则下每圈可部署 4 MJ 电动能 (MGU-K) 与回收 2.5 MJ (FIA 2026
技术规则 §5.4). ERS 部署策略对圈速影响 0.3-0.5 s/lap — 真实车队
(Mercedes/Red Bull/Racing Bulls 等) 都会基于赛道特性为每段直道分配能量.

本模块实现一个工程化的 ERS 部署模型:

- 每圈 4 MJ 部署预算 + 2.5 MJ 回收预算 (FIA 限制).
- 电池状态 (State of Charge, SoC): 0-100%, 起始 50%, 不能超出.
- 部署区间: 按赛道最长直道优先分配 (drag-limited 直道收益最大).
- 部署效率: 1 MJ ≈ 0.08 s 直道收益 (基于公开 F1 工程估算).
- 回收效率: 刹车区每 MJ ≈ 0.05 s 圈速代价 (发动机 + 电池充电阻力).
- 模式: attack (主动多部署) / balanced / conserve (省电池).

公开 API:
    - :class:`ERSTrackProfile` — 单赛道 ERS 直道分布.
    - :class:`ERSDeploymentModel` — 单圈 ERS 仿真.
    - :data:`ERS_TRACK_PROFILES` — 12 条主要赛道的 ERS profile.

参考 (FIA 公开技术规则, 无学术论文):
    FIA Formula 1 Technical Regulations 2026 §5 (Power Unit).
    Mercedes-AMG HPP 公开技术简报 (ERS 效率估算).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# FIA 2026 限制
_MAX_DEPLOY_MJ_PER_LAP = 4.0
_MAX_HARVEST_MJ_PER_LAP = 2.5
_BATTERY_CAPACITY_MJ = 4.0           # 电池容量 (MJ), 同单圈最大部署量
_DEFAULT_INITIAL_SOC = 0.5            # 起始 SoC 50%

# 工程化系数
_DEPLOY_GAIN_S_PER_MJ = 0.08          # 1 MJ 部署 ~0.08 s 直道收益
_HARVEST_DRAG_S_PER_MJ = 0.05         # 1 MJ 回收 ~0.05 s 圈速代价
_DEPLOY_BOOST_FACTOR = 1.3            # attack 模式效率增益
_CONSERVE_FACTOR = 0.7                # conserve 模式效率折扣
_DRAG_LIMITED_MIN_LENGTH_M = 600.0    # 长直道阈值 (m)


# --------------------------------------------------------------------------- #
# 赛道 ERS profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ERSDeploymentZone:
    """单条直道的 ERS 部署区段.

    - ``start_m`` / ``end_m``: 直道在赛道上的位置 (米, 起跑线为 0).
    - ``length_m``: 直道长度 (= end - start).
    - ``is_drag_limited``: 是否为 "drag-limited" (长直道, 末端速度受空气
      阻力限制, ERS 收益最大).
    - ``priority_weight``: 部署优先级 (0-1, 1=最高优先).
    """

    start_m: float
    end_m: float
    length_m: float
    is_drag_limited: bool
    priority_weight: float

    @classmethod
    def from_segment(
        cls, start_m: float, end_m: float, priority_weight: float | None = None
    ) -> ERSDeploymentZone:
        length = max(0.0, end_m - start_m)
        is_dl = length >= _DRAG_LIMITED_MIN_LENGTH_M
        if priority_weight is None:
            # 默认按长度归一化: 长直道权重高
            priority_weight = min(1.0, length / 1500.0)
        return cls(start_m=start_m, end_m=end_m, length_m=length,
                   is_drag_limited=is_dl, priority_weight=float(priority_weight))


@dataclass(frozen=True)
class ERSTrackProfile:
    """单赛道 ERS 直道分布 + 刹车回收区.

    - ``track_id``: 赛道 id.
    - ``deployment_zones``: 部署直道列表.
    - ``harvest_zones``: 回收 (刹车) 区列表, 每个含回收量 (MJ).
    - ``lap_length_m``: 单圈长度 (m), 用于校验.
    """

    track_id: str
    lap_length_m: float
    deployment_zones: tuple[ERSDeploymentZone, ...]
    harvest_zones: tuple[tuple[float, float, float], ...]  # (start, end, MJ)

    def total_deploy_length_m(self) -> float:
        return sum(z.length_m for z in self.deployment_zones)

    def total_harvest_mj(self) -> float:
        return sum(mj for _, _, mj in self.harvest_zones)

    def longest_drag_limited_zone(self) -> ERSDeploymentZone | None:
        dl = [z for z in self.deployment_zones if z.is_drag_limited]
        if not dl:
            return None
        return max(dl, key=lambda z: z.length_m)


# 12 条主要赛道 ERS profile (基于公开赛道图 + 直道位置估算)
ERS_TRACK_PROFILES: dict[str, ERSTrackProfile] = {
    "monza": ERSTrackProfile(
        track_id="monza", lap_length_m=5793.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 1100, 1.0),       # 主直道
            ERSDeploymentZone.from_segment(3300, 4100, 0.9),    # Curva Grande
            ERSDeploymentZone.from_segment(4800, 5400, 0.7),    # Ascari 前
        ),
        harvest_zones=((1000, 1100, 0.8), (4000, 4100, 0.6), (5300, 5400, 0.5)),
    ),
    "spa": ERSTrackProfile(
        track_id="spa", lap_length_m=7004.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 800, 1.0),       # Raidillon 后
            ERSDeploymentZone.from_segment(4400, 5600, 0.95),  # Kemmel 后到 Les Combes
            ERSDeploymentZone.from_segment(6200, 6800, 0.6),   # Blanchimont
        ),
        harvest_zones=((700, 800, 0.7), (5500, 5600, 0.5), (6700, 6800, 0.4)),
    ),
    "jeddah": ERSTrackProfile(
        track_id="jeddah", lap_length_m=6174.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 700, 0.9),
            ERSDeploymentZone.from_segment(2500, 3300, 0.95),
            ERSDeploymentZone.from_segment(4900, 5800, 1.0),    # 最长直道
        ),
        harvest_zones=((600, 700, 0.6), (3200, 3300, 0.5), (5700, 5800, 0.8)),
    ),
    "bahrain": ERSTrackProfile(
        track_id="bahrain", lap_length_m=5412.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 600, 0.85),
            ERSDeploymentZone.from_segment(2300, 3000, 0.75),
            ERSDeploymentZone.from_segment(4000, 4700, 0.9),
        ),
        harvest_zones=((500, 600, 0.6), (2900, 3000, 0.4), (4600, 4700, 0.6)),
    ),
    "silverstone": ERSTrackProfile(
        track_id="silverstone", lap_length_m=5891.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 700, 0.85),
            ERSDeploymentZone.from_segment(2500, 3100, 0.65),
            ERSDeploymentZone.from_segment(4900, 5400, 0.7),
        ),
        harvest_zones=((600, 700, 0.5), (3000, 3100, 0.4), (5300, 5400, 0.5)),
    ),
    "monaco": ERSTrackProfile(
        track_id="monaco", lap_length_m=3337.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 350, 0.55),     # 主直道 (短)
            ERSDeploymentZone.from_segment(1200, 1500, 0.4),
            ERSDeploymentZone.from_segment(2500, 2800, 0.45),
        ),
        harvest_zones=((300, 350, 0.3), (1450, 1500, 0.25), (2750, 2800, 0.3)),
    ),
    "suzuka": ERSTrackProfile(
        track_id="suzuka", lap_length_m=5807.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 600, 0.75),
            ERSDeploymentZone.from_segment(2800, 3400, 0.6),
            ERSDeploymentZone.from_segment(4700, 5300, 0.7),    # 130R 后
        ),
        harvest_zones=((500, 600, 0.5), (3300, 3400, 0.4), (5200, 5300, 0.5)),
    ),
    "melbourne": ERSTrackProfile(
        track_id="melbourne", lap_length_m=5303.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 700, 0.8),
            ERSDeploymentZone.from_segment(2400, 2900, 0.55),
            ERSDeploymentZone.from_segment(4200, 4800, 0.7),
        ),
        harvest_zones=((600, 700, 0.5), (2800, 2900, 0.4), (4700, 4800, 0.5)),
    ),
    "yas_marina": ERSTrackProfile(
        track_id="yas_marina", lap_length_m=5281.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 700, 0.85),
            ERSDeploymentZone.from_segment(2600, 3200, 0.75),
            ERSDeploymentZone.from_segment(4500, 5100, 0.8),
        ),
        harvest_zones=((600, 700, 0.6), (3100, 3200, 0.5), (5000, 5100, 0.6)),
    ),
    "shanghai": ERSTrackProfile(
        track_id="shanghai", lap_length_m=5451.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 1100, 1.0),     # 最长直道
            ERSDeploymentZone.from_segment(3200, 3700, 0.6),
            ERSDeploymentZone.from_segment(4700, 5200, 0.7),
        ),
        harvest_zones=((1000, 1100, 0.8), (3600, 3700, 0.4), (5100, 5200, 0.5)),
    ),
    "austin": ERSTrackProfile(
        track_id="austin", lap_length_m=5513.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 1000, 0.95),
            ERSDeploymentZone.from_segment(2800, 3300, 0.65),
            ERSDeploymentZone.from_segment(4700, 5200, 0.7),
        ),
        harvest_zones=((900, 1000, 0.7), (3200, 3300, 0.4), (5100, 5200, 0.5)),
    ),
    "interlagos": ERSTrackProfile(
        track_id="interlagos", lap_length_m=4309.0,
        deployment_zones=(
            ERSDeploymentZone.from_segment(0, 800, 0.95),     # 主直道
            ERSDeploymentZone.from_segment(2400, 2800, 0.6),
            ERSDeploymentZone.from_segment(3800, 4200, 0.75), # Subida do Lago
        ),
        harvest_zones=((700, 800, 0.6), (2700, 2800, 0.4), (4100, 4200, 0.5)),
    ),
}

_DEFAULT_PROFILE = ERSTrackProfile(
    track_id="unknown", lap_length_m=5000.0,
    deployment_zones=(
        ERSDeploymentZone.from_segment(0, 700, 0.7),
        ERSDeploymentZone.from_segment(2500, 3000, 0.55),
    ),
    harvest_zones=((600, 700, 0.5), (2900, 3000, 0.4)),
)


def get_ers_profile(track_id: str) -> ERSTrackProfile:
    """获取赛道 ERS profile, 未知返回默认 (单圈长 5000 m)."""
    return ERS_TRACK_PROFILES.get(track_id, _DEFAULT_PROFILE)


# --------------------------------------------------------------------------- #
# ERS 部署模型
# --------------------------------------------------------------------------- #
@dataclass
class ERSDeploymentModel:
    """单圈 ERS 部署 + 回收仿真, 跟踪电池 SoC.

    用法::

        m = ERSDeploymentModel(track_id="monza", mode="balanced",
                               initial_soc=0.6)
        result = m.simulate_lap()
        # result = {deploy_mj, harvest_mj, soc_after, lap_gain_s, ...}
        laps = m.simulate_stint(laps=20)
    """

    track_id: str
    mode: str = "balanced"        # attack / balanced / conserve
    initial_soc: float = _DEFAULT_INITIAL_SOC
    profile: ERSTrackProfile = field(init=False, repr=False)
    _soc: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("attack", "balanced", "conserve"):
            self.mode = "balanced"
        self.initial_soc = max(0.0, min(1.0, float(self.initial_soc)))
        self.profile = get_ers_profile(self.track_id)
        self._soc = self.initial_soc

    @property
    def soc(self) -> float:
        return self._soc

    def reset(self) -> None:
        self._soc = self.initial_soc

    # ------------------------------------------------------------------ #
    # 部署分配 (MJ per zone)
    # ------------------------------------------------------------------ #
    def _allocate_deployment(self) -> list[tuple[ERSDeploymentZone, float]]:
        """按 priority_weight 在所有部署区之间分配 4 MJ 预算."""
        zones = list(self.profile.deployment_zones)
        if not zones:
            return []
        total_weight = sum(z.priority_weight for z in zones)
        if total_weight <= 0:
            total_weight = float(len(zones))
        # 模式影响预算
        if self.mode == "attack":
            budget = _MAX_DEPLOY_MJ_PER_LAP * _DEPLOY_BOOST_FACTOR
        elif self.mode == "conserve":
            budget = _MAX_DEPLOY_MJ_PER_LAP * _CONSERVE_FACTOR
        else:
            budget = _MAX_DEPLOY_MJ_PER_LAP

        # 受 SoC 限制: 最多用完电池剩余
        soc_available_mj = self._soc * _BATTERY_CAPACITY_MJ
        budget = min(budget, soc_available_mj)

        # 按权重分配
        out: list[tuple[ERSDeploymentZone, float]] = []
        for z in zones:
            share = (z.priority_weight / total_weight) * budget
            out.append((z, share))
        return out

    # ------------------------------------------------------------------ #
    # 回收分配
    # ------------------------------------------------------------------ #
    def _allocate_harvest(self) -> float:
        """总回收量, 受 FIA 限额 2.5 MJ 限制."""
        requested = self.profile.total_harvest_mj()
        # 攻击模式: 减少回收 (省刹车阻力), 保守模式: 增加回收
        if self.mode == "attack":
            requested *= 0.8
        elif self.mode == "conserve":
            requested *= 1.1
        return min(requested, _MAX_HARVEST_MJ_PER_LAP)

    # ------------------------------------------------------------------ #
    # 单圈仿真
    # ------------------------------------------------------------------ #
    def simulate_lap(self) -> dict[str, Any]:
        """仿真单圈 ERS 部署/回收, 更新 SoC, 返回圈速收益 (s)."""
        soc_before = self._soc  # capture before any update
        alloc = self._allocate_deployment()
        harvest = self._allocate_harvest()

        # 部署收益
        total_deploy = sum(mj for _, mj in alloc)
        if self.mode == "attack":
            gain = total_deploy * _DEPLOY_GAIN_S_PER_MJ * 1.1   # 攻击模式效率更高
        elif self.mode == "conserve":
            gain = total_deploy * _DEPLOY_GAIN_S_PER_MJ * 0.95
        else:
            gain = total_deploy * _DEPLOY_GAIN_S_PER_MJ

        # 回收代价 (发动机阻力 + 电池充电损耗)
        drag_cost = harvest * _HARVEST_DRAG_S_PER_MJ

        # 更新 SoC
        net_mj = harvest - total_deploy
        self._soc = max(0.0, min(1.0, self._soc + net_mj / _BATTERY_CAPACITY_MJ))

        # 按区输出明细
        zone_breakdown = [
            {
                "start_m": z.start_m,
                "end_m": z.end_m,
                "length_m": z.length_m,
                "is_drag_limited": z.is_drag_limited,
                "mj_deployed": mj,
                "gain_s": mj * _DEPLOY_GAIN_S_PER_MJ,
            }
            for z, mj in alloc
        ]

        return {
            "track_id": self.track_id,
            "mode": self.mode,
            "deploy_mj": float(total_deploy),
            "harvest_mj": float(harvest),
            "net_mj": float(net_mj),
            "soc_before": float(soc_before),
            "soc_after": float(self._soc),
            "lap_gain_s": float(gain),                  # 正 = 圈速更快
            "lap_drag_cost_s": float(drag_cost),         # 正 = 圈速更慢
            "net_lap_gain_s": float(gain - drag_cost),
            "zone_breakdown": zone_breakdown,
        }

    # ------------------------------------------------------------------ #
    # 多圈仿真 (stint)
    # ------------------------------------------------------------------ #
    def simulate_stint(self, laps: int) -> list[dict[str, Any]]:
        """仿真多圈 ERS, SoC 跨圈传递."""
        self.reset()
        out: list[dict[str, Any]] = []
        for k in range(int(laps)):
            r = self.simulate_lap()
            r["lap"] = k + 1
            out.append(r)
        return out

    # ------------------------------------------------------------------ #
    # 推荐模式 (基于当前 SoC)
    # ------------------------------------------------------------------ #
    def recommend_mode(self, target_soc: float = 0.5) -> str:
        """基于当前 SoC 与目标 SoC 推荐模式.

        - SoC > target+0.15: attack (有多余电量)
        - SoC < target-0.15: conserve (省电)
        - 否则: balanced
        """
        if self._soc > target_soc + 0.15:
            return "attack"
        if self._soc < target_soc - 0.15:
            return "conserve"
        return "balanced"
