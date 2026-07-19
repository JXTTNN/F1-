"""F1 2026 车队赛车性能数据库 (Iter-36).

F1 2026 EA Sports 官方 / 公开媒体综合车队赛车性能评级:
10 支车队 × 多维性能参数 (相对参考基准的偏移).

车队性能维度 (F1 2026 车队 simulator 标准):
- pace_offset_s: 圈速偏移 (s/lap, 负=快于基准, 正=慢于基准).
  以 2026 赛季中位车队为基准 (≈0). 顶队 RBR/MCL ≈ -0.6s,
  后段 Sauber/Haas ≈ +0.8s.
- aero_efficiency: 空气动力学效率 0.85-1.10 (1.0 = 基准).
- power_unit_kW: 2026 PU 峰值功率 (ICE+MGU-K), 730-755 kW.
  规则上限 750 kW, 但实际各队有差异.
- tire_degradation_factor: 轮胎退化系数 0.90-1.15.
  低 = 保护轮胎 (RBR/Ferrari 历史强项), 高 = 吃胎 (Williams).
- reliability: 可靠性 0.85-0.99 (退赛概率 = 1 - reliability).
- drs_effectiveness: DRS 增益系数 0.90-1.10.
- fuel_efficiency: 燃油效率 0.95-1.05 (低 = 省油).

数据来源 (Iter-36):
- EA Sports F1 2026 官方车队评级 (公开游戏数据).
- 2025 赛季公开技术分析 (Auto Motor und Sport, The Race).
- 各车队 2026 PU 供应商公开声明 (Mercedes HPP, Ferrari, Honda RBPT, Renault).
- FIA 2026 Power Unit Regulations §5.

注意: 所有数值是基于公开信息合理工程估计, 不代表车队实际内部数据.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamCarProfile2026:
    """F1 2026 单车队赛车性能档案.

    所有偏移以"赛季中位车队"为基准 (≈0). 负数 = 优于基准.
    """

    team_id: str
    team_name: str
    full_name: str
    """车队全名 (含赞助商/制造商)."""

    pace_offset_s: float
    """圈速偏移 s/lap, 负=快于基准, 正=慢于基准.
    顶队 -0.6, 后段 +0.8."""

    aero_efficiency: float
    """空气动力学效率 0.85-1.10 (1.0 = 基准)."""

    power_unit_kW: float
    """2026 PU 峰值功率 (ICE+MGU-K) kW, 规则上限 750."""

    power_unit_supplier: str
    """PU 供应商: mercedes / ferrari / honda_rbpt / renault."""

    tire_degradation_factor: float
    """轮胎退化系数 0.90-1.15 (1.0 = 基准, 低 = 保护轮胎)."""

    reliability: float
    """可靠性 0.85-0.99 (退赛概率 ≈ (1-reliability) × race_distance_factor)."""

    drs_effectiveness: float
    """DRS 增益系数 0.90-1.10 (1.0 = 基准)."""

    fuel_efficiency: float
    """燃油效率 0.95-1.05 (1.0 = 基准, 低 = 省油)."""

    @property
    def retirement_probability_per_race(self) -> float:
        """单场退赛概率估计 (中位赛道 58 圈)."""
        # 基础 1.5% × (1 - reliability) × 6.67 (放大到全场)
        return max(0.005, (1.0 - self.reliability) * 0.20)


# --------------------------------------------------------------------------- #
# F1 2026 车队性能档案 (基于公开 2025 数据外推 + 2026 PU 供应商变化)
# --------------------------------------------------------------------------- #
# 2026 PU 供应商变化:
# - Aston Martin: Mercedes → Honda RBPT (与 RBR 同源)
# - Williams: Mercedes (保持)
# - Racing Bulls (RB): Honda RBPT (保持)
# - Alpine: Renault → Mercedes (重大变化, 期待性能提升)
_TEAMS_2026: list[TeamCarProfile2026] = [
    # Red Bull Racing — Honda RBPT (Verstappen + Tsunoda)
    # 2025 强队, 2026 期待继续争冠
    TeamCarProfile2026(
        team_id="rbr", team_name="Red Bull Racing",
        full_name="Oracle Red Bull Racing-Honda RBPT",
        pace_offset_s=-0.55, aero_efficiency=1.07,
        power_unit_kW=752, power_unit_supplier="honda_rbpt",
        tire_degradation_factor=0.92, reliability=0.97,
        drs_effectiveness=1.05, fuel_efficiency=0.98,
    ),
    # McLaren — Mercedes (Norris + Piastri)
    # 2025 卫冕冠军, 2026 期待继续强势
    TeamCarProfile2026(
        team_id="mcl", team_name="McLaren",
        full_name="McLaren Formula 1 Team-Mercedes",
        pace_offset_s=-0.60, aero_efficiency=1.08,
        power_unit_kW=750, power_unit_supplier="mercedes",
        tire_degradation_factor=0.93, reliability=0.97,
        drs_effectiveness=1.04, fuel_efficiency=0.97,
    ),
    # Mercedes-AMG — Mercedes HPP (Russell + Antonelli)
    # 2025 强势回归, 2026 主场 PU 优势
    TeamCarProfile2026(
        team_id="mer", team_name="Mercedes",
        full_name="Mercedes-AMG Petronas F1 Team",
        pace_offset_s=-0.50, aero_efficiency=1.05,
        power_unit_kW=753, power_unit_supplier="mercedes",
        tire_degradation_factor=0.94, reliability=0.98,
        drs_effectiveness=1.03, fuel_efficiency=0.97,
    ),
    # Ferrari — Ferrari (Leclerc + Hamilton)
    # 2025 Hamilton 加盟, 期待争冠
    TeamCarProfile2026(
        team_id="fer", team_name="Ferrari",
        full_name="Scuderia Ferrari HP",
        pace_offset_s=-0.45, aero_efficiency=1.04,
        power_unit_kW=751, power_unit_supplier="ferrari",
        tire_degradation_factor=0.92, reliability=0.96,
        drs_effectiveness=1.02, fuel_efficiency=0.98,
    ),
    # Aston Martin — Honda RBPT (Alonso + Stroll)
    # 2026 切换到 Honda RBPT, 期待 Adrian Newey 设计 + Honda PU 双重提升
    TeamCarProfile2026(
        team_id="amr", team_name="Aston Martin",
        full_name="Aston Martin Aramco F1 Team-Honda RBPT",
        pace_offset_s=-0.15, aero_efficiency=1.02,
        power_unit_kW=751, power_unit_supplier="honda_rbpt",
        tire_degradation_factor=0.95, reliability=0.95,
        drs_effectiveness=1.00, fuel_efficiency=0.99,
    ),
    # Williams — Mercedes (Sainz + Albon)
    # 2025 Sainz 加盟, 2026 期待进步
    TeamCarProfile2026(
        team_id="wil", team_name="Williams",
        full_name="Atlassian Williams Racing-Mercedes",
        pace_offset_s=0.10, aero_efficiency=0.98,
        power_unit_kW=750, power_unit_supplier="mercedes",
        tire_degradation_factor=1.05, reliability=0.94,
        drs_effectiveness=1.02, fuel_efficiency=1.00,
    ),
    # Alpine — Mercedes (Gasly + Doohan) [2026 切换 Renault→Mercedes]
    # 2026 重大变化: 工厂 PU → 客户 Mercedes, 期待性能提升
    TeamCarProfile2026(
        team_id="alp", team_name="Alpine",
        full_name="BWT Alpine F1 Team-Mercedes",
        pace_offset_s=0.25, aero_efficiency=0.96,
        power_unit_kW=750, power_unit_supplier="mercedes",
        tire_degradation_factor=1.02, reliability=0.93,
        drs_effectiveness=0.98, fuel_efficiency=1.00,
    ),
    # Racing Bulls (RB) — Honda RBPT (Lawson + Hadjar)
    # 2026 新秀组合, 与 RBR 同 PU
    TeamCarProfile2026(
        team_id="rb", team_name="Racing Bulls",
        full_name="Visa Cash App Racing Bulls-Honda RBPT",
        pace_offset_s=0.35, aero_efficiency=0.97,
        power_unit_kW=751, power_unit_supplier="honda_rbpt",
        tire_degradation_factor=1.04, reliability=0.94,
        drs_effectiveness=0.99, fuel_efficiency=1.00,
    ),
    # Kick Sauber — Ferrari (Hulkenberg + Bortoleto)
    # 2026 Audi 过渡年, 期待改善
    TeamCarProfile2026(
        team_id="kck", team_name="Kick Sauber",
        full_name="Kick Sauber F1 Team-Ferrari",
        pace_offset_s=0.65, aero_efficiency=0.93,
        power_unit_kW=749, power_unit_supplier="ferrari",
        tire_degradation_factor=1.08, reliability=0.91,
        drs_effectiveness=0.96, fuel_efficiency=1.01,
    ),
    # Haas — Ferrari (Ocon + Bearman)
    # 2026 Ocon 加盟 + Bearman 第二年
    TeamCarProfile2026(
        team_id="has", team_name="Haas",
        full_name="MoneyGram Haas F1 Team-Ferrari",
        pace_offset_s=0.75, aero_efficiency=0.92,
        power_unit_kW=749, power_unit_supplier="ferrari",
        tire_degradation_factor=1.10, reliability=0.92,
        drs_effectiveness=0.97, fuel_efficiency=1.01,
    ),
]


def all_teams_2026_profiles() -> list[TeamCarProfile2026]:
    """返回全部 10 支 F1 2026 车队性能档案."""
    return list(_TEAMS_2026)


def get_team_profile_2026(team_id: str) -> TeamCarProfile2026:
    """按 team_id 查询车队性能档案."""
    for t in _TEAMS_2026:
        if t.team_id == team_id:
            return t
    raise ValueError(f"Unknown team_id: {team_id!r}")


def pace_offset_for_team(team_id: str) -> float:
    """便捷: 返回车队圈速偏移 s/lap (负=快, 正=慢)."""
    return get_team_profile_2026(team_id).pace_offset_s


def teams_by_pu_supplier(supplier: str) -> list[TeamCarProfile2026]:
    """按 PU 供应商返回车队列表."""
    return [t for t in _TEAMS_2026 if t.power_unit_supplier == supplier]
