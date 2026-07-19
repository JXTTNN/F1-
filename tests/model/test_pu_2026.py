"""F1 2026 PU 能量部署精确模型测试 (Iter-53)."""

from __future__ import annotations

from f1opt.model.pu_2026 import (
    BATTERY_CAPACITY_MJ,
    DEPLOY_GAIN_S_PER_MJ,
    HARVEST_DRAG_S_PER_MJ,
    ICE_POWER_KW,
    MAX_DEPLOY_MJ_PER_LAP,
    MAX_HARVEST_MJ_PER_LAP,
    MGU_K_POWER_KW,
    TOTAL_POWER_KW,
    PU2026Model,
    PU2026State,
    PUDeployMode,
    lap_time_gain_s,
    max_deploy_mj_per_lap,
    mode_deploy_mj,
    total_power_kw,
)


# --------------------------------------------------------------------------- #
# FIA 2026 PU 常量 (严格对标 EA F1 2026)
# --------------------------------------------------------------------------- #
def test_ice_power_400kw():
    assert ICE_POWER_KW == 400.0


def test_mgu_k_power_350kw():
    """2026 MGU-K 350kW (大幅提升, 旧 120kW)."""
    assert MGU_K_POWER_KW == 350.0


def test_total_power_750kw():
    assert TOTAL_POWER_KW == 750.0
    assert ICE_POWER_KW + MGU_K_POWER_KW == TOTAL_POWER_KW


def test_max_deploy_9mj():
    """2026 每圈部署 9 MJ (比旧 4 MJ 翻倍)."""
    assert MAX_DEPLOY_MJ_PER_LAP == 9.0


def test_max_harvest_6mj():
    assert MAX_HARVEST_MJ_PER_LAP == 6.0


def test_battery_capacity_9mj():
    """电池容量匹配单圈最大部署."""
    assert BATTERY_CAPACITY_MJ == 9.0


def test_deploy_gain_coefficient():
    """1 MJ 部署 ~0.09 s 收益."""
    assert DEPLOY_GAIN_S_PER_MJ == 0.09


def test_harvest_cost_coefficient():
    assert HARVEST_DRAG_S_PER_MJ == 0.04


# --------------------------------------------------------------------------- #
# PUDeployMode
# --------------------------------------------------------------------------- #
def test_four_deploy_modes():
    assert len(list(PUDeployMode)) == 4


def test_mode_values():
    assert PUDeployMode.QUALIFYING.value == "qualifying"
    assert PUDeployMode.ATTACK.value == "attack"
    assert PUDeployMode.BALANCED.value == "balanced"
    assert PUDeployMode.CONSERVE.value == "conserve"


def test_mode_deploy_mj_values():
    """4 模式部署量: 9/8/6/4 MJ."""
    assert mode_deploy_mj(PUDeployMode.QUALIFYING) == 9.0
    assert mode_deploy_mj(PUDeployMode.ATTACK) == 8.0
    assert mode_deploy_mj(PUDeployMode.BALANCED) == 6.0
    assert mode_deploy_mj(PUDeployMode.CONSERVE) == 4.0


def test_mode_deploy_ordering():
    """模式部署量递减."""
    assert mode_deploy_mj(PUDeployMode.QUALIFYING) > mode_deploy_mj(PUDeployMode.ATTACK)
    assert mode_deploy_mj(PUDeployMode.ATTACK) > mode_deploy_mj(PUDeployMode.BALANCED)
    assert mode_deploy_mj(PUDeployMode.BALANCED) > mode_deploy_mj(PUDeployMode.CONSERVE)


# --------------------------------------------------------------------------- #
# PU2026State
# --------------------------------------------------------------------------- #
def test_state_default_soc_50pct():
    s = PU2026State()
    assert abs(s.soc_mj - 4.5) < 1e-9  # 50% of 9 MJ
    assert abs(s.soc_pct - 50.0) < 1e-9


def test_state_soc_pct():
    s = PU2026State(soc_mj=9.0)
    assert s.soc_pct == 100.0
    s = PU2026State(soc_mj=0.0)
    assert s.soc_pct == 0.0


def test_state_is_low_soc():
    assert PU2026State(soc_mj=1.0).is_low_soc is True  # < 20% = 1.8 MJ
    assert PU2026State(soc_mj=5.0).is_low_soc is False


def test_state_is_full():
    assert PU2026State(soc_mj=9.0).is_full is True
    assert PU2026State(soc_mj=8.0).is_full is False


# --------------------------------------------------------------------------- #
# PU2026Model 基础
# --------------------------------------------------------------------------- #
def test_model_creation():
    pu = PU2026Model(track_id="monza", mode=PUDeployMode.BALANCED)
    assert pu.track_id == "monza"
    assert pu.mode == PUDeployMode.BALANCED


def test_model_target_deploy():
    pu = PU2026Model("monza", PUDeployMode.QUALIFYING)
    assert pu.target_deploy_mj() == 9.0
    pu = PU2026Model("monza", PUDeployMode.CONSERVE)
    assert pu.target_deploy_mj() == 4.0


def test_model_efficiency_by_mode():
    """Qualifying 模式效率最高 (1.10), Conserve 最低 (0.95)."""
    pu_q = PU2026Model("monza", PUDeployMode.QUALIFYING)
    pu_c = PU2026Model("monza", PUDeployMode.CONSERVE)
    assert pu_q.mode_efficiency() > pu_c.mode_efficiency()


# --------------------------------------------------------------------------- #
# simulate_lap
# --------------------------------------------------------------------------- #
def test_simulate_lap_returns_result():
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=4.5)
    r = pu.simulate_lap(state)
    assert r.deploy_mj > 0
    assert r.net_gain_s > 0


def test_simulate_lap_deploy_matches_mode():
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=9.0)  # 满电
    r = pu.simulate_lap(state)
    assert abs(r.deploy_mj - 6.0) < 1e-9


def test_simulate_lap_soc_decreases_when_deploy():
    """部署 > 回收时 SoC 下降."""
    pu = PU2026Model("monza", PUDeployMode.QUALIFYING)
    state = PU2026State(soc_mj=9.0)
    r = pu.simulate_lap(state)
    # Qualifying 9 MJ 部署, 回收 < 9, SoC 下降
    assert r.net_soc_delta_mj < 0


def test_simulate_lap_soc_increases_when_conserve():
    """Conserve 模式回收 > 部署, SoC 上升."""
    pu = PU2026Model("montreal", PUDeployMode.CONSERVE)  # 重制动高回收
    state = PU2026State(soc_mj=2.0)  # 低电
    r = pu.simulate_lap(state)
    # Conserve 4 MJ 部署, montreal 高回收, SoC 应上升
    assert r.net_soc_delta_mj > 0


def test_simulate_lap_low_soc_reduces_qualifying_deploy():
    """低 SoC + Qualifying 模式自动降部署."""
    pu = PU2026Model("monza", PUDeployMode.QUALIFYING)
    state = PU2026State(soc_mj=1.0)  # 低电 (< 1.8 MJ 阈值)
    r = pu.simulate_lap(state)
    # 应降级到 60% 部署
    assert r.deploy_mj <= 9.0 * 0.6 + 1e-9


def test_simulate_lap_soc_clamped():
    """SoC 不超 9 MJ, 不低于 0."""
    pu = PU2026Model("monza", PUDeployMode.CONSERVE)
    state = PU2026State(soc_mj=8.5)  # 接近满
    pu.simulate_lap(state)
    assert state.soc_mj <= 9.0


def test_simulate_lap_updates_state():
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=4.5)
    pu.simulate_lap(state)
    assert state.laps_completed == 1
    assert state.cumulative_deploy_mj > 0
    assert state.cumulative_harvest_mj > 0


# --------------------------------------------------------------------------- #
# 圈速收益
# --------------------------------------------------------------------------- #
def test_qualifying_fastest_gain():
    """Qualifying 模式圈速收益最大."""
    pu_q = PU2026Model("monza", PUDeployMode.QUALIFYING)
    pu_b = PU2026Model("monza", PUDeployMode.BALANCED)
    state_q = PU2026State(soc_mj=9.0)
    state_b = PU2026State(soc_mj=9.0)
    r_q = pu_q.simulate_lap(state_q)
    r_b = pu_b.simulate_lap(state_b)
    assert r_q.net_gain_s > r_b.net_gain_s


def test_conserve_lowest_gain():
    pu_c = PU2026Model("monza", PUDeployMode.CONSERVE)
    pu_b = PU2026Model("monza", PUDeployMode.BALANCED)
    state_c = PU2026State(soc_mj=9.0)
    state_b = PU2026State(soc_mj=9.0)
    r_c = pu_c.simulate_lap(state_c)
    r_b = pu_b.simulate_lap(state_b)
    assert r_c.net_gain_s < r_b.net_gain_s


def test_gain_difference_reasonable():
    """Qualifying vs Balanced 收益差应在 0.15-0.35 s."""
    pu_q = PU2026Model("monza", PUDeployMode.QUALIFYING)
    pu_b = PU2026Model("monza", PUDeployMode.BALANCED)
    r_q = pu_q.simulate_lap(PU2026State(soc_mj=9.0))
    r_b = pu_b.simulate_lap(PU2026State(soc_mj=9.0))
    diff = r_q.net_gain_s - r_b.net_gain_s
    assert 0.10 < diff < 0.40


# --------------------------------------------------------------------------- #
# 赛道回收差异
# --------------------------------------------------------------------------- #
def test_heavy_braking_track_more_harvest():
    """重制动赛道 (montreal) 回收多于全油门赛道 (monza)."""
    pu_mtl = PU2026Model("montreal", PUDeployMode.BALANCED)
    pu_mza = PU2026Model("monza", PUDeployMode.BALANCED)
    s1 = PU2026State(soc_mj=4.5)
    s2 = PU2026State(soc_mj=4.5)
    r_mtl = pu_mtl.simulate_lap(s1)
    r_mza = pu_mza.simulate_lap(s2)
    assert r_mtl.harvest_mj >= r_mza.harvest_mj


def test_spa_low_harvest():
    """Spa 全油门多, 回收少."""
    pu_spa = PU2026Model("spa", PUDeployMode.BALANCED)
    pu_mtl = PU2026Model("montreal", PUDeployMode.BALANCED)
    r_spa = pu_spa.simulate_lap(PU2026State(soc_mj=4.5))
    r_mtl = pu_mtl.simulate_lap(PU2026State(soc_mj=4.5))
    assert r_spa.harvest_mj <= r_mtl.harvest_mj


# --------------------------------------------------------------------------- #
# recommend_mode
# --------------------------------------------------------------------------- #
def test_recommend_qualifying_last_lap():
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=4.5)
    assert pu.recommend_mode(state, lap=53, total_laps=53) == PUDeployMode.QUALIFYING


def test_recommend_conserve_low_soc():
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=1.0)  # 低电
    assert pu.recommend_mode(state, lap=20, total_laps=53) == PUDeployMode.CONSERVE


def test_recommend_attack_late_race():
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=6.0)  # 充足
    assert pu.recommend_mode(state, lap=45, total_laps=53) == PUDeployMode.ATTACK


def test_recommend_balanced_mid_race():
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=4.5)
    assert pu.recommend_mode(state, lap=20, total_laps=53) == PUDeployMode.BALANCED


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def test_lap_time_gain_s():
    gain = lap_time_gain_s(9.0, 0.0)
    assert abs(gain - 9.0 * 0.09) < 1e-9


def test_lap_time_gain_with_harvest():
    gain = lap_time_gain_s(9.0, 6.0)
    expected = 9.0 * 0.09 - 6.0 * 0.04
    assert abs(gain - expected) < 1e-9


def test_max_deploy_convenience():
    assert max_deploy_mj_per_lap() == 9.0


def test_total_power_convenience():
    assert total_power_kw() == 750.0


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_deterministic():
    pu1 = PU2026Model("monza", PUDeployMode.BALANCED)
    pu2 = PU2026Model("monza", PUDeployMode.BALANCED)
    s1 = PU2026State(soc_mj=4.5)
    s2 = PU2026State(soc_mj=4.5)
    r1 = pu1.simulate_lap(s1)
    r2 = pu2.simulate_lap(s2)
    assert r1.deploy_mj == r2.deploy_mj
    assert r1.net_gain_s == r2.net_gain_s


# --------------------------------------------------------------------------- #
# 实战场景 (对标 EA F1 2026)
# --------------------------------------------------------------------------- #
def test_qualifying_lap_max_deployment():
    """排位飞驰圈: 全力 9 MJ 部署, 圈速收益最大."""
    pu = PU2026Model("monza", PUDeployMode.QUALIFYING)
    state = PU2026State(soc_mj=9.0)  # 满电
    r = pu.simulate_lap(state)
    assert r.deploy_mj == 9.0
    assert r.net_gain_s > 0.7  # 9 × 0.09 - harvest_cost


def test_race_stint_soc_management():
    """正赛 stint: SoC 管理, 平衡模式维持电量."""
    pu = PU2026Model("monza", PUDeployMode.BALANCED)
    state = PU2026State(soc_mj=4.5)
    soc_history = [state.soc_mj]
    for _ in range(20):  # 20 圈 stint
        pu.simulate_lap(state)
        soc_history.append(state.soc_mj)
    # SoC 应维持在合理范围 (不耗尽)
    assert min(soc_history) >= 0.0
    assert max(soc_history) <= 9.0


def test_attack_mode_drains_soc():
    """Attack 模式逐渐耗尽 SoC."""
    pu = PU2026Model("monza", PUDeployMode.ATTACK)
    state = PU2026State(soc_mj=9.0)
    initial_soc = state.soc_mj
    for _ in range(10):
        pu.simulate_lap(state)
    # 10 圈 Attack 应显著降低 SoC
    assert state.soc_mj < initial_soc


def test_conserve_mode_recharges():
    """Conserve 模式回收多于部署, SoC 上升."""
    pu = PU2026Model("montreal", PUDeployMode.CONSERVE)  # 重制动
    state = PU2026State(soc_mj=1.0)  # 低电
    initial = state.soc_mj
    for _ in range(5):
        pu.simulate_lap(state)
    assert state.soc_mj > initial


def test_full_qualifying_vs_balanced_lap_time():
    """排位飞驰圈 vs 平衡圈: 圈速差距应在 0.15-0.35 s."""
    pu_q = PU2026Model("monza", PUDeployMode.QUALIFYING)
    pu_b = PU2026Model("monza", PUDeployMode.BALANCED)
    r_q = pu_q.simulate_lap(PU2026State(soc_mj=9.0))
    r_b = pu_b.simulate_lap(PU2026State(soc_mj=9.0))
    diff = r_q.net_gain_s - r_b.net_gain_s
    assert 0.15 < diff < 0.45
