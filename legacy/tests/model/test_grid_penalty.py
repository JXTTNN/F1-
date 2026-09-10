"""F1 2026 PU 组件配额与罚位模型测试 (Iter-44)."""

from __future__ import annotations

import pytest

from f1opt.model.grid_penalty import (
    PU_COMPONENTS_2026,
    GridPenaltyCalculator,
    PUComponentInventory,
    apply_grid_penalties,
)


# --------------------------------------------------------------------------- #
# PU_COMPONENTS_2026 常量
# --------------------------------------------------------------------------- #
def test_components_quota_values():
    assert PU_COMPONENTS_2026["ice"] == 4
    assert PU_COMPONENTS_2026["mgu_k"] == 4
    assert PU_COMPONENTS_2026["tc"] == 4
    assert PU_COMPONENTS_2026["es"] == 4
    assert PU_COMPONENTS_2026["ce"] == 4
    assert PU_COMPONENTS_2026["exhaust"] == 8


def test_components_count_six_types():
    """2026 PU 5 大件 + 排气 = 6 类组件."""
    assert len(PU_COMPONENTS_2026) == 6


def test_no_mgu_h_in_2026():
    """2026 移除 MGU-H."""
    assert "mgu_h" not in PU_COMPONENTS_2026


# --------------------------------------------------------------------------- #
# PUComponentInventory 基础
# --------------------------------------------------------------------------- #
def test_inventory_starts_zero():
    inv = PUComponentInventory(driver_id="ver")
    for c in PU_COMPONENTS_2026:
        assert inv.used[c] == 0
        assert inv.exceeded[c] == 0
    assert inv.remaining("ice") == 4
    assert inv.remaining("exhaust") == 8
    assert inv.total_grid_penalty() == 0
    assert not inv.needs_pit_lane_start()


def test_install_within_quota_no_penalty():
    inv = PUComponentInventory(driver_id="ver")
    # 装 4 个 ICE (配额内)
    for i in range(4):
        triggered = inv.install_component("ice", race_idx=i)
        assert triggered is False
    assert inv.is_over_limit("ice") is False
    assert inv.remaining("ice") == 0
    assert inv.total_grid_penalty() == 0


def test_install_first_exceed_triggers_penalty():
    inv = PUComponentInventory(driver_id="ver")
    for i in range(4):
        inv.install_component("ice", race_idx=i)
    # 第 5 个超额
    triggered = inv.install_component("ice", race_idx=4)
    assert triggered is True
    assert inv.is_over_limit("ice") is True
    assert inv.exceeded["ice"] == 1
    assert inv.penalty_for_component("ice") == 5
    assert inv.total_grid_penalty() == 5
    assert not inv.needs_pit_lane_start()  # 5 < 15


def test_install_second_exceed_accumulates():
    inv = PUComponentInventory(driver_id="ver")
    for i in range(6):  # 4 + 2 超额
        inv.install_component("ice", race_idx=i)
    assert inv.exceeded["ice"] == 2
    assert inv.penalty_for_component("ice") == 10
    assert inv.total_grid_penalty() == 10


def test_install_third_exceed_pit_lane():
    inv = PUComponentInventory(driver_id="ver")
    for i in range(8):  # 4 + 4 超额
        inv.install_component("ice", race_idx=i)
    assert inv.exceeded["ice"] == 4
    assert inv.total_grid_penalty() == 20
    assert inv.needs_pit_lane_start()  # 20 > 15


def test_install_unknown_component_raises():
    inv = PUComponentInventory(driver_id="ver")
    with pytest.raises(ValueError, match="Unknown PU component"):
        inv.install_component("mgu_h")  # 2026 不存在


def test_install_unknown_component_random():
    inv = PUComponentInventory(driver_id="ver")
    with pytest.raises(ValueError):
        inv.install_component("nonexistent")


def test_history_recorded():
    inv = PUComponentInventory(driver_id="ver")
    inv.install_component("ice", race_idx=0)
    inv.install_component("mgu_k", race_idx=0)
    inv.install_component("ice", race_idx=3)
    assert len(inv.history) == 3
    assert inv.history[0] == (0, "ice")
    assert inv.history[1] == (0, "mgu_k")
    assert inv.history[2] == (3, "ice")


def test_remaining_decreases():
    inv = PUComponentInventory(driver_id="ver")
    assert inv.remaining("ice") == 4
    inv.install_component("ice")
    assert inv.remaining("ice") == 3
    inv.install_component("ice")
    assert inv.remaining("ice") == 2


def test_remaining_zero_after_full_use():
    inv = PUComponentInventory(driver_id="ver")
    for _ in range(4):
        inv.install_component("ice")
    assert inv.remaining("ice") == 0


def test_remaining_negative_clipped():
    """超额后 remaining 仍返回 0 (不返回负数)."""
    inv = PUComponentInventory(driver_id="ver")
    for _ in range(6):
        inv.install_component("ice")
    assert inv.remaining("ice") == 0


# --------------------------------------------------------------------------- #
# 多组件罚位累计
# --------------------------------------------------------------------------- #
def test_multiple_components_accumulate():
    inv = PUComponentInventory(driver_id="ver")
    # ICE 超 1 次 (5 位)
    for _ in range(5):
        inv.install_component("ice")
    # MGU-K 超 1 次 (5 位)
    for _ in range(5):
        inv.install_component("mgu_k")
    assert inv.total_grid_penalty() == 10
    assert not inv.needs_pit_lane_start()


def test_three_components_pit_lane():
    inv = PUComponentInventory(driver_id="ver")
    # ICE 超 1 (5)
    for _ in range(5):
        inv.install_component("ice")
    # MGU-K 超 1 (5)
    for _ in range(5):
        inv.install_component("mgu_k")
    # TC 超 1 (5)
    for _ in range(5):
        inv.install_component("tc")
    assert inv.total_grid_penalty() == 15
    # 15 不 > 15, 所以不算 pit lane start
    assert not inv.needs_pit_lane_start()


def test_four_components_pit_lane():
    inv = PUComponentInventory(driver_id="ver")
    for _ in range(5):
        inv.install_component("ice")
    for _ in range(5):
        inv.install_component("mgu_k")
    for _ in range(5):
        inv.install_component("tc")
    for _ in range(5):
        inv.install_component("es")
    assert inv.total_grid_penalty() == 20
    assert inv.needs_pit_lane_start()


def test_exhaust_higher_quota():
    """排气配额 8, 前 8 次不罚."""
    inv = PUComponentInventory(driver_id="ver")
    for i in range(8):
        triggered = inv.install_component("exhaust", race_idx=i)
        assert triggered is False
    assert inv.total_grid_penalty() == 0
    # 第 9 次超额
    triggered = inv.install_component("exhaust", race_idx=8)
    assert triggered is True
    assert inv.total_grid_penalty() == 5


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def test_summary_structure():
    inv = PUComponentInventory(driver_id="ver")
    inv.install_component("ice")
    inv.install_component("ice")
    s = inv.summary()
    assert s["driver_id"] == "ver"
    assert s["used"]["ice"] == 2
    assert s["quota"]["ice"] == 4
    assert s["exceeded"]["ice"] == 0
    assert s["total_grid_penalty"] == 0
    assert s["needs_pit_lane_start"] is False
    assert s["n_installs"] == 2


def test_summary_with_penalty():
    inv = PUComponentInventory(driver_id="ver")
    for _ in range(5):
        inv.install_component("ice")
    s = inv.summary()
    assert s["exceeded"]["ice"] == 1
    assert s["total_grid_penalty"] == 5


# --------------------------------------------------------------------------- #
# GridPenaltyCalculator.apply_penalties
# --------------------------------------------------------------------------- #
def test_apply_no_penalties():
    order = ["a", "b", "c", "d"]
    final = apply_grid_penalties(order, {})
    assert final == order


def test_apply_single_penalty():
    order = ["a", "b", "c", "d"]
    final = apply_grid_penalties(order, {"b": 1})
    # b 退后 1 位
    assert final == ["a", "c", "b", "d"]


def test_apply_large_penalty_caps_at_end():
    order = ["a", "b", "c", "d"]
    final = apply_grid_penalties(order, {"a": 10})  # 10 位罚
    # a 退到最后 (位置 3)
    assert final == ["b", "c", "d", "a"]


def test_apply_multiple_penalties():
    order = ["a", "b", "c", "d", "e"]
    final = apply_grid_penalties(order, {"a": 2, "b": 1})
    # a (P1, pen 2) -> target P3 (idx 2); b (P2, pen 1) -> target P3 (idx 2)
    # a 优先占 idx 2, b 冲突后移到 idx 3
    # 非罚位 c, d, e 填充 0, 1, 4
    assert final == ["c", "d", "a", "b", "e"]


def test_apply_pit_lane_start():
    order = ["a", "b", "c", "d", "e"]
    final = apply_grid_penalties(order, {"c": 20}, pit_lane_starts={"c"})
    # c 维修区发车, 放最后
    assert final[-1] == "c"
    assert "c" not in final[:-1]


def test_apply_multiple_pit_lane_starts():
    order = ["a", "b", "c", "d", "e"]
    final = apply_grid_penalties(order,
                                 {"c": 20, "d": 20},
                                 pit_lane_starts={"c", "d"})
    # c, d 维修区发车
    assert final[-2:] == ["c", "d"] or final[-2:] == ["d", "c"]
    assert "a" in final[:3]


def test_apply_preserves_length():
    order = [f"d{i}" for i in range(20)]
    penalties = {f"d{i}": i for i in range(20)}
    final = apply_grid_penalties(order, penalties)
    assert len(final) == 20
    assert set(final) == set(order)


def test_apply_tiebreak_by_original_position():
    """罚位相同时, 原排位靠前的优先占位."""
    order = ["a", "b", "c", "d"]
    final = apply_grid_penalties(order, {"a": 1, "b": 1})
    # a (P1, pen 1) -> target idx 1; b (P2, pen 1) -> target idx 2
    # a 占 idx 1, b 占 idx 2, 非罚位 c, d 填 0, 3
    assert final == ["c", "a", "b", "d"]


# --------------------------------------------------------------------------- #
# GridPenaltyCalculator.summary
# --------------------------------------------------------------------------- #
def test_calculator_summary_structure():
    calc = GridPenaltyCalculator()
    order = ["a", "b", "c", "d"]
    s = calc.summary(order, {"b": 1})
    assert "qualifying_order" in s
    assert "final_grid" in s
    assert "penalties" in s
    assert "pit_lane_starts" in s
    assert "position_changes" in s
    assert s["final_grid"] == ["a", "c", "b", "d"]
    assert s["position_changes"]["b"] == {"from": 2, "to": 3, "change": 1}


def test_calculator_summary_pit_lane():
    calc = GridPenaltyCalculator()
    order = ["a", "b", "c"]
    s = calc.summary(order, {"c": 20}, pit_lane_starts={"c"})
    assert s["position_changes"]["c"]["to"] == 3
    assert "c" in s["pit_lane_starts"]


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_realistic_season_pu_management():
    """模拟赛季中合理 PU 管理: 4 ICE 在 24 场中分配."""
    inv = PUComponentInventory(driver_id="ver")
    # 平均每个 ICE 用 6 场
    ice_changes = [0, 6, 12, 18]  # 4 个 ICE, 不超额
    for r in ice_changes:
        inv.install_component("ice", race_idx=r)
    assert inv.total_grid_penalty() == 0
    assert inv.remaining("ice") == 0


def test_aggressive_pu_usage_penalty():
    """激进使用 (频繁换新) 导致超额罚位."""
    inv = PUComponentInventory(driver_id="lec")
    # 每 3 场换 ICE → 8 个 ICE, 超 4
    for r in range(0, 24, 3):
        inv.install_component("ice", race_idx=r)
    assert inv.exceeded["ice"] == 4
    assert inv.total_grid_penalty() == 20
    assert inv.needs_pit_lane_start()


def test_full_grid_20_drivers_with_penalties():
    """20 车手中部分有罚位的发车位."""
    drivers = [f"d{i:02d}" for i in range(20)]
    # 3 个车手有罚位
    penalties = {"d00": 3, "d05": 5, "d10": 10}
    final = apply_grid_penalties(drivers, penalties)
    assert len(final) == 20
    # d10 退 10 位
    assert final.index("d10") >= 10
    # d00 退 3 位
    assert final.index("d00") >= 3


def test_pit_lane_start_does_not_block_others():
    """维修区发车的车手不占用正常发车位."""
    order = ["a", "b", "c", "d"]
    final = apply_grid_penalties(order, {"b": 20}, pit_lane_starts={"b"})
    # b 维修区发车, a/c/d 占据前 3 位
    assert final[:3] == ["a", "c", "d"]
    assert final[3] == "b"
