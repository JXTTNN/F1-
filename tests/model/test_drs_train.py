"""F1 2026 DRS Train 模型测试 (Iter-41)."""

from __future__ import annotations

from f1opt.model.drs_train import (
    DRSTrainModel,
    detect_drs_train,
    drs_train_lap_penalty,
)


# --------------------------------------------------------------------------- #
# 基础检测
# --------------------------------------------------------------------------- #
def test_no_train_with_few_cars():
    """少于 3 车不形成列车."""
    gaps = [("d1", 0.0), ("d2", 0.5)]
    trains = detect_drs_train(gaps)
    assert trains == []


def test_no_train_with_large_gaps():
    """间隔 > 1.0s 不形成列车."""
    gaps = [("d1", 0.0), ("d2", 2.0), ("d3", 2.5), ("d4", 3.0)]
    trains = detect_drs_train(gaps)
    assert trains == []


def test_detects_3_car_train():
    """3 车在 1.0s 内形成列车."""
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    trains = detect_drs_train(gaps)
    assert len(trains) == 1
    assert trains[0].size == 3


def test_detects_5_car_train():
    """5 车 DRS 列车 (巴库 2018 案例)."""
    gaps = [("d1", 0.0), ("d2", 0.4), ("d3", 0.6), ("d4", 0.8), ("d5", 0.9)]
    trains = detect_drs_train(gaps)
    assert len(trains) == 1
    assert trains[0].size == 5


def test_leader_is_first_car():
    """列车领头是第一位车."""
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    trains = detect_drs_train(gaps)
    assert trains[0].leader_id == "d1"


def test_tail_is_last_car():
    """列车末尾是最后一位车."""
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    trains = detect_drs_train(gaps)
    assert trains[0].tail_id == "d3"


# --------------------------------------------------------------------------- #
# 多列车
# --------------------------------------------------------------------------- #
def test_detects_two_separate_trains():
    """两组列车 (中间有大间隔)."""
    # 这个测试用例语义不清晰, 跳过到 test_two_trains_with_clear_separation
    pass


def test_two_trains_with_clear_separation():
    """两组列车, 中间明确大间隔."""
    gaps = [
        ("d1", 0.0), ("d2", 0.5), ("d3", 0.7),  # 列车 1
        ("d4", 5.0), ("d5", 0.4), ("d6", 0.6),  # d4 大间隔, d5-d6 在 d4 后但 d5 gap=0.4
    ]
    # 语义: d2 与 d1 间隔 0.5, d3 与 d2 间隔 0.7, d4 与 d3 间隔 5.0, d5 与 d4 间隔 0.4, d6 与 d5 间隔 0.6
    # 列车 1: d1-d2-d3 (d2 0.5, d3 0.7 都 ≤ 1.0)
    # d4 gap=5.0 > 1.0, 断开
    # d5 gap=0.4 ≤ 1.0 → d4-d5? 但 d4 gap=5.0 表示 d4 与前车 d3 间隔大
    # 列车 2: d4-d5-d6? d5 gap=0.4 ≤ 1.0, d6 gap=0.6 ≤ 1.0 → d4-d5-d6 列车
    trains = detect_drs_train(gaps)
    assert len(trains) == 2
    assert trains[0].size == 3  # d1-d2-d3
    assert trains[1].size == 3  # d4-d5-d6


# --------------------------------------------------------------------------- #
# 圈速损失
# --------------------------------------------------------------------------- #
def test_leader_penalty_smallest():
    """列车领头损失最少."""
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7), ("d4", 0.9)]
    trains = detect_drs_train(gaps)
    penalties = [c.lap_penalty_s for c in trains[0].cars]
    # 领头 (d1) 损失最少, 末尾 (d4) 损失最多
    assert penalties[0] < penalties[-1]
    assert penalties == sorted(penalties)  # 递增


def test_penalty_increases_with_position():
    """列车内位置越后, 损失越大."""
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7), ("d4", 0.9), ("d5", 0.8)]
    trains = detect_drs_train(gaps)
    penalties = [c.lap_penalty_s for c in trains[0].cars]
    for i in range(len(penalties) - 1):
        assert penalties[i] <= penalties[i + 1]


def test_penalty_capped_at_max():
    """列车内单圈损失有上限."""
    model = DRSTrainModel(penalty_max_s=0.30)
    # 10 车列车
    gaps = [("d1", 0.0)] + [(f"d{i}", 0.5) for i in range(2, 11)]
    trains = model.detect_trains(gaps)
    for car in trains[0].cars:
        assert car.lap_penalty_s <= 0.30


def test_penalty_for_specific_driver():
    """查询特定车手的列车损失."""
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    trains = detect_drs_train(gaps)
    # d2 在列车内
    assert trains[0].penalty_for("d2") > 0
    # d4 不在列车内
    assert trains[0].penalty_for("d4") == 0.0


def test_convenience_penalty_function():
    """drs_train_lap_penalty 便捷函数."""
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    # d2 在列车内
    assert drs_train_lap_penalty(gaps, "d2") > 0
    # d4 不在
    assert drs_train_lap_penalty(gaps, "d4") == 0.0


# --------------------------------------------------------------------------- #
# 状态追踪
# --------------------------------------------------------------------------- #
def test_get_active_train_for_driver():
    """查询车手当前所在列车."""
    model = DRSTrainModel(track_id="baku")
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    model.detect_trains(gaps, current_lap=10)
    train = model.get_active_train_for("d2")
    assert train is not None
    assert train.size == 3
    # d4 不在
    assert model.get_active_train_for("d4") is None


def test_penalty_for_driver_method():
    """model.penalty_for_driver 返回当前损失."""
    model = DRSTrainModel()
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    model.detect_trains(gaps)
    assert model.penalty_for_driver("d1") > 0
    assert model.penalty_for_driver("d2") > 0
    assert model.penalty_for_driver("d4") == 0.0


def test_summary_returns_dict():
    model = DRSTrainModel(track_id="baku")
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    model.detect_trains(gaps, current_lap=10)
    s = model.summary()
    assert s["n_active_trains"] == 1
    assert s["trains"][0]["size"] == 3
    assert s["trains"][0]["leader"] == "d1"


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #
def test_empty_gaps_returns_empty():
    assert detect_drs_train([]) == []


def test_single_car_no_train():
    assert detect_drs_train([("d1", 0.0)]) == []


def test_two_cars_close_no_train():
    """2 车接近但 < 3 车不形成列车."""
    gaps = [("d1", 0.0), ("d2", 0.5)]
    assert detect_drs_train(gaps) == []


def test_threshold_boundary():
    """间隔 = 1.0s (阈值边界) 应包含在列车内."""
    gaps = [("d1", 0.0), ("d2", 1.0), ("d3", 1.0)]
    trains = detect_drs_train(gaps)
    assert len(trains) == 1
    assert trains[0].size == 3


def test_just_over_threshold_no_train():
    """间隔 = 1.1s (刚超阈值) 不在列车内."""
    gaps = [("d1", 0.0), ("d2", 1.1), ("d3", 0.5)]
    # d2 gap=1.1 > 1.0, 断开
    # d3 gap=0.5 ≤ 1.0, 但 d2-d3 需要 d2 也在列车内
    # 实际: d2 gap=1.1 > 1.0, 所以 d2 不加入 d1 的列车
    # d3 单独检查: 从 i=2, gaps[2]=(d3, 0.5), i-1=1 (d2). d3-d2 间隔 0.5 ≤ 1.0
    # 但 d2 没在前一列车内... 检测逻辑: 从 i=2 开始, leader=gaps[1]=d2
    # d2 gap=1.1 > 1.0, 所以 d2 不会被加入新列车 (因为它是新领头, gap 是它与 d1 的间隔)
    # 实际只有 d2-d3 两车, 不足 3 车
    trains = detect_drs_train(gaps)
    assert trains == []


# --------------------------------------------------------------------------- #
# formed_lap 追踪
# --------------------------------------------------------------------------- #
def test_formed_lap_recorded():
    model = DRSTrainModel(track_id="baku")
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    trains = model.detect_trains(gaps, current_lap=15)
    assert trains[0].formed_lap == 15


def test_track_id_recorded():
    model = DRSTrainModel(track_id="spa")
    gaps = [("d1", 0.0), ("d2", 0.5), ("d3", 0.7)]
    trains = model.detect_trains(gaps)
    assert trains[0].track_id == "spa"
