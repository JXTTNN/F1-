"""车手风格定向（第二层）验收测试。

覆盖三件事，每一件都对应一个真实失效：

1. **风格规则本身**：中性/缺失/长度不足 → 不产生动作；极端开法 → 动作方向正确；
   同一参数被多条规则命中时只认领一次（不能叠成 2 档）。
2. **风格真的改变了最终建议**：不只改中间量 —— 实测旧实现 6 条规则里 4 条在
   24 站上对最终 delta 零影响（乘 0 还是 0 / 被 max_delta 夹平）。
3. **走完整使用流程**：3 圈遥测 → 风格向量落库 → 车手反馈 → ``/api/v1/suggest``，
   报告里 ``style_targeting`` 真的生效；不足 3 圈时如实不启用。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from setup_tuner.app import _init_app_services, _make_packet_handler, create_app
from setup_tuner.config import Config
from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.style_coefficients import (
    _DIM_LABELS,
    STYLE_VECTOR_LEN,
    describe_style,
    gain_for_style,
    style_trims,
)
from setup_tuner.engine.engine import generate_suggestion
from setup_tuner.telemetry.style_extractor import STYLE_DIMS

NEUTRAL = [0.5] * STYLE_VECTOR_LEN
#: 粗糙型：转向修正多、胎温离散大、循迹刹车多
ROUGH = [0.3, 0.2, 0.3, 0.3, 0.8, 0.3, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5]
#: 激进型：攻弯凶、给油早、滑移多
AGGRESSIVE = [0.9, 0.8, 0.9, 0.8, 0.3, 0.9, 0.7, 0.9, 0.5, 0.5, 0.5, 0.5]
_SYMPTOMS = [("understeer", 3), ("turnin_unresponsive", 2)]


# ===========================================================================
# 1) 规则层
# ===========================================================================
class TestStyleTrims:
    def test_dim_labels_match_extractor_order(self) -> None:
        """报告用的中文标签必须与风格向量维度同序（复制一份 ≠ 可以不锁）。"""
        assert [name for name, _ in _DIM_LABELS] == STYLE_DIMS
        assert len(_DIM_LABELS) == STYLE_VECTOR_LEN

    def test_neutral_and_missing_vectors_produce_nothing(self) -> None:
        assert style_trims(NEUTRAL) == []
        assert style_trims(None) == []
        assert style_trims([]) == []
        assert style_trims([0.5] * 5) == []
        assert style_trims([float("nan")] * STYLE_VECTOR_LEN) == []
        # 中性向量必须逐位等于 1.0（否则"没样本"会被当成"有倾向"）
        assert set(gain_for_style(NEUTRAL).values()) == {1.0}

    def test_aggressive_gets_front_wing_and_throttle_actions(self) -> None:
        trims = style_trims(AGGRESSIVE)
        moves = {p: s for t in trims for p, s in t.moves}
        assert moves["front_wing"] == 1.0
        assert moves["on_throttle_diff"] == 1.0
        assert moves["rear_wing"] == 1.0

    def test_rough_gets_brake_and_pressure_actions(self) -> None:
        trims = style_trims(ROUGH)
        moves = {p: s for t in trims for p, s in t.moves}
        assert moves["brake_pressure"] == -1.0
        # 胎压必须四轮成组、且是"按档位"（-1 档 = -0.1 bar）
        pressure = [p for p in moves if "tyre_pressure" in p]
        assert len(pressure) == 4
        assert {moves[p] for p in pressure} == {-1.0}
        groups = [t for t in trims if "front_left_tyre_pressure" in t.params]
        assert len(groups) == 1 and len(groups[0].params) == 4

    def test_same_param_never_claimed_twice(self) -> None:
        """给油早 + 滑移多都指向锁止差速 → 只能动 1 档，不能叠成 2 档。"""
        trims = style_trims(AGGRESSIVE)
        claims = [p for t in trims for p in t.params]
        assert len(claims) == len(set(claims))
        on_throttle = [p for p in claims if p == "on_throttle_diff"]
        assert len(on_throttle) == 1

    def test_describe_style_is_sorted_by_strength(self) -> None:
        profile = describe_style(AGGRESSIVE)
        assert profile
        assert profile[0]["strength"] >= profile[-1]["strength"]
        # 中性不产生倾向描述（否则报告会给所有人贴标签）
        assert describe_style(NEUTRAL) == []


# ===========================================================================
# 2) 引擎：风格必须改变**最终**建议且不破契约
# ===========================================================================
@pytest.mark.parametrize("track_id", ["melbourne", "monza", "monaco"])
@pytest.mark.parametrize("style", [ROUGH, AGGRESSIVE], ids=["rough", "aggressive"])
def test_style_changes_final_delta(track_id: str, style: list[float]) -> None:
    base = CarSetup.default().to_dict()
    neutral = generate_suggestion(_SYMPTOMS, base, track_id, model_type="nn",
                                  style_vector=NEUTRAL)
    styled = generate_suggestion(_SYMPTOMS, base, track_id, model_type="nn",
                                 style_vector=style)
    assert styled["setup_delta"] != neutral["setup_delta"], (
        "换了开法，最终建议必须不同（旧实现有 4/6 条规则零影响）"
    )
    block = styled["holistic"]["style_targeting"]
    assert block["enabled"] is True
    assert block["survived"] >= 1, "至少要有一处活到最终建议里"
    assert block["cost_ms"] is not None
    assert abs(float(block["cost_ms"])) <= block["budget_ms"] + 1e-6


@pytest.mark.parametrize("style", [ROUGH, AGGRESSIVE], ids=["rough", "aggressive"])
def test_style_keeps_usability_contract(style: list[float]) -> None:
    """风格动作不能越界、不能超 max_delta、不能把胎压弄成左右不等。"""
    base = CarSetup.default().to_dict()
    for track in ("melbourne", "spa", "monaco"):
        r = generate_suggestion(_SYMPTOMS, base, track, model_type="nn",
                                style_vector=style)
        delta = r["setup_delta"]
        for spec in ALL_SETUP_FIELDS:
            d = float(delta[spec.name])
            final = base[spec.name] + d
            assert abs(d) <= spec.max_delta + 1e-6, f"{track}/{spec.name} 超 max_delta"
            assert spec.min_val - 1e-6 <= final <= spec.max_val + 1e-6
            steps = (final - spec.min_val) / spec.step
            assert abs(steps - round(steps)) <= 1e-6, (
                f"{track}/{spec.name} 终值 {final:g} 不在 {spec.step:g} 档位上"
            )
        for left, right in (
            ("front_left_tyre_pressure", "front_right_tyre_pressure"),
            ("rear_left_tyre_pressure", "rear_right_tyre_pressure"),
        ):
            assert abs(float(delta[left]) - float(delta[right])) <= 1e-9


def test_no_diagnosis_means_no_style_moves() -> None:
    """无症状/无遥测 → 不得因为"车手风格"改车。"""
    base = CarSetup.default().to_dict()
    for style in (ROUGH, AGGRESSIVE):
        r = generate_suggestion([], base, "melbourne", model_type="nn",
                                style_vector=style)
        assert all(abs(float(v)) < 1e-9 for v in r["setup_delta"].values())
        block = r["holistic"]["style_targeting"]
        assert block["enabled"] is False


def test_rule_only_model_does_not_apply_style_trims() -> None:
    """model_type=rule 不该偷偷用调教性能模型评估/采纳风格动作。"""
    base = CarSetup.default().to_dict()
    r = generate_suggestion(_SYMPTOMS, base, "melbourne", model_type="rule",
                            style_vector=AGGRESSIVE)
    block = r["holistic"]["style_targeting"]
    assert block["enabled"] is False
    assert "模型" in block["reason"] or "未覆盖" in block["reason"]


# ===========================================================================
# 3) 使用流程端到端：遥测圈 → 风格 → 反馈 → /suggest
# ===========================================================================
def _frame(i: int) -> dict:
    """一帧遥测（Packet 6），带一点车手开法特征（转向/油门有变化）。"""
    return {
        "packet_id": 6,
        "m_speed": 150 + (i % 180),
        "m_throttle": 0.9 if i % 5 else 0.1,
        "m_brake": 0.7 if i % 7 == 0 else 0.0,
        "m_steer": 0.35 * ((i % 40) - 20) / 20.0,
        "m_tyresSurfaceTemperature": [98, 104, 92, 96],
        "m_brakesTemperature": [430, 445, 400, 415],
    }


def _run_laps(app, handler, laps: int, lap_time_ms: int = 90000) -> None:
    """按真实链路跑 laps 圈（3000+ 帧/圈 → 触发整圈落库与风格 EMA）。"""
    for lap in range(1, laps + 1):
        handler({"packet_id": 2, "m_currentLapNum": lap, "m_sector": 0})
        for i in range(3000):
            handler(_frame(i + lap))
        handler({"packet_id": 2, "m_currentLapNum": lap + 1, "m_sector": 0,
                 "m_lastLapTimeInMS": lap_time_ms})


def _wait(predicate, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _boot(tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path / "data"), udp_port=0)
    app = create_app(cfg)
    _init_app_services(app, cfg)
    app.state.current_track_id = "melbourne"
    return app, _make_packet_handler(app)


def test_usage_flow_applies_driver_style(tmp_path: Path) -> None:
    """3 圈遥测 → 风格 ≥3 样本 → 反馈 → /suggest：报告如实标出风格定向。"""
    app, handler = _boot(tmp_path)
    _run_laps(app, handler, 3)
    client = TestClient(app)

    store = app.state.store
    assert _wait(lambda: (store.get_driver_style(1, "melbourne") or {}).get(
        "sample_count", 0) >= 3), "3 圈后风格样本应达 3"

    # 用车手真实产生的向量跑一次（不伪造向量：这条测的是接线，不是规则）
    vec = list(store.get_driver_style(1, "melbourne")["vector"])
    assert len(vec) == STYLE_VECTOR_LEN

    r = client.post("/api/v1/feedback", json={
        "track_id": "melbourne",
        "feedbacks": [{"corner_number": 1, "symptom": "understeer", "strength": 3}],
    })
    assert r.status_code == 200, r.text

    resp = client.post("/api/v1/suggest", json={
        "track_id": "melbourne", "model_type": "nn",
    })
    assert resp.status_code == 200, resp.text
    report = resp.json()["data"]["report"]
    block = report["holistic"]["style_targeting"]
    # 由真实遥测产生的向量可能接近中性 → 允许"中性、无需微调"，但必须
    # 明确给出结论，不能静默消失（旧实现直接不写这个字段）。
    assert "enabled" in block
    assert block["reason"]
    assert report["holistic"]["simulation"]["available"] is True


def test_usage_flow_without_enough_laps_skips_style(tmp_path: Path) -> None:
    """没跑够 3 圈时上游传 None：风格定向必须如实"未启用"。"""
    app, handler = _boot(tmp_path)
    _run_laps(app, handler, 1)
    client = TestClient(app)
    store = app.state.store
    assert _wait(lambda: (store.get_driver_style(1, "melbourne") or {}).get(
        "sample_count", 0) == 1)

    client.post("/api/v1/feedback", json={
        "track_id": "melbourne",
        "feedbacks": [{"corner_number": 1, "symptom": "understeer", "strength": 3}],
    })
    resp = client.post("/api/v1/suggest", json={
        "track_id": "melbourne", "model_type": "nn",
    })
    assert resp.status_code == 200
    block = resp.json()["data"]["report"]["holistic"]["style_targeting"]
    assert block["enabled"] is False
    assert "3" in block["reason"] or "样本" in block["reason"]


def test_style_vector_stored_per_track_not_global(tmp_path: Path) -> None:
    """风格是「这条赛道上的开法」：别的赛道没跑过 → 不串用。"""
    app, handler = _boot(tmp_path)
    _run_laps(app, handler, 3)
    store = app.state.store
    assert _wait(lambda: (store.get_driver_style(1, "melbourne") or {}).get(
        "sample_count", 0) >= 3)
    assert store.get_driver_style(1, "monza") is None
