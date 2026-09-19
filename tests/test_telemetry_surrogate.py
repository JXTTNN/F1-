"""遥测代理模型 / 自动诊断 / 整体性收口 的测试。

覆盖本轮新增的四件事：
1. 纯标准库 MLP 真的能训练（不是空壳）；
2. 代理模型在**权重文件缺失时中性降级**（不得抛错、不得影响主流程）；
3. 遥测能自动发现车手未反馈的问题，且不重复计入 Dx；
4. 机械抓地参与度收口：牵引型赛道上悬挂/几何不得被优化器清零，
   且收口是**预算中性**的（总改动幅度不增长）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from setup_tuner.domain.setup import CarSetup
from setup_tuner.engine.holistic import (
    holistic_coherence,
    track_demand,
)
from setup_tuner.engine.pure_nn import MLP, r2_score
from setup_tuner.engine.surrogate import SurrogateModel
from setup_tuner.engine.telemetry_diagnosis import (
    ENGINE_OWNED_SOURCES,
    diagnose_from_telemetry,
    implicit_for_engine,
    merge_with_driver_feedbacks,
)


# --------------------------------------------------------------------------- #
# 1. 纯标准库 MLP
# --------------------------------------------------------------------------- #
class TestPureNN:
    def test_learns_linear_function(self) -> None:
        """线性关系必须被学到（R² > 0.9）—— 否则"能训练"是假的。"""
        x = [[i * 0.01, (i % 7) * 0.1, 1.0 - i * 0.005] for i in range(200)]
        y = [[r[0] * 2.0 - r[1] + 0.5] for r in x]
        model = MLP([3, 8, 8, 1], seed=7)
        model.fit(x, y, epochs=200, lr=0.02, batch_size=32)
        pred = [v[0] for v in model.predict(x)]
        truth = [v[0] for v in y]
        assert r2_score(truth, pred) > 0.9

    def test_untrained_model_refuses_to_predict(self) -> None:
        """未训练就推理必须报错，而不是静默返回垃圾。"""
        model = MLP([2, 4, 1])
        with pytest.raises(RuntimeError):
            model.predict([[1.0, 2.0]])

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        x = [[float(i), float(i % 5)] for i in range(60)]
        y = [[r[0] * 0.5] for r in x]
        model = MLP([2, 6, 1], seed=1)
        model.fit(x, y, epochs=80, lr=0.02)
        path = tmp_path / "m.json"
        model.save(path)
        loaded = MLP.load(path)
        assert loaded.predict([[3.0, 3.0]])[0][0] == pytest.approx(
            model.predict([[3.0, 3.0]])[0][0], rel=1e-9,
        )

    def test_dimension_mismatch_is_rejected(self) -> None:
        model = MLP([3, 4, 1])
        with pytest.raises(ValueError):
            model.fit([[1.0, 2.0]], [[1.0]], epochs=1)


# --------------------------------------------------------------------------- #
# 2. 代理模型中性地降级
# --------------------------------------------------------------------------- #
class TestSurrogateDegradation:
    def test_missing_file_is_neutral(self, tmp_path: Path) -> None:
        model = SurrogateModel(tmp_path / "nope.json")
        assert model.available is False
        assert model.corner_importance("monza") is None
        assert model.expected_corner_s("monza") is None
        assert model.predict_corner_share("monza", 1, {}) is None
        assert model.lap_head_usable is False
        assert "不可用" in model.describe()

    def test_corrupt_file_is_neutral(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        model = SurrogateModel(bad)
        assert model.available is False

    def test_loads_importance_and_answers(self, tmp_path: Path) -> None:
        payload = {
            "corner_head": {
                "pair_keys": ["monza#1", "monza#2"],
                "feature_keys": ["pair=monza#1", "pair=monza#2", "cls_slow"],
                "corner_importance": {"monza": {"1": 0.4, "2": 0.6}},
                "expected_corner_time_s": {"monza": {"1": 10.0, "2": 15.0}},
                "n_samples": 123,
                "active": {"kind": "ridge", "metrics": {"val_mae": 0.1, "val_r2": 0.8}},
                "family": {"ridge": {"model": {
                    "kind": "ridge", "alpha": 1.0, "intercept": 1.0,
                    "coef": [1.0, 1.0, 0.0], "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                }}},
            },
            "lap_head": {"available": False},
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = SurrogateModel(path)
        assert model.available is True
        assert model.corner_importance("monza") == {1: 0.4, 2: 0.6}
        assert model.expected_corner_s("monza") == {1: 10.0, 2: 15.0}
        # 已知弯：one-hot 命中 → 预测 2.0；未知弯：整行留零 → 预测 1.0（不抛错）
        assert model.predict_corner_share("monza", 1, {}) == pytest.approx(2.0)
        assert model.predict_corner_share("__unknown__", 9, {}) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. 遥测自动诊断
# --------------------------------------------------------------------------- #
def _kerb_lap_agg(**overrides) -> dict:
    base = {
        "kerb_corners": [
            {"corner": 5, "ratio": 10.08, "frames": 299, "side": "left"},
            {"corner": 2, "ratio": 1.7, "frames": 50, "side": "right"},
        ],
        "plank_bottoming": True,
        "plank_bottoming_ratio": 0.08,
        "m_tyresSurfaceTemperature": [88.0, 89.0, 96.0, 97.0],
        "m_brakesTemperature": [400.0, 410.0, 430.0, 440.0],
    }
    base.update(overrides)
    return base


class TestTelemetryDiagnosis:
    def test_detects_kerb_with_corner_attribution(self) -> None:
        items = diagnose_from_telemetry(_kerb_lap_agg(), "monza")
        kerbs = [i for i in items if i["symptom"] == "kerb_instability"]
        assert {i["corner_number"] for i in kerbs} == {5, 2}
        strong = next(i for i in kerbs if i["corner_number"] == 5)
        assert strong["strength"] == 3
        mild = next(i for i in kerbs if i["corner_number"] == 2)
        assert mild["strength"] == 2
        assert all(i["implicit"] is True for i in items)

    def test_detects_bottoming(self) -> None:
        items = diagnose_from_telemetry(_kerb_lap_agg(), "monza")
        assert any(i["symptom"] == "bottoming" for i in items)

    def test_detects_tyre_and_brake_issues(self) -> None:
        agg = _kerb_lap_agg(
            m_tyresSurfaceTemperature=[80.0, 80.0, 110.0, 112.0],
            m_brakesTemperature=[700.0, 710.0, 720.0, 730.0],
        )
        symptoms = {i["symptom"] for i in diagnose_from_telemetry(agg, "monza")}
        assert "midcorner_understeer" in symptoms   # 前胎更热
        assert "brake_fade" in symptoms             # 刹车过热

    def test_no_telemetry_is_neutral(self) -> None:
        assert diagnose_from_telemetry(None, "monza") == []
        assert diagnose_from_telemetry({}, "monza") == []

    def test_engine_owned_subset_only(self) -> None:
        """只有接管来源才进反馈路径，避免与聚合规则重复计入 Dx。"""
        items = diagnose_from_telemetry(_kerb_lap_agg(), "monza")
        owned = implicit_for_engine(items)
        assert owned, "路肩条目应进入反馈路径"
        assert all(i["source"] in ENGINE_OWNED_SOURCES for i in owned)
        assert all(i["corner_number"] is not None for i in owned)

    def test_driver_feedback_wins(self) -> None:
        driver = [{"corner_number": 5, "symptom": "kerb_instability",
                   "strength": 1, "category": "global"}]
        items = diagnose_from_telemetry(_kerb_lap_agg(), "monza")
        merged = merge_with_driver_feedbacks(driver, implicit_for_engine(items))
        same = [m for m in merged
                if m["corner_number"] == 5 and m["symptom"] == "kerb_instability"]
        assert len(same) == 1
        assert same[0]["strength"] == 1, "车手反馈优先，不应被隐式条目覆盖"

    def test_surrogate_residual_requires_corner_times(self) -> None:
        """没有逐弯用时就没有模型证据（不得凭空造问题）。"""
        items = diagnose_from_telemetry(_kerb_lap_agg(), "monza", corner_times=None)
        assert not any(i["source"].startswith("surrogate") for i in items)


# --------------------------------------------------------------------------- #
# 4. 机械抓地参与度收口
# --------------------------------------------------------------------------- #
_MECH_KEYS = (
    "front_camber", "rear_camber", "front_toe", "rear_toe",
    "front_suspension", "rear_suspension",
    "front_anti_roll_bar", "rear_anti_roll_bar",
)


class TestMechanicalParticipation:
    def test_restores_mechanical_grip_on_traction_track(self) -> None:
        demand = track_demand("monza")
        assert demand.traction_index >= 0.35, "前置条件：monza 属牵引型赛道"
        dx = {"front_grip_req": 0.9, "exit_traction_req": 0.6}
        optimized = {"front_wing": 2.0, "rear_wing": -1.0}   # 无任何机械手段
        fallback = {"front_toe": 0.05, "rear_anti_roll_bar": 1.0}
        closed, notes = holistic_coherence(
            optimized, dx, demand, mechanical_fallback=fallback,
        )
        assert any(closed.get(k) for k in _MECH_KEYS), "机械抓地手段必须参与"
        assert any("机械抓地参与度" in n for n in notes)

    def test_restoration_is_budget_neutral(self) -> None:
        """恢复机械手段不得抬高总改动幅度（湿地保守性依赖这条不变量）。"""
        demand = track_demand("monza")
        dx = {"front_grip_req": 0.9}
        optimized = {"front_wing": 3.0, "rear_wing": 3.0, "on_throttle_diff": 4.0}
        before = sum(abs(v) for v in optimized.values())
        fallback = {"front_suspension": 2.0, "rear_suspension": 2.0}
        closed, _ = holistic_coherence(
            optimized, dx, demand, mechanical_fallback=fallback,
        )
        assert sum(abs(v) for v in closed.values()) <= before + 1e-9

    def test_no_restoration_without_demand(self) -> None:
        """没有机械抓地需求时不得凭空塞入改动。"""
        demand = track_demand("monza")
        optimized = {"front_wing": 2.0}
        closed, _ = holistic_coherence(
            optimized, {}, demand,
            mechanical_fallback={"front_toe": 0.05, "rear_anti_roll_bar": 1.0},
        )
        assert not any(closed.get(k) for k in _MECH_KEYS)

    def test_untouched_when_mechanical_already_present(self) -> None:
        demand = track_demand("monza")
        dx = {"front_grip_req": 0.9}
        optimized = {"front_anti_roll_bar": -1.0}
        closed, notes = holistic_coherence(
            optimized, dx, demand, mechanical_fallback={"front_toe": 0.05},
        )
        assert closed["front_anti_roll_bar"] == -1.0
        assert closed.get("front_toe", 0.0) == 0.0, "已有机械手段时不应再补"
        assert not any("机械抓地参与度收口" in n for n in notes)


# --------------------------------------------------------------------------- #
# 5. 引擎端到端：隐式反馈进入报告
# --------------------------------------------------------------------------- #
class TestEngineImplicitFeedbacks:
    def test_report_contains_implicit_feedbacks(self) -> None:
        from setup_tuner.engine.engine import generate_suggestion

        result = generate_suggestion(
            symptoms=[("understeer", 2)],
            current_setup=CarSetup.default().to_dict(),
            track_id="monza",
            telemetry=_kerb_lap_agg(),
            model_type="rule",
        )
        implicit = result["holistic"]["implicit_feedbacks"]
        assert implicit, "遥测里明明有路肩/刮底信号，报告必须体现"
        assert all({"corner", "symptom", "strength", "source", "evidence"} <= set(i)
                   for i in implicit)
        assert "surrogate" in result["holistic"]

    def test_kerb_not_double_counted(self) -> None:
        """接管路肩后，聚合规则不得再叠加一次 ride_height_req。"""
        from setup_tuner.engine.engine import _derive_telemetry_dx

        raw = _derive_telemetry_dx(_kerb_lap_agg())
        flagged = dict(_kerb_lap_agg())
        flagged["_implicit_kerb"] = True
        handled = _derive_telemetry_dx(flagged)
        assert raw["ride_height_req"] > handled["ride_height_req"]

    def test_no_telemetry_still_works(self) -> None:
        from setup_tuner.engine.engine import generate_suggestion

        result = generate_suggestion(
            symptoms=[("understeer", 2)],
            current_setup=CarSetup.default().to_dict(),
            track_id="monza",
            telemetry=None,
            model_type="rule",
        )
        assert result["holistic"]["implicit_feedbacks"] == []


# --------------------------------------------------------------------------- #
# 6. 逐弯重要度：模型优先、启发式兜底
# --------------------------------------------------------------------------- #
class TestCornerImportanceSource:
    def test_falls_back_to_heuristic_without_model(self) -> None:
        """模型不可用时，重要度必须回退为 120/参考速度 的归一化启发式。"""
        from setup_tuner.domain.track import get_track_by_id
        from setup_tuner.engine import surrogate as surrogate_mod
        from setup_tuner.engine.lap_model import _TIME_REF_SPEED

        saved = surrogate_mod.get_surrogate().importance
        surrogate_mod.get_surrogate().importance = {}
        try:
            evals = {e.number: e for e in _evals("suzuka")}
            track = get_track_by_id("suzuka")
            raw = {
                c.number: _TIME_REF_SPEED / max(30.0, float(c.speed_kmh))
                for c in track.corners
            }
            total = sum(raw.values())
            for num, e in evals.items():
                # 代码内会 round(...,6)，用绝对容差而不是相对容差
                assert e.importance == pytest.approx(raw[num] / total, abs=1e-6)
        finally:
            surrogate_mod.get_surrogate().importance = saved

    def test_uses_model_weights_when_available(self) -> None:
        from setup_tuner.engine import surrogate as surrogate_mod
        from setup_tuner.engine.lap_model import _TIME_REF_SPEED

        model = surrogate_mod.get_surrogate()
        if not model.available:
            pytest.skip("代理模型权重不存在（CI 环境正常）")
        evals = {e.number: e for e in _evals("monza")}
        # 至少有一个弯的权重与启发式显著不同，证明模型确实接管了权重
        from setup_tuner.domain.track import get_track_by_id

        different = 0
        for c in get_track_by_id("monza").corners:
            raw = _TIME_REF_SPEED / max(30.0, float(c.speed_kmh))
            if abs(evals[c.number].importance - raw) > 0.01:
                different += 1
        assert different > 0


def _evals(track_id: str):
    from setup_tuner.engine.lap_model import corner_evaluations

    return corner_evaluations(track_id)


# --------------------------------------------------------------------------- #
# 7. 引擎规则没覆盖的盲区信号（车损 / 胎耗 / ABS / 牵引介入）
# --------------------------------------------------------------------------- #
class TestBlindSpotSignals:
    def test_damage_is_detected(self) -> None:
        """车翼/底板损伤：引擎里没有任何聚合规则，必须由本层接管。"""
        agg = {
            "wing_damage_max": 55,
            "floor_damage": 45,
            "damage_severe": True,
        }
        items = diagnose_from_telemetry(agg, "monza")
        symptoms = {i["symptom"] for i in items}
        assert "understeer" in symptoms            # 前翼损伤 → 前轴失压
        assert "high_speed_instability" in symptoms  # 底板/严重损伤 → 失稳
        assert all(
            i["source"] == "telemetry:damage"
            for i in items if i["symptom"] in {"understeer", "high_speed_instability"}
            and "损伤" in i["evidence"]
        )
        assert implicit_for_engine(items), "车损必须进入反馈路径"

    def test_blind_spot_sources_are_engine_owned(self) -> None:
        """引擎无规则的信号必须全部进入反馈路径（否则问题永远进不了优化）。"""
        # 注意：同一 (弯, 症状) 只保留一条来源（避免重复计入 Dx），
        # 因此这里逐个信号单独验证，而不是一次全给。
        wear = implicit_for_engine(diagnose_from_telemetry(
            {"tyres_wear": [30.0, 30.0, 12.0, 12.0]}, "monza",
        ))
        assert "telemetry:tyre_wear" in {i["source"] for i in wear}
        abs_items = implicit_for_engine(diagnose_from_telemetry(
            {"anti_lock_brakes": 1}, "monza",
        ))
        assert "telemetry:abs" in {i["source"] for i in abs_items}
        tc_items = implicit_for_engine(diagnose_from_telemetry(
            {"traction_control": 2}, "monza",
        ))
        assert "telemetry:tc" in {i["source"] for i in tc_items}

    def test_abs_detected_without_brake_temps(self) -> None:
        """ABS 判定与刹车温度无关 —— 缺温度时也必须能检测到（实测回归）。"""
        items = diagnose_from_telemetry({"anti_lock_brakes": 1}, "monza")
        assert any(i["symptom"] == "lockup" for i in items)

    def test_covered_sources_stay_out_of_feedback_path(self) -> None:
        """已有聚合规则覆盖的信号不得再注入反馈路径（防重复计入 Dx）。"""
        agg = {
            "m_tyresSurfaceTemperature": [88.0, 89.0, 110.0, 112.0],
            "m_brakesTemperature": [700.0, 710.0, 720.0, 730.0],
            "plank_bottoming": True,
        }
        sources = {i["source"] for i in implicit_for_engine(
            diagnose_from_telemetry(agg, "monza"),
        )}
        assert sources == set(), f"这些信号已由聚合规则覆盖：{sources}"


# --------------------------------------------------------------------------- #
# 8. 「问题 → 优化方案」映射
# --------------------------------------------------------------------------- #
class TestImplicitPlan:
    def test_maps_finding_to_changed_params(self) -> None:
        from setup_tuner.engine.telemetry_diagnosis import link_findings_to_changes

        findings = [{
            "symptom": "bottoming", "corner_number": None, "strength": 3,
            "source": "telemetry:plank", "evidence": "底板触地", "category": "global",
        }]
        delta = {"front_ride_height": 2.0, "rear_ride_height": 2.0, "front_wing": 0.0}
        plan = link_findings_to_changes(findings, delta)
        assert len(plan) == 1
        assert plan[0]["resolved"] is True
        params = [c["param"] for c in plan[0]["changes"]]
        assert "front_ride_height" in params
        assert "front_wing" not in params, "零改动不应出现在方案里"

    def test_unresolved_is_reported_honestly(self) -> None:
        from setup_tuner.engine.telemetry_diagnosis import link_findings_to_changes

        findings = [{
            "symptom": "bottoming", "corner_number": None, "strength": 3,
            "source": "telemetry:plank", "evidence": "底板触地", "category": "global",
        }]
        plan = link_findings_to_changes(findings, {})
        assert plan[0]["resolved"] is False
        assert plan[0]["changes"] == []


# --------------------------------------------------------------------------- #
# 9. 优化器质量：精修只会更好 + 需求满足度指标
# --------------------------------------------------------------------------- #
class TestOptimizerQuality:
    @staticmethod
    def _result(track: str):
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import compute_setup_delta
        from setup_tuner.engine.holistic import class_weighted_dx
        from setup_tuner.engine.optimizer import optimize_setup

        feedbacks = [{"corner_number": 1, "symptom": "understeer",
                      "strength": 3, "category": "entry"}]
        cdx = class_weighted_dx(feedbacks, track)
        setup = CarSetup.default().to_dict()
        initial = compute_setup_delta(cdx.dx, setup)
        return optimize_setup(
            cdx.dx, setup, track, feedbacks=feedbacks,
            initial_delta=initial, needs_by_class=cdx.by_class,
        )

    def test_satisfaction_is_a_ratio(self) -> None:
        res = self._result("monza")
        for table in (res.satisfaction_before, res.satisfaction_after):
            assert set(table) == {"slow", "medium", "fast"}
            assert all(0.0 <= v <= 1.0 for v in table.values())

    def test_fine_stage_never_worsens_objective(self) -> None:
        """二级精修只接受更优解（关掉它也只能得到更差或相同的目标值）。"""
        import setup_tuner.engine.optimizer as opt

        with_fine = self._result("suzuka")
        saved = opt._GRID_FINE
        opt._GRID_FINE = opt._GRID      # 精修退化为粗搜 → 不产生改善
        try:
            without_fine = self._result("suzuka")
        finally:
            opt._GRID_FINE = saved
        assert with_fine.after.total <= without_fine.after.total + 1e-12

    def test_same_input_same_output(self) -> None:
        """优化器必须保持确定性（同输入必得同输出）。"""
        a, b = self._result("spa"), self._result("spa")
        assert a.delta == b.delta


# --------------------------------------------------------------------------- #
# 10. 真实 API 链路：车手零反馈时也必须能出方案
# --------------------------------------------------------------------------- #
def _synthetic_monza_lap(handler, lap_no: int) -> None:
    """通过 app 的真实包处理链喂进一圈合成遥测（含路肩/刮底/车损/ABS/TC）。"""
    handler({"packet_id": 1, "m_trackId": 11, "m_trackLength": 5793.0,
             "m_sessionUID": 20260918})
    steps = 480
    for i in range(steps):
        frac = i / steps
        dist = frac * 5793.0
        is_kerb = 0.045 <= frac < 0.075 or 0.33 <= frac < 0.36
        rough = 6.0 if is_kerb else 0.4
        handler({"packet_id": 2, "m_currentLapNum": lap_no, "m_lapDistance": dist,
                 "m_sector": i * 3 // steps, "m_lastLapTimeInMS": 95000})
        handler({"packet_id": 6, "m_speed": 120.0, "m_throttle": 0.6,
                 "m_brake": 0.2, "m_steer": 0.30 if is_kerb else 0.10,
                 "m_tyresSurfaceTemperature": [88, 89, 106, 107],
                 "m_tyresInnerTemperature": [95, 96, 113, 114],
                 "m_brakesTemperature": [420, 425, 690, 700],
                 "m_tyresPressure": [22.0, 22.1, 23.2, 23.3]})
        handler({"packet_id": 13,
                 "m_frontAeroHeight": 0.006 if is_kerb else 0.030,
                 "m_rearAeroHeight": 0.008 if is_kerb else 0.035,
                 "m_suspensionPosition": [0.02] * 4,
                 "m_suspensionAcceleration": [rough] * 4})
        handler({"packet_id": 0, "m_gForceLateral": 3.2,
                 "m_gForceLongitudinal": -4.1})
    handler({"packet_id": 10, "m_tyresWear": [12.0, 12.0, 26.0, 26.0],
             "m_floorDamage": 45, "m_frontLeftWingDamage": 30,
             "m_frontRightWingDamage": 28, "m_rearWingDamage": 5})
    handler({"packet_id": 7, "m_antiLockBrakes": 1, "m_tractionControl": 2,
             "m_actualTyreCompound": 16, "m_frontBrakeBias": 57,
             "m_tyresAgeLaps": 6})


class TestSuggestApiTelemetryOnly:
    def _client_with_telemetry(self, tmp_path: Path):
        from fastapi.testclient import TestClient

        from setup_tuner.app import _make_packet_handler, create_app
        from setup_tuner.config import Config

        app = create_app(Config(data_dir=str(tmp_path / "data")))
        handler = _make_packet_handler(app)
        client = TestClient(app)
        return app, handler, client

    def test_suggest_works_without_driver_feedback(self, tmp_path: Path) -> None:
        """核心回归：零车手反馈 + 有遥测发现 → 必须出方案（原来 400）。"""
        _app, handler, client = self._client_with_telemetry(tmp_path)
        with client:
            _synthetic_monza_lap(handler, 1)
            _synthetic_monza_lap(handler, 2)
            resp = client.post("/api/v1/suggest",
                               json={"track_id": "monza", "model_type": "hybrid"})
        assert resp.status_code == 200, resp.text[:400]
        report = resp.json()["data"]["report"]
        assert report.get("telemetry_only") is True
        assert report.get("telemetry_discovered", 0) > 0
        holistic = report["holistic"]
        assert holistic["implicit_feedbacks"], "报告必须列出遥测发现的问题"
        assert any(p.get("resolved") for p in holistic["implicit_plan"])
        mech = {"front_suspension", "rear_suspension", "front_anti_roll_bar",
                "rear_anti_roll_bar", "front_ride_height", "rear_ride_height"}
        delta = report["setup_delta"]
        assert any(delta.get(k) for k in mech), "整体性：悬挂类参数必须参与"

    def test_suggest_without_feedback_and_without_telemetry_still_400(
        self, tmp_path: Path,
    ) -> None:
        """回归：什么都没有时必须继续拒绝，不能变成"什么都出方案"。"""
        _app, _handler, client = self._client_with_telemetry(tmp_path)
        with client:
            resp = client.post("/api/v1/suggest", json={"track_id": "monza"})
        assert resp.status_code == 400
