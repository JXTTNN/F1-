"""setup_sim 模块测试 —— 调教性能神经网络（纯标准库，无 PyTorch）。

覆盖：

1. 归一化：half_range / normalize_setup / denormalize_param（真实值 ↔ n）
2. 特征向量：build_feature_row 结构与 one-hot；未知赛道 → None
3. 模型加载：缺文件 / 损坏文件 / 合法模型 → available 与 reason
4. 推理：已知赛道出数值、未知赛道与不可用 → None；输出缩放口径
5. predict_for_setup：真实调教入口
6. 覆盖判定 covers / describe
7. 端到端：sim_refine 在模型可用时确实会改动 delta（模拟优化生效）

测试全部自建临时模型文件（不依赖 data/ 下的真实权重），
保证 CI 数据目录隔离下确定性通过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field
from setup_tuner.engine.pure_nn import MLP
from setup_tuner.engine.setup_sim import (
    CONDITION_KEYS,
    TRACK_FEATURE_KEYS,
    SetupSimModel,
    build_feature_row,
    denormalize_param,
    feature_names,
    half_range,
    normalize_setup,
)

_TRACK_A = "alpha"
_TRACK_B = "beta"


def _make_model_payload(tmp_path: Path) -> Path:
    """构造一个自包含的小模型文件（2 赛道，恒定输出 0.5 / -0.5）。"""
    track_ids = [_TRACK_A, _TRACK_B]
    track_features = {
        _TRACK_A: {
            "traction_index": 0.4, "aero_index": 0.2, "braking_index": 0.1,
            "slow_share": 0.3, "fast_share": 0.2,
            "tyre_softness": 0.6, "wet": 0.0, "track_temp": 35.0, "air_temp": 24.0,
        },
        _TRACK_B: {
            "traction_index": 0.1, "aero_index": 0.5, "braking_index": 0.2,
            "slow_share": 0.1, "fast_share": 0.4,
            "tyre_softness": 0.3, "wet": 1.0, "track_temp": 20.0, "air_temp": 15.0,
        },
    }
    n_in = len(track_ids) + len(TRACK_FEATURE_KEYS) + len(ALL_SETUP_FIELDS) \
        + len(CONDITION_KEYS)
    mlp = MLP([n_in, 8, 1], seed=1)
    # 用少量样本把模型标记为 fitted（结构性测试不关心拟合质量）
    x = [[0.0] * n_in, [1.0] * n_in]
    mlp.fit(x, [[0.2], [0.8]], epochs=3, batch_size=2)
    payload = {
        "schema": "f1opt-setup-sim-nn/1",
        "track_ids": track_ids,
        "track_features": track_features,
        "feature_keys": feature_names(track_ids),
        "n_samples": 2,
        "metrics": {"val_mae_ms": 1.0, "val_r2": 0.99},
        "mlp": {"model": mlp.to_dict()},
        "y_mean": 0.5,
        "y_scale": 2.0,
        "created": "2026-09-19T00:00:00",
    }
    path = tmp_path / "setup_sim_nn.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ===========================================================================
# 1. 归一化
# ===========================================================================
class TestNormalization:
    def test_half_range_covers_far_end(self) -> None:
        """半幅取到较远一端，保证 n ∈ [-1, 1]。"""
        spec = get_field("front_wing")  # 0..50, default 25
        assert half_range(spec) == pytest.approx(25.0)
        spec2 = get_field("front_camber")  # -3.5..-2.5, default -3.5
        assert half_range(spec2) == pytest.approx(1.0)

    def test_normalize_default_is_zero(self) -> None:
        """默认调教 → 全零（n=0）。"""
        from setup_tuner.domain.setup import CarSetup

        n = normalize_setup(CarSetup.default().to_dict())
        assert all(abs(v) < 1e-12 for v in n.values())

    def test_normalize_extremes_within_unit_range(self) -> None:
        """合法区间内任意值 → |n| ≤ 1。"""
        n_max = normalize_setup({f.name: f.max_val for f in ALL_SETUP_FIELDS})
        n_min = normalize_setup({f.name: f.min_val for f in ALL_SETUP_FIELDS})
        for f in ALL_SETUP_FIELDS:
            assert abs(n_max[f.name]) <= 1.0 + 1e-12
            assert abs(n_min[f.name]) <= 1.0 + 1e-12

    def test_denormalize_round_trip(self) -> None:
        """denormalize_param 是 normalize 的逆（在合法区间内）。"""
        spec = get_field("front_wing")
        for raw in (0.0, 12.0, 25.0, 40.0, 50.0):
            n = (raw - spec.default) / half_range(spec)
            assert denormalize_param(spec.name, n) == pytest.approx(raw)

    def test_denormalize_clamps(self) -> None:
        """超出区间的 n 会被夹取。"""
        spec = get_field("front_wing")
        assert denormalize_param(spec.name, 5.0) == pytest.approx(spec.max_val)
        assert denormalize_param(spec.name, -5.0) == pytest.approx(spec.min_val)


# ===========================================================================
# 2. 特征向量
# ===========================================================================
class TestFeatureRow:
    def test_shape_and_onehot(self) -> None:
        track_ids = [_TRACK_A, _TRACK_B]
        tf = {t: {} for t in track_ids}
        row = build_feature_row(track_ids, tf, _TRACK_B, normalize_setup({}))
        assert row is not None
        n_in = len(track_ids) + 5 + len(ALL_SETUP_FIELDS) + 4
        assert len(row) == n_in
        # one-hot：B 位置 1、A 位置 0
        assert row[0] == 0.0 and row[1] == 1.0

    def test_unknown_track_returns_none(self) -> None:
        row = build_feature_row([_TRACK_A], {_TRACK_A: {}}, "nope", {})
        assert row is None

    def test_setup_values_embedded(self) -> None:
        track_ids = [_TRACK_A]
        tf = {_TRACK_A: {}}
        n = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
        n["front_wing"] = -0.5
        row = build_feature_row(track_ids, tf, _TRACK_A, n)
        assert row is not None
        idx = len(track_ids) + 5 + [f.name for f in ALL_SETUP_FIELDS].index("front_wing")
        assert row[idx] == pytest.approx(-0.5)

    def test_feature_names_length_matches(self) -> None:
        names = feature_names([_TRACK_A, _TRACK_B])
        row = build_feature_row([_TRACK_A, _TRACK_B], {_TRACK_A: {}}, _TRACK_A, {})
        assert row is not None
        assert len(names) == len(row)


# ===========================================================================
# 3. 模型加载
# ===========================================================================
class TestModelLoading:
    def test_missing_file_unavailable(self, tmp_path: Path) -> None:
        model = SetupSimModel(tmp_path / "nope.json")
        assert model.available is False
        assert "不存在" in model.reason

    def test_corrupt_file_unavailable(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        model = SetupSimModel(p)
        assert model.available is False

    def test_missing_mlp_block_unavailable(self, tmp_path: Path) -> None:
        p = tmp_path / "incomplete.json"
        p.write_text(json.dumps({"track_ids": ["a"]}), encoding="utf-8")
        model = SetupSimModel(p)
        assert model.available is False
        assert "mlp" in model.reason

    def test_valid_model_available(self, tmp_path: Path) -> None:
        model = SetupSimModel(_make_model_payload(tmp_path))
        assert model.available is True
        assert model.reason == "ok"
        assert model.track_ids == [_TRACK_A, _TRACK_B]
        assert model.n_samples == 2
        assert model.y_scale == pytest.approx(2.0)

    def test_describe_mentions_state(self, tmp_path: Path) -> None:
        model = SetupSimModel(_make_model_payload(tmp_path))
        text = model.describe()
        assert "调教性能 NN" in text
        unavailable = SetupSimModel(tmp_path / "nope.json")
        assert "不可用" in unavailable.describe()


# ===========================================================================
# 4. 推理
# ===========================================================================
class TestPredict:
    def test_predict_known_track_returns_float(self, tmp_path: Path) -> None:
        model = SetupSimModel(_make_model_payload(tmp_path))
        value = model.predict_delta_s(_TRACK_A, normalize_setup({}))
        assert isinstance(value, float)

    def test_predict_applies_output_scaling(self, tmp_path: Path) -> None:
        """输出 = mlp_raw × y_scale + y_mean（缩放口径必须一致）。"""
        p = _make_model_payload(tmp_path)
        model = SetupSimModel(p)
        raw = float(model._mlp.predict_one(
            model.build_features(_TRACK_A, normalize_setup({})),
        )[0])
        expected = raw * 2.0 + 0.5
        assert model.predict_delta_s(_TRACK_A, normalize_setup({})) == pytest.approx(expected)

    def test_predict_unknown_track_none(self, tmp_path: Path) -> None:
        model = SetupSimModel(_make_model_payload(tmp_path))
        assert model.predict_delta_s("no_such_track", normalize_setup({})) is None

    def test_predict_unavailable_none(self, tmp_path: Path) -> None:
        model = SetupSimModel(tmp_path / "nope.json")
        assert model.predict_delta_s(_TRACK_A, normalize_setup({})) is None

    def test_predict_for_setup_accepts_real_values(self, tmp_path: Path) -> None:
        from setup_tuner.domain.setup import CarSetup

        model = SetupSimModel(_make_model_payload(tmp_path))
        value = model.predict_for_setup(_TRACK_A, CarSetup.default().to_dict())
        assert isinstance(value, float)

    def test_covers(self, tmp_path: Path) -> None:
        model = SetupSimModel(_make_model_payload(tmp_path))
        assert model.covers(_TRACK_A) is True
        assert model.covers("nope") is False


# ===========================================================================
# 5. 端到端：模拟优化在模型可用时确实会改动 delta
# ===========================================================================
class TestSimRefineEndToEnd:
    def test_refine_changes_delta_with_model(self, tmp_path: Path, monkeypatch) -> None:
        """模型可用时，sim_refine 至少能接受一个让模拟圈速变快的候选。"""
        from dataclasses import dataclass

        import setup_tuner.engine.sim_optimizer as opt_mod

        @dataclass
        class _Model:
            available = True
            reason = "ok"
            track_ids = ("alpha", "beta")

            def covers(self, track_id: str) -> bool:
                return track_id in self.track_ids

            def predict_delta_s(self, track_id: str, setup_norm: dict) -> float:
                # 单调函数：front_wing 的 n 越负（减翼）模拟圈速越快 —— 给优化器
                # 一个明确可下降的方向
                return 0.1 + setup_norm.get("front_wing", 0.0)

            def describe(self) -> str:
                return "test"

        monkeypatch.setattr(opt_mod, "get_setup_sim", lambda *a, **k: _Model())

        from setup_tuner.domain.setup import CarSetup

        setup = CarSetup.default().to_dict()
        delta0 = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
        result = opt_mod.sim_refine(delta0, setup, "alpha", dx=None)
        assert result.available is True
        assert result.accepted >= 1
        assert result.delta["front_wing"] < 0.0     # 减翼方向被采纳
        assert result.gain_s is not None and result.gain_s > 0.0

    def test_refine_deterministic(self, tmp_path: Path, monkeypatch) -> None:
        """相同输入两次运行逐位一致（可复现约定）。"""
        from dataclasses import dataclass

        import setup_tuner.engine.sim_optimizer as opt_mod

        @dataclass
        class _Model:
            available = True
            reason = "ok"
            track_ids = ("alpha",)

            def covers(self, track_id: str) -> bool:
                return track_id in self.track_ids

            def predict_delta_s(self, track_id: str, setup_norm: dict) -> float:
                return setup_norm.get("front_wing", 0.0) + setup_norm.get("rear_wing", 0.0) * 0.5

            def describe(self) -> str:
                return "test"

        monkeypatch.setattr(opt_mod, "get_setup_sim", lambda *a, **k: _Model())

        from setup_tuner.domain.setup import CarSetup

        setup = CarSetup.default().to_dict()
        delta0 = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
        r1 = opt_mod.sim_refine(delta0, setup, "alpha", dx=None)
        r2 = opt_mod.sim_refine(delta0, setup, "alpha", dx=None)
        assert r1.delta == r2.delta
        assert r1.trace == r2.trace


# ===========================================================================
# 6. 引擎级契约：nn 模式真的跑模拟优化，rule 模式不跑
# ===========================================================================
class _FakeSim:
    """确定性假模型：只有 front_wing 影响圈速（前翼越大越快）。

    方向刻意与"推头（understeer）"的正解一致（加前翼 = 加前轴抓地 = 缓解推头），
    这样模拟优化既满足车手需求又"更快"，目标代价判据不会拦下它 ——
    用来验证引擎确实把 NN 模拟优化接进了主链路。
    """

    available = True
    reason = "ok"
    track_ids = ("suzuka",)
    y_mean = 0.0
    y_scale = 1.0
    n_samples = 1
    metrics = {"val_mae_ms": 0.0, "val_r2": 1.0}

    def covers(self, track_id: str) -> bool:
        return track_id in self.track_ids

    def predict_delta_s(self, track_id: str, setup_norm: dict) -> float | None:
        if track_id not in self.track_ids:
            return None
        return 0.05 - 3.0 * setup_norm.get("front_wing", 0.0)

    def describe(self) -> str:
        return "fake-sim"


@pytest.fixture()
def _fake_engine_model(monkeypatch):
    """把引擎与模拟优化器里的 get_setup_sim 都替换为确定性假模型。"""
    import setup_tuner.engine.engine as eng_mod
    import setup_tuner.engine.sim_optimizer as opt_mod

    monkeypatch.setattr(opt_mod, "get_setup_sim", lambda *a, **k: _FakeSim())
    monkeypatch.setattr(eng_mod, "get_setup_sim", lambda *a, **k: _FakeSim())
    return _FakeSim()


class TestEngineSimulationContract:
    """``generate_suggestion`` 在 nn/hybrid 下必须真的跑模拟优化。"""

    def test_nn_mode_runs_simulation(self, _fake_engine_model) -> None:
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        res = generate_suggestion(
            [("understeer", 2)], CarSetup.default().to_dict(), "suzuka", None,
            model_type="nn",
        )
        assert res["model_type"] == "nn"
        assert res["nn_available"] is True
        sim = res["holistic"]["simulation"]
        assert sim["available"] is True
        assert sim["accepted"] >= 1
        assert sim["gain_ms"] is not None and sim["gain_ms"] > 0
        assert sim["trace"], "模拟优化应留下改动轨迹"
        # 假模型只奖励加前翼 → 结果必须真的加了前翼
        assert res["setup_delta"]["front_wing"] > 0.0

    def test_rule_mode_skips_simulation(self, _fake_engine_model) -> None:
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        res = generate_suggestion(
            [("understeer", 2)], CarSetup.default().to_dict(), "suzuka", None,
            model_type="rule",
        )
        assert res["model_type"] == "rule"
        sim = res["holistic"]["simulation"]
        assert sim["available"] is False
        assert "rule" in sim["reason"]

    def test_zero_dx_skips_simulation(self, _fake_engine_model) -> None:
        """无症状/无遥测发现 → 不改车（保持既有契约）。"""
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        res = generate_suggestion([], CarSetup.default().to_dict(), "suzuka", None)
        assert res["model_type"] == "rule"
        assert all(abs(v) < 1e-12 for v in res["setup_delta"].values())
        assert "无诊断输入" in res["holistic"]["simulation"]["reason"]

    def test_uncovered_track_degrades(self, _fake_engine_model) -> None:
        """赛道不在模型覆盖范围 → 不跑模拟，也不改变 delta。"""
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        setup = CarSetup.default().to_dict()
        nn_res = generate_suggestion([("understeer", 2)], setup, "monza", None,
                                     model_type="nn")
        rule_res = generate_suggestion([("understeer", 2)], setup, "monza", None,
                                       model_type="rule")
        assert nn_res["model_type"] == "rule"
        assert "覆盖范围" in nn_res["holistic"]["simulation"]["reason"]
        assert nn_res["setup_delta"] == rule_res["setup_delta"]

    def test_determinism_with_simulation(self, _fake_engine_model) -> None:
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        setup = CarSetup.default().to_dict()
        a = generate_suggestion([("understeer", 2)], setup, "suzuka", None,
                                model_type="nn")
        b = generate_suggestion([("understeer", 2)], setup, "suzuka", None,
                                model_type="nn")
        assert a["setup_delta"] == b["setup_delta"]
        assert a["holistic"]["simulation"]["trace"] == b["holistic"]["simulation"]["trace"]

    def test_result_respects_bounds_and_pressure_symmetry(self, _fake_engine_model) -> None:
        """模拟优化结果仍受参数上下限/max_delta 约束，左右胎压一致。"""
        from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        setup = CarSetup.default().to_dict()
        res = generate_suggestion([("understeer", 2)], setup, "suzuka", None,
                                  model_type="nn")
        for spec in ALL_SETUP_FIELDS:
            nxt = setup[spec.name] + res["setup_delta"][spec.name]
            assert spec.min_val - 1e-6 <= nxt <= spec.max_val + 1e-6
            assert abs(res["setup_delta"][spec.name]) <= spec.max_delta + 1e-6
        assert (res["setup_delta"]["front_left_tyre_pressure"]
                == res["setup_delta"]["front_right_tyre_pressure"])
        assert (res["setup_delta"]["rear_left_tyre_pressure"]
                == res["setup_delta"]["rear_right_tyre_pressure"])
