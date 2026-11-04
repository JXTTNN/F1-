"""真实圈速校准层（real_lap_calibration）的加载与融合测试。

覆盖：

1. `build_norm_vector` 与 `normalize_setup` 同口径（训练/推理不漂移）
2. 校准文件缺失 → 纯仿真，行为与旧版完全一致
3. 校准文件损坏 → 静默降级，不抛异常
4. 融合：``(1−w)·仿真 + w·真实``，且默认调教恒为 0
5. 逐站权重覆盖全局权重；权重越界被夹到 [0, 1]
6. 系数按**名字**映射（文件里的字段顺序变了也不会错位）

测试全部自建临时模型/校准文件，不依赖 data/ 下的真实权重。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup, get_field
from setup_tuner.engine.pure_nn import MLP
from setup_tuner.engine.setup_sim import (
    CONDITION_KEYS,
    TRACK_FEATURE_KEYS,
    SetupSimModel,
    build_norm_vector,
    feature_names,
    normalize_setup,
)

_TRACK = "alpha"


def _zero_model(tmp_path: Path) -> Path:
    """构造一个输出恒为 0 的模型：让 `sim_delta` 归零，便于隔离校准层。"""
    track_ids = [_TRACK]
    n_in = len(track_ids) + len(TRACK_FEATURE_KEYS) + len(ALL_SETUP_FIELDS) \
        + len(CONDITION_KEYS)
    mlp = MLP([n_in, 4, 1], seed=3)
    # 先 fit 一次以写入标准化口径（x_mean/x_std）与 fitted 标记，再把权重清零：
    # 零权重 → 输出恒为 0，便于把「仿真分量」隔离掉，只看校准层的贡献。
    mlp.fit([[0.0] * n_in, [1.0] * n_in], [[0.0], [0.0]], epochs=1, batch_size=2)
    for layer in mlp.weights:
        for row in layer:
            for i in range(len(row)):
                row[i] = 0.0
    for layer in mlp.biases:
        for i in range(len(layer)):
            layer[i] = 0.0
    payload = {
        "track_ids": track_ids,
        "track_features": {_TRACK: {"aero_index": 0.5, "traction_index": 0.4}},
        "feature_keys": feature_names(track_ids),
        "n_samples": 1,
        "metrics": {},
        "mlp": {"model": mlp.to_dict()},
        "y_mean": 0.0,
        "y_scale": 1.0,
    }
    path = tmp_path / "setup_sim_nn.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _calibration(tmp_path: Path, **overrides) -> Path:
    coef = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
    coef["front_wing"] = -0.2   # 前翼越大 → 圈速越快（负增量）
    coef["rear_wing"] = 0.1
    payload = {
        "schema": "f1opt-real-lap-calibration/1",
        "n_samples": 1234,
        "blend_weight": 0.8,
        "blend_weight_by_track": {_TRACK: 0.8},
        "track_coef": {_TRACK: coef},
        "metrics": {"median_pair_acc_real": 0.56},
        "features": [f.name for f in ALL_SETUP_FIELDS],
    }
    payload.update(overrides)
    path = tmp_path / "real_lap_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestNormVector:
    def test_matches_normalize_setup(self) -> None:
        setup = CarSetup.default().to_dict()
        vec = build_norm_vector(setup)
        norm = normalize_setup(setup)
        assert len(vec) == len(ALL_SETUP_FIELDS)
        assert vec == pytest.approx([norm[f.name] for f in ALL_SETUP_FIELDS])

    def test_default_setup_is_zero_vector(self) -> None:
        vec = build_norm_vector(CarSetup.default().to_dict())
        assert all(abs(v) < 1e-12 for v in vec)


class TestLoading:
    def test_missing_calibration_keeps_pure_sim(self, tmp_path: Path) -> None:
        model = SetupSimModel(
            _zero_model(tmp_path), calibration_path=tmp_path / "nope.json",
        )
        assert model.real_track_coef == {}
        assert model.real_weight == 0.0
        assert "不存在" in model.real_reason
        # 纯仿真：前翼拉满也应当是 0（模型恒零）
        setup = {"front_wing": get_field("front_wing").max_val}
        assert model.predict_for_setup(_TRACK, setup) == pytest.approx(0.0)

    def test_corrupt_calibration_degrades(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=bad)
        assert model.real_track_coef == {}
        assert "失败" in model.real_reason

    def test_empty_coef_is_rejected(self, tmp_path: Path) -> None:
        empty = _calibration(tmp_path, track_coef={})
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=empty)
        assert model.real_track_coef == {}
        assert "没有可用" in model.real_reason


class TestBlending:
    def test_default_setup_stays_zero(self, tmp_path: Path) -> None:
        model = SetupSimModel(
            _zero_model(tmp_path), calibration_path=_calibration(tmp_path),
        )
        assert model.real_weight == pytest.approx(0.8)
        pred = model.predict_for_setup(_TRACK, CarSetup.default().to_dict())
        assert pred == pytest.approx(0.0, abs=1e-12)

    def test_blend_uses_weight(self, tmp_path: Path) -> None:
        model = SetupSimModel(
            _zero_model(tmp_path), calibration_path=_calibration(tmp_path),
        )
        setup = CarSetup.default().to_dict()
        setup["front_wing"] = get_field("front_wing").max_val  # n = +1
        # 真实校准 = -0.2 s；仿真 = 0 → 融合 = 0.8 × -0.2
        assert model.predict_for_setup(_TRACK, setup) == pytest.approx(-0.16)

    def test_track_weight_overrides_global(self, tmp_path: Path) -> None:
        cal = _calibration(tmp_path, blend_weight_by_track={_TRACK: 0.5})
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=cal)
        setup = {"front_wing": get_field("front_wing").max_val}
        assert model.predict_for_setup(_TRACK, setup) == pytest.approx(-0.10)

    def test_weight_is_clamped(self, tmp_path: Path) -> None:
        cal = _calibration(tmp_path, blend_weight=3.5,
                           blend_weight_by_track={_TRACK: -2.0})
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=cal)
        assert model.real_weight == pytest.approx(1.0)
        assert model.real_weight_by_track[_TRACK] == pytest.approx(0.0)

    def test_unknown_track_falls_back_to_sim(self, tmp_path: Path) -> None:
        model = SetupSimModel(
            _zero_model(tmp_path), calibration_path=_calibration(tmp_path),
        )
        assert model.predict_for_setup("not-a-track", {}) is None
        assert model.predict_real_delta_s("not-a-track", {}) is None

    def test_feature_order_in_file_does_not_matter(self, tmp_path: Path) -> None:
        """文件里 features 顺序与代码不同，系数仍按名字对上。"""
        coef = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
        coef["front_wing"] = -0.2
        payload = {
            "blend_weight": 1.0,
            "blend_weight_by_track": {_TRACK: 1.0},
            "track_coef": {_TRACK: coef},
            "features": [f.name for f in reversed(ALL_SETUP_FIELDS)],
        }
        path = tmp_path / "reordered.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=path)
        setup = {"front_wing": get_field("front_wing").max_val}
        assert model.predict_for_setup(_TRACK, setup) == pytest.approx(-0.20)


class TestDescribe:
    def test_describe_mentions_calibration(self, tmp_path: Path) -> None:
        model = SetupSimModel(
            _zero_model(tmp_path), calibration_path=_calibration(tmp_path),
        )
        text = model.describe()
        assert "真实校准" in text
        assert "1234" in text

    def test_describe_without_calibration_is_unchanged(self, tmp_path: Path) -> None:
        model = SetupSimModel(
            _zero_model(tmp_path), calibration_path=tmp_path / "nope.json",
        )
        assert "真实校准" not in model.describe()


class TestTrackAccuracy:
    """逐站准确率：面板要区分「听谁的」（weight）与「有多准」（accuracy）。"""

    def test_reads_track_accuracy(self, tmp_path: Path) -> None:
        path = _calibration(tmp_path, track_accuracy={_TRACK: 0.615})
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=path)
        assert model.real_accuracy_for(_TRACK) == pytest.approx(0.615)

    def test_missing_accuracy_returns_none(self, tmp_path: Path) -> None:
        model = SetupSimModel(
            _zero_model(tmp_path), calibration_path=_calibration(tmp_path),
        )
        assert model.real_accuracy_for(_TRACK) is None

    def test_below_chance_is_discarded(self, tmp_path: Path) -> None:
        """≤0.5 等于随机，不能拿到面板上冒充「有精度」。"""
        path = _calibration(tmp_path, track_accuracy={_TRACK: 0.48})
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=path)
        assert model.real_accuracy_for(_TRACK) is None

    def test_unknown_track_returns_none(self, tmp_path: Path) -> None:
        path = _calibration(tmp_path, track_accuracy={_TRACK: 0.7})
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=path)
        assert model.real_accuracy_for("nowhere") is None

    def test_malformed_accuracy_is_ignored(self, tmp_path: Path) -> None:
        path = _calibration(tmp_path, track_accuracy={_TRACK: "high"})
        model = SetupSimModel(_zero_model(tmp_path), calibration_path=path)
        assert model.real_accuracy_for(_TRACK) is None
