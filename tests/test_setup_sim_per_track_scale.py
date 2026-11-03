"""逐赛道输出标准化（v1.6.14）—— 训练端与推理端必须成对生效。

背景：24 条赛道的 ``lap_delta_ms`` 不是同一个尺度（Monaco 慢 1s 与 Monza 慢 1s
不是一回事）。早先只有**全局** ``y_mean``/``y_scale``，站间差异吃掉动态范围，
站内几十毫秒的决策区间被压得很窄。v1.6.14 改为逐站 z-score：训练端按站标定、
推理端按站还原。

本测试不依赖真实数据集（`data/training` 不入库），用合成数据跑完整训练入口，
锁定三件事：
1. 产物里逐站 ``mean``/``scale`` 齐全且**逐站不同**（不是把全局值抄一遍）；
2. 推理端 ``raw = z * scale[track] + mean[track]``，解码口径一致；
3. 旧模型（没有逐站字段）继续按全局口径解码，且差分校准仍使默认点恒为 0。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.engine.setup_sim import (
    TRACK_FEATURE_KEYS,
    SetupSimModel,
    normalize_setup,
)

ROOT = Path(__file__).resolve().parents[1]

_TRACKS = ["alpha", "beta", "gamma"]
#: 每站「1 单位代价 = 多少 ms」—— 故意拉成 1 : 2 : 3.5，逼出逐站 std 的差异
_SCALES = {"alpha": 1.0, "beta": 2.0, "gamma": 3.5}
_ACTIVE = ("front_wing", "rear_wing", "front_toe", "brake_bias")


def _write_dataset(tmp_path: Path, n_per_track: int = 60) -> tuple[Path, Path]:
    """合成数据集：标签 = 每站尺度 × 少数参数平方和（默认调教 → 0）。"""
    track_features: dict[str, dict[str, float]] = {}
    for i, tid in enumerate(_TRACKS):
        feats = {k: 0.2 + 0.1 * i for k in TRACK_FEATURE_KEYS}
        feats.update({"tyre_softness": 0.6, "wet": 0.0, "track_temp": 35.0, "air_temp": 24.0})
        track_features[tid] = feats

    rng = random.Random(7)
    lines: list[str] = []
    for tid in _TRACKS:
        for _ in range(n_per_track):
            norm = {f.name: rng.uniform(-1.0, 1.0) for f in ALL_SETUP_FIELDS}
            cost = sum(norm[k] ** 2 for k in _ACTIVE)
            lines.append(json.dumps({
                "track_id": tid,
                "setup_norm": norm,
                "lap_delta_ms": round(320.0 * _SCALES[tid] * cost, 3),
                "tier": "tube" if cost < 0.25 else "wide",
            }))

    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(lines), encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({
        "tracks": _TRACKS,
        "track_features": track_features,
        "anchor_sources": {t: "telemetry" for t in _TRACKS[:2]} | {_TRACKS[2]: "authored"},
    }), encoding="utf-8")
    return ds, meta


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """跑一遍真实训练入口，产出模型文件（模块级复用，只训一次）。"""
    tmp = tmp_path_factory.mktemp("per_track")
    ds, meta = _write_dataset(tmp)
    out = tmp / "model.json"
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_setup_sim_nn as tr

    rc = tr.main([
        "--dataset", str(ds), "--meta", str(meta), "--out", str(out),
        "--epochs", "3", "--val-ratio", "0.25",
    ])
    assert rc == 0
    assert out.exists()
    return out


def test_training_emits_per_track_scaling(trained: Path) -> None:
    """产物必须带逐站 mean/scale，且三者互不相同（真的按站标定过）。"""
    payload = json.loads(trained.read_text("utf-8"))
    assert payload["metrics"]["output_scaling"] == "per_track"
    assert set(payload["y_mean_by_track"]) == set(_TRACKS)
    assert set(payload["y_scale_by_track"]) == set(_TRACKS)
    scales = [payload["y_scale_by_track"][t] for t in _TRACKS]
    assert len({round(v, 6) for v in scales}) == len(_TRACKS), f"逐站 std 不应相同：{scales}"
    # 尺度顺序应与标签尺度一致（alpha < beta < gamma）
    assert scales == sorted(scales)
    # 全局口径仍保留，供未知赛道/旧模型兜底
    assert payload["y_scale"] > 0


def test_inference_decodes_with_track_scale(trained: Path) -> None:
    """推理端必须按**该赛道**的 mean/scale 还原，而不是全局那一对。"""
    model = SetupSimModel(trained)
    assert model.available
    norm = normalize_setup(CarSetup.default().to_dict())
    payload = json.loads(trained.read_text("utf-8"))
    for tid in _TRACKS:
        row = model.build_features(tid, norm)
        assert row is not None
        z = model._mlp.predict_one(row)[0]
        mean, scale = payload["y_mean_by_track"][tid], payload["y_scale_by_track"][tid]
        assert model.output_scale(tid) == pytest.approx((mean, scale))
        assert model.predict_raw_delta_s(tid, norm) == pytest.approx(z * scale + mean)
    # 全局口径与之不同 → 证明真的走了逐站分支
    assert model.output_scale(_TRACKS[0]) != (model.y_mean, model.y_scale)


def test_default_setup_still_predicts_zero(trained: Path) -> None:
    """差分校准不受缩放口径影响：默认调教在每条赛道都必须预测 0。"""
    model = SetupSimModel(trained)
    norm = normalize_setup(CarSetup.default().to_dict())
    for tid in _TRACKS:
        delta = model.predict_delta_s(tid, norm)
        assert delta is not None
        assert abs(delta) < 1e-9, f"{tid} 默认点未归零：{delta}"


def test_legacy_model_falls_back_to_global_scale(trained: Path, tmp_path: Path) -> None:
    """旧模型（无逐站字段）必须按全局口径继续可用 —— 不能让老用户静默降级。"""
    payload = json.loads(trained.read_text("utf-8"))
    legacy = {k: v for k, v in payload.items()
              if k not in ("y_mean_by_track", "y_scale_by_track")}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    model = SetupSimModel(path)
    assert model.available
    assert model.y_scale_by_track == {}
    assert model.output_scale(_TRACKS[0]) == pytest.approx((model.y_mean, model.y_scale))
    norm = normalize_setup(CarSetup.default().to_dict())
    row = model.build_features(_TRACKS[0], norm)
    z = model._mlp.predict_one(row)[0]
    assert model.predict_raw_delta_s(_TRACKS[0], norm) == pytest.approx(
        z * model.y_scale + model.y_mean,
    )
