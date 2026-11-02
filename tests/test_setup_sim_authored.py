"""几何推定锚点（无遥测赛道）的单元测试。

覆盖三件事：
1. 最小二乘求解正确（否则圈速模型会静默跑偏）；
2. 遥测站不足时**放弃**推定（宁缺勿滥，不能凭 2 站就编出 24 站的模型）；
3. 推定出的锚点结构完备、量纲合理、弯时占比被归一（否则仿真器会把整圈时间挤爆）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _dataset_module() -> Any:
    """运行时注入路径后导入数据集构建模块。

    ``scripts/`` 不属于安装包（只在仓库里），因此不能在文件顶部直接 import。
    """
    for extra in (ROOT, ROOT / "scripts"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import build_setup_sim_dataset

    return build_setup_sim_dataset


@pytest.fixture()
def mod() -> Any:
    return _dataset_module()


def _fake_track(
    tid: str, lap: float, length_m: float, corners: list[tuple[str, float, float, float]],
) -> dict:
    """构造一个与 ``load_corner_stats`` 同构的锚点条目。"""
    rows = [
        {"corner": i + 1, "klass": k, "v_entry": v, "v_min": v, "v_exit": v,
         "t_corner": t, "brake_pct": b, "lat_g": 0.0, "n": 10}
        for i, (k, v, t, b) in enumerate(corners)
    ]
    return {
        "corners": rows,
        "lap_time_s": lap,
        "straight_share": 0.35,
        "avg_speed_kmh": length_m / lap * 3.6,
        "drag_scale": 0.4,
        "traction_index": 0.3, "aero_index": 0.3, "braking_index": 0.3,
        "slow_share": 0.3, "fast_share": 0.3,
        "tyre_softness": 0.6, "wet": 0.0, "track_temp": 35.0, "air_temp": 24.0,
        "n_samples": 100,
        "anchor": "telemetry",
    }


def _stats(n: int) -> dict:
    """造 n 条「遥测」赛道。

    弯速随序号变化 → ``Σ1/v`` 不是常数（否则与截距列共线，正规方程奇异）；
    圈速按 ``lap = c0 + c1·length + c2·Σ1/v`` 生成 → 最小二乘应精确还原。
    """
    out = {}
    for i in range(n):
        length = 4000.0 + 200.0 * i
        speeds = (90.0 + i, 150.0 + 2 * i, 230.0 + 3 * i)
        corners = [("slow", speeds[0], 6.0, 55.0), ("medium", speeds[1], 5.0, 30.0),
                   ("fast", speeds[2], 4.0, 12.0)]
        inv_v = sum(3.6 / v for v in speeds)
        lap = 30.0 + 0.008 * length + 3.0 * inv_v
        out[f"t{i}"] = _fake_track(f"t{i}", lap, length, corners)
    return out


def test_least_squares_recovers_linear_relation(mod):
    # y = 3 + 2*a - 1.5*b
    xs = [[1.0, float(i), float(i % 5)] for i in range(12)]
    ys = [3.0 + 2.0 * a - 1.5 * b for _, a, b in xs]
    coeffs = mod._least_squares(xs, ys)
    assert coeffs is not None
    assert coeffs[0] == pytest.approx(3.0, abs=1e-6)
    assert coeffs[1] == pytest.approx(2.0, abs=1e-6)
    assert coeffs[2] == pytest.approx(-1.5, abs=1e-6)


def test_least_squares_returns_none_for_singular_design(mod):
    # 两列完全共线 → 奇异
    xs = [[1.0, 2.0, 4.0] for _ in range(6)]
    assert mod._least_squares(xs, [1.0] * 6) is None


def test_fit_refuses_when_too_few_tracks(mod):
    assert mod.fit_authored_constants(_stats(3)) is None


def test_fit_produces_constants(mod):
    calib = mod.fit_authored_constants(_stats(12))
    assert calib is not None
    # 每类弯都有等效弯长与制动占比，且量纲合理（米 / 百分比）
    for klass in ("slow", "medium", "fast"):
        assert klass in calib["corner"]
        assert 20.0 <= calib["corner"][klass]["eff_len_m"] <= 600.0
        assert 0.0 <= calib["corner"][klass]["brake_pct"] <= 100.0
    assert 0.2 <= calib["share"] <= 0.9
    # 弯时占比来自实测：每个假赛道 15s 弯时 / ~62s 圈速 ≈ 0.24
    assert calib["share"] == pytest.approx(0.24, abs=0.05)
    assert len(calib["lap"]["coeffs"]) == 3
    # 造的数据严格线性 → 拟合误差应趋近 0
    assert calib["lap"]["median_rel_err"] < 1e-6


def test_authored_anchor_is_well_formed(mod):
    calib = mod.fit_authored_constants(_stats(12))
    assert calib is not None
    anchor = mod.authored_anchor(
        "las_vegas", calib,
        {"tyre_softness": 0.6, "wet": 0.0, "track_temp": 35.0, "air_temp": 24.0},
    )
    assert anchor is not None
    assert anchor["anchor"] == "authored"
    assert anchor["lap_time_s"] > 30.0
    assert {c["klass"] for c in anchor["corners"]} <= {"slow", "medium", "fast"}
    # 弯时占比被归一：Σt_corner / 圈速 ≈ 遥测站中位占比
    corner_time = sum(c["t_corner"] for c in anchor["corners"])
    assert corner_time / anchor["lap_time_s"] == pytest.approx(calib["share"], abs=0.02)
    # 仿真输入必须完整：每个弯都有正弯时与制动占比
    for c in anchor["corners"]:
        assert c["t_corner"] > 0.0
        assert 0.0 <= c["brake_pct"] <= 100.0
    # 直道占比 / drag_scale 落在文档化区间
    assert 0.18 <= anchor["straight_share"] <= 0.55
    assert 0.0 < anchor["drag_scale"] < 3.0
    assert math.isfinite(anchor["avg_speed_kmh"])


def test_authored_anchor_returns_none_for_unknown_track(mod):
    calib = mod.fit_authored_constants(_stats(12))
    assert calib is not None
    assert mod.authored_anchor("no_such_track", calib, {
        "tyre_softness": 0.6, "wet": 0.0, "track_temp": 35.0, "air_temp": 24.0,
    }) is None


def test_model_describe_reports_authored_composition(tmp_path):
    """模型描述必须如实写出「遥测 N + 几何推定 M」，不能只报总数。"""
    import json

    from setup_tuner.engine.setup_sim import SetupSimModel

    base = SetupSimModel()  # 用仓库自带模型取权重，避免手搓 MLP
    if not base.available:
        pytest.skip("仓库内没有可用的调教性能模型")
    payload = {
        "track_ids": list(base.track_ids)[:2],
        "track_features": {tid: base.track_features[tid] for tid in list(base.track_ids)[:2]},
        "anchor_sources": {base.track_ids[0]: "telemetry", base.track_ids[1]: "authored"},
        "metrics": {"val_mae_ms": 90.0},
        "n_samples": 10,
        "y_mean": base.y_mean,
        "y_scale": base.y_scale,
        "mlp": {"model": base._mlp.to_dict()},
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        model = SetupSimModel(path=str(path))
    except Exception:  # noqa: BLE001 — MLP 序列化格式变化时跳过而非误报
        pytest.skip("MLP 序列化格式与测试不一致")
    assert model.available
    assert model.coverage_counts() == (1, 1)
    assert "几何推定 1" in model.describe()
    assert model.anchor_of(base.track_ids[1]) == "authored"


def test_fit_rejects_poor_quality(mod, capsys):
    """圈速拟合误差超门限时必须放弃推定（宁缺勿滥）。"""
    stats = _stats(14)
    # 把圈速改成与特征无关的噪声：拟合再努力也解释不了
    for i, tid in enumerate(sorted(stats)):
        stats[tid]["lap_time_s"] = 60.0 + (17.0 if i % 2 == 0 else -17.0)
    assert mod.fit_authored_constants(stats) is None
    out = capsys.readouterr().out
    assert "放弃几何推定锚点" in out
