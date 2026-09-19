"""验证「遥测训练的模型确实进入了本地 F1OPT 调教优化链路」。

四步验证（全部可复现、只读）：

1. **模型是否加载**：打印代理模型的学习器、样本量、留出集指标；
2. **模型是否改变优化权重**：对比"数据驱动逐弯重要度"与旧的
   ``120/参考速度`` 启发式，证明两者不同（即遥测真的在起作用）；
3. **车手未反馈的问题是否被发现**：喂入只含遥测（无任何车手反馈）的工况，
   检查报告里是否出现隐式症状 + 悬挂/几何类改动；
4. **悬挂几何是否参与**：确认最终建议里出现了悬挂几何/悬挂/防倾杆参数。

用法::

    python scripts/verify_model_in_engine.py
    python scripts/verify_model_in_engine.py --track monza
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MECH = (
    "front_camber", "rear_camber", "front_toe", "rear_toe",
    "front_suspension", "rear_suspension",
    "front_anti_roll_bar", "rear_anti_roll_bar",
    "front_ride_height", "rear_ride_height",
)


def _section(title: str) -> None:
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="验证模型已接入调教优化")
    ap.add_argument("--track", default="monza")
    args = ap.parse_args(argv)
    ok = True

    from setup_tuner.domain.track import get_track_by_id
    from setup_tuner.engine.lap_model import _TIME_REF_SPEED, corner_evaluations
    from setup_tuner.engine.surrogate import get_surrogate

    _section("1. 代理模型加载状态")
    model = get_surrogate()
    print(f"  权重文件 : {model.path}（存在={model.path.exists()}）")
    print(f"  可用     : {model.available}")
    print(f"  摘要     : {model.describe()}")
    if not model.available:
        print("  ✗ 模型未加载 —— 请先跑 python scripts/retrain_all.py")
        return 1
    print(f"  训练样本 : {model.trained_samples}")
    print(f"  弯级指标 : {model.corner_metrics}")

    _section("2. 模型是否改变了优化权重（vs 旧启发式）")
    evals = corner_evaluations(args.track)
    learned = model.corner_importance(args.track)
    track = get_track_by_id(args.track)
    if not evals or not learned:
        print(f"  ✗ 该赛道缺少模型数据：{args.track}")
        return 1
    print(f"  {'弯':>4}{'类型':>8}{'模型权重':>12}{'启发式':>12}{'差异':>10}")
    diffs = 0
    for e in evals[:8]:
        c = next(c for c in track.corners if c.number == e.number)
        heuristic_raw = _TIME_REF_SPEED / max(30.0, float(c.speed_kmh))
        print(f"  {e.number:>4}{e.klass:>8}{e.importance:>12.4f}"
              f"{heuristic_raw:>12.4f}{e.importance - heuristic_raw:>10.4f}")
        diffs += 1
    # 等价性检验：把模型权重换成启发式，优化输入必然不同
    from setup_tuner.engine import surrogate as surrogate_mod
    saved = dict(surrogate_mod.get_surrogate().importance)
    surrogate_mod.get_surrogate().importance = {}
    heur = {e.number: e.importance for e in corner_evaluations(args.track)}
    surrogate_mod.get_surrogate().importance = saved
    diff_count = sum(
        1 for n, w in heur.items()
        if abs(w - next(e.importance for e in evals if e.number == n)) > 1e-6
    )
    print(f"  → 与启发式不同的弯：{diff_count}/{len(heur)}")
    ok = ok and diff_count > 0

    _section("3. 车手零反馈时，遥测能否自动发现问题")
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.engine import generate_suggestion

    telemetry = {
        "kerb_corners": [
            {"corner": 5, "ratio": 10.08, "frames": 299, "side": "left"},
            {"corner": 2, "ratio": 7.10, "frames": 770, "side": "right"},
        ],
        "plank_bottoming": True,
        "plank_bottoming_ratio": 0.08,
        "m_tyresSurfaceTemperature": [88.0, 89.0, 104.0, 106.0],
        "m_brakesTemperature": [420.0, 430.0, 700.0, 710.0],
        "avg_steer": 0.24,
        "m_tyresPressure": [22.0, 22.0, 23.0, 23.0],
    }
    result = generate_suggestion(
        symptoms=[],                      # ← 车手什么都没点
        current_setup=CarSetup.default().to_dict(),
        track_id=args.track,
        telemetry=telemetry,
        model_type="rule",
    )
    implicit = result["holistic"]["implicit_feedbacks"]
    print("  车手反馈条数 : 0")
    print(f"  自动发现条数 : {len(implicit)}")
    for it in implicit[:8]:
        where = f"T{it['corner']}" if it["corner"] else "全局"
        print(f"    - {where:<6}{it['symptom']:<24}强度{it['strength']}  "
              f"[{it['source']}] {it['evidence']}")
    ok = ok and len(implicit) > 0
    print(f"  Dx（由遥测+隐式诊断共同驱动）: "
          f"{ {k: round(v, 3) for k, v in result['dx'].items() if v} }")

    _section("4. 最终调教里是否有悬挂几何类改动")
    delta = result["setup_delta"]
    changed = {k: v for k, v in delta.items() if v}
    mech = {k: v for k, v in delta.items() if v and k in _MECH}
    print(f"  非零改动 {len(changed)} 项：")
    for k, v in changed.items():
        mark = "★" if k in _MECH else " "
        print(f"    {mark} {k:<28}{v:+.2f}")
    print(f"  其中悬挂/几何/防倾杆：{len(mech)} 项 {mech if mech else '（无）'}")
    notes = result["holistic"].get("coherence_notes") or []
    for n in notes:
        if "机械抓地" in n or "rake" in n or "配对" in n:
            print(f"  收口说明：{n}")
    ok = ok and len(mech) > 0

    _section("结论")
    print("  ✅ 四项验证全部通过：模型已接入本地 F1OPT 调教优化链路"
          if ok else "  ✗ 存在未通过项，见上文")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
