"""诊断脚本：追踪调教建议在各阶段（规则 → 优化 → 收口）里参数是怎么被改掉的。

用途：定位"悬挂几何参数在最终建议里被清零"这类整体性缺陷。
只读，不写任何数据。

用法::

    python scripts/trace_setup_pipeline.py
    python scripts/trace_setup_pipeline.py --track monza --symptom understeer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 关注的整体性参数（悬挂几何 / 悬挂 / 防倾杆 / 离地）
SUSPENSION_PARAMS = (
    "front_camber", "rear_camber", "front_toe", "rear_toe",
    "front_suspension", "rear_suspension",
    "front_anti_roll_bar", "rear_anti_roll_bar",
    "front_ride_height", "rear_ride_height",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="追踪调教建议管线各阶段")
    ap.add_argument("--track", default="monza")
    ap.add_argument("--symptom", default="understeer")
    ap.add_argument("--strength", type=int, default=3)
    ap.add_argument("--param", default=None, help="只关注某个参数")
    args = ap.parse_args(argv)

    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.coupling import nonzero_cells_for_param
    from setup_tuner.engine.engine import compute_setup_delta
    from setup_tuner.engine.holistic import class_weighted_dx, holistic_coherence, track_demand
    from setup_tuner.engine.lap_model import corner_evaluations, objective, track_context
    from setup_tuner.engine.optimizer import optimize_setup

    current = CarSetup.default().to_dict()
    feedbacks = [{
        "corner_number": 1, "symptom": args.symptom,
        "strength": args.strength, "category": "entry",
    }]
    weighted = class_weighted_dx(feedbacks, args.track)
    dx = weighted.dx
    initial = compute_setup_delta(dx, current)
    demand = track_demand(args.track)
    ctx = track_context(args.track, None, dx)

    res = optimize_setup(
        dx, current, args.track, feedbacks=feedbacks,
        initial_delta=initial, needs_by_class=weighted.by_class,
    )
    closed, notes = holistic_coherence(
        res.delta, dx, demand, mechanical_fallback=initial,
    )

    print(f"=== 管线追踪：{args.track} / {args.symptom}@{args.strength} ===")
    print(f"Dx: { {k: round(v, 3) for k, v in dx.items() if v} }")
    print(f"逐弯重要度前 5: "
          f"{[(e.number, e.klass, round(e.importance, 4)) for e in corner_evaluations(args.track)[:5]]}")
    print(f"目标函数: before={res.before.total:.6f} after={res.after.total:.6f} "
          f"改善={res.improvement:.6f}")
    print()
    header = f"{'参数':<26}{'规则初值':>10}{'优化后':>10}{'收口后':>10}  耦合来源"
    print(header)
    print("-" * len(header))
    params = [args.param] if args.param else list(SUSPENSION_PARAMS)
    for p in params:
        cells = nonzero_cells_for_param(p)
        src = ",".join(sorted({c.diag for c in cells})) or "(无耦合)"
        print(f"{p:<26}{initial.get(p, 0.0):>10.2f}{res.delta.get(p, 0.0):>10.2f}"
              f"{closed.get(p, 0.0):>10.2f}  {src}")

    print()
    nonzero_init = [p for p, v in initial.items() if v]
    nonzero_opt = [p for p, v in res.delta.items() if v]
    nonzero_closed = [p for p, v in closed.items() if v]
    print(f"非零项数：规则 {len(nonzero_init)} → 优化 {len(nonzero_opt)} "
          f"→ 收口 {len(nonzero_closed)}")
    print("优化轨迹（前 12 条）：")
    for line in res.trace[:12]:
        print(f"  - {line}")
    if notes:
        print("收口说明：")
        for n in notes:
            print(f"  - {n}")
    print()
    print("目标函数在三个阶段的取值：")
    print(f"  规则初值 : {objective(_units(initial, current), weighted.by_class, args.track, ctx).total:.6f}")
    print(f"  优化后   : {res.after.total:.6f}")
    return 0


def _units(delta, current):
    """把真实 delta 折算回归一化单位（与优化器口径一致）。"""
    from setup_tuner.domain.setup import get_field

    out = {}
    for name, value in (delta or {}).items():
        if not value:
            continue
        spec = get_field(name)
        out[name] = value / (spec.max_delta or 1.0)
    return out


if __name__ == "__main__":
    sys.exit(main())
