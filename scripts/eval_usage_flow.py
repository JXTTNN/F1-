"""按「系统使用流程」评测调教模型是否真的生效（端到端，走引擎真实入口）。

为什么需要这个脚本
------------------
``eval_setup_sim.py`` 量的是「模型对物理仿真器的保真度」——那是必要条件，
不是用户拿到的东西。用户真正经历的是：

    选赛道 → 赛道图点问题弯 / 录反馈（症状 + 强度）→ 点「生成建议」
    → 报告里的「神经网络模拟」面板（模型 / 迭代 / 采纳 / 模拟圈速提升）
    → 照着改车 → 下一圈再反馈

这条链上任何一环断了，保真度再高，实际收益都是 0：

- 模型没覆盖该赛道 → 面板显示「未启用」，用户白选了一条赛道；
- 无诊断输入却动了车 → 违反「无症状不出建议」的契约；
- 坐标上升一步都没采纳 → 模型跑了一遍却什么也没改，用户看不出区别；
- 采纳后用 ``max_delta`` / 上下限 / 胎压对称性去夹，把收益夹没了；
- 报告里承诺的 ``gain_ms`` 与最终 ``setup_delta`` 对不上。

因此本脚本**直接调用** ``engine.generate_suggestion``（与
``POST /api/v1/suggest`` 完全同一条代码路径），对「全部赛道 × 典型车手反馈」
做全矩阵跑测，把上面每一条变成可断言的数字。

指标
----
====================  ==================================================
``nn_active_rate``    实际走「神经网络模拟优化」的赛道-场景比例（目标 1.0）
``accept_rate``       至少采纳 1 步改动的比例（否则模型等于没作用）
``gain_ms``           报告承诺的模拟圈速提升（中位数 / p10 / p90；已按
                      单步 80ms、单轮 120ms 的可信上限记账）
``mean_sim_moves``    模型自己采纳的步数（= 它让用户多拧的螺丝数，门禁看这个）
``changed_params``    整条建议的改动参数个数（含规则层，仅作观察口径 ——
                      规则层的 ``_CHANGE_BUDGET`` 就占掉 8~12 项）
``violations``        越界 / 超 ``max_delta`` / 胎压不对称 / 无输入改车（必须 0）
====================  ==================================================

用法::

    python scripts/eval_usage_flow.py                  # 全量 24 赛道 × 5 场景
    python scripts/eval_usage_flow.py --quick          # 3 赛道快跑（CI 冒烟用）
    python scripts/eval_usage_flow.py --strict         # 门禁不通过退出码 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup  # noqa: E402
from setup_tuner.domain.track import ALL_TRACKS  # noqa: E402
from setup_tuner.engine.engine import generate_suggestion  # noqa: E402
from setup_tuner.engine.setup_sim import get_setup_sim  # noqa: E402

DEFAULT_JSON = ROOT / "setup_tuner" / "resources" / "models" / "usage_flow_eval.json"
DEFAULT_MD = ROOT / "setup_tuner" / "resources" / "models" / "usage_flow_eval.md"

#: 真实车手反馈场景（症状, 强度 1~3）。覆盖入弯/弯中/出弯/全局四类，
#: 以及「只有 1 条反馈」与「多条打架」两种真实录入习惯。
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "entry_understeer",
        "label": "入弯推头 + 转向不灵敏",
        "symptoms": [("understeer", 3), ("turnin_unresponsive", 2)],
    },
    {
        "id": "exit_traction",
        "label": "出弯打滑 + 弯中不能稳定加速",
        "symptoms": [("exit_wheelspin", 3), ("midcorner_traction", 2)],
    },
    {
        "id": "high_speed",
        "label": "高速不稳 + 刹车中车身不稳",
        "symptoms": [("high_speed_instability", 2), ("brake_instability", 2)],
    },
    {
        "id": "global_mixed",
        "label": "胎温过高 + 刮底 + 直道速度低",
        "symptoms": [("tyre_overheat", 2), ("bottoming", 2), ("straight_slow", 2)],
    },
    {
        "id": "single_symptom",
        "label": "单条反馈（弯中推头，强度 2）",
        "symptoms": [("midcorner_understeer", 2)],
    },
)

#: 门禁（``--strict`` 时生效）。数字口径见模块文档。
GATES: dict[str, tuple[str, float]] = {
    "nn_active_rate": (">=", 1.0),      # 24/24 赛道都必须能跑模型
    "accept_rate": (">=", 0.90),        # 至少采纳一步的比例
    "median_gain_ms": (">=", 10.0),     # 中位模拟收益要有实际意义（>10ms）
    # 模型自己加了几步 —— 这才是「模型贡献的改动处数」。
    # 口径修正（v1.6.13）：此前卡的是 ``mean_changed_params``（整条建议的
    # 改动总条数），但实测规则层（holistic._CHANGE_BUDGET=12）就占掉 8~12 项，
    # 模型只加 1~4 步 —— 拿总数当门禁等于用规则层的输出否决模型，且会让
    # 每一轮重训都红。模型该管的是它自己多拧了几颗螺丝（_MAX_MOVES=3）。
    "mean_sim_moves": ("<=", 3.0),
    # 收益上限只做「量级兜底」：sim_optimizer 已按单步 80ms / 单轮 120ms
    # 记账，这里留 200ms 兜住「回退或换模型后上限失效」。
    "p90_gain_ms": ("<=", 200.0),
    "violations": ("<=", 0),            # 越界/不对称/无输入改车：绝对不允许
    "zero_input_moves": ("<=", 0),      # 无诊断输入不得改车
}

#: 胎压左右必须一致的成对参数
_PRESSURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("front_left_tyre_pressure", "front_right_tyre_pressure"),
    ("rear_left_tyre_pressure", "rear_right_tyre_pressure"),
)


def _violations(
    track_id: str, scenario: str, result: dict[str, Any],
    current_setup: dict[str, float],
) -> list[str]:
    """检查一次建议是否违反可用性契约（用户的「能不能照着拧」底线）。"""
    out: list[str] = []
    delta = result.get("setup_delta") or {}
    for spec in ALL_SETUP_FIELDS:
        d = float(delta.get(spec.name, 0.0))
        final = float(current_setup.get(spec.name, spec.default)) + d
        if abs(d) > spec.max_delta + 1e-6:
            out.append(
                f"{track_id}/{scenario}: {spec.name} 改动 {d:+.3f} "
                f"超 max_delta {spec.max_delta}"
            )
        if final < spec.min_val - 1e-6 or final > spec.max_val + 1e-6:
            out.append(
                f"{track_id}/{scenario}: {spec.name} 终值 {final:g} 越界 "
                f"[{spec.min_val:g}, {spec.max_val:g}]"
            )
        # 档位对齐：终值必须落在真实档位上（否则用户设不出来）
        steps = (final - spec.min_val) / spec.step
        if abs(steps - round(steps)) > 1e-6:
            out.append(f"{track_id}/{scenario}: {spec.name} 终值 {final:g} 未对齐档位 {spec.step:g}")
    for left, right in _PRESSURE_PAIRS:
        dl = float(delta.get(left, 0.0))
        dr = float(delta.get(right, 0.0))
        if abs(dl - dr) > 1e-6:
            out.append(f"{track_id}/{scenario}: 胎压不对称 {left}={dl:+.3f} / {right}={dr:+.3f}")
    return out


def _run_one(
    track_id: str, symptoms: list[tuple[str, int]], current_setup: dict[str, float],
) -> dict[str, Any]:
    """走一次完整建议生成（与 /suggest 同一入口）。"""
    result = generate_suggestion(
        symptoms, current_setup, track_id,
        telemetry=None, model_type="nn", style_vector=None, feedbacks=None,
    )
    sim = (result.get("holistic") or {}).get("simulation") or {}
    delta = result.get("setup_delta") or {}
    changed = sum(
        1 for spec in ALL_SETUP_FIELDS if abs(float(delta.get(spec.name, 0.0))) > 1e-9
    )
    return {
        "model_type": result.get("model_type"),
        "nn_available": bool(result.get("nn_available")),
        "sim": sim,
        "changed_params": changed,
        "delta": {k: round(float(v), 4) for k, v in delta.items()},
        "summary": result.get("summary"),
    }


def evaluate(tracks: list[str], scenarios: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """跑全矩阵，返回机器可读结果。"""
    current_setup = CarSetup.default().to_dict()
    sim_model = get_setup_sim()

    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    gains: list[float] = []
    n_active = 0
    n_accept = 0

    for tid in tracks:
        for sc in scenarios:
            r = _run_one(tid, sc["symptoms"], current_setup)
            rows.append({
                "track_id": tid,
                "scenario": sc["id"],
                "model_type": r["model_type"],
                "nn_available": r["nn_available"],
                "reason": (r["sim"] or {}).get("reason"),
                "iterations": (r["sim"] or {}).get("iterations"),
                "accepted": (r["sim"] or {}).get("accepted"),
                "gain_ms": (r["sim"] or {}).get("gain_ms"),
                "changed_params": r["changed_params"],
            })
            if r["nn_available"]:
                n_active += 1
                acc = int((r["sim"] or {}).get("accepted") or 0)
                if acc > 0:
                    n_accept += 1
                g = (r["sim"] or {}).get("gain_ms")
                if isinstance(g, (int, float)):
                    gains.append(float(g))
            violations.extend(_violations(tid, sc["id"], r, current_setup))

    # ── 契约：无诊断输入不得改车（用户点了生成但没有反馈/遥测）──────────────
    zero_input_moves = 0
    zero_input_detail: list[str] = []
    for tid in tracks:
        r = _run_one(tid, [], current_setup)
        changed = r["changed_params"]
        zero_input_moves += changed
        if changed:
            zero_input_detail.append(f"{tid}: 无输入却改了 {changed} 个参数")

    gains_sorted = sorted(gains)

    def _pct(p: float) -> float:
        if not gains_sorted:
            return 0.0
        k = min(len(gains_sorted) - 1, max(0, int(round(p * (len(gains_sorted) - 1)))))
        return round(gains_sorted[k], 2)

    n_runs = len(rows)
    metrics: dict[str, Any] = {
        "n_tracks": len(tracks),
        "n_scenarios": len(scenarios),
        "n_runs": n_runs,
        "nn_active_rate": round(n_active / n_runs, 4) if n_runs else 0.0,
        "accept_rate": round(n_accept / max(1, n_active), 4) if n_active else 0.0,
        "median_gain_ms": _pct(0.50),
        "p10_gain_ms": _pct(0.10),
        "p90_gain_ms": _pct(0.90),
        "mean_gain_ms": round(sum(gains) / len(gains), 2) if gains else 0.0,
        "mean_changed_params": round(
            sum(r["changed_params"] for r in rows) / max(1, n_runs), 2
        ),
        # 模型自己采纳的步数（= 它让用户多拧的螺丝数）；门禁看这个。
        "mean_sim_moves": round(
            sum(int(r["accepted"] or 0) for r in rows) / max(1, n_runs), 2
        ),
        "violations": len(violations),
        "zero_input_moves": zero_input_moves,
        "model_tracks": len(sim_model.track_ids),
        "model_desc": sim_model.describe(),
        "tracks_without_model": [t for t in tracks if not sim_model.covers(t)],
    }
    return {
        "schema": "f1opt-usage-flow-eval/1",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "entry": "engine.generate_suggestion（与 POST /api/v1/suggest 同路径）",
        "metrics": metrics,
        "gate": _check_gates(metrics),
        "violation_details": violations[:40],
        "zero_input_details": zero_input_detail[:20],
        "rows": rows,
    }


def _check_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    for key, (op, threshold) in GATES.items():
        value = metrics.get(key)
        ok = value is not None and (
            (value <= threshold) if op == "<=" else (value >= threshold)
        )
        checks[key] = {"value": value, "threshold": f"{op}{threshold}", "ok": bool(ok)}
        passed = passed and bool(ok)
    return {"passed": passed, "checks": checks}


def render_md(result: dict[str, Any]) -> str:
    m = result["metrics"]
    g = result["gate"]
    lines = [
        "# 调教模型「使用流程」评测",
        "",
        f"- 入口：{result['entry']}",
        f"- 评测时间：{result['evaluated_at']}",
        f"- 矩阵：{m['n_tracks']} 赛道 × {m['n_scenarios']} 场景 = {m['n_runs']} 次生成",
        f"- 模型：{m['model_desc']}",
        "",
        "## 门禁",
        "",
        "| 指标 | 值 | 门槛 | 结论 |",
        "|---|---|---|---|",
    ]
    for key, chk in g["checks"].items():
        lines.append(
            f"| `{key}` | {chk['value']} | {chk['threshold']} | "
            f"{'PASS' if chk['ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 用户实际拿到的东西",
        "",
        f"- 模型生效率：**{m['nn_active_rate']:.0%}**"
        f"（{m['n_tracks']} 赛道中 {m['n_tracks'] - len(m['tracks_without_model'])} 条覆盖模型）",
        f"- 至少采纳 1 步改动的比例：**{m['accept_rate']:.0%}**",
        f"- 模拟圈速提升：中位 **{m['median_gain_ms']:.1f} ms**"
        f"（p10 {m['p10_gain_ms']:.1f} / p90 {m['p90_gain_ms']:.1f} ms）"
        f" —— 已按单步 80ms / 单轮 120ms 的可信上限记账",
        f"- 模型自己采纳的步数：平均 **{m['mean_sim_moves']:.1f} 步**"
        f"（它让用户多拧的螺丝数，上限 3）",
        f"- 整条建议的改动参数：平均 {m['mean_changed_params']:.1f} 个"
        f"（含规则层改动预算，仅作观察）",
        f"- 越界/超幅/胎压不对称：{m['violations']} 条",
        f"- 无诊断输入却改车：{m['zero_input_moves']} 处",
        "",
    ]
    if m["tracks_without_model"]:
        lines += [
            "## 未覆盖模型（用户选到这些赛道时面板显示「未启用」）",
            "",
            "、".join(f"`{t}`" for t in m["tracks_without_model"]),
            "",
        ]
    if result["violation_details"]:
        lines += ["## 违规明细（前 40 条）", ""]
        lines += [f"- {v}" for v in result["violation_details"]]
        lines.append("")
    lines += [
        "## 说明",
        "",
        "- 本表衡量的是**使用流程是否通**（模型跑没跑、改动落没落地、报告是否自洽），"
        "不衡量绝对圈速精度；精度见 `setup_sim_eval.md`。",
        "- 场景取自 UI 里真实存在的症状项与强度档位（1~3），不是构造数据。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按系统使用流程评测调教模型")
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--md", default=str(DEFAULT_MD))
    ap.add_argument("--tracks", default="", help="逗号分隔；缺省=全部 24 赛道")
    ap.add_argument("--quick", action="store_true", help="只在 3 条赛道跑（CI 冒烟）")
    ap.add_argument("--strict", action="store_true", help="门禁不通过则退出码 1")
    args = ap.parse_args(argv)

    all_ids = [t.track_id for t in ALL_TRACKS]
    if args.tracks:
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    elif args.quick:
        # 一条遥测覆盖、一条后段新增、一条无遥测 —— 三类都验到
        tracks = [t for t in ("monza", "austin", "las_vegas") if t in all_ids]
    else:
        tracks = all_ids

    t0 = time.time()
    result = evaluate(tracks, SCENARIOS)
    elapsed = time.time() - t0

    m = result["metrics"]
    print(f"使用流程评测完成（{elapsed:.1f}s，{len(tracks)} 赛道 × {len(SCENARIOS)} 场景）")
    print(f"  模型生效率   : {m['nn_active_rate']:.1%}"
          f"（未覆盖 {len(m['tracks_without_model'])} 条：{m['tracks_without_model']}）")
    print(f"  采纳率       : {m['accept_rate']:.1%}（至少采纳 1 步）")
    print(f"  模拟收益     : 中位 {m['median_gain_ms']:.1f} ms | "
          f"p10 {m['p10_gain_ms']:.1f} | p90 {m['p90_gain_ms']:.1f}")
    print(f"  平均改动参数 : {m['mean_changed_params']:.1f}（含规则层）| "
          f"模型采纳 {m['mean_sim_moves']:.1f} 步")
    print(f"  可用性违规   : {m['violations']} 条 | 无输入改车：{m['zero_input_moves']} 处")
    print(f"  门禁         : {'PASS' if result['gate']['passed'] else 'FAIL'}")
    for key, chk in result["gate"]["checks"].items():
        if not chk["ok"]:
            print(f"    - {key} = {chk['value']} 未达 {chk['threshold']}")

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    out_md = Path(args.md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(result), encoding="utf-8")
    print(f"写出: {out_json} / {out_md}")

    if args.strict and not result["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
