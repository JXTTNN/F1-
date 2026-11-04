"""按「使用流程」评测车手风格定向是否真的改变了建议（端到端）。

为什么单独一个脚本
------------------
``eval_usage_flow.py`` 量的是「模型有没有被用上、收益是否可信」，
``eval_setup_sim.py`` 量的是「模型对物理仿真器的保真度」。
两者都**不看车手**：同样的赛道与反馈，给任何人同一套车。

而产品承诺的是「按车手驾驶风格针对性优化」。这条承诺有两个失效模式，
在真实数据上都出现过：

1. **风格层空转**：``gain_for_style`` 只把 ``Dx×C`` 的幅度乘上 0.85~1.25，
   一旦参数顶到 ``max_delta``（实测 20 项里 6 项常年饱和）或诊断本来没动
   这个参数（乘 0 还是 0），风格倍数就完全失效 —— 6 条风格规则里 4 条在
   24 站上零影响，用户看到的是「选了风格，建议一字不差」。
2. **风格层乱花圈速**：为了"顺手"把车调得慢几十毫秒，还把它算成收益。

因此本脚本对「24 赛道 × 5 场景 × 3 种典型开法」跑全矩阵，直接调用
``engine.generate_suggestion``（与 ``POST /api/v1/suggest`` 同一条代码路径），
把上面两条变成可断言的数字：

====================  ==================================================
``effect_rate``       换了开法后，最终建议（不是中间量）确实变了的比例
``survive_rate``      风格动作至少活过模拟优化一处的比例
``cost_ms``           风格动作净代价（中位 / p90；正=为顺手让出的圈速）
``dim_coverage``      6 条风格规则里有几条真的能落地（诚实统计，不藏）
``violations``        越界 / 超 max_delta / 档位未对齐 / 胎压不对称
``silent_moves``      无诊断输入却因风格改了车（必须 0）
====================  ==================================================

用法::

    python scripts/eval_style_targeting.py            # 全量
    python scripts/eval_style_targeting.py --quick    # 3 赛道冒烟
    python scripts/eval_style_targeting.py --strict   # 门禁不过则退出码 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_usage_flow import SCENARIOS, _violations  # noqa: E402

from setup_tuner.domain.setup import CarSetup  # noqa: E402
from setup_tuner.domain.style_coefficients import (  # noqa: E402
    STYLE_VECTOR_LEN,
    TRIM_MIN_DEVIATION,
    describe_style,
    style_trims,
)
from setup_tuner.domain.track import ALL_TRACKS  # noqa: E402
from setup_tuner.engine.engine import generate_suggestion  # noqa: E402
from setup_tuner.engine.setup_sim import get_setup_sim  # noqa: E402

DEFAULT_JSON = ROOT / "setup_tuner" / "resources" / "models" / "style_flow_eval.json"
DEFAULT_MD = ROOT / "setup_tuner" / "resources" / "models" / "style_flow_eval.md"

#: 中性基线：所有维度 0.5（没见过足够样本时的退化行为）
NEUTRAL = [0.5] * STYLE_VECTOR_LEN

#: 三种"典型开法"（维度顺序见 telemetry.style_extractor.STYLE_DIMS）
ARCHETYPES: dict[str, list[float]] = {
    "neutral": NEUTRAL,
    # 粗糙型：转向修正多、胎温离散大、循迹刹车多 → 需要容错与制动余量
    "rough": [0.3, 0.2, 0.3, 0.3, 0.8, 0.3, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5],
    # 激进型：攻弯凶、给油早、滑移多 → 需要入弯响应与牵引
    "aggressive": [0.9, 0.8, 0.9, 0.8, 0.3, 0.9, 0.7, 0.9, 0.5, 0.5, 0.5, 0.5],
}

#: 门禁
GATES: dict[str, str] = {
    "effect_rate": ">=0.85",
    "survive_rate": ">=0.85",
    "cost_p90_ms": "<=25.0",
    "cost_max_ms": "<=30.0",
    "violations": "<=0",
    "silent_moves": "<=0",
    "dim_coverage": ">=2",
}


def _nonzero(delta: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in delta.items() if abs(float(v)) > 1e-9}


def _run(track_id: str, symptoms: list[tuple[str, int]],
         style: list[float] | None) -> dict[str, Any]:
    result = generate_suggestion(
        symptoms, CarSetup.default().to_dict(), track_id,
        telemetry=None, model_type="nn", style_vector=style, feedbacks=None,
    )
    holistic = result.get("holistic") or {}
    style_block = holistic.get("style_targeting") or {}
    sim = holistic.get("simulation") or {}
    return {
        "delta": _nonzero(result.get("setup_delta") or {}),
        "model_type": result.get("model_type"),
        "style": style_block,
        "sim": sim,
        "raw": result,
    }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return round(ordered[k], 2)


def evaluate(tracks: list[str],
             scenarios: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    current_setup = CarSetup.default().to_dict()
    sim = get_setup_sim()
    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    per_style: dict[str, dict[str, Any]] = {}

    for name in ("rough", "aggressive"):
        per_style[name] = {
            "runs": 0, "effect": 0, "applied": 0, "survived": 0,
            "costs": [], "trim_hits": {}, "trim_dims": set(), "dead_reasons": {},
        }

    for track_id in tracks:
        for scenario in scenarios:
            base = _run(track_id, scenario["symptoms"], NEUTRAL)
            violations.extend(
                _violations(track_id, f"{scenario['id']}/neutral", base["raw"],
                            current_setup)
            )
            for name in ("rough", "aggressive"):
                styled = _run(track_id, scenario["symptoms"], ARCHETYPES[name])
                stats = per_style[name]
                stats["runs"] += 1
                if styled["delta"] != base["delta"]:
                    stats["effect"] += 1
                block = styled["style"] or {}
                applied = len(block.get("applied") or [])
                survived = int(block.get("survived") or 0)
                if applied:
                    stats["applied"] += 1
                if survived:
                    stats["survived"] += 1
                cost = block.get("cost_ms")
                if isinstance(cost, (int, float)) and applied:
                    stats["costs"].append(float(cost))
                for entry in block.get("applied") or []:
                    if entry.get("survived"):
                        stats["trim_dims"].add(entry.get("dim_name"))
                        for param in entry.get("params") or {}:
                            stats["trim_hits"][param] = (
                                stats["trim_hits"].get(param, 0) + 1
                            )
                for entry in block.get("rejected") or []:
                    key = str(entry.get("blocked_by") or "未知")
                    stats["dead_reasons"][key] = stats["dead_reasons"].get(key, 0) + 1
                violations.extend(
                    _violations(track_id, f"{scenario['id']}/{name}",
                                styled["raw"], current_setup)
                )
                rows.append({
                    "track_id": track_id,
                    "scenario": scenario["id"],
                    "style": name,
                    "changed_params": len(styled["delta"]),
                    "applied": applied,
                    "survived": survived,
                    "cost_ms": cost,
                    "model_type": styled["model_type"],
                    "gain_ms": (styled["sim"] or {}).get("gain_ms"),
                    "notes": block.get("notes"),
                })

    # 契约：无诊断输入时风格不得改车
    silent_moves = 0
    silent_detail: list[str] = []
    for track_id in tracks:
        for name in ("rough", "aggressive"):
            r = _run(track_id, [], ARCHETYPES[name])
            if r["delta"]:
                silent_moves += len(r["delta"])
                silent_detail.append(f"{track_id}/{name}: 无输入却改了 {len(r['delta'])} 项")
            block = r["style"] or {}
            if block.get("enabled"):
                silent_moves += 1
                silent_detail.append(f"{track_id}/{name}: 无输入却启用了风格定向")

    metrics: dict[str, Any] = {"n_tracks": len(tracks), "n_scenarios": len(scenarios)}
    for name, stats in per_style.items():
        runs = max(1, stats["runs"])
        # 每种开法覆盖到的风格维度（按 style_trims 触发的维度统计）
        dims = {
            t.dim for t in style_trims(ARCHETYPES[name])
        }
        metrics[name] = {
            "effect_rate": round(stats["effect"] / runs, 4),
            "survive_rate": round(stats["survived"] / runs, 4),
            "applied_rate": round(stats["applied"] / runs, 4),
            "cost_median_ms": _pct(stats["costs"], 0.5),
            "cost_p90_ms": _pct(stats["costs"], 0.9),
            "cost_max_ms": round(max(stats["costs"]), 2) if stats["costs"] else 0.0,
            "trim_hits": dict(sorted(stats["trim_hits"].items(),
                                    key=lambda kv: -kv[1])),
            "rejected_reasons": dict(sorted(stats["dead_reasons"].items(),
                                            key=lambda kv: -kv[1])),
            "dims_triggered": sorted(dims),
            "profile": describe_style(ARCHETYPES[name]),
        }
        metrics[name]["dim_coverage"] = len(stats["trim_dims"])
        metrics[name]["dims_landed"] = sorted(stats["trim_dims"])

    metrics["violations"] = len(violations)
    metrics["violation_sample"] = violations[:10]
    metrics["silent_moves"] = silent_moves
    metrics["silent_detail"] = silent_detail[:5]
    metrics["trims_defined"] = len(style_trims(ARCHETYPES["aggressive"])) + len(
        style_trims(ARCHETYPES["rough"]))
    metrics["min_deviation"] = TRIM_MIN_DEVIATION
    metrics["model"] = sim.describe()
    return {"rows": rows, "metrics": metrics}


def _check_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in ("rough", "aggressive"):
        block = metrics[name]
        checks[f"{name}.effect_rate"] = {
            "value": block["effect_rate"], "threshold": ">=0.85",
            "ok": block["effect_rate"] >= 0.85,
        }
        checks[f"{name}.survive_rate"] = {
            "value": block["survive_rate"], "threshold": ">=0.85",
            "ok": block["survive_rate"] >= 0.85,
        }
        checks[f"{name}.cost_p90_ms"] = {
            "value": block["cost_p90_ms"], "threshold": "<=25.0",
            "ok": block["cost_p90_ms"] <= 25.0,
        }
        checks[f"{name}.cost_max_ms"] = {
            "value": block["cost_max_ms"], "threshold": "<=30.0",
            "ok": block["cost_max_ms"] <= 30.0,
        }
        checks[f"{name}.dim_coverage"] = {
            "value": block["dim_coverage"], "threshold": ">=2",
            "ok": block["dim_coverage"] >= 2,
        }
    checks["violations"] = {
        "value": metrics["violations"], "threshold": "<=0",
        "ok": metrics["violations"] <= 0,
    }
    checks["silent_moves"] = {
        "value": metrics["silent_moves"], "threshold": "<=0",
        "ok": metrics["silent_moves"] <= 0,
    }
    return {"checks": checks, "passed": all(c["ok"] for c in checks.values())}


def render_md(result: dict[str, Any]) -> str:
    m = result["metrics"]
    gate = result["gate"]
    lines = [
        "# 车手风格定向评测（24 赛道 × 5 场景 × 2 种典型开法）",
        "",
        "- 生成时间：" + result["evaluated_at"],
        f"- 模型：{m['model']}",
        f"- 覆盖：{m['n_tracks']} 站 × {m['n_scenarios']} 场景",
        f"- 门禁：{'全部通过' if gate['passed'] else '**未通过**'}",
        "",
        "## 换了开法，建议是否真的变了",
        "",
        "| 开法 | 生效比例 | 至少一处存活 | 采纳比例 | 代价中位 | 代价 p90 | 代价最大 | 落地的风格维度 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("rough", "aggressive"):
        b = m[name]
        lines.append(
            f"| {name} | {b['effect_rate']:.1%} | {b['survive_rate']:.1%} "
            f"| {b['applied_rate']:.1%} | {b['cost_median_ms']:+.1f} ms "
            f"| {b['cost_p90_ms']:+.1f} ms | {b['cost_max_ms']:+.1f} ms "
            f"| {b['dim_coverage']} 维 |"
        )
    lines += [
        "",
        "## 落在哪些参数上（存活次数）",
        "",
    ]
    for name in ("rough", "aggressive"):
        hits = m[name]["trim_hits"]
        lines.append(f"- **{name}**：" + (
            "、".join(f"`{p}` ×{n}" for p, n in hits.items()) or "（无）"
        ))
    lines += [
        "",
        "## 被拒绝的动作与原因",
        "",
    ]
    for name in ("rough", "aggressive"):
        reasons = m[name]["rejected_reasons"]
        if not reasons:
            lines.append(f"- **{name}**：（无）")
            continue
        lines.append(f"- **{name}**：")
        for reason, count in reasons.items():
            lines.append(f"  - {reason} ×{count}")
    lines += [
        "",
        "## 契约检查",
        "",
        f"- 越界/超 max_delta/档位未对齐/胎压不对称：**{m['violations']}** 次",
        f"- 无诊断输入却因风格改车：**{m['silent_moves']}** 次",
        "",
        "## 门禁明细",
        "",
        "| 检查 | 值 | 门槛 | 结果 |",
        "|---|---|---|---|",
    ]
    for key, chk in gate["checks"].items():
        lines.append(
            f"| {key} | {chk['value']} | {chk['threshold']} "
            f"| {'PASS' if chk['ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        "> 口径说明：`effect_rate` 比的是**最终建议**（含模拟优化），不是中间量；",
        "> `cost_ms` 用调教性能 NN 的差分校准预测，正数表示「为了让车更顺手让出的圈速」，",
        "> 单条动作超过 ±30ms 的预测视为量级不可信（代理模型误差），不采纳。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="车手风格定向端到端评测")
    ap.add_argument("--quick", action="store_true", help="3 赛道冒烟")
    ap.add_argument("--strict", action="store_true", help="门禁不过退出码 1")
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--md", default=str(DEFAULT_MD))
    ap.add_argument("--no-write", action="store_true", help="只打印，不写文件")
    args = ap.parse_args(argv)

    tracks = [t.track_id for t in ALL_TRACKS]
    scenarios = SCENARIOS
    if args.quick:
        tracks = tracks[:3]
        scenarios = SCENARIOS[:2]

    import datetime as _dt
    result = evaluate(tracks, scenarios)
    result["evaluated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    result["gate"] = _check_gates(result["metrics"])

    if not args.no_write:
        Path(args.json).write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        Path(args.md).write_text(render_md(result), encoding="utf-8")

    m = result["metrics"]
    print(f"赛道 {m['n_tracks']} × 场景 {m['n_scenarios']}")
    for name in ("rough", "aggressive"):
        b = m[name]
        print(f"  {name:11s} effect={b['effect_rate']:.1%} "
              f"survive={b['survive_rate']:.1%} "
              f"cost p90={b['cost_p90_ms']:+.1f}ms max={b['cost_max_ms']:+.1f}ms "
              f"dims={b['dim_coverage']}")
    print(f"  violations={m['violations']} silent={m['silent_moves']}")
    print(f"  gate: {'PASS' if result['gate']['passed'] else 'FAIL'}")
    if not result["gate"]["passed"]:
        for key, chk in result["gate"]["checks"].items():
            if not chk["ok"]:
                print(f"    FAIL {key}: {chk['value']} (要求 {chk['threshold']})")
    return 0 if (result["gate"]["passed"] or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
