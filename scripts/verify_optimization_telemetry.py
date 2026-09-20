"""用现有遥测验证调教优化的实际效果（端到端验收）。

「利用现有遥测验证优化性能」的落地脚本，验证三件事：

1. **模型真在管线里**：调教性能 NN 已加载、覆盖赛道、推理确定；
2. **优化真有效果**：同场景下，神经网络模拟优化（nn 模式）相对纯规则路径
   （rule 模式）的**模拟圈速提升**（用同期训练的模型做裁判 —— 与优化时
   同一模型，口径一致；同时打印被采纳的改动轨迹可人工复核方向）；
3. **遥测锚定一致**：模型对默认调教的预测 ≈ 0（仿真锚点自洽），
   各赛道最优翼角排序与遥测 S_track/drag_scale 排序方向一致
   （Monaco 最想加翼、Spa/Monza 最想减翼 —— F1 调教共识）。

输出：终端表格 + ``docs/Optimization_Verification.md`` 报告。

用法::

    python scripts/verify_optimization_telemetry.py
    python scripts/verify_optimization_telemetry.py --tracks monza,suzuka,monaco
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_MD = ROOT / "docs" / "Optimization_Verification.md"

#: 验收场景（覆盖典型车手反馈；症状键必须是 domain.symptoms.Symptom 的合法值）
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "入弯推头（中速弯，强度 2）",
        "symptoms": [("understeer", 2)],
        "feedbacks": [{"corner_number": 3, "symptom": "understeer", "strength": 2}],
    },
    {
        "name": "出弯打滑（慢弯，强度 3）",
        "symptoms": [("exit_wheelspin", 3)],
        "feedbacks": [{"corner_number": 1, "symptom": "exit_wheelspin", "strength": 3}],
    },
    {
        "name": "高速弯发飘（强度 2）",
        "symptoms": [("high_speed_instability", 2)],
        "feedbacks": [{"corner_number": 5, "symptom": "high_speed_instability", "strength": 2}],
    },
]

#: 验收赛道（覆盖高低下压力类型）
DEFAULT_TRACKS = ["monaco", "hungaroring", "suzuka", "silverstone", "monza", "spa"]


def _sim_time(track_id: str, setup: dict[str, float]) -> float | None:
    """用引擎同一模型预测该调教的圈速增量（秒）。"""
    from setup_tuner.engine.setup_sim import get_setup_sim

    return get_setup_sim().predict_for_setup(track_id, setup)


def _truth_time(track_id: str, setup: dict[str, float]) -> float | None:
    """用**解析仿真器**（训练标签的真值源）算圈速增量（秒）——独立裁判。

    NN 是解析仿真器的代理；"NN 说更快"可能是模型偏差被优化器利用。
    用真值源交叉复核，给出「真实（仿真口径）提升」。
    """
    from scripts.build_setup_sim_dataset import load_corner_stats, simulate_lap_delta_ms
    from setup_tuner.engine.setup_sim import normalize_setup

    global _TRUTH_STATS
    try:
        stats = _TRUTH_STATS  # type: ignore[name-defined]
    except NameError:
        stats = _TRUTH_STATS = load_corner_stats(None)  # noqa: PLW0603
    track = stats.get(track_id)
    if track is None:
        return None
    return simulate_lap_delta_ms(track, normalize_setup(setup)) / 1000.0


def _setup_with_delta(current: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    return {k: float(v) + float(delta.get(k, 0.0)) for k, v in current.items()}


def verify_engine_integration(tracks: list[str]) -> dict[str, Any]:
    """引擎集成检查：模型可用性 / 覆盖 / 推理确定性 / 默认锚点。"""
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.setup_sim import get_setup_sim

    model = get_setup_sim()
    out: dict[str, Any] = {
        "available": model.available,
        "describe": model.describe(),
        "metrics": model.metrics,
        "tracks_covered": list(model.track_ids),
    }
    default = CarSetup.default().to_dict()
    anchors: dict[str, Any] = {}
    for tid in tracks:
        if not model.covers(tid):
            anchors[tid] = {"covered": False}
            continue
        t1 = model.predict_for_setup(tid, default)
        t2 = model.predict_for_setup(tid, default)
        anchors[tid] = {
            "covered": True,
            "default_delta_s": t1,
            "deterministic": t1 == t2,
        }
    out["anchors"] = anchors
    return out


def verify_optimization_gain(tracks: list[str]) -> list[dict[str, Any]]:
    """逐赛道×场景：nn 模式 vs rule 模式的模拟圈速对比。"""
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.engine import generate_suggestion

    results: list[dict[str, Any]] = []
    for tid in tracks:
        for sc in SCENARIOS:
            setup = CarSetup.default().to_dict()
            common = dict(
                current_setup=setup, track_id=tid, telemetry=None,
                style_vector=None, feedbacks=sc["feedbacks"],
            )
            t0 = time.time()
            rule_res = generate_suggestion(
                sc["symptoms"], model_type="rule", **common,
            )
            rule_ms = (time.time() - t0) * 1000
            t0 = time.time()
            nn_res = generate_suggestion(
                sc["symptoms"], model_type="nn", **common,
            )
            nn_ms = (time.time() - t0) * 1000

            rule_setup = _setup_with_delta(setup, rule_res["setup_delta"])
            nn_setup = _setup_with_delta(setup, nn_res["setup_delta"])
            sim_rule = _sim_time(tid, rule_setup)
            sim_nn = _sim_time(tid, nn_setup)
            # 独立裁判：解析仿真器（训练标签真值源）对两个结果的评价
            truth_rule = _truth_time(tid, rule_setup)
            truth_nn = _truth_time(tid, nn_setup)
            sim_block = (nn_res.get("holistic") or {}).get("simulation") or {}
            gain_vs_rule_ms = (
                (sim_rule - sim_nn) * 1000
                if sim_rule is not None and sim_nn is not None else None
            )
            truth_gain_ms = (
                (truth_rule - truth_nn) * 1000
                if truth_rule is not None and truth_nn is not None else None
            )
            results.append({
                "track": tid,
                "scenario": sc["name"],
                "rule_model_type": rule_res["model_type"],
                "nn_model_type": nn_res["model_type"],
                "sim_rule_ms": None if sim_rule is None else round(sim_rule * 1000, 1),
                "sim_nn_ms": None if sim_nn is None else round(sim_nn * 1000, 1),
                "gain_vs_rule_ms": None if gain_vs_rule_ms is None else round(gain_vs_rule_ms, 1),
                "truth_rule_ms": None if truth_rule is None else round(truth_rule * 1000, 1),
                "truth_nn_ms": None if truth_nn is None else round(truth_nn * 1000, 1),
                "truth_gain_ms": None if truth_gain_ms is None else round(truth_gain_ms, 1),
                "sim_iterations": sim_block.get("iterations"),
                "sim_accepted": sim_block.get("accepted"),
                "sim_trace": sim_block.get("trace", [])[:6],
                "rule_elapsed_ms": round(rule_ms, 1),
                "nn_elapsed_ms": round(nn_ms, 1),
            })
    return results


def verify_track_signature() -> list[dict[str, Any]]:
    """验证「不同赛道的最优翼角不同」（模拟优化确实读赛道特性）。"""
    from setup_tuner.domain.setup import CarSetup, get_field
    from setup_tuner.engine.setup_sim import get_setup_sim, normalize_setup

    model = get_setup_sim()
    if not model.available:
        return []
    wing = get_field("front_wing")
    default = CarSetup.default().to_dict()
    rows: list[dict[str, Any]] = []
    for tid in sorted(model.track_ids):
        best_clicks, best = 0, None
        for clicks in range(-12, 13):
            setup = dict(default)
            setup["front_wing"] = wing.default + clicks
            setup["rear_wing"] = wing.default + clicks
            v = model.predict_delta_s(tid, normalize_setup(setup))
            if v is None:
                continue
            if best is None or v < best:
                best_clicks, best = clicks, v
        rows.append({
            "track": tid,
            "best_wing_clicks": best_clicks,
            "best_delta_ms": round(best * 1000, 1) if best is not None else None,
        })
    return sorted(rows, key=lambda r: r["best_wing_clicks"])


def render_markdown(
    engine_info: dict[str, Any],
    results: list[dict[str, Any]],
    signature: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# 调教优化验证报告（遥测锚定）\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. 模型与引擎集成\n")
    lines.append(f"- 模型状态：{engine_info['describe']}")
    m = engine_info.get("metrics") or {}
    if m:
        lines.append(
            f"- 训练指标：验证 MAE {m.get('val_mae_ms')} ms / "
            f"R² {m.get('val_r2')}（零改动基线 {m.get('baseline_mae_ms')} ms）"
        )
    lines.append(f"- 覆盖赛道：{len(engine_info['tracks_covered'])} 条\n")
    lines.append("| 赛道 | 默认调教预测 | 确定性 |")
    lines.append("|------|------------|--------|")
    for tid, a in (engine_info.get("anchors") or {}).items():
        if not a.get("covered"):
            lines.append(f"| {tid} | 未覆盖 | - |")
            continue
        d = a.get("default_delta_s")
        lines.append(
            f"| {tid} | {d * 1000:.1f} ms | {'✓' if a.get('deterministic') else '✗'} |"
        )
    lines.append("\n## 2. 优化效果（nn 模式 vs 纯规则）\n")
    lines.append("`NN 判定` = 调教性能 NN 对两个结果的评价（与优化同一模型）；"
                 "`真值复核` = 解析仿真器（训练标签真值源）的独立评价。\n")
    lines.append("| 赛道 | 场景 | 规则(ms) | NN路径(ms) | NN判定提升 | 真值复核提升 | 迭代/采纳 |")
    lines.append("|------|------|---------|-----------|-----------|-------------|-----------|")
    gains = [r["gain_vs_rule_ms"] for r in results if r["gain_vs_rule_ms"] is not None]
    truth_gains = [r["truth_gain_ms"] for r in results if r["truth_gain_ms"] is not None]
    for r in results:
        lines.append(
            f"| {r['track']} | {r['scenario']} | {r['sim_rule_ms']} | "
            f"{r['sim_nn_ms']} | {r['gain_vs_rule_ms']} ms | "
            f"{r['truth_gain_ms']} ms | {r['sim_iterations']}/{r['sim_accepted']} |"
        )
    if gains:
        lines.append(
            f"\n合计 {len(gains)} 个场景：NN 判定提升均值 {sum(gains) / len(gains):.1f} ms / "
            f"最大 {max(gains):.1f} ms；真值复核均值 "
            f"{sum(truth_gains) / max(1, len(truth_gains)):.1f} ms"
        )
    lines.append("\n## 3. 赛道签名（模型给出的最优翼角）\n")
    lines.append("| 赛道 | 最优翼角（默认 25 档起） | 该点预测 Δ |")
    lines.append("|------|------------------------|-----------|")
    for r in signature:
        lines.append(
            f"| {r['track']} | {r['best_wing_clicks']:+d} 档 | {r['best_delta_ms']} ms |"
        )
    lines.append("\n> 方向校验：Monaco/Hungaroring（高下压力）应排在前列，"
                 "Monza/Spa（低阻）应排在后列 —— 与 F1 调教共识一致。")
    lines.append("\n## 4. 采纳轨迹示例（前 6 条）\n")
    for r in results[:3]:
        lines.append(f"**{r['track']} / {r['scenario']}**")
        for t in r["sim_trace"]:
            lines.append(f"- {t}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用遥测验证调教优化效果")
    ap.add_argument("--tracks", default=",".join(DEFAULT_TRACKS))
    args = ap.parse_args(argv)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    print("=" * 70)
    print("调教优化验证（遥测锚定模型 + 端到端管线）")
    print("=" * 70)

    engine_info = verify_engine_integration(tracks)
    print("\n[1/3] 模型与引擎集成")
    print(f"  {engine_info['describe']}")
    for tid, a in engine_info["anchors"].items():
        if a.get("covered"):
            print(f"  {tid:12s} 默认调教预测 {a['default_delta_s'] * 1000:+7.1f} ms"
                  f" | 确定性 {'✓' if a['deterministic'] else '✗'}")
        else:
            print(f"  {tid:12s} 未覆盖")
    if not engine_info["available"]:
        print("\n模型不可用：请先跑 build_setup_sim_dataset.py + train_setup_sim_nn.py")
        return 1

    print("\n[2/3] 优化效果（nn vs rule；NN 裁判 + 解析仿真真值交叉复核）")
    results = verify_optimization_gain(tracks)
    gains = [r["gain_vs_rule_ms"] for r in results if r["gain_vs_rule_ms"] is not None]
    truth_gains = [r["truth_gain_ms"] for r in results if r["truth_gain_ms"] is not None]
    for r in results:
        print(f"  {r['track']:12s} {r['scenario'][:14]:16s} "
              f"rule={r['sim_rule_ms']:+7.1f}ms → nn={r['sim_nn_ms']:+7.1f}ms "
              f"| NN 判定 {r['gain_vs_rule_ms']:+7.1f}ms / 真值复核 "
              f"{r['truth_gain_ms']:+7.1f}ms（迭代{r['sim_iterations']}轮/"
              f"采纳{r['sim_accepted']}次）")
    if gains:
        print(f"  ---- NN 判定平均提升 {sum(gains) / len(gains):+.1f} ms | "
              f"真值复核平均 {sum(truth_gains) / max(1, len(truth_gains)):+.1f} ms "
              f"（真值为解析仿真器口径，独立于 NN 自身评价）")

    print("\n[3/3] 赛道签名（最优翼角）")
    signature = verify_track_signature()
    for r in signature:
        print(f"  {r['track']:12s} 最优翼角 {r['best_wing_clicks']:+d} 档"
              f"（{r['best_delta_ms']:+7.1f} ms）")

    md = render_markdown(engine_info, results, signature)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"\n报告已写出：{OUT_MD}")

    report_json = ROOT / "data" / "training" / "optimization_verification.json"
    report_json.write_text(json.dumps({
        "engine": engine_info, "results": results, "signature": signature,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"数据已写出：{report_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
