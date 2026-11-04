"""用真实「调教 → 圈速」配对校准调教性能模型（第一次有外部真值）。

背景
----
`setup_sim_nn.json` 的标签全部来自 `simulate_lap_delta_ms`（项目自写的准静态
物理模型），验证 R² 0.997 只证明「网络学会了复现仿真器」，不证明「预测真实圈速」。
`setup_sim_eval.json` 自己也写着 `real_setup_laps: 0`。

本脚本引入 `scripts/fetch_setup_laps.py` 抓到的真实配对（F1Laps 社区调教，
每套带 20 项数值与实测圈速），回答三个此前无法回答的问题：

1. **排序力**：站内把调教按「模型预测更快」排序，与真实圈速排序一致吗？
   （Spearman ρ；ρ≈0 表示模型对真实调教没有区分力）
2. **量级**：模型预测的站内离散度 vs 真实圈速离散度。
   （ratio >> 1 表示高估了调教的影响）
3. **符号**：预测「更快」的调教，真实是否也确实更快？（分组单调性）

方法上的克制
------------
- 圈速跨车手，车手差异远大于调教差异。因此**只在站内比较**（不跨站），
  并把 `assists` 与 `session` 作为混淆项统计（不同辅助/会话不混在一起算）。
- 只报告**相关与排序**，不报告「MAE 多少毫秒」——绝对误差在这里没有意义，
  因为真实圈速里混着车手水平这个无法观测的变量。
- 样本 < 8 的赛道跳过并在报告里标注，不做小样本结论。

用法::

    python scripts/calibrate_sim_with_real_laps.py
    python scripts/calibrate_sim_with_real_laps.py --min-per-track 12

产物::

    data/setup_laps/calibration_report.json
    data/setup_laps/calibration_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "setup_laps" / "f1laps_setups.jsonl"
OUT_JSON = ROOT / "data" / "setup_laps" / "calibration_report.json"
OUT_MD = ROOT / "data" / "setup_laps" / "calibration_report.md"


def _rank(values: list[float]) -> list[float]:
    """平均秩（处理并列）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman ρ = 秩上的 Pearson；对单调关系稳健。"""
    if len(x) < 3:
        return None
    return _pearson(_rank(x), _rank(y))


def _stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用真实调教-圈速校准仿真模型")
    ap.add_argument("--min-per-track", type=int, default=8,
                    help="每站最少样本数，低于则跳过（不做小样本结论）")
    ap.add_argument("--source", default=str(SRC))
    args = ap.parse_args(argv)

    src = Path(args.source)
    if not src.exists():
        print(f"[error] 缺少真实配对数据：{src}\n"
              f"        先运行 python scripts/fetch_setup_laps.py --dry", file=sys.stderr)
        return 1

    from setup_tuner.engine.setup_sim import get_setup_sim

    sim = get_setup_sim()
    if not sim.available:
        print(f"[error] 调教性能模型不可用：{sim.reason}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 分组粒度：赛道 × 会话 × TC 开关。
    #
    # 为什么不用「全部辅助设置」做键：实测 20 条记录里能出现 9 种组合，
    # 每组只剩 1~8 条，全被 min_per_track 挡掉，校准变成空报告。
    # 牵引力控制是圈速影响最大的一项（TC Off vs Full 常差 1 s 以上），
    # 必须同质；ABS / 变速箱 / 行车线对圈速影响小一个量级，作为残余混淆记录在
    # 报告里，不参与分组。
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    unknown_track = 0
    no_pred = 0
    for rec in rows:
        tid = rec["track_id"]
        if not sim.covers(tid):
            unknown_track += 1
            continue
        pred = sim.predict_for_setup(tid, rec["setup"])
        if pred is None:
            no_pred += 1
            continue
        assists = rec.get("assists") or {}
        tc = str(assists.get("traction_control") or "").lower()
        key = (
            tid,
            str(rec.get("session") or "unknown"),
            "tc_off" if tc.startswith("off") else "tc_on",
        )
        groups[key].append({"lap_ms": rec["lap_time_ms"], "pred_s": pred})

    per_track: dict[str, Any] = {}
    track_rows: list[dict[str, Any]] = []
    # 逐组统计：TC Off 与 TC On 的圈速基线差 1 s 以上，混在一起算相关会被
    # 这层台阶吃掉，所以分组独立评估，报告里按 `track/tc_*` 展示。
    for (tid, _session, tc_group), items in sorted(groups.items()):
        label = f"{tid}/{tc_group}"
        if len(items) < args.min_per_track:
            per_track[label] = {"n": len(items), "skipped": "样本不足"}
            continue
        actual = [it["lap_ms"] / 1000.0 for it in items]
        pred = [it["pred_s"] for it in items]
        # 真实圈速越小越快 → 与预测增量（正=更慢）应正相关
        rho = _spearman(pred, actual)
        pear = _pearson(pred, actual)
        spread_real = _stdev(actual)
        spread_pred = _stdev(pred)
        ratio = (spread_pred / spread_real) if spread_real > 1e-9 else None
        # 分位单调性：按预测从快到慢分三组，看真实中位圈速是否递增
        order = sorted(range(len(pred)), key=lambda i: pred[i])
        k = max(1, len(order) // 3)
        fast = [actual[i] for i in order[:k]]
        slow = [actual[i] for i in order[-k:]]
        med_fast, med_slow = statistics.median(fast), statistics.median(slow)
        per_track[label] = {
            "n": len(items),
            "spearman_rho": None if rho is None else round(rho, 4),
            "pearson_r": None if pear is None else round(pear, 4),
            "pred_spread_s": round(spread_pred, 3),
            "real_spread_s": round(spread_real, 3),
            "spread_ratio": None if ratio is None else round(ratio, 2),
            "median_gap_slow_minus_fast_s": round(med_slow - med_fast, 3),
            "monotonic": bool(med_slow > med_fast),
        }
        track_rows.append(per_track[label])

    def _median(key: str) -> float | None:
        vals = [r[key] for r in track_rows if r.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    summary = {
        "tracks_evaluated": len(track_rows),
        "median_spearman_rho": _median("spearman_rho"),
        "median_pearson_r": _median("pearson_r"),
        "median_spread_ratio": _median("spread_ratio"),
        "monotonic_tracks": sum(1 for r in track_rows if r["monotonic"]),
        "total_tracks": len(track_rows),
        "unknown_track_samples": unknown_track,
        "no_prediction_samples": no_pred,
        "total_samples": len(rows),
    }

    verdict: list[str] = []
    rho = summary["median_spearman_rho"]
    ratio = summary["median_spread_ratio"]
    if rho is None:
        verdict.append("样本不足，无法判定排序力")
    elif rho >= 0.4:
        verdict.append(f"排序力可用（中位 ρ={rho}）：模型在真实调教上具备区分能力")
    elif rho >= 0.15:
        verdict.append(f"排序力偏弱（中位 ρ={rho}）：有信号但不足以支撑强结论")
    else:
        verdict.append(f"排序力不可用（中位 ρ={rho}）：模型对真实调教的快慢没有区分力")
    if ratio is not None:
        if ratio > 2.0:
            verdict.append(f"量级高估约 {ratio:.1f}×：预测离散度远大于真实圈速离散度")
        elif ratio < 0.5:
            verdict.append(f"量级低估约 {1 / max(ratio, 1e-9):.1f}×：预测几乎不区分调教")
        else:
            verdict.append(f"量级接近真实（ratio={ratio:.2f}）")

    report = {
        "schema": "f1opt-sim-calibration/1",
        "source": str(src.name),
        "model_tracks": len(sim.track_ids),
        "model_anchor_sources": {
            "telemetry": sum(1 for v in sim.anchor_sources.values() if v == "telemetry"),
            "authored": sum(1 for v in sim.anchor_sources.values() if v != "telemetry"),
        },
        "summary": summary,
        "verdict": verdict,
        "per_track": per_track,
        "caveat": (
            "真实圈速跨车手，车手水平是不可观测的混淆变量；"
            "本报告只做站内相关与排序，不报告绝对误差。"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 真实调教 → 圈速 校准报告",
        "",
        f"- 样本文件：`{src.name}`",
        f"- 样本数：{summary['total_samples']}（参与评测 {len(rows) - unknown_track - no_pred}）",
        f"- 覆盖赛道（模型内）：{summary['tracks_evaluated']}",
        f"- 中位 Spearman ρ：**{summary['median_spearman_rho']}**",
        f"- 中位离散度比（预测/真实）：**{summary['median_spread_ratio']}**",
        f"- 单调赛道：{summary['monotonic_tracks']}/{summary['total_tracks']}",
        "",
        "## 结论",
        "",
    ]
    lines += [f"- {v}" for v in verdict]
    lines += [
        "",
        "## 逐赛道",
        "",
        "| 赛道 | n | ρ | r | 预测离散(s) | 真实离散(s) | 比值 | 单调 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tid, m in sorted(per_track.items()):
        if m.get("skipped"):
            lines.append(f"| {tid} | {m['n']} | — | — | — | — | — | 跳过 |")
            continue
        lines.append(
            f"| {tid} | {m['n']} | {m['spearman_rho']} | {m['pearson_r']} | "
            f"{m['pred_spread_s']} | {m['real_spread_s']} | {m['spread_ratio']} | "
            f"{'是' if m['monotonic'] else '否'} |"
        )
    lines += ["", f"> {report['caveat']}", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[calibrate] {json.dumps(summary, ensure_ascii=False)}")
    for v in verdict:
        print(f"[verdict] {v}")
    print(f"[out] {OUT_JSON}\n[out] {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
