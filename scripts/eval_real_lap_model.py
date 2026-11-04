"""真实圈速校准模型的**零泄漏**泛化检验（训练脚本之外的独立验收）。

为什么单独一个脚本
------------------
`train_real_lap_model.py` 的留出指标来自 K 折交叉验证：同队同会话的配对
可能分散在不同折，折之间的标准化偏移会给指标带进一丝水分。要让「这个
模型到底学到了什么」站得住，必须再做两层完全不共享标签的检验：

A. **留一赛道（leave-one-track-out）**
   把整站样本全部移出训练集，用剩下 23 站拟合，再预测这一站内部的
   同队同会话配对。这是对「部署到一条新赛道」的直接模拟 —— 模型只能靠
   `aero_index` / `fast_share` / `traction_index` / `braking_index`
   这类赛道特征把规律外推过去，记不住任何单站答案。

B. **跨年 / 跨切面**
   2026 训练 → 2025 测试（车与规则都变了）、干地训练 → 湿地测试。

判定标准：配对准确率显著高于 50%（二项 95% 区间），且优于纯仿真基线。

用法::

    python scripts/eval_real_lap_model.py
    python scripts/eval_real_lap_model.py --source data/setup_laps/f1laps_setups.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "setup_laps" / "f1laps_setups.jsonl"
MODEL = ROOT / "setup_tuner" / "resources" / "models" / "real_lap_calibration.json"
OUT_JSON = ROOT / "data" / "setup_laps" / "real_lap_holdout.json"
OUT_MD = ROOT / "data" / "setup_laps" / "real_lap_holdout.md"

#: 少于这个配对数的结论不写进报告（噪声太大）
MIN_PAIRS = 30


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 区间（比正态近似在小样本下稳）。"""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - r) / d, (c + r) / d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="零泄漏泛化检验")
    ap.add_argument("--source", default=str(SRC))
    args = ap.parse_args(argv)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "trm", ROOT / "scripts" / "train_real_lap_model.py",
    )
    if spec is None or spec.loader is None:
        print("[error] 无法加载训练脚本", file=sys.stderr)
        return 1
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    from setup_tuner.engine.setup_sim import get_setup_sim, normalize_setup

    sim = get_setup_sim()
    if not sim.available:
        print(f"[error] 调教性能模型不可用：{sim.reason}", file=sys.stderr)
        return 1

    lam = 30000.0
    if MODEL.exists():
        try:
            lam = float(json.loads(MODEL.read_text(encoding="utf-8"))["ridge_lambda"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    def prep(recs: list[dict[str, Any]]):
        rows = [r for r in recs if sim.covers(r["track_id"])]
        if not rows:
            return rows, [], [], []
        residual = tr.strip_group_effects(rows)
        zero: dict[str, float | None] = {}
        sim_pred: list[float | None] = []
        for rec in rows:
            tid = rec["track_id"]
            if tid not in zero:
                zero[tid] = sim.predict_raw_delta_s(tid, {})
            raw = sim.predict_raw_delta_s(tid, normalize_setup(rec["setup"]))
            base = zero[tid]
            sim_pred.append(None if raw is None or base is None else raw - base)
        design = tr.build_design(rows, sim.track_features, sim_pred)
        return rows, residual, sim_pred, design

    def fit(rows, residual, design):
        means, stds = tr.column_stats(design)
        beta = tr.ridge_fit(tr.standardize(design, means, stds), residual, lam)
        return tr.expand_track_coef(beta, means, stds, sim.track_features)

    def predict(rows, design, coef, simc):
        out = []
        for i, rec in enumerate(rows):
            c = coef.get(rec["track_id"])
            out.append(
                0.0 if c is None
                else sum(cc * design[i][1 + j] for j, cc in enumerate(c))
                + simc.get(rec["track_id"], 0.0) * design[i][0],
            )
        return out

    def score(rows, pred, sim_pred) -> dict[str, Any] | None:
        hit, pairs = tr.matched_pair_counts(rows, list(range(len(rows))), pred)
        svec = [0.0 if v is None else float(v) for v in sim_pred]
        shit, spairs = tr.matched_pair_counts(rows, list(range(len(rows))), svec)
        if pairs < MIN_PAIRS:
            return None
        acc = hit / pairs
        sacc = shit / spairs if spairs else None
        lo, hi = wilson(hit, pairs)
        return {
            "n": len(rows),
            "pairs": pairs,
            "acc": round(acc, 4),
            "sim_acc": None if sacc is None else round(sacc, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0.5),
        }

    raw, dropped = tr.load_rows(Path(args.source))
    if not raw:
        print(f"[error] 无有效样本：{dropped}", file=sys.stderr)
        return 1
    full_rows, full_res, full_sim, full_design = prep(raw)
    if not full_rows:
        print("[error] 没有赛道被仿真模型覆盖", file=sys.stderr)
        return 1

    # ---- A. 留一赛道 ----
    by_track: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(full_rows):
        by_track[rec["track_id"]].append(i)
    loto: dict[str, dict[str, Any]] = {}
    pooled_hit = pooled_pairs = 0
    for tid, idxs in sorted(by_track.items()):
        keep = set(idxs)
        rest = [i for i in range(len(full_rows)) if i not in keep]
        if not rest:
            continue
        coef, simc = fit(
            [full_rows[i] for i in rest], [full_res[i] for i in rest],
            [full_design[i] for i in rest],
        )
        sub_rows = [full_rows[i] for i in idxs]
        pred = predict(sub_rows, [full_design[i] for i in idxs], coef, simc)
        res = score(sub_rows, pred, [full_sim[i] for i in idxs])
        if res is None:
            continue
        loto[tid] = res
        pooled_hit += res["acc"] * res["pairs"]
        pooled_pairs += res["pairs"]

    loto_pooled = None
    if pooled_pairs:
        lo, hi = wilson(pooled_hit, pooled_pairs)
        loto_pooled = {
            "pairs": pooled_pairs,
            "acc": round(pooled_hit / pooled_pairs, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0.5),
            "tracks_above_55": sum(1 for v in loto.values() if v["acc"] > 0.55),
            "tracks": len(loto),
        }

    # ---- B. 跨年 / 跨切面 ----
    transfers: list[dict[str, Any]] = []

    def transfer(tag, train_sel, test_sel):
        tr_recs = [r for r in raw if train_sel(r)]
        te_recs = [r for r in raw if test_sel(r)]
        if len(tr_recs) < 40 or len(te_recs) < 20:
            return
        a_rows, a_res, _a_sim, a_design = prep(tr_recs)
        b_rows, _b_res, b_sim, b_design = prep(te_recs)
        if not a_rows or not b_rows:
            return
        coef, simc = fit(a_rows, a_res, a_design)
        res = score(b_rows, predict(b_rows, b_design, coef, simc), b_sim)
        if res is None:
            return
        transfers.append({"tag": tag, "train": len(a_rows), **res})

    years = sorted({r.get("game_year") for r in raw if r.get("game_year")})
    for ty in years:
        for ey in years:
            if ty != ey:
                transfer(
                    f"{ty} 干 → {ey} 干",
                    lambda r, ty=ty: r.get("game_year") == ty and r.get("surface") == "dry",
                    lambda r, ey=ey: r.get("game_year") == ey and r.get("surface") == "dry",
                )
    for gy in years:
        transfer(
            f"{gy} 干 → {gy} 湿",
            lambda r, gy=gy: r.get("game_year") == gy and r.get("surface") == "dry",
            lambda r, gy=gy: r.get("game_year") == gy and r.get("surface") == "wet",
        )

    # ---- 自训自测上限（同集 K 折，作为对照）----
    cv_pred = tr.cross_validate(full_rows, full_res, full_design, 5, lam)
    cv_res = score(full_rows, cv_pred, full_sim)

    payload = {
        "schema": "f1opt-real-lap-holdout/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": Path(args.source).name,
        "ridge_lambda": lam,
        "n_samples": len(full_rows),
        "n_tracks": len(by_track),
        "dropped": dropped,
        "cv": cv_res,
        "leave_one_track_out": {"pooled": loto_pooled, "per_track": loto},
        "transfer": transfers,
        "caveat": (
            "配对口径 = 同赛道同车队同会话内的两套调教谁更快；"
            "留一赛道时该站完全不参与训练，模型只能靠赛道特征外推。"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    def ci(d):
        return f"[{d['ci95'][0]:.1%}, {d['ci95'][1]:.1%}]" if d else "n/a"

    lines = [
        "# 真实圈速校准 · 零泄漏泛化检验",
        "",
        f"- 样本：**{len(full_rows)}** 条 / **{len(by_track)}** 站（λ={lam:g}）",
        "- 配对口径：同赛道 + 同车队 + 同会话内的两套调教谁更快",
        "",
        "## A. 留一赛道（该站完全不参与训练）",
        "",
        "| 口径 | 配对 | 准确率 | 95% 区间 | 显著优于随机 |",
        "|---|---|---|---|---|",
    ]
    if loto_pooled:
        lines.append(
            f"| **合并 {loto_pooled['tracks']} 站** | {loto_pooled['pairs']} | "
            f"**{pct(loto_pooled['acc'])}** | {ci(loto_pooled)} | "
            f"{'是' if loto_pooled['significant'] else '否'} |"
        )
    if cv_res:
        lines.append(
            f"| 同集 5 折（对照上限） | {cv_res['pairs']} | {pct(cv_res['acc'])} | "
            f"{ci(cv_res)} | {'是' if cv_res['significant'] else '否'} |"
        )
    lines += ["", "### 逐赛道", "",
              "| 赛道 | 配对 | 仿真 | 留出站 | 95% 区间 |", "|---|---|---|---|---|"]
    for tid, d in sorted(loto.items(), key=lambda kv: -kv[1]["acc"]):
        lines.append(
            f"| {tid} | {d['pairs']} | {pct(d['sim_acc'])} | {pct(d['acc'])} | {ci(d)} |"
        )

    lines += ["", "## B. 跨年 / 跨切面迁移", ""]
    if transfers:
        lines += ["| 迁移 | 配对 | 仿真 | 真实校准 | 95% 区间 | 显著 |", "|---|---|---|---|---|---|"]
        for d in transfers:
            lines.append(
                f"| {d['tag']} | {d['pairs']} | {pct(d['sim_acc'])} | {pct(d['acc'])} | "
                f"{ci(d)} | {'是' if d['significant'] else '否'} |"
            )
    else:
        lines.append("_可用切面不足，未做迁移检验。_")
    lines += ["", f"> {payload['caveat']}", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[holdout] 样本 {len(full_rows)} / {len(by_track)} 站；λ={lam:g}")
    if loto_pooled:
        print(f"[holdout] 留一赛道合并：{loto_pooled['acc']:.1%} "
              f"({loto_pooled['pairs']} 对, 95% {ci(loto_pooled)}, "
              f"{loto_pooled['tracks_above_55']}/{loto_pooled['tracks']} 站 >55%)")
    if cv_res:
        print(f"[holdout] 同集 5 折对照：{cv_res['acc']:.1%} ({cv_res['pairs']} 对)")
    for d in transfers:
        print(f"[holdout] {d['tag']}: {d['acc']:.1%} ({d['pairs']} 对)")
    print(f"[out] {OUT_JSON}\n[out] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
