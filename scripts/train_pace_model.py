"""在「本地 UDP 全量样本 + 外部 2026 专业基准」融合数据上训练配速模型。

输入：``data/training/merged_dataset.json``（由 merge_features.py 生成）
输出：``data/training/pace_model.json``（权重 + 归一化参数 + 交叉验证指标）

模型：岭回归（纯标准库实现，零第三方依赖 —— venv 内没有 numpy/torch，
不引入新依赖）。特征 = 本地 29 维（赛道画像/工况/风格）+ 外部 2026 基准 8 维。
目标 = ``lap_time_ms / track_length_m``（每公里用时，秒/km）。

**诚实守门**：样本数不足时脚本照常运行并给出可复现的基线指标，但会在
报告里明确标注「仅供管线验证，不可用于决策」——避免把 14 条样本的拟合
当成真实模型。

用法::

    python scripts/train_pace_model.py
    python scripts/train_pace_model.py --l2 1.0 --cv loo
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "training" / "merged_dataset.json"
MODEL_OUT = ROOT / "data" / "training" / "pace_model.json"

#: 低于该样本数只输出管线验证结论（不宣称可用）
MIN_USABLE_SAMPLES = 30


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """高斯消元（部分主元）解线性方程组 —— 纯标准库。"""
    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] = 1e-12  # 奇异兜底（ridge 下不应发生）
        a[col], a[pivot] = a[pivot], a[col]
        pv = a[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / pv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def _fit_ridge(
    x: list[list[float]], y: list[float], l2: float,
) -> list[float]:
    """岭回归闭式解 w = (XᵀX + λI)⁻¹Xᵀy（含截距列，截距不惩罚）。"""
    d = len(x[0])
    xtx = [[0.0] * d for _ in range(d)]
    xty = [0.0] * d
    for row, target in zip(x, y, strict=True):
        for i in range(d):
            xty[i] += row[i] * target
            for j in range(d):
                xtx[i][j] += row[i] * row[j]
    for i in range(1, d):  # 跳过截距（index 0）
        xtx[i][i] += l2
    return _solve(xtx, xty)


def _predict(weights: list[float], row: list[float]) -> float:
    return sum(w * v for w, v in zip(weights, row, strict=True))


def _standardize(
    x: list[list[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    """列标准化（截距列保持 1）。"""
    d = len(x[0])
    means = [0.0] * d
    stds = [1.0] * d
    for j in range(1, d):
        col = [r[j] for r in x]
        means[j] = sum(col) / len(col)
        var = sum((v - means[j]) ** 2 for v in col) / max(1, len(col) - 1)
        stds[j] = math.sqrt(var) or 1.0
    out = [
        [1.0] + [(r[j] - means[j]) / stds[j] for j in range(1, d)]
        for r in x
    ]
    return out, means, stds


def _mae(a: list[float], b: list[float]) -> float:
    return sum(abs(u - v) for u, v in zip(a, b, strict=True)) / len(a)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="融合数据配速模型（岭回归，纯标准库）")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--out", default=str(MODEL_OUT))
    ap.add_argument("--l2", type=float, default=1.0, help="岭正则强度")
    ap.add_argument("--cv", choices=("loo", "kfold"), default="loo")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    path = Path(args.dataset)
    if not path.exists():
        print(f"数据集不存在：{path}\n先运行 python scripts/merge_features.py")
        return 2
    ds = json.loads(path.read_text(encoding="utf-8"))
    keys: list[str] = ds["feature_keys"]
    rows = [r for r in ds["rows"] if r.get("target") is not None]
    if len(rows) < 3:
        print(f"样本不足（{len(rows)} 条），无法训练。")
        return 3

    x_raw = [[1.0] + [float((r["features"] or {}).get(k, 0.0)) for k in keys]
             for r in rows]
    y = [float(r["target"]) for r in rows]
    x, means, stds = _standardize(x_raw)

    # ---- 交叉验证 ----
    preds = [0.0] * len(rows)
    folds = len(rows) if args.cv == "loo" else args.folds
    for f in range(folds):
        idx_test = [
            i for i in range(len(rows)) if i % folds == f
        ] if args.cv == "kfold" else [f]
        idx_train = [i for i in range(len(rows)) if i not in idx_test]
        if not idx_train:
            continue
        w = _fit_ridge([x[i] for i in idx_train], [y[i] for i in idx_train], args.l2)
        for i in idx_test:
            preds[i] = _predict(w, x[i])
    cv_mae = _mae(y, preds) if args.cv == "kfold" else _mae(
        [y[i] for i in range(len(rows))], preds,
    )
    baseline = statistics.mean(y)
    baseline_mae = sum(abs(v - baseline) for v in y) / len(y)

    weights = _fit_ridge(x, y, args.l2)
    in_sample = _mae(y, [_predict(weights, r) for r in x])

    usable = len(rows) >= MIN_USABLE_SAMPLES
    report: dict[str, Any] = {
        "schema": "f1opt-pace-model/1",
        "trained_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(rows),
        "feature_keys": keys,
        "l2": args.l2,
        "cv": {"kind": args.cv, "folds": folds, "mae_s_per_km": round(cv_mae, 4)},
        "baseline": {"mean_s_per_km": round(baseline, 4),
                     "mae_s_per_km": round(baseline_mae, 4)},
        "in_sample_mae_s_per_km": round(in_sample, 4),
        "usable_for_decision": usable,
        "weights": {"intercept": round(weights[0], 6), **{
            k: round(w, 6) for k, w in zip(keys, weights[1:], strict=True)
        }},
        "normalization": {"mean": [round(v, 6) for v in means],
                          "std": [round(v, 6) for v in stds]},
        "note": (
            "外部特征来自 TracingInsights 2026（仅 2026，年检通过）；"
            "本地特征来自 F1 2025 2026 Season Pack UDP 录制。"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 配速模型训练（岭回归，纯标准库）===")
    print(f"样本 {len(rows)} 条 / 特征 {len(keys)} 维（含外部 2026 基准 "
          f"{sum(1 for r in rows if r.get('has_external_ref'))} 条）")
    print(f"交叉验证（{args.cv}）MAE = {cv_mae:.4f} s/km"
          f"   基线(均值) MAE = {baseline_mae:.4f} s/km")
    print(f"样本内 MAE = {in_sample:.4f} s/km")
    verdict = "是" if usable else f"否（样本 < {MIN_USABLE_SAMPLES}，仅供管线验证）"
    print(f"决策可用性: {verdict}")
    print(f"模型写出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
