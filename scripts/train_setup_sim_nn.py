"""训练调教性能神经网络（纯标准库，无 PyTorch）。

数据流::

    data/training/setup_sim_dataset.jsonl   （遥测锚定仿真样本，见
        scripts/build_setup_sim_dataset.py —— 80k+ 真实 2026 逐弯遥测做锚点）
    data/training/setup_sim_meta.json       （每赛道聚合特征表）
        → 特征：[赛道 one-hot | 聚合特征 | 调教 20 | 工况]
        → 纯标准库 MLP（setup_tuner.engine.pure_nn）拟合 lap_delta_ms
        → data/models/setup_sim_nn.json      （引擎 sim_optimizer 使用）

评估：80/20 留出集 MAE（ms）/ R²；同时给出「零改动基线」（恒预测 0ms）
对照，让指标可解释 —— MAE 必须显著小于样本标准差才算学到东西。

用法::

    python scripts/train_setup_sim_nn.py
    python scripts/train_setup_sim_nn.py --epochs 120 --hidden 48,24
    python scripts/train_setup_sim_nn.py --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data" / "training" / "setup_sim_dataset.jsonl"
META = ROOT / "data" / "training" / "setup_sim_meta.json"
OUT = ROOT / "data" / "models" / "setup_sim_nn.json"


def _mae(y: list[float], p: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y, p, strict=True)) / max(1, len(y))


def _r2(y: list[float], p: list[float]) -> float:
    if not y:
        return 0.0
    m = sum(y) / len(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, p, strict=True))
    ss_tot = sum((a - m) ** 2 for a in y)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def load_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not DATASET.exists():
        raise SystemExit(f"缺少仿真数据集 {DATASET}（先跑 build_setup_sim_dataset.py）")
    if not META.exists():
        raise SystemExit(f"缺少数据集元信息 {META}")
    rows: list[dict[str, Any]] = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    meta = json.loads(META.read_text(encoding="utf-8"))
    return rows, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练调教性能神经网络（纯标准库）")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", default="32,16", help="隐藏层（逗号分隔）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    from setup_tuner.engine.pure_nn import MLP, mean_abs_error, r2_score
    from setup_tuner.engine.setup_sim import build_feature_row, feature_names

    rows, meta = load_dataset()
    track_ids: list[str] = list(meta.get("tracks") or [])
    track_features: dict[str, dict[str, float]] = meta.get("track_features") or {}
    if not rows or not track_ids:
        raise SystemExit("数据集为空")

    # 特征矩阵（训练/推理共用 build_feature_row，口径一致）
    x: list[list[float]] = []
    y: list[float] = []
    skipped = 0
    for r in rows:
        row = build_feature_row(
            track_ids, track_features, str(r.get("track_id")), r.get("setup_norm") or {},
        )
        if row is None:
            skipped += 1
            continue
        x.append(row)
        y.append(float(r.get("lap_delta_ms") or 0.0) / 1000.0)  # ms → s
    if skipped:
        print(f"跳过 {skipped} 行（赛道不在特征表）")
    n_in = len(x[0])
    print(f"样本 {len(x)} | 输入 {n_in} 维 | 赛道 {len(track_ids)} | "
          f"样本标准差 {math.sqrt(sum((v - sum(y)/len(y))**2 for v in y)/len(y)):.4f} s")

    # 留出集（固定种子确定性划分）
    rng = random.Random(args.seed)
    idx = list(range(len(x)))
    rng.shuffle(idx)
    n_val = max(200, int(args.val_ratio * len(x)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_tr = [x[i] for i in train_idx]
    y_tr = [y[i] for i in train_idx]
    x_va = [x[i] for i in val_idx]
    y_va = [y[i] for i in val_idx]

    hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
    sizes = [n_in, *hidden, 1]
    print(f"网络结构: {sizes} | epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch_size} seed={args.seed}")

    # 输出缩放：目标量纲 ±2.5s 直接进 MSE 会使梯度条件数差（小尺度局部曲率
    # 淹没在大尺度样本里）。按训练集标准差缩放到 ~1 量级，推理时乘回 ——
    # 训练质量显著提升（实测差分方向正确率 +8pt 量级）。
    y_mean = sum(y_tr) / len(y_tr)
    y_var = sum((v - y_mean) ** 2 for v in y_tr) / len(y_tr)
    y_scale = math.sqrt(y_var) if y_var > 1e-12 else 1.0
    y_tr_scaled = [(v - y_mean) / y_scale for v in y_tr]
    print(f"输出缩放: offset={y_mean:.4f}s scale={y_scale:.4f}s")

    t0 = time.time()
    model = MLP(sizes, seed=args.seed)
    model.fit(
        x_tr, [[v] for v in y_tr_scaled],
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    pred = [(v[0] * y_scale + y_mean) for v in model.predict(x_va)]
    mae_s = mean_abs_error(y_va, pred)
    r2 = r2_score(y_va, pred)
    # 零改动基线（恒预测 0s）：模型必须显著更好
    base_mae = mean_abs_error(y_va, [0.0] * len(y_va))
    print(f"验证集：MAE={mae_s * 1000:.2f} ms | R²={r2:.4f} | "
          f"零改动基线 MAE={base_mae * 1000:.2f} ms | 训练耗时 {elapsed:.1f}s")

    payload = {
        "schema": "f1opt-setup-sim-nn/1",
        "source": "telemetry-anchored QSS simulation (TracingInsights 2026 anchors)",
        "track_ids": track_ids,
        "track_features": track_features,
        "feature_keys": feature_names(track_ids),
        "n_samples": len(x),
        "metrics": {
            "val_mae_ms": round(mae_s * 1000.0, 3),
            "val_r2": round(r2, 5),
            "baseline_mae_ms": round(base_mae * 1000.0, 3),
            "n_train": len(x_tr),
            "n_val": len(x_va),
            "epochs": args.epochs,
            "seed": args.seed,
            "hidden": hidden,
            "train_seconds": round(elapsed, 2),
        },
        "mlp": {"model": model.to_dict(), "metrics": {"val_mae_ms": round(mae_s * 1000.0, 3)}},
        # 输出缩放口径（推理端必须乘回；见 setup_tuner.engine.setup_sim）
        "y_mean": round(y_mean, 6),
        "y_scale": round(y_scale, 6),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"写出: {out}（{out.stat().st_size / 1e3:.0f} KB）")

    # 端到端自检：加载并做一次推理（与引擎同一入口）
    sys.path.insert(0, str(ROOT))
    from setup_tuner.engine.setup_sim import SetupSimModel

    m = SetupSimModel(out)
    assert m.available, f"模型加载失败：{m.reason}"
    sample = rows[-1]
    pred_s = m.predict_delta_s(sample["track_id"], sample["setup_norm"])
    print(f"自检：{sample['track_id']} 预测 {pred_s * 1000 if pred_s is not None else None:.1f} ms "
          f"vs 仿真标注 {sample['lap_delta_ms']:.1f} ms")
    print(f"自检：未知赛道返回 {m.predict_delta_s('no_such_track', sample['setup_norm'])}（应为 None）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
