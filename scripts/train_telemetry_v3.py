"""重训遥测代理模型 v3：经验贝叶斯基准 + 残差岭回归。

相对 `train_telemetry_surrogate.py` 的三处改动
------------------------------------------------
1. **换数据源**：直接吃 `data/f1_2026/train/corner_samples_clean.jsonl.gz`
   （21 站 / 314,891 条干净逐弯样本），不再依赖 `data/training/corner_dataset.jsonl`。
2. **先基准、再残差**：弯时由「赛道×弯号」主导，先用经验贝叶斯收缩估出基准
   （小样本弯向同弯型均值收缩），岭回归只学工况残差。原实现把 344 维 one-hot
   丢给一个全局岭回归，小样本弯的系数被噪声拉偏，且未见赛道整行 one-hot 为 0。
3. **按分站留出评估**：`splits.json` 的 15/3/3，同时保留随机留出便于与旧版对比。

兼容性：产出的 JSON 仍是 `f1opt-telemetry-surrogate/1`，新增 `corner_baseline`
块；引擎侧 `surrogate.py` 若读到 baseline 就走「baseline + 残差」，读不到则维持旧行为。

用法::

    python scripts/train_telemetry_v3.py --out setup_tuner/resources/models/telemetry_surrogate.json
    python scripts/train_telemetry_v3.py --experiment-only   # 只出对照报告，不写模型
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "data" / "f1_2026" / "train"
CORNER_GZ = TRAIN_DIR / "corner_samples_clean.jsonl.gz"
LAPS_JSONL = TRAIN_DIR / "laps_clean.jsonl"
SPLITS = TRAIN_DIR / "splits.json"
DEFAULT_OUT = ROOT / "setup_tuner" / "resources" / "models" / "telemetry_surrogate.json"

#: 引擎侧固定的 14 个工况特征（顺序不可变，surrogate._corner_row 与此一致）
CONDITION_KEYS: tuple[str, ...] = (
    "cls_slow", "cls_medium", "cls_fast",
    "tyre_softness", "tyre_medium", "tyre_wet",
    "wet", "track_temp", "air_temp",
    "traction_index", "aero_index", "braking_index",
    "slow_share", "fast_share",
)

#: 仅用于离线对照实验（证明轮胎上下文值不值得接进引擎）
TYRE_KEYS: tuple[str, ...] = ("tyre_life_norm", "fresh_tyre", "stint_norm")


def load_rows(with_tyre: bool) -> list[dict[str, Any]]:
    laps: dict[tuple[str, str, int], dict[str, Any]] = {}
    if with_tyre and LAPS_JSONL.exists():
        with LAPS_JSONL.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                laps[(r["track_id"], r["driver"], int(r["lap"]))] = r
    rows: list[dict[str, Any]] = []
    with gzip.open(CORNER_GZ, "rt", encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            lap = laps.get((o["track_id"], o["driver"], int(o["lap"]))) if laps else None
            o["_tyre_life"] = (lap or {}).get("tyre_life")
            o["_fresh"] = (lap or {}).get("fresh_tyre")
            o["_stint"] = (lap or {}).get("stint")
            rows.append(o)
    return rows


def condition_vector(r: dict[str, Any]) -> list[float]:
    cls = r.get("corner_class")
    tyre = r.get("tyre_class")
    return [
        1.0 if cls == "slow" else 0.0,
        1.0 if cls == "medium" else 0.0,
        1.0 if cls == "fast" else 0.0,
        float(r.get("tyre_softness") or 0.5),
        1.0 if tyre == "medium" else 0.0,
        1.0 if tyre in ("wet", "inter") else 0.0,
        float(r.get("wet") or 0.0),
        float(r.get("track_temp") or 35.0),
        float(r.get("air_temp") or 24.0),
        float(r.get("track_traction_index") or 0.0),
        float(r.get("track_aero_index") or 0.0),
        float(r.get("track_braking_index") or 0.0),
        float(r.get("track_slow_share") or 0.0),
        float(r.get("track_fast_share") or 0.0),
    ]


def tyre_vector(r: dict[str, Any]) -> list[float]:
    life = r.get("_tyre_life")
    try:
        life = float(life)
    except (TypeError, ValueError):
        life = 8.0
    stint = r.get("_stint")
    try:
        stint = float(stint)
    except (TypeError, ValueError):
        stint = 1.0
    return [min(life, 40.0) / 40.0, 1.0 if r.get("_fresh") else 0.0, min(stint, 5.0) / 5.0]


def pair_key(r: dict[str, Any]) -> str:
    return f"{r['track_id']}#{r['corner']}"


def baseline_table(rows: list[dict[str, Any]], k: float = 20.0) -> tuple[dict[str, float], dict[str, float]]:
    """经验贝叶斯收缩：(赛道,弯号) 中位数 → 该弯型中位数（k≈20 圈）。"""
    by_pair: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        share = float(r["corner_time_s"]) / float(r["lap_time_s"])
        by_pair[pair_key(r)].append(share)
        by_class[str(r.get("corner_class"))].append(share)
    class_med = {c: statistics.median(v) for c, v in by_class.items()}
    pair_med = {p: statistics.median(v) for p, v in by_pair.items()}
    pair_n = {p: len(v) for p, v in by_pair.items()}
    shrunk: dict[str, float] = {}
    cls_of: dict[str, str] = {}
    for r in rows:
        cls_of[pair_key(r)] = str(r.get("corner_class"))
    global_med = statistics.median([v for vs in by_pair.values() for v in vs])
    for p, med in pair_med.items():
        n = pair_n[p]
        prior = class_med.get(cls_of.get(p), global_med)
        shrunk[p] = (n / (n + k)) * med + (k / (n + k)) * prior
    return shrunk, class_med


class Ridge:
    """标准化 + 闭式解岭回归（numpy），保留 to_dict 以写入产物。"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> Ridge:
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        z = (x - self.mean) / self.std
        self.intercept = float(y.mean())
        yc = y - self.intercept
        d = z.shape[1]
        xtx = z.T @ z + self.alpha * np.eye(d)
        xty = z.T @ yc
        self.coef = np.linalg.solve(xtx, xty)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None and self.coef is not None
        return ((x - self.mean) / self.std) @ self.coef + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ridge", "alpha": self.alpha,
            "coef": [float(v) for v in self.coef],
            "intercept": float(self.intercept),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
        }


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def eval_pack(rows: list[dict[str, Any]], y: np.ndarray, pred: np.ndarray, label: str) -> dict[str, Any]:
    lap = np.array([float(r["lap_time_s"]) for r in rows])
    err_s = np.abs(pred - y) * lap
    return {
        "label": label,
        "n": int(len(y)),
        "mae_share": round(mae(y, pred), 6),
        "r2": round(r2(y, pred), 4),
        "mae_seconds": round(float(err_s.mean()), 4),
    }


def build_design(rows: list[dict[str, Any]], pair_index: dict[str, int], extra: list[str]) -> np.ndarray:
    n = len(rows)
    width = len(pair_index) + len(CONDITION_KEYS) + len(extra) * len(TYRE_KEYS)
    x = np.zeros((n, width), dtype=np.float64)
    for i, r in enumerate(rows):
        pi = pair_index.get(pair_key(r))
        if pi is not None:
            x[i, pi] = 1.0
        base = len(pair_index)
        x[i, base:base + len(CONDITION_KEYS)] = condition_vector(r)
        if extra:
            off = base + len(CONDITION_KEYS)
            x[i, off:off + len(TYRE_KEYS)] = tyre_vector(r)
    return x


def load_old_model(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    head = d.get("corner_head") or {}
    return head or None


def predict_old(head: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray | None:
    """复刻旧产物的推理口径（pair one-hot + 14 工况）。"""
    model = ((head.get("family") or {}).get("ridge") or {}).get("model")
    pair_keys = list(head.get("pair_keys") or [])
    if not model or not pair_keys:
        return None
    pi = {k: i for i, k in enumerate(pair_keys)}
    x = build_design(rows, pi, extra=[])
    coef = np.array(model["coef"], dtype=np.float64)
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    if x.shape[1] != len(coef):
        return None
    return ((x - mean) / std) @ coef + float(model["intercept"])


def split_of(track: str, splits: dict[str, list[str]]) -> str:
    for name in ("train", "val", "test"):
        if track in splits.get(name, []):
            return name
    return "unassigned"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="重训遥测代理 v3（基准 + 残差）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--experiment-only", action="store_true")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--shrink-k", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    rows = load_rows(with_tyre=True)
    print(f"样本 {len(rows)} / {len({r['track_id'] for r in rows})} 站 / "
          f"{len({pair_key(r) for r in rows})} 个弯对")

    y = np.array([float(r["corner_time_s"]) / float(r["lap_time_s"]) for r in rows])
    old_head = load_old_model(DEFAULT_OUT)
    report: dict[str, Any] = {"dataset": {
        "rows": len(rows), "tracks": sorted({r["track_id"] for r in rows}),
        "pairs": len({pair_key(r) for r in rows}),
    }, "splits": splits, "experiments": {}}

    def run_experiment(name: str, train_rows: list[int], test_rows: list[int]) -> dict[str, Any]:
        tr = [rows[i] for i in train_rows]
        te = [rows[i] for i in test_rows]
        y_tr, y_te = y[train_rows], y[test_rows]
        pairs = sorted({pair_key(r) for r in tr})
        pi = {k: i for i, k in enumerate(pairs)}
        out: dict[str, Any] = {"train_n": len(tr), "test_n": len(te)}

        # 1) 旧模型（同一批测试行）
        if old_head:
            p_old = predict_old(old_head, te)
            if p_old is not None:
                out["old_model"] = eval_pack(te, y_te, p_old, "旧产物（19 站 one-hot）")

        # 2) 仅基准（查表 + 收缩）
        base_shrunk, class_med = baseline_table(tr, k=args.shrink_k)
        global_med = statistics.median(list(base_shrunk.values()))
        p_base = np.array([
            base_shrunk.get(pair_key(r), class_med.get(str(r.get("corner_class")), global_med))
            for r in te
        ])
        out["baseline_only"] = eval_pack(te, y_te, p_base, "基准（收缩查表）")

        # 3) 基准 + 残差岭（14 工况）
        resid = y_tr - np.array([
            base_shrunk.get(pair_key(r), class_med.get(str(r.get("corner_class")), global_med))
            for r in tr
        ])
        x_tr = np.array([condition_vector(r) for r in tr])
        x_te = np.array([condition_vector(r) for r in te])
        m = Ridge(alpha=args.alpha).fit(x_tr, resid)
        p_res = p_base + m.predict(x_te)
        out["baseline_plus_ridge"] = eval_pack(te, y_te, p_res, "基准 + 残差岭")

        # 4) 基准 + 残差岭 + 轮胎（离线对照，暂不上线）
        ty_tr = np.array([tyre_vector(r) for r in tr])
        ty_te = np.array([tyre_vector(r) for r in te])
        m2 = Ridge(alpha=args.alpha).fit(np.hstack([x_tr, ty_tr]), resid)
        p_res2 = p_base + m2.predict(np.hstack([x_te, ty_te]))
        out["baseline_plus_ridge_tyre"] = eval_pack(te, y_te, p_res2, "基准 + 残差岭 + 轮胎上下文")
        return out

    tr_idx = [i for i, r in enumerate(rows) if r["track_id"] in splits["train"]]
    va_idx = [i for i, r in enumerate(rows) if r["track_id"] in splits["val"]]
    te_idx = [i for i, r in enumerate(rows) if r["track_id"] in splits["test"]]

    report["experiments"]["by_track_val"] = run_experiment("val", tr_idx, va_idx)
    report["experiments"]["by_track_test"] = run_experiment("test", tr_idx, te_idx)
    report["experiments"]["by_track_valtest"] = run_experiment("valtest", tr_idx, va_idx + te_idx)

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = int(0.2 * len(rows))
    report["experiments"]["random_holdout"] = run_experiment("random", idx[n_val:], idx[:n_val])

    for exp_name, exp in report["experiments"].items():
        print(f"\n== {exp_name} (train {exp['train_n']} / test {exp['test_n']})")
        for key, val in exp.items():
            if isinstance(val, dict) and "mae_share" in val:
                print(f"  {key:26s} MAE {val['mae_share']:.6f}  R² {val['r2']:+.4f}  "
                      f"±{val['mae_seconds']:.4f}s")

    # ---------------- 正式产物：全 21 站训练 ---------------- #
    if args.experiment_only:
        (ROOT / "work").mkdir(exist_ok=True)
        (ROOT / "work" / "telemetry_v3_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("仅实验，未写模型")
        return 0

    all_shrunk, all_class = baseline_table(rows, k=args.shrink_k)
    global_med = statistics.median(list(all_shrunk.values()))
    base_all = np.array([
        all_shrunk.get(pair_key(r), all_class.get(str(r.get("corner_class")), global_med))
        for r in rows
    ])
    resid_all = y - base_all
    x_all = np.array([condition_vector(r) for r in rows])
    final = Ridge(alpha=args.alpha).fit(x_all, resid_all)

    # 逐弯期望时间与重要度（全量中位数预测）
    pred_all = base_all + final.predict(x_all)
    by_track_corner: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r, p in zip(rows, pred_all, strict=True):
        by_track_corner[r["track_id"]][str(r["corner"])].append(float(p))
    median_lap = {}
    for tid in {r["track_id"] for r in rows}:
        median_lap[tid] = statistics.median(
            [float(r["lap_time_s"]) for r in rows if r["track_id"] == tid])
    expected = {
        tid: {c: round(statistics.median(v) * median_lap[tid], 4) for c, v in cs.items()}
        for tid, cs in by_track_corner.items()
    }
    importance = {}
    for tid, cs in by_track_corner.items():
        med = {c: statistics.median(v) for c, v in cs.items()}
        tot = sum(med.values()) or 1.0
        importance[tid] = {c: round(v / tot, 6) for c, v in med.items()}
    cls_share: dict[str, float] = defaultdict(float)
    for r in rows:
        cls_share[str(r.get("corner_class"))] += float(r["corner_time_s"])
    tot = sum(cls_share.values()) or 1.0
    cls_share = {k: round(v / tot, 4) for k, v in cls_share.items()}

    metrics = report["experiments"]["random_holdout"]["baseline_plus_ridge"]
    bt = report["experiments"]["by_track_valtest"]["baseline_plus_ridge"]
    payload = {
        "schema": "f1opt-telemetry-surrogate/1",
        "source": "F1 2026 official timing (21 rounds, clean laps) + OpenF1 cross-check",
        "corner_head": {
            "feature_keys": list(CONDITION_KEYS),
            "pair_keys": [],
            "track_ids": sorted({r["track_id"] for r in rows}),
            "target": "corner_time_share",
            "n_samples": len(rows),
            "tracks": sorted(by_track_corner),
            "corner_count_by_track": {
                t: max(int(c) for c in cs) for t, cs in by_track_corner.items()
            },
            "family": {
                "ridge": {"model": final.to_dict(), "metrics": {
                    "val_mae": metrics["mae_share"], "val_r2": metrics["r2"],
                    "train_samples": len(rows), "n_samples_total": len(rows),
                    "holdout": "random 20% in-distribution",
                    "by_track_mae": bt["mae_share"], "by_track_r2": bt["r2"],
                    "by_track_note": "15 训练站 / 6 留出站；跨站不迁移是已知边界",
                }},
                "winner": "ridge",
            },
            "mlp": None,
            "active": {"kind": "ridge", "metrics": {
                "val_mae": metrics["mae_share"], "val_r2": metrics["r2"],
                "train_samples": len(rows), "holdout": "random 20% in-distribution",
                "by_track_mae": bt["mae_share"], "by_track_r2": bt["r2"],
            }},
            "corner_baseline": {
                "shrink_k": args.shrink_k,
                "by_pair": {k: round(v, 6) for k, v in all_shrunk.items()},
                "by_class": {k: round(v, 6) for k, v in all_class.items()},
                "global": round(global_med, 6),
            },
            "expected_corner_time_s": expected,
            "corner_importance": importance,
            "corner_class_time_share": cls_share,
        },
        "lap_head": (old_head and json.loads(DEFAULT_OUT.read_text(encoding="utf-8")).get("lap_head")) or {},
    }

    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    report["artifact"] = {
        "path": str(out), "bytes": out.stat().st_size,
        "tracks": len(payload["corner_head"]["tracks"]),
        "pairs": len(all_shrunk),
        "expected_metrics": metrics,
    }
    (ROOT / "work" / "telemetry_v3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out} ({out.stat().st_size/1e6:.2f} MB)，覆盖 {len(payload['corner_head']['tracks'])} 站")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
