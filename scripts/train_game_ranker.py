"""用 F1 26 游戏调教语料训练「调教 → 圈速分位」排序模型。

数据
----
- 主：`data/f1_2026/game_setups/f1laps_26.jsonl`（本次抓取，含圈速/工况/车队）
- 辅：`data/f1_2026/train/setups_f1_26.jsonl`（仓库已有 899 条，无工况字段）

标签口径
--------
同一 (赛道, 工况=干, 会话组) 内按圈速排名，转成 0–1 分位（**0 = 最快**）。
分位化可以消掉赛道长度、车、驾驶水平带来的绝对值差异，只保留"这套调教在该站
处于什么位置"。

模型
----
1. **kNN（主）**：同赛道内按归一化调教距离取 k 个邻居，距离加权平均分位。
   不需要假设函数形式，直接复现"这站快的人都怎么调"。
2. **岭回归（兜底）**：全局线性模型 + 赛道哑变量，用于没见过的赛道或样本不足。

评估
----
- 赛道内 5 折：预测分位与实际分位的 Spearman 相关、top25% 命中率；
- 赛道留出：全局岭回归在没见过的站上的表现（冷启动能力）。

产物
----
`setup_tuner/resources/models/game_ranker.json`（每站样本 + 全局系数）。
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from setup_tuner.domain.setup import ALL_SETUP_FIELDS  # noqa: E402

MAIN = ROOT / "data" / "f1_2026" / "game_setups" / "f1laps_26.jsonl"
AUX = ROOT / "data" / "f1_2026" / "train" / "setups_f1_26.jsonl"
OUT = ROOT / "setup_tuner" / "resources" / "models" / "game_ranker.json"
REPORT = ROOT / "work" / "game_ranker_report.json"

NAMES = [f.name for f in ALL_SETUP_FIELDS]
#: 哪些会话的圈速可以当"纯速度"参考（Race 有油量/轮胎差异，排除）
PACE_SESSIONS = {"Time Trial", "Qualifying", "unknown"}
K_NEIGHBORS = 12


def load_rows() -> list[dict]:
    rows: list[dict] = []
    if MAIN.exists():
        for line in MAIN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("conditions") != "dry":
                continue
            if r.get("session") not in PACE_SESSIONS:
                continue
            rows.append({"track": r["track_id"], "setup": r["setup"],
                         "lap": r["lap_time_ms"], "src": "f1laps"})
    if AUX.exists():
        for line in AUX.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            setup = r.get("setup") or {}
            lap = r.get("lap_time_ms")
            if not lap or len(setup) < 20:
                continue
            rows.append({"track": r["track_id"], "setup": setup, "lap": lap,
                         "src": "f1laps-899"})
    return rows


def normalized(setup: dict) -> np.ndarray:
    out = np.zeros(len(NAMES), dtype=np.float64)
    for i, f in enumerate(ALL_SETUP_FIELDS):
        v = float(setup.get(f.name, f.default))
        half = max(f.max_val - f.default, f.default - f.min_val) or 1.0
        out[i] = (v - f.default) / half
    return out


def percentiles(rows: list[dict]) -> dict[str, list[float]]:
    """按赛道内圈速排名 → 0–1 分位。"""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["track"]].append(r)
    out: dict[str, list[float]] = {}
    for track, items in by.items():
        items.sort(key=lambda r: r["lap"])
        n = len(items)
        out[track] = [i / max(1, n - 1) for i in range(n)]
    return out


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


def knn_predict(xq: np.ndarray, xs: np.ndarray, ys: np.ndarray, k: int) -> float:
    d = np.sqrt(((xs - xq) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 1e-3)
    return float((w * ys[idx]).sum() / w.sum())


class Ridge:
    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        z = (x - self.mean) / self.std
        self.intercept = float(y.mean())
        d = z.shape[1]
        self.coef = np.linalg.solve(z.T @ z + self.alpha * np.eye(d), z.T @ (y - self.intercept))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return z @ self.coef + self.intercept


def main() -> int:
    rows = load_rows()
    pcts = percentiles(rows)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["track"]].append(r)
    for track, items in by.items():
        for r, p in zip(sorted(items, key=lambda r: r["lap"]), pcts[track], strict=False):
            r["pct"] = p

    per_track: dict[str, dict] = {}
    report: dict = {"tracks": {}, "overall": {}}
    rho_all: list[float] = []
    hit_all: list[int] = []
    for track, items in sorted(by.items()):
        if len(items) < 12:
            continue
        x = np.array([normalized(r["setup"]) for r in items])
        y = np.array([r["pct"] for r in items])
        # 5 折交叉验证（确定性取模分组）
        pred = np.zeros_like(y)
        for fold in range(5):
            mask = np.arange(len(y)) % 5 == fold
            if mask.all() or (~mask).sum() < K_NEIGHBORS:
                pred[mask] = y[mask]
                continue
            for i in np.where(mask)[0]:
                pred[i] = knn_predict(x[i], x[~mask], y[~mask], K_NEIGHBORS)
        rho = spearman(list(pred), list(y))
        top = y <= 0.25
        hit = int((pred[top] <= 0.35).sum())
        rho_all.append(rho)
        hit_all.append(hit)
        per_track[track] = {
            "n": len(items),
            "rho": round(rho, 3),
            "top25_hit": round(hit / max(1, int(top.sum())), 3),
            "setups": [[round(v, 3) for v in row] for row in x],
            "pcts": [round(v, 4) for v in y],
        }
        report["tracks"][track] = {"n": len(items), "rho": round(rho, 3),
                                   "top25_hit": round(hit / max(1, int(top.sum())), 3)}

    # 全局兜底：赛道哑变量 + 调教
    tracks = sorted(per_track)
    xs, ys = [], []
    for ti, t in enumerate(tracks):
        for row, p in zip(per_track[t]["setups"], per_track[t]["pcts"], strict=False):
            onehot = [0.0] * len(tracks)
            onehot[ti] = 1.0
            xs.append(row + onehot)
            ys.append(p)
    xg = np.array(xs)
    yg = np.array(ys)
    ridge = Ridge(alpha=5.0).fit(xg, yg)
    pred_g = ridge.predict(xg)
    rho_global = spearman(list(pred_g), list(yg))

    payload = {
        "schema": "f1opt-game-ranker/1",
        "source": "f1laps F1 26 dry pace sessions",
        "features": NAMES,
        "k": K_NEIGHBORS,
        "tracks": per_track,
        "ridge": {
            "mean": [round(float(v), 4) for v in ridge.mean],
            "std": [round(float(v), 4) for v in ridge.std],
            "coef": [round(float(v), 4) for v in ridge.coef],
            "intercept": round(float(ridge.intercept), 4),
            "track_ids": tracks,
        },
        "metrics": {
            "median_rho": round(statistics.median(rho_all), 3) if rho_all else 0.0,
            "tracks": len(per_track),
            "rows": len(rows),
            "global_rho_in_sample": round(rho_global, 3),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    report["overall"] = payload["metrics"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本 {len(rows)} 条 / {len(per_track)} 站 → {OUT}（{OUT.stat().st_size/1024:.0f} KB）")
    print(f"赛道内 Spearman 中位 {payload['metrics']['median_rho']}；全局 {rho_global:.3f}")
    for t in tracks[:6]:
        m = report["tracks"][t]
        print(f"  {t:12s} n={m['n']:3d} rho={m['rho']:+.3f} top25命中={m['top25_hit']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
