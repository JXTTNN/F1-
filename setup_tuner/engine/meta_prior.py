"""游戏 meta 先验：把「该站快调教长什么样」接进优化器。

数据来源：`setup_tuner/resources/models/game_ranker.json`（由
`scripts/train_game_ranker.py` 从 F1 26 游戏调教语料训练）。

提供三件事
----------
1. ``percentile(track, setup)`` —— 这套调教在该站的历史分位（0 = 最快）；
2. ``meta_target(track)`` —— 该站前 25% 调教的中位数（归一化坐标）；
3. ``pull_toward_meta(track, delta, current_setup)`` —— 把当前建议按比例拉向
   meta 目标，并按各参数 ``max_delta`` 夹紧。

为什么需要它
------------
物理模型对翼角的敏感度被压平（上一轮为防止"加翼永远更快"而加阻力），
导致建议长期停在默认附近：摩纳哥前后翼 25/25，而游戏内 41 套快调教的中位数是 50/50。
语料统计显示"贴近该站 meta 中位数"的调教在干地条件下中位快 206 ms（19/24 站为正），
所以 meta 先验是比继续调物理系数更强的信号。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ..domain.setup import ALL_SETUP_FIELDS

logger = logging.getLogger(__name__)

MODEL_NAME = "game_ranker.json"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "resources" / "models" / MODEL_NAME
#: 拉向 meta 的比例（0 = 不动，1 = 直接跳到 meta 中位数）
PULL_RATIO = 0.55
#: 只在建议明显偏慢时动手（分位阈值，0.5 = 比历史中位数还慢）
TRIGGER_PERCENTILE = 0.5


class GameRanker:
    """游戏 meta 排序模型的运行时封装（懒加载单例）。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self.available = False
        self.tracks: dict[str, dict] = {}
        self._ridge: dict | None = None
        self._targets: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            logger.info("游戏 meta 先验不存在（%s），按纯物理规则运行", self.path)
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("meta 先验读取失败：%s", self.path, exc_info=True)
            return
        self.tracks = data.get("tracks") or {}
        self._ridge = data.get("ridge")
        self.k = int(data.get("k") or 12)
        for track, entry in self.tracks.items():
            rows = entry.get("setups") or []
            pcts = entry.get("pcts") or []
            top = [r for r, p in zip(rows, pcts, strict=False) if p <= 0.25]
            if not top:
                continue
            n = len(top[0])
            self._targets[track] = [
                sorted(r[i] for r in top)[len(top) // 2] for i in range(n)
            ]
        self.available = bool(self.tracks)

    # ---------------- 推理 ---------------- #
    def _vector(self, setup: dict[str, float]) -> list[float]:
        out = []
        for f in ALL_SETUP_FIELDS:
            v = float(setup.get(f.name, f.default))
            half = max(f.max_val - f.default, f.default - f.min_val) or 1.0
            out.append((v - f.default) / half)
        return out

    def percentile(self, track_id: str, setup: dict[str, float]) -> float | None:
        """该调教在该站的历史分位（0 = 最快，1 = 最慢）。无数据返回 None。"""
        vec = self._vector(setup)
        entry = self.tracks.get(track_id)
        if entry and entry.get("setups"):
            rows = entry["setups"]
            pcts = entry["pcts"]
            dist = [
                sum((a - b) ** 2 for a, b in zip(vec, row, strict=False)) ** 0.5
                for row in rows
            ]
            order = sorted(range(len(dist)), key=lambda i: dist[i])[: self.k]
            w = [1.0 / (dist[i] + 1e-3) for i in order]
            return float(sum(wi * pcts[i] for wi, i in zip(w, order, strict=False)) / sum(w))
        if self._ridge:
            return self._ridge_predict(track_id, vec)
        return None

    def _ridge_predict(self, track_id: str, vec: list[float]) -> float | None:
        r = self._ridge or {}
        ids = r.get("track_ids") or []
        if track_id not in ids:
            return None
        mean, std = r["mean"], r["std"]
        coef, intercept = r["coef"], r["intercept"]
        onehot = [1.0 if t == track_id else 0.0 for t in ids]
        x = vec + onehot
        s = float(intercept)
        for j, c in enumerate(coef):
            s += c * (x[j] - mean[j]) / std[j]
        return float(s)

    def meta_target(self, track_id: str) -> list[float] | None:
        """该站前 25% 调教的中位数（归一化坐标）。"""
        return self._targets.get(track_id)

    def pull_toward_meta(
        self, track_id: str, current_setup: dict[str, float],
        ratio: float = PULL_RATIO,
    ) -> dict[str, float] | None:
        """返回拉向 meta 之后的绝对调教（未夹紧由调用方负责）。"""
        target = self.meta_target(track_id)
        if target is None:
            return None
        out = dict(current_setup)
        for i, f in enumerate(ALL_SETUP_FIELDS):
            half = max(f.max_val - f.default, f.default - f.min_val) or 1.0
            goal = f.default + target[i] * half
            goal = max(f.min_val, min(f.max_val, goal))
            cur = float(current_setup.get(f.name, f.default))
            out[f.name] = cur + ratio * (goal - cur)
        return out


_RANKER: GameRanker | None = None
_LOCK = threading.Lock()


def get_game_ranker(path: str | Path | None = None) -> GameRanker:
    global _RANKER
    if _RANKER is None:
        with _LOCK:
            if _RANKER is None:
                _RANKER = GameRanker(path)
    return _RANKER


def reset_game_ranker() -> None:
    global _RANKER
    with _LOCK:
        _RANKER = None
