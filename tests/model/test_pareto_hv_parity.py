"""Opt-041/042 parity tests — 云端全量验证新算法与原始实现逐位一致。

* ``_hv_3d``: 增量 skyline z-sweep (Opt-041) vs 参考实现
  (Opt-041 前的每片全量 sort + ``_hv_2d`` 扫描算法, 此处以参考实现形式保留)。
* ``compute_front`` / ``compute_front_with_metadata``: numpy 向量化
  (Opt-042) vs 纯 Python 回退路径 (``_compute_front_loop``)。

fuzz 覆盖: 随机点、重复点、坐标并列、最大化方向翻转、参考点贴近/远离前沿。
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from f1opt.model.pareto import ParetoFront, _hv_2d, _hv_3d


# --------------------------------------------------------------------------- #
# 参考实现: Opt-041 前的原始 _hv_3d (z 片 sweep, 每片对累积点集全量 _hv_2d)。
# 仅用于 parity 对照, 不进入生产路径。
# --------------------------------------------------------------------------- #
def _hv_3d_reference(points: list[list[float]], ref: list[float]) -> float:
    pts = sorted(points, key=lambda p: (p[2], p[0], p[1]))
    ref2d = [ref[0], ref[1]]
    hv = 0.0
    prev_z = pts[0][2]
    front2d: list[list[float]] = [[pts[0][0], pts[0][1]]]
    for i in range(1, len(pts)):
        z_i = pts[i][2]
        hv += _hv_2d(front2d, ref2d) * (z_i - prev_z)
        front2d.append([pts[i][0], pts[i][1]])
        prev_z = z_i
    hv += _hv_2d(front2d, ref2d) * (ref[2] - prev_z)
    return float(hv)


def _random_points(
    rng: random.Random, n: int, tie_prob: float = 0.2
) -> list[list[float]]:
    """生成 [0, 10]³ 内的随机点; 以 tie_prob 概率复用旧坐标制造并列/重复。"""
    pts: list[list[float]] = []
    for _ in range(n):
        if pts and rng.random() < tie_prob:
            src = rng.choice(pts)
            pts.append([src[0], src[1], src[2]])
        else:
            pts.append([rng.uniform(0.0, 10.0) for _ in range(3)])
    return pts


class TestHv3dParity:
    """Opt-041: 增量 skyline 实现与参考实现 fuzz 一致。"""

    @pytest.mark.parametrize("seed", range(40))
    def test_fuzz_parity_random(self, seed: int) -> None:
        rng = random.Random(seed)
        n = rng.randint(1, 60)
        pts = _random_points(rng, n)
        ref = [rng.uniform(9.0, 12.0), rng.uniform(9.0, 12.0), rng.uniform(9.0, 12.0)]
        got = _hv_3d([list(p) for p in pts], list(ref))
        want = _hv_3d_reference([list(p) for p in pts], list(ref))
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)

    def test_exact_small_cases(self) -> None:
        # 单点立方体
        assert _hv_3d([[1.0, 1.0, 1.0]], [5.0, 5.0, 5.0]) == pytest.approx(64.0)
        # 两点并列 z → 与参考一致
        pts = [[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]]
        assert _hv_3d(pts, [5.0, 5.0, 5.0]) == pytest.approx(
            _hv_3d_reference(pts, [5.0, 5.0, 5.0])
        )
        # 完全重复点
        pts = [[1.0, 2.0, 3.0]] * 5
        assert _hv_3d(pts, [6.0, 6.0, 6.0]) == pytest.approx(
            _hv_3d_reference(pts, [6.0, 6.0, 6.0])
        )
        # 支配链 (1,1,1) 支配 (2,2,2) 支配 (3,3,3) → 并集 = (1,1,1)→ref 的盒子
        pts = [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]
        want = 3.0 * 4.0 * 5.0
        assert _hv_3d(pts, [4.0, 5.0, 6.0]) == pytest.approx(
            _hv_3d_reference(pts, [4.0, 5.0, 6.0])
        )
        assert _hv_3d(pts, [4.0, 5.0, 6.0]) == pytest.approx(want)

    def test_via_paretofront_maximize(self) -> None:
        """经 ParetoFront 入口 (含 maximize 方向翻转) 与参考一致。"""
        rng = random.Random(7)
        pf = ParetoFront(["a", "b", "c"], maximize=[True, False, True])
        for _ in range(30):
            pf.add_sample([rng.uniform(0, 10) for _ in range(3)])
        ref = [rng.uniform(9, 12) for _ in range(3)]
        # 最小化空间变换后走参考实现。
        pts = [
            [
                -v[0] if pf.maximize[0] else v[0],
                -v[1] if pf.maximize[1] else v[1],
                -v[2] if pf.maximize[2] else v[2],
            ]
            for v, _ in pf._samples
        ]
        ref_min = [
            -ref[0] if pf.maximize[0] else ref[0],
            -ref[1] if pf.maximize[1] else ref[1],
            -ref[2] if pf.maximize[2] else ref[2],
        ]
        assert pf.hypervolume(ref) == pytest.approx(
            _hv_3d_reference(pts, ref_min), rel=1e-12, abs=1e-12
        )


class TestComputeFrontParity:
    """Opt-042: 向量化非支配排序 vs 纯 Python 回退路径。"""

    @pytest.mark.parametrize("seed", range(40))
    def test_fuzz_parity_front(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 80))
        vals = rng.uniform(0.0, 10.0, size=(n, 3)).round(1)  # round 制造并列
        pf = ParetoFront(["x", "y", "z"])
        for row in vals:
            pf.add_sample(row.tolist())

        vec_front = pf.compute_front()
        loop_front = pf._compute_front_loop()
        assert vec_front == loop_front

        vec_meta = pf.compute_front_with_metadata()
        for i in vec_front:
            entry = next(m for m in vec_meta if m["index"] == i)
            dominated_by = sum(
                1
                for j in range(n)
                if j != i and pf._dominates(j, i)
            )
            assert entry["dominated_by_count"] == dominated_by
            assert entry["values"] == [float(v) for v in vals[i]]

    def test_maximize_direction_parity(self) -> None:
        rng = np.random.default_rng(3)
        pf = ParetoFront(["a", "b"], maximize=[True, False])
        for row in rng.uniform(0, 10, size=(25, 2)).round(1):
            pf.add_sample(row.tolist())
        assert pf.compute_front() == pf._compute_front_loop()

    def test_empty_front(self) -> None:
        pf = ParetoFront(["x"])
        assert pf.compute_front() == []
        assert pf.compute_front_with_metadata() == []
