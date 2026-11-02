"""采样策略的单元测试（路径管密采样 + 近似最优点）。

背景：模型以前在全量程均匀采样上训练，学到的精度集中在「糟糕 vs 极糟糕」，
而调教优化只在最优点附近几十毫秒的尺度上做决策。这里锁死新的采样契约：
候选值必须落在真实档位上、样本必须沿「默认→最优」路径展开、
近似最优搜索必须确定性且不劣于默认调教。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def _dataset_module() -> Any:
    """运行时注入路径后导入数据集构建模块。

    ``scripts/`` 不属于安装包（只在仓库里），因此不能在文件顶部直接 import。
    """
    for extra in (ROOT, ROOT / "scripts"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import build_setup_sim_dataset

    return build_setup_sim_dataset


B = _dataset_module()


def _fake_track() -> dict[str, Any]:
    """中性混合赛道（与 print_calibration 的演示赛道同构）。"""
    return {
        "corners": [
            {"corner": 1, "klass": "slow", "t_corner": 4.5, "brake_pct": 55.0},
            {"corner": 2, "klass": "medium", "t_corner": 5.0, "brake_pct": 30.0},
            {"corner": 3, "klass": "fast", "t_corner": 4.0, "brake_pct": 12.0},
            {"corner": 4, "klass": "medium", "t_corner": 5.5, "brake_pct": 35.0},
        ],
        "lap_time_s": 90.0,
        "straight_share": 0.40,
        "drag_scale": 0.40,
    }


class TestSnapToStep:
    """候选值必须落在车手真的能设出来的档位上。"""

    def test_lands_on_step_grid_and_within_bounds(self) -> None:
        for spec in ALL_SETUP_FIELDS:
            for probe in (-1.9, -0.37, 0.0, 0.13, 0.62, 3.1):
                norm = B.snap_to_step(probe, spec)
                real = spec.default + norm * B._half_range(spec)
                assert spec.min_val - 1e-9 <= real <= spec.max_val + 1e-9, spec.name
                steps = (real - spec.min_val) / spec.step
                assert abs(steps - round(steps)) < 1e-6, (spec.name, real)
                assert -1.0 - 1e-9 <= norm <= 1.0 + 1e-9, spec.name

    def test_is_idempotent(self) -> None:
        for spec in ALL_SETUP_FIELDS:
            once = B.snap_to_step(0.314, spec)
            assert B.snap_to_step(once, spec) == pytest.approx(once, abs=1e-12)


class TestCoarseOptimum:
    """近似最优搜索：确定性、不劣于默认、结果在合法域内。"""

    def test_is_deterministic(self) -> None:
        track = _fake_track()
        specs = list(ALL_SETUP_FIELDS)
        first = B.coarse_optimum(track, specs)
        second = B.coarse_optimum(track, specs)
        assert first[0] == second[0]
        assert first[1] == pytest.approx(second[1], abs=1e-9)

    def test_never_worse_than_default(self) -> None:
        track = _fake_track()
        specs = list(ALL_SETUP_FIELDS)
        _, best = B.coarse_optimum(track, specs)
        default = B.simulate_lap_delta_ms(track, {s.name: 0.0 for s in specs})
        assert default == pytest.approx(0.0, abs=1e-6)
        assert best <= 1e-9

    def test_optimum_stays_in_domain(self) -> None:
        track = _fake_track()
        specs = list(ALL_SETUP_FIELDS)
        optimum, _ = B.coarse_optimum(track, specs)
        assert set(optimum) == {s.name for s in specs}
        for name, norm in optimum.items():
            assert -1.0 - 1e-9 <= norm <= 1.0 + 1e-9, name


class TestTierCounts:
    """分档配额必须恰好用满，且比例符合声明。"""

    def test_sums_to_total(self) -> None:
        for total in (0, 1, 3, 100, 396, 400):
            counts = B._tier_counts(total)
            assert len(counts) == len(B._SAMPLE_TIERS)
            assert sum(counts) == total
            assert all(c >= 0 for c in counts)

    def test_first_tier_dominates(self) -> None:
        counts = B._tier_counts(396)
        # 最内档（路径管主力区）应占大头：调优决策就发生在这里
        assert counts[0] == pytest.approx(396 * B._SAMPLE_TIERS[0][1], abs=2)
        assert counts[0] > counts[-1]


class TestPathTube:
    """采样覆盖「默认→最优」路径的管状邻域，而非最优解一个点。"""

    def test_tiers_are_well_formed(self) -> None:
        assert len(B._SAMPLE_TIERS) == 3
        for half_width, share, t_max in B._SAMPLE_TIERS:
            assert share > 0
            if half_width is None:
                continue
            assert half_width > 0
            # 路径位置上限必须覆盖整条路径（t=1 即近似最优解）
            assert t_max >= 1.0

    def test_shares_sum_to_one(self) -> None:
        assert sum(t[1] for t in B._SAMPLE_TIERS) == pytest.approx(1.0)

    def test_dominant_tier_is_the_narrow_one(self) -> None:
        # 主力档必须是最窄抖动的那一档：调优决策就发生在窄邻域里
        assert B._SAMPLE_TIERS[0][0] == min(
            t[0] for t in B._SAMPLE_TIERS if t[0] is not None
        )
        assert B._SAMPLE_TIERS[0][1] == max(t[1] for t in B._SAMPLE_TIERS)
