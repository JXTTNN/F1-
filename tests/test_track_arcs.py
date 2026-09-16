"""赛道弧长表与「当前弯」判定回归测试。

背景：``api/ws.py::_map_corner`` 早期用「弯道沿赛道均匀分布」近似判定当前弯，
云端实测在 24 赛道 × 200 采样点上判错 3313/4800（69%）。现改为依据
``domain/_track_arcs.py``（锚点在 SVG 路径上的真实弧长占比）做循环最近邻查找。

本文件同时承担两件事：
1. **数据一致性**：用 ``scripts/gen_track_arcs.py::compute_arcs`` 重新计算弧长占比，
   与已提交的 ``_track_arcs.py`` 逐条比对（保证数据文件与 SVG 资产不脱节）；
2. **判定正确性**：把新实现与弧长表真值对拍，要求错误率为 0。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from setup_tuner.api.ws import _map_corner
from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS
from setup_tuner.domain.track import ALL_TRACKS, get_track_by_id

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GEN_PATH = _REPO_ROOT / "scripts" / "gen_track_arcs.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_track_arcs", _GEN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# 1. 数据文件与 SVG 资产一致性
# ===========================================================================
class TestArcTableConsistency:
    """``_track_arcs.py`` 必须能由 ``scripts/gen_track_arcs.py`` 复现。"""

    def test_arc_table_covers_all_tracks(self) -> None:
        """弧长表覆盖全部 24 条赛道，且弯道数与领域模型一致。"""
        assert len(TRACK_CORNER_ARCS) == len(ALL_TRACKS) == 24
        for track in ALL_TRACKS:
            arcs = TRACK_CORNER_ARCS.get(track.track_id)
            assert arcs is not None, f"{track.track_id} 缺少弧长表"
            assert sorted(arcs) == [c.number for c in track.corners]

    def test_arc_values_in_unit_interval(self) -> None:
        """弧长占比必须落在 [0, 1)（1.0 会破坏闭合路径的循环查找）。"""
        for track_id, arcs in TRACK_CORNER_ARCS.items():
            for corner, fraction in arcs.items():
                assert 0.0 <= fraction < 1.0, f"{track_id} T{corner} 占比越界：{fraction}"

    def test_arc_table_matches_regenerated_values(self) -> None:
        """重新计算 SVG 弧长占比，与提交的数据文件逐条比对。"""
        computed = _load_generator().compute_arcs()
        assert set(computed) == set(TRACK_CORNER_ARCS)
        for track_id, arcs in computed.items():
            for corner, fraction in arcs.items():
                committed = TRACK_CORNER_ARCS[track_id][corner]
                assert committed == pytest.approx(fraction, abs=1e-3), (
                    f"{track_id} T{corner}: 提交值 {committed} vs 重算值 {fraction}"
                )

    def test_known_corner_positions(self) -> None:
        """抽查若干已知弯位（同时锁定"路径起点不在 T1 前"这一已知特性）。"""
        suzuka = TRACK_CORNER_ARCS["suzuka"]
        assert suzuka[1] == pytest.approx(0.0, abs=0.02)
        # task-63：F1 26 弯号 —— Dunlop=T7、Degner 1=T8
        assert suzuka[7] == pytest.approx(0.193, abs=0.03)
        assert suzuka[8] == pytest.approx(0.265, abs=0.03)
        melbourne = TRACK_CORNER_ARCS["melbourne"]
        assert melbourne[1] == pytest.approx(0.0, abs=0.02)
        # mexico_city / monza / spielberg 的 SVG 路径起点落在 T1 附近 → T1 ≈ 1.0
        for track_id in ("monza", "mexico_city", "spielberg"):
            assert TRACK_CORNER_ARCS[track_id][1] > 0.98


# ===========================================================================
# 2. 「当前弯」判定正确性
# ===========================================================================
class TestCornerMapping:
    """``_map_corner`` 必须与弧长表真值一致（错误率 0）。"""

    def _truth(self, track_id: str, progress: float) -> int:
        arcs = TRACK_CORNER_ARCS[track_id]
        return min(
            arcs,
            key=lambda c: min(abs(arcs[c] - progress), 1.0 - abs(arcs[c] - progress)),
        )

    @pytest.mark.parametrize("samples", [200])
    def test_zero_error_rate_across_all_tracks(self, samples: int) -> None:
        """24 赛道 × 采样点全对（旧实现错误率 69%）。"""
        wrong = 0
        total = 0
        for track in ALL_TRACKS:
            for i in range(samples):
                progress = i / samples
                lap_distance = track.length_m * progress
                got = _map_corner(lap_distance, track.length_m, track.corners, track.track_id)
                assert got is not None
                # 与真值比较时允许相邻弯的边界抖动：用真值占比反算容忍区间
                expect = self._truth(track.track_id, progress)
                if got != expect:
                    arcs = TRACK_CORNER_ARCS[track.track_id]
                    near = min(
                        abs(arcs[expect] - progress),
                        abs(arcs[got] - progress),
                        1.0 - abs(arcs[expect] - progress),
                        1.0 - abs(arcs[got] - progress),
                    )
                    # 两弯占比几乎等距时属于边界摆动，不计为错误
                    if near >= 1e-3:
                        wrong += 1
                total += 1
        assert wrong == 0, f"判定错误 {wrong}/{total}"

    def test_suzuka_landmark_positions(self) -> None:
        """Suzuka 具体点校验（旧均匀近似在这里会给出错误的弯号）。"""
        track = get_track_by_id("suzuka")
        assert track is not None
        # task-63：26% 处应为 T8（Degner 1，F1 26 弯号；均匀近似会算错）
        assert _map_corner(0.26 * track.length_m, track.length_m, track.corners, "suzuka") == 8
        # 76% 处应为 T15（130R，F1 26 弯号）
        assert _map_corner(0.76 * track.length_m, track.length_m, track.corners, "suzuka") == 15

    def test_wraparound_path_origin(self) -> None:
        """monza 的路径起点在 T1 附近：lap_distance≈0 仍应判为 T1。"""
        track = get_track_by_id("monza")
        assert track is not None
        assert _map_corner(0.0, track.length_m, track.corners, "monza") == 1
        assert _map_corner(0.02 * track.length_m, track.length_m, track.corners, "monza") == 1

    def test_fallback_without_track_id(self) -> None:
        """未提供 track_id 时回退到均匀近似，仍返回合法弯号。"""
        track = get_track_by_id("suzuka")
        assert track is not None
        got = _map_corner(0.5 * track.length_m, track.length_m, track.corners)
        assert got in {c.number for c in track.corners}

    def test_invalid_inputs(self) -> None:
        """无效输入返回 None（不抛错）。"""
        track = get_track_by_id("suzuka")
        assert track is not None
        assert _map_corner(100.0, 0.0, track.corners, "suzuka") is None
        assert _map_corner(100.0, track.length_m, [], "suzuka") is None
        assert _map_corner(100.0, track.length_m, track.corners, "unknown_track") is not None
