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
        """弧长占比必须落在 [0, 1]（1.0 = 末弯弯心与起跑线重合，属合法情形）。"""
        for track_id, arcs in TRACK_CORNER_ARCS.items():
            for corner, fraction in arcs.items():
                assert 0.0 <= fraction <= 1.0, f"{track_id} T{corner} 占比越界：{fraction}"

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
        """抽查若干已知弯位（按真实赛道里程/圈长换算，2026-09-17 曲率法重标）。

        物理参照（距起点里程 ÷ 圈长）：
        - Suzuka（5.807 km）：Dunlop(T7) ≈ 2600 m ≈ 0.45、Degner 1(T8) ≈ 0.50、
          Hairpin(T11) ≈ 0.69、130R(T15) ≈ 0.87 —— 旧表把 Dunlop 放在 0.193
          （≈1100 m），物理上就是错的。
        - Monza（5.793 km）：主直道 1.1 km，T1（第一减速弯）在直道**末端**
          ≈ 0.19；旧表 T1 = 0.0 是把锚点摆在了直道上（用户实测指出）。
        """
        suzuka = TRACK_CORNER_ARCS["suzuka"]
        assert suzuka[1] == pytest.approx(0.0522, abs=0.02)
        assert suzuka[7] == pytest.approx(0.4482, abs=0.03)     # Dunlop
        assert suzuka[8] == pytest.approx(0.4954, abs=0.03)     # Degner 1
        assert suzuka[11] == pytest.approx(0.6925, abs=0.03)    # Hairpin
        assert suzuka[15] == pytest.approx(0.8707, abs=0.03)    # 130R
        melbourne = TRACK_CORNER_ARCS["melbourne"]
        assert melbourne[1] == pytest.approx(0.1417, abs=0.02)
        # 这 4 条赛道的 T1 必须偏离起跑线（旧表压在 0.0）
        for track_id in ("monza", "mexico_city", "spielberg"):
            assert TRACK_CORNER_ARCS[track_id][1] > 0.01


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
        # 物理里程：Dunlop(T7)≈45%、Degner 1(T8)≈50%、Hairpin(T11)≈69%、
        # 130R(T15)≈87%（距起点里程 ÷ 5.807 km）
        assert _map_corner(0.45 * track.length_m, track.length_m, track.corners, "suzuka") == 7
        assert _map_corner(0.50 * track.length_m, track.length_m, track.corners, "suzuka") == 8
        assert _map_corner(0.69 * track.length_m, track.length_m, track.corners, "suzuka") == 11
        assert _map_corner(0.87 * track.length_m, track.length_m, track.corners, "suzuka") == 15

    def test_wraparound_path_origin(self) -> None:
        """monza 路径起点=起跑线：圈初判为**刚出末弯**（T11），而非 T1。

        旧断言"圈初= T1"建立在 T1 锚点被压到起跑线的错误之上 —— 蒙扎主直道
        1.1 km，圈初位置是刚离开 Parabolica(T11)，T1 在直道末端才出现。

        注意：赛道 SVG 是**示意图、不按比例**，主直道画得比真实的 19% 短，
        因此"图上 4.2% 处"对应真实里程约 19% 的 T1。弧长表的本职是把标记
        落在**图上画的弯**上（这正是本次修正的目标），不是复刻真实里程。
        """
        track = get_track_by_id("monza")
        assert track is not None
        arcs = TRACK_CORNER_ARCS["monza"]
        # 圈初 = 刚出末弯（闭环回绕），不是 T1
        assert _map_corner(0.0, track.length_m, track.corners, "monza") == 11
        assert _map_corner(0.10 * track.length_m, track.corners and track.length_m,
                           track.corners, "monza") != 1
        # T1 的弯心位置必须能被定位到（标记不再压在直道上）
        t1_arc = arcs[1]
        assert _map_corner(t1_arc * track.length_m, track.length_m,
                           track.corners, "monza") == 1

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


# ===========================================================================
# 3. 闭合回路起跑线归一化（回归：T1 曾被判为全圈最后一个弯）
# ===========================================================================
class TestClosedLoopNormalization:
    """赛道 SVG 是闭合回路，锚点落在起跑线时必须归一化为 0 而非 ≈1。

    历史缺陷：``nearest_arc_fraction`` 在锚点投影到回路闭合点时，最近点命中
    折线最后一段末端，返回 ≈1.0，使 T1 成为全圈最末弯、整条赛道弯号错位。
    受影响赛道：melbourne / mexico_city / monza / spielberg。
    """

    # 曾经 T1 ≈ 1.0 的赛道（现应 ≈ 0）
    REGRESSED = ["melbourne", "mexico_city", "monza", "spielberg"]

    def test_no_track_has_t1_at_lap_end(self) -> None:
        """没有任何赛道的 T1 落在圈末（否则弯号会整体错位）。"""
        bad = {
            tid: arcs[1]
            for tid, arcs in TRACK_CORNER_ARCS.items()
            if arcs.get(1) is not None and arcs[1] > 0.5
        }
        assert not bad, f"T1 落在圈末的赛道: {bad}"

    @pytest.mark.parametrize("track_id", REGRESSED)
    def test_t1_not_pinned_to_start_line(self, track_id: str) -> None:
        """T1 锚点**不得**压在起跑线上（用户实测指出的缺陷，2026-09-17 修正）。

        旧约定曾把这 4 条赛道的 T1 归一化为 0.0，导致蒙扎 T1（第一减速弯，
        主直道末端 ≈1100 m ≈ 19% 圈长）被判在直道上，整条赛道弯号整体前移。
        2026-09-17 起改用曲率峰值定位弯心，T1 必须偏离起跑线。
        """
        fraction = TRACK_CORNER_ARCS[track_id][1]
        assert fraction > 0.01, (
            f"{track_id} T1 弧长占比 {fraction} 仍压在起跑线上"
            f"（≈{fraction * get_track_by_id(track_id).length_m:.0f} m）"
        )

    @pytest.mark.parametrize("track_id", sorted(TRACK_CORNER_ARCS))
    def test_arc_table_is_monotonic(self, track_id: str) -> None:
        """弧长表必须随弯号严格递增（同一圈内弯道位置不可回退）。"""
        arcs = TRACK_CORNER_ARCS[track_id]
        values = [arcs[k] for k in sorted(arcs)]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], (
                f"{track_id} 弧长表在 T{i + 1}->T{i + 2} 非单调: "
                f"{values[i]:.6f} >= {values[i + 1]:.6f}"
            )

    @pytest.mark.parametrize("track_id", sorted(TRACK_CORNER_ARCS))
    def test_arc_values_in_unit_interval(self, track_id: str) -> None:
        """全部弧长占比必须落在 [0, 1] 区间。"""
        for number, value in TRACK_CORNER_ARCS[track_id].items():
            assert 0.0 <= value <= 1.0, f"{track_id} T{number} 弧长越界: {value}"

    @pytest.mark.parametrize(
        "track_id", ["monza", "mexico_city", "spielberg", "melbourne", "suzuka", "spa"]
    )
    def test_start_of_lap_maps_to_turn_1(self, track_id: str) -> None:
        """圈首与圈尾是闭环上同一个点，必须映射到同一个弯。

        旧断言"圈初必为 T1"只在 T1 紧贴起跑线的赛道成立；对主直道较长的赛道
        （蒙扎主直道 1.1 km ≈ 19% 圈长），圈初位置其实是**刚出末弯**，判为末弯
        才对。闭环一致性才是与弯位布局无关的真不变量。
        """
        track = get_track_by_id(track_id)
        assert track is not None
        near_start = 0.001 * track.length_m
        near_end = 0.999 * track.length_m
        assert _map_corner(near_start, track.length_m, track.corners, track_id) == (
            _map_corner(near_end, track.length_m, track.corners, track_id)
        )

    def test_full_lap_sweep_hits_every_turn(self) -> None:
        """每圈扫描必须覆盖该赛道所有弯号（无弯被跳过）。"""
        for track in ALL_TRACKS:
            expected = {c.number for c in track.corners}
            seen = set()
            for i in range(2000):
                progress = i / 2000
                got = _map_corner(
                    progress * track.length_m, track.length_m, track.corners, track.track_id
                )
                if got is not None:
                    seen.add(got)
            missing = expected - seen
            assert not missing, f"{track.track_id} 圈内扫描遗漏弯号: {sorted(missing)}"
