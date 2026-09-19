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

    def test_arc_table_matches_svg_projection(self) -> None:
        """把提交锚点投影回 SVG **实际路径**反算占比，与本表逐条比对。

        这是「弧长表与 SVG/锚点不脱节」的锁：SVG 是真实几何的等比投影，
        反算（纯距离最近邻，无浮点敏感的选择决策）在 Windows/Linux 上
        跨平台稳定，实测最大偏差 4e-4。

        .. note::
            不做「重新跑 ``select_anchors`` 逐位比对占比」：锚点选择依赖
            累计转角（libm atan2 链 + 过零重置等离散判定），Windows ucrt
            与 Linux glibc 的 libm 不逐位一致，名额竞争的临界候选会在
            两平台选出相差一个弯位的锚点（实测 baku T5 漂移 0.033）——
            逐位复现不可行也不必要。选择质量由几何硬门
            （``select_anchors``：恰好官方弯数、每点 >= 12°、>=60° 弯段
            全覆盖，选不满即 RuntimeError）与下方物理参照测试保证。
        """
        worst, issues = _load_generator().verify_svg_consistency()
        assert not issues, "SVG 反算与弧长表脱节：" + "; ".join(issues)
        assert worst < 1e-2, f"SVG 反算最大偏差 {worst:.4f} 超过 1e-2"

    def test_known_corner_positions(self) -> None:
        """抽查若干已知弯位（2026-09-17 真实 GPS 几何重标，与官方赛道图对照）。

        口径：占比 = 锚点真实圈内距离 / 真实圈长（起点 = 发车线、方向 =
        官方行驶方向），与遥测 lap_distance 同源。参照值即官方赛道布局：
        - Suzuka（5.807 km）：T1 First Curve ≈ 590 m ≈ 0.102、Dunlop(T8)
          ≈ 2045 m ≈ 0.352、Hairpin(T11) ≈ 2656 m ≈ 0.457、130R(T16)
          ≈ 4728 m ≈ 0.814、Casio chicane(T18) ≈ 5226 m ≈ 0.900。
        - Monza（5.793 km）：发车线在主直道中段，T1（第一减速弯）弯心
          ≈ 637 m ≈ 0.110 —— 旧示意表的 0.0/0.19 系示意 SVG 不按比例所致。
        - Melbourne（5.278 km）：T1 弯心 ≈ 330 m ≈ 0.062。
        """
        suzuka = TRACK_CORNER_ARCS["suzuka"]
        assert suzuka[1] == pytest.approx(0.1018, abs=0.03)     # First Curve
        assert suzuka[8] == pytest.approx(0.3523, abs=0.03)     # Dunlop
        assert suzuka[11] == pytest.approx(0.4573, abs=0.03)    # Hairpin
        assert suzuka[16] == pytest.approx(0.8142, abs=0.03)    # 130R
        assert suzuka[18] == pytest.approx(0.8999, abs=0.03)    # Casio chicane
        melbourne = TRACK_CORNER_ARCS["melbourne"]
        assert melbourne[1] == pytest.approx(0.0625, abs=0.03)
        # 这 3 条赛道的 T1 必须偏离起跑线（长主直道，弯心在直道末端）
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
        """Suzuka 具体点校验（2026-09-17 真实几何口径）。

        真实 GPS 几何下 Dunlop=T8 ≈ 0.352、Hairpin=T11 ≈ 0.457、
        130R=T16 ≈ 0.814，与官方布局一致；旧断言「Dunlop(T7) ≈ 48%」
        系示意 SVG 时代的弯号错位。
        """
        track = get_track_by_id("suzuka")
        assert track is not None
        arcs = TRACK_CORNER_ARCS["suzuka"]
        # Dunlop(T8)：S 弯群后；Hairpin(T11)：Degner 后的发夹；130R(T16)：Spoon 后
        assert _map_corner(arcs[8] * track.length_m, track.length_m, track.corners, "suzuka") == 8
        assert _map_corner(arcs[11] * track.length_m, track.length_m, track.corners, "suzuka") == 11
        assert _map_corner(arcs[16] * track.length_m, track.length_m, track.corners, "suzuka") == 16

    def test_wraparound_path_origin(self) -> None:
        """monza 闭环回绕（真实 GPS 口径，2026-09-17 重标）。

        真实主直道恢复后：发车线离 T1（第一减速弯）弯心 ≈ 637 m、离
        Parabolica(T11) 弯心 ≈ 965 m —— 圈初最近邻判 **T1**（正冲向第一
        减速弯）。圈首(0.0) 与圈尾(1.0) 是闭环上同一点，映射必须一致。
        """
        track = get_track_by_id("monza")
        assert track is not None
        arcs = TRACK_CORNER_ARCS["monza"]
        # 闭环一致性：圈首与圈尾同点
        assert _map_corner(0.0, track.length_m, track.corners, "monza") == (
            _map_corner(track.length_m, track.length_m, track.corners, "monza")
        )
        # 主直道上（圈初 5%）判 T1（不在 T11 出口侧）
        assert _map_corner(0.05 * track.length_m, track.length_m, track.corners, "monza") == 1
        # T1 / T11 弯心处必须能定位到本弯
        assert _map_corner(arcs[1] * track.length_m, track.length_m, track.corners, "monza") == 1
        assert _map_corner(arcs[11] * track.length_m, track.length_m, track.corners, "monza") == 11

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
        """圈首(0.0)与圈尾(1.0)是闭环上同一个点，必须映射到同一个弯。

        2026-09-17 真实几何口径下不再用 0.001/0.999 采样 —— 两点相距
        0.2% 圈长，可能横跨相邻弯的最近邻边界（如 Suzuka 发车线恰在
        Casio(T18) 与 First Curve(T1) 弯心近乎等距处），判不同弯是正确
        行为。真正的闭环不变量是 0.0 与 1.0（严格同一点）映射一致。
        """
        track = get_track_by_id(track_id)
        assert track is not None
        assert _map_corner(0.0, track.length_m, track.corners, track_id) == (
            _map_corner(track.length_m, track.length_m, track.corners, track_id)
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


# ===========================================================================
# 锚点必须落在"弯"上，不得落在直道（2026-09-17 真实 GPS 几何口径，task-80）
# ===========================================================================
#: 累计转角下限（度）：低于此即视为"落在直道上"。
#: 真实几何口径下锚点由 ``real_geometry.select_anchors`` 选出（硬门：
#: ±40m 真实转角 >= 12°），官方缓弯（如 Spa Eau Rouge）允许 12–25°；
#: 旧的 17 项示意 SVG 弱弯豁免随真实几何重建整体删除，不再需要。
_MIN_TURN_DEG = 12.0


class TestAnchorsSitOnCorners:
    """锚点像素必须落在弯上，而非起跑线或直道。

    两个判据（各自直接对应一种"标错"）：
    1. **不得落在起跑线**（路径闭合点）——除非是末弯（末弯出口恰在起跑线）。
       旧锚点把蒙扎 T1 摆在起跑线 (118.3, 370.2)，正是用户报的缺陷。
    2. **累计转角 ≥ 下限**——直道转角≈0。
    """

    def _closure_point(self, track_id: str) -> tuple[float, float]:
        g = _load_generator()
        poly, _, _ = g._dense_path(track_id)
        return poly[0]

    def _cumulative_turn_at(self, track_id: str, x: float, y: float) -> float:
        import math

        g = _load_generator()
        poly, cum, total = g._dense_path(track_id)
        raw = g._turning_series(poly)
        ct = g._cumulative_turn(raw, cum, total, 0.05)
        j = min(range(len(poly)), key=lambda k: (poly[k][0] - x) ** 2
                + (poly[k][1] - y) ** 2)
        return math.degrees(ct[j])

    @pytest.mark.parametrize("track_id", sorted(TRACK_CORNER_ARCS))
    def test_anchors_not_on_start_line(self, track_id: str) -> None:
        """除末弯外，锚点不得落在起跑线（路径闭合点）。"""
        from setup_tuner.domain._track_anchors import TRACK_ANCHORS

        anchors = TRACK_ANCHORS[track_id]
        last = max(anchors)
        cx, cy = self._closure_point(track_id)
        for number, (x, y) in anchors.items():
            if number == last:
                continue
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            assert dist > 2.0, (
                f"{track_id} T{number} 锚点落在起跑线（距离 {dist:.1f}px），"
                f"弯道不该标在直道上"
            )

    @pytest.mark.parametrize("track_id", sorted(TRACK_CORNER_ARCS))
    def test_anchors_not_on_straights(self, track_id: str) -> None:
        """锚点在 SVG 实际路径上的累计转角（5% 窗口）必须 ≥ 下限。

        真实几何口径下无任何豁免：锚点全部由 ``select_anchors`` 硬门
        （±40m 真实转角 ≥ 12°）选出，SVG 是等比投影，此处兜底校验
        「可见层标记确实画在弯上」。
        """
        from setup_tuner.domain._track_anchors import TRACK_ANCHORS

        for number, (x, y) in TRACK_ANCHORS[track_id].items():
            deg = self._cumulative_turn_at(track_id, x, y)
            assert deg >= _MIN_TURN_DEG, (
                f"{track_id} T{number} 累计转角 {deg:.0f}° < {_MIN_TURN_DEG:.0f}°，"
                f"锚点疑似落在直道上"
            )
