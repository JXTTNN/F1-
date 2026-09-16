"""弯道段（corner group）—— 连续弯的一等表达（task-63）。

问题（2026-09-16 用户实测「弯道特别是连续弯直接不行」）：
    全 24 赛道 380 个相邻弯对中 **39% 间距 < 4% 圈长**——连续弯/组合弯是常态
    而非例外。在"单弯粒度"下连续弯不可用：

    - 定位：SVG 弧长与游戏 ``m_lapDistance`` 间存在 2–4% 系统性偏移，与相邻
      弯间距同量级 → S 弯段内"当前弯"高频跳变/错位（如 Suzuka T2-T7，相邻
      间距仅 1.5–3.6%）；
    - 反馈：车手感知的是"这一段弯"，旧 UI 却只能绑单个弯号；
    - 建议：段内其余弯完全不受反馈影响。

整改：弯道子系统升级为**段粒度**——
    - 段推导：相邻间距 ≤ :data:`GROUP_GAP_THRESHOLD` 链成段（断链即分段），
      段跨度上限 :data:`GROUP_MAX_SPAN`；单弯也是段（members=[n]），全赛道
      覆盖，调用方无需判空；
    - 映射：进度 → 段（前向区间归属，无跳变歧义）→ 段内最近成员；
    - 反馈：段级反馈由前端展开为段内逐弯反馈（复用现有批量接口与引擎，
      不改 DB schema、不改 Dx 聚合）。

环形坐标系（重要）：
    以最大环形间距处为切点展开成**线性序列**（保证切点落在组间空隙，
    即任何段都不被切点穿过）。段的 ``arc_start``/``arc_end`` 是**展开坐标**
    ——``arc_end`` 可能 ≥ 1.0（段合法地跨 0/1 边界，如 monza）。containment
    判断用前向距离：``cyclic_gap(arc_start, p) ≤ (arc_end - arc_start)``。
    纯函数、不落库（段由 TRACK_CORNER_ARCS 推导，避免迁移）。
"""

from __future__ import annotations

from typing import Any

# 相邻弯弧长间距 ≤ 5% 圈长 → 并入同一段（S 弯/组合弯的典型密度）
GROUP_GAP_THRESHOLD = 0.05
# 段的最大弧长跨度 18% 圈长（防止把半个赛道并成一段）
GROUP_MAX_SPAN = 0.18


def _cyclic_gap(a: float, b: float) -> float:
    """前向环形间距：从 a 前进到 b 的比例距离（∈ [0, 1)）。"""
    gap = b - a
    if gap < 0:
        gap += 1.0
    return gap


def _cyclic_dist(a: float, b: float) -> float:
    """环形最短距离（∈ [0, 0.5]）。"""
    gap = _cyclic_gap(a, b)
    return min(gap, 1.0 - gap)


def build_corner_groups(arcs: dict[int, float]) -> list[dict[str, Any]]:
    """把单弯锚点合并为「弯道段」。

    Args:
        arcs: ``{corner_number: arc_fraction}``（``TRACK_CORNER_ARCS[track_id]``）。

    Returns:
        段列表（按 arc_start 升序），每段::

            {
              "name": "T2-T7",          # 单弯段为 "T3"
              "members": [2, 3, 4, 5, 6, 7],
              "arc_start": 0.031,        # 展开坐标：段首弯锚点
              "arc_end": 1.02,           # 展开坐标：段尾弯锚点（可能 ≥ 1.0）
              "size": 6,
            }

    算法：按弧长排序 → 找最大环形间距作切点 → 展开为线性序列（切点必落在
    组间空隙）→ 沿线扫描链段（间距与跨度双阈值）。
    """
    if not arcs:
        return []
    numbers = sorted(arcs)
    if len(numbers) == 1:
        n = numbers[0]
        return [{
            "name": f"T{n}", "members": [n],
            "arc_start": arcs[n], "arc_end": arcs[n], "size": 1,
        }]

    # 环形展开：最大间距处为切点（该间距必为组间空隙或超长间隙）
    gaps = [
        (numbers[i], _cyclic_gap(arcs[numbers[i]], arcs[numbers[(i + 1) % len(numbers)]]))
        for i in range(len(numbers))
    ]
    start_idx = max(gaps, key=lambda kv: kv[1])[0] and (
        next(i for i, (n, g) in enumerate(gaps) if n == max(gaps, key=lambda kv: kv[1])[0]) + 1
    )
    ordered = [numbers[(start_idx + i) % len(numbers)] for i in range(len(numbers))]

    # 展开坐标：pos 单调递增（累计前向间距）
    positions = {ordered[0]: arcs[ordered[0]]}
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        positions[cur] = positions[prev] + _cyclic_gap(arcs[prev], arcs[cur])

    segments: list[list[int]] = []
    current: list[int] = [ordered[0]]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        gap = positions[cur] - positions[prev]
        span = positions[cur] - positions[current[0]]
        if gap <= GROUP_GAP_THRESHOLD and span <= GROUP_MAX_SPAN:
            current.append(cur)
        else:
            segments.append(current)
            current = [cur]
    segments.append(current)

    result: list[dict[str, Any]] = []
    for members in segments:
        first, last = members[0], members[-1]
        result.append({
            "name": f"T{first}-T{last}" if len(members) > 1 else f"T{first}",
            "members": members,
            "arc_start": positions[first],
            "arc_end": positions[last],
            "size": len(members),
        })
    return result


def corner_group_index(
    groups: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """``{corner_number: group}`` —— 成员弯号到所属段的索引。"""
    index: dict[int, dict[str, Any]] = {}
    for group in groups:
        for member in group["members"]:
            index[member] = group
    return index


def _in_group(p: float, group: dict[str, Any]) -> bool:
    """进度是否落在段的展开区间内（前向距离 ≤ 段跨度）。"""
    start = group["arc_start"]
    span = group["arc_end"] - group["arc_start"]
    return _cyclic_gap(start, p % 1.0) <= span + 1e-9


def group_for_progress(
    progress: float, groups: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """把圈内进度映射到段：先区间归属，落在组间空隙时取最近段。

    Args:
        progress: 圈内进度 ∈ [0, 1)。
        groups: :func:`build_corner_groups` 的输出。

    Returns:
        所属段（最稳定的表达：段级高亮/反馈不随微小偏移跳变）。
    """
    if not groups:
        return None
    p = progress % 1.0
    for group in groups:
        if _in_group(p, group):
            return group

    def dist_to_range(g: dict[str, Any]) -> float:
        start, end = g["arc_start"], g["arc_end"]
        if _cyclic_gap(start, p) <= end - start:
            return 0.0
        return min(_cyclic_dist(p, start % 1.0), _cyclic_dist(p, end % 1.0))

    return min(groups, key=dist_to_range)


def nearest_member(
    progress: float, group: dict[str, Any], arcs: dict[int, float],
) -> int:
    """段内取与进度环形距离最近的成员弯号。"""
    p = progress % 1.0
    return min(group["members"], key=lambda n: _cyclic_dist(p, arcs[n] % 1.0))
