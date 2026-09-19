"""圈内进度 → 弯道编号的**唯一定位实现**。

为什么提取到 domain 层
----------------------
该算法原先只存在于 ``api/ws.py::_map_corner``（API 层）。遥测层要"按弯累积
数据"（供后续优化消费）时同样需要它 —— 若在遥测层再写一份，就会出现两份
"进度→弯号"实现，迟早漂移。本项目已因同类分叉实现返工过（同一语义只允许
一处实现），故提取到 domain 层供 API 与遥测共用。

算法（保持既有行为，勿轻改）
--------------------------
``domain/_track_arcs.TRACK_CORNER_ARCS`` 给出每个弯道锚点在 SVG 路径上的累计
弧长占比；把 ``lap_distance`` 归一化为圈内进度后，在该占比序列上做**循环最近邻**
二分查找。赛道是闭合回路，故首尾相邻，比较时要按循环距离。

早期实现按"弯道沿赛道均匀分布"近似，云端实测 24 赛道 × 200 采样点中判定错误
69%。无弧长表的赛道仍回退均匀近似（兼容未知赛道）。
"""

from __future__ import annotations

from bisect import bisect_left
from functools import cache
from typing import Any

from ._track_arcs import TRACK_CORNER_ARCS


@cache
def arc_table(track_id: str) -> tuple[tuple[float, ...], tuple[int, ...]] | None:
    """把弧长表预处理为 (排序后的占比, 对应弯号) 二元组，供二分查找。

    缓存：赛道数据静态，且该查询在原实现中单次约 0.2–0.3 µs。
    """
    arcs = TRACK_CORNER_ARCS.get(track_id)
    if not arcs:
        return None
    items = sorted(arcs.items())
    return tuple(f for _, f in items), tuple(n for n, _ in items)


def locate_corner(
    lap_distance: float,
    track_length: float,
    corners: list[Any],
    track_id: str | None = None,
) -> int | None:
    """由圈内距离定位当前弯道编号。

    Args:
        lap_distance: 当前圈距离（米）。
        track_length: 赛道长度（米）。
        corners: 弯道列表（``domain.Corner``），仅回退路径使用。
        track_id: 赛道标识；有弧长表时走精确路径。

    Returns:
        当前弯道编号（1-based）；无法映射时返回 ``None``。
    """
    if track_length <= 0 or not corners:
        return None
    progress = (lap_distance % track_length) / track_length

    table = arc_table(track_id) if track_id else None
    if table is not None:
        fractions, numbers = table
        i = bisect_left(fractions, progress)
        prev_fraction = fractions[i - 1] if i > 0 else fractions[-1] - 1.0
        next_fraction = fractions[i] if i < len(fractions) else fractions[0] + 1.0
        # 循环比较（路径闭合，末段与首段相邻）
        if (progress - prev_fraction) <= (next_fraction - progress):
            return numbers[i - 1]
        return numbers[i % len(numbers)]

    # 回退：无弧长表的赛道仍用均匀分布近似（兼容未知赛道）
    total = len(corners)
    idx = int(progress * total)
    if idx >= total:
        idx = total - 1
    return corners[idx].number
