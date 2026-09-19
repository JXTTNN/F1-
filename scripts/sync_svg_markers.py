"""把 ``ui/tracks/*.svg`` 内嵌的可见弯道圆点/标签同步到 ``_track_anchors`` 锚点。

背景（task-79）：SVG 内嵌的 ``<circle>/<text>`` 是早期手工摆放的旧坐标，
与 2026-09-17 由累计转角窗口重建的 ``_track_anchors.py`` 脱节——用户在
地图上看到的圆点错位（蒙扎 T1 被画在起跑线直道上），而前端热区层已用
新锚点，导致同一弯道出现两个错开的标记。

本脚本以 ``_track_anchors`` 为唯一权威，成对改写每个 ``<circle>`` 及其
紧跟的 ``<text>``（text x = 圆心 x，text y = 圆心 y - 9，与现有版式一致）；
文件其余字节保持不变。幂等：重复运行不会再产生变化。

用法::

    python scripts/sync_svg_markers.py          # 同步全部 24 条赛道
    python scripts/sync_svg_markers.py monza    # 只同步指定赛道
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from setup_tuner.domain._track_anchors import TRACK_ANCHORS  # noqa: E402

TRACKS_DIR = REPO / "setup_tuner" / "ui" / "tracks"

# circle 与紧跟的 text 成对匹配（text 序号即弯号）； caption 类 <text>（MONZA 等）
# 属性不同，不会被误匹配。
PAIR_RE = re.compile(
    r'<circle cx="(?P<cx>[\d.]+)" cy="(?P<cy>[\d.]+)" r="6" fill="none" '
    r'stroke="#3B9EFF" stroke-width="1\.5" opacity="0\.85"/>'
    r'(?P<sep>\s*)<text x="(?P<tx>[\d.]+)" y="(?P<ty>[\d.]+)" '
    r'text-anchor="middle" fill="#9494a8" font-size="9" '
    r'font-family="monospace">(?P<num>\d+)</text>'
)

TEXT_Y_OFFSET = 9.0  # 数字标在圆环上方 9px（与现有文件版式一致）


def sync_track(track_id: str, anchors: dict[int, tuple[float, float]]) -> int:
    """同步单个赛道 SVG，返回改写的圆点对数；数量不符则中止不写盘。"""
    svg_path = TRACKS_DIR / f"{track_id}.svg"
    if not svg_path.exists():
        raise SystemExit(f"{track_id}: SVG 文件缺失：{svg_path}")
    content = svg_path.read_text(encoding="utf-8")
    matched = {"n": 0}

    def repl(m: re.Match[str]) -> str:
        num = int(m.group("num"))
        if num not in anchors:
            raise SystemExit(f"{track_id}: _track_anchors 缺少弯道 {num}")
        x, y = anchors[num]
        matched["n"] += 1
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="none" '
            f'stroke="#3B9EFF" stroke-width="1.5" opacity="0.85"/>'
            f'{m.group("sep")}<text x="{x:.2f}" y="{y - TEXT_Y_OFFSET:.2f}" '
            f'text-anchor="middle" fill="#9494a8" font-size="9" '
            f'font-family="monospace">{num}</text>'
        )

    new_content = PAIR_RE.sub(repl, content)
    if matched["n"] != len(anchors):
        raise SystemExit(
            f"{track_id}: 仅匹配到 {matched['n']} 对圆点，期望 {len(anchors)}"
            " —— 已中止，未写盘"
        )
    svg_path.write_text(new_content, encoding="utf-8", newline="\n")
    return matched["n"]


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only and only not in TRACK_ANCHORS:
        raise SystemExit(f"未知赛道标识：{only}")
    changed: list[str] = []
    for track_id, anchors in sorted(TRACK_ANCHORS.items()):
        if only and track_id != only:
            continue
        path = TRACKS_DIR / f"{track_id}.svg"
        before = path.read_bytes()
        n = sync_track(track_id, anchors)
        if path.read_bytes() != before:
            changed.append(track_id)
        print(f"  {track_id}: {n} 个圆点已同步")
    print(f"完成：{len(changed)} 个文件更新{'（' + ', '.join(changed) + '）' if changed else '（全部无变化，已同步）'}")


if __name__ == "__main__":
    main()
