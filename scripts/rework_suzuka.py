"""Suzuka 弯道数据按 F1 26 游内弯号全面重做（task-63 连续弯整改模板）。

依据（F1 26 官方弯号，来源 simracingsetup/planetf1 赛道指南）：
    T1 First Curve | T2 Second Curve | T3-T6 S-Curves | T7 Dunlop |
    T8 Degner 1 | T9 Degner 2 | T10 Bridge Right | T11 Hairpin |
    T12 Sweep Right | T13/T14 Spoon | T15 130R | T16-T18 Casio Triangle

步骤：
    1. 新弯道表（18 弯，含各弯的圈内进度比例）；
    2. 用 SVG 路径把比例投影回像素坐标 → 更新 _track_anchors["suzuka"]；
    3. 重新生成 _track_arcs（与 CI 校验共用同一实现）。
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import gen_track_arcs as gta  # noqa: E402

# ------------------------------------------------------------------ #
# F1 26 游内弯号 → (name, corner_type, speed_kmh, lap_fraction)
# fraction 为该弯锚点在圈内的比例位置（依据真实 Suzuka 圈结构与
# 旧锚点位置校准；interpolated = 相邻真实锚点间插值）。
# ------------------------------------------------------------------ #
TURNS: list[tuple[int, str, str, float, float]] = [
    (1, "First Curve", "fast", 230, 0.003),
    (2, "Second Curve", "medium", 150, 0.045),
    (3, "S-Curves 1", "medium", 165, 0.088),
    (4, "S-Curves 2", "medium", 175, 0.128),
    (5, "S-Curves 3", "medium", 180, 0.157),
    (6, "S-Curves 4", "medium", 170, 0.175),
    (7, "Dunlop", "fast", 240, 0.193),
    (8, "Degner 1", "medium", 115, 0.265),
    (9, "Degner 2", "slow", 85, 0.328),
    (10, "Bridge Right", "fast", 210, 0.365),
    (11, "Hairpin", "slow", 70, 0.439),
    (12, "Sweep Right", "fast", 250, 0.480),
    (13, "Spoon Curve", "medium", 140, 0.538),
    (14, "Spoon Exit", "medium", 155, 0.600),
    (15, "130R", "fast", 295, 0.756),
    (16, "Casio Chicane Right", "slow", 75, 0.800),
    (17, "Casio Chicane Left", "slow", 80, 0.845),
    (18, "Final Right", "medium", 140, 0.900),
]


def point_at_fraction(
    fraction: float,
    poly: list[tuple[float, float]],
    cum: list[float],
    total: float,
) -> tuple[float, float]:
    """圈内比例 → SVG 折线上的像素坐标（沿累计弧长定位）。"""
    target = fraction * total
    for k in range(len(poly) - 1):
        ax, ay = poly[k]
        bx, by = poly[k + 1]
        seg = math.hypot(bx - ax, by - ay)
        if cum[k] + seg >= target or k == len(poly) - 2:
            t = 0.0 if seg == 0 else (target - cum[k]) / seg
            t = max(0.0, min(1.0, t))
            return (ax + t * (bx - ax), ay + t * (by - ay))
    return poly[-1]


def main() -> int:
    svg = (REPO / "setup_tuner" / "ui" / "tracks" / "suzuka.svg").read_text(
        encoding="utf-8"
    )
    paths = re.findall(r'<path[^>]*\sd="([^"]+)"', svg)
    poly = gta.parse_path(max(paths, key=len))
    cum, total = gta._cumulative(poly)

    anchors: dict[int, tuple[float, float]] = {}
    for number, name, ctype, speed, fraction in TURNS:
        px, py = point_at_fraction(fraction, poly, cum, total)
        anchors[number] = (round(px, 1), round(py, 1))
        print(f"T{number:>2} {name:<20} {ctype:<6} {speed:>3}km/h "
              f"@ {fraction*100:5.1f}% -> ({px:.1f}, {py:.1f})")

    # 更新 _track_anchors["suzuka"]（保持其他赛道不动）
    anchors_path = REPO / "setup_tuner" / "domain" / "_track_anchors.py"
    text = anchors_path.read_text(encoding="utf-8")
    block = re.compile(r'    "suzuka": \{.*?\n    \},', re.S)
    lines = ",\n".join(f"        {n}: ({x}, {y})" for n, (x, y) in anchors.items())
    text = block.sub(f'    "suzuka": {{\n{lines},\n    }},', text, count=1)
    anchors_path.write_text(text, encoding="utf-8", newline="\n")
    print("\n_track_anchors[suzuka] 已更新")

    # 更新 domain/track.py 的 Suzuka 种子元数据
    track_py = REPO / "setup_tuner" / "domain" / "track.py"
    ttext = track_py.read_text(encoding="utf-8")
    old_block = re.compile(
        r'def _suzuka_corners\(\) -> list\[Corner\]:\n'
        r'    """Suzuka International Racing Course \(5\.807 km, 18 弯, mixed\)。"""\n'
        r'    return _build_corners\(\[\n(?:.*?\n)*?    \]\)',
    )
    rows = ",\n".join(
        f'        ({n}, "{name}", "{ctype}", {speed})'
        for n, name, ctype, speed, _ in TURNS
    )
    new_block = (
        'def _suzuka_corners() -> list[Corner]:\n'
        '    """Suzuka International Racing Course (5.807 km, 18 弯, mixed)。\n\n'
        '    task-63：按 F1 26 游内弯号重排（S-Curves=T3-T6、Dunlop=T7、\n'
        '    Degner=T8/T9、Bridge Right=T10、Hairpin=T11、Spoon=T13/T14、\n'
        '    130R=T15、Casio=T16-T18）。锚点取自 _track_anchors（真实 SVG 投影）。\n'
     '    """\n'
        f'    return _build_corners([\n{rows},\n    ])'
    )
    ttext, n_sub = old_block.subn(new_block, ttext, count=1)
    assert n_sub == 1, "track.py Suzuka 种子块未匹配"
    track_py.write_text(ttext, encoding="utf-8", newline="\n")
    print("domain/track.py Suzuka 种子已重写")

    # 重新生成弧长表
    arcs = gta.compute_arcs()
    (REPO / "setup_tuner" / "domain" / "_track_arcs.py").write_text(
        gta.render_module(arcs), encoding="utf-8", newline="\n",
    )
    print(f"_track_arcs 已再生（suzuka 弧长："
          f"{ {k: round(v, 3) for k, v in list(arcs['suzuka'].items())[:6]} } ...）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
