#!/usr/bin/env python3
"""Generate SVG track map placeholders — standalone, no project imports needed.

Reads control points directly from f1opt/data/track_maps/__init__.py via
regex extraction, then generates SVG files for each track.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACK_MAPS_FILE = PROJECT_ROOT / "f1opt" / "data" / "track_maps" / "__init__.py"
OUTPUT_DIR = PROJECT_ROOT / "f1opt" / "ui" / "static"

# Track data: (track_id, canvas_width, canvas_height, corners_count, sector_boundaries_str)
TRACK_META = [
    ("melbourne", 800, 600, 14, "1760,3520,5278"),
    ("shanghai", 800, 600, 16, "1817,3634,5451"),
    ("suzuka", 800, 600, 18, "1936,3872,5807"),
    ("sakhir", 800, 600, 13, "1804,3608,5412"),
    ("jeddah", 800, 600, 27, "2058,4116,6174"),
    ("miami", 800, 600, 11, "1804,3608,5412"),
    ("montreal", 800, 600, 14, "1454,2907,4361"),
    ("monaco", 800, 600, 13, "1112,2225,3337"),
    ("barcelona", 800, 600, 14, "1552,3105,4657"),
    ("spielberg", 800, 600, 10, "1439,2879,4318"),
    ("silverstone", 800, 600, 18, "1964,3927,5891"),
    ("spa", 800, 600, 19, "2335,4669,7004"),
    ("hungaroring", 800, 600, 14, "1460,2921,4381"),
    ("zandvoort", 800, 600, 14, "1420,2840,4259"),
    ("monza", 800, 600, 11, "1931,3862,5793"),
    ("madrid", 800, 600, 22, "1805,3611,5416"),
    ("baku", 800, 600, 20, "2001,4002,6003"),
    ("singapore", 800, 600, 19, "1647,3293,4940"),
    ("austin", 800, 600, 20, "1838,3675,5513"),
    ("mexico_city", 800, 600, 17, "1435,2869,4304"),
    ("sao_paulo", 800, 600, 15, "1436,2873,4309"),
    ("las_vegas", 800, 600, 17, "2067,4134,6201"),
    ("lusail", 800, 600, 16, "1806,3613,5419"),
    ("yas_marina", 800, 600, 16, "1760,3521,5281"),
]


def extract_control_points(track_id: str) -> list[tuple[float, float, float]]:
    """Extract control points for a given track from the source file."""
    content = TRACK_MAPS_FILE.read_text(encoding="utf-8")

    # Find the section for this track
    pattern = rf'TRACK_MAPS\["{re.escape(track_id)}"\]\s*=\s*TrackMapData\((.*?)\n\)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []

    block = match.group(1)

    # Extract control_points list
    cp_pattern = r'control_points=\[(.*?)\]'
    cp_match = re.search(cp_pattern, block, re.DOTALL)
    if not cp_match:
        return []

    cp_block = cp_match.group(1)
    # Parse tuples like (0, 400, 50)
    points = re.findall(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', cp_block)
    return [(float(d), float(x), float(y)) for d, x, y in points]


def extract_corners(track_id: str) -> list[tuple[int, str, float]]:
    """Extract corner data: (corner_id, name, x_px, y_px)."""
    content = TRACK_MAPS_FILE.read_text(encoding="utf-8")

    pattern = rf'TRACK_MAPS\["{re.escape(track_id)}"\]\s*=\s*TrackMapData\((.*?)\n\)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []

    block = match.group(1)

    # Extract corners list
    corners_pattern = r'corners=\[(.*?)\]'
    corners_match = re.search(corners_pattern, block, re.DOTALL)
    if not corners_match:
        return []

    corner_block = corners_match.group(1)

    # Check if it's a list comprehension (for jeddah, madrid, baku, singapore)
    if "for i in range" in corner_block:
        # Extract range parameters
        range_match = re.search(r'range\((\d+),\s*(\d+)\)', corner_block)
        start = int(range_match.group(1)) if range_match else 1
        end = int(range_match.group(2)) if range_match else 20

        # Extract formula for x, y
        x_pattern = re.search(r'x_px:\s*(.*?)(?:,|\n)', corner_block)
        y_pattern = re.search(r'y_px:\s*(.*?)(?:,|\n)', corner_block)
        if not x_pattern or not y_pattern:
            return []

        corners = []
        for i in range(start, end):
            # Evaluate the formula
            x_formula = x_pattern.group(1).strip()
            y_formula = y_pattern.group(1).strip()
            x_formula = x_formula.replace("i", str(i))
            y_formula = y_formula.replace("i", str(i))
            try:
                x_px = eval(x_formula)
                y_px = eval(y_formula)
                corners.append((i, f"Turn {i}", x_px, y_px))
            except Exception:
                corners.append((i, f"Turn {i}", 400, 300))
        return corners
    else:
        # Regular CornerDef entries
        corners = []
        entries = re.findall(
            r'CornerDef\(\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
            corner_block
        )
        for cid, name, x_px, y_px in entries:
            corners.append((int(cid), name, float(x_px), float(y_px)))
        return corners


def extract_start_finish(track_id: str) -> tuple[float, float]:
    """Extract start/finish line position."""
    content = TRACK_MAPS_FILE.read_text(encoding="utf-8")
    pattern = rf'TRACK_MAPS\["{re.escape(track_id)}"\]\s*=\s*TrackMapData\((.*?)\n\)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return (400, 100)

    block = match.group(1)
    sf_match = re.search(r'start_finish_line_px=\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', block)
    if sf_match:
        return (float(sf_match.group(1)), float(sf_match.group(2)))
    return (400, 100)


def generate_svg(track_id: str, cp: list, corners: list, sectors: list,
                 w: int, h: int, sf_pos: tuple) -> str:
    """Generate a simple SVG track map."""
    points_str = " ".join(f"{p[1]:.1f},{p[2]:.1f}" for p in cp)

    corner_elements = []
    for cid, name, cx, cy in corners:
        corner_elements.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
            f'fill="none" stroke="#00e5ff" stroke-width="1.5" opacity="0.7"/>'
        )
        corner_elements.append(
            f'  <text x="{cx:.1f}" y="{cy - 8:.1f}" text-anchor="middle" '
            f'fill="#8a909c" font-size="8" font-family="monospace">{cid}</text>'
        )

    sector_elements = []
    for sd in sectors:
        best = cp[0]
        best_dist = 999999
        for p in cp:
            d = abs(p[0] - sd)
            if d < best_dist:
                best_dist = d
                best = p
        sx, sy = best[1], best[2]
        sector_elements.append(
            f'  <line x1="{sx-6:.1f}" y1="{sy-6:.1f}" x2="{sx+6:.1f}" y2="{sy+6:.1f}" '
            f'stroke="#ff2d2d" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.6"/>'
        )

    sf_x, sf_y = sf_pos
    sf_element = (
        f'  <rect x="{sf_x - 4:.1f}" y="{sf_y - 4:.1f}" width="8" height="8" '
        f'fill="#ff2d2d" rx="1"/>'
    )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <style>
      .track-bg {{ fill: #111317; }}
      .track-line {{ fill: none; stroke: #2a2e36; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }}
      .track-line-accent {{ fill: none; stroke: rgba(0,229,255,0.25); stroke-width: 2; stroke-linecap: round; stroke-dasharray: 6,4; }}
    </style>
  </defs>
  <rect class="track-bg" width="{w}" height="{h}" rx="8"/>
  <polyline class="track-line" points="{points_str}"/>
  <polyline class="track-line-accent" points="{points_str}"/>
{chr(10).join(sector_elements)}
{chr(10).join(corner_elements)}
{sf_element}
  <text x="{w/2:.0f}" y="{h - 12:.0f}" text-anchor="middle" fill="#5b616e"
        font-size="10" font-family="monospace">{track_id.upper()}</text>
</svg>'''


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0

    for track_id, w, h, corner_count, sector_str in TRACK_META:
        cp = extract_control_points(track_id)
        if not cp:
            print(f"  SKIP: {track_id} (no control points found)")
            continue

        corners = extract_corners(track_id)
        sectors = [float(s) for s in sector_str.split(",")]
        sf_pos = extract_start_finish(track_id)

        svg = generate_svg(track_id, cp, corners, sectors, w, h, sf_pos)
        out_path = OUTPUT_DIR / f"{track_id}.svg"
        out_path.write_text(svg, encoding="utf-8")
        generated += 1
        print(f"  Generated: {out_path.name} ({len(corners)} corners, {len(cp)} control points)")

    print(f"\nDone: {generated} SVG track maps generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
