#!/usr/bin/env python3
"""Generate SVG track map placeholders from control points data.

Creates simple SVG representations of each track's layout using the control
points defined in f1opt/data/track_maps/__init__.py. These serve as visual
placeholders until official high-resolution track images are provided.
"""

import sys
from pathlib import Path

# Add the project root to sys.path so we can import track_maps
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from f1opt.data.track_maps import TRACK_MAPS, TrackMapData


def generate_svg(track_data: TrackMapData) -> str:
    """Generate a simple SVG track map from control points."""
    w = track_data.canvas_width
    h = track_data.canvas_height
    cp = track_data.control_points

    # Build the track outline path
    points_str = " ".join(f"{p[1]:.1f},{p[2]:.1f}" for p in cp)

    # Build corner markers SVG elements
    corner_elements = []
    for c in track_data.corners:
        cx, cy = c.x_px, c.y_px
        corner_elements.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
            f'fill="none" stroke="#00e5ff" stroke-width="1.5" opacity="0.7"/>'
        )
        corner_elements.append(
            f'  <text x="{cx:.1f}" y="{cy - 8:.1f}" text-anchor="middle" '
            f'fill="#8a909c" font-size="8" font-family="monospace">{c.corner_id}</text>'
        )

    # Build sector boundary markers
    sector_elements = []
    for sd in track_data.sector_boundaries:
        # Find nearest control point
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

    # Start/finish marker
    sf_element = ""
    if track_data.start_finish_line_px:
        sfx, sfy = track_data.start_finish_line_px
        sf_element = (
            f'  <rect x="{sfx - 4:.1f}" y="{sfy - 4:.1f}" width="8" height="8" '
            f'fill="#ff2d2d" rx="1"/>'
        )

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <style>
      .track-bg {{ fill: #111317; }}
      .track-line {{ fill: none; stroke: #2a2e36; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }}
      .track-line-accent {{ fill: none; stroke: rgba(0,229,255,0.25); stroke-width: 2; stroke-linecap: round; stroke-dasharray: 6,4; }}
    </style>
  </defs>

  <!-- Background -->
  <rect class="track-bg" width="{w}" height="{h}" rx="8"/>

  <!-- Track outline (wide shadow line) -->
  <polyline class="track-line" points="{points_str}"/>

  <!-- Track accent line -->
  <polyline class="track-line-accent" points="{points_str}"/>

{chr(10).join(sector_elements)}

{chr(10).join(corner_elements)}

{sf_element}

  <!-- Track label -->
  <text x="{w/2:.0f}" y="{h - 12:.0f}" text-anchor="middle" fill="#5b616e"
        font-size="10" font-family="monospace">{track_data.track_id.upper()}</text>
</svg>'''
    return svg


def main():
    output_dir = PROJECT_ROOT / "f1opt" / "ui" / "static"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for track_id, track_data in TRACK_MAPS.items():
        svg_content = generate_svg(track_data)
        out_path = output_dir / f"{track_id}.svg"
        out_path.write_text(svg_content, encoding="utf-8")
        generated += 1
        print(f"  Generated: {out_path.name} ({track_data.canvas_width}x{track_data.canvas_height})")

    print(f"\nDone: {generated} SVG track maps generated in {output_dir}")


if __name__ == "__main__":
    main()
