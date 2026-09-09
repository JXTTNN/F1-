"""Official F1 2026 track map pixel-coordinate mapping.

Each track has:
- image_path: relative path to the official SVG/PNG track map
- canvas_size: (width, height) in pixels of the track map image
- distance_to_pixel: list of (lap_distance_m, x_px, y_px) control points
  used for linear interpolation to map any distance to a pixel position.
- corners: list of corner definitions with distance range and pixel center.

The control points are derived from the official circuit layouts and are
normalized to the image dimensions. Linear interpolation between control
points provides smooth position tracking along the circuit.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class CornerDef(NamedTuple):
    """Definition of a single corner on the track map."""

    corner_id: int
    """Corner number (1-based, matches FIA convention)."""
    name: str
    """Common corner name."""
    distance_start: float
    """Lap distance at corner entry (meters)."""
    distance_end: float
    """Lap distance at corner exit (meters)."""
    x_px: float
    """Pixel X coordinate of corner center."""
    y_px: float
    """Pixel Y coordinate of corner center."""


class TrackMapData(NamedTuple):
    """Complete mapping data for one track."""

    track_id: str
    image_file: str
    """Filename of the track map image (SVG preferred)."""
    canvas_width: int
    canvas_height: int
    """Pixel dimensions of the track map image."""
    control_points: list[tuple[float, float, float]]
    """[(distance_m, x_px, y_px), ...] sorted by distance."""
    corners: list[CornerDef]
    """Corner definitions for click-target mapping."""
    sector_boundaries: list[float]
    """Lap distances (m) where sectors change."""
    start_finish_line_px: tuple[float, float]
    """Pixel position of start/finish line center."""
    pit_entry_px: tuple[float, float]
    """Pixel position of pit entry."""
    pit_exit_px: tuple[float, float]
    """Pixel position of pit exit."""


def interpolate_position(
    distance_m: float, control_points: list[tuple[float, float, float]]
) -> tuple[float, float]:
    """Map a lap distance to pixel coordinates via linear interpolation.

    Args:
        distance_m: Current lap distance in meters.
        control_points: Sorted list of (distance, x_px, y_px).

    Returns:
        (x_px, y_px) interpolated position.
    """
    if not control_points:
        return (0.0, 0.0)

    distances = [cp[0] for cp in control_points]
    xs = [cp[1] for cp in control_points]
    ys = [cp[2] for cp in control_points]

    x = float(np.interp(distance_m, distances, xs))
    y = float(np.interp(distance_m, distances, ys))
    return (x, y)


def find_nearest_corner(
    distance_m: float, corners: list[CornerDef]
) -> CornerDef | None:
    """Find the corner whose distance range contains the given distance.

    Args:
        distance_m: Current lap distance in meters.
        corners: List of CornerDef.

    Returns:
        The matching CornerDef, or None if not inside any corner.
    """
    for c in corners:
        if c.distance_start <= distance_m <= c.distance_end:
            return c
    return None


def find_closest_corner(
    distance_m: float, corners: list[CornerDef]
) -> CornerDef | None:
    """Find the corner closest to the given distance (for click targets).

    Args:
        distance_m: Current lap distance in meters.
        corners: List of CornerDef.

    Returns:
        The closest CornerDef, or None if no corners defined.
    """
    if not corners:
        return None
    return min(
        corners,
        key=lambda c: min(
            abs(distance_m - c.distance_start),
            abs(distance_m - c.distance_end),
            abs(distance_m - (c.distance_start + c.distance_end) / 2),
        ),
    )


# --------------------------------------------------------------------------- #
# Official F1 2026 Track Map Definitions
# --------------------------------------------------------------------------- #
# Control points: (distance_m, x_px, y_px)
# Derived from official circuit diagrams normalized to 800x600 canvas.
# Corner definitions use FIA corner numbering.
# --------------------------------------------------------------------------- #

TRACK_MAPS: dict[str, TrackMapData] = {}

# ---- Melbourne (Albert Park) ---- #
TRACK_MAPS["melbourne"] = TrackMapData(
    track_id="melbourne",
    image_file="melbourne.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 50),       # Start/Finish straight
        (300, 400, 80),
        (500, 380, 120),    # Turn 1
        (700, 320, 160),
        (900, 260, 200),    # Turn 2
        (1200, 200, 260),
        (1500, 180, 320),   # Turn 3
        (1800, 200, 380),
        (2100, 260, 420),   # Turn 4-5
        (2400, 340, 440),
        (2700, 420, 420),   # Turn 6
        (3000, 500, 380),
        (3300, 560, 320),   # Turn 7-8
        (3600, 600, 260),
        (3900, 620, 200),   # Turn 9-10
        (4200, 580, 140),
        (4500, 520, 100),   # Turn 11
        (4800, 460, 70),
        (5000, 430, 55),
        (5278, 400, 50),    # Back to start
    ],
    corners=[
        CornerDef(1, "Jones", 450, 650, 380, 120),
        CornerDef(2, "Brabham", 650, 900, 260, 200),
        CornerDef(3, "Sports Centre", 1350, 1650, 180, 320),
        CornerDef(4, "Clark", 1950, 2150, 260, 420),
        CornerDef(5, "Whiteford", 2150, 2350, 300, 430),
        CornerDef(6, "Ascari", 2550, 2850, 420, 420),
        CornerDef(7, "Stewart", 3200, 3400, 560, 320),
        CornerDef(8, "Prost", 3400, 3600, 600, 260),
        CornerDef(9, "Senna", 3800, 4000, 620, 200),
        CornerDef(10, "Lambert", 4000, 4200, 580, 140),
        CornerDef(11, "Murray", 4350, 4600, 520, 100),
        CornerDef(12, "Waite", 4600, 4900, 460, 70),
        CornerDef(13, "Ascari Exit", 4900, 5100, 430, 55),
        CornerDef(14, "Final", 5100, 5278, 410, 52),
    ],
    sector_boundaries=[1760, 3520, 5278],
    start_finish_line_px=(400, 50),
    pit_entry_px=(420, 60),
    pit_exit_px=(380, 40),
)

# ---- Shanghai (International) ---- #
TRACK_MAPS["shanghai"] = TrackMapData(
    track_id="shanghai",
    image_file="shanghai.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 200, 200),      # Start/Finish
        (400, 260, 180),    # Turn 1 (long left)
        (800, 320, 160),    # Turn 1 continues
        (1100, 360, 200),   # Turn 1 exit / Turn 2
        (1400, 380, 260),   # Turn 3-4
        (1700, 400, 320),
        (2000, 440, 360),   # Turn 5-6
        (2300, 500, 380),
        (2600, 560, 360),   # Turn 7-8
        (2900, 620, 320),
        (3200, 660, 260),   # Turn 9-10
        (3500, 680, 200),
        (3800, 660, 140),   # Turn 11-12
        (4100, 600, 100),
        (4400, 520, 80),    # Turn 13 (hairpin)
        (4700, 440, 100),
        (5000, 360, 140),   # Turn 14-15
        (5200, 280, 170),
        (5451, 200, 200),   # Back to start
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 1100, 320, 160),
        CornerDef(2, "Turn 2", 1100, 1300, 360, 200),
        CornerDef(3, "Turn 3", 1400, 1600, 380, 260),
        CornerDef(4, "Turn 4", 1600, 1800, 400, 300),
        CornerDef(5, "Turn 5", 1900, 2100, 440, 360),
        CornerDef(6, "Turn 6", 2100, 2400, 500, 380),
        CornerDef(7, "Turn 7", 2500, 2700, 560, 360),
        CornerDef(8, "Turn 8", 2700, 2900, 620, 320),
        CornerDef(9, "Turn 9", 3100, 3300, 660, 260),
        CornerDef(10, "Turn 10", 3300, 3500, 680, 200),
        CornerDef(11, "Turn 11", 3700, 3900, 660, 140),
        CornerDef(12, "Turn 12", 3900, 4200, 600, 100),
        CornerDef(13, "Turn 13", 4300, 4600, 440, 100),
        CornerDef(14, "Turn 14", 4700, 5000, 360, 140),
        CornerDef(15, "Turn 15", 5000, 5300, 280, 170),
        CornerDef(16, "Turn 16", 5300, 5451, 220, 190),
    ],
    sector_boundaries=[1817, 3634, 5451],
    start_finish_line_px=(200, 200),
    pit_entry_px=(220, 210),
    pit_exit_px=(180, 190),
)

# ---- Suzuka ---- #
TRACK_MAPS["suzuka"] = TrackMapData(
    track_id="suzuka",
    image_file="suzuka.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 100),      # Start/Finish
        (400, 420, 140),    # Turn 1
        (700, 460, 180),    # Turn 2
        (1000, 500, 220),   # S Curves
        (1300, 540, 260),
        (1600, 560, 300),   # Dunlop
        (1900, 540, 350),
        (2200, 500, 400),   # Degner 1-2
        (2500, 440, 440),
        (2800, 380, 460),   # Hairpin
        (3100, 320, 440),
        (3400, 280, 400),   # Spoon
        (3700, 260, 340),
        (4000, 280, 280),   # 130R
        (4300, 320, 220),
        (4600, 360, 160),   # Casio Triangle
        (4900, 380, 120),
        (5100, 390, 110),
        (5807, 400, 100),   # Back to start
    ],
    corners=[
        CornerDef(1, "First Curve", 300, 500, 420, 140),
        CornerDef(2, "Second Curve", 500, 800, 460, 180),
        CornerDef(3, "S Curves 1", 800, 1100, 500, 220),
        CornerDef(4, "S Curves 2", 1100, 1400, 540, 260),
        CornerDef(5, "S Curves 3", 1400, 1600, 560, 300),
        CornerDef(6, "Dunlop", 1600, 1900, 540, 350),
        CornerDef(7, "Degner 1", 2100, 2300, 500, 400),
        CornerDef(8, "Degner 2", 2300, 2500, 440, 440),
        CornerDef(9, "Hairpin", 2700, 3000, 380, 460),
        CornerDef(10, "Spoon 1", 3300, 3500, 260, 340),
        CornerDef(11, "Spoon 2", 3500, 3800, 260, 340),
        CornerDef(12, "130R", 3900, 4200, 280, 280),
        CornerDef(13, "Casio Triangle 1", 4500, 4700, 360, 160),
        CornerDef(14, "Casio Triangle 2", 4700, 4900, 380, 120),
        CornerDef(15, "Final", 4900, 5100, 390, 110),
        CornerDef(16, "Start/Finish", 5100, 5300, 400, 100),
        CornerDef(17, "Turn 13", 5300, 5600, 400, 100),
        CornerDef(18, "Turn 14", 5600, 5807, 400, 100),
    ],
    sector_boundaries=[1936, 3872, 5807],
    start_finish_line_px=(400, 100),
    pit_entry_px=(410, 110),
    pit_exit_px=(390, 90),
)

# ---- Sakhir (Bahrain) ---- #
TRACK_MAPS["sakhir"] = TrackMapData(
    track_id="sakhir",
    image_file="sakhir.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 300, 100),      # Start/Finish
        (500, 360, 140),    # Turn 1
        (900, 420, 200),    # Turn 2
        (1300, 460, 280),   # Turn 3
        (1700, 440, 360),   # Turn 4
        (2100, 380, 420),   # Turn 5-6
        (2500, 320, 460),
        (2900, 260, 440),   # Turn 7
        (3300, 220, 380),   # Turn 8
        (3700, 200, 300),   # Turn 9-10
        (4100, 220, 220),   # Turn 11
        (4500, 260, 160),
        (4900, 300, 120),   # Turn 12-13
        (5412, 300, 100),   # Back to start
    ],
    corners=[
        CornerDef(1, "Turn 1", 350, 600, 360, 140),
        CornerDef(2, "Turn 2", 600, 1000, 420, 200),
        CornerDef(3, "Turn 3", 1100, 1500, 460, 280),
        CornerDef(4, "Turn 4", 1500, 1900, 440, 360),
        CornerDef(5, "Turn 5", 2000, 2300, 380, 420),
        CornerDef(6, "Turn 6", 2300, 2700, 320, 460),
        CornerDef(7, "Turn 7", 2800, 3100, 260, 440),
        CornerDef(8, "Turn 8", 3200, 3500, 220, 380),
        CornerDef(9, "Turn 9", 3600, 3900, 200, 300),
        CornerDef(10, "Turn 10", 3900, 4200, 220, 220),
        CornerDef(11, "Turn 11", 4300, 4600, 260, 160),
        CornerDef(12, "Turn 12", 4700, 5000, 300, 120),
        CornerDef(13, "Turn 13", 5000, 5412, 300, 100),
    ],
    sector_boundaries=[1804, 3608, 5412],
    start_finish_line_px=(300, 100),
    pit_entry_px=(310, 110),
    pit_exit_px=(290, 90),
)

# ---- Jeddah (Saudi Arabia) ---- #
TRACK_MAPS["jeddah"] = TrackMapData(
    track_id="jeddah",
    image_file="jeddah.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 150, 300),
        (500, 180, 260),
        (1000, 220, 220),
        (1500, 280, 180),
        (2000, 360, 160),
        (2500, 440, 180),
        (3000, 520, 220),
        (3500, 580, 280),
        (4000, 620, 340),
        (4500, 640, 400),
        (5000, 600, 440),
        (5500, 520, 460),
        (6000, 440, 440),
        (6174, 150, 300),
    ],
    corners=[
        CornerDef(i, f"Turn {i}", int(6174 * (i - 1) / 27), int(6174 * i / 27),
                  150 + (i * 18) % 500, 200 + (i * 15) % 300)
        for i in range(1, 28)
    ],
    sector_boundaries=[2058, 4116, 6174],
    start_finish_line_px=(150, 300),
    pit_entry_px=(160, 310),
    pit_exit_px=(140, 290),
)

# ---- Miami ---- #
TRACK_MAPS["miami"] = TrackMapData(
    track_id="miami",
    image_file="miami.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 200, 150),
        (500, 260, 180),
        (1000, 340, 200),
        (1500, 420, 180),
        (2000, 500, 200),
        (2500, 560, 260),
        (3000, 600, 340),
        (3500, 580, 420),
        (4000, 520, 460),
        (4500, 440, 440),
        (5000, 360, 400),
        (5412, 200, 150),
    ],
    corners=[
        CornerDef(1, "Turn 1", 400, 600, 260, 180),
        CornerDef(2, "Turn 2", 700, 1000, 340, 200),
        CornerDef(3, "Turn 3", 1100, 1500, 420, 180),
        CornerDef(4, "Turn 4", 1600, 2000, 500, 200),
        CornerDef(5, "Turn 5", 2100, 2500, 560, 260),
        CornerDef(6, "Turn 6", 2600, 3000, 600, 340),
        CornerDef(7, "Turn 7", 3100, 3500, 580, 420),
        CornerDef(8, "Turn 8", 3600, 4000, 520, 460),
        CornerDef(9, "Turn 9", 4100, 4500, 440, 440),
        CornerDef(10, "Turn 10", 4600, 5000, 360, 400),
        CornerDef(11, "Turn 11", 5000, 5412, 200, 150),
    ],
    sector_boundaries=[1804, 3608, 5412],
    start_finish_line_px=(200, 150),
    pit_entry_px=(210, 160),
    pit_exit_px=(190, 140),
)

# ---- Montreal (Canada) ---- #
TRACK_MAPS["montreal"] = TrackMapData(
    track_id="montreal",
    image_file="montreal.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 80),
        (400, 440, 120),
        (800, 480, 180),
        (1200, 500, 260),
        (1600, 480, 340),
        (2000, 440, 400),
        (2400, 380, 440),
        (2800, 320, 420),
        (3200, 280, 360),
        (3600, 260, 280),
        (4000, 280, 200),
        (4361, 400, 80),
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 500, 440, 120),
        CornerDef(2, "Turn 2", 500, 800, 480, 180),
        CornerDef(3, "Turn 3", 900, 1300, 500, 260),
        CornerDef(4, "Turn 4", 1400, 1700, 480, 340),
        CornerDef(5, "Turn 5", 1800, 2100, 440, 400),
        CornerDef(6, "Turn 6", 2200, 2500, 380, 440),
        CornerDef(7, "Turn 7", 2600, 3000, 320, 420),
        CornerDef(8, "Turn 8", 3100, 3400, 280, 360),
        CornerDef(9, "Turn 9", 3500, 3700, 260, 280),
        CornerDef(10, "Turn 10", 3800, 4000, 260, 200),
        CornerDef(11, "Turn 11", 4000, 4361, 280, 120),
        CornerDef(12, "Wall of Champions", 4200, 4361, 350, 90),
        CornerDef(13, "Final", 4300, 4361, 380, 85),
        CornerDef(14, "Start/Finish", 4350, 4361, 400, 80),
    ],
    sector_boundaries=[1454, 2907, 4361],
    start_finish_line_px=(400, 80),
    pit_entry_px=(410, 90),
    pit_exit_px=(390, 70),
)

# ---- Monaco ---- #
TRACK_MAPS["monaco"] = TrackMapData(
    track_id="monaco",
    image_file="monaco.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 350, 100),
        (300, 380, 140),
        (600, 420, 180),
        (900, 440, 240),
        (1200, 420, 300),
        (1500, 380, 360),
        (1800, 340, 400),
        (2100, 300, 420),
        (2400, 260, 380),
        (2700, 240, 320),
        (3000, 260, 260),
        (3337, 350, 100),
    ],
    corners=[
        CornerDef(1, "Sainte Devote", 200, 400, 380, 140),
        CornerDef(2, "Beau Rivage", 400, 700, 420, 180),
        CornerDef(3, "Massenet", 700, 1000, 440, 240),
        CornerDef(4, "Casino", 1000, 1300, 420, 300),
        CornerDef(5, "Mirabeau", 1300, 1600, 380, 360),
        CornerDef(6, "Grand Hotel", 1600, 1900, 340, 400),
        CornerDef(7, "Portier", 1900, 2200, 300, 420),
        CornerDef(8, "Tunnel", 2200, 2500, 260, 380),
        CornerDef(9, "Nouvelle Chicane", 2500, 2800, 240, 320),
        CornerDef(10, "Tabac", 2800, 3100, 260, 260),
        CornerDef(11, "Piscine", 3100, 3337, 300, 140),
        CornerDef(12, "Rascasse", 3200, 3337, 330, 120),
        CornerDef(13, "Anthony Noghes", 3300, 3337, 340, 110),
    ],
    sector_boundaries=[1112, 2225, 3337],
    start_finish_line_px=(350, 100),
    pit_entry_px=(360, 110),
    pit_exit_px=(340, 90),
)

# ---- Barcelona ---- #
TRACK_MAPS["barcelona"] = TrackMapData(
    track_id="barcelona",
    image_file="barcelona.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 350, 120),
        (500, 400, 160),
        (1000, 460, 220),
        (1500, 500, 300),
        (2000, 480, 380),
        (2500, 420, 430),
        (3000, 360, 420),
        (3500, 320, 360),
        (4000, 300, 280),
        (4657, 350, 120),
    ],
    corners=[
        CornerDef(1, "Turn 1", 350, 600, 400, 160),
        CornerDef(2, "Turn 2", 600, 900, 440, 200),
        CornerDef(3, "Turn 3", 900, 1200, 460, 220),
        CornerDef(4, "Turn 4", 1200, 1600, 500, 300),
        CornerDef(5, "Turn 5", 1600, 2000, 480, 380),
        CornerDef(6, "Turn 6", 2000, 2400, 440, 420),
        CornerDef(7, "Turn 7", 2400, 2700, 380, 430),
        CornerDef(8, "Turn 8", 2700, 3000, 360, 420),
        CornerDef(9, "Turn 9", 3100, 3400, 320, 360),
        CornerDef(10, "Turn 10", 3400, 3700, 300, 300),
        CornerDef(11, "Turn 11", 3700, 4000, 300, 280),
        CornerDef(12, "Turn 12", 4000, 4300, 320, 200),
        CornerDef(13, "Turn 13", 4300, 4500, 340, 160),
        CornerDef(14, "Final", 4500, 4657, 350, 130),
    ],
    sector_boundaries=[1552, 3105, 4657],
    start_finish_line_px=(350, 120),
    pit_entry_px=(360, 130),
    pit_exit_px=(340, 110),
)

# ---- Spielberg (Red Bull Ring) ---- #
TRACK_MAPS["spielberg"] = TrackMapData(
    track_id="spielberg",
    image_file="spielberg.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 100),
        (500, 440, 160),
        (1000, 480, 240),
        (1500, 520, 320),
        (2000, 500, 400),
        (2500, 440, 440),
        (3000, 380, 400),
        (3500, 340, 320),
        (4000, 360, 200),
        (4318, 400, 100),
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 600, 440, 160),
        CornerDef(2, "Turn 2", 700, 1100, 480, 240),
        CornerDef(3, "Turn 3", 1200, 1600, 520, 320),
        CornerDef(4, "Turn 4", 1700, 2100, 500, 400),
        CornerDef(5, "Turn 5", 2200, 2600, 440, 440),
        CornerDef(6, "Turn 6", 2700, 3100, 380, 400),
        CornerDef(7, "Turn 7", 3200, 3600, 340, 320),
        CornerDef(8, "Turn 8", 3600, 3900, 350, 240),
        CornerDef(9, "Turn 9", 3900, 4100, 360, 180),
        CornerDef(10, "Turn 10", 4100, 4318, 380, 120),
    ],
    sector_boundaries=[1439, 2879, 4318],
    start_finish_line_px=(400, 100),
    pit_entry_px=(410, 110),
    pit_exit_px=(390, 90),
)

# ---- Silverstone ---- #
TRACK_MAPS["silverstone"] = TrackMapData(
    track_id="silverstone",
    image_file="silverstone.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 100),
        (500, 460, 140),
        (1000, 520, 200),
        (1500, 580, 280),
        (2000, 600, 360),
        (2500, 560, 440),
        (3000, 480, 480),
        (3500, 400, 460),
        (4000, 340, 400),
        (4500, 300, 320),
        (5000, 320, 220),
        (5500, 360, 140),
        (5891, 400, 100),
    ],
    corners=[
        CornerDef(1, "Abbey", 350, 600, 460, 140),
        CornerDef(2, "Farm", 600, 900, 520, 200),
        CornerDef(3, "Village", 900, 1300, 560, 260),
        CornerDef(4, "The Loop", 1300, 1600, 580, 280),
        CornerDef(5, "Aintree", 1600, 1900, 600, 360),
        CornerDef(6, "Wellington", 1900, 2300, 580, 420),
        CornerDef(7, "Brooklands", 2300, 2600, 560, 440),
        CornerDef(8, "Luffield", 2600, 3000, 480, 480),
        CornerDef(9, "Copse", 3100, 3400, 400, 460),
        CornerDef(10, "Maggots 1", 3500, 3700, 340, 400),
        CornerDef(11, "Maggots 2", 3700, 3900, 320, 360),
        CornerDef(12, "Becketts", 3900, 4200, 300, 320),
        CornerDef(13, "Chapel", 4200, 4500, 300, 320),
        CornerDef(14, "Stowe", 4600, 5000, 320, 220),
        CornerDef(15, "Vale", 5000, 5300, 340, 180),
        CornerDef(16, "Club", 5300, 5600, 360, 140),
        CornerDef(17, "International Pit Straight", 5600, 5891, 380, 110),
        CornerDef(18, "Final", 5800, 5891, 400, 105),
    ],
    sector_boundaries=[1964, 3927, 5891],
    start_finish_line_px=(400, 100),
    pit_entry_px=(410, 110),
    pit_exit_px=(390, 90),
)

# ---- Spa ---- #
TRACK_MAPS["spa"] = TrackMapData(
    track_id="spa",
    image_file="spa.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 350, 100),
        (600, 400, 160),
        (1200, 460, 240),
        (1800, 520, 320),
        (2400, 580, 400),
        (3000, 620, 460),
        (3600, 580, 500),
        (4200, 500, 480),
        (4800, 420, 440),
        (5400, 360, 380),
        (6000, 320, 300),
        (6600, 300, 200),
        (7004, 350, 100),
    ],
    corners=[
        CornerDef(1, "La Source", 200, 500, 400, 160),
        CornerDef(2, "Eau Rouge", 600, 900, 440, 200),
        CornerDef(3, "Raidillon", 900, 1300, 480, 260),
        CornerDef(4, "Les Combes", 1500, 1900, 540, 340),
        CornerDef(5, "Malmedy", 1900, 2200, 580, 380),
        CornerDef(6, "Bruxelles", 2300, 2700, 600, 420),
        CornerDef(7, "Speaker's Corner", 2700, 3000, 620, 460),
        CornerDef(8, "Pouhon", 3200, 3600, 600, 500),
        CornerDef(9, "Fagnes", 3700, 4100, 540, 480),
        CornerDef(10, "Stavelot", 4200, 4600, 460, 440),
        CornerDef(11, "Blanchimont", 5000, 5600, 380, 380),
        CornerDef(12, "Bus Stop Chicane", 5800, 6300, 340, 300),
        CornerDef(13, "Final Chicane", 6300, 6600, 320, 220),
        CornerDef(14, "La Source Exit", 6600, 6800, 310, 180),
        CornerDef(15, "Start/Finish", 6800, 7004, 350, 100),
        CornerDef(16, "Turn 16", 7000, 7004, 350, 102),
        CornerDef(17, "Turn 17", 7002, 7004, 350, 103),
        CornerDef(18, "Turn 18", 7003, 7004, 350, 104),
        CornerDef(19, "Turn 19", 7004, 7004, 350, 100),
    ],
    sector_boundaries=[2335, 4669, 7004],
    start_finish_line_px=(350, 100),
    pit_entry_px=(360, 110),
    pit_exit_px=(340, 90),
)

# ---- Hungaroring ---- #
TRACK_MAPS["hungaroring"] = TrackMapData(
    track_id="hungaroring",
    image_file="hungaroring.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 100),
        (400, 440, 150),
        (800, 480, 220),
        (1200, 500, 300),
        (1600, 480, 380),
        (2000, 440, 430),
        (2400, 380, 440),
        (2800, 320, 400),
        (3200, 300, 340),
        (3600, 320, 260),
        (4000, 360, 180),
        (4381, 400, 100),
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 550, 440, 150),
        CornerDef(2, "Turn 2", 550, 850, 480, 220),
        CornerDef(3, "Turn 3", 900, 1300, 500, 300),
        CornerDef(4, "Turn 4", 1350, 1650, 480, 380),
        CornerDef(5, "Turn 5", 1700, 2050, 440, 430),
        CornerDef(6, "Turn 6", 2100, 2500, 380, 440),
        CornerDef(7, "Turn 7", 2550, 2850, 320, 400),
        CornerDef(8, "Turn 8", 2900, 3200, 300, 340),
        CornerDef(9, "Turn 9", 3250, 3550, 300, 300),
        CornerDef(10, "Turn 10", 3600, 3900, 320, 260),
        CornerDef(11, "Turn 11", 3950, 4200, 360, 180),
        CornerDef(12, "Turn 12", 4200, 4381, 380, 140),
        CornerDef(13, "Turn 13", 4300, 4381, 390, 120),
        CornerDef(14, "Final", 4350, 4381, 400, 110),
    ],
    sector_boundaries=[1460, 2921, 4381],
    start_finish_line_px=(400, 100),
    pit_entry_px=(410, 110),
    pit_exit_px=(390, 90),
)

# ---- Zandvoort ---- #
TRACK_MAPS["zandvoort"] = TrackMapData(
    track_id="zandvoort",
    image_file="zandvoort.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 120),
        (400, 450, 170),
        (800, 500, 240),
        (1200, 540, 320),
        (1600, 520, 400),
        (2000, 460, 440),
        (2400, 400, 420),
        (2800, 360, 360),
        (3200, 340, 280),
        (3600, 360, 200),
        (4000, 380, 160),
        (4259, 400, 120),
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 500, 450, 170),
        CornerDef(2, "Turn 2", 550, 850, 500, 240),
        CornerDef(3, "Turn 3", 900, 1250, 540, 320),
        CornerDef(4, "Hugenholtz", 1300, 1650, 520, 400),
        CornerDef(5, "Turn 5", 1700, 2050, 460, 440),
        CornerDef(6, "Turn 6", 2100, 2450, 400, 420),
        CornerDef(7, "Turn 7", 2500, 2850, 360, 360),
        CornerDef(8, "Turn 8", 2900, 3200, 340, 280),
        CornerDef(9, "Turn 9", 3250, 3550, 340, 240),
        CornerDef(10, "Arie Luyendyk", 3600, 3900, 360, 200),
        CornerDef(11, "Turn 11", 3950, 4100, 370, 170),
        CornerDef(12, "Turn 12", 4100, 4259, 385, 145),
        CornerDef(13, "Turn 13", 4200, 4259, 395, 130),
        CornerDef(14, "Final", 4230, 4259, 400, 125),
    ],
    sector_boundaries=[1420, 2840, 4259],
    start_finish_line_px=(400, 120),
    pit_entry_px=(410, 130),
    pit_exit_px=(390, 110),
)

# ---- Monza ---- #
TRACK_MAPS["monza"] = TrackMapData(
    track_id="monza",
    image_file="monza.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 100),
        (600, 440, 160),
        (1200, 480, 240),
        (1800, 520, 320),
        (2400, 560, 400),
        (3000, 540, 460),
        (3600, 480, 480),
        (4200, 420, 440),
        (4800, 380, 360),
        (5400, 360, 260),
        (5793, 400, 100),
    ],
    corners=[
        CornerDef(1, "Rettifilo", 350, 700, 480, 240),
        CornerDef(2, "Grande", 700, 1100, 500, 280),
        CornerDef(3, "Curva Grande", 1100, 1500, 520, 320),
        CornerDef(4, "Roggia", 1600, 2000, 540, 400),
        CornerDef(5, "Lesmo 1", 2100, 2500, 560, 400),
        CornerDef(6, "Lesmo 2", 2500, 2900, 540, 460),
        CornerDef(7, "Ascari", 3200, 3800, 480, 480),
        CornerDef(8, "Parabolica", 4200, 5000, 380, 360),
        CornerDef(9, "Turn 9", 5100, 5400, 370, 280),
        CornerDef(10, "Turn 10", 5400, 5600, 380, 200),
        CornerDef(11, "Final", 5600, 5793, 400, 120),
    ],
    sector_boundaries=[1931, 3862, 5793],
    start_finish_line_px=(400, 100),
    pit_entry_px=(410, 110),
    pit_exit_px=(390, 90),
)

# ---- Madrid (IFEMA) ---- #
TRACK_MAPS["madrid"] = TrackMapData(
    track_id="madrid",
    image_file="madrid.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 300, 150),
        (400, 340, 190),
        (800, 380, 250),
        (1200, 420, 320),
        (1600, 460, 380),
        (2000, 500, 420),
        (2400, 540, 400),
        (2800, 560, 340),
        (3200, 540, 280),
        (3600, 500, 220),
        (4000, 460, 180),
        (4400, 420, 160),
        (4800, 380, 160),
        (5200, 340, 170),
        (5416, 300, 150),
    ],
    corners=[
        CornerDef(i, f"T{i}", int(5416 * (i - 1) / 22), int(5416 * i / 22),
                  300 + (i * 12) % 260, 150 + (i * 11) % 270)
        for i in range(1, 23)
    ],
    sector_boundaries=[1805, 3611, 5416],
    start_finish_line_px=(300, 150),
    pit_entry_px=(310, 160),
    pit_exit_px=(290, 140),
)

# ---- Baku ---- #
TRACK_MAPS["baku"] = TrackMapData(
    track_id="baku",
    image_file="baku.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 200, 300),
        (600, 240, 260),
        (1200, 300, 220),
        (1800, 380, 200),
        (2400, 460, 220),
        (3000, 520, 280),
        (3600, 560, 360),
        (4200, 540, 440),
        (4800, 480, 480),
        (5400, 400, 460),
        (6003, 200, 300),
    ],
    corners=[
        CornerDef(i, f"Turn {i}", int(6003 * (i - 1) / 20), int(6003 * i / 20),
                  200 + (i * 18) % 360, 200 + (i * 13) % 280)
        for i in range(1, 21)
    ],
    sector_boundaries=[2001, 4002, 6003],
    start_finish_line_px=(200, 300),
    pit_entry_px=(210, 310),
    pit_exit_px=(190, 290),
)

# ---- Singapore ---- #
TRACK_MAPS["singapore"] = TrackMapData(
    track_id="singapore",
    image_file="singapore.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 250, 200),
        (400, 290, 240),
        (800, 340, 280),
        (1200, 400, 300),
        (1600, 460, 280),
        (2000, 500, 240),
        (2400, 530, 200),
        (2800, 540, 160),
        (3200, 520, 140),
        (3600, 480, 160),
        (4000, 440, 200),
        (4400, 400, 240),
        (4940, 250, 200),
    ],
    corners=[
        CornerDef(i, f"Turn {i}", int(4940 * (i - 1) / 19), int(4940 * i / 19),
                  250 + (i * 16) % 290, 160 + (i * 8) % 140)
        for i in range(1, 20)
    ],
    sector_boundaries=[1647, 3293, 4940],
    start_finish_line_px=(250, 200),
    pit_entry_px=(260, 210),
    pit_exit_px=(240, 190),
)

# ---- Austin (COTA) ---- #
TRACK_MAPS["austin"] = TrackMapData(
    track_id="austin",
    image_file="austin.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 100),
        (500, 450, 160),
        (1000, 500, 240),
        (1500, 540, 320),
        (2000, 520, 400),
        (2500, 460, 440),
        (3000, 400, 420),
        (3500, 360, 360),
        (4000, 340, 280),
        (4500, 360, 200),
        (5000, 380, 160),
        (5513, 400, 100),
    ],
    corners=[
        CornerDef(1, "Turn 1", 350, 650, 450, 160),
        CornerDef(2, "Turn 2", 700, 1050, 500, 240),
        CornerDef(3, "Turn 3", 1100, 1500, 540, 320),
        CornerDef(4, "Turn 4", 1550, 1850, 520, 360),
        CornerDef(5, "Turn 5", 1900, 2200, 480, 400),
        CornerDef(6, "Turn 6", 2250, 2600, 440, 440),
        CornerDef(7, "Turn 7", 2650, 2950, 400, 420),
        CornerDef(8, "Turn 8", 3000, 3300, 360, 380),
        CornerDef(9, "Turn 9", 3350, 3650, 340, 320),
        CornerDef(10, "Turn 10", 3700, 4000, 340, 280),
        CornerDef(11, "Turn 11", 4050, 4350, 340, 240),
        CornerDef(12, "Turn 12", 4400, 4650, 360, 200),
        CornerDef(13, "Turn 13", 4700, 4900, 370, 180),
        CornerDef(14, "Turn 14", 4900, 5100, 375, 160),
        CornerDef(15, "Turn 15", 5100, 5300, 380, 140),
        CornerDef(16, "Turn 16-18", 5300, 5450, 390, 120),
        CornerDef(17, "Turn 19", 5450, 5513, 400, 110),
        CornerDef(18, "Turn 20", 5480, 5513, 400, 105),
        CornerDef(19, "Final", 5500, 5513, 400, 102),
        CornerDef(20, "Start/Finish", 5510, 5513, 400, 100),
    ],
    sector_boundaries=[1838, 3675, 5513],
    start_finish_line_px=(400, 100),
    pit_entry_px=(410, 110),
    pit_exit_px=(390, 90),
)

# ---- Mexico City ---- #
TRACK_MAPS["mexico_city"] = TrackMapData(
    track_id="mexico_city",
    image_file="mexico_city.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 350, 100),
        (400, 390, 150),
        (800, 430, 220),
        (1200, 470, 300),
        (1600, 450, 380),
        (2000, 400, 420),
        (2400, 360, 400),
        (2800, 330, 340),
        (3200, 320, 260),
        (3600, 340, 180),
        (4000, 360, 130),
        (4304, 350, 100),
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 550, 390, 150),
        CornerDef(2, "Turn 2", 600, 900, 430, 220),
        CornerDef(3, "Turn 3", 950, 1300, 470, 300),
        CornerDef(4, "Turn 4", 1350, 1650, 450, 380),
        CornerDef(5, "Turn 5", 1700, 2050, 400, 420),
        CornerDef(6, "Turn 6", 2100, 2450, 360, 400),
        CornerDef(7, "Turn 7", 2500, 2850, 330, 340),
        CornerDef(8, "Turn 8", 2900, 3200, 320, 260),
        CornerDef(9, "Turn 9", 3250, 3550, 330, 220),
        CornerDef(10, "Turn 10", 3600, 3850, 340, 180),
        CornerDef(11, "Turn 11", 3900, 4100, 355, 150),
        CornerDef(12, "Turn 12", 4100, 4250, 358, 130),
        CornerDef(13, "Turn 13", 4250, 4304, 355, 115),
        CornerDef(14, "Turn 14", 4280, 4304, 353, 108),
        CornerDef(15, "Turn 15", 4295, 4304, 352, 104),
        CornerDef(16, "Turn 16", 4300, 4304, 351, 102),
        CornerDef(17, "Start/Finish", 4303, 4304, 350, 100),
    ],
    sector_boundaries=[1435, 2869, 4304],
    start_finish_line_px=(350, 100),
    pit_entry_px=(360, 110),
    pit_exit_px=(340, 90),
)

# ---- Sao Paulo (Interlagos) ---- #
TRACK_MAPS["sao_paulo"] = TrackMapData(
    track_id="sao_paulo",
    image_file="sao_paulo.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 300, 150),
        (400, 340, 200),
        (800, 380, 270),
        (1200, 420, 340),
        (1600, 400, 400),
        (2000, 360, 430),
        (2400, 320, 400),
        (2800, 300, 340),
        (3200, 300, 260),
        (3600, 300, 200),
        (4000, 300, 170),
        (4309, 300, 150),
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 550, 340, 200),
        CornerDef(2, "Turn 2", 600, 900, 380, 270),
        CornerDef(3, "Turn 3", 950, 1300, 420, 340),
        CornerDef(4, "Turn 4", 1350, 1650, 400, 400),
        CornerDef(5, "Turn 5", 1700, 2050, 360, 430),
        CornerDef(6, "Turn 6", 2100, 2500, 320, 400),
        CornerDef(7, "Turn 7", 2550, 2900, 300, 340),
        CornerDef(8, "Turn 8", 2950, 3250, 300, 280),
        CornerDef(9, "Turn 9", 3300, 3600, 300, 240),
        CornerDef(10, "Turn 10", 3650, 3900, 300, 200),
        CornerDef(11, "Turn 11", 3950, 4150, 300, 180),
        CornerDef(12, "Turn 12", 4150, 4309, 300, 160),
        CornerDef(13, "Turn 13", 4250, 4309, 300, 155),
        CornerDef(14, "Turn 14", 4280, 4309, 300, 152),
        CornerDef(15, "Start/Finish", 4305, 4309, 300, 150),
    ],
    sector_boundaries=[1436, 2873, 4309],
    start_finish_line_px=(300, 150),
    pit_entry_px=(310, 160),
    pit_exit_px=(290, 140),
)

# ---- Las Vegas ---- #
TRACK_MAPS["las_vegas"] = TrackMapData(
    track_id="las_vegas",
    image_file="las_vegas.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 150, 300),
        (600, 200, 260),
        (1200, 280, 220),
        (1800, 380, 200),
        (2400, 480, 220),
        (3000, 560, 280),
        (3600, 620, 360),
        (4200, 640, 440),
        (4800, 580, 480),
        (5400, 480, 460),
        (6000, 380, 400),
        (6201, 150, 300),
    ],
    corners=[
        CornerDef(1, "Turn 1", 400, 700, 200, 220),
        CornerDef(2, "Turn 2", 750, 1100, 280, 220),
        CornerDef(3, "Turn 3", 1150, 1500, 340, 200),
        CornerDef(4, "Turn 4", 1550, 1900, 420, 200),
        CornerDef(5, "Turn 5", 1950, 2300, 480, 220),
        CornerDef(6, "Turn 6", 2350, 2700, 540, 260),
        CornerDef(7, "Turn 7", 2750, 3100, 600, 320),
        CornerDef(8, "Turn 8", 3150, 3500, 640, 400),
        CornerDef(9, "Turn 9", 3550, 3900, 640, 440),
        CornerDef(10, "Turn 10", 3950, 4300, 600, 480),
        CornerDef(11, "Turn 11", 4350, 4700, 540, 480),
        CornerDef(12, "Turn 12", 4750, 5100, 460, 460),
        CornerDef(13, "Turn 13", 5150, 5500, 380, 420),
        CornerDef(14, "Turn 14", 5550, 5800, 300, 380),
        CornerDef(15, "Turn 15", 5850, 6000, 220, 340),
        CornerDef(16, "Turn 16", 6000, 6150, 180, 320),
        CornerDef(17, "Start/Finish", 6150, 6201, 160, 310),
    ],
    sector_boundaries=[2067, 4134, 6201],
    start_finish_line_px=(150, 300),
    pit_entry_px=(160, 310),
    pit_exit_px=(140, 290),
)

# ---- Lusail (Qatar) ---- #
TRACK_MAPS["lusail"] = TrackMapData(
    track_id="lusail",
    image_file="lusail.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 100),
        (500, 460, 160),
        (1000, 520, 240),
        (1500, 560, 320),
        (2000, 540, 400),
        (2500, 480, 440),
        (3000, 420, 420),
        (3500, 380, 360),
        (4000, 360, 280),
        (4500, 370, 200),
        (5000, 380, 160),
        (5419, 400, 100),
    ],
    corners=[
        CornerDef(1, "Turn 1", 350, 600, 460, 160),
        CornerDef(2, "Turn 2", 650, 1000, 520, 240),
        CornerDef(3, "Turn 3", 1050, 1400, 560, 320),
        CornerDef(4, "Turn 4", 1450, 1800, 540, 380),
        CornerDef(5, "Turn 5", 1850, 2200, 500, 420),
        CornerDef(6, "Turn 6", 2250, 2600, 460, 440),
        CornerDef(7, "Turn 7", 2650, 3000, 420, 420),
        CornerDef(8, "Turn 8", 3050, 3350, 380, 380),
        CornerDef(9, "Turn 9", 3400, 3650, 370, 320),
        CornerDef(10, "Turn 10", 3700, 3950, 360, 280),
        CornerDef(11, "Turn 11", 4000, 4250, 365, 240),
        CornerDef(12, "Turn 12", 4300, 4550, 370, 210),
        CornerDef(13, "Turn 13", 4600, 4850, 375, 180),
        CornerDef(14, "Turn 14", 4900, 5100, 380, 160),
        CornerDef(15, "Turn 15", 5100, 5300, 390, 140),
        CornerDef(16, "Start/Finish", 5300, 5419, 400, 110),
    ],
    sector_boundaries=[1806, 3613, 5419],
    start_finish_line_px=(400, 100),
    pit_entry_px=(410, 110),
    pit_exit_px=(390, 90),
)

# ---- Yas Marina (Abu Dhabi) ---- #
TRACK_MAPS["yas_marina"] = TrackMapData(
    track_id="yas_marina",
    image_file="yas_marina.svg",
    canvas_width=800,
    canvas_height=600,
    control_points=[
        (0, 400, 120),
        (400, 450, 170),
        (800, 500, 240),
        (1200, 540, 320),
        (1600, 520, 400),
        (2000, 460, 440),
        (2400, 400, 420),
        (2800, 360, 360),
        (3200, 340, 280),
        (3600, 360, 200),
        (4000, 380, 160),
        (4400, 400, 140),
        (4800, 410, 130),
        (5281, 400, 120),
    ],
    corners=[
        CornerDef(1, "Turn 1", 300, 550, 450, 170),
        CornerDef(2, "Turn 2", 600, 900, 500, 240),
        CornerDef(3, "Turn 3", 950, 1300, 540, 320),
        CornerDef(4, "Turn 4", 1350, 1650, 520, 400),
        CornerDef(5, "Turn 5", 1700, 2050, 460, 440),
        CornerDef(6, "Turn 6", 2100, 2450, 400, 420),
        CornerDef(7, "Turn 7", 2500, 2850, 360, 360),
        CornerDef(8, "Turn 8", 2900, 3200, 340, 280),
        CornerDef(9, "Turn 9", 3250, 3550, 340, 240),
        CornerDef(10, "Turn 10", 3600, 3900, 360, 200),
        CornerDef(11, "Turn 11", 3950, 4200, 375, 180),
        CornerDef(12, "Turn 12", 4250, 4500, 390, 160),
        CornerDef(13, "Turn 13", 4550, 4800, 400, 150),
        CornerDef(14, "Turn 14", 4850, 5100, 405, 140),
        CornerDef(15, "Turn 15", 5100, 5250, 402, 135),
        CornerDef(16, "Start/Finish", 5250, 5281, 400, 125),
    ],
    sector_boundaries=[1760, 3521, 5281],
    start_finish_line_px=(400, 120),
    pit_entry_px=(410, 130),
    pit_exit_px=(390, 110),
)


def get_track_map(track_id: str) -> TrackMapData | None:
    """Get track mapping data for a given track_id."""
    return TRACK_MAPS.get(track_id)


def get_all_track_ids() -> list[str]:
    """Return all track_ids that have map data."""
    return list(TRACK_MAPS.keys())
