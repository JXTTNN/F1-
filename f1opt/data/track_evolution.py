"""Track evolution + weather effect models for F1 setup optimization.

Models how grip, track temperature, rubber build-up, and weather conditions
change across a session and impact car performance / setup choices.

References (textbook F1 engineering knowledge, no papers):
    - Benson, "F1 Track Evolution" (BBC Sport technical primer).
    - F1 official sporting regulations: tyre compound allocation rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from f1opt.numeric import clamp as _clamp

__all__ = [
    "TrackConditionSnapshot",
    "TrackEvolutionModel",
    "SessionTimeline",
    "WeatherCondition",
    "WeatherImpactModel",
    "WeatherForecast",
    "WindModel",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --------------------------------------------------------------------------- #
# Track condition snapshot
# --------------------------------------------------------------------------- #
@dataclass
class TrackConditionSnapshot:
    """Track state captured at a single moment."""

    grip_level: float = 0.85       # [0,1], 1 = max grip
    track_temp_c: float = 30.0     # track surface temperature °C
    ambient_temp_c: float = 25.0   # ambient air temperature °C
    rubber_level: float = 0.0      # [0,1], 0 = green track, 1 = fully rubbered
    wetness: float = 0.0           # [0,1], 0 = dry, 1 = soaking
    wind_speed_ms: float = 0.0
    wind_dir_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grip_level": self.grip_level,
            "track_temp_c": self.track_temp_c,
            "ambient_temp_c": self.ambient_temp_c,
            "rubber_level": self.rubber_level,
            "wetness": self.wetness,
            "wind_speed_ms": self.wind_speed_ms,
            "wind_dir_deg": self.wind_dir_deg,
        }


# --------------------------------------------------------------------------- #
# Track evolution model
# --------------------------------------------------------------------------- #
class TrackEvolutionModel:
    """Models grip / track-temp / rubber build-up across a session."""

    # Per-session growth rate constants (higher = faster rubber build-up).
    _SESSION_RUBBER_RATE = {
        "practice": 0.06,
        "qualifying": 0.10,
        "sprint": 0.08,
        "race": 0.04,
    }
    _SESSION_LENGTH_LAPS = {
        "practice": 60,
        "qualifying": 18,
        "sprint": 24,
        "race": 60,
    }

    def __init__(self, track_id: str, session_type: str = "race") -> None:
        self.track_id = track_id
        self.session_type = session_type if session_type in self._SESSION_RUBBER_RATE else "race"
        self._rate = self._SESSION_RUBBER_RATE[self.session_type]
        self._session_len = self._SESSION_LENGTH_LAPS[self.session_type]

    # ------------------------------------------------------------------ #
    def rubber_buildup(self, lap: int) -> float:
        """Rubber level [0,1] at a given lap. Asymptotic growth."""
        lap = max(0, lap)
        return _clamp(1.0 - math.exp(-self._rate * lap), 0.0, 1.0)

    def grip_level(self, lap: int) -> float:
        """Grip level [0,1]. Starts ~0.85, rises toward ~0.97 with rubber."""
        r = self.rubber_buildup(lap)
        # Grip improves with rubber, then plateaus.
        return _clamp(0.85 + 0.12 * r, 0.0, 1.0)

    def track_temp_progression(self, lap: int, ambient_c: float) -> float:
        """Track surface temp °C. Rises mid-session, cools late."""
        lap = max(0, lap)
        # Peak around 50% of session, falls off late.
        frac = lap / max(1, self._session_len)
        peak_boost = 12.0 * math.sin(math.pi * _clamp(frac, 0.0, 1.0))
        return ambient_c + 8.0 + peak_boost  # baseline +8 + sine bump

    def lap_time_delta_from_evolution(self, lap: int) -> float:
        """Seconds vs lap 1 from track evolution (negative = faster)."""
        grip0 = self.grip_level(0)
        grip = self.grip_level(lap)
        # More grip → faster. Map grip delta to seconds (empirical ~1.5s total).
        delta_grip = grip - grip0  # 0 → ~0.12
        return -1.5 * (delta_grip / 0.12)  # up to -1.5s

    def optimal_lap_window(self) -> tuple[int, int]:
        """Peak-grip lap window per session type."""
        windows = {
            "practice": (15, 40),
            "qualifying": (3, 12),
            "sprint": (8, 18),
            "race": (10, 30),
        }
        return windows.get(self.session_type, (10, 30))

    def marbles_offline_grip_penalty(self, distance_from_racing_line_m: float) -> float:
        """Off-line grip penalty (marbles). 0 on line, up to -0.15 at 2m+ off."""
        d = max(0.0, distance_from_racing_line_m)
        # Linear ramp to -0.15 at 2m, then flat.
        return _clamp(-0.075 * d, -0.15, 0.0)


# --------------------------------------------------------------------------- #
# Session timeline
# --------------------------------------------------------------------------- #
class SessionTimeline:
    """Track condition snapshots across a session, with interpolation."""

    def __init__(self, track_id: str) -> None:
        self.track_id = track_id
        self._records: dict[int, TrackConditionSnapshot] = {}

    def record(self, lap: int, snapshot: TrackConditionSnapshot) -> None:
        self._records[int(lap)] = snapshot

    def at(self, lap: int) -> TrackConditionSnapshot:
        """Return snapshot at lap, interpolating between recorded laps."""
        if not self._records:
            return TrackConditionSnapshot()
        lap = int(lap)
        if lap in self._records:
            return self._records[lap]
        # Find bracketing recorded laps.
        laps_sorted = sorted(self._records.keys())
        if lap <= laps_sorted[0]:
            return self._records[laps_sorted[0]]
        if lap >= laps_sorted[-1]:
            return self._records[laps_sorted[-1]]
        # Interpolate.
        lo = max(lp for lp in laps_sorted if lp <= lap)
        hi = min(lp for lp in laps_sorted if lp >= lap)
        if lo == hi:
            return self._records[lo]
        frac = (lap - lo) / (hi - lo)
        s_lo = self._records[lo]
        s_hi = self._records[hi]
        return TrackConditionSnapshot(
            grip_level=s_lo.grip_level + frac * (s_hi.grip_level - s_lo.grip_level),
            track_temp_c=s_lo.track_temp_c + frac * (s_hi.track_temp_c - s_lo.track_temp_c),
            ambient_temp_c=s_lo.ambient_temp_c + frac * (s_hi.ambient_temp_c - s_lo.ambient_temp_c),
            rubber_level=s_lo.rubber_level + frac * (s_hi.rubber_level - s_lo.rubber_level),
            wetness=s_lo.wetness + frac * (s_hi.wetness - s_lo.wetness),
            wind_speed_ms=s_lo.wind_speed_ms + frac * (s_hi.wind_speed_ms - s_lo.wind_speed_ms),
            wind_dir_deg=s_lo.wind_dir_deg + frac * (s_hi.wind_dir_deg - s_lo.wind_dir_deg),
        )

    def trend(self, metric: str) -> str:
        """Trend of a metric across recorded laps: improving/degrading/stable."""
        if len(self._records) < 2:
            return "stable"
        laps_sorted = sorted(self._records.keys())
        first = getattr(self._records[laps_sorted[0]], metric)
        last = getattr(self._records[laps_sorted[-1]], metric)
        delta = last - first
        if abs(delta) < 1e-3:
            return "stable"
        # For grip_level/rubber_level: increasing = improving; for wetness: decreasing = improving.
        improving_metrics = {"grip_level", "rubber_level", "visibility_m"}
        if metric in improving_metrics:
            return "improving" if delta > 0 else "degrading"
        # For wetness/track_temp (degrading metrics): decreasing = improving.
        return "improving" if delta < 0 else "degrading"


# --------------------------------------------------------------------------- #
# Weather condition
# --------------------------------------------------------------------------- #
@dataclass
class WeatherCondition:
    """Weather state at a moment."""

    ambient_temp_c: float = 25.0
    track_temp_c: float = 30.0
    humidity_pct: float = 50.0
    precipitation_mm: float = 0.0  # mm/h
    wind_speed_ms: float = 0.0
    wind_dir_deg: float = 0.0
    visibility_m: float = 10000.0
    pressure_hpa: float = 1013.0

    def wetness(self) -> float:
        """Derived wetness [0,1]: 0 dry, 1 soaking."""
        # Combines precipitation + humidity above 85%.
        precip_term = _clamp(self.precipitation_mm / 10.0, 0.0, 1.0)  # 10mm/h = soaking
        humidity_term = _clamp((self.humidity_pct - 85.0) / 15.0, 0.0, 1.0)
        return _clamp(max(precip_term, 0.5 * humidity_term), 0.0, 1.0)

    def is_dry(self) -> bool:
        return self.wetness() < 0.15

    def is_intermediate(self) -> bool:
        w = self.wetness()
        return 0.15 <= w < 0.6

    def is_wet(self) -> bool:
        return self.wetness() >= 0.6

    def compound_recommendation(self) -> str:
        """Tyre compound recommendation based on conditions."""
        w = self.wetness()
        if w >= 0.6:
            return "wet"
        if w >= 0.15:
            return "intermediate"
        # Dry: choose by temp.
        if self.track_temp_c >= 35:
            return "hard"  # hot → hard to manage temps
        if self.track_temp_c <= 15:
            return "soft"  # cold → soft for warm-up
        return "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambient_temp_c": self.ambient_temp_c,
            "track_temp_c": self.track_temp_c,
            "humidity_pct": self.humidity_pct,
            "precipitation_mm": self.precipitation_mm,
            "wind_speed_ms": self.wind_speed_ms,
            "wind_dir_deg": self.wind_dir_deg,
            "visibility_m": self.visibility_m,
            "pressure_hpa": self.pressure_hpa,
            "wetness": self.wetness(),
        }


# --------------------------------------------------------------------------- #
# Weather impact model
# --------------------------------------------------------------------------- #
class WeatherImpactModel:
    """How weather affects car performance."""

    def grip_multiplier(self, weather: WeatherCondition) -> float:
        """Grip multiplier [0,1]: 1.0 dry, lower in wet."""
        w = weather.wetness()
        # Dry=1.0, light rain ~0.7, heavy ~0.4.
        return _clamp(1.0 - 0.6 * w, 0.3, 1.0)

    def lap_time_delta(self, weather: WeatherCondition, base_lap_time: float) -> float:
        """Seconds added by weather. Dry: 0. Wet: +15-25s."""
        w = weather.wetness()
        if w < 0.01:
            return 0.0
        # Scale wet penalty by wetness.
        return 25.0 * w

    def downforce_loss(self, weather: WeatherCondition) -> float:
        """Aero efficiency loss [0,1] in wet (water drag + air density)."""
        w = weather.wetness()
        return _clamp(0.10 * w, 0.0, 0.15)

    def tire_temp_impact(self, weather: WeatherCondition) -> float:
        """Delta to tire operating temp (°C). Hot +10, wet -20."""
        ambient_delta = weather.ambient_temp_c - 25.0
        wet_penalty = -20.0 * weather.wetness()
        return _clamp(ambient_delta * 0.4 + wet_penalty, -30.0, 15.0)

    def brake_temp_impact(self, weather: WeatherCondition) -> float:
        """Delta to brake operating temp (°C)."""
        ambient_delta = weather.ambient_temp_c - 25.0
        wet_penalty = -30.0 * weather.wetness()
        return _clamp(ambient_delta * 0.5 + wet_penalty, -40.0, 20.0)

    def visibility_impact(self, weather: WeatherCondition) -> float:
        """Visibility quality [0,1]: 1 clear, 0 fog."""
        return _clamp(weather.visibility_m / 10000.0, 0.0, 1.0)

    def setup_adjustment_recommendations(
        self, weather: WeatherCondition, track_type: str
    ) -> list[dict[str, Any]]:
        """Setup adjustments for weather conditions."""
        recs: list[dict[str, Any]] = []
        w = weather.wetness()
        if w < 0.15:
            # Dry: minor temp-based adjustments.
            if weather.track_temp_c > 35:
                recs.append({
                    "field": "front_tyre_pressure",
                    "direction": "decrease",
                    "magnitude": 1,
                    "reason": "高温赛道降低胎压防止过热",
                })
            return recs
        # Wet/intermediate: raise ride height, more wing, softer springs.
        recs.append({
            "field": "front_ride_height",
            "direction": "increase",
            "magnitude": 2,
            "reason": "升高离地间隙防止底盘触底积水",
        })
        recs.append({
            "field": "rear_ride_height",
            "direction": "increase",
            "magnitude": 2,
            "reason": "同步升高后部保持rake",
        })
        recs.append({
            "field": "rear_wing",
            "direction": "increase",
            "magnitude": 2,
            "reason": "增加下压力弥补湿地抓地力不足",
        })
        recs.append({
            "field": "front_suspension",
            "direction": "decrease",
            "magnitude": 1,
            "reason": "软化悬挂提升湿地路感适应",
        })
        recs.append({
            "field": "rear_suspension",
            "direction": "decrease",
            "magnitude": 1,
            "reason": "软化后悬挂提升出弯牵引力",
        })
        if w >= 0.6:
            recs.append({
                "field": "front_arb",
                "direction": "decrease",
                "magnitude": 2,
                "reason": "大雨软化前防倾杆减少突然滑移",
            })
        return recs


# --------------------------------------------------------------------------- #
# Weather forecast
# --------------------------------------------------------------------------- #
class WeatherForecast:
    """Forecast weather across a session with interpolation."""

    def __init__(self, initial: WeatherCondition) -> None:
        self._points: dict[int, WeatherCondition] = {0: initial}

    def add_change(self, lap: int, new: WeatherCondition) -> None:
        self._points[int(lap)] = new

    def forecast_at(self, lap: int) -> WeatherCondition:
        """Linearly interpolate weather at lap."""
        if not self._points:
            return WeatherCondition()
        lap = int(lap)
        if lap in self._points:
            return self._points[lap]
        laps_sorted = sorted(self._points.keys())
        if lap <= laps_sorted[0]:
            return self._points[laps_sorted[0]]
        if lap >= laps_sorted[-1]:
            return self._points[laps_sorted[-1]]
        lo = max(lp for lp in laps_sorted if lp <= lap)
        hi = min(lp for lp in laps_sorted if lp >= lap)
        if lo == hi:
            return self._points[lo]
        frac = (lap - lo) / (hi - lo)
        w_lo = self._points[lo]
        w_hi = self._points[hi]
        return WeatherCondition(
            ambient_temp_c=w_lo.ambient_temp_c + frac * (w_hi.ambient_temp_c - w_lo.ambient_temp_c),
            track_temp_c=w_lo.track_temp_c + frac * (w_hi.track_temp_c - w_lo.track_temp_c),
            humidity_pct=w_lo.humidity_pct + frac * (w_hi.humidity_pct - w_lo.humidity_pct),
            precipitation_mm=(
                w_lo.precipitation_mm + frac * (w_hi.precipitation_mm - w_lo.precipitation_mm)
            ),
            wind_speed_ms=w_lo.wind_speed_ms + frac * (w_hi.wind_speed_ms - w_lo.wind_speed_ms),
            wind_dir_deg=w_lo.wind_dir_deg + frac * (w_hi.wind_dir_deg - w_lo.wind_dir_deg),
            visibility_m=w_lo.visibility_m + frac * (w_hi.visibility_m - w_lo.visibility_m),
            pressure_hpa=w_lo.pressure_hpa + frac * (w_hi.pressure_hpa - w_lo.pressure_hpa),
        )

    def will_change_dry_to_wet(self) -> bool:
        """Detect dry → wet transition in forecast."""
        laps_sorted = sorted(self._points.keys())
        if len(laps_sorted) < 2:
            return False
        first = self._points[laps_sorted[0]]
        last = self._points[laps_sorted[-1]]
        return first.is_dry() and last.is_wet()

    def will_change_wet_to_dry(self) -> bool:
        laps_sorted = sorted(self._points.keys())
        if len(laps_sorted) < 2:
            return False
        first = self._points[laps_sorted[0]]
        last = self._points[laps_sorted[-1]]
        return first.is_wet() and last.is_dry()

    def strategy_recommendation(self) -> str:
        """Chinese strategy recommendation based on forecast."""
        if self.will_change_dry_to_wet():
            return "即将下雨，考虑提前进站换雨胎"
        if self.will_change_wet_to_dry():
            return "雨势减弱，可考虑换干胎"
        if self._points:
            current = self._points[min(self._points.keys())]
            if current.is_wet():
                return "持续雨天，保持雨胎并降低节奏"
        return "天气稳定，保持当前轮胎"


# --------------------------------------------------------------------------- #
# Wind model
# --------------------------------------------------------------------------- #
class WindModel:
    """Wind effects on F1 lap time + aero balance."""

    def headwind_component(
        self, wind_speed_ms: float, wind_dir_deg: float, car_dir_deg: float
    ) -> float:
        """Headwind (+) or tailwind (-) in m/s."""
        # Relative angle between wind and car direction.
        rel = math.radians(wind_dir_deg - car_dir_deg)
        # cos(rel) > 0 → headwind (wind blowing toward car).
        return wind_speed_ms * math.cos(rel)

    def crosswind_component(
        self, wind_speed_ms: float, wind_dir_deg: float, car_dir_deg: float
    ) -> float:
        """Crosswind in m/s (perpendicular to car)."""
        rel = math.radians(wind_dir_deg - car_dir_deg)
        return wind_speed_ms * math.sin(rel)

    def lap_time_wind_effect(
        self, wind_speed_ms: float, wind_dir_deg: float, track_bearing_deg: float
    ) -> float:
        """Net seconds effect around a lap (positive = slower).

        Approximation: headwind on main straight costs (drag), tailwind helps
        less (drag reduction smaller at high speed), crosswind hurts aero
        balance. Net effect usually positive (slower) for strong winds.
        """
        head = self.headwind_component(wind_speed_ms, wind_dir_deg, track_bearing_deg)
        cross = self.crosswind_component(wind_speed_ms, wind_dir_deg, track_bearing_deg)
        # Headwind costs ~0.3s per m/s on a long straight; tailwind recovers ~0.15s.
        straight_effect = 0.3 * max(head, 0) - 0.15 * max(-head, 0)
        # Crosswind hurts aero balance + driver confidence.
        cross_effect = 0.05 * abs(cross)
        return straight_effect + cross_effect

    def aero_balance_shift(self, crosswind_ms: float) -> float:
        """How crosswind shifts aero balance [-1,1]."""
        return _clamp(crosswind_ms / 20.0, -1.0, 1.0)
