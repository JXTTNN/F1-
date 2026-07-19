"""Race strategy planning: pit/tire strategy + stint simulation.

Provides:

- :class:`RaceStrategyPlanner` — plan 0/1/2-stop strategies, pick the optimal
  one by estimated total time, project tire wear and fuel burn.
- :class:`StintSimulator` — per-lap simulation of a single stint (lap time,
  tire wear, fuel, cumulative time) with compound-dependent degradation.
- :class:`StrategyComparator` — rank / compare candidate strategies and emit a
  Chinese recommendation summary.

Pure-python (only depends on :mod:`f1opt.data.tracks` for track metadata); no
DNN surrogate required.
"""

from __future__ import annotations

from f1opt.data.tracks import TRACKS_BY_ID

__all__ = [
    "RaceStrategyPlanner",
    "StintSimulator",
    "StrategyComparator",
]

# --------------------------------------------------------------------------- #
# Compound / track constants
# --------------------------------------------------------------------------- #
# Cumulative tire-wear rate (% per lap, linear). Soft is fastest but wears
# quickest; hard is most durable.
_COMPOUND_WEAR_RATE: dict[str, float] = {
    "soft": 3.0,
    "medium": 1.8,
    "hard": 1.0,
}
_DEFAULT_WEAR_RATE = 1.8
# Lap-time penalty (seconds) per 1% of cumulative tire wear.
_DEGRADATION_PENALTY = 0.1
# Nominal fuel at the start of a simulated stint (kg). StintSimulator does not
# receive a fuel load, so it projects from this nominal value.
_NOMINAL_STINT_FUEL = 80.0
# Base fuel burn rate (kg/lap) at a 5 km reference track.
_FUEL_BURN_BASE = 1.6
_REFERENCE_LENGTH_M = 5000.0

# Average speed prior (m/s) by track type, used to derive base lap time.
_AVG_SPEED: dict[str, float] = {
    "high_speed_low_downforce": 80.0,
    "street": 50.0,
    "high_downforce": 65.0,
    "medium": 70.0,
    "mixed": 72.0,
}
_DEFAULT_AVG_SPEED = 70.0

# Pit-stop time loss (seconds) by track id (curated). Monaco is tightest
# (lowest average speed → smallest pit loss); Monza is highest.
_PIT_LOSS: dict[str, float] = {
    "monaco": 22.0,
    "singapore": 23.5,
    "melbourne": 23.0,
    "monza": 24.0,
    "spa": 23.5,
    "silverstone": 23.0,
    "suzuka": 23.0,
    "shanghai": 23.5,
    "baku": 23.0,
    "jeddah": 23.5,
    "miami": 23.0,
    "montreal": 23.0,
    "barcelona": 23.0,
    "spielberg": 22.5,
    "hungaroring": 22.5,
    "zandvoort": 22.5,
    "madrid": 23.0,
    "austin": 23.0,
    "mexico_city": 22.5,
    "sao_paulo": 23.0,
    "las_vegas": 23.5,
    "lusail": 23.5,
    "yas_marina": 23.5,
    "sakhir": 23.5,
}
_PIT_LOSS_BY_TYPE: dict[str, float] = {
    "street": 22.0,
    "high_speed_low_downforce": 24.0,
    "high_downforce": 23.0,
    "medium": 23.0,
    "mixed": 23.0,
}
_DEFAULT_PIT_LOSS = 23.0


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #
def _wear_rate(compound: str) -> float:
    return _COMPOUND_WEAR_RATE.get(compound, _DEFAULT_WEAR_RATE)


def _wear_projection(compound: str, stint_length: int) -> list[float]:
    """Cumulative tire wear (%) at the end of each lap (1-indexed)."""
    rate = _wear_rate(compound)
    return [rate * (k + 1) for k in range(int(stint_length))]


def _base_lap_time(track_id: str) -> float:
    """Estimate a baseline dry lap time (s) from track length / type."""
    track = TRACKS_BY_ID.get(track_id)
    if track is None:
        return 90.0
    speed = _AVG_SPEED.get(track.track_type, _DEFAULT_AVG_SPEED)
    return float(track.length_m / speed)


def _fuel_burn_rate(track_id: str) -> float:
    """Fuel burn (kg/lap) scaled by track length vs the 5 km reference."""
    track = TRACKS_BY_ID.get(track_id)
    scale = (track.length_m / _REFERENCE_LENGTH_M) if track is not None else 1.0
    return _FUEL_BURN_BASE * scale


def _pit_loss(track_id: str) -> float:
    if track_id in _PIT_LOSS:
        return _PIT_LOSS[track_id]
    track = TRACKS_BY_ID.get(track_id)
    if track is None:
        return _DEFAULT_PIT_LOSS
    return _PIT_LOSS_BY_TYPE.get(track.track_type, _DEFAULT_PIT_LOSS)


# --------------------------------------------------------------------------- #
# StintSimulator
# --------------------------------------------------------------------------- #
class StintSimulator:
    """Simulate a single stint on one compound.

    Lap time degrades linearly with cumulative tire wear
    (``base + wear_pct * 0.1``); fuel decreases by the track-scaled burn rate
    each lap.
    """

    def __init__(
        self,
        compound: str,
        stint_length: int,
        track_id: str,
        base_lap_time: float,
    ) -> None:
        self.compound = compound
        self.stint_length = int(stint_length)
        self.track_id = track_id
        self.base_lap_time = float(base_lap_time)

    # ------------------------------------------------------------------ #
    def _wear_curve(self) -> list[float]:
        return _wear_projection(self.compound, self.stint_length)

    # ------------------------------------------------------------------ #
    def degradation_curve(self) -> list[float]:
        """Lap time (s) per lap — shows the degradation pattern."""
        return [
            self.base_lap_time + w * _DEGRADATION_PENALTY
            for w in self._wear_curve()
        ]

    # ------------------------------------------------------------------ #
    def simulate(self) -> list[dict]:
        """Per-lap records: ``{lap, lap_time, tire_wear_pct, fuel_kg,
        cumulative_time}``."""
        wear = self._wear_curve()
        burn = _fuel_burn_rate(self.track_id)
        fuel = _NOMINAL_STINT_FUEL
        cumulative = 0.0
        out: list[dict] = []
        for k in range(self.stint_length):
            lap_time = self.base_lap_time + wear[k] * _DEGRADATION_PENALTY
            cumulative += lap_time
            fuel = max(0.0, fuel - burn)
            out.append(
                {
                    "lap": k + 1,
                    "lap_time": float(lap_time),
                    "tire_wear_pct": float(wear[k]),
                    "fuel_kg": float(fuel),
                    "cumulative_time": float(cumulative),
                }
            )
        return out

    # ------------------------------------------------------------------ #
    def total_time(self) -> float:
        """Total stint time (s) = sum of per-lap times."""
        return float(sum(self.degradation_curve()))

    # ------------------------------------------------------------------ #
    def avg_lap_time(self) -> float:
        """Average lap time (s); 0.0 for an empty stint."""
        if self.stint_length <= 0:
            return 0.0
        return self.total_time() / self.stint_length


# --------------------------------------------------------------------------- #
# RaceStrategyPlanner
# --------------------------------------------------------------------------- #
class RaceStrategyPlanner:
    """Plan pit + tire strategies for a race and pick the optimal one."""

    def __init__(self, track_id: str, total_laps: int, fuel_load_kg: float) -> None:
        self.track_id = track_id
        self.total_laps = int(total_laps)
        self.fuel_load_kg = float(fuel_load_kg)

    # ------------------------------------------------------------------ #
    def tire_wear_projection(self, compound: str, stint_length: int) -> list[float]:
        """Projected cumulative wear (%) per lap for a compound/stint."""
        return _wear_projection(compound, stint_length)

    # ------------------------------------------------------------------ #
    def fuel_projection(self) -> list[float]:
        """Fuel (kg) at each lap boundary (length ``total_laps + 1``)."""
        burn = _fuel_burn_rate(self.track_id)
        return [
            max(0.0, self.fuel_load_kg - k * burn) for k in range(self.total_laps + 1)
        ]

    # ------------------------------------------------------------------ #
    def pit_loss_time(self, track_id: str) -> float:
        """Pit-stop time loss (s), track-dependent (Monaco ~22s, Monza ~24s)."""
        return _pit_loss(track_id)

    # ------------------------------------------------------------------ #
    def plan_no_stop(self, compound: str = "hard") -> dict:
        """Plan a 0-stop strategy on a single compound."""
        base = _base_lap_time(self.track_id)
        stint = StintSimulator(compound, self.total_laps, self.track_id, base)
        return {
            "stops": [],
            "total_time_est": stint.total_time(),
            "tire_wear_projection": stint._wear_curve(),
            "fuel_projection": self.fuel_projection(),
        }

    # ------------------------------------------------------------------ #
    def plan_one_stop(self, tire_compounds: list[str] | None = None) -> dict:
        """Plan a 1-stop strategy across two compounds."""
        if tire_compounds is None:
            tire_compounds = ["medium", "hard"]
        c0 = tire_compounds[0]
        c1 = tire_compounds[-1]
        if self.total_laps <= 0:
            n1, n2 = 0, 0
        else:
            n1 = max(1, self.total_laps // 2)
            n2 = self.total_laps - n1
        base = _base_lap_time(self.track_id)
        s1 = StintSimulator(c0, n1, self.track_id, base)
        s2 = StintSimulator(c1, n2, self.track_id, base)
        pit = self.pit_loss_time(self.track_id) if (n1 > 0 and n2 > 0) else 0.0
        total = s1.total_time() + pit + s2.total_time()
        stops: list[dict] = []
        if n1 > 0 and n2 > 0:
            stops.append(
                {
                    "lap": n1,
                    "compound_in": c0,
                    "compound_out": c1,
                    "reason": f"第{n1}圈进站, 由 {c0} 换 {c1}, 平衡圈速与胎耗",
                }
            )
        return {
            "stops": stops,
            "total_time_est": float(total),
            "tire_wear_projection": s1._wear_curve() + s2._wear_curve(),
            "fuel_projection": self.fuel_projection(),
        }

    # ------------------------------------------------------------------ #
    def plan_two_stop(self, tire_compounds: list[str] | None = None) -> dict:
        """Plan a 2-stop strategy across three compounds."""
        if tire_compounds is None:
            tire_compounds = ["soft", "medium", "hard"]
        comps = list(tire_compounds)
        while len(comps) < 3:
            comps.append(comps[-1] if comps else "hard")
        c0, c1, c2 = comps[0], comps[1], comps[2]
        if self.total_laps <= 0:
            n1 = n2 = n3 = 0
        else:
            n1 = max(1, self.total_laps // 3)
            n2 = max(1, self.total_laps // 3)
            n3 = self.total_laps - n1 - n2
            if n3 < 0:
                n3 = 0
        base = _base_lap_time(self.track_id)
        s1 = StintSimulator(c0, n1, self.track_id, base)
        s2 = StintSimulator(c1, n2, self.track_id, base)
        s3 = StintSimulator(c2, n3, self.track_id, base)
        pit = self.pit_loss_time(self.track_id)
        n_pits = (1 if n1 > 0 and (n2 > 0 or n3 > 0) else 0) + (
            1 if n2 > 0 and n3 > 0 else 0
        )
        total = s1.total_time() + s2.total_time() + s3.total_time() + n_pits * pit
        stops: list[dict] = []
        if n1 > 0 and (n2 > 0 or n3 > 0):
            stops.append(
                {
                    "lap": n1,
                    "compound_in": c0,
                    "compound_out": c1,
                    "reason": f"第{n1}圈进站, 由 {c0} 换 {c1}, 利用软胎起速",
                }
            )
        if n2 > 0 and n3 > 0:
            stops.append(
                {
                    "lap": n1 + n2,
                    "compound_in": c1,
                    "compound_out": c2,
                    "reason": f"第{n1 + n2}圈进站, 由 {c1} 换 {c2}, 收尾求稳",
                }
            )
        return {
            "stops": stops,
            "total_time_est": float(total),
            "tire_wear_projection": (
                s1._wear_curve() + s2._wear_curve() + s3._wear_curve()
            ),
            "fuel_projection": self.fuel_projection(),
        }

    # ------------------------------------------------------------------ #
    def _pick_compounds(
        self, available: list[str], n: int, default: list[str]
    ) -> list[str]:
        out: list[str] = []
        for i in range(n):
            if i < len(available):
                out.append(available[i])
            elif i < len(default):
                out.append(default[i])
            else:
                out.append(out[-1] if out else "hard")
        return out

    # ------------------------------------------------------------------ #
    def optimal_strategy(self, available_compounds: list[str]) -> dict:
        """Pick the best of 0/1/2-stop by estimated total time."""
        avail = list(available_compounds) if available_compounds else ["hard"]
        no_compound = (
            "hard" if "hard" in avail else ("medium" if "medium" in avail else avail[-1])
        )
        no_plan = self.plan_no_stop(no_compound)
        one_plan = self.plan_one_stop(
            self._pick_compounds(avail, 2, ["medium", "hard"])
        )
        two_plan = self.plan_two_stop(
            self._pick_compounds(avail, 3, ["soft", "medium", "hard"])
        )
        candidates: list[tuple[str, dict]] = [
            ("0-stop", no_plan),
            ("1-stop", one_plan),
            ("2-stop", two_plan),
        ]
        best_type, best_plan = min(
            candidates, key=lambda c: c[1]["total_time_est"]
        )
        parts: list[str] = []
        for typ, plan in candidates:
            gap = plan["total_time_est"] - best_plan["total_time_est"]
            parts.append(f"{typ} 总耗时 {plan['total_time_est']:.2f}s (差距 {gap:+.2f}s)")
        reason = f"推荐 {best_type} 策略: " + "；".join(parts) + "。"
        return {
            "strategy_type": best_type,
            "plan": best_plan,
            "total_time_est": best_plan["total_time_est"],
            "recommendation_reason": reason,
        }


# --------------------------------------------------------------------------- #
# StrategyComparator
# --------------------------------------------------------------------------- #
class StrategyComparator:
    """Rank and compare candidate strategy dicts (each with ``total_time_est``)."""

    def __init__(self, strategies: list[dict]) -> None:
        self.strategies: list[dict] = list(strategies)

    # ------------------------------------------------------------------ #
    def rank(self) -> list[tuple[int, float]]:
        """Return ``(original_index, total_time)`` ascending by total time."""
        indexed = [
            (i, float(s.get("total_time_est", float("inf"))))
            for i, s in enumerate(self.strategies)
        ]
        indexed.sort(key=lambda t: t[1])
        return indexed

    # ------------------------------------------------------------------ #
    def best(self) -> dict:
        """Return the strategy dict with the lowest total time."""
        ranked = self.rank()
        if not ranked:
            raise ValueError("no strategies to compare")
        return self.strategies[ranked[0][0]]

    # ------------------------------------------------------------------ #
    def gap_to_best(self, idx: int) -> float:
        """Seconds gap between strategy ``idx`` and the best (>= 0)."""
        ranked = self.rank()
        if not ranked:
            raise ValueError("no strategies to compare")
        best_time = ranked[0][1]
        return float(self.strategies[idx].get("total_time_est", float("inf"))) - best_time

    # ------------------------------------------------------------------ #
    def recommendation(self) -> str:
        """Chinese summary ranking the candidate strategies."""
        ranked = self.rank()
        if not ranked:
            return "无策略可比。"
        best_i, best_t = ranked[0]
        lines = [f"排名第1: 策略 {best_i} 总耗时 {best_t:.2f}s (基准)"]
        for pos, (i, t) in enumerate(ranked[1:], start=2):
            lines.append(
                f"排名第{pos}: 策略 {i} 总耗时 {t:.2f}s (落后 {t - best_t:.2f}s)"
            )
        return "；".join(lines) + "。"
