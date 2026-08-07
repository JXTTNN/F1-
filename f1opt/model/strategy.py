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

    # ------------------------------------------------------------------ #
    # Tire compound degradation crossover analysis (Iter-201)
    # ------------------------------------------------------------------ #
    def degradation_crossover(
        self,
        compound_a: str = "soft",
        compound_b: str = "medium",
    ) -> dict[str, Any]:
        """Find the lap where compound B becomes faster than compound A (Iter-201).

        In F1 strategy, softer compounds start faster but degrade more quickly.
        This method computes the crossover lap where the cumulative lap time of
        compound B exceeds that of compound A, indicating the optimal pit window
        for switching compounds.

        Args:
            compound_a: The softer/faster compound (e.g., "soft").
            compound_b: The harder/durable compound (e.g., "medium").

        Returns:
            Dict with ``crossover_lap``, ``lap_times`` per lap for both
            compounds, ``cumulative_times``, and ``recommendation``.
        """
        wear_a = _wear_rate(compound_a)
        wear_b = _wear_rate(compound_b)
        base = _base_lap_time(self.track_id)

        # Simulate lap-by-lap degradation for both compounds over full race.
        lap_times_a: list[float] = []
        lap_times_b: list[float] = []
        cum_a: list[float] = []
        cum_b: list[float] = []

        cum_a_total = 0.0
        cum_b_total = 0.0
        crossover_lap: int | None = None

        for lap in range(1, self.total_laps + 1):
            t_a = base + wear_a * lap * _DEGRADATION_PENALTY
            t_b = base + wear_b * lap * _DEGRADATION_PENALTY
            lap_times_a.append(round(t_a, 3))
            lap_times_b.append(round(t_b, 3))
            cum_a_total += t_a
            cum_b_total += t_b
            cum_a.append(round(cum_a_total, 2))
            cum_b.append(round(cum_b_total, 2))

            if crossover_lap is None and cum_b_total > cum_a_total:
                crossover_lap = lap

        if crossover_lap is None:
            crossover_lap = self.total_laps

        # Recommendation based on crossover
        if crossover_lap <= 0:
            recommendation = f"{compound_b} 从未优于 {compound_a}，全程使用 {compound_a}"
        elif crossover_lap >= self.total_laps:
            recommendation = (
                f"{compound_a} 全程优于 {compound_b}，"
                f"建议全程使用 {compound_a} 或考虑 0-stop 策略"
            )
        else:
            recommendation = (
                f"第 {crossover_lap} 圈后 {compound_b} 累积时间优于 {compound_a}，"
                f"建议在第 {max(1, crossover_lap - 2)}-{crossover_lap} 圈窗口进站换胎"
            )

        return {
            "compound_a": compound_a,
            "compound_b": compound_b,
            "crossover_lap": crossover_lap,
            "lap_times_a": lap_times_a,
            "lap_times_b": lap_times_b,
            "cumulative_time_a_s": cum_a,
            "cumulative_time_b_s": cum_b,
            "recommendation": recommendation,
            "total_laps": self.total_laps,
        }

    # ------------------------------------------------------------------ #
    # Fuel saving mode analysis (Iter-216)
    # ------------------------------------------------------------------ #
    def fuel_saving_analysis(self) -> dict[str, Any]:
        """Analyze fuel consumption and recommend fuel-saving strategies (Iter-216).

        Estimates whether the current fuel load is sufficient for the race
        distance and recommends fuel-saving modes (lift-and-coast, short-shift)
        if needed.

        Returns:
            Dict with ``fuel_sufficient``, ``fuel_deficit_kg``,
            ``fuel_saving_laps_needed``, ``recommended_mode``, and
            ``estimated_time_saved``.
        """
        burn = _fuel_burn_rate(self.track_id)
        total_burn = burn * self.total_laps
        deficit = total_burn - self.fuel_load_kg

        if deficit <= 0:
            return {
                "fuel_sufficient": True,
                "fuel_deficit_kg": 0.0,
                "fuel_saving_laps_needed": 0,
                "recommended_mode": "none",
                "estimated_time_saved": 0.0,
                "fuel_burn_per_lap_kg": round(burn, 3),
                "total_fuel_needed_kg": round(total_burn, 1),
                "fuel_load_kg": round(self.fuel_load_kg, 1),
            }

        # Fuel saving modes and their impact
        # Lift-and-coast: ~0.3s loss per lap, saves ~0.3 kg/lap
        # Short-shift: ~0.15s loss per lap, saves ~0.15 kg/lap
        modes = [
            {
                "mode": "lift_and_coast",
                "description": "Lift and coast before braking zones",
                "time_loss_per_lap_s": 0.3,
                "fuel_saved_per_lap_kg": 0.3,
                "max_laps": self.total_laps,
            },
            {
                "mode": "short_shift",
                "description": "Short-shift (early upshift) to reduce fuel flow",
                "time_loss_per_lap_s": 0.15,
                "fuel_saved_per_lap_kg": 0.15,
                "max_laps": self.total_laps,
            },
            {
                "mode": "combined",
                "description": "Lift-and-coast + short-shift combined",
                "time_loss_per_lap_s": 0.4,
                "fuel_saved_per_lap_kg": 0.4,
                "max_laps": self.total_laps,
            },
        ]

        best_mode = None
        best_laps = 0
        best_time_loss = float("inf")

        for mode in modes:
            save_per_lap = mode["fuel_saved_per_lap_kg"]
            laps_needed = int(min(deficit / save_per_lap, self.total_laps) + 0.99)
            time_loss = laps_needed * mode["time_loss_per_lap_s"]
            if time_loss < best_time_loss and laps_needed <= mode["max_laps"]:
                best_time_loss = time_loss
                best_laps = laps_needed
                best_mode = mode

        if best_mode is None:
            return {
                "fuel_sufficient": False,
                "fuel_deficit_kg": round(deficit, 1),
                "fuel_saving_laps_needed": self.total_laps,
                "recommended_mode": "insufficient",
                "estimated_time_saved": 0.0,
                "fuel_burn_per_lap_kg": round(burn, 3),
                "total_fuel_needed_kg": round(total_burn, 1),
                "fuel_load_kg": round(self.fuel_load_kg, 1),
            }

        return {
            "fuel_sufficient": False,
            "fuel_deficit_kg": round(deficit, 1),
            "fuel_saving_laps_needed": best_laps,
            "recommended_mode": best_mode["mode"],
            "recommended_description": best_mode["description"],
            "estimated_time_loss_s": round(best_time_loss, 1),
            "fuel_burn_per_lap_kg": round(burn, 3),
            "total_fuel_needed_kg": round(total_burn, 1),
            "fuel_load_kg": round(self.fuel_load_kg, 1),
        }

    # ------------------------------------------------------------------ #
    # Undercut/overcut analysis (Iter-216)
    # ------------------------------------------------------------------ #
    def undercut_overcut_analysis(
        self,
        compound: str = "soft",
        opponent_lap: int | None = None,
    ) -> dict[str, Any]:
        """Analyze undercut and overcut potential (Iter-216).

        In F1, the undercut (pitting earlier than opponent) can gain positions
        if the fresh tyre advantage outweighs the dirty-air loss. The overcut
        (pitting later) works when the opponent struggles on worn tyres.

        Args:
            compound: The fresh compound to compare against worn tyres.
            opponent_lap: The lap the opponent is expected to pit. If
                ``None``, defaults to half race distance.

        Returns:
            Dict with ``undercut_gain_s``, ``overcut_gain_s``,
            ``recommended_strategy``, and ``analysis``.
        """
        base = _base_lap_time(self.track_id)
        wear_rate = _wear_rate(compound)
        pit_loss = self.pit_loss_time(self.track_id)

        if opponent_lap is None:
            opponent_lap = max(1, self.total_laps // 2)

        # Fresh tyre lap time advantage
        fresh_lap = base  # fresh compound, no wear

        # Worn tyre lap time (at opponent's pit lap)
        worn_wear = wear_rate * opponent_lap
        worn_lap = base + worn_wear * _DEGRADATION_PENALTY

        # Per-lap delta on fresh vs worn tyres
        fresh_advantage_per_lap = worn_lap - fresh_lap

        # Undercut: pit 1 lap earlier, gain fresh-tyre advantage for 1 out-lap
        # Out-lap after pit stop is typically ~1.5s slower than a normal lap
        out_lap_penalty = 1.5
        undercut_gain = fresh_advantage_per_lap * 2 - out_lap_penalty  # 2 laps of advantage

        # Overcut: stay out longer, opponent comes out behind on fresh tyres
        # but you have clean air. The worn tyre penalty increases each lap.
        overcut_gain = fresh_advantage_per_lap * 1.5  # clean air advantage

        # Recommendation
        if undercut_gain > overcut_gain and undercut_gain > 0.5:
            recommended = "undercut"
            rationale = (
                f"推荐 Undercut：提前 1-2 圈进站，利用新胎优势在出站后超越对手。"
                f"预估净收益 {undercut_gain:.1f}s。"
            )
        elif overcut_gain > undercut_gain and overcut_gain > 0.5:
            recommended = "overcut"
            rationale = (
                f"推荐 Overcut：保持赛道位置，对手进站后利用干净空气刷快圈速。"
                f"预估净收益 {overcut_gain:.1f}s。"
            )
        else:
            recommended = "neutral"
            rationale = (
                f"Undercut 和 Overcut 收益相近（{undercut_gain:.1f}s vs {overcut_gain:.1f}s），"
                f"建议根据赛道实时情况决定。"
            )

        return {
            "undercut_gain_s": round(undercut_gain, 1),
            "overcut_gain_s": round(overcut_gain, 1),
            "fresh_advantage_per_lap_s": round(fresh_advantage_per_lap, 2),
            "worn_lap_time_at_opponent_pit_s": round(worn_lap, 2),
            "fresh_lap_time_s": round(fresh_lap, 2),
            "pit_loss_s": round(pit_loss, 1),
            "recommended_strategy": recommended,
            "rationale": rationale,
            "opponent_pit_lap": opponent_lap,
            "compound": compound,
        }

    # ------------------------------------------------------------------ #
    # Weather-aware strategy adjustments (Iter-202)
    # ------------------------------------------------------------------ #
    def weather_impact_on_strategy(
        self,
        track_wetness: float = 0.0,
        rain_intensity: float = 0.0,
        track_temp_c: float = 35.0,
        wind_speed_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Adjust race strategy based on weather conditions (Iter-202).

        Weather dramatically changes optimal strategy: wet conditions favor
        fewer stops (inter/wet tyres last longer), rain increases pit loss
        time, and extreme temperatures shift compound preferences.

        Args:
            track_wetness: 0.0 (dry) to 1.0 (fully wet).
            rain_intensity: Rain in mm/h.
            track_temp_c: Track temperature in Celsius.
            wind_speed_ms: Wind speed in m/s.

        Returns:
            Dict with ``adjusted_compounds``, ``recommended_stops``,
            ``pit_loss_modifier``, ``compound_shift``, and ``rationale``.
        """
        is_wet = track_wetness > 0.2
        is_heavy_rain = rain_intensity > 10.0
        is_cold = track_temp_c < 25.0
        is_hot = track_temp_c > 45.0
        is_windy = wind_speed_ms > 10.0

        adjustments: list[str] = []
        compound_shift: dict[str, str] = {}
        pit_loss_modifier = 1.0

        # Wet conditions → wet compounds, fewer stops, higher pit loss
        if is_wet:
            adjustments.append("湿地条件：推荐半雨胎或全雨胎，减少进站次数")
            compound_shift = {"soft": "intermediate", "medium": "intermediate",
                              "hard": "intermediate"}
            pit_loss_modifier = 1.3 if track_wetness > 0.5 else 1.15
            if is_heavy_rain:
                compound_shift = {"soft": "wet", "medium": "wet", "hard": "wet"}
                pit_loss_modifier = 1.5
                adjustments.append("大雨条件：全雨胎推荐，进站时间损失显著增加")

        # Temperature effects
        if is_hot:
            adjustments.append("高温条件：硬胎耐久性优势放大，软胎退化加速")
            if not is_wet:
                compound_shift["soft"] = "medium"
        elif is_cold:
            adjustments.append("低温条件：软胎升温更快，适合短 stints")
            if not is_wet:
                compound_shift["hard"] = "medium"

        # Wind effects
        if is_windy:
            adjustments.append("强风条件：下压力重要性增加，策略保守化")
            pit_loss_modifier = max(pit_loss_modifier, 1.05)

        # Determine recommended stops
        if is_heavy_rain:
            recommended_stops = 0  # Safety car likely, fewer stops
        elif is_wet and track_wetness > 0.4:
            recommended_stops = 1
        elif is_hot:
            recommended_stops = 2  # More stops to manage degradation
        else:
            recommended_stops = 1  # Default

        return {
            "is_wet": is_wet,
            "is_heavy_rain": is_heavy_rain,
            "is_cold": is_cold,
            "is_hot": is_hot,
            "is_windy": is_windy,
            "compound_shift": compound_shift,
            "recommended_stops": recommended_stops,
            "pit_loss_modifier": round(pit_loss_modifier, 2),
            "adjustments": adjustments,
            "rationale": "；".join(adjustments) if adjustments else "正常天气，标准策略适用",
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
