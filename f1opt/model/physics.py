"""Physics-based sub-models for F1 2026 setup optimization (Iter-15 .. Iter-20).

This module complements the data-driven :mod:`f1opt.model.surrogate` DNN with
*interpretable, first-principles* sub-models covering the six physical domains
that dominate F1 setup sensitivity. Each sub-model is pure numpy/python (no
torch), deterministic, and clamps out-of-range inputs so it never raises on
edge cases (negative speeds, oversized loads, etc.).

The six sub-areas (one per iteration):

- **Iter-15** :class:`TireThermalModel` — surface/core tire thermodynamics and
  grip-vs-temperature curve. F1 tires operate in a narrow ~80-100 °C window;
  outside it grip falls off rapidly (graining/blistering).
- **Iter-16** :class:`TireDegradationModel` — per-lap wear increment from slip,
  load and thermal activation, plus wear→laptime penalty with blowout.
- **Iter-17** :class:`AeroModel` — quadratic-in-speed downforce/drag with wing
  angle and ground-effect ride-height terms (stall below 5 mm).
- **Iter-18** :class:`PowertrainModel` — 2026 ERS deployment, energy→laptime
  benefit, fuel-mass laptime penalty and manual boost modes.
- **Iter-19** :class:`BrakeThermalModel` — per-axle brake temperatures, bias
  balance classification and lockup risk.
- **Iter-20** :class:`MassModel` — lateral weight transfer, total mass and
  per-lap fuel burn.

All public methods return floats or dicts of floats (``brake_bias_balance``
returns a ``str`` per its spec) and never ``None``. Tunable parameters are
exposed as module-level constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from f1opt.numeric import clamp as _clamp

# === Tire thermodynamics (Iter-15) =========================================
# F1 tires produce peak grip in a narrow window centred on ~90 °C. Below ~70 °C
# the compound is too cold (graining); above ~110 °C it overheats (blistering).
PEAK_GRIP_TEMP_C = 90.0
"""Temperature (°C) of maximum grip — the apex of the grip curve."""
GRIP_WINDOW_LOW_C = 70.0
"""Lower edge of the usable grip window (grip ≈ 0.7 here)."""
GRIP_WINDOW_HIGH_C = 110.0
"""Upper edge of the usable grip window (grip ≈ 0.7 here)."""
GRIP_AT_WINDOW_EDGE = 0.7
"""Grip factor at the usable-window edges (70 °C and 110 °C)."""
GRIP_OVERSHOOT_SPAN_C = 40.0
"""Temperature span (°C) above 110 °C over which grip falls from 0.7 to 0."""
PEAK_OPERATING_WINDOW_LOW_C = 80.0
"""Lower bound of the near-peak operating window (grip ≳ 0.9)."""
PEAK_OPERATING_WINDOW_HIGH_C = 100.0
"""Upper bound of the near-peak operating window (grip ≳ 0.9)."""

TIRE_THERMAL_MASS_J_PER_C = 9000.0
"""Effective thermal capacity of one tire+carcass (J/°C)."""
K_SURFACE_AIR = 0.025
"""Surface-to-air cooling rate (1/s)."""
K_SURFACE_CORE = 0.020
"""Surface-to-core conduction rate (1/s)."""
K_CORE_AIR = 0.006
"""Core-to-air (slow) cooling rate (1/s)."""
DEFAULT_TIRE_TEMP_C = 90.0
"""Default per-tire temperature for a fresh :class:`TireThermalState`."""
_TIRE_IDS: tuple[str, ...] = ("FL", "FR", "RL", "RR")

# === Tire degradation (Iter-16) ============================================
# Wear grows cubically with slip angle, with an exponential term dominating at
# high slip (the "exponential falloff"). Load is linear. Above 100 °C track
# temperature a thermal-activation multiplier accelerates wear (Arrhenius-style).
C_SLIP = 0.06
"""Cubic coefficient on slip angle (deg^-3)."""
C_SLIP_EXP = 0.05
"""Amplitude of the exponential high-slip wear term."""
SLIP_EXP_SCALE = 6.0
"""Slip-angle scale (deg) of the exponential wear term."""
C_LOAD = 0.40
"""Linear coefficient on vertical load (g^-1)."""
THERMAL_ACTIVATION_C = 100.0
"""Track-temperature threshold (°C) above which wear thermally activates."""
THERMAL_TAU = 30.0
"""Thermal-activation time constant (°C)."""
C_WEAR_FEEDBACK = 0.3
"""Fraction by which per-lap wear grows as the tire approaches 100 % wear."""
BLOWOUT_WEAR_PCT = 90.0
"""Wear percentage (0-100) at which the tire is considered to blow out."""
BLOWOUT_PENALTY_S = 6.0
"""Laptime penalty (s) representing imminent blowout / failure."""
WEAR_PENALTY_AT_50 = 0.8
"""Anchor laptime penalty (s) at 50 % wear."""

# === Aerodynamics (Iter-17) ================================================
# F1 aero forces scale with v^2. Downforce coefficient rises with wing angle
# and with ground effect (lower ride height), but the floor stalls below ~5 mm
# and downforce collapses — the classic porpoising/ground-effect cliff.
STALL_RIDE_HEIGHT_MM = 5.0
"""Ride height (mm) below which the floor stalls and downforce collapses."""
STALL_GE_MAX = 1.0
"""Ground-effect factor reached at the stall boundary (just before collapse)."""
RH_REF_MM = 25.0
"""Reference ride height (mm) at which ground effect is taken as unity."""
GE_GAIN = 1.5
"""Ground-effect gain: lower ride height → proportionally more downforce."""
GE_MAX = 2.5
"""Cap on the ground-effect multiplier (prevents unphysical blow-up)."""
CL_BASE = 0.5
"""Baseline lift coefficient (no wings, no ground effect)."""
CL_FRONT = 0.6
"""Front-wing lift-coefficient contribution at max wing (50 clicks)."""
CL_REAR = 0.8
"""Rear-wing lift-coefficient contribution at max wing (50 clicks)."""
CD_BASE = 0.4
"""Baseline drag coefficient."""
CD_FRONT = 0.3
"""Front-wing drag contribution at max wing."""
CD_REAR = 0.4
"""Rear-wing drag contribution at max wing."""
CD_GE = 0.2
"""Drag contribution from underbody ground effect (lower = more drag)."""

# === Powertrain / ERS (Iter-18) ============================================
# 2026 regs: ERS deploys a per-lap energy budget; manual boost gives a fixed
# laptime benefit; every kg of fuel costs ~0.035 s of laptime (F1 rule of thumb).
ERS_BASE_KJ: dict[int, float] = {0: 0.0, 1: 120.0, 2: 200.0, 3: 350.0}
"""Per-lap ERS deployment energy (kJ) by mode: none/medium/hotlap/deployment."""
ERS_LAYOUT_FACTOR: dict[str, float] = {
    "high_speed_low_downforce": 1.2,
    "street": 0.9,
    "high_downforce": 0.85,
    "medium": 1.0,
    "mixed": 1.05,
}
"""Track-layout multiplier on ERS deployment (long straights favour deployment)."""
ERS_KJ_TO_S_COEFF = 0.015
"""Converts deployed energy (kJ) to laptime benefit (s·km/kJ)."""
FUEL_TANK_MAX_KG = 110.0
"""2026 fuel tank capacity (kg) — fuel load is clamped to this."""
FUEL_MIN_KG = 5.0
"""Minimum meaningful fuel load (kg); fuel below this incurs no penalty."""
FUEL_PENALTY_PER_KG = 0.035
"""Laptime cost per kg of fuel above the minimum (s/kg)."""
BOOST_BENEFIT_S: dict[int, float] = {0: 0.0, 1: 0.05, 2: 0.12, 3: 0.20}
"""2026 manual boost laptime benefit (s) by mode 0/1/2/3."""

# === Brake thermal (Iter-19) ===============================================
# F1 carbon brakes run 400-800 °C. Brake bias shifts heat front/rear; too much
# front bias biases the car toward turn-in oversteer, too little toward
# understeer (per this module's convention).
BRAKE_HEAT_COEFF = 5000.0
"""Brake heat-input coefficient (°C at full pressure, full braking fraction)."""
BRAKE_TEMP_REF_C = 100.0
"""Reference brake temperature (°C) below which lockup thermal factor is 0."""
BRAKE_TEMP_SPAN = 700.0
"""Brake-temperature span (°C) over which the thermal lockup factor ramps 0→1."""
BRAKE_SLIP_REF = 10.0
"""Slip value at which the slip lockup factor saturates to 1."""

# === Mass & weight transfer (Iter-20) ======================================
# Lateral load transfer: ΔW = m · g · h_cg / tw · a_lat. 2026 minimum car mass
# is 798 kg; fuel burn is ~1.6 L/lap scaled by track length.
G_EARTH = 9.81
"""Gravitational acceleration (m/s^2)."""
CAR_MIN_MASS_KG = 798.0
"""2026 minimum car mass (kg), fuel excluded."""
FUEL_BURN_L_PER_LAP = 1.6
"""Baseline fuel burn (L/lap) at the reference lap length."""
REF_LAP_LENGTH_M = 5000.0
"""Reference lap length (m) for the fuel-burn scaling."""
FUEL_DENSITY_KG_PER_L = 0.75
"""F1 fuel density (kg/L)."""
FUEL_BURN_FUEL_COEFF = 0.05
"""Extra burn fraction at full tank vs. empty (heavier car burns more)."""


# === Iter-15: Tire thermodynamics ==========================================
@dataclass
class TireThermalState:
    """Evolving per-tire thermal state (surface + core temperatures, °C).

    A two-node thermal model is used per tire: the surface receives slip-work
    heat input and cools to ambient air, while the core is coupled to the
    surface by conduction and cools more slowly to ambient. The state is
    mutated in place by :meth:`TireThermalModel.temperature` so it can be
    integrated lap-over-lap by a simulation loop.
    """

    surface_temps: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(_TIRE_IDS, DEFAULT_TIRE_TEMP_C)
    )
    core_temps: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(_TIRE_IDS, DEFAULT_TIRE_TEMP_C)
    )


class TireThermalModel:
    """Tire thermodynamics: slip-work heating, ambient cooling and grip curve.

    The temperature evolution follows ``dT/dt = heat_in - k*(T - ambient)``
    applied to a two-node (surface/core) model per tire. Grip is a smooth
    unimodal function of temperature peaking at 90 °C.
    """

    def temperature(
        self,
        state: TireThermalState,
        ambient_track_temp_c: float,
        slip_work: float,
        duration_s: float,
    ) -> dict[str, dict[str, float]]:
        """Integrate per-tire surface/core temperatures over ``duration_s``.

        Parameters
        ----------
        state:
            Evolving :class:`TireThermalState` (mutated in place).
        ambient_track_temp_c:
            Ambient / track temperature (°C), clamped to [-50, 150].
        slip_work:
            Total mechanical work dissipated at the contact patch over the
            duration, per tire (Joules). Clamped to ≥ 0. Applied uniformly to
            all four tires.
        duration_s:
            Integration duration (s). Clamped to ≥ 0; ≤ 0 returns current state.

        Returns
        -------
        dict
            ``{"FL": {"surface": ..., "core": ...}, "FR": {...}, ...}``.
        """
        amb = _clamp(ambient_track_temp_c, -50.0, 150.0)
        work = max(0.0, float(slip_work))
        if duration_s <= 0.0:
            return {
                tid: {
                    "surface": float(state.surface_temps[tid]),
                    "core": float(state.core_temps[tid]),
                }
                for tid in _TIRE_IDS
            }
        # Slip-work power (W) → surface heating rate (°C/s).
        heat_rate = (work / duration_s) / TIRE_THERMAL_MASS_J_PER_C
        # Sub-step for numerical stability of the explicit Euler integration.
        n_steps = max(1, min(int(math.ceil(duration_s / 0.5)), 1000))
        dt = duration_s / n_steps
        for tid in _TIRE_IDS:
            ts = float(state.surface_temps[tid])
            tc = float(state.core_temps[tid])
            for _ in range(n_steps):
                dts = heat_rate - K_SURFACE_AIR * (ts - amb) - K_SURFACE_CORE * (ts - tc)
                dtc = K_SURFACE_CORE * (ts - tc) - K_CORE_AIR * (tc - amb)
                ts += dts * dt
                tc += dtc * dt
            state.surface_temps[tid] = ts
            state.core_temps[tid] = tc
        return {
            tid: {
                "surface": float(state.surface_temps[tid]),
                "core": float(state.core_temps[tid]),
            }
            for tid in _TIRE_IDS
        }

    def grip_factor(self, temp_c: float) -> float:
        """Normalized grip in ``[0, 1]`` for a tire surface temperature.

        Returns 1.0 at 90 °C (peak), 0.7 at the 70 °C / 110 °C window edges,
        and falls off quadratically outside the window (toward 0). The near-peak
        operating window is 80-100 °C (grip ≳ 0.9).
        """
        t = float(temp_c)
        if t <= 0.0:
            return 0.0
        if t < GRIP_WINDOW_LOW_C:
            # Quadratic falloff: 0.7 at 70 °C → 0 at 0 °C.
            return GRIP_AT_WINDOW_EDGE * (t / GRIP_WINDOW_LOW_C) ** 2
        if t <= GRIP_WINDOW_HIGH_C:
            # Peak window: 1.0 at 90 °C, 0.7 at 70/110 °C (quadratic).
            x = (t - PEAK_GRIP_TEMP_C) / (PEAK_GRIP_TEMP_C - GRIP_WINDOW_LOW_C)
            return 1.0 - (1.0 - GRIP_AT_WINDOW_EDGE) * x * x
        # Above 110 °C: quadratic falloff from 0.7 toward 0 over a 40 °C span.
        over = (t - GRIP_WINDOW_HIGH_C) / GRIP_OVERSHOOT_SPAN_C
        if over >= 1.0:
            return 0.0
        return GRIP_AT_WINDOW_EDGE * (1.0 - over) ** 2


# === Iter-16: Tire degradation =============================================
@dataclass
class TireWearState:
    """Evolving tire-wear state (cumulative wear percentage, 0-100).

    Mutated in place by :meth:`TireDegradationModel.wear_lap`; worn tires
    degrade slightly faster (positive feedback), modelling thermal/structural
    degradation acceleration.
    """

    wear_pct: float = 0.0


class TireDegradationModel:
    """Per-lap tire wear from slip, load and thermal activation."""

    def wear_lap(
        self,
        state: TireWearState,
        slip_angle_deg: float,
        tyre_load_g: float,
        track_temp_c: float,
    ) -> float:
        """Return the per-lap wear increment (percentage points, 0-100).

        The slip contribution is cubic with an exponential high-slip term
        (rapid "falloff" growth at large slip angles). Load is linear. Above
        100 °C track temperature a thermal-activation multiplier accelerates
        wear. The increment is added to ``state.wear_pct`` (clamped to 100).

        Parameters
        ----------
        state:
            Evolving :class:`TireWearState` (mutated in place).
        slip_angle_deg:
            Tire slip angle (deg), clamped to ≥ 0.
        tyre_load_g:
            Vertical load (g), clamped to ≥ 0.
        track_temp_c:
            Track temperature (°C); thermal activation triggers above 100 °C.
        """
        slip = max(0.0, float(slip_angle_deg))
        load = max(0.0, float(tyre_load_g))
        tt = float(track_temp_c)
        slip_term = C_SLIP * slip ** 3 + C_SLIP_EXP * (math.exp(slip / SLIP_EXP_SCALE) - 1.0)
        load_term = C_LOAD * load
        if tt > THERMAL_ACTIVATION_C:
            thermal = math.exp((tt - THERMAL_ACTIVATION_C) / THERMAL_TAU)
        else:
            thermal = 1.0
        wear_boost = 1.0 + C_WEAR_FEEDBACK * (_clamp(state.wear_pct, 0.0, 100.0) / 100.0)
        inc = max(0.0, (slip_term + load_term) * thermal * wear_boost)
        state.wear_pct = min(100.0, state.wear_pct + inc)
        return inc

    def wear_to_laptime_penalty(self, wear_pct: float) -> float:
        """Map cumulative wear (0-100 %) to a laptime penalty (seconds).

        Returns 0 s at 0 % wear, ~0.8 s at 50 % wear (quadratic), and the
        blowout penalty at/above the 90 % blowout threshold.
        """
        w = _clamp(float(wear_pct), 0.0, 100.0)
        if w >= BLOWOUT_WEAR_PCT:
            return BLOWOUT_PENALTY_S
        return WEAR_PENALTY_AT_50 * (w / 50.0) ** 2


# === Iter-17: Aerodynamics =================================================
def _ground_effect_factor(ride_height_mm: float) -> float:
    """Ground-effect multiplier on downforce for a ride height (mm).

    Above the 5 mm stall threshold, lower ride height yields more downforce
    (ground effect). Below 5 mm the floor stalls and downforce collapses
    linearly toward 0 as the ride height approaches 0.
    """
    h = max(0.0, float(ride_height_mm))
    if h < STALL_RIDE_HEIGHT_MM:
        # Stalled: downforce ramps down from the stall value toward 0.
        return STALL_GE_MAX * (h / STALL_RIDE_HEIGHT_MM)
    # Attached flow: lower ride height → more downforce.
    ge = 1.0 + GE_GAIN * max(0.0, (RH_REF_MM - h) / RH_REF_MM)
    return min(ge, GE_MAX)


class AeroModel:
    """Quadratic-in-speed downforce/drag with wing-angle and ground-effect terms.

    Forces follow ``F = C * v^2``. The lift coefficient ``C_L`` is a linear
    function of front/rear wing angle (clicks, 0-50) multiplied by a per-axle
    ground-effect factor; the drag coefficient ``C_D`` is a linear function of
    wing angle plus a mild ground-effect drag term. Ride heights are taken in
    **millimetres**.
    """

    def downforce(
        self,
        front_wing: float,
        rear_wing: float,
        ride_height_f: float,
        ride_height_r: float,
        speed_ms: float,
    ) -> float:
        """Return total aerodynamic downforce (Newtons).

        ``downforce = C_L(front_wing, rear_wing, ride heights) * v^2``.
        Negative speeds are clamped to 0 (no reverse aero). Wing clicks are
        clamped to [0, 50]; ride heights are in mm.
        """
        v = max(0.0, float(speed_ms))
        fw = _clamp(float(front_wing), 0.0, 50.0) / 50.0
        rw = _clamp(float(rear_wing), 0.0, 50.0) / 50.0
        ge_f = _ground_effect_factor(ride_height_f)
        ge_r = _ground_effect_factor(ride_height_r)
        c_l = CL_BASE + CL_FRONT * fw * ge_f + CL_REAR * rw * ge_r
        return c_l * v * v

    def drag(
        self,
        front_wing: float,
        rear_wing: float,
        ride_height_avg: float,
        speed_ms: float,
    ) -> float:
        """Return aerodynamic drag (Newtons).

        ``drag = C_D(front_wing, rear_wing, ride_height_avg) * v^2``. Lower
        average ride height adds a small ground-effect drag term. Negative
        speeds are clamped to 0; ride height is in mm.
        """
        v = max(0.0, float(speed_ms))
        fw = _clamp(float(front_wing), 0.0, 50.0) / 50.0
        rw = _clamp(float(rear_wing), 0.0, 50.0) / 50.0
        ge_drag = CD_GE * max(0.0, (RH_REF_MM - float(ride_height_avg)) / RH_REF_MM)
        c_d = CD_BASE + CD_FRONT * fw + CD_REAR * rw + ge_drag
        return c_d * v * v

    def downforce_balance(self, front: float, rear: float) -> float:
        """Return aero balance in ``[-1, 1]`` from front/rear downforce (N).

        -1 = understeer (rear-dominated), 0 = neutral, +1 = oversteer
        (front-dominated). Returns 0 when both inputs are non-positive.
        """
        total = float(front) + float(rear)
        if total <= 0.0:
            return 0.0
        return _clamp((float(front) - float(rear)) / total, -1.0, 1.0)


# === Iter-18: Powertrain / ERS =============================================
def _layout_factor(track_layout: object) -> float:
    """Resolve the ERS track-layout multiplier from a string or Track-like."""
    if track_layout is None:
        return 1.0
    if hasattr(track_layout, "track_type"):
        track_layout = track_layout.track_type
    return ERS_LAYOUT_FACTOR.get(str(track_layout), 1.0)


class PowertrainModel:
    """2026 ERS deployment, energy→laptime benefit, fuel penalty and boost."""

    def ers_deploy_per_lap(
        self,
        ers_deploy_mode: int,
        track_layout: object,
    ) -> float:
        """Return per-lap ERS deployment energy (kJ).

        Modes 0/1/2/3 = none/medium/hotlap/deployment, scaled by a track-layout
        factor (long straights favour deployment). The mode is clamped to [0, 3].
        """
        mode = int(_clamp(float(ers_deploy_mode), 0.0, 3.0))
        return ERS_BASE_KJ[mode] * _layout_factor(track_layout)

    def laptime_benefit_kj_to_s(self, kj: float, track_length_m: float) -> float:
        """Convert deployed ERS energy (kJ) to a laptime benefit (seconds).

        Benefit grows with energy and shrinks with track length (the boost is
        spread over more distance on longer tracks). Track length is clamped to
        ≥ 500 m to avoid division blow-up.
        """
        k = max(0.0, float(kj))
        length_km = max(0.5, float(track_length_m) / 1000.0)
        return ERS_KJ_TO_S_COEFF * k / length_km

    def fuel_effect_laptime(self, fuel_load_kg: float) -> float:
        """Return the laptime penalty (s) from carrying fuel mass.

        ``0.035 s`` per kg of fuel above the minimum, clamped to the 110 kg
        tank capacity. Fuel below the minimum incurs no penalty.
        """
        fuel = _clamp(float(fuel_load_kg), 0.0, FUEL_TANK_MAX_KG)
        excess = max(0.0, fuel - FUEL_MIN_KG)
        return FUEL_PENALTY_PER_KG * excess

    def boost_mode_laptime(self, mode: int) -> float:
        """Return the 2026 manual boost laptime benefit (seconds).

        Modes 0/1/2/3 → 0 / 0.05 / 0.12 / 0.20 s. The mode is clamped to [0, 3].
        """
        m = int(_clamp(float(mode), 0.0, 3.0))
        return BOOST_BENEFIT_S[m]


# === Iter-19: Brake thermal ================================================
class BrakeThermalModel:
    """Per-axle brake temperatures, bias balance and lockup risk."""

    def brake_temp(
        self,
        brake_pressure: float,
        brake_bias: float,
        ambient_c: float,
        lap_frac_braking: float,
    ) -> dict[str, float]:
        """Return per-axle brake temperatures (°C) as ``{"front": ..., "rear": ...}``.

        Heat input scales with ``brake_pressure^2`` and the lap fraction spent
        braking, then splits between axles by ``brake_bias`` (front fraction).
        Carbon brakes typically reach 400-800 °C. ``brake_pressure`` is a
        percentage (0-100), ``brake_bias`` a front fraction (0-1), and
        ``lap_frac_braking`` a fraction (0-1).
        """
        bp = _clamp(float(brake_pressure), 0.0, 100.0) / 100.0
        bias = _clamp(float(brake_bias), 0.0, 1.0)
        frac = _clamp(float(lap_frac_braking), 0.0, 1.0)
        amb = float(ambient_c)
        heat = BRAKE_HEAT_COEFF * (bp ** 2) * frac
        return {
            "front": amb + heat * bias,
            "rear": amb + heat * (1.0 - bias),
        }

    def brake_bias_balance(self, bias: float) -> str:
        """Classify brake balance from the front bias fraction.

        Returns ``"understeer"`` for bias < 0.50, ``"neutral"`` for 0.50-0.56,
        and ``"oversteer"`` for bias > 0.56 (this module's convention).
        """
        b = float(bias)
        if b < 0.50:
            return "understeer"
        if b > 0.56:
            return "oversteer"
        return "neutral"

    def lockup_risk(self, temp_c: float, bias: float, slip: float) -> float:
        """Return brake lockup risk in ``[0, 1]``.

        Risk rises with brake temperature (overheated, grabby discs), with
        wheel slip, and with front bias. Each factor is normalized to [0, 1]
        and combined; the result is clamped to [0, 1].
        """
        temp_factor = _clamp(
            (float(temp_c) - BRAKE_TEMP_REF_C) / BRAKE_TEMP_SPAN, 0.0, 1.0
        )
        bias_factor = _clamp(float(bias), 0.0, 1.0)
        slip_factor = _clamp(float(slip) / BRAKE_SLIP_REF, 0.0, 1.0)
        risk = 0.35 * temp_factor + 0.50 * slip_factor + 0.15 * bias_factor
        return _clamp(risk, 0.0, 1.0)


# === Iter-20: Mass & weight transfer =======================================
class MassModel:
    """Lateral weight transfer, total car mass and per-lap fuel burn."""

    def weight_transfer(
        self,
        mass_kg: float,
        cg_height_m: float,
        track_width_m: float,
        lat_g: float,
    ) -> float:
        """Return lateral load transfer (Newtons).

        Standard formula ``ΔW = m · g · h_cg / tw · a_lat``. The track width is
        guarded against zero; mass and CG height are clamped to ≥ 0. The result
        is linear in lateral acceleration ``lat_g``.
        """
        m = max(0.0, float(mass_kg))
        cg = max(0.0, float(cg_height_m))
        tw = float(track_width_m) if track_width_m > 0.0 else 1e-6
        return m * G_EARTH * cg / tw * float(lat_g)

    def total_mass(self, fuel_load_kg: float) -> float:
        """Return total car mass (kg): 798 kg minimum car mass + fuel.

        Fuel is clamped to [0, 110] kg (tank capacity).
        """
        fuel = _clamp(float(fuel_load_kg), 0.0, FUEL_TANK_MAX_KG)
        return CAR_MIN_MASS_KG + fuel

    def fuel_burn_per_lap(self, fuel_load_kg: float, track_length_m: float) -> float:
        """Return per-lap fuel burn (kg).

        Baseline ~1.6 L/lap at a 5 km reference lap, scaled linearly by track
        length and slightly increased by current fuel load (heavier car burns
        more). Converted to kg at 0.75 kg/L. Track length is clamped to ≥ 1 m.
        """
        fuel = _clamp(float(fuel_load_kg), 0.0, FUEL_TANK_MAX_KG)
        length = max(1.0, float(track_length_m))
        mass_factor = 1.0 + FUEL_BURN_FUEL_COEFF * (fuel / FUEL_TANK_MAX_KG)
        burn_liters = FUEL_BURN_L_PER_LAP * (length / REF_LAP_LENGTH_M) * mass_factor
        return burn_liters * FUEL_DENSITY_KG_PER_L


__all__ = [
    "AeroModel",
    "BrakeThermalModel",
    "MassModel",
    "PowertrainModel",
    "TireDegradationModel",
    "TireThermalModel",
    "TireThermalState",
    "TireWearState",
]
