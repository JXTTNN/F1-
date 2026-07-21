"""Pacejka Magic Formula tire force model (Iter-30/31).

Implements the industry-standard "Magic Formula" tire force curves used in
vehicle dynamics simulation. Pure-python (no numpy required) so it can be
imported in lightweight contexts (feedback engine, what-if analyzer).

The Magic Formula has the form::

    F = D * sin(C * atan(B * x - E * (B * x - atan(B * x))))

where:
    B = stiffness factor (initial slope)
    C = shape factor (controls width of peak)
    D = peak factor (= mu * Fz, the load on the tire)
    E = curvature factor (controls post-peak falloff)

Sub-modules:
    - :class:`MagicFormulaTire` — single-tire longitudinal/lateral/combined force
    - :class:`TireSet` — four-tire set with balance diagnostics
    - :data:`COMPOUND_PARAMS` — per-compound (mu, stiffness, thermal window)

References (textbook formulas, no papers):
    Pacejka H.B. "Tire and Vehicle Dynamics" (3rd ed., SAE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from f1opt.numeric import clamp as _clamp

__all__ = [
    "COMPOUND_PARAMS",
    "MagicFormulaTire",
    "TireSet",
    "CompoundParams",
]


# --------------------------------------------------------------------------- #
# Per-compound parameters (F1 2026 representative values)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompoundParams:
    """Per-compound tire force parameters."""

    name: str
    mu_peak: float           # peak friction coefficient at reference conditions
    b_long: float            # longitudinal stiffness factor B
    b_lat: float             # lateral stiffness factor B (per degree)
    peak_temp_c: float       # optimum operating temperature
    thermal_window_c: float  # half-width of high-grip window


COMPOUND_PARAMS: dict[str, CompoundParams] = {
    "soft": CompoundParams("soft", 1.90, 28.0, 0.30, 90.0, 20.0),
    "medium": CompoundParams("medium", 1.70, 27.0, 0.28, 90.0, 22.0),
    "hard": CompoundParams("hard", 1.50, 26.0, 0.27, 95.0, 25.0),
    "intermediate": CompoundParams("intermediate", 1.30, 22.0, 0.24, 80.0, 30.0),
    "wet": CompoundParams("wet", 0.90, 20.0, 0.20, 70.0, 35.0),
}

_DEFAULT_COMPOUND = "soft"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _magic_formula(x: float, b: float, c: float, d: float, e: float) -> float:
    """Core Pacejka: F = D*sin(C*atan(B*x - E*(B*x - atan(B*x)))).

    Returns the force magnitude (D * sin(...)). ``d`` already encodes the load.
    """
    bx = b * x
    inner = bx - e * (bx - math.atan(bx))
    return d * math.sin(c * math.atan(inner))


# --------------------------------------------------------------------------- #
# Magic Formula single tire
# --------------------------------------------------------------------------- #
class MagicFormulaTire:
    """Single-tire Pacejka Magic Formula model.

    State (slip_ratio, slip_angle, load, camber, temp, compound) drives all
    force calculations. All inputs are clamped to safe ranges; no method
    raises on extreme values.
    """

    # Shape factors (well-known Magic Formula constants for racing tires).
    C_LONG = 1.65    # longitudinal shape factor
    E_LONG = 0.97    # longitudinal curvature factor
    C_LAT = 1.30     # lateral shape factor
    E_LAT = -0.04    # lateral curvature factor

    # Aligning moment (Mz) shape factors.
    B_ALIGN = 2.0
    C_ALIGN = 1.2
    E_ALIGN = -0.10

    # Load-sensitivity slope: mu decreases ~0.5% per kN above 4 kN reference.
    _LOAD_REF_N = 4000.0
    _LOAD_SENS_PER_KN = -0.005

    # Camber: lateral grip peaks near -3.5 deg (mild negative camber).
    _CAMBER_PEAK_DEG = -3.5
    _CAMBER_WINDOW_DEG = 6.0  # ±6 deg around peak before grip drops significantly

    def __init__(
        self,
        slip_ratio: float = 0.0,
        slip_angle: float = 0.0,
        load_n: float = 4000.0,
        camber_deg: float = 0.0,
        temp_c: float = 90.0,
        compound: str = _DEFAULT_COMPOUND,
    ) -> None:
        self.slip_ratio = _clamp(slip_ratio, -1.0, 1.0)
        self.slip_angle = _clamp(slip_angle, -30.0, 30.0)
        self.load_n = max(0.0, load_n)
        self.camber_deg = _clamp(camber_deg, -15.0, 15.0)
        self.temp_c = _clamp(temp_c, 0.0, 200.0)
        self.compound = compound if compound in COMPOUND_PARAMS else _DEFAULT_COMPOUND

    # ----- compound access ------------------------------------------------ #
    @property
    def _params(self) -> CompoundParams:
        return COMPOUND_PARAMS[self.compound]

    # ----- grip modifiers ------------------------------------------------- #
    def grip_vs_temp(self, temp_c: float) -> float:
        """Grip multiplier vs temperature.

        Peaks at compound peak_temp, quadratic falloff to 0.7 at
        peak±thermal_window, to 0 at peak±2*window.
        """
        t = _clamp(temp_c, 0.0, 200.0)
        p = self._params
        delta = abs(t - p.peak_temp_c)
        if delta <= p.thermal_window_c:
            # 1.0 at peak → 0.7 at window edge: linear in delta.
            return 1.0 - 0.3 * (delta / p.thermal_window_c)
        # Beyond window: continue down to 0 at 2*window.
        extra = delta - p.thermal_window_c
        return max(0.0, 0.7 * (1.0 - extra / p.thermal_window_c))

    def grip_vs_load(self, load_n: float) -> float:
        """Load-sensitivity multiplier (more load → slightly less mu)."""
        load = max(0.0, load_n)
        delta_kn = (load - self._LOAD_REF_N) / 1000.0
        mult = 1.0 + self._LOAD_SENS_PER_KN * delta_kn
        return _clamp(mult, 0.85, 1.05)

    def grip_vs_camber(self, camber_deg: float) -> float:
        """Lateral-grip camber multiplier. Peak near -3.5 deg."""
        c = _clamp(camber_deg, -15.0, 15.0)
        delta = abs(c - self._CAMBER_PEAK_DEG)
        # Gaussian-ish: 1.0 at peak, drops as exp(-(delta/window)^2).
        return max(0.6, math.exp(-(delta / self._CAMBER_WINDOW_DEG) ** 2))

    # ----- effective mu --------------------------------------------------- #
    def _effective_mu(self, load_n: float | None = None, temp_c: float | None = None,
                       camber_deg: float | None = None) -> float:
        """Combine compound peak mu with all grip modifiers."""
        load = self.load_n if load_n is None else load_n
        temp = self.temp_c if temp_c is None else temp_c
        cam = self.camber_deg if camber_deg is None else camber_deg
        return (
            self._params.mu_peak
            * self.grip_vs_temp(temp)
            * self.grip_vs_load(load)
            # Note: camber mainly affects lateral grip; include mild effect on mu.
            * (0.9 + 0.1 * self.grip_vs_camber(cam))
        )

    # ----- pure-slip forces ---------------------------------------------- #
    def pure_longitudinal(self, slip_ratio: float | None = None) -> float:
        """Longitudinal force Fx in Newtons (Pacejka MF)."""
        sr = self.slip_ratio if slip_ratio is None else _clamp(slip_ratio, -1.0, 1.0)
        mu = self._effective_mu()
        d = mu * self.load_n  # peak force
        if d <= 0.0:
            return 0.0
        return _magic_formula(sr, self._params.b_long, self.C_LONG, d, self.E_LONG)

    def pure_lateral(self, slip_angle_deg: float | None = None) -> float:
        """Lateral force Fy in Newtons (Pacejka MF). slip_angle in degrees."""
        alpha = self.slip_angle if slip_angle_deg is None else _clamp(slip_angle_deg, -30.0, 30.0)
        mu = self._effective_mu()
        # Lateral grip gets full camber benefit (vs the 10% in _effective_mu).
        cam_grip = self.grip_vs_camber(self.camber_deg)
        mu_lat = mu / (0.9 + 0.1 * cam_grip) * cam_grip
        d = mu_lat * self.load_n
        if d <= 0.0:
            return 0.0
        return _magic_formula(alpha, self._params.b_lat, self.C_LAT, d, self.E_LAT)

    # ----- combined slip -------------------------------------------------- #
    def combined_force(self, slip_ratio: float | None = None,
                       slip_angle_deg: float | None = None) -> tuple[float, float]:
        """Combined-slip (Fx, Fy) via friction-ellipse reduction.

        Each pure force is attenuated by the other slip component so the
        resultant vector stays inside the friction circle.
        """
        sr = self.slip_ratio if slip_ratio is None else _clamp(slip_ratio, -1.0, 1.0)
        alpha = self.slip_angle if slip_angle_deg is None else _clamp(slip_angle_deg, -30.0, 30.0)

        fx0 = self.pure_longitudinal(sr)
        fy0 = self.pure_lateral(alpha)

        # Friction ellipse: when both slips present, each is reduced so that
        # (Fx/Fx_max)^2 + (Fy/Fy_max)^2 <= 1.
        # Use a simple reduction: tan(alpha) provides the lateral coupling.
        alpha_rad = math.radians(alpha)
        tan_a = math.tan(alpha_rad) if abs(alpha) < 89 else 1e6
        # Coupling weight: how much lateral slip reduces longitudinal force.
        long_factor = 1.0 / math.sqrt(1.0 + (0.15 * tan_a) ** 2) if abs(tan_a) < 1e6 else 0.0
        # Symmetric: longitudinal slip reduces lateral force.
        slip_factor = 1.0 / math.sqrt(1.0 + (5.0 * sr) ** 2)

        fx = fx0 * long_factor
        fy = fy0 * slip_factor
        return (fx, fy)

    # ----- aligning moment ------------------------------------------------ #
    def self_aligning_torque(self, slip_angle_deg: float | None = None) -> float:
        """Aligning moment Mz in N·m (Pacejka MF for Mz). 0 at slip=0."""
        alpha = self.slip_angle if slip_angle_deg is None else _clamp(slip_angle_deg, -30.0, 30.0)
        mu = self._effective_mu()
        # Peak Mz ~ 0.05 * Fz * pneumatic trail (~0.05m).
        d = 0.05 * mu * self.load_n
        if d <= 0.0:
            return 0.0
        return _magic_formula(alpha, self.B_ALIGN, self.C_ALIGN, d, self.E_ALIGN)

    # ----- optima --------------------------------------------------------- #
    def optimal_slip_ratio(self) -> float:
        """Slip ratio at peak longitudinal force (typically 0.08-0.12 for F1).

        Found by golden-section search on pure_longitudinal over [0, 0.5].
        """
        lo, hi = 0.0, 0.5
        # Save current slip_ratio so we don't mutate state.
        for _ in range(40):
            m1 = lo + 0.382 * (hi - lo)
            m2 = lo + 0.618 * (hi - lo)
            f1 = self.pure_longitudinal(m1)
            f2 = self.pure_longitudinal(m2)
            if f1 < f2:
                lo = m1
            else:
                hi = m2
        return 0.5 * (lo + hi)

    def optimal_slip_angle(self) -> float:
        """Slip angle (deg) at peak lateral force (typically 5-8 deg)."""
        lo, hi = 0.0, 20.0
        for _ in range(40):
            m1 = lo + 0.382 * (hi - lo)
            m2 = lo + 0.618 * (hi - lo)
            f1 = self.pure_lateral(m1)
            f2 = self.pure_lateral(m2)
            if f1 < f2:
                lo = m1
            else:
                hi = m2
        return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Four-tire set
# --------------------------------------------------------------------------- #
class TireSet:
    """Four-tire set with balance diagnostics.

    Front tires typically run more negative camber; rear tires carry more load
    (rearward weight bias + aero). The default layout reflects F1 practice.
    """

    def __init__(self, compound: str = _DEFAULT_COMPOUND, track_temp_c: float = 30.0) -> None:
        self.compound = compound if compound in COMPOUND_PARAMS else _DEFAULT_COMPOUND
        self.track_temp_c = _clamp(track_temp_c, 0.0, 80.0)
        # Four tires: FL, FR, RL, RR.
        self.tires: list[MagicFormulaTire] = [
            MagicFormulaTire(compound=self.compound, temp_c=90.0) for _ in range(4)
        ]
        # Default mild F1-ish setup: front more camber, rear more load.
        self.tires[0].camber_deg = -3.5  # FL
        self.tires[1].camber_deg = -3.5  # FR
        self.tires[2].camber_deg = -2.0  # RL
        self.tires[3].camber_deg = -2.0  # RR
        self.tires[0].load_n = 3500.0
        self.tires[1].load_n = 3500.0
        self.tires[2].load_n = 4500.0
        self.tires[3].load_n = 4500.0

    def update(
        self,
        slip_ratios: list[float],
        slip_angles: list[float],
        loads_n: list[float],
        cambers_deg: list[float],
        temps_c: list[float],
    ) -> None:
        """Update all four tires' state. Lists must be length 4."""
        if not (len(slip_ratios) == len(slip_angles) == len(loads_n)
                == len(cambers_deg) == len(temps_c) == 4):
            raise ValueError("All input lists must have length 4")
        for i, t in enumerate(self.tires):
            t.slip_ratio = _clamp(slip_ratios[i], -1.0, 1.0)
            t.slip_angle = _clamp(slip_angles[i], -30.0, 30.0)
            t.load_n = max(0.0, loads_n[i])
            t.camber_deg = _clamp(cambers_deg[i], -15.0, 15.0)
            t.temp_c = _clamp(temps_c[i], 0.0, 200.0)

    def total_longitudinal(self) -> float:
        """Sum of Fx across all 4 tires (N)."""
        return sum(t.pure_longitudinal() for t in self.tires)

    def total_lateral(self) -> float:
        """Sum of Fy across all 4 tires (N)."""
        return sum(t.pure_lateral() for t in self.tires)

    def lateral_balance(self) -> float:
        """(front_lateral - rear_lateral) / total_lateral in [-1, 1].

        Negative → front grips less relative to rear → understeer tendency
        (front axle saturates first).
        Positive → rear grips less relative to front → oversteer tendency.
        """
        front = self.tires[0].pure_lateral() + self.tires[1].pure_lateral()
        rear = self.tires[2].pure_lateral() + self.tires[3].pure_lateral()
        total = front + rear
        if total < 1e-6:
            return 0.0
        return _clamp((front - rear) / total, -1.0, 1.0)

    def traction_capacity(self) -> dict[str, float]:
        """Maximum force capacity at current state."""
        return {
            "fx_max": sum(t.pure_longitudinal(t.optimal_slip_ratio()) for t in self.tires),
            "fy_max": sum(t.pure_lateral(t.optimal_slip_angle()) for t in self.tires),
            "combined_max": sum(
                max(abs(t.combined_force(t.optimal_slip_ratio(), t.optimal_slip_angle())[0]),
                    abs(t.combined_force(t.optimal_slip_ratio(), t.optimal_slip_angle())[1]))
                for t in self.tires
            ),
        }

    def wear_rate(self) -> float:
        """Aggregate wear indicator from slip magnitudes + temps."""
        total = 0.0
        for t in self.tires:
            # Slip contributes quadratically; thermal activation above 100°C.
            slip_term = (t.slip_ratio ** 2) * 100.0 + (t.slip_angle ** 2) * 0.5
            temp_factor = 1.0 + max(0.0, (t.temp_c - 100.0) / 30.0)
            total += slip_term * temp_factor
        return total

    def overheating_risk(self) -> float:
        """Overheating risk in [0,1] based on max tire temp vs 110°C threshold."""
        max_temp = max(t.temp_c for t in self.tires)
        if max_temp <= 100.0:
            return 0.0
        if max_temp >= 130.0:
            return 1.0
        return (max_temp - 100.0) / 30.0

    def grip_balance_diagnosis(self) -> str:
        """Chinese balance diagnosis string."""
        bal = self.lateral_balance()
        if bal < -0.15:
            return "前轴抓地力不足(推头倾向)"
        if bal > 0.15:
            return "后轴抓地力不足(甩尾倾向)"
        return "平衡良好"

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for logging/inspection."""
        return {
            "compound": self.compound,
            "track_temp_c": self.track_temp_c,
            "tires": [
                {
                    "slip_ratio": t.slip_ratio,
                    "slip_angle": t.slip_angle,
                    "load_n": t.load_n,
                    "camber_deg": t.camber_deg,
                    "temp_c": t.temp_c,
                }
                for t in self.tires
            ],
            "total_fx": self.total_longitudinal(),
            "total_fy": self.total_lateral(),
            "lateral_balance": self.lateral_balance(),
            "overheating_risk": self.overheating_risk(),
        }
