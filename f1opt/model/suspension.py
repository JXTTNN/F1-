"""F1 suspension + vehicle dynamics sub-models (Iter-21).

Complements :mod:`f1opt.model.physics` with three interpretable, first-principles
sub-models covering the suspension / vehicle-dynamics domain:

- :class:`SuspensionModel` — spring/damper/ARB behaviour, ride height, rake,
  corner weights, roll & pitch stiffness distribution, heave natural frequency,
  damping ratio and lateral load-transfer distribution.
- :class:`VehicleDynamicsModel` — whole-car transient behaviour: steady-state
  understeer gradient, yaw inertia, yaw response time, stability classification,
  trail-brake balance, aero sensitivity and a Chinese setup-balance diagnosis.
- :class:`SetupHarmonics` — internal-consistency checks (spring/ARB harmony,
  ride-height/rake window, camber alignment).

All public methods return floats / dicts / strs (never ``None``) and never raise
on edge-case inputs — out-of-range game-unit values are clamped. The suspension
setup dict uses :class:`~f1opt.data.setup_schema.CarSetup` field names
(``front_suspension`` / ``rear_suspension`` / ``front_arb`` / ``rear_arb`` /
``front_ride_height`` / ``rear_ride_height`` / ``front_camber`` / ``rear_camber``
/ ``front_toe`` / ``rear_toe``); the conceptual aliases ``front_spring`` /
``rear_spring`` / ``front_anti_roll_bar`` / ``rear_anti_roll_bar`` are also
accepted for convenience. Spring / ARB / ride-height game units are taken in
``[1, 11]`` and mapped linearly to physical SI ranges; camber and toe are
already supplied in degrees.
"""

from __future__ import annotations

import math

from f1opt.numeric import clamp as _clamp

# === Physical constants ====================================================
G_EARTH = 9.81
"""Gravitational acceleration (m/s^2)."""

# === Game-unit -> physical scaling (input game units in [1, 11]) ===========
SPRING_MIN_G = 1.0
SPRING_MAX_G = 11.0
SPRING_RATE_MIN_N_PER_MM = 80.0
"""Spring rate (N/mm) at game unit 1 (softest)."""
SPRING_RATE_MAX_N_PER_MM = 200.0
"""Spring rate (N/mm) at game unit 11 (stiffest)."""

ARB_MIN_G = 1.0
ARB_MAX_G = 11.0
ARB_STIFFNESS_MIN_N_M_PER_DEG = 10.0
"""Anti-roll bar stiffness (N·m/deg) at game unit 1."""
ARB_STIFFNESS_MAX_N_M_PER_DEG = 100.0
"""Anti-roll bar stiffness (N·m/deg) at game unit 11."""

RIDE_HEIGHT_MIN_G = 1.0
RIDE_HEIGHT_MAX_G = 11.0
RIDE_HEIGHT_MIN_MM = 5.0
"""Ride height (mm) at game unit 1 (lowest)."""
RIDE_HEIGHT_MAX_MM = 40.0
"""Ride height (mm) at game unit 11 (highest)."""

RAKE_WHEELBASE_M = 3.0
"""Wheelbase (m) assumed when converting ride-height difference to a rake angle."""

F1_CORNERING_STIFFNESS_N_PER_RAD = 2.0e5
"""Representative total F1 cornering stiffness (N/rad) for yaw response time."""

# Ground-effect model for aero_sensitivity (mirrors physics.AeroModel behaviour).
STALL_RIDE_HEIGHT_MM = 5.0
"""Ride height (mm) below which the floor stalls and downforce collapses."""
RH_REF_MM = 25.0
"""Reference ride height (mm) at which ground effect is unity."""
GE_GAIN = 1.5
"""Ground-effect gain: lower ride height -> proportionally more downforce."""
GE_MAX = 2.5
"""Cap on the ground-effect multiplier."""

# Setup dict key aliases: CarSetup field name first, conceptual alias second.
_SPRING_KEYS = {
    "front": ("front_suspension", "front_spring"),
    "rear": ("rear_suspension", "rear_spring"),
}
_ARB_KEYS = {
    "front": ("front_arb", "front_anti_roll_bar"),
    "rear": ("rear_arb", "rear_anti_roll_bar"),
}
_RH_KEYS = {
    "front": ("front_ride_height",),
    "rear": ("rear_ride_height",),
}
_CAMBER_KEYS = {"front": ("front_camber",), "rear": ("rear_camber",)}
_TOE_KEYS = {"front": ("front_toe",), "rear": ("rear_toe",)}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _lookup(setup: dict, keys: tuple[str, ...], default: float = 0.0) -> float:
    """Return the first present key in ``setup`` (as float), else ``default``."""
    for k in keys:
        if k in setup:
            return float(setup[k])
    return float(default)


def _linear_scale(
    value: float, in_min: float, in_max: float, out_min: float, out_max: float
) -> float:
    """Linearly map ``value`` from ``[in_min, in_max]`` to ``[out_min, out_max]``.

    ``value`` is clamped to the input range first.
    """
    v = _clamp(float(value), in_min, in_max)
    frac = (v - in_min) / (in_max - in_min) if in_max > in_min else 0.0
    return out_min + frac * (out_max - out_min)


def _ground_effect_factor(ride_height_mm: float) -> float:
    """Ground-effect multiplier on downforce for a ride height (mm).

    Above the 5 mm stall threshold, lower ride height yields more downforce;
    below 5 mm the floor stalls and downforce collapses linearly toward 0.
    """
    h = max(0.0, float(ride_height_mm))
    if h < STALL_RIDE_HEIGHT_MM:
        return h / STALL_RIDE_HEIGHT_MM
    ge = 1.0 + GE_GAIN * max(0.0, (RH_REF_MM - h) / RH_REF_MM)
    return min(ge, GE_MAX)


# === Iter-21a: SuspensionModel =============================================
class SuspensionModel:
    """Spring / damper / ARB behaviour, ride height, rake and load distribution."""

    def __init__(self, setup: dict) -> None:
        self._setup = dict(setup)
        # Raw game-unit values (springs/ARB/ride height in [1, 11]; camber/toe in deg).
        self.front_spring_g = _lookup(self._setup, _SPRING_KEYS["front"])
        self.rear_spring_g = _lookup(self._setup, _SPRING_KEYS["rear"])
        self.front_arb_g = _lookup(self._setup, _ARB_KEYS["front"])
        self.rear_arb_g = _lookup(self._setup, _ARB_KEYS["rear"])
        self.front_ride_height_g = _lookup(self._setup, _RH_KEYS["front"])
        self.rear_ride_height_g = _lookup(self._setup, _RH_KEYS["rear"])
        self.front_camber = _lookup(self._setup, _CAMBER_KEYS["front"])
        self.rear_camber = _lookup(self._setup, _CAMBER_KEYS["rear"])
        self.front_toe = _lookup(self._setup, _TOE_KEYS["front"])
        self.rear_toe = _lookup(self._setup, _TOE_KEYS["rear"])

    def spring_rate(self, kind: str) -> float:
        """Return spring rate (N/mm) for ``kind`` in {"front", "rear"}.

        Game units ``[1, 11]`` map linearly to ``[80, 200]`` N/mm.
        """
        g = self.front_spring_g if kind == "front" else self.rear_spring_g
        return _linear_scale(
            g, SPRING_MIN_G, SPRING_MAX_G,
            SPRING_RATE_MIN_N_PER_MM, SPRING_RATE_MAX_N_PER_MM,
        )

    def arb_stiffness(self, kind: str) -> float:
        """Return anti-roll bar stiffness (N·m/deg) for ``kind``.

        Game units ``[1, 11]`` map linearly to ``[10, 100]`` N·m/deg.
        """
        g = self.front_arb_g if kind == "front" else self.rear_arb_g
        return _linear_scale(
            g, ARB_MIN_G, ARB_MAX_G,
            ARB_STIFFNESS_MIN_N_M_PER_DEG, ARB_STIFFNESS_MAX_N_M_PER_DEG,
        )

    def ride_height(self, kind: str) -> float:
        """Return ride height (mm) for ``kind``.

        Game units ``[1, 11]`` map linearly to ``[5, 40]`` mm.
        """
        g = self.front_ride_height_g if kind == "front" else self.rear_ride_height_g
        return _linear_scale(
            g, RIDE_HEIGHT_MIN_G, RIDE_HEIGHT_MAX_G,
            RIDE_HEIGHT_MIN_MM, RIDE_HEIGHT_MAX_MM,
        )

    def rake_angle(self) -> float:
        """Return rake angle (degrees) = atan((rear_rh - front_rh) / wheelbase).

        Uses a 3 m wheelbase. Positive when the rear rides higher than the front.
        """
        diff_mm = self.ride_height("rear") - self.ride_height("front")
        return math.degrees(math.atan2(diff_mm / 1000.0, RAKE_WHEELBASE_M))

    def corner_weights(self, mass_kg: float, cg_longitudinal_pct: float = 0.52) -> dict:
        """Return static corner weights (Newtons) as ``{"fl","fr","rl","rr"}``.

        ``cg_longitudinal_pct`` is the front-axle mass fraction (front bias).
        Lateral distribution is assumed symmetric (50/50); the four weights sum
        to ``mass_kg * g``.
        """
        m = max(0.0, float(mass_kg))
        front_frac = _clamp(float(cg_longitudinal_pct), 0.0, 1.0)
        total_n = m * G_EARTH
        front_axle = total_n * front_frac
        rear_axle = total_n * (1.0 - front_frac)
        return {
            "fl": front_axle / 2.0,
            "fr": front_axle / 2.0,
            "rl": rear_axle / 2.0,
            "rr": rear_axle / 2.0,
        }

    def roll_stiffness_distribution(self) -> float:
        """Return front / total roll stiffness in ``[0, 1]``.

        Spring and ARB contributions are summed per axle (treated as relative
        stiffness contributions). ``> 0.5`` = stiffer front = more understeer.
        """
        front = self.spring_rate("front") + self.arb_stiffness("front")
        rear = self.spring_rate("rear") + self.arb_stiffness("rear")
        total = front + rear
        if total <= 0.0:
            return 0.5
        return _clamp(front / total, 0.0, 1.0)

    def pitch_stiffness_distribution(self) -> float:
        """Return front / total spring rate in ``[0, 1]`` (pitch is spring-dominated)."""
        front = self.spring_rate("front")
        rear = self.spring_rate("rear")
        total = front + rear
        if total <= 0.0:
            return 0.5
        return _clamp(front / total, 0.0, 1.0)

    def natural_frequency(self, kind: str, mass_per_corner_kg: float) -> float:
        """Return heave natural frequency (Hz) = sqrt(k/m) / (2π).

        ``k`` is the per-axle spring rate converted to N/m. F1 cars typically
        run 4-7 Hz.
        """
        k_n_per_m = self.spring_rate(kind) * 1000.0  # N/mm -> N/m
        m = max(1e-6, float(mass_per_corner_kg))
        return math.sqrt(k_n_per_m / m) / (2.0 * math.pi)

    def damping_ratio(self, crit_damping_pct: float = 0.7) -> float:
        """Return assumed damping ratio (setup exposes no damping explicitly).

        Modelled as ``crit_damping_pct`` +/- a small ARB influence: stiffer ARBs
        slightly raise the effective damping ratio.
        """
        base = _clamp(float(crit_damping_pct), 0.0, 2.0)
        avg_arb = 0.5 * (self.arb_stiffness("front") + self.arb_stiffness("rear"))
        # Normalise ARB to [-1, 1] around the midpoint of [10, 100].
        arb_dev = (avg_arb - 55.0) / 45.0
        return _clamp(base + 0.02 * arb_dev, 0.0, 2.0)

    def load_transfer_distribution(self, lat_g: float) -> float:
        """Return the fraction of lateral load transfer through the front axle.

        Elastic load transfer follows the roll-stiffness distribution; the
        geometric (CG-driven) component splits ~50/50 and grows in relative
        weight at high lateral acceleration, pulling the result toward 0.5.
        Returns a value in ``[0, 1]``.
        """
        elastic = self.roll_stiffness_distribution()
        geo_weight = _clamp(max(0.0, float(lat_g)) / 4.0, 0.0, 1.0) * 0.3
        ltd = (1.0 - geo_weight) * elastic + geo_weight * 0.5
        return _clamp(ltd, 0.0, 1.0)


# === Iter-21b: VehicleDynamicsModel ========================================
class VehicleDynamicsModel:
    """Whole-car transient behaviour built on top of :class:`SuspensionModel`."""

    def __init__(
        self,
        setup: dict,
        mass_kg: float = 798.0,
        track_width_m: float = 2.0,
        wheelbase_m: float = 3.5,
    ) -> None:
        self.susp = SuspensionModel(setup)
        self.mass_kg = max(0.0, float(mass_kg))
        self.track_width_m = max(1e-6, float(track_width_m))
        self.wheelbase_m = max(1e-6, float(wheelbase_m))

    def steady_state_balance(self, lat_g: float, speed_ms: float) -> float:
        """Return the steady-state understeer gradient (rad/g).

        Positive = understeer, negative = oversteer. Combines a mechanical term
        (roll-stiffness distribution), an aero term (rake-driven balance, scaled
        by speed) and a camber term.
        """
        # Mechanical: >0.5 front roll stiffness -> understeer.
        mech = (self.susp.roll_stiffness_distribution() - 0.5) * 2.0  # [-1, 1]
        # Aero: rake drives a forward aero bias; amplified with speed.
        rake = self.susp.rake_angle()  # degrees, positive = rear higher
        aero_raw = _clamp(rake / 1.5, -1.0, 1.0)
        speed_factor = _clamp(max(0.0, float(speed_ms)) / 80.0, 0.0, 1.0)
        aero = aero_raw * speed_factor
        # Camber: more negative front camber -> better front grip -> oversteer.
        front_neg = -self.susp.front_camber
        rear_neg = -self.susp.rear_camber
        camber = _clamp((rear_neg - front_neg) * 0.5, -1.0, 1.0)
        # Combine into an understeer gradient (rad/g).
        k = 0.020 * mech + 0.012 * aero + 0.006 * camber
        # Mild lateral-acceleration influence (tyre load sensitivity).
        k *= 1.0 + 0.05 * _clamp(max(0.0, float(lat_g)), 0.0, 3.0)
        return float(k)

    def yaw_inertia(self) -> float:
        """Return approximate yaw moment of inertia (kg·m²) = m·(wb² + tw²) / 12."""
        return self.mass_kg * (self.wheelbase_m ** 2 + self.track_width_m ** 2) / 12.0

    def response_time(self, lat_g_step: float) -> float:
        """Return yaw response time constant (s) ≈ sqrt(I_z / (C_total · wb)).

        Effective cornering stiffness reduces slightly with lateral step (tyre
        load sensitivity), lengthening the response. F1 cars: ~0.05-0.15 s.
        """
        i_z = self.yaw_inertia()
        c_eff = F1_CORNERING_STIFFNESS_N_PER_RAD * (
            1.0 - 0.10 * _clamp(max(0.0, float(lat_g_step)), 0.0, 3.0)
        )
        c_eff = max(1e-6, c_eff)
        tau = math.sqrt(i_z / (c_eff * self.wheelbase_m))
        return float(tau)

    def stability_factor(self) -> str:
        """Return ``"understeer"`` / ``"oversteer"`` / ``"neutral"`` at 1 G."""
        k = self.steady_state_balance(1.0, 50.0)
        if k > 0.002:
            return "understeer"
        if k < -0.002:
            return "oversteer"
        return "neutral"

    def trail_brake_balance(self, brake_bias: float, lat_g: float) -> dict:
        """Return trail-brake force distribution and rear-lock safety.

        ``brake_bias`` is the front brake fraction (0-1). The rear is more prone
        to locking under trail braking (lateral load transfer unloads the
        inside-rear); ``trail_brake_safe`` is True when the front bias is high
        enough to keep the rear from locking first.
        """
        bias = _clamp(float(brake_bias), 0.0, 1.0)
        front_share = bias
        rear_share = 1.0 - bias
        # Required front bias rises with lateral acceleration.
        threshold = 0.50 + 0.04 * _clamp(max(0.0, float(lat_g)), 0.0, 2.5)
        safe = bias >= threshold
        return {
            "front_share": front_share,
            "rear_share": rear_share,
            "trail_brake_safe": bool(safe),
        }

    def aero_sensitivity(self, speed_ms: float) -> dict:
        """Return total downforce (N) and the aero balance shift vs low speed.

        Downforce scales with v² and ground effect (ride height). At high speed
        the rake drives a forward aero-balance shift (positive = forward).
        """
        v = max(0.0, float(speed_ms))
        ge_f = _ground_effect_factor(self.susp.ride_height("front"))
        ge_r = _ground_effect_factor(self.susp.ride_height("rear"))
        c_l = 1.5 * 0.5 * (ge_f + ge_r)
        downforce = c_l * v * v
        rake = self.susp.rake_angle()
        speed_factor = _clamp(v / 80.0, 0.0, 1.0)
        balance_shift = _clamp(rake / 1.5, -1.0, 1.0) * 0.3 * speed_factor
        return {
            "downforce_total_n": float(downforce),
            "balance_shift": float(balance_shift),
        }

    def setup_balance_diagnosis(self) -> dict:
        """Return a Chinese setup-balance diagnosis.

        Keys: ``mechanical_balance``, ``aero_balance``, ``overall``,
        ``recommendation``.
        """
        dist = self.susp.roll_stiffness_distribution()
        rake = self.susp.rake_angle()
        stab = self.stability_factor()

        if dist > 0.55:
            mech = "偏向推头(前轴滚转刚度偏高)"
        elif dist < 0.45:
            mech = "偏向甩尾(后轴滚转刚度偏高)"
        else:
            mech = "机械平衡中性"

        if rake > 1.0:
            aero = "仰角过大(后扩散器失速风险)"
        elif rake < 0.1:
            aero = "仰角不足(偏推头)"
        else:
            aero = "空气动力学平衡良好"

        if stab == "understeer":
            overall = "入弯轻微推头"
            rec = "可适当软化前防倾杆或降低前翼"
        elif stab == "oversteer":
            overall = "入弯轻微甩尾"
            rec = "可适当加硬前防倾杆或升高前翼"
        else:
            overall = "平衡中性"
            rec = "当前调教平衡良好,无需调整"

        return {
            "mechanical_balance": mech,
            "aero_balance": aero,
            "overall": overall,
            "recommendation": rec,
        }


# === Iter-21c: SetupHarmonics ==============================================
class SetupHarmonics:
    """Internal-consistency checks for a suspension setup."""

    def __init__(self, setup: dict) -> None:
        self.susp = SuspensionModel(setup)

    def check_spring_arb_harmony(self) -> dict:
        """Flag stiff-spring/soft-ARB (poor roll control) and the reverse (harsh).

        Per axle, the spring and ARB are each normalised to ``[0, 1]``; a large
        mismatch either way raises a warning. Returns ``{"ok": bool, "warnings"}``.
        """
        warnings: list[str] = []
        for kind, label in (("front", "前"), ("rear", "后")):
            s = self.susp.spring_rate(kind)
            a = self.susp.arb_stiffness(kind)
            ns = _clamp(
                (s - SPRING_RATE_MIN_N_PER_MM)
                / (SPRING_RATE_MAX_N_PER_MM - SPRING_RATE_MIN_N_PER_MM),
                0.0,
                1.0,
            )
            na = _clamp(
                (a - ARB_STIFFNESS_MIN_N_M_PER_DEG)
                / (ARB_STIFFNESS_MAX_N_M_PER_DEG - ARB_STIFFNESS_MIN_N_M_PER_DEG),
                0.0,
                1.0,
            )
            diff = ns - na
            if diff > 0.35:
                warnings.append(f"{label}轴弹簧偏硬而防倾杆偏软,滚转控制不足")
            elif diff < -0.35:
                warnings.append(f"{label}轴防倾杆偏硬而弹簧偏软,路感过硬")
        return {"ok": len(warnings) == 0, "warnings": warnings}

    def check_ride_height_rake(self) -> dict:
        """Flag rake outside the 5-15 mm window (stall / understeer).

        Returns ``{"ok": bool, "rake_mm": float, "warnings": list[str]}``.
        """
        front_rh = self.susp.ride_height("front")
        rear_rh = self.susp.ride_height("rear")
        rake_mm = rear_rh - front_rh
        warnings: list[str] = []
        if rake_mm > 15.0:
            warnings.append(f"仰角过大({rake_mm:.1f}mm),后扩散器有失速风险")
        elif rake_mm < 5.0:
            warnings.append(f"仰角不足({rake_mm:.1f}mm),容易推头且下压力不足")
        return {"ok": len(warnings) == 0, "rake_mm": float(rake_mm), "warnings": warnings}

    def check_camber_alignment(self) -> dict:
        """Flag front camber not more negative than rear (and insufficient camber).

        Returns ``{"ok": bool, "warnings": list[str]}``.
        """
        warnings: list[str] = []
        fc = self.susp.front_camber
        rc = self.susp.rear_camber
        # Front should be more negative than rear (front_camber < rear_camber).
        if fc >= rc:
            warnings.append("前轮外倾角应比后轮更负(前轮外倾不足)")
        if fc > -2.0:
            warnings.append("前轮外倾角偏正,弯道抓地力不足")
        return {"ok": len(warnings) == 0, "warnings": warnings}

    def all_checks(self) -> dict:
        """Run all three checks and aggregate.

        Returns ``{"ok": bool, "checks": dict, "total_warnings": int}``.
        """
        checks = {
            "spring_arb_harmony": self.check_spring_arb_harmony(),
            "ride_height_rake": self.check_ride_height_rake(),
            "camber_alignment": self.check_camber_alignment(),
        }
        total = sum(len(c["warnings"]) for c in checks.values())
        ok = all(c["ok"] for c in checks.values())
        return {"ok": ok, "checks": checks, "total_warnings": total}


__all__ = [
    "SetupHarmonics",
    "SuspensionModel",
    "VehicleDynamicsModel",
]
