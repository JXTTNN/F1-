"""Model validation + verification suite for the F1 setup optimizer.

Three validator classes that sanity-check the DNN surrogate, the physics
sub-models and a :class:`~f1opt.data.setup_schema.CarSetup` before it is
applied to a track. Pure-python, deterministic, never raises on edge cases.

- :class:`SurrogateValidator` — checks the DNN surrogate's predictions are
  monotonic in setup fields, in a realistic lap-time range, sector-consistent,
  have physically valid response ranges and produce different lap times on
  different tracks.
- :class:`PhysicsValidator` — checks the first-principles physics sub-models
  (aero v^2 scaling, tire thermal grip curve, Magic Formula tire force,
  suspension corner weights / natural frequency, ERS energy balance).
- :class:`SetupSanityChecker` — checks a setup is sane before applying it to a
  track: range compliance, track appropriateness, internal consistency.
"""

from __future__ import annotations

import math
import random

from f1opt.data.setup_schema import (
    ALL_SETUP_FIELDS,
    DEFAULT_SETUP,
    SETUP_FIELDS,
    CarSetup,
)
from f1opt.data.tracks import TRACKS_BY_ID
from f1opt.model.physics import (
    ERS_LAYOUT_FACTOR,
    AeroModel,
    PowertrainModel,
    TireThermalModel,
)
from f1opt.model.surrogate import RESPONSE_NAMES, SurrogateModel, _get_default_model
from f1opt.model.suspension import SetupHarmonics, SuspensionModel
from f1opt.model.tire_dynamics import MagicFormulaTire

__all__ = [
    "PhysicsValidator",
    "SetupSanityChecker",
    "SurrogateValidator",
]

# CarSetup field order (matches ALL_SETUP_FIELDS() ordering).
_FIELD_ORDER = list(SETUP_FIELDS.keys())

# Suspension-model game-unit range is [1, 11] while CarSetup clicks are [1, 50];
# convert clicks -> suspension game units so SetupHarmonics sees meaningful
# values rather than clamped-to-max values.
_SUSP_CLICKS_MIN = 1.0
_SUSP_CLICKS_MAX = 50.0
_SUSP_GAME_MIN = 1.0
_SUSP_GAME_MAX = 11.0


def _random_setup(rng: random.Random) -> CarSetup:
    """Build a random valid CarSetup via the public ``from_vector`` API."""
    vec = [rng.random() for _ in range(len(SETUP_FIELDS))]
    return CarSetup.from_vector(vec)


def _sweep_setup(field: str, norm_value: float) -> CarSetup:
    """Return ``DEFAULT_SETUP`` with one field set to ``norm_value`` (in [0,1])."""
    idx = _FIELD_ORDER.index(field)
    vec = list(DEFAULT_SETUP.to_vector())
    vec[idx] = float(norm_value)
    return CarSetup.from_vector(vec)


def _clicks_to_suspension(v: float) -> float:
    """Map a CarSetup clicks value in [1, 50] to suspension game units [1, 11]."""
    frac = (float(v) - _SUSP_CLICKS_MIN) / (_SUSP_CLICKS_MAX - _SUSP_CLICKS_MIN)
    return _SUSP_GAME_MIN + frac * (_SUSP_GAME_MAX - _SUSP_GAME_MIN)


def _setup_to_suspension_dict(setup: CarSetup) -> dict:
    """Convert a CarSetup to the dict shape expected by :class:`SuspensionModel`."""
    return {
        "front_suspension": _clicks_to_suspension(setup.front_suspension),
        "rear_suspension": _clicks_to_suspension(setup.rear_suspension),
        "front_arb": _clicks_to_suspension(setup.front_arb),
        "rear_arb": _clicks_to_suspension(setup.rear_arb),
        "front_ride_height": _clicks_to_suspension(setup.front_ride_height),
        "rear_ride_height": _clicks_to_suspension(setup.rear_ride_height),
        "front_camber": float(setup.front_camber),
        "rear_camber": float(setup.rear_camber),
        "front_toe": float(setup.front_toe),
        "rear_toe": float(setup.rear_toe),
    }


# === SurrogateValidator =====================================================
class SurrogateValidator:
    """Validate the DNN surrogate's predictions for physical sanity.

    All methods return plain dicts and never raise; ``model=None`` uses the
    cached default surrogate model (loaded from disk if available, else
    untrained and returning the analytic prior).
    """

    LAP_TIME_MIN = 60.0
    LAP_TIME_MAX = 180.0

    RESPONSE_VALID_RANGES: dict[str, tuple[float, float]] = {
        "speed_avg": (10.0, 120.0),
        "speed_max": (20.0, 150.0),
        "slip_angle": (-15.0, 15.0),
        "tyre_load_spread": (0.0, 1.0),
        "rake": (-2.0, 5.0),
        "tyre_temp": (40.0, 150.0),
        "g_lat_max": (0.0, 6.0),
    }

    def __init__(self, model: SurrogateModel | None = None) -> None:
        self.model = model if model is not None else _get_default_model()

    def validate_monotonicity(self, field: str, track_id: str = "melbourne") -> dict:
        """Sweep ``field`` from min to max in 5 steps; check lap_time direction.

        Returns ``{field, track_id, monotonic, direction, samples}`` where
        ``direction`` is ``"increasing"`` / ``"decreasing"`` / ``"mixed"`` and
        ``samples`` is a list of ``(value, lap_time)`` tuples.
        """
        n_steps = 5
        samples: list[tuple[float, float]] = []
        for i in range(n_steps):
            norm = i / (n_steps - 1)
            setup = _sweep_setup(field, norm)
            lap = float(self.model.predict_lap_time(setup, track_id))
            value = float(getattr(setup, field))
            samples.append((value, lap))
        lap_times = [s[1] for s in samples]
        deltas = [lap_times[i + 1] - lap_times[i] for i in range(len(lap_times) - 1)]
        eps = 1e-9
        nonneg = all(d >= -eps for d in deltas)
        nonpos = all(d <= eps for d in deltas)
        if nonneg and nonpos:
            direction = "mixed"
            monotonic = True
        elif nonneg:
            direction = "increasing"
            monotonic = True
        elif nonpos:
            direction = "decreasing"
            monotonic = True
        else:
            direction = "mixed"
            monotonic = False
        return {
            "field": field,
            "track_id": track_id,
            "monotonic": monotonic,
            "direction": direction,
            "samples": samples,
        }

    def validate_range(
        self, track_id: str = "melbourne", n_samples: int = 20, seed: int = 0
    ) -> dict:
        """Check predicted lap_times are in a realistic F1 range [60, 180] s."""
        rng = random.Random(seed)
        laps = [
            float(self.model.predict_lap_time(_random_setup(rng), track_id))
            for _ in range(n_samples)
        ]
        out_of_range = sum(
            1 for lap in laps if lap < self.LAP_TIME_MIN or lap > self.LAP_TIME_MAX
        )
        return {
            "track_id": track_id,
            "n_samples": n_samples,
            "min_lap": float(min(laps)) if laps else 0.0,
            "max_lap": float(max(laps)) if laps else 0.0,
            "avg_lap": float(sum(laps) / len(laps)) if laps else 0.0,
            "all_in_range": out_of_range == 0,
            "out_of_range_count": out_of_range,
        }

    def validate_sector_consistency(
        self, track_id: str = "melbourne", n_samples: int = 10, seed: int = 0
    ) -> dict:
        """Check sector_times sum to lap_time within 0.01 s."""
        rng = random.Random(seed)
        errors: list[float] = []
        for _ in range(n_samples):
            out = self.model.predict(_random_setup(rng), track_id)
            errors.append(abs(float(out["lap_time"]) - float(sum(out["sectors"]))))
        max_err = max(errors) if errors else 0.0
        return {
            "track_id": track_id,
            "n_samples": n_samples,
            "consistent": max_err < 0.01,
            "max_error_s": float(max_err),
        }

    def validate_response_ranges(
        self, track_id: str = "melbourne", n_samples: int = 5, seed: int = 0
    ) -> dict:
        """Check response predictions are in physically valid ranges."""
        rng = random.Random(seed)
        observed: dict[str, list[float]] = {name: [] for name in RESPONSE_NAMES}
        for _ in range(n_samples):
            out = self.model.predict(_random_setup(rng), track_id)
            for name, value in out["responses"].items():
                if name in observed:
                    observed[name].append(float(value))
        responses: dict[str, dict] = {}
        for name in RESPONSE_NAMES:
            vals = observed[name]
            lo, hi = self.RESPONSE_VALID_RANGES.get(name, (-math.inf, math.inf))
            v_min = min(vals) if vals else 0.0
            v_max = max(vals) if vals else 0.0
            responses[name] = {
                "min": float(v_min),
                "max": float(v_max),
                "valid_range": [float(lo), float(hi)],
                "in_range": bool(lo <= v_min and v_max <= hi),
            }
        return {"responses": responses}

    def cross_track_consistency(self, setup: CarSetup | None = None) -> dict:
        """Check the same setup produces different lap times on different tracks.

        Monaco and Monza should produce different lap times (different layouts).
        Returns ``{consistent, monaco_vs_monza_delta, samples}``.
        """
        setup = setup if setup is not None else DEFAULT_SETUP
        track_ids = ["monaco", "monza", "melbourne", "spa", "silverstone", "jeddah"]
        samples: list[dict] = []
        for tid in track_ids:
            if tid not in TRACKS_BY_ID:
                continue
            lap = float(self.model.predict_lap_time(setup, tid))
            samples.append({"track_id": tid, "lap_time": lap})
        monaco_lap = next(
            (s["lap_time"] for s in samples if s["track_id"] == "monaco"), 0.0
        )
        monza_lap = next(
            (s["lap_time"] for s in samples if s["track_id"] == "monza"), 0.0
        )
        delta = monaco_lap - monza_lap
        lap_times = [s["lap_time"] for s in samples]
        consistent = len(set(lap_times)) > 1
        return {
            "consistent": bool(consistent),
            "monaco_vs_monza_delta": float(delta),
            "samples": samples,
        }

    def full_report(self) -> dict:
        """Run all surrogate validations; return ``{passed, checks, summary}``."""
        checks = {
            "monotonicity_front_wing": self.validate_monotonicity("front_wing"),
            "monotonicity_fuel_load": self.validate_monotonicity("fuel_load"),
            "range_melbourne": self.validate_range("melbourne"),
            "sector_consistency_melbourne": self.validate_sector_consistency("melbourne"),
            "response_ranges": self.validate_response_ranges(),
            "cross_track": self.cross_track_consistency(),
        }
        range_ok = checks["range_melbourne"]["all_in_range"]
        sector_ok = checks["sector_consistency_melbourne"]["consistent"]
        cross_ok = checks["cross_track"]["consistent"]
        response_ok = all(
            r["in_range"] for r in checks["response_ranges"]["responses"].values()
        )
        passed = bool(range_ok and sector_ok and cross_ok and response_ok)
        status = "全部通过" if passed else "存在异常"
        summary = (
            f"代理模型校验: {status}; "
            f"圈速范围={'通过' if range_ok else '异常'}, "
            f"分段一致性={'通过' if sector_ok else '异常'}, "
            f"跨赛道={'通过' if cross_ok else '异常'}, "
            f"响应范围={'通过' if response_ok else '异常'}"
        )
        return {"passed": passed, "checks": checks, "summary": summary}


# === PhysicsValidator =======================================================
class PhysicsValidator:
    """Validate that the physics sub-models are physically sane."""

    # ERS energy-balance assumptions (kJ): per-lap recovery ceiling + store.
    _ERS_MAX_RECOVER_KJ = 350.0
    _ERS_INITIAL_STORE_KJ = 100.0

    def __init__(self) -> None:
        self.aero = AeroModel()
        self.tire_thermal = TireThermalModel()
        self.powertrain = PowertrainModel()

    def validate_aero(self) -> dict:
        """Check downforce and drag scale with v^2 (doubling speed -> 4x force)."""
        fw, rw, rh_f, rh_r, rh_avg = 25.0, 27.0, 10.0, 12.0, 11.0
        v_lo, v_hi = 25.0, 50.0
        df_lo = self.aero.downforce(fw, rw, rh_f, rh_r, v_lo)
        df_hi = self.aero.downforce(fw, rw, rh_f, rh_r, v_hi)
        dr_lo = self.aero.drag(fw, rw, rh_avg, v_lo)
        dr_hi = self.aero.drag(fw, rw, rh_avg, v_hi)
        df_ratio = df_hi / df_lo if df_lo > 0 else 0.0
        dr_ratio = dr_hi / dr_lo if dr_lo > 0 else 0.0
        samples = [
            {"speed_ms": v_lo, "downforce_n": float(df_lo), "drag_n": float(dr_lo)},
            {"speed_ms": v_hi, "downforce_n": float(df_hi), "drag_n": float(dr_hi)},
        ]
        return {
            "downforce_quadratic": bool(abs(df_ratio - 4.0) < 0.05),
            "drag_quadratic": bool(abs(dr_ratio - 4.0) < 0.05),
            "samples": samples,
        }

    def validate_tire_thermal(self) -> dict:
        """Check grip peaks at 90 °C and falls off at temperature extremes."""
        temps = [0.0, 30.0, 50.0, 70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 200.0]
        grips = [self.tire_thermal.grip_factor(t) for t in temps]
        peak_idx = grips.index(max(grips))
        peak_at_90 = temps[peak_idx] == 90.0
        grip_90 = self.tire_thermal.grip_factor(90.0)
        falloff = (
            self.tire_thermal.grip_factor(0.0) < grip_90
            and self.tire_thermal.grip_factor(200.0) < grip_90
        )
        return {
            "peak_at_90": bool(peak_at_90),
            "falloff_at_extremes": bool(falloff),
        }

    def validate_tire_dynamics(self) -> dict:
        """Check Magic Formula force peaks at optimal slip and is 0 at zero slip."""
        tire = MagicFormulaTire(load_n=4000.0, compound="soft")
        f_zero = float(tire.pure_longitudinal(0.0))
        opt = tire.optimal_slip_ratio()
        f_opt = float(tire.pure_longitudinal(opt))
        f_high = float(tire.pure_longitudinal(0.5))
        peak_exists = f_opt > f_zero and f_opt >= f_high
        return {
            "peak_exists": bool(peak_exists),
            "zero_at_zero_slip": bool(abs(f_zero) < 1e-6),
        }

    def validate_suspension(self) -> dict:
        """Check corner_weights sum to mass*g and natural frequency is in range."""
        setup_dict = {
            "front_suspension": 6,
            "rear_suspension": 4,
            "front_arb": 7,
            "rear_arb": 5,
            "front_ride_height": 4,
            "rear_ride_height": 8,
            "front_camber": -3.5,
            "rear_camber": -2.0,
            "front_toe": 0.05,
            "rear_toe": 0.20,
        }
        susp = SuspensionModel(setup_dict)
        mass_kg = 798.0
        cw = susp.corner_weights(mass_kg)
        total = sum(cw.values())
        expected = mass_kg * 9.81
        sum_correct = abs(total - expected) < 1e-6
        mass_per_corner = mass_kg / 4.0
        freq_front = susp.natural_frequency("front", mass_per_corner)
        freq_rear = susp.natural_frequency("rear", mass_per_corner)
        freq_in_range = all(2.0 <= f <= 10.0 for f in (freq_front, freq_rear))
        return {
            "corner_weights_sum_correct": bool(sum_correct),
            "natural_freq_in_range": bool(freq_in_range),
            "natural_freq_front_hz": float(freq_front),
            "natural_freq_rear_hz": float(freq_rear),
        }

    def validate_energy_consistency(self) -> dict:
        """Check ERS deploy per lap does not exceed recover + initial store."""
        layouts = list(ERS_LAYOUT_FACTOR.keys())
        max_deploy = 0.0
        for mode in range(4):
            for layout in layouts:
                d = float(self.powertrain.ers_deploy_per_lap(mode, layout))
                if d > max_deploy:
                    max_deploy = d
        balanced = max_deploy <= self._ERS_MAX_RECOVER_KJ + self._ERS_INITIAL_STORE_KJ
        return {
            "balanced": bool(balanced),
            "max_deploy_kj": float(max_deploy),
            "max_recover_kj": float(self._ERS_MAX_RECOVER_KJ),
            "initial_store_kj": float(self._ERS_INITIAL_STORE_KJ),
        }

    def full_report(self) -> dict:
        """Run all physics validations; return ``{passed, checks, summary}``."""
        checks = {
            "aero": self.validate_aero(),
            "tire_thermal": self.validate_tire_thermal(),
            "tire_dynamics": self.validate_tire_dynamics(),
            "suspension": self.validate_suspension(),
            "energy": self.validate_energy_consistency(),
        }
        aero_ok = checks["aero"]["downforce_quadratic"] and checks["aero"]["drag_quadratic"]
        thermal_ok = (
            checks["tire_thermal"]["peak_at_90"]
            and checks["tire_thermal"]["falloff_at_extremes"]
        )
        tire_ok = (
            checks["tire_dynamics"]["peak_exists"]
            and checks["tire_dynamics"]["zero_at_zero_slip"]
        )
        susp_ok = (
            checks["suspension"]["corner_weights_sum_correct"]
            and checks["suspension"]["natural_freq_in_range"]
        )
        energy_ok = checks["energy"]["balanced"]
        passed = bool(aero_ok and thermal_ok and tire_ok and susp_ok and energy_ok)
        status = "全部通过" if passed else "存在异常"
        summary = (
            f"物理模型校验: {status}; "
            f"气动={'通过' if aero_ok else '异常'}, "
            f"胎温={'通过' if thermal_ok else '异常'}, "
            f"轮胎动力学={'通过' if tire_ok else '异常'}, "
            f"悬挂={'通过' if susp_ok else '异常'}, "
            f"能量平衡={'通过' if energy_ok else '异常'}"
        )
        return {"passed": passed, "checks": checks, "summary": summary}


# === SetupSanityChecker =====================================================
class SetupSanityChecker:
    """Sanity-check a setup before applying it to a track.

    Range compliance flags fields at their extreme (min/max) values; track
    appropriateness flags setup-track mismatches (e.g. low wing at Monaco);
    internal consistency delegates to :class:`SetupHarmonics`.
    """

    HIGH_DOWNFORCE_TRACK_TYPES = {"street", "high_downforce"}
    LOW_DOWNFORCE_TRACK_TYPES = {"high_speed_low_downforce"}

    def __init__(self, setup: CarSetup, track_id: str) -> None:
        self.setup = setup
        self.track_id = track_id
        self.track = TRACKS_BY_ID.get(track_id)

    def check_range_compliance(self) -> list[str]:
        """Return warnings for fields at extreme (0th/100th percentile) values."""
        warnings: list[str] = []
        for spec in ALL_SETUP_FIELDS():
            value = float(getattr(self.setup, spec.name))
            if value <= spec.min:
                warnings.append(
                    f"{spec.name} 处于最小值 {spec.min:g} (单位: {spec.unit})"
                )
            elif value >= spec.max:
                warnings.append(
                    f"{spec.name} 处于最大值 {spec.max:g} (单位: {spec.unit})"
                )
        return warnings

    def check_track_appropriateness(self) -> list[str]:
        """Return warnings for setup-track mismatches."""
        warnings: list[str] = []
        if self.track is None:
            return warnings
        tt = self.track.track_type
        fw = float(self.setup.front_wing)
        rw = float(self.setup.rear_wing)
        avg_wing = 0.5 * (fw + rw)
        if tt in self.HIGH_DOWNFORCE_TRACK_TYPES and avg_wing < 15.0:
            warnings.append(
                f"赛道 {self.track_id} 为高下压力赛道, 平均翼面 {avg_wing:.1f} 偏低"
            )
        if tt in self.LOW_DOWNFORCE_TRACK_TYPES and avg_wing > 35.0:
            warnings.append(
                f"赛道 {self.track_id} 为低下压力赛道, 平均翼面 {avg_wing:.1f} 偏高"
            )
        fuel = float(self.setup.fuel_load)
        if fuel < 10.0:
            warnings.append(f"燃油装载 {fuel:.1f}kg 偏低, 可能在比赛中耗尽")
        front_rh = float(self.setup.front_ride_height)
        if tt in self.HIGH_DOWNFORCE_TRACK_TYPES and front_rh > 40.0:
            warnings.append(
                f"高下压力赛道前离地间隙 {front_rh:.0f} 偏高, 下压力不足"
            )
        bp = float(self.setup.brake_pressure)
        if bp < 90.0:
            warnings.append(f"制动压力 {bp:.0f} 偏低, 制动力不足")
        return warnings

    def check_internal_consistency(self) -> list[str]:
        """Use :class:`SetupHarmonics` to check spring/ARB/ride-height harmony."""
        harmonics = SetupHarmonics(_setup_to_suspension_dict(self.setup))
        result = harmonics.all_checks()
        warnings: list[str] = []
        for check_name, check_result in result["checks"].items():
            for w in check_result.get("warnings", []):
                warnings.append(f"[{check_name}] {w}")
        return warnings

    def overall_warnings(self) -> list[str]:
        """All warnings combined (range + track + internal)."""
        return [
            *self.check_range_compliance(),
            *self.check_track_appropriateness(),
            *self.check_internal_consistency(),
        ]

    def is_sane(self) -> bool:
        """True if no critical (track-appropriateness or internal) warnings."""
        critical = self.check_track_appropriateness() + self.check_internal_consistency()
        return len(critical) == 0

    def recommendation(self) -> str:
        """Return a Chinese summary of the setup's sanity."""
        warnings = self.overall_warnings()
        if not warnings:
            return "调教状态良好, 无需调整"
        lines = [f"共发现 {len(warnings)} 项警告:"]
        for w in warnings:
            lines.append(f"  - {w}")
        if self.is_sane():
            lines.append("建议: 警告多为边界值, 可酌情调整以增加调教裕度")
        else:
            lines.append("建议: 存在严重不匹配, 请优先处理赛道适配与内部一致性问题")
        return "\n".join(lines)
