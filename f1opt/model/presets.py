"""Track- and condition-specific setup presets + feedback-driven auto-tuner.

Two higher-level helpers built on top of the existing setup schema, track
database, suspension harmony checks and surrogate optimizer:

- :class:`SetupPresets` — curated baseline :class:`~f1opt.data.setup_schema.CarSetup`
  per ``track_type`` (Monza/Monaco-style / street / mixed / medium), plus
  condition-specific (wet/hot/cold), compound-specific, aggressive /
  conservative and driver-archetype variants. :meth:`SetupPresets.list_presets`
  exposes a UI-friendly summary.
- :class:`SetupAutoTuner` — turns a list of driver feedback dimensions
  (``balance`` / ``tyres`` / ``braking`` / ``confidence`` / ``lap_time_potential``)
  into a tuned :class:`~f1opt.data.setup_schema.CarSetup`, with a structured
  change log (:meth:`SetupAutoTuner.tune_diff`), an internal-consistency pass
  (:meth:`SetupAutoTuner.apply_constraints` via
  :class:`~f1opt.model.suspension.SetupHarmonics`) and a
  :meth:`SetupAutoTuner.confidence_score`.

The module is pure-python and deterministic. The optional
``lap_time_potential="gap"`` path lazily imports
:func:`~f1opt.model.optimizer.search_setup` so merely importing this module
does not force the surrogate / scipy stack to load for the common heuristic
paths.
"""

from __future__ import annotations

import math
from typing import Any

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.data.track_evolution import WeatherCondition
from f1opt.data.tracks import get_track
from f1opt.model.suspension import SetupHarmonics
from f1opt.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["SetupPresets", "SetupAutoTuner"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _snap(name: str, value: float) -> int | float:
    """Clamp ``value`` to the legal range / step of setup field ``name``.

    Mirrors :func:`f1opt.data.setup_schema._snap_to_step` but returns a native
    ``int`` for integer fields so the result round-trips through
    :class:`CarSetup` without type-coercion surprises.
    """
    spec = SETUP_FIELDS[name]
    lo = float(spec.min)
    hi = float(spec.max)
    step = float(spec.step)
    v = float(value)
    if v < lo:
        v = lo
    elif v > hi:
        v = hi
    steps = round((v - lo) / step)
    v = lo + steps * step
    if spec.kind == "int":
        return int(round(v))
    decimals = max(0, -int(math.floor(math.log10(step))))
    return round(v, decimals)


def _apply_delta(
    state: dict[str, Any],
    field: str,
    delta: float,
    reason: str,
    changes: list[dict],
) -> None:
    """Apply ``delta`` to ``state[field]`` (clamped/snapped) and record a change."""
    before = state[field]
    after = _snap(field, float(before) + float(delta))
    if after != before:
        state[field] = after
        changes.append(
            {"field": field, "before": before, "after": after, "reason": reason}
        )


def _clone_state(setup: CarSetup) -> dict[str, Any]:
    """Return a mutable, properly-typed copy of ``setup`` (int fields stay int)."""
    return setup.model_dump()


# --------------------------------------------------------------------------- #
# Preset registry (module-level)
# --------------------------------------------------------------------------- #
# Each entry encodes a curated baseline setup for a track_type. ``compound`` is
# a recommendation hint reflected in the chosen pressures / camber (the F1 25
# CarSetup schema exposes no explicit compound field), used by list_presets and
# as documentation; for_track returns a plain CarSetup.
_TRACK_PRESETS: dict[str, dict[str, Any]] = {
    "high_speed_low_downforce": {
        "name": "Low Downforce (Monza-style)",
        "compound": "hard",
        "description": (
            "Minimal wing for top speed, stiff springs for braking stability "
            "and a hard compound tendency (higher pressures, less camber)."
        ),
        "setup": {
            "front_wing": 8,
            "rear_wing": 6,
            "on_throttle_diff": 75,
            "off_throttle_diff": 60,
            "front_camber": -3.20,
            "rear_camber": -1.80,
            "front_toe": 0.05,
            "rear_toe": 0.15,
            "front_suspension": 35,
            "rear_suspension": 30,
            "front_arb": 25,
            "rear_arb": 15,
            "front_ride_height": 15,
            "rear_ride_height": 30,
            "brake_pressure": 100,
            "front_brake_bias": 55,
            "front_tyre_pressure": 25.5,
            "rear_tyre_pressure": 21.5,
            "fuel_load": 30.0,
        },
    },
    "high_downforce": {
        "name": "High Downforce (Monaco-style)",
        "compound": "soft",
        "description": (
            "Maximum wing for cornering grip, soft springs to soak up bumps "
            "and kerbs, soft compound tendency (lower pressures)."
        ),
        "setup": {
            "front_wing": 45,
            "rear_wing": 48,
            "on_throttle_diff": 85,
            "off_throttle_diff": 50,
            "front_camber": -3.50,
            "rear_camber": -2.00,
            "front_toe": 0.05,
            "rear_toe": 0.20,
            "front_suspension": 8,
            "rear_suspension": 6,
            "front_arb": 8,
            "rear_arb": 12,
            "front_ride_height": 20,
            "rear_ride_height": 40,
            "brake_pressure": 100,
            "front_brake_bias": 55,
            "front_tyre_pressure": 23.0,
            "rear_tyre_pressure": 20.0,
            "fuel_load": 30.0,
        },
    },
    "street": {
        "name": "Street (bumpy, medium-high wing)",
        "compound": "medium",
        "description": (
            "Medium-high wing for tight corners, softer suspension to cope "
            "with bumps and uneven street surfaces, medium compound."
        ),
        "setup": {
            "front_wing": 35,
            "rear_wing": 37,
            "on_throttle_diff": 82,
            "off_throttle_diff": 52,
            "front_camber": -3.40,
            "rear_camber": -1.90,
            "front_toe": 0.05,
            "rear_toe": 0.20,
            "front_suspension": 12,
            "rear_suspension": 8,
            "front_arb": 12,
            "rear_arb": 18,
            "front_ride_height": 25,
            "rear_ride_height": 42,
            "brake_pressure": 100,
            "front_brake_bias": 55,
            "front_tyre_pressure": 24.0,
            "rear_tyre_pressure": 20.5,
            "fuel_load": 30.0,
        },
    },
    "mixed": {
        "name": "Mixed (balanced compromise)",
        "compound": "medium",
        "description": (
            "Balanced setup for circuits with very different sectors; a "
            "neutral compromise across aero, suspension and tyres."
        ),
        "setup": DEFAULT_SETUP.model_dump(),
    },
    "medium": {
        "name": "Medium (balanced)",
        "compound": "medium",
        "description": (
            "Balanced medium-downforce baseline; attack and defence in "
            "equilibrium across aero and handling."
        ),
        "setup": DEFAULT_SETUP.model_dump(),
    },
}


# --------------------------------------------------------------------------- #
# SetupPresets
# --------------------------------------------------------------------------- #
class SetupPresets:
    """Curated baseline setups per track_type / condition / driver archetype."""

    def __init__(self) -> None:
        """Load the module-level preset registry (a shallow copy)."""
        self._registry: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in _TRACK_PRESETS.items()
        }

    # ------------------------------------------------------------------ #
    def for_track(self, track_id: str) -> CarSetup:
        """Return a starting-point setup for ``track_id`` adjusted by track_type.

        Looks up the track's ``track_type`` and returns the matching curated
        baseline. Unknown ``track_id`` propagates :class:`ValueError` from
        :func:`f1opt.data.tracks.get_track`.
        """
        track = get_track(track_id)
        preset = self._registry.get(track.track_type, self._registry["medium"])
        return CarSetup(**preset["setup"])

    # ------------------------------------------------------------------ #
    def for_condition(
        self, weather: WeatherCondition, track_id: str
    ) -> CarSetup:
        """Adjust the track baseline for weather (wet / hot / cold).

        - wet / intermediate: raise ride height, more wing, softer springs.
        - hot: lower tyre pressures (and a harder-compound hint).
        - cold: softer-compound hint, more (negative) camber for heat.
        """
        state = _clone_state(self.for_track(track_id))
        if weather.is_wet() or weather.is_intermediate():
            _snap_inplace(state, "front_ride_height", +4)
            _snap_inplace(state, "rear_ride_height", +4)
            _snap_inplace(state, "rear_wing", +4)
            _snap_inplace(state, "front_wing", +2)
            _snap_inplace(state, "front_suspension", -3)
            _snap_inplace(state, "rear_suspension", -3)
        elif weather.track_temp_c >= 35.0:
            # Hot: lower pressures to limit thermal growth.
            _snap_inplace(state, "front_tyre_pressure", -0.5)
            _snap_inplace(state, "rear_tyre_pressure", -0.4)
        elif weather.track_temp_c <= 15.0:
            # Cold: more (negative) camber to help generate tyre heat.
            _snap_inplace(state, "front_camber", -0.10)
            _snap_inplace(state, "rear_camber", -0.05)
        return CarSetup(**state)

    # ------------------------------------------------------------------ #
    def for_compound(self, compound: str, base: CarSetup) -> CarSetup:
        """Adjust pressures / camber for a tyre ``compound``.

        Softer compounds run lower pressures and more negative camber; harder
        compounds run higher pressures and less camber. ``medium`` is a no-op.
        """
        state = _clone_state(base)
        if compound == "soft":
            _snap_inplace(state, "front_tyre_pressure", -0.4)
            _snap_inplace(state, "rear_tyre_pressure", -0.3)
            _snap_inplace(state, "front_camber", -0.10)
        elif compound == "hard":
            _snap_inplace(state, "front_tyre_pressure", +0.4)
            _snap_inplace(state, "rear_tyre_pressure", +0.3)
            _snap_inplace(state, "front_camber", +0.10)
        elif compound == "intermediate":
            _snap_inplace(state, "front_tyre_pressure", -0.6)
            _snap_inplace(state, "rear_tyre_pressure", -0.4)
            _snap_inplace(state, "front_camber", +0.05)
        elif compound == "wet":
            _snap_inplace(state, "front_tyre_pressure", -0.8)
            _snap_inplace(state, "rear_tyre_pressure", -0.5)
            _snap_inplace(state, "front_camber", +0.10)
        # "medium" (or anything unrecognised): leave as-is.
        return CarSetup(**state)

    # ------------------------------------------------------------------ #
    def aggressive_variant(self, base: CarSetup) -> CarSetup:
        """Return a more aggressive variant: stiffer, less rear wing, more camber."""
        state = _clone_state(base)
        _snap_inplace(state, "front_suspension", +3)
        _snap_inplace(state, "rear_suspension", +3)
        _snap_inplace(state, "front_arb", +3)
        _snap_inplace(state, "rear_arb", +3)
        _snap_inplace(state, "rear_wing", -3)
        _snap_inplace(state, "front_camber", -0.10)
        return CarSetup(**state)

    # ------------------------------------------------------------------ #
    def conservative_variant(self, base: CarSetup) -> CarSetup:
        """Return a more conservative variant: softer, more wing, less camber."""
        state = _clone_state(base)
        _snap_inplace(state, "front_suspension", -3)
        _snap_inplace(state, "rear_suspension", -3)
        _snap_inplace(state, "front_arb", -3)
        _snap_inplace(state, "rear_arb", -3)
        _snap_inplace(state, "rear_wing", +3)
        _snap_inplace(state, "front_wing", +2)
        _snap_inplace(state, "front_camber", +0.10)
        return CarSetup(**state)

    # ------------------------------------------------------------------ #
    def for_driver_archetype(
        self, archetype: str, base: CarSetup
    ) -> CarSetup:
        """Adjust ``base`` for a driver-style archetype.

        - ``AGGRESSIVE``: stiffer rear, more rear brake bias.
        - ``DEVELOPMENT``: softer, more forgiving.
        - ``TIRE_WHISPERER``: less camber, softer springs (preserve tyres).
        """
        state = _clone_state(base)
        if archetype == "AGGRESSIVE":
            _snap_inplace(state, "rear_suspension", +3)
            _snap_inplace(state, "rear_arb", +3)
            _snap_inplace(state, "front_brake_bias", -1)  # more rear bias
        elif archetype == "DEVELOPMENT":
            _snap_inplace(state, "front_suspension", -2)
            _snap_inplace(state, "rear_suspension", -2)
            _snap_inplace(state, "front_arb", -2)
            _snap_inplace(state, "rear_arb", -2)
        elif archetype == "TIRE_WHISPERER":
            _snap_inplace(state, "front_camber", +0.20)  # less negative
            _snap_inplace(state, "front_suspension", -2)
            _snap_inplace(state, "rear_suspension", -2)
        # Unknown archetype: leave as-is.
        return CarSetup(**state)

    # ------------------------------------------------------------------ #
    def list_presets(self) -> list[dict]:
        """Return a UI-friendly list of preset summaries.

        Each entry is ``{name, track_type, description, setup_diff}`` where
        ``setup_diff`` is the diff from :data:`DEFAULT_SETUP`.
        """
        result: list[dict] = []
        for track_type, preset in self._registry.items():
            setup = CarSetup(**preset["setup"])
            result.append(
                {
                    "name": preset["name"],
                    "track_type": track_type,
                    "description": preset["description"],
                    "setup_diff": DEFAULT_SETUP.diff(setup),
                }
            )
        return result


def _snap_inplace(state: dict[str, Any], field: str, delta: float) -> None:
    """Apply a clamped/snapped delta to ``state[field]`` in place."""
    state[field] = _snap(field, float(state[field]) + float(delta))


# --------------------------------------------------------------------------- #
# SetupAutoTuner
# --------------------------------------------------------------------------- #
# Recognised feedback dimensions and their signal values.
_RECOGNIZED_FEEDBACK: dict[str, set[str]] = {
    "balance": {"understeer", "oversteer"},
    "tyres": {"overheating", "too_cold"},
    "braking": {"lockup"},
    "confidence": {"low"},
    "lap_time_potential": {"gap"},
}


class SetupAutoTuner:
    """Automated setup tuning driven by a list of feedback dimensions.

    Parameters
    ----------
    current_setup
        The setup to start tuning from.
    track_id
        Track identifier (used for the optional ``lap_time_potential="gap"``
        optimizer path).
    feedback
        List of ``{name, value, advice}`` dimension dicts. ``name`` is the
        dimension (``balance`` / ``tyres`` / ``braking`` / ``confidence`` /
        ``lap_time_potential``), ``value`` is the signal (e.g. ``understeer``,
        ``overheating``, ``lockup``, ``low``, ``gap``) and ``advice`` is
        optional free-form text.
    """

    def __init__(
        self,
        current_setup: CarSetup,
        track_id: str,
        feedback: list[dict],
    ) -> None:
        self.current_setup = current_setup
        self.track_id = track_id
        self.feedback: list[dict] = list(feedback) if feedback else []
        self._diff: list[dict] | None = None

    # ------------------------------------------------------------------ #
    def _valid_dims(self) -> int:
        """Count feedback dims with a recognised name + value."""
        n = 0
        for item in self.feedback:
            name = item.get("name")
            value = item.get("value")
            if name in _RECOGNIZED_FEEDBACK and value in _RECOGNIZED_FEEDBACK[name]:
                n += 1
        return n

    # ------------------------------------------------------------------ #
    def confidence_score(self) -> float:
        """Return a confidence in ``[0, 1]``.

        More recognised feedback dimensions with clear signals yield higher
        confidence. Empty / unrecognised feedback → ``0.0``.
        """
        return min(1.0, 0.25 * self._valid_dims())

    # ------------------------------------------------------------------ #
    def tune(self) -> CarSetup:
        """Return a setup adjusted from ``current_setup`` per the feedback.

        Adjustments (each recorded for :meth:`tune_diff`):

        - ``balance="understeer"`` → increase front_wing, decrease rear_wing,
          soften front_arb.
        - ``balance="oversteer"`` → decrease front_wing, increase rear_wing,
          stiffen front_arb.
        - ``tyres="overheating"`` → lower pressures, less camber.
        - ``tyres="too_cold"`` → more camber, higher pressures.
        - ``braking="lockup"`` → move brake_bias rearward, less brake pressure.
        - ``confidence="low"`` → more rear wing, softer front.
        - ``lap_time_potential="gap"`` → run ``search_setup`` for optimization.

        The result is finally passed through :meth:`apply_constraints`.
        """
        state = _clone_state(self.current_setup)
        changes: list[dict] = []
        gap_requested = False

        for item in self.feedback:
            name = item.get("name")
            value = item.get("value")
            if name == "balance" and value == "understeer":
                _apply_delta(state, "front_wing", +2,
                             "understeer: increase front wing for front grip", changes)
                _apply_delta(state, "rear_wing", -2,
                             "understeer: decrease rear wing", changes)
                _apply_delta(state, "front_arb", -2,
                             "understeer: soften front ARB", changes)
            elif name == "balance" and value == "oversteer":
                _apply_delta(state, "front_wing", -2,
                             "oversteer: decrease front wing", changes)
                _apply_delta(state, "rear_wing", +2,
                             "oversteer: increase rear wing for rear stability", changes)
                _apply_delta(state, "front_arb", +2,
                             "oversteer: stiffen front ARB", changes)
            elif name == "tyres" and value == "overheating":
                _apply_delta(state, "front_tyre_pressure", -0.4,
                             "overheating: lower front pressure", changes)
                _apply_delta(state, "rear_tyre_pressure", -0.3,
                             "overheating: lower rear pressure", changes)
                _apply_delta(state, "front_camber", +0.10,
                             "overheating: less camber to reduce heat", changes)
            elif name == "tyres" and value == "too_cold":
                _apply_delta(state, "front_camber", -0.10,
                             "too_cold: more camber to generate heat", changes)
                _apply_delta(state, "front_tyre_pressure", +0.4,
                             "too_cold: higher front pressure", changes)
                _apply_delta(state, "rear_tyre_pressure", +0.3,
                             "too_cold: higher rear pressure", changes)
            elif name == "braking" and value == "lockup":
                _apply_delta(state, "front_brake_bias", -2,
                             "lockup: move brake bias rearward", changes)
                _apply_delta(state, "brake_pressure", -2,
                             "lockup: reduce brake pressure", changes)
            elif name == "confidence" and value == "low":
                _apply_delta(state, "rear_wing", +3,
                             "low confidence: more rear wing for stability", changes)
                _apply_delta(state, "front_suspension", -2,
                             "low confidence: softer front", changes)
            elif name == "lap_time_potential" and value == "gap":
                gap_requested = True

        setup = CarSetup(**state)

        if gap_requested:
            setup = self._apply_optimizer_search(setup, changes)

        pre_constraints = setup
        setup = self.apply_constraints(setup)
        for d in pre_constraints.diff(setup):
            changes.append(
                {
                    "field": d["name"],
                    "before": d["before"],
                    "after": d["after"],
                    "reason": "internal consistency (SetupHarmonics)",
                }
            )

        self._diff = changes
        return setup

    # ------------------------------------------------------------------ #
    def _apply_optimizer_search(
        self, setup: CarSetup, changes: list[dict]
    ) -> CarSetup:
        """Run ``search_setup`` and record the resulting diff. Best-effort."""
        try:
            from f1opt.model.optimizer import search_setup

            result = search_setup(
                self.track_id, baseline=setup, iterations=30, seed=0
            )
            recommended = CarSetup(**result.recommended)
            for d in setup.diff(recommended):
                changes.append(
                    {
                        "field": d["name"],
                        "before": d["before"],
                        "after": d["after"],
                        "reason": "lap_time_potential: surrogate optimizer search",
                    }
                )
            return recommended
        except Exception:
            log.debug("lap_time_potential optimizer search failed; keeping setup", exc_info=True)
            return setup

    # ------------------------------------------------------------------ #
    def tune_diff(self) -> list[dict]:
        """Return the list of ``{field, before, after, reason}`` changes.

        Computes :meth:`tune` on demand if it has not been called yet.
        """
        if self._diff is None:
            self.tune()
        assert self._diff is not None
        return self._diff

    # ------------------------------------------------------------------ #
    def apply_constraints(self, setup: CarSetup) -> CarSetup:
        """Ensure ``setup`` is internally consistent via :class:`SetupHarmonics`.

        Runs the spring/ARB harmony, ride-height/rake and camber-alignment
        checks and nudges offending fields back into a consistent window over
        a small number of iterations. Always returns a valid
        :class:`CarSetup`.
        """
        state = _clone_state(setup)
        for _ in range(3):
            harmonics = SetupHarmonics(state)
            result = harmonics.all_checks()
            if result["ok"]:
                break
            for check_name, check_result in result["checks"].items():
                if check_result["ok"]:
                    continue
                self._fix_check(state, check_name, check_result)
        return CarSetup(**state)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fix_check(
        state: dict[str, Any], check_name: str, check_result: dict
    ) -> None:
        """Nudge ``state`` to resolve a single failed harmony check."""
        if check_name == "spring_arb_harmony":
            # Local import avoids pulling suspension constants at module load.
            from f1opt.model.suspension import (
                ARB_STIFFNESS_MAX_N_M_PER_DEG,
                ARB_STIFFNESS_MIN_N_M_PER_DEG,
                SPRING_RATE_MAX_N_PER_MM,
                SPRING_RATE_MIN_N_PER_MM,
                SuspensionModel,
            )

            susp = SuspensionModel(state)
            for kind, f_spring, f_arb in (
                ("front", "front_suspension", "front_arb"),
                ("rear", "rear_suspension", "rear_arb"),
            ):
                ns = (susp.spring_rate(kind) - SPRING_RATE_MIN_N_PER_MM) / (
                    SPRING_RATE_MAX_N_PER_MM - SPRING_RATE_MIN_N_PER_MM
                )
                na = (susp.arb_stiffness(kind) - ARB_STIFFNESS_MIN_N_M_PER_DEG) / (
                    ARB_STIFFNESS_MAX_N_M_PER_DEG - ARB_STIFFNESS_MIN_N_M_PER_DEG
                )
                ns = max(0.0, min(1.0, ns))
                na = max(0.0, min(1.0, na))
                if ns - na > 0.35:
                    state[f_spring] = _snap(f_spring, float(state[f_spring]) - 1)
                    state[f_arb] = _snap(f_arb, float(state[f_arb]) + 1)
                elif na - ns > 0.35:
                    state[f_spring] = _snap(f_spring, float(state[f_spring]) + 1)
                    state[f_arb] = _snap(f_arb, float(state[f_arb]) - 1)
        elif check_name == "ride_height_rake":
            rake_mm = float(check_result.get("rake_mm", 0.0))
            if rake_mm > 15.0:
                state["rear_ride_height"] = _snap(
                    "rear_ride_height", float(state["rear_ride_height"]) - 2
                )
            elif rake_mm < 5.0:
                new_rear = _snap(
                    "rear_ride_height", float(state["rear_ride_height"]) + 2
                )
                if new_rear == state["rear_ride_height"]:
                    state["front_ride_height"] = _snap(
                        "front_ride_height", float(state["front_ride_height"]) - 2
                    )
                else:
                    state["rear_ride_height"] = new_rear
        elif check_name == "camber_alignment":
            fc = float(state["front_camber"])
            rc = float(state["rear_camber"])
            if fc >= rc:
                state["front_camber"] = _snap("front_camber", fc - 0.10)
            if float(state["front_camber"]) > -2.0:
                state["front_camber"] = _snap(
                    "front_camber", float(state["front_camber"]) - 0.10
                )
