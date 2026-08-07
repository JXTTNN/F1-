"""Causal-explanation & what-if setup analysis for F1OPT feedback (Iter-26..28).

Adds a *physically-grounded* causal layer on top of the 19-field
:class:`~f1opt.data.setup_schema.CarSetup` schema:

- :data:`CAUSAL_RULES` — a rules table covering ALL 19 setup fields. Each
  entry maps an increase / decrease of the field's raw value to a primary
  physical effect, 1-3 cascading secondary effects, and signed per-metric
  delta coefficients (``expected = coefficient * (proposed - current)``).
- :class:`CausalExplanationEngine` — given a single (field, current,
  proposed) change, produce a causal explanation dict: change direction,
  magnitude %, primary effect, secondary effects, expected metric deltas,
  risk classification, and a Chinese causal-chain sentence.
- :class:`WhatIfAnalyzer` — combine a causal explanation with a surrogate
  ``predict_lap_time`` delta (lazy-imported), a confidence estimate, and
  recommended accompanying field adjustments; also supports batch multi-field
  changes.

Conventions
-----------
``delta = proposed - current`` is SIGNED and expressed in the field's natural
units (clicks / percent / degrees / psi / kg). Metric-delta coefficients in
:data:`CAUSAL_RULES` are "effect per unit *increase* in the field's raw value";
for camber fields (which are negative degrees) the coefficient sign is chosen
so that a more-negative value (larger camber) yields the physically correct
direction (e.g. corner grip increases). This keeps the model monotonic and a
single coefficient per metric is sufficient.

Risk classification is keyed off the number of snapped "clicks" of change
(``|delta| / step``): ``<=2`` low, ``<=5`` medium, else high.

The module imports ``CarSetup`` / ``SETUP_FIELDS`` / ``_snap_to_step`` at
module load (cheap, pydantic-only); the torch surrogate is imported lazily
inside :class:`WhatIfAnalyzer` methods so merely importing this module (or
inspecting :data:`CAUSAL_RULES`) never loads the model.
"""

from __future__ import annotations

from typing import Any

from f1opt.data.setup_schema import SETUP_FIELDS, CarSetup, SetupField, _snap_to_step

__all__ = [
    "ACCOMPANYING_RULES",
    "CAUSAL_RULES",
    "CausalExplanationEngine",
    "WhatIfAnalyzer",
]

#: Clicks-of-change thresholds for risk classification (|delta| / spec.step).
RISK_LOW_STEPS: float = 2.0
RISK_HIGH_STEPS: float = 5.0


# --------------------------------------------------------------------------- #
# CAUSAL_RULES — physically-reasonable rules for all 19 setup fields.
# Each entry:
#   primary_effect_inc / _dec : Chinese primary effect (for value inc / dec)
#   secondary_inc / _dec      : 1-3 cascading secondary effects
#   metric_deltas              : {metric: coef} where expected = coef * delta
# (delta = proposed - current, signed; for camber the coef sign already
# encodes the more-negative-is-more-camber direction.)
# --------------------------------------------------------------------------- #
CAUSAL_RULES: dict[str, dict[str, Any]] = {
    # --- Aerodynamics ---
    "front_wing": {
        "primary_effect_inc": "增加前轴下压力",
        "primary_effect_dec": "降低前轴下压力",
        "secondary_inc": ["前轮机械抓地力提升", "入弯响应更直接", "直道尾速下降"],
        "secondary_dec": ["前轮机械抓地力下降", "入弯响应变迟钝", "直道尾速提升"],
        "metric_deltas": {"grip": 0.03, "top_speed": -0.8, "tyre_temp_f": 2.0},
    },
    "rear_wing": {
        "primary_effect_inc": "增加后轴下压力",
        "primary_effect_dec": "降低后轴下压力",
        "secondary_inc": ["后轮抓地力提升", "出弯牵引力改善", "直道尾速下降"],
        "secondary_dec": ["后轮抓地力下降", "出弯牵引力减弱", "直道尾速提升"],
        "metric_deltas": {"grip": 0.02, "top_speed": -1.0, "tyre_temp_r": 1.5},
    },
    # --- Transmission ---
    "on_throttle_diff": {
        "primary_effect_inc": "提高油门差速锁止率",
        "primary_effect_dec": "降低油门差速锁止率",
        "secondary_inc": ["出弯牵引力增强", "后轮协同性提升", "低速推头倾向加剧"],
        "secondary_dec": ["出弯后轮更易滑动", "低速转向更灵活", "牵引力减弱"],
        "metric_deltas": {"traction": 0.05, "understeer": 0.03, "tyre_wear_r": 1.0},
    },
    "off_throttle_diff": {
        "primary_effect_inc": "提高收油差速率",
        "primary_effect_dec": "降低收油差速率",
        "secondary_inc": ["入弯后轴锁止增强", "收油稳定性提升", "转向不足略增"],
        "secondary_dec": ["收油后轴更灵活", "入弯转向更敏锐", "收油稳定性下降"],
        "metric_deltas": {"stability": 0.04, "understeer": 0.02},
    },
    # --- Suspension Geometry ---
    "front_camber": {
        # value increase (less negative -> less camber) -> less grip/temp/wear
        "primary_effect_inc": "减小负外倾角(更接近垂直)",
        "primary_effect_dec": "增加负外倾角",
        "secondary_inc": ["弯中胎面接触减小", "直行胎温降低", "轮胎内侧磨损减轻"],
        "secondary_dec": ["弯中外倾侧胎面接触改善", "直行胎温升高", "轮胎内侧磨损加剧"],
        # coef negative so more-negative (delta<0) -> positive corner grip gain
        "metric_deltas": {"corner_grip": -0.04, "tyre_temp": -1.5, "tyre_wear_inner": -3.0},
    },
    "rear_camber": {
        "primary_effect_inc": "减小后轮负外倾角",
        "primary_effect_dec": "增加后轮负外倾角",
        "secondary_inc": ["后弯中胎面接触减小", "后直行胎温降低", "后轮内侧磨损减轻"],
        "secondary_dec": ["后弯中外倾侧胎面接触改善", "后直行胎温升高", "后轮内侧磨损加剧"],
        "metric_deltas": {"corner_grip_r": -0.03, "tyre_temp": -1.0, "tyre_wear_inner_r": -2.0},
    },
    "front_toe": {
        "primary_effect_inc": "增大前轮前束(外束)",
        "primary_effect_dec": "减小前轮前束",
        "secondary_inc": ["入弯初始响应更敏锐", "直行滚动阻力略增", "前胎外侧磨损加剧"],
        "secondary_dec": ["入弯初始响应变钝", "直行稳定性提升", "前胎外侧磨损减轻"],
        "metric_deltas": {"response": 0.04, "tyre_wear_outer": 2.0, "top_speed": -0.2},
    },
    "rear_toe": {
        "primary_effect_inc": "增大后轮前束(内束)",
        "primary_effect_dec": "减小后轮前束",
        "secondary_inc": ["后直行稳定性提升", "后轮内侧磨损加剧", "转向灵活性略降"],
        "secondary_dec": ["后直行稳定性下降", "转向灵活性提升", "后轮内侧磨损减轻"],
        "metric_deltas": {"stability": 0.03, "tyre_wear_inner_r": 2.0},
    },
    # --- Suspension ---
    "front_suspension": {
        "primary_effect_inc": "提高前弹簧硬度",
        "primary_effect_dec": "降低前弹簧硬度",
        "secondary_inc": ["前轴支撑性增强", "前轮压缩下沉减小", "路感与颠簸传递增加"],
        "secondary_dec": ["前轴支撑性减弱", "前轮压缩下沉增大", "颠簸传递减弱"],
        "metric_deltas": {"response": 0.03, "grip_f": -0.01, "tyre_temp_f": 1.0},
    },
    "rear_suspension": {
        "primary_effect_inc": "提高后弹簧硬度",
        "primary_effect_dec": "降低后弹簧硬度",
        "secondary_inc": ["后轴支撑性增强", "出弯牵引力改善", "后轮颠簸传递增加"],
        "secondary_dec": ["后轴支撑性减弱", "出弯牵引力减弱", "后轮颠簸传递减弱"],
        "metric_deltas": {"traction": 0.03, "grip_r": -0.01, "tyre_temp_r": 1.0},
    },
    "front_arb": {
        "primary_effect_inc": "提高前防倾杆硬度",
        "primary_effect_dec": "降低前防倾杆硬度",
        "secondary_inc": ["前轴侧倾刚度增加", "弯中前轮载荷转移增大", "推头倾向加剧"],
        "secondary_dec": ["前轴侧倾刚度减小", "弯中前轮载荷转移减小", "推头倾向缓解"],
        "metric_deltas": {"understeer": 0.04, "corner_grip_f": -0.02},
    },
    "rear_arb": {
        "primary_effect_inc": "提高后防倾杆硬度",
        "primary_effect_dec": "降低后防倾杆硬度",
        "secondary_inc": ["后轴侧倾刚度增加", "弯中后轮载荷转移增大", "甩尾倾向加剧"],
        "secondary_dec": ["后轴侧倾刚度减小", "弯中后轮载荷转移减小", "甩尾倾向缓解"],
        "metric_deltas": {"oversteer": 0.04, "corner_grip_r": -0.02},
    },
    "front_ride_height": {
        "primary_effect_inc": "提高前离地间隙",
        "primary_effect_dec": "降低前离地间隙",
        "secondary_inc": ["前翼下压力效率下降", "底盘气流泄漏增加", "前轴抓地力下降"],
        "secondary_dec": ["前翼下压力效率提升", "底盘气流密封改善", "触底风险增加"],
        "metric_deltas": {"grip_f": -0.02, "top_speed": 0.3, "downforce_f": -0.05},
    },
    "rear_ride_height": {
        "primary_effect_inc": "提高后离地间隙",
        "primary_effect_dec": "降低后离地间隙",
        "secondary_inc": ["后轴下压力效率下降", "rake 角增大", "后轴抓地力下降"],
        "secondary_dec": ["后轴下压力效率提升", "rake 角减小", "后轴触底风险增加"],
        "metric_deltas": {"grip_r": -0.02, "top_speed": 0.3, "rake": 0.5},
    },
    # --- Brakes ---
    "brake_pressure": {
        "primary_effect_inc": "提高制动压力上限",
        "primary_effect_dec": "降低制动压力上限",
        "secondary_inc": ["制动力增强", "锁死风险增加", "制动距离缩短"],
        "secondary_dec": ["制动力减弱", "锁死风险降低", "制动距离增加"],
        "metric_deltas": {"brake_force": 0.05, "lockup_risk": 0.08, "brake_temp": 5.0},
    },
    "front_brake_bias": {
        "primary_effect_inc": "前制动力分配增加",
        "primary_effect_dec": "后制动力分配增加",
        "secondary_inc": ["入弯前轴更容易锁死", "入弯稳定性提升", "出弯后轴负载减小"],
        "secondary_dec": ["入弯前轴锁死风险降低", "入弯稳定性下降", "出弯后轴负载增大"],
        "metric_deltas": {"lockup_risk_f": 0.1, "stability": 0.05, "turn_in": 0.03},
    },
    # --- Tyres ---
    "front_tyre_pressure": {
        "primary_effect_inc": "前胎气压升高",
        "primary_effect_dec": "前胎气压降低",
        "secondary_inc": ["胎面接触面积减小", "胎温工作区间升高", "直行响应更锐利"],
        "secondary_dec": ["胎面接触面积增大", "胎温工作区间降低", "直行响应变钝"],
        "metric_deltas": {"tyre_temp": 2.0, "grip": -0.02, "response": 0.04},
    },
    "rear_tyre_pressure": {
        "primary_effect_inc": "后胎气压升高",
        "primary_effect_dec": "后胎气压降低",
        "secondary_inc": ["后胎面接触面积减小", "后胎温工作区间升高", "后直行响应更锐利"],
        "secondary_dec": ["后胎面接触面积增大", "后胎温工作区间降低", "后直行响应变钝"],
        "metric_deltas": {"tyre_temp_r": 2.0, "grip_r": -0.02, "response_r": 0.04},
    },
    # --- Fuel ---
    "fuel_load": {
        "primary_effect_inc": "增加燃油装载量",
        "primary_effect_dec": "减少燃油装载量",
        "secondary_inc": ["车重增加", "圈速下降", "制动距离增加"],
        "secondary_dec": ["车重减轻", "圈速提升", "制动距离缩短"],
        "metric_deltas": {"lap_time": 0.03, "top_speed": -0.1, "tyre_wear": 0.5},
    },
    # --- Active Aero (Iter-219) ---
    "active_aero_mode": {
        "primary_effect_inc": "切换至更高下压力模式",
        "primary_effect_dec": "切换至更低阻力模式",
        "secondary_inc": ["弯中抓地力提升", "直道尾速下降", "胎耗增加"],
        "secondary_dec": ["直道尾速提升", "弯中抓地力下降", "胎耗降低"],
        "metric_deltas": {"grip": 0.05, "top_speed": -2.0, "tyre_wear": 1.0},
    },
    "x_mode_activations": {
        "primary_effect_inc": "增加 X-Mode 激活次数",
        "primary_effect_dec": "减少 X-Mode 激活次数",
        "secondary_inc": ["直道尾速提升幅度增大", "出弯加速后更快切换低阻", "弯中下压力不足风险增加"],
        "secondary_dec": ["弯中下压力保持更稳定", "直道尾速提升幅度减小", "连续弯段稳定性提升"],
        "metric_deltas": {"top_speed": 0.5, "grip": -0.02, "lap_time": -0.01},
    },
}


# --------------------------------------------------------------------------- #
# Recommended accompanying field adjustments (used by WhatIfAnalyzer).
# Each value is a list of (other_field, reason) suggestions.
# --------------------------------------------------------------------------- #
ACCOMPANYING_RULES: dict[str, list[tuple[str, str]]] = {
    "front_wing": [("rear_wing", "同向微调以维持前后气动平衡")],
    "rear_wing": [("front_wing", "同向微调以维持前后气动平衡")],
    "front_brake_bias": [("brake_pressure", "视锁死情况同步调整制动压力")],
    "brake_pressure": [("front_brake_bias", "调整制动分配以匹配新制动力")],
    "front_arb": [("rear_arb", "反向调整以维持侧向平衡")],
    "rear_arb": [("front_arb", "反向调整以维持侧向平衡")],
    "front_suspension": [("rear_suspension", "同步调整以维持前后弹簧平衡")],
    "rear_suspension": [("front_suspension", "同步调整以维持前后弹簧平衡")],
    "front_ride_height": [("rear_ride_height", "同向调整以维持 rake")],
    "rear_ride_height": [("front_ride_height", "同向调整以维持 rake")],
    "front_tyre_pressure": [("rear_tyre_pressure", "视胎温同步调整后轮胎压")],
    "rear_tyre_pressure": [("front_tyre_pressure", "视胎温同步调整前轮胎压")],
    "front_camber": [("rear_camber", "同步调整以维持前后外倾平衡")],
    "rear_camber": [("front_camber", "同步调整以维持前后外倾平衡")],
    "on_throttle_diff": [("off_throttle_diff", "评估收油差速以保持入弯一致性")],
    "off_throttle_diff": [("on_throttle_diff", "评估油门差速以保持出弯一致性")],
    "front_toe": [("rear_toe", "视转向响应同步调整后束")],
    "rear_toe": [("front_toe", "视转向响应同步调整前束")],
    "fuel_load": [],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _magnitude_pct(current: float, proposed: float, spec: SetupField) -> float:
    """Relative change in % (|delta| / |current|; falls back to range span)."""
    delta = abs(float(proposed) - float(current))
    cur = abs(float(current))
    if cur > 1e-9:
        return delta / cur * 100.0
    span = float(spec.max) - float(spec.min)
    if span > 0:
        return delta / span * 100.0
    return 0.0


def _num_steps(delta: float, spec: SetupField) -> float:
    """Number of snapped "clicks" of change (|delta| / spec.step)."""
    step = float(spec.step)
    return abs(float(delta)) / step if step > 0 else 0.0


def _classify_risk(delta: float, spec: SetupField) -> str:
    """Risk = low (<=2 clicks) / medium (<=5) / high (>5)."""
    clicks = _num_steps(delta, spec)
    if clicks <= RISK_LOW_STEPS:
        return "low"
    if clicks <= RISK_HIGH_STEPS:
        return "medium"
    return "high"


def _risk_reason(risk: str, delta: float, spec: SetupField) -> str:
    clicks = _num_steps(delta, spec)
    return f"变化约 {clicks:.1f} 个档位 ({spec.unit}), 风险等级={risk}"


def _build_causal_chain(
    field: str,
    spec: SetupField,
    change: str,
    primary: str,
    secondary: list[str],
    delta: float,
    magnitude_pct: float,
    track_type: str | None,
) -> str:
    """Render the Chinese causal-chain sentence.

    Format: ``<verb><field_zh>→<primary>→<sec1>→<sec2>→<sec3> (变化 X unit, 相对 Y%)``
    plus an optional track-context tail. For ``unchanged`` returns a neutral
    marker (no causal chain since there is no effect).
    """
    if change == "unchanged":
        return f"{spec.description}未变化, 无因果效应"
    verb = "增大" if change == "increased" else "减小"
    chain_parts = [f"{verb}{spec.description}", primary, *secondary]
    chain = "→".join(chain_parts)
    delta_abs = abs(float(delta))
    tail = f" (变化 {delta_abs:g} {spec.unit}, 相对 {magnitude_pct:.1f}%)"
    if track_type:
        tail += f" [赛道类型={track_type}]"
    return chain + tail


# --------------------------------------------------------------------------- #
# CausalExplanationEngine
# --------------------------------------------------------------------------- #
class CausalExplanationEngine:
    """Produce causal explanations for single-field setup changes.

    Parameters
    ----------
    setup_schema
        Mapping ``{field_name: SetupField}`` (typically
        :data:`~f1opt.data.setup_schema.SETUP_FIELDS`). Used for unit / range /
        step metadata and the Chinese field description.
    track
        Optional :class:`~f1opt.data.tracks.Track` for context (currently only
        surfaced in the explanation tail).
    """

    def __init__(self, setup_schema: dict[str, SetupField], track: Any = None) -> None:
        self.setup_schema = setup_schema
        self.track = track

    def explain(
        self,
        field: str,
        current: float,
        proposed: float,
        current_metrics: dict | None = None,
        track_type: str | None = None,
    ) -> dict:
        """Explain a (field, current -> proposed) change as a causal dict.

        Returns a dict with keys: ``field``, ``change`` ("increased" /
        "decreased" / "unchanged"), ``magnitude_pct``, ``primary_effect``,
        ``secondary_effects``, ``expected_metric_deltas``, ``risk``,
        ``risk_reason``, ``explanation_text``.

        ``current_metrics`` is accepted for forward-compat (e.g. grounding the
        expected deltas in observed telemetry) but not required.
        """
        spec = self.setup_schema.get(field)
        if spec is None:
            raise KeyError(f"未知调教字段: {field!r} (不在 setup_schema 中)")
        if field not in CAUSAL_RULES:
            raise KeyError(f"CAUSAL_RULES 缺少字段 {field!r} 的因果规则")

        delta = float(proposed) - float(current)
        rule = CAUSAL_RULES[field]

        if delta > 1e-12:
            change = "increased"
            primary = rule["primary_effect_inc"]
            secondary = list(rule["secondary_inc"])
        elif delta < -1e-12:
            change = "decreased"
            primary = rule["primary_effect_dec"]
            secondary = list(rule["secondary_dec"])
        else:
            change = "unchanged"
            # No change -> no direction; surface the increase-side templates
            # only as a "what would an increase do" reference is not needed, so
            # we keep primary/secondary from the increase template but zero the
            # numeric deltas below.
            primary = rule["primary_effect_inc"]
            secondary = list(rule["secondary_inc"])

        magnitude_pct = _magnitude_pct(current, proposed, spec)

        coefs: dict[str, float] = rule["metric_deltas"]
        if change == "unchanged":
            expected_metric_deltas = {k: 0.0 for k in coefs}
        else:
            expected_metric_deltas = {k: float(v) * delta for k, v in coefs.items()}

        risk = _classify_risk(delta, spec)
        risk_reason = _risk_reason(risk, delta, spec)
        explanation_text = _build_causal_chain(
            field, spec, change, primary, secondary, delta, magnitude_pct, track_type
        )

        return {
            "field": field,
            "change": change,
            "magnitude_pct": magnitude_pct,
            "primary_effect": primary,
            "secondary_effects": secondary,
            "expected_metric_deltas": expected_metric_deltas,
            "risk": risk,
            "risk_reason": risk_reason,
            "explanation_text": explanation_text,
        }


# --------------------------------------------------------------------------- #
# WhatIfAnalyzer
# --------------------------------------------------------------------------- #
def _coerce_setup(setup: CarSetup | dict) -> CarSetup:
    """Accept a ``CarSetup`` or a dict; return a validated ``CarSetup``."""
    if isinstance(setup, CarSetup):
        return setup
    if isinstance(setup, dict):
        return CarSetup(**setup)
    raise TypeError(
        f"setup 必须是 CarSetup 或 dict, 收到 {type(setup).__name__}"
    )


# Iter-183: Compound sensitivity matrix — maps tyre compound to grip multiplier.
# Soft compounds have higher grip but amplify setup sensitivity; hard compounds
# are more forgiving but have lower absolute grip.
_COMPOUND_GRIP_MULT: dict[str, float] = {
    "soft": 1.10,
    "medium": 1.00,
    "hard": 0.90,
    "intermediate": 0.70,
    "wet": 0.55,
}

# Iter-183: Weather sensitivity — maps weather condition to confidence modifier.
# Rain reduces predictability significantly.
_WEATHER_CONFIDENCE_MOD: dict[str, float] = {
    "dry": 1.00,
    "damp": 0.92,
    "wet": 0.80,
    "storm": 0.65,
}


class WhatIfAnalyzer:
    """Combine causal explanation with a surrogate lap-time prediction.

    ``analyze_change`` lazily imports :func:`~f1opt.model.surrogate.predict_lap_time`
    so importing this module never loads the torch model.

    Parameters
    ----------
    setup
        Baseline setup (``CarSetup`` or dict) to perturb.
    track_id
        Track id forwarded to the surrogate (e.g. ``"melbourne"``).
    driver_profile
        Optional driver profile forwarded to the surrogate.
    compound
        Optional tyre compound (``"soft"`` / ``"medium"`` / ``"hard"`` /
        ``"intermediate"`` / ``"wet"``). Affects grip multiplier and
        confidence scoring.
    weather
        Optional weather condition (``"dry"`` / ``"damp"`` / ``"wet"`` /
        ``"storm"``). Affects confidence scoring.
    """

    def __init__(
        self,
        setup: CarSetup | dict,
        track_id: str,
        driver_profile: Any = None,
        compound: str = "medium",
        weather: str = "dry",
    ) -> None:
        self.setup_obj = _coerce_setup(setup)
        self.track_id = track_id
        self.driver_profile = driver_profile
        self.compound = compound
        self.weather = weather
        self.causal_engine = CausalExplanationEngine(SETUP_FIELDS, track=None)
        self._base_lap_cached: float | None = None

    # -- internals --------------------------------------------------------- #
    def _track_type(self) -> str | None:
        try:
            from f1opt.data.tracks import get_track

            return get_track(self.track_id).track_type
        except Exception:
            return None

    def _base_lap(self) -> float:
        """Cached baseline lap time (predicts once for the analyzer's setup)."""
        if self._base_lap_cached is None:
            from f1opt.model.surrogate import predict_lap_time

            self._base_lap_cached = float(
                predict_lap_time(self.setup_obj, self.track_id, self.driver_profile)
            )
        return self._base_lap_cached

    @staticmethod
    def _snapped_value(spec: SetupField, new_value: float) -> float:
        """Snap ``new_value`` to the legal grid; coerce to int for int fields."""
        snapped = _snap_to_step(float(new_value), spec)
        if spec.kind == "int":
            return float(int(round(snapped)))
        return float(snapped)

    # -- public API -------------------------------------------------------- #
    def analyze_change(self, field: str, new_value: float) -> dict:
        """What-if predict a single-field change against the baseline setup.

        Returns a dict with: ``field``, ``current``, ``proposed``,
        ``delta``, ``causal`` (the :class:`CausalExplanationEngine` dict),
        ``lap_time_delta`` (signed seconds, modified - baseline),
        ``confidence`` (0..1, lower for larger changes), and
        ``recommended_accompanying`` (list of dicts from
        :meth:`suggest_accompanying`).
        """
        spec = SETUP_FIELDS[field]
        current = float(getattr(self.setup_obj, field))
        proposed = self._snapped_value(spec, new_value)
        delta = proposed - current

        causal = self.causal_engine.explain(
            field, current, proposed, {}, self._track_type()
        )

        # Surrogate lap-time delta (lazy import; torch only loaded on first call).
        from f1opt.model.surrogate import predict_lap_time

        modified = self.setup_obj.model_copy(update={field: proposed})
        base_lap = self._base_lap()
        new_lap = float(predict_lap_time(modified, self.track_id, self.driver_profile))
        lap_time_delta = new_lap - base_lap

        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
        accompanying = self.suggest_accompanying(field, direction)

        clicks = _num_steps(delta, spec)
        # Iter-183: Non-linear confidence decay (exponential) with track-awareness.
        # Confidence = 0.95 * exp(-0.15 * clicks), floor 0.3 (was 0.4).
        # Small changes (≤2 clicks) retain > 0.7 confidence; large changes (≥10)
        # asymptote to 0.3.  Track-specific modifiers adjust confidence further.
        import math as _math
        confidence = max(0.3, min(0.98, 0.95 * _math.exp(-0.15 * clicks)))
        # Track-type modifier: high-speed tracks amplify setup sensitivity
        track_type = self._track_type()
        if track_type in ("high_speed", "street"):
            confidence *= 0.95  # high-speed tracks are less predictable
        elif track_type in ("technical", "mixed"):
            confidence *= 0.98  # technical tracks have more predictable sensitivities
        # Compound modifier: soft tyres amplify setup effects, hard tyres dampen
        compound_mod = _COMPOUND_GRIP_MULT.get(self.compound, 1.0)
        if compound_mod > 1.0:
            confidence *= 0.93  # soft compounds: wider operating window = less predictable
        elif compound_mod < 1.0:
            confidence *= 1.02  # hard compounds: narrower window = more predictable
        # Weather modifier: rain reduces confidence substantially
        weather_mod = _WEATHER_CONFIDENCE_MOD.get(self.weather, 1.0)
        confidence *= weather_mod
        confidence = max(0.25, min(0.98, confidence))

        return {
            "field": field,
            "current": current,
            "proposed": proposed,
            "delta": delta,
            "causal": causal,
            "lap_time_delta": lap_time_delta,
            "confidence": confidence,
            "recommended_accompanying": accompanying,
        }

    def analyze_multi_change(self, changes: dict[str, float]) -> dict:
        """Batch what-if: apply several field changes together.

        Returns a dict with: ``changes`` (list of per-field causal dicts),
        ``lap_time_delta`` (combined modified - baseline), ``n_fields``,
        and ``combined_setup_delta`` (summary of per-field signed deltas).
        """
        updates: dict[str, float] = {}
        per_field_causal: list[dict] = []
        per_field_delta: list[dict] = []
        track_type = self._track_type()

        for field, new_value in changes.items():
            spec = SETUP_FIELDS[field]
            current = float(getattr(self.setup_obj, field))
            proposed = self._snapped_value(spec, new_value)
            updates[field] = proposed
            causal = self.causal_engine.explain(
                field, current, proposed, {}, track_type
            )
            per_field_causal.append(causal)
            per_field_delta.append(
                {
                    "field": field,
                    "current": current,
                    "proposed": proposed,
                    "delta": proposed - current,
                }
            )

        from f1opt.model.surrogate import predict_lap_time

        modified = self.setup_obj.model_copy(update=updates)
        base_lap = self._base_lap()
        new_lap = float(predict_lap_time(modified, self.track_id, self.driver_profile))
        lap_time_delta = new_lap - base_lap

        return {
            "changes": per_field_causal,
            "lap_time_delta": lap_time_delta,
            "n_fields": len(changes),
            "combined_setup_delta": per_field_delta,
        }

    def suggest_accompanying(self, field: str, direction: int) -> list[dict]:
        """Suggest 0-1 accompanying field adjustments for ``field``.

        ``direction`` is ``+1`` (increase), ``-1`` (decrease) or ``0``.
        Returns a list of ``{"field", "direction", "reason"}`` dicts. For
        balance-paired fields (front/rear wing, ARBs, suspension, ride
        height, tyres, camber) the suggestion mirrors the requested direction;
        the caller may invert it for fields documented as reverse-coupled
        (e.g. front/rear ARB) — here we keep the sign simple and document the
        intent in ``reason``.
        """
        suggestions: list[dict] = []
        for other_field, reason in ACCOMPANYING_RULES.get(field, []):
            suggestions.append(
                {
                    "field": other_field,
                    "direction": int(direction),
                    "reason": reason,
                }
            )
        return suggestions
