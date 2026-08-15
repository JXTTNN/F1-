"""LLM driver-feedback engine.

Iter-03 Task 3.1: produces evidence-grounded feedback covering ALL 18 spec
dimensions (balance / grip / tyres / braking / ers_deployment / drs_usage /
throttle_brake_smoothness / confidence / lap_time_potential / sector_compare /
setup_advice). Works without an LLM API key via a comprehensive rule-based
fallback, and supports a pluggable LLM backend (``config.llm_backend``) for
richer natural language — the LLM-enhance path stays gated (default off) but
its prompt already lists all 18 dimensions.

Pipeline:

1. :func:`extract_metrics` — compute aggregate telemetry metrics WITH evidence
   (every metric stores the frame_t + field + value it was derived from).
   Iter-03 adds braking (lockup_proxy, brake_bias_assessment), confidence
   (steering_correction_freq, g_lat_stability) and sector_compare
   (sector_times derived from lap_distance crossings, else nominal split).
2. :func:`rule_based_feedback` — emit ALL 18 dimension entries + setup
   suggestions via F1 setup knowledge encoded as rules; the ``setup_advice``
   dimension consumes :func:`f1opt.model.optimizer.search_setup` to present the
   model-driven recommended setup diff.
3. :func:`llm_enhance` — optionally rewrite the summary into natural prose and
   answer the driver's question via an OpenAI-compatible chat completions API.

Every numeric claim in the rule-based path is traceable to an entry in
``sources`` (frame_t + field + value). The rule-based path never invents
numbers. The ``setup_advice`` dimension documents its model-derived numbers in
its ``evidence`` field (the search_setup call signature + recommended lap).

Public entry: :func:`generate_feedback` (constructs a cached default engine).
"""

from __future__ import annotations

import gc as _gc
import logging as _logging
import threading as _threading
import time as _time
from collections import deque as _deque
from dataclasses import dataclass as _dataclass
from functools import lru_cache
from typing import Any

_logger = _logging.getLogger(__name__)

import numpy as np

from f1opt.config import Settings, get_settings
from f1opt.data.setup_schema import SETUP_FIELDS, _snap_to_step
from f1opt.data.tracks import TRACKS_BY_ID, Track, get_track
from f1opt.driver.profile import DriverProfile

from .conversation import ConversationSession, get_session
from .prompts import (
    FEEDBACK_DIMENSIONS,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_driver_profile,
)
from .quality import assess_response_quality

__all__ = [
    "FEEDBACK_DIMENSIONS",
    "FeedbackEngine",
    "extract_metrics",
    "generate_feedback",
    "llm_enhance",
    "rule_based_feedback",
]

#: All 12 feedback dimensions required by the setup-optimizer spec (Iter-03).
#: Re-exported from :mod:`f1opt.feedback.prompts` (single source of truth) so
#: the rule-based path and the LLM prompt dimension list never drift apart.
#: The rule-based path always emits one entry per name in this order.

# Nominal F1 2026 reference lap pace (seconds) per track_type, scaled per-track
# by length relative to a 5 km nominal circuit.
_REF_LAP_BY_TYPE: dict[str, float] = {
    "high_speed_low_downforce": 95.0,
    "street": 95.0,
    "high_downforce": 78.0,
    "medium": 90.0,
    "mixed": 95.0,
}

# Nominal sector split weights (sum to 1.0); used when sector boundaries cannot
# be derived from telemetry (single-lap / sparse lap_distance).
_SECTOR_PRIOR_WEIGHTS: tuple[float, float, float] = (0.34, 0.33, 0.33)

# OpenAI-compatible chat completions endpoints per backend.
_LLM_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "local": "http://localhost:11434/v1/chat/completions",
}
_LLM_DEFAULT_MODEL: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "local": "llama3.1",
}


# --------------------------------------------------------------------------- #
# Iter-138: LLM token usage tracking
# --------------------------------------------------------------------------- #
@_dataclass
class TokenUsageRecord:
    """A single LLM call's token usage (one row in the tracker's log).

    - ``backend``: ``"openai"`` / ``"local"`` / etc.
    - ``model``: Model name reported in the request payload.
    - ``prompt_tokens``: Tokens in the prompt (input).
    - ``completion_tokens``: Tokens in the completion (output).
    - ``total_tokens``: ``prompt + completion`` (as reported by the API).
    - ``success``: Whether the call returned a usable completion.
    - ``streamed``: Whether the call used the streaming endpoint.
    - ``timestamp``: ``time.time()`` when the record was logged.
    """

    backend: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    streamed: bool
    timestamp: float


class TokenUsageTracker:
    """Thread-safe accumulator for LLM token usage (Iter-138).

    Records per-call usage (prompt / completion / total tokens) and exposes
    cumulative totals, per-backend breakdowns, and a bounded recent-records
    log. Designed for cost monitoring: an EA F1 2026 engineer running the
    feedback loop over a session wants to know how many tokens (and roughly
    how many cents) the LLM backend has consumed.

    Thread-safe via an internal ``threading.Lock`` — safe to share across the
    sync and async feedback paths, which may run concurrently under FastAPI.

    Public API:

    * :meth:`record` — log one LLM call's usage.
    * :meth:`totals` — cumulative token counts + call count.
    * :meth:`per_backend` — breakdown by backend.
    * :meth:`recent` — bounded list of recent :class:`TokenUsageRecord`.
    * :meth:`reset` — clear all state.
    * :meth:`cost_estimate` — rough USD cost given per-backend token rates.
    """

    def __init__(self, max_records: int = 1000) -> None:
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        self._lock = _threading.Lock()
        self._records: _deque[TokenUsageRecord] = _deque(maxlen=max_records)
        self._max_records = max_records

    def record(
        self,
        backend: str,
        model: str,
        usage: dict[str, Any] | None,
        *,
        success: bool = True,
        streamed: bool = False,
    ) -> TokenUsageRecord | None:
        """Log one LLM call's token usage.

        ``usage`` is the OpenAI-compatible ``usage`` dict (``prompt_tokens``,
        ``completion_tokens``, ``total_tokens``). When ``usage`` is None or
        missing the expected keys, a zero-token record is still logged (so
        call counts stay accurate) unless ``success`` is False AND ``usage``
        is None — in which case a zero-token failed record is logged.

        Returns the created :class:`TokenUsageRecord` (or None when ``usage``
        is None and ``success`` is True — nothing to record on a usage-less
        success, e.g. a backend that doesn't report usage).
        """
        pt = ct = tt = 0
        if usage is not None:
            try:
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                tt = int(usage.get("total_tokens", pt + ct) or (pt + ct))
            except (TypeError, ValueError):
                pt = ct = tt = 0
        # Skip logging only when: success but no usage reported (nothing to
        # track). Failed calls + zero-usage calls are still logged for
        # call-count accuracy.
        if usage is None and success:
            return None
        rec = TokenUsageRecord(
            backend=backend,
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            success=success,
            streamed=streamed,
            timestamp=_time.time(),
        )
        with self._lock:
            self._records.append(rec)
        return rec

    def totals(self) -> dict[str, Any]:
        """Cumulative token usage across all recorded calls.

        Returns a dict with: ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``, ``calls`` (total records), ``successful_calls``,
        ``failed_calls``, ``streamed_calls``.
        """
        with self._lock:
            recs = list(self._records)
        if not recs:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "streamed_calls": 0,
            }
        return {
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "total_tokens": sum(r.total_tokens for r in recs),
            "calls": len(recs),
            "successful_calls": sum(1 for r in recs if r.success),
            "failed_calls": sum(1 for r in recs if not r.success),
            "streamed_calls": sum(1 for r in recs if r.streamed),
        }

    def per_backend(self) -> dict[str, dict[str, Any]]:
        """Breakdown of :meth:`totals` by backend name."""
        with self._lock:
            recs = list(self._records)
        out: dict[str, list[TokenUsageRecord]] = {}
        for r in recs:
            out.setdefault(r.backend, []).append(r)
        result: dict[str, dict[str, Any]] = {}
        for backend, rs in out.items():
            result[backend] = {
                "prompt_tokens": sum(r.prompt_tokens for r in rs),
                "completion_tokens": sum(r.completion_tokens for r in rs),
                "total_tokens": sum(r.total_tokens for r in rs),
                "calls": len(rs),
                "successful_calls": sum(1 for r in rs if r.success),
                "failed_calls": sum(1 for r in rs if not r.success),
                "streamed_calls": sum(1 for r in rs if r.streamed),
            }
        return result

    def recent(self, n: int = 50) -> list[TokenUsageRecord]:
        """Return the N most-recent records (newest last)."""
        if n < 1:
            return []
        with self._lock:
            recs = list(self._records)
        return recs[-n:]

    def reset(self) -> None:
        """Clear all recorded usage (for testing / session restart)."""
        with self._lock:
            self._records.clear()

    def cost_estimate(
        self,
        rates: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Rough USD cost estimate per backend.

        Args:
            rates: ``{backend: (prompt_per_1m_usd, completion_per_1m_usd)}``.
                When None, uses conservative public list prices for known
                backends (``openai`` gpt-4o-mini: $0.15/$0.60 per 1M tokens).

        Returns:
            ``{backend: usd}`` dict. Unknown backends use the openai rate.
        """
        if rates is None:
            rates = {
                "openai": (0.15, 0.60),
                "local": (0.0, 0.0),
            }
        default_rate = rates.get("openai", (0.15, 0.60))
        per_b = self.per_backend()
        out: dict[str, float] = {}
        for backend, agg in per_b.items():
            p_rate, c_rate = rates.get(backend, default_rate)
            out[backend] = (
                agg["prompt_tokens"] * p_rate / 1_000_000.0
                + agg["completion_tokens"] * c_rate / 1_000_000.0
            )
        return out


_default_token_tracker: TokenUsageTracker | None = None


# --------------------------------------------------------------------------- #
# Iter-142: Feedback memory — ring buffer of recent feedback snapshots
# --------------------------------------------------------------------------- #
@_dataclass
class FeedbackMemoryEntry:
    """A single feedback snapshot stored in the memory ring buffer (Iter-142).

    Captures the key outputs of one feedback generation cycle so the LLM can
    reference past feedback when answering follow-up questions.
    """

    summary: str
    lap_time: float | None
    track_id: str
    dimension_names: list[str]
    dimension_values: dict[str, str]
    setup_changes: list[str]
    timestamp: float


class FeedbackMemory:
    """Bounded ring buffer of recent feedback results (Iter-142).

    Stores up to ``max_entries`` feedback snapshots. When the LLM enhance
    function is called, the memory is queried for recent context and injected
    into the prompt so the model can answer follow-up questions like:

    - "How did the last change affect braking?"
    - "Compare the current balance with the previous lap."
    - "What was the trend over the last 3 laps?"

    Thread-safe via an internal ``threading.Lock`` — safe to share across
    sync and async feedback paths.
    """

    def __init__(self, max_entries: int = 20) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._lock = _threading.Lock()
        self._entries: _deque[FeedbackMemoryEntry] = _deque(maxlen=max_entries)
        self._max_entries = max_entries

    def add(
        self,
        summary: str,
        lap_time: float | None,
        track_id: str,
        dimension_names: list[str],
        dimension_values: dict[str, str],
        setup_changes: list[str],
    ) -> FeedbackMemoryEntry:
        """Store a feedback snapshot.

        Args:
            summary: The feedback summary text.
            lap_time: The lap time this feedback was generated for (seconds).
            track_id: Track identifier (e.g. "melbourne").
            dimension_names: Ordered list of dimension names (10 entries).
            dimension_values: ``{name: value}`` for each dimension.
            setup_changes: List of ``"name: before -> after"`` strings.

        Returns:
            The stored :class:`FeedbackMemoryEntry`.
        """
        entry = FeedbackMemoryEntry(
            summary=summary,
            lap_time=lap_time,
            track_id=track_id,
            dimension_names=list(dimension_names),
            dimension_values=dict(dimension_values),
            setup_changes=list(setup_changes),
            timestamp=_time.time(),
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def recent(self, n: int = 5) -> list[FeedbackMemoryEntry]:
        """Return the N most-recent entries (newest last)."""
        if n < 1:
            return []
        with self._lock:
            return list(self._entries)[-n:]

    def context_for_prompt(self, n: int = 5) -> str:
        """Generate a context string for injection into the LLM prompt.

        Includes:
        - Lap time trend (last N laps).
        - Setup changes applied in recent feedback.
        - Key dimension flags that were raised.

        Args:
            n: Number of recent entries to include.

        Returns:
            A multi-line string suitable for appending to the LLM user prompt.
        """
        entries = self.recent(n)
        if not entries:
            return ""

        parts: list[str] = ["# Recent Feedback History"]

        # Lap time trend.
        lts = [e.lap_time for e in entries if e.lap_time is not None]
        if len(lts) >= 2:
            trend = "improving" if lts[-1] < lts[0] else "slowing"
            parts.append(
                f"Lap time trend: {lts[0]:.3f}s -> {lts[-1]:.3f}s ({trend})"
            )
        elif lts:
            parts.append(f"Last lap time: {lts[-1]:.3f}s")

        # Setup changes.
        all_changes: list[str] = []
        seen: set[str] = set()
        for e in reversed(entries):
            for ch in e.setup_changes:
                if ch not in seen:
                    all_changes.append(ch)
                    seen.add(ch)
        if all_changes:
            parts.append("Recent setup changes: " + "; ".join(all_changes[:10]))

        # Key dimension flags.
        flagged: dict[str, str] = {}
        for e in entries:
            for name, val in e.dimension_values.items():
                if val and val not in ("neutral", "normal", "ok", "good", ""):
                    flagged[name] = val
        if flagged:
            items = [f"{k}: {v}" for k, v in list(flagged.items())[:8]]
            parts.append("Flagged dimensions: " + "; ".join(items))

        return "\n".join(parts)

    def reset(self) -> None:
        """Clear all stored entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def get_default_token_tracker() -> TokenUsageTracker:
    """Lazy module-level singleton tracker (created on first call)."""
    global _default_token_tracker
    if _default_token_tracker is None:
        _default_token_tracker = TokenUsageTracker()
    return _default_token_tracker


def _extract_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the OpenAI-compatible ``usage`` dict from a non-streaming response.

    Returns None when absent (some backends omit usage).
    """
    usage = data.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


def _extract_usage_from_stream_chunk(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Pull ``usage`` from a streaming SSE chunk (final chunk, empty choices).

    When ``stream_options.include_usage`` is set, the API emits a final chunk
    with ``choices: []`` and a populated ``usage`` field. Returns None for
    normal content chunks (which have non-empty choices and no usage).
    """
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


# Lockup detection thresholds.
_BRAKE_HIGH_THRESHOLD = 0.5      # brake input above this = sustained braking.
_LOCKUP_SPEED_DROP_KMH = 0.5     # <0.5 km/h drop per frame @ high brake = lockup.

# Confidence normalisation references.
_CORRECTION_FREQ_REF_HZ = 2.0    # 2 Hz of steer sign-changes = max micromanagement.
_G_LAT_STABILITY_REF = 3.0       # 3G std = fully unstable lateral load.

# Iter-2v2: cache for _dim_setup_advice to avoid re-running search_setup
# (1.4s scipy DE) on every feedback call with the same track+setup+profile.
_SETUP_ADVICE_CACHE: dict[tuple, dict] = {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _num(frame: dict[str, Any], key: str) -> float | None:
    """Read a numeric field from a unified frame, ``None`` if absent/invalid."""
    v = frame.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _resolve_track(track_id: str) -> Track | None:
    """Resolve a track_id, returning ``None`` for unknown ids (no crash)."""
    try:
        return get_track(track_id)
    except ValueError:
        return None


def _format_ref(ref: dict[str, Any]) -> str:
    """Format an evidence reference as ``field=value at t=frame_t``."""
    return f"{ref['field']}={ref['value']:.2f} at t={ref['frame_t']:.2f}s"


def _ref_lap_for(track: Track | None) -> float:
    """Reference lap time (s) for a track, scaled by length vs a 5 km circuit."""
    track_length = track.length_m if track else 5000.0
    ref_base = _REF_LAP_BY_TYPE.get(track.track_type, 90.0) if track else 90.0
    return ref_base * (track_length / 5000.0)


def _data_insufficient(name: str, advice: str | None = None) -> dict[str, Any]:
    """Build a dimension entry for the no-data case."""
    return {"name": name, "value": "数据不足", "evidence": "", "advice": advice}


# --------------------------------------------------------------------------- #
# Driver-profile personalisation helpers (Iter-05)
# --------------------------------------------------------------------------- #
def _normalize_driver_profile(
    driver_profile: DriverProfile | dict[str, Any] | list[float] | None,
) -> DriverProfile | None:
    """Coerce a flexible profile input into a :class:`DriverProfile` (or None).

    Accepts a :class:`DriverProfile` (passthrough), a field-keyed ``dict``
    (constructed via ``DriverProfile(**dict)``), an 8-element ``list``/
    ``tuple`` (via :meth:`DriverProfile.from_vector`), or ``None``. Invalid
    inputs fall back to ``None`` so the pipeline never crashes on a bad
    profile — personalisation is best-effort.
    """
    if driver_profile is None:
        return None
    if isinstance(driver_profile, DriverProfile):
        return driver_profile
    if isinstance(driver_profile, dict):
        try:
            return DriverProfile(**driver_profile)
        except Exception:
            return None
    if isinstance(driver_profile, (list, tuple)):
        try:
            return DriverProfile.from_vector(list(driver_profile))
        except Exception:
            return None
    return None


def _profile_style_label(profile: DriverProfile | None) -> str:
    """Compact style tag for evidence injection, e.g. ``AGGRESSIVE(aggression=0.90)``."""
    if profile is None:
        return "NONE"
    aggr = float(profile.aggression_score)
    if aggr >= 0.6:
        style = "AGGRESSIVE"
    elif aggr <= 0.4:
        style = "CONSERVATIVE"
    else:
        style = "BALANCED"
    return f"{style}(aggression={aggr:.2f})"


def _braking_personal(profile: DriverProfile) -> str:
    """Personalised braking advice clause keyed on ``aggression_score``."""
    aggr = float(profile.aggression_score)
    if aggr >= 0.6:
        return (
            f"车手画像偏激进(aggression={aggr:.2f})：保持晚刹车但注意锁死风险。"
        )
    if aggr <= 0.4:
        return (
            f"车手画像偏保守(aggression={aggr:.2f})：可尝试稍晚刹车提升圈速，"
            "当前刹车点偏保守。"
        )
    return f"车手画像中性(aggression={aggr:.2f})：维持当前刹车节奏。"


def _balance_personal(profile: DriverProfile) -> str:
    """Personalised balance advice clause keyed on ``corner_balance_pref``."""
    cbp = float(profile.corner_balance_pref)
    if cbp >= 0.6:
        return (
            f"车手偏好转向过度(corner_balance_pref={cbp:.2f})：当前平衡偏推头，"
            "可适度加重后部以释放车尾。"
        )
    if cbp <= 0.4:
        return (
            f"车手偏好转向不足(corner_balance_pref={cbp:.2f})：当前平衡若偏推头"
            "则与驾驶习惯一致，注意入弯信心。"
        )
    return f"车手平衡偏好中性(corner_balance_pref={cbp:.2f})：保持当前平衡基调。"


def _confidence_personal(profile: DriverProfile) -> str:
    """Personalised confidence advice clause keyed on ``consistency_score``."""
    cons = float(profile.consistency_score)
    if cons <= 0.4:
        return (
            f"车手一致性偏低(consistency={cons:.2f})：圈速波动大，"
            "建议先稳住节奏再推极限。"
        )
    if cons >= 0.6:
        return (
            f"车手一致性较高(consistency={cons:.2f})：节奏稳定，"
            "可尝试更激进的入弯。"
        )
    return f"车手一致性中性(consistency={cons:.2f})：保持当前节奏逐步推进。"


def _smoothness_personal(profile: DriverProfile) -> str:
    """Personalised smoothness advice clause keyed on ``throttle_smoothness``."""
    ts = float(profile.throttle_smoothness)
    if ts <= 0.4:
        return (
            f"车手油门渐进度不足(throttle_smoothness={ts:.2f})："
            "出弯牵引力损失，建议更线性给油。"
        )
    if ts >= 0.6:
        return (
            f"车手油门较平顺(throttle_smoothness={ts:.2f})：维持线性给油节奏。"
        )
    return f"车手油门平顺度中性(throttle_smoothness={ts:.2f})：可进一步细化出弯给油。"


def _apply_personal_advice(
    dim: dict[str, Any],
    profile: DriverProfile | None,
    suffix_fn: Any,
) -> dict[str, Any]:
    """Append a profile-driven advice clause to a dimension entry.

    Skips ``数据不足`` entries (no data to personalise) and ``None`` profiles
    (no personalisation requested). The objective ``value`` / ``evidence``
    fields are never modified — only the advisory ``advice`` text is extended
    so the personalised guidance is observable without altering the facts.
    """
    if profile is None:
        return dim
    if dim.get("value") == "数据不足":
        return dim
    suffix = suffix_fn(profile)
    if not suffix:
        return dim
    base = dim.get("advice")
    new_advice = suffix if not base else f"{base} {suffix}"
    return {**dim, "advice": new_advice}


# --------------------------------------------------------------------------- #
# 1. Metric extraction
# --------------------------------------------------------------------------- #
def extract_metrics(
    frames: list[dict[str, Any]], setup: dict[str, Any], track_id: str
) -> dict[str, Any]:
    """Compute aggregate telemetry metrics WITH evidence.

    Returns a dict with:

    - ``values``: computed metric scalars / dicts.
    - ``refs``: per-metric evidence references ``{frame_t, field, value}``.
    - ``sources``: flat deduped list of evidence entries (the contract's
      ``sources`` field).
    - ``n_frames``, ``track_id``, ``setup``: passthroughs.

    Iter-03 additions: ``lockup_proxy`` / ``brake_bias_setup``,
    ``steering_correction_freq`` / ``g_lat_stability`` / ``confidence_score``,
    ``max_g_lat``, ``sector_times``.
    """
    metrics: dict[str, Any] = {
        "n_frames": len(frames),
        "track_id": track_id,
        "setup": setup,
        "values": {},
        "refs": {},
        "sources": [],
    }
    if not frames:
        return metrics

    seen: set[tuple[float, str]] = set()

    def add_ref(key: str, frame_t: float, field: str, value: float) -> None:
        ref = {"frame_t": float(frame_t), "field": field, "value": float(value)}
        metrics["refs"][key] = ref
        dedup = (round(float(frame_t), 6), field)
        if dedup not in seen:
            seen.add(dedup)
            metrics["sources"].append(ref)

    # Iter-229: batch extraction of multiple fields in a single pass.
    def col_multi(*fields: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Extract multiple fields from frames in one pass (O(n) instead of O(n*k))."""
        result: dict[str, tuple[list[float], list[float]]] = {
            f: ([], []) for f in fields
        }
        for frm in frames:
            ts_val = float(frm.get("session_time") or 0.0)
            for field in fields:
                v = _num(frm, field)
                if v is not None:
                    result[field][0].append(v)
                    result[field][1].append(ts_val)
        return {
            f: (np.asarray(vals, dtype=np.float64), np.asarray(ts, dtype=np.float64))
            for f, (vals, ts) in result.items()
        }

    cols = col_multi("speed", "throttle", "brake", "steer", "g_lat", "ers_store", "ers_deployed", "ers_harvested", "ers_deploy_mode", "ers_mgu_k_deploy", "drs_allowed", "drs_active", "drs_zone", "lap_time", "lap_distance", "fuel_in_tank", "brake_temp_fl", "brake_temp_rl", "brake_temp_fr", "brake_temp_rr", "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr", "tyre_wear_fl", "tyre_wear_fr", "tyre_wear_rl", "tyre_wear_rr", "tyre_inner_temp_fl", "tyre_inner_temp_fr", "tyre_outer_temp_fl", "tyre_outer_temp_fr", "active_aero_x", "active_aero_z")

    # --- Speed ---
    speed, speed_t = cols["speed"]
    if len(speed):
        avg_speed = float(np.mean(speed))
        max_idx = int(np.argmax(speed))
        max_speed = float(speed[max_idx])
        max_speed_t = float(speed_t[max_idx])
        metrics["values"]["avg_speed"] = avg_speed
        metrics["values"]["max_speed"] = max_speed
        metrics["values"]["max_speed_t"] = max_speed_t
        add_ref("max_speed", max_speed_t, "speed", max_speed)
        mid_idx = int(np.argmin(np.abs(speed - avg_speed)))
        add_ref("avg_speed", float(speed_t[mid_idx]), "speed", float(speed[mid_idx]))

    # --- Throttle smoothness: 1 - std(throttle gradient) ---
    throttle, throttle_t = cols["throttle"]
    if len(throttle) >= 2:
        std_grad = float(np.std(np.diff(throttle)))
        smoothness = max(0.0, min(1.0, 1.0 - std_grad))
        metrics["values"]["throttle_smoothness"] = smoothness
        mid = len(throttle) // 2
        add_ref(
            "throttle_smoothness",
            float(throttle_t[mid]),
            "throttle",
            float(throttle[mid]),
        )

    # --- Brake aggression: max positive brake gradient ---
    brake, brake_t = cols["brake"]
    if len(brake) >= 2:
        grad = np.diff(brake)
        max_grad = float(np.max(grad))
        gi = int(np.argmax(grad))
        metrics["values"]["brake_aggression"] = max_grad
        idx = min(gi + 1, len(brake_t) - 1)
        add_ref("brake_aggression", float(brake_t[idx]), "brake", float(brake[idx]))

    # --- Braking: lockup detection (NEW in Iter-03) ---
    # A lockup proxy: among sustained high-brake frames (current AND next frame
    # both brake > threshold), fraction where speed barely drops (< threshold
    # km/h per frame). High brake + near-zero deceleration => wheels locked /
    # car not slowing proportionally. brake_bias assessment is setup-driven.
    if len(brake) >= 2 and len(speed) >= 2:
        n = min(len(brake), len(speed))
        b = brake[:n]
        sp = speed[:n]
        bt = brake_t[:n]
        high_brake = b > _BRAKE_HIGH_THRESHOLD
        sustained = high_brake[:-1] & high_brake[1:]
        if sustained.any():
            hb_idx = np.where(sustained)[0]
            sp_diff = np.diff(sp)
            drops = sp_diff[hb_idx]
            lockup_frames = drops > -_LOCKUP_SPEED_DROP_KMH
            lockup_proxy = float(np.mean(lockup_frames))
            metrics["values"]["lockup_proxy"] = lockup_proxy
            metrics["values"]["lockup_frame_count"] = int(np.sum(lockup_frames))
            metrics["values"]["lockup_total_brake_frames"] = int(len(hb_idx))
            rep = int(hb_idx[0])
            add_ref("lockup_brake", float(bt[rep]), "brake", float(b[rep]))
            add_ref("lockup_speed", float(bt[rep]), "speed", float(sp[rep]))
    bb = setup.get("front_brake_bias")
    if bb is not None:
        try:
            metrics["values"]["brake_bias_setup"] = float(bb)
        except (TypeError, ValueError):
            pass

    # --- Understeer indicator: low g_lat relative to |steer| mid-corner ---
    steer, steer_t = cols["steer"]
    g_lat, g_lat_t = cols["g_lat"]
    if len(steer) and len(g_lat):
        n = min(len(steer), len(g_lat))
        s = steer[:n]
        g = g_lat[:n]
        t = steer_t[:n]
        mask = np.abs(s) > 0.3
        if mask.any():
            s_c = np.abs(s[mask])
            g_c = np.abs(g[mask])
            t_c = t[mask]
            # ratio: g_lat per unit steer; ~5G at full lock = neutral/good.
            ratio = float(np.mean(g_c) / (np.mean(s_c) + 1e-6))
            understeer = max(0.0, min(1.0, 1.0 - ratio / 5.0))
            metrics["values"]["understeer_indicator"] = understeer
            metrics["values"]["corner_g_lat_per_steer"] = ratio
            wi = int(np.argmax(s_c))
            wt = float(t_c[wi])
            add_ref("understeer_steer", wt, "steer", float(s_c[wi]))
            add_ref("understeer_g_lat", wt, "g_lat", float(g_c[wi]))

    # --- Confidence: steering correction freq + g_lat stability (NEW Iter-03) ---
    # steering_correction_freq = sign changes in steer first-difference / time
    # span (more reversals => more micromanagement). g_lat_stability =
    # 1 - std(g_lat)/ref. confidence = 1 - correction_norm * (1 - stability).
    if len(steer) >= 3 and len(steer_t) >= 2:
        steer_diff = np.diff(steer)
        signs = np.sign(steer_diff)
        sign_changes = 0
        prev_sign = 0
        for sg in signs:
            si = int(sg)
            if si != 0:
                if prev_sign != 0 and si != prev_sign:
                    sign_changes += 1
                prev_sign = si
        time_span = float(steer_t[-1] - steer_t[0])
        corr_freq = sign_changes / time_span if time_span > 0 else 0.0
        metrics["values"]["steering_correction_freq"] = float(corr_freq)
        metrics["values"]["steering_correction_count"] = int(sign_changes)
        mid = len(steer) // 2
        add_ref(
            "steering_correction",
            float(steer_t[mid]),
            "steer",
            float(steer[mid]),
        )
    if len(g_lat) >= 2:
        g_lat_std = float(np.std(g_lat))
        g_lat_stability = max(0.0, min(1.0, 1.0 - g_lat_std / _G_LAT_STABILITY_REF))
        metrics["values"]["g_lat_stability"] = g_lat_stability
        metrics["values"]["g_lat_std"] = g_lat_std
        mid = len(g_lat) // 2
        add_ref(
            "g_lat_stability",
            float(g_lat_t[mid]),
            "g_lat",
            float(g_lat[mid]),
        )
    corr_freq = metrics["values"].get("steering_correction_freq")
    stability = metrics["values"].get("g_lat_stability")
    if corr_freq is not None and stability is not None:
        corr_norm = max(0.0, min(1.0, corr_freq / _CORRECTION_FREQ_REF_HZ))
        confidence = max(0.0, min(1.0, 1.0 - corr_norm * (1.0 - stability)))
        metrics["values"]["confidence_score"] = confidence

    # --- Oversteer indicator: rear tyre wear imbalance vs front ---
    wear_fields = ("tyre_wear_fl", "tyre_wear_fr", "tyre_wear_rl", "tyre_wear_rr")
    wears: list[float] = []
    for k in wear_fields:
        w, _ = cols[k]
        if len(w):
            wears.append(float(w[-1]))
    if len(wears) == 4:
        front_avg = (wears[0] + wears[1]) / 2.0
        rear_avg = (wears[2] + wears[3]) / 2.0
        oversteer = max(0.0, min(1.0, (rear_avg - front_avg) / 20.0))
        metrics["values"]["oversteer_indicator"] = float(oversteer)
        metrics["values"]["tyre_wear_balance"] = {
            "fl": wears[0],
            "fr": wears[1],
            "rl": wears[2],
            "rr": wears[3],
            "front_avg": front_avg,
            "rear_avg": rear_avg,
        }
        last_t = float(frames[-1].get("session_time") or 0.0)
        add_ref("oversteer_wear_rl", last_t, "tyre_wear_rl", float(wears[2]))
        add_ref("oversteer_wear_rr", last_t, "tyre_wear_rr", float(wears[3]))

    # --- Tyre temp spread ---
    temp_fields = ("tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr")
    temps: list[float] = []
    for k in temp_fields:
        t_arr, _ = cols[k]
        if len(t_arr):
            temps.append(float(np.mean(t_arr)))
    if len(temps) == 4:
        spread = float(max(temps) - min(temps))
        metrics["values"]["tyre_temp_spread"] = spread
        metrics["values"]["tyre_temps_avg"] = list(temps)
        mid_t = float(frames[len(frames) // 2].get("session_time") or 0.0)
        add_ref("tyre_temp_fl", mid_t, "tyre_temp_fl", float(temps[0]))
        add_ref("tyre_temp_fr", mid_t, "tyre_temp_fr", float(temps[1]))
        add_ref("tyre_temp_rl", mid_t, "tyre_temp_rl", float(temps[2]))
        add_ref("tyre_temp_rr", mid_t, "tyre_temp_rr", float(temps[3]))

    # --- Tyre temp gradient (inner/outer spread) — Iter-198 ---
    # Detects camber misalignment: if inner tyre temp is much hotter than
    # outer, the camber is too negative (excessive inner edge wear).
    inner_temp_fields = ("tyre_inner_temp_fl", "tyre_inner_temp_fr")
    outer_temp_fields = ("tyre_outer_temp_fl", "tyre_outer_temp_fr")
    inner_temps: list[float] = []
    outer_temps: list[float] = []
    for k in inner_temp_fields:
        t_arr, _ = cols[k]
        if len(t_arr):
            inner_temps.append(float(np.mean(t_arr)))
    for k in outer_temp_fields:
        t_arr, _ = cols[k]
        if len(t_arr):
            outer_temps.append(float(np.mean(t_arr)))
    if len(inner_temps) == 2 and len(outer_temps) == 2:
        # Average inner vs outer across FL+FR.
        avg_inner = sum(inner_temps) / 2.0
        avg_outer = sum(outer_temps) / 2.0
        temp_gradient = avg_inner - avg_outer
        metrics["values"]["tyre_temp_gradient"] = temp_gradient
        mid_t = float(frames[len(frames) // 2].get("session_time") or 0.0)
        add_ref("tyre_inner_temp_fl", mid_t, "tyre_inner_temp_fl", float(inner_temps[0]))
        add_ref("tyre_outer_temp_fl", mid_t, "tyre_outer_temp_fl", float(outer_temps[0]))

    # --- Aero balance (downforce) — Iter-214 ---
    # Compares g_lat at high speed (>250 km/h) vs low speed (<150 km/h) while
    # cornering to determine if the car is aero-dominant or mechanical-grip-dominant.
    if len(g_lat) >= 2 and len(speed) >= 2 and len(steer) >= 2:
        n_aero = min(len(g_lat), len(speed), len(steer))
        g_a = g_lat[:n_aero]
        sp_a = speed[:n_aero]
        st_a = steer[:n_aero]
        corner_mask = np.abs(st_a) > 0.3
        if corner_mask.any():
            high_mask = corner_mask & (sp_a > 250.0)
            low_mask = corner_mask & (sp_a < 150.0)
            high_g = float(np.mean(np.abs(g_a[high_mask]))) if high_mask.any() else 0.0
            low_g = float(np.mean(np.abs(g_a[low_mask]))) if low_mask.any() else 0.0
            aero_ratio = high_g / low_g if low_g > 0 else 0.0
            metrics["values"]["high_speed_g_lat_avg"] = high_g
            metrics["values"]["low_speed_g_lat_avg"] = low_g
            metrics["values"]["aero_balance_ratio"] = aero_ratio
            if aero_ratio > 1.8:
                diag = "aero-dominant: strong downforce, may need more mechanical grip"
            elif aero_ratio < 1.1:
                diag = "mechanical-dominant: good mechanical grip, may need more downforce"
            else:
                diag = "balanced aero/mechanical grip"
            metrics["values"]["aero_balance_diagnosis"] = diag
            mid_ab = len(g_a) // 2
            add_ref("aero_balance_high_g", float(g_lat_t[min(mid_ab, len(g_lat_t)-1)]), "g_lat", high_g)
            add_ref("aero_balance_low_g", float(g_lat_t[min(mid_ab, len(g_lat_t)-1)]), "g_lat", low_g)

    # --- Active aero usage (F1 2026 X-Mode / Z-Mode) — Iter-256 ---
    # X-Mode = 低阻直道 (m_activeAeroX), Z-Mode = 高下压弯道 (m_activeAeroZ).
    # 以 position > 0.5 判定该模式激活, 计算各自占帧比 (duty cycle).
    aero_x, aero_x_t = cols["active_aero_x"]
    aero_z, aero_z_t = cols["active_aero_z"]
    if len(aero_x) or len(aero_z):
        x_frac = (
            float(np.count_nonzero(np.asarray(aero_x) > 0.5)) / len(aero_x)
            if len(aero_x) else 0.0
        )
        z_frac = (
            float(np.count_nonzero(np.asarray(aero_z) > 0.5)) / len(aero_z)
            if len(aero_z) else 0.0
        )
        metrics["values"]["active_aero_x_fraction"] = x_frac
        metrics["values"]["active_aero_z_fraction"] = z_frac
        metrics["values"]["active_aero_mean_x"] = float(np.mean(aero_x)) if len(aero_x) else 0.0
        metrics["values"]["active_aero_mean_z"] = float(np.mean(aero_z)) if len(aero_z) else 0.0
        if len(aero_x):
            xi = int(np.argmax(aero_x))
            add_ref("active_aero_x", float(aero_x_t[xi]), "active_aero_x", float(aero_x[xi]))
        if len(aero_z):
            zi = int(np.argmax(aero_z))
            add_ref("active_aero_z", float(aero_z_t[zi]), "active_aero_z", float(aero_z[zi]))

    # --- Max g_lat (grip proxy) ---
    if len(g_lat):
        abs_g = np.abs(g_lat)
        max_glat = float(np.max(abs_g))
        mi = int(np.argmax(abs_g))
        metrics["values"]["max_g_lat"] = max_glat
        add_ref("max_g_lat", float(g_lat_t[mi]), "g_lat", float(g_lat[mi]))

    # --- ers_deployment: ERS store level, harvest/deploy totals, mode efficiency ---
    ers, ers_t = cols["ers_store"]
    if len(ers) >= 2:
        ers_store_mean_val = float(np.mean(ers))
        metrics["values"]["ers_store_mean"] = ers_store_mean_val
        add_ref("ers_store_mean", float(ers_t[len(ers_t)//2]), "ers_store", ers_store_mean_val)
        xm = ers_t - ers_t.mean()
        denom = float(np.dot(xm, xm))
        slope = float(np.dot(xm, ers - ers.mean()) / denom) if denom > 0 else 0.0
        metrics["values"]["ers_slope_per_s"] = slope
        add_ref("ers_start", float(ers_t[0]), "ers_store", float(ers[0]))
        add_ref("ers_end", float(ers_t[-1]), "ers_store", float(ers[-1]))
    ers_deploy, deploy_t = cols["ers_deployed"]
    deployed_total = None
    if len(ers_deploy) >= 2:
        deployed_total = float(ers_deploy[-1] - ers_deploy[0])
        metrics["values"]["ers_deployed_total"] = deployed_total
        add_ref("ers_deployed_total", float(deploy_t[-1]), "ers_deployed", float(ers_deploy[-1]))
    ers_harv, harv_t = cols["ers_harvested"]
    if len(ers_harv) >= 2:
        harvested_total = float(ers_harv[-1] - ers_harv[0])
        metrics["values"]["ers_harvested_total"] = harvested_total
        add_ref("ers_harvested_total", float(harv_t[-1]), "ers_harvested", float(ers_harv[-1]))
    deploy_mode, mode_t = cols["ers_deploy_mode"]
    if len(deploy_mode) >= 2:
        hotlap_mask = deploy_mode == 1.0
        hotlap_pct = float(np.mean(hotlap_mask)) * 100.0
        metrics["values"]["deploy_mode_hotlap_pct"] = hotlap_pct
        add_ref("deploy_mode_hotlap_pct", float(mode_t[0]), "ers_deploy_mode", float(hotlap_pct))
    if deployed_total is not None:
        ers_consum, consum_t = cols["ers_mgu_k_deploy"]
        if len(ers_consum) >= 2:
            total_consumed = float(ers_consum[-1] - ers_consum[0])
            if total_consumed > 0:
                metrics["values"]["ers_efficiency"] = deployed_total / total_consumed

    # --- drs_usage: DRS activation timing, zone utilisation ---
    drs, drs_t = cols["drs_allowed"]
    if len(drs) >= 2:
        drs_int = np.round(drs).astype(int)
        diffs = np.diff(drs_int)
        transitions = int(np.sum(diffs > 0))
        metrics["values"]["drs_activation_count"] = transitions
        if transitions > 0:
            idx = int(np.where(diffs > 0)[0][0])
            add_ref("drs_first_activation", float(drs_t[idx + 1]), "drs_allowed", 1.0)
    drs_active, drs_at = cols["drs_active"]
    if len(drs_active) >= 2:
        active_mask = drs_active >= 0.5
        active_pct = float(np.mean(active_mask)) * 100.0
        metrics["values"]["drs_active_pct"] = active_pct
        add_ref("drs_active_pct", float(drs_at[0]), "drs_active", active_pct)
    drs_zone, zone_t = cols["drs_zone"]
    if len(drs_zone) >= 1:
        zone_ids = np.unique(np.round(drs_zone).astype(int))
        zone_ids = zone_ids[zone_ids > 0]
        metrics["values"]["drs_zone_count"] = len(zone_ids)
        add_ref("drs_zone_count", float(zone_t[0]), "drs_zone", float(len(zone_ids)))
    if len(drs) >= 2 and len(drs_active) >= 2:
        d_changes = np.diff(np.round(drs).astype(int))
        d_up = np.where(d_changes > 0)[0] + 1
        a_changes = np.diff(np.round(drs_active).astype(int))
        a_up = np.where(a_changes > 0)[0] + 1
        delays: list[float] = []
        for d_idx in d_up:
            d_t_val = float(drs_t[d_idx])
            later = a_up[a_up >= d_idx]
            if len(later) > 0:
                a_t_val = float(drs_at[later[0]])
                delays.append(a_t_val - d_t_val)
        if delays:
            metrics["values"]["drs_activation_delay_mean"] = float(np.mean(delays))

    # --- Lap time (from last frame) ---
    lap_times, lap_t = cols["lap_time"]
    if len(lap_times):
        last_lap = float(lap_times[-1])
        metrics["values"]["lap_time"] = last_lap
        add_ref("lap_time", float(lap_t[-1]), "lap_time", last_lap)

    # --- Sector compare: derive sector times from lap_distance crossings ---
    # (NEW Iter-03). If lap_distance spans >= 2/3 of track_length, interpolate
    # lap_time at L/3 and 2L/3 boundaries to get sector splits. Otherwise the
    # sector_compare dimension falls back to the nominal track_type split.
    lap_dist, _ = cols["lap_distance"]
    track = _resolve_track(track_id)
    track_length = track.length_m if track else 5000.0
    if (
        len(lap_dist) >= 2
        and len(lap_times) >= 2
        and track_length > 0
    ):
        order = np.argsort(lap_dist)
        ld_sorted = lap_dist[order]
        lt_sorted = lap_times[order]
        span = float(ld_sorted[-1] - ld_sorted[0])
        if span >= 2.0 * track_length / 3.0:
            s1_bound = track_length / 3.0
            s2_bound = 2.0 * track_length / 3.0
            t1 = float(np.interp(s1_bound, ld_sorted, lt_sorted))
            t2 = float(np.interp(s2_bound, ld_sorted, lt_sorted))
            t_start = float(lt_sorted[0])
            t_end = float(lt_sorted[-1])
            s1 = t1 - t_start
            s2 = t2 - t1
            s3 = t_end - t2
            if s1 > 0 and s2 > 0 and s3 > 0:
                metrics["values"]["sector_times"] = [s1, s2, s3]
                ref_t = float(lap_t[0]) if len(lap_t) else 0.0
                add_ref("sector_s1", ref_t, "lap_time", s1)
                add_ref("sector_s2", ref_t, "lap_time", s2)
                add_ref("sector_s3", ref_t, "lap_time", s3)

    # --- Fuel used ---
    fuel, fuel_t = cols["fuel_in_tank"]
    if len(fuel) >= 2:
        fuel_used = max(0.0, float(fuel[0] - fuel[-1]))
        metrics["values"]["fuel_used"] = fuel_used
        add_ref("fuel_start", float(fuel_t[0]), "fuel_in_tank", float(fuel[0]))
        add_ref("fuel_end", float(fuel_t[-1]), "fuel_in_tank", float(fuel[-1]))
        # Iter-200: fuel consumption rate (kg/lap) and per-sector efficiency.
        lap_dist, _ = cols["lap_distance"]
        if len(lap_dist) >= 2:
            track_len = float(lap_dist[-1] - lap_dist[0])
            if track_len > 0:
                fuel_per_km = fuel_used / (track_len / 1000.0) if fuel_used > 0 else 0.0
                metrics["values"]["fuel_consumption_rate_kg_per_km"] = fuel_per_km
                metrics["values"]["fuel_consumption_rate_kg_per_lap"] = fuel_used

    # --- Fuel per sector (Iter-246) ---
    # Use TelemetryAnalytics fuel_per_sector_analysis to get sector-level
    # fuel consumption proxy data for richer fuel feedback.
    try:
        from f1opt.telemetry.analytics import TelemetryAnalytics as _TA
        ta = _TA(frames, track_length_m=track_len)
        sector_fuel = ta.fuel_per_sector_analysis()
        metrics["values"]["fuel_sector_data"] = sector_fuel
        metrics["values"]["fuel_total_proxy"] = sector_fuel.get("total_fuel_proxy", 0.0)
        metrics["values"]["fuel_proxy_per_km"] = sector_fuel.get("fuel_proxy_per_km", 0.0)
        highest = sector_fuel.get("highest_consumption_sector", -1)
        if highest >= 0:
            metrics["values"]["fuel_highest_sector"] = highest
    except Exception:
        pass  # Fallback gracefully if analytics unavailable.

    # --- Throttle/brake overlap (Iter-206) ---
    thr_arr, _ = cols["throttle"]
    brk_arr, _ = cols["brake"]
    if len(thr_arr) >= 2 and len(brk_arr) >= 2:
        n_overlap = min(len(thr_arr), len(brk_arr))
        overlap_mask = (thr_arr[:n_overlap] > 0.3) & (brk_arr[:n_overlap] > 0.3)
        overlap_count = int(np.sum(overlap_mask))
        metrics["values"]["throttle_brake_overlap_count"] = overlap_count
        metrics["values"]["throttle_brake_overlap_pct"] = float(overlap_count / n_overlap) if n_overlap > 0 else 0.0

    # --- Mechanical grip trend (Iter-217) ---
    # Slope of low-speed g_lat over lap_distance to detect grip degradation.
    if len(g_lat) >= 3 and len(speed) >= 3 and len(steer) >= 3:
        lap_dist_m, _ = cols["lap_distance"]
        if len(lap_dist_m) >= 3:
            n_m = min(len(g_lat), len(speed), len(steer), len(lap_dist_m))
            g_m = g_lat[:n_m]
            sp_m = speed[:n_m]
            st_m = steer[:n_m]
            ld_m = lap_dist_m[:n_m]
            corner_mask_m = np.abs(st_m) > 0.3
            low_mask_m = corner_mask_m & (sp_m < 150.0)
            if low_mask_m.sum() >= 3:
                g_low_m = np.abs(g_m[low_mask_m])
                d_low_m = ld_m[low_mask_m]
                x_mean_m = float(np.mean(d_low_m))
                y_mean_m = float(np.mean(g_low_m))
                num_m = float(np.sum((d_low_m - x_mean_m) * (g_low_m - y_mean_m)))
                den_m = float(np.sum((d_low_m - x_mean_m) ** 2))
                slope_m = num_m / den_m if den_m > 0 else 0.0
                if slope_m < -0.001:
                    metrics["values"]["mech_grip_trend"] = "decaying"
                elif slope_m > 0.001:
                    metrics["values"]["mech_grip_trend"] = "improving"
                else:
                    metrics["values"]["mech_grip_trend"] = "stable"
                metrics["values"]["mech_grip_slope"] = slope_m

    # --- Brake temperature balance (Iter-223) ---
    # Front/rear brake temp ratio to detect brake bias imbalance.
    btf_arr, _ = cols["brake_temp_fl"]
    btr_arr, _ = cols["brake_temp_rl"]
    if len(btf_arr) >= 2 and len(btr_arr) >= 2:
        btf_fr_arr, _ = cols["brake_temp_fr"]
        btr_rr_arr, _ = cols["brake_temp_rr"]
        n_bt = min(len(btf_arr), len(btf_fr_arr), len(btr_arr), len(btr_rr_arr))
        front_avg = float((np.mean(btf_arr[:n_bt]) + np.mean(btf_fr_arr[:n_bt])) / 2.0)
        rear_avg = float((np.mean(btr_arr[:n_bt]) + np.mean(btr_rr_arr[:n_bt])) / 2.0)
        if rear_avg > 0:
            bt_ratio = front_avg / rear_avg
            metrics["values"]["brake_temp_f_r_ratio"] = bt_ratio
            if bt_ratio > 1.3:
                metrics["values"]["brake_temp_balance_diag"] = "front-bias"
            elif bt_ratio < 0.7:
                metrics["values"]["brake_temp_balance_diag"] = "rear-bias"
            else:
                metrics["values"]["brake_temp_balance_diag"] = "balanced"

    # --- Tyre temperature gradient (Iter-227) ---
    ts_fl_arr, _ = cols["tyre_temp_fl"]
    ts_fr_arr, _ = cols["tyre_temp_fr"]
    ts_rl_arr, _ = cols["tyre_temp_rl"]
    ts_rr_arr, _ = cols["tyre_temp_rr"]
    if len(ts_fl_arr) >= 2 and len(ts_fr_arr) >= 2:
        n_tt = min(len(ts_fl_arr), len(ts_fr_arr), len(ts_rl_arr), len(ts_rr_arr))
        lr_front = float(np.mean(ts_fl_arr[:n_tt] - ts_fr_arr[:n_tt]))
        lr_rear = float(np.mean(ts_rl_arr[:n_tt] - ts_rr_arr[:n_tt]))
        lr_delta = (lr_front + lr_rear) / 2.0
        fr_left = float(np.mean(ts_fl_arr[:n_tt] - ts_rl_arr[:n_tt]))
        fr_right = float(np.mean(ts_fr_arr[:n_tt] - ts_rr_arr[:n_tt]))
        fr_delta = (fr_left + fr_right) / 2.0
        metrics["values"]["tyre_temp_left_right_delta"] = lr_delta
        metrics["values"]["tyre_temp_front_rear_delta"] = fr_delta
        diag_parts = []
        if abs(lr_delta) > 3.0:
            diag_parts.append(f"{'左' if lr_delta > 0 else '右'}侧偏热")
        if abs(fr_delta) > 5.0:
            diag_parts.append(f"{'前' if fr_delta > 0 else '后'}轴偏热")
        metrics["values"]["tyre_temp_gradient_diag"] = "; ".join(diag_parts) if diag_parts else "balanced"

    # --- Grip consistency (Iter-241) ---
    # g_lat standard deviation during cornering as a consistency measure.
    if len(g_lat) >= 3 and len(steer) >= 3:
        corner_mask_gc = np.abs(steer) > 0.3
        if corner_mask_gc.any():
            g_corner = np.abs(g_lat[corner_mask_gc])
            gc_std = float(np.std(g_corner))
            metrics["values"]["grip_consistency_overall_std"] = gc_std
            # Per-sector worst determination via lap_distance.
            lap_dist_gc, _ = cols["lap_distance"]
            if len(lap_dist_gc) >= 2:
                track_len = 5000.0  # default
                track_info = TRACKS_BY_ID.get(track_id)
                if track_info is not None:
                    track_len = track_info.length_m
                n_gc = min(len(g_lat), len(steer), len(lap_dist_gc))
                order_gc = np.argsort(lap_dist_gc[:n_gc])
                g_gc = np.abs(g_lat[:n_gc][order_gc])
                st_gc = steer[:n_gc][order_gc]
                ld_gc = lap_dist_gc[:n_gc][order_gc]
                cm_gc = np.abs(st_gc) > 0.3
                boundaries_gc = [track_len / 3.0, 2.0 * track_len / 3.0, track_len]
                worst_std = 0.0
                worst_sec = -1
                prev_gc = 0.0
                for si, bound in enumerate(boundaries_gc):
                    mask = (ld_gc >= prev_gc) & (ld_gc < bound + 1e-6)
                    prev_gc = bound
                    corner_in = mask & cm_gc
                    if corner_in.any():
                        s_std = float(np.std(g_gc[corner_in]))
                        if s_std > worst_std:
                            worst_std = s_std
                            worst_sec = si
                metrics["values"]["grip_consistency_worst_sector"] = worst_sec

    return metrics


# --------------------------------------------------------------------------- #
# 2. Rule-based feedback
# --------------------------------------------------------------------------- #
def _make_suggestion(
    setup: dict[str, Any],
    name: str,
    delta: float,
    *,
    reason: str,
    expected_gain: str,
) -> dict[str, Any]:
    """Build a setup suggestion, clamping ``after`` to a valid CarSetup step."""
    spec = SETUP_FIELDS[name]
    before = float(setup.get(name, spec.min))
    proposed = before + delta
    clamped = max(spec.min, min(spec.max, proposed))
    snapped = float(_snap_to_step(clamped, spec))
    return {
        "name": name,
        "before": before,
        "after": snapped,
        "unit": spec.unit,
        "expected_gain": expected_gain,
        "reason": reason,
    }


def _answer_question(
    question: str,
    metrics: dict[str, Any],
    track: Track | None,
    conversation: ConversationSession | None = None,
) -> str:
    """Produce a canned but evidence-grounded answer to the driver's question.

    Iter-07: when a ``conversation`` session is supplied and the question
    contains a demonstrative reference (刚才/那个/它/上面) AND the session
    already has prior history, the answer is prefixed with
    ``[引用上文] <previous user content>`` so follow-up questions can resolve
    their references against the dialogue. Without a conversation the
    behaviour is unchanged (backward compatible).
    """
    answer = _build_answer(question, metrics, track)
    if conversation is None:
        return answer
    recent = conversation.recent(5)
    if not recent:
        return answer
    if not any(w in question for w in ("刚才", "那个", "它", "上面")):
        return answer
    # Resolve the most recent PRIOR user turn (the question being referenced).
    # The current question is appended to history AFTER _answer_question runs,
    # so recent() here only contains earlier turns.
    last_user_content: str | None = None
    for t in reversed(recent):
        if t.get("role") == "user":
            last_user_content = t.get("content")
            break
    if not last_user_content:
        return answer
    snippet = (
        last_user_content
        if len(last_user_content) <= 40
        else last_user_content[:40]
    )
    return f"[引用上文] {snippet}\n{answer}"


def _build_answer(question: str, metrics: dict[str, Any], track: Track | None) -> str:
    """Build the canned evidence-grounded answer (pre-Iter-07 logic)."""
    values = metrics.get("values", {})
    refs = metrics.get("refs", {})
    q = question.lower()

    if "推头" in question or "understeer" in q or "推" in question:
        u = values.get("understeer_indicator")
        s_ref = refs.get("understeer_steer")
        g_ref = refs.get("understeer_g_lat")
        if u is not None and s_ref and g_ref:
            return (
                f"关于「{question}」：从遥测看，入弯阶段 {_format_ref(s_ref)} "
                f"但横向G力仅 {_format_ref(g_ref)}，转向输入大、实际横向加速度"
                f"不足——前轴出现推头（understeer indicator={u:.2f}）。建议增加"
                "前翼下压力或软化前防倾杆，让前轮更早建立抓地力。"
            )
        return (
            f"关于「{question}」：当前窗口未捕获到足够的弯中转向数据，无法定量"
            "定位推头来源；建议重放入弯段遥测。"
        )
    if "甩尾" in question or "oversteer" in q:
        o = values.get("oversteer_indicator")
        rl = refs.get("oversteer_wear_rl")
        if o is not None and rl:
            return (
                f"关于「{question}」：后轴磨损偏高（{_format_ref(rl)}），"
                f"oversteer indicator={o:.2f}，提示后轮在加速出弯时打滑。"
                "建议降低 on_throttle_diff 或软化后防倾杆。"
            )
        return f"关于「{question}」：未观察到明显甩尾证据。"
    if "锁死" in question or "lockup" in q or "制动" in question or "刹车" in question:
        lp = values.get("lockup_proxy")
        b_ref = refs.get("lockup_brake")
        if lp is not None and b_ref:
            return (
                f"关于「{question}」：制动段 {_format_ref(b_ref)}，lockup proxy="
                f"{lp:.2f}，{'前轮锁死风险偏高' if lp > 0.3 else '锁死风险可控'}。"
                "建议降低 brake_pressure 或后移 front_brake_bias。"
            )
        return f"关于「{question}」：当前窗口无制动锁死证据。"
    if "胎" in question or "tyre" in q or "tire" in q:
        twb = values.get("tyre_wear_balance")
        if twb:
            return (
                f"关于「{question}」：四轮磨损 FL={twb['fl']:.1f}%、"
                f"FR={twb['fr']:.1f}%、RL={twb['rl']:.1f}%、RR={twb['rr']:.1f}%，"
                f"后轴比前轴高 {twb['rear_avg'] - twb['front_avg']:.1f}%，"
                "建议适当提高后轮胎压或降低差速锁止率。"
            )
        return f"关于「{question}」：当前遥测窗口无胎损数据。"
    if "ers" in q or "电池" in question or "能量" in question:
        slope = values.get("ers_slope_per_s")
        if slope is not None:
            return (
                f"关于「{question}」：ERS 储能斜率 {slope:.2f}/s，若为负表示在"
                "主动部署；建议把部署集中在出弯与直道前段。"
            )
        return f"关于「{question}」：当前窗口无 ERS 数据。"
    if "drs" in q:
        c = values.get("drs_activation_count")
        if c is not None:
            return (
                f"关于「{question}」：本段共记录 {c} 次 DRS 激活；确保在每个检测"
                "区段都对准前车 1 秒以内。"
            )
        return f"关于「{question}」：当前窗口无 DRS 数据。"
    return (
        f"关于「{question}」：基于当前 {metrics.get('n_frames', 0)} 帧遥测，已"
        "生成下方各维度反馈与调教建议，可对照证据帧时间戳复盘。"
    )


def _general_summary(
    metrics: dict[str, Any],
    dimensions: list[dict[str, Any]],
    track: Track | None,
    ref_lap: float,
) -> str:
    n = metrics.get("n_frames", 0)
    values = metrics.get("values", {})
    if n == 0:
        return (
            "数据不足：当前无可用遥测帧，无法生成定量反馈；请确认采集与对齐"
            "流水线已运行。（已输出全部 10 维度结构与空 sources）"
        )
    parts: list[str] = []
    track_name = track.circuit_name if track else "未知赛道"
    parts.append(f"赛道：{track_name}。基于 {n} 帧遥测分析。")
    lap = values.get("lap_time")
    if lap is not None:
        parts.append(f"本段末圈速 {lap:.2f}s，参考圈速约 {ref_lap:.1f}s。")
    max_spd = values.get("max_speed")
    if max_spd is not None:
        parts.append(f"最高速度 {max_spd:.1f} km/h。")
    for d in dimensions:
        if d["name"] == "balance" and "understeer" in d["value"]:
            parts.append("主要瓶颈：弯中推头。")
            break
        if d["name"] == "balance" and "oversteer" in d["value"]:
            parts.append("主要瓶颈：后轴不稳定。")
            break
    parts.append("详见下方维度与调教建议；每条结论均可在 sources 中追溯证据帧。")
    return " ".join(parts)


def _dim_balance(
    values: dict[str, Any], refs: dict[str, Any], setup: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """balance (推头/甩尾): understeer + oversteer indicators."""
    understeer = values.get("understeer_indicator")
    oversteer = values.get("oversteer_indicator")
    suggestions: list[dict[str, Any]] = []
    if understeer is None and oversteer is None:
        return _data_insufficient("balance"), suggestions

    if understeer is not None and understeer > 0.4:
        s_ref = refs.get("understeer_steer")
        g_ref = refs.get("understeer_g_lat")
        ev = "; ".join(_format_ref(r) for r in (s_ref, g_ref) if r)
        # Iter-193: enhanced understeer assessment with corner context.
        severity = "严重" if understeer > 0.7 else ("中等" if understeer > 0.55 else "轻微")
        phase_hint = ""
        if s_ref and g_ref:
            steer_val = s_ref.get("value", 0)
            g_lat_val = g_ref.get("value", 0)
            if abs(steer_val) > 0.7 and abs(g_lat_val) < 1.5:
                phase_hint = " (高速弯中转向输入大但横向G不足, 前轴气动抓地力不足)"
            elif abs(steer_val) < 0.5:
                phase_hint = " (低速弯中前轮机械抓地力不足)"
        dim = {
            "name": "balance",
            "value": f"understeer indicator {understeer:.2f} ({severity}推头){phase_hint}",
            "evidence": ev or f"understeer={understeer:.2f}",
            "advice": "增加前轴抓地: 提高 front_wing 或软化 front_arb。",
        }
        extra = ""
        if s_ref and g_ref:
            extra = (
                f" Understeer indicator {understeer:.2f}: "
                f"{_format_ref(s_ref)} vs {_format_ref(g_ref)} shows the front "
                "axle sliding mid-corner."
            )
        suggestions.append(
            _make_suggestion(
                setup,
                "front_wing",
                +2,
                reason=("Raise front_wing to add front downforce and reduce understeer." + extra),
                expected_gain="~0.1-0.3s/lap from higher minimum corner speed",
            )
        )
        return dim, suggestions

    if oversteer is not None and oversteer > 0.4:
        rl_ref = refs.get("oversteer_wear_rl")
        rr_ref = refs.get("oversteer_wear_rr")
        ev = "; ".join(_format_ref(r) for r in (rl_ref, rr_ref) if r)
        # Iter-194: enhanced oversteer assessment with severity level.
        severity = "严重" if oversteer > 0.7 else ("中等" if oversteer > 0.55 else "轻微")
        rear_wear_info = ""
        twb = values.get("tyre_wear_balance")
        if twb and isinstance(twb, dict):
            rear_avg = twb.get("rear_avg", 0)
            front_avg = twb.get("front_avg", 0)
            if rear_avg > front_avg + 5:
                rear_wear_info = f" (后轴磨损比前轴高{rear_avg - front_avg:.1f}%)"
        dim = {
            "name": "balance",
            "value": f"oversteer indicator {oversteer:.2f} ({severity}甩尾{rear_wear_info})",
            "evidence": ev or f"oversteer={oversteer:.2f}",
            "advice": "稳定后轴: 提高 rear_wing 或软化 rear_arb。",
        }
        extra = ""
        if rl_ref and rr_ref:
            extra = (
                f" Oversteer indicator {oversteer:.2f}: {_format_ref(rl_ref)}, "
                f"{_format_ref(rr_ref)} show the rear axle is overworking."
            )
        suggestions.append(
            _make_suggestion(
                setup,
                "rear_arb",
                -2,
                reason=("Soften rear_arb to give the rear axle more mechanical grip." + extra),
                expected_gain="~0.1-0.2s/lap from improved traction on exit",
            )
        )
        return dim, suggestions

    u = 0.0 if understeer is None else understeer
    o = 0.0 if oversteer is None else oversteer
    s_ref = refs.get("understeer_steer")
    ev = _format_ref(s_ref) if s_ref else f"understeer={u:.2f}, oversteer={o:.2f}"
    return (
        {
            "name": "balance",
            "value": f"balance neutral (understeer {u:.2f}, oversteer {o:.2f})",
            "evidence": ev,
            "advice": None,
        },
        suggestions,
    )


def _dim_grip(
    values: dict[str, Any],
    refs: dict[str, Any],
    track: Track | None,
    setup: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """grip (整体抓地): max g_lat + speed-through-corners proxy."""
    max_glat = values.get("max_g_lat")
    max_speed = values.get("max_speed")
    if max_glat is None and max_speed is None:
        return _data_insufficient("grip"), []
    parts: list[str] = []
    ev_parts: list[str] = []
    if max_glat is not None:
        parts.append(f"max g_lat {max_glat:.2f}G")
        r = refs.get("max_g_lat")
        if r:
            ev_parts.append(_format_ref(r))
    if max_speed is not None:
        parts.append(f"max speed {max_speed:.1f} km/h")
        r = refs.get("max_speed")
        if r:
            ev_parts.append(_format_ref(r))
    advice: str | None = None
    suggestions: list[dict[str, Any]] = []
    # Low-downforce track with low top speed => reduce drag.
    if (
        track
        and track.track_type == "high_speed_low_downforce"
        and max_speed is not None
        and max_speed < 320.0
    ):
        m_ref = refs.get("max_speed")
        evidence = _format_ref(m_ref) if m_ref else f"max_speed={max_speed:.1f}"
        advice = "降低 drag: 减小 rear_wing 以提升直道尾速。"
        suggestions.append(
            _make_suggestion(
                setup,
                "rear_wing",
                -2,
                reason=(
                    f"Top speed {max_speed:.1f} km/h ({evidence}) is low for a "
                    f"{track.track_type} circuit; lowering rear_wing reduces drag "
                    "on long straights."
                ),
                expected_gain="~0.1-0.2s/lap from higher straight-line speed",
            )
        )
    elif max_glat is not None and max_glat < 3.0:
        advice = "弯中横向 G 偏低，可提高下压力以增加整体抓地。"
    # Iter-217: mechanical grip trend in grip dimension.
    mech_trend = values.get("mech_grip_trend")
    if mech_trend and mech_trend != "stable":
        parts.append(f"mech_grip_trend={mech_trend}")
        if mech_trend == "decaying":
            if advice is None:
                advice = "低速弯机械抓地力在衰退, 注意轮胎管理。"
            else:
                advice += " 低速弯机械抓地力在衰退, 注意轮胎管理。"
        elif mech_trend == "improving":
            if advice is None:
                advice = "机械抓地力趋势向好, 轮胎正在进入工作窗口。"
            else:
                advice += " 机械抓地力趋势向好, 轮胎正在进入工作窗口。"
    return (
        {
            "name": "grip",
            "value": "; ".join(parts),
            "evidence": "; ".join(ev_parts) if ev_parts else "no grip samples",
            "advice": advice,
        },
        suggestions,
    )


def _dim_tyres(
    values: dict[str, Any], refs: dict[str, Any], setup: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """tyres (四轮轮胎温度与磨损): wear balance FL/FR/RL/RR + temp spread."""
    twb = values.get("tyre_wear_balance")
    spread = values.get("tyre_temp_spread")
    if twb is None and spread is None:
        return _data_insufficient("tyres"), []
    parts: list[str] = []
    ev_parts: list[str] = []
    if twb:
        parts.append(
            f"wear FL={twb['fl']:.1f}% FR={twb['fr']:.1f}% "
            f"RL={twb['rl']:.1f}% RR={twb['rr']:.1f}%"
        )
        for k in ("oversteer_wear_rl", "oversteer_wear_rr"):
            r = refs.get(k)
            if r:
                ev_parts.append(_format_ref(r))
    if spread is not None:
        parts.append(f"temp spread {spread:.1f}°C")
        r = refs.get("tyre_temp_fl")
        if r:
            ev_parts.append(_format_ref(r))
    advice: str | None = None
    suggestions: list[dict[str, Any]] = []
    if twb and twb["rear_avg"] - twb["front_avg"] > 5.0:
        advice = "降低后轴负载: 减小 on_throttle_diff 或提高 rear_tyre_pressure。"
        suggestions.append(
            _make_suggestion(
                setup,
                "on_throttle_diff",
                -5,
                reason=(
                    f"Rear tyres wearing "
                    f"{twb['rear_avg'] - twb['front_avg']:.1f}% faster than "
                    "fronts; lowering on_throttle_diff reduces rear locking "
                    "under throttle application."
                ),
                expected_gain="~0.05-0.15s/lap + improved tyre life",
            )
        )
    # Iter-212: tyre temp gradient feedback (inner vs outer edge).
    temp_gradient = values.get("tyre_temp_gradient")
    if temp_gradient is not None and abs(temp_gradient) > 5.0:
        if temp_gradient > 0:
            camber_advice = (
                f"前轮内肩比外肩热 {temp_gradient:.1f}°C, 负外倾角过大; "
                "建议 front_camber +0.3° (减少负倾角) 以均匀胎面温度。"
            )
        else:
            camber_advice = (
                f"前轮外肩比内肩热 {abs(temp_gradient):.1f}°C, 负外倾角不足; "
                "建议 front_camber -0.3° (增加负倾角) 以提升弯中抓地。"
            )
        advice = camber_advice if advice is None else f"{advice} {camber_advice}"
    return (
        {
            "name": "tyres",
            "value": "; ".join(parts),
            "evidence": "; ".join(ev_parts) if ev_parts else "no tyre samples",
            "advice": advice,
        },
        suggestions,
    )


def _dim_braking(
    values: dict[str, Any], refs: dict[str, Any], setup: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """braking (制动表现: 锁死/抱死点): lockup_proxy + brake_bias assessment."""
    lockup = values.get("lockup_proxy")
    if lockup is None:
        # No sustained braking observed.
        bb = values.get("brake_bias_setup")
        if bb is None:
            return _data_insufficient("braking"), []
        return (
            {
                "name": "braking",
                "value": f"no sustained braking; front_brake_bias={bb:.0f}%",
                "evidence": f"front_brake_bias={bb:.0f}% (setup)",
                "advice": None,
            },
            [],
        )
    b_ref = refs.get("lockup_brake")
    s_ref = refs.get("lockup_speed")
    ev_parts = [f"lockup_proxy={lockup:.2f}"]
    if b_ref:
        ev_parts.append(_format_ref(b_ref))
    if s_ref:
        ev_parts.append(_format_ref(s_ref))
    bb = values.get("brake_bias_setup")
    parts = [f"lockup proxy {lockup:.2f}"]
    if bb is not None:
        parts.append(f"brake bias {bb:.0f}%")
    advice: str | None = None
    suggestions: list[dict[str, Any]] = []
    if lockup > 0.3:
        bias_note = ""
        if bb is not None and bb >= 53:
            bias_note = f" (front_brake_bias={bb:.0f}% 偏前)"
        # Iter-195: classify lockup severity and add progressive braking advice.
        sev_label = "严重" if lockup > 0.6 else ("中等" if lockup > 0.45 else "轻微")
        advice = (
            f"前轮锁死(lockup)风险{sev_label}{bias_note}：降低 brake_pressure 或后移 "
            f"front_brake_bias，并采用渐近刹车。建议刹车时先轻踩再逐步加重 (trail braking)。"
        )
        suggestions.append(
            _make_suggestion(
                setup,
                "brake_pressure",
                -3,
                reason=(
                    f"lockup_proxy={lockup:.2f} indicates wheel lockup under "
                    f"braking{_format_ref(b_ref) if b_ref else ''}; lowering "
                    "brake_pressure reduces lockup risk."
                ),
                expected_gain="~0.05-0.15s/lap from cleaner braking",
            )
        )
    return (
        {
            "name": "braking",
            "value": "; ".join(parts),
            "evidence": "; ".join(ev_parts),
            "advice": advice,
        },
        suggestions,
    )


def _dim_brake_temp(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-222: brake temperature balance dimension."""
    bt_ratio = values.get("brake_temp_f_r_ratio")
    bt_diag = values.get("brake_temp_balance_diag") or ""
    if bt_ratio is None:
        return _data_insufficient("brake_temp")
    if bt_ratio > 1.3:
        advice = "前刹车温度过高, 建议后移刹车偏置 1-2% 以平衡热负荷。"
    elif bt_ratio < 0.7:
        advice = "后刹车温度过高, 建议前移刹车偏置 1-2% 以平衡热负荷。"
    else:
        advice = "前后刹车温度分布均匀, 刹车偏置设置合理。"
    return {
        "name": "brake_temp",
        "value": f"brake_temp_f_r_ratio={bt_ratio:.2f} ({bt_diag})",
        "evidence": f"brake_temp_f_r_ratio={bt_ratio:.2f}",
        "advice": advice,
    }


def _dim_tyre_temp_gradient(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-227: tyre temperature gradient dimension (left-right / front-rear)."""
    lr_delta = values.get("tyre_temp_left_right_delta")
    fr_delta = values.get("tyre_temp_front_rear_delta")
    diag = values.get("tyre_temp_gradient_diag") or ""
    if lr_delta is None and fr_delta is None:
        return _data_insufficient("tyre_temp_gradient")
    parts = []
    if lr_delta is not None:
        parts.append(f"left_right_delta={lr_delta:.1f}°C")
    if fr_delta is not None:
        parts.append(f"front_rear_delta={fr_delta:.1f}°C")
    if diag:
        parts.append(f"({diag})")
    if abs(lr_delta or 0) > 3.0:
        side = "左" if (lr_delta or 0) > 0 else "右"
        advice = f"{side}侧轮胎温度偏高, 检查 camber 和胎压对称性。"
    elif abs(fr_delta or 0) > 5.0:
        axle = "前" if (fr_delta or 0) > 0 else "后"
        advice = f"{axle}轴轮胎温度偏高, 检查前后气动平衡和刹车偏置。"
    else:
        advice = "轮胎温度梯度分布均匀, 胎压和 camber 设置合理。"
    return {
        "name": "tyre_temp_gradient",
        "value": "; ".join(parts),
        "evidence": "; ".join(parts),
        "advice": advice,
    }


def _dim_ers_sector_efficiency(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-231: ERS per-sector efficiency dimension."""
    sector_ers = values.get("ers_sector_efficiency")
    if sector_ers is None:
        return _data_insufficient("ers_sector_efficiency")
    parts = []
    worst_sector = -1
    worst_eff = 1.0
    for i, se in enumerate(sector_ers):
        eff = se.get("efficiency", 0.0)
        parts.append(f"S{i+1}_eff={eff:.2f}")
        if eff < worst_eff:
            worst_eff = eff
            worst_sector = i
    if worst_sector >= 0 and worst_eff < 0.5:
        advice = f"S{worst_sector+1} ERS 效率偏低 ({worst_eff:.2f}), 建议优化该扇区部署策略。"
    else:
        advice = "各扇区 ERS 效率良好, 部署策略合理。"
    return {
        "name": "ers_sector_efficiency",
        "value": "; ".join(parts),
        "evidence": "; ".join(parts),
        "advice": advice,
    }


def _dim_grip_consistency(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-241: grip consistency dimension (g_lat std deviation across sectors)."""
    overall_std = values.get("grip_consistency_overall_std")
    worst_sector = values.get("grip_consistency_worst_sector")
    if overall_std is None:
        return _data_insufficient("grip_consistency")
    if overall_std > 0.5:
        advice = (
            f"弯中 g_lat 波动较大 (std={overall_std:.2f}g), 抓地力一致性不足. "
            "建议检查轮胎温度和胎压, 或降低入弯激进程度."
        )
    elif overall_std > 0.3:
        advice = "弯中抓地力有一定波动, 关注轮胎管理. 继续维持当前驾驶风格."
    else:
        advice = "弯中抓地力一致性良好, 轮胎工作状态稳定."
    sector_info = f" (worst=S{worst_sector+1})" if worst_sector is not None and worst_sector >= 0 else ""
    return {
        "name": "grip_consistency",
        "value": f"g_lat_std={overall_std:.3f}g{sector_info}",
        "evidence": f"grip_consistency_overall_std={overall_std:.3f}",
        "advice": advice,
    }


def _dim_ers_deployment(values: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    """ers_deployment (ERS部署与回收): ERS SOC level, deploy vs harvest balance, MGU-K efficiency."""
    store_mean = values.get("ers_store_mean")
    slope = values.get("ers_slope_per_s")
    deployed = values.get("ers_deployed_total")
    harvested = values.get("ers_harvested_total")
    efficiency = values.get("ers_efficiency")
    hotlap_pct = values.get("deploy_mode_hotlap_pct")
    if store_mean is None and slope is None and deployed is None:
        return _data_insufficient("ers_deployment")
    parts: list[str] = []
    ev_parts: list[str] = []
    if store_mean is not None:
        parts.append(f"ERS SOC mean {store_mean:.1f}")
        r = refs.get("ers_store_mean")
        if r:
            ev_parts.append(_format_ref(r))
    if slope is not None:
        direction = "harvesting" if slope > 0 else "deploying"
        parts.append(f"ERS trend {slope:.2f}/s ({direction})")
        for k in ("ers_start", "ers_end"):
            r = refs.get(k)
            if r:
                ev_parts.append(_format_ref(r))
    if deployed is not None and harvested is not None:
        balance = "deploy-heavy" if deployed > harvested * 1.1 else ("harvest-heavy" if harvested > deployed * 1.1 else "balanced")
        parts.append(f"deploy {deployed:.1f} vs harvest {harvested:.1f} ({balance})")
        for k in ("ers_deployed_total", "ers_harvested_total"):
            r = refs.get(k)
            if r:
                ev_parts.append(_format_ref(r))
    if efficiency is not None:
        parts.append(f"ERS efficiency {efficiency:.2f}")
    if hotlap_pct is not None:
        parts.append(f"hotlap mode {hotlap_pct:.0f}%")
        r = refs.get("deploy_mode_hotlap_pct")
        if r:
            ev_parts.append(_format_ref(r))
    advice: str = "平衡 ERS 部署与回收; 优先在出弯与直道前段使用 Hotlap 模式."
    if slope is not None and slope < -0.5:
        advice = "ERS 消耗过快, 增加制动回收或减少低效区段部署."
    elif deployed is not None and harvested is not None and deployed > harvested * 1.3:
        advice = "部署量显著高于回收, 关注 MGU-K 效率及大直道前的 SoC 储备."
    # Iter-204: low SOC warning.
    if store_mean is not None and store_mean < 30.0:
        advice = f"ERS 电量偏低 (SOC={store_mean:.0f}%), 建议切换到回收模式, 减少部署。"
    elif store_mean is not None and store_mean < 50.0:
        advice = f"ERS 电量偏低 (SOC={store_mean:.0f}%), 注意大直道前储备电量。"
    return {
        "name": "ers_deployment",
        "value": "; ".join(parts),
        "evidence": "; ".join(ev_parts) if ev_parts else "no ERS deployment data",
        "advice": advice,
    }


def _dim_drs_usage(values: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    """drs_usage (DRS使用): DRS activation timing, zone utilisation."""
    drs_count = values.get("drs_activation_count")
    active_pct = values.get("drs_active_pct")
    delay_mean = values.get("drs_activation_delay_mean")
    zone_count = values.get("drs_zone_count")
    if drs_count is None and active_pct is None:
        return _data_insufficient("drs_usage")
    parts: list[str] = []
    ev_parts: list[str] = []
    if drs_count is not None:
        parts.append(f"DRS activations: {drs_count}")
        r = refs.get("drs_first_activation")
        if r:
            ev_parts.append(_format_ref(r))
    if active_pct is not None:
        parts.append(f"DRS active {active_pct:.1f}%")
        r = refs.get("drs_active_pct")
        if r:
            ev_parts.append(_format_ref(r))
    if zone_count is not None:
        parts.append(f"DRS zones: {zone_count}")
        r = refs.get("drs_zone_count")
        if r:
            ev_parts.append(_format_ref(r))
    if delay_mean is not None:
        parts.append(f"DRS activation delay mean {delay_mean:.3f}s")
    advice: str = "充分利用 DRS 区段; 确保在每个检测区段都对准前车 1 秒以内."
    if delay_mean is not None and delay_mean > 0.1:
        advice = f"DRS 激活延迟 {delay_mean:.3f}s, 建议提前预判 DRS 窗口以减少延迟."
    return {
        "name": "drs_usage",
        "value": "; ".join(parts),
        "evidence": "; ".join(ev_parts) if ev_parts else "no DRS usage data",
        "advice": advice,
    }


def _dim_smoothness(values: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    """throttle_brake_smoothness (油门/刹车渐进度)."""
    smoothness = values.get("throttle_smoothness")
    brake_aggr = values.get("brake_aggression")
    if smoothness is None and brake_aggr is None:
        return _data_insufficient("throttle_brake_smoothness")
    parts: list[str] = []
    ev_parts: list[str] = []
    if smoothness is not None:
        parts.append(f"throttle smoothness {smoothness:.2f}")
        r = refs.get("throttle_smoothness")
        if r:
            ev_parts.append(_format_ref(r))
    if brake_aggr is not None:
        parts.append(f"brake aggression {brake_aggr:.2f}")
        r = refs.get("brake_aggression")
        if r:
            ev_parts.append(_format_ref(r))
    advice: str | None = None
    if brake_aggr is not None and brake_aggr > 0.5:
        advice = "缓释刹车入弯以助旋转; 油门渐进施加以保牵引。"
    elif smoothness is not None and smoothness < 0.5:
        advice = "油门输入偏顿挫, 建议更渐进施加。"
    return {
        "name": "throttle_brake_smoothness",
        "value": "; ".join(parts),
        "evidence": "; ".join(ev_parts) if ev_parts else "no smoothness samples",
        "advice": advice,
    }


def _dim_confidence(values: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    """confidence (操控信心): correction_freq × g_lat_stability."""
    corr_freq = values.get("steering_correction_freq")
    stability = values.get("g_lat_stability")
    confidence = values.get("confidence_score")
    if corr_freq is None and stability is None:
        return _data_insufficient("confidence")
    parts: list[str] = []
    ev_parts: list[str] = []
    if corr_freq is not None:
        parts.append(f"corrections {corr_freq:.2f} Hz")
        r = refs.get("steering_correction")
        if r:
            ev_parts.append(_format_ref(r))
    if stability is not None:
        parts.append(f"g_lat stability {stability:.2f}")
        r = refs.get("g_lat_stability")
        if r:
            ev_parts.append(_format_ref(r))
    if confidence is not None:
        parts.append(f"confidence {confidence:.2f}")
    advice: str | None = None
    if confidence is not None and confidence < 0.5:
        advice = "操控信心偏低: 减少弯中转向修正, 信任底盘抓地, 平滑输入。"
    return {
        "name": "confidence",
        "value": "; ".join(parts),
        "evidence": "; ".join(ev_parts) if ev_parts else "no confidence samples",
        "advice": advice,
    }


def _dim_lap_time_potential(
    values: dict[str, Any], refs: dict[str, Any], ref_lap: float
) -> dict[str, Any]:
    """lap_time_potential (圈速潜力): gap to reference lap time per track_type."""
    lap_time = values.get("lap_time")
    if lap_time is None or lap_time <= 0:
        dim = _data_insufficient("lap_time_potential")
        dim["value"] = f"insufficient lap-time data (reference {ref_lap:.1f}s)"
        return dim
    gap = lap_time - ref_lap
    ref = refs.get(
        "lap_time",
        {"frame_t": 0.0, "field": "lap_time", "value": float(lap_time)},
    )
    if gap > 0:
        value = f"~{gap:.2f}s above reference ({ref_lap:.1f}s)"
        advice = "Carry more minimum speed through corner exit to close the gap."
    else:
        value = f"~{abs(gap):.2f}s under reference ({ref_lap:.1f}s)"
        advice = "Pace is strong; focus on consistency and tyre preservation."
    return {
        "name": "lap_time_potential",
        "value": value,
        "evidence": _format_ref(ref),
        "advice": advice,
    }


def _dim_sector_compare(
    values: dict[str, Any], refs: dict[str, Any], track: Track | None, ref_lap: float
) -> dict[str, Any]:
    """sector_compare (分段对比): derived sector times vs nominal, else nominal."""
    sector_times = values.get("sector_times")
    nominal = [ref_lap * w for w in _SECTOR_PRIOR_WEIGHTS]
    lap_time = values.get("lap_time")
    if sector_times:
        s1, s2, s3 = sector_times
        n1, n2, n3 = nominal
        deltas = [s1 - n1, s2 - n2, s3 - n3]
        worst = int(np.argmax(deltas))
        ev_parts: list[str] = []
        for k, label in (("sector_s1", "S1"), ("sector_s2", "S2"), ("sector_s3", "S3")):
            r = refs.get(k)
            if r:
                ev_parts.append(f"{label}={r['value']:.2f}s at t={r['frame_t']:.2f}s")
        value = (
            f"S1={s1:.2f}s (nom {n1:.2f}s, {deltas[0]:+.2f}); "
            f"S2={s2:.2f}s (nom {n2:.2f}s, {deltas[1]:+.2f}); "
            f"S3={s3:.2f}s (nom {n3:.2f}s, {deltas[2]:+.2f})"
        )
        advice = f"最慢分段: S{worst + 1} (delta {deltas[worst]:+.2f}s), 优先优化该段。"
        return {
            "name": "sector_compare",
            "value": value,
            "evidence": "; ".join(ev_parts) if ev_parts else "no sector refs",
            "advice": advice,
        }
    # Fallback: no derivable sector boundaries — report nominal split + lap gap.
    nom_str = (
        f"nominal S1={nominal[0]:.2f}s S2={nominal[1]:.2f}s S3={nominal[2]:.2f}s"
    )
    if lap_time is not None and lap_time > 0:
        gap = lap_time - ref_lap
        value = f"sectors not separable; {nom_str}; lap gap {gap:+.2f}s"
        r = refs.get("lap_time")
        evidence = _format_ref(r) if r else f"lap_time={lap_time:.2f}"
    else:
        value = f"sectors not separable; {nom_str}"
        evidence = ""
    return {
        "name": "sector_compare",
        "value": value,
        "evidence": evidence,
        "advice": None,
    }


def _dim_setup_advice(
    setup: dict[str, Any],
    track_id: str,
    rule_suggestions: list[dict[str, Any]],
    driver_profile: DriverProfile | None = None,
) -> dict[str, Any]:
    """setup_advice (调教建议): consume search_setup result as model-driven advice.

    Lazy-imports the optimizer so the feedback module stays decoupled from the
    (heavier) model stack. On any failure (import error, model unavailable,
    invalid setup) falls back to summarising the rule-based suggestions and
    notes the fallback. ``search_setup`` itself handles unknown track_id
    gracefully (gain ≈ 0), so invalid track_id does NOT trigger fallback.

    Iter-05: forwards ``driver_profile`` to :func:`search_setup` so the
    model-driven recommendation is conditioned on the driver's style, and
    documents the resolved profile tag in ``evidence``.

    Iter-2v2: caches search_setup result by (track_id, setup_hash, profile_tag)
    to avoid re-running the 1.4s scipy DE on every feedback call during a session.

    Iter-164.06: 调用 ``search_setup`` 时显式传 ``holistic=True`` (R6.3 修复).
    EA F1 2026 专业车队调教流程始终考虑胎耗保育, LLM→优化器桥接也应使用
    多目标模式 (lap_time + 0.3*tire_wear_proxy), 而非单目标圈速. 旧版未传
    holistic, 导致 FeedbackEngine.run 永远走单目标路径, 与 "整体性思维" (R6)
    要求矛盾. 现在 holistic=True 让 setup_advice 维度的推荐既考虑圈速又
    考虑胎耗, 与 search_setup(holistic=True) 行为一致.

    Iter-164.07: cache_key 加入 holistic 标志 ("holi" / "single"), 避免
    holistic 与 single 模式的结果互相污染缓存.
    """
    # Build cache key from track + setup + profile + holistic mode.
    import hashlib
    import json as _json
    setup_hash = hashlib.md5(
        _json.dumps(setup, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    profile_tag = _profile_style_label(driver_profile)
    # Iter-164.07: holistic 模式标志加入 cache_key.
    holistic_flag = True  # Iter-164.06: LLM 管线默认 holistic
    mode_tag = "holi" if holistic_flag else "single"
    cache_key = (track_id, setup_hash, profile_tag, mode_tag)

    # Check cache.
    cached = _SETUP_ADVICE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        from f1opt.data.setup_schema import CarSetup
        from f1opt.model.optimizer import search_setup

        baseline = CarSetup(**setup)
        result = search_setup(
            track_id,
            driver_profile=driver_profile,
            baseline=baseline,
            iterations=25,
            seed=42,
            holistic=holistic_flag,  # Iter-164.06: R6.3 修复
        )
        gain = float(result.predicted_gain_s)
        diff = result.diff or []
        # Iter-164.06: evidence 中加入 tire_wear / weight 信息 (透明性).
        tire_wear_norm = float(result.tire_wear)
        tire_wear_weight = float(result.tire_wear_weight)
        evidence = (
            f"search_setup(track_id={track_id!r}, iterations=25, seed=42, "
            f"driver_profile={profile_tag}, holistic={holistic_flag}): "
            f"recommended_lap={float(result.recommended_lap_time):.3f}s, "
            f"baseline_lap={float(result.baseline_lap_time):.3f}s, "
            f"tire_wear={tire_wear_norm:.3f} (norm [0,1]), "
            f"tire_wear_weight={tire_wear_weight:.2f}"
        )
        if diff:
            head = diff[:5]
            changes = ", ".join(
                f"{d['name']} {d['before']}→{d['after']}" for d in head
            )
            more = f" (+{len(diff) - 5} more)" if len(diff) > 5 else ""
            value = (
                f"模型推荐 {len(diff)} 项改动; predicted_gain={gain:.3f}s; "
                f"胎耗={tire_wear_norm:.2f}"
            )
            advice = (
                f"应用模型推荐 (predicted_gain={gain:.3f}s, 胎耗={tire_wear_norm:.2f}): "
                f"{changes}{more}"
            )
        else:
            value = (
                f"模型推荐与当前调教一致; predicted_gain={gain:.3f}s; "
                f"胎耗={tire_wear_norm:.2f}"
            )
            advice = (
                f"当前调教已接近模型最优 (predicted_gain={gain:.3f}s, "
                f"胎耗={tire_wear_norm:.2f}); 可参考规则建议微调。"
            )
        result_dim = {
            "name": "setup_advice",
            "value": value,
            "evidence": evidence,
            "advice": advice,
        }
        # Cache the result (bounded LRU via dict size check).
        _SETUP_ADVICE_CACHE[cache_key] = dict(result_dim)
        if len(_SETUP_ADVICE_CACHE) > 50:
            # Evict oldest (first key).
            _SETUP_ADVICE_CACHE.pop(next(iter(_SETUP_ADVICE_CACHE)))
        return result_dim
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback.
        if rule_suggestions:
            names = ", ".join(s["name"] for s in rule_suggestions[:5])
            value = f"模型不可用; 规则建议 {len(rule_suggestions)} 项"
            advice = f"模型不可用 ({type(exc).__name__}); 参考规则建议: {names}"
        else:
            value = "数据不足 / 模型不可用"
            advice = None
        return {
            "name": "setup_advice",
            "value": value,
            "evidence": "",
            "advice": advice,
        }


def _dim_throttle_brake_overlap(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-206: throttle/brake overlap detection dimension."""
    overlap_count = values.get("throttle_brake_overlap_count")
    if overlap_count is None:
        return _data_insufficient("throttle_brake_overlap")
    overlap_pct = values.get("throttle_brake_overlap_pct", 0.0)
    parts = [f"overlap frames: {overlap_count}"]
    if overlap_pct:
        parts.append(f"{overlap_pct:.1%} of lap")
    advice: str | None = None
    if overlap_count > 10:
        advice = "油门刹车同时踩踏次数过多, 浪费燃料和刹车; 提高入弯前收油习惯。"
    return {
        "name": "throttle_brake_overlap",
        "value": "; ".join(parts),
        "evidence": f"overlap_count={overlap_count}",
        "advice": advice,
    }


def _dim_fuel_consumption(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-202, Iter-246: fuel consumption efficiency dimension with sector data."""
    fuel_used = values.get("fuel_used")
    fuel_rate = values.get("fuel_consumption_rate_kg_per_km")
    if fuel_used is None:
        return _data_insufficient("fuel_consumption")
    parts = [f"fuel used {fuel_used:.3f} kg/lap"]
    if fuel_rate is not None:
        parts.append(f"{fuel_rate:.4f} kg/km")
    advice: str | None = None
    # F1 2026 race fuel budget: ~100 kg, ~305 km race → ~0.33 kg/km target.
    if fuel_rate is not None and fuel_rate > 0.35:
        advice = "燃油消耗偏高, 建议 lift-and-coast 或提前升档以节省燃油。"
    elif fuel_rate is not None and fuel_rate < 0.28:
        advice = "燃油消耗偏低, 可考虑提高 ERS 部署功率或增加引擎出力。"
    # Iter-246: add sector-level fuel insight.
    highest_sector = values.get("fuel_highest_sector")
    if highest_sector is not None and highest_sector >= 0:
        if advice:
            advice += f" S{highest_sector+1} 油耗最高, 重点关注该扇区油门控制。"
        else:
            advice = f"S{highest_sector+1} 油耗最高, 建议在该扇区 lift-and-coast。"
    return {
        "name": "fuel_consumption",
        "value": "; ".join(parts),
        "evidence": f"fuel_used={fuel_used:.3f} kg/lap",
        "advice": advice,
    }


def _dim_corner_analysis(
    metrics: dict[str, Any],
    track_id: str,
) -> dict[str, Any]:
    """Iter-164.14: 逐弯分析 (corner_analysis) — 把圈级遥测指标映射到
    最可能产生问题的弯角, 给出逐弯调教建议 (R5 全程动态).

    用 :func:`f1opt.data.corners.get_corners` 拿赛道逐弯数据, 用
    :func:`f1opt.data.corners.problematic_corner_heuristic` 把圈级
    understeer/oversteer/lockup/tire_wear 指标映射到逐弯问题清单.

    返回 ``{"name": "corner_analysis", "value", "evidence", "advice"}``,
    其中 ``value`` 汇总问题弯角数, ``advice`` 列出 top-3 问题弯角 + 建议.
    """
    values = metrics.get("values", {})
    understeer = float(values.get("understeer_indicator", 0.0) or 0.0)
    oversteer = float(values.get("oversteer_indicator", 0.0) or 0.0)
    lockup = float(values.get("lockup_proxy", 0.0) or 0.0)
    # 胎耗过高判断: 用 tyre_wear_balance 或 tyre_temp
    tyre_wear_balance = values.get("tyre_wear_balance", {})
    avg_wear = 0.0
    if isinstance(tyre_wear_balance, dict) and tyre_wear_balance:
        wears = [float(v) for v in tyre_wear_balance.values() if v is not None]
        if wears:
            avg_wear = sum(wears) / len(wears)
    high_tire_wear = avg_wear > 0.5

    try:
        from f1opt.data.corners import (
            corner_demand_summary,
            corner_setup_recommendations,
            get_corners,
            problematic_corner_heuristic,
        )
        corners = get_corners(track_id)
        if not corners:
            return {
                "name": "corner_analysis",
                "value": "赛道逐弯数据不可用",
                "evidence": f"get_corners({track_id!r}) returned empty",
                "advice": None,
            }
        issues = problematic_corner_heuristic(
            corners,
            understeer_indicator=understeer,
            oversteer_indicator=oversteer,
            lockup_proxy=lockup,
            high_tire_wear=high_tire_wear,
        )
        # Iter-164.16: 把弯角问题映射成具体 setup 参数变更
        setup_recs = corner_setup_recommendations(issues)
        demand_summary = corner_demand_summary(track_id)
        top_demands = sorted(demand_summary.items(), key=lambda x: -x[1])[:3]
        demand_str = ", ".join(f"{d}={p:.0%}" for d, p in top_demands)

        n_corners = len(corners)
        n_issues = len(issues)
        if n_issues == 0:
            value = (
                f"逐弯分析: {n_corners} 弯全过, 无显著问题弯角 "
                f"(understeer={understeer:.2f}, oversteer={oversteer:.2f}, "
                f"lockup={lockup:.2f})"
            )
            advice = (
                f"赛道需求: {demand_str}; 圈级指标均在正常范围, 无逐弯调教建议."
            )
        else:
            top3 = issues[:3]
            value = (
                f"逐弯分析: {n_corners} 弯中 {n_issues} 个有潜在问题 "
                f"(understeer={understeer:.2f}, oversteer={oversteer:.2f}, "
                f"lockup={lockup:.2f})"
            )
            parts = []
            for iss in top3:
                parts.append(
                    f"第{iss['corner']}弯({iss['name']}, {iss['corner_type']}, "
                    f"{iss['speed_kmh']:.0f}km/h): {iss['issue']} "
                    f"[sev={iss['severity']:.2f}] → {iss['suggestion']}"
                )
            advice = "逐弯建议: " + "; ".join(parts)
            if n_issues > 3:
                advice += f" (+{n_issues - 3} 更多)"
            # Iter-164.16: 追加具体 setup 参数变更
            if setup_recs:
                rec_parts = []
                for rec in setup_recs[:5]:
                    sign = "+" if rec["delta"] > 0 else ""
                    rec_parts.append(
                        f"{rec['name']}{sign}{rec['delta']}"
                    )
                advice += f" | 逐弯调教变更: {', '.join(rec_parts)}"
        evidence = (
            f"get_corners({track_id!r}) → {n_corners} corners; "
            f"problematic_corner_heuristic(understeer={understeer:.2f}, "
            f"oversteer={oversteer:.2f}, lockup={lockup:.2f}, "
            f"high_tire_wear={high_tire_wear}) → {n_issues} issues; "
            f"corner_setup_recommendations → {len(setup_recs)} setup changes; "
            f"demand_summary: {demand_str}"
        )
        return {
            "name": "corner_analysis",
            "value": value,
            "evidence": evidence,
            "advice": advice,
        }
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback.
        return {
            "name": "corner_analysis",
            "value": f"逐弯分析不可用 ({type(exc).__name__})",
            "evidence": f"corner_analysis fallback: {exc}",
            "advice": None,
        }


def _dim_aero_balance(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-214: 下压力平衡 (aero balance) 维度.

    从 extract_metrics 中读取 high_speed_g_lat_avg / low_speed_g_lat_avg /
    aero_balance_ratio / aero_balance_diagnosis, 判断车辆是气动主导还是
    机械抓地主导, 给出前后翼配比建议.
    """
    aero_ratio = values.get("aero_balance_ratio")
    diag = values.get("aero_balance_diagnosis") or ""
    high_g = values.get("high_speed_g_lat_avg")
    low_g = values.get("low_speed_g_lat_avg")
    if aero_ratio is None:
        return _data_insufficient("aero_balance")
    if aero_ratio > 1.8:
        advice = (
            "气动主导: 高速下压力充足但低速机械抓地力不足. "
            "建议软化前防倾杆 +1 档, 增加前束角以提升机械抓地力."
        )
    elif aero_ratio < 1.1:
        advice = (
            "机械主导: 低速机械抓地力良好但高速气动抓地力不足. "
            "建议前翼+2 档, 后翼+1 档以增加整体下压力."
        )
    else:
        advice = "气动/机械抓地力平衡良好, 继续维持当前前后翼配比."
    return {
        "name": "aero_balance",
        "value": (
            f"aero_balance_ratio {aero_ratio:.2f} ({diag}), "
            f"high_speed_g={high_g:.2f}g, low_speed_g={low_g:.2f}g"
        ),
        "evidence": (
            f"aero_balance_ratio={aero_ratio:.2f}, "
            f"high_speed_g_lat_avg={high_g:.2f}, low_speed_g_lat_avg={low_g:.2f}"
        ),
        "advice": advice,
    }


def _dim_active_aero_usage(values: dict[str, Any]) -> dict[str, Any]:
    """Iter-256: F1 2026 主动空力 (X-Mode / Z-Mode) 使用维度.

    从 extract_metrics 读取 active_aero_x_fraction / active_aero_z_fraction,
    判断 X-Mode (低阻直道) 与 Z-Mode (高下压弯道) 的激活占比, 给出切换时机建议.
    F1 2026 主动空力是核心玩法: Z-Mode 为默认弯道模式, X-Mode 仅在直道可用.
    """
    x_frac = values.get("active_aero_x_fraction")
    z_frac = values.get("active_aero_z_fraction")
    if x_frac is None and z_frac is None:
        return _data_insufficient("active_aero_usage")
    x_frac = float(x_frac or 0.0)
    z_frac = float(z_frac or 0.0)
    if x_frac < 0.2 and z_frac > 0.5:
        advice = (
            "X-Mode (低阻直道) 使用不足: 直道上翼片未充分放平, 阻力偏高损失尾速. "
            "建议在长直道 (尤其 DRS 区) 提前切入 X-Mode, 配合 ERS 超车模式."
        )
    elif x_frac > 0.7:
        advice = (
            "X-Mode 使用过于激进: 若在弯道仍保持低阻, 会损失下压力导致抓地不足. "
            "确保仅在直道使用 X-Mode, 进弯前切回 Z-Mode."
        )
    else:
        advice = "X/Z-Mode 切换节奏合理, 直道低阻与弯道下压力利用均衡, 继续维持."
    return {
        "name": "active_aero_usage",
        "value": f"X-Mode {x_frac:.0%} / Z-Mode {z_frac:.0%}",
        "evidence": (
            f"active_aero_x_fraction={x_frac:.3f}, active_aero_z_fraction={z_frac:.3f}"
        ),
        "advice": advice,
    }


def rule_based_feedback(
    metrics: dict[str, Any],
    setup: dict[str, Any],
    track_id: str,
    driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
) -> dict[str, Any]:
    """Produce ALL 18 dimension entries + setup suggestions from F1 setup rules.

    Each dimension is ``{"name", "value", "evidence", "advice"}`` and every
    numeric claim in ``value`` / ``advice`` traces to an entry in
    ``metrics["sources"]`` (for telemetry-derived dims) or is documented in the
    dimension's ``evidence`` field (for the model-driven ``setup_advice`` dim).
    The ``setup_advice`` dimension calls :func:`search_setup` to present the
    model-driven recommended setup diff; rule-based ``setup_suggestions`` are
    still populated for the UI.

    Iter-05: when ``driver_profile`` is supplied (a :class:`DriverProfile`,
    field-keyed ``dict``, or 8-element ``list``) it is normalised and used to
    personalise the ``advice`` text of the braking / balance /
    throttle_brake_smoothness / confidence dimensions AND forwarded to
    :func:`search_setup` so the model-driven setup_advice is style-conditioned.
    ``driver_profile=None`` reproduces the pre-Iter-05 behaviour exactly.
    """
    profile = _normalize_driver_profile(driver_profile)
    values = metrics.get("values", {})
    refs = metrics.get("refs", {})
    sources: list[dict[str, Any]] = list(metrics.get("sources", []))
    suggestions: list[dict[str, Any]] = []
    track = _resolve_track(track_id)
    ref_lap = _ref_lap_for(track)

    # Build the 18 dimensions in FEEDBACK_DIMENSIONS order. Rule-based
    # suggestions accumulate from balance / grip / tyres / braking dims.
    dim_balance, sug = _dim_balance(values, refs, setup)
    dim_balance = _apply_personal_advice(dim_balance, profile, _balance_personal)
    suggestions.extend(sug)
    dim_grip, sug = _dim_grip(values, refs, track, setup)
    suggestions.extend(sug)
    dim_tyres, sug = _dim_tyres(values, refs, setup)
    suggestions.extend(sug)
    dim_braking, sug = _dim_braking(values, refs, setup)
    dim_braking = _apply_personal_advice(dim_braking, profile, _braking_personal)
    suggestions.extend(sug)
    dim_ers_deployment = _dim_ers_deployment(values, refs)
    dim_drs_usage = _dim_drs_usage(values, refs)
    dim_smoothness = _dim_smoothness(values, refs)
    dim_smoothness = _apply_personal_advice(
        dim_smoothness, profile, _smoothness_personal
    )
    dim_confidence = _dim_confidence(values, refs)
    dim_confidence = _apply_personal_advice(
        dim_confidence, profile, _confidence_personal
    )
    dim_lap_time = _dim_lap_time_potential(values, refs, ref_lap)
    dim_sector = _dim_sector_compare(values, refs, track, ref_lap)

    # Fallback: ensure at least one rule-based suggestion when there is
    # telemetry but no specific rule fired.
    if not suggestions and values.get("lap_time") is not None and track is not None:
        if track.track_type in ("high_downforce", "street"):
            suggestions.append(
                _make_suggestion(
                    setup,
                    "front_wing",
                    +1,
                    reason=f"Default for {track.track_type}: add front-end grip.",
                    expected_gain="marginal",
                )
            )
        elif track.track_type == "high_speed_low_downforce":
            suggestions.append(
                _make_suggestion(
                    setup,
                    "rear_wing",
                    -1,
                    reason=f"Default for {track.track_type}: reduce drag.",
                    expected_gain="marginal",
                )
            )
        else:
            suggestions.append(
                _make_suggestion(
                    setup,
                    "front_arb",
                    +1,
                    reason="Default: sharpen front-end response.",
                    expected_gain="marginal",
                )
            )

    dim_setup = _dim_setup_advice(
        setup, track_id, suggestions, driver_profile=profile
    )

    # Iter-164.14: 逐弯分析维度 (R5 全程动态)
    dim_corner = _dim_corner_analysis(metrics, track_id)

    # Iter-202: fuel consumption dimension
    dim_fuel = _dim_fuel_consumption(values)

    # Iter-206: throttle/brake overlap dimension
    dim_overlap = _dim_throttle_brake_overlap(values)

    # Iter-214: aero balance (downforce) dimension
    dim_aero = _dim_aero_balance(values)

    # Iter-222: brake temperature balance dimension
    dim_brake_temp = _dim_brake_temp(values)

    # Iter-227: tyre temperature gradient dimension
    dim_tyre_temp_grad = _dim_tyre_temp_gradient(values)

    # Iter-241: grip consistency dimension
    dim_grip_consistency = _dim_grip_consistency(values)

    # Iter-256: F1 2026 active aero (X-Mode / Z-Mode) usage dimension
    dim_active_aero = _dim_active_aero_usage(values)

    dimensions = [
        dim_balance,
        dim_grip,
        dim_tyres,
        dim_braking,
        dim_ers_deployment,
        dim_drs_usage,
        dim_smoothness,
        dim_confidence,
        dim_lap_time,
        dim_sector,
        dim_setup,
        dim_corner,  # Iter-164.14: 逐弯分析 (第 12 维)
        dim_fuel,   # Iter-202: 燃油消耗 (第 13 维)
        dim_overlap,  # Iter-206: 油门刹车重叠 (第 14 维)
        dim_aero,   # Iter-214: 下压力平衡 (第 15 维)
        dim_brake_temp,  # Iter-222: 刹车温度平衡 (第 16 维)
        dim_tyre_temp_grad,  # Iter-227: 轮胎温度梯度 (第 17 维)
        dim_grip_consistency,  # Iter-241: 抓地力一致性 (第 18 维)
        dim_active_aero,  # Iter-256: 主动空力使用 (第 19 维)
    ]

    summary = _general_summary(metrics, dimensions, track, ref_lap)

    return {
        "summary": summary,
        "dimensions": dimensions,
        "setup_suggestions": suggestions,
        "sources": sources,
    }


# --------------------------------------------------------------------------- #
# 3. LLM enhancement (optional, gated by config)
# --------------------------------------------------------------------------- #
def _metrics_for_prompt(feedback: dict[str, Any]) -> str:
    sources = feedback.get("sources", [])
    return "\n".join(f"t={s['frame_t']:.2f}s {s['field']}={s['value']:.2f}" for s in sources[:40])


def _assess_quality(
    feedback: dict[str, Any], sources: list[dict[str, Any]]
) -> dict[str, Any]:
    """Iter-146: assess response quality and return a dict for embedding."""
    report = assess_response_quality(feedback, sources)
    return {
        "groundedness": report.groundedness,
        "completeness": report.completeness,
        "actionability": report.actionability,
        "overall": report.overall,
        "label": report.label,
        "issues": report.issues[:10],  # Cap at 10 issues for size
    }


def llm_enhance(
    feedback: dict[str, Any],
    question: str | None,
    config: Settings,
    driver_profile: DriverProfile | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    tracker: TokenUsageTracker | None = None,
) -> dict[str, Any]:
    """Optionally rewrite the summary into natural prose via an LLM.

    Gated by ``config.llm_backend != "none"`` AND a non-empty API key. On any
    error (network, parse, missing endpoint) falls back to the rule-based
    summary unchanged. The structured dimensions / sources are never modified —
    the LLM only rephrases the summary and answers the question.

    Iter-05: ``driver_profile`` is rendered into the user prompt (via
    :func:`format_driver_profile`) so the LLM can personalise tone/emphasis.
    ``None`` yields the ``default (no personalisation)`` marker.

    Iter-125: ``conversation_history`` injects prior multi-turn dialogue as
    alternating ``user``/``assistant`` messages BEFORE the current user prompt,
    enabling the LLM to maintain continuity across turns (resolve "刚才/那个/
    它" references, build on prior advice). ``None`` / empty reproduces the
    pre-Iter-125 single-turn behaviour exactly. Callers should pass
    ``ConversationSession.recent(n)`` captured BEFORE appending the current
    turn, so the current question is not duplicated.

    Iter-138: ``tracker`` (optional) records token usage from the API
    ``usage`` field. When None, the module-level default tracker
    (:func:`get_default_token_tracker`) is used. Failed calls are logged as
    zero-token failed records for call-count accuracy.
    """
    backend = config.llm_backend
    if backend == "none" or (backend == "openai" and not config.llm_api_key):
        return feedback
    endpoint = _LLM_ENDPOINTS.get(backend)
    if endpoint is None:
        return feedback
    model_name = _LLM_DEFAULT_MODEL.get(backend, "gpt-4o-mini")
    tk = tracker if tracker is not None else get_default_token_tracker()
    try:
        import httpx

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question or "(none)",
            granularity="overall",
            granularity_hint="",
            summary=feedback.get("summary", ""),
            dimensions="\n".join(
                f"- {d['name']}: {d['value']} (evidence: {d['evidence']})"
                for d in feedback.get("dimensions", [])
            ),
            metrics_summary=_metrics_for_prompt(feedback),
            driver_profile=format_driver_profile(driver_profile),
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        # Iter-125: inject prior conversation turns as alternating
        # user/assistant messages so the LLM sees multi-turn context.
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role")
                content = turn.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.3,
        }
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {config.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            # Iter-138: record token usage.
            tk.record(
                backend, model_name, _extract_usage(data),
                success=True, streamed=False,
            )
        text = (content or "").strip()
        if text:
            feedback = {**feedback, "summary": text}
    except Exception:
        # Fall back silently to the rule-based summary. Iter-138: log the
        # failed call (zero tokens) for call-count accuracy.
        tk.record(backend, model_name, None, success=False, streamed=False)
    return feedback


async def llm_enhance_async(
    feedback: dict[str, Any],
    question: str | None,
    config: Settings,
    driver_profile: DriverProfile | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    tracker: TokenUsageTracker | None = None,
) -> dict[str, Any]:
    """Async version of :func:`llm_enhance` — Iter-122.

    Uses ``httpx.AsyncClient`` + ``await client.post(...)`` to avoid blocking
    the event loop. The sync :func:`llm_enhance` uses ``httpx.Client`` which
    blocks the entire event loop for up to 10s (timeout) when called from an
    async context (e.g. FastAPI ``async def feedback`` endpoint). This async
    version yields control back to the loop during I/O wait, so WebSocket
    pushes, other HTTP requests, and UDP telemetry dispatch continue running.

    Behaviour is identical to :func:`llm_enhance`: gated by
    ``config.llm_backend != "none"`` + non-empty key; falls back silently on
    any error; only rephrases ``summary``, never touches ``dimensions`` /
    ``sources``.

    Iter-125: ``conversation_history`` injects prior multi-turn dialogue as
    alternating ``user``/``assistant`` messages (same as the sync version).

    Iter-138: ``tracker`` (optional) records token usage; defaults to the
    module-level default tracker. See :func:`llm_enhance`.

    Args:
        feedback: Rule-based feedback dict (with ``summary`` / ``dimensions`` /
            ``sources``).
        question: Optional user question for the LLM.
        config: Settings (provides ``llm_backend`` / ``llm_api_key``).
        driver_profile: Optional driver profile for personalisation.
        conversation_history: Optional list of prior ``{"role", "content"}``
            turns to inject before the current user prompt (Iter-125).
        tracker: Optional :class:`TokenUsageTracker` (Iter-138).

    Returns:
        Feedback dict with ``summary`` possibly rewritten by the LLM.
    """
    backend = config.llm_backend
    if backend == "none" or (backend == "openai" and not config.llm_api_key):
        return feedback
    endpoint = _LLM_ENDPOINTS.get(backend)
    if endpoint is None:
        return feedback
    model_name = _LLM_DEFAULT_MODEL.get(backend, "gpt-4o-mini")
    tk = tracker if tracker is not None else get_default_token_tracker()
    try:
        import httpx

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question or "(none)",
            granularity="overall",
            granularity_hint="",
            summary=feedback.get("summary", ""),
            dimensions="\n".join(
                f"- {d['name']}: {d['value']} (evidence: {d['evidence']})"
                for d in feedback.get("dimensions", [])
            ),
            metrics_summary=_metrics_for_prompt(feedback),
            driver_profile=format_driver_profile(driver_profile),
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        # Iter-125: inject prior conversation turns as alternating
        # user/assistant messages so the LLM sees multi-turn context.
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role")
                content = turn.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.3,
        }
        # Iter-122: AsyncClient + await — non-blocking I/O.
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {config.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            # Iter-138: record token usage.
            tk.record(
                backend, model_name, _extract_usage(data),
                success=True, streamed=False,
            )
        text = (content or "").strip()
        if text:
            feedback = {**feedback, "summary": text}
    except Exception:
        # Fall back silently. Iter-138: log failed call for call-count accuracy.
        tk.record(backend, model_name, None, success=False, streamed=False)
    return feedback


# --------------------------------------------------------------------------- #
# 3b. LLM streaming enhancement (Iter-134)
# --------------------------------------------------------------------------- #
def _build_llm_messages(
    feedback: dict[str, Any],
    question: str | None,
    config: Settings,
    driver_profile: DriverProfile | None,
    conversation_history: list[dict[str, str]] | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the OpenAI-compatible ``messages`` list + payload base.

    Shared by the streaming and non-streaming LLM enhance paths so prompt
    construction stays single-sourced. Returns ``(messages, payload_base)``
    where ``payload_base`` already includes ``model``, ``messages``,
    ``temperature`` — the caller may add ``"stream": True`` for streaming.

    Iter-171: detects feedback granularity (corner/sector/overall) from
    the driver's question and injects it into the user prompt so the LLM
    matches its answer's precision to the driver's intent.
    """
    # Iter-171: classify granularity from the question (default overall).
    granularity_str = "overall"
    granularity_hint = ""
    corner_ref = ""
    if question:
        try:
            from f1opt.feedback.intent import classify_granularity
            gres = classify_granularity(question)
            granularity_str = gres.granularity
            corner_ref = gres.corner_ref
            if gres.confidence < 1.0:
                granularity_hint = (
                    "  (no explicit granularity keyword detected — defaulting "
                    "to overall; mention the most affected corner/sector as "
                    "a concrete example)"
                )
            elif granularity_str == "corner" and corner_ref:
                granularity_hint = (
                    f"  (driver asked about {corner_ref} — cite telemetry "
                    f"from {corner_ref} specifically and give {corner_ref}-"
                    "specific setup advice)"
                )
            elif granularity_str == "sector":
                granularity_hint = (
                    "  (driver asked about a sector — cover that sector's "
                    "time range and give sector-level advice)"
                )
            else:
                granularity_hint = (
                    "  (driver asked about the whole lap — give holistic "
                    "summary with overall setup recommendations)"
                )
        except Exception:
            # Best-effort: if granularity classification fails, default to overall.
            pass
    # Stash granularity on the feedback dict so run() can surface it.
    feedback["granularity"] = granularity_str
    if corner_ref:
        feedback["corner_ref"] = corner_ref

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question or "(none)",
        granularity=granularity_str,
        granularity_hint=granularity_hint,
        summary=feedback.get("summary", ""),
        dimensions="\n".join(
            f"- {d['name']}: {d['value']} (evidence: {d['evidence']})"
            for d in feedback.get("dimensions", [])
        ),
        metrics_summary=_metrics_for_prompt(feedback),
        driver_profile=format_driver_profile(driver_profile),
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    backend = config.llm_backend
    payload = {
        "model": _LLM_DEFAULT_MODEL.get(backend, "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0.3,
    }
    return messages, payload


def _parse_sse_data_line(data_str: str) -> str | None:
    """Parse a single SSE ``data:`` payload string into a content delta text.

    Returns None for [DONE] sentinel, empty deltas, or unparseable JSON.
    Tolerates missing ``choices`` / ``delta`` / ``content`` keys (some servers
    emit role-only or empty deltas at stream start).
    """
    import json as _json

    if not data_str or data_str == "[DONE]":
        return None
    try:
        obj = _json.loads(data_str)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    choices = obj.get("choices")
    if not choices or not isinstance(choices[0], dict):
        return None
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    if not isinstance(content, str) or not content:
        return None
    return content


def llm_enhance_stream(
    feedback: dict[str, Any],
    question: str | None,
    config: Settings,
    driver_profile: DriverProfile | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    tracker: TokenUsageTracker | None = None,
):
    """Streaming version of :func:`llm_enhance` — Iter-134.

    Yields content delta strings (``str``) as they arrive from the
    OpenAI-compatible streaming endpoint (SSE ``data:`` lines). The caller
    concatenates the deltas and, if non-empty, uses the result as the new
    ``summary``; otherwise it falls back to the rule-based ``summary``.

    Gated identically to :func:`llm_enhance`: when ``config.llm_backend ==
    "none"``, the API key is empty, or the backend is unknown, the generator
    produces no items (empty stream) so the caller keeps the rule-based
    summary unchanged. On any error (network, parse, non-200) the generator
    stops silently — partial deltas already yielded are kept by the caller.
    Only ``summary`` is affected; ``dimensions`` / ``sources`` are never
    modified.

    Uses :func:`_build_llm_messages` for prompt construction (single-sourced
    with the non-streaming path) and adds ``"stream": True`` +
    ``"stream_options": {"include_usage": True}`` to the payload so the API
    emits a final chunk with token ``usage`` (Iter-138).

    Iter-138: ``tracker`` records token usage from the final stream chunk.
    """
    import json as _json

    backend = config.llm_backend
    if backend == "none" or (backend == "openai" and not config.llm_api_key):
        return
    endpoint = _LLM_ENDPOINTS.get(backend)
    if endpoint is None:
        return
    model_name = _LLM_DEFAULT_MODEL.get(backend, "gpt-4o-mini")
    tk = tracker if tracker is not None else get_default_token_tracker()
    _, payload = _build_llm_messages(
        feedback, question, config, driver_profile, conversation_history
    )
    payload["stream"] = True
    # Iter-138: request usage in the final stream chunk.
    payload["stream_options"] = {"include_usage": True}
    recorded = False
    try:
        import httpx

        with httpx.Client(timeout=30.0) as client:
            with client.stream(
                "POST",
                endpoint,
                headers={
                    "Authorization": f"Bearer {config.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].lstrip()
                    delta = _parse_sse_data_line(data_str)
                    if delta:
                        yield delta
                    # Iter-138: capture usage from the final chunk (it has
                    # empty choices + a populated usage field).
                    if not recorded and data_str and data_str != "[DONE]":
                        try:
                            obj = _json.loads(data_str)
                        except (ValueError, TypeError):
                            obj = None
                        usage = _extract_usage_from_stream_chunk(obj)
                        if usage is not None:
                            tk.record(
                                backend, model_name, usage,
                                success=True, streamed=True,
                            )
                            recorded = True
    except Exception:
        # Fall back silently: stop yielding; caller keeps rule-based summary.
        # Iter-138: log failed call only if we never recorded a success.
        if not recorded:
            tk.record(backend, model_name, None, success=False, streamed=True)
        return


async def llm_enhance_stream_async(
    feedback: dict[str, Any],
    question: str | None,
    config: Settings,
    driver_profile: DriverProfile | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    tracker: TokenUsageTracker | None = None,
):
    """Async streaming version of :func:`llm_enhance` — Iter-134.

    Async counterpart of :func:`llm_enhance_stream`: yields content delta
    strings (``str``) using ``httpx.AsyncClient.stream(...)`` so the event
    loop is not blocked while waiting for SSE chunks. Behaviour is otherwise
    identical — gated by ``config.llm_backend != "none"`` + non-empty key,
    falls back silently on any error, only rephrases ``summary``.

    Uses :func:`_build_llm_messages` for prompt construction (single-sourced
    with the non-streaming path) and adds ``"stream": True`` +
    ``"stream_options": {"include_usage": True}`` to the payload (Iter-138).

    Iter-138: ``tracker`` records token usage from the final stream chunk.
    """
    import json as _json

    backend = config.llm_backend
    if backend == "none" or (backend == "openai" and not config.llm_api_key):
        return
    endpoint = _LLM_ENDPOINTS.get(backend)
    if endpoint is None:
        return
    model_name = _LLM_DEFAULT_MODEL.get(backend, "gpt-4o-mini")
    tk = tracker if tracker is not None else get_default_token_tracker()
    _, payload = _build_llm_messages(
        feedback, question, config, driver_profile, conversation_history
    )
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    recorded = False
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                endpoint,
                headers={
                    "Authorization": f"Bearer {config.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].lstrip()
                    delta = _parse_sse_data_line(data_str)
                    if delta:
                        yield delta
                    if not recorded and data_str and data_str != "[DONE]":
                        try:
                            obj = _json.loads(data_str)
                        except (ValueError, TypeError):
                            obj = None
                        usage = _extract_usage_from_stream_chunk(obj)
                        if usage is not None:
                            tk.record(
                                backend, model_name, usage,
                                success=True, streamed=True,
                            )
                            recorded = True
    except Exception:
        # Fall back silently: stop yielding; caller keeps rule-based summary.
        if not recorded:
            tk.record(backend, model_name, None, success=False, streamed=True)
        return


# --------------------------------------------------------------------------- #
# 4. Engine + public entry (Iter-134: streaming added)
# --------------------------------------------------------------------------- #
class FeedbackEngine:
    """Wraps the feedback pipeline with a configurable LLM backend."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings | None = config
        # Iter-138: per-engine token usage tracker. When None, the LLM
        # enhance functions fall back to the module-level default tracker.
        self._token_tracker: TokenUsageTracker | None = None
        # Iter-142: per-engine feedback memory for context-aware LLM responses.
        self._feedback_memory: FeedbackMemory | None = None
        # Lazy LLM loading: the LLM backend is NOT loaded at init time.
        # It must be explicitly loaded via preload_llm() before any LLM
        # enhancement runs. This ensures the LLM stays out of memory
        # during gameplay (telemetry collection) and is only loaded when
        # the user stops telemetry and requests feedback analysis.
        self._llm_loaded: bool = False

    @property
    def config(self) -> Settings:
        if self._config is None:
            self._config = get_settings()
        return self._config

    @property
    def token_tracker(self) -> TokenUsageTracker:
        """The token-usage tracker used by this engine's LLM calls (Iter-138).

        Lazily binds to the module-level default tracker on first access so
        all FeedbackEngine instances share one cumulative log unless a
        custom tracker is set via :meth:`set_token_tracker`.
        """
        if self._token_tracker is None:
            self._token_tracker = get_default_token_tracker()
        return self._token_tracker

    def set_token_tracker(self, tracker: TokenUsageTracker | None) -> None:
        """Attach a custom tracker (or None to revert to the default)."""
        self._token_tracker = tracker

    def token_usage(self) -> dict[str, Any]:
        """Cumulative LLM token usage for this engine (Iter-138).

        Returns the :meth:`TokenUsageTracker.totals` dict (prompt /
        completion / total tokens, call counts). Convenience accessor for
        monitoring dashboards.
        """
        return self.token_tracker.totals()

    def token_usage_per_backend(self) -> dict[str, dict[str, Any]]:
        """Per-backend breakdown of token usage (Iter-138)."""
        return self.token_tracker.per_backend()

    def token_cost_estimate(
        self,
        rates: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Rough USD cost estimate per backend (Iter-138)."""
        return self.token_tracker.cost_estimate(rates)

    @property
    def feedback_memory(self) -> FeedbackMemory:
        """The feedback-memory ring buffer used by this engine (Iter-142).

        Lazily creates a default :class:`FeedbackMemory` on first access.
        """
        if self._feedback_memory is None:
            self._feedback_memory = FeedbackMemory()
        return self._feedback_memory

    def set_feedback_memory(self, memory: FeedbackMemory | None) -> None:
        """Attach a custom feedback memory (or None to revert to default)."""
        self._feedback_memory = memory

    def reset_token_usage(self) -> None:
        """Clear the engine's token-usage log (Iter-138)."""
        self.token_tracker.reset()

    def _get_memory_usage_bytes(self) -> int:
        """Return the current process RSS (resident set size) in bytes.

        Uses ``/proc/self/status`` on Linux; falls back to 0 on other
        platforms or when the file is unreadable.
        """
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024  # kB → bytes
        except Exception:
            pass
        return 0

    def preload_llm(self) -> dict[str, Any]:
        """Explicitly load the LLM backend for feedback enhancement.

        Validates the backend configuration (``llm_backend``, ``llm_api_key``),
        records the pre-load memory snapshot, and sets ``_llm_loaded = True``.
        After this call, ``run()`` / ``run_async()`` / ``run_stream()`` /
        ``run_stream_async()`` will invoke the LLM enhance step.

        Must be called explicitly after gameplay ends (telemetry stopped).
        During gameplay the LLM stays unloaded so it never consumes memory
        or CPU while the game is running.

        Returns:
            A dict with ``loaded`` (bool), ``backend``, ``memory_before_bytes``,
            and ``memory_after_bytes``.
        """
        mem_before = self._get_memory_usage_bytes()
        backend = self.config.llm_backend
        api_key = self.config.llm_api_key

        if backend == "none" or (backend == "openai" and not api_key):
            _logger.info(
                "preload_llm: backend=%s, api_key_set=%s — LLM disabled, "
                "skipping load",
                backend, bool(api_key),
            )
            self._llm_loaded = False
            return {
                "loaded": False,
                "backend": backend,
                "reason": "LLM backend is 'none' or API key is empty",
                "memory_before_bytes": mem_before,
                "memory_after_bytes": mem_before,
            }

        endpoint = _LLM_ENDPOINTS.get(backend)
        model_name = _LLM_DEFAULT_MODEL.get(backend)
        if endpoint is None:
            _logger.warning(
                "preload_llm: unknown backend=%s, skipping load", backend,
            )
            self._llm_loaded = False
            return {
                "loaded": False,
                "backend": backend,
                "reason": f"Unknown backend: {backend}",
                "memory_before_bytes": mem_before,
                "memory_after_bytes": mem_before,
            }

        # Iter-254: for the local (Ollama) backend, verify reachability before
        # claiming loaded — otherwise preload reports success while every
        # feedback call then times out (10s) before silently falling back.
        if backend == "local":
            try:
                import httpx
                with httpx.Client(timeout=2.0) as client:
                    r = client.get("http://localhost:11434/api/tags")
                    r.raise_for_status()
            except Exception as exc:
                self._llm_loaded = False
                return {
                    "loaded": False,
                    "backend": backend,
                    "reason": (
                        "Ollama not reachable at http://localhost:11434 "
                        f"({type(exc).__name__})"
                    ),
                    "memory_before_bytes": mem_before,
                    "memory_after_bytes": mem_before,
                }

        self._llm_loaded = True
        mem_after = self._get_memory_usage_bytes()
        _logger.info(
            "preload_llm: backend=%s model=%s endpoint=%s loaded. "
            "memory: %d → %d bytes (delta=%+d)",
            backend, model_name, endpoint,
            mem_before, mem_after, mem_after - mem_before,
        )
        return {
            "loaded": True,
            "backend": backend,
            "model": model_name,
            "endpoint": endpoint,
            "memory_before_bytes": mem_before,
            "memory_after_bytes": mem_after,
        }

    def unload_llm(self) -> dict[str, Any]:
        """Release the LLM backend and free associated memory.

        Sets ``_llm_loaded = False``, clears any cached backend state,
        triggers Python garbage collection, and logs the memory delta.
        After this call, ``run()`` / ``run_async()`` / ``run_stream()`` /
        ``run_stream_async()`` will skip the LLM enhance step and return
        rule-based feedback only.

        Calling this when the LLM is already unloaded is a no-op (returns
        ``loaded=False`` with the current memory snapshot).

        Returns:
            A dict with ``loaded`` (False), ``memory_before_bytes``,
            ``memory_after_bytes``, and ``memory_delta_bytes``.
        """
        mem_before = self._get_memory_usage_bytes()
        if not self._llm_loaded:
            _logger.debug("unload_llm: already unloaded, no-op")
            return {
                "loaded": False,
                "memory_before_bytes": mem_before,
                "memory_after_bytes": mem_before,
                "memory_delta_bytes": 0,
            }

        self._llm_loaded = False
        # Trigger garbage collection to release any cached objects
        # (e.g. httpx connection pools, cached prompt templates).
        _gc.collect()
        mem_after = self._get_memory_usage_bytes()
        delta = mem_after - mem_before
        _logger.info(
            "unload_llm: LLM unloaded. memory: %d → %d bytes (delta=%+d)",
            mem_before, mem_after, delta,
        )
        return {
            "loaded": False,
            "memory_before_bytes": mem_before,
            "memory_after_bytes": mem_after,
            "memory_delta_bytes": delta,
        }

    def run(
        self,
        frames: list[dict[str, Any]],
        setup: dict[str, Any],
        track_id: str,
        question: str | None = None,
        driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        profile = _normalize_driver_profile(driver_profile)
        metrics = extract_metrics(frames, setup, track_id)
        feedback = rule_based_feedback(
            metrics, setup, track_id, driver_profile=profile
        )
        # Iter-07: resolve a conversation session when session_id is provided.
        # The session is consulted (read-only) inside _answer_question so that
        # demonstrative references (刚才/那个/它/上面) resolve against prior
        # turns. The current turn (user question + assistant answer) is
        # appended AFTER the answer is built, so recent() inside
        # _answer_question sees only prior context.
        conversation = get_session(session_id) if session_id is not None else None
        # Iter-125: capture prior turns BEFORE adding the current turn, so the
        # LLM receives multi-turn context without duplicating the current
        # question (which is already embedded in the user prompt template).
        prior_history = conversation.recent(6) if conversation is not None else None
        if question:
            track = _resolve_track(track_id)
            answer = _answer_question(
                question, metrics, track, conversation=conversation
            )
            feedback = {**feedback, "summary": answer + "\n\n" + feedback["summary"]}
            if conversation is not None:
                conversation.add("user", question)
                conversation.add("assistant", answer)
        # LLM enhancement is gated by _llm_loaded: during gameplay the LLM
        # stays unloaded (rule-based only). The caller must explicitly call
        # preload_llm() after stopping telemetry to enable LLM enhancement.
        if self._llm_loaded:
            feedback = llm_enhance(
                feedback,
                question,
                self.config,
                driver_profile=profile,
                conversation_history=prior_history,
                tracker=self.token_tracker,
            )
        # Iter-146: attach quality assessment to every feedback response.
        feedback["_quality"] = _assess_quality(
            feedback, metrics.get("sources", [])
        )
        return feedback

    async def run_async(
        self,
        frames: list[dict[str, Any]],
        setup: dict[str, Any],
        track_id: str,
        question: str | None = None,
        driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`run` — Iter-122.

        Identical to :meth:`run` except the LLM enhancement step uses
        :func:`llm_enhance_async` (``httpx.AsyncClient``) instead of the
        blocking :func:`llm_enhance` (``httpx.Client``). This prevents the
        LLM HTTP call from blocking the event loop when invoked from an async
        context (e.g. FastAPI ``async def`` endpoint).

        When ``config.llm_backend == "none"`` (default), the LLM step is
        short-circuited and this method behaves identically to :meth:`run`
        (no I/O, no await overhead beyond the coroutine creation).
        """
        profile = _normalize_driver_profile(driver_profile)
        metrics = extract_metrics(frames, setup, track_id)
        feedback = rule_based_feedback(
            metrics, setup, track_id, driver_profile=profile
        )
        conversation = get_session(session_id) if session_id is not None else None
        # Iter-125: capture prior turns BEFORE adding the current turn, so the
        # LLM receives multi-turn context without duplicating the current
        # question (which is already embedded in the user prompt template).
        prior_history = conversation.recent(6) if conversation is not None else None
        if question:
            track = _resolve_track(track_id)
            answer = _answer_question(
                question, metrics, track, conversation=conversation
            )
            feedback = {**feedback, "summary": answer + "\n\n" + feedback["summary"]}
            if conversation is not None:
                conversation.add("user", question)
                conversation.add("assistant", answer)
        # Iter-122: async LLM enhancement (non-blocking).
        # Iter-125: pass prior conversation history for multi-turn context.
        # Gated by _llm_loaded (lazy loading).
        if self._llm_loaded:
            feedback = await llm_enhance_async(
                feedback,
                question,
                self.config,
                driver_profile=profile,
                conversation_history=prior_history,
                tracker=self.token_tracker,
            )
        # Iter-146: attach quality assessment.
        feedback["_quality"] = _assess_quality(feedback, metrics.get("sources", []))
        return feedback

    def run_stream(
        self,
        frames: list[dict[str, Any]],
        setup: dict[str, Any],
        track_id: str,
        question: str | None = None,
        driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
        session_id: str | None = None,
    ):
        """Streaming version of :meth:`run` — Iter-134.

        Yields typed event dicts:

        - ``{"type": "chunk", "text": "..."}`` for each LLM token delta
          (concatenate ``text`` to render the live summary).
        - ``{"type": "done", "feedback": <feedback dict>}`` exactly once at
          the end. The ``feedback`` dict carries the final ``summary``: when
          the LLM produced at least one chunk, the concatenated text replaces
          the rule-based summary; otherwise the rule-based summary is kept.

        The rule-based feedback (dimensions / sources / setup_suggestions) is
        built once up-front and never modified by streaming. Conversation
        memory is updated identically to :meth:`run`.

        When ``config.llm_backend == "none"`` (default), only the ``done``
        event is yielded with the rule-based feedback (no chunks).
        """
        profile = _normalize_driver_profile(driver_profile)
        metrics = extract_metrics(frames, setup, track_id)
        feedback = rule_based_feedback(
            metrics, setup, track_id, driver_profile=profile
        )
        conversation = get_session(session_id) if session_id is not None else None
        prior_history = conversation.recent(6) if conversation is not None else None
        if question:
            track = _resolve_track(track_id)
            answer = _answer_question(
                question, metrics, track, conversation=conversation
            )
            feedback = {**feedback, "summary": answer + "\n\n" + feedback["summary"]}
            if conversation is not None:
                conversation.add("user", question)
                conversation.add("assistant", answer)
        # Iter-134: stream LLM chunks; accumulate and replace summary on done.
        # Gated by _llm_loaded (lazy loading).
        accumulated: list[str] = []
        if self._llm_loaded:
            for delta in llm_enhance_stream(
                feedback,
                question,
                self.config,
                driver_profile=profile,
                conversation_history=prior_history,
                tracker=self.token_tracker,
            ):
                accumulated.append(delta)
                yield {"type": "chunk", "text": delta}
        if accumulated:
            feedback = {**feedback, "summary": "".join(accumulated)}
        # Iter-146: attach quality assessment.
        feedback["_quality"] = _assess_quality(feedback, metrics.get("sources", []))
        yield {"type": "done", "feedback": feedback}

    async def run_stream_async(
        self,
        frames: list[dict[str, Any]],
        setup: dict[str, Any],
        track_id: str,
        question: str | None = None,
        driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
        session_id: str | None = None,
    ):
        """Async streaming version of :meth:`run` — Iter-134.

        Async counterpart of :meth:`run_stream`: yields the same typed event
        dicts (``{"type": "chunk", "text": ...}`` then ``{"type": "done",
        "feedback": ...}``) but uses :func:`llm_enhance_stream_async` so the
        event loop is not blocked while waiting for SSE chunks.

        When ``config.llm_backend == "none"`` (default), only the ``done``
        event is yielded with the rule-based feedback (no chunks, no I/O).
        """
        profile = _normalize_driver_profile(driver_profile)
        metrics = extract_metrics(frames, setup, track_id)
        feedback = rule_based_feedback(
            metrics, setup, track_id, driver_profile=profile
        )
        conversation = get_session(session_id) if session_id is not None else None
        prior_history = conversation.recent(6) if conversation is not None else None
        if question:
            track = _resolve_track(track_id)
            answer = _answer_question(
                question, metrics, track, conversation=conversation
            )
            feedback = {**feedback, "summary": answer + "\n\n" + feedback["summary"]}
            if conversation is not None:
                conversation.add("user", question)
                conversation.add("assistant", answer)
        accumulated: list[str] = []
        if self._llm_loaded:
            async for delta in llm_enhance_stream_async(
                feedback,
                question,
                self.config,
                driver_profile=profile,
                conversation_history=prior_history,
                tracker=self.token_tracker,
            ):
                accumulated.append(delta)
                yield {"type": "chunk", "text": delta}
        if accumulated:
            feedback = {**feedback, "summary": "".join(accumulated)}
        # Iter-146: attach quality assessment.
        feedback["_quality"] = _assess_quality(feedback, metrics.get("sources", []))
        yield {"type": "done", "feedback": feedback}


@lru_cache(maxsize=1)
def _default_engine() -> FeedbackEngine:
    return FeedbackEngine()


def generate_feedback(
    frames: list[dict[str, Any]],
    setup: dict[str, Any],
    track_id: str,
    question: str | None = None,
    driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Public entry point: produce evidence-grounded driver feedback.

    Returns a dict with ``summary``, ``dimensions`` (all 10 FEEDBACK_DIMENSIONS),
    ``setup_suggestions`` and ``sources`` (see module docstring). Works without
    an LLM API key via the rule-based path; uses ``config.llm_backend`` for
    richer prose when set.

    Iter-05: ``driver_profile`` (a :class:`~f1opt.driver.profile.DriverProfile`,
    field-keyed ``dict``, or 8-element ``list``) personalises the braking /
    balance / smoothness / confidence ``advice`` text and conditions the
    model-driven ``setup_advice`` on the driver's style. ``None`` (default)
    reproduces the pre-Iter-05 behaviour exactly.

    Iter-07: ``session_id`` enables multi-turn dialogue memory. When non-None
    the engine resolves (or lazily creates) a :class:`ConversationSession` and
    records the question/answer turn so subsequent calls with the same id can
    resolve demonstrative references (刚才/那个/它/上面) against prior turns.
    ``session_id=None`` (default) reproduces the pre-Iter-07 behaviour exactly.
    """
    return _default_engine().run(
        frames,
        setup,
        track_id,
        question,
        driver_profile=driver_profile,
        session_id=session_id,
    )


async def generate_feedback_async(
    frames: list[dict[str, Any]],
    setup: dict[str, Any],
    track_id: str,
    question: str | None = None,
    driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Async public entry point — Iter-122.

    Identical to :func:`generate_feedback` but uses
    :meth:`FeedbackEngine.run_async` (non-blocking ``httpx.AsyncClient`` for
    the LLM step). Preferred over :func:`generate_feedback` when called from
    an async context (FastAPI endpoints, asyncio tasks) to avoid blocking the
    event loop during LLM HTTP calls.

    When ``config.llm_backend == "none"`` (default), the behaviour is
    identical to :func:`generate_feedback` (no network I/O).
    """
    return await _default_engine().run_async(
        frames,
        setup,
        track_id,
        question,
        driver_profile=driver_profile,
        session_id=session_id,
    )


def generate_feedback_stream(
    frames: list[dict[str, Any]],
    setup: dict[str, Any],
    track_id: str,
    question: str | None = None,
    driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
    session_id: str | None = None,
):
    """Streaming public entry point — Iter-134.

    Yields typed event dicts (see :meth:`FeedbackEngine.run_stream`):

    - ``{"type": "chunk", "text": "..."}`` for each LLM token delta.
    - ``{"type": "done", "feedback": <feedback dict>}`` exactly once at end.

    When ``config.llm_backend == "none"`` (default), only the ``done`` event
    is yielded with the rule-based feedback (no chunks, no network I/O).
    Suitable for rendering live LLM output in a UI (e.g. Server-Sent Events
    endpoint) while still delivering the structured feedback dict at the end.
    """
    yield from _default_engine().run_stream(
        frames,
        setup,
        track_id,
        question,
        driver_profile=driver_profile,
        session_id=session_id,
    )


async def generate_feedback_stream_async(
    frames: list[dict[str, Any]],
    setup: dict[str, Any],
    track_id: str,
    question: str | None = None,
    driver_profile: DriverProfile | dict[str, Any] | list[float] | None = None,
    session_id: str | None = None,
):
    """Async streaming public entry point — Iter-134.

    Async counterpart of :func:`generate_feedback_stream`: yields the same
    typed event dicts but uses :meth:`FeedbackEngine.run_stream_async` so the
    event loop is not blocked while waiting for SSE chunks. Preferred in async
    contexts (FastAPI streaming endpoints, asyncio tasks).
    """
    async for event in _default_engine().run_stream_async(
        frames,
        setup,
        track_id,
        question,
        driver_profile=driver_profile,
        session_id=session_id,
    ):
        yield event
