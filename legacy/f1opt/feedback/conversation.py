"""Multi-turn conversation memory for F1OPT feedback (Iter-07 / Iter-25..28 / Iter-175).

Iter-07 ships the :class:`ConversationSession` dataclass (bounded history,
prompt rendering, process-wide registry).

Iter-25..28 extend it with cross-turn intelligence used by the feedback engine
and the new what-if / causal layer:

- :meth:`ConversationSession.remember_setup_change` /
  :meth:`get_setup_history` — track proposed setup edits across turns so the
  causal/what-if analyser can explain a chain of changes.
- :meth:`summarize_focus` — cluster recent driver messages by handling keyword
  (推头/甩尾/胎/刹车/油门/ERS/DRS/平衡/温度/磨损) into a Chinese focus summary.
- :meth:`is_followup` / :meth:`resolve_reference` — detect anaphoric follow-up
  questions (这/那/它/再/还/继续/为什么/怎么样) and resolve demonstratives against
  the most-recent prior user turn.
- :class:`ConversationMemory` — a bounded LRU-style message store with topic
  extraction and serialization, for callers that want a lighter-weight memory
  than a full :class:`ConversationSession`.

Iter-175 adds lap-time trend tracking to :class:`ConversationSession`:

- :meth:`remember_lap` — record a completed lap (lap_time, track_id, position).
- :meth:`get_lap_trend` — compute trend over the last N laps (improving / stable
  / slowing + slope in s/lap).
- :meth:`lap_trend_for_prompt` — format trend as LLM-injectable context text.
- :attr:`lap_history` — bounded list of ``LapRecord`` dataclass instances.

The trend engine uses a simple linear regression on the last N lap times to
compute a per-lap delta; when N < 3 the result is neutral. This bridges the
feedback engine's ring buffer (engine.FeedbackMemory) and the conversation
session so the LLM can answer questions like "趋势怎么样?" / "上圈改善了吗?" /
"Is the car getting faster across the stint?" with real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple


class LapRecord(NamedTuple):
    """A completed lap entry stored in the session's lap history (Iter-175)."""
    lap_number: int
    lap_time_s: float
    track_id: str

_FOCUS_KEYWORDS: tuple[str, ...] = (
    "推头",
    "甩尾",
    "胎",
    "刹车",
    "油门",
    "ERS",
    "DRS",
    "平衡",
    "温度",
    "磨损",
)

_FOLLOWUP_TRIGGERS: tuple[str, ...] = (
    "怎么样",
    "为什么",
    "这样",
    "那样",
    "再调",
    "还有",
    "继续",
    "它",
    "这个",
    "那个",
    "再",
    "还",
    "这",
    "那",
)

_REFERENCE_NOUNS: tuple[str, ...] = (
    "前翼",
    "后翼",
    "前制动",
    "后制动",
    "制动",
    "刹车",
    "前胎压",
    "后胎压",
    "胎压",
    "轮胎",
    "胎温",
    "胎",
    "油门",
    "差速器",
    "防倾杆",
    "弹簧",
    "离地间隙",
    "外倾角",
    "前束",
    "ERS",
    "DRS",
    "燃油",
    "推头",
    "甩尾",
    "下压力",
    "平衡",
    "磨损",
)

_MAX_LAP_HISTORY = 100


@dataclass
class ConversationSession:
    """A single driver's conversation history (last N turns)."""

    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    max_turns: int = 10
    setup_changes: list[dict] = field(default_factory=list)
    lap_history: list[LapRecord] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def recent(self, n: int = 5) -> list[dict[str, str]]:
        return self.history[-n:] if n < len(self.history) else self.history[:]

    def format_for_prompt(self, n: int = 5) -> str:
        turns = self.recent(n)
        if not turns:
            return ""
        lines = []
        for t in turns:
            role = "车手" if t["role"] == "user" else "工程师"
            lines.append(f"{role}: {t['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Iter-175: lap-time trend tracking
    # ------------------------------------------------------------------ #
    def remember_lap(
        self, lap_time_s: float, track_id: str = "", lap_number: int = -1
    ) -> None:
        """Record a completed lap in the session's lap history.

        If ``lap_number`` is negative (default), the next sequential lap
        number is auto-assigned (``len(lap_history) + 1``). The lap history
        is bounded to :data:`_MAX_LAP_HISTORY` entries, keeping the most
        recent ones on overflow.
        """
        if lap_number < 0:
            lap_number = len(self.lap_history) + 1
        record = LapRecord(
            lap_number=lap_number,
            lap_time_s=float(lap_time_s),
            track_id=track_id,
        )
        self.lap_history.append(record)
        if len(self.lap_history) > _MAX_LAP_HISTORY:
            self.lap_history = self.lap_history[-_MAX_LAP_HISTORY:]

    def get_lap_trend(self, n: int = 5) -> dict:
        """Compute the lap-time trend over the most recent ``n`` laps.

        Uses simple linear regression (y = a + b*x) on the last n lap times
        with x = lap index (0..n-1). The slope ``b`` is the per-lap delta
        (negative = improving). When ``n < 3`` or lap history is empty, the
        trend is neutral with zero confidence.

        Returns:
            ``{"trend": str, "slope_s_per_lap": float, "first_lap_s": float,
            "last_lap_s": float, "n_laps": int, "delta_total_s": float}``

        ``trend`` is one of ``"improving"`` (slope < -0.05),
        ``"slowing"`` (slope > +0.05), or ``"stable"`` (within ±0.05).
        """
        if not self.lap_history or n < 3:
            return {
                "trend": "stable",
                "slope_s_per_lap": 0.0,
                "first_lap_s": 0.0,
                "last_lap_s": 0.0,
                "n_laps": len(self.lap_history),
                "delta_total_s": 0.0,
            }
        laps = self.lap_history[-min(n, len(self.lap_history)):]
        times = [r.lap_time_s for r in laps]
        m = len(times)
        if m < 2:
            return {
                "trend": "stable",
                "slope_s_per_lap": 0.0,
                "first_lap_s": times[0],
                "last_lap_s": times[-1],
                "n_laps": m,
                "delta_total_s": 0.0,
            }
        x_mean = (m - 1) / 2.0
        y_mean = sum(times) / m
        num = sum((i - x_mean) * (times[i] - y_mean) for i in range(m))
        den = sum((i - x_mean) ** 2 for i in range(m))
        slope = num / den if den != 0 else 0.0
        delta_total = times[-1] - times[0]
        if slope < -0.05:
            trend = "improving"
        elif slope > 0.05:
            trend = "slowing"
        else:
            trend = "stable"
        return {
            "trend": trend,
            "slope_s_per_lap": round(slope, 4),
            "first_lap_s": round(times[0], 3),
            "last_lap_s": round(times[-1], 3),
            "n_laps": m,
            "delta_total_s": round(delta_total, 3),
        }

    def lap_trend_for_prompt(self, n: int = 5) -> str:
        """Format the lap-time trend as prompt-injectable context text.

        When there are >= 3 laps, returns a line like:
        ``"圈速趋势 (5圈): 92.103s → 91.870s, 改善中 (-0.047s/圈, 合计 -0.233s)"``

        When there are < 3 laps or the trend is stable (< ±0.05 s/lap),
        returns a simpler summary. Returns an empty string when there is no
        lap history at all.
        """
        info = self.get_lap_trend(n)
        if info["n_laps"] == 0:
            return ""
        if info["n_laps"] < 3:
            return (
                f"Lap time ({info['n_laps']} lap{'s' if info['n_laps']>1 else ''}): "
                f"{info['last_lap_s']:.3f}s"
            )
        delta_label = (
            "改善中" if info["trend"] == "improving"
            else "恶化中" if info["trend"] == "slowing"
            else "稳定"
        )
        return (
            f"圈速趋势 ({info['n_laps']}圈): "
            f"{info['first_lap_s']:.3f}s → {info['last_lap_s']:.3f}s, "
            f"{delta_label} ({info['slope_s_per_lap']:+.3f}s/圈, "
            f"合计 {info['delta_total_s']:+.3f}s)"
        )

    # ------------------------------------------------------------------ #
    # Iter-25..28: cross-turn intelligence (setup tracking / focus /
    # anaphora resolution)
    # ------------------------------------------------------------------ #
    def remember_setup_change(
        self, field: str, before: float, after: float, reason: str
    ) -> None:
        self.setup_changes.append(
            {
                "field": field,
                "before": float(before),
                "after": float(after),
                "reason": reason,
                "turn": len(self.history) + 1,
            }
        )

    def get_setup_history(self) -> list[dict]:
        return [dict(entry) for entry in self.setup_changes]

    def summarize_focus(self) -> str:
        user_msgs = [t["content"] for t in self.history if t.get("role") == "user"]
        if not user_msgs:
            return "本次对话焦点: (暂无对话)"
        counts: dict[str, int] = {}
        for kw in _FOCUS_KEYWORDS:
            hits = sum(m.count(kw) for m in user_msgs)
            if hits > 0:
                counts[kw] = hits
        if not counts:
            return "本次对话焦点: (无明显关键词)"
        ranked = sorted(counts, key=lambda k: (-counts[k], k))
        return "本次对话焦点: " + "、".join(ranked)

    def is_followup(self, question: str) -> bool:
        return any(tr in question for tr in _FOLLOWUP_TRIGGERS)

    def resolve_reference(self, question: str) -> str:
        prior_user: str | None = None
        for t in reversed(self.history):
            if t.get("role") == "user":
                prior_user = t.get("content")
                break
        if not prior_user:
            return question
        noun = _extract_reference_noun(prior_user)
        if not noun:
            return question
        for demo in ("这个", "那个", "这", "那", "它"):
            if demo in question:
                return question.replace(demo, noun, 1)
        return question


def _extract_reference_noun(text: str) -> str | None:
    for noun in _REFERENCE_NOUNS:
        if noun in text:
            return noun
    return None


class ConversationMemory:
    """Bounded LRU-style multi-turn memory (Iter-27)."""

    def __init__(self, max_turns: int = 20) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        self.max_turns = max_turns
        self._user: list[str] = []
        self._bot: list[str] = []

    def add_user_message(self, text: str) -> None:
        self._user.append(text)
        if len(self._user) > self.max_turns:
            self._user = self._user[-self.max_turns :]

    def add_bot_message(self, text: str) -> None:
        self._bot.append(text)
        if len(self._bot) > self.max_turns:
            self._bot = self._bot[-self.max_turns :]

    def recent_user_messages(self, n: int = 3) -> list[str]:
        if n < len(self._user):
            return list(self._user[-n:])
        return list(self._user)

    def topics(self) -> list[str]:
        recent = self.recent_user_messages(self.max_turns)
        counts: dict[str, int] = {}
        for kw in _FOCUS_KEYWORDS:
            hits = sum(m.count(kw) for m in recent)
            if hits > 0:
                counts[kw] = hits
        return sorted(counts, key=lambda k: (-counts[k], k))

    def to_dict(self) -> dict:
        return {
            "max_turns": self.max_turns,
            "user_messages": list(self._user),
            "bot_messages": list(self._bot),
        }


_sessions: dict[str, ConversationSession] = {}


def get_session(session_id: str = "default") -> ConversationSession:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationSession(session_id=session_id)
    return _sessions[session_id]


def reset_sessions() -> None:
    _sessions.clear()


def list_sessions() -> list[dict[str, Any]]:
    """Iter-183: List all active sessions with stats."""
    result: list[dict[str, Any]] = []
    for sid, session in _sessions.items():
        result.append({
            "session_id": sid,
            "turn_count": len(session.history),
            "setup_changes": len(session.setup_changes),
            "lap_count": len(session.lap_history),
            "focus": session.summarize_focus(),
        })
    return result


def session_stats() -> dict[str, Any]:
    """Iter-183: Aggregate stats across all sessions."""
    total_turns = 0
    total_setup_changes = 0
    total_laps = 0
    for session in _sessions.values():
        total_turns += len(session.history)
        total_setup_changes += len(session.setup_changes)
        total_laps += len(session.lap_history)
    return {
        "total_sessions": len(_sessions),
        "total_turns": total_turns,
        "total_setup_changes": total_setup_changes,
        "total_laps": total_laps,
    }


# --------------------------------------------------------------------------- #
# Iter-150: context window management
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    ascii_chars = sum(1 for c in text if c.isascii() and c.isalnum())
    return int(cjk + ascii_chars * 0.75)


def trim_conversation_history(
    history: list[dict[str, str]],
    max_tokens: int = 4096,
    *,
    preserve_system: bool = True,
    min_recent: int = 3,
) -> list[dict[str, str]]:
    if not history:
        return []
    n = len(history)
    total = sum(estimate_tokens(m.get("content", "")) for m in history)
    if total <= max_tokens:
        return list(history)
    protected: set[int] = set()
    for i in range(max(0, n - min_recent), n):
        protected.add(i)
    if preserve_system:
        for i, m in enumerate(history):
            if m.get("role") == "system":
                protected.add(i)
                break
    result: list[dict[str, str]] = []
    result_tokens = 0
    for i, m in enumerate(history):
        if i in protected:
            result.append(m)
            result_tokens += estimate_tokens(m.get("content", ""))
    if result_tokens > max_tokens:
        result = []
        result_tokens = 0
        for m in history:
            if preserve_system and m.get("role") == "system":
                sys_tokens = estimate_tokens(m.get("content", ""))
                if sys_tokens <= max_tokens:
                    result.append(m)
                    result_tokens += sys_tokens
                break
        for m in history[-min_recent:]:
            t = estimate_tokens(m.get("content", ""))
            if result_tokens + t <= max_tokens:
                result.append(m)
                result_tokens += t
            else:
                break
    return result


def conversation_token_usage(
    history: list[dict[str, str]],
) -> dict[str, int]:
    result = {"total": 0, "system": 0, "user": 0, "assistant": 0, "message_count": len(history)}
    for m in history:
        t = estimate_tokens(m.get("content", ""))
        result["total"] += t
        role = m.get("role", "unknown")
        if role in result:
            result[role] += t
    return result
