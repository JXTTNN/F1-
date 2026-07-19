"""Multi-turn conversation memory for F1OPT feedback (Iter-07 / Iter-25..28).

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
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Driver-focus keywords used by :meth:`ConversationSession.summarize_focus`
#: and :meth:`ConversationMemory.topics`. Order is roughly handling-area then
#: subsystem; clustering counts occurrences across recent user messages.
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

#: Anaphora / follow-up triggers used by :meth:`ConversationSession.is_followup`.
#: Ordered longest-first so multi-character triggers (怎么样 / 为什么) win over
#: bare 这 / 那 — a bare "怎么调" (fresh how-to question) must NOT match "怎么样".
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

#: Candidate reference noun phrases (setup / handling terms) scanned from a
#: prior user turn by :meth:`ConversationSession.resolve_reference`. Ordered
#: most-specific-first so e.g. "前翼" / "胎压" match before bare "胎".
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


@dataclass
class ConversationSession:
    """A single driver's conversation history (last N turns)."""

    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    max_turns: int = 10
    # Iter-27: tracked setup-change proposals across turns (fed to the causal /
    # what-if analyser so a chain of edits can be explained together).
    setup_changes: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """Append a turn (role='user'/'assistant'); trim to max_turns."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def recent(self, n: int = 5) -> list[dict[str, str]]:
        """Return the last n turns (or all if fewer)."""
        return self.history[-n:] if n < len(self.history) else self.history[:]

    def format_for_prompt(self, n: int = 5) -> str:
        """Render recent turns as context text for _answer_question / LLM prompt."""
        turns = self.recent(n)
        if not turns:
            return ""
        lines = []
        for t in turns:
            role = "车手" if t["role"] == "user" else "工程师"
            lines.append(f"{role}: {t['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Iter-25..28: cross-turn intelligence (setup tracking / focus /
    # anaphora resolution). All additive — existing API unchanged.
    # ------------------------------------------------------------------ #
    def remember_setup_change(
        self, field: str, before: float, after: float, reason: str
    ) -> None:
        """Record a proposed setup change across turns.

        Appends ``{field, before, after, reason, turn}`` to
        :attr:`setup_changes`. ``turn`` is the 1-based conversation turn index
        at proposal time (``len(history) + 1``), so a chain of edits can be
        ordered for causal / what-if analysis.
        """
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
        """Return tracked setup-change proposals as a list of dicts.

        Returns a shallow copy so callers cannot mutate the session's internal
        log accidentally.
        """
        return [dict(entry) for entry in self.setup_changes]

    def summarize_focus(self) -> str:
        """Cluster recent user messages by handling keyword into a focus line.

        Scans every ``role == "user"`` turn and counts occurrences of each
        :data:`_FOCUS_KEYWORDS` term (推头/甩尾/胎/刹车/油门/ERS/DRS/平衡/
        温度/磨损). Returns ``"本次对话焦点: <kw1>、<kw2> ..."`` with keywords
        ranked by frequency (most-discussed first). Returns a neutral marker
        when there are no user turns or no keyword hits.
        """
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
        """Detect an anaphoric / follow-up question.

        Returns ``True`` when ``question`` contains any follow-up trigger
        (:data:`_FOLLOWUP_TRIGGERS`: 这/那/它/再/还/继续/为什么/怎么样 ...).
        A fresh how-to question such as ``"前翼怎么调"`` does NOT contain a
        trigger (``"怎么"`` without ``"样"``) and returns ``False``.
        """
        return any(tr in question for tr in _FOLLOWUP_TRIGGERS)

    def resolve_reference(self, question: str) -> str:
        """Resolve a demonstrative (这/那) against the prior user turn.

        Extracts the most-recent prior ``role == "user"`` message, finds the
        first matching reference noun (:data:`_REFERENCE_NOUNS`) in it, and
        replaces the first demonstrative occurrence in ``question`` with that
        noun. If there is no prior user turn, no extractable noun, or no
        demonstrative, ``question`` is returned unchanged.
        """
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
        # Longer demonstratives first so "这个" / "那个" are replaced before a
        # bare "这" / "那" would match their leading character.
        for demo in ("这个", "那个", "这", "那"):
            if demo in question:
                return question.replace(demo, noun, 1)
        return question


def _extract_reference_noun(text: str) -> str | None:
    """Return the first :data:`_REFERENCE_NOUNS` term found in ``text``."""
    for noun in _REFERENCE_NOUNS:
        if noun in text:
            return noun
    return None


class ConversationMemory:
    """Bounded LRU-style multi-turn memory (Iter-27).

    Stores user / assistant messages in two independent bounded lists
    (each trimmed to ``max_turns`` on overflow, keeping the most recent). This
    is a lighter-weight alternative to :class:`ConversationSession` for
    callers that only need message recall + topic extraction (no session
    registry, no prompt rendering).

    Topic extraction reuses :data:`_FOCUS_KEYWORDS` so :meth:`topics` aligns
    with :meth:`ConversationSession.summarize_focus`.
    """

    def __init__(self, max_turns: int = 20) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        self.max_turns = max_turns
        self._user: list[str] = []
        self._bot: list[str] = []

    def add_user_message(self, text: str) -> None:
        """Append a user message; trim to ``max_turns`` (keep most-recent)."""
        self._user.append(text)
        if len(self._user) > self.max_turns:
            self._user = self._user[-self.max_turns :]

    def add_bot_message(self, text: str) -> None:
        """Append a bot/assistant message; trim to ``max_turns``."""
        self._bot.append(text)
        if len(self._bot) > self.max_turns:
            self._bot = self._bot[-self.max_turns :]

    def recent_user_messages(self, n: int = 3) -> list[str]:
        """Return the last ``n`` user messages (or all if fewer)."""
        if n < len(self._user):
            return list(self._user[-n:])
        return list(self._user)

    def topics(self) -> list[str]:
        """Extract focus keywords from recent user turns, ranked by frequency.

        Returns unique keywords found across the most recent ``max_turns``
        user messages, sorted by occurrence count (desc) then keyword. Empty
        list when nothing matches.
        """
        recent = self.recent_user_messages(self.max_turns)
        counts: dict[str, int] = {}
        for kw in _FOCUS_KEYWORDS:
            hits = sum(m.count(kw) for m in recent)
            if hits > 0:
                counts[kw] = hits
        return sorted(counts, key=lambda k: (-counts[k], k))

    def to_dict(self) -> dict:
        """Serialize the memory (max_turns + bounded user/bot message lists)."""
        return {
            "max_turns": self.max_turns,
            "user_messages": list(self._user),
            "bot_messages": list(self._bot),
        }


# Global session registry (process-wide, lazy create)
_sessions: dict[str, ConversationSession] = {}


def get_session(session_id: str = "default") -> ConversationSession:
    """Get or create a ConversationSession by id."""
    if session_id not in _sessions:
        _sessions[session_id] = ConversationSession(session_id=session_id)
    return _sessions[session_id]


def reset_sessions() -> None:
    """Clear all sessions (for testing)."""
    _sessions.clear()


# --------------------------------------------------------------------------- #
# Iter-150: context window management
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string (Iter-150).

    Uses a language-aware heuristic:
    - CJK characters: ~1 token each
    - ASCII words: ~0.75 tokens each (4 chars ≈ 3 tokens)
    - Digits/punctuation: bundled into adjacent words

    This is a fast approximation (no tokenizer dependency) suitable for
    context-window gating. For exact counts, use the backend's tokenizer.
    """
    if not text:
        return 0
    # Count CJK characters (Unicode range U+4E00–U+9FFF, plus common extensions)
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    # Count ASCII word characters
    ascii_chars = sum(1 for c in text if c.isascii() and c.isalnum())
    # Approximate: CJK ≈ 1 token each, ASCII ≈ 0.75 tokens each
    return int(cjk + ascii_chars * 0.75)


def trim_conversation_history(
    history: list[dict[str, str]],
    max_tokens: int = 4096,
    *,
    preserve_system: bool = True,
    min_recent: int = 3,
) -> list[dict[str, str]]:
    """Trim a conversation history to fit within a token budget (Iter-150).

    Strategy:
    1. Always keep the most recent ``min_recent`` messages (recency bias).
    2. Always keep the system message (if ``preserve_system`` is True).
    3. Remove older messages from the middle until the total estimated
       tokens fit within ``max_tokens``.
    4. Never remove the system message or the most recent messages.

    This is a structural trim — it does not summarise or rewrite content.
    For semantic compression, combine with a separate summarisation step.

    Args:
        history: List of ``{"role": str, "content": str}`` messages.
        max_tokens: Maximum token budget for the trimmed history.
        preserve_system: If True, never remove the ``role="system"`` message.
        min_recent: Minimum number of most-recent messages to keep.

    Returns:
        A new trimmed list (does not mutate the original).
    """
    if not history:
        return []

    n = len(history)
    # Nothing to trim if already within budget
    total = sum(estimate_tokens(m.get("content", "")) for m in history)
    if total <= max_tokens:
        return list(history)

    # Identify protected indices
    protected: set[int] = set()
    # Always keep the most recent min_recent messages
    for i in range(max(0, n - min_recent), n):
        protected.add(i)
    # Always keep the system message
    if preserve_system:
        for i, m in enumerate(history):
            if m.get("role") == "system":
                protected.add(i)
                break

    # Build trimmed list: keep protected messages + trim from middle
    result: list[dict[str, str]] = []
    result_tokens = 0
    # Phase 1: collect protected messages (in order)
    for i, m in enumerate(history):
        if i in protected:
            result.append(m)
            result_tokens += estimate_tokens(m.get("content", ""))

    # Phase 2: if still over budget, we can't trim further (all remaining are protected)
    if result_tokens > max_tokens:
        # Extreme case: even protected messages exceed budget.
        # Keep only system + last min_recent, truncating if needed.
        result = []
        result_tokens = 0
        # System first
        for m in history:
            if preserve_system and m.get("role") == "system":
                sys_tokens = estimate_tokens(m.get("content", ""))
                if sys_tokens <= max_tokens:
                    result.append(m)
                    result_tokens += sys_tokens
                break
        # Add recent messages until budget is exhausted
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
    """Compute token usage statistics for a conversation history (Iter-150).

    Returns:
        ``{"total": int, "system": int, "user": int, "assistant": int, "message_count": int}``
    """
    result = {"total": 0, "system": 0, "user": 0, "assistant": 0, "message_count": len(history)}
    for m in history:
        t = estimate_tokens(m.get("content", ""))
        result["total"] += t
        role = m.get("role", "unknown")
        if role in result:
            result[role] += t
    return result
