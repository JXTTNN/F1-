"""Tests for :mod:`f1opt.feedback.conversation` (Iter-07 multi-turn memory).

Covers the :class:`ConversationSession` dataclass (``add`` / ``recent`` /
``format_for_prompt`` / trim-to-``max_turns``) and the process-wide session
registry helpers (:func:`get_session` lazy create + :func:`reset_sessions`).

These tests are pure unit tests: they do not exercise the feedback engine or
the rule-based answer builder, only the memory primitive that the engine
consults when ``session_id`` is supplied.
"""

from __future__ import annotations

import pytest

from f1opt.feedback import ConversationSession, get_session, reset_sessions
from f1opt.feedback.conversation import _sessions


# --------------------------------------------------------------------------- #
# ConversationSession.add / .recent
# --------------------------------------------------------------------------- #
def test_add_appends_user_and_assistant_turns_in_order() -> None:
    sess = ConversationSession(session_id="s1")
    assert sess.history == []
    sess.add("user", "为什么推头")
    sess.add("assistant", "前轴出现推头 indicator=0.6")
    assert sess.history == [
        {"role": "user", "content": "为什么推头"},
        {"role": "assistant", "content": "前轴出现推头 indicator=0.6"},
    ]
    # recent() with no args returns the last 5 turns (default n=5).
    assert sess.recent() == sess.history


def test_recent_returns_last_n_or_all_when_fewer() -> None:
    sess = ConversationSession(session_id="s2")
    for i in range(8):
        sess.add("user" if i % 2 == 0 else "assistant", f"turn-{i}")
    # Asking for the last 3 turns yields the trailing 3 entries only.
    last3 = sess.recent(3)
    assert len(last3) == 3
    assert last3[0]["content"] == "turn-5"
    assert last3[-1]["content"] == "turn-7"
    # Asking for more than available returns the full history (no padding).
    assert sess.recent(50) == sess.history[:]
    # recent(1) returns just the last turn.
    assert sess.recent(1) == [{"role": "assistant", "content": "turn-7"}]
    # recent() default (n=5) returns the trailing 5 turns.
    assert len(sess.recent()) == 5
    assert sess.recent()[0]["content"] == "turn-3"


# --------------------------------------------------------------------------- #
# ConversationSession.format_for_prompt
# --------------------------------------------------------------------------- #
def test_format_for_prompt_renders_role_labels() -> None:
    sess = ConversationSession(session_id="s3")
    sess.add("user", "为什么推头")
    sess.add("assistant", "前轴抓地不足")
    text = sess.format_for_prompt(5)
    # user role -> 车手, assistant role -> 工程师.
    assert "车手: 为什么推头" in text
    assert "工程师: 前轴抓地不足" in text
    # Two turns joined by a single newline.
    assert text.count("\n") == 1
    # Empty session renders as empty string.
    assert ConversationSession(session_id="s4").format_for_prompt() == ""


# --------------------------------------------------------------------------- #
# ConversationSession trim-to-max_turns
# --------------------------------------------------------------------------- #
def test_add_trims_to_max_turns_keeping_most_recent() -> None:
    sess = ConversationSession(session_id="s5", max_turns=4)
    for i in range(10):
        sess.add("user" if i % 2 == 0 else "assistant", f"turn-{i}")
    # History must be capped at max_turns and contain only the latest 4 turns.
    assert len(sess.history) == 4
    assert [t["content"] for t in sess.history] == [
        "turn-6",
        "turn-7",
        "turn-8",
        "turn-9",
    ]
    # Subsequent adds still cap (no growth past max_turns).
    sess.add("user", "follow-up")
    assert len(sess.history) == 4
    assert sess.history[-1] == {"role": "user", "content": "follow-up"}
    assert sess.history[0]["content"] == "turn-7"


# --------------------------------------------------------------------------- #
# Registry: get_session lazy create + reset_sessions
# --------------------------------------------------------------------------- #
def test_get_session_lazy_creates_and_reuses_same_instance() -> None:
    reset_sessions()
    assert _sessions == {}
    a = get_session("reg-1")
    b = get_session("reg-1")
    # Same id -> same instance (no duplicate).
    assert a is b
    assert a.session_id == "reg-1"
    assert "reg-1" in _sessions
    # Different id -> different instance.
    c = get_session("reg-2")
    assert c is not a
    assert c.session_id == "reg-2"
    # Mutating one session must not bleed into the other.
    a.add("user", "hello")
    assert a.history == [{"role": "user", "content": "hello"}]
    assert c.history == []
    reset_sessions()


def test_get_session_default_id_is_lazy_created() -> None:
    reset_sessions()
    s = get_session()  # default id "default"
    assert s.session_id == "default"
    assert "default" in _sessions
    assert get_session() is s  # subsequent call returns the same instance.
    reset_sessions()
    assert _sessions == {}


def test_reset_sessions_clears_all_sessions() -> None:
    reset_sessions()
    get_session("a")
    get_session("b")
    assert len(_sessions) == 2
    reset_sessions()
    assert _sessions == {}
    # After reset, the same id yields a fresh instance with empty history.
    fresh = get_session("a")
    assert fresh.history == []


def test_reset_sessions_is_idempotent_when_already_empty() -> None:
    reset_sessions()
    reset_sessions()  # calling on empty registry must not raise.
    assert _sessions == {}


# --------------------------------------------------------------------------- #
# Package-level re-exports
# --------------------------------------------------------------------------- #
def test_package_reexports_conversation_symbols() -> None:
    from f1opt.feedback import ConversationSession as CS2
    from f1opt.feedback import get_session as gs2
    from f1opt.feedback import reset_sessions as rs2

    assert CS2 is ConversationSession
    assert gs2 is get_session
    assert rs2 is reset_sessions


# --------------------------------------------------------------------------- #
# Fixture: keep the registry clean across tests so they don't leak state.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clean_sessions() -> None:
    reset_sessions()
    yield
    reset_sessions()
