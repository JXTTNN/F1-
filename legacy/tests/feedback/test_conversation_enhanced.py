"""Tests for the Iter-25..28 conversation-memory enhancements.

Covers the new :class:`~f1opt.feedback.conversation.ConversationSession`
methods (``remember_setup_change`` / ``get_setup_history``,
``summarize_focus``, ``is_followup``, ``resolve_reference``) and the new
:class:`~f1opt.feedback.conversation.ConversationMemory` class
(add/recent/topics/to_dict + bounded trimming).

These are pure unit tests; no feedback engine or surrogate is exercised.
"""

from __future__ import annotations

import pytest

from f1opt.feedback import ConversationMemory, ConversationSession, reset_sessions


# --------------------------------------------------------------------------- #
# remember_setup_change / get_setup_history
# --------------------------------------------------------------------------- #
def test_remember_setup_change_and_get_history() -> None:
    sess = ConversationSession(session_id="s1")
    sess.add("user", "前翼怎么调")
    sess.remember_setup_change("front_wing", 25, 27, "缓解推头")
    hist = sess.get_setup_history()
    assert len(hist) == 1
    entry = hist[0]
    assert entry["field"] == "front_wing"
    assert entry["before"] == 25.0
    assert entry["after"] == 27.0
    assert entry["reason"] == "缓解推头"
    # turn is the 1-based index at proposal time (1 prior turn -> turn 2).
    assert entry["turn"] == 2


def test_get_setup_history_returns_independent_copy() -> None:
    sess = ConversationSession(session_id="s2")
    sess.remember_setup_change("rear_wing", 27, 30, "平衡")
    hist = sess.get_setup_history()
    hist.append({"field": "evil", "before": 0, "after": 0, "reason": "", "turn": 99})
    hist[0]["field"] = "mutated"
    # internal log must NOT be affected by caller mutation.
    internal = sess.get_setup_history()
    assert internal[0]["field"] == "rear_wing"
    assert len(internal) == 1


# --------------------------------------------------------------------------- #
# summarize_focus
# --------------------------------------------------------------------------- #
def test_summarize_focus_with_keywords() -> None:
    sess = ConversationSession(session_id="s3")
    sess.add("user", "为什么推头")
    sess.add("assistant", "前轴抓地不足")
    sess.add("user", "前轮胎温偏高")
    summary = sess.summarize_focus()
    assert summary.startswith("本次对话焦点: ")
    assert "推头" in summary
    assert "胎" in summary


def test_summarize_focus_empty_session() -> None:
    sess = ConversationSession(session_id="s4")
    summary = sess.summarize_focus()
    assert "暂无" in summary


def test_summarize_focus_no_keyword_match() -> None:
    sess = ConversationSession(session_id="s5")
    sess.add("user", "你好")
    summary = sess.summarize_focus()
    assert "无明显" in summary


# --------------------------------------------------------------------------- #
# is_followup
# --------------------------------------------------------------------------- #
def test_is_followup_true_for_followup_patterns() -> None:
    sess = ConversationSession(session_id="s6")
    assert sess.is_followup("怎么样") is True
    assert sess.is_followup("为什么这样") is True
    assert sess.is_followup("再调一下") is True
    assert sess.is_followup("继续说") is True
    assert sess.is_followup("那个怎么处理") is True


def test_is_followup_false_for_fresh_question() -> None:
    sess = ConversationSession(session_id="s7")
    # "怎么调" without 样 is a fresh how-to question, NOT a follow-up.
    assert sess.is_followup("前翼怎么调") is False
    assert sess.is_followup("胎压多少") is False
    assert sess.is_followup("你好") is False


# --------------------------------------------------------------------------- #
# resolve_reference
# --------------------------------------------------------------------------- #
def test_resolve_reference_replaces_demonstrative() -> None:
    sess = ConversationSession(session_id="s8")
    sess.add("user", "前翼角度怎么样")
    sess.add("assistant", "前翼当前 25 clicks")
    # "这个" should resolve to the prior noun "前翼".
    resolved = sess.resolve_reference("这个再调高一点")
    assert resolved == "前翼再调高一点"


def test_resolve_reference_replaces_na_with_prior_noun() -> None:
    sess = ConversationSession(session_id="s9")
    sess.add("user", "后翼下压力够吗")
    resolved = sess.resolve_reference("那个怎么样")
    assert resolved == "后翼怎么样"


def test_resolve_reference_no_prior_returns_unchanged() -> None:
    sess = ConversationSession(session_id="s10")
    # no history -> nothing to resolve against
    assert sess.resolve_reference("这个再调高") == "这个再调高"


def test_resolve_reference_no_demonstrative_returns_unchanged() -> None:
    sess = ConversationSession(session_id="s11")
    sess.add("user", "前翼角度怎么样")
    assert sess.resolve_reference("前翼再调高") == "前翼再调高"


# --------------------------------------------------------------------------- #
# ConversationMemory
# --------------------------------------------------------------------------- #
def test_conversation_memory_add_and_recent() -> None:
    mem = ConversationMemory(max_turns=20)
    mem.add_user_message("为什么推头")
    mem.add_bot_message("前轴抓地不足")
    mem.add_user_message("前翼怎么调")
    recent = mem.recent_user_messages(3)
    assert recent == ["为什么推头", "前翼怎么调"]


def test_conversation_memory_recent_returns_all_when_fewer() -> None:
    mem = ConversationMemory()
    mem.add_user_message("only one")
    # asking for 10 but only 1 present -> returns all (no padding).
    assert mem.recent_user_messages(10) == ["only one"]


def test_conversation_memory_topics() -> None:
    mem = ConversationMemory()
    mem.add_user_message("推头怎么解决")
    mem.add_user_message("推头严重，前轮胎温也高")
    topics = mem.topics()
    assert isinstance(topics, list)
    assert "推头" in topics
    # most-discussed keyword ranks first (推头 x2 > 胎 x1).
    assert topics[0] == "推头"


def test_conversation_memory_to_dict() -> None:
    mem = ConversationMemory(max_turns=5)
    mem.add_user_message("u1")
    mem.add_bot_message("b1")
    d = mem.to_dict()
    assert d["max_turns"] == 5
    assert d["user_messages"] == ["u1"]
    assert d["bot_messages"] == ["b1"]


def test_conversation_memory_bounded_max_turns() -> None:
    mem = ConversationMemory(max_turns=3)
    for i in range(5):
        mem.add_user_message(f"u{i}")
    # trimmed to the 3 most-recent.
    assert len(mem.recent_user_messages(10)) == 3
    assert mem.recent_user_messages(10) == ["u2", "u3", "u4"]
    # to_dict also reflects the bound.
    assert len(mem.to_dict()["user_messages"]) == 3


def test_conversation_memory_rejects_invalid_max_turns() -> None:
    with pytest.raises(ValueError):
        ConversationMemory(max_turns=0)


def test_conversation_memory_reexported_from_package() -> None:
    import f1opt.feedback as fb

    assert fb.ConversationMemory is ConversationMemory


# --------------------------------------------------------------------------- #
# Fixture: keep the process-wide session registry clean across tests.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clean_sessions() -> None:
    reset_sessions()
    yield
    reset_sessions()
