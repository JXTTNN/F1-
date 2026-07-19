"""Tests for conversation context window management (Iter-150)."""
from __future__ import annotations

from f1opt.feedback.conversation import (
    conversation_token_usage,
    estimate_tokens,
    trim_conversation_history,
)


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_english(self) -> None:
        t = estimate_tokens("hello world")
        assert t > 0

    def test_chinese(self) -> None:
        t = estimate_tokens("你好世界")
        assert t == 4  # 4 CJK chars = ~4 tokens

    def test_mixed(self) -> None:
        t = estimate_tokens("前翼 angle +2 deg")
        assert t > 0

    def test_long_text(self) -> None:
        t = estimate_tokens("前翼角度增加2度，后翼角度减少1度，胎压调整0.5 bar")
        assert t > 10


class TestTrimConversationHistory:
    def test_no_trim_needed(self) -> None:
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = trim_conversation_history(history, max_tokens=10000)
        assert len(result) == 2

    def test_empty_history(self) -> None:
        result = trim_conversation_history([], max_tokens=100)
        assert result == []

    def test_preserve_system_message(self) -> None:
        history = [
            {"role": "system", "content": "You are an F1 engineer."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = trim_conversation_history(history, max_tokens=10000)
        assert result[0]["role"] == "system"

    def test_trim_large_history(self) -> None:
        # Create a long history that needs trimming
        history = [{"role": "system", "content": "F1 engineer system prompt"}] + [
            {"role": "user", "content": "Msg" + str(i) * 10}
            for i in range(20)
        ]
        result = trim_conversation_history(
            history, max_tokens=500, preserve_system=True, min_recent=3
        )
        # Should at least preserve system + last 3 messages
        assert len(result) >= 4

    def test_trim_removes_middle(self) -> None:
        history = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "X" * 200},
            {"role": "assistant", "content": "Y" * 200},
            {"role": "user", "content": "Z" * 200},
            {"role": "assistant", "content": "W" * 200},
            {"role": "user", "content": "recent1"},
            {"role": "assistant", "content": "recent2"},
            {"role": "user", "content": "recent3"},
        ]
        result = trim_conversation_history(
            history, max_tokens=100, preserve_system=True, min_recent=3
        )
        # Should have system + last 3
        assert len(result) <= 4
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "recent3"

    def test_extreme_case_protected_exceeds_budget(self) -> None:
        # Even protected messages (system + last 3) exceed budget
        history = [
            {"role": "system", "content": "S" * 30},
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "C" * 200},
            {"role": "assistant", "content": "D" * 200},
        ]
        result = trim_conversation_history(
            history, max_tokens=50, preserve_system=True, min_recent=3
        )
        # Should still return system (fits in budget)
        assert len(result) >= 1
        assert result[0]["role"] == "system"

    def test_no_system_preserve(self) -> None:
        history = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "X" * 500},
            {"role": "assistant", "content": "Y" * 500},
            {"role": "user", "content": "recent"},
        ]
        result = trim_conversation_history(
            history, max_tokens=50, preserve_system=False, min_recent=1
        )
        # Should trim system if not preserved
        assert len(result) <= 1 or result[0]["role"] != "system"


class TestConversationTokenUsage:
    def test_basic_usage(self) -> None:
        history = [
            {"role": "system", "content": "F1 engineer"},
            {"role": "user", "content": "前翼角度"},
            {"role": "assistant", "content": "建议调整 +2"},
        ]
        usage = conversation_token_usage(history)
        assert usage["message_count"] == 3
        assert usage["total"] > 0
        assert usage["system"] > 0
        assert usage["user"] > 0
        assert usage["assistant"] > 0

    def test_empty_history(self) -> None:
        usage = conversation_token_usage([])
        assert usage["total"] == 0
        assert usage["message_count"] == 0

    def test_unknown_role(self) -> None:
        history = [{"role": "unknown", "content": "test"}]
        usage = conversation_token_usage(history)
        assert usage["total"] > 0  # Still counts tokens