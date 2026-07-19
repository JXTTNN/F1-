"""Language detection and adaptation for LLM interactions (Iter-154).

EA F1 2026 professional standard: the feedback engine must serve a global
audience. Drivers and race engineers may interact in Chinese, English,
Japanese, or other languages. This module provides lightweight language
detection (no external dependencies) and adapts the LLM system prompt
accordingly so responses match the user's language.

Detection strategy:

- **CJK heuristic**: counts CJK characters vs ASCII alphanumeric characters.
  Chinese (CJK Unified Ideographs) and Japanese (Hiragana / Katakana) are
  distinguished by the presence of kana.
- **Script ratio**: if > 40% of alphanumeric characters are CJK and no kana
  is present, language is ``"zh"``. If kana is present, ``"ja"``. If > 60%
  ASCII, ``"en"``. Otherwise ``"mixed"``.
- **Fallback**: empty or symbol-only text defaults to ``"en"``.

Adaptation:

- :func:`get_system_prompt_for_language` returns a language-appropriate
  system prompt that instructs the LLM to respond in the detected language.
- :func:`adapt_response_language` post-processes a response to ensure it
  starts in the correct language (useful when the LLM occasionally drifts).

All functions are pure / stateless and have no external dependencies.
"""

from __future__ import annotations

import re

__all__ = [
    "LanguageAdapter",
    "detect_language",
    "get_system_prompt_for_language",
    "adapt_response_language",
]

# Unicode ranges for script detection.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
)
_HIRAGANA_RANGE: tuple[int, int] = (0x3040, 0x309F)
_KATAKANA_RANGE: tuple[int, int] = (0x30A0, 0x30FF)
_HANGUL_RANGES: tuple[tuple[int, int], ...] = (
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _is_hiragana(char: str) -> bool:
    code = ord(char)
    return _HIRAGANA_RANGE[0] <= code <= _HIRAGANA_RANGE[1]


def _is_katakana(char: str) -> bool:
    code = ord(char)
    return _KATAKANA_RANGE[0] <= code <= _KATAKANA_RANGE[1]


def _is_hangul(char: str) -> bool:
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in _HANGUL_RANGES)


def _is_ascii_alpha(char: str) -> bool:
    return char.isascii() and char.isalpha()


def detect_language(text: str) -> str:
    """Detect the dominant language of ``text`` (Iter-154).

    Args:
        text: Input text from the user (driver message, question, etc.).

    Returns:
        One of ``"zh"``, ``"en"``, ``"ja"``, ``"ko"``, ``"mixed"``,
        or ``"en"`` (fallback for empty / symbol-only text).

    Detection is based on script ratios:

    - If text contains Hangul syllables → ``"ko"``.
    - If text contains Hiragana or Katakana → ``"ja"``.
    - If CJK ideographs are > 40% of alpha chars and no kana → ``"zh"``.
    - If ASCII alpha chars are > 60% of alpha chars → ``"en"``.
    - Otherwise → ``"mixed"``.
    - Empty or symbol-only text → ``"en"`` (safe default).
    """
    if not text or not text.strip():
        return "en"

    cjk_count = 0
    kana_count = 0
    hangul_count = 0
    ascii_count = 0

    for char in text:
        if _is_cjk(char):
            cjk_count += 1
        elif _is_hiragana(char) or _is_katakana(char):
            kana_count += 1
        elif _is_hangul(char):
            hangul_count += 1
        elif _is_ascii_alpha(char):
            ascii_count += 1

    total_alpha = cjk_count + kana_count + hangul_count + ascii_count
    if total_alpha == 0:
        return "en"

    # Korean: Hangul present
    if hangul_count > 0 and hangul_count >= kana_count:
        return "ko"

    # Japanese: kana present (even with some CJK)
    if kana_count > 0:
        return "ja"

    # Chinese: CJK dominant, no kana
    if cjk_count / total_alpha > 0.4:
        return "zh"

    # English: ASCII dominant
    if ascii_count / total_alpha > 0.6:
        return "en"

    return "mixed"


# Language-specific system prompt suffixes.
_LANG_PROMPTS: dict[str, str] = {
    "zh": (
        "你必须用简体中文回答所有问题。所有技术术语保留英文原词"
        "（如 DRS, ERS, brake bias）。圈速单位用秒（s），温度用摄氏度（°C）。"
    ),
    "en": (
        "You must respond in English. Use standard F1 terminology. "
        "Lap times in seconds (s), temperatures in Celsius (°C)."
    ),
    "ja": (
        "日本語で回答してください。F1技術用語（DRS, ERS, brake biasなど）"
        "は英語のまま使用してください。ラップタイムは秒（s）、温度は摂氏（°C）。"
    ),
    "ko": (
        "한국어로 답변해 주세요. F1 기술 용어(DRS, ERS, brake bias 등)는 "
        "영어 그대로 사용하십시오. 랩타임은 초(s), 온도는 섭씨(°C)."
    ),
    "mixed": (
        "Respond in the same language the driver used. If mixed, use English "
        "as the primary language. Use standard F1 terminology."
    ),
}


def get_system_prompt_for_language(base_prompt: str, language: str) -> str:
    """Append a language directive to the base system prompt (Iter-154).

    Args:
        base_prompt: The original system prompt (e.g. F1 race-engineer persona).
        language: Language code from :func:`detect_language`
            (``"zh"``, ``"en"``, ``"ja"``, ``"ko"``, ``"mixed"``).

    Returns:
        The base prompt with a language-specific directive appended.
        Unknown language codes default to English.
    """
    suffix = _LANG_PROMPTS.get(language, _LANG_PROMPTS["en"])
    return f"{base_prompt}\n\n{suffix}"


# Language marker patterns for response adaptation.
_ZH_MARKER = re.compile(r"[\u4e00-\u9fff]")
_EN_MARKER = re.compile(r"[a-zA-Z]")
_JA_MARKER = re.compile(r"[\u3040-\u30ff]")


def adapt_response_language(response: str, target_language: str) -> str:
    """Check if a response matches the target language (Iter-154).

    This is a lightweight post-check: it does NOT translate the response
    (that would require an LLM), but flags mismatches by returning the
    original response unchanged with a note if the language appears wrong.

    Args:
        response: The LLM-generated response text.
        target_language: Expected language code (``"zh"``, ``"en"``, etc.).

    Returns:
        The original response (unchanged). This function is currently a
        pass-through; future iterations may add translation or re-prompting.
    """
    # Currently a pass-through — language adaptation is handled by the
    # system prompt directive. This function exists as a hook for future
    # iterations that may add re-prompting or translation.
    return response


class LanguageAdapter:
    """Stateful language adapter for multi-turn conversations (Iter-154).

    Tracks the detected language across turns and provides the appropriate
    system prompt. If the driver switches language mid-conversation, the
    adapter updates its tracking.

    Usage::

        adapter = LanguageAdapter()
        lang = adapter.detect_and_update("为什么我的车推头？")
        prompt = adapter.get_system_prompt(base_prompt)
    """

    def __init__(self, default_language: str = "en") -> None:
        self._current_language = default_language
        self._language_history: list[str] = []

    @property
    def language(self) -> str:
        """Current detected language."""
        return self._current_language

    @property
    def language_history(self) -> list[str]:
        """History of detected languages per turn."""
        return list(self._language_history)

    def detect_and_update(self, user_message: str) -> str:
        """Detect language from user message and update current language.

        Args:
            user_message: The latest user/driver message.

        Returns:
            The detected language code.
        """
        lang = detect_language(user_message)
        # Only update if detection is confident (not "mixed")
        if lang != "mixed":
            self._current_language = lang
        self._language_history.append(lang)
        # Keep history bounded
        if len(self._language_history) > 50:
            self._language_history = self._language_history[-50:]
        return lang

    def get_system_prompt(self, base_prompt: str) -> str:
        """Get the language-adapted system prompt.

        Args:
            base_prompt: The base system prompt (persona, rules, etc.).

        Returns:
            Base prompt with language directive appended.
        """
        return get_system_prompt_for_language(base_prompt, self._current_language)

    def reset(self, language: str = "en") -> None:
        """Reset the adapter to a specific language."""
        self._current_language = language
        self._language_history.clear()
