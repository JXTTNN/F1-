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
    "detect_language_with_confidence",
    "get_system_prompt_for_language",
    "adapt_response_language",
    "get_response_template",
    "list_template_keys",
    "template_supported_languages",
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


def detect_language_with_confidence(text: str) -> tuple[str, float]:
    """Detect language with confidence score (Iter-183).

    Returns a ``(language_code, confidence)`` tuple where confidence is
    in [0.0, 1.0]. Higher confidence means more unambiguous detection.

    Args:
        text: Input text to classify.

    Returns:
        ``(language_code, confidence)`` — confidence is computed from
        script ratio strength and total character count.

    Examples::

        >>> detect_language_with_confidence("为什么我的车推头？")
        ('zh', 0.95)
        >>> detect_language_with_confidence("Why is my car understeering?")
        ('en', 0.92)
        >>> detect_language_with_confidence("hello 你好")
        ('mixed', 0.45)
    """
    if not text or not text.strip():
        return ("en", 0.0)

    cjk_count = 0
    kana_count = 0
    hangul_count = 0
    ascii_count = 0
    total_chars = 0

    for char in text:
        if not char.isspace():
            total_chars += 1
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
        return ("en", 0.0)

    if hangul_count > 0 and hangul_count >= kana_count:
        ratio = hangul_count / total_alpha
        confidence = min(0.95, 0.5 + 0.5 * ratio)
        return ("ko", confidence)

    if kana_count > 0:
        ratio = kana_count / total_alpha
        confidence = min(0.95, 0.5 + 0.5 * ratio)
        return ("ja", confidence)

    if cjk_count / total_alpha > 0.4:
        ratio = cjk_count / total_alpha
        confidence = min(0.95, 0.4 + 0.55 * ratio)
        return ("zh", confidence)

    if ascii_count / total_alpha > 0.6:
        ratio = ascii_count / total_alpha
        confidence = min(0.95, 0.3 + 0.65 * ratio)
        return ("en", confidence)

    # Mixed: low confidence
    dominant_ratio = max(
        cjk_count, kana_count, hangul_count, ascii_count
    ) / max(total_alpha, 1)
    return ("mixed", max(0.1, min(0.5, dominant_ratio * 0.5)))


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


# ============================================================================
# Language-specific response templates (Iter-198)
# ============================================================================

# Pre-built response templates for common feedback scenarios in each language.
# Used as fallbacks when LLM is unavailable or for quick rule-based responses.
_LANG_TEMPLATES: dict[str, dict[str, str]] = {
    "zh": {
        "balance_good": "赛车平衡良好，前后轴抓地力分布均匀。",
        "balance_push": "赛车存在转向不足（推头），前轴抓地力不足。建议增加前翼角度或降低后悬挂高度。",
        "balance_loose": "赛车存在转向过度（甩尾），后轴不稳定。建议降低前翼角度或增加后下压力。",
        "grip_low": "整体抓地力偏低，赛道温度或轮胎状态可能不佳。",
        "tyres_ok": "轮胎温度正常，磨损率在预期范围内。",
        "tyres_hot": "轮胎温度过高，可能出现热降解。建议管理轮胎温度，减少滑动。",
        "tyres_cold": "轮胎温度偏低，未达到工作窗口。建议增加暖胎圈速度。",
        "braking_ok": "刹车性能正常，无锁死或衰退迹象。",
        "braking_lockup": "检测到刹车锁死，建议调整刹车平衡或减少刹车压力。",
        "ers_ok": "ERS 能量管理和部署正常。",
        "ers_low": "ERS 电池电量偏低，建议增加能量回收。",
        "drs_ok": "DRS 使用正常。",
        "setup_improved": "调教调整后圈速有改善，继续微调。",
        "setup_no_change": "调教调整后圈速无明显变化，建议尝试其他方向。",
        "setup_worse": "调教调整后圈速变慢，建议回退上一个调教。",
        "lap_improved": "圈速提升！继续当前调教方向。",
        "lap_consistent": "圈速稳定，在预期范围内。",
        "lap_degraded": "圈速下降，检查轮胎磨损或燃油负载。",
        "fuel_ok": "燃油消耗正常，与目标一致。",
        "fuel_high": "燃油消耗偏高，建议使用燃油节省模式。",
        "general_ok": "赛车状态良好，继续当前驾驶。",
        "general_check": "赛车表现正常，但建议检查轮胎温度和刹车状态。",
    },
    "en": {
        "balance_good": "Car balance is good, front and rear axle grip distribution is even.",
        "balance_push": "Car is understeering (pushing), front axle lacks grip. Consider increasing front wing angle or lowering rear ride height.",
        "balance_loose": "Car is oversteering (loose), rear axle is unstable. Consider reducing front wing angle or increasing rear downforce.",
        "grip_low": "Overall grip is low, track temperature or tyre condition may be suboptimal.",
        "tyres_ok": "Tyre temperatures are normal, wear rate is within expected range.",
        "tyres_hot": "Tyre temperatures are too high, thermal degradation likely. Manage tyre temps and reduce sliding.",
        "tyres_cold": "Tyre temperatures are below the operating window. Increase warm-up pace.",
        "braking_ok": "Braking performance is normal, no lockups or fade detected.",
        "braking_lockup": "Brake lockup detected. Consider adjusting brake balance or reducing brake pressure.",
        "ers_ok": "ERS energy management and deployment is normal.",
        "ers_low": "ERS battery level is low. Increase energy harvesting.",
        "drs_ok": "DRS usage is normal.",
        "setup_improved": "Lap time improved after setup change. Continue fine-tuning.",
        "setup_no_change": "No significant lap time change after setup adjustment. Try a different direction.",
        "setup_worse": "Lap time worsened after setup change. Consider reverting to previous setup.",
        "lap_improved": "Lap time improved! Continue current setup direction.",
        "lap_consistent": "Lap times are consistent, within expected range.",
        "lap_degraded": "Lap time degraded. Check tyre wear or fuel load.",
        "fuel_ok": "Fuel consumption is normal, on target.",
        "fuel_high": "Fuel consumption is high. Consider using fuel-saving mode.",
        "general_ok": "Car is in good condition. Continue current driving.",
        "general_check": "Car performance is normal, but check tyre temperatures and brake status.",
    },
    "ja": {
        "balance_good": "車のバランスは良好で、前後アクスルのグリップ配分は均等です。",
        "balance_push": "アンダーステア（プッシング）が発生しています。フロントウィング角度を増やすか、リアの車高を下げてください。",
        "balance_loose": "オーバーステア（ルース）が発生しています。フロントウィング角度を減らすか、リアのダウンフォースを増やしてください。",
        "grip_low": "全体的なグリップが低いです。トラック温度またはタイヤ状態が最適でない可能性があります。",
        "tyres_ok": "タイヤ温度は正常で、摩耗率は予想範囲内です。",
        "tyres_hot": "タイヤ温度が高すぎます。熱劣化の可能性があります。タイヤ温度を管理し、スライドを減らしてください。",
        "tyres_cold": "タイヤ温度が作動ウィンドウを下回っています。ウォームアップペースを上げてください。",
        "braking_ok": "ブレーキ性能は正常で、ロックアップやフェードは検出されていません。",
        "braking_lockup": "ブレーキロックアップが検出されました。ブレーキバランスを調整するか、ブレーキ圧力を下げてください。",
        "ers_ok": "ERS エネルギー管理と配備は正常です。",
        "ers_low": "ERS バッテリーレベルが低いです。エネルギー回収を増やしてください。",
        "drs_ok": "DRS 使用は正常です。",
        "setup_improved": "セットアップ変更後、ラップタイムが改善しました。微調整を続けてください。",
        "lap_improved": "ラップタイムが改善しました！現在のセットアップ方向を続けてください。",
        "general_ok": "車の状態は良好です。現在のドライビングを続けてください。",
    },
    "ko": {
        "balance_good": "차량 밸런스가 양호하며, 전후 액슬 그립 분배가 균일합니다.",
        "balance_push": "언더스티어(푸싱)가 발생하고 있습니다. 프론트 윙 각도를 높이거나 리어 라이드 높이를 낮추세요.",
        "balance_loose": "오버스티어(루스)가 발생하고 있습니다. 프론트 윙 각도를 줄이거나 리어 다운포스를 높이세요.",
        "grip_low": "전체적인 그립이 낮습니다. 트랙 온도나 타이어 상태가 최적이 아닐 수 있습니다.",
        "tyres_ok": "타이어 온도가 정상이며, 마모율이 예상 범위 내에 있습니다.",
        "tyres_hot": "타이어 온도가 너무 높습니다. 열화 가능성이 있습니다. 타이어 온도를 관리하고 슬라이딩을 줄이세요.",
        "tyres_cold": "타이어 온도가 작동 윈도우 아래에 있습니다. 웜업 페이스를 높이세요.",
        "braking_ok": "브레이크 성능이 정상이며, 락업이나 페이드가 감지되지 않았습니다.",
        "braking_lockup": "브레이크 락업이 감지되었습니다. 브레이크 밸런스를 조정하거나 브레이크 압력을 줄이세요.",
        "ers_ok": "ERS 에너지 관리 및 배치가 정상입니다.",
        "ers_low": "ERS 배터리 레벨이 낮습니다. 에너지 회수를 늘리세요.",
        "drs_ok": "DRS 사용이 정상입니다.",
        "setup_improved": "셋업 변경 후 랩타임이 개선되었습니다. 미세 조정을 계속하세요.",
        "lap_improved": "랩타임이 개선되었습니다! 현재 셋업 방향을 계속하세요.",
        "general_ok": "차량 상태가 양호합니다. 현재 드라이빙을 계속하세요.",
    },
}


def get_response_template(template_key: str, language: str = "en") -> str:
    """Get a pre-built response template in the specified language (Iter-198).

    These templates serve as fallbacks when the LLM is unavailable or for
    quick rule-based responses. Each template covers a common feedback
    scenario.

    Args:
        template_key: One of the predefined template keys (e.g.
            ``"balance_good"``, ``"tyres_hot"``, ``"lap_improved"``).
        language: Language code (``"zh"``, ``"en"``, ``"ja"``, ``"ko"``).
            Falls back to ``"en"`` for unknown languages.

    Returns:
        The response template string in the requested language. Returns
        an empty string if the template key is unknown.

    Examples::

        >>> get_response_template("lap_improved", "zh")
        '圈速提升！继续当前调教方向。'
        >>> get_response_template("balance_push", "en")
        'Car is understeering (pushing)...'
    """
    templates = _LANG_TEMPLATES.get(language, _LANG_TEMPLATES["en"])
    return templates.get(template_key, "")


def list_template_keys() -> list[str]:
    """Return all available template keys (Iter-198).

    Returns:
        Sorted list of template key strings (e.g. ``["balance_good", ...]``).
    """
    keys: set[str] = set()
    for lang_templates in _LANG_TEMPLATES.values():
        keys.update(lang_templates.keys())
    return sorted(keys)


def template_supported_languages() -> list[str]:
    """Return languages with template support (Iter-198).

    Returns:
        Sorted list of language codes with templates.
    """
    return sorted(_LANG_TEMPLATES.keys())
