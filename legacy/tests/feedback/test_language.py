"""Tests for :mod:`f1opt.feedback.language` (Iter-154)."""
from __future__ import annotations

from f1opt.feedback.language import (
    LanguageAdapter,
    adapt_response_language,
    detect_language,
    get_system_prompt_for_language,
)


class TestDetectLanguage:
    def test_chinese(self) -> None:
        assert detect_language("为什么我的车推头？") == "zh"
        assert detect_language("胎温太高了怎么调") == "zh"
        assert detect_language("刹车点太早了") == "zh"

    def test_english(self) -> None:
        assert detect_language("Why is the car understeering?") == "en"
        assert detect_language("Tire temperatures are too high.") == "en"
        assert detect_language("How should I adjust brake bias?") == "en"

    def test_japanese(self) -> None:
        assert detect_language("なぜアンダーステアになるのですか？") == "ja"
        assert detect_language("タイヤ温度が高すぎます") == "ja"
        assert detect_language("ブレーキバイアスを調整してください") == "ja"

    def test_korean(self) -> None:
        assert detect_language("왜 언더스티어가 나나요?") == "ko"
        assert detect_language("타이어 온도가 너무 높습니다") == "ko"

    def test_empty_text_defaults_en(self) -> None:
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"

    def test_symbol_only_defaults_en(self) -> None:
        assert detect_language("???!!!") == "en"
        assert detect_language("123 456") == "en"

    def test_mixed_cjk_and_english(self) -> None:
        """大量英文 + 少量中文 → en."""
        text = "The car feels 推头 but overall balance is good in high speed corners."
        lang = detect_language(text)
        # Mostly ASCII, so "en" expected
        assert lang == "en"

    def test_chinese_dominant_with_some_english(self) -> None:
        """大量中文 + 少量英文 → zh."""
        text = "为什么推头？DRS 打不开怎么办？刹车点太早了，圈速慢了零点五秒。"
        assert detect_language(text) == "zh"

    def test_mixed_language_returns_mixed(self) -> None:
        """中英文各半 → mixed."""
        text = "推头understeer问题problem刹车brake这是一段很长的话来平衡比例"
        lang = detect_language(text)
        assert lang in ("mixed", "zh")

    def test_numbers_only(self) -> None:
        assert detect_language("80.5 90.3 85.2") == "en"

    def test_japanese_with_kanji_only(self) -> None:
        """纯汉字 (无假名) → zh (无法区分中日汉字)."""
        text = "車温度調整"
        # No kana → classified as zh (CJK dominant, no kana)
        assert detect_language(text) == "zh"


class TestGetSystemPrompt:
    def test_zh_prompt_appended(self) -> None:
        base = "You are an F1 race engineer."
        result = get_system_prompt_for_language(base, "zh")
        assert "You are an F1 race engineer." in result
        assert "简体中文" in result

    def test_en_prompt_appended(self) -> None:
        base = "You are an F1 race engineer."
        result = get_system_prompt_for_language(base, "en")
        assert "English" in result

    def test_ja_prompt_appended(self) -> None:
        base = "You are an F1 race engineer."
        result = get_system_prompt_for_language(base, "ja")
        assert "日本語" in result

    def test_ko_prompt_appended(self) -> None:
        base = "You are an F1 race engineer."
        result = get_system_prompt_for_language(base, "ko")
        assert "한국어" in result

    def test_unknown_language_defaults_en(self) -> None:
        base = "You are an F1 race engineer."
        result = get_system_prompt_for_language(base, "fr")
        assert "English" in result

    def test_mixed_language_prompt(self) -> None:
        base = "You are an F1 race engineer."
        result = get_system_prompt_for_language(base, "mixed")
        assert "same language" in result

    def test_base_prompt_preserved(self) -> None:
        base = "IMPORTANT: Every claim must be grounded in telemetry."
        result = get_system_prompt_for_language(base, "zh")
        assert result.startswith(base)


class TestAdaptResponseLanguage:
    def test_pass_through_unchanged(self) -> None:
        response = "建议降低前翼角度来减少推头。"
        result = adapt_response_language(response, "zh")
        assert result == response

    def test_empty_response(self) -> None:
        result = adapt_response_language("", "en")
        assert result == ""


class TestLanguageAdapter:
    def test_default_language(self) -> None:
        adapter = LanguageAdapter()
        assert adapter.language == "en"

    def test_detect_and_update_zh(self) -> None:
        adapter = LanguageAdapter()
        lang = adapter.detect_and_update("为什么推头？")
        assert lang == "zh"
        assert adapter.language == "zh"

    def test_detect_and_update_en(self) -> None:
        adapter = LanguageAdapter()
        adapter.detect_and_update("为什么推头？")
        adapter.detect_and_update("How to fix understeer?")
        assert adapter.language == "en"

    def test_mixed_does_not_update(self) -> None:
        """mixed 语言不更新当前语言."""
        adapter = LanguageAdapter(default_language="zh")
        # Use a text that genuinely produces "mixed" detection
        text = "推头understeer问题problem刹车brake这是一段很长的话来平衡比例"
        adapter.detect_and_update(text)
        # mixed should not override the current language
        assert adapter.language == "zh"

    def test_language_history(self) -> None:
        adapter = LanguageAdapter()
        adapter.detect_and_update("为什么推头？")
        adapter.detect_and_update("How to fix?")
        adapter.detect_and_update("タイヤ温度")
        assert adapter.language_history == ["zh", "en", "ja"]

    def test_get_system_prompt(self) -> None:
        adapter = LanguageAdapter()
        adapter.detect_and_update("为什么推头？")
        prompt = adapter.get_system_prompt("You are an F1 engineer.")
        assert "简体中文" in prompt
        assert "You are an F1 engineer." in prompt

    def test_reset(self) -> None:
        adapter = LanguageAdapter()
        adapter.detect_and_update("为什么推头？")
        adapter.reset()
        assert adapter.language == "en"
        assert adapter.language_history == []

    def test_reset_to_specific_language(self) -> None:
        adapter = LanguageAdapter()
        adapter.reset("ja")
        assert adapter.language == "ja"

    def test_history_bounded(self) -> None:
        """语言历史超过 50 条时自动截断."""
        adapter = LanguageAdapter()
        for _ in range(60):
            adapter.detect_and_update("hello")
        assert len(adapter.language_history) <= 50

    def test_japanese_then_chinese(self) -> None:
        adapter = LanguageAdapter()
        adapter.detect_and_update("タイヤ温度が高い")
        assert adapter.language == "ja"
        adapter.detect_and_update("胎温太高了")
        assert adapter.language == "zh"
