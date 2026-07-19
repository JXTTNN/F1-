"""Tests for :mod:`f1opt.feedback.intent` (Iter-158)."""
from __future__ import annotations

from f1opt.feedback.intent import IntentResult, classify_intent


class TestSetupAdvice:
    def test_chinese_setup_advice(self) -> None:
        assert classify_intent("怎么调整前翼来减少推头？").intent == "setup_advice"
        assert classify_intent("应该如何调教刹车？").intent == "setup_advice"
        assert classify_intent("怎么改能更快？").intent == "setup_advice"

    def test_english_setup_advice(self) -> None:
        assert classify_intent("How should I adjust the front wing?").intent == "setup_advice"
        assert classify_intent("What setup changes do you recommend?").intent == "setup_advice"
        assert classify_intent("how to improve lap time?").intent == "setup_advice"


class TestProblemReport:
    def test_chinese_problem(self) -> None:
        assert classify_intent("车推头很严重").intent == "problem_report"
        assert classify_intent("甩尾了").intent == "problem_report"
        assert classify_intent("胎温太高了").intent == "problem_report"
        assert classify_intent("没有抓地力").intent == "problem_report"

    def test_english_problem(self) -> None:
        assert classify_intent("The car is understeering badly").intent == "problem_report"
        assert classify_intent("I have no grip in the rear").intent == "problem_report"
        assert classify_intent("tyres overheating quickly").intent == "problem_report"


class TestTelemetryQuestion:
    def test_chinese_telemetry(self) -> None:
        assert classify_intent("我的圈速是多少？").intent == "telemetry_question"
        assert classify_intent("分段速度多少？").intent == "telemetry_question"

    def test_english_telemetry(self) -> None:
        assert classify_intent("What was my lap time?").intent == "telemetry_question"
        assert classify_intent("How fast was I in sector 2?").intent == "telemetry_question"


class TestStrategyQuestion:
    def test_chinese_strategy(self) -> None:
        assert classify_intent("什么时候进站？").intent == "strategy_question"
        assert classify_intent("还剩几圈？").intent == "strategy_question"

    def test_english_strategy(self) -> None:
        assert classify_intent("When should I pit?").intent == "strategy_question"
        assert classify_intent("How many laps left?").intent == "strategy_question"


class TestFeedback:
    def test_chinese_feedback(self) -> None:
        assert classify_intent("好多了，有改善").intent == "feedback"
        assert classify_intent("还是推头").intent == "feedback"

    def test_english_feedback(self) -> None:
        assert classify_intent("That worked, much better").intent == "feedback"
        assert classify_intent("Still understeering").intent == "feedback"


class TestGreeting:
    def test_chinese_greeting(self) -> None:
        assert classify_intent("你好").intent == "greeting"
        assert classify_intent("早上好").intent == "greeting"

    def test_english_greeting(self) -> None:
        assert classify_intent("hello").intent == "greeting"
        assert classify_intent("Hi there").intent == "greeting"
        assert classify_intent("good morning").intent == "greeting"


class TestStatusCheck:
    def test_chinese_status(self) -> None:
        assert classify_intent("现在状态怎么样？").intent == "status_check"
        assert classify_intent("情况如何了？").intent == "status_check"

    def test_english_status(self) -> None:
        assert classify_intent("How am I doing?").intent == "status_check"
        assert classify_intent("What's the situation?").intent == "status_check"


class TestOther:
    def test_unrecognized(self) -> None:
        assert classify_intent("xyz abc def").intent == "other"
        assert classify_intent("...").intent == "other"

    def test_empty_message(self) -> None:
        assert classify_intent("").intent == "other"
        assert classify_intent("   ").intent == "other"

    def test_numbers_only(self) -> None:
        result = classify_intent("123 456 789")
        assert result.intent in ("other", "telemetry_question")


class TestConfidence:
    def test_matched_confidence_is_one(self) -> None:
        """有匹配的模式 confidence=1.0."""
        result = classify_intent("怎么调前翼？")
        assert result.confidence == 1.0

    def test_no_match_confidence_is_zero(self) -> None:
        """无匹配 confidence=0.0."""
        result = classify_intent("xyzqwerty")
        assert result.confidence == 0.0


class TestIntentResult:
    def test_repr(self) -> None:
        result = IntentResult("setup_advice", 1.0)
        assert "setup_advice" in repr(result)
        assert "1.00" in repr(result)

    def test_equality(self) -> None:
        r1 = IntentResult("greeting", 1.0)
        r2 = IntentResult("greeting", 1.0)
        assert r1 == r2

    def test_inequality(self) -> None:
        r1 = IntentResult("greeting", 1.0)
        r2 = IntentResult("other", 0.0)
        assert r1 != r2


class TestPriority:
    def test_setup_advice_over_problem(self) -> None:
        """当消息同时包含 setup_advice 和 problem_report 关键词时, setup_advice 优先."""
        # "怎么调" (setup) + "推头" (problem) → setup_advice should win
        result = classify_intent("怎么调推头问题？")
        assert result.intent == "setup_advice"

    def test_problem_over_telemetry(self) -> None:
        """problem_report 优先于 telemetry_question."""
        # "推头" (problem) + "速度" could be telemetry → problem wins
        result = classify_intent("推头导致速度慢")
        assert result.intent == "problem_report"


# ---------------------------------------------------------------------------
# Iter-162.6: 对话式逐弯反馈盲区测试 (RED-first TDD)
# 每个测试对应 Iter-162.5 1000 条语料库测试中发现的一类盲区.
# ---------------------------------------------------------------------------

class TestConversationalCornerFeedback:
    """对话式逐弯反馈句的意图分类 (Iter-162.6).

    每个测试对应 Iter-162.5 测试揭示的一类盲区. 这些句子使用真实 F1
    无线电风格的描述, 涉及具体弯道 (T1/T12/1 号弯) 和具体动态
    (DRS/ERS/路肩/牵引/锁死等).
    """

    # --- 盲区 1: DRS 关键词缺失 -------------------------------------------
    def test_drs_not_opening_is_problem(self) -> None:
        """'DRS 没开' / 'DRS failed' 是 problem_report, 不是 other."""
        assert classify_intent("T12 DRS 没开").intent == "problem_report"
        assert classify_intent("DRS failed at Turn 5").intent == "problem_report"

    def test_drs_slow_to_react_is_problem(self) -> None:
        """'DRS 反应慢' 是 problem_report."""
        assert classify_intent("长弯 15 号弯 DRS 反应慢").intent == "problem_report"

    # --- 盲区 2: ERS 关键词缺失 -------------------------------------------
    def test_ers_deployment_weak_is_problem(self) -> None:
        """'ERS 部署不够' 是 problem_report."""
        assert classify_intent("4 号弯出弯 ERS 用完了").intent == "problem_report"
        assert classify_intent("ERS deployment out of T4 is too short").intent == "problem_report"

    def test_ers_no_boost_is_problem(self) -> None:
        """'ERS 没有 boost' 是 problem_report."""
        assert classify_intent("电池没电了, ERS 没有 boost").intent == "problem_report"

    # --- 盲区 3: 路肩冲击关键词缺失 ---------------------------------------
    def test_kerb_loose_rear_is_problem(self) -> None:
        """'压路肩车尾跳' 是 problem_report."""
        assert classify_intent("中速左手弯压路肩车身不稳").intent == "problem_report"
        assert classify_intent("over the kerbs at T3 the rear is loose").intent == "problem_report"

    def test_kerb_bottoming_is_problem(self) -> None:
        """'I'm bottoming on the kerbs' 是 problem_report."""
        assert classify_intent("I'm bottoming on the kerbs at Turn 8").intent == "problem_report"

    # --- 盲区 4: 出弯牵引关键词缺失 ---------------------------------------
    def test_exit_traction_poor_is_problem(self) -> None:
        """'出弯牵引不行' 是 problem_report."""
        assert classify_intent("T7 出弯牵引不行").intent == "problem_report"
        assert classify_intent("traction out of T9 is poor").intent == "problem_report"

    def test_no_traction_on_exit_is_problem(self) -> None:
        """'I have no traction on exit' 是 problem_report."""
        assert classify_intent("I have no traction on exit of Turn 4").intent == "problem_report"

    # --- 盲区 5: 弯道名 + 动态组合 fall through ---------------------------
    def test_corner_name_plus_understeer_is_problem(self) -> None:
        """'T3 的入弯转向感很模糊' 含弯道名 + 动态, 应为 problem_report."""
        assert classify_intent("T3 的入弯转向感很模糊").intent == "problem_report"

    def test_tyre_temp_at_corner_is_problem(self) -> None:
        """'tyre temperatures are out of control at Turn 11' 是 problem_report."""
        assert classify_intent("tyre temperatures are out of control at Turn 11").intent == "problem_report"

    def test_rotation_at_corner_is_problem(self) -> None:
        """'the car won't rotate through complex' 是 problem_report."""
        assert classify_intent("the car won't rotate through complex").intent == "problem_report"

    def test_front_lockup_at_corner_is_problem(self) -> None:
        """'I'm locking up the fronts at T1' 是 problem_report."""
        assert classify_intent("I'm locking up the fronts at T1").intent == "problem_report"

    def test_braking_rear_step_out_is_problem(self) -> None:
        """'braking into T2 the rear is stepping out' 是 problem_report."""
        assert classify_intent("braking into T2 the rear is stepping out").intent == "problem_report"

    def test_no_front_bite_is_problem(self) -> None:
        """'I have no front bite on entry to T7' 是 problem_report."""
        assert classify_intent("I have no front bite on entry to T7").intent == "problem_report"

    def test_exit_snap_is_problem(self) -> None:
        """'I had a snap on exit of Turn 5' 是 problem_report."""
        assert classify_intent("I had a snap on exit of Turn 5").intent == "problem_report"

    def test_graining_at_corner_is_problem(self) -> None:
        """'I have heavy graining at T6' 是 problem_report."""
        assert classify_intent("I have heavy graining at T6").intent == "problem_report"

    def test_car_wont_rotate_is_problem(self) -> None:
        """'the car is too pointy at T3' 是 problem_report."""
        assert classify_intent("the car is too pointy at T3").intent == "problem_report"

    # --- 盲区 6: "Engineer," 前缀误触发 greeting --------------------------
    def test_engineer_prefix_does_not_trigger_greeting(self) -> None:
        """'Engineer, ...' 开头不应被识别为 greeting.

        错误样本: 'Engineer, the exit of quick chicane is unstable on power
        application' 之前被错误地匹配为 greeting.
        """
        result = classify_intent(
            "Engineer, the exit of quick chicane is unstable on power application"
        )
        assert result.intent != "greeting"
        # 应该是 problem_report (含 unstable + power)
        assert result.intent == "problem_report"

    def test_engineer_prefix_chinese_no_greeting(self) -> None:
        """'工程师, ...' 开头的中文描述不应被识别为 greeting."""
        result = classify_intent("工程师, 高速弯前胎磨掉了")
        assert result.intent != "greeting"
        assert result.intent == "problem_report"

    # --- 盲区 7: 情绪化后缀 "这车没法开!" 误触发 status_check -------------
    def test_emotional_undrivable_is_problem(self) -> None:
        """'卧槽, 1 号弯前胎过热啊, 这车没法开!' 是 problem_report, 不是 status_check.

        错误样本之前因含 '开' 被错误匹配为 status_check (无此关键词但有 '情况'
        / '状态' 才触发).
        """
        result = classify_intent("卧槽, 1 号弯前胎过热啊, 这车没法开!")
        assert result.intent == "problem_report"

    def test_emotional_fuck_undrivable_is_problem(self) -> None:
        """'Fuck, tyre temps are too high at blind entry, the car is undrivable!'
        是 problem_report, 不是 greeting."""
        result = classify_intent(
            "Fuck, tyre temps are too high at blind entry, the car is undrivable!"
        )
        assert result.intent == "problem_report"

    # --- 盲区 8: setup_advice 含 problem 关键词被抢匹配 -------------------
    def test_setup_advice_with_problem_keyword_wins(self) -> None:
        ''''16 号弯出弯打滑, 调油门锁止还是后翼?' 是 setup_advice, 不是 problem_report.

        含 "调" + "X 还是 Y" 的调教选择问句, 应优先于 problem_report
        (问题已显式提出调教请求).
        '''
        result = classify_intent("16 号弯出弯打滑, 调油门锁止还是后翼?")
        assert result.intent == "setup_advice"

    def test_how_to_tune_diff_is_setup_advice(self) -> None:
        """'How should I tune the diff for T8?' 是 setup_advice."""
        result = classify_intent("How should I tune the diff for T8?")
        assert result.intent == "setup_advice"

    # --- 盲区 9: strategy_question 含 "几圈" 被抢匹配 ---------------------
    def test_strategy_question_how_many_laps_at_corner(self) -> None:
        """'高速度弯 8 号弯之后还有几圈?' 是 strategy_question, 不是 telemetry.

        错误样本之前因 telemetry 模式先匹配, 但 '几圈' 应明确触发 strategy.
        """
        result = classify_intent("高速度弯 8 号弯之后还有几圈?")
        assert result.intent == "strategy_question"

    def test_strategy_question_laps_left_at_corner_en(self) -> None:
        """'How many laps left after T4?' 是 strategy_question."""
        result = classify_intent("How many laps left after T4?")
        assert result.intent == "strategy_question"

    # --- 盲区 10: 短遥测句 ---------------------------------------------
    def test_short_tyre_temp_question_is_telemetry(self) -> None:
        """'Rear tyre temp at T3?' 是 telemetry_question."""
        result = classify_intent("Rear tyre temp at T3?")
        assert result.intent == "telemetry_question"

    def test_short_speed_question_is_telemetry(self) -> None:
        """'Speed at T1?' 是 telemetry_question."""
        result = classify_intent("Speed at T1?")
        assert result.intent == "telemetry_question"

    # --- 盲区 11: feedback "X 改善了" 模式 --------------------------------
    def test_lockup_improved_is_feedback(self) -> None:
        ''''T11 的前轮锁死了改善了' 是 feedback.

        含 problem 关键词 '锁死' 但后缀 '改善了' 表明是对前次调教的反馈.
        '''
        result = classify_intent("T11 的前轮锁死了改善了")
        assert result.intent == "feedback"

    def test_dynamic_improved_is_feedback(self) -> None:
        """'{corner} 的 {dynamic} 改善了' 是 feedback."""
        result = classify_intent("T3 的入弯转向改善了")
        assert result.intent == "feedback"
