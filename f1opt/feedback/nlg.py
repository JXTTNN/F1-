"""Natural-language generation for F1OPT driver feedback.

Turns the structured ``{name, value, evidence, advice}`` dimension entries
produced by :mod:`f1opt.feedback.engine` into coherent Chinese prose, adapts
the tone to a driver archetype, generates causal explanations for observed
handling phenomena, and manages multi-turn conversation flow.

This module is purely deterministic (no LLM calls); it complements the
optional LLM-enhancement path in :mod:`f1opt.feedback.engine` by providing
reliable rule-based narration that works fully offline.

Public classes:

- :class:`FeedbackNarrator` — narrate dimensions / setup changes / sessions.
- :class:`ToneAdapter` — adjust phrasing per driver archetype.
- :class:`ExplanationGenerator` — causal "why" + "how to fix" explanations.
- :class:`ConversationFlow` — opening / acknowledge / transition / closing.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "ConversationFlow",
    "ExplanationGenerator",
    "FeedbackNarrator",
    "ToneAdapter",
]


#: Dimension narration priority (lower = narrated first). Dimensions absent
#: from this map default to tier 9. Spec ordering:
#:   balance/grip/tyres -> braking/ers_drs -> confidence/lap_time_potential ->
#:   sector_compare/setup_advice.
_NARRATION_PRIORITY: dict[str, int] = {
    "balance": 0,
    "grip": 0,
    "tyres": 0,
    "braking": 1,
    "ers_drs": 1,
    "throttle_brake_smoothness": 2,
    "confidence": 2,
    "lap_time_potential": 2,
    "sector_compare": 3,
    "setup_advice": 3,
}

#: Short Chinese names for the 19 setup fields (used by narrate_setup_change).
_PARAM_ZH: dict[str, str] = {
    "front_wing": "前翼",
    "rear_wing": "后翼",
    "on_throttle_diff": "油门差速器",
    "off_throttle_diff": "收油差速器",
    "front_camber": "前轮外倾角",
    "rear_camber": "后轮外倾角",
    "front_toe": "前轮前束",
    "rear_toe": "后轮前束",
    "front_suspension": "前弹簧",
    "rear_suspension": "后弹簧",
    "front_arb": "前防倾杆",
    "rear_arb": "后防倾杆",
    "front_ride_height": "前离地间隙",
    "rear_ride_height": "后离地间隙",
    "brake_pressure": "制动压力",
    "front_brake_bias": "前制动分配",
    "front_tyre_pressure": "前轮胎压",
    "rear_tyre_pressure": "后轮胎压",
    "fuel_load": "燃油装载量",
}

#: Chinese labels for the 10 feedback dimensions.
_DIM_LABEL_ZH: dict[str, str] = {
    "balance": "平衡",
    "grip": "抓地力",
    "tyres": "轮胎",
    "braking": "制动",
    "ers_drs": "ERS/DRS",
    "throttle_brake_smoothness": "油门刹车平顺性",
    "confidence": "操控信心",
    "lap_time_potential": "圈速潜力",
    "sector_compare": "分段对比",
    "setup_advice": "调教建议",
}

#: Opening-line prefix per driver archetype.
_TONE_PREFIX: dict[str, str] = {
    "AGGRESSIVE_OVERTAKER": "作为进攻型车手，",
    "AGGRESSIVE": "作为进攻型车手，",
    "DEVELOPMENT": "作为新晋车手，",
    "RACE_CRAFT": "从比赛策略角度，",
    "TIRE_WHISPERER": "作为轮胎管理专家，",
}

#: Extracts the first numeric lap-time gap (e.g. "0.5" from "~0.5s above ...").
_GAP_RE = re.compile(r"~?\s*([0-9]+\.?[0-9]*)\s*s")


def _extract_gap(value: str) -> str | None:
    """Return the first numeric gap found in ``value`` (e.g. ``"0.5"``)."""
    m = _GAP_RE.search(value)
    return m.group(1) if m else None


def _append_advice(base: str, advice: str) -> str:
    """Append an advice clause, normalising the trailing full stop."""
    if not advice:
        return base
    return base + advice.rstrip("。") + "。"


# --------------------------------------------------------------------------- #
# Per-dimension Chinese narrators (value, advice, evidence) -> sentence.
# --------------------------------------------------------------------------- #
def _narrate_balance(value: str, advice: str, evidence: str) -> str:
    v = value.lower()
    if "neutral" in v:
        base = "赛车平衡表现中性，前后轴抓地力匹配良好。"
    elif "understeer" in v or "推头" in value:
        base = "赛车在入弯阶段存在推头倾向，前轮抓地力不足。"
    elif "oversteer" in v or "甩尾" in value:
        base = "赛车存在甩尾倾向，后轮抓地力不足。"
    else:
        base = f"平衡方面：{value}。"
    return _append_advice(base, advice)


def _narrate_grip(value: str, advice: str, evidence: str) -> str:
    base = f"整体抓地力方面：{value}。"
    return _append_advice(base, advice)


def _narrate_tyres(value: str, advice: str, evidence: str) -> str:
    base = f"轮胎方面：{value}。请关注轮胎温度与磨损管理。"
    return _append_advice(base, advice)


def _narrate_braking(value: str, advice: str, evidence: str) -> str:
    v = value.lower()
    if "lockup" in v or "锁死" in value:
        base = f"制动方面存在锁死风险({value})。"
    else:
        base = f"制动方面：{value}。"
    return _append_advice(base, advice)


def _narrate_ers_drs(value: str, advice: str, evidence: str) -> str:
    base = f"ERS/DRS 使用方面：{value}。"
    return _append_advice(base, advice)


def _narrate_smoothness(value: str, advice: str, evidence: str) -> str:
    base = f"油门与刹车平顺性方面：{value}。"
    return _append_advice(base, advice)


def _narrate_confidence(value: str, advice: str, evidence: str) -> str:
    base = f"操控信心方面：{value}。"
    return _append_advice(base, advice)


def _narrate_lap_time(value: str, advice: str, evidence: str) -> str:
    v = value.lower()
    gap = _extract_gap(value)
    if "above reference" in v and gap:
        base = f"当前圈速距离参考节奏约{gap}秒，仍有提升空间。"
    elif "under reference" in v and gap:
        base = f"圈速表现优于参考节奏约{gap}秒，节奏良好。"
    else:
        base = f"圈速潜力方面：{value}。"
    return _append_advice(base, advice)


def _narrate_sector(value: str, advice: str, evidence: str) -> str:
    base = f"分段对比方面：{value}。"
    return _append_advice(base, advice)


def _narrate_setup_advice(value: str, advice: str, evidence: str) -> str:
    base = f"调教建议方面：{value}。"
    return _append_advice(base, advice)


#: Dispatch table mapping dimension name -> narrator function.
_NARRATORS_ZH: dict[str, Any] = {
    "balance": _narrate_balance,
    "grip": _narrate_grip,
    "tyres": _narrate_tyres,
    "braking": _narrate_braking,
    "ers_drs": _narrate_ers_drs,
    "throttle_brake_smoothness": _narrate_smoothness,
    "confidence": _narrate_confidence,
    "lap_time_potential": _narrate_lap_time,
    "sector_compare": _narrate_sector,
    "setup_advice": _narrate_setup_advice,
}


# --------------------------------------------------------------------------- #
# FeedbackNarrator
# --------------------------------------------------------------------------- #
class FeedbackNarrator:
    """Turn structured feedback dimensions into natural-language prose.

    Parameters
    ----------
    language
        ``"zh"`` (default, full coverage) or ``"en"`` (basic English fallback).
    """

    def __init__(self, language: str = "zh") -> None:
        self.language = language

    def narrate_dimension(self, dim: dict) -> str:
        """Narrate a single ``{name, value, evidence, advice}`` dimension.

        Returns a Chinese sentence describing the dimension. Handles the
        ``数据不足`` (insufficient data) marker and unknown dimension names
        gracefully.
        """
        if not dim:
            return ""
        name = str(dim.get("name", ""))
        value = str(dim.get("value", ""))
        advice = dim.get("advice") or ""
        evidence = dim.get("evidence") or ""

        if self.language != "zh":
            return self._narrate_dimension_en(name, value, advice)

        label = _DIM_LABEL_ZH.get(name, name)
        if not value or value == "数据不足":
            return f"{label}数据不足，暂无法给出定量结论。"

        handler = _NARRATORS_ZH.get(name)
        if handler is None:
            return f"{label}方面：{value}。"
        return handler(value, advice, evidence)

    @staticmethod
    def _narrate_dimension_en(name: str, value: str, advice: str) -> str:
        if not value or value == "数据不足":
            return f"{name}: insufficient data."
        base = f"{name}: {value}."
        if advice:
            base += f" Advice: {advice}."
        return base

    def narrate_all(self, dimensions: list[dict]) -> str:
        """Concatenate dimension narrations into a coherent paragraph.

        Dimensions are ordered by :data:`_NARRATION_PRIORITY`
        (balance/grip/tyres first ... sector_compare/setup_advice last) and
        joined with Chinese transitions (首先 / 其次 / 此外 / 最后). The
        original relative order is preserved within a priority tier (stable
        sort). Returns ``""`` for an empty list.
        """
        if not dimensions:
            return ""
        ordered = sorted(
            enumerate(dimensions),
            key=lambda pair: (
                _NARRATION_PRIORITY.get(pair[1].get("name", ""), 9),
                pair[0],
            ),
        )
        n = len(ordered)
        parts: list[str] = []
        for i, (_, dim) in enumerate(ordered):
            sentence = self.narrate_dimension(dim)
            if not sentence:
                continue
            if i == 0:
                trans = "首先"
            elif i == 1:
                trans = "其次"
            elif i == n - 1:
                trans = "最后"
            else:
                trans = "此外"
            parts.append(f"{trans}，{sentence}")
        return "".join(parts)

    def narrate_setup_change(self, suggestion: dict) -> str:
        """Narrate a setup suggestion.

        Accepts ``{pname, before, after, expected_gain, reason}`` (also falls
        back to a ``name`` key for ``pname``). Produces prose such as
        ``"建议将前翼从25调至28(变化+3档)，预计~0.3s/lap。原因：..."``.
        """
        pname = suggestion.get("pname") or suggestion.get("name") or "参数"
        before = suggestion.get("before")
        after = suggestion.get("after")
        gain = suggestion.get("expected_gain", "") or ""
        reason = suggestion.get("reason", "") or ""
        pname_zh = _PARAM_ZH.get(str(pname), str(pname))

        delta_str = ""
        try:
            delta = float(after) - float(before)
            if delta != 0:
                delta_str = f"(变化{delta:+g}档)"
        except (TypeError, ValueError):
            pass

        parts = [f"建议将{pname_zh}从{before}调至{after}{delta_str}"]
        parts.append(f"，预计{gain}。" if gain else "。")
        if reason:
            parts.append(f"原因：{reason}。")
        return "".join(parts)

    def summarize_session(self, feedback: dict) -> str:
        """2-3 sentence executive summary of a feedback session.

        Covers the lap-time gap, 1-2 key issues (balance/tyres/braking), and
        the top setup recommendation. Always returns exactly 3 sentences.
        """
        dimensions = (feedback or {}).get("dimensions", []) or []
        suggestions = (feedback or {}).get("setup_suggestions", []) or []

        sentences: list[str] = []

        # Sentence 1: lap-time potential.
        lap_dim = next(
            (d for d in dimensions if d.get("name") == "lap_time_potential"), None
        )
        if (
            lap_dim
            and lap_dim.get("value")
            and "数据不足" not in str(lap_dim.get("value"))
        ):
            sentences.append(f"圈速方面，{lap_dim['value']}。")
        else:
            sentences.append("圈速数据暂不可用。")

        # Sentence 2: key issues (balance / tyres / braking).
        issues: list[str] = []
        for name in ("balance", "tyres", "braking"):
            d = next((dd for dd in dimensions if dd.get("name") == name), None)
            if d and d.get("value") and "数据不足" not in str(d.get("value")):
                issues.append(str(d["value"]))
        if issues:
            sentences.append(f"主要问题：{'；'.join(issues[:2])}。")
        else:
            sentences.append("未发现明显异常。")

        # Sentence 3: top recommendation.
        if suggestions:
            s = suggestions[0]
            pname = s.get("pname") or s.get("name") or "参数"
            pname_zh = _PARAM_ZH.get(str(pname), str(pname))
            sentences.append(
                f"建议优先调整{pname_zh}({s.get('before')}→{s.get('after')})。"
            )
        else:
            sentences.append("暂无调教建议。")

        return "".join(sentences)


# --------------------------------------------------------------------------- #
# ToneAdapter
# --------------------------------------------------------------------------- #
class ToneAdapter:
    """Adjust feedback tone per driver archetype.

    Parameters
    ----------
    archetype
        One of ``"AGGRESSIVE_OVERTAKER"`` (or ``"AGGRESSIVE"``),
        ``"DEVELOPMENT"`` (rookie), ``"RACE_CRAFT"``, ``"TIRE_WHISPERER"``.
        Unknown archetypes fall back to a neutral tone.
    """

    def __init__(self, archetype: str) -> None:
        self.archetype = archetype

    def prefix(self) -> str:
        """Opening line per archetype (empty for unknown archetypes)."""
        return _TONE_PREFIX.get(self.archetype, "")

    def adapt(self, text: str) -> str:
        """Adjust phrasing per archetype.

        - AGGRESSIVE: direct, confident (prefix only).
        - DEVELOPMENT (rookie): explanatory, gentle, invites questions.
        - RACE_CRAFT: balanced, data-focused.
        - TIRE_WHISPERER: emphasises tyre preservation.
        """
        pre = self.prefix()
        if self.archetype in ("AGGRESSIVE_OVERTAKER", "AGGRESSIVE"):
            return f"{pre}{text}"
        if self.archetype == "DEVELOPMENT":
            return f"{pre}我们来一起分析：{text}如有疑问随时沟通。"
        if self.archetype == "RACE_CRAFT":
            return f"{pre}从数据来看，{text}"
        if self.archetype == "TIRE_WHISPERER":
            if "胎" in text:
                return f"{pre}{text}"
            return f"{pre}注意保护轮胎。{text}"
        return f"{pre}{text}"


# --------------------------------------------------------------------------- #
# ExplanationGenerator
# --------------------------------------------------------------------------- #
class ExplanationGenerator:
    """Generate causal explanations for observed handling phenomena."""

    def __init__(self) -> None:
        pass

    def explain_why(self, observation: str, metrics: dict) -> str:
        """Generate a multi-sentence causal explanation for ``observation``.

        ``observation`` is a Chinese handling term (推头/甩尾/锁死/胎...) or
        English equivalent (understeer/oversteer/lockup/tyre). ``metrics`` is a
        flat dict of supporting measurements (e.g. ``{"front_wing": 22}``).
        Always returns a non-empty multi-sentence string even when ``metrics``
        is empty.
        """
        m = metrics or {}
        obs_l = observation.lower()
        if "推头" in observation or "understeer" in obs_l:
            return self._explain_understeer(m)
        if "甩尾" in observation or "oversteer" in obs_l:
            return self._explain_oversteer(m)
        if "锁死" in observation or "lockup" in obs_l:
            return self._explain_lockup(m)
        if "胎" in observation or "tyre" in obs_l or "tire" in obs_l:
            return self._explain_tyre(m)
        return self._explain_generic(observation, m)

    def _explain_understeer(self, m: dict) -> str:
        parts: list[str] = []
        fw = m.get("front_wing") or m.get("front_wing_level")
        if fw is not None:
            parts.append(
                f"推头现象的原因是：前翼下压力不足(当前{fw}档)，"
                "导致入弯时前轮抓地力不够。"
            )
        else:
            parts.append("推头现象的原因是：前翼下压力不足，导致入弯时前轮抓地力不够。")
        temp = (
            m.get("front_tyre_temp")
            or m.get("tyre_temp_fl")
            or m.get("front_tyre_temp_fl")
        )
        if temp is not None:
            try:
                t = float(temp)
                if t < 90:
                    parts.append(f"同时前胎温度偏低({t:g}°C)也降低了前轮机械抓地力。")
                else:
                    parts.append(f"同时前胎温度偏高({t:g}°C)影响了前轮工作窗口。")
            except (TypeError, ValueError):
                pass
        parts.append("建议增加前翼下压力以提升前轴抓地。")
        return "".join(parts)

    def _explain_oversteer(self, m: dict) -> str:
        parts = ["甩尾现象的原因是：后轴抓地力不足。"]
        rw = m.get("rear_wing") or m.get("rear_wing_level")
        if rw is not None:
            parts.append(f"后翼下压力(当前{rw}档)可能偏低。")
        parts.append("建议提高后翼或软化后防倾杆以稳定后轴。")
        return "".join(parts)

    def _explain_lockup(self, m: dict) -> str:
        parts = ["制动锁死的原因是：制动力分配偏前或制动压力过高。"]
        bb = m.get("front_brake_bias") or m.get("brake_bias")
        if bb is not None:
            parts.append(f"当前前制动分配为{bb}%，可能偏前。")
        parts.append("建议后移前制动分配或降低制动压力。")
        return "".join(parts)

    def _explain_tyre(self, m: dict) -> str:
        parts = ["轮胎问题的原因是：胎温或磨损偏离工作窗口。"]
        temp = m.get("tyre_temp_fl") or m.get("front_tyre_temp")
        if temp is not None:
            parts.append(f"前胎温度为{temp}°C。")
        parts.append("建议调整胎压或驾驶节奏以恢复轮胎工作区间。")
        return "".join(parts)

    def _explain_generic(self, observation: str, m: dict) -> str:
        parts = [
            f"关于「{observation}」：根据当前遥测，"
            "主要原因是车辆平衡与抓地力匹配不足。"
        ]
        if m:
            keys = list(m.keys())[:3]
            parts.append(f"相关指标包括{'、'.join(keys)}。")
        parts.append("建议结合调教与驾驶风格进一步定位。")
        return "".join(parts)

    def explain_how_to_fix(self, problem: str, setup: dict) -> list[str]:
        """Return a list of Chinese action items for ``problem``.

        ``problem`` is a Chinese handling term (推头/甩尾/锁死...) or English
        equivalent. ``setup`` is the current setup dict (used to compute
        before→after deltas when the relevant field is present).
        """
        s = setup or {}
        prob_l = problem.lower()
        if "推头" in problem or "understeer" in prob_l:
            return self._fix_understeer(s)
        if "甩尾" in problem or "oversteer" in prob_l:
            return self._fix_oversteer(s)
        if "锁死" in problem or "lockup" in prob_l:
            return self._fix_lockup(s)
        return self._fix_generic(s)

    def _fix_understeer(self, s: dict) -> list[str]:
        actions: list[str] = []
        fw = s.get("front_wing")
        if fw is not None:
            try:
                new_fw = float(fw) + 2
                actions.append(f"增加前翼2档({fw:g}→{new_fw:g})")
            except (TypeError, ValueError):
                actions.append("增加前翼下压力2档")
        else:
            actions.append("增加前翼下压力2档")
        actions.append("检查前胎气压，可适当降低0.2bar")
        actions.append("如持续推头，软化前防倾杆1档")
        return actions

    def _fix_oversteer(self, s: dict) -> list[str]:
        actions: list[str] = []
        rw = s.get("rear_wing")
        if rw is not None:
            try:
                new_rw = float(rw) + 2
                actions.append(f"增加后翼2档({rw:g}→{new_rw:g})")
            except (TypeError, ValueError):
                actions.append("增加后翼下压力2档")
        else:
            actions.append("增加后翼下压力2档")
        actions.append("软化后防倾杆1档以释放车尾")
        actions.append("适当降低on_throttle_diff以减少后轮打滑")
        return actions

    def _fix_lockup(self, s: dict) -> list[str]:
        actions: list[str] = []
        bb = s.get("front_brake_bias")
        if bb is not None:
            try:
                new_bb = float(bb) - 1
                actions.append(f"后移前制动分配1档({bb:g}→{new_bb:g})")
            except (TypeError, ValueError):
                actions.append("后移前制动分配1档")
        else:
            actions.append("后移前制动分配1档")
        actions.append("降低制动压力2-3%")
        actions.append("采用渐近刹车，避免瞬时满刹")
        return actions

    @staticmethod
    def _fix_generic(s: dict) -> list[str]:
        return [
            "复核车辆平衡，确认前后轴抓地匹配",
            "检查轮胎温度与磨损是否处于工作窗口",
            "结合赛道类型调整下压力级别",
        ]

    def rank_explanations(
        self, explanations: list[str], relevance: list[float]
    ) -> list[str]:
        """Sort explanations by relevance descending.

        Ties are broken by the explanation text (ascending) for determinism.
        Returns a new list; inputs are not mutated. When the two lists differ
        in length, the shorter one bounds the result (excess entries ignored).
        """
        n = min(len(explanations), len(relevance))
        paired = sorted(
            zip(explanations[:n], relevance[:n], strict=True),
            key=lambda pair: (-float(pair[1]), pair[0]),
        )
        return [e for e, _ in paired]


# --------------------------------------------------------------------------- #
# ConversationFlow
# --------------------------------------------------------------------------- #
class ConversationFlow:
    """Manage multi-turn feedback conversation flow."""

    def __init__(self) -> None:
        pass

    def opening(self, track_id: str, driver_name: str = "车手") -> str:
        """Opening greeting with track context.

        Resolves ``track_id`` to the circuit name via
        :func:`f1opt.data.tracks.get_track` (lazy import); falls back to the
        raw ``track_id`` when the track is unknown.
        """
        track_name = self._track_name(track_id)
        return (
            f"{driver_name}你好，欢迎来到{track_name}。"
            "让我们基于本次遥测数据分析你的驾驶表现，并给出针对性的调教建议。"
        )

    def acknowledge(self, user_input: str) -> str:
        """Acknowledge the driver's question or concern."""
        if not user_input:
            return "好的，让我分析一下当前数据。"
        return f"好的，关于「{user_input}」，让我结合遥测数据分析一下。"

    def transition(self, from_topic: str, to_topic: str) -> str:
        """Smooth transition between topics (mentions both topics)."""
        return f"关于{from_topic}的分析就到这里，接下来我们看一下{to_topic}的情况。"

    def closing(self) -> str:
        """Closing summary + next steps."""
        return (
            "以上是本次遥测反馈的全部内容。建议先落实高优先级调教，"
            "并在下一 stint 复盘关键指标变化。如需进一步分析，随时沟通。"
        )

    def clarify(self, ambiguous_input: str) -> str:
        """Ask for clarification on an ambiguous input (ends with a question)."""
        return (
            f"你提到的「{ambiguous_input}」我需要进一步确认——"
            "你是指入弯阶段还是出弯阶段的表现？请补充更多细节。"
        )

    @staticmethod
    def _track_name(track_id: str) -> str:
        """Resolve ``track_id`` to a circuit name; fall back to the id."""
        try:
            from f1opt.data.tracks import get_track

            return get_track(track_id).circuit_name
        except Exception:
            return track_id
