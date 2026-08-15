"""Natural-language generation for F1OPT driver feedback.

Turns the structured ``{name, value, evidence, advice}`` dimension entries
produced by :mod:`f1opt.feedback.engine` into coherent Chinese prose, adapts
the tone to a driver archetype, generates causal explanations for observed
handling phenomena, and manages multi-turn conversation flow.

This module is purely deterministic (no LLM calls); it complements the
optional LLM-enhancement path in :mod:`f1opt.feedback.engine` by providing
reliable rule-based narration that works fully offline.

Iter-182: add urgency labels (immediate/recommended/optional) to dimension
narrations. Each dimension now prepends a priority tag based on the time
gap from ideal, helping drivers triage the most impactful issues first.

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

_NARRATION_PRIORITY: dict[str, int] = {
    "balance": 0,
    "grip": 0,
    "tyres": 0,
    "braking": 1,
    "ers_deployment": 1,
    "aero_balance": 1,  # Iter-214
    "brake_temp": 1,  # Iter-225
    "tyre_temp_gradient": 1,  # Iter-232
    "ers_sector_efficiency": 1,  # Iter-232
    "drs_usage": 2,
    "throttle_brake_smoothness": 2,
    "confidence": 2,
    "lap_time_potential": 2,
    "sector_compare": 3,
    "setup_advice": 3,
    "corner_analysis": 3,
    "fuel_consumption": 3,  # Iter-203
    "throttle_brake_overlap": 2,  # Iter-210
}

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

_DIM_LABEL_ZH: dict[str, str] = {
    "balance": "平衡",
    "grip": "抓地力",
    "tyres": "轮胎",
    "braking": "制动",
    "ers_deployment": "ERS部署",
    "drs_usage": "DRS使用",
    "throttle_brake_smoothness": "油门刹车平顺性",
    "confidence": "操控信心",
    "lap_time_potential": "圈速潜力",
    "sector_compare": "分段对比",
    "setup_advice": "调教建议",
    "corner_analysis": "逐弯分析",
    "fuel_consumption": "燃油消耗",  # Iter-203
    "throttle_brake_overlap": "油门刹车重叠",  # Iter-210
    "aero_balance": "下压力平衡",  # Iter-214
    "ers_sector_efficiency": "ERS扇区效率",  # Iter-242
}


_URGENCY_ZH: dict[str, str] = {
    "immediate": "[紧急]",
    "recommended": "[建议]",
    "optional": "[可选]",
}

_TONE_PREFIX: dict[str, str] = {
    "AGGRESSIVE_OVERTAKER": "作为进攻型车手，",
    "AGGRESSIVE": "作为进攻型车手，",
    "DEVELOPMENT": "作为新晋车手，",
    "RACE_CRAFT": "从比赛策略角度，",
    "TIRE_WHISPERER": "作为轮胎管理专家，",
}

_GAP_RE = re.compile(r"~?\s*([0-9]+\.?[0-9]*)\s*s")


def _extract_gap(value: str) -> str | None:
    m = _GAP_RE.search(value)
    return m.group(1) if m else None


def _append_advice(base_: str, advice: str) -> str:
    if not advice:
        return base_
    return base_ + advice.rstrip("。") + "。"


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


def _narrate_ers_deployment(value: str, advice: str, evidence: str) -> str:
    base = f"ERS部署方面：{value}。"
    return _append_advice(base, advice)


def _narrate_drs_usage(value: str, advice: str, evidence: str) -> str:
    base = f"DRS使用方面：{value}。"
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


def _narrate_corner_analysis(value: str, advice: str, evidence: str) -> str:
    """Iter-201: corner-specific Chinese narration with弯道问题详情."""
    if "全过" in value or "无显著问题" in value:
        base = f"逐弯分析方面：{value}。"
    else:
        base = f"逐弯分析方面：{value}。"
    if advice and "逐弯建议" in advice:
        return f"{base}{advice}"
    return _append_advice(base, advice)


def _narrate_fuel_consumption(value: str, advice: str, evidence: str) -> str:
    """Iter-203: fuel consumption Chinese narration."""
    base = f"燃油消耗方面：{value}。"
    return _append_advice(base, advice)


def _narrate_throttle_brake_overlap(value: str, advice: str, evidence: str) -> str:
    """Iter-210: throttle/brake overlap Chinese narration."""
    base = f"油门刹车重叠方面：{value}。"
    return _append_advice(base, advice)


def _narrate_aero_balance(value: str, advice: str, evidence: str) -> str:
    """Iter-214: aero balance Chinese narration."""
    v = value.lower()
    if "aero-dominant" in v:
        base = "下压力平衡方面：气动主导, 高速下压力充足但低速机械抓地力不足。"
    elif "mechanical-dominant" in v:
        base = "下压力平衡方面：机械主导, 低速抓地力好但高速下压力不足。"
    else:
        base = f"下压力平衡方面：{value}。"
    return _append_advice(base, advice)


def _narrate_ers_sector_efficiency(value: str, advice: str, evidence: str) -> str:
    """Iter-242: ERS sector efficiency Chinese narration."""
    base = f"ERS扇区效率方面：{value}。"
    return _append_advice(base, advice)


_NARRATORS_ZH: dict[str, Any] = {
    "balance": _narrate_balance,
    "grip": _narrate_grip,
    "tyres": _narrate_tyres,
    "braking": _narrate_braking,
    "ers_deployment": _narrate_ers_deployment,
    "drs_usage": _narrate_drs_usage,
    "throttle_brake_smoothness": _narrate_smoothness,
    "confidence": _narrate_confidence,
    "lap_time_potential": _narrate_lap_time,
    "sector_compare": _narrate_sector,
    "setup_advice": _narrate_setup_advice,
    "corner_analysis": _narrate_corner_analysis,
    "fuel_consumption": _narrate_fuel_consumption,  # Iter-203
    "throttle_brake_overlap": _narrate_throttle_brake_overlap,  # Iter-210
    "aero_balance": _narrate_aero_balance,  # Iter-214
    "ers_sector_efficiency": _narrate_ers_sector_efficiency,  # Iter-242
}


class FeedbackNarrator:
    """Turn structured feedback dimensions into natural-language prose."""

    def __init__(self, language: str = "zh") -> None:
        self.language = language

    def narrate_dimension(self, dim: dict) -> str:
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

    @staticmethod
    def _delta_from_dim(dim: dict) -> float:
        """Extract the time gap from a dimension dict (Iter-182).

        Tries ``value`` string numeric extraction first (e.g. ``"~0.8s above"``)
        then falls back to ``delta_from_ideal`` key. Returns 0.0 when undetectable.
        """
        v = str(dim.get("value", ""))
        m = _GAP_RE.search(v)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return float(dim.get("delta_from_ideal", 0.0) or 0.0)

    @staticmethod
    def urgency_label(delta_from_ideal: float) -> str:
        """Classify urgency from the time gap to ideal (Iter-182).

        Returns ``"immediate"`` (>0.5s), ``"recommended"`` (>0.1s), ``"optional"`` otherwise.
        """
        if delta_from_ideal > 0.5:
            return "immediate"
        if delta_from_ideal > 0.1:
            return "recommended"
        return "optional"

    def narrate_all(self, dimensions: list[dict]) -> str:
        """Concatenate dimension narrations with urgency labels (Iter-182)."""
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
            delta_val = self._delta_from_dim(dim)
            urgency = self.urgency_label(delta_val)
            tag = _URGENCY_ZH.get(urgency, "")
            prefixed = f"{tag}{sentence}" if tag else sentence
            if i == 0:
                trans = "首先"
            elif i == 1:
                trans = "其次"
            elif i == n - 1:
                trans = "最后"
            else:
                trans = "此外"
            parts.append(f"{trans}，{prefixed}")
        return "".join(parts)

    def narrate_all_with_urgency(self, dimensions: list[dict]) -> str:
        """Enhanced narration with detailed urgency lines (Iter-182).

        Each dimension is preceded by a standalone urgency line:
        ``首先，[紧急] 圈速潜力：immediate (+0.80s)。...``
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
            delta_val = self._delta_from_dim(dim)
            urgency = self.urgency_label(delta_val)
            tag = _URGENCY_ZH.get(urgency, "")
            label = _DIM_LABEL_ZH.get(str(dim.get("name", "")), "")
            urgency_line = f"{tag} {label}：{urgency} ({delta_val:+.2f}s)" if tag else ""
            if i == 0:
                trans = "首先"
            elif i == 1:
                trans = "其次"
            elif i == n - 1:
                trans = "最后"
            else:
                trans = "此外"
            if urgency_line:
                parts.append(f"{trans}，{urgency_line}。{sentence}")
            else:
                parts.append(f"{trans}，{sentence}")
        return "".join(parts)

    def narrate_setup_change(self, suggestion: dict) -> str:
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
        dimensions = (feedback or {}).get("dimensions", []) or []
        suggestions = (feedback or {}).get("setup_suggestions", []) or []

        sentences: list[str] = []

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

        issues: list[str] = []
        for name in ("balance", "tyres", "braking"):
            d = next((dd for dd in dimensions if dd.get("name") == name), None)
            if d and d.get("value") and "数据不足" not in str(d.get("value")):
                issues.append(str(d["value"]))
        if issues:
            sentences.append(f"主要问题：{'；'.join(issues[:2])}。")
        else:
            sentences.append("未发现明显异常。")

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


class ToneAdapter:
    """Adjust feedback tone per driver archetype."""

    def __init__(self, archetype: str) -> None:
        self.archetype = archetype

    def prefix(self) -> str:
        return _TONE_PREFIX.get(self.archetype, "")

    def adapt(self, text: str) -> str:
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


class ExplanationGenerator:
    """Generate causal explanations for observed handling phenomena."""

    def __init__(self) -> None:
        pass

    def explain_why(self, observation: str, metrics: dict) -> str:
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
        if "ers" in obs_l or "电池" in observation or "能量" in observation:
            return self._explain_ers(m)
        if "下压力" in observation or "downforce" in obs_l or "气动" in observation:
            return self._explain_downforce_balance(m)
        if "抓地" in observation or "grip" in obs_l or "机械" in observation:
            return self._explain_mech_grip(m)
        if "刹车温度" in observation or "brake temp" in obs_l or "制动温度" in observation:
            return self._explain_brake_temp(m)
        if "ers扇区" in observation or "ers sector" in obs_l or "扇区ers" in observation or "sector ers" in obs_l:
            return self._explain_ers_sector_efficiency(m)
        if "油耗" in observation or "燃油" in observation or "fuel" in obs_l or "lift" in obs_l:
            return self._explain_fuel_consumption(m)
        if "重叠" in observation or "overlap" in obs_l or "同时踩" in observation:
            return self._explain_throttle_brake_overlap(m)
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

    def _explain_ers(self, m: dict) -> str:
        """Iter-207: ERS efficiency explanation."""
        parts = ["ERS问题的原因是：部署与回收策略不匹配。"]
        slope = m.get("ers_slope_per_s") or m.get("ers_trend")
        if slope is not None:
            try:
                s = float(slope)
                if s < -0.5:
                    parts.append(f"ERS消耗速率 {s:.2f}/s, 部署过于激进。")
                elif s < 0:
                    parts.append(f"ERS消耗速率 {s:.2f}/s, 部署略高于回收。")
                else:
                    parts.append(f"ERS回收速率 {s:.2f}/s, 回收正常。")
            except (TypeError, ValueError):
                pass
        store = m.get("ers_store_mean")
        if store is not None:
            try:
                store_val = float(store)
                parts.append(f"当前平均电量 {store_val:.0f}%。")
            except (TypeError, ValueError):
                pass
        parts.append("建议出弯部署 Hotlap 模式, 直道末端回收 MGU-K。")
        return "".join(parts)

    def _explain_downforce_balance(self, m: dict) -> str:
        """Iter-213: 下压力平衡 (aero balance) 中文解释."""
        parts = ["下压力问题的原因是：前后气动平衡不匹配。"]
        aero_ratio = m.get("aero_balance_ratio")
        diag = m.get("diagnosis") or ""
        if aero_ratio is not None:
            try:
                r = float(aero_ratio)
                if r > 1.8:
                    parts.append(
                        f"高速下压力比低速大 {r:.1f} 倍, 气动主导。"
                        "机械抓地力不足, 建议软化悬挂或增加前束角。"
                    )
                elif r < 1.1:
                    parts.append(
                        f"高低速下压力比仅 {r:.2f}, 机械主导。"
                        "高速下压力不足, 建议增加前后翼角度。"
                    )
                else:
                    parts.append(
                        f"气动/机械平衡良好 (ratio={r:.2f})。"
                    )
            except (TypeError, ValueError):
                pass
        else:
            parts.append(diag if diag else "请提供前后翼设置和高速弯g_lat数据。")
        if m.get("front_wing") is not None:
            parts.append(
                f"前翼 {m['front_wing']} 档, 后翼 {m.get('rear_wing', '?')} 档。"
            )
        parts.append("建议根据赛道类型调整前后翼比例以优化气动效率。")
        return "".join(parts)

    def _explain_mech_grip(self, m: dict) -> str:
        """Iter-218: 机械抓地力趋势中文解释."""
        parts = ["机械抓地力问题的原因是：低速弯横向G力在衰退。"]
        trend = m.get("mech_grip_trend") or ""
        slope = m.get("mech_grip_slope")
        if trend == "decaying":
            parts.append(
                "低速弯 g_lat 随圈数呈下降趋势, 轮胎可能过热或产生 graining。"
            )
            if slope is not None:
                try:
                    s = float(slope)
                    parts.append(f"衰退斜率 {s:.4f} G/m, 机械抓地力持续减弱。")
                except (TypeError, ValueError):
                    pass
            parts.append(
                "建议降低胎压 0.3 psi 或减少入弯速度以保护轮胎表面。"
            )
        elif trend == "improving":
            parts.append("低速弯 g_lat 趋势向好, 轮胎正在进入最佳工作窗口。")
        elif trend == "stable":
            parts.append("低速弯机械抓地力保持稳定, 轮胎工作状态良好。")
        else:
            parts.append("暂无足够低速弯数据进行趋势分析。")
        return "".join(parts)

    def _explain_brake_temp(self, m: dict) -> str:
        """Iter-225: 刹车温度平衡中文解释."""
        parts = ["刹车温度问题的原因是：前后刹车热负荷不均衡。"]
        bt_ratio = m.get("brake_temp_f_r_ratio")
        bt_diag = m.get("brake_temp_balance_diag") or ""
        if bt_ratio is not None:
            try:
                r = float(bt_ratio)
                if r > 1.3:
                    parts.append(
                        f"前刹车温度是后刹的 {r:.1f} 倍 (front-bias), "
                        "前刹承受过多热负荷, 可能导致前轮刹车衰减。"
                    )
                    parts.append(
                        "建议后移刹车偏置 1-2% 以将热负荷分配至后轴。"
                    )
                elif r < 0.7:
                    parts.append(
                        f"后刹车温度显著高于前刹 (ratio={r:.2f}), "
                        "后刹过热可能导致制动不稳定。"
                    )
                    parts.append(
                        "建议前移刹车偏置 1-2% 以平衡热负荷。"
                    )
                else:
                    parts.append(
                        f"前后刹车温度分布均匀 (ratio={r:.2f}), 刹车偏置设置合理。"
                    )
            except (TypeError, ValueError):
                pass
        else:
            parts.append(bt_diag if bt_diag else "请提供刹车温度数据以进行诊断。")
        return "".join(parts)

    def _explain_ers_sector_efficiency(self, m: dict) -> str:
        """Iter-242: ERS扇区效率中文解释."""
        parts = ["ERS扇区效率问题的原因是：部署与回收在不同扇区不匹配。"]
        sector_ers = m.get("ers_sector_efficiency")
        if sector_ers and isinstance(sector_ers, list):
            worst_sector = -1
            worst_eff = 1.0
            details: list[str] = []
            for i, se in enumerate(sector_ers):
                eff = se.get("efficiency", 0.0)
                dep = se.get("deploy_total", 0.0)
                rec = se.get("recover_total", 0.0)
                details.append(f"S{i+1}: deploy={dep:.1f}, recover={rec:.1f}, eff={eff:.2f}")
                if eff < worst_eff:
                    worst_eff = eff
                    worst_sector = i
            if worst_sector >= 0 and worst_eff < 0.5:
                parts.append(
                    f"S{worst_sector+1} ERS效率最低 (eff={worst_eff:.2f}), "
                    "部署量远大于回收量, 电池在该扇区过度消耗。"
                )
                parts.append(
                    f"建议S{worst_sector+1}减少Hotlap模式时长, 增加MGU-K回收。"
                )
            elif worst_sector >= 0:
                parts.append("各扇区ERS效率基本均衡, 部署策略合理。")
            parts.append("各扇区详情: " + "; ".join(details) + "。")
        elif m.get("ers_sector_worst") is not None:
            worst = m["ers_sector_worst"]
            parts.append(f"最差扇区S{worst+1}, 建议优化该扇区ERS部署策略。")
        else:
            parts.append("请提供各扇区ERS部署数据以进行诊断。")
        return "".join(parts)

    def _explain_fuel_consumption(self, m: dict) -> str:
        """Iter-247: 燃油消耗中文解释."""
        parts = ["燃油消耗问题的原因是：油门输入和ERS部署策略影响油耗。"]
        fuel_used = m.get("fuel_used")
        fuel_rate = m.get("fuel_consumption_rate_kg_per_km")
        if fuel_used is not None:
            try:
                fu = float(fuel_used)
                parts.append(f"当前燃油消耗 {fu:.3f} kg/圈。")
            except (TypeError, ValueError):
                pass
        if fuel_rate is not None:
            try:
                fr = float(fuel_rate)
                if fr > 0.35:
                    parts.append(
                        f"油耗率 {fr:.4f} kg/km 偏高, 超过目标 0.33 kg/km。"
                        "建议 lift-and-coast 或提前升档以节省燃油。"
                    )
                elif fr < 0.28:
                    parts.append(
                        f"油耗率 {fr:.4f} kg/km 偏低, 低于目标 0.33 kg/km。"
                        "如有余量可提高 ERS 部署功率或增加引擎出力。"
                    )
                else:
                    parts.append(
                        f"油耗率 {fr:.4f} kg/km 在目标范围内。"
                    )
            except (TypeError, ValueError):
                pass
        highest_sector = m.get("fuel_highest_sector")
        if highest_sector is not None and highest_sector >= 0:
            parts.append(f"S{highest_sector+1} 油耗最高, 建议在该扇区 lift-and-coast。")
        if fuel_used is None and fuel_rate is None:
            parts.append("请提供燃油数据以进行诊断。")
        return "".join(parts)

    def _explain_throttle_brake_overlap(self, m: dict) -> str:
        """Iter-249: 油门刹车重叠中文解释."""
        parts = ["油门刹车重叠的原因是：同时踩下油门和刹车, 浪费燃油且磨损刹车。"]
        overlap_count = m.get("throttle_brake_overlap_count")
        overlap_pct = m.get("throttle_brake_overlap_pct")
        overlap_duration = m.get("overlap_duration_s")
        if overlap_count is not None:
            try:
                oc = int(overlap_count)
                if oc == 0:
                    parts.append("当前未检测到油门刹车重叠, 驾驶习惯良好。")
                else:
                    parts.append(f"检测到 {oc} 帧油门刹车重叠。")
                    if overlap_pct is not None:
                        try:
                            op = float(overlap_pct)
                            parts.append(f"重叠比例 {op*100:.1f}%。")
                        except (TypeError, ValueError):
                            pass
                    if overlap_duration is not None:
                        try:
                            od = float(overlap_duration)
                            parts.append(f"重叠总时长 {od:.2f}s。")
                        except (TypeError, ValueError):
                            pass
                    parts.append(
                        "建议在刹车前完全松开油门, 避免左脚刹车时右脚仍带油门。"
                    )
            except (TypeError, ValueError):
                pass
        else:
            parts.append("请提供油门刹车重叠数据以进行诊断。")
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
        if "推头" in problem or "understeer" in problem.lower():
            return [
                "增加前翼下压力 2-3 档以提升前轴抓地",
                "降低前胎压 0.2-0.4 psi 以增大接地面积",
                "适当提前刹车点，让前轮有更多时间建立抓地",
            ]
        if "甩尾" in problem or "oversteer" in problem.lower():
            return [
                "增加后翼下压力 2-3 档以稳定后轴",
                "软化后防倾杆 1-2 档以增加后轴机械抓地",
                "收窄 off-throttle diff 5-10% 以稳定入弯尾部",
            ]
        if "锁死" in problem or "lockup" in problem.lower():
            return [
                "后移前制动分配 1-2% 以减少前轮锁死风险",
                "降低制动压力 2-5% 以增加制动线性度",
                "尝试更早更轻的制动方式而非急刹",
            ]
        return ["请提供更具体的车辆行为描述以便精准定位问题。"]


class ConversationFlow:
    """Multi-turn conversation flow manager."""

    def __init__(self) -> None:
        self._turn = 0

    def opening(self, driver_name: str = "") -> str:
        self._turn = 1
        if driver_name:
            return f"{driver_name}，随时为你分析遥测数据。有什么需要？"
        return "随时为你分析遥测数据。有什么需要？"

    def acknowledge(self) -> str:
        self._turn += 1
        return "收到。"

    def transition(self, from_topic: str = "", to_topic: str = "") -> str:
        self._turn += 1
        if from_topic and to_topic:
            return f"关于{from_topic}还有什么问题吗？我们可以转到{to_topic}。"
        return "还有什么需要分析的吗？"

    def closing(self) -> str:
        self._turn += 1
        return "随时呼唤，持续关注你的表现。"
