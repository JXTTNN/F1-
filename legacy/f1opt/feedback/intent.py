"""Driver intent classification for LLM interactions (Iter-158).

EA F1 2026 professional standard: the feedback engine must understand what
the driver is asking for before generating a response. This module provides
lightweight, rule-based intent classification from the driver's natural
language message, enabling the feedback engine to route the response
appropriately (e.g. setup advice vs. telemetry explanation vs. simple
acknowledgment).

Intent categories:

- ``"setup_advice"``: Driver wants setup recommendations ("how should I
  adjust...", "what changes for...").
- ``"teammate_comparison"``: Driver wants to compare with teammate or another
  car ("队友比我快多少?", "compare sector times with car #2", "为什么队友在S2更快?").
  Iter-174: new intent, routed to :mod:`f1opt.feedback.comparison`.
- ``"problem_report"``: Driver reports a handling problem ("understeer",
  "oversteer", "no grip", "tyres overheating").
- ``"telemetry_question"``: Driver asks about telemetry data ("what was my
  sector time", "how fast was I in turn 3").
- ``"strategy_question"``: Driver asks about strategy ("when should I pit",
  "how many laps left").
- ``"status_check"``: Driver checks overall status ("how am I doing",
  "what's the situation").
- ``"feedback"``: Driver gives feedback on previous advice ("that worked",
  "still pushing").
- ``"greeting"``: Opening / greeting message.
- ``"other"``: Anything that doesn't fit the above.

Detection uses keyword + pattern matching (no ML model required, works
fully offline). Supports Chinese and English inputs.
"""

from __future__ import annotations

import re

__all__ = [
    "IntentResult",
    "GranularityResult",
    "SubIntentResult",
    "classify_intent",
    "classify_granularity",
    "classify_sub_intent",
    "GRANULARITIES",
]

GRANULARITIES: tuple[str, ...] = ("corner", "sector", "overall")

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "setup_advice",
        re.compile(
            r"(?:怎么调|如何调|怎么改|如何改|应该调|建议调|setup|调整|调教建议|"
            r"how.*(should|do).*adjust|what.*change|recommend.*setup|"
            r"what.*setup|how.*improve|怎么提升|如何改善|"
            r"调.*(还是|或者)|"
            r"how.*(tune|set).*for|tune.*diff.*for|"
            r"调教|调一下|怎么改)",
            re.IGNORECASE,
        ),
    ),
    # ---- teammate_comparison: 对比队友 (Iter-174, priority 2 — above feedback/problem_report) ----
    # NOTE: must be above feedback because "比上次快" would match feedback, and
    # above problem_report because "队友S2比我快" contains "快" which falls into
    # telemetry_question patterns.
    (
        "teammate_comparison",
        re.compile(
            r"(?:"
            # 中文: 队友 / 对比 / 差距
            r"队友|对比.*(队友|车手|他|car|第[一二三四五六七八九十\d]+车)|"
            r"比.*(队友|他|其他车|别[人的])|差距.*(队友|在哪|多少)|"
            r"和.*(队友|他|car\s*\d+|#[2-9]|二号|二号车).*比|"
            r"跟.*(队友|他).*比|"
            r"为什么.*(队友|他).*比.*快|他怎么.*快|"
            r"比.*(我|自己).*快.*(多少|在哪|什么|原因)|"
            r"快.*我.*(多少|为什么|哪里)|"
            r"慢.*我.*(多少|为什么|哪里)|"
            r"和.*第[一二三四五六七八九十\d]+车.*比|"
            r"跟.*第[一二三四五六七八九十\d]+车.*比|"
            # 英文: teammate / compare / gap / delta / vs
            r"\bteammate\b|"
            r"compare.*(with|to).*(teammate|cars?\s*\d+|#[2-9])|"
            r"how.*(faster|slower).*(am|i.am|is).*(teammate|car)|"
            r"\bgap\s+(to|from|with).*(teammate|next car)|"
            r"\bdelta\s+(to|from|with).*(teammate)|"
            r"\bvs\.?\s*(teammate|team.?mate|car\s*\d+)|"
            r"car\s*\d+\s*(faster|slower|better)|"
            r"what.*(teammate|other car).*doing|"
            r"why.*(teammate|he|they).*(faster|slower|better)|"
            r"head[- ]to[- ]head|"
            r"side[- ]by[- ]side.*(compare|comparison)"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Feedback on previous advice (must be before problem_report)
    (
        "feedback",
        re.compile(
            r"(?:好多了|有改善|有效果|没用|没效果|还是.*推头|还是.*甩尾|"
            r"worked|better|worse|no change|still.*same|fixed|"
            r"that.*help|didn.*help|feel.*good|feel.*bad|"
            r"still.*understeer|still.*oversteer|"
            r"改善了|好多了|"
            r"improved|better now|fixed it)",
            re.IGNORECASE,
        ),
    ),
    # Strategy question — Iter-162.6: 提前到 telemetry 之前
    (
        "strategy_question",
        re.compile(
            r"(?:进站|换胎|策略|几圈|剩.*圈|何时|什么时候|pit|stop|strategy|"
            r"tire.*change|how.*many.*lap|when.*pit|race.*plan|"
            r"laps left|laps remaining)",
            re.IGNORECASE,
        ),
    ),
    # Problem report — describing a handling issue
    (
        "problem_report",
        re.compile(
            r"(?:推头|甩尾|甩|转向不足|转向过度|打滑|抓地力|没抓地|胎温.*高|"
            r"车尾.*松|尾.*松|后轴.*松|车尾.*滑|尾部.*滑|"
            r"温度.*高|磨损.*快|刹车.*不好|油门.*响应|ERS.*不够|动力.*不足|"
            r"understeer|oversteer|no grip|overheating|wear.*fast|"
            r"loose|tight|pushing|sliding|wheel spin|tire.*wear|"
            r"brake.*issue|throttle.*response|power.*delivery|"
            r"DRS.*(没开|不开|失灵|慢|fail|slow|late|early|反应)|"
            r"ERS.*(不够|用完|没.*boost|响应慢|部署.*不对|weak|short|empty|"
            r"slow|out)|电池.*没电|no.*ERS|out of battery|"
            r"路肩|kerb|bottoming|压路肩|过路肩|"
            r"牵引.*(不行|不够|断|差|poor|bad)|no traction|"
            r"traction.*(poor|bad|out of|inconsistent)|"
            r"can.*t.*put.*power|wheelspin|wheel spin|spinning up|"
            r"锁死|lockup|locking up|lock up|locked up|"
            r"\bsnap\b|甩了一下|差点失控|lost it|"
            r"graining|起粒|掉得快|磨损.*快|胎.*平了|胎.*磨掉|磨平|gone|"
            r"rotate|rotation|pointy|旋转|转不过来|"
            r"bite|biting|不咬|没咬|floating|漂浮|"
            r"undrivable|没法开|没法驾驶|开不了|"
            r"unstable|nervous|不稳|晃|跳|"
            r"stepping out|step out|rear.*moving|"
            r"模糊|感觉钝|heavy|lazy|vague|numb|"
            r"out of control|失控|飙|过热|烫|"
            r"balance.*off|平衡.*不好|不平衡)",
            re.IGNORECASE,
        ),
    ),
    # Telemetry question — asking about data
    (
        "telemetry_question",
        re.compile(
            r"(?:分段|圈速|速度|多少|数据|遥测|sector|lap.?time|speed|"
            r"what.*time|how.*fast|telemetry|data|"
            r"我的.*多少|多快|多高|多少度|"
            r"(speed|temp|temperature|rpm|g-?force|pressure).*(at|@)\s*"
            r"(t\b|turn|T\d|\d+\s*号弯)|"
            r"(tyre|tire).*(temp|temperature).*(at|@))",
            re.IGNORECASE,
        ),
    ),
    # Greeting (Iter-162.6: 英文加 \b 词边界避免 "hi" 匹配 "chicane"/"high")
    (
        "greeting",
        re.compile(
            r"(?:你好|嗨|嘿|早上好|下午好|晚上好|"
            r"\bhello\b|\bhi\b|\bhey\b|\bgood (morning|afternoon|evening)\b)",
            re.IGNORECASE,
        ),
    ),
    # Status check — asking about overall situation
    (
        "status_check",
        re.compile(
            r"(?:怎么样|状态|情况|如何了|how.*doing|how.*going|what.*situation|"
            r"status|overview|summary|怎么样了)",
            re.IGNORECASE,
        ),
    ),
]


class IntentResult:
    """Result of intent classification (Iter-158).

    Attributes:
        intent: The classified intent label (e.g. ``"setup_advice"``).
        confidence: Confidence score in [0, 1] (based on pattern match
            strength). 1.0 = exact keyword match, 0.5 = partial match.
        matched_pattern: The regex pattern that matched (for debugging).
    """

    __slots__ = ("intent", "confidence", "matched_pattern")

    def __init__(self, intent: str, confidence: float, matched_pattern: str = "") -> None:
        self.intent = intent
        self.confidence = confidence
        self.matched_pattern = matched_pattern

    def __repr__(self) -> str:
        return f"IntentResult(intent={self.intent!r}, confidence={self.confidence:.2f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntentResult):
            return NotImplemented
        return self.intent == other.intent and abs(self.confidence - other.confidence) < 1e-6


def classify_intent(message: str) -> IntentResult:
    """Classify the driver's intent from their message (Iter-158).

    Args:
        message: The driver's natural language input (Chinese or English).

    Returns:
        :class:`IntentResult` with the classified intent and confidence.
        Returns ``IntentResult("other", 0.0)`` if no pattern matches.

    Examples::

        >>> classify_intent("怎么调整前翼来减少推头？")
        IntentResult(intent='setup_advice', confidence=1.0)
        >>> classify_intent("我的圈速是多少？")
        IntentResult(intent='telemetry_question', confidence=1.0)
        >>> classify_intent("hello")
        IntentResult(intent='greeting', confidence=1.0)
        >>> classify_intent("队友比我快多少?")
        IntentResult(intent='teammate_comparison', confidence=1.0)
    """
    if not message or not message.strip():
        return IntentResult("other", 0.0)

    # Iter-183: Try all patterns, collect confidence scores.
    # First match wins (highest priority), but we also look for secondary
    # matches to calibrate confidence when multiple intents overlap.
    matches: list[tuple[str, float, str]] = []
    for intent, pattern in _INTENT_PATTERNS:
        match = pattern.search(message)
        if match:
            matches.append((intent, 1.0, match.group()))

    if not matches:
        return IntentResult("other", 0.0)

    if len(matches) == 1:
        return IntentResult(
            intent=matches[0][0],
            confidence=matches[0][1],
            matched_pattern=matches[0][2],
        )

    # Multiple intents matched: primary = first (highest priority),
    # confidence reduced by number of competing matches.
    primary = matches[0]
    confidence = max(0.5, 1.0 - 0.1 * (len(matches) - 1))
    return IntentResult(
        intent=primary[0],
        confidence=confidence,
        matched_pattern=primary[2],
    )


# ============================================================================
# Granularity classification (Iter-171)
# ============================================================================

_GRANULARITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "corner",
        re.compile(
            r"(?:"
            r"\bT\d+[A-Z]*\b|"
            r"\d*号弯|发卡弯|回头弯|急弯|慢弯|快弯|那个弯|这弯|弯心|"
            r"\bhairpin\b|\bchicane\b|\bsweeper\b|\bapex\b|"
            r"130R|勺子弯| spoon|parabolica|casino|loews|portier|"
            r"第[一二三四五六七八九十\d]+弯|"
            r"\b(first|second|third|last|final)\s+corner\b|"
            r"\bturn\s+\d+\b"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "sector",
        re.compile(
            r"(?:"
            r"\bS[123]\b|\bsector\s*\d\b|"
            r"扇区|那一段|这一段|连续弯段|直道段|高速段|低速段|中段|"
            r"\besses\b|\bstraight\b|\bcomplex\b|\bsector\b|"
            r"出弯时|入弯时|刹车点|"
            r"\b(first|middle|last)\s+sector\b|"
            r"前半段|后半段"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "overall",
        re.compile(
            r"(?:"
            r"整体|全圈|整车|总体|全部|整圈|"
            r"圈速|总时间|lap\s*time|lap\s*pace|"
            r"平衡|balance|"
            r"胎温|tyre\s*temp|tire\s*temp|"
            r"ERS|燃油|fuel|deploy|"
            r"还能优化|感觉车|车还行|车不行|"
            r"\boverall\b|\bwhole\s+lap\b|\bentire\s+lap\b|"
            r"还能快|快多少|潜力|potential"
            r")",
            re.IGNORECASE,
        ),
    ),
]


class GranularityResult:
    """Result of granularity classification (Iter-171).

    Attributes:
        granularity: One of ``"corner"`` / ``"sector"`` / ``"overall"``.
        confidence: Confidence in [0, 1] — 1.0 for explicit keyword match,
            0.3 for default fallback (no keyword matched → overall).
        matched_pattern: The matched substring (empty for fallback).
        corner_ref: If granularity=="corner" and a turn number was
            detected, the turn reference (e.g. ``"T1"``, ``"T130R"``);
            empty string otherwise.
    """

    __slots__ = ("granularity", "confidence", "matched_pattern", "corner_ref")

    def __init__(
        self,
        granularity: str,
        confidence: float,
        matched_pattern: str = "",
        corner_ref: str = "",
    ) -> None:
        self.granularity = granularity
        self.confidence = confidence
        self.matched_pattern = matched_pattern
        self.corner_ref = corner_ref

    def __repr__(self) -> str:
        return (
            f"GranularityResult(granularity={self.granularity!r}, "
            f"confidence={self.confidence:.2f}, corner_ref={self.corner_ref!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GranularityResult):
            return NotImplemented
        return (
            self.granularity == other.granularity
            and abs(self.confidence - other.confidence) < 1e-6
        )


def classify_granularity(message: str) -> GranularityResult:
    """Classify the driver's feedback granularity (Iter-171).

    Determines whether the driver is asking about a specific corner
    (``"corner"``), a sector/section (``"sector"``), or the whole
    lap/car (``"overall"``). Defaults to ``"overall"`` with low
    confidence when no keyword matches.

    Args:
        message: The driver's natural language input (Chinese or English).

    Returns:
        :class:`GranularityResult` with granularity, confidence, and
        (for corner granularity) the detected corner reference.

    Examples::

        >>> classify_granularity("为什么 T1 入弯总推头?")
        GranularityResult(granularity='corner', confidence=1.0, corner_ref='T1')
        >>> classify_granularity("S2 连续弯那一段车头太钝")
        GranularityResult(granularity='sector', confidence=1.0, corner_ref='')
        >>> classify_granularity("圈速能再快多少?")
        GranularityResult(granularity='overall', confidence=1.0, corner_ref='')
        >>> classify_granularity("hello")
        GranularityResult(granularity='overall', confidence=0.3, corner_ref='')
    """
    if not message or not message.strip():
        return GranularityResult("overall", 0.3)

    for granularity, pattern in _GRANULARITY_PATTERNS:
        match = pattern.search(message)
        if match:
            matched = match.group()
            corner_ref = ""
            if granularity == "corner":
                t_match = re.search(r"T\d+[A-Z]*", message, re.IGNORECASE)
                if t_match:
                    corner_ref = t_match.group().upper()
            return GranularityResult(
                granularity=granularity,
                confidence=1.0,
                matched_pattern=matched,
                corner_ref=corner_ref,
            )

    return GranularityResult("overall", 0.3)


# ============================================================================
# Sub-intent classification (Iter-197)
# ============================================================================

# Sub-intents for setup_advice: maps sub-intent to regex patterns
_SETUP_SUB_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "aero",
        re.compile(
            r"(?:前翼|后翼|尾翼|下压力|扰流|扩散器|"
            r"front.?wing|rear.?wing|downforce|diffuser|spoiler|"
            r"drag|风阻|空力|aero|wing.*angle|"
            r"beam.?wing|地板|floor|底盘|floor.*edge|underbody)",
            re.IGNORECASE,
        ),
    ),
    (
        "mechanical",
        re.compile(
            r"(?:悬挂|弹簧|避震|阻尼|anti.?roll.?bar|防倾杆|arb|"
            r"suspension|spring|damper|shock|ride.?height|车高|"
            r"toe|camber|外倾|束角|wheel.*rate|"
            r"roll.?bar|bump.*stop|行程|travel|"
            r"push.?rod|pull.?rod|rocker)",
            re.IGNORECASE,
        ),
    ),
    (
        "differential",
        re.compile(
            r"(?:差速|diff.*(?:lock|coast|power|preload|entry|mid|exit)|"
            r"lsd|limited.?slip|open.*diff|"
            r"on.?throttle|off.?throttle|驱动分配|"
            r"torque.*vectoring|扭矩.*分配)",
            re.IGNORECASE,
        ),
    ),
    (
        "brakes",
        re.compile(
            r"(?:刹车|制动|brake|bias|压力|disc|pad|"
            r"brake.*pressure|brake.*balance|brake.*duct|"
            r"brake.*migration|brake.*magic|"
            r"engine.?braking|engine.*brake|再生.*制动|"
            r"brake.*temp|制动.*温度)",
            re.IGNORECASE,
        ),
    ),
    (
        "tyre",
        re.compile(
            r"(?:轮胎|胎压|胎温|胎|tyre|tire|compound|"
            r"pressure|psi|温度.*胎|胎.*压|"
            r"camber.*tire|tire.*camber|toe.*tire|"
            r"soft|medium|hard|inter|wet|"
            r"wear|磨损|degradation)",
            re.IGNORECASE,
        ),
    ),
]

# Sub-intents for problem_report: what specific problem
_PROBLEM_SUB_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "understeer",
        re.compile(
            r"(?:推头|转向不足|understeer|won.*turn|tight.*corner|"
            r"not.*turning|missing.*apex|wide.*exit|"
            r"push.*on|doesn.*rotate|won.*rotate|"
            r"numb|vague|front.*wash|wash.*out)",
            re.IGNORECASE,
        ),
    ),
    (
        "oversteer",
        re.compile(
            r"(?:甩尾|甩|转向过度|oversteer|snap|loose|tail.*happy|"
            r"rear.*slide|rear.*loose|rear.*moving|stepping.*out|"
            r"step.*out|unstable|nervous|twitchy|"
            r"rotate.*too.*much|too.*pointy|snap.*oversteer|"
            r"lift.?off.*oversteer|power.*oversteer|"
            r"车尾.*松|尾.*松|后轴.*松|出弯.*甩|尾部.*滑|后.*滑)",
            re.IGNORECASE,
        ),
    ),
    (
        "tyre_wear",
        re.compile(
            r"(?:磨损|掉粒|graining|blistering|起泡|flat.?spot|平斑|"
            r"wear.*fast|going.*off|dropped.*off|fell.*off|"
            r"worn|gone|dead|finished|"
            r"rear.*tyre.*wear|front.*tyre.*wear|"
            r"deg|degradation)",
            re.IGNORECASE,
        ),
    ),
    (
        "brake",
        re.compile(
            r"(?:刹车|锁死|lockup|locking|brake.*fade|brake.*fail|"
            r"brake.*long|brake.*soft|brake.*spongy|"
            r"brake.*temp|brake.*hot|brake.*smoke|"
            r"brake.*bias.*wrong|brake.*balance.*off|"
            r"rear.*locking|front.*locking)",
            re.IGNORECASE,
        ),
    ),
    (
        "ers",
        re.compile(
            r"(?:ERS|电池|能量|deploy|harvest|boost|"
            r"out.*of.*battery|no.*deploy|deploy.*late|deploy.*early|"
            r"energy.*store|ES|MGU.?K|MGU.?H|"
            r"battery.*empty|no.*boost|no.*ERS|"
            r"ERS.*weak|ERS.*slow|power.*delivery)",
            re.IGNORECASE,
        ),
    ),
    (
        "traction",
        re.compile(
            r"(?:牵引|traction|wheelspin|wheel.*spin|spinning.*up|"
            r"no.*grip|can.*put.*power|can.*get.*power|"
            r"sliding|slipping|spin|"
            r"low.*speed.*grip|exit.*grip|corner.*exit.*grip|"
            r"poor.*traction|bad.*traction|"
            r"no.*bite|浮|漂浮)",
            re.IGNORECASE,
        ),
    ),
    (
        "balance",
        re.compile(
            r"(?:平衡|balance.*off|not.*balanced|不平衡|"
            r"too.*much.*understeer.*then.*oversteer|"
            r"push.*then.*loose|inconsistent.*balance|"
            r"entry.*push.*exit.*loose|entry.*loose.*exit.*push)",
            re.IGNORECASE,
        ),
    ),
]

_SUB_INTENT_ALL: set[str] = {
    sub for sub, _ in _SETUP_SUB_INTENT_PATTERNS
} | {
    sub for sub, _ in _PROBLEM_SUB_INTENT_PATTERNS
}


class SubIntentResult:
    """Result of sub-intent classification (Iter-197).

    Attributes:
        sub_intent: The classified sub-intent label (e.g. ``"aero"``, ``"understeer"``).
        confidence: Confidence score in [0, 1].
        category: The parent intent category (``"setup"`` or ``"problem"``).
        matched_pattern: The regex pattern that matched (for debugging).
    """

    __slots__ = ("sub_intent", "confidence", "category", "matched_pattern")

    def __init__(
        self,
        sub_intent: str,
        confidence: float,
        category: str = "",
        matched_pattern: str = "",
    ) -> None:
        self.sub_intent = sub_intent
        self.confidence = confidence
        self.category = category
        self.matched_pattern = matched_pattern

    def __repr__(self) -> str:
        return (
            f"SubIntentResult(sub_intent={self.sub_intent!r}, "
            f"confidence={self.confidence:.2f}, category={self.category!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubIntentResult):
            return NotImplemented
        return (
            self.sub_intent == other.sub_intent
            and abs(self.confidence - other.confidence) < 1e-6
        )


def classify_sub_intent(message: str, intent: str) -> SubIntentResult:
    """Classify the sub-intent for a given parent intent (Iter-197).

    For ``"setup_advice"``, returns the specific setup area
    (aero/mechanical/differential/brakes/tyre). For ``"problem_report"``,
    returns the specific problem type (understeer/oversteer/tyre_wear/brake/ers/traction/balance).

    Args:
        message: The driver's natural language input.
        intent: The parent intent from :func:`classify_intent`.

    Returns:
        :class:`SubIntentResult` with the classified sub-intent.
        Returns ``SubIntentResult("general", 0.3)`` if no specific sub-intent
        matches.

    Examples::

        >>> classify_sub_intent("怎么调整前翼来减少推头？", "setup_advice")
        SubIntentResult(sub_intent='aero', confidence=1.0, category='setup')
        >>> classify_sub_intent("车尾太松了, 出弯总是甩", "problem_report")
        SubIntentResult(sub_intent='oversteer', confidence=1.0, category='problem')
        >>> classify_sub_intent("hello", "greeting")
        SubIntentResult(sub_intent='general', confidence=0.3, category='')
    """
    if not message or not message.strip():
        return SubIntentResult("general", 0.0)

    if intent == "setup_advice":
        patterns = _SETUP_SUB_INTENT_PATTERNS
        category = "setup"
    elif intent == "problem_report":
        patterns = _PROBLEM_SUB_INTENT_PATTERNS
        category = "problem"
    else:
        return SubIntentResult("general", 0.3)

    matches: list[tuple[str, float, str]] = []
    for sub_intent, pattern in patterns:
        match = pattern.search(message)
        if match:
            matches.append((sub_intent, 1.0, match.group()))

    if not matches:
        return SubIntentResult("general", 0.3, category)

    if len(matches) == 1:
        return SubIntentResult(
            sub_intent=matches[0][0],
            confidence=matches[0][1],
            category=category,
            matched_pattern=matches[0][2],
        )

    # Multiple sub-intents: primary = first, confidence reduced
    primary = matches[0]
    confidence = max(0.4, 1.0 - 0.15 * (len(matches) - 1))
    return SubIntentResult(
        sub_intent=primary[0],
        confidence=confidence,
        category=category,
        matched_pattern=primary[2],
    )
