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
    "classify_intent",
    "classify_granularity",
    "GRANULARITIES",
]

#: 三级反馈精确度 (Iter-171).
#: - "corner"  — 精确到某个弯道 (T1、T130R、发卡弯等)
#: - "sector"  — 某一段/扇区 (S2、直道段、连续弯段等)
#: - "overall" — 整体感受 (全圈、整车平衡、总体策略)
GRANULARITIES: tuple[str, ...] = ("corner", "sector", "overall")

# Intent keyword patterns (ordered by priority — first match wins).
# NOTE: "feedback" must be checked BEFORE "problem_report" because feedback
# phrases like "还是推头" / "Still understeering" contain problem keywords.
# Similarly, "setup_advice" is checked first because "怎么调推头" contains
# both setup and problem keywords.
#
# Iter-162.6 升级: 补全 11 类对话式逐弯反馈盲区 (来自 1000 条语料库测试):
#   - DRS / ERS / 路肩 / 牵引 / 锁死 / snap / graining / rotation / biting
#   - 弯道名 + 动态组合 (T3 入弯转向) 不再 fall through 到 other
#   - "Engineer," 前缀不再误触发 greeting (英文 greeting 加 \b 词边界)
#   - 情绪化后缀 "这车没法开" 不再误触发 status_check (改用 problem_report 关键词)
#   - setup_advice "调 X 还是 Y" 优先于 problem_report
#   - strategy "几圈" / "laps left" 优先于 telemetry "速度"
#   - feedback "X 改善了" 模式补全
#   - telemetry 短句 "Speed at T1" / "Temp at T3" 模式补全
_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Setup advice — asking for recommendations (priority 1)
    (
        "setup_advice",
        re.compile(
            r"(?:怎么调|如何调|怎么改|如何改|应该调|建议调|setup|调整|调教建议|"
            r"how.*(should|do).*adjust|what.*change|recommend.*setup|"
            r"what.*setup|how.*improve|怎么提升|如何改善|"
            # Iter-162.6: 调教选择问句 "调 X 还是 Y" / "调 X 或者 Y"
            r"调.*(还是|或者)|"
            # Iter-162.6: "How should I tune the diff for T8?"
            r"how.*(tune|set).*for|tune.*diff.*for|"
            r"调教|调一下|怎么改)",
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
            # Iter-162.6: "X 改善了" / "X 好多了" / "X 的 Y 改善了"
            r"改善了|好多了|"
            r"improved|better now|fixed it)",
            re.IGNORECASE,
        ),
    ),
    # Strategy question — Iter-162.6: 提前到 telemetry 之前
    # ("几圈" / "laps left" / "pit" 不能被 telemetry "速度" 抢匹配)
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
            r"(?:推头|甩尾|转向不足|转向过度|打滑|抓地力|没抓地|胎温.*高|"
            r"温度.*高|磨损.*快|刹车.*不好|油门.*响应|ERS.*不够|动力.*不足|"
            r"understeer|oversteer|no grip|overheating|wear.*fast|"
            r"loose|tight|pushing|sliding|wheel spin|tire.*wear|"
            r"brake.*issue|throttle.*response|power.*delivery|"
            # Iter-162.6: DRS 关键词
            r"DRS.*(没开|不开|失灵|慢|fail|slow|late|early|反应)|"
            # Iter-162.6: ERS 关键词
            r"ERS.*(不够|用完|没.*boost|响应慢|部署.*不对|weak|short|empty|"
            r"slow|out)|电池.*没电|no.*ERS|out of battery|"
            # Iter-162.6: 路肩 kerb
            r"路肩|kerb|bottoming|压路肩|过路肩|"
            # Iter-162.6: 出弯牵引 traction
            r"牵引.*(不行|不够|断|差|poor|bad)|no traction|"
            r"traction.*(poor|bad|out of|inconsistent)|"
            r"can.*t.*put.*power|wheelspin|wheel spin|spinning up|"
            # Iter-162.6: 锁死 lockup
            r"锁死|lockup|locking up|lock up|locked up|"
            # Iter-162.6: snap oversteer
            r"\bsnap\b|甩了一下|差点失控|lost it|"
            # Iter-162.6: graining / 粒化
            r"graining|起粒|掉得快|磨损.*快|胎.*平了|胎.*磨掉|磨平|gone|"
            # Iter-162.6: rotation / pointy
            r"rotate|rotation|pointy|旋转|转不过来|"
            # Iter-162.6: biting / 不咬住
            r"bite|biting|不咬|没咬|floating|漂浮|"
            # Iter-162.6: undrivable / 没法开
            r"undrivable|没法开|没法驾驶|开不了|"
            # Iter-162.6: 不稳定 unstable / nervous
            r"unstable|nervous|不稳|晃|跳|"
            # Iter-162.6: braking rear step out
            r"stepping out|step out|rear.*moving|"
            # Iter-162.6: 模糊 / 钝 / heavy wheel
            r"模糊|感觉钝|heavy|lazy|vague|numb|"
            # Iter-162.6: 胎温 out of control
            r"out of control|失控|飙|过热|烫|"
            # Iter-162.6: 平衡 off / balance
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
            # Iter-162.6: 短遥测句 "Speed at T1" / "Temp at T3" / "RPM at T4"
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
    """
    if not message or not message.strip():
        return IntentResult("other", 0.0)

    # Try each pattern in priority order — first match wins
    for intent, pattern in _INTENT_PATTERNS:
        match = pattern.search(message)
        if match:
            # Confidence: 1.0 for a clear match. We could differentiate
            # based on match length / position, but for a rule-based
            # classifier, 1.0 is appropriate for any keyword hit.
            return IntentResult(
                intent=intent,
                confidence=1.0,
                matched_pattern=match.group(),
            )

    return IntentResult("other", 0.0)


# ============================================================================
# Granularity classification (Iter-171)
# ============================================================================

# Granularity patterns — first match wins (corner > sector > overall).
# Corner: explicit turn references (T1, T130R, 弯, 发卡弯, hairpin, etc.)
# Sector: explicit sector references (S1, S2, S3, 扇区, 段, esses, straight, etc.)
# Overall: lap-wide or car-wide terms (圈速, 平衡, 整车, 全圈, overall, balance, etc.)
#
# Patterns are intentionally specific — a message like "T3 入弯推头" matches
# corner (not sector) because the corner pattern is checked first. A message
# like "S2 那段车头钝" matches sector. A message like "圈速能再快多少" matches
# overall (no corner/sector keyword).
_GRANULARITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # ---- corner: 精确到单弯 ----
    (
        "corner",
        re.compile(
            r"(?:"
            # 英文弯道编号: T1, T2, ... T130R, T1-T3
            r"\bT\d+[A-Z]*\b|"
            # 中文弯道词: 弯, 发卡弯, 回头弯, 急弯, 慢弯, 快弯
            # NOTE: "入弯"/"出弯" 不放这里 (太宽泛, 匹配任意弯道进出);
            # 它们的 "入弯时"/"出弯时" 形式在 sector 模式里.
            r"\d*号弯|发卡弯|回头弯|急弯|慢弯|快弯|那个弯|这弯|弯心|"
            # 英文弯道类型
            r"\bhairpin\b|\bchicane\b|\bsweeper\b|\bapex\b|"
            # 中文具体弯道 (常见赛道)
            r"130R|勺子弯| spoon|parabolica|casino|loews|portier|"
            # "the first/last corner" / "第N弯"
            r"第[一二三四五六七八九十\d]+弯|"
            r"\b(first|second|third|last|final)\s+corner\b|"
            r"\bturn\s+\d+\b"
            r")",
            re.IGNORECASE,
        ),
    ),
    # ---- sector: 某一段/扇区 ----
    (
        "sector",
        re.compile(
            r"(?:"
            # 英文 sector: S1, S2, S3, sector 1
            r"\bS[123]\b|\bsector\s*\d\b|"
            # 中文扇区/段
            r"扇区|那一段|这一段|连续弯段|直道段|高速段|低速段|中段|"
            # 英文段名
            r"\besses\b|\bstraight\b|\bcomplex\b|\bsector\b|"
            # "出弯时" / "刹车点" (隐含某段, 但无具体弯道编号 → sector)
            r"出弯时|入弯时|刹车点|"
            # "the middle sector" / "first sector"
            r"\b(first|middle|last)\s+sector\b|"
            r"前半段|后半段"
            r")",
            re.IGNORECASE,
        ),
    ),
    # ---- overall: 整体感受 ----
    (
        "overall",
        re.compile(
            r"(?:"
            # 整体/全圈/整车
            r"整体|全圈|整车|总体|全部|整圈|"
            # 圈速/总时间
            r"圈速|总时间|lap\s*time|lap\s*pace|"
            # 平衡 (整车)
            r"平衡|balance|"
            # 轮胎温度 (整车四轮)
            r"胎温|tyre\s*temp|tire\s*temp|"
            # ERS/燃油策略 (整车)
            r"ERS|燃油|fuel|deploy|"
            # "还能优化吗" / "感觉车还行" (整体)
            r"还能优化|感觉车|车还行|车不行|"
            # "How much faster" / "overall"
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

    # Try each pattern in priority order (corner > sector > overall).
    for granularity, pattern in _GRANULARITY_PATTERNS:
        match = pattern.search(message)
        if match:
            matched = match.group()
            corner_ref = ""
            # Extract corner reference for corner granularity.
            if granularity == "corner":
                # Look for T<number>[letters] pattern (T1, T130R, etc.)
                t_match = re.search(r"T\d+[A-Z]*", message, re.IGNORECASE)
                if t_match:
                    corner_ref = t_match.group().upper()
            return GranularityResult(
                granularity=granularity,
                confidence=1.0,
                matched_pattern=matched,
                corner_ref=corner_ref,
            )

    # No keyword matched → default to overall with low confidence.
    return GranularityResult("overall", 0.3)
