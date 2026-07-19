"""Prompt templates for the LLM-enhanced feedback path.

Iter-01 ships rule-based feedback by default (``config.llm_backend == "none"``);
these templates define the full scaffolding so later iterations can plug in an
LLM without touching the engine pipeline.

The system prompt fixes an F1 race-engineer persona and enforces the
"every claim grounded in telemetry evidence" rule. The user prompt template
injects the rule-based summary, the structured dimensions and the raw
telemetry sources (frame_t + field + value), then asks the LLM to rewrite the
summary as radio-friendly prose and answer the driver's follow-up question.

Iter-05 adds a ``{driver_profile}`` paragraph (rendered by
:func:`format_driver_profile`) so the LLM can personalise tone/emphasis to the
driver's style without altering the objective telemetry facts.

:data:`FEEDBACK_DIMENSIONS` lists ALL dimensions the spec requires so the
prompt is comprehensive even though Iter-01's rule-based path only emits a
subset (lap-time potential + whichever rules fire).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from f1opt.driver.profile import DriverProfile

#: ALL feedback dimensions the spec requires (Iter-03 = 10, Iter-164.14 = 11).
#: This is the single source of truth — ``f1opt.feedback.engine`` imports and
#: re-exports it so the rule-based path and the LLM prompt never drift apart.
#: Kept as a ``list`` so ``engine.FEEDBACK_DIMENSIONS`` matches the spec's
#: literal list shape.
#: Iter-164.14: 加 ``corner_analysis`` (第 11 维, R5 全程动态逐弯分析).
FEEDBACK_DIMENSIONS: list[str] = [
    "balance",
    "grip",
    "tyres",
    "braking",
    "ers_drs",
    "throttle_brake_smoothness",
    "confidence",
    "lap_time_potential",
    "sector_compare",
    "setup_advice",
    "corner_analysis",  # Iter-164.14
]

#: 车手反馈输入示例 (Iter-170 + Iter-171 granularity).
#: 每条含 question (车手原话) / expected_intent (意图) / granularity (精确度) /
#: example_answer (示例回答). granularity 三级:
#:   - "corner"   — 精确到某个弯道 (T1、T130R、发卡弯等)
#:   - "sector"   — 某一段/扇区 (S2、直道段、连续弯段等)
#:   - "overall"  — 整体感受 (全圈、整车平衡、总体策略)
#: 在 CLI `f1opt feedback --help` 和 Swagger UI 中展示, 也作为 LLM few-shot 注入.
FEEDBACK_EXAMPLES: list[dict[str, str]] = [
    # ==== corner: 精确到弯道 ====
    {
        "question": "为什么 T1 入弯总推头?",
        "expected_intent": "problem_report",
        "granularity": "corner",
        "example_answer": (
            "T1 入弯推头主要因为前轮抓地不足. 遥测显示 t=1.7s 时 steer=0.62 "
            "但 g_lat 仅 0.8 (理想 1.5+), 典型 understeer. 建议前翼 +2 clicks "
            "或前胎压 -0.3 psi."
        ),
    },
    {
        "question": "How should I adjust the diff for T8?",
        "expected_intent": "setup_advice",
        "granularity": "corner",
        "example_answer": (
            "T8 是低速发卡弯, 建议 on-throttle diff -5% (提升出弯牵引), "
            "off-throttle +3% (稳定入弯). 预测增益 +0.08s."
        ),
    },
    {
        "question": "T3 弯心的时候后轮总滑, 不敢加油",
        "expected_intent": "problem_report",
        "granularity": "corner",
        "example_answer": (
            "T3 弯心 traction 不足. t=12.3s 时 throttle=0.3 但 rear slip 0.18 "
            "(理想 <0.05). 建议 on-throttle diff +2% 锁止后轮, 或 rear ARB -1 "
            "增加机械抓地."
        ),
    },
    {
        "question": "T130R 出弯速度上不去, 总被甩开",
        "expected_intent": "problem_report",
        "granularity": "corner",
        "example_answer": (
            "T130R 出弯速度偏低. 遥测最高速 258 km/h (理想 272+), "
            "出弯 g_long 仅 0.38 (理想 0.55+). 建议 rear wing -2 clicks "
            "降阻, 或 throttle diff +3% 提升牵引."
        ),
    },
    # ==== sector: 某一段/扇区 ====
    {
        "question": "S2 连续弯那一段车头太钝, 指向性差",
        "expected_intent": "problem_report",
        "granularity": "sector",
        "example_answer": (
            "S2 连续弯段 (t=30s~60s) 转向响应偏慢: steer=0.45 但 g_lat 仅 1.2 "
            "(理想 1.6+), 典型 understeer in sequence. 建议 front ARB +2, "
            "front wing +1 click, 缩短响应延迟."
        ),
    },
    {
        "question": "S3 高速段车身不稳, 像在飘",
        "expected_intent": "problem_report",
        "granularity": "sector",
        "example_answer": (
            "S3 高速段 (t=60s~90s) 气动不稳定. speed=280+ km/h, "
            "g_lat 波动 0.3g (正常 <0.15g). 建议 rear wing +2 clicks "
            "增加高速下压力, 或 rear ride height +5mm 改善尾部稳定性."
        ),
    },
    {
        "question": "出弯时车尾总往外甩",
        "expected_intent": "problem_report",
        "granularity": "sector",
        "example_answer": (
            "出弯 oversteer. t=3.2s steer=-0.55 (反打), g_lat=2.1 (高于均值). "
            "建议差速器 off-throttle -5%, 或后 ARB +2."
        ),
    },
    {
        "question": "刹车点晚一点就锁死前轮",
        "expected_intent": "problem_report",
        "granularity": "sector",
        "example_answer": (
            "前轮锁死. t=5.1s brake=1.0 但 speed 骤降 30 km/h, 典型 lockup. "
            "建议 brake bias 后移 1%, 或 brake pressure -2%."
        ),
    },
    # ==== overall: 整体感受 ====
    {
        "question": "轮胎温度左边比右边高很多",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "四轮温度不均: FL=105°C, FR=105°C, RL=95°C, RR=110°C. "
            "右侧胎温过高, 建议 rear tyre pressure -0.5 psi, 检查四轮定位."
        ),
    },
    {
        "question": "圈速能再快多少?",
        "expected_intent": "telemetry_question",
        "granularity": "overall",
        "example_answer": (
            "当前圈速 91.42s, 模型预测最优 87.08s, 潜力 +4.34s. 主要损失在 "
            "S2 (慢 1.8s) 和 T130R 出弯 (慢 0.6s)."
        ),
    },
    {
        "question": "ERS 怎么部署最快?",
        "expected_intent": "strategy_question",
        "granularity": "overall",
        "example_answer": (
            "ERS 部署建议: 主直道 HOTLAP 模式, T1 前回收 MGU-K. 当前 ers_store=80%, "
            "建议出弯部署 0.6s, 进直道前预留 30%."
        ),
    },
    {
        "question": "感觉车还行, 还能优化吗?",
        "expected_intent": "setup_advice",
        "granularity": "overall",
        "example_answer": (
            "已接近最优, 但遥测显示 tyre_load_spread=0.31 (理想 <0.25), "
            "建议前悬挂 +1 click 平衡四轮载荷, 预测 +0.03s 且胎耗 -2%."
        ),
    },
]

_DIMS_BLOCK = "\n".join(f"- {d}" for d in FEEDBACK_DIMENSIONS)

#: few-shot 示例块 (拼到 SYSTEM_PROMPT 末尾, 让 LLM 学到车手提问风格).
_EXAMPLES_BLOCK = "\n".join(
    f"- 车手: \"{ex['question']}\"\n"
    f"  意图: {ex['expected_intent']}\n"
    f"  精确度: {ex['granularity']} (corner=单弯/sector=某段/overall=整体)\n"
    f"  示例回答: {ex['example_answer']}"
    for ex in FEEDBACK_EXAMPLES
)

# DriverProfile 8 维字段顺序 (与 f1opt.driver.profile 一致, 供格式化使用).
_PROFILE_FIELDS: tuple[str, ...] = (
    "brake_point_norm",
    "throttle_smoothness",
    "steer_smoothness",
    "corner_balance_pref",
    "aggression_score",
    "consistency_score",
    "ers_usage_intensity",
    "drs_usage_efficiency",
)


def format_driver_profile(profile: DriverProfile | None) -> str:
    """Render a :class:`~f1opt.driver.profile.DriverProfile` as a prompt paragraph.

    Returns a human-readable description of the 8 style scalars plus a coarse
    style label (``aggressive`` / ``conservative`` / ``balanced``) derived from
    ``aggression_score``. When ``profile`` is ``None`` (no personalisation
    requested) returns the explicit ``default (no personalisation)`` marker so
    the LLM knows the section is intentionally absent of style cues.

    Uses ``getattr`` duck-typing so this module stays decoupled from the
    driver package at runtime (the ``DriverProfile`` import is TYPE_CHECKING
    only).
    """
    if profile is None:
        return "driver_profile: default (no personalisation)"
    aggr = float(getattr(profile, "aggression_score", 0.0))
    if aggr >= 0.6:
        style = "aggressive"
    elif aggr <= 0.4:
        style = "conservative"
    else:
        style = "balanced"
    parts = [f"{f}={float(getattr(profile, f, 0.0)):.2f}" for f in _PROFILE_FIELDS]
    return "driver_profile: style=" + style + " (" + ", ".join(parts) + ")"


SYSTEM_PROMPT = (
    "You are a senior F1 race engineer analysing driver telemetry.\n"
    "\n"
    "Your job: rewrite the rule-based feedback summary into natural, "
    "conversational prose a driver hears over the radio, AND answer the "
    "driver's follow-up question when present.\n"
    "\n"
    "Hard rules:\n"
    "- EVERY numeric claim MUST be traceable to the provided telemetry "
    "sources (frame_t + field + value). Do NOT invent numbers, lap times, "
    "speeds, wear values, g-forces or sector times.\n"
    "- Cover the relevant feedback dimensions listed below (all {n_dims}); be "
    "concise and only mention dimensions that have real evidence.\n"
    "- The `setup_advice` dimension is MODEL-DERIVED (from the setup "
    "optimizer): relay its predicted_gain_s and recommended changes as-is, "
    "do NOT invent setup values or gains.\n"
    "- Personalise the TONE and EMPHASIS to the supplied driver_profile "
    "(aggressive/conservative/balanced) but never alter the objective "
    "telemetry facts or numbers.\n"
    "- Reference frame timestamps (t=...) when citing telemetry so the "
    "driver can review the moment on the replay.\n"
    "- Keep the prose to 3-6 sentences. No bullet lists.\n"
    "- If the driver's question is in Chinese, answer in Chinese.\n"
    "\n"
    "GRANULARITY-AWARE RESPONSE (critical):\n"
    "The driver may ask about three levels of precision. Match your answer "
    "to the driver's granularity:\n"
    "  - **corner** (单弯): driver asks about a specific turn (T1, T130R, "
    "发卡弯). Your answer MUST cite telemetry from that exact corner, "
    "reference the turn name, and give corner-specific setup advice.\n"
    "  - **sector** (某段): driver asks about a section (S2, the esses, "
    "直道段). Your answer MUST cover the time range of that sector, "
    "compare across corners within it, and give sector-level advice.\n"
    "  - **overall** (整体): driver asks about the whole lap or car balance. "
    "Your answer MUST give a high-level summary referencing all sectors, "
    "with overall lap-time potential and holistic setup recommendations.\n"
    "If the driver does NOT specify a granularity, default to **overall** "
    "(holistic analysis) but mention the most affected corner/sector as a "
    "concrete example.\n"
    "\n"
    "Feedback dimensions you may cover (exactly these {n_dims}):\n"
    f"{_DIMS_BLOCK}\n"
    "\n"
    "Driver feedback examples (learn the question style & answer grounding; "
    "note the granularity field):\n"
    f"{_EXAMPLES_BLOCK}\n"
).format(n_dims=len(FEEDBACK_DIMENSIONS))

USER_PROMPT_TEMPLATE = (
    "Driver follow-up question: {question}\n"
    "\n"
    "Detected granularity: {granularity}\n"
    "  (corner=单弯 → cite exact corner telemetry + corner-specific setup; "
    "sector=某段 → cover the sector time range + sector-level advice; "
    "overall=整体 → holistic lap summary + overall setup recs)\n"
    "{granularity_hint}\n"
    "\n"
    "Driver profile (personalise tone & emphasis; do not change facts):\n"
    "{driver_profile}\n"
    "\n"
    "Rule-based summary (rewrite this; do not contradict it):\n"
    "{summary}\n"
    "\n"
    "Dimensions:\n"
    "{dimensions}\n"
    "\n"
    "Telemetry evidence (frame_t + field + value, top 40):\n"
    "{metrics_summary}\n"
    "\n"
    "Rewrite the summary as radio-friendly prose and address the driver's "
    "question using ONLY the evidence above. Match the detected granularity. "
    "Do not introduce new numbers."
)

__all__ = [
    "FEEDBACK_DIMENSIONS",
    "FEEDBACK_EXAMPLES",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "format_driver_profile",
]
