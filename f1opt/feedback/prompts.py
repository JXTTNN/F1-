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
#: ALL feedback dimensions the spec requires (Iter-03 = 10, 现为 18).
#: This is the single source of truth --- ``f1opt.feedback.engine`` imports and
#: re-exports it so the rule-based path and the LLM prompt never drift apart.
#: Kept as a ``list`` so ``engine.FEEDBACK_DIMENSIONS`` matches the spec's
#: literal list shape.
#: Iter-164.14: 加 ``corner_analysis`` (第 11 维, R5 全程动态逐弯分析).
FEEDBACK_DIMENSIONS: list[str] = [
    "balance",
    "grip",
    "tyres",
    "braking",
    "ers_deployment",
    "drs_usage",
    "throttle_brake_smoothness",
    "confidence",
    "lap_time_potential",
    "sector_compare",
    "setup_advice",
    "corner_analysis",  # Iter-164.14
    "fuel_consumption",  # Iter-203
    "throttle_brake_overlap",  # Iter-210
    "aero_balance",  # Iter-214
    "brake_temp",  # Iter-222: 刹车温度平衡
    "tyre_temp_gradient",  # Iter-227: 轮胎温度梯度
    "grip_consistency",  # Iter-241: 抓地力一致性
    "active_aero_usage",  # Iter-256: F1 2026 主动空力 (X/Z-Mode) 使用
]
#: 车手反馈输入示例 (Iter-170 + Iter-171 granularity, Iter-176 expand 14->20).
#: 每条含 question (车手原话) / expected_intent (意图) / granularity (精确度) /
#: example_answer (示例回答). granularity 三级:
#:   - "corner"   --- 精确到某个弯道 (T1、T130R、发卡弯等)
#:   - "sector"   --- 某一段/扇区 (S2、直道段、连续弯段等)
#:   - "overall"  --- 整体感受 (全圈、整车平衡、总体策略)
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
    {
        "question": "T5出弯ERS Overtake模式够不够?",
        "expected_intent": "strategy_question",
        "granularity": "corner",
        "example_answer": (
            "T5 出弯 ERS Overtake 模式部署 0.5s, 消耗 12% 电量, 出弯速度 +6 km/h. "
            "剩余电量 55%, 可维持当前策略至圈末. 若要在T5超车, 建议延长部署至 0.8s "
            "(多耗 5% 电量, 出弯速度再 +3 km/h)."
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
    {
        "question": "队友S2比我快0.3s, 差在哪?",
        "expected_intent": "telemetry_question",
        "granularity": "sector",
        "example_answer": (
            "队友S2 (t=30s~60s) 比你快 0.28s, 主要差距在 T8 出弯 (队友出弯速度 +8 km/h) "
            "和 T9 刹车点 (队友晚刹 0.05s). 建议 T8 throttle diff +3% 提升出弯牵引, "
            "T9 brake bias 前移 1% 增强入弯信心."
        ),
    },
    {
        "question": "DRS zone detection为什么比队友慢?",
        "expected_intent": "problem_report",
        "granularity": "sector",
        "example_answer": (
            "DRS 激活延迟 0.12s (队友 0.05s). 遥测显示 detection point 前 "
            "speed=289 km/h (队友 294), 车距差 0.08s. 建议 rear wing -1 click "
            "降低直道阻力, 或优化前弯出弯速度确保 detection 点时已到极速."
        ),
    },
    {
        "question": "S1慢在T3出弯还是T4刹车?",
        "expected_intent": "telemetry_question",
        "granularity": "sector",
        "example_answer": (
            "S1 损失 0.22s: T3 出弯慢 0.08s (出弯速度 178 vs 理想 185 km/h), "
            "T4 刹车慢 0.14s (刹车点晚 6m, 入弯速度过高导致出弯推头). "
            "重点优化 T4: brake bias 前移 1%, brake pressure -2%, 预测增益 +0.10s."
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
    {
        "question": "前轮起粒了怎么办",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "前轮 graining 检测: FL surf temp=112°C, FR=110°C (理想 95~105°C), "
            "胎面过热导致橡胶颗粒化. 建议 front tyre pressure -0.3 psi, "
            "front brake bias 后移 2% 减少前轮热负荷, 入弯速度 -3 km/h 保护胎面, "
            "预计 2 圈后 graining 清除."
        ),
    },
    {
        "question": "雨天这调教能用吗",
        "expected_intent": "setup_advice",
        "granularity": "overall",
        "example_answer": (
            "当前调教偏干地设定, 雨天需调整: front wing +3 clicks (增加前部下压力), "
            "rear ride height +8mm (防 aquaplaning), diff preload -15% (柔和动力输出), "
            "brake bias 前移 2% (补偿低抓地力). 预测湿地圈速 +3.5s, 若维持干地调教 "
            "将额外损失 1.2s."
        ),
    },
    # Iter-209: 新增燃油/ERS/档位相关示例
    {
        "question": "这圈油用得太快了吧",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "燃油消耗率 0.42 kg/km, 比目标 0.33 kg/km 高 27%. 建议 lift-and-coast "
            "在 T1/T3 刹车区, 提前升档至 8 档节省燃油. 预测节省 0.8 kg/圈."
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
        "question": "感觉升档转速不够, 加速慢",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "升档时平均 RPM 11200, 红线区 90% 仅 35% 的升档触发. 建议延迟升档 "
            "至 12000+ RPM 以获得最大功率输出. 预测增益 +0.08s 每直道."
        ),
    },
    {
        "question": "轮胎内肩比外肩热太多",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "前轮内肩温度比外肩高 8°C, 负外倾角过大. 建议 front camber +0.5° "
            "(减少负倾角), 改善胎面均匀接触. 预测胎耗 -3% 且弯中稳定性提升."
        ),
    },
    # Iter-215: 下压力/气动相关示例
    {
        "question": "高速弯下压力不够, 车在飘",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "高速弯中 g_lat 仅 2.8g (理想 3.5+), 前后翼下压力不足. "
            "建议 front wing +2 档, rear wing +1 档, 预计高速弯速 +5 km/h."
        ),
    },
    {
        "question": "低速弯机械抓地力不足, 总推头",
        "expected_intent": "problem_report",
        "granularity": "corner",
        "example_answer": (
            "低速弯中机械抓地力不足, aero_balance_ratio=1.9 表明气动主导. "
            "建议 front ARB -1 档, front camber +0.3° 提升机械抓地力."
        ),
    },
    # Iter-220: active aero 相关示例
    {
        "question": "Active-X 模式直道尾速还是不够快",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "Active-X 模式下直道尾速 328 km/h (理想 340+), 低阻模式效果不足. "
            "建议 rear wing -2 档, front wing -1 档以降低整体阻力, 预测尾速 +8 km/h."
        ),
    },
    {
        "question": "什么时候切 Active-Z 模式?",
        "expected_intent": "strategy_question",
        "granularity": "overall",
        "example_answer": (
            "Active-Z 模式应在连续弯段 (S2) 使用, 提供额外下压力. "
            "当前 S2 弯中 g_lat 仅 2.5g, 建议 T5-T8 段启用 Active-Z, 预测弯速 +3 km/h."
        ),
    },
    # Iter-226: brake temp 相关示例
    {
        "question": "前刹车温度太高了，制动有点软",
        "expected_intent": "problem_report",
        "granularity": "overall",
        "example_answer": (
            "前刹车温度偏高 (front/rear ratio=1.5), 前刹承受过多热负荷. "
            "建议后移刹车偏置 1-2% 以将热负荷分配至后轴, 同时检查刹车导管是否通畅."
        ),
    },
    {
        "question": "刹车温度怎么样？需要调整偏置吗？",
        "expected_intent": "status_inquiry",
        "granularity": "overall",
        "example_answer": (
            "前后刹车温度分布均匀 (ratio=1.05), 刹车偏置设置合理. "
            "前刹车峰值 800°C, 后刹车 760°C, 均在安全工作范围内."
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
REFLECTION_PROMPT_TEMPLATE = (
    "You just produced this answer to the driver:\n"
    "{previous}\n"
    "\n"
    "Now critically re-read it against the telemetry evidence below. "
    "Reflect step by step:\n"
    "1. Does every numeric claim (speed, g-force, wear, lap time, sector) "
    "trace to a concrete evidence row?\n"
    "2. Is any setup advice unsafe or unsupported by the evidence?\n"
    "3. Did you stay within the detected granularity and the driver's question?\n"
    "\n"
    "Telemetry evidence (frame_t + field + value):\n"
    "{evidence}\n"
    "\n"
    "Produce a CORRECTED final answer (3-6 sentences, no bullet lists), "
    "fixing any unsupported numbers. If the original answer is already "
    "correct, repeat it unchanged. Answer in the same language as before."
)
__all__ = [
    "FEEDBACK_DIMENSIONS",
    "FEEDBACK_EXAMPLES",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "REFLECTION_PROMPT_TEMPLATE",
    "format_driver_profile",
    "DRIVER_FEEDBACK_TEMPLATES",
    "FEEDBACK_TEMPLATE_GROUPS",
    "render_feedback_template",
]

# --------------------------------------------------------------------------- #
# Iter-183: 车手反馈模板系统 — 赛后结构化反馈收集
# --------------------------------------------------------------------------- #
# 为车手提供结构化模板, 在跑完比赛后收集反馈输入 LLM 分析.
# 模板按粒度分为三级: corner (弯道), sector (赛段), overall (整体).
# 每个模板包含中文和英文版本, 包含问题提示和示例回答.

DRIVER_FEEDBACK_TEMPLATES: dict[str, dict[str, str]] = {
    # ==== corner 级模板 ====
    "corner_understeer": {
        "id": "corner_understeer",
        "granularity": "corner",
        "category": "handling",
        "zh": """【弯道推头反馈】
弯道: _____ (如 T1, T130R, 发卡弯)
问题: 入弯时车头推出去 (understeer), 转不过弯
严重程度: □ 轻微 □ 中等 □ 严重
发生阶段: □ 入弯刹车时 □ 弯中稳住油门时 □ 出弯加油时
你的感受: _______________________________________________
建议调整方向: 前翼, 前防倾杆, 前胎压, 差速器""",
        "en": """[Corner Understeer Feedback]
Corner: _____ (e.g. T1, T130R, Hairpin)
Issue: Front end pushes wide on entry (understeer)
Severity: □ Mild □ Moderate □ Severe
Phase: □ Braking into corner □ Mid-corner steady throttle □ Exit on throttle
Your feeling: _______________________________________________
Suggested adjustment: Front wing, Front ARB, Front tyre pressure, Differential""",
    },
    "corner_oversteer": {
        "id": "corner_oversteer",
        "granularity": "corner",
        "category": "handling",
        "zh": """【弯道甩尾反馈】
弯道: _____ (如 T1, T130R, 发卡弯)
问题: 车尾甩出去 (oversteer), 需要反打方向盘
严重程度: □ 轻微 □ 中等 □ 严重
发生阶段: □ 入弯刹车时 □ 弯中稳住油门时 □ 出弯加油时
你的感受: _______________________________________________
建议调整方向: 后翼, 后防倾杆, 后胎压, 差速器""",
        "en": """[Corner Oversteer Feedback]
Corner: _____ (e.g. T1, T130R, Hairpin)
Issue: Rear end steps out (oversteer), need counter-steer
Severity: □ Mild □ Moderate □ Severe
Phase: □ Braking into corner □ Mid-corner steady throttle □ Exit on throttle
Your feeling: _______________________________________________
Suggested adjustment: Rear wing, Rear ARB, Rear tyre pressure, Differential""",
    },
    "corner_braking": {
        "id": "corner_braking",
        "granularity": "corner",
        "category": "braking",
        "zh": """【弯道刹车反馈】
弯道: _____ (如 T1, T130R, 发卡弯)
问题: □ 刹车点太早 □ 刹车点太晚 □ 前轮锁死 □ 后轮锁死 □ 刹车距离太长
你的感受: _______________________________________________
刹车时方向盘: □ 稳定 □ 抖动 □ 偏左 □ 偏右
建议调整方向: 刹车偏置, 刹车压力, 前后刹车平衡""",
        "en": """[Corner Braking Feedback]
Corner: _____ (e.g. T1, T130R, Hairpin)
Issue: □ Braking too early □ Braking too late □ Front lockup □ Rear lockup □ Long stopping distance
Your feeling: _______________________________________________
Steering under braking: □ Stable □ Vibrating □ Pulls left □ Pulls right
Suggested adjustment: Brake bias, Brake pressure, Front/rear brake balance""",
    },
    "corner_traction": {
        "id": "corner_traction",
        "granularity": "corner",
        "category": "traction",
        "zh": """【弯道牵引力反馈】
弯道: _____ (如 T1, T130R, 发卡弯)
问题: 出弯加速时后轮打滑, 牵引力不足
失去牵引的时刻: □ 刚给油时 □ 半油门时 □ 全油门时
你的感受: _______________________________________________
建议调整方向: 差速器 (on-throttle), 后悬挂, 后胎压, 后翼""",
        "en": """[Corner Traction Feedback]
Corner: _____ (e.g. T1, T130R, Hairpin)
Issue: Rear wheel spin on exit, insufficient traction
Moment of traction loss: □ Initial throttle □ Mid throttle □ Full throttle
Your feeling: _______________________________________________
Suggested adjustment: Differential (on-throttle), Rear suspension, Rear tyre pressure, Rear wing""",
    },
    # Iter-196: 新增弯道级模板
    "corner_apex_speed": {
        "id": "corner_apex_speed",
        "granularity": "corner",
        "category": "speed",
        "zh": """【弯道弯心速度反馈】
弯道: _____ (如 T1, T130R, 发卡弯)
问题: 弯心速度偏低, 出弯后加速距离不够
最低速度: _____ km/h (理想: _____ km/h)
你的感受: _______________________________________________
建议调整方向: 增加前翼下压力, 优化刹车点, 调整入弯线路""",
        "en": """[Corner Apex Speed Feedback]
Corner: _____ (e.g. T1, T130R, Hairpin)
Issue: Apex speed too low, insufficient acceleration distance after exit
Minimum speed: _____ km/h (target: _____ km/h)
Your feeling: _______________________________________________
Suggested adjustment: Increase front wing, Optimize braking point, Adjust entry line""",
    },
    "corner_kerb_usage": {
        "id": "corner_kerb_usage",
        "granularity": "corner",
        "category": "track_usage",
        "zh": """【弯道路肩使用反馈】
弯道: _____ (如 T1, T130R, 发卡弯)
路肩使用: □ 不敢压路肩 □ 压路肩后车不稳 □ 路肩使用充分
你的感受: _______________________________________________
建议调整方向: 悬挂柔顺性, 离地间隙, 轮胎压力""",
        "en": """[Corner Kerb Usage Feedback]
Corner: _____ (e.g. T1, T130R, Hairpin)
Kerb usage: □ Avoid kerbs □ Car unstable on kerbs □ Full kerb usage
Your feeling: _______________________________________________
Suggested adjustment: Suspension compliance, Ride height, Tyre pressure""",
    },
    # ==== sector 级模板 ====
    "sector_balance": {
        "id": "sector_balance",
        "granularity": "sector",
        "category": "handling",
        "zh": """【赛段平衡反馈】
赛段: _____ (S1 / S2 / S3)
整体感受: □ 推头为主 □ 甩尾为主 □ 中性平衡
弯道类型: □ 低速弯最多 □ 中速弯最多 □ 高速弯最多
直道表现: □ 加速好 □ 加速差 □ 尾速好 □ 尾速差
你的感受: _______________________________________________
建议调整方向: 前后翼平衡, 差速器, 悬挂几何""",
        "en": """[Sector Balance Feedback]
Sector: _____ (S1 / S2 / S3)
Overall feel: □ Understeer dominant □ Oversteer dominant □ Neutral
Corner types: □ Mostly slow □ Mostly medium □ Mostly fast
Straight performance: □ Good acceleration □ Poor acceleration □ Good top speed □ Poor top speed
Your feeling: _______________________________________________
Suggested adjustment: Front/rear wing balance, Differential, Suspension geometry""",
    },
    "sector_tyres": {
        "id": "sector_tyres",
        "granularity": "sector",
        "category": "tyres",
        "zh": """【赛段轮胎反馈】
赛段: _____ (S1 / S2 / S3)
轮胎温度: □ 太冷 (抓地力差) □ 刚好 □ 太热 (开始滑动)
磨损状况: □ 前轮磨损更快 □ 后轮磨损更快 □ 左侧更多 □ 右侧更多
轮胎类型: □ 软胎 □ 中等胎 □ 硬胎 □ 半雨胎 □ 全雨胎
你的感受: _______________________________________________
建议调整方向: 胎压, 外倾角, 前束角, 驾驶风格""",
        "en": """[Sector Tyre Feedback]
Sector: _____ (S1 / S2 / S3)
Tyre temperature: □ Too cold (poor grip) □ Just right □ Too hot (sliding)
Wear pattern: □ Front wearing faster □ Rear wearing faster □ Left side more □ Right side more
Compound: □ Soft □ Medium □ Hard □ Intermediate □ Wet
Your feeling: _______________________________________________
Suggested adjustment: Tyre pressure, Camber, Toe, Driving style""",
    },
    # ==== overall 级模板 ====
    "overall_general": {
        "id": "overall_general",
        "granularity": "overall",
        "category": "general",
        "zh": """【整体驾驶反馈】
赛道: _____
圈速: _____ (你的最佳圈速)
天气: □ 晴天 □ 多云 □ 小雨 □ 大雨 □ 变化中
赛道温度: □ 冷 □ 适中 □ 热
整体感受:
  1. 车的平衡性如何? _________________________________
  2. 哪个赛段最有信心? _______________________________
  3. 哪个赛段最没信心? _______________________________
  4. 轮胎表现如何? ___________________________________
  5. 刹车表现如何? ___________________________________
  6. ERS 使用感觉如何? _______________________________
  7. 最大的一个问题是什么? ____________________________
  8. 希望在哪些方面改进? _____________________________
其他反馈: _______________________________________________""",
        "en": """[Overall Driving Feedback]
Track: _____
Lap time: _____ (your best lap)
Weather: □ Clear □ Overcast □ Light rain □ Heavy rain □ Changing
Track temperature: □ Cold □ Moderate □ Hot
Overall feel:
  1. How is the car balance? ______________________________
  2. Which sector do you feel most confident? ______________
  3. Which sector do you feel least confident? _____________
  4. How are the tyres performing? ________________________
  5. How are the brakes performing? _______________________
  6. How does ERS deployment feel? ________________________
  7. What is the single biggest issue? _____________________
  8. What areas would you like to improve? ________________
Additional feedback: _______________________________________""",
    },
    "overall_setup": {
        "id": "overall_setup",
        "granularity": "overall",
        "category": "setup",
        "zh": """【调教反馈】
当前调教满意度: □ 很满意 □ 还行 □ 不太满意 □ 需要大改
需要调整的方向 (可多选):
  □ 前翼下压力 (+ / -)
  □ 后翼下压力 (+ / -)
  □ 差速器 (油门/收油)
  □ 悬挂硬度 (前/后)
  □ 防倾杆 (前/后)
  □ 离地间隙 (前/后)
  □ 刹车 (压力/偏置)
  □ 胎压 (前/后)
  □ 外倾角/前束角
  □ 燃油负载
具体需求: _______________________________________________
期望圈速提升: _____ 秒""",
        "en": """[Setup Feedback]
Current setup satisfaction: □ Very satisfied □ Okay □ Not satisfied □ Needs major changes
Areas needing adjustment (select multiple):
  □ Front wing downforce (+ / -)
  □ Rear wing downforce (+ / -)
  □ Differential (on/off throttle)
  □ Suspension stiffness (front/rear)
  □ Anti-roll bar (front/rear)
  □ Ride height (front/rear)
  □ Brakes (pressure/bias)
  □ Tyre pressure (front/rear)
  □ Camber/Toe
  □ Fuel load
Specific needs: _______________________________________________
Expected lap time improvement: _____ seconds""",
    },
    "overall_ers": {
        "id": "overall_ers",
        "granularity": "overall",
        "category": "ers",
        "zh": """【ERS 使用反馈】
ERS 部署模式: □ 自动 □ Hotlap □ Overtake □ 手动
电池管理: □ 总是不够用 □ 刚好够用 □ 总是有剩余
回收效率: □ 回收不够 □ 回收刚好 □ 回收过多
关键部署点:
  1. 主直道: □ 全功率 □ 部分功率 □ 不使用
  2. 出弯: □ 全功率 □ 部分功率 □ 不使用
你的感受: _______________________________________________
建议调整: _______________________________________________""",
        "en": """[ERS Usage Feedback]
ERS deployment mode: □ Auto □ Hotlap □ Overtake □ Manual
Battery management: □ Always running low □ Just enough □ Always surplus
Recovery efficiency: □ Not enough recovery □ Just right □ Too much recovery
Key deployment points:
  1. Main straight: □ Full power □ Partial power □ No deployment
  2. Corner exit: □ Full power □ Partial power □ No deployment
Your feeling: _______________________________________________
Suggested adjustment: _______________________________________""",
    },
    "overall_comparison": {
        "id": "overall_comparison",
        "granularity": "overall",
        "category": "comparison",
        "zh": """【对比反馈】
对比对象: □ 队友 □ 上一圈 □ 上一节练习 □ 理想圈速
差距: _____ 秒 (比你快/慢)
主要差距在哪:
  □ 直道尾速
  □ 刹车点/刹车距离
  □ 弯中速度
  □ 出弯加速
  □ 赛道利用
  □ 轮胎管理
  □ ERS 部署
你的感受: _______________________________________________
你认为可以在哪些方面改进: _________________________________""",
        "en": """[Comparison Feedback]
Comparison target: □ Teammate □ Previous lap □ Previous session □ Ideal lap
Gap: _____ seconds (faster/slower than you)
Where is the main gap:
  □ Straight-line top speed
  □ Braking point/distance
  □ Mid-corner speed
  □ Corner exit acceleration
  □ Track usage
  □ Tyre management
  □ ERS deployment
Your feeling: _______________________________________________
Where do you think you can improve: _________________________""",
    },
}

# 模板分组 (按使用场景)
FEEDBACK_TEMPLATE_GROUPS: dict[str, list[str]] = {
    "corner": ["corner_understeer", "corner_oversteer", "corner_braking", "corner_traction"],
    "sector": ["sector_balance", "sector_tyres"],
    "overall": ["overall_general", "overall_setup", "overall_ers", "overall_comparison"],
    "all": [
        "corner_understeer", "corner_oversteer", "corner_braking", "corner_traction",
        "sector_balance", "sector_tyres",
        "overall_general", "overall_setup", "overall_ers", "overall_comparison",
    ],
}


def render_feedback_template(
    template_id: str,
    language: str = "zh",
    prefill: dict[str, str] | None = None,
) -> str:
    """渲染车手反馈模板.

    Args:
        template_id: 模板 ID (如 ``"overall_general"``).
        language: 语言 (``"zh"`` 或 ``"en"``).
        prefill: 可选的预填值字典 (如 ``{"track": "Suzuka"}``).

    Returns:
        渲染后的模板文本.

    Raises:
        KeyError: 模板 ID 不存在.
    """
    template = DRIVER_FEEDBACK_TEMPLATES.get(template_id)
    if template is None:
        available = ", ".join(sorted(DRIVER_FEEDBACK_TEMPLATES.keys()))
        raise KeyError(
            f"Unknown template '{template_id}'. Available: {available}"
        )
    text = template.get(language, template.get("zh", ""))
    if prefill:
        for key, value in prefill.items():
            text = text.replace(f"_____ ({key})", f"{value}")
            text = text.replace(f"_____ ({key})", f"{value}")
    return text
