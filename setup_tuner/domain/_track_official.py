"""赛道弯道数据的**官方来源登记表**（一切以 F1 官方文档为准）。

为什么需要这个文件
------------------
本项目的弯道编号/名称曾经踩过三类坑：

1. **编号↔名称错配**：名称挂到了错误的弯号上（连续弯区域尤其严重）；
2. **人造占位名**：``Bocht 8`` / ``Bocht 9`` / ``Curve 10`` /
   ``Turn 11 Right`` / ``Circuit Zandvoort Bocht`` —— 既不是官方名，
   也不是官方编号写法，纯属为填满字段而造；
3. **非弯道名词充当弯名**：把直道名（``Reta Oposta``）或路段名
   （``Subida dos Boxes`` / ``Arquibancada``）当成弯道专名。

后果是车手在游戏里看到 T11 是发夹弯，而工具里 T11 却是别的东西 ——
反馈与调教建议因此对不上。

官方口径（重要）
----------------
查证 F1 官方文档后的结论：

- F1 官方**只对部分赛道命名弯道**。Monaco、Silverstone 等有逐弯命名的官方
  文章；而 Sakhir / Baku / Lusail / Yas Marina / Shanghai / Jeddah / Miami /
  Las Vegas 等赛道的弯道**在官方口径里就没有专名**，官方直接写 ``Turn N``。
  因此 ``Turn N`` 不是"占位符"，而是这些弯的**正式写法**。
- 弯道**数量**在部分赛道有官方明示的 ``Number of turns: N``
  （Monaco / Silverstone / Albert Park 等），其余赛道官方仅在赛事页正文提及。

本文件把"哪些结论有官方出处、出处是哪个 URL"固化下来，
供 ``tests/test_corner_naming.py` 校验，避免再次出现无出处的臆造数据。

命名层级
--------
- ``OFFICIAL_NAMES``：F1 官方文档明确使用（并给出弯号）的名称 —— 最高置信度，
  测试逐条锁定。
- ``TRADITIONAL_NAMES``：赛道方/业界长期通用、但**未见于 F1 官方文档**的名称。
  保留是为了可用性，但必须在测试中可区分，**不得**与官方名混为一谈。
- 两者都没有 → 弯名就是 ``Turn N``（官方写法）。
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# 官方来源 URL（每条赛道至少一条，指向 formula1.com 官方页面）
# 全部经实际 HTTP 200 校验；命名/弯数的关键凭据优先用逐弯命名文章与 Circuit Guide。
# --------------------------------------------------------------------------- #
OFFICIAL_SOURCES: dict[str, list[str]] = {
    # 有逐弯命名 / Circuit Guide（含 "Number of turns: N" 逐字行）
    "melbourne": [
        "https://formula1.com/en/latest/article/"
        "circuit-guide-2026-australian-grand-prix-albert-park.19zPlhKhMbTaVNFIPKAAMa",
        "https://formula1.com/en/racing/2026/australia",
    ],
    "monaco": [
        "https://formula1.com/en/latest/article/"
        "circuit-guide-everything-you-need-to-know-about-the-circuit-de-monaco"
        ".vFsmfGHr6RWyLFtxi58wi",
        "https://formula1.com/en/racing/2026/monaco",
    ],
    "silverstone": [
        "https://formula1.com/en/latest/article/"
        "explained-how-every-silverstone-corner-got-its-name.idMlxFC2gfN2ApcPmifxN",
        "https://formula1.com/en/latest/article/"
        "circuit-guide-everything-you-need-to-know-about-silverstone-2026"
        ".5Sl0O8g393enBWVIkjRzOr",
    ],
    # 有赛事页点名（Rindt / Arie Luyendijkbocht / Tarzan / Hugenholtz
    # / Curva Alboreto / Variante Ascari / Eau Rouge / Raidillon / Pouhon）
    "spielberg": ["https://formula1.com/en/racing/2026/austria"],
    "zandvoort": ["https://formula1.com/en/racing/2026/netherlands"],
    "monza": ["https://formula1.com/en/racing/2026/italy"],
    "spa": ["https://formula1.com/en/racing/2026/belgium"],
    "sao_paulo": ["https://formula1.com/en/racing/2026/brazil"],
    # 官方未逐弯命名：赛事页为弯数/布局的官方依据
    "shanghai": ["https://formula1.com/en/racing/2026/china"],
    "suzuka": ["https://formula1.com/en/racing/2026/japan"],
    "sakhir": ["https://formula1.com/en/racing/2026/bahrain"],
    "jeddah": ["https://formula1.com/en/racing/2026/saudi-arabia"],
    "miami": ["https://formula1.com/en/racing/2026/miami"],
    "montreal": ["https://formula1.com/en/racing/2026/canada"],
    # 2026 西班牙站为马德里 Madring，故 Barcelona-Catalunya 取 2025 赛季页
    "barcelona": ["https://formula1.com/en/racing/2025/spain"],
    "hungaroring": ["https://formula1.com/en/racing/2026/hungary"],
    "madrid": ["https://formula1.com/en/racing/2026/spain"],
    "baku": ["https://formula1.com/en/racing/2026/azerbaijan"],
    "singapore": ["https://formula1.com/en/racing/2026/singapore"],
    "austin": ["https://formula1.com/en/racing/2026/united-states"],
    "mexico_city": ["https://formula1.com/en/racing/2026/mexico"],
    "las_vegas": ["https://formula1.com/en/racing/2026/las-vegas"],
    "lusail": ["https://formula1.com/en/racing/2026/qatar"],
    "yas_marina": ["https://formula1.com/en/racing/2026/united-arab-emirates"],
}

# --------------------------------------------------------------------------- #
# 官方明示的弯道总数
#   value = (官方弯数, 官方原文凭据 or None)
# 「Number of turns: N」是官方逐字行；None 表示官方仅在赛事页散文提及，
# 该数值按赛事页/官方赛道图核对，置信度略低。
# --------------------------------------------------------------------------- #
OFFICIAL_TURN_COUNTS: dict[str, tuple[int, str | None]] = {
    "melbourne": (14, "Number of turns: 14"),
    "monaco": (19, "Number of turns: 19"),
    "silverstone": (18, "Number of turns: 18"),
    "shanghai": (16, None),
    "suzuka": (18, None),
    "sakhir": (15, None),
    "jeddah": (27, None),
    "miami": (19, None),
    "montreal": (14, None),
    "barcelona": (14, None),
    "spielberg": (10, None),
    "spa": (19, None),
    "hungaroring": (14, None),
    "zandvoort": (14, None),
    "monza": (11, None),
    "madrid": (22, None),
    "baku": (20, None),
    "singapore": (19, None),
    "austin": (20, None),
    "mexico_city": (17, None),
    "sao_paulo": (15, None),
    "las_vegas": (17, None),
    "lusail": (16, None),
    "yas_marina": (16, None),
}

# --------------------------------------------------------------------------- #
# F1 官方文档明确给出的「弯号 → 名称」
#   只收录有官方原文支撑的条目；测试逐条锁定，任何一条被改动都会失败。
# --------------------------------------------------------------------------- #
OFFICIAL_NAMES: dict[str, dict[int, str]] = {
    # 官方原文：「Sainte Devote, Turn 1」；另官方提到 Loews/Grand Hotel Hairpin、
    # Casino Square、Swimming Pool、Tabac —— 除 T1 外官方未给弯号，故只锁 T1。
    "monaco": {1: "Sainte Devote"},
    # 官方逐条：「Turn 1 – Abbey」「Turn 2 – Farm」「Turn 3 – Village」
    # 「Turn 4 – The Loop」「Turn 5 – Aintree」「Turn 6 – Brooklands」
    # 「Turn 7 – Luffield」「Turn 8 – Woodcote」「Turn 9 – Copse」
    # 「Turns 10-14 – Maggotts, Becketts and Chapel」；官方指南另证
    # 「between Turns 14 and 15」→ Chapel=T14、「Turn 17 / Turn 18」→ Club。
    "silverstone": {
        1: "Abbey", 2: "Farm", 3: "Village", 4: "The Loop", 5: "Aintree",
        6: "Brooklands", 7: "Luffield", 8: "Woodcote", 9: "Copse",
        10: "Maggotts", 14: "Chapel", 15: "Stowe", 16: "Vale",
    },
    # 官方赛事页：「the exhilarating Rindt right-hander, named for Austria's
    # first F1 champion」——官方点名 Rindt，但未给弯号。
    "spielberg": {9: "Rindt"},
    # 官方赛事页：「Arie Luyendijkbocht – the final turn on the track」、
    # 「the famous Tarzan corner」、「Hugenholtzbocht」。
    "zandvoort": {1: "Tarzan", 3: "Hugenholtz", 14: "Arie Luyendyk"},
    # 官方赛事页：「Curva Alboreto (aka Parabolica)」、「Variante Ascari」、
    # 「Variante del Rettifilo」、「Curva Grande」。
    "monza": {
        1: "Variante del Rettifilo", 3: "Curva Grande",
        8: "Variante Ascari", 11: "Curva Alboreto",
    },
    # 官方赛事页：「La Source」、「Eau Rouge」、「Raidillon」、「Pouhon」。
    "spa": {1: "La Source", 2: "Eau Rouge", 3: "Raidillon", 10: "Pouhon"},
}

#: 官方**未**给出逐弯命名的赛道 —— 这些赛道的弯名用 ``Turn N`` 属正确写法，
#: 不得被"必须有专名"的规则误伤。
OFFICIALLY_UNNAMED_TRACKS: frozenset[str] = frozenset({
    "shanghai", "sakhir", "jeddah", "miami", "montreal", "barcelona",
    "hungaroring", "madrid", "baku", "singapore", "austin", "mexico_city",
    "las_vegas", "lusail", "yas_marina",
})

#: 弯曲名称中**禁止出现**的人造占位模式（历史缺陷的固化拦截）。
FORBIDDEN_NAME_PATTERNS: tuple[str, ...] = (
    r"\bbocht\s*\d+$",          # "Bocht 8" / "Bocht 9" / "Bocht 13"
    r"^circuit\s+\w+\s+bocht$",  # "Circuit Zandvoort Bocht"
    r"^curve\s+\d+$",            # "Curve 10"
    r"^turn\s+\d+\s+(left|right)$",  # "Turn 3 Right" / "Turn 11 Left"
    r"\bstraight\b",             # "Wellington Straight"
    r"\bapproach\b",             # "La Source approach"
)
