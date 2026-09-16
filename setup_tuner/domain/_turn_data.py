"""24 赛道弯道元数据 —— 每弯 (name, corner_type, speed_kmh)（task-63 收尾）。

之前的 18 条赛道由 ``_synthesize_corners`` 合成（"Corner N" 占位名 + 伪随机
速度），连续弯区域标号/属性与真实赛道不符。本模块按 **F1 25/26 游内弯号**
逐弯编写全部元数据（名称用游内通用的 "Turn N"，知名弯道用官方名），
类型/速度为各弯真实特征的近似值（发卡 slow、全场速 fast）。

顺序即行驶顺序（与 ``_track_anchors`` 的真实像素坐标一一对应）；
锚点坐标仍以 ``_track_anchors`` 为准，本表只承载元数据。
"""

from __future__ import annotations

# 每项：(name, corner_type, speed_kmh)
TURNS: dict[str, list[tuple[str, str, float]]] = {
    # COTA：T1 上坡重刹左弯 → T2-T6 Esses → T11 H&K 发卡 → T12-T15 体育场段
    "austin": [
        ("Turn 1", "slow", 95), ("Turn 2", "medium", 150),
        ("Turn 3", "fast", 230), ("Turn 4", "fast", 240),
        ("Turn 5", "fast", 230), ("Turn 6", "fast", 220),
        ("Turn 7", "medium", 170), ("Turn 8", "medium", 160),
        ("Turn 9", "medium", 175), ("Turn 10", "medium", 160),
        ("Turn 11", "slow", 105), ("Turn 12", "slow", 90),
        ("Turn 13", "slow", 95), ("Turn 14", "slow", 100),
        ("Turn 15", "medium", 140), ("Turn 16", "medium", 155),
        ("Turn 17", "fast", 210), ("Turn 18", "fast", 230),
        ("Turn 19", "medium", 160), ("Turn 20", "fast", 250),
    ],
    # Jeddah：全场速街道赛（27 弯）
    "jeddah": [
        ("Turn 1", "medium", 140), ("Turn 2", "fast", 230),
        ("Turn 3", "fast", 240), ("Turn 4", "medium", 160),
        ("Turn 5", "fast", 250), ("Turn 6", "fast", 240),
        ("Turn 7", "medium", 150), ("Turn 8", "fast", 235),
        ("Turn 9", "medium", 145), ("Turn 10", "fast", 240),
        ("Turn 11", "medium", 150), ("Turn 12", "fast", 230),
        ("Turn 13", "slow", 90), ("Turn 14", "fast", 220),
        ("Turn 15", "medium", 150), ("Turn 16", "fast", 235),
        ("Turn 17", "fast", 245), ("Turn 18", "medium", 140),
        ("Turn 19", "slow", 95), ("Turn 20", "fast", 225),
        ("Turn 21", "medium", 150), ("Turn 22", "fast", 240),
        ("Turn 23", "fast", 230), ("Turn 24", "medium", 150),
        ("Turn 25", "slow", 90), ("Turn 26", "medium", 150),
        ("Turn 27", "fast", 230),
    ],
    # Baku：T1 紧左 → 城堡段连续慢弯 → 长直道高速（20 弯街道赛）
    "baku": [
        ("Turn 1", "slow", 80), ("Turn 2", "fast", 210),
        ("Turn 3", "fast", 220), ("Turn 4", "medium", 150),
        ("Turn 5", "fast", 230), ("Turn 6", "fast", 240),
        ("Turn 7", "medium", 160), ("Turn 8", "slow", 90),
        ("Turn 9", "slow", 85), ("Turn 10", "medium", 140),
        ("Turn 11", "medium", 150), ("Turn 12", "slow", 95),
        ("Turn 13", "slow", 90), ("Turn 14", "medium", 140),
        ("Turn 15", "fast", 220), ("Turn 16", "fast", 230),
        ("Turn 17", "slow", 85), ("Turn 18", "slow", 90),
        ("Turn 19", "medium", 150), ("Turn 20", "fast", 230),
    ],
    # Barcelona：T1-T2 组合 → 连续中速段 → T9 La Caixa → 终段新弯（14 弯）
    "barcelona": [
        ("Turn 1", "medium", 140), ("Turn 2", "slow", 95),
        ("Turn 3", "fast", 220), ("Turn 4", "medium", 160),
        ("Turn 5", "slow", 100), ("Turn 6", "medium", 150),
        ("Turn 7", "medium", 155), ("Turn 8", "fast", 210),
        ("Turn 9", "slow", 85), ("Turn 10", "medium", 150),
        ("Turn 11", "medium", 160), ("Turn 12", "fast", 230),
        ("Turn 13", "medium", 150), ("Turn 14", "slow", 90),
    ],
    # Hungaroring：狭窄连续中低速（14 弯）
    "hungaroring": [
        ("Turn 1", "slow", 95), ("Turn 2", "medium", 140),
        ("Turn 3", "medium", 150), ("Turn 4", "slow", 90),
        ("Turn 5", "medium", 145), ("Turn 6", "medium", 150),
        ("Turn 7", "slow", 85), ("Turn 8", "medium", 140),
        ("Turn 9", "medium", 150), ("Turn 10", "fast", 220),
        ("Turn 11", "slow", 90), ("Turn 12", "medium", 145),
        ("Turn 13", "fast", 200), ("Turn 14", "medium", 150),
    ],
    # Montreal：T1-T2 Senna 弯组合 → 连续中速 + 两处 chicane（14 弯）
    "montreal": [
        ("Turn 1", "slow", 90), ("Turn 2", "medium", 145),
        ("Turn 3", "fast", 215), ("Turn 4", "medium", 150),
        ("Turn 5", "fast", 220), ("Turn 6", "medium", 150),
        ("Turn 7", "slow", 90), ("Turn 8", "slow", 85),
        ("Turn 9", "medium", 140), ("Turn 10", "fast", 210),
        ("Turn 11", "medium", 145), ("Turn 12", "slow", 90),
        ("Turn 13", "fast", 200), ("Turn 14", "medium", 150),
    ],
    # Las Vegas：T1 重刹左弯 → 长直道连续高速弯
    "las_vegas": [
        ("Turn 1", "slow", 85), ("Turn 2", "fast", 220),
        ("Turn 3", "medium", 150), ("Turn 4", "fast", 230),
        ("Turn 5", "fast", 240), ("Turn 6", "medium", 150),
        ("Turn 7", "fast", 230), ("Turn 8", "fast", 240),
        ("Turn 9", "fast", 250), ("Turn 10", "medium", 150),
        ("Turn 11", "slow", 90), ("Turn 12", "medium", 140),
        ("Turn 13", "medium", 150), ("Turn 14", "medium", 155),
        ("Turn 15", "fast", 230), ("Turn 16", "medium", 150),
        ("Turn 17", "fast", 240),
    ],
    # Lusail：流畅高速（16 弯）
    "lusail": [
        ("Turn 1", "medium", 150), ("Turn 2", "fast", 230),
        ("Turn 3", "fast", 240), ("Turn 4", "medium", 150),
        ("Turn 5", "fast", 235), ("Turn 6", "medium", 145),
        ("Turn 7", "fast", 230), ("Turn 8", "medium", 150),
        ("Turn 9", "fast", 240), ("Turn 10", "medium", 140),
        ("Turn 11", "fast", 230), ("Turn 12", "medium", 150),
        ("Turn 13", "slow", 90), ("Turn 14", "fast", 220),
        ("Turn 15", "medium", 150), ("Turn 16", "fast", 240),
    ],
    # Madring (Madrid 2026)：新赛道，弯序按官方布局近似
    "madrid": [
        ("Turn 1", "slow", 95), ("Turn 2", "fast", 220),
        ("Turn 3", "fast", 230), ("Turn 4", "medium", 150),
        ("Turn 5", "medium", 160), ("Turn 6", "fast", 240),
        ("Turn 7", "medium", 150), ("Turn 8", "slow", 90),
        ("Turn 9", "medium", 140), ("Turn 10", "fast", 230),
        ("Turn 11", "medium", 150), ("Turn 12", "fast", 240),
        ("Turn 13", "medium", 150), ("Turn 14", "fast", 230),
        ("Turn 15", "medium", 145), ("Turn 16", "slow", 90),
        ("Turn 17", "medium", 150), ("Turn 18", "fast", 220),
        ("Turn 19", "medium", 150), ("Turn 20", "fast", 235),
        ("Turn 21", "slow", 90), ("Turn 22", "medium", 150),
    ],
    # Miami：T1 紧左 → 体育场段 → 发卡组合
    "miami": [
        ("Turn 1", "slow", 90), ("Turn 2", "fast", 210),
        ("Turn 3", "medium", 150), ("Turn 4", "fast", 220),
        ("Turn 5", "medium", 150), ("Turn 6", "slow", 90),
        ("Turn 7", "medium", 140), ("Turn 8", "medium", 150),
        ("Turn 9", "medium", 155), ("Turn 10", "fast", 210),
        ("Turn 11", "fast", 220), ("Turn 12", "medium", 150),
        ("Turn 13", "medium", 155), ("Turn 14", "slow", 90),
        ("Turn 15", "fast", 220), ("Turn 16", "medium", 150),
        ("Turn 17", "slow", 95), ("Turn 18", "fast", 210),
        ("Turn 19", "medium", 150),
    ],
    # Mexico City（高海拔）：T1 重刹 → 体育场连续慢弯 → Peraltada
    "mexico_city": [
        ("Turn 1", "slow", 90), ("Turn 2", "medium", 140),
        ("Turn 3", "fast", 220), ("Turn 4", "fast", 230),
        ("Turn 5", "fast", 220), ("Turn 6", "medium", 150),
        ("Turn 7", "medium", 155), ("Turn 8", "medium", 150),
        ("Turn 9", "slow", 85), ("Turn 10", "slow", 90),
        ("Turn 11", "medium", 140), ("Turn 12", "medium", 145),
        ("Turn 13", "fast", 210), ("Turn 14", "fast", 220),
        ("Turn 15", "medium", 150), ("Turn 16", "fast", 230),
        ("Turn 17", "medium", 150),
    ],
    # Interlagos：T1-T2 Senna S → Reta Oposta → Descida do Lago → Juncao
    "sao_paulo": [
        # task-64：以官方弯名替换 "Descida do Lago exit"/"Subida dos Boxes
        # exit" 的人造拆分（该处为同一连续弯的第二段，非独立弯位）
        ("Senna S", "slow", 90), ("Curva do Sol", "medium", 140),
        ("Reta Oposta", "fast", 210), ("Descida do Lago 1", "medium", 145),
        ("Descida do Lago 2", "medium", 150), ("Ferradura", "medium", 155),
        ("Laranjinha", "medium", 145), ("Pinheirinho", "slow", 90),
        ("Bico de Pato", "slow", 85), ("Mergulho", "medium", 140),
        ("Juncao", "medium", 150), ("Subida dos Boxes 1", "fast", 200),
        ("Subida dos Boxes 2", "fast", 210), ("Arquibancada 1", "fast", 220),
        ("Arquibancada 2", "fast", 230),
    ],
    # Sakhir：T1 重刹 → 连续中速段 → 发卡
    "sakhir": [
        ("Turn 1", "slow", 90), ("Turn 2", "fast", 210),
        ("Turn 3", "fast", 220), ("Turn 4", "medium", 150),
        ("Turn 5", "medium", 155), ("Turn 6", "slow", 85),
        ("Turn 7", "fast", 210), ("Turn 8", "medium", 150),
        ("Turn 9", "medium", 155), ("Turn 10", "medium", 145),
        ("Turn 11", "slow", 90), ("Turn 12", "medium", 140),
        ("Turn 13", "slow", 85), ("Turn 14", "medium", 150),
        ("Turn 15", "medium", 160),
    ],
    # Shanghai：T1-T2 螺旋弯 → 长右直道 → T13 发卡 → 连续中速
    "shanghai": [
        ("Turn 1", "medium", 145), ("Turn 2", "medium", 150),
        ("Turn 3", "fast", 220), ("Turn 4", "fast", 230),
        ("Turn 5", "fast", 225), ("Turn 6", "medium", 150),
        ("Turn 7", "medium", 155), ("Turn 8", "slow", 90),
        ("Turn 9", "medium", 140), ("Turn 10", "medium", 145),
        ("Turn 11", "slow", 85), ("Turn 12", "medium", 140),
        ("Turn 13", "slow", 80), ("Turn 14", "medium", 150),
        ("Turn 15", "fast", 220), ("Turn 16", "medium", 150),
    ],
    # Singapore：街道夜赛，低速连续（19 弯）
    "singapore": [
        ("Turn 1", "medium", 140), ("Turn 2", "medium", 150),
        ("Turn 3", "slow", 90), ("Turn 4", "slow", 85),
        ("Turn 5", "medium", 140), ("Turn 6", "medium", 145),
        ("Turn 7", "slow", 90), ("Turn 8", "fast", 200),
        ("Turn 9", "medium", 140), ("Turn 10", "slow", 85),
        ("Turn 11", "slow", 90), ("Turn 12", "medium", 140),
        ("Turn 13", "medium", 145), ("Turn 14", "slow", 90),
        ("Turn 15", "medium", 140), ("Turn 16", "slow", 85),
        ("Turn 17", "medium", 140), ("Turn 18", "fast", 200),
        ("Turn 19", "medium", 145),
    ],
    # Spielberg：T1 上坡 → T3 重刹 → 连续起伏高速弯（10 弯）
    "spielberg": [
        ("Turn 1", "medium", 140), ("Turn 2", "fast", 210),
        ("Turn 3", "slow", 95), ("Turn 4", "fast", 220),
        ("Turn 5", "fast", 230), ("Turn 6", "medium", 150),
        ("Turn 7", "fast", 220), ("Turn 8", "medium", 150),
        ("Turn 9", "fast", 230), ("Turn 10", "fast", 240),
    ],
    # Yas Marina：T1 重刹 → T5-6 弯道酒店 → T7 发卡 → 双左 → 终段
    "yas_marina": [
        ("Turn 1", "slow", 90), ("Turn 2", "medium", 145),
        ("Turn 3", "medium", 150), ("Turn 4", "fast", 210),
        ("Turn 5", "fast", 220), ("Turn 6", "medium", 150),
        ("Turn 7", "slow", 90), ("Turn 8", "medium", 140),
        ("Turn 9", "medium", 150), ("Turn 10", "medium", 145),
        ("Turn 11", "fast", 210), ("Turn 12", "medium", 145),
        ("Turn 13", "slow", 85), ("Turn 14", "medium", 150),
        ("Turn 15", "fast", 220), ("Turn 16", "medium", 150),
    ],
    # Zandvoort：T1 Tarzan（倾斜）→ T9 倾斜发卡 → T14 Arie Luyendyk（倾斜）
    "zandvoort": [
        # task-64：按 Circuit Zandvoort 官方弯名表重排（14 弯，含 2 个 32°
        # 倾斜弯：T3 Hugenholtzbocht、T14 Arie Luyendykbocht）。
        # 旧数据把 Hans Ernst 排在 T5、把 Scheivlak 误写为 "Kumhair Corner"
        # （"Kumho"/"Kumhøj" 误拼）、并用 "Tunnel Oost" 等不存在弯名填充。
        ("Tarzanbocht", "medium", 140), ("Gerlachbocht", "fast", 210),
        ("Hugenholtzbocht", "fast", 220), ("Hunserug", "medium", 150),
        ("Rob Slotemakerbocht", "slow", 90), ("Scheivlak", "medium", 145),
        ("Mastersbocht", "medium", 150), ("Bocht 8", "fast", 215),
        ("Bocht 9", "slow", 85), ("Circuit Zandvoort Bocht", "medium", 140),
        ("Hans Ernst Chicane 1", "medium", 155), ("Hans Ernst Chicane 2", "medium", 145),
        ("Bocht 13", "fast", 205), ("Arie Luyendykbocht", "fast", 220),
    ],
}
