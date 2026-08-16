"""F1 2026 赛道数据库.

定义 :class:`Track` pydantic 模型以及 24 条 FIA / Formula 1 官方
2026 赛历赛道元数据 (3 月 6 日墨尔本揭幕 → 12 月 6 日阿布扎比收官,
6 个 Sprint 周末, 马德里 IFEMA 首秀). 所有数据按赛历轮次顺序排列.

数据来源为 FIA 世界汽车运动理事会批准的官方 2026 赛历
(https://www.formula1.com/en/latest/article/formula-1-reveals-calendar-for-2026-season).
赛道长度 / 弯角数 / 海拔变化采用当前已知官方值; 马德里 IFEMA 新赛道
待 FIA 赛道认证, 相关数值为估算并在 :attr:`Track.notes` 中显式标注.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

TrackType = Literal[
    "high_speed_low_downforce",
    "street",
    "high_downforce",
    "medium",
    "mixed",
]
"""赛道调教分类.

- ``high_speed_low_downforce``: 长直道为主, 低下压力 (Monza/Baku/Jeddah/
  Montreal/Las Vegas)
- ``street``: 街道赛 / 街道-半永久混合 (Monaco/Singapore/Madrid/Miami)
- ``high_downforce``: 高下压力技术赛道 (Hungaroring/Zandvoort)
- ``medium``: 中等下压力, 攻守均衡
- ``mixed``: 三段差异大, 需要折中调教 (如 Suzuka/Silverstone/Spa/COTA)
"""

# Iter-192: 赛道特性扩展数据 (grip_level, kerb_sensitivity, top_speed_kmh)
_TRACK_CHARACTERISTICS: dict[str, dict[str, float | None]] = {
    "melbourne": {"grip_level": 0.75, "kerb_sensitivity": 0.4, "top_speed_kmh": 325},
    "shanghai": {"grip_level": 0.80, "kerb_sensitivity": 0.5, "top_speed_kmh": 330},
    "suzuka": {"grip_level": 0.85, "kerb_sensitivity": 0.7, "top_speed_kmh": 320},
    "sakhir": {"grip_level": 0.70, "kerb_sensitivity": 0.3, "top_speed_kmh": 325},
    "jeddah": {"grip_level": 0.90, "kerb_sensitivity": 0.2, "top_speed_kmh": 335},
    "miami": {"grip_level": 0.65, "kerb_sensitivity": 0.5, "top_speed_kmh": 320},
    "montreal": {"grip_level": 0.60, "kerb_sensitivity": 0.8, "top_speed_kmh": 330},
    "monaco": {"grip_level": 0.55, "kerb_sensitivity": 0.9, "top_speed_kmh": 290},
    "barcelona": {"grip_level": 0.80, "kerb_sensitivity": 0.5, "top_speed_kmh": 325},
    "spielberg": {"grip_level": 0.75, "kerb_sensitivity": 0.6, "top_speed_kmh": 330},
    "silverstone": {"grip_level": 0.85, "kerb_sensitivity": 0.5, "top_speed_kmh": 325},
    "spa": {"grip_level": 0.80, "kerb_sensitivity": 0.6, "top_speed_kmh": 330},
    "hungaroring": {"grip_level": 0.75, "kerb_sensitivity": 0.8, "top_speed_kmh": 310},
    "zandvoort": {"grip_level": 0.80, "kerb_sensitivity": 0.7, "top_speed_kmh": 315},
    "monza": {"grip_level": 0.70, "kerb_sensitivity": 0.4, "top_speed_kmh": 350},
    "madrid": {"grip_level": 0.70, "kerb_sensitivity": 0.5, "top_speed_kmh": 340},
    "baku": {"grip_level": 0.65, "kerb_sensitivity": 0.4, "top_speed_kmh": 345},
    "singapore": {"grip_level": 0.60, "kerb_sensitivity": 0.9, "top_speed_kmh": 305},
    "austin": {"grip_level": 0.80, "kerb_sensitivity": 0.6, "top_speed_kmh": 325},
    "mexico_city": {"grip_level": 0.65, "kerb_sensitivity": 0.5, "top_speed_kmh": 320},
    "sao_paulo": {"grip_level": 0.70, "kerb_sensitivity": 0.7, "top_speed_kmh": 320},
    "las_vegas": {"grip_level": 0.60, "kerb_sensitivity": 0.3, "top_speed_kmh": 345},
    "lusail": {"grip_level": 0.80, "kerb_sensitivity": 0.5, "top_speed_kmh": 330},
    "yas_marina": {"grip_level": 0.75, "kerb_sensitivity": 0.4, "top_speed_kmh": 330},
}


def get_track_characteristics(track_id: str) -> dict[str, float | None]:
    """Iter-192: 获取赛道特性 (grip_level, kerb_sensitivity, top_speed_kmh)."""
    return _TRACK_CHARACTERISTICS.get(track_id, {"grip_level": 0.75, "kerb_sensitivity": 0.5, "top_speed_kmh": 325})


class Track(BaseModel):
    """单条 F1 赛道元数据.

    所有字段对应 FIA / Formula 1 官方 2026 赛历条目; 估算字段
    (主要涉及马德里 IFEMA 新赛道) 在 :attr:`notes` 中标注.
    """

    model_config = ConfigDict(frozen=True)

    track_id: str
    """短规范 id, 例如 ``"melbourne"`` / ``"yas_marina"``."""

    official_name: str
    """官方大奖赛名称, 例如 ``"Australian Grand Prix"``."""

    circuit_name: str
    """官方赛道名称."""

    city: str
    country: str
    country_code: str
    """ISO-3166 alpha-2 国家/地区代码 (小写), 例如 ``"au"``."""

    round_number: int
    """2026 赛历轮次 (1..24)."""

    date_range: str
    """比赛周末日期区间 (周五/周日), 格式 ``YYYY-MM-DD/YYYY-MM-DD``."""

    is_sprint: bool
    """是否为 Sprint 周末."""

    length_m: float
    """当前赛道布局官方长度 (米)."""

    corners: int
    """官方弯角数."""

    elevation_change_m: float
    """赛道最大海拔变化 (米), 估算值见 :attr:`notes`."""

    track_type: TrackType
    notes: str


ALL_TRACKS: list[Track] = [
    Track(
        track_id="melbourne",
        official_name="Australian Grand Prix",
        circuit_name="Albert Park Grand Prix Circuit",
        city="Melbourne",
        country="Australia",
        country_code="au",
        round_number=1,
        date_range="2026-03-06/2026-03-08",
        is_sprint=False,
        length_m=5278.0,
        corners=14,
        elevation_change_m=5.0,
        track_type="medium",
        notes="2026 season opener at Albert Park, semi-street flowing layout.",
    ),
    Track(
        track_id="shanghai",
        official_name="Chinese Grand Prix",
        circuit_name="Shanghai International Circuit",
        city="Shanghai",
        country="China",
        country_code="cn",
        round_number=2,
        date_range="2026-03-13/2026-03-15",
        is_sprint=True,
        length_m=5451.0,
        corners=16,
        elevation_change_m=6.0,
        track_type="medium",
        notes="Sprint weekend; long straights combined with slow hairpins.",
    ),
    Track(
        track_id="suzuka",
        official_name="Japanese Grand Prix",
        circuit_name="Suzuka International Racing Course",
        city="Suzuka",
        country="Japan",
        country_code="jp",
        round_number=3,
        date_range="2026-03-27/2026-03-29",
        is_sprint=False,
        length_m=5807.0,
        corners=18,
        elevation_change_m=40.0,
        track_type="mixed",
        notes="Only F1 figure-eight circuit; fast S-curves require balanced setup.",
    ),
    Track(
        track_id="sakhir",
        official_name="Bahrain Grand Prix",
        circuit_name="Bahrain International Circuit",
        city="Sakhir",
        country="Bahrain",
        country_code="bh",
        round_number=4,
        date_range="2026-04-10/2026-04-12",
        is_sprint=False,
        length_m=5412.0,
        corners=15,
        elevation_change_m=6.0,
        track_type="medium",
        notes="Three DRS zones; medium-downforce, heavy braking and traction demands.",
    ),
    Track(
        track_id="jeddah",
        official_name="Saudi Arabian Grand Prix",
        circuit_name="Jeddah Corniche Circuit",
        city="Jeddah",
        country="Saudi Arabia",
        country_code="sa",
        round_number=5,
        date_range="2026-04-17/2026-04-19",
        is_sprint=False,
        length_m=6174.0,
        corners=27,
        elevation_change_m=3.0,
        track_type="high_speed_low_downforce",
        notes="Fastest street circuit on the calendar; 27 flat-out corners.",
    ),
    Track(
        track_id="miami",
        official_name="Miami Grand Prix",
        circuit_name="Miami International Autodrome",
        city="Miami",
        country="United States",
        country_code="us",
        round_number=6,
        date_range="2026-05-01/2026-05-03",
        is_sprint=True,
        length_m=5412.0,
        corners=19,
        elevation_change_m=3.0,
        track_type="street",
        notes="Sprint weekend; street-hybrid layout around Hard Rock Stadium.",
    ),
    Track(
        track_id="montreal",
        official_name="Canadian Grand Prix",
        circuit_name="Circuit Gilles Villeneuve",
        city="Montreal",
        country="Canada",
        country_code="ca",
        round_number=7,
        date_range="2026-05-22/2026-05-24",
        is_sprint=True,
        length_m=4361.0,
        corners=14,
        elevation_change_m=3.0,
        track_type="high_speed_low_downforce",
        notes="Sprint weekend; semi-street layout with long straights and heavy braking.",
    ),
    Track(
        track_id="monaco",
        official_name="Monaco Grand Prix",
        circuit_name="Circuit de Monaco",
        city="Monte Carlo",
        country="Monaco",
        country_code="mc",
        round_number=8,
        date_range="2026-06-05/2026-06-07",
        is_sprint=False,
        length_m=3337.0,
        corners=19,
        elevation_change_m=42.0,
        track_type="street",
        notes="Tightest circuit on the calendar; maximum downforce, qualifying-critical.",
    ),
    Track(
        track_id="barcelona",
        official_name="Gran Premio de Barcelona-Catalunya",
        circuit_name="Circuit de Barcelona-Catalunya",
        city="Barcelona",
        country="Spain",
        country_code="es",
        round_number=9,
        date_range="2026-06-12/2026-06-14",
        is_sprint=False,
        length_m=4657.0,
        corners=14,
        elevation_change_m=8.0,
        track_type="medium",
        notes="remains on 2026 calendar alongside Madrid.",
    ),
    Track(
        track_id="spielberg",
        official_name="Austrian Grand Prix",
        circuit_name="Red Bull Ring",
        city="Spielberg",
        country="Austria",
        country_code="at",
        round_number=10,
        date_range="2026-06-26/2026-06-28",
        is_sprint=False,
        length_m=4318.0,
        corners=10,
        elevation_change_m=65.0,
        track_type="medium",
        notes="Only 10 corners with three long straights; significant elevation change.",
    ),
    Track(
        track_id="silverstone",
        official_name="British Grand Prix",
        circuit_name="Silverstone Circuit",
        city="Silverstone",
        country="United Kingdom",
        country_code="gb",
        round_number=11,
        date_range="2026-07-03/2026-07-05",
        is_sprint=True,
        length_m=5891.0,
        corners=18,
        elevation_change_m=10.0,
        track_type="mixed",
        notes="Sprint weekend; high-speed Maggotts-Becketts-Chapel complex.",
    ),
    Track(
        track_id="spa",
        official_name="Belgian Grand Prix",
        circuit_name="Circuit de Spa-Francorchamps",
        city="Spa-Francorchamps",
        country="Belgium",
        country_code="be",
        round_number=12,
        date_range="2026-07-17/2026-07-19",
        is_sprint=False,
        length_m=7004.0,
        corners=19,
        elevation_change_m=102.0,
        track_type="mixed",
        notes="Longest circuit on the calendar; ~102m elevation, Eau Rouge-Raidillon.",
    ),
    Track(
        track_id="hungaroring",
        official_name="Hungarian Grand Prix",
        circuit_name="Hungaroring",
        city="Budapest",
        country="Hungary",
        country_code="hu",
        round_number=13,
        date_range="2026-07-24/2026-07-26",
        is_sprint=False,
        length_m=4381.0,
        corners=14,
        elevation_change_m=25.0,
        track_type="high_downforce",
        notes="Twisty and technical; 'Monaco without the walls'.",
    ),
    Track(
        track_id="zandvoort",
        official_name="Dutch Grand Prix",
        circuit_name="Circuit Zandvoort",
        city="Zandvoort",
        country="Netherlands",
        country_code="nl",
        round_number=14,
        date_range="2026-08-21/2026-08-23",
        is_sprint=True,
        length_m=4259.0,
        corners=14,
        elevation_change_m=10.0,
        track_type="high_downforce",
        notes="Sprint weekend; banked Hugenholtz and Arie Luyendyk corners.",
    ),
    Track(
        track_id="monza",
        official_name="Italian Grand Prix",
        circuit_name="Autodromo Nazionale Monza",
        city="Monza",
        country="Italy",
        country_code="it",
        round_number=15,
        date_range="2026-09-04/2026-09-06",
        is_sprint=False,
        length_m=5793.0,
        corners=11,
        elevation_change_m=15.0,
        track_type="high_speed_low_downforce",
        notes="'Temple of Speed'; lowest downforce configuration of the season.",
    ),
    Track(
        track_id="madrid",
        official_name="Spanish Grand Prix",
        circuit_name="Circuito de Madring",
        city="Madrid",
        country="Spain",
        country_code="es",
        round_number=16,
        date_range="2026-09-11/2026-09-13",
        is_sprint=False,
        length_m=5416.0,
        corners=22,
        elevation_change_m=26.0,
        track_type="street",
        notes=(
            "2026 debut (F1 returns to Madrid since 1981). Hybrid street-"
            "permanent layout at IFEMA + Valdebebas; FIA Grade 1. Signature "
            "'La Monumental' banked corner (T12): 550m, 24% banking, ~6s flat. "
            "Top speed ~340 km/h. Params verified via madring.com official + "
            "formula1.com 2026 calendar + multi-source cross-check (Iter-09)."
        ),
    ),
    Track(
        track_id="baku",
        official_name="Azerbaijan Grand Prix",
        circuit_name="Baku City Circuit",
        city="Baku",
        country="Azerbaijan",
        country_code="az",
        round_number=17,
        date_range="2026-09-24/2026-09-26",
        is_sprint=False,
        length_m=6003.0,
        corners=20,
        elevation_change_m=10.0,
        track_type="high_speed_low_downforce",
        notes="Longest straight on calendar (~2.2km); Saturday race in 2026.",
    ),
    Track(
        track_id="singapore",
        official_name="Singapore Grand Prix",
        circuit_name="Marina Bay Street Circuit",
        city="Singapore",
        country="Singapore",
        country_code="sg",
        round_number=18,
        date_range="2026-10-09/2026-10-11",
        is_sprint=True,
        length_m=4940.0,
        corners=19,
        elevation_change_m=5.0,
        track_type="street",
        notes="Sprint weekend; hot and humid night street race.",
    ),
    Track(
        track_id="austin",
        official_name="United States Grand Prix",
        circuit_name="Circuit of the Americas",
        city="Austin",
        country="United States",
        country_code="us",
        round_number=19,
        date_range="2026-10-23/2026-10-25",
        is_sprint=False,
        length_m=5513.0,
        corners=20,
        elevation_change_m=41.0,
        track_type="mixed",
        notes="COTA; sector 1 fast sweeps, sector 2 tight, sector 3 long straight.",
    ),
    Track(
        track_id="mexico_city",
        official_name="Mexico City Grand Prix",
        circuit_name="Autodromo Hermanos Rodriguez",
        city="Mexico City",
        country="Mexico",
        country_code="mx",
        round_number=20,
        date_range="2026-10-30/2026-11-01",
        is_sprint=False,
        length_m=4304.0,
        corners=17,
        elevation_change_m=3.0,
        track_type="mixed",
        notes="High altitude (~2286m); thin air reduces aero and cooling.",
    ),
    Track(
        track_id="sao_paulo",
        official_name="Sao Paulo Grand Prix",
        circuit_name="Autodromo Jose Carlos Pace",
        city="Sao Paulo",
        country="Brazil",
        country_code="br",
        round_number=21,
        date_range="2026-11-06/2026-11-08",
        is_sprint=False,
        length_m=4309.0,
        corners=15,
        elevation_change_m=20.0,
        track_type="mixed",
        notes="Interlagos; technical infield with long start/finish straight.",
    ),
    Track(
        track_id="las_vegas",
        official_name="Las Vegas Grand Prix",
        circuit_name="Las Vegas Strip Circuit",
        city="Las Vegas",
        country="United States",
        country_code="us",
        round_number=22,
        date_range="2026-11-19/2026-11-21",
        is_sprint=False,
        length_m=6201.0,
        corners=17,
        elevation_change_m=0.0,
        track_type="high_speed_low_downforce",
        notes="Saturday night street race along the Strip; ~6.2km, near-flat.",
    ),
    Track(
        track_id="lusail",
        official_name="Qatar Grand Prix",
        circuit_name="Lusail International Circuit",
        city="Lusail",
        country="Qatar",
        country_code="qa",
        round_number=23,
        date_range="2026-11-27/2026-11-29",
        is_sprint=False,
        length_m=5419.0,
        corners=16,
        elevation_change_m=5.0,
        track_type="medium",
        notes="Medium-fast flowing layout with single long straight; night race.",
    ),
    Track(
        track_id="yas_marina",
        official_name="Abu Dhabi Grand Prix",
        circuit_name="Yas Marina Circuit",
        city="Abu Dhabi",
        country="United Arab Emirates",
        country_code="ae",
        round_number=24,
        date_range="2026-12-04/2026-12-06",
        is_sprint=False,
        length_m=5281.0,
        corners=16,
        elevation_change_m=10.0,
        track_type="medium",
        notes="2026 season finale; redesigned in 2021 with more flowing corners.",
    ),
]


# --------------------------------------------------------------------------- #
# 中文本地化 (Iter-293): 赛道 / 国家 / 赛道类型中文名, 供 UI 与 API 展示.
# --------------------------------------------------------------------------- #
TRACK_NAME_CN: dict[str, str] = {
    "melbourne": "澳大利亚大奖赛 · 墨尔本",
    "shanghai": "中国大奖赛 · 上海",
    "suzuka": "日本大奖赛 · 铃鹿",
    "sakhir": "巴林大奖赛 · 萨基尔",
    "jeddah": "沙特阿拉伯大奖赛 · 吉达",
    "miami": "迈阿密大奖赛",
    "montreal": "加拿大大奖赛 · 蒙特利尔",
    "monaco": "摩纳哥大奖赛",
    "barcelona": "西班牙大奖赛 · 巴塞罗那",
    "spielberg": "奥地利大奖赛 · 红牛环",
    "silverstone": "英国大奖赛 · 银石",
    "spa": "比利时大奖赛 · 斯帕",
    "hungaroring": "匈牙利大奖赛 · 布达佩斯",
    "zandvoort": "荷兰大奖赛 · 赞德沃特",
    "monza": "意大利大奖赛 · 蒙扎",
    "madrid": "西班牙大奖赛 · 马德里",
    "baku": "阿塞拜疆大奖赛 · 巴库",
    "singapore": "新加坡大奖赛",
    "austin": "美国大奖赛 · 奥斯汀",
    "mexico_city": "墨西哥城大奖赛",
    "sao_paulo": "圣保罗大奖赛",
    "las_vegas": "拉斯维加斯大奖赛",
    "lusail": "卡塔尔大奖赛 · 卢赛尔",
    "yas_marina": "阿布扎比大奖赛 · 亚斯码头",
}

TRACK_COUNTRY_CN: dict[str, str] = {
    "Australia": "澳大利亚",
    "China": "中国",
    "Japan": "日本",
    "Bahrain": "巴林",
    "Saudi Arabia": "沙特阿拉伯",
    "United States": "美国",
    "Canada": "加拿大",
    "Monaco": "摩纳哥",
    "Spain": "西班牙",
    "Austria": "奥地利",
    "United Kingdom": "英国",
    "Belgium": "比利时",
    "Hungary": "匈牙利",
    "Netherlands": "荷兰",
    "Italy": "意大利",
    "Azerbaijan": "阿塞拜疆",
    "Singapore": "新加坡",
    "Mexico": "墨西哥",
    "Brazil": "巴西",
    "Qatar": "卡塔尔",
    "United Arab Emirates": "阿联酋",
}

TRACK_TYPE_CN: dict[str, str] = {
    "high_speed_low_downforce": "高速 · 低下压力",
    "street": "街道赛",
    "high_downforce": "高下压力",
    "medium": "中等下压力",
    "mixed": "混合型",
}


def track_name_cn(track_id: str) -> str:
    """返回赛道中文名, 未知 id 回退为原始 id."""
    return TRACK_NAME_CN.get(track_id, track_id)


def country_name_cn(country: str) -> str:
    """返回国家中文名, 未知国家回退为英文原名."""
    return TRACK_COUNTRY_CN.get(country, country)


def track_type_cn(track_type: str) -> str:
    """返回赛道调教分类中文名, 未知类型回退为原始值."""
    return TRACK_TYPE_CN.get(track_type, track_type)


TRACKS_BY_ID: dict[str, Track] = {t.track_id: t for t in ALL_TRACKS}


def all_tracks() -> list[Track]:
    """返回全部 24 条赛道 (按赛历轮次顺序)."""
    return list(ALL_TRACKS)


def sprint_tracks() -> list[Track]:
    """返回 6 条 Sprint 周末赛道 (按赛历轮次顺序)."""
    return [t for t in ALL_TRACKS if t.is_sprint]


def get_track(track_id: str) -> Track:
    """按 ``track_id`` 查询赛道; 不存在则抛出 :class:`ValueError`."""
    track = TRACKS_BY_ID.get(track_id)
    if track is None:
        raise ValueError(f"Unknown track_id: {track_id!r}")
    return track


def get_track_by_round(round_number: int) -> Track:
    """按赛历轮次查询赛道; 不存在则抛出 :class:`ValueError`."""
    for track in ALL_TRACKS:
        if track.round_number == round_number:
            return track
    raise ValueError(f"Unknown round_number: {round_number!r}")
