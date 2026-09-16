"""赛道 / 弯道领域模型与 24 条 F1 2026 赛历静态数据.

本模块为 F1OPT 调教优化助手的领域层（纯数据，无 IO），定义：
    - :class:`CornerAnchor` — SVG 归一化坐标锚点（0~1），用于可点击热区定位
    - :class:`Corner`       — 单个弯道定义（编号 / 名称 / 类型 / 预计速度 / 锚点）
    - :class:`Track`        — 单条赛道定义（元数据 + 弯道列表 + UDP 映射 + SVG 路径）
    - :data:`ALL_TRACKS`    — 24 条 F1 2026 赛历赛道（按赛历轮次顺序）

数据来源与核对：
    - 赛道元数据（track_id / official_name / circuit_name / city / country /
      track_type / length_m / 弯道数）核对自 ``legacy/f1opt/data/tracks.py``
      （FIA / Formula 1 官方 2026 赛历）。
    - 6 条手工录入赛道的弯道序列（melbourne / suzuka / monaco / silverstone /
      monza / spa）核对自 ``legacy/f1opt/data/corners.py`` 的手填数据
      （弯角编号 / 名称 / 类型 / apex 速度）。
    - 其余 18 条赛道的逐弯元数据由 :mod:`_turn_data` 按游内弯号提供（task-63）
      （量级准确，弯角名称用编号占位）。
    - ``udp_track_id`` 为 EA UDP 规范 Session 包 ``m_trackId`` 枚举值。
      取值来自官方枚举（F1 22–25 一致，由 f1-game-packet-parser 的 ``TrackId``
      枚举、raweceek f1-22-udp 附录、f1-telemetry-go ``tracks.md`` 三处交叉核对）：
      ``0 Melbourne / 2 Shanghai / 3 Sakhir / 4 Catalunya / 5 Monaco / 6 Montreal /
      7 Silverstone / 9 Hungaroring / 10 Spa / 11 Monza / 12 Marina Bay /
      13 Suzuka / 14 Yas Marina / 15 COTA / 16 Interlagos / 17 Red Bull Ring /
      19 Mexico City / 20 Baku / 26 Zandvoort / 29 Jeddah / 30 Miami /
      31 Las Vegas / 32 Losail``。
      **不再使用「赛历轮次 - 1」的占位分配**（那会让 24 条赛道里 20 条认错赛道）。
      ``madrid``（F1 25 2026 Season Pack 新增的 Madring）官方枚举尚无公开值，
      暂按 33 并记入 ``UDP_TRACK_ID_UNVERIFIED``，待官方规范确认后更正。
    - 弯道锚点 (anchor_x / anchor_y) 优先取自
      ``legacy/f1opt/data/track_maps/__init__.py`` 中各赛道 ``corners`` 的真实
      像素坐标（由 :func:`_apply_real_anchors` 用 ``x_px/canvas_width``、
      ``y_px/canvas_height`` 归一化到 [0, 1]）。未在 track_maps 中出现的弯道
      退回 :func:`_estimate_anchor` 的椭圆分布估算。
 
SVG 资产：24 条赛道 SVG 由 ``scripts/convert_track_svgs.py`` 从
``julesr0y/f1-circuits-svg`` 的真实赛道 SVG 提取并归一化到 800×600 画布，
输出到 ``setup_tuner/ui/tracks/``，文件名 = track_id（如 ``suzuka.svg``）；
对应的弯道锚点像素坐标由同一脚本生成到 ``_track_anchors.py``。
各弯道在路径上的弧长占比由 ``scripts/gen_track_arcs.py`` 生成到 ``_track_arcs.py``。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ._track_anchors import TRACK_ANCHORS, TRACK_CANVAS
from ._turn_data import TURNS as _AUTHORED_TURN_DATA

# UDP m_trackId 尚未经官方规范确认的赛道（用于测试与文档显式标注，避免"看起来已校准"）
UDP_TRACK_ID_UNVERIFIED: frozenset[str] = frozenset({"madrid"})

# --------------------------------------------------------------------------- #
# 数据类定义
# --------------------------------------------------------------------------- #

CornerType = Literal["slow", "medium", "fast"]
"""弯道类型：slow (<100 km/h) / medium (100-200) / fast (>200)。"""

TrackType = Literal[
    "high_speed_low_downforce",
    "street",
    "high_downforce",
    "medium",
    "mixed",
]
"""赛道调教分类（对齐 design.md 2.3.2 track.track_type）。"""


@dataclass(frozen=True)
class CornerAnchor:
    """SVG 归一化坐标锚点，用于可点击热区定位。

    坐标系：SVG 画布左上角为原点 (0, 0)，右下角为 (1, 1)。
    前端在 (anchor_x, anchor_y) 处绘制透明可点击圆形命中区。
    """

    anchor_x: float  # 0.0 ~ 1.0
    anchor_y: float  # 0.0 ~ 1.0


@dataclass(frozen=True)
class Corner:
    """弯道定义。

    对应 design.md 2.3.2 ``corner`` 表（不含 track_id 外键）。
    """

    number: int           # 1-based 弯道编号（按 F1 官方赛道图）
    name: str             # 弯道名称（如 "T1", "S-Curve", "Eau Rouge"）
    corner_type: CornerType  # slow | medium | fast
    speed_kmh: float      # 预计通过速度（apex 速度, km/h）
    anchor: CornerAnchor  # SVG 热区锚点（归一化坐标 0~1）


@dataclass
class Track:
    """赛道定义。

    对应 design.md 2.3.2 ``track`` 表 + 嵌套弯道列表。
    """

    track_id: str         # 短规范 id，如 'suzuka'
    official_name: str    # 官方大奖赛全名
    circuit_name: str     # 官方赛道名
    city: str
    country: str
    track_type: TrackType  # high_speed_low_downforce | street | high_downforce | medium | mixed
    length_m: float       # 赛道官方长度（米）
    corners: list[Corner]  # 弯道列表（1-based 编号，长度 = 弯道数）
    udp_track_id: int     # 遥测 m_trackId 数值（用于自动识别）
    svg_path: str         # SVG 资源路径（相对 ui/，如 "tracks/suzuka.svg"）


# --------------------------------------------------------------------------- #
# 锚点估算与真实坐标校准
# --------------------------------------------------------------------------- #

def _estimate_anchor(number: int, total: int) -> CornerAnchor:
    """沿椭圆分布估算弯道锚点（fallback，仅用于无真实坐标的弯道）。

    将弯道按编号沿一个中心椭圆均匀分布，模拟赛道轮廓形状。
    anchor_x / anchor_y 均落在 [0.1, 0.9] 区间内，严格在 (0, 1) 开区间内。

    .. note::
        真实坐标优先取自 ``_track_anchors.TRACK_ANCHORS``；本函数仅作为
        fallback，用于 track_maps 未覆盖的弯道（如 sakhir/miami/monaco
        中 track.py 弯道数多于 track_maps 的部分）。
    """
    angle = 2.0 * math.pi * (number - 1) / total
    return CornerAnchor(
        anchor_x=0.5 + 0.4 * math.cos(angle),
        anchor_y=0.5 + 0.4 * math.sin(angle),
    )


def _clamp01(v: float, eps: float = 1e-4) -> float:
    """把值夹到 (0, 1) 开区间内，避免边界 0/1 导致热区贴边。"""
    if v <= 0.0:
        return eps
    if v >= 1.0:
        return 1.0 - eps
    return v


def _apply_real_anchors(track_id: str, corners: list[Corner]) -> list[Corner]:
    """用 track_maps 真实像素坐标覆盖 corners 的 anchor。

    策略：
        1. 若 ``track_id`` 不在 ``TRACK_ANCHORS``，原样返回（保留椭圆估算）。
        2. 否则按 ``corner.number`` 在 ``TRACK_ANCHORS[track_id]`` 中查真实坐标；
           命中则用 ``x_px/canvas_width`` / ``y_px/canvas_height`` 归一化并夹到
           (0, 1) 开区间；未命中保留原 ``_estimate_anchor`` 估算值。

    返回新的 Corner 列表（不就地修改，保持纯函数）。
    """
    real = TRACK_ANCHORS.get(track_id)
    if not real:
        return corners
    canvas = TRACK_CANVAS.get(track_id, (800, 600))
    cw, ch = float(canvas[0]), float(canvas[1])
    out: list[Corner] = []
    for c in corners:
        px = real.get(c.number)
        if px is None:
            out.append(c)
            continue
        ax = _clamp01(px[0] / cw)
        ay = _clamp01(px[1] / ch)
        out.append(
            Corner(
                number=c.number,
                name=c.name,
                corner_type=c.corner_type,
                speed_kmh=c.speed_kmh,
                anchor=CornerAnchor(anchor_x=ax, anchor_y=ay),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 弯道数据构建辅助
# --------------------------------------------------------------------------- #

# 手工录入赛道的原始弯道数据：(number, name, corner_type, speed_kmh)
# 核对自 legacy/f1opt/data/corners.py 的手填函数。
_RawCorner = tuple[int, str, str, float]


def _build_corners(raw: list[_RawCorner]) -> list[Corner]:
    """从原始元组列表构建 Corner 列表（带估算锚点）。"""
    total = len(raw)
    return [
        Corner(
            number=n,
            name=name,
            corner_type=ct,  # type: ignore[arg-type]
            speed_kmh=spd,
            anchor=_estimate_anchor(n, total),
        )
        for n, name, ct, spd in raw
    ]


# --------------------------------------------------------------------------- #
# 6 条手工录入赛道的弯道原始数据（核对自 legacy corners.py）
# --------------------------------------------------------------------------- #

def _melbourne_corners() -> list[Corner]:
    """Albert Park Grand Prix Circuit (5.278 km, 14 弯, medium)。"""
    return _build_corners([
        (1, "Turn 1", "slow", 95),
        (2, "Turn 2", "medium", 130),
        (3, "Turn 3", "fast", 230),
        (4, "Turn 4", "slow", 85),
        (5, "Turn 5", "medium", 160),
        (6, "Turn 6", "fast", 210),
        (7, "Turn 7", "slow", 80),
        (8, "Turn 8", "medium", 150),
        (9, "Turn 9", "fast", 220),
        (10, "Turn 10", "slow", 90),
        (11, "Turn 11", "fast", 240),
        (12, "Turn 12", "medium", 155),
        (13, "Turn 13", "slow", 85),
        (14, "Turn 14", "medium", 140),
    ])


def _suzuka_corners() -> list[Corner]:
    """Suzuka International Racing Course (5.807 km, 18 弯, mixed)。

    task-63：按 F1 26 游内弯号重排（S-Curves=T3-T6、Dunlop=T7、
    Degner=T8/T9、Bridge Right=T10、Hairpin=T11、Spoon=T13/T14、
    130R=T15、Casio=T16-T18）。锚点取自 _track_anchors（真实 SVG 投影）。
    """
    return _build_corners([
        (1, "First Curve", "fast", 230),
        (2, "Second Curve", "medium", 150),
        (3, "S-Curves 1", "medium", 165),
        (4, "S-Curves 2", "medium", 175),
        (5, "S-Curves 3", "medium", 180),
        (6, "S-Curves 4", "medium", 170),
        (7, "Dunlop", "fast", 240),
        (8, "Degner 1", "medium", 115),
        (9, "Degner 2", "slow", 85),
        (10, "Bridge Right", "fast", 210),
        (11, "Hairpin", "slow", 70),
        (12, "Sweep Right", "fast", 250),
        (13, "Spoon Curve", "medium", 140),
        (14, "Spoon Exit", "medium", 155),
        (15, "130R", "fast", 295),
        (16, "Casio Chicane Right", "slow", 75),
        (17, "Casio Chicane Left", "slow", 80),
        (18, "Final Right", "medium", 140),
    ])


def _monaco_corners() -> list[Corner]:
    """Circuit de Monaco (3.337 km, 19 弯, street)。"""
    return _build_corners([
        (1, "Sainte-Devote", "slow", 80),
        (2, "Beau Rivage", "medium", 130),
        (3, "Massenet", "medium", 120),
        (4, "Casino", "medium", 135),
        (5, "Mirabeau Haut", "slow", 75),
        (6, "Mirabeau Bas", "slow", 65),
        (7, "Grand Hotel Hairpin", "slow", 45),
        (8, "Portier", "slow", 80),
        (9, "Tunnel entry", "medium", 145),
        (10, "Tunnel", "fast", 200),
        (11, "Tunnel exit", "medium", 155),
        (12, "Chicane", "slow", 70),
        (13, "Chicane exit", "slow", 75),
        (14, "Tabac", "medium", 140),
        (15, "Piscine entry", "medium", 150),
        (16, "Piscine", "medium", 135),
        (17, "Rascasse entry", "slow", 80),
        (18, "Rascasse", "slow", 70),
        (19, "Anthony Noghes", "medium", 125),
    ])


def _silverstone_corners() -> list[Corner]:
    """Silverstone Circuit (5.891 km, 18 弯, mixed)。"""
    return _build_corners([
        (1, "Abbey", "fast", 240),
        (2, "Farm", "medium", 175),
        (3, "Village", "medium", 140),
        (4, "The Loop", "slow", 90),
        (5, "Aintree", "medium", 165),
        (6, "Wellington Straight", "fast", 280),
        (7, "Brooklands", "slow", 95),
        (8, "Luffield", "medium", 130),
        (9, "Woodcote", "fast", 230),
        (10, "Copse", "fast", 245),
        (11, "Maggotts 1", "fast", 220),
        (12, "Maggotts 2", "fast", 210),
        (13, "Becketts 1", "medium", 175),
        (14, "Becketts 2", "medium", 160),
        (15, "Chapel", "fast", 215),
        (16, "Stowe", "medium", 165),
        (17, "Vale", "slow", 90),
        (18, "Club", "medium", 155),
    ])


def _monza_corners() -> list[Corner]:
    """Autodromo Nazionale Monza (5.793 km, 11 弯, high_speed_low_downforce)。

    task-63：按 F1 26 游内弯号重排 —— T1 Variante del Rettifilo、T2 Curva
    Grande、T3 Variante della Roggia、T4/T5 Lesmo、T6 Serraglio、
    T7-T9 Variante Ascari、T10/T11 Parabolica（旧版把 Curva Grande 排在
    T10 且缺 Lesmo，与游内弯号不符）。
    """
    return _build_corners([
        (1, "Variante del Rettifilo", "slow", 80),
        (2, "Curva Grande", "fast", 250),
        (3, "Variante della Roggia", "slow", 85),
        (4, "Lesmo 1", "medium", 145),
        (5, "Lesmo 2", "medium", 150),
        (6, "Serraglio", "fast", 230),
        (7, "Variante Ascari entry", "slow", 90),
        (8, "Variante Ascari", "slow", 85),
        (9, "Variante Ascari exit", "medium", 140),
        (10, "Parabolica entry", "medium", 160),
        (11, "Parabolica", "medium", 200),
    ])


def _spa_corners() -> list[Corner]:
    """Circuit de Spa-Francorchamps (7.004 km, 19 弯, mixed)。"""
    return _build_corners([
        (1, "La Source", "slow", 85),
        (2, "Eau Rouge", "fast", 260),
        (3, "Raidillon", "fast", 270),
        (4, "Kemmel Straight", "fast", 280),
        (5, "Les Combes", "slow", 95),
        (6, "Malmedy", "medium", 145),
        (7, "Rivage", "slow", 90),
        (8, "Pouhon", "fast", 225),
        (9, "Fagnes", "medium", 155),
        (10, "Campus", "slow", 88),
        (11, "Stavelot", "fast", 235),
        (12, "Blanchimont 1", "fast", 275),
        (13, "Blanchimont 2", "fast", 280),
        (14, "Bus Stop entry", "slow", 82),
        (15, "Bus Stop", "slow", 78),
        (16, "Bus Stop exit", "slow", 85),
        (17, "La Source approach", "medium", 160),
        (18, "Eau Rouge approach", "fast", 255),
        (19, "Kemmel approach", "fast", 265),
    ])


# 手工录入赛道弯道构建器映射
_MANUAL_CORNER_BUILDERS: dict[str, Callable[[], list[Corner]]] = {
    "melbourne": _melbourne_corners,
    "suzuka": _suzuka_corners,
    "monaco": _monaco_corners,
    "silverstone": _silverstone_corners,
    "monza": _monza_corners,
    "spa": _spa_corners,
}


def _make_corners(track_id: str, track_type: str, n_corners: int) -> list[Corner]:
    """获取赛道弯道列表：手工录入优先，其次逐弯元数据表；再用真实坐标校准锚点。

    task-63：全部 24 条赛道的逐弯元数据均已真实化（:mod:`_turn_data`），
    ``_synthesize_corners`` 合成路径已删除（"Corner N" 占位数据是
    连续弯标号错误之外的另一处假数据源）。
    """
    builder = _MANUAL_CORNER_BUILDERS.get(track_id)
    if builder is not None:
        corners = builder()
    elif track_id in _AUTHORED_TURN_DATA:
        corners = _build_corners([
            (i + 1, name, ct, spd)
            for i, (name, ct, spd) in enumerate(_AUTHORED_TURN_DATA[track_id])
        ])
    else:  # pragma: no cover - 防御分支（24 赛道已全覆盖）
        raise KeyError(f"赛道 {track_id!r} 缺少弯道元数据（_turn_data.TURNS）")
    return _apply_real_anchors(track_id, corners)


# --------------------------------------------------------------------------- #
# 24 条 F1 2026 赛历赛道（按赛历轮次顺序）
# --------------------------------------------------------------------------- #
# 元数据核对自 legacy/f1opt/data/tracks.py ALL_TRACKS。
# udp_track_id 按赛历轮次顺序分配（round_number - 1），需对照 F1 25
# 官方 UDP 规范 m_trackId 枚举校准。
# svg_path 相对 setup_tuner/ui/ 目录。

ALL_TRACKS: list[Track] = [
    Track(
        track_id="melbourne",
        official_name="Australian Grand Prix",
        circuit_name="Albert Park Grand Prix Circuit",
        city="Melbourne",
        country="Australia",
        track_type="medium",
        length_m=5278.0,
        corners=_make_corners("melbourne", "medium", 14),
        udp_track_id=0,
        svg_path="tracks/melbourne.svg",
    ),
    Track(
        track_id="shanghai",
        official_name="Chinese Grand Prix",
        circuit_name="Shanghai International Circuit",
        city="Shanghai",
        country="China",
        track_type="medium",
        length_m=5451.0,
        corners=_make_corners("shanghai", "medium", 16),
        udp_track_id=2,
        svg_path="tracks/shanghai.svg",
    ),
    Track(
        track_id="suzuka",
        official_name="Japanese Grand Prix",
        circuit_name="Suzuka International Racing Course",
        city="Suzuka",
        country="Japan",
        track_type="mixed",
        length_m=5807.0,
        corners=_make_corners("suzuka", "mixed", 18),
        udp_track_id=13,
        svg_path="tracks/suzuka.svg",
    ),
    Track(
        track_id="sakhir",
        official_name="Bahrain Grand Prix",
        circuit_name="Bahrain International Circuit",
        city="Sakhir",
        country="Bahrain",
        track_type="medium",
        length_m=5412.0,
        corners=_make_corners("sakhir", "medium", 15),
        udp_track_id=3,
        svg_path="tracks/sakhir.svg",
    ),
    Track(
        track_id="jeddah",
        official_name="Saudi Arabian Grand Prix",
        circuit_name="Jeddah Corniche Circuit",
        city="Jeddah",
        country="Saudi Arabia",
        track_type="high_speed_low_downforce",
        length_m=6174.0,
        corners=_make_corners("jeddah", "high_speed_low_downforce", 27),
        udp_track_id=29,
        svg_path="tracks/jeddah.svg",
    ),
    Track(
        track_id="miami",
        official_name="Miami Grand Prix",
        circuit_name="Miami International Autodrome",
        city="Miami",
        country="United States",
        track_type="street",
        length_m=5412.0,
        corners=_make_corners("miami", "street", 19),
        udp_track_id=30,
        svg_path="tracks/miami.svg",
    ),
    Track(
        track_id="montreal",
        official_name="Canadian Grand Prix",
        circuit_name="Circuit Gilles Villeneuve",
        city="Montreal",
        country="Canada",
        track_type="high_speed_low_downforce",
        length_m=4361.0,
        corners=_make_corners("montreal", "high_speed_low_downforce", 14),
        udp_track_id=6,
        svg_path="tracks/montreal.svg",
    ),
    Track(
        track_id="monaco",
        official_name="Monaco Grand Prix",
        circuit_name="Circuit de Monaco",
        city="Monte Carlo",
        country="Monaco",
        track_type="street",
        length_m=3337.0,
        corners=_make_corners("monaco", "street", 19),
        udp_track_id=5,
        svg_path="tracks/monaco.svg",
    ),
    Track(
        track_id="barcelona",
        official_name="Gran Premio de Barcelona-Catalunya",
        circuit_name="Circuit de Barcelona-Catalunya",
        city="Barcelona",
        country="Spain",
        track_type="medium",
        length_m=4657.0,
        corners=_make_corners("barcelona", "medium", 14),
        udp_track_id=4,
        svg_path="tracks/barcelona.svg",
    ),
    Track(
        track_id="spielberg",
        official_name="Austrian Grand Prix",
        circuit_name="Red Bull Ring",
        city="Spielberg",
        country="Austria",
        track_type="medium",
        length_m=4318.0,
        corners=_make_corners("spielberg", "medium", 10),
        udp_track_id=17,
        svg_path="tracks/spielberg.svg",
    ),
    Track(
        track_id="silverstone",
        official_name="British Grand Prix",
        circuit_name="Silverstone Circuit",
        city="Silverstone",
        country="United Kingdom",
        track_type="mixed",
        length_m=5891.0,
        corners=_make_corners("silverstone", "mixed", 18),
        udp_track_id=7,
        svg_path="tracks/silverstone.svg",
    ),
    Track(
        track_id="spa",
        official_name="Belgian Grand Prix",
        circuit_name="Circuit de Spa-Francorchamps",
        city="Spa-Francorchamps",
        country="Belgium",
        track_type="mixed",
        length_m=7004.0,
        corners=_make_corners("spa", "mixed", 19),
        udp_track_id=10,
        svg_path="tracks/spa.svg",
    ),
    Track(
        track_id="hungaroring",
        official_name="Hungarian Grand Prix",
        circuit_name="Hungaroring",
        city="Budapest",
        country="Hungary",
        track_type="high_downforce",
        length_m=4381.0,
        corners=_make_corners("hungaroring", "high_downforce", 14),
        udp_track_id=9,
        svg_path="tracks/hungaroring.svg",
    ),
    Track(
        track_id="zandvoort",
        official_name="Dutch Grand Prix",
        circuit_name="Circuit Zandvoort",
        city="Zandvoort",
        country="Netherlands",
        track_type="high_downforce",
        length_m=4259.0,
        corners=_make_corners("zandvoort", "high_downforce", 14),
        udp_track_id=26,
        svg_path="tracks/zandvoort.svg",
    ),
    Track(
        track_id="monza",
        official_name="Italian Grand Prix",
        circuit_name="Autodromo Nazionale Monza",
        city="Monza",
        country="Italy",
        track_type="high_speed_low_downforce",
        length_m=5793.0,
        corners=_make_corners("monza", "high_speed_low_downforce", 11),
        udp_track_id=11,
        svg_path="tracks/monza.svg",
    ),
    Track(
        track_id="madrid",
        official_name="Spanish Grand Prix",
        circuit_name="Circuito de Madring",
        city="Madrid",
        country="Spain",
        track_type="street",
        length_m=5416.0,
        corners=_make_corners("madrid", "street", 22),
        udp_track_id=33,
        svg_path="tracks/madrid.svg",
    ),
    Track(
        track_id="baku",
        official_name="Azerbaijan Grand Prix",
        circuit_name="Baku City Circuit",
        city="Baku",
        country="Azerbaijan",
        track_type="high_speed_low_downforce",
        length_m=6003.0,
        corners=_make_corners("baku", "high_speed_low_downforce", 20),
        udp_track_id=20,
        svg_path="tracks/baku.svg",
    ),
    Track(
        track_id="singapore",
        official_name="Singapore Grand Prix",
        circuit_name="Marina Bay Street Circuit",
        city="Singapore",
        country="Singapore",
        track_type="street",
        length_m=4940.0,
        corners=_make_corners("singapore", "street", 19),
        udp_track_id=12,
        svg_path="tracks/singapore.svg",
    ),
    Track(
        track_id="austin",
        official_name="United States Grand Prix",
        circuit_name="Circuit of the Americas",
        city="Austin",
        country="United States",
        track_type="mixed",
        length_m=5513.0,
        corners=_make_corners("austin", "mixed", 20),
        udp_track_id=15,
        svg_path="tracks/austin.svg",
    ),
    Track(
        track_id="mexico_city",
        official_name="Mexico City Grand Prix",
        circuit_name="Autodromo Hermanos Rodriguez",
        city="Mexico City",
        country="Mexico",
        track_type="mixed",
        length_m=4304.0,
        corners=_make_corners("mexico_city", "mixed", 17),
        udp_track_id=19,
        svg_path="tracks/mexico_city.svg",
    ),
    Track(
        track_id="sao_paulo",
        official_name="Sao Paulo Grand Prix",
        circuit_name="Autodromo Jose Carlos Pace",
        city="Sao Paulo",
        country="Brazil",
        track_type="mixed",
        length_m=4309.0,
        corners=_make_corners("sao_paulo", "mixed", 15),
        udp_track_id=16,
        svg_path="tracks/sao_paulo.svg",
    ),
    Track(
        track_id="las_vegas",
        official_name="Las Vegas Grand Prix",
        circuit_name="Las Vegas Strip Circuit",
        city="Las Vegas",
        country="United States",
        track_type="high_speed_low_downforce",
        length_m=6201.0,
        corners=_make_corners("las_vegas", "high_speed_low_downforce", 17),
        udp_track_id=31,
        svg_path="tracks/las_vegas.svg",
    ),
    Track(
        track_id="lusail",
        official_name="Qatar Grand Prix",
        circuit_name="Lusail International Circuit",
        city="Lusail",
        country="Qatar",
        track_type="medium",
        length_m=5419.0,
        corners=_make_corners("lusail", "medium", 16),
        udp_track_id=32,
        svg_path="tracks/lusail.svg",
    ),
    Track(
        track_id="yas_marina",
        official_name="Abu Dhabi Grand Prix",
        circuit_name="Yas Marina Circuit",
        city="Abu Dhabi",
        country="United Arab Emirates",
        track_type="medium",
        length_m=5281.0,
        corners=_make_corners("yas_marina", "medium", 16),
        udp_track_id=14,
        svg_path="tracks/yas_marina.svg",
    ),
]


# --------------------------------------------------------------------------- #
# 查询索引与辅助函数
# --------------------------------------------------------------------------- #

_TRACKS_BY_ID: dict[str, Track] = {t.track_id: t for t in ALL_TRACKS}
_TRACKS_BY_UDP: dict[int, Track] = {t.udp_track_id: t for t in ALL_TRACKS}


def get_all_tracks() -> list[Track]:
    """返回全部 24 条赛道（按赛历轮次顺序）。"""
    return list(ALL_TRACKS)


def get_track_by_id(track_id: str) -> Track | None:
    """按 track_id 查询赛道；不存在返回 None。"""
    return _TRACKS_BY_ID.get(track_id)


def get_track_by_udp_id(udp_track_id: int) -> Track | None:
    """按遥测 m_trackId (udp_track_id) 查询赛道；不存在返回 None。"""
    return _TRACKS_BY_UDP.get(udp_track_id)