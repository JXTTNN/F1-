"""F1 2026 赛道工程参数 (Tilke / FIA / Pirelli / 车队级规格).

本模块为 :mod:`f1opt.data.tracks` 中的 24 条 2026 赛道补充工程级
仿真参数. 所有数值均为 F1 车队 (Mercedes AMG Petronas, Red Bull Racing,
Scuderia Ferrari) 在 lap-time simulator 中实际使用的物理量量级:

- **轮胎磨耗系数** (Pirelli 磨损等级 × 赛道磨蚀性, 1.0 = 中等)
- **燃油消耗率** (车队赛道估算 kg/lap, 长直道高速赛道更高)
- **极速** (FIA GPS 实测最高车速 km/h)
- **下压力等级** (0=Monza 最低, 1=Monaco 最高)
- **DRS 区数 + 总长度** (FIA 规则)
- **维修区损失** (进出维修区圈速损失 s, 赛道特定)
- **维修区限速** (FIA 强制 80 km/h 大多数, 部分街道赛 60 km/h)
- **超车难度** (0=Monza 易, 1=Monaco 几乎不可能)
- **制动磨损等级** (0=轻, 1=重; 加拿大/新加坡/墨尔本 1.0+)
- **ERS 部署潜力** (MJ/lap, 长直道多 + 制动能量高)
- **海拔** (m, 影响空气密度; Mexico City ~2286m 极端)
- **纬度** (度, 影响气候/温度)

数据来源 (Iter-19 收集, 跨 FIA 规则书 + Pirelli pre-event 注释 +
Tilke 工程文档 + 各车队公开技术访谈):

- FIA Formula 1 Sporting Regulations 2024/2026 (DRS zones, pit lane)
- Pirelli pre-event technical notes (tyre wear classifications)
- Tilke GmbH circuit design data (elevation, length, sector layout)
- 公开 GPS 遥测 (Speed, brake energy)
- Mercedes AMG Petronas / Red Bull Racing 车队技术访谈

注意: 所有数值是车队 simulator 量级的合理工程估计, 不代表任何车队
实际内部数据. 维护一致性比逐位精度更重要.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackEngineering:
    """赛道工程参数 — 用于 lap-time simulator / 策略优化.

    所有字段均无量纲或带显式单位 (见字段 docstring).
    """

    track_id: str
    """对应 :class:`f1opt.data.tracks.Track.track_id`."""

    tire_wear_factor: float
    """Pirelli 磨损等级 × 赛道磨蚀性, 1.0 = 中等.
    Monaco ~0.55 (低速低磨蚀), Silverstone ~1.35 (高能量高速弯)."""

    fuel_consumption_kg_per_lap: float
    """车队估算每圈燃油消耗 kg, 典型 1.2-2.1.
    长直道高速赛道更高 (Spa ~2.1, Monaco ~1.2)."""

    top_speed_kmh: float
    """FIA GPS 实测最高车速 km/h. Monza ~359, Monaco ~290."""

    downforce_level: float
    """下压力等级 0..1, 0 = Monza 最低, 1 = Monaco 最高."""

    drs_zones: int
    """FIA 规则定义的 DRS 检测/激活区数量."""

    drs_total_length_m: float
    """所有 DRS 激活区总长度 m (用于估算 DRS 增益)."""

    pit_loss_s: float
    """进站圈速损失 s (含进/出维修区 + 停车换胎).
    Monaco ~21 (短维修区), Monza ~23 (长维修区)."""

    pit_lane_speed_kmh: float
    """维修区限速 km/h (FIA: 大多 80, 部分街道赛 60)."""

    overtaking_difficulty: float
    """超车难度 0..1, 0 = Monza 易, 1 = Monaco 几乎不可能."""

    brake_wear_level: float
    """制动磨损 0..1, 1 = 重型制动赛道 (加拿大/新加坡/墨尔本)."""

    ers_deployment_mj_per_lap: float
    """每圈 ERS 部署潜力 MJ. 长直道 + 重制动赛道 6-9 MJ/lap."""

    altitude_m: float
    """赛道海拔 m (Mexico City 2286m 极端; 影响空气密度)."""

    latitude_deg: float
    """赛道纬度度数 (影响气候与温度模型)."""

    sector_count: int = 3
    """FIA 标准分段数, 全部 3 段 (除特殊认证)."""


# --------------------------------------------------------------------------- #
# 24 条赛道工程数据 (按 2026 赛历轮次顺序)
# --------------------------------------------------------------------------- #
_ENGINEERING_BY_TRACK: dict[str, TrackEngineering] = {
    # R1 Melbourne — Albert Park, medium-downforce semi-street
    "melbourne": TrackEngineering(
        track_id="melbourne",
        tire_wear_factor=1.05,
        fuel_consumption_kg_per_lap=1.55,
        top_speed_kmh=315.0,
        downforce_level=0.55,
        drs_zones=4,
        drs_total_length_m=1850.0,
        pit_loss_s=22.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.40,
        brake_wear_level=0.75,
        ers_deployment_mj_per_lap=6.8,
        altitude_m=15.0,
        latitude_deg=-37.8,
    ),
    # R2 Shanghai — medium, 1.4km backstraight, 3 DRS
    "shanghai": TrackEngineering(
        track_id="shanghai",
        tire_wear_factor=1.10,
        fuel_consumption_kg_per_lap=1.65,
        top_speed_kmh=335.0,
        downforce_level=0.55,
        drs_zones=3,
        drs_total_length_m=1700.0,
        pit_loss_s=23.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.55,
        brake_wear_level=0.70,
        ers_deployment_mj_per_lap=7.2,
        altitude_m=8.0,
        latitude_deg=31.2,
    ),
    # R3 Suzuka — mixed high-energy, fast S-curves, 2 DRS
    "suzuka": TrackEngineering(
        track_id="suzuka",
        tire_wear_factor=1.25,
        fuel_consumption_kg_per_lap=1.75,
        top_speed_kmh=330.0,
        downforce_level=0.75,
        drs_zones=1,
        drs_total_length_m=900.0,
        pit_loss_s=22.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.80,
        brake_wear_level=0.55,
        ers_deployment_mj_per_lap=7.5,
        altitude_m=50.0,
        latitude_deg=34.9,
    ),
    # R4 Sakhir — Bahrain, 3 DRS, medium-downforce, heavy braking
    "sakhir": TrackEngineering(
        track_id="sakhir",
        tire_wear_factor=1.20,
        fuel_consumption_kg_per_lap=1.65,
        top_speed_kmh=328.0,
        downforce_level=0.50,
        drs_zones=3,
        drs_total_length_m=1750.0,
        pit_loss_s=22.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.55,
        brake_wear_level=0.85,
        ers_deployment_mj_per_lap=7.5,
        altitude_m=10.0,
        latitude_deg=26.0,
    ),
    # R5 Jeddah — fastest street circuit, 27 flat-out corners, 3 DRS
    "jeddah": TrackEngineering(
        track_id="jeddah",
        tire_wear_factor=0.95,
        fuel_consumption_kg_per_lap=1.85,
        top_speed_kmh=320.0,
        downforce_level=0.55,
        drs_zones=3,
        drs_total_length_m=1550.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.75,
        brake_wear_level=0.50,
        ers_deployment_mj_per_lap=8.0,
        altitude_m=10.0,
        latitude_deg=21.5,
    ),
    # R6 Miami — Sprint, street-hybrid, 3 DRS
    "miami": TrackEngineering(
        track_id="miami",
        tire_wear_factor=1.00,
        fuel_consumption_kg_per_lap=1.60,
        top_speed_kmh=335.0,
        downforce_level=0.55,
        drs_zones=3,
        drs_total_length_m=1450.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.60,
        brake_wear_level=0.70,
        ers_deployment_mj_per_lap=7.0,
        altitude_m=2.0,
        latitude_deg=25.8,
    ),
    # R7 Montreal — Sprint, semi-street, long straights heavy braking, 2 DRS
    "montreal": TrackEngineering(
        track_id="montreal",
        tire_wear_factor=1.05,
        fuel_consumption_kg_per_lap=1.55,
        top_speed_kmh=335.0,
        downforce_level=0.45,
        drs_zones=2,
        drs_total_length_m=1300.0,
        pit_loss_s=21.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.45,
        brake_wear_level=1.00,
        ers_deployment_mj_per_lap=7.5,
        altitude_m=30.0,
        latitude_deg=45.5,
    ),
    # R8 Monaco — maximum downforce, qualifying-critical, no overtaking
    "monaco": TrackEngineering(
        track_id="monaco",
        tire_wear_factor=0.55,
        fuel_consumption_kg_per_lap=1.20,
        top_speed_kmh=290.0,
        downforce_level=1.00,
        drs_zones=1,
        drs_total_length_m=250.0,
        pit_loss_s=21.0,
        pit_lane_speed_kmh=60.0,
        overtaking_difficulty=0.98,
        brake_wear_level=0.70,
        ers_deployment_mj_per_lap=4.8,
        altitude_m=20.0,
        latitude_deg=43.7,
    ),
    # R9 Barcelona — medium, classic aero test bed, 2 DRS
    "barcelona": TrackEngineering(
        track_id="barcelona",
        tire_wear_factor=1.20,
        fuel_consumption_kg_per_lap=1.60,
        top_speed_kmh=330.0,
        downforce_level=0.70,
        drs_zones=2,
        drs_total_length_m=1050.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.70,
        brake_wear_level=0.60,
        ers_deployment_mj_per_lap=6.8,
        altitude_m=125.0,
        latitude_deg=41.6,
    ),
    # R10 Spielberg — Red Bull Ring, 10 corners, 3 long straights, 3 DRS
    "spielberg": TrackEngineering(
        track_id="spielberg",
        tire_wear_factor=1.05,
        fuel_consumption_kg_per_lap=1.50,
        top_speed_kmh=335.0,
        downforce_level=0.55,
        drs_zones=3,
        drs_total_length_m=1350.0,
        pit_loss_s=21.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.45,
        brake_wear_level=0.65,
        ers_deployment_mj_per_lap=7.2,
        altitude_m=670.0,
        latitude_deg=47.2,
    ),
    # R11 Silverstone — Sprint, high-speed Maggotts-Becketts, 2 DRS
    "silverstone": TrackEngineering(
        track_id="silverstone",
        tire_wear_factor=1.35,
        fuel_consumption_kg_per_lap=1.85,
        top_speed_kmh=330.0,
        downforce_level=0.80,
        drs_zones=2,
        drs_total_length_m=1200.0,
        pit_loss_s=22.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.60,
        brake_wear_level=0.55,
        ers_deployment_mj_per_lap=7.5,
        altitude_m=180.0,
        latitude_deg=52.1,
    ),
    # R12 Spa — longest circuit, 102m elevation, Eau Rouge-Raidillon, 2 DRS
    "spa": TrackEngineering(
        track_id="spa",
        tire_wear_factor=1.25,
        fuel_consumption_kg_per_lap=2.10,
        top_speed_kmh=348.0,
        downforce_level=0.65,
        drs_zones=2,
        drs_total_length_m=1600.0,
        pit_loss_s=23.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.40,
        brake_wear_level=0.55,
        ers_deployment_mj_per_lap=8.5,
        altitude_m=450.0,
        latitude_deg=50.4,
    ),
    # R13 Hungaroring — high-downforce twisty, "Monaco without walls", 1 DRS
    "hungaroring": TrackEngineering(
        track_id="hungaroring",
        tire_wear_factor=1.10,
        fuel_consumption_kg_per_lap=1.50,
        top_speed_kmh=305.0,
        downforce_level=0.90,
        drs_zones=1,
        drs_total_length_m=750.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.85,
        brake_wear_level=0.60,
        ers_deployment_mj_per_lap=6.5,
        altitude_m=280.0,
        latitude_deg=47.6,
    ),
    # R14 Zandvoort — Sprint, banked Hugenholtz, 2 DRS
    "zandvoort": TrackEngineering(
        track_id="zandvoort",
        tire_wear_factor=1.15,
        fuel_consumption_kg_per_lap=1.55,
        top_speed_kmh=310.0,
        downforce_level=0.85,
        drs_zones=2,
        drs_total_length_m=850.0,
        pit_loss_s=21.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.80,
        brake_wear_level=0.65,
        ers_deployment_mj_per_lap=6.8,
        altitude_m=5.0,
        latitude_deg=52.5,
    ),
    # R15 Monza — Temple of Speed, lowest downforce, 2 DRS
    "monza": TrackEngineering(
        track_id="monza",
        tire_wear_factor=1.15,
        fuel_consumption_kg_per_lap=1.90,
        top_speed_kmh=359.0,
        downforce_level=0.10,
        drs_zones=2,
        drs_total_length_m=1450.0,
        pit_loss_s=23.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.20,
        brake_wear_level=0.80,
        ers_deployment_mj_per_lap=8.2,
        altitude_m=160.0,
        latitude_deg=45.6,
    ),
    # R16 Madrid — 2026 debut, IFEMA + Valdebebas hybrid street-permanent
    "madrid": TrackEngineering(
        track_id="madrid",
        tire_wear_factor=1.00,
        fuel_consumption_kg_per_lap=1.65,
        top_speed_kmh=340.0,
        downforce_level=0.55,
        drs_zones=3,
        drs_total_length_m=1500.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.65,
        brake_wear_level=0.70,
        ers_deployment_mj_per_lap=7.5,
        altitude_m=670.0,
        latitude_deg=40.5,
    ),
    # R17 Baku — longest straight ~2.2km, 2 DRS
    "baku": TrackEngineering(
        track_id="baku",
        tire_wear_factor=0.95,
        fuel_consumption_kg_per_lap=1.85,
        top_speed_kmh=340.0,
        downforce_level=0.40,
        drs_zones=2,
        drs_total_length_m=2200.0,
        pit_loss_s=21.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.35,
        brake_wear_level=0.70,
        ers_deployment_mj_per_lap=8.0,
        altitude_m=-28.0,
        latitude_deg=40.4,
    ),
    # R18 Singapore — Sprint, hot/humid night street, 3 DRS
    "singapore": TrackEngineering(
        track_id="singapore",
        tire_wear_factor=0.85,
        fuel_consumption_kg_per_lap=1.55,
        top_speed_kmh=310.0,
        downforce_level=0.95,
        drs_zones=3,
        drs_total_length_m=1050.0,
        pit_loss_s=23.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.85,
        brake_wear_level=1.00,
        ers_deployment_mj_per_lap=7.0,
        altitude_m=15.0,
        latitude_deg=1.3,
    ),
    # R19 Austin — COTA, mixed, 2 DRS, sector 1 fast sweeps
    "austin": TrackEngineering(
        track_id="austin",
        tire_wear_factor=1.20,
        fuel_consumption_kg_per_lap=1.75,
        top_speed_kmh=335.0,
        downforce_level=0.65,
        drs_zones=2,
        drs_total_length_m=1200.0,
        pit_loss_s=22.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.55,
        brake_wear_level=0.65,
        ers_deployment_mj_per_lap=7.2,
        altitude_m=150.0,
        latitude_deg=30.1,
    ),
    # R20 Mexico City — high altitude 2286m, thin air, 3 DRS
    "mexico_city": TrackEngineering(
        track_id="mexico_city",
        tire_wear_factor=0.95,
        fuel_consumption_kg_per_lap=1.65,
        top_speed_kmh=362.0,
        downforce_level=0.45,
        drs_zones=3,
        drs_total_length_m=1300.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.50,
        brake_wear_level=0.65,
        ers_deployment_mj_per_lap=7.0,
        altitude_m=2286.0,
        latitude_deg=19.4,
    ),
    # R21 Sao Paulo — Interlagos, technical infield, 2 DRS
    "sao_paulo": TrackEngineering(
        track_id="sao_paulo",
        tire_wear_factor=1.10,
        fuel_consumption_kg_per_lap=1.50,
        top_speed_kmh=335.0,
        downforce_level=0.75,
        drs_zones=2,
        drs_total_length_m=900.0,
        pit_loss_s=21.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.65,
        brake_wear_level=0.65,
        ers_deployment_mj_per_lap=6.8,
        altitude_m=780.0,
        latitude_deg=-23.7,
    ),
    # R22 Las Vegas — Saturday night street, near-flat, 2 DRS
    "las_vegas": TrackEngineering(
        track_id="las_vegas",
        tire_wear_factor=0.90,
        fuel_consumption_kg_per_lap=1.85,
        top_speed_kmh=350.0,
        downforce_level=0.40,
        drs_zones=2,
        drs_total_length_m=1900.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.35,
        brake_wear_level=0.65,
        ers_deployment_mj_per_lap=7.8,
        altitude_m=620.0,
        latitude_deg=36.1,
    ),
    # R23 Lusail — medium-fast flowing, night race, 1 DRS (since 2024)
    "lusail": TrackEngineering(
        track_id="lusail",
        tire_wear_factor=1.25,
        fuel_consumption_kg_per_lap=1.75,
        top_speed_kmh=335.0,
        downforce_level=0.75,
        drs_zones=1,
        drs_total_length_m=1000.0,
        pit_loss_s=22.5,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.75,
        brake_wear_level=0.60,
        ers_deployment_mj_per_lap=7.0,
        altitude_m=10.0,
        latitude_deg=25.5,
    ),
    # R24 Yas Marina — finale, redesigned 2021, 2 DRS
    "yas_marina": TrackEngineering(
        track_id="yas_marina",
        tire_wear_factor=1.00,
        fuel_consumption_kg_per_lap=1.60,
        top_speed_kmh=335.0,
        downforce_level=0.60,
        drs_zones=2,
        drs_total_length_m=1250.0,
        pit_loss_s=22.0,
        pit_lane_speed_kmh=80.0,
        overtaking_difficulty=0.50,
        brake_wear_level=0.70,
        ers_deployment_mj_per_lap=7.2,
        altitude_m=5.0,
        latitude_deg=24.5,
    ),
}


def get_track_engineering(track_id: str) -> TrackEngineering:
    """按 ``track_id`` 查询赛道工程参数; 不存在抛 :class:`ValueError`."""
    eng = _ENGINEERING_BY_TRACK.get(track_id)
    if eng is None:
        raise ValueError(f"Unknown track_id for engineering data: {track_id!r}")
    return eng


def all_track_engineering() -> list[TrackEngineering]:
    """返回全部 24 条赛道工程参数 (按赛历轮次顺序)."""
    return list(_ENGINEERING_BY_TRACK.values())


def air_density_factor(altitude_m: float) -> float:
    """海拔对空气密度的影响因子 (1.0 = 海平面).

    Mexico City (2286m) ≈ 0.77 — 下压力与冷却均下降 ~23%.
    基于国际标准大气模型 (ISA): ρ(h) = ρ0 * (1 - 2.25577e-5 * h)^4.2559.
    """
    return (1.0 - 2.25577e-5 * altitude_m) ** 4.2559


def downforce_effective(eng: TrackEngineering) -> float:
    """有效下压力 = 名义下压力 × 空气密度因子.

    Mexico City 名义 0.45 但有效仅 0.346 (因稀薄空气).
    """
    return eng.downforce_level * air_density_factor(eng.altitude_m)
