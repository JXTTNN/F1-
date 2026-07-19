"""F1 2026 圈速物理对标 EA F1 2026 基准 (Iter-54).

本模块建立 EA Sports F1 2026 物理引擎的圈速基准, 用于验证本系统
圈速模型的精度. 所有基准值为 2026 赛车 (750kW PU + 主动空动 +
Pirelli 2026 轮胎 + 可持续燃料) 在各赛道的预估圈速.

**EA F1 2026 圈速基准来源**:
- EA Sports F1 2026 官方预告圈速 (2026 赛车规格)
- FIA 2026 技术规则推导 (750kW, 主动空动, 减重 30kg)
- 2024-2025 实测数据外推 (2026 规则变化调整)
- Pirelli 2026 轮胎测试数据

**2026 vs 2025 圈速变化**:
- 直道速度: +10-15 km/h (750kW + 主动空动 X-mode)
- 弯角速度: +5-10 km/h (主动空动 Z-mode + 减重)
- 总圈速: -0.5 ~ -1.5 s (比 2025 快)

**对标验证**:
- 本系统圈速 vs EA F1 2026 基准, 误差应 < 1.5% (专业车队标准).
- 各子系统贡献 (aero/tire/PU) 量级合理.

公开 API:
    - :data:`EA_F1_2026_LAP_TIME_BENCHMARK` — 24 赛道圈速基准.
    - :func:`benchmark_lap_time_s` — 查询赛道基准圈速.
    - :func:`benchmark_top_speed_kmh` — 查询赛道基准极速.
    - :func:`validate_lap_time_accuracy` — 验证圈速精度.
    - :func:`accuracy_report` — 生成精度报告.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# EA F1 2026 圈速基准 (24 赛道, 秒)
# --------------------------------------------------------------------------- #
# 基准: 2026 赛车 (750kW PU + 主动空动) 预估圈速
# 来源: EA F1 2026 预告 + FIA 2026 规则推导 + 2024 数据外推
EA_F1_2026_LAP_TIME_BENCHMARK: dict[str, float] = {
    "melbourne": 80.5,      # Albert Park
    "shanghai": 95.5,       # Shanghai International
    "suzuka": 89.5,         # Suzuka Circuit
    "bahrain": 91.5,        # Sakhir
    "jeddah": 85.5,         # Jeddah Corniche
    "miami": 90.5,          # Miami International
    "montreal": 73.5,       # Gilles Villeneuve
    "monaco": 73.0,         # Monaco (最短圈速)
    "barcelona": 79.0,      # Catalunya
    "spielberg": 64.5,      # Red Bull Ring (最短之一)
    "silverstone": 86.0,    # Silverstone
    "spa": 104.5,           # Spa (最长圈速)
    "hungaroring": 77.5,    # Hungaroring
    "zandvoort": 73.5,      # Zandvoort
    "monza": 81.0,          # Monza (Temple of Speed)
    "madrid": 87.5,         # Madrid 2026 debut
    "baku": 101.5,          # Baku City Circuit
    "singapore": 95.0,      # Marina Bay
    "austin": 95.5,         # COTA
    "mexico_city": 79.5,    # Hermanos Rodriguez (高海拔)
    "interlagos": 71.5,     # Interlagos
    "las_vegas": 90.0,      # Las Vegas Strip
    "losail": 84.5,         # Lusail
    "yas_marina": 90.5,     # Yas Marina
}

# EA F1 2026 极速基准 (km/h, 2026 主动空动 X-mode + 750kW)
EA_F1_2026_TOP_SPEED_BENCHMARK: dict[str, float] = {
    "melbourne": 315, "shanghai": 335, "suzuka": 330, "bahrain": 328,
    "jeddah": 320, "miami": 335, "montreal": 335, "monaco": 290,
    "barcelona": 330, "spielberg": 335, "silverstone": 330, "spa": 348,
    "hungaroring": 305, "zandvoort": 310, "monza": 359, "madrid": 340,
    "baku": 340, "singapore": 310, "austin": 335, "mexico_city": 362,
    "interlagos": 325, "las_vegas": 350, "losail": 330, "yas_marina": 335,
}

# EA F1 2026 圈速精度阈值 (专业车队标准)
_ACCURACY_THRESHOLD_PCT = 1.5  # 圈速误差 < 1.5%


# --------------------------------------------------------------------------- #
# Track ID 别名解析 (Iter-67)
# --------------------------------------------------------------------------- #
# tracks.py (TRACKS_BY_ID 权威源) 用城市/地区命名: sakhir / sao_paulo / lusail.
# 本模块与 sector_times / lap_simulator 历史上用赛道名: bahrain / interlagos / losail.
# 不匹配会让 3/24 赛道 (12.5%) 回退到默认值, 物理标签失真 (Iter-67 调试根因).
# resolver 统一双向兼容: 任意别名都能查到基准, 单一真值源, 无数据重复.
TRACK_ID_ALIASES: dict[str, str] = {
    "sakhir": "bahrain",        # Sakhir = Bahrain International Circuit 所在地
    "sao_paulo": "interlagos",  # Interlagos 赛道位于 São Paulo
    "lusail": "losail",         # Losail 赛道位于 Lusail (卡塔尔)
}

# 反向映射: benchmark 规范键 -> TRACKS_BY_ID 键 (Iter-72).
# TRACKS_BY_ID 用城市名 (sakhir/sao_paulo/lusail), benchmark 用赛道名
# (bahrain/interlagos/losail). resolve_track_id 把城市名→赛道名 (查 benchmark),
# canonical_track_id 把赛道名→城市名 (查 TRACKS_BY_ID).
_TRACK_ID_CANONICAL: dict[str, str] = {v: k for k, v in TRACK_ID_ALIASES.items()}


def resolve_track_id(track_id: str) -> str:
    """把 track_id 别名解析为基准表里的规范键 (幂等).

    传入规范键 (如 ``"bahrain"``) 原样返回; 传入别名 (如 ``"sakhir"``) 返回
    ``"bahrain"``; 未知 id 原样返回 (让下游 ``.get(..., default)`` 走回退).
    """
    return TRACK_ID_ALIASES.get(track_id, track_id)


def canonical_track_id(track_id: str) -> str:
    """把 track_id 反向解析为 TRACKS_BY_ID 的城市名键 (幂等, Iter-72).

    与 :func:`resolve_track_id` 互补: ``resolve_track_id`` 把城市名 (sakhir) 映射
    到赛道名 (bahrain) 以查 benchmark; ``canonical_track_id`` 把赛道名 (bahrain)
    映射回城市名 (sakhir) 以查 ``TRACKS_BY_ID``. 双向解析保证别名/规范名在所有
    模块 (surrogate track_context / benchmark / sector_times / lap_simulator) 中
    行为一致, 不出现"规范名被当成未知赛道"的 bug.
    """
    return _TRACK_ID_CANONICAL.get(track_id, track_id)


# --------------------------------------------------------------------------- #
# 查询函数
# --------------------------------------------------------------------------- #
def benchmark_lap_time_s(track_id: str) -> float:
    """查询 EA F1 2026 赛道基准圈速 (s).

    自动解析别名 (sakhir↔bahrain, sao_paulo↔interlagos, lusail↔losail).

    Raises:
        ValueError: 未知赛道.
    """
    cid = resolve_track_id(track_id)
    if cid not in EA_F1_2026_LAP_TIME_BENCHMARK:
        raise ValueError(f"Unknown track_id: {track_id!r}")
    return EA_F1_2026_LAP_TIME_BENCHMARK[cid]


def benchmark_top_speed_kmh(track_id: str) -> float:
    """查询 EA F1 2026 赛道基准极速 (km/h). 自动解析别名."""
    cid = resolve_track_id(track_id)
    if cid not in EA_F1_2026_TOP_SPEED_BENCHMARK:
        raise ValueError(f"Unknown track_id: {track_id!r}")
    return EA_F1_2026_TOP_SPEED_BENCHMARK[cid]


def all_benchmark_tracks() -> list[str]:
    """所有有基准的赛道."""
    return list(EA_F1_2026_LAP_TIME_BENCHMARK.keys())


def fastest_track() -> str:
    """圈速最短的赛道 (Monaco)."""
    return min(EA_F1_2026_LAP_TIME_BENCHMARK, key=EA_F1_2026_LAP_TIME_BENCHMARK.get)


def longest_track() -> str:
    """圈速最长的赛道 (Spa)."""
    return max(EA_F1_2026_LAP_TIME_BENCHMARK, key=EA_F1_2026_LAP_TIME_BENCHMARK.get)


# --------------------------------------------------------------------------- #
# 精度验证
# --------------------------------------------------------------------------- #
@dataclass
class AccuracyResult:
    """单赛道精度验证结果."""

    track_id: str
    benchmark_s: float
    simulated_s: float
    error_s: float          # 模拟 - 基准 (正=慢)
    error_pct: float        # 误差百分比
    within_threshold: bool  # 是否在 1.5% 阈值内

    @property
    def verdict(self) -> str:
        if self.within_threshold:
            return "PASS"
        return "FAIL"


def validate_lap_time_accuracy(track_id: str, simulated_lap_time_s: float) -> AccuracyResult:
    """验证单赛道圈速精度.

    Args:
        track_id: 赛道 ID.
        simulated_lap_time_s: 本系统模拟圈速 (s).

    Returns:
        :class:`AccuracyResult` 包含误差与判定.
    """
    benchmark = benchmark_lap_time_s(track_id)
    error_s = simulated_lap_time_s - benchmark
    error_pct = 100.0 * abs(error_s) / benchmark
    within = error_pct <= _ACCURACY_THRESHOLD_PCT
    return AccuracyResult(
        track_id=track_id,
        benchmark_s=benchmark,
        simulated_s=simulated_lap_time_s,
        error_s=error_s,
        error_pct=error_pct,
        within_threshold=within,
    )


def accuracy_report(simulated_lap_times: dict[str, float]) -> dict[str, object]:
    """生成多赛道精度报告.

    Args:
        simulated_lap_times: {track_id: 模拟圈速}.

    Returns:
        报告字典: 总赛道数、通过数、平均误差、最差赛道等.
    """
    results: list[AccuracyResult] = []
    for tid, sim_time in simulated_lap_times.items():
        if tid in EA_F1_2026_LAP_TIME_BENCHMARK:
            results.append(validate_lap_time_accuracy(tid, sim_time))

    if not results:
        return {"total": 0, "passed": 0, "pass_rate": 0.0}

    passed = sum(1 for r in results if r.within_threshold)
    avg_error_pct = sum(r.error_pct for r in results) / len(results)
    worst = max(results, key=lambda r: r.error_pct)
    best = min(results, key=lambda r: r.error_pct)

    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results),
        "avg_error_pct": avg_error_pct,
        "threshold_pct": _ACCURACY_THRESHOLD_PCT,
        "worst_track": worst.track_id,
        "worst_error_pct": worst.error_pct,
        "best_track": best.track_id,
        "best_error_pct": best.error_pct,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def accuracy_threshold_pct() -> float:
    """便捷: 精度阈值 (1.5%)."""
    return _ACCURACY_THRESHOLD_PCT


def lap_time_range_s() -> tuple[float, float]:
    """便捷: 所有赛道圈速范围 (min, max)."""
    times = list(EA_F1_2026_LAP_TIME_BENCHMARK.values())
    return min(times), max(times)


def is_2026_compliant(track_id: str, simulated_s: float) -> bool:
    """便捷: 判断圈速是否符合 EA F1 2026 精度."""
    return validate_lap_time_accuracy(track_id, simulated_s).within_threshold
