"""SetupDelta 计算引擎（核心，6 步确定性流水线 + 混合模型）。

实现 design 2.7.4 的 ``SetupDelta = clamp(Dx × C)`` 计算流水线：

    1. 矩阵乘法：raw[p] = Σ_d Dx[d] × C[d][p]
    2. 遥测校准：raw[p] *= telemetry_gain[p]（默认 1.0）
    3. 单次上限约束：raw[p] = clip(raw[p], -max_delta[p], +max_delta[p])
    4. 合法区间约束：next[p] = clip(current[p] + raw[p], min[p], max[p])，
       Δ[p] = next[p] - current[p]
    5. 档位对齐：整数参数 round；浮点参数 round 到 step
    6. 输出 SetupDelta = {param: Δ_value}

纯函数、零 IO、零随机、零时间依赖，满足 FR-ENG-05 / FR-NFR-R1（可复现）。
任意单症状产出 SetupDelta 覆盖全部 20 参数；相同输入输出完全一致。

混合模型扩展（task-43）：
    ``generate_suggestion`` 支持 ``model_type`` 参数：
        - ``"rule"``：纯规则引擎（默认确定性流水线）
        - ``"nn"``：纯神经网络（PyTorch 不可用时自动降级为规则引擎）
        - ``"hybrid"``：混合模型（规则 60% + 神经网络 40%，神经网络不可用时降级）

    神经网络分支由 :mod:`setup_tuner.engine.nn_model` 提供，
    PyTorch 为可选依赖，不可用时自动降级为纯规则引擎。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

logger = logging.getLogger(__name__)

# 浮点比较 epsilon（用于 delta 零值判定与边界容差）
DELTA_ZERO_EPSILON = 1e-12
BOUND_TOLERANCE_EPSILON = 1e-6

# 底板（plank）触地上限，单位米。与
# ``setup_tuner.telemetry.lap_aggregator._PLANK_BOTTOMING_MAX_M`` 保持一致：
# MotionEx 未提供布尔标志时，用最小离地高度兜底判断。
_PLANK_BOTTOMING_MAX_M = 0.012

from .confidence import assess_confidence
from .coupling import nonzero_cells_for_param_cached
from .diagnostic import (
    DIAG_DIMS,
    DIAG_DIMS_POSITIVE_SEMANTICS,
    DIAG_DIMS_ZH,
    compute_dx,
    is_zero_dx,
)
from .holistic import class_weighted_dx, holistic_coherence, track_demand
from .optimizer import describe_tradeoff, optimize_setup


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------
def _clip(value: float, lo: float, hi: float) -> float:
    """将 value 裁剪到 [lo, hi] 区间。"""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _align_to_step(value: float, step: float, lo: float) -> float:
    """将 value 对齐到 step 档位（相对 lo 的整数倍）。

    Args:
        value: 待对齐值。
        step: 档位步长。
        lo: 区间下界（档位基准）。

    Returns:
        对齐到 step 档位的值。
    """
    if step <= 0:
        return value
    steps = round((value - lo) / step)
    return lo + steps * step


# ---------------------------------------------------------------------------
# 遥测校准增益提取（design 2.7.5）
# ---------------------------------------------------------------------------
def _is_wet_weather(telemetry: dict[str, Any] | None) -> bool:
    """判断遥测指示湿滑天气。

    接受 ``None``（"没有遥测"是合法输入）：早期直接 ``telemetry.get(...)``，
    调用方一旦传 None 就 ``AttributeError``；引擎整体对缺失遥测是宽容的，
    这里不得例外。
    """
    if not telemetry:
        return False
    weather = telemetry.get("weather") or telemetry.get("m_weather")
    if isinstance(weather, str):
        return weather.lower() in {"wet", "rainy", "rain", "drizzle"}
    if isinstance(weather, (int, float)):
        # F1 UDP m_weather: 0=clear, 1=4light rain, 2=heavy rain, 3=storm
        return weather >= 1
    return False


def _apply_tyre_temp_gain(telemetry: dict[str, Any], gain: dict[str, float]) -> None:
    """根据胎温校准胎压参数增益（原地修改 gain）。"""
    tyre_temps = telemetry.get("m_tyresSurfaceTemperature")
    if not (isinstance(tyre_temps, list) and len(tyre_temps) >= 4):
        return
    avg_tyre_temp = sum(tyre_temps[:4]) / 4.0
    tyre_pressure_params = (
        "front_left_tyre_pressure", "front_right_tyre_pressure",
        "rear_left_tyre_pressure", "rear_right_tyre_pressure",
    )
    if avg_tyre_temp > 100.0:
        # 胎温过高 → 胎压调整更敏感
        for p in tyre_pressure_params:
            gain[p] *= 1.3
    elif avg_tyre_temp < 80.0:
        # 胎温过低 → 胎压调整保守
        for p in tyre_pressure_params:
            gain[p] *= 0.8


def _apply_throttle_brake_gain(telemetry: dict[str, Any], gain: dict[str, float]) -> None:
    """根据油门/刹车均值校准差速器/刹车参数增益（原地修改 gain）。"""
    throttle = telemetry.get("m_throttle")
    if isinstance(throttle, (int, float)) and throttle > 0.7:
        for p in ("on_throttle_diff", "off_throttle_diff"):
            gain[p] *= 1.2
    brake = telemetry.get("m_brake")
    if isinstance(brake, (int, float)) and brake > 0.5:
        for p in ("brake_pressure", "brake_bias"):
            gain[p] *= 1.2


def _derive_telemetry_gain(telemetry: dict[str, Any] | None) -> dict[str, float]:
    """从遥测字典提取每参数的幅度增益（参数级增益）。

    规则（扩展版，利用更多遥测数据校准）：
        - 天气湿（``telemetry["weather"]`` 为 ``"wet"``/``"rainy"`` 或
          ``m_weather`` >= 1）→ 全参数 ×0.7（湿地下保守调整）；
        - 胎温过高（``m_tyresSurfaceTemperature`` 均值 > 100°C）→ 胎压参数 ×1.3
          （胎温过高时胎压调整更敏感）；
        - 胎温过低（``m_tyresSurfaceTemperature`` 均值 < 80°C）→ 胎压参数 ×0.8
          （胎温过低时胎压调整保守）；
        - 油门均值高（``m_throttle`` 均值 > 0.7）→ 差速器参数 ×1.2
          （高油门占比时差速器调整更关键）；
        - 刹车均值高（``m_brake`` 均值 > 0.5）→ 刹车参数 ×1.2
          （高刹车占比时刹车调整更关键）；
        - 其余默认 1.0（不臆测）。

    Args:
        telemetry: 遥测字典；None 时返回全 1.0 增益。

    Returns:
        {param: gain} 字典，覆盖全部 20 参数。
    """
    gain = {f.name: 1.0 for f in ALL_SETUP_FIELDS}
    if not telemetry:
        return gain

    if _is_wet_weather(telemetry):
        for name in gain:
            gain[name] = 0.7
    _apply_tyre_temp_gain(telemetry, gain)
    _apply_throttle_brake_gain(telemetry, gain)
    return gain


# ---------------------------------------------------------------------------
# 遥测诊断向量提取（遥测数据作为诊断输入）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 遥测诊断向量提取辅助函数（task-104：15条规则覆盖全9维Dx + 方向修正因子）
# ---------------------------------------------------------------------------
def _apply_tyre_temperature_rules(
    telemetry: dict[str, Any], dx: dict[str, float],
) -> None:
    """规则1/2/12/13：胎温相关规则（原地修改 dx）。

    规则1 - 胎温过高（均值 > 100°C）：
        tyre_life_req += 0.3，前轮更高时 front_grip_req += 0.2，反之 rear_grip_req += 0.2。
    规则2 - 胎温过低（均值 < 80°C）：
        front_grip_req += 0.3, rear_grip_req += 0.3。
    规则12 - 胎温不均（四轮偏差 > 15°C）：
        tyre_life_req += 0.2。出处：Pirelli官方——胎温不均加速轮胎磨损。
    规则13 - 胎温严重不足（湿地，均值 < 60°C 且 weather_code >= 1）：
        front_grip_req += 0.4, rear_grip_req += 0.4。
        出处：Pirelli官方——湿地条件下胎温严重不足需大幅增加抓地力。
    """
    tyre_temps = telemetry.get("m_tyresSurfaceTemperature")
    if not (isinstance(tyre_temps, list) and len(tyre_temps) >= 4):
        return
    temps = tyre_temps[:4]
    avg_tyre_temp = sum(temps) / 4.0
    # 车轮顺序由官方规范固定为 RL, RR, FL, FR（**不是** FL, FR, RL, RR）：
    # "All wheel arrays have the following order: RL, RR, FL, FR"
    #   [0]=RL [1]=RR [2]=FL [3]=FR
    # 早期实现把 [:2] 当前轮、[2:4] 当后轮，导致前后轮判断完全颠倒。
    rear_avg = (temps[0] + temps[1]) / 2.0   # RL, RR
    front_avg = (temps[2] + temps[3]) / 2.0  # FL, FR

    # 规则1：胎温过高
    if avg_tyre_temp > 100.0:
        dx["tyre_life_req"] += 0.3
        if front_avg > rear_avg:
            dx["front_grip_req"] += 0.2
        else:
            dx["rear_grip_req"] += 0.2
    # 规则2：胎温过低
    elif avg_tyre_temp < 80.0:
        dx["front_grip_req"] += 0.3
        dx["rear_grip_req"] += 0.3

    # 规则12：胎温不均（四轮最大偏差 > 15°C）
    temp_spread = max(temps) - min(temps)
    if temp_spread > 15.0:
        dx["tyre_life_req"] += 0.2

    # 规则13：胎温严重不足（湿地条件）
    if avg_tyre_temp < 60.0 and _is_wet_weather(telemetry):
        dx["front_grip_req"] += 0.4
        dx["rear_grip_req"] += 0.4


def _apply_tyre_pressure_rules(
    telemetry: dict[str, Any], dx: dict[str, float],
) -> None:
    """规则3：胎压异常（原地修改 dx）。

    m_tyresPressure 任一轮胎 > 26.0 或 < 22.0 psi：
        前轮异常 front_grip_req += 0.15，后轮异常 rear_grip_req += 0.15。
    出处：Pirelli官方胎压工作窗口 22-26 psi。
    """
    tyre_pressures = telemetry.get("m_tyresPressure")
    if not (isinstance(tyre_pressures, list) and len(tyre_pressures) >= 4):
        return
    # 官方车轮顺序 RL, RR, FL, FR：前轮 = 索引 2,3；后轮 = 索引 0,1
    front_abnormal = any(p > 26.0 or p < 22.0 for p in tyre_pressures[2:4])
    rear_abnormal = any(p > 26.0 or p < 22.0 for p in tyre_pressures[0:2])
    if front_abnormal:
        dx["front_grip_req"] += 0.15
    if rear_abnormal:
        dx["rear_grip_req"] += 0.15


def _apply_brake_rules(
    telemetry: dict[str, Any], dx: dict[str, float],
) -> None:
    """规则4/8/14：刹车相关规则（原地修改 dx）。

    规则4 - 刹车温度过高（``m_brakesTemperature`` 均值超阈值）：
        brake_stab_req += 0.3。
        阈值随轮胎配方浮动：软胎 450°C / 中性 500°C / 硬胎 550°C
        （配方来自 Packet 7 ``m_actualTyreCompound``）。出处：F1 官方刹车
        工作窗口 200-500°C，软胎工作窗口整体偏低。
    规则8 - 制动力不足（m_brake > 0.8，高刹车输入但减速不明显）：
        brake_power_req += 0.3。出处：F1官方调教指南。
        注意：整圈统计摘要无逐帧 speed 变化率，仅用高刹车输入作为简化条件。
    规则14 - 刹车持续过热（均值超严格阈值）：
        brake_stab_req += 0.4, brake_power_req -= 0.2（方向修正因子——过热需减弱刹车压力）。
        阈值同样随配方浮动：软胎 550°C / 中性 600°C / 硬胎 660°C。
    规则4b - 前后轴刹车温度失衡（单轴均值差 > 120°C）：
        brake_stab_req += 0.2 —— 提示刹车平衡或前后制动分配需要复核。
    """
    brake_temps = telemetry.get("m_brakesTemperature")
    if isinstance(brake_temps, list) and len(brake_temps) >= 4:
        avg_brake_temp = sum(brake_temps[:4]) / 4.0
        # 阈值按轮胎配方调整：软胎升温快、工作窗口低 → 更早告警；
        # 硬胎工作窗口更高 → 阈值上调，避免把正常高温误判为过热。
        # 配方来自 Packet 7 m_actualTyreCompound（聚合器换算为布尔标记）。
        warn_temp = 500.0
        severe_temp = 600.0
        if telemetry.get("is_soft_compound"):
            warn_temp, severe_temp = 450.0, 550.0
        elif telemetry.get("is_hard_compound"):
            warn_temp, severe_temp = 550.0, 660.0
        if avg_brake_temp > warn_temp:
            dx["brake_stab_req"] += 0.3
        if avg_brake_temp > severe_temp:
            dx["brake_stab_req"] += 0.4
            dx["brake_power_req"] -= 0.2
        # 刹车温度前后轴严重失衡（单轴 > 另一轴 120°C）→ 刹车平衡偏置
        front_bt = (brake_temps[2] + brake_temps[3]) / 2.0  # 官方顺序 RL,RR,FL,FR
        rear_bt = (brake_temps[0] + brake_temps[1]) / 2.0
        if abs(front_bt - rear_bt) > 120.0:
            dx["brake_stab_req"] += 0.2

    # 规则8：制动力不足（高刹车输入）
    brake = telemetry.get("m_brake")
    if isinstance(brake, (int, float)) and brake > 0.8:
        dx["brake_power_req"] += 0.3


def _apply_corner_rules(
    telemetry: dict[str, Any], dx: dict[str, float],
) -> None:
    """规则5/7/10/11：弯道相关规则（原地修改 dx）。

    规则5 - 出弯段油门低（sector == 3 且 m_throttle < 0.3）：
        exit_traction_req += 0.3。出处：F1官方调教指南。
        sector 约定为 **1 基**（1/2/3）。UDP 原始 ``m_sector`` 是 0 基，
        由 ``telemetry.packets.to_sector_1based`` 在生成遥测摘要时统一转换；
        早期直传 0 基值（上限 2）导致本规则永不触发。
    规则7 - 入弯响应差（max_steer > 0.3，转向幅度大）：
        turnin_req += 0.3。出处：F1官方调教指南。
        需要整圈统计的 ``max_steer``（由 LapAggregator 提供）。
    规则10 - 弯中不稳定（avg_steer > 0.2，转向反复修正的简化条件）：
        hi_speed_stab_req += 0.3。
        需要整圈统计的 ``avg_steer``（由 LapAggregator 提供）。
    规则11 - 出弯打滑（m_throttle > 0.7 且 m_speed < 150，高油门但速度低）：
        exit_traction_req += 0.3。出处：F1官方调教指南。
    """
    # 规则5：出弯段油门低
    sector = telemetry.get("sector") or telemetry.get("m_sector")
    throttle = telemetry.get("m_throttle")
    if (isinstance(sector, (int, float)) and sector == 3
            and isinstance(throttle, (int, float)) and throttle < 0.3):
        dx["exit_traction_req"] += 0.3

    # 规则7：入弯响应差（转向幅度大）
    max_steer = telemetry.get("max_steer") or telemetry.get("m_max_steer")
    if isinstance(max_steer, (int, float)) and max_steer > 0.3:
        dx["turnin_req"] += 0.3

    # 规则10：弯中不稳定（平均转向绝对值大）
    avg_steer = telemetry.get("avg_steer") or telemetry.get("m_avg_steer")
    if isinstance(avg_steer, (int, float)) and avg_steer > 0.2:
        dx["hi_speed_stab_req"] += 0.3

    # 规则11：出弯打滑（高油门但速度低）
    if isinstance(throttle, (int, float)) and throttle > 0.7:
        speed = telemetry.get("speed") or telemetry.get("m_speed")
        if isinstance(speed, (int, float)) and speed < 150:
            dx["exit_traction_req"] += 0.3


def _apply_speed_rules(
    telemetry: dict[str, Any], dx: dict[str, float],
) -> None:
    """规则6/15：速度相关规则（原地修改 dx）。

    规则6 - 直道速度低（speed < 200 且在直道）：
        hi_speed_stab_req -= 0.3（方向修正因子——需减阻/减翼）。
        若无法判断 sector 类型，保守地不触发。出处：F1官方调教指南。
    规则15 - 直道极速低（max_speed < 280 干地 / < 200 湿地）：
        hi_speed_stab_req -= 0.3（方向修正因子——下压力过大或齿比不当）。
        出处：F1官方调教指南。
    """
    # 规则6：直道速度低
    speed = telemetry.get("speed") or telemetry.get("m_speed")
    if isinstance(speed, (int, float)) and speed < 200:
        on_straight = (
            telemetry.get("m_on_straight")
            or telemetry.get("on_straight")
            or telemetry.get("is_straight")
        )
        sector_type = (
            telemetry.get("sector_type")
            or telemetry.get("m_sector_type")
        )
        is_on_straight = False
        if isinstance(on_straight, bool):
            is_on_straight = on_straight
        elif isinstance(on_straight, (int, float)) and on_straight == 1:
            is_on_straight = True
        elif isinstance(sector_type, str) and sector_type.lower() in ("straight", "straightaway"):
            is_on_straight = True
        if is_on_straight:
            dx["hi_speed_stab_req"] -= 0.3

    # 规则15：直道极速低
    max_speed = telemetry.get("max_speed") or telemetry.get("m_max_speed")
    if isinstance(max_speed, (int, float)):
        threshold = 200 if _is_wet_weather(telemetry) else 280
        if max_speed < threshold:
            dx["hi_speed_stab_req"] -= 0.3


def _apply_ride_height_rules(
    telemetry: dict[str, Any], dx: dict[str, float],
) -> None:
    """规则9：刮底（底板触地）检测，原地修改 ``dx``。

    信号源：Packet 13 (MotionEx) 的 ``m_frontAeroHeight`` / ``m_rearAeroHeight``
    —— 规范定义为底板前/后缘离地高度（plank edge height above road surface），
    由 :class:`setup_tuner.telemetry.lap_aggregator.LapAggregator` 整圈累积后
    以 ``plank_bottoming`` / ``plank_*_height_min`` 等键传入。

    判定条件（满足其一）：
    - 整圈 ``plank_bottoming`` 为真（该圈至少一帧底板离地高度落入触地带）；
    - ``plank_*_height_min`` 低于触地上限（兜底：未累积布尔标志也能量化判断）。

    影响：``ride_height_req += 0.4``（抬高底盘 / 加硬弹簧以缓解刮底）。
    无 MotionEx 数据时不触发（保持既有行为，避免误报）。
    """
    bottoming = telemetry.get("plank_bottoming")
    if bottoming is None:
        # 退一步用最小离地高度判断（若提供方只给了量化值）
        minima = [
            telemetry.get("plank_front_height_min"),
            telemetry.get("plank_rear_height_min"),
        ]
        numeric = [
            float(m) for m in minima
            if isinstance(m, (int, float)) and not isinstance(m, bool)
        ]
        if numeric and min(numeric) <= _PLANK_BOTTOMING_MAX_M:
            bottoming = True
    if bottoming:
        dx["ride_height_req"] += 0.4


def _derive_telemetry_dx(telemetry: dict[str, Any] | None) -> dict[str, float]:
    """从遥测性能数据提取诊断向量贡献（9维Dx，16条规则）。

    与车手反馈Dx叠加后共同驱动调教优化模型。所有规则基于官方数据，
    覆盖全部9维Dx并引入方向修正因子（负值表示该能力过强需减弱）。

    16条规则分组：
        - 胎温规则（1/2/12/13）：胎温过高/过低/不均/湿地严重不足
        - 胎压规则（3）：胎压异常
        - 刹车规则（4/4b/8/14）：刹车过热/前后轴失衡/制动力不足/持续过热+方向修正
        - 刹车平衡规则（16）：游戏内实际读数与写入调教不一致
        - 弯道规则（5/7/10/11）：出弯油门低/入弯响应差/弯中不稳定/出弯打滑
        - 速度规则（6/15）：直道速度低/直道极速低（均含方向修正因子）
        - 底盘规则（9）：刮底检测（底板离地高度，来自 Packet 13 MotionEx）

    配方相关阈值：刹车温度告警阈值随 Packet 7 ``m_actualTyreCompound``
    换算的软/硬标记浮动（软胎更早告警、硬胎阈值上调）。

    方向修正因子：
        hi_speed_stab_req 取负值 = 需减阻/减翼（下压力过大）
        brake_power_req 取负值 = 制动力过强需减弱

    Args:
        telemetry: 遥测字典；None 时返回全 0.0 的 Dx。

    Returns:
        9 维 Dx 字典 {dim_key: value}，覆盖全部 DIAG_DIMS，无触发维度为 0.0。
    """
    dx: dict[str, float] = dict.fromkeys(DIAG_DIMS, 0.0)
    if not telemetry:
        return dx
    _apply_tyre_temperature_rules(telemetry, dx)
    _apply_tyre_pressure_rules(telemetry, dx)
    _apply_brake_rules(telemetry, dx)
    _apply_brake_bias_consistency(telemetry, dx)
    _apply_corner_rules(telemetry, dx)
    _apply_speed_rules(telemetry, dx)
    _apply_ride_height_rules(telemetry, dx)
    return dx


def _apply_brake_bias_consistency(
    telemetry: dict[str, Any], dx: dict[str, float],
) -> None:
    """规则16：刹车平衡实际生效核对。

    遥测字典里同时可能带两个量：
    - ``front_brake_bias``：游戏内实际读数（Packet 7 ``m_frontBrakeBias``）；
    - ``brake_bias_setup``：本工具写入调教的前轴刹车平衡值。

    若两者偏差 > 2 个百分点，说明调教未真正生效（例如游戏内手动覆盖、
    或赛车不允许该设置）——此时不应继续基于差值调参，而是抬高
    ``brake_stab_req`` 提示用户先在游戏内确认设置。

    任一侧缺失时不做任何判断（避免在无遥测时误报）。
    """
    actual = telemetry.get("front_brake_bias")
    setting = telemetry.get("brake_bias_setup")
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return
    if not isinstance(setting, (int, float)) or isinstance(setting, bool):
        return
    if abs(float(actual) - float(setting)) > 2.0:
        dx["brake_stab_req"] += 0.25


# ---------------------------------------------------------------------------
# 核心：compute_setup_delta（6 步确定性流水线）
# ---------------------------------------------------------------------------
def _compute_param_raw_delta(
    p: str, dx: dict[str, float], telemetry_gain: dict[str, float],
    track_gain: dict[str, float] | None = None,
    style_gain: dict[str, float] | None = None,
) -> float:
    """步骤 1-2：矩阵乘法 + 遥测/赛道/风格校准，返回校准后 raw delta。"""
    raw = 0.0
    for cell in nonzero_cells_for_param_cached(p):
        dx_val = dx.get(cell.diag, 0.0)
        if dx_val == 0.0:
            continue
        raw += dx_val * cell.value
    gain = telemetry_gain.get(p, 1.0)
    if track_gain:
        gain *= track_gain.get(p, 1.0)
    if style_gain:
        gain *= style_gain.get(p, 1.0)
    return raw * gain


def _align_param_delta(spec: Any, raw: float, current: float) -> float:
    """步骤 3-6：单次上限 + 合法区间 + 档位对齐，返回最终 delta。"""
    raw = _clip(raw, -spec.max_delta, spec.max_delta)
    next_val = _clip(current + raw, spec.min_val, spec.max_val)
    next_aligned = _align_to_step(next_val, spec.step, spec.min_val)
    next_aligned = _clip(next_aligned, spec.min_val, spec.max_val)
    delta_aligned = next_aligned - current
    # 整数参数（step >= 1 且为整数）round 到整数
    if spec.step >= 1.0 and float(spec.step).is_integer():
        delta_aligned = float(round(delta_aligned))
    return delta_aligned


def compute_setup_delta(
    dx: dict[str, float],
    current_setup: dict[str, float],
    telemetry_gain: dict[str, float] | None = None,
    track_gain: dict[str, float] | None = None,
    style_gain: dict[str, float] | None = None,
) -> dict[str, float]:
    """执行完整的 6 步 SetupDelta 计算流水线。

    确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。

    Args:
        dx: 诊断向量字典 {dim_key: value}（9 维，来自 compute_dx）。
        current_setup: 当前调教快照 {param: value}（21 参数）。
        telemetry_gain: 每参数的遥测幅度增益 {param: gain}；
            None 时全部默认 1.0。
        track_gain: 每参数的**赛道敏感度增益** {param: gain}（见
            ``domain.track_coefficients``）；None 时全部默认 1.0。
        style_gain: 每参数的**车手风格敏感度增益** {param: gain}（见
            ``domain.style_coefficients``）；None 时全部默认 1.0。

    Returns:
        SetupDelta 字典 {param: delta_value}，覆盖全部 21 参数。
        每参数 delta 满足：
            - |delta| <= max_delta[p]（单次上限）；
            - current[p] + delta ∈ [min[p], max[p]（合法区间）；
            - delta 对齐到 step 档位。

    Raises:
        KeyError: current_setup 缺失某参数。
    """
    if telemetry_gain is None:
        telemetry_gain = {f.name: 1.0 for f in ALL_SETUP_FIELDS}
    if track_gain is None:
        track_gain = {}
    if style_gain is None:
        style_gain = {}

    setup_delta: dict[str, float] = {}
    for spec in ALL_SETUP_FIELDS:
        p = spec.name
        raw = _compute_param_raw_delta(p, dx, telemetry_gain, track_gain, style_gain)
        current = float(current_setup[p])
        setup_delta[p] = _align_param_delta(spec, raw, current)
    return setup_delta


def _derive_style_gain(style_vector: list[float] | None) -> dict[str, float]:
    """按车手风格向量推导参数敏感度增益（L1 风格调制）。

    实现委托给 ``domain.style_coefficients``；向量缺失时返回全 1.0。
    """
    from setup_tuner.domain.style_coefficients import gain_for_style

    return gain_for_style(style_vector)

def _derive_track_gain(track_id: str) -> dict[str, float]:
    """按赛道标识推导参数敏感度增益（让建议因赛道而异）。

    实现委托给 ``domain.track_coefficients``（按 track_type 分组的温和倍数，
    未知赛道返回全 1.0）。此前 track_id 只传给神经网络分支，
    规则引擎与赛道无关 → suzuka 与 monza 输出逐位相同。
    """
    from setup_tuner.domain.track_coefficients import track_gain_for

    return track_gain_for(track_id)


# ---------------------------------------------------------------------------
# 完整建议生成
# ---------------------------------------------------------------------------
# 完整的参数 tradeoff 提示表（参数 → 方向 → tradeoff 文案）
_TRADEOFF_NOTES: dict[str, dict[str, str]] = {
    "front_wing": {
        "increase": "可能牺牲直道极速",
        "decrease": "可能提升直道极速但降低前轴下压力",
    },
    "rear_wing": {
        "increase": "可能牺牲直道极速",
        "decrease": "可能提升直道极速但降低后轴下压力",
    },
    "on_throttle_diff": {
        "increase": "可能增加出弯转向不足倾向",
        "decrease": "可能增加出弯转向过度倾向",
    },
    "off_throttle_diff": {
        "increase": "可能增加入弯转向过度倾向",
        "decrease": "可能增加入弯转向不足倾向",
    },
    "front_camber": {
        "increase": "可能增加直道轮胎内缘磨耗",
        "decrease": "可能降低弯中前轮抓地",
    },
    "rear_camber": {
        "increase": "可能增加直道轮胎内缘磨耗",
        "decrease": "可能降低弯中后轮抓地",
    },
    "front_toe": {
        "increase": "可能增加直道轮胎磨耗和阻力",
        "decrease": "可能降低前轴指向精度",
    },
    "rear_toe": {
        "increase": "可能增加直道轮胎磨耗和阻力",
        "decrease": "可能降低后轴稳定性",
    },
    "front_suspension": {
        "increase": "可能降低机械抓地但增高速稳定性",
        "decrease": "可能增加车身侧倾但增机械抓地",
    },
    "rear_suspension": {
        "increase": "可能降低机械抓地但增高速稳定性",
        "decrease": "可能增加车身侧倾但增机械抓地",
    },
    "front_anti_roll_bar": {
        "increase": "可能降低前轴独立抓地但增侧倾刚度",
        "decrease": "可能增加车身侧倾但增前轴独立抓地",
    },
    "rear_anti_roll_bar": {
        "increase": "可能降低后轴独立抓地但增侧倾刚度",
        "decrease": "可能增加车身侧倾但增后轴独立抓地",
    },
    "front_ride_height": {
        "increase": "可能降低前轴下压力中心",
        "decrease": "可能增加刮底风险",
    },
    "rear_ride_height": {
        "increase": "可能降低后轴下压力中心",
        "decrease": "可能增加刮底风险",
    },
    "brake_pressure": {
        "increase": "可能增加轮胎锁死风险",
        "decrease": "可能延长制动距离",
    },
    "brake_bias": {
        "increase": "可能增加前轮锁死倾向",
        "decrease": "可能增加后轮锁死倾向",
    },
    "engine_braking": {
        "increase": "可能增加弯中后轴扰动但助减速",
        "decrease": "可能减少弯中后轴扰动但降减速辅助",
    },
    "front_left_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
    "front_right_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
    "rear_left_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
    "rear_right_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
}


def _active_cells(spec_name: str, dx: dict[str, float]) -> list[Any]:
    """返回该参数上 Dx 分量非零的耦合单元（保持耦合矩阵原始顺序）。

    性能说明：报告组装此前对同一参数调用 ``nonzero_cells_for_param_cached``
    两次（联动说明 + 中文语义各一次）。抽出本函数后只取一次，减少
    报告生成耗时（实测该步骤占 generate_suggestion 的约 7 成）。
    """
    return [
        cell for cell in nonzero_cells_for_param_cached(spec_name)
        if dx.get(cell.diag, 0.0) != 0.0
    ]


def _collect_param_linkages(
    cells: list[Any], dx: dict[str, float],
) -> tuple[list[str], list[str]]:
    """收集参数的诊断维度联动描述与出处列表。"""
    linkages: list[str] = []
    sources: list[str] = []
    for cell in cells:
        dx_val = dx.get(cell.diag, 0.0)
        linkages.append(
            f"{cell.diag}({DIAG_DIMS_ZH[cell.diag]}) Dx={dx_val:+.2f} × C={cell.value:+.2f}",
        )
        if cell.source not in sources:
            sources.append(cell.source)
    return linkages, sources


def _build_linked_notes(cells: list[Any]) -> str:
    """构造参数的中文联动说明。"""
    if not cells:
        return "本次无需调整"
    linked_notes = "、".join(
        f"{DIAG_DIMS_POSITIVE_SEMANTICS.get(cell.diag, cell.diag)}"
        for cell in cells
    )
    return linked_notes or "由多个诊断维度联动调整"


def _build_param_detail(
    spec_name: str,
    current: float,
    delta: float,
    dx: dict[str, float],
) -> dict[str, Any]:
    """组装单参数的报告详情（联动说明 / 出处 / tradeoff）。"""
    cells = _active_cells(spec_name, dx)
    linkages, sources = _collect_param_linkages(cells, dx)
    linked_notes = _build_linked_notes(cells)
    source = ",".join(sources) if sources else ""

    tradeoff: str | None = None
    if abs(delta) > DELTA_ZERO_EPSILON:
        direction = "increase" if delta > 0 else "decrease"
        tradeoff = _TRADEOFF_NOTES.get(spec_name, {}).get(direction)

    return {
        "param": spec_name,
        "current": current,
        "next": current + delta,
        "setup_delta": delta,
        "linkages": linkages,
        "linked_notes": linked_notes,
        "source": source,
        "tradeoff": tradeoff,
    }


def _compute_nn_delta(
    model_type: str, symptoms: list, dx: dict, current_setup: dict[str, float],
    track_id: str,
) -> tuple[dict[str, float] | None, bool]:
    """计算神经网络 delta（若需要且可用），返回 (nn_delta, nn_available)。"""
    if model_type not in ("nn", "hybrid"):
        return None, False
    nn_manager = _get_nn_manager()
    if nn_manager is None or not nn_manager.available:
        return None, False
    nn_delta = nn_manager.predict(symptoms, dx, current_setup, track_id)
    return nn_delta, True


def _build_param_details(
    final_delta: dict[str, float], current_setup: dict[str, float], dx: dict,
) -> list[dict[str, Any]]:
    """逐参数构造报告详情列表。"""
    parameters: list[dict[str, Any]] = []
    for spec in ALL_SETUP_FIELDS:
        current = float(current_setup[spec.name])
        delta = final_delta[spec.name]
        parameters.append(_build_param_detail(spec.name, current, delta, dx))
    return parameters


def _build_suggestion_summary(
    final_delta: dict[str, float], dx: dict, parameters: list[dict[str, Any]],
) -> str:
    """构造建议摘要文本。"""
    nonzero_count = sum(1 for d in final_delta.values() if abs(d) > DELTA_ZERO_EPSILON)
    if is_zero_dx(dx):
        return "未检测到有效症状，本次无调整建议"
    return (
        f"本次建议共关联 {len(parameters)} 项参数，"
        f"其中 {nonzero_count} 项非零调整，整体性调教"
    )


_GENERATE_SUGGESTION_DOC = """完整建议生成（Dx → SetupDelta → 报告组装）。

确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。
时间戳由报告落库层（T7）在持久化时补充，本函数不引入时间依赖。

混合模型（task-43）：
    ``model_type`` 控制使用哪种模型分支：
        - ``"rule"``：纯规则引擎（6 步确定性流水线）
        - ``"nn"``：纯神经网络（不可用时降级为规则引擎）
        - ``"hybrid"``：混合（规则 60% + 神经网络 40%，不可用时降级）

    神经网络分支为可选依赖（PyTorch），不可用时自动降级为纯规则引擎，
    保证向后兼容。

阶段敏感扩展（task-60）：
    ``symptoms`` 支持二元组和三元组混合：
        - 二元组 ``(symptom, strength)``：使用症状的默认阶段（向后兼容）；
        - 三元组 ``(symptom, strength, stage)``：使用指定阶段的 Dx 映射。

遥测诊断叠加（task-89）：
    遥测数据不仅用于增益校准（``_derive_telemetry_gain``），还独立产生
    诊断向量贡献（``_derive_telemetry_dx``）。车手反馈Dx与遥测Dx按维度
    代数叠加后共同驱动调教优化模型：
        ``dx[dim] = feedback_dx[dim] + telemetry_dx[dim]``
    遥测Dx规则基于Pirelli官方轮胎数据与F1官方调教指南。

Args:
    symptoms: 症状列表，每项为二元组或三元组：
        - ``(symptom_key, strength)``：使用默认阶段；
        - ``(symptom_key, strength, stage)``：使用指定阶段。
    current_setup: 当前调教快照 {param: value}（20 参数）。
    track_id: 赛道标识。
    telemetry: 遥测客观数据（用于校准增益与置信度）；None 表示无遥测。
    model_type: 模型类型，``"rule"`` | ``"nn"`` | ``"hybrid"``。
        默认 ``"hybrid"``。未知值按 ``"rule"`` 处理。

Returns:
    建议报告字典，结构对齐 design 2.7.7：
    ::
        {{
          "track_id": str,
          "dx": {{dim: value}},
          "setup_delta": {{param: delta}},
          "parameters": [param_detail, ...],   # 21 项
          "confidence": "high|medium|low",
          "summary": str,
          "model_type": str,            # 实际使用的模型类型
          "nn_available": bool,         # 神经网络是否可用
        }}
"""


def generate_suggestion(
    symptoms: list[tuple[str, int]] | list[tuple[str, int, str]],
    current_setup: dict[str, float],
    track_id: str,
    telemetry: dict[str, Any] | None = None,
    model_type: str = "hybrid",
    style_vector: list[float] | None = None,
    feedbacks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """完整建议生成（Dx → SetupDelta → 整体收口 → 报告组装）。详见模块级文档。

    Args:
        symptoms: ``[(symptom, strength[, stage]), ...]``。
        current_setup: 当前调教（21 参数）。
        track_id: 赛道标识。
        telemetry: 遥测摘要。
        model_type: 模型类型。
        style_vector: 车手风格向量。
        feedbacks: **逐弯原始反馈**（含 ``corner_number``）。提供时改走
            :func:`holistic.class_weighted_dx` —— 用弯道号查到该弯的类别
            （慢/中/快）后重加权 Dx，使"慢发夹推头"与"高速弯推头"得到
            不同的调教动作。缺省时退化为原 ``compute_dx(symptoms)`` 路径。

    Returns:
        建议结果字典（在原有键之外新增 ``holistic``：赛道需求画像、
        逐弯加权说明、跨类别冲突、整体收口说明）。
    """
    demand = track_demand(track_id)
    holistic_block: dict[str, Any] = {
        "demand": demand.describe(),
        "corner_notes": [],
        "conflicts": [],
        "coherence_notes": [],
    }

    weighted = None
    if feedbacks:
        weighted = class_weighted_dx(feedbacks, track_id)
        feedback_dx = weighted.dx
        holistic_block["corner_notes"] = list(weighted.corner_notes)
        holistic_block["conflicts"] = [c.describe() for c in weighted.conflicts]
    else:
        feedback_dx = compute_dx(symptoms)

    telemetry_dx = _derive_telemetry_dx(telemetry)
    dx = {dim: feedback_dx[dim] + telemetry_dx[dim] for dim in DIAG_DIMS}
    telemetry_gain = _derive_telemetry_gain(telemetry)
    track_gain = _derive_track_gain(track_id)
    style_gain = _derive_style_gain(style_vector)
    rule_delta = compute_setup_delta(
        dx, current_setup, telemetry_gain, track_gain, style_gain,
    )

    nn_delta, nn_available = _compute_nn_delta(
        model_type, symptoms, dx, current_setup, track_id,
    )
    blended_delta, actual_model_type = _blend_delta(
        rule_delta, nn_delta, model_type, nn_available,
    )

    # 圈级整体优化：在"逐弯需求残差 + 显式代价（阻力/刮底/胎温/改动幅度）"
    # 这个目标函数上做确定性搜索，把一次线性步换成整圈净收益最大的解。
    # 关键：需求按弯道类别分列（来自 class_weighted_dx().by_class），
    # 于是慢弯与快弯各自带着不同诉求，单一参数集不可能同时满足 ——
    # 赛道弯型占比与车手反馈才真正决定取向，并产生可解释的取舍。
    optimized = optimize_setup(
        dx, current_setup, track_id,
        telemetry=telemetry, feedbacks=feedbacks,
        initial_delta=blended_delta,
        needs_by_class=(weighted.by_class if weighted else None),
        wet=_is_wet_weather(telemetry),
    )
    holistic_block["optimization"] = {
        "objective_before": round(optimized.before.total, 6),
        "objective_after": round(optimized.after.total, 6),
        "improvement": round(optimized.improvement, 6),
        "per_class_gain": optimized.per_class_gain,
        "costs": {
            "drag": round(optimized.after.drag_cost, 6),
            "bottoming": round(optimized.after.bottoming_cost, 6),
            "tyre_heat": round(optimized.after.tyre_heat_cost, 6),
            "effort": round(optimized.after.effort_cost, 6),
        },
        "trace": list(optimized.trace),
        "tradeoff": describe_tradeoff(optimized),
    }

    # 整体性收口：胎压左右对称 / 前后翼平衡窗口 / 改动预算（保证可用性）
    final_delta, coherence_notes = holistic_coherence(optimized.delta, dx, demand)
    holistic_block["coherence_notes"] = coherence_notes

    parameters = _build_param_details(final_delta, current_setup, dx)
    confidence = assess_confidence(symptoms, telemetry)
    summary = _build_suggestion_summary(final_delta, dx, parameters)

    return {
        "track_id": track_id,
        "dx": dx,
        "setup_delta": final_delta,
        "parameters": parameters,
        "confidence": confidence,
        "summary": summary,
        "model_type": actual_model_type,
        "nn_available": nn_available,
        "holistic": holistic_block,
    }


generate_suggestion.__doc__ = _GENERATE_SUGGESTION_DOC


def _blend_delta(
    rule_delta: dict[str, float],
    nn_delta: dict[str, float] | None,
    model_type: str,
    nn_available: bool,
) -> tuple[dict[str, float], str]:
    """根据 model_type 混合规则引擎与神经网络结果。

    Args:
        rule_delta: 规则引擎的 SetupDelta。
        nn_delta: 神经网络的 SetupDelta（None 表示不可用）。
        model_type: 请求的模型类型。
        nn_available: 神经网络是否可用。

    Returns:
        (final_delta, actual_model_type) 二元组。
        actual_model_type 为实际使用的模型类型（可能因降级而与请求不同）。
    """
    # 权重：hybrid 模式下规则 60% + 神经网络 40%
    RULE_WEIGHT = 0.6
    NN_WEIGHT = 0.4

    if model_type == "nn" and nn_delta is not None:
        # 纯神经网络模式
        return nn_delta, "nn"

    if model_type == "hybrid" and nn_delta is not None:
        # 混合模式：加权平均
        blended = {
            p: rule_delta[p] * RULE_WEIGHT + nn_delta[p] * NN_WEIGHT
            for p in rule_delta
        }
        return blended, "hybrid"

    # 降级为纯规则引擎
    if model_type in ("nn", "hybrid") and not nn_available:
        # 请求了 nn/hybrid 但神经网络不可用，降级
        return rule_delta, "rule"
    # model_type == "rule" 或未知值
    return rule_delta, "rule"


# ---------------------------------------------------------------------------
# 神经网络管理器单例（延迟加载，PyTorch 不可用时返回 None）
# ---------------------------------------------------------------------------
# 哨兵：表示"已尝试加载且失败"，用于区分"尚未尝试"（None）。
# 若失败后置回 None，则每个请求都会重新 import + 构造 NNModelManager
# （PyTorch 可用但权重缺失时仍会构造完整 F1SetupNet），造成反复开销。
_LOAD_FAILED: Any = object()
_NN_MANAGER: Any = None
# 初始化锁：FastAPI 默认线程池并发处理请求，串行化首次加载
_NN_MANAGER_LOCK = threading.Lock()


def _get_nn_manager() -> Any:
    """获取神经网络模型管理器单例（延迟加载）。

    PyTorch 不可用或首次加载失败时返回 None，引擎自动降级为纯规则引擎；
    加载结果（成功或失败）均被缓存，不在每次请求时重试。

    Returns:
        NNModelManager 实例或 None。
    """
    global _NN_MANAGER
    if _NN_MANAGER is None:
        with _NN_MANAGER_LOCK:
            # 双重检查：等待锁期间可能已有线程完成加载
            if _NN_MANAGER is None:
                try:
                    from .nn_model import NNModelManager
                    _NN_MANAGER = NNModelManager()
                except Exception:
                    logger.warning(
                        "神经网络管理器加载失败，降级为纯规则引擎", exc_info=True,
                    )
                    _NN_MANAGER = _LOAD_FAILED
    # 失败哨兵对外统一表现为 None
    if _NN_MANAGER is _LOAD_FAILED:
        return None
    return _NN_MANAGER


def reset_nn_manager() -> None:
    """重置神经网络管理器单例（供测试使用）。"""
    global _NN_MANAGER
    with _NN_MANAGER_LOCK:
        _NN_MANAGER = None


# ---------------------------------------------------------------------------
# 确定性自校验（任意单症状覆盖全部 20 参数）
# ---------------------------------------------------------------------------
def _validate_symptom_invariants(
    symptom: str, result1: dict, result2: dict, default_setup: dict,
) -> None:
    """校验单症状的覆盖性、确定性、不越界、出处完整性。"""
    # 1. 覆盖全部 20 参数
    assert set(result1["setup_delta"].keys()) == set(default_setup.keys()), (
        f"症状 {symptom!r} SetupDelta 未覆盖全部 20 参数"
    )
    # 2. 确定性
    assert result1["setup_delta"] == result2["setup_delta"], (
        f"症状 {symptom!r} 两次运行结果不一致（非确定性）"
    )
    # 3. 不越界 + 4. 非零 delta 有出处
    for spec in ALL_SETUP_FIELDS:
        _validate_param_in_bounds(symptom, spec, result1, default_setup)


def _validate_param_in_bounds(
    symptom: str, spec: Any, result1: dict, default_setup: dict,
) -> None:
    """校验单参数 next 在 [min, max] 且 |delta| <= max_delta，非零 delta 有出处。"""
    p = spec.name
    current = default_setup[p]
    delta = result1["setup_delta"][p]
    next_val = current + delta
    assert next_val >= spec.min_val - BOUND_TOLERANCE_EPSILON, (
        f"症状 {symptom!r} 参数 {p!r} next={next_val} < min={spec.min_val}"
    )
    assert next_val <= spec.max_val + BOUND_TOLERANCE_EPSILON, (
        f"症状 {symptom!r} 参数 {p!r} next={next_val} > max={spec.max_val}"
    )
    assert abs(delta) <= spec.max_delta + BOUND_TOLERANCE_EPSILON, (
        f"症状 {symptom!r} 参数 {p!r} |delta|={abs(delta)} > max_delta={spec.max_delta}"
    )
    if abs(delta) > BOUND_TOLERANCE_EPSILON:
        detail = next(pd for pd in result1["parameters"] if pd["param"] == p)
        assert detail["source"], (
            f"症状 {symptom!r} 参数 {p!r} 非零 delta 但 source 为空"
        )


def validate_engine() -> None:
    """构建期校验引擎确定性与覆盖性。

    校验项：
        1. 任意单症状（强度 3）产出 SetupDelta 覆盖全部 20 参数；
        2. 确定性：相同输入跑两次结果完全一致；
        3. 不越界：current + delta ∈ [min, max] 且 |delta| <= max_delta；
        4. 每参数非零 delta 的 source 非空。

    Raises:
        AssertionError: 任一校验不通过。
    """
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.domain.symptoms import Symptom

    default_setup = CarSetup.default().to_dict()

    for symptom in (s.value for s in Symptom):
        # 单症状强度 3，使用 rule 模式保持确定性自校验
        result1 = generate_suggestion(
            [(symptom, 3)], default_setup, "validate_track", None,
            model_type="rule",
        )
        result2 = generate_suggestion(
            [(symptom, 3)], default_setup, "validate_track", None,
            model_type="rule",
        )
        _validate_symptom_invariants(symptom, result1, result2, default_setup)