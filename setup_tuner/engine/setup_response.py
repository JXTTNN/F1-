"""调教 → 遥测响应模型（F1 25 / 2026 Season Pack 游戏口径）。

为什么需要这一层
----------------
诊断引擎 :func:`setup_tuner.engine.telemetry_diagnosis.diagnose_from_telemetry`
只会"读症状"，但系统里没有任何模块把**调教变化**翻译成**遥测症状变化**。
缺了这一层会导致：

1. 无法验证诊断方向是否正确（不知道"前翼调低"该出现什么症状）；
2. 无法做反事实推演（改这个调教，遥测会怎么变）；
3. 优化器只能依赖标量圈速代理，而该代理在真实数据上中位 ρ 仅 0.16。

设计原则：**以缺省调教为原点**
--------------------------------
游戏里的缺省调教是一套均衡设置，跑标准工况应当"几乎没有症状"。所以本模块
所有症状都由**相对缺省值的偏移量**驱动，而不是由绝对物理量驱动：

- 缺省调教 + 标准工况 → 症状数 ≈ 0（信噪比的前提）；
- 单项拉到极端 → 必然跨过诊断阈值，且方向与游戏物理一致。

这样既保证"没调坏就没症状"，也保证"调坏了能被查出来"。

物理口径（与游戏 Garage 语义一致）
--------------------------------
- 前翼/后翼 0–50 级：翼片越大下压力越大、直线阻力越大。**前翼高 → 前轴下
  压力占比高 → 转向更灵（推头减轻、高速易转向过度）**；后翼高则相反。
- 差速器（开/关油门）10–100%：开油门差速越大，出弯越容易甩尾。
- 前束 0–0.20°：toe-out 越大入弯越灵，但直道不稳、胎温更高。
- 悬挂 1–41 级、防倾杆 1–21 级：数值越大越硬。**前防倾杆越硬 → 前轴侧倾
  越小 → 前轮载荷转移越大 → 推头**；后防倾杆越硬 → 转向过度。
- 车高：前 15–35 / 后 40–60 级。越低空力平台越好，但底板触地风险越高。
- 刹车压力 80–100%、刹车平衡 50–70%：压力越大越易锁死；平衡越偏前，
  入弯越稳但前轮负担越大。
- 胎压：前 22.5–29.5 psi / 后 20.5–26.5 psi，偏离工作窗口即出症状。

用法::

    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.setup_response import simulate_lap_agg, predict_symptoms

    agg = simulate_lap_agg(CarSetup(), "monza", {"wet": 0.0, "track_temp": 35.0})
    symptoms = predict_symptoms(CarSetup(), "monza")
"""

from __future__ import annotations

from typing import Any

from setup_tuner.domain.setup import CarSetup
from setup_tuner.domain.track import ALL_TRACKS, Track
from setup_tuner.engine.holistic import track_demand

#: 车轮顺序统一为官方 UDP：RL, RR, FL, FR（[0][1]=后轴，[2][3]=前轴）
WHEEL_ORDER = ("RL", "RR", "FL", "FR")

#: 缺省胎温（°C）：缺省调教在标准工况下的工作温度
BASE_TYRE_TEMP = 92.0
#: 缺省刹车温度（°C）
BASE_BRAKE_TEMP_FRONT = 420.0
BASE_BRAKE_TEMP_REAR = 400.0


def _track(track_id: str) -> Track:
    for t in ALL_TRACKS:
        if t.track_id == track_id:
            return t
    raise KeyError(f"未知赛道 {track_id!r}")


def _dev(value: float, default: float, lo: float, hi: float) -> float:
    """把游戏取值转成**相对缺省的有符号归一化偏移**（-1..1）。

    这是整个模块的核心：症状只由偏移驱动，缺省即原点。
    """
    if hi <= lo:
        return 0.0
    if value >= default:
        span = hi - default
    else:
        span = default - lo
    if span <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, (value - default) / span))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def balance_deviation(setup: CarSetup) -> dict[str, float]:
    """前后轴失衡度（相对缺省）。

    Returns:
        ``balance``：>0 前轴相对更强（转向过度倾向），<0 推头倾向。
        ``df_dev``：总下压力相对缺省的偏移（>0 阻力更大）。
    """
    fw = _dev(setup.front_wing, 25.0, 0.0, 50.0)
    rw = _dev(setup.rear_wing, 25.0, 0.0, 50.0)
    farb = _dev(setup.front_anti_roll_bar, 6.0, 1.0, 21.0)
    rarb = _dev(setup.rear_anti_roll_bar, 6.0, 1.0, 21.0)
    fs = _dev(setup.front_suspension, 6.0, 1.0, 41.0)
    rs = _dev(setup.rear_suspension, 6.0, 1.0, 41.0)
    frh = _dev(setup.front_ride_height, 25.0, 15.0, 35.0)
    rrh = _dev(setup.rear_ride_height, 50.0, 40.0, 60.0)
    toe = _dev(setup.front_toe, 0.0, 0.0, 0.20)

    balance = (
        0.50 * fw - 0.50 * rw          # 翼片：前翼增推头减轻，后翼增推头
        - 0.40 * farb + 0.40 * rarb    # 防倾杆：前硬→推头，后硬→过度
        - 0.15 * fs + 0.15 * rs        # 弹簧：同向但弱于防倾杆
        + 0.12 * frh - 0.12 * rrh      # 车高：抬高前轴→前轴响应变差→推头
        + 0.40 * toe                   # 前束 toe-out：前轴响应更灵→过度倾向
    )
    return {
        "balance": _clamp(balance, -2.0, 2.0),
        "df_dev": (fw + rw) / 2.0,
        "wing_dev_front": fw,
        "wing_dev_rear": rw,
    }


def _slide_amounts(setup: CarSetup) -> tuple[float, float]:
    """``(slide_front, slide_rear)``：0..1 滑移强度，互斥。

    推头（前轴先到极限）→ ``slide_front``；转向过度 → ``slide_rear``。
    """
    bal = balance_deviation(setup)["balance"]
    if bal >= 0.0:
        return 0.0, _clamp(bal * 2.2, 0.0, 1.0)
    return _clamp(-bal * 2.2, 0.0, 1.0), 0.0


def _tyre_temps(
    setup: CarSetup,
    conditions: dict[str, Any],
    slide_front: float,
    slide_rear: float,
) -> tuple[list[float], list[float]]:
    """胎面/胎内温度（°C），顺序 RL, RR, FL, FR。"""
    wet = float(conditions.get("wet") or 0.0)
    track_temp = float(conditions.get("track_temp") or 35.0)
    bal = balance_deviation(setup)

    # 工况只整体平移，不改变前后差
    base = BASE_TYRE_TEMP + 0.25 * (track_temp - 35.0) - 22.0 * wet
    base += 3.0 * bal["df_dev"]

    # 束角/外倾偏离缺省 → 轮胎偏磨、温度上升
    toe_dev = _dev(setup.front_toe, 0.0, 0.0, 0.20)
    camber_dev = _dev(setup.front_camber, -3.5, -3.5, -2.5)
    front = base + 5.0 * toe_dev + 3.0 * camber_dev + 15.0 * slide_front - 9.0 * slide_rear
    rear = base + 4.0 * _dev(setup.rear_toe, 0.20, 0.10, 0.35) \
        + 15.0 * slide_rear - 9.0 * slide_front

    # 胎压偏离缺省 → 该轴温度整体偏移（高压升温更快、接地更小）
    fp = (float(setup.front_left_tyre_pressure) + float(setup.front_right_tyre_pressure)) / 2.0
    rp = (float(setup.rear_left_tyre_pressure) + float(setup.rear_right_tyre_pressure)) / 2.0
    front += 6.0 * _dev(fp, 23.5, 22.5, 29.5)
    rear += 6.0 * _dev(rp, 22.0, 20.5, 26.5)

    # 前束造成左右温差（toe-out 让外侧轮负担更重）
    lateral = 4.0 * abs(toe_dev)
    surf = [rear - lateral, rear + lateral, front - lateral, front + lateral]
    inner = [t + 6.0 for t in surf]
    return surf, inner


def _tyre_pressures(setup: CarSetup) -> list[float]:
    return [
        float(setup.rear_left_tyre_pressure),
        float(setup.rear_right_tyre_pressure),
        float(setup.front_left_tyre_pressure),
        float(setup.front_right_tyre_pressure),
    ]


def _tyre_wear(
    setup: CarSetup,
    conditions: dict[str, Any],
    slide_front: float,
    slide_rear: float,
) -> list[float]:
    """单圈胎耗（%）：缺省约 1%，滑移越强耗得越快。"""
    wet = float(conditions.get("wet") or 0.0)
    bal = balance_deviation(setup)
    base = 1.0 - 0.4 * wet + 0.4 * bal["df_dev"]
    front = base + 2.4 * slide_front
    rear = base + 2.4 * slide_rear
    return [rear, rear, front, front]


def _brake_temps(
    setup: CarSetup,
    track_id: str,
    conditions: dict[str, Any],
) -> list[float]:
    """刹车温度（°C）：压力/平衡偏离缺省才升温。"""
    pressure = _dev(setup.brake_pressure, 90.0, 80.0, 100.0)
    bias = _dev(setup.brake_bias, 58.0, 50.0, 70.0)
    braking = getattr(track_demand(track_id), "braking_index", 0.35)
    wet = float(conditions.get("wet") or 0.0)
    front = BASE_BRAKE_TEMP_FRONT + 170.0 * pressure + 110.0 * bias \
        + 45.0 * braking - 90.0 * wet
    rear = BASE_BRAKE_TEMP_REAR + 85.0 * pressure - 75.0 * bias \
        + 30.0 * braking - 90.0 * wet
    return [rear, rear, front, front]


def _plank(setup: CarSetup, track_id: str) -> dict[str, Any]:
    """底板触地：只有车高低于缺省才出现。"""
    frh = _dev(setup.front_ride_height, 25.0, 15.0, 35.0)
    rrh = _dev(setup.rear_ride_height, 50.0, 40.0, 60.0)
    aero = getattr(track_demand(track_id), "aero_index", 0.3)
    ratio = _clamp(
        max(0.0, -frh) * 1.0 + max(0.0, -rrh) * 0.6
        + 0.10 * aero * max(0.0, -frh),
        0.0, 1.0,
    )
    front_m = 0.045 + frh * 0.0125
    rear_m = 0.048 + rrh * 0.0100
    return {
        "plank_bottoming": ratio >= 0.08,
        "plank_bottoming_ratio": round(ratio, 4),
        "plank_front_height_min": round(front_m, 4),
        "plank_rear_height_min": round(rear_m, 4),
        "suspension_height_min": round(min(front_m, rear_m), 4),
    }


def _kerbs(setup: CarSetup, track_id: str) -> list[dict[str, Any]]:
    """按弯给路肩冲击比：悬挂越硬、路肩越多，冲击越大。"""
    fs = _dev(setup.front_suspension, 6.0, 1.0, 41.0)
    rs = _dev(setup.rear_suspension, 6.0, 1.0, 41.0)
    track = _track(track_id)
    # 低速弯多、路肩多的赛道冲击更明显
    slow_share = getattr(track_demand(track_id), "slow_share", 0.3)
    ratio = 1.0 + 0.80 * max(0.0, fs) + 0.60 * max(0.0, rs) + 0.25 * slow_share
    corners = getattr(track, "corners", []) or []
    return [
        {"corner": int(c.number), "ratio": round(ratio, 3)}
        for c in corners
        if getattr(c, "corner_type", "") in ("slow", "medium")
    ]


def _inputs(
    setup: CarSetup,
    slide_front: float,
    slide_rear: float,
) -> dict[str, Any]:
    """转向/油门特征：推头要打更多方向，转向过度要反复修正。"""
    bal = balance_deviation(setup)
    toe = _dev(setup.front_toe, 0.0, 0.0, 0.20)
    max_steer = _clamp(0.14 + 0.24 * slide_front + 0.08 * slide_rear, 0.0, 1.0)
    avg_steer = _clamp(
        0.06 + 0.15 * slide_front + 0.12 * slide_rear + 0.12 * toe, 0.0, 1.0,
    )
    df = bal["df_dev"]
    straight_ratio = 0.32 + 0.22 * df
    return {
        "max_steer": round(max_steer, 4),
        "avg_steer": round(avg_steer, 4),
        "avg_speed": round(245.0 - 40.0 * df, 1),
        "straight_ratio": round(straight_ratio, 4),
        "aero_straight_ratio": round(straight_ratio, 4),
        # 开油门差速偏离缺省越多，出弯越容易打滑
        "traction_control": bool(
            _dev(setup.on_throttle_diff, 50.0, 10.0, 100.0) > 0.75
        ),
    }


def simulate_lap_agg(
    setup: CarSetup,
    track_id: str,
    conditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """由调教生成整圈遥测摘要（与 UDP 解析后的 ``lap_agg`` 同构）。

    Args:
        setup: 20 项调教。
        track_id: 赛道标识（``ALL_TRACKS`` 之一）。
        conditions: 可选工况，支持 ``wet`` / ``track_temp`` / ``air_temp``。

    Returns:
        可直接喂给 ``diagnose_from_telemetry`` 的字典。
    """
    cond = dict(conditions or {})
    slide_front, slide_rear = _slide_amounts(setup)
    surf, inner = _tyre_temps(setup, cond, slide_front, slide_rear)

    bias_dev = _dev(setup.brake_bias, 58.0, 50.0, 70.0)
    pressure_dev = _dev(setup.brake_pressure, 90.0, 80.0, 100.0)

    agg: dict[str, Any] = {
        "track_id": track_id,
        "m_tyresSurfaceTemperature": [round(v, 1) for v in surf],
        "m_tyresInnerTemperature": [round(v, 1) for v in inner],
        "m_tyresPressure": [round(v, 2) for v in _tyre_pressures(setup)],
        "tyres_wear": [round(v, 2) for v in
                       _tyre_wear(setup, cond, slide_front, slide_rear)],
        "m_brakesTemperature": [round(v, 0) for v in
                                _brake_temps(setup, track_id, cond)],
        # 锁死：刹车压力或平衡显著偏离缺省
        "anti_lock_brakes": bool(
            pressure_dev > 0.90 or bias_dev > 0.85
            or (pressure_dev > 0.5 and bias_dev > 0.5)
        ),
        "kerb_corners": _kerbs(setup, track_id),
        "floor_damage": 0.0,
        "wing_damage_max": 0.0,
        "damage_severe": False,
        "is_soft_compound": bool(cond.get("soft_compound")),
        "is_hard_compound": bool(cond.get("hard_compound")),
    }
    agg.update(_plank(setup, track_id))
    agg.update(_inputs(setup, slide_front, slide_rear))
    return agg


def predict_symptoms(
    setup: CarSetup,
    track_id: str,
    conditions: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """一步得到诊断症状（调教 → 遥测 → 症状）。"""
    from setup_tuner.engine.telemetry_diagnosis import diagnose_from_telemetry

    return diagnose_from_telemetry(
        simulate_lap_agg(setup, track_id, conditions), track_id,
    )
