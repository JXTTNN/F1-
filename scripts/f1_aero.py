"""F1 气动模型（训练侧真值）：地效曲线 + 气动平衡 + 诱导阻力 + 平台稳定性。

为什么需要它
------------
原仿真把翼角和离地间隙当成线性能力项（``EFFECTS`` 里 ``A: ±0.035``），
于是「越低越快」单调成立，模型里不存在最优离地间隙，也不存在前后翼配平。
真实赛车不是这样：

1. **地效（floor）下压力随离地间隙是非线性的**：先随高度下降而上升，
   在某个窗口达到峰值，再往下进入失速（2022–25 的喷气/海豚跳，2026 的
   平底+小扩散器把这条曲线拉平缓，但依然存在峰值与失速）。
2. **气动平衡**：前后翼与前后离地间隙共同决定重心前后的下压力分配；
   偏离该赛道需要的平衡点会先在高速弯里丢时间（推头/甩尾）。
3. **诱导阻力**：CD = CD0 + k·CL²，下压力本身带来阻力，所以"翼角拉满"
   必然被惩罚，最优解在内部。
4. **平台稳定性**：软悬挂 + 低离地间隙时，制动俯冲让离地间隙瞬态变化，
   气动平台失稳，高速弯丢时间。

2026 规则依据（写死在常数里，便于复核）
----------------------------------------
- 地效通道取消、平底 + 加长扩散器：总下压力比 2022–25 低约 30%，
  阻力低约 40–55%；离地间隙窗口更宽、曲线更平缓
  （F1 官方 2026 规则说明；f1briefing 对"downforce curve far less sudden"的解释）。
- 主动气动：直道 X-mode（低阻）/ 弯道 Z-mode（高下压），两翼均可变。
  因此**直道阻力对翼角的敏感度显著低于旧规则**，本模型用 ``ACTIVE_AERO_DRAG`` 折减。
- 轮胎：前胎窄 25 mm、后胎窄 30 mm → 机械抓地略降，本模型通过既有 M 项体现。

归一化约定：``n[p] = (value - default) / half_range``，n ∈ [-1, 1]，
与 ``build_setup_sim_dataset.capability_deltas`` 完全一致。
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# 常数（全部有物理含义；改动必须重跑 self_test）
# --------------------------------------------------------------------------- #
#: 地效峰值位置（归一化平台高度，负 = 比默认更低）
FLOOR_PEAK_H = -0.30
#: 地效曲线半宽：h 偏离峰值 1.0 时下压力降到约 0（2026 平底 → 比旧规则宽）
FLOOR_HALF_WIDTH = 1.25
#: 失速起点（归一化平台高度）；低于此值下压力骤降并出现海豚跳罚
FLOOR_STALL_H = -0.78
#: 失速区下压力损失比例（从失速点往下到 -1.0 线性增加）
FLOOR_STALL_LOSS = 0.45
#: 底板在总下压力中的占比（2026 平底：比 2022–25 的 0.6 低）
FLOOR_CL_SHARE = 0.50
#: 前后翼每单位归一化行程贡献的相对下压力
WING_CL_FRONT = 0.042
WING_CL_REAR = 0.036
#: 车高每下降 1 个归一化单位，气动平台增益（仅在线性段生效）
PLATFORM_GAIN = 0.10
#: 前翼每单位行程带来的气动平衡前移量（占前轴份额）
BALANCE_FRONT_GAIN = 0.055
BALANCE_REAR_GAIN = 0.048
#: 平衡偏差罚（秒 / 单位平衡偏差，乘高速弯占比与圈速因子）
BALANCE_PENALTY_K = 5.2
#: 失速罚（秒）与海豚跳附加阻力
STALL_PENALTY_S = 0.18
STALL_DRAG_GAIN = 0.35
#: 平台失稳罚（秒 / (低车高 × 软悬挂 × 高速弯占比)）
PLATFORM_PENALTY_K = 0.22
#: 2026 主动气动：直道 X-mode 把翼角阻力折减到原来的这个比例
ACTIVE_AERO_DRAG = 0.65
#: 诱导阻力系数（CD = CD0 + k·CL²；k ≈ 1/(π·AR·e)，AR≈3.2、e≈0.75）
INDUCED_DRAG_K = 0.13
#: 单位 CD 变化对应的圈速代价（ms / CD，再乘 drag_scale）。
#: 标定依据：前后翼各加满（CL +约 8%）在 Monza 量级约 +0.25–0.35 s。
DRAG_MS_PER_CD = 90_000.0
#: 底板诱导阻力系数：地效下压力比翼面"省阻"（L/D 更高），故低于机翼的 0.13
FLOOR_DRAG_K = 0.06
#: 下压力 → 圈速的全局折算。公开工程量级：1 点下压力（1%）≈ 0.02–0.03 s/圈，
#: 即 1% CL 约合 0.02 s；未折算前本模型约强 3 倍（6.7% CL 给到 0.58 s），故取 0.35。
AERO_GAIN_SCALE = 0.35

#: 总开关。默认 False：
#: v2 气动标尺训出的 NN 在本仓库评测门禁上是 482 ms 局部 MAE（门槛 120），
#: 尚不足以随包发布；用 AERO_V2=True 跑实验、验证通过后再改默认值。
AERO_V2 = False

_RH_KEYS = ("front_ride_height", "rear_ride_height")
_SUSP_KEYS = ("front_suspension", "rear_suspension")


def _avg(n: dict[str, float], keys: tuple[str, ...]) -> float:
    return sum(float(n.get(k, 0.0)) for k in keys) / max(1, len(keys))


def floor_downforce(h: float) -> tuple[float, float]:
    """平台高度 h（归一化）→ (地效下压力系数, 失速程度 0..1)。

    曲线：峰值在 ``FLOOR_PEAK_H``，向两侧二次衰减；低于 ``FLOOR_STALL_H``
    进入失速，下压力按 ``FLOOR_STALL_LOSS`` 线性损失。
    """
    if h >= FLOOR_STALL_H:
        x = (h - FLOOR_PEAK_H) / FLOOR_HALF_WIDTH
        val = max(0.0, 1.0 - x * x)
        stall = 0.0
    else:
        # 失速区：先取失速点的值，再按深度衰减
        x = (FLOOR_STALL_H - FLOOR_PEAK_H) / FLOOR_HALF_WIDTH
        base = max(0.0, 1.0 - x * x)
        depth = min(1.0, (FLOOR_STALL_H - h) / 0.22)
        val = base * (1.0 - FLOOR_STALL_LOSS * depth)
        stall = depth
    return val, stall


def _floor_reference() -> float:
    """默认调教（h=0）的地效系数，用于把 A 增量归一到 0。"""
    return floor_downforce(0.0)[0]


def aero_state(n: dict[str, float]) -> dict[str, float]:
    """归一化调教 → 气动状态。

    Returns:
        ``cl`` 总下压力系数（相对默认，1.0 = 默认）、``balance`` 前轴份额、
        ``stall`` 失速程度、``rake`` 前后车高差、``platform`` 平台失稳度。
    """
    f_rh = float(n.get("front_ride_height", 0.0))
    r_rh = float(n.get("rear_ride_height", 0.0))
    h = 0.5 * (f_rh + r_rh)
    rake = r_rh - f_rh

    floor, stall = floor_downforce(h)
    floor_ref = _floor_reference()
    floor_rel = floor / floor_ref if floor_ref > 1e-9 else 1.0

    wing_front = WING_CL_FRONT * float(n.get("front_wing", 0.0))
    wing_rear = WING_CL_REAR * float(n.get("rear_wing", 0.0))

    # 总下压力：底板份额 + 翼份额；底板随车高非线性变化，翼角线性
    cl = FLOOR_CL_SHARE * floor_rel + (1.0 - FLOOR_CL_SHARE) * (
        1.0 + wing_front + wing_rear
    )
    # 平台下压带来的额外增益（线性段），仅当未失速
    if stall <= 0.0:
        cl += PLATFORM_GAIN * max(0.0, -h)

    # 气动平衡（前轴份额）：前翼↑ → 前移；后翼↑ → 后移；前后车高差（rake）也影响
    balance = (
        0.44
        + BALANCE_FRONT_GAIN * float(n.get("front_wing", 0.0))
        - BALANCE_REAR_GAIN * float(n.get("rear_wing", 0.0))
        + 0.02 * (f_rh - r_rh)
    )

    # 平台失稳：低车高 + 软悬挂
    soft = max(0.0, -_avg(n, _SUSP_KEYS))
    platform = max(0.0, -h) * soft

    return {
        "cl": cl,
        "balance": balance,
        "stall": stall,
        "rake": rake,
        "platform": platform,
        "h": h,
    }


def balance_target(track: dict[str, Any]) -> float:
    """该赛道需要的前轴气动平衡份额。

    高速弯越多越需要前移（前翼多给一点，避免高速推头）；慢弯多的街道赛
    反而需要略偏后（后轴稳定、出弯牵引）。
    """
    fast = float(track.get("fast_share", 0.3) or 0.3)
    slow = float(track.get("slow_share", 0.3) or 0.3)
    return 0.435 + 0.05 * fast - 0.03 * slow


def aero_capability(n: dict[str, float]) -> float:
    """气动抓地力相对默认的变化量（喂给 ``capability_deltas`` 的 A 项）。"""
    st = aero_state(n)
    return (st["cl"] - 1.0) * AERO_GAIN_SCALE


def _penalty_raw(track: dict[str, Any], st: dict[str, float]) -> float:
    lap = float(track.get("lap_time_s", 90.0) or 90.0)
    fast = float(track.get("fast_share", 0.3) or 0.3)
    bal_err = abs(st["balance"] - balance_target(track))
    penalty = BALANCE_PENALTY_K * bal_err * (0.25 + 0.75 * fast) * (lap / 90.0)
    if st["stall"] > 0.0:
        penalty += STALL_PENALTY_S * st["stall"]
    penalty += PLATFORM_PENALTY_K * st["platform"] * (0.3 + 0.7 * fast) * (lap / 90.0)
    return penalty


def aero_penalty_s(track: dict[str, Any], n: dict[str, float]) -> float:
    """气动平衡 / 失速 / 平台失稳的时间罚（秒，正 = 更慢；默认调教为 0）。"""
    st = aero_state(n)
    # 这是「相对默认调教」的增量模型：默认工况的基准罚必须扣掉，
    # 否则空调教也会凭空多出几十毫秒。
    return _penalty_raw(track, st) - _penalty_raw(track, aero_state({}))


def aero_drag_ms(track: dict[str, Any], n: dict[str, float]) -> float:
    """诱导阻力带来的圈速代价（ms，正 = 更慢）。

    CD = CD0 + k·CL²；2026 主动气动在直道切 X-mode，翼角那部分阻力被
    ``ACTIVE_AERO_DRAG`` 折减，底板阻力不受益（平底仍在气流里）。
    """
    st = aero_state(n)
    drag_scale = float(track.get("drag_scale", 0.0) or 0.0)
    if drag_scale <= 0.0:
        return 0.0
    # 分别算翼面与底板的诱导阻力：两者 L/D 不同，混在一起会让"降车高"
    # 和"加翼角"付出同样的阻力代价，与真实气动不符。
    wing_cl = (1.0 - FLOOR_CL_SHARE) * (1.0 + WING_CL_FRONT * float(n.get("front_wing", 0.0))
                                        + WING_CL_REAR * float(n.get("rear_wing", 0.0)))
    wing_ref = (1.0 - FLOOR_CL_SHARE) * 1.0
    floor_cl = FLOOR_CL_SHARE * (st["cl"] - (1.0 - FLOOR_CL_SHARE) * (
        1.0 + WING_CL_FRONT * float(n.get("front_wing", 0.0))
        + WING_CL_REAR * float(n.get("rear_wing", 0.0)))) / max(1e-9, FLOOR_CL_SHARE)
    floor_ref = FLOOR_CL_SHARE * 1.0
    d_cd_wing = INDUCED_DRAG_K * (wing_cl * wing_cl - wing_ref * wing_ref)
    d_cd_floor = FLOOR_DRAG_K * (floor_cl * floor_cl - floor_ref * floor_ref)
    # 主动气动只削翼角那部分阻力
    total = d_cd_wing * ACTIVE_AERO_DRAG + d_cd_floor
    # 失速时额外阻力（气流分离）
    if st["stall"] > 0.0:
        total += abs(d_cd_floor) * STALL_DRAG_GAIN * st["stall"]
    # drag_scale 的量纲与旧模型一致：ms = 系数 × drag_scale
    return total * DRAG_MS_PER_CD * drag_scale


def self_test() -> dict[str, Any]:
    """常数自检：返回关键敏感性，供回归对比。"""
    base = {k: 0.0 for k in (
        "front_ride_height", "rear_ride_height", "front_wing", "rear_wing",
        "front_suspension", "rear_suspension",
    )}
    out: dict[str, Any] = {"default_cl": round(aero_state(base)["cl"], 4)}

    # 1) 地效曲线：从最高到最低扫一遍，确认存在峰值与失速
    curve = []
    for i in range(21):
        h = -1.0 + i * 0.1
        val, stall = floor_downforce(h)
        curve.append((round(h, 2), round(val, 3), round(stall, 2)))
    out["floor_curve"] = curve

    # 2) 前翼 +50%（归一化 +0.5）的平衡与下压力变化
    n = dict(base); n["front_wing"] = 0.5
    st = aero_state(n)
    out["front_wing_up"] = {
        "cl": round(st["cl"], 4), "balance": round(st["balance"], 4),
    }
    # 3) 后翼 +50%
    n = dict(base); n["rear_wing"] = 0.5
    st = aero_state(n)
    out["rear_wing_up"] = {"cl": round(st["cl"], 4), "balance": round(st["balance"], 4)}
    # 4) 车高降到 -0.8（失速区）
    n = dict(base); n["front_ride_height"] = -0.8; n["rear_ride_height"] = -0.8
    st = aero_state(n)
    out["stall_point"] = {
        "cl": round(st["cl"], 4), "stall": round(st["stall"], 3),
    }
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), ensure_ascii=False, indent=2))
