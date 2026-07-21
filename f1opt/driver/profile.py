"""车手画像 (driver profile) — 从统一遥测帧提取驾驶风格特征.

将 :mod:`f1opt.telemetry.aligner` 产出的统一帧序列 (``list[dict]``, 键见
``UNIFIED_KEYS``) 压缩为一个固定长度 8 的归一化向量, 用作调教代理模型
(:mod:`f1opt.model.surrogate`) 的条件输入.

公开 API:

- :class:`DriverProfile` — 8 个 [0,1] 风格标量 + ``to_vector`` / ``from_vector``.
- :func:`extract_driver_profile` — 从帧序列计算画像 (含证据).
- :data:`DEFAULT_PROFILE` / :data:`AGGRESSIVE_PROFILE` /
  :data:`CONSERVATIVE_PROFILE` — 手工设定样例 (供下游模型差异化测试).

向量顺序 (``to_vector`` 固定 8 维, 与 ``from_vector`` 互逆)::

    [brake_point_norm, throttle_smoothness, steer_smoothness,
     corner_balance_pref, aggression_score, consistency_score,
     ers_usage_intensity, drs_usage_efficiency]

实现说明: 仅依赖 ``numpy`` 与 ``pydantic`` (v2). 缺失字段 (None) 被跳过,
空帧返回全零画像. 各指标的取证信息保存在 pydantic 私有属性中, 不参与序列化.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from f1opt.numeric import clamp01 as _clamp01

# to_vector / from_vector 固定字段顺序 (8 维).
_FIELD_ORDER: tuple[str, ...] = (
    "brake_point_norm",
    "throttle_smoothness",
    "steer_smoothness",
    "corner_balance_pref",
    "aggression_score",
    "consistency_score",
    "ers_usage_intensity",
    "drs_usage_efficiency",
)

# 归一化参考常数.
_G_LAT_REF = 6.0       # 横向 g 力参考上限 (F1 量级约 6g).
_CORNER_RATIO_K = 2.0  # corner_balance_pref 饱和常数。
_VAR_K = 0.05          # consistency 方差饱和常数。
_BRAKE_ONSET = 0.3     # 制动触发阈值 (brake > 该值视为开始制动)。


def _arr(frames: list[dict[str, Any]], key: str) -> np.ndarray:
    """返回 ``key`` 字段全部非 None 值组成的 float64 数组 (保持原序)。"""
    raw = [f.get(key) for f in frames]
    return np.array([float(v) for v in raw if v is not None], dtype=np.float64)


def _first_t(frames: list[dict[str, Any]], key: str) -> float:
    """返回 ``key`` 字段首个非 None 帧的 session_time (无则首帧或 0.0)。"""
    for f in frames:
        if f.get(key) is not None:
            t = f.get("session_time")
            return float(t) if t is not None else 0.0
    if frames:
        t = frames[0].get("session_time")
        return float(t) if t is not None else 0.0
    return 0.0


class DriverProfile(BaseModel):
    """车手驾驶风格画像 (8 个 [0,1] 标量)。

    所有字段默认 0.0, 因此空数据构造的画像是合法的全零画像。通过
    :func:`extract_driver_profile` 计算的画像会附带证据 (见 :meth:`evidence`)。
    """

    brake_point_norm: float = Field(default=0.0, ge=0.0, le=1.0)
    throttle_smoothness: float = Field(default=0.0, ge=0.0, le=1.0)
    steer_smoothness: float = Field(default=0.0, ge=0.0, le=1.0)
    corner_balance_pref: float = Field(default=0.0, ge=0.0, le=1.0)
    aggression_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ers_usage_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    drs_usage_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)

    # 各指标的取证信息 (metric_name -> {frame_t, field, value}); 不参与序列化。
    _evidence: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    # --- 向量互转 -----------------------------------------------------------
    def to_vector(self) -> list[float]:
        """返回固定顺序的 8 维 [0,1] 向量 (顺序见模块文档)。"""
        return [float(getattr(self, name)) for name in _FIELD_ORDER]

    @classmethod
    def from_vector(cls, vec: list[float]) -> DriverProfile:
        """:meth:`to_vector` 的逆运算: 钳位到 [0,1] 后构造画像。"""
        if len(vec) != len(_FIELD_ORDER):
            raise ValueError(
                f"向量长度 {len(vec)} 与车手画像维度 {len(_FIELD_ORDER)} 不一致"
            )
        kwargs = {
            name: _clamp01(float(v))
            for name, v in zip(_FIELD_ORDER, vec, strict=True)
        }
        return cls(**kwargs)

    # --- 证据 ---------------------------------------------------------------
    def evidence(self) -> dict[str, dict[str, Any]]:
        """返回各指标的取证信息 (metric_name -> {frame_t, field, value})。"""
        return dict(self._evidence)


# --- 单指标计算 (每个返回 (value, frame_t)) ---------------------------------
def _brake_point_norm(
    frames: list[dict[str, Any]], track_length_m: float | None
) -> tuple[float, float]:
    """brake_point_norm = clamp(1 - mean(onset_d) / normalizer, 0, 1)。

    onset_d = brake 由 <= 阈值上升至 > 阈值 (0.3) 的边沿处的 lap_distance；
    normalizer = ``track_length_m`` (若提供且 > 0) 否则帧中最大 lap_distance。
    0 = 很晚制动, 1 = 很早制动。
    """
    onsets_d: list[float] = []
    prev_brake: float | None = None
    for f in frames:
        b = f.get("brake")
        if b is None:
            continue
        b = float(b)
        d = f.get("lap_distance")
        if (
            d is not None
            and b > _BRAKE_ONSET
            and (prev_brake is None or prev_brake <= _BRAKE_ONSET)
        ):
            onsets_d.append(float(d))
        prev_brake = b
    t0 = _first_t(frames, "brake")
    if not onsets_d:
        return 0.0, t0
    if track_length_m is not None and track_length_m > 0.0:
        normalizer = track_length_m
    else:
        ld = _arr(frames, "lap_distance")
        normalizer = float(np.max(ld)) if ld.size else 0.0
    if normalizer <= 0.0:
        return 0.0, t0
    return _clamp01(1.0 - float(np.mean(onsets_d)) / normalizer), t0


def _throttle_smoothness(frames: list[dict[str, Any]]) -> tuple[float, float]:
    """throttle_smoothness = clamp(1 - std(positive throttle gradient), 0, 1)。

    正向油门梯度 (逐帧 diff > 0) 的标准差越小, 油门越平顺。
    """
    t0 = _first_t(frames, "throttle")
    thr = _arr(frames, "throttle")
    if thr.size < 2:
        return 0.0, t0
    grad = np.diff(thr)
    pos = grad[grad > 0.0]
    if pos.size == 0:
        return 1.0, t0
    return _clamp01(1.0 - float(np.std(pos))), t0


def _steer_smoothness(frames: list[dict[str, Any]]) -> tuple[float, float]:
    """steer_smoothness = clamp(1 - std(steer first-difference), 0, 1)。

    转向一阶差分标准差越小, 转向越平顺。
    """
    t0 = _first_t(frames, "steer")
    st = _arr(frames, "steer")
    if st.size < 2:
        return 0.0, t0
    diff = np.diff(st)
    if diff.size == 0:
        return 1.0, t0
    return _clamp01(1.0 - float(np.std(diff))), t0


def _corner_balance_pref(frames: list[dict[str, Any]]) -> tuple[float, float]:
    """corner_balance_pref = ratio / (ratio + k)。

    ratio = avg|g_lat| / avg|steer| (仅在 |steer| > 0.2 的弯中帧上统计)；
    0 = 低 g_lat/转角 (偏推头), 1 = 高承诺。
    """
    glats: list[float] = []
    steers: list[float] = []
    for f in frames:
        s = f.get("steer")
        g = f.get("g_lat")
        if s is None or g is None:
            continue
        s = float(s)
        if abs(s) > 0.2:
            glats.append(abs(float(g)))
            steers.append(abs(s))
    t0 = _first_t(frames, "g_lat")
    if not steers:
        return 0.0, t0
    avg_g = float(np.mean(glats))
    avg_s = float(np.mean(steers))
    if avg_s <= 0.0:
        return 0.0, t0
    ratio = avg_g / avg_s
    return _clamp01(ratio / (ratio + _CORNER_RATIO_K)), t0


def _aggression_score(
    frames: list[dict[str, Any]], throttle_smoothness: float
) -> tuple[float, float]:
    """aggression = clamp((brake_aggr + thr_comp + glat_comp) / 3, 0, 1)。

    - brake_aggr = clamp(max(brake 正向梯度), 0, 1) — 制动猛踩程度；
    - thr_comp = 1 - throttle_smoothness — 油门不平顺贡献；
    - glat_comp = clamp(max|g_lat| / 6, 0, 1) — 横向 g 力极限。
    """
    t0 = _first_t(frames, "brake")
    brake = _arr(frames, "brake")
    brake_aggr = _clamp01(float(np.max(np.diff(brake)))) if brake.size >= 2 else 0.0
    glat = _arr(frames, "g_lat")
    glat_comp = (
        _clamp01(float(np.max(np.abs(glat))) / _G_LAT_REF) if glat.size else 0.0
    )
    thr_comp = 1.0 - _clamp01(throttle_smoothness)
    return _clamp01((brake_aggr + thr_comp + glat_comp) / 3.0), t0


def _consistency_score(frames: list[dict[str, Any]]) -> tuple[float, float]:
    """consistency = clamp(1 - 0.5 * (tv/(tv+k) + sv/(sv+k)), 0, 1)。

    tv / sv = throttle / steer 的逐帧方差, 越小越一致 → 一致性越高。
    """
    t0 = _first_t(frames, "throttle")
    thr = _arr(frames, "throttle")
    st = _arr(frames, "steer")
    tv = float(np.var(thr)) if thr.size else 0.0
    sv = float(np.var(st)) if st.size else 0.0
    tn = tv / (tv + _VAR_K)
    sn = sv / (sv + _VAR_K)
    return _clamp01(1.0 - 0.5 * (tn + sn)), t0


def _ers_usage_intensity(frames: list[dict[str, Any]]) -> tuple[float, float]:
    """ers_usage_intensity = clamp(max(frac_deploy, store_decrease), 0, 1)。

    frac_deploy = ers_deploy_mode > 0 的帧占比;
    store_decrease = clamp((ers_store[0] - ers_store[-1]) / max(ers_store[0], eps), 0, 1)。
    """
    t0 = _first_t(frames, "ers_deploy_mode")
    mode = _arr(frames, "ers_deploy_mode")
    frac_deploy = float(np.mean(mode > 0.0)) if mode.size else 0.0
    store = _arr(frames, "ers_store")
    store_decrease = 0.0
    if store.size >= 2:
        first = store[0]
        last = store[-1]
        denom = first if first > 1e-6 else 1e-6
        store_decrease = _clamp01((first - last) / denom)
    return _clamp01(max(frac_deploy, store_decrease)), t0


def _drs_usage_efficiency(frames: list[dict[str, Any]]) -> tuple[float, float]:
    """drs_usage_efficiency = clamp(count(drs_allowed > 0) / total, 0, 1)。

    统一帧只暴露 ``drs_allowed`` (是否允许), 无独立激活标志, 故以允许帧占比
    作为 DRS 使用效率代理。
    """
    t0 = _first_t(frames, "drs_allowed")
    drs = _arr(frames, "drs_allowed")
    if drs.size == 0:
        return 0.0, t0
    return _clamp01(float(np.mean(drs > 0.0))), t0


# --- 主入口 -----------------------------------------------------------------
def extract_driver_profile(
    frames: list[dict[str, Any]],
    setup: dict[str, Any] | None = None,
    track_length_m: float | None = None,
) -> DriverProfile:
    """从统一遥测帧序列计算车手画像。

    Parameters
    ----------
    frames
        :func:`f1opt.telemetry.aligner.TelemetryAligner.sample_60hz` 风格的
        帧字典列表 (键见 ``UNIFIED_KEYS``); 缺失字段可为 None (被跳过)。
    setup
        调教字典 (保留参数, 当前未参与计算, 供未来细化 brake_point 等)。
    track_length_m
        赛道长度 (米); 提供则用作 brake_point_norm 的归一化分母, 否则用帧中
        最大 lap_distance。

    Returns
    -------
    DriverProfile
        含 8 个 [0,1] 指标与证据; 空帧返回全零画像 (无证据)。
    """
    if not frames:
        return DriverProfile()

    bp, bp_t = _brake_point_norm(frames, track_length_m)
    ts, ts_t = _throttle_smoothness(frames)
    ss, ss_t = _steer_smoothness(frames)
    cb, cb_t = _corner_balance_pref(frames)
    ag, ag_t = _aggression_score(frames, ts)
    co, co_t = _consistency_score(frames)
    ers, ers_t = _ers_usage_intensity(frames)
    drs, drs_t = _drs_usage_efficiency(frames)

    profile = DriverProfile(
        brake_point_norm=bp,
        throttle_smoothness=ts,
        steer_smoothness=ss,
        corner_balance_pref=cb,
        aggression_score=ag,
        consistency_score=co,
        ers_usage_intensity=ers,
        drs_usage_efficiency=drs,
    )
    profile._evidence = {
        "brake_point_norm": {"frame_t": bp_t, "field": "lap_distance", "value": bp},
        "throttle_smoothness": {"frame_t": ts_t, "field": "throttle", "value": ts},
        "steer_smoothness": {"frame_t": ss_t, "field": "steer", "value": ss},
        "corner_balance_pref": {"frame_t": cb_t, "field": "g_lat", "value": cb},
        "aggression_score": {"frame_t": ag_t, "field": "brake", "value": ag},
        "consistency_score": {"frame_t": co_t, "field": "throttle", "value": co},
        "ers_usage_intensity": {
            "frame_t": ers_t,
            "field": "ers_deploy_mode",
            "value": ers,
        },
        "drs_usage_efficiency": {
            "frame_t": drs_t,
            "field": "drs_allowed",
            "value": drs,
        },
    }
    return profile


# --- 手工样例 (供下游模型差异化测试) ----------------------------------------
# DEFAULT_PROFILE = 全零画像 = "无证据/空画像" 语义 (与 extract_driver_profile
# 空帧返回值一致). 注意: 全零在 driver_physical_offset_s 计算中 = 最差车手
# (+0.75s 慢), 不是中性车手. 调用方若需中性车手应传 None (Iter-93: None -> 
# [0.5]*8 中性) 而非 DEFAULT_PROFILE.
DEFAULT_PROFILE = DriverProfile()

AGGRESSIVE_PROFILE = DriverProfile(
    brake_point_norm=0.15,
    throttle_smoothness=0.20,
    steer_smoothness=0.25,
    corner_balance_pref=0.85,
    aggression_score=0.90,
    consistency_score=0.30,
    ers_usage_intensity=0.80,
    drs_usage_efficiency=0.75,
)

CONSERVATIVE_PROFILE = DriverProfile(
    brake_point_norm=0.85,
    throttle_smoothness=0.85,
    steer_smoothness=0.80,
    corner_balance_pref=0.30,
    aggression_score=0.15,
    consistency_score=0.80,
    ers_usage_intensity=0.30,
    drs_usage_efficiency=0.35,
)


__all__ = [
    "DriverProfile",
    "extract_driver_profile",
    "DEFAULT_PROFILE",
    "AGGRESSIVE_PROFILE",
    "CONSERVATIVE_PROFILE",
]
