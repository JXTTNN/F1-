"""批量 / 网格 / 敏感度 / Pareto 推理辅助 (perf 层).

向量化多 setup 推理, 构建在 :mod:`f1opt.model.surrogate` 之上:

- :func:`batch_predict_lap_times` — N 个 setup 单次 forward 批量圈速.
- :func:`batch_predict_full` — N 个 setup 批量富预测 (圈速 / 分段 / 响应).
- :func:`predict_lap_time_grid` — 扫描单个调教字段, 返回各档位圈速.
- :func:`sensitivity_analysis` — 19 个字段逐维敏感度排序.
- :func:`setup_pareto_front` — 非支配 setup 索引 (圈速 vs 胎耗代理).

所有函数确定性, 空列表输入返回 ``[]``. 批量路径经由
:meth:`SurrogateModel.predict_batch` (内部构建 ``(N, 37)`` 输入张量并做单次
``forward``); 本模块不在顶层 ``import torch`` (torch 由 surrogate 侧惰性加载).
批量路径抛异常时回退到逐条 :func:`predict_lap_time` / :func:`predict_full`.
"""

from __future__ import annotations

import math
from typing import Any

from f1opt.data.setup_schema import SETUP_FIELDS, CarSetup
from f1opt.model.surrogate import (
    _get_default_model,
    predict_full,
    predict_lap_time,
)

# 胎耗代理参考常数 (与 f1opt.model.optimizer 一致, 自然单位).
_TYRE_TEMP_REF = 90.0
_TYRE_TEMP_SPAN = 30.0
_SLIP_REF = 5.0


def _tire_wear_proxy_from_full(pred: dict[str, Any]) -> float:
    """从 predict_full 响应字典计算胎耗代理 (无量纲, 越大胎耗越快).

    = (tyre_temp - 90) / 30 + slip_angle / 5 + tyre_load_spread
    (胎温偏离基准 + 后轴滑移 + 四轮载荷离散).
    """
    resp = pred["responses"]
    temp = float(resp["tyre_temp"])
    slip = float(resp["slip_angle"])
    spread = float(resp["tyre_load_spread"])
    return (temp - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN + slip / _SLIP_REF + spread


def _batch_predict_full_vec(
    setups: list[CarSetup],
    track_id: str,
    driver_profile: Any,
) -> list[dict[str, Any]]:
    """单次 forward 批量富预测 (共享 track_id / driver).

    使用 :meth:`SurrogateModel.predict_batch` 内部构建 ``(N, 37)`` 张量并做
    单次 ``forward``. 失败时抛异常, 由调用方回退到逐条路径.
    """
    items: list[tuple[CarSetup, str, Any]] = [
        (s, track_id, driver_profile) for s in setups
    ]
    model = _get_default_model()
    return model.predict_batch(items)


def batch_predict_lap_times(
    setups: list[CarSetup],
    track_id: str,
    driver_profile: Any = None,
) -> list[float]:
    """批量圈速预测: 单次 forward, 返回 N 个 float.

    空列表返回 ``[]``. 批量路径异常时回退到逐条 :func:`predict_lap_time`.
    """
    if not setups:
        return []
    try:
        results = _batch_predict_full_vec(setups, track_id, driver_profile)
        return [float(r["lap_time"]) for r in results]
    except Exception:
        return [
            float(predict_lap_time(s, track_id, driver_profile)) for s in setups
        ]


def batch_predict_full(
    setups: list[CarSetup],
    track_id: str,
    driver_profile: Any = None,
) -> list[dict[str, Any]]:
    """批量富预测: 单次 forward, 返回 N 个 predict_full 风格字典.

    每个字典含 ``lap_time`` / ``sectors`` / ``responses`` / ``model_version``.
    空列表返回 ``[]``. 批量路径异常时回退到逐条 :func:`predict_full`.
    """
    if not setups:
        return []
    try:
        return _batch_predict_full_vec(setups, track_id, driver_profile)
    except Exception:
        return [predict_full(s, track_id, driver_profile) for s in setups]


def _snap_to_spec(value: float, spec: Any) -> int | float:
    """把原始值对齐到字段档位网格 (与 setup_schema._snap_to_step 等价)."""
    n_steps = round((value - spec.min) / spec.step)
    snapped = spec.min + n_steps * spec.step
    if snapped < spec.min:
        snapped = spec.min
    elif snapped > spec.max:
        snapped = spec.max
    if spec.kind == "int":
        return int(round(snapped))
    decimals = 0 if spec.step >= 1.0 else max(0, -int(math.floor(math.log10(spec.step))))
    return round(snapped, decimals)


def predict_lap_time_grid(
    setup_field: str,
    values: list[float],
    base_setup: CarSetup,
    track_id: str,
    driver_profile: Any = None,
) -> list[float]:
    """扫描单个调教字段 ``setup_field`` 取 ``values`` 各档, 返回对应圈速列表.

    其余字段保持 ``base_setup`` 不变; 每个值经档位对齐 (非法值 snap 到最近
    合法档). 空列表返回 ``[]``. 未知字段名抛 ``KeyError``. 内部使用批量路径
    单次 forward 评估全部档位.
    """
    if not values:
        return []
    if setup_field not in SETUP_FIELDS:
        raise KeyError(f"未知调教字段: {setup_field!r}")
    spec = SETUP_FIELDS[setup_field]
    setups = [
        base_setup.model_copy(update={setup_field: _snap_to_spec(float(v), spec)})
        for v in values
    ]
    return batch_predict_lap_times(setups, track_id, driver_profile)


def sensitivity_analysis(
    setup: CarSetup,
    track_id: str,
    driver_profile: Any = None,
    delta_steps: int = 1,
) -> dict[str, float]:
    """逐字段敏感度: 对 19 个字段各扰 ±delta_steps 档, 返回 ``{field: max_abs_delta}``.

    全部 38 个扰动 setup (19 字段 × ±) 在单次批量 forward 中评估; 基线圈速
    单独计算. 返回字典键为 19 个调教字段名, 值为该字段 +/- 扰动下圈速最大
    绝对变化 (>= 0, 用于字段敏感度排序). ``delta_steps`` 必须 >= 1.
    """
    if delta_steps < 1:
        raise ValueError("delta_steps 必须 >= 1")
    fields = list(SETUP_FIELDS.items())
    perturbed: list[CarSetup] = []
    for name, spec in fields:
        base_val = float(getattr(setup, name))
        for sign in (1, -1):
            new_val = _snap_to_spec(base_val + sign * delta_steps * spec.step, spec)
            perturbed.append(setup.model_copy(update={name: new_val}))
    times = batch_predict_lap_times(perturbed, track_id, driver_profile)
    base_time = float(predict_lap_time(setup, track_id, driver_profile))
    out: dict[str, float] = {}
    for fidx, (name, _spec) in enumerate(fields):
        plus_t = times[fidx * 2]
        minus_t = times[fidx * 2 + 1]
        out[name] = float(max(abs(plus_t - base_time), abs(minus_t - base_time)))
    return out


def setup_pareto_front(
    setups: list[CarSetup],
    track_id: str,
    driver_profile: Any = None,
    tire_wear_weight: float = 0.0,
) -> list[int]:
    """返回非支配 setup 的索引列表 (Pareto 前沿).

    - ``tire_wear_weight == 0``: 单目标圈速, 前沿 = 圈速最小的索引集合
      (无其他 setup 圈速严格更小).
    - ``tire_wear_weight > 0``: 双目标 (圈速, 胎耗代理) Pareto 前沿, 二者均
      最小化; 胎耗代理取自 :func:`predict_full` 响应 (胎温偏离 + 滑移 + 载荷
      离散).

    返回索引按升序排列; 空列表输入返回 ``[]``.
    """
    if not setups:
        return []
    fulls = batch_predict_full(setups, track_id, driver_profile)
    lap = [float(f["lap_time"]) for f in fulls]
    n = len(setups)
    if tire_wear_weight > 0.0:
        tire = [_tire_wear_proxy_from_full(f) for f in fulls]
        objs: list[tuple[float, ...]] = [(lap[i], tire[i]) for i in range(n)]
    else:
        objs = [(lap[i],) for i in range(n)]
    front: list[int] = []
    for i in range(n):
        oi = objs[i]
        dominated = False
        for j in range(n):
            if j == i:
                continue
            oj = objs[j]
            # j 支配 i: oj 各维 <= oi 且至少一维严格 <.
            if all(oj[k] <= oi[k] for k in range(len(oj))) and any(
                oj[k] < oi[k] for k in range(len(oj))
            ):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front)
