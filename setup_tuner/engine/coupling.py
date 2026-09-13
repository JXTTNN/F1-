"""耦合矩阵 C（9 诊断维度 × 20 调教参数）。

C[d][p] = 诊断维度 d 每 +1 单位 → 参数 p 的调整量（带符号档位数）。

**结构**：9×20 矩阵（诊断维度 × 参数，适配 ``SetupDelta = Dx × C`` 的行向量乘序）。
**元素三要素**：
    - 符号（方向）：``+`` 表示 d 需求增加时 p 应增大；``-`` 表示应减小。
    - 强度（幅度）：每 +1 诊断单位的参数档位调整量。
    - 出处（source）：每个非零元素附带出处键。

参数对齐 EA F1 游戏（F1 25，UDP format=2026）CarSetups 包真实字段：
前翼/后翼/差速器×2/倾角×2/束角×2/悬挂×2/防倾杆×2/行驶高度×2/刹车×2/胎压×4。

**工程硬约束（构建期校验，``validate_matrix``）**：
    1. 每个参数（列）至少含 1 个非零元素；
    2. 每个诊断维度（行）至少含 1 个非零元素；
    3. 矩阵非零密度 ≥ 30%。

出处枚举（内部标签，标识调教机理来源域）：
    - ``F1_SETUP_DOMAIN``：F1 调教领域机理
    - ``F1_UDP_SPEC``     ：F1 UDP 规范字段映射
    - ``PIRELLI_TYRE``    ：Pirelli 轮胎工作窗口
"""

from __future__ import annotations

from dataclasses import dataclass

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

from .diagnostic import DIAG_DIMS

# ---------------------------------------------------------------------------
# 出处键
# ---------------------------------------------------------------------------
F1_SETUP_DOMAIN = "F1_SETUP_DOMAIN"  # F1 调教领域机理
F1_UDP_SPEC = "F1_UDP_SPEC"          # F1 UDP 规范字段映射
PIRELLI_TYRE = "PIRELLI_TYRE"        # Pirelli 轮胎工作窗口

VALID_SOURCES: frozenset[str] = frozenset({F1_SETUP_DOMAIN, F1_UDP_SPEC, PIRELLI_TYRE})


@dataclass(frozen=True, slots=True)
class CouplingCell:
    """耦合矩阵单个非零元素。"""

    param: str
    diag: str
    sign: int
    magnitude: float
    source: str

    @property
    def value(self) -> float:
        return self.sign * self.magnitude


# ---------------------------------------------------------------------------
# 20 个参数名（固定顺序，从 ALL_SETUP_FIELDS 读取）
# ---------------------------------------------------------------------------
PARAM_NAMES: list[str] = [f.name for f in ALL_SETUP_FIELDS]
_PARAM_INDEX: dict[str, int] = {name: i for i, name in enumerate(PARAM_NAMES)}


# ---------------------------------------------------------------------------
# 原始矩阵定义（9 行 × 20 列）
# 每个诊断维度一行，值为 {param_name: (sign, magnitude, source)}
# ---------------------------------------------------------------------------
G = F1_SETUP_DOMAIN
U = F1_UDP_SPEC
P = PIRELLI_TYRE

_Cell = tuple[int, float, str]

_RAW_MATRIX: dict[str, dict[str, _Cell]] = {
    # 1. front_grip_req（前轴抓地不足）
    "front_grip_req": {
        "front_wing":           (+1, 1.5, G),  # 增前翼下压力
        "rear_wing":            (+1, 0.5, G),  # 略增后翼平衡空力
        "on_throttle_diff":     (-1, 0.2, G),  # 降加速差速减前轴推力损失
        "off_throttle_diff":    (+1, 0.2, G),  # 收油差速略调稳前轴入弯
        "front_camber":         (-1, 0.3, G),  # 更负外倾增弯中前轮接地
        "front_toe":            (+1, 0.1, G),  # 微增前束稳前轴指向
        "front_suspension":     (-1, 0.2, G),  # 软前悬挂增机械抓地
        "front_anti_roll_bar":  (-1, 0.3, G),  # 软前防倾杆让前轴独立抓地
        "rear_anti_roll_bar":   (+1, 0.4, G),  # 增后防倾杆转移侧倾刚度至后轴
        "front_ride_height":    (-1, 0.2, G),  # 降前离地增前轴下压力中心
        "brake_bias":           (+1, 0.2, G),  # 略前移刹车配比
        "front_left_tyre_pressure":  (+1, 0.3, P),  # 增前左胎压增弯中响应
        "front_right_tyre_pressure": (+1, 0.3, P),  # 增前右胎压增弯中响应
    },

    # 2. rear_grip_req（后轴抓地不足）
    "rear_grip_req": {
        "front_wing":           (+1, 0.5, G),
        "rear_wing":            (+1, 1.5, G),  # 增后翼下压力
        "on_throttle_diff":     (-1, 0.3, G),  # 降加速差速减后轮打滑
        "off_throttle_diff":    (+1, 0.2, G),  # 收油差速影响后轴稳定性
        "rear_camber":          (-1, 0.3, G),  # 更负后外倾增后轮弯中接地
        "rear_toe":             (+1, 0.1, G),
        "rear_suspension":      (-1, 0.2, G),  # 软后悬挂增后机械抓地
        "rear_anti_roll_bar":   (-1, 0.6, G),  # 降后防倾杆让后轴独立抓地
        "rear_ride_height":     (-1, 0.2, G),
        "brake_bias":           (-1, 0.2, G),  # 略后移刹车配比
        "rear_left_tyre_pressure":  (+1, 0.3, P),
        "rear_right_tyre_pressure": (+1, 0.3, P),
    },

    # 3. turnin_req（入弯响应不足）
    "turnin_req": {
        "front_wing":           (+1, 0.6, G),  # 增前翼增前轴指向
        "on_throttle_diff":     (-1, 0.1, G),  # 降加速差速增入弯指向
        "off_throttle_diff":    (+1, 0.3, G),  # 增收油差速增入弯响应
        "front_camber":         (-1, 0.4, G),  # 更负前外倾增前轮指向
        "front_toe":            (+1, 0.2, G),  # 增前束增入弯指向
        "front_suspension":     (-1, 0.2, G),  # 软前悬增入弯响应
        "front_anti_roll_bar":  (-1, 0.2, G),  # 软前防倾杆增入弯响应
        "rear_anti_roll_bar":   (+1, 0.8, G),  # 增后防倾杆让前轴更快响应
        "brake_bias":           (+1, 0.1, G),  # 略前移刹车配比助入弯
        "front_left_tyre_pressure":  (+1, 0.2, P),
        "front_right_tyre_pressure": (+1, 0.2, P),
    },

    # 4. hi_speed_stab_req（高速/弯中不稳定）
    "hi_speed_stab_req": {
        "front_wing":           (+1, 0.4, G),
        "rear_wing":            (+1, 0.6, G),
        "on_throttle_diff":     (+1, 0.1, G),  # 增加速差速稳高速
        "off_throttle_diff":    (+1, 0.1, G),  # 增收油差速稳高速
        "rear_camber":          (-1, 0.2, G),
        "front_toe":            (+1, 0.1, G),
        "rear_toe":             (+1, 0.2, G),
        "front_suspension":     (+1, 0.3, G),  # 硬前悬挂稳高速
        "rear_suspension":      (+1, 0.3, G),  # 硬后悬挂稳高速
        "front_anti_roll_bar":  (+1, 0.3, G),
        "rear_anti_roll_bar":   (+1, 0.3, G),
        "front_ride_height":    (+1, 0.2, G),
        "rear_ride_height":     (+1, 0.2, G),
        "front_left_tyre_pressure":  (+1, 0.2, P),
        "front_right_tyre_pressure": (+1, 0.2, P),
        "rear_left_tyre_pressure":   (+1, 0.2, P),
        "rear_right_tyre_pressure":  (+1, 0.2, P),
    },

    # 5. brake_stab_req（制动不稳定/易锁死）
    "brake_stab_req": {
        "on_throttle_diff":     (+1, 0.1, G),  # 差速影响制动稳定性
        "off_throttle_diff":    (+1, 0.2, G),  # 收油差速影响制动稳定性
        "front_camber":         (-1, 0.1, G),  # 外倾影响制动时轮胎接地
        "rear_camber":          (-1, 0.1, G),  # 后外倾影响制动时后轮接地
        "front_suspension":     (+1, 0.2, G),  # 硬前悬减制动俯冲
        "rear_suspension":      (+1, 0.2, G),  # 硬后悬减制动俯冲
        "front_anti_roll_bar":  (+1, 0.2, G),  # 防倾杆影响制动侧倾刚度
        "rear_anti_roll_bar":   (+1, 0.2, G),
        "brake_pressure":       (-1, 0.8, G),  # 降刹车压力防锁死
        "brake_bias":           (-1, 0.3, G),  # 略后移刹车配比
        "front_left_tyre_pressure":  (+1, 0.2, P),
        "front_right_tyre_pressure": (+1, 0.2, P),
        "rear_left_tyre_pressure":   (+1, 0.2, P),  # 后胎压也影响制动稳定性
        "rear_right_tyre_pressure":  (+1, 0.2, P),
    },

    # 6. brake_power_req（制动力不足）
    "brake_power_req": {
        "front_suspension":     (+1, 0.2, G),  # 硬前悬增制动重量转移效率
        "rear_suspension":      (+1, 0.2, G),  # 硬后悬增制动重量转移效率
        "brake_pressure":       (+1, 2.0, G),  # 增刹车压力
        "brake_bias":           (+1, 0.3, G),  # 略前移刹车配比
        "front_left_tyre_pressure":  (+1, 0.3, P),  # 胎压影响制动接触面积
        "front_right_tyre_pressure": (+1, 0.3, P),
        "rear_left_tyre_pressure":   (+1, 0.2, P),  # 后胎压也影响制动力
        "rear_right_tyre_pressure":  (+1, 0.2, P),
    },

    # 7. exit_traction_req（出弯牵引不足）
    "exit_traction_req": {
        "front_wing":           (+1, 0.2, G),  # 前翼影响出弯平衡
        "rear_wing":            (+1, 0.3, G),  # 增后翼增出弯后轴下压力
        "on_throttle_diff":     (-1, 0.5, G),  # 降加速差速减后轮出弯打滑
        "off_throttle_diff":    (+1, 0.1, G),  # 收油差速影响出弯过渡
        "rear_camber":          (-1, 0.2, G),
        "rear_toe":             (+1, 0.1, G),
        "rear_suspension":      (-1, 0.3, G),
        "rear_anti_roll_bar":   (-1, 0.4, G),
        "rear_ride_height":     (-1, 0.1, G),
        "rear_left_tyre_pressure":  (-1, 0.2, P),  # 降后胎压增出弯接触面积
        "rear_right_tyre_pressure": (-1, 0.2, P),
    },

    # 8. tyre_life_req（胎耗过高）
    "tyre_life_req": {
        "front_wing":           (-1, 0.2, G),
        "rear_wing":            (-1, 0.2, G),
        "on_throttle_diff":     (+1, 0.2, G),
        "off_throttle_diff":    (+1, 0.2, G),
        "front_camber":         (+1, 0.2, G),  # 略减外倾负值减边缘磨耗
        "rear_camber":          (+1, 0.2, G),
        "front_suspension":     (+1, 0.2, G),
        "rear_suspension":      (+1, 0.2, G),
        "brake_pressure":       (-1, 0.3, G),
        "front_left_tyre_pressure":  (+1, 0.2, P),
        "front_right_tyre_pressure": (+1, 0.2, P),
        "rear_left_tyre_pressure":   (+1, 0.2, P),
        "rear_right_tyre_pressure":  (+1, 0.2, P),
    },

    # 9. ride_height_req（底盘离地不足/刮底）
    "ride_height_req": {
        "front_wing":           (-1, 0.2, G),  # 降前翼减下压力压迫
        "rear_wing":            (-1, 0.2, G),  # 降后翼减下压力压迫
        "front_suspension":     (+1, 0.5, G),  # 硬前悬防刮底
        "rear_suspension":      (+1, 0.5, G),  # 硬后悬防刮底
        "front_anti_roll_bar":  (+1, 0.3, G),  # 增防倾杆减侧倾刮底
        "rear_anti_roll_bar":   (+1, 0.3, G),
        "front_ride_height":    (+1, 1.0, G),  # 增高前离地
        "rear_ride_height":     (+1, 1.0, G),
        "front_left_tyre_pressure":  (+1, 0.2, P),
        "front_right_tyre_pressure": (+1, 0.2, P),
        "rear_left_tyre_pressure":   (+1, 0.2, P),
        "rear_right_tyre_pressure":  (+1, 0.2, P),
    },
}


# ---------------------------------------------------------------------------
# 构建内存态耦合矩阵
# ---------------------------------------------------------------------------
def _build_matrix() -> dict[str, dict[str, CouplingCell | None]]:
    matrix: dict[str, dict[str, CouplingCell | None]] = {}
    for diag in DIAG_DIMS:
        row: dict[str, CouplingCell | None] = {}
        raw_row = _RAW_MATRIX.get(diag, {})
        for param in PARAM_NAMES:
            cell = raw_row.get(param)
            if cell is None:
                row[param] = None
            else:
                sign, magnitude, source = cell
                if source not in VALID_SOURCES:
                    raise AssertionError(
                        f"耦合矩阵出处非法: diag={diag!r} param={param!r} source={source!r}",
                    )
                if sign not in (+1, -1):
                    raise AssertionError(
                        f"耦合矩阵符号非法: diag={diag!r} param={param!r} sign={sign!r}",
                    )
                if magnitude <= 0.0:
                    raise AssertionError(
                        f"耦合矩阵幅度非法: diag={diag!r} param={param!r} magnitude={magnitude!r}",
                    )
                row[param] = CouplingCell(
                    param=param, diag=diag, sign=sign, magnitude=float(magnitude), source=source,
                )
        matrix[diag] = row
    return matrix


COUPLING_MATRIX: dict[str, dict[str, CouplingCell | None]] = _build_matrix()


# 预计算：参数 → 非零耦合单元列表
_PARAM_NONZERO_CELLS: dict[str, list[CouplingCell]] = {
    param: [
        COUPLING_MATRIX[diag][param]
        for diag in DIAG_DIMS
        if COUPLING_MATRIX[diag][param] is not None
    ]
    for param in PARAM_NAMES
}


def nonzero_cells_for_param_cached(param: str) -> list[CouplingCell]:
    return _PARAM_NONZERO_CELLS.get(param, [])


def get_coupling(diag: str, param: str) -> CouplingCell | None:
    row = COUPLING_MATRIX.get(diag)
    if row is None:
        return None
    return row.get(param)


def get_row(diag: str) -> dict[str, CouplingCell | None]:
    return COUPLING_MATRIX.get(diag, dict.fromkeys(PARAM_NAMES))


def get_column(param: str) -> dict[str, CouplingCell | None]:
    return {diag: COUPLING_MATRIX[diag].get(param) for diag in DIAG_DIMS}


def nonzero_cells_for_diag(diag: str) -> list[CouplingCell]:
    row = COUPLING_MATRIX.get(diag, {})
    return [cell for cell in row.values() if cell is not None]


def nonzero_cells_for_param(param: str) -> list[CouplingCell]:
    result: list[CouplingCell] = []
    for diag in DIAG_DIMS:
        cell = COUPLING_MATRIX[diag].get(param)
        if cell is not None:
            result.append(cell)
    return result


def validate_matrix() -> None:
    """构建期校验耦合矩阵 3 条硬约束。"""
    assert len(COUPLING_MATRIX) == len(DIAG_DIMS), (
        f"耦合矩阵行数 {len(COUPLING_MATRIX)} != 诊断维度数 {len(DIAG_DIMS)}"
    )
    for diag in DIAG_DIMS:
        row = COUPLING_MATRIX[diag]
        assert len(row) == len(PARAM_NAMES), (
            f"耦合矩阵 {diag!r} 行列数 {len(row)} != 参数数 {len(PARAM_NAMES)}"
        )

    empty_params: list[str] = []
    for param in PARAM_NAMES:
        nonzero_count = sum(
            1 for diag in DIAG_DIMS if COUPLING_MATRIX[diag][param] is not None
        )
        if nonzero_count == 0:
            empty_params.append(param)
    assert not empty_params, (
        f"约束1失败：以下参数在所有诊断维度上均为零（被遗忘）: {empty_params}"
    )

    empty_diags: list[str] = []
    for diag in DIAG_DIMS:
        nonzero_count = sum(
            1 for cell in COUPLING_MATRIX[diag].values() if cell is not None
        )
        if nonzero_count == 0:
            empty_diags.append(diag)
    assert not empty_diags, (
        f"约束2失败：以下诊断维度在所有参数上均为零（无效维度）: {empty_diags}"
    )

    total_cells = len(DIAG_DIMS) * len(PARAM_NAMES)
    nonzero_total = sum(
        1
        for diag in DIAG_DIMS
        for cell in COUPLING_MATRIX[diag].values()
        if cell is not None
    )
    density = nonzero_total / total_cells
    assert density >= 0.30, (
        f"约束3失败：矩阵非零密度 {density:.2%} < 30% "
        f"({nonzero_total}/{total_cells})"
    )


def matrix_stats() -> dict[str, object]:
    total_cells = len(DIAG_DIMS) * len(PARAM_NAMES)
    nonzero_total = sum(
        1
        for diag in DIAG_DIMS
        for cell in COUPLING_MATRIX[diag].values()
        if cell is not None
    )
    return {
        "diag_dims": len(DIAG_DIMS),
        "params": len(PARAM_NAMES),
        "total_cells": total_cells,
        "nonzero_cells": nonzero_total,
        "density": nonzero_total / total_cells,
        "sources_used": sorted(
            {
                cell.source
                for diag in DIAG_DIMS
                for cell in COUPLING_MATRIX[diag].values()
                if cell is not None
            },
        ),
    }


validate_matrix()