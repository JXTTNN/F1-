"""耦合矩阵 C（9 诊断维度 × 23 调教参数）。

C[d][p] = 诊断维度 d 每 +1 单位 → 参数 p 的调整量（带符号档位数）。

**结构**：9×23 矩阵（诊断维度 × 参数，适配 ``SetupDelta = Dx × C`` 的行向量乘序）。
**元素三要素**：
    - 符号（方向）：``+`` 表示 d 需求增加时 p 应增大；``-`` 表示应减小。
    - 强度（幅度）：每 +1 诊断单位的参数档位调整量。
    - 出处（source）：每个非零元素附带官方出处键。

**工程硬约束（构建期校验，``validate_matrix``）**：
    1. 每个参数（列）至少含 1 个非零元素（无被遗忘的参数）；
    2. 每个诊断维度（行）至少含 1 个非零元素（无无效维度）；
    3. 矩阵非零密度 ≥ 30%（保证单症状即可波及大部分参数类别）。

出处枚举（对齐 FR-RPT-02）：
    - ``EA_SETUP_GUIDE``：EA F1 2026 官方调教指南
    - ``EA_UDP_2026``    ：EA F1 2026 官方 UDP 规范
    - ``PIRELLI``        ：Pirelli 官方轮胎数据

本模块为纯静态数据 + 纯函数，零 IO、零随机，满足 FR-ENG-05 / FR-NFR-R1。
"""

from __future__ import annotations

from dataclasses import dataclass

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

from .diagnostic import DIAG_DIMS


# ---------------------------------------------------------------------------
# 官方出处键（受限枚举，对齐 FR-RPT-02）
# ---------------------------------------------------------------------------
EA_SETUP_GUIDE = "EA_SETUP_GUIDE"  # EA F1 2026 官方调教指南
EA_UDP_2026 = "EA_UDP_2026"        # EA F1 2026 官方 UDP 规范
PIRELLI = "PIRELLI"                # Pirelli 官方轮胎数据

# 合法出处枚举（构建期校验用）
VALID_SOURCES: frozenset[str] = frozenset({EA_SETUP_GUIDE, EA_UDP_2026, PIRELLI})


# ---------------------------------------------------------------------------
# 耦合矩阵单元
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CouplingCell:
    """耦合矩阵单个非零元素。

    Attributes:
        param: 参数名（对应 setup_tuner.domain.setup 的字段名）。
        diag: 诊断维度 key（对应 setup_tuner.engine.diagnostic.DIAG_DIMS）。
        sign: 方向符号，+1 表示 d 需求增加时 p 应增大，-1 表示应减小。
        magnitude: 每 +1 诊断单位的参数档位调整量（非负浮点）。
        source: 官方出处键（VALID_SOURCES 之一）。
    """

    param: str
    diag: str
    sign: int
    magnitude: float
    source: str

    @property
    def value(self) -> float:
        """带符号的耦合值 = sign × magnitude。"""
        return self.sign * self.magnitude


# ---------------------------------------------------------------------------
# 23 个参数名（固定顺序，与 setup_tuner.domain.setup.ALL_SETUP_FIELDS 一致）
# ---------------------------------------------------------------------------
PARAM_NAMES: list[str] = [f.name for f in ALL_SETUP_FIELDS]
_PARAM_INDEX: dict[str, int] = {name: i for i, name in enumerate(PARAM_NAMES)}


# ---------------------------------------------------------------------------
# 原始矩阵定义（9 行 × 23 列）
# 每个诊断维度一行，值为 {param_name: (sign, magnitude, source)}；
# 未列出的参数视为零格（None）。
#
# 工程理由逐行注释：
# - 符号遵循「d 需求 +1 → p 应如何调整」的官方调教机理；
# - 幅度反映该参数对该诊断维度的响应灵敏度（档位数/诊断单位）；
# - 出处标注每格的官方依据。
# ---------------------------------------------------------------------------
# 简写：G=EA_SETUP_GUIDE, U=EA_UDP_2026, P=PIRELLI
G = EA_SETUP_GUIDE
U = EA_UDP_2026
P = PIRELLI

# 单元类型：(sign, magnitude, source)
_Cell = tuple[int, float, str]

_RAW_MATRIX: dict[str, dict[str, _Cell]] = {
    # =====================================================================
    # 1. front_grip_req（前轴抓地不足，需增强前轴抓地）
    #    机理：增前翼下压力 / 更负前外倾增弯中接地 / 软前弹簧增机械抓地 /
    #          增后ARB转移侧倾刚度至后轴让前轴相对软 / 降前离地增前轴下压力中心
    #    片段锚点（design 2.7.3）：
    #       front_wing +1.5, rear_wing +0.5, front_camber -0.3,
    #       rear_arb +0.4, brake_pressure 0, front_tyre_pressure +0.3,
    #       ride_height -0.2
    # =====================================================================
    "front_grip_req": {
        "front_wing":          (+1, 1.5, G),  # 增前翼下压力（片段锚点）
        "rear_wing":           (+1, 0.5, G),  # 略增后翼平衡空力（片段锚点）
        "active_aero_z":       (+1, 0.3, G),  # 弯道空力偏前轴
        "off_throttle_diff":   (+1, 0.2, G),  # 收油差速略调稳前轴入弯
        "front_camber":        (-1, 0.3, G),  # 更负外倾增弯中前轮接地（片段锚点）
        "front_toe":           (+1, 0.1, G),  # 微增前束稳前轴指向
        "front_spring":        (-1, 0.2, G),  # 软前弹簧增机械抓地
        "front_anti_roll_bar": (-1, 0.3, G),  # 软前ARB让前轴更独立抓地
        "rear_anti_roll_bar":  (+1, 0.4, G),  # 增后ARB转移侧倾刚度至后轴（片段锚点）
        "front_ride_height":   (-1, 0.2, G),  # 降前离地增前轴下压力中心（片段锚点 ride_height -0.2）
        "brake_bias":          (+1, 0.2, G),  # 略前移刹车配比稳前轴入弯负荷
        "front_tyre_pressure": (+1, 0.3, P),  # 增前胎压增弯中响应（片段锚点，Pirelli 工作窗口）
        "ballast":             (+1, 0.1, G),  # 略前移配重增前轴负荷
    },

    # =====================================================================
    # 2. rear_grip_req（后轴抓地不足，需增强后轴抓地）
    #    机理：增后翼下压力 / 更负后外倾 / 软后弹簧 / 降后ARB让后轴独立抓地 /
    #          降加速差速减后轮打滑 / 降后离地增后轴下压力中心
    #    片段锚点：
    #       front_wing +0.5, rear_wing +1.5, front_camber 0,
    #       rear_arb -0.6, brake_pressure 0, front_tyre_pressure 0,
    #       ride_height -0.2
    # =====================================================================
    "rear_grip_req": {
        "front_wing":          (+1, 0.5, G),  # 略增前翼以平衡（片段锚点）
        "rear_wing":           (+1, 1.5, G),  # 增后翼下压力（片段锚点）
        "active_aero_z":       (+1, 0.3, G),  # 弯道空力增下压力
        "on_throttle_diff":    (-1, 0.3, G),  # 降加速差速减后轮打滑
        "rear_camber":         (-1, 0.3, G),  # 更负后外倾增后轮弯中接地
        "rear_toe":            (+1, 0.1, G),  # 微增后束稳后轴
        "rear_spring":         (-1, 0.2, G),  # 软后弹簧增后机械抓地
        "rear_anti_roll_bar":  (-1, 0.6, G),  # 降后ARB让后轴更独立抓地（片段锚点）
        "rear_ride_height":    (-1, 0.2, G),  # 降后离地增后轴下压力中心（片段锚点 ride_height -0.2）
        "brake_bias":          (-1, 0.2, G),  # 略后移刹车配比减后轴入弯负荷
        "rear_tyre_pressure":  (+1, 0.3, P),  # 增后胎压增弯中响应（Pirelli 工作窗口）
        "engine_braking":      (+1, 0.2, G),  # 增发动机制动稳后轴入弯
        "ballast":             (-1, 0.1, G),  # 略后移配重增后轴负荷
    },

    # =====================================================================
    # 3. turnin_req（入弯响应不足，需提升入弯指向）
    #    机理：增前翼增前轴指向 / 更负前外倾增前轮指向 / 增前束 /
    #          增后ARB转移侧倾刚度至后轴让前轴更快响应 / 增收油差速增入弯响应
    #    片段锚点：
    #       front_wing +0.6, rear_wing 0, front_camber -0.4,
    #       rear_arb +0.8, brake_pressure 0, front_tyre_pressure +0.2,
    #       ride_height 0
    # =====================================================================
    "turnin_req": {
        "front_wing":          (+1, 0.6, G),  # 增前翼增前轴指向（片段锚点）
        "active_aero_z":       (+1, 0.2, G),  # 弯道空力偏前轴
        "off_throttle_diff":   (+1, 0.3, G),  # 增收油差速增入弯响应
        "front_camber":        (-1, 0.4, G),  # 更负前外倾增前轮指向（片段锚点）
        "front_toe":           (+1, 0.2, G),  # 增前束增入弯指向
        "rear_anti_roll_bar":  (+1, 0.8, G),  # 增后ARB转移侧倾刚度至后轴让前轴更快响应（片段锚点）
        "front_tyre_pressure": (+1, 0.2, P),  # 增前胎压增入弯响应（片段锚点，Pirelli）
        "ballast":             (+1, 0.1, G),  # 略前移配重增前轴指向
    },

    # =====================================================================
    # 4. hi_speed_stab_req（高速/弯中不稳定，需增强高速稳定性）
    #    机理：增前后翼下压力 / 硬弹簧 / 增ARB抑制侧倾 / 增阻尼 /
    #          增束角稳指向 / 增胎压稳高速
    # =====================================================================
    "hi_speed_stab_req": {
        "front_wing":          (+1, 0.4, G),  # 增前翼增高速下压力
        "rear_wing":           (+1, 0.6, G),  # 增后翼增高速后轴稳定
        "active_aero_z":       (+1, 0.2, G),  # 弯道空力增下压力
        "active_aero_x":       (+1, 0.3, G),  # 直道空力增下压力稳高速
        "rear_camber":         (-1, 0.2, G),  # 更负后外倾稳后轴高速
        "front_toe":           (+1, 0.1, G),  # 微增前束稳高速指向
        "rear_toe":            (+1, 0.2, G),  # 增后束稳高速后轴
        "front_spring":        (+1, 0.3, G),  # 硬前弹簧稳高速
        "rear_spring":         (+1, 0.3, G),  # 硬后弹簧稳高速
        "front_anti_roll_bar": (+1, 0.3, G),  # 增前ARB稳高速侧倾
        "rear_anti_roll_bar":  (+1, 0.3, G),  # 增后ARB稳高速侧倾
        "front_ride_height":   (+1, 0.2, G),  # 略增高前离地稳高速空力平衡
        "rear_ride_height":    (+1, 0.2, G),  # 略增高后离地稳高速空力平衡
        "damping":             (+1, 0.4, G),  # 增阻尼稳高速
        "front_tyre_pressure": (+1, 0.2, P),  # 增前胎压稳高速（Pirelli）
        "rear_tyre_pressure":  (+1, 0.2, P),  # 增后胎压稳高速（Pirelli）
    },

    # =====================================================================
    # 5. brake_stab_req（制动不稳定/易锁死，需增强制动稳定性）
    #    机理：降刹车压力防锁死 / 略后移刹车配比减前轴锁死 /
    #          增发动机制动辅助减速 / 增阻尼稳制动俯仰 / 增前胎压防锁死
    # =====================================================================
    "brake_stab_req": {
        "off_throttle_diff":   (+1, 0.2, G),  # 增收油差速稳制动
        "damping":             (+1, 0.3, G),  # 增阻尼稳制动俯仰
        "brake_pressure":      (-1, 0.8, G),  # 降刹车压力防锁死
        "brake_bias":          (-1, 0.3, G),  # 略后移刹车配比减前轴锁死
        "front_tyre_pressure": (+1, 0.2, P),  # 增前胎压防锁死（Pirelli）
        "engine_braking":      (+1, 0.3, G),  # 增发动机制动稳制动
    },

    # =====================================================================
    # 6. brake_power_req（制动力不足，需增大制动力）
    #    机理：增刹车压力 / 略前移刹车配比增前轴制动 / 增发动机制动辅助减速
    #    片段锚点：brake_pressure +2.0
    # =====================================================================
    "brake_power_req": {
        "brake_pressure":      (+1, 2.0, G),  # 增刹车压力（片段锚点）
        "brake_bias":          (+1, 0.3, G),  # 略前移刹车配比增前轴制动
        "engine_braking":      (+1, 0.2, G),  # 增发动机制动辅助减速
    },

    # =====================================================================
    # 7. exit_traction_req（出弯牵引不足，需增强出弯牵引）
    #    机理：略增后翼增后轴出弯下压力 / 降加速差速减后轮出弯打滑 /
    #          更负后外倾 / 软后弹簧 / 降后ARB让后轴独立抓地 /
    #          降后胎压增后轮出弯接地面积 / 略后移配重
    # =====================================================================
    "exit_traction_req": {
        "rear_wing":           (+1, 0.3, G),  # 略增后翼增后轴出弯下压力
        "on_throttle_diff":    (-1, 0.5, G),  # 降加速差速减后轮出弯打滑
        "rear_camber":         (-1, 0.2, G),  # 更负后外倾增后轮出弯接地
        "rear_toe":            (+1, 0.1, G),  # 微增后束稳出弯
        "rear_spring":         (-1, 0.3, G),  # 软后弹簧增后机械抓地
        "rear_anti_roll_bar":  (-1, 0.4, G),  # 降后ARB让后轴更独立抓地
        "rear_ride_height":    (-1, 0.1, G),  # 略降后离地增后轴下压力中心
        "damping":             (+1, 0.2, G),  # 增阻尼控出弯牵引
        "rear_tyre_pressure":  (-1, 0.2, P),  # 降后胎压增后轮出弯接地面积（Pirelli）
        "ballast":             (-1, 0.1, G),  # 略后移配重增后轴出弯牵引
    },

    # =====================================================================
    # 8. tyre_life_req（胎耗过高，需延长轮胎寿命）
    #    机理：略降前后翼减胎负荷 / 略增加速收油差速减滑移磨胎 /
    #          略减外倾负值减边缘磨耗 / 略硬弹簧减过载磨胎 /
    #          增阻尼减胎面振荡磨耗 / 略降刹车压力减制动磨胎 /
    #          增胎压至 Pirelli 工作窗口减过磨
    # =====================================================================
    "tyre_life_req": {
        "front_wing":          (-1, 0.2, G),  # 略降前翼减前胎负荷
        "rear_wing":           (-1, 0.2, G),  # 略降后翼减后胎负荷
        "on_throttle_diff":    (+1, 0.2, G),  # 略增加速差速减后轮滑移磨胎
        "off_throttle_diff":   (+1, 0.2, G),  # 略增收油差速减滑移
        "front_camber":        (+1, 0.2, G),  # 略减前外倾负值减前胎边缘磨耗
        "rear_camber":         (+1, 0.2, G),  # 略减后外倾负值减后胎边缘磨耗
        "front_spring":        (+1, 0.2, G),  # 略硬前弹簧减前胎过载磨耗
        "rear_spring":         (+1, 0.2, G),  # 略硬后弹簧减后胎过载磨耗
        "damping":             (+1, 0.3, G),  # 增阻尼减胎面振荡磨耗
        "brake_pressure":      (-1, 0.3, G),  # 略降刹车压力减制动磨胎
        "front_tyre_pressure": (+1, 0.2, P),  # 增前胎压至工作窗口减过磨（Pirelli）
        "rear_tyre_pressure":  (+1, 0.2, P),  # 增后胎压至工作窗口（Pirelli）
    },

    # =====================================================================
    # 9. ride_height_req（底盘离地不足/刮底，需增高底盘离地）
    #    机理：增高前后行驶高度 / 硬弹簧防刮底 / 增阻尼防刮底 /
    #          增胎压增高底盘
    #    片段锚点：ride_height +1.0
    # =====================================================================
    "ride_height_req": {
        "front_spring":        (+1, 0.5, G),  # 硬前弹簧防刮底
        "rear_spring":         (+1, 0.5, G),  # 硬后弹簧防刮底
        "front_ride_height":   (+1, 1.0, G),  # 增高前离地（片段锚点 ride_height +1.0）
        "rear_ride_height":    (+1, 1.0, G),  # 增高后离地
        "damping":             (+1, 0.3, G),  # 增阻尼防刮底
        "front_tyre_pressure": (+1, 0.2, P),  # 增前胎压增高底盘（Pirelli）
        "rear_tyre_pressure":  (+1, 0.2, P),  # 增后胎压增高底盘（Pirelli）
    },
}


# ---------------------------------------------------------------------------
# 构建内存态耦合矩阵：dict[diag][param] -> CouplingCell | None
# ---------------------------------------------------------------------------
def _build_matrix() -> dict[str, dict[str, CouplingCell | None]]:
    """从 _RAW_MATRIX 构建完整的 9×23 CouplingCell 矩阵（含 None 零格）。"""
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
                # 构建期校验：出处合法 + sign ∈ {+1,-1} + magnitude > 0
                if source not in VALID_SOURCES:
                    raise AssertionError(
                        f"耦合矩阵出处非法: diag={diag!r} param={param!r} source={source!r}"
                    )
                if sign not in (+1, -1):
                    raise AssertionError(
                        f"耦合矩阵符号非法: diag={diag!r} param={param!r} sign={sign!r}"
                    )
                if magnitude <= 0.0:
                    raise AssertionError(
                        f"耦合矩阵幅度非法: diag={diag!r} param={param!r} magnitude={magnitude!r}"
                    )
                row[param] = CouplingCell(
                    param=param, diag=diag, sign=sign, magnitude=float(magnitude), source=source
                )
        matrix[diag] = row
    return matrix


COUPLING_MATRIX: dict[str, dict[str, CouplingCell | None]] = _build_matrix()


# ---------------------------------------------------------------------------
# 查询接口
# ---------------------------------------------------------------------------
def get_coupling(diag: str, param: str) -> CouplingCell | None:
    """查询单个耦合单元。

    Args:
        diag: 诊断维度 key。
        param: 参数名。

    Returns:
        对应的 CouplingCell；若该格为零或维度/参数不存在则返回 None。
    """
    row = COUPLING_MATRIX.get(diag)
    if row is None:
        return None
    return row.get(param)


def get_row(diag: str) -> dict[str, CouplingCell | None]:
    """查询某诊断维度对应的整行（23 个参数的耦合单元）。"""
    return COUPLING_MATRIX.get(diag, {p: None for p in PARAM_NAMES})


def get_column(param: str) -> dict[str, CouplingCell | None]:
    """查询某参数对应的整列（9 个诊断维度的耦合单元）。"""
    return {diag: COUPLING_MATRIX[diag].get(param) for diag in DIAG_DIMS}


def nonzero_cells_for_diag(diag: str) -> list[CouplingCell]:
    """返回某诊断维度下所有非零耦合单元（用于报告联动说明）。"""
    row = COUPLING_MATRIX.get(diag, {})
    return [cell for cell in row.values() if cell is not None]


def nonzero_cells_for_param(param: str) -> list[CouplingCell]:
    """返回某参数被所有诊断维度影响的非零耦合单元。"""
    result: list[CouplingCell] = []
    for diag in DIAG_DIMS:
        cell = COUPLING_MATRIX[diag].get(param)
        if cell is not None:
            result.append(cell)
    return result


# ---------------------------------------------------------------------------
# 构建期硬约束校验
# ---------------------------------------------------------------------------
def validate_matrix() -> None:
    """构建期校验耦合矩阵 3 条硬约束（design 2.7.3）。

    约束：
        1. 每个参数（列）至少含 1 个非零元素（无被遗忘的参数）；
        2. 每个诊断维度（行）至少含 1 个非零元素（无无效维度）；
        3. 矩阵非零密度 ≥ 30%。

    Raises:
        AssertionError: 任一约束不通过。
    """
    # 维度数与参数数一致性
    assert len(COUPLING_MATRIX) == len(DIAG_DIMS), (
        f"耦合矩阵行数 {len(COUPLING_MATRIX)} != 诊断维度数 {len(DIAG_DIMS)}"
    )
    for diag in DIAG_DIMS:
        row = COUPLING_MATRIX[diag]
        assert len(row) == len(PARAM_NAMES), (
            f"耦合矩阵 {diag!r} 行列数 {len(row)} != 参数数 {len(PARAM_NAMES)}"
        )

    # 约束 1：每个参数（列）至少含 1 个非零元素
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

    # 约束 2：每个诊断维度（行）至少含 1 个非零元素
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

    # 约束 3：矩阵非零密度 ≥ 30%
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
    """返回矩阵统计信息（用于调试/报告，非校验）。"""
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
            }
        ),
    }


# 模块导入时即执行构建期校验，确保矩阵始终合法
validate_matrix()