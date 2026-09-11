"""切片 4 深度测试：诊断向量(9维)与耦合矩阵(9×23)（diagnostic + coupling）。

覆盖：
    - setup_tuner.engine.diagnostic（9 维 Dx 向量，compute_dx 等纯函数）
    - setup_tuner.engine.coupling（9×23 耦合矩阵，CouplingCell，validate_matrix）

5 种测试方式：
    1. unit     — compute_dx / get_coupling / empty_dx 等正常输入正确性
    2. boundary — 未知症状、强度越界、非法类型、空输入
    3. property — 确定性、线性叠加性、is_zero_dx 不变量
    4. static   — 9 维 / 23 参数 / 矩阵密度 / 出处枚举约束
    5. smoke    — 真实症状列表 → compute_dx → dx_to_vector → 矩阵乘法链路
"""

from __future__ import annotations

import pytest

from setup_tuner.engine.coupling import (
    COUPLING_MATRIX,
    EA_SETUP_GUIDE,
    EA_UDP_2026,
    PARAM_NAMES,
    PIRELLI,
    VALID_SOURCES,
    CouplingCell,
    get_column,
    get_coupling,
    get_row,
    matrix_stats,
    nonzero_cells_for_diag,
    nonzero_cells_for_param,
    validate_matrix,
)
from setup_tuner.engine.diagnostic import (
    DIAG_DIMS,
    DIAG_DIMS_POSITIVE_SEMANTICS,
    DIAG_DIMS_ZH,
    SYMPTOM_TO_DX,
    compute_dx,
    dx_to_vector,
    empty_dx,
    is_zero_dx,
)


# ===========================================================================
# 1. 单元测试 (unit) — 每个公开函数的正常输入正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证诊断向量与耦合矩阵公开函数的正常输入正确性。"""

    def test_unit_compute_dx_single_symptom(self) -> None:
        """compute_dx 对单症状应返回正确的 Dx 向量。"""
        dx = compute_dx([("understeer", 3)])
        # understeer 映射: front_grip_req=0.80, turnin_req=0.30
        assert dx["front_grip_req"] == pytest.approx(0.80 * 3)
        assert dx["turnin_req"] == pytest.approx(0.30 * 3)
        # 其余维度为 0
        assert dx["rear_grip_req"] == 0.0

    def test_unit_compute_dx_multiple_symptoms(self) -> None:
        """compute_dx 对多症状应代数求和。"""
        dx = compute_dx([("understeer", 2), ("oversteer", 2)])
        # understeer: front_grip_req=0.80, turnin_req=0.30
        # oversteer: rear_grip_req=0.80, hi_speed_stab_req=0.30
        assert dx["front_grip_req"] == pytest.approx(0.80 * 2)
        assert dx["rear_grip_req"] == pytest.approx(0.80 * 2)
        assert dx["turnin_req"] == pytest.approx(0.30 * 2)
        assert dx["hi_speed_stab_req"] == pytest.approx(0.30 * 2)

    def test_unit_compute_dx_zero_strength_no_contribution(self) -> None:
        """compute_dx 对强度 0 的症状应不产生贡献。"""
        dx = compute_dx([("understeer", 0)])
        assert is_zero_dx(dx)

    def test_unit_empty_dx(self) -> None:
        """empty_dx 应返回全零 9 维向量。"""
        dx = empty_dx()
        assert len(dx) == 9
        assert all(v == 0.0 for v in dx.values())
        assert is_zero_dx(dx)

    def test_unit_is_zero_dx_true_for_empty(self) -> None:
        """is_zero_dx 对全零向量应返回 True。"""
        assert is_zero_dx(empty_dx())

    def test_unit_is_zero_dx_false_for_nonzero(self) -> None:
        """is_zero_dx 对非零向量应返回 False。"""
        dx = compute_dx([("understeer", 1)])
        assert not is_zero_dx(dx)

    def test_unit_dx_to_vector_order(self) -> None:
        """dx_to_vector 应按 DIAG_DIMS 顺序返回列表。"""
        dx = compute_dx([("understeer", 1)])
        vec = dx_to_vector(dx)
        assert len(vec) == 9
        # 第 0 个是 front_grip_req
        assert vec[0] == pytest.approx(0.80)
        # 第 2 个是 turnin_req
        assert vec[2] == pytest.approx(0.30)

    def test_unit_get_coupling_returns_cell(self) -> None:
        """get_coupling 应返回对应 CouplingCell。"""
        cell = get_coupling("front_grip_req", "front_wing")
        assert cell is not None
        assert isinstance(cell, CouplingCell)
        assert cell.param == "front_wing"
        assert cell.diag == "front_grip_req"

    def test_unit_get_coupling_none_for_zero(self) -> None:
        """get_coupling 对零格应返回 None。"""
        # front_grip_req 对 active_aero_x 无耦合
        cell = get_coupling("front_grip_req", "active_aero_x")
        assert cell is None

    def test_unit_get_row(self) -> None:
        """get_row 应返回某诊断维度的整行（23 个参数）。"""
        row = get_row("front_grip_req")
        assert len(row) == 23

    def test_unit_get_column(self) -> None:
        """get_column 应返回某参数的整列（9 个诊断维度）。"""
        col = get_column("front_wing")
        assert len(col) == 9

    def test_unit_nonzero_cells_for_diag(self) -> None:
        """nonzero_cells_for_diag 应返回非零耦合单元列表。"""
        cells = nonzero_cells_for_diag("front_grip_req")
        assert len(cells) > 0
        assert all(isinstance(c, CouplingCell) for c in cells)

    def test_unit_nonzero_cells_for_param(self) -> None:
        """nonzero_cells_for_param 应返回非零耦合单元列表。"""
        cells = nonzero_cells_for_param("front_wing")
        assert len(cells) > 0

    def test_unit_coupling_cell_value_property(self) -> None:
        """CouplingCell.value 应为 sign × magnitude。"""
        cell = get_coupling("front_grip_req", "front_wing")
        assert cell is not None
        assert cell.value == cell.sign * cell.magnitude


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 未知症状、强度越界、非法类型
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证异常输入的处理。"""

    def test_boundary_compute_dx_unknown_symptom_raises(self) -> None:
        """compute_dx 对未知症状应抛 KeyError。"""
        with pytest.raises(KeyError, match="未知症状"):
            compute_dx([("nonexistent_symptom", 3)])

    def test_boundary_compute_dx_strength_below_zero_raises(self) -> None:
        """compute_dx 对强度 < 0 应抛 ValueError。"""
        with pytest.raises(ValueError, match="越界"):
            compute_dx([("understeer", -1)])

    def test_boundary_compute_dx_strength_above_five_raises(self) -> None:
        """compute_dx 对强度 > 5 应抛 ValueError。"""
        with pytest.raises(ValueError, match="越界"):
            compute_dx([("understeer", 6)])

    def test_boundary_compute_dx_invalid_strength_type_raises(self) -> None:
        """compute_dx 对非法强度类型应抛 ValueError。"""
        with pytest.raises(ValueError, match="类型非法"):
            compute_dx([("understeer", "3")])  # type: ignore[list-item]

    def test_boundary_compute_dx_empty_input(self) -> None:
        """compute_dx 对空症状列表应返回全零 Dx。"""
        dx = compute_dx([])
        assert is_zero_dx(dx)

    def test_boundary_compute_dx_strength_zero(self) -> None:
        """compute_dx 对强度 0 应等同于未触发。"""
        dx = compute_dx([("understeer", 0)])
        assert is_zero_dx(dx)

    def test_boundary_compute_dx_strength_at_boundary(self) -> None:
        """compute_dx 对边界强度 0 和 5 应正常计算。"""
        dx0 = compute_dx([("understeer", 0)])
        assert is_zero_dx(dx0)
        dx5 = compute_dx([("understeer", 5)])
        assert dx5["front_grip_req"] == pytest.approx(0.80 * 5)

    def test_boundary_get_coupling_unknown_diag_returns_none(self) -> None:
        """get_coupling 对未知诊断维度应返回 None。"""
        assert get_coupling("unknown_diag", "front_wing") is None

    def test_boundary_get_coupling_unknown_param_returns_none(self) -> None:
        """get_coupling 对未知参数应返回 None。"""
        assert get_coupling("front_grip_req", "unknown_param") is None

    def test_boundary_get_row_unknown_diag_returns_empty_dict(self) -> None:
        """get_row 对未知诊断维度应返回全 None 的 23 槽字典。"""
        row = get_row("unknown_diag")
        assert len(row) == 23
        assert all(v is None for v in row.values())

    def test_boundary_dx_to_vector_missing_dims_defaults_zero(self) -> None:
        """dx_to_vector 对缺失维度应默认 0.0。"""
        vec = dx_to_vector({})
        assert len(vec) == 9
        assert all(v == 0.0 for v in vec)


# ===========================================================================
# 3. 属性不变量测试 (property) — 确定性、线性叠加性
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证确定性与线性叠加性。"""

    def test_property_compute_dx_deterministic(self) -> None:
        """确定性：相同输入两次计算结果完全一致。"""
        symptoms = [("understeer", 3), ("oversteer", 2), ("lockup", 4)]
        dx1 = compute_dx(symptoms)
        dx2 = compute_dx(symptoms)
        assert dx1 == dx2

    def test_property_compute_dx_linear_superposition(self) -> None:
        """线性叠加性：Dx(s1+s2) == Dx(s1) + Dx(s2)（逐分量）。"""
        dx1 = compute_dx([("understeer", 2)])
        dx2 = compute_dx([("oversteer", 3)])
        dx_sum = compute_dx([("understeer", 2), ("oversteer", 3)])
        for dim in DIAG_DIMS:
            assert dx_sum[dim] == pytest.approx(dx1[dim] + dx2[dim])

    def test_property_compute_dx_scaling(self) -> None:
        """线性缩放：Dx(s×k) == Dx(s) × k（对单症状）。"""
        dx1 = compute_dx([("understeer", 1)])
        dx3 = compute_dx([("understeer", 3)])
        for dim in DIAG_DIMS:
            assert dx3[dim] == pytest.approx(dx1[dim] * 3)

    def test_property_empty_dx_all_nine_dims(self) -> None:
        """不变量：empty_dx 含全部 9 个维度。"""
        dx = empty_dx()
        assert set(dx.keys()) == set(DIAG_DIMS)

    def test_property_compute_dx_covers_all_dims(self) -> None:
        """不变量：compute_dx 结果含全部 9 个维度。"""
        dx = compute_dx([("understeer", 3)])
        assert set(dx.keys()) == set(DIAG_DIMS)

    def test_property_is_zero_dx_symmetric(self) -> None:
        """不变量：is_zero_dx 对全零返回 True，对任一非零返回 False。"""
        assert is_zero_dx(empty_dx())
        dx = empty_dx()
        dx["front_grip_req"] = 0.001
        assert not is_zero_dx(dx)

    def test_property_coupling_cell_value_sign_consistency(self) -> None:
        """不变量：CouplingCell.value 符号与 sign 一致。"""
        for diag in DIAG_DIMS:
            for cell in nonzero_cells_for_diag(diag):
                if cell.sign > 0:
                    assert cell.value > 0
                else:
                    assert cell.value < 0

    def test_property_matrix_validate_passes(self) -> None:
        """不变量：validate_matrix 应通过（构建期已校验）。"""
        validate_matrix()  # 不抛异常

    def test_property_dx_to_vector_inverse(self) -> None:
        """不变量：dx_to_vector 长度与 DIAG_DIMS 一致。"""
        dx = compute_dx([("understeer", 3)])
        vec = dx_to_vector(dx)
        assert len(vec) == len(DIAG_DIMS)


# ===========================================================================
# 4. 静态分析 (static) — 9 维 / 23 参数 / 矩阵密度 / 出处枚举
# ===========================================================================
class TestStatic:
    """静态分析：验证维度、参数、矩阵密度与出处约束。"""

    def test_static_diag_dims_count_is_9(self) -> None:
        """数量约束：DIAG_DIMS 恰好 9 维。"""
        assert len(DIAG_DIMS) == 9

    def test_static_param_names_count_is_23(self) -> None:
        """数量约束：PARAM_NAMES 恰好 23 项。"""
        assert len(PARAM_NAMES) == 23

    def test_static_matrix_shape_9x23(self) -> None:
        """形状约束：耦合矩阵为 9×23。"""
        assert len(COUPLING_MATRIX) == 9
        for diag in DIAG_DIMS:
            assert len(COUPLING_MATRIX[diag]) == 23

    def test_static_matrix_density_above_30_percent(self) -> None:
        """密度约束：矩阵非零密度 ≥ 30%。"""
        stats = matrix_stats()
        assert stats["density"] >= 0.30

    def test_static_valid_sources(self) -> None:
        """枚举约束：VALID_SOURCES 含 3 个合法出处。"""
        assert VALID_SOURCES == frozenset({EA_SETUP_GUIDE, EA_UDP_2026, PIRELLI})

    def test_static_all_cells_use_valid_sources(self) -> None:
        """不变量约束：所有非零单元的 source 在 VALID_SOURCES 内。"""
        for diag in DIAG_DIMS:
            for cell in COUPLING_MATRIX[diag].values():
                if cell is not None:
                    assert cell.source in VALID_SOURCES

    def test_static_all_cells_sign_in_plus_minus_1(self) -> None:
        """不变量约束：所有非零单元 sign ∈ {+1, -1}。"""
        for diag in DIAG_DIMS:
            for cell in COUPLING_MATRIX[diag].values():
                if cell is not None:
                    assert cell.sign in (+1, -1)

    def test_static_all_cells_magnitude_positive(self) -> None:
        """不变量约束：所有非零单元 magnitude > 0。"""
        for diag in DIAG_DIMS:
            for cell in COUPLING_MATRIX[diag].values():
                if cell is not None:
                    assert cell.magnitude > 0

    def test_static_each_param_has_nonzero_cell(self) -> None:
        """约束1：每个参数至少含 1 个非零元素。"""
        for param in PARAM_NAMES:
            cells = nonzero_cells_for_param(param)
            assert len(cells) >= 1, f"参数 {param} 在所有维度上为零"

    def test_static_each_diag_has_nonzero_cell(self) -> None:
        """约束2：每个诊断维度至少含 1 个非零元素。"""
        for diag in DIAG_DIMS:
            cells = nonzero_cells_for_diag(diag)
            assert len(cells) >= 1, f"诊断维度 {diag} 在所有参数上为零"

    @pytest.mark.parametrize("dim", DIAG_DIMS)
    def test_static_each_dim_has_zh_name(self, dim: str) -> None:
        """参数化：每个维度有中文名称。"""
        assert dim in DIAG_DIMS_ZH
        assert DIAG_DIMS_ZH[dim]

    @pytest.mark.parametrize("dim", DIAG_DIMS)
    def test_static_each_dim_has_positive_semantics(self, dim: str) -> None:
        """参数化：每个维度有正值语义说明。"""
        assert dim in DIAG_DIMS_POSITIVE_SEMANTICS
        assert DIAG_DIMS_POSITIVE_SEMANTICS[dim]

    @pytest.mark.parametrize("symptom", list(SYMPTOM_TO_DX.keys()))
    def test_static_each_symptom_maps_to_valid_dims(self, symptom: str) -> None:
        """参数化：每个症状映射到的维度都在 DIAG_DIMS 内。"""
        for dim in SYMPTOM_TO_DX[symptom]:
            assert dim in DIAG_DIMS

    def test_static_coupling_cell_is_frozen(self) -> None:
        """类型约束：CouplingCell 为 frozen dataclass。"""
        cell = get_coupling("front_grip_req", "front_wing")
        assert cell is not None
        with pytest.raises((AttributeError, TypeError)):
            cell.param = "modified"  # type: ignore[misc]


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 真实症状 → Dx → 矩阵乘法链路
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：用真实症状数据走完整 Dx → C 链路。"""

    def test_smoke_full_dx_computation_chain(self) -> None:
        """冒烟：多症状 → compute_dx → dx_to_vector → 验证 9 维。"""
        symptoms = [
            ("understeer", 3), ("oversteer", 2), ("brake_long", 4),
            ("exit_wheelspin", 3), ("bottoming", 2),
        ]
        dx = compute_dx(symptoms)
        vec = dx_to_vector(dx)
        assert len(vec) == 9
        # 至少有 1 个非零分量
        assert any(abs(v) > 1e-9 for v in vec)

    def test_smoke_dx_matrix_multiplication(self) -> None:
        """冒烟：Dx × C 矩阵乘法产出 23 参数增量。"""
        dx = compute_dx([("understeer", 3)])
        # 手动矩阵乘法：raw[p] = Σ_d Dx[d] × C[d][p]
        raw: dict[str, float] = {}
        for param in PARAM_NAMES:
            total = 0.0
            for dim in DIAG_DIMS:
                dx_val = dx.get(dim, 0.0)
                if dx_val == 0.0:
                    continue
                cell = COUPLING_MATRIX[dim][param]
                if cell is None:
                    continue
                total += dx_val * cell.value
            raw[param] = total
        # understeer 应对 front_wing 产生非零增量
        assert raw["front_wing"] != 0.0

    def test_smoke_all_12_symptoms_produce_valid_dx(self) -> None:
        """冒烟：全部 12 症状各自计算 Dx 均产出 9 维向量。"""
        for symptom in SYMPTOM_TO_DX:
            dx = compute_dx([(symptom, 3)])
            assert set(dx.keys()) == set(DIAG_DIMS)

    def test_smoke_matrix_stats_real(self) -> None:
        """冒烟：matrix_stats 返回真实统计信息。"""
        stats = matrix_stats()
        assert stats["diag_dims"] == 9
        assert stats["params"] == 23
        assert stats["total_cells"] == 9 * 23
        assert stats["nonzero_cells"] > 0
        assert stats["density"] > 0.0
        assert len(stats["sources_used"]) >= 1

    def test_smoke_conflicting_symptoms_cancellation(self) -> None:
        """冒烟：矛盾症状对（如 understeer + straight_slow）在 front_grip_req 上部分抵消。"""
        # understeer: front_grip_req += 0.80 × s
        # straight_slow: front_grip_req += -0.50 × s
        dx = compute_dx([("understeer", 2), ("straight_slow", 2)])
        # 0.80*2 + (-0.50)*2 = 0.60
        assert dx["front_grip_req"] == pytest.approx(0.80 * 2 - 0.50 * 2)

    def test_smoke_lockup_negative_brake_power(self) -> None:
        """冒烟：lockup 症状对 brake_power_req 产生负贡献（需减压）。"""
        dx = compute_dx([("lockup", 3)])
        # lockup: brake_power_req = -0.30 × 3 = -0.90
        assert dx["brake_power_req"] == pytest.approx(-0.30 * 3)

    def test_smoke_full_matrix_row_column_access(self) -> None:
        """冒烟：遍历全部 9 行 23 列，所有单元可访问。"""
        for diag in DIAG_DIMS:
            row = get_row(diag)
            assert len(row) == 23
        for param in PARAM_NAMES:
            col = get_column(param)
            assert len(col) == 9