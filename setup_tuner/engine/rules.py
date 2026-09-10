"""规则库：每症状的 delta_table（Dx×C 的物化结果）。

规则库为 ``{症状, 全参数增量 delta_table, 依据:官方出处}`` 的集合，
其中 ``delta_table`` 是「症状 → 全参数增量」的**物化结果**，
由该症状的 Dx（强度 s=1 时）经耦合矩阵 C 预计算导出：

    delta_table[param] = Σ_d Dx_s1[d] × C[d][param]

运行时可「查表 + 遥测校准」快速产建议，同时每条增量都携带 ``source``，
满足 AC-04（每条建议可追溯官方出处）。

数据格式对齐 design 2.7.6，``source`` 取值受限枚举：
    ``EA_SETUP_GUIDE`` / ``EA_UDP_2026`` / ``PIRELLI``（对齐 FR-RPT-02）。

本模块为纯函数、零 IO、零随机，规则库在内存中由 Dx 与 C 物化构建，
满足 FR-ENG-05 / FR-NFR-R1（可复现）。
"""

from __future__ import annotations

from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.domain.symptoms import (
    Symptom,
    get_symptom_category,
    get_symptom_label,
)

from .coupling import COUPLING_MATRIX, EA_SETUP_GUIDE, CouplingCell
from .diagnostic import DIAG_DIMS, DIAG_DIMS_ZH, SYMPTOM_TO_DX

# ---------------------------------------------------------------------------
# 规则级主出处（症状→Dx 映射的出处，来自官方调教指南）
# ---------------------------------------------------------------------------
RULE_SOURCE = EA_SETUP_GUIDE  # 症状-机理对应章节出处


# ---------------------------------------------------------------------------
# 物化单症状的 delta_table（强度 s=1 时的全参数增量）
# ---------------------------------------------------------------------------
def _materialize_symptom(symptom: str) -> dict[str, Any]:
    """物化单症状的规则（强度 s=1 时的 Dx×C 结果）。

    Args:
        symptom: 症状标识字符串。

    Returns:
        规则字典，结构对齐 design 2.7.6：
        ::
            {
              "id": "rule_understeer",
              "symptom": "understeer",
              "category": "entry",
              "name_zh": "转向不足",
              "dx": {dim: coef, ...},          # s=1 时的 Dx 贡献
              "delta_table": {
                param: {
                  "value": float,              # s=1 时的增量（带符号）
                  "sources": [str, ...],       # 贡献该参数的出处枚举列表
                  "linkages": [str, ...]       # 贡献该参数的诊断维度说明
                },
                ...23 个参数...
              },
              "source": "EA_SETUP_GUIDE"       # 规则级主出处
            }

    Raises:
        KeyError: 症状未知。
    """
    if symptom not in SYMPTOM_TO_DX:
        raise KeyError(f"未知症状标识: {symptom!r}")

    dx_coefs = SYMPTOM_TO_DX[symptom]  # {dim: coef} 强度 s=1 时的 Dx

    # 矩阵乘法：raw[p] = Σ_d Dx[d] × C[d][p]
    delta_table: dict[str, dict[str, Any]] = {}
    for spec in ALL_SETUP_FIELDS:
        param = spec.name
        total = 0.0
        sources: list[str] = []
        linkages: list[str] = []
        for dim in DIAG_DIMS:
            dx_val = dx_coefs.get(dim, 0.0)
            if dx_val == 0.0:
                continue
            cell: CouplingCell | None = COUPLING_MATRIX[dim][param]
            if cell is None:
                continue
            total += dx_val * cell.value
            if cell.source not in sources:
                sources.append(cell.source)
            linkages.append(
                f"{dim}({DIAG_DIMS_ZH[dim]}) Dx={dx_val:+.2f} × C={cell.value:+.2f}"
            )
        delta_table[param] = {
            "value": total,
            "sources": sources,
            "linkages": linkages,
        }

    # 症状元信息
    sym_enum = Symptom(symptom)
    return {
        "id": f"rule_{symptom}",
        "symptom": symptom,
        "category": get_symptom_category(sym_enum),
        "name_zh": get_symptom_label(sym_enum),
        "dx": dict(dx_coefs),
        "delta_table": delta_table,
        "source": RULE_SOURCE,
    }


# ---------------------------------------------------------------------------
# 规则库加载与查询
# ---------------------------------------------------------------------------
def load_rules() -> dict[str, dict[str, Any]]:
    """加载规则库（在内存中由 Dx 与 C 物化构建，纯函数无 IO）。

    Returns:
        {symptom_key: rule_dict} 字典，覆盖全部 12 症状。
        每条规则结构见 :func:`_materialize_symptom`。
    """
    return {symptom: _materialize_symptom(symptom) for symptom in SYMPTOM_TO_DX}


def get_rule(symptom: str) -> dict[str, Any] | None:
    """查询单症状规则。

    Args:
        symptom: 症状标识字符串。

    Returns:
        规则字典；若症状未知返回 None。
    """
    if symptom not in SYMPTOM_TO_DX:
        return None
    return _materialize_symptom(symptom)


def get_all_rules() -> dict[str, dict[str, Any]]:
    """获取全部规则（load_rules 的别名，语义清晰）。"""
    return load_rules()


# ---------------------------------------------------------------------------
# 规则库构建期自校验
# ---------------------------------------------------------------------------
def validate_rules() -> None:
    """构建期校验规则库完整性。

    校验项：
        1. 规则数 == 12（覆盖全部症状）；
        2. 每条规则的 delta_table 覆盖全部 23 参数；
        3. 每条规则 source 非空；
        4. 每条非零 delta 的参数 sources 非空（可追溯出处）。

    Raises:
        AssertionError: 任一校验不通过。
    """
    rules = load_rules()
    assert len(rules) == len(SYMPTOM_TO_DX), (
        f"规则数 {len(rules)} != 症状数 {len(SYMPTOM_TO_DX)}"
    )

    expected_params = {f.name for f in ALL_SETUP_FIELDS}
    for symptom, rule in rules.items():
        dt = rule["delta_table"]
        assert set(dt.keys()) == expected_params, (
            f"规则 {symptom!r} delta_table 参数集不匹配: "
            f"缺 {expected_params - set(dt.keys())}, "
            f"多 {set(dt.keys()) - expected_params}"
        )
        assert rule["source"], f"规则 {symptom!r} source 为空"
        for param, entry in dt.items():
            if abs(entry["value"]) > 1e-12:
                assert entry["sources"], (
                    f"规则 {symptom!r} 参数 {param!r} 非零增量但 sources 为空"
                )


# 模块导入时即构建并校验规则库
validate_rules()