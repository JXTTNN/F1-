"""切片8 深度测试 —— setup_tuner/report/builder.py 报告生成。

覆盖报告组装主入口 build_report 与 4 个辅助函数：
    - format_linkages       — 联动说明格式化
    - build_summary         — 摘要生成
    - extract_setup_from_packet5 — 从遥测包提取调教参数
    - extract_telemetry_summary  — 遥测摘要提取
    - feedbacks_to_symptoms     — 反馈转症状列表

5 种测试方式（每个 class 对应一种，注释明确标注）：
    1. TestUnit     — 单元测试：每个函数的正常输入正确性
    2. TestBoundary — 边界/异常测试：空输入、缺字段、越界值
    3. TestProperty — 属性不变量测试：确定性、往返一致性、字段完整性
    4. TestStatic   — 静态分析：值域约束、字段约束、source 枚举
    5. TestSmoke    — 实际运行冒烟：真实引擎输出 → 报告 → 落库
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.engine.engine import generate_suggestion
from setup_tuner.report.builder import (
    build_report,
    build_summary,
    extract_setup_from_packet5,
    extract_telemetry_summary,
    feedbacks_to_symptoms,
    format_linkages,
)


# ===========================================================================
# 1. 单元测试 (unit) — 每个函数的正常输入正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证报告组装各函数在正常输入下的正确行为。"""

    # ---------- format_linkages ----------
    def test_format_linkages_list(self) -> None:
        """format_linkages 列表输入原样返回（确保为 str）。"""
        result = format_linkages({
            "linkages": ["front_grip_req(前轴抓地)", "turnin_req(入弯响应)"],
        })
        assert result == ["front_grip_req(前轴抓地)", "turnin_req(入弯响应)"]

    def test_format_linkages_string(self) -> None:
        """format_linkages 字符串输入返回单元素列表。"""
        result = format_linkages({"linkages": "single_string"})
        assert result == ["single_string"]

    # ---------- build_summary ----------
    def test_build_summary_with_nonzero(self) -> None:
        """build_summary 含非零调整参数的摘要。"""
        params = [
            {"setup_delta": 1.0},
            {"setup_delta": 0.0},
            {"setup_delta": -2.0},
        ]
        summary = build_summary(params)
        assert "3" in summary  # 总参数 3
        assert "2" in summary  # 非零 2

    def test_build_summary_all_zero(self) -> None:
        """build_summary 全零调整的摘要。"""
        params = [{"setup_delta": 0.0}, {"setup_delta": 0.0}]
        summary = build_summary(params)
        assert "均无需调整" in summary

    # ---------- build_report ----------
    def test_build_report_structure(self) -> None:
        """build_report 返回含 design 2.7.7 全部顶层字段。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=CarSetup.default().to_dict(),
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka", setup_id=1)
        for field in ("track_id", "setup_id", "generated_at", "parameters",
                      "summary", "confidence", "setup_delta", "dx"):
            assert field in report, f"报告缺字段: {field}"

    def test_build_report_track_id(self) -> None:
        """build_report 顶层 track_id 与入参一致。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=CarSetup.default().to_dict(),
            track_id="monza",
        )
        report = build_report(suggestion, track_id="monza")
        assert report["track_id"] == "monza"

    def test_build_report_setup_id(self) -> None:
        """build_report setup_id 透传。"""
        suggestion = {"parameters": [], "confidence": "medium"}
        report = build_report(suggestion, track_id="suzuka", setup_id=42)
        assert report["setup_id"] == 42

    def test_build_report_generated_at_iso8601(self) -> None:
        """build_report generated_at 为 ISO8601 格式（含 Z 后缀）。"""
        suggestion = {"parameters": [], "confidence": "medium"}
        report = build_report(suggestion, track_id="suzuka")
        assert report["generated_at"].endswith("Z")
        # 应可被 datetime 解析
        dt = datetime.strptime(report["generated_at"], "%Y-%m-%dT%H:%M:%SZ")
        assert dt is not None

    def test_build_report_param_entry_fields(self) -> None:
        """build_report 每个参数项含 design 2.7.7 八字段。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=CarSetup.default().to_dict(),
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        for p in report["parameters"]:
            for field in ("param", "current", "setup_delta", "linkages",
                          "linked_notes", "source", "confidence"):
                assert field in p, f"参数项缺字段: {field}"

    # ---------- extract_setup_from_packet5 ----------
    def test_extract_setup_returns_23_fields(self, sample_packet5: dict) -> None:
        """extract_setup_from_packet5 返回 23 项参数。"""
        result = extract_setup_from_packet5(sample_packet5)
        assert len(result) == 23
        # 字段名与 ALL_SETUP_FIELDS 一致
        assert set(result.keys()) == {f.name for f in ALL_SETUP_FIELDS}

    def test_extract_setup_front_wing(self, sample_packet5: dict) -> None:
        """extract_setup_from_packet5 正确映射 m_frontWing → front_wing。"""
        result = extract_setup_from_packet5(sample_packet5)
        assert result["front_wing"] == 6.0

    def test_extract_setup_aero_mode_z(self, sample_packet5: dict) -> None:
        """extract_setup_from_packet5 m_activeAeroMode=0 → active_aero_z=0.6。"""
        result = extract_setup_from_packet5(sample_packet5)
        assert result["active_aero_z"] == 0.6
        assert result["active_aero_x"] == 0.4

    def test_extract_setup_tyre_pressure(self, sample_packet5: dict) -> None:
        """extract_setup_from_packet5 胎压取前/后轴均值。"""
        result = extract_setup_from_packet5(sample_packet5)
        # tyresPressure = [25.5, 25.6, 25.4, 25.5]
        assert abs(result["front_tyre_pressure"] - 25.55) < 1e-6
        assert abs(result["rear_tyre_pressure"] - 25.45) < 1e-6

    # ---------- extract_telemetry_summary ----------
    def test_extract_telemetry_summary_session(self) -> None:
        """extract_telemetry_summary 从 Packet 1 提取天气/温度。"""
        all_latest = {1: {"m_weather": 1, "m_trackTemperature": 30, "m_airTemperature": 25}}
        summary = extract_telemetry_summary(all_latest)
        assert summary["weather"] == 1
        assert summary["track_temp"] == 30
        assert summary["air_temp"] == 25

    def test_extract_telemetry_summary_telemetry(self) -> None:
        """extract_telemetry_summary 从 Packet 6 提取速度/油门。"""
        all_latest = {6: {"m_speed": 300, "m_throttle": 0.8, "m_gear": 5}}
        summary = extract_telemetry_summary(all_latest)
        assert summary["speed"] == 300
        assert summary["throttle"] == 0.8
        assert summary["gear"] == 5

    # ---------- feedbacks_to_symptoms ----------
    def test_feedbacks_to_symptoms(self) -> None:
        """feedbacks_to_symptoms 正确转换为 (symptom, strength) 列表。"""
        feedbacks = [
            {"symptom": "understeer", "strength": 4},
            {"symptom": "oversteer", "strength": 3},
        ]
        result = feedbacks_to_symptoms(feedbacks)
        assert result == [("understeer", 4), ("oversteer", 3)]


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 空输入、缺字段、越界值
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证各函数在空/缺字段/越界输入下的健壮行为。"""

    # ---------- format_linkages ----------
    def test_format_linkages_empty_list(self) -> None:
        """format_linkages 空列表返回空列表。"""
        assert format_linkages({"linkages": []}) == []

    def test_format_linkages_missing_key(self) -> None:
        """format_linkages 缺 linkages 键返回空列表。"""
        assert format_linkages({}) == []

    def test_format_linkages_none(self) -> None:
        """format_linkages linkages=None 返回空列表。"""
        assert format_linkages({"linkages": None}) == []

    def test_format_linkages_empty_string(self) -> None:
        """format_linkages 空字符串返回空列表（falsy）。"""
        assert format_linkages({"linkages": ""}) == []

    # ---------- build_summary ----------
    def test_build_summary_empty(self) -> None:
        """build_summary 空参数列表返回特定提示。"""
        assert build_summary([]) == "本次建议无参数"

    def test_build_summary_missing_setup_delta(self) -> None:
        """build_summary 缺 setup_delta 字段视为 0.0。"""
        summary = build_summary([{}, {}])
        assert "均无需调整" in summary

    # ---------- build_report ----------
    def test_build_report_empty_suggestion(self) -> None:
        """build_report 空 suggestion 返回空参数列表报告。"""
        report = build_report({}, track_id="suzuka")
        assert report["parameters"] == []
        assert report["track_id"] == "suzuka"
        assert report["confidence"] == "medium"  # 缺省

    def test_build_report_missing_parameters(self) -> None:
        """build_report suggestion 缺 parameters 键不抛异常。"""
        report = build_report({"confidence": "high"}, track_id="suzuka")
        assert report["parameters"] == []

    def test_build_report_setup_id_none(self) -> None:
        """build_report setup_id=None 透传。"""
        report = build_report({}, track_id="suzuka", setup_id=None)
        assert report["setup_id"] is None

    # ---------- extract_setup_from_packet5 ----------
    def test_extract_setup_empty_packet(self) -> None:
        """extract_setup_from_packet5 空字典返回全缺省值。"""
        result = extract_setup_from_packet5({})
        defaults = {f.name: f.default for f in ALL_SETUP_FIELDS}
        assert result == defaults

    def test_extract_setup_missing_fields(self) -> None:
        """extract_setup_from_packet5 缺部分字段取缺省。"""
        result = extract_setup_from_packet5({"m_frontWing": 7.0})
        assert result["front_wing"] == 7.0
        # rear_wing 缺省
        assert result["rear_wing"] == 5.0

    def test_extract_setup_out_of_range_clamped(self) -> None:
        """extract_setup_from_packet5 越界值被 clamp 到合法区间。"""
        # front_wing 范围 0-11，传 99 应 clamp 到 11
        result = extract_setup_from_packet5({"m_frontWing": 99.0})
        assert result["front_wing"] == 11.0

    def test_extract_setup_negative_clamped(self) -> None:
        """extract_setup_from_packet5 负数越界被 clamp 到 min。"""
        result = extract_setup_from_packet5({"m_frontWing": -99.0})
        assert result["front_wing"] == 0.0

    def test_extract_setup_aero_mode_unknown(self) -> None:
        """extract_setup_from_packet5 未知 aero_mode 不修改缺省值。"""
        result = extract_setup_from_packet5({"m_activeAeroMode": 99})
        # 缺省 0.5
        assert result["active_aero_z"] == 0.5
        assert result["active_aero_x"] == 0.5

    def test_extract_setup_aero_mode_x(self) -> None:
        """extract_setup_from_packet5 m_activeAeroMode=1 → X 直道模式。"""
        result = extract_setup_from_packet5({"m_activeAeroMode": 1})
        assert result["active_aero_z"] == 0.4
        assert result["active_aero_x"] == 0.6

    def test_extract_setup_tyre_pressure_short(self) -> None:
        """extract_setup_from_packet5 胎压列表不足 4 个取缺省。"""
        result = extract_setup_from_packet5({"tyresPressure": [25.0, 25.0]})
        # 缺省胎压
        assert result["front_tyre_pressure"] == 25.5

    # ---------- extract_telemetry_summary ----------
    def test_extract_telemetry_summary_empty(self) -> None:
        """extract_telemetry_summary 空字典返回空摘要。"""
        assert extract_telemetry_summary({}) == {}

    def test_extract_telemetry_summary_partial(self) -> None:
        """extract_telemetry_summary 仅含部分 packet 不抛异常。"""
        summary = extract_telemetry_summary({1: {"m_weather": 0}})
        assert summary["weather"] == 0
        assert "speed" not in summary  # Packet 6 缺失

    # ---------- feedbacks_to_symptoms ----------
    def test_feedbacks_to_symptoms_empty(self) -> None:
        """feedbacks_to_symptoms 空列表返回空列表。"""
        assert feedbacks_to_symptoms([]) == []

    def test_feedbacks_to_symptoms_filter_invalid(self) -> None:
        """feedbacks_to_symptoms 过滤掉缺 symptom/strength 的项。"""
        feedbacks = [
            {"symptom": "understeer", "strength": 3},
            {"symptom": None, "strength": 3},  # symptom None 应过滤
            {"symptom": "oversteer", "strength": None},  # strength None 应过滤
            {"strength": 3},  # 缺 symptom 应过滤
        ]
        result = feedbacks_to_symptoms(feedbacks)
        assert result == [("understeer", 3)]


# ===========================================================================
# 3. 属性不变量测试 (property) — 确定性、往返一致性、字段完整性
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证报告生成的确定性与字段完整性。"""

    def test_build_report_deterministic_except_timestamp(
        self, sample_params: dict,
    ) -> None:
        """build_report 除 generated_at 外确定性（相同输入相同输出）。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        r1 = build_report(suggestion, track_id="suzuka", setup_id=1)
        r2 = build_report(suggestion, track_id="suzuka", setup_id=1)
        # 排除时间戳后应完全一致
        r1.pop("generated_at")
        r2.pop("generated_at")
        assert r1 == r2

    def test_build_report_param_count_23(self, sample_params: dict) -> None:
        """build_report parameters 列表含 23 项（与 ALL_SETUP_FIELDS 对应）。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        assert len(report["parameters"]) == 23

    def test_build_report_setup_delta_keys_match_params(
        self, sample_params: dict,
    ) -> None:
        """build_report setup_delta 键集与 parameters 的 param 字段一致。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        param_names = {p["param"] for p in report["parameters"]}
        delta_names = set(report["setup_delta"].keys())
        assert param_names == delta_names

    def test_extract_setup_idempotent_with_defaults(self) -> None:
        """extract_setup_from_packet5 空字典等价于全缺省。"""
        r1 = extract_setup_from_packet5({})
        r2 = extract_setup_from_packet5({})
        assert r1 == r2

    def test_extract_setup_all_fields_in_range(self, sample_packet5: dict) -> None:
        """extract_setup_from_packet5 所有结果在 [min, max] 区间。"""
        result = extract_setup_from_packet5(sample_packet5)
        for spec in ALL_SETUP_FIELDS:
            v = result[spec.name]
            assert spec.min_val <= v <= spec.max_val, \
                f"{spec.name}={v} 越界 [{spec.min_val}, {spec.max_val}]"

    def test_format_linkages_preserves_count(self) -> None:
        """format_linkages 列表输入保持元素数量。"""
        lst = ["a", "b", "c", "d"]
        assert len(format_linkages({"linkages": lst})) == len(lst)

    def test_build_summary_total_count_consistent(self) -> None:
        """build_summary 摘要中总参数数与输入列表长度一致。"""
        params = [{"setup_delta": 1.0}, {"setup_delta": 0.0}, {"setup_delta": 2.0}]
        summary = build_summary(params)
        assert "3" in summary  # 总数 3

    def test_feedbacks_to_symptoms_preserves_order(self) -> None:
        """feedbacks_to_symptoms 保持输入顺序。"""
        feedbacks = [
            {"symptom": s.value, "strength": 3}
            for s in [Symptom.UNDERSTEER, Symptom.OVERSTEER, Symptom.LOCKUP]
        ]
        result = feedbacks_to_symptoms(feedbacks)
        assert [r[0] for r in result] == ["understeer", "oversteer", "lockup"]


# ===========================================================================
# 4. 静态分析 (static) — 值域约束、字段约束、source 枚举
# ===========================================================================
class TestStatic:
    """静态分析：验证报告字段的值域约束与枚举值。"""

    def test_confidence_in_enum(self, sample_params: dict) -> None:
        """build_report confidence 取值 high|medium|low。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        assert report["confidence"] in {"high", "medium", "low"}

    def test_param_confidence_in_enum(self, sample_params: dict) -> None:
        """每个参数项的 confidence 取值 high|medium|low。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        for p in report["parameters"]:
            assert p["confidence"] in {"high", "medium", "low"}

    def test_generated_at_format(self) -> None:
        """generated_at 格式为 YYYY-MM-DDTHH:MM:SSZ。"""
        report = build_report({}, track_id="suzuka")
        ts = report["generated_at"]
        # 长度应为 20（YYYY-MM-DDTHH:MM:SSZ）
        assert len(ts) == 20
        assert ts[4] == "-" and ts[7] == "-" and ts[10] == "T"
        assert ts[13] == ":" and ts[16] == ":"
        assert ts[-1] == "Z"

    def test_setup_delta_values_are_float(self, sample_params: dict) -> None:
        """setup_delta 各值为 float 类型。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        for v in report["setup_delta"].values():
            assert isinstance(v, (int, float))

    def test_param_names_match_setup_fields(self, sample_params: dict) -> None:
        """parameters 的 param 字段名与 ALL_SETUP_FIELDS 一致。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        param_names = {p["param"] for p in report["parameters"]}
        expected = {f.name for f in ALL_SETUP_FIELDS}
        assert param_names == expected

    def test_extract_setup_keys_match_all_fields(self) -> None:
        """extract_setup_from_packet5 键集等于 ALL_SETUP_FIELDS 名集。"""
        result = extract_setup_from_packet5({})
        assert set(result.keys()) == {f.name for f in ALL_SETUP_FIELDS}

    def test_linkages_is_list_of_str(self, sample_params: dict) -> None:
        """每个参数项 linkages 为 list[str]。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        for p in report["parameters"]:
            assert isinstance(p["linkages"], list)
            for item in p["linkages"]:
                assert isinstance(item, str)

    def test_tradeoff_optional(self, sample_params: dict) -> None:
        """tradeoff 字段可选（无副作用时省略）。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        # tradeoff 可有可无，但若有则必须是字符串
        for p in report["parameters"]:
            if "tradeoff" in p:
                assert isinstance(p["tradeoff"], str)


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 真实引擎输出 → 报告 → 落库
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：真实引擎输出 → 报告组装 → JSON 序列化。"""

    def test_real_engine_to_report_to_json(self, sample_params: dict) -> None:
        """真实引擎输出 → build_report → json.dumps 可序列化。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 4), ("oversteer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka", setup_id=1)
        # 应可 JSON 序列化（含中文）
        text = json.dumps(report, ensure_ascii=False, sort_keys=True)
        assert "suzuka" in text
        # 反序列化应一致
        restored = json.loads(text)
        assert restored["track_id"] == "suzuka"

    def test_all_12_symptoms_report(self, sample_params: dict) -> None:
        """全部 12 症状输入 → 报告生成不抛异常。"""
        symptoms = [(s.value, 3) for s in Symptom]
        suggestion = generate_suggestion(
            symptoms=symptoms,
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka")
        assert len(report["parameters"]) == 23

    def test_telemetry_aware_report(
        self, sample_params: dict, sample_packet5: dict,
    ) -> None:
        """含遥测摘要的报告生成不抛异常。"""
        telemetry = extract_telemetry_summary({
            1: {"m_weather": 1, "m_trackTemperature": 30, "m_trackId": 2},
            6: {"m_speed": 300, "m_engineRPM": 11000},
        })
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
            telemetry=telemetry,
        )
        report = build_report(suggestion, track_id="suzuka")
        assert report["confidence"] in {"high", "medium", "low"}

    def test_full_packet5_extraction(self, sample_packet5: dict) -> None:
        """完整 Packet 5 → 23 参数提取 → 可作为 CarSetup.from_dict 输入。"""
        params = extract_setup_from_packet5(sample_packet5)
        setup = CarSetup.from_dict(params)
        # 验证可构造合法 CarSetup
        assert setup.front_wing == 6.0
        assert setup.rear_wing == 4.0

    def test_report_roundtrip_via_json(self, sample_params: dict) -> None:
        """报告 JSON 往返：build_report → json.dumps → json.loads 结构一致。"""
        suggestion = generate_suggestion(
            symptoms=[("understeer", 3)],
            current_setup=sample_params,
            track_id="suzuka",
        )
        report = build_report(suggestion, track_id="suzuka", setup_id=1)
        text = json.dumps(report, ensure_ascii=False, sort_keys=True)
        restored = json.loads(text)
        # 顶层字段一致
        assert restored["track_id"] == report["track_id"]
        assert restored["setup_id"] == report["setup_id"]
        assert restored["confidence"] == report["confidence"]
        assert len(restored["parameters"]) == len(report["parameters"])

    def test_generated_at_is_utc(self) -> None:
        """generated_at 时间戳为 UTC（与本地时区无关）。"""
        report = build_report({}, track_id="suzuka")
        ts = report["generated_at"]
        # 解析为 naive datetime，验证接近当前 UTC 时间（±60 秒容差）
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta = abs((now - dt).total_seconds())
        assert delta < 60, f"generated_at 与当前 UTC 偏差 {delta}s 过大"