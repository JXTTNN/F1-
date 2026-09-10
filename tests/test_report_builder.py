"""报告组装模块单元测试。

覆盖验收标准：
1. format_linkages — 联动说明格式化（空/字符串/列表）
2. build_summary — 摘要生成（空/全零/有非零）
3. build_report — 报告组装主入口（结构/字段/时间戳）
4. extract_setup_from_packet5 — 从遥测包提取调教参数（含主动空力/胎压/clamp）
5. extract_telemetry_summary — 遥测摘要提取（各 packet_id）
6. feedbacks_to_symptoms — 反馈转症状列表（含空/无效过滤）
"""

from __future__ import annotations

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
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
# 1. format_linkages
# ===========================================================================
class TestFormatLinkages:
    """联动说明格式化。"""

    def test_empty_linkages(self) -> None:
        """空 linkages 返回空列表。"""
        assert format_linkages({"linkages": []}) == []
        assert format_linkages({}) == []
        assert format_linkages({"linkages": None}) == []

    def test_string_linkages(self) -> None:
        """字符串 linkages 返回单元素列表。"""
        result = format_linkages({"linkages": "front_grip_req(前轴抓地)"})
        assert result == ["front_grip_req(前轴抓地)"]

    def test_list_linkages(self) -> None:
        """列表 linkages 原样返回（确保为 str）。"""
        result = format_linkages({
            "linkages": [
                "front_grip_req(前轴抓地) Dx=+0.80 × C=+1.50",
                "turnin_req(入弯响应) Dx=+0.30 × C=+0.60",
            ],
        })
        assert len(result) == 2
        assert "front_grip_req" in result[0]

    def test_non_string_items_converted(self) -> None:
        """非字符串元素转为 str。"""
        result = format_linkages({"linkages": [123, 45.6]})
        assert result == ["123", "45.6"]


# ===========================================================================
# 2. build_summary
# ===========================================================================
class TestBuildSummary:
    """摘要生成。"""

    def test_empty_parameters(self) -> None:
        """空参数列表返回特定摘要。"""
        assert build_summary([]) == "本次建议无参数"

    def test_all_zero_delta(self) -> None:
        """全零 delta 返回「均无需调整」。"""
        params = [
            {"param": "front_wing", "setup_delta": 0.0},
            {"param": "rear_wing", "setup_delta": 0.0},
        ]
        summary = build_summary(params)
        assert "2 项参数" in summary
        assert "均无需调整" in summary

    def test_with_nonzero_delta(self) -> None:
        """有非零 delta 返回「非零调整」。"""
        params = [
            {"param": "front_wing", "setup_delta": 2.0},
            {"param": "rear_wing", "setup_delta": 0.0},
            {"param": "brake_pressure", "setup_delta": -1.0},
        ]
        summary = build_summary(params)
        assert "3 项参数" in summary
        assert "2 项非零调整" in summary

    def test_missing_setup_delta_field(self) -> None:
        """缺 setup_delta 字段视为 0。"""
        params = [{"param": "front_wing"}, {"param": "rear_wing", "setup_delta": 1.0}]
        summary = build_summary(params)
        assert "2 项参数" in summary
        assert "1 项非零调整" in summary


# ===========================================================================
# 3. build_report
# ===========================================================================
class TestBuildReport:
    """报告组装主入口。"""

    @staticmethod
    def _make_suggestion_result() -> dict:
        """构造一个合法的 suggestion_result。"""
        default_setup = CarSetup.default().to_dict()
        return generate_suggestion(
            [("understeer", 3)], default_setup, "suzuka", None,
        )

    def test_report_structure(self) -> None:
        """报告结构对齐 design 2.7.7。"""
        result = self._make_suggestion_result()
        report = build_report(result, track_id="suzuka", setup_id=1)

        assert report["track_id"] == "suzuka"
        assert report["setup_id"] == 1
        assert "generated_at" in report
        assert "parameters" in report
        assert "summary" in report
        assert "confidence" in report
        assert "setup_delta" in report
        assert "dx" in report

    def test_report_generated_at_is_iso8601(self) -> None:
        """generated_at 应为 ISO8601 格式（带 Z 后缀）。"""
        result = self._make_suggestion_result()
        report = build_report(result, track_id="suzuka")
        ts = report["generated_at"]
        assert ts.endswith("Z")
        assert "T" in ts
        # 格式 YYYY-MM-DDTHH:MM:SSZ
        assert len(ts) == 20

    def test_report_parameters_count_23(self) -> None:
        """参数列表 23 项。"""
        result = self._make_suggestion_result()
        report = build_report(result, track_id="suzuka")
        assert len(report["parameters"]) == 23

    def test_report_param_entry_fields(self) -> None:
        """每参数项含必要字段。"""
        result = self._make_suggestion_result()
        report = build_report(result, track_id="suzuka")
        for p in report["parameters"]:
            assert "param" in p
            assert "current" in p
            assert "setup_delta" in p
            assert "linkages" in p
            assert "linked_notes" in p
            assert "source" in p
            assert "confidence" in p

    def test_report_tradeoff_only_when_nonzero(self) -> None:
        """tradeoff 仅在非零 delta 且有副作用时出现。"""
        # 空症状 → 全零 delta → 无 tradeoff
        default_setup = CarSetup.default().to_dict()
        result = generate_suggestion([], default_setup, "suzuka", None)
        report = build_report(result, track_id="suzuka")
        for p in report["parameters"]:
            assert "tradeoff" not in p or p.get("tradeoff") is None

    def test_report_confidence_from_engine(self) -> None:
        """confidence 取引擎结果。"""
        result = self._make_suggestion_result()
        report = build_report(result, track_id="suzuka")
        assert report["confidence"] == result["confidence"]

    def test_report_setup_id_none(self) -> None:
        """setup_id=None 时报告含 setup_id=None。"""
        result = self._make_suggestion_result()
        report = build_report(result, track_id="suzuka", setup_id=None)
        assert report["setup_id"] is None

    def test_report_empty_suggestion(self) -> None:
        """空 suggestion_result 不崩溃。"""
        report = build_report({}, track_id="empty")
        assert report["track_id"] == "empty"
        assert report["parameters"] == []
        assert report["confidence"] == "medium"
        assert report["setup_delta"] == {}
        assert report["dx"] == {}


# ===========================================================================
# 4. extract_setup_from_packet5
# ===========================================================================
class TestExtractSetupFromPacket5:
    """从遥测 CarSetups 包提取 23 参数快照。"""

    def test_extract_all_23_params(self) -> None:
        """提取结果覆盖全部 23 参数。"""
        packet5 = {
            "m_frontWing": 7,
            "m_rearWing": 6,
            "m_onThrottleDiff": 50,
            "m_offThrottleDiff": 50,
            "m_frontCamber": -3.0,
            "m_rearCamber": -2.5,
            "m_frontToe": 0.25,
            "m_rearToe": 0.25,
            "m_frontSuspension": 300,
            "m_rearSuspension": 250,
            "m_frontAntiRollBar": 200,
            "m_rearAntiRollBar": 200,
            "m_frontSuspensionHeight": 20,
            "m_rearSuspensionHeight": 20,
            "m_brakePressure": 80,
            "m_brakeBias": 60,
            "m_engineBraking": 50,
            "m_ballast": 50,
        }
        params = extract_setup_from_packet5(packet5)
        assert len(params) == 23
        assert params["front_wing"] == 7.0
        assert params["rear_wing"] == 6.0
        assert params["brake_pressure"] == 80.0

    def test_extract_missing_fields_use_default(self) -> None:
        """缺失字段取 SetupField.default。"""
        params = extract_setup_from_packet5({})
        defaults = CarSetup.default().to_dict()
        for name, val in defaults.items():
            assert params[name] == val, f"{name}: {params[name]} != {val}"

    def test_extract_active_aero_mode_0(self) -> None:
        """m_activeAeroMode=0 (Z/弯道) → active_aero_z=0.6, active_aero_x=0.4。"""
        params = extract_setup_from_packet5({"m_activeAeroMode": 0})
        assert params["active_aero_z"] == 0.6
        assert params["active_aero_x"] == 0.4

    def test_extract_active_aero_mode_1(self) -> None:
        """m_activeAeroMode=1 (X/直道) → active_aero_z=0.4, active_aero_x=0.6。"""
        params = extract_setup_from_packet5({"m_activeAeroMode": 1})
        assert params["active_aero_z"] == 0.4
        assert params["active_aero_x"] == 0.6

    def test_extract_tyre_pressure_from_list(self) -> None:
        """从 tyresPressure[4] 取前轴/后轴均值。"""
        packet5 = {
            "tyresPressure": [25.0, 26.0, 27.0, 28.0],
        }
        params = extract_setup_from_packet5(packet5)
        # front = (25+26)/2 = 25.5, rear = (27+28)/2 = 27.5
        assert params["front_tyre_pressure"] == 25.5
        assert params["rear_tyre_pressure"] == 27.5

    def test_extract_tyre_pressure_from_m_field(self) -> None:
        """从 m_tyresPressure&lsqb;4&rsqb; 取胎压。"""
        packet5 = {
            "m_tyresPressure": [24.0, 26.0, 28.0, 30.0],
        }
        params = extract_setup_from_packet5(packet5)
        assert params["front_tyre_pressure"] == 25.0
        assert params["rear_tyre_pressure"] == 29.0

    def test_extract_clamp_to_valid_range(self) -> None:
        """UDP 值越界时 clamp 到合法区间。"""
        params = extract_setup_from_packet5({
            "m_frontWing": 999,  # 越界（max=11）
            "m_brakePressure": -50,  # 越界（min=50）
        })
        spec_fw = next(f for f in ALL_SETUP_FIELDS if f.name == "front_wing")
        spec_bp = next(f for f in ALL_SETUP_FIELDS if f.name == "brake_pressure")
        assert params["front_wing"] == spec_fw.max_val
        assert params["brake_pressure"] == spec_bp.min_val

    def test_extract_tyre_pressure_invalid_ignored(self) -> None:
        """胎压列表不足 4 个时取缺省。"""
        params = extract_setup_from_packet5({"tyresPressure": [25.0, 26.0]})
        defaults = CarSetup.default().to_dict()
        assert params["front_tyre_pressure"] == defaults["front_tyre_pressure"]


# ===========================================================================
# 5. extract_telemetry_summary
# ===========================================================================
class TestExtractTelemetrySummary:
    """遥测摘要提取。"""

    def test_empty_input(self) -> None:
        """空输入返回空字典。"""
        assert extract_telemetry_summary({}) == {}

    def test_session_packet(self) -> None:
        """Packet 1 Session 提取天气/温度。"""
        summary = extract_telemetry_summary({
            1: {"m_weather": 2, "m_trackTemperature": 30, "m_airTemperature": 25, "m_trackId": 7},
        })
        assert summary["weather"] == 2
        assert summary["track_temp"] == 30
        assert summary["air_temp"] == 25
        assert summary["track_id_udp"] == 7

    def test_telemetry_packet(self) -> None:
        """Packet 6 CarTelemetry 提取速度/油门等。"""
        summary = extract_telemetry_summary({
            6: {
                "m_speed": 300, "m_throttle": 0.8, "m_brake": 0.0,
                "m_gear": 7, "m_engineRPM": 12000,
            },
        })
        assert summary["speed"] == 300
        assert summary["throttle"] == 0.8
        assert summary["brake"] == 0.0
        assert summary["gear"] == 7
        assert summary["engine_rpm"] == 12000

    def test_status_packet(self) -> None:
        """Packet 7 CarStatus 提取轮胎配方/胎龄/燃油。"""
        summary = extract_telemetry_summary({
            7: {
                "m_visualTyreCompound": 3, "m_tyresAgeLaps": 5, "m_fuelInTank": 50.0,
            },
        })
        assert summary["tyre_compound"] == 3
        assert summary["tyres_age_laps"] == 5
        assert summary["fuel_in_tank"] == 50.0

    def test_lap_data_packet(self) -> None:
        """Packet 2 LapData 提取圈速/扇区/距离。"""
        summary = extract_telemetry_summary({
            2: {
                "m_lapDistance": 1500.0, "m_sector": 1,
                "m_currentLapNum": 3, "m_lastLapTimeInMS": 90000,
            },
        })
        assert summary["lap_distance"] == 1500.0
        assert summary["sector"] == 1
        assert summary["current_lap_num"] == 3
        assert summary["last_lap_time_ms"] == 90000

    def test_all_packets_combined(self) -> None:
        """所有包同时存在时合并提取。"""
        summary = extract_telemetry_summary({
            1: {"m_weather": 1, "m_trackTemperature": 20, "m_airTemperature": 18, "m_trackId": 0},
            2: {"m_lapDistance": 500.0, "m_sector": 0, "m_currentLapNum": 1, "m_lastLapTimeInMS": 60000},
            6: {"m_speed": 250, "m_throttle": 1.0, "m_brake": 0.0, "m_gear": 6, "m_engineRPM": 11000},
            7: {"m_visualTyreCompound": 2, "m_tyresAgeLaps": 3, "m_fuelInTank": 80.0},
        })
        assert summary["weather"] == 1
        assert summary["speed"] == 250
        assert summary["tyre_compound"] == 2
        assert summary["lap_distance"] == 500.0


# ===========================================================================
# 6. feedbacks_to_symptoms
# ===========================================================================
class TestFeedbacksToSymptoms:
    """反馈记录转症状列表。"""

    def test_empty_feedbacks(self) -> None:
        """空列表返回空。"""
        assert feedbacks_to_symptoms([]) == []

    def test_single_feedback(self) -> None:
        """单条反馈转换。"""
        result = feedbacks_to_symptoms([
            {"symptom": "understeer", "strength": 3},
        ])
        assert result == [("understeer", 3)]

    def test_multiple_feedbacks(self) -> None:
        """多条反馈转换。"""
        result = feedbacks_to_symptoms([
            {"symptom": "understeer", "strength": 3},
            {"symptom": "oversteer", "strength": 4},
            {"symptom": "tyre_wear", "strength": 5},
        ])
        assert len(result) == 3
        assert result[0] == ("understeer", 3)
        assert result[1] == ("oversteer", 4)
        assert result[2] == ("tyre_wear", 5)

    def test_filter_invalid_entries(self) -> None:
        """过滤无 symptom 或无 strength 的条目。"""
        result = feedbacks_to_symptoms([
            {"symptom": "understeer", "strength": 3},
            {"symptom": None, "strength": 3},  # 无 symptom
            {"symptom": "oversteer", "strength": None},  # 无 strength
            {"symptom": "", "strength": 2},  # 空 symptom
        ])
        assert len(result) == 1
        assert result[0] == ("understeer", 3)

    def test_strength_converted_to_int(self) -> None:
        """strength 转为 int。"""
        result = feedbacks_to_symptoms([
            {"symptom": "understeer", "strength": "3"},  # 字符串
        ])
        assert result[0] == ("understeer", 3)
        assert isinstance(result[0][1], int)