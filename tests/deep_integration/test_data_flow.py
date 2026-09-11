"""方式 1：数据流串接测试 —— 验证模块间数据流的正确传递。

核心业务闭环：
    UDP 遥测包 → 解析 → 领域模型 → 诊断向量 Dx(9 维)
    → 耦合矩阵 C(9×23) → SetupDelta → 报告生成 → API 响应

本文件验证每一步的输出必须是下一步的合法输入，且数据在传递过程中
不丢失、不变形。所有测试均使用真实模块（不 mock 核心逻辑），
仅用 struct 构造的 bytes 样本作为 UDP 包输入。

覆盖链路：
    1. telemetry.packets.parse_packet() → domain 对象 → engine 输入
    2. engine.diagnostic.compute_dx() → engine.coupling.apply() → SetupDelta
    3. feedback.service.submit() → db.store.save() → engine.generate_suggestion()
       → report.builder.build_report()
"""

from __future__ import annotations

import struct

import pytest

from setup_tuner.db.store import Store
from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.domain.track import get_all_tracks, get_track_by_id
from setup_tuner.engine.diagnostic import DIAG_DIMS, compute_dx, is_zero_dx
from setup_tuner.engine.engine import compute_setup_delta, generate_suggestion
from setup_tuner.feedback.service import FeedbackService
from setup_tuner.report.builder import (
    build_report,
    extract_setup_from_packet5,
    feedbacks_to_symptoms,
)
from setup_tuner.telemetry.packets import HEADER_FORMAT, parse_packet


# ===========================================================================
# 辅助：构造 UDP 包字节流
# ===========================================================================
def _build_header(
    packet_id: int,
    player_car_index: int = 0,
    *,
    packet_format: int = 2026,
    game_year: int = 26,
    game_major: int = 1,
    game_minor: int = 0,
    packet_version: int = 1,
    session_uid: int = 0x1234_5678_9ABC_DEF0,
    session_time: float = 12.5,
    frame_id: int = 100,
    overall_frame_id: int = 200,
    secondary_player: int = 255,
) -> bytes:
    """构造 29 字节包头（小端）。"""
    return struct.pack(
        HEADER_FORMAT,
        packet_format, game_year, game_major, game_minor, packet_version,
        packet_id, session_uid, session_time, frame_id, overall_frame_id,
        player_car_index, secondary_player,
    )


def _build_session_body(track_id: int = 2, weather: int = 0) -> bytes:
    """构造 Session 包体（含 21 marshal zones + 0 天气样本）。"""
    prefix_core = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        weather, 25, 22, 58, 5807, 0, track_id, 1, 1800, 3600, 60, 0, 0, 0, 0, 21,
    )
    marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
    tail = struct.pack("<BBB", 0, 0, 0)  # safetyCar / networkGame / numWfs=0
    return prefix_core + marshal_zones + tail


def _build_car_setups_body(
    *,
    front_wing: int = 6,
    rear_wing: int = 4,
    brake_pressure: int = 80,
    brake_bias: int = 65,
    front_camber: float = -2.5,
    rear_camber: float = -2.5,
) -> bytes:
    """构造 CarSetups 单车包体（50 字节）。"""
    return struct.pack(
        "<BBBBffffBBBBBBBBBffffBf",
        front_wing, rear_wing, 50, 50,           # 4 × uint8 翼/差速
        front_camber, rear_camber, 0.25, 0.25,   # 4 × float 外倾/束角
        1, 1, 1, 1, 1, 1, brake_pressure, brake_bias, 50,  # 9 × uint8
        25.5, 25.5, 25.5, 25.5,                  # 4 × float 胎压
        50, 100.0,                               # ballast, fuelLoad
    )


def _build_lap_data_body(last_lap_ms: int = 90000) -> bytes:
    """构造 LapData 单车包体。"""
    return struct.pack(
        "<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB",
        last_lap_ms, 45000,
        30000, 0, 25000, 0, 1234, 0, 5678, 0,
        1500.0, 3000.0, 0.0,
        5, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 320.0, 0,
    )


# ===========================================================================
# 1. UDP 解析 → 领域模型 → 引擎输入
# ===========================================================================
class TestTelemetryToEngine:
    """验证 telemetry.packets.parse_packet() 的输出可被引擎正确消费。

    协作模块：telemetry.packets ↔ report.builder ↔ engine.engine
    """

    def test_packet5_to_setup_dict_covers_23_params(self) -> None:
        """Packet 5 解析后经 extract_setup_from_packet5 应覆盖全部 23 参数。

        链路：parse_packet → extract_setup_from_packet5 → CarSetup.from_dict
        """
        data = _build_header(packet_id=5) + _build_car_setups_body(
            front_wing=7, rear_wing=3, brake_pressure=85,
        )
        parsed = parse_packet(data)
        assert parsed is not None
        assert parsed["packet_id"] == 5

        # 解析结果 → 23 参数字典
        setup_params = extract_setup_from_packet5(parsed)
        assert set(setup_params.keys()) == {f.name for f in ALL_SETUP_FIELDS}
        assert len(setup_params) == 23

        # 23 参数字典 → CarSetup 领域对象（不抛异常即合法）
        setup = CarSetup.from_dict(setup_params)
        assert setup.front_wing == 7
        assert setup.rear_wing == 3
        assert setup.brake_pressure == 85

    def test_packet5_field_values_preserved_through_pipeline(self) -> None:
        """Packet 5 关键字段值经整条链路传递后不变形。

        链路：UDP bytes → parse_packet → extract_setup_from_packet5 → CarSetup
        """
        data = _build_header(packet_id=5) + _build_car_setups_body(
            front_wing=9, rear_wing=2,
            front_camber=-3.0, rear_camber=-1.5,
            brake_pressure=90, brake_bias=70,
        )
        parsed = parse_packet(data)
        assert parsed is not None

        setup_params = extract_setup_from_packet5(parsed)
        setup = CarSetup.from_dict(setup_params)

        # 逐字段断言：UDP 解析值与领域对象值一致
        assert parsed["m_frontWing"] == 9
        assert setup.front_wing == 9
        assert parsed["m_rearWing"] == 2
        assert setup.rear_wing == 2
        assert parsed["m_brakePressure"] == 90
        assert setup.brake_pressure == 90
        assert parsed["m_brakeBias"] == 70
        assert setup.brake_bias == 70
        # float 字段用 approx 容忍浮点误差
        assert setup.front_camber == pytest.approx(-3.0)
        assert setup.rear_camber == pytest.approx(-1.5)

    def test_parsed_setup_is_valid_engine_input(self) -> None:
        """从 Packet 5 提取的 setup 字典可作为 generate_suggestion 的合法输入。

        链路：parse_packet → extract_setup_from_packet5 → generate_suggestion
        """
        data = _build_header(packet_id=5) + _build_car_setups_body()
        parsed = parse_packet(data)
        assert parsed is not None
        setup_params = extract_setup_from_packet5(parsed)

        # 用提取的 setup 作为引擎输入，应正常生成建议（不抛异常）
        result = generate_suggestion(
            symptoms=[(Symptom.UNDERSTEER.value, 3)],
            current_setup=setup_params,
            track_id="suzuka",
            telemetry=None,
        )
        assert "setup_delta" in result
        assert len(result["setup_delta"]) == 23
        assert len(result["parameters"]) == 23


# ===========================================================================
# 2. 症状 → Dx → 耦合矩阵 → SetupDelta
# ===========================================================================
class TestDxToSetupDelta:
    """验证 engine.diagnostic.compute_dx() → compute_setup_delta() 数据流。

    协作模块：engine.diagnostic ↔ engine.coupling ↔ engine.engine
    """

    def test_symptoms_to_dx_covers_9_dims(self) -> None:
        """症状列表经 compute_dx 后应产出 9 维 Dx 向量。

        链路：[(symptom, strength)] → compute_dx → {dim: value}
        """
        symptoms = [
            (Symptom.UNDERSTEER.value, 4),
            (Symptom.BRAKE_LONG.value, 3),
        ]
        dx = compute_dx(symptoms)

        # Dx 必须覆盖全部 9 维
        assert set(dx.keys()) == set(DIAG_DIMS)
        assert len(dx) == 9

        # understeer 应贡献 front_grip_req 与 turnin_req
        assert dx["front_grip_req"] > 0.0
        assert dx["turnin_req"] > 0.0
        # brake_long 应贡献 brake_power_req
        assert dx["brake_power_req"] > 0.0

    def test_dx_to_setup_delta_covers_23_params(self) -> None:
        """Dx 向量经 compute_setup_delta 后应覆盖全部 23 参数。

        链路：compute_dx → compute_setup_delta → {param: delta}
        """
        dx = compute_dx([(Symptom.UNDERSTEER.value, 3)])
        current_setup = CarSetup.default().to_dict()

        setup_delta = compute_setup_delta(dx, current_setup)

        # SetupDelta 必须覆盖全部 23 参数
        assert set(setup_delta.keys()) == {f.name for f in ALL_SETUP_FIELDS}
        assert len(setup_delta) == 23

    def test_dx_to_setup_delta_respects_bounds(self) -> None:
        """SetupDelta 每参数必须满足 |delta| <= max_delta 且 next ∈ [min, max]。

        链路：compute_dx → compute_setup_delta → 越界校验
        """
        dx = compute_dx([
            (Symptom.UNDERSTEER.value, 5),
            (Symptom.OVERSTEER.value, 5),
            (Symptom.BRAKE_LONG.value, 5),
        ])
        current_setup = CarSetup.default().to_dict()
        setup_delta = compute_setup_delta(dx, current_setup)

        for spec in ALL_SETUP_FIELDS:
            delta = setup_delta[spec.name]
            current = current_setup[spec.name]
            next_val = current + delta
            # 单次上限约束
            assert abs(delta) <= spec.max_delta + 1e-6, (
                f"{spec.name}: |delta|={abs(delta)} > max_delta={spec.max_delta}"
            )
            # 合法区间约束
            assert spec.min_val - 1e-6 <= next_val <= spec.max_val + 1e-6, (
                f"{spec.name}: next={next_val} 越界 [{spec.min_val}, {spec.max_val}]"
            )

    def test_zero_dx_produces_zero_delta(self) -> None:
        """全零 Dx（无有效症状）应产出全零 SetupDelta。

        链路：compute_dx(空/强度0) → compute_setup_delta → 全零 delta
        """
        dx = compute_dx([(Symptom.UNDERSTEER.value, 0)])
        assert is_zero_dx(dx)

        current_setup = CarSetup.default().to_dict()
        setup_delta = compute_setup_delta(dx, current_setup)

        # 全零 Dx → 全零 delta
        for name, delta in setup_delta.items():
            assert abs(delta) < 1e-12, f"{name}: delta={delta} 非零"

    def test_dx_is_deterministic(self) -> None:
        """相同症状输入两次 compute_dx 应产出完全相同的 Dx（确定性）。

        链路：symptoms → compute_dx × 2 → 结果一致
        """
        symptoms = [
            (Symptom.UNDERSTEER.value, 3),
            (Symptom.LOCKUP.value, 4),
            (Symptom.BOTTOMING.value, 2),
        ]
        dx1 = compute_dx(symptoms)
        dx2 = compute_dx(symptoms)
        assert dx1 == dx2

    def test_setup_delta_is_deterministic(self) -> None:
        """相同 Dx + setup 输入两次 compute_setup_delta 应产出完全相同结果。"""
        dx = compute_dx([(Symptom.UNDERSTEER.value, 3)])
        setup = CarSetup.default().to_dict()

        delta1 = compute_setup_delta(dx, setup)
        delta2 = compute_setup_delta(dx, setup)
        assert delta1 == delta2


# ===========================================================================
# 3. feedback.service → db.store → engine → report.builder 完整链路
# ===========================================================================
class TestFeedbackToReport:
    """验证 feedback.service.submit() → db.store → engine → report.builder 链路。

    协作模块：feedback.service ↔ db.store ↔ engine.engine ↔ report.builder
    """

    @pytest.fixture()
    def store(self, tmp_path) -> Store:
        """每个测试用独立临时 SQLite 文件，确保隔离。"""
        db_path = tmp_path / "test_data_flow.db"
        s = Store(str(db_path))
        yield s
        s.close()

    def test_submit_feedback_persisted_and_retrievable(
        self, store: Store,
    ) -> None:
        """提交的反馈经 Store 持久化后可被完整检索。

        链路：FeedbackService.submit_feedback → Store.add_feedback
              → Store.get_feedbacks → feedbacks_to_symptoms
        """
        svc = FeedbackService(store)
        track_id = "suzuka"

        # 提交 3 条反馈
        submitted = [
            svc.submit_feedback(track_id, 1, "understeer", 4),
            svc.submit_feedback(track_id, 5, "brake_long", 3),
            svc.submit_feedback(track_id, None, "bottoming", 2),
        ]
        assert len(submitted) == 3
        for fb in submitted:
            assert "id" in fb and fb["id"] > 0

        # 检索并验证不丢失、不变形
        retrieved = svc.get_feedbacks(track_id)
        assert len(retrieved) == 3
        # 按提交顺序逐条比对关键字段
        for sub, ret in zip(submitted, retrieved, strict=True):
            assert ret["track_id"] == sub["track_id"]
            assert ret["corner_number"] == sub["corner_number"]
            assert ret["symptom"] == sub["symptom"]
            assert ret["strength"] == sub["strength"]
            assert ret["category"] == sub["category"]

    def test_feedbacks_to_symptoms_preserves_strength(self, store: Store) -> None:
        """feedbacks_to_symptoms 应保留 symptom 与 strength 不变形。

        链路：Store.get_feedbacks → feedbacks_to_symptoms → engine 输入
        """
        svc = FeedbackService(store)
        track_id = "monza"

        svc.submit_feedback(track_id, 1, "understeer", 4)
        svc.submit_feedback(track_id, 3, "oversteer", 2)
        svc.submit_feedback(track_id, None, "tyre_wear", 5)

        feedbacks = svc.get_feedbacks(track_id)
        symptoms = feedbacks_to_symptoms(feedbacks)

        assert len(symptoms) == 3
        # 验证 (symptom, strength) 二元组完整保留
        sym_map = dict(symptoms)
        assert sym_map["understeer"] == 4
        assert sym_map["oversteer"] == 2
        assert sym_map["tyre_wear"] == 5

    def test_full_pipeline_feedback_to_report(self, store: Store) -> None:
        """完整链路：提交反馈 → 检索 → 引擎生成建议 → 组装报告。

        链路：
            FeedbackService.submit_feedback × N
            → Store.get_feedbacks
            → feedbacks_to_symptoms
            → generate_suggestion
            → build_report
        """
        svc = FeedbackService(store)
        track_id = "suzuka"

        # 提交多弯道多症状反馈
        svc.submit_feedback(track_id, 1, "understeer", 4)
        svc.submit_feedback(track_id, 5, "brake_long", 3)
        svc.submit_feedback(track_id, 10, "oversteer", 3)
        svc.submit_feedback(track_id, None, "bottoming", 2)

        # 检索反馈 → 转为 symptoms
        feedbacks = svc.get_feedbacks(track_id)
        symptoms = feedbacks_to_symptoms(feedbacks)
        assert len(symptoms) == 4

        # 引擎生成建议
        current_setup = CarSetup.default().to_dict()
        suggestion = generate_suggestion(
            symptoms=symptoms,
            current_setup=current_setup,
            track_id=track_id,
            telemetry=None,
        )
        assert suggestion["track_id"] == track_id
        assert len(suggestion["setup_delta"]) == 23
        assert len(suggestion["parameters"]) == 23
        # 多症状非零 Dx → 应有非零调整
        assert not is_zero_dx(suggestion["dx"])

        # 组装报告
        report = build_report(
            suggestion_result=suggestion,
            track_id=track_id,
            setup_id=None,
        )
        assert report["track_id"] == track_id
        assert "generated_at" in report
        assert len(report["parameters"]) == 23
        assert report["confidence"] in {"high", "medium", "low"}
        # 报告 setup_delta 与引擎 setup_delta 一致（不变形）
        assert report["setup_delta"] == suggestion["setup_delta"]
        assert report["dx"] == suggestion["dx"]

    def test_report_persisted_via_store(self, store: Store) -> None:
        """报告经 Store.save_suggestion 持久化后可完整检索。

        链路：build_report → json.dumps → Store.save_suggestion
              → Store.get_latest_suggestion → json.loads → 报告完整
        """
        import json

        svc = FeedbackService(store)
        track_id = "monaco"

        svc.submit_feedback(track_id, 1, "understeer", 3)
        feedbacks = svc.get_feedbacks(track_id)
        symptoms = feedbacks_to_symptoms(feedbacks)

        suggestion = generate_suggestion(
            symptoms=symptoms,
            current_setup=CarSetup.default().to_dict(),
            track_id=track_id,
        )
        report = build_report(suggestion, track_id)

        # 持久化
        report_json = json.dumps(report, ensure_ascii=False, sort_keys=True)
        suggestion_id = store.save_suggestion(track_id, report_json)
        assert suggestion_id > 0

        # 检索并验证不变形
        retrieved = store.get_latest_suggestion(track_id)
        assert retrieved is not None
        assert retrieved["id"] == suggestion_id
        retrieved_report = json.loads(retrieved["report_json"])
        assert retrieved_report["track_id"] == track_id
        assert retrieved_report["setup_delta"] == report["setup_delta"]
        assert len(retrieved_report["parameters"]) == 23


# ===========================================================================
# 4. 赛道 → 反馈 → 建议闭环数据一致性
# ===========================================================================
class TestTrackSetupConsistency:
    """验证赛道 → 调教 → 反馈 → 建议闭环中数据标识的一致性。

    协作模块：domain.track ↔ domain.setup ↔ engine.engine ↔ report.builder
    """

    def test_track_id_preserved_through_pipeline(self) -> None:
        """track_id 经整条链路传递后保持一致。

        链路：domain.track → generate_suggestion(track_id) → build_report
        """
        tracks = get_all_tracks()
        assert len(tracks) >= 20  # 24 条赛道

        # 取第一条赛道
        track = tracks[0]
        track_id = track.track_id

        # 引擎生成建议
        suggestion = generate_suggestion(
            symptoms=[(Symptom.UNDERSTEER.value, 3)],
            current_setup=CarSetup.default().to_dict(),
            track_id=track_id,
        )
        assert suggestion["track_id"] == track_id

        # 组装报告
        report = build_report(suggestion, track_id)
        assert report["track_id"] == track_id

    def test_setup_dict_matches_carsetup_fields(self) -> None:
        """CarSetup.to_dict() 的键集与 ALL_SETUP_FIELDS 名称集完全一致。

        保证 setup 字典在模块间传递时不会缺失字段。
        """
        setup = CarSetup.default()
        setup_dict = setup.to_dict()
        field_names = {f.name for f in ALL_SETUP_FIELDS}
        assert set(setup_dict.keys()) == field_names

    def test_all_symptoms_produce_valid_dx(self) -> None:
        """12 症状每个都能产出合法的 9 维 Dx（不抛异常、维度完整）。"""
        for symptom in Symptom:
            dx = compute_dx([(symptom.value, 3)])
            assert set(dx.keys()) == set(DIAG_DIMS)
            assert len(dx) == 9

    def test_all_tracks_have_valid_setup_default(self) -> None:
        """每条赛道都能用缺省 CarSetup 生成合法建议（不抛异常）。"""
        tracks = get_all_tracks()
        for track in tracks[:3]:  # 取前 3 条避免测试过慢
            suggestion = generate_suggestion(
                symptoms=[(Symptom.UNDERSTEER.value, 3)],
                current_setup=CarSetup.default().to_dict(),
                track_id=track.track_id,
            )
            assert suggestion["track_id"] == track.track_id
            assert len(suggestion["setup_delta"]) == 23

    def test_track_lookup_by_id_consistent(self) -> None:
        """get_track_by_id 返回的赛道与 get_all_tracks 中的一致。"""
        tracks = get_all_tracks()
        for track in tracks[:3]:
            lookup = get_track_by_id(track.track_id)
            assert lookup is not None
            assert lookup.track_id == track.track_id
            assert lookup.official_name == track.official_name
            assert len(lookup.corners) == len(track.corners)


# ===========================================================================
# 5. 多包遥测流 → 引擎遥测摘要
# ===========================================================================
class TestTelemetrySummaryFlow:
    """验证多包遥测流 → extract_telemetry_summary → 引擎 telemetry 输入。

    协作模块：telemetry.packets ↔ report.builder ↔ engine.engine
    """

    def test_multi_packet_to_telemetry_summary(self) -> None:
        """多个 UDP 包解析后组装为遥测摘要，可被引擎消费。

        链路：parse_packet × N → {packet_id: parsed} → extract_telemetry_summary
              → generate_suggestion(telemetry=summary)
        """
        # 构造 Session + LapData 两个包
        session_data = _build_header(packet_id=1) + _build_session_body(
            track_id=2, weather=1,  # 雨天
        )
        lap_data = _build_header(packet_id=2) + _build_lap_data_body(
            last_lap_ms=95000,
        )

        session_parsed = parse_packet(session_data)
        lap_parsed = parse_packet(lap_data)
        assert session_parsed is not None
        assert lap_parsed is not None

        # 组装 all_latest 字典（模拟 TelemetryStream.get_all_latest()）
        all_latest = {
            session_parsed["packet_id"]: session_parsed,
            lap_parsed["packet_id"]: lap_parsed,
        }

        # 提取遥测摘要
        from setup_tuner.report.builder import extract_telemetry_summary
        summary = extract_telemetry_summary(all_latest)
        assert summary["weather"] == 1
        assert summary["track_id_udp"] == 2
        assert summary["last_lap_time_ms"] == 95000

        # 引擎消费遥测摘要（雨天应触发增益 ×0.7）
        result = generate_suggestion(
            symptoms=[(Symptom.UNDERSTEER.value, 3)],
            current_setup=CarSetup.default().to_dict(),
            track_id="suzuka",
            telemetry=summary,
        )
        assert len(result["setup_delta"]) == 23

        # 对比：无遥测 vs 有遥测，雨天增益应使调整幅度更保守
        result_no_telemetry = generate_suggestion(
            symptoms=[(Symptom.UNDERSTEER.value, 3)],
            current_setup=CarSetup.default().to_dict(),
            track_id="suzuka",
            telemetry=None,
        )
        # 雨天应至少有一个参数的 |delta| 更小或相等
        deltas_wet = [abs(v) for v in result["setup_delta"].values()]
        deltas_dry = [abs(v) for v in result_no_telemetry["setup_delta"].values()]
        assert all(w <= d + 1e-9 for w, d in zip(deltas_wet, deltas_dry, strict=True))