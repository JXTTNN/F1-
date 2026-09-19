"""Packet 7 (CarStatus) 端到端可达性测试。

背景：``parse_car_status`` 自项目早期即已实现，但 ``app._on_packet``
从未分发 ``packet_id == 7``，聚合器也没有 ``on_car_status`` 方法 ——
轮胎配方 / 胎龄 / 燃油 / ERS / 刹车平衡这些字段**解析出来即被丢弃**，
与 task-65 的「规则9 空实现」属同一类缺陷。

本文件锁定修复后的完整链路：
    parse_car_status → _on_packet(7) → LapAggregator.on_car_status
    → snapshot 字段 → engine 配方相关阈值 / 刹车平衡一致性规则
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config
from setup_tuner.engine.engine import _derive_telemetry_dx, generate_suggestion
from setup_tuner.telemetry.lap_aggregator import (
    LapAggregator,
    _compound_label,
)
from setup_tuner.telemetry.packets import HEADER_FORMAT, parse_packet


def _isolated_app(tmp_path: Path):
    """构建使用隔离数据目录的 app。

    测试**不得**读写仓库的 ``./data``：那会让断言依赖本机残留数据
    （本地绿 / CI 红），并反过来污染用户真实数据库。
    """
    return create_app(Config(data_dir=str(tmp_path / "data")))


# ===========================================================================
# 1. 模块级辅助
# ===========================================================================
class TestCompoundHelpers:
    """配方代码 → 名称映射。"""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (16, "C5"), (17, "C4"), (18, "C3"), (19, "C2"),
            (20, "C1"), (21, "C0"), (22, "C6"),
            (7, "Inter"), (8, "Wet"),
            (9, "Soft(Classic)"), (10, "Medium(Classic)"), (11, "Hard(Classic)"),
        ],
    )
    def test_known_codes(self, code: int, expected: str) -> None:
        assert _compound_label(code) == expected

    def test_unknown_code_is_descriptive(self) -> None:
        assert _compound_label(999) == "Unknown(999)"

    def test_none_is_unknown(self) -> None:
        assert _compound_label(None) == "Unknown"


# ===========================================================================
# 2. 聚合器：on_car_status 累积
# ===========================================================================
def _status_frame(**over: object) -> dict:
    frame = {
        "m_actualTyreCompound": 18,
        "m_visualTyreCompound": 18,
        "m_tyresAgeLaps": 3,
        "m_fuelInTank": 82.5,
        "m_fuelRemainingLaps": 34.4,
        "m_frontBrakeBias": 57.5,
        "m_ersStoreEnergy": 2_500_000.0,
        "m_ersDeployMode": 2,
        "m_tractionControl": 1,
        "m_antiLockBrakes": 1,
        "m_fuelMix": 1,
        "m_maxRPM": 15000,
        "m_drsAllowed": 1,
    }
    frame.update(over)
    return frame


class TestOnCarStatusAccumulation:
    """on_car_status 累积与快照。"""

    def test_single_frame_populates_snapshot(self) -> None:
        agg = LapAggregator()
        agg.on_car_status(_status_frame())
        snap = agg.snapshot()
        assert snap["car_status_frames"] == 1
        assert snap["tyre_compound"] == 18
        assert snap["tyre_compound_name"] == "C3"
        assert snap["visual_tyre_compound"] == 18
        assert snap["tyres_age_laps"] == 3
        assert snap["fuel_in_tank"] == pytest.approx(82.5, abs=1e-6)
        assert snap["fuel_remaining_laps"] == pytest.approx(34.4, abs=1e-6)
        assert snap["front_brake_bias"] == pytest.approx(57.5, abs=1e-6)
        assert snap["ers_store_energy"] == pytest.approx(2_500_000.0, abs=1e-3)
        assert snap["ers_deploy_mode"] == 2
        assert snap["traction_control"] == 1
        assert snap["anti_lock_brakes"] == 1
        assert snap["fuel_mix"] == 1
        assert snap["max_rpm"] == 15000
        assert snap["drs_allowed"] == 1

    def test_soft_compound_flag(self) -> None:
        """C5(16) 应标记为软胎 —— 引擎据此下调刹车温度阈值。"""
        agg = LapAggregator()
        agg.on_car_status(_status_frame(m_actualTyreCompound=16))
        snap = agg.snapshot()
        assert snap["is_soft_compound"] is True
        assert snap["is_hard_compound"] is False

    def test_hard_compound_flag(self) -> None:
        """C1(20) 应标记为硬胎 —— 引擎据此上调刹车温度阈值。"""
        agg = LapAggregator()
        agg.on_car_status(_status_frame(m_actualTyreCompound=20))
        snap = agg.snapshot()
        assert snap["is_hard_compound"] is True
        assert snap["is_soft_compound"] is False

    def test_medium_compound_neither_flag(self) -> None:
        """C3(18) 为中性，两个标记都应为假。"""
        agg = LapAggregator()
        agg.on_car_status(_status_frame(m_actualTyreCompound=18))
        snap = agg.snapshot()
        assert snap["is_soft_compound"] is False
        assert snap["is_hard_compound"] is False

    def test_last_valid_value_wins(self) -> None:
        """后续帧的合法值覆盖旧值。"""
        agg = LapAggregator()
        agg.on_car_status(_status_frame(m_actualTyreCompound=18))
        agg.on_car_status(_status_frame(m_actualTyreCompound=16))
        assert agg.snapshot()["tyre_compound"] == 16

    def test_invalid_values_keep_previous(self) -> None:
        """非法值（None/bool/字符串）不覆盖上一次有效值。"""
        agg = LapAggregator()
        agg.on_car_status(_status_frame(m_actualTyreCompound=18, m_tyresAgeLaps=3))
        agg.on_car_status(_status_frame(
            m_actualTyreCompound=None, m_tyresAgeLaps=True,
            m_fuelInTank="not a number", m_frontBrakeBias=None,
        ))
        snap = agg.snapshot()
        assert snap["tyre_compound"] == 18, "None 不应覆盖有效配方"
        assert snap["tyres_age_laps"] == 3, "bool 不应被当作整数接受"
        assert snap["fuel_in_tank"] == pytest.approx(82.5, abs=1e-6)
        assert snap["front_brake_bias"] == pytest.approx(57.5, abs=1e-6)
        assert snap["car_status_frames"] == 2, "帧数仍应累加"

    def test_no_status_frames_no_snapshot_keys(self) -> None:
        """从未收到 Packet 7 时，快照不含这些键（避免用 0 冒充真实值）。"""
        agg = LapAggregator()
        agg.on_telemetry({"m_speed": 100.0})
        snap = agg.snapshot()
        for key in ("car_status_frames", "tyre_compound", "fuel_in_tank",
                    "front_brake_bias", "is_soft_compound"):
            assert key not in snap, f"{key} 不应在无数据时出现"

    def test_reset_clears_status(self) -> None:
        """跨圈重置后状态清空（新一圈重新累积）。"""
        agg = LapAggregator()
        agg.on_lap_data({"m_currentLapNum": 1})
        agg.on_car_status(_status_frame(m_actualTyreCompound=16))
        assert agg.snapshot()["tyre_compound"] == 16
        # 圈号变化 → 固化并重置
        agg.on_lap_data({"m_currentLapNum": 2})
        agg.on_telemetry({"m_speed": 200.0})
        snap = agg.snapshot()
        assert "tyre_compound" not in snap
        assert "car_status_frames" not in snap

    def test_thread_safety_smoke(self) -> None:
        """并发调用不崩且帧数守恒。"""
        import threading

        agg = LapAggregator()
        n_threads, per_thread = 8, 50
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(per_thread):
                agg.on_car_status(_status_frame())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert agg.snapshot()["car_status_frames"] == n_threads * per_thread


# ===========================================================================
# 3. parse_packet 分发
# ===========================================================================
def _build_status_packet(player_car: int = 0) -> bytes:
    """构造合法 Packet 7 字节流（22 车，只填玩家车）。

    复用 packets 模块自身的 ``_STATUS_PER_STRUCT`` 打包，避免手写格式串
    与解析侧漂移（手写副本一旦错位，测的是副本而非真实协议）。
    """
    from setup_tuner.telemetry import packets as pk

    header = struct.pack(
        HEADER_FORMAT, 2026, 26, 1, 0, 1, 7, 0x1234, 12.5, 100, 200,
        player_car, 255,
    )
    per = pk._STATUS_PER_STRUCT  # noqa: SLF001 — 测试需与解析侧同源
    blank = per.pack(
        0, 0, 0, 0, 0,                          # tc/abs/fuelMix/bias(uint8)/pitLimiter
        0.0, 0.0, 0.0,                          # fuel
        0, 0,                                   # rpm
        0, 0,                                   # maxGears, drsAllowed
        0,                                      # drsActivationDistance
        0, 0,                                   # compound
        0,                                      # tyresAgeLaps
        0,                                      # fiaFlags
        0.0, 0.0, 0.0,                          # power ICE/MGUK, ersStore
        0,                                      # ersDeployMode
        0.0, 0.0, 0.0, 0.0,                     # ers harvested/limit/deployed
        0,                                      # networkPaused
    )
    player = per.pack(
        1, 1, 1, 58, 0,                        # frontBrakeBias 是 uint8（整数百分比）
        82.5, 110.0, 34.4,
        15000, 4000,
        8, 1,
        500,
        18, 18,
        3,
        0,
        800.0, 120.0, 2_500_000.0,
        2,
        1.0, 1.0, 4_000_000.0, 1.0,
        0,
    )
    body = b"".join(
        player if i == player_car else blank for i in range(22)
    )
    return header + body


class TestParsePacketDispatch:
    """parse_packet 对 Packet 7 的分发。"""

    def test_packet_id_7_dispatched(self) -> None:
        parsed = parse_packet(_build_status_packet())
        assert parsed is not None
        assert parsed["packet_id"] == 7

    def test_player_fields_parsed(self) -> None:
        parsed = parse_packet(_build_status_packet(player_car=0))
        assert parsed["m_actualTyreCompound"] == 18
        assert parsed["m_visualTyreCompound"] == 18
        assert parsed["m_tyresAgeLaps"] == 3
        assert parsed["m_frontBrakeBias"] == 58
        assert parsed["m_fuelInTank"] == pytest.approx(82.5, abs=1e-4)
        assert parsed["m_maxRPM"] == 15000
        assert parsed["m_drsAllowed"] == 1

    def test_player_car_index_shifts_slice(self) -> None:
        """player_car_index=3 时取第 4 车的字段（而非第 1 车）。"""
        parsed = parse_packet(_build_status_packet(player_car=3))
        assert parsed["m_actualTyreCompound"] == 18
        assert parsed["m_frontBrakeBias"] == 58

    def test_short_packet_raises(self) -> None:
        from setup_tuner.telemetry.packets import PacketTooShortError

        with pytest.raises(PacketTooShortError):
            parse_packet(_build_status_packet()[:60])


# ===========================================================================
# 4. app 分发链路
# ===========================================================================
class TestAppDispatch:
    """_on_packet 把 Packet 7 交给聚合器。"""

    def test_car_status_reaches_aggregator(self, tmp_path: Path) -> None:
        app = _isolated_app(tmp_path)
        with TestClient(app) as c:  # 进入 lifespan 才会初始化 state
            handler = app.state.packet_handler
            agg = app.state.lap_aggregator
            before = agg.snapshot().get("car_status_frames", 0)
            handler(parse_packet(_build_status_packet()))
            after = agg.snapshot()
            assert after.get("car_status_frames", 0) == before + 1
            assert after["tyre_compound"] == 18
            del c

    def test_unknown_packet_id_is_ignored(self, tmp_path: Path) -> None:
        """不支持的 packet_id（如 99）不得抛异常。"""
        app = _isolated_app(tmp_path)
        with TestClient(app):
            app.state.packet_handler({"packet_id": 99, "name": "Nope"})


# ===========================================================================
# 5. 模拟器覆盖
# ===========================================================================
class TestSimulatorCoverage:
    """模拟器必须产出 Packet 7，否则无真实游戏时该链路永不激活。"""

    def test_simulator_emits_car_status(self, tmp_path: Path) -> None:
        app = _isolated_app(tmp_path)
        with TestClient(app) as c:
            r = c.post("/api/v1/telemetry/simulate",
                       json={"action": "start", "track_id": "suzuka"})
            assert r.status_code == 200
            snap: dict = {}
            for _ in range(100):
                snap = app.state.lap_aggregator.snapshot()
                if snap.get("car_status_frames", 0) > 0:
                    break
                time.sleep(0.1)
            c.post("/api/v1/telemetry/simulate",
                   json={"action": "stop", "track_id": "suzuka"})
        assert snap.get("car_status_frames", 0) > 0, "模拟器未产出 Packet 7"
        assert "tyre_compound" in snap
        assert "front_brake_bias" in snap


# ===========================================================================
# 6. 引擎消费：配方阈值与刹车平衡一致性
# ===========================================================================
class TestEngineBrakeRules:
    """刹车温度阈值随配方浮动 + 刹车平衡一致性。"""

    def _base(self, **over: object) -> dict:
        telemetry = {
            "m_brakesTemperature": [520.0, 520.0, 520.0, 520.0],
        }
        telemetry.update(over)
        return telemetry

    def test_neutral_compound_520_is_over_threshold(self) -> None:
        """中性配方：520°C > 500°C → 触发 brake_stab_req。"""
        dx = _derive_telemetry_dx(self._base())
        assert dx["brake_stab_req"] > 0.0

    def test_hard_compound_520_is_normal(self) -> None:
        """硬胎：阈值上调到 550°C，520°C 不再触发（避免误报）。"""
        dx = _derive_telemetry_dx(self._base(is_hard_compound=True))
        assert dx["brake_stab_req"] == 0.0

    def test_soft_compound_470_triggers_early(self) -> None:
        """软胎：阈值下调到 450°C，470°C 即告警（中性配方下不告警）。"""
        assert _derive_telemetry_dx(self._base(
            m_brakesTemperature=[470.0] * 4,
        ))["brake_stab_req"] == 0.0
        assert _derive_telemetry_dx(self._base(
            m_brakesTemperature=[470.0] * 4, is_soft_compound=True,
        ))["brake_stab_req"] > 0.0

    def test_severe_threshold_hard_compound(self) -> None:
        """硬胎严重阈值 660°C：650°C 不应触发方向修正。"""
        dx = _derive_telemetry_dx(self._base(
            m_brakesTemperature=[650.0] * 4, is_hard_compound=True,
        ))
        # 650 < warn(550)? 否 → 触发 warn 分支，但不触发 severe 的负向修正
        assert dx["brake_power_req"] == 0.0
        dx_soft = _derive_telemetry_dx(self._base(
            m_brakesTemperature=[650.0] * 4, is_soft_compound=True,
        ))
        assert dx_soft["brake_power_req"] < 0.0, "软胎 650°C 应触发严重过热修正"

    def test_front_rear_imbalance_flagged(self) -> None:
        """前后轴刹车温度差 > 120°C → 额外叠加失衡提示项。

        前轴 700 / 后轴 500：均值 600 > 500 → warn 0.3；600 不 > 600
        → 不触发 severe。再加失衡 0.2，合计应为 0.5。
        """
        dx = _derive_telemetry_dx(self._base(
            # 官方顺序 RL, RR, FL, FR：前轴 700/700，后轴 500/500 → 差 200
            m_brakesTemperature=[500.0, 500.0, 700.0, 700.0],
        ))
        assert dx["brake_stab_req"] == pytest.approx(0.5, abs=1e-9)
        # 对照组：轴间均衡时只有 warn 的 0.3
        balanced = _derive_telemetry_dx(self._base(
            m_brakesTemperature=[600.0, 600.0, 600.0, 600.0],
        ))
        assert balanced["brake_stab_req"] == pytest.approx(0.3, abs=1e-9)

    def test_balanced_axles_not_flagged(self) -> None:
        """前后轴接近（差 < 120°C）不触发失衡分支。"""
        dx = _derive_telemetry_dx(self._base(
            m_brakesTemperature=[500.0, 500.0, 560.0, 560.0],
        ))
        # 仅因 530°C 均值 > 500 触发 0.3，不应再多出 0.2 的失衡项
        assert dx["brake_stab_req"] == pytest.approx(0.3, abs=1e-9)

    def test_brake_bias_mismatch_flagged(self) -> None:
        """写入 58 但游戏内实际 54 → 偏差 4 > 2，提示设置未生效。"""
        dx = _derive_telemetry_dx({
            "front_brake_bias": 54.0, "brake_bias_setup": 58.0,
        })
        assert dx["brake_stab_req"] > 0.0

    def test_brake_bias_match_not_flagged(self) -> None:
        """写入 58 且实际 58.5 → 偏差 0.5 < 2，不告警。"""
        dx = _derive_telemetry_dx({
            "front_brake_bias": 58.5, "brake_bias_setup": 58.0,
        })
        assert dx["brake_stab_req"] == 0.0

    def test_brake_bias_missing_side_is_silent(self) -> None:
        """任一侧缺失不做判断（无遥测时不误报）。"""
        assert _derive_telemetry_dx({"front_brake_bias": 54.0})["brake_stab_req"] == 0.0
        assert _derive_telemetry_dx({"brake_bias_setup": 58.0})["brake_stab_req"] == 0.0

    def test_bool_brake_bias_ignored(self) -> None:
        """bool 不应被当作数值接受。"""
        dx = _derive_telemetry_dx({
            "front_brake_bias": True, "brake_bias_setup": 58.0,
        })
        assert dx["brake_stab_req"] == 0.0


# ===========================================================================
# 7. 端到端：模拟 → 建议
# ===========================================================================
class TestEndToEnd:
    """模拟遥测驱动的完整建议链路。"""

    def test_simulated_lap_reaches_suggestion(self, tmp_path: Path) -> None:
        """模拟一圈后 /suggest 应能生成完整报告。

        必须使用隔离的数据目录：``/suggest`` 的入口校验要求"该赛道已有反馈"，
        若沿用默认 ``./data``，本测试会因本地累积的反馈而**假绿**，
        在 CI 的空数据库上必然 400 —— 这正是 PR #79 上 CI 红、本地绿的原因。
        反馈由本测试经 API 显式录入，不再依赖任何环境残留。
        """
        app = _isolated_app(tmp_path)
        with TestClient(app) as c:
            c.post("/api/v1/telemetry/simulate",
                   json={"action": "start", "track_id": "suzuka"})
            # 轮询等待首帧，避免固定 sleep 在慢速 CI 上不稳定
            snap: dict = {}
            for _ in range(100):
                snap = app.state.lap_aggregator.snapshot()
                if snap.get("car_status_frames", 0) > 0:
                    break
                time.sleep(0.1)
            c.post("/api/v1/telemetry/simulate",
                   json={"action": "stop", "track_id": "suzuka"})

            # Packet 7 状态已进入聚合器
            assert snap.get("car_status_frames", 0) > 0
            assert "front_brake_bias" in snap

            # 先录入反馈（/suggest 的前置条件），再生成建议
            fb = c.post("/api/v1/feedback", json={
                "track_id": "suzuka",
                "feedbacks": [{
                    "corner_number": 1,
                    "symptom": "midcorner_understeer",
                    "strength": 3,
                }],
            })
            assert fb.status_code == 200, fb.text

            r = c.post("/api/v1/suggest", json={"track_id": "suzuka"})
            assert r.status_code == 200, r.text
            assert r.json()["code"] == 0

            report = r.json()["data"]["report"]
            assert len(report["parameters"]) == 20

    def test_suggestion_deterministic_with_new_rule(self) -> None:
        """新增规则不得破坏确定性。"""
        from setup_tuner.domain.setup import CarSetup

        setup = CarSetup().to_dict()
        telemetry = {
            "front_brake_bias": 54.0,
            "brake_bias_setup": float(setup["brake_bias"]),
            "m_brakesTemperature": [520.0] * 4,
        }
        r1 = generate_suggestion(
            symptoms=[("midcorner_understeer", 2)],
            current_setup=setup,
            track_id="suzuka", telemetry=telemetry,
        )
        r2 = generate_suggestion(
            symptoms=[("midcorner_understeer", 2)],
            current_setup=setup,
            track_id="suzuka", telemetry=telemetry,
        )
        assert r1["setup_delta"] == r2["setup_delta"]
        # 一致性规则确实生效（偏差 4 > 2）：dx 是「反馈 + 遥测」叠加结果，
        # 其中 brake_stab_req 应包含遥测侧贡献
        assert r1["dx"]["brake_stab_req"] > 0.0
        # 遥测侧单独核算：520°C 均值 → warn 0.3；刹车平衡偏差 4 → 0.25
        assert _derive_telemetry_dx(telemetry)["brake_stab_req"] == pytest.approx(
            0.55, abs=1e-9,
        )
        # 去掉刹车温度后，只剩一致性规则的 0.25
        only_bias = {"front_brake_bias": 54.0, "brake_bias_setup": 58.0}
        assert _derive_telemetry_dx(only_bias)["brake_stab_req"] == pytest.approx(
            0.25, abs=1e-9,
        )
