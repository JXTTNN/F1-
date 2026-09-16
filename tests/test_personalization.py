"""task-62 个性化（M1–M3）验收测试。

覆盖：
1. Store 数据层：driver / lap_record / driver_style CRUD 与 iteration 效果回写；
2. StyleExtractor：12 维向量、确定性、圈号固化；
3. style_coefficients：中性=全 1.0、区间 0.85–1.25；
4. 引擎接入：风格改变建议、中性风格与 L0 逐位一致。
"""

from __future__ import annotations

import pytest

from setup_tuner.db.store import Store


@pytest.fixture
def store() -> Store:
    return Store(":memory:")
from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.style_coefficients import (
    GAIN_MAX,
    GAIN_MIN,
    NEUTRAL_GAIN,
    gain_for_style,
)
from setup_tuner.engine.engine import generate_suggestion
from setup_tuner.telemetry.style_extractor import STYLE_DIMS, StyleExtractor

_SYMPTOMS = [("understeer", 3)]


# ===========================================================================
# 1. 数据层
# ===========================================================================
class TestStorePersonalization:
    """车手 / 整圈落库 / 风格向量 / 效果回写。"""

    def test_default_driver_exists(self, store: Store) -> None:
        """默认车手（id=1）由初始化自动创建。"""
        driver_id = store.get_or_create_driver("默认车手")
        assert driver_id == 1

    def test_get_or_create_driver_idempotent(self, store: Store) -> None:
        """同名车手幂等。"""
        assert store.get_or_create_driver("张三") == store.get_or_create_driver("张三")

    def test_lap_record_roundtrip(self, store: Store) -> None:
        """整圈落库后可按车手+赛道检索。"""
        snapshot = {"lap_frames": 5400, "max_speed": 320.0, "lap_number": 1}
        store.save_lap_record(
            1, "suzuka", snapshot,
            lap_number=1, lap_time_ms=91000, is_valid=True,
        )
        records = store.get_lap_records("suzuka", 1)
        assert len(records) == 1
        assert records[0]["lap_time_ms"] == 91000
        body = records[0]["telemetry_json"]
        assert "max_speed" in body

    def test_driver_style_upsert(self, store: Store) -> None:
        """风格向量 UPSERT：同车手同赛道只保留一行。"""
        store.save_driver_style(1, "suzuka", [0.5] * 12, sample_count=3)
        store.save_driver_style(1, "suzuka", [0.6] * 12, sample_count=4)
        entry = store.get_driver_style(1, "suzuka")
        assert entry is not None
        assert entry["vector"] == [0.6] * 12
        assert entry["sample_count"] == 4
        assert store.get_driver_style(1, "monza") is None

    def test_corrupt_driver_style_json_is_logged(self, store: Store, caplog) -> None:
        """库内风格向量 JSON 损坏时按"无记录"处理，但必须留痕。

        否则个性化会静默退化为全局默认，且看不出是数据坏了还是样本不足。
        """
        store.save_driver_style(1, "suzuka", [0.5] * 12, sample_count=3)
        # 直接把 vector_json 写坏
        with store._lock:  # noqa: SLF001 — 测试需绕过正常写入路径
            store._conn.execute(  # noqa: SLF001
                "UPDATE driver_style SET vector_json = ? "
                "WHERE driver_id = 1 AND track_id = 'suzuka'",
                ("{not json",),
            )
            store._conn.commit()  # noqa: SLF001

        with caplog.at_level("WARNING", logger="setup_tuner.db.store"):
            entry = store.get_driver_style(1, "suzuka")

        assert entry is None, "损坏数据应表现为无记录"
        assert any(
            "损坏" in r.getMessage() for r in caplog.records if r.levelno >= 30
        ), "损坏向量被静默吞掉"

    def test_iteration_outcome_writeback(self, store: Store) -> None:
        """迭代记录可回写实际圈时（S4 闭环数据入口）。"""
        round_no = store.get_latest_round("suzuka") + 1
        iteration_id = store.save_iteration(
            track_id="suzuka", round_no=round_no,
            before_setup_id=None, after_setup_id=None, suggestion_id=None,
        )
        store.update_iteration_outcome(
            iteration_id,
            lap_time_before_ms=92000,
            lap_time_after_ms=90700,
            outcome_delta_ms=-1300.0,
        )
        rows = store.get_iterations("suzuka")
        target = next(r for r in rows if r["id"] == iteration_id)
        assert target["lap_time_before_ms"] == 92000
        assert target["lap_time_after_ms"] == 90700
        assert target["outcome_delta_ms"] == -1300.0


# ===========================================================================
# 2. 风格提取器
# ===========================================================================
class TestStyleExtractor:
    """逐帧累积与圈号固化。"""

    def _frame(self, speed=200.0, throttle=0.6, brake=0.0, steer=0.1,
               temps=(90, 92, 88, 90), btemps=(420, 430, 400, 410)):
        return {
            "m_speed": speed, "m_throttle": throttle, "m_brake": brake,
            "m_steer": steer,
            "m_tyresSurfaceTemperature": list(temps),
            "m_brakesTemperature": list(btemps),
        }

    def test_empty_returns_none(self) -> None:
        assert StyleExtractor().snapshot() is None

    def test_vector_length_and_dims(self) -> None:
        agg = StyleExtractor()
        agg.on_telemetry(self._frame())
        vector = agg.snapshot()
        # 每圈输出 11 维；第 12 维（圈速一致性）由 Store 层依据圈史追加
        assert vector is not None and len(vector) == len(STYLE_DIMS) - 1
        assert all(0.0 <= v <= 1.0 for v in vector)

    def test_non_numeric_wheel_temps_are_skipped(self, caplog) -> None:
        """非数值胎温/刹车温度不崩，跳过该帧信号并留 debug 痕迹。"""
        agg = StyleExtractor()
        agg.on_telemetry(self._frame())
        bad = agg.on_telemetry({
            "m_speed": 200.0, "m_throttle": 0.5, "m_brake": 0.0, "m_steer": 0.1,
            "m_tyresSurfaceTemperature": ["x", None, {}, []],
            "m_brakesTemperature": ["y", "z", "w", "v"],
        })
        # 不抛异常，向量仍可产出且仍在值域内
        vector = agg.snapshot()
        assert bad is None or isinstance(bad, list)
        assert vector is not None
        assert all(0.0 <= v <= 1.0 for v in vector)

    def test_short_wheel_arrays_are_ignored(self) -> None:
        """少于 4 个元素的车轮数组不参与统计（不做越界访问）。"""
        agg = StyleExtractor()
        agg.on_telemetry({
            "m_speed": 200.0, "m_throttle": 0.5, "m_brake": 0.0, "m_steer": 0.1,
            "m_tyresSurfaceTemperature": [90, 92],
            "m_brakesTemperature": [420, 430],
        })
        assert agg.snapshot() is not None

    def test_extremes_distinguish_drivers(self) -> None:
        """两个极端开法 → 向量显著不同（这就是"风格"可分性的依据）。"""
        smooth = StyleExtractor()
        for _ in range(50):
            smooth.on_telemetry(self._frame(speed=280, throttle=0.95,
                                            steer=0.02, temps=(85, 85, 85, 85)))
        rough = StyleExtractor()
        for _ in range(50):
            rough.on_telemetry(self._frame(speed=180, throttle=0.1, brake=0.6,
                                           steer=0.45, temps=(110, 100, 95, 105)))
        v_smooth, v_rough = smooth.snapshot(), rough.snapshot()
        assert v_smooth is not None and v_rough is not None
        # 循迹刹车：rough 高、smooth 低
        assert v_rough[4] > v_smooth[4]
        # 攻弯强度：rough 高
        assert v_rough[0] > v_smooth[0]
        # 轮胎管理：smooth 更好（离散小 → 维度值高）
        assert v_smooth[6] > v_rough[6]

    def test_lap_rollover_freezes(self) -> None:
        """圈号变化时固化上一圈向量。"""
        agg = StyleExtractor()
        agg.on_lap_data({"m_currentLapNum": 1})
        agg.on_telemetry(self._frame(speed=250, throttle=0.8))
        agg.on_lap_data({"m_currentLapNum": 2})
        assert agg.take_completed() is not None
        assert agg.take_completed() is None
        assert agg.snapshot() is None

    def test_deterministic(self) -> None:
        """相同输入两次结果一致。"""
        a, b = StyleExtractor(), StyleExtractor()
        frames = [self._frame(speed=150 + i, throttle=0.5 + i / 200,
                              steer=0.1 * (i % 3)) for i in range(30)]
        for f in frames:
            a.on_telemetry(f)
            b.on_telemetry(f)
        assert a.snapshot() == b.snapshot()


# ===========================================================================
# 3. 风格系数
# ===========================================================================
class TestStyleGain:
    """风格 → 增益映射的约束。"""

    def test_none_is_neutral(self) -> None:
        assert gain_for_style(None) == NEUTRAL_GAIN

    def test_short_vector_is_neutral(self) -> None:
        assert gain_for_style([0.5] * 5) == NEUTRAL_GAIN

    def test_neutral_vector_is_identity(self) -> None:
        """各维 0.5（无倾向）→ 全 1.0。"""
        assert gain_for_style([0.5] * 12) == NEUTRAL_GAIN

    def test_all_gains_in_bounds(self) -> None:
        import random
        rng = random.Random(7)
        for _ in range(20):
            vector = [rng.random() for _ in range(12)]
            gain = gain_for_style(vector)
            assert len(gain) == len(ALL_SETUP_FIELDS)
            for v in gain.values():
                assert GAIN_MIN <= v <= GAIN_MAX

    def test_nan_is_neutral(self) -> None:
        """含 NaN 的向量 → 全 1.0（防御）。"""
        gain = gain_for_style([float("nan")] * 12)
        assert gain == NEUTRAL_GAIN


# ===========================================================================
# 4. 引擎接入
# ===========================================================================
class TestEngineStyleIntegration:
    """风格调制接入 generate_suggestion。"""

    def test_neutral_style_matches_l0(self) -> None:
        """中性风格（全 0.5）与不带风格的结果逐位一致。"""
        setup = CarSetup.default().to_dict()
        without_style = generate_suggestion(_SYMPTOMS, setup, "monza",
                                            model_type="rule")
        with_neutral = generate_suggestion(_SYMPTOMS, setup, "monza",
                                           model_type="rule",
                                           style_vector=[0.5] * 12)
        assert without_style["setup_delta"] == with_neutral["setup_delta"]

    def test_style_changes_suggestion(self) -> None:
        """同一症状，激进风格与温和风格产生不同建议。"""
        setup = CarSetup.default().to_dict()
        aggressive = generate_suggestion(
            _SYMPTOMS, setup, "monza", model_type="rule",
            style_vector=[1.0] * 12,
        )
        conservative = generate_suggestion(
            _SYMPTOMS, setup, "monza", model_type="rule",
            style_vector=[0.0] * 12,
        )
        assert aggressive["setup_delta"] != conservative["setup_delta"]

    def test_style_respects_bounds(self) -> None:
        """极端风格下输出仍满足 clamp 与档位（护栏不被绕过）。"""
        setup = CarSetup.default().to_dict()
        spec = {f.name: f for f in ALL_SETUP_FIELDS}
        result = generate_suggestion(
            _SYMPTOMS, setup, "monza", model_type="rule",
            style_vector=[1.0] * 12,
        )
        for name, delta in result["setup_delta"].items():
            assert abs(delta) <= spec[name].max_delta + 1e-9
            current = setup[name]
            assert spec[name].min_val - 1e-9 <= current + delta <= spec[name].max_val + 1e-9

    def test_deterministic_with_style(self) -> None:
        """相同风格向量两次结果一致。"""
        setup = CarSetup.default().to_dict()
        a = generate_suggestion(_SYMPTOMS, setup, "monza", model_type="rule",
                                style_vector=[0.9, 0.2, 0.8, 0.4, 0.9, 0.7,
                                              0.3, 0.6, 0.8, 0.5, 0.4, 0.7])
        b = generate_suggestion(_SYMPTOMS, setup, "monza", model_type="rule",
                                style_vector=[0.9, 0.2, 0.8, 0.4, 0.9, 0.7,
                                              0.3, 0.6, 0.8, 0.5, 0.4, 0.7])
        assert a["setup_delta"] == b["setup_delta"]
