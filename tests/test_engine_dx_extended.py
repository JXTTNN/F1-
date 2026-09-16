"""引擎Dx扩展规则单元测试。

覆盖验收标准：
1. 原有6条规则（胎温过高/过低、胎压异常、刹车过热、出弯油门低、直道速度低）保持不变
2. 新增9条规则：
   - 规则7: max_steer > 0.3 → turnin_req += 0.3
   - 规则8: m_brake > 0.8 → brake_power_req += 0.3
   - 规则10: avg_steer > 0.2 → hi_speed_stab_req += 0.3
   - 规则11: m_throttle > 0.7 且 m_speed < 150 → exit_traction_req += 0.3
   - 规则12: 四轮胎温偏差 > 15°C → tyre_life_req += 0.2
   - 规则13: 胎温均值 < 60°C 且湿地 → front/rear_grip_req += 0.4
   - 规则14: 刹车均值 > 600°C → brake_stab_req += 0.4, brake_power_req -= 0.2
   - 规则15: max_speed < 280(干地)/200(湿地) → hi_speed_stab_req -= 0.3
3. 方向修正因子叠加效果（规则4+规则14同时触发）
4. 向后兼容：None输入返回全0.0，空字典返回全0.0
"""

from __future__ import annotations

import pytest

from setup_tuner.engine.diagnostic import DIAG_DIMS
from setup_tuner.engine.engine import _derive_telemetry_dx


# ===========================================================================
# 辅助函数
# ===========================================================================
def _zero_dx() -> dict[str, float]:
    """构造全零 Dx 字典。"""
    return dict.fromkeys(DIAG_DIMS, 0.0)


def _make_telemetry(**kwargs) -> dict:
    """构造遥测字典，默认值为不触发任何规则的值。"""
    defaults = {
        "m_tyresSurfaceTemperature": [90.0, 90.0, 90.0, 90.0],
        "m_tyresInnerTemperature": [85.0, 85.0, 85.0, 85.0],
        "m_brakesTemperature": [400.0, 400.0, 400.0, 400.0],
        "m_tyresPressure": [23.5, 24.0, 23.0, 23.8],
        "m_throttle": 0.5,
        "m_brake": 0.3,
        "m_speed": 250.0,
        "weather": "clear",
        "m_weather": 0,
        "track_temp": 35.0,
        "air_temp": 28.0,
        "tyre_compound": "Soft",
        "sector": 0,
        "lap_distance": 0.0,
        "avg_steer": 0.05,
        "max_steer": 0.1,
        "max_speed": 320.0,
    }
    defaults.update(kwargs)
    return defaults


# ===========================================================================
# 1. 向后兼容测试
# ===========================================================================
class TestBackwardCompatibility:
    """向后兼容：None 和空字典返回全0.0。"""

    def test_none_input_returns_all_zero(self) -> None:
        """None 输入返回全 0.0 的 Dx。"""
        dx = _derive_telemetry_dx(None)
        assert set(dx.keys()) == set(DIAG_DIMS)
        for val in dx.values():
            assert val == 0.0

    def test_empty_dict_returns_all_zero(self) -> None:
        """空字典输入返回全 0.0 的 Dx。"""
        dx = _derive_telemetry_dx({})
        assert set(dx.keys()) == set(DIAG_DIMS)
        for val in dx.values():
            assert val == 0.0

    def test_dx_covers_all_9_dims(self) -> None:
        """Dx 覆盖全部9维。"""
        dx = _derive_telemetry_dx(None)
        assert len(dx) == 9
        assert set(dx.keys()) == set(DIAG_DIMS)


# ===========================================================================
# 2. 原有6条规则测试
# ===========================================================================
class TestOriginalRules:
    """原有6条规则保持不变。"""

    def test_rule1_tyre_temp_too_high(self) -> None:
        """规则1: 胎温均值 > 100°C → tyre_life_req += 0.3。"""
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[110, 110, 110, 110])
        dx = _derive_telemetry_dx(telemetry)
        assert dx["tyre_life_req"] == pytest.approx(0.3)

    def test_rule1_tyre_temp_high_front(self) -> None:
        """规则1: 胎温过高且前轮更高 → front_grip_req += 0.2。"""
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[115, 115, 105, 105])
        dx = _derive_telemetry_dx(telemetry)
        assert dx["tyre_life_req"] == pytest.approx(0.3)
        assert dx["front_grip_req"] == pytest.approx(0.2)

    def test_rule1_tyre_temp_high_rear(self) -> None:
        """规则1: 胎温过高且后轮更高 → rear_grip_req += 0.2。"""
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[105, 105, 115, 115])
        dx = _derive_telemetry_dx(telemetry)
        assert dx["tyre_life_req"] == pytest.approx(0.3)
        assert dx["rear_grip_req"] == pytest.approx(0.2)

    def test_rule2_tyre_temp_too_low(self) -> None:
        """规则2: 胎温均值 < 80°C → front_grip_req += 0.3, rear_grip_req += 0.3。"""
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[70, 70, 70, 70])
        dx = _derive_telemetry_dx(telemetry)
        assert dx["front_grip_req"] == pytest.approx(0.3)
        assert dx["rear_grip_req"] == pytest.approx(0.3)

    def test_rule3_tyre_pressure_abnormal(self) -> None:
        """规则3: 胎压 > 26.0 或 < 22.0 → 对应轴 grip_req += 0.15。"""
        telemetry = _make_telemetry(m_tyresPressure=[27.0, 24.0, 21.0, 23.0])
        dx = _derive_telemetry_dx(telemetry)
        assert dx["front_grip_req"] == pytest.approx(0.15)  # FL > 26
        assert dx["rear_grip_req"] == pytest.approx(0.15)   # RL < 22

    def test_rule4_brake_temp_too_high(self) -> None:
        """规则4: 刹车温度均值 > 500°C → brake_stab_req += 0.3。"""
        telemetry = _make_telemetry(m_brakesTemperature=[510, 510, 510, 510])
        dx = _derive_telemetry_dx(telemetry)
        assert dx["brake_stab_req"] == pytest.approx(0.3)

    def test_rule5_exit_low_throttle(self) -> None:
        """规则5: sector == 3 且 m_throttle < 0.3 → exit_traction_req += 0.3。"""
        telemetry = _make_telemetry(sector=3, m_throttle=0.2)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["exit_traction_req"] == pytest.approx(0.3)

    def test_rule6_straight_low_speed(self) -> None:
        """规则6: speed < 200 且在直道 → hi_speed_stab_req -= 0.3。"""
        telemetry = _make_telemetry(m_speed=150, m_on_straight=True)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == pytest.approx(-0.3)


# ===========================================================================
# 3. 新增9条规则测试
# ===========================================================================
class TestNewRules:
    """新增9条规则测试。"""

    # ---- 规则7: 入弯响应差 ----
    def test_rule7_max_steer_high(self) -> None:
        """规则7: max_steer > 0.3 → turnin_req += 0.3。"""
        telemetry = _make_telemetry(max_steer=0.4)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["turnin_req"] == pytest.approx(0.3)

    def test_rule7_max_steer_not_triggered(self) -> None:
        """规则7: max_steer <= 0.3 不触发。"""
        telemetry = _make_telemetry(max_steer=0.3)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["turnin_req"] == 0.0

    # ---- 规则8: 制动力不足 ----
    def test_rule8_brake_input_high(self) -> None:
        """规则8: m_brake > 0.8 → brake_power_req += 0.3。"""
        telemetry = _make_telemetry(m_brake=0.9)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["brake_power_req"] == pytest.approx(0.3)

    def test_rule8_brake_not_triggered(self) -> None:
        """规则8: m_brake <= 0.8 不触发。"""
        telemetry = _make_telemetry(m_brake=0.8)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["brake_power_req"] == 0.0

    # ---- 规则10: 弯中不稳定 ----
    def test_rule10_avg_steer_high(self) -> None:
        """规则10: avg_steer > 0.2 → hi_speed_stab_req += 0.3。"""
        telemetry = _make_telemetry(avg_steer=0.3)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == pytest.approx(0.3)

    def test_rule10_avg_steer_not_triggered(self) -> None:
        """规则10: avg_steer <= 0.2 不触发。"""
        telemetry = _make_telemetry(avg_steer=0.2)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == 0.0

    # ---- 规则11: 出弯打滑 ----
    def test_rule11_throttle_high_speed_low(self) -> None:
        """规则11: m_throttle > 0.7 且 m_speed < 150 → exit_traction_req += 0.3。"""
        telemetry = _make_telemetry(m_throttle=0.8, m_speed=120)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["exit_traction_req"] == pytest.approx(0.3)

    def test_rule11_throttle_high_speed_high_not_triggered(self) -> None:
        """规则11: m_throttle > 0.7 但 m_speed >= 150 不触发。"""
        telemetry = _make_telemetry(m_throttle=0.8, m_speed=200)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["exit_traction_req"] == 0.0

    def test_rule11_throttle_low_not_triggered(self) -> None:
        """规则11: m_throttle <= 0.7 不触发。"""
        telemetry = _make_telemetry(m_throttle=0.5, m_speed=100)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["exit_traction_req"] == 0.0

    # ---- 规则12: 胎温不均 ----
    def test_rule12_tyre_temp_spread_high(self) -> None:
        """规则12: 四轮胎温偏差 > 15°C → tyre_life_req += 0.2。"""
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[100, 80, 90, 85])
        dx = _derive_telemetry_dx(telemetry)
        # 偏差 = 100 - 80 = 20 > 15
        assert dx["tyre_life_req"] == pytest.approx(0.2)

    def test_rule12_tyre_temp_spread_not_triggered(self) -> None:
        """规则12: 四轮胎温偏差 <= 15°C 不触发。"""
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[95, 85, 90, 88])
        dx = _derive_telemetry_dx(telemetry)
        # 偏差 = 95 - 85 = 10 <= 15
        assert dx["tyre_life_req"] == 0.0

    # ---- 规则13: 胎温严重不足（湿地） ----
    def test_rule13_cold_tyres_wet(self) -> None:
        """规则13: 胎温均值 < 60°C 且湿地 → front/rear_grip_req += 0.4。

        注意：胎温55°C同时触发规则2（<80°C, +0.3）和规则13（<60°C湿地, +0.4），
        叠加结果为 front/rear_grip_req = 0.3 + 0.4 = 0.7。
        """
        telemetry = _make_telemetry(
            m_tyresSurfaceTemperature=[55, 55, 55, 55],
            weather="wet",
            m_weather=1,
        )
        dx = _derive_telemetry_dx(telemetry)
        # 规则2(<80°C): +0.3 + 规则13(<60°C湿地): +0.4 = 0.7
        assert dx["front_grip_req"] == pytest.approx(0.7)
        assert dx["rear_grip_req"] == pytest.approx(0.7)

    def test_rule13_cold_tyres_dry_not_triggered(self) -> None:
        """规则13: 胎温均值 < 60°C 但干地不触发。"""
        telemetry = _make_telemetry(
            m_tyresSurfaceTemperature=[55, 55, 55, 55],
            weather="clear",
            m_weather=0,
        )
        dx = _derive_telemetry_dx(telemetry)
        # 规则2会触发（< 80°C），但规则13不会
        assert dx["front_grip_req"] == pytest.approx(0.3)  # 规则2
        assert dx["rear_grip_req"] == pytest.approx(0.3)   # 规则2

    def test_rule13_warm_tyres_wet_not_triggered(self) -> None:
        """规则13: 胎温均值 >= 60°C 且湿地不触发规则13。

        注意：胎温65°C仍触发规则2（<80°C），所以 front/rear_grip_req = 0.3。
        但规则13（<60°C）不触发，不会额外 +0.4。
        """
        telemetry = _make_telemetry(
            m_tyresSurfaceTemperature=[65, 65, 65, 65],
            weather="wet",
            m_weather=1,
        )
        dx = _derive_telemetry_dx(telemetry)
        # 规则2(<80°C): +0.3，规则13不触发
        assert dx["front_grip_req"] == pytest.approx(0.3)
        assert dx["rear_grip_req"] == pytest.approx(0.3)

    # ---- 规则14: 刹车持续过热 ----
    def test_rule14_brake_temp_very_high(self) -> None:
        """规则14: 刹车均值 > 600°C → brake_stab_req += 0.4, brake_power_req -= 0.2。

        注意：刹车610°C同时触发规则4（>500°C, brake_stab_req += 0.3）和规则14（>600°C），
        叠加结果为 brake_stab_req = 0.3 + 0.4 = 0.7。
        """
        telemetry = _make_telemetry(m_brakesTemperature=[610, 610, 610, 610])
        dx = _derive_telemetry_dx(telemetry)
        # 规则4(>500°C): +0.3 + 规则14(>600°C): +0.4 = 0.7
        assert dx["brake_stab_req"] == pytest.approx(0.7)
        assert dx["brake_power_req"] == pytest.approx(-0.2)

    def test_rule14_brake_temp_not_triggered(self) -> None:
        """规则14: 刹车均值 <= 600°C 不触发规则14。

        注意：刹车590°C仍触发规则4（>500°C, brake_stab_req += 0.3），
        但规则14（>600°C）不触发，不会额外 +0.4 和 -0.2。
        """
        telemetry = _make_telemetry(m_brakesTemperature=[590, 590, 590, 590])
        dx = _derive_telemetry_dx(telemetry)
        # 规则4(>500°C): +0.3，规则14不触发
        assert dx["brake_stab_req"] == pytest.approx(0.3)
        assert dx["brake_power_req"] == 0.0

    # ---- 规则15: 直道极速低 ----
    def test_rule15_max_speed_low_dry(self) -> None:
        """规则15: max_speed < 280(干地) → hi_speed_stab_req -= 0.3。"""
        telemetry = _make_telemetry(max_speed=250, weather="clear", m_weather=0)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == pytest.approx(-0.3)

    def test_rule15_max_speed_low_wet(self) -> None:
        """规则15: max_speed < 200(湿地) → hi_speed_stab_req -= 0.3。"""
        telemetry = _make_telemetry(max_speed=180, weather="wet", m_weather=1)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == pytest.approx(-0.3)

    def test_rule15_max_speed_high_not_triggered(self) -> None:
        """规则15: max_speed >= 阈值不触发。"""
        telemetry = _make_telemetry(max_speed=300, weather="clear", m_weather=0)
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == 0.0


# ===========================================================================
# 4. 方向修正因子叠加效果
# ===========================================================================
class TestDirectionalCorrectionStacking:
    """方向修正因子叠加效果测试。"""

    def test_rule4_plus_rule14_stacking(self) -> None:
        """规则4 + 规则14 同时触发：brake_stab_req = 0.3 + 0.4 = 0.7。"""
        # 刹车均值 > 500 → 规则4: brake_stab_req += 0.3
        # 刹车均值 > 600 → 规则14: brake_stab_req += 0.4, brake_power_req -= 0.2
        # 两者叠加：brake_stab_req = 0.3 + 0.4 = 0.7
        telemetry = _make_telemetry(m_brakesTemperature=[610, 610, 610, 610])
        dx = _derive_telemetry_dx(telemetry)
        assert dx["brake_stab_req"] == pytest.approx(0.7)  # 0.3 + 0.4
        assert dx["brake_power_req"] == pytest.approx(-0.2)  # 方向修正

    def test_rule6_plus_rule15_stacking(self) -> None:
        """规则6 + 规则15 同时触发：hi_speed_stab_req = -0.3 + (-0.3) = -0.6。"""
        # speed < 200 且直道 → 规则6: hi_speed_stab_req -= 0.3
        # max_speed < 280(干地) → 规则15: hi_speed_stab_req -= 0.3
        telemetry = _make_telemetry(
            m_speed=150,
            m_on_straight=True,
            max_speed=250,
            weather="clear",
            m_weather=0,
        )
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == pytest.approx(-0.6)  # -0.3 + -0.3

    def test_rule10_plus_rule15_conflict(self) -> None:
        """规则10(+) + 规则15(-) 同时触发：hi_speed_stab_req = 0.3 + (-0.3) = 0.0。"""
        # avg_steer > 0.2 → 规则10: hi_speed_stab_req += 0.3
        # max_speed < 280 → 规则15: hi_speed_stab_req -= 0.3
        telemetry = _make_telemetry(
            avg_steer=0.3,
            max_speed=250,
            weather="clear",
            m_weather=0,
        )
        dx = _derive_telemetry_dx(telemetry)
        assert dx["hi_speed_stab_req"] == pytest.approx(0.0)  # 0.3 + (-0.3)

    def test_rule8_plus_rule14_stacking(self) -> None:
        """规则8(+) + 规则14(-) 同时触发：brake_power_req = 0.3 + (-0.2) = 0.1。"""
        # m_brake > 0.8 → 规则8: brake_power_req += 0.3
        # 刹车均值 > 600 → 规则14: brake_power_req -= 0.2
        telemetry = _make_telemetry(
            m_brake=0.9,
            m_brakesTemperature=[610, 610, 610, 610],
        )
        dx = _derive_telemetry_dx(telemetry)
        assert dx["brake_power_req"] == pytest.approx(0.1)  # 0.3 + (-0.2)


# ===========================================================================
# 5. 多规则叠加综合测试
# ===========================================================================
class TestMultipleRulesStacking:
    """多规则叠加综合测试。"""

    def test_tyre_temp_rules_1_and_12_stacking(self) -> None:
        """规则1(胎温过高) + 规则12(胎温不均) 叠加。"""
        # 胎温均值 > 100 → 规则1: tyre_life_req += 0.3
        # 四轮偏差 > 15 → 规则12: tyre_life_req += 0.2
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[120, 100, 110, 105])
        dx = _derive_telemetry_dx(telemetry)
        # 均值 = 108.75 > 100 → 规则1
        # 偏差 = 120 - 100 = 20 > 15 → 规则12
        assert dx["tyre_life_req"] == pytest.approx(0.5)  # 0.3 + 0.2

    def test_tyre_temp_rules_2_and_13_stacking(self) -> None:
        """规则2(胎温过低) + 规则13(湿地严重不足) 叠加。"""
        # 胎温均值 < 80 → 规则2: front_grip_req += 0.3, rear_grip_req += 0.3
        # 胎温均值 < 60 且湿地 → 规则13: front_grip_req += 0.4, rear_grip_req += 0.4
        telemetry = _make_telemetry(
            m_tyresSurfaceTemperature=[55, 55, 55, 55],
            weather="wet",
            m_weather=1,
        )
        dx = _derive_telemetry_dx(telemetry)
        assert dx["front_grip_req"] == pytest.approx(0.7)  # 0.3 + 0.4
        assert dx["rear_grip_req"] == pytest.approx(0.7)   # 0.3 + 0.4

    def test_all_brake_rules_stacking(self) -> None:
        """规则4 + 规则8 + 规则14 全部叠加。"""
        # 刹车均值 > 500 → 规则4: brake_stab_req += 0.3
        # m_brake > 0.8 → 规则8: brake_power_req += 0.3
        # 刹车均值 > 600 → 规则14: brake_stab_req += 0.4, brake_power_req -= 0.2
        telemetry = _make_telemetry(
            m_brake=0.9,
            m_brakesTemperature=[610, 610, 610, 610],
        )
        dx = _derive_telemetry_dx(telemetry)
        assert dx["brake_stab_req"] == pytest.approx(0.7)   # 0.3 + 0.4
        assert dx["brake_power_req"] == pytest.approx(0.1)   # 0.3 + (-0.2)

    def test_no_rules_triggered_baseline(self) -> None:
        """所有值在正常范围内，无规则触发。"""
        telemetry = _make_telemetry()  # 默认值不触发任何规则
        dx = _derive_telemetry_dx(telemetry)
        for dim in DIAG_DIMS:
            assert dx[dim] == 0.0, f"{dim} should be 0.0 but got {dx[dim]}"


# ===========================================================================
# 6. 属性不变量测试
# ===========================================================================
class TestDxInvariants:
    """Dx 属性不变量测试。"""

    def test_dx_always_has_9_dims(self) -> None:
        """Dx 始终包含9维。"""
        dx = _derive_telemetry_dx(_make_telemetry())
        assert len(dx) == 9
        assert set(dx.keys()) == set(DIAG_DIMS)

    def test_dx_values_are_floats(self) -> None:
        """Dx 值均为浮点类型。"""
        dx = _derive_telemetry_dx(_make_telemetry(m_tyresSurfaceTemperature=[110, 110, 110, 110]))
        for val in dx.values():
            assert isinstance(val, float)

    def test_none_and_empty_dict_same_result(self) -> None:
        """None 和空字典返回相同结果。"""
        dx_none = _derive_telemetry_dx(None)
        dx_empty = _derive_telemetry_dx({})
        assert dx_none == dx_empty

    def test_deterministic_same_input_same_output(self) -> None:
        """相同输入产生相同输出（确定性）。"""
        telemetry = _make_telemetry(m_tyresSurfaceTemperature=[110, 110, 110, 110])
        dx1 = _derive_telemetry_dx(telemetry)
        dx2 = _derive_telemetry_dx(telemetry)
        assert dx1 == dx2