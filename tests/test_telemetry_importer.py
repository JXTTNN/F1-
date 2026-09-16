"""遥测导入模块单元测试。

覆盖验收标准：
1. import_lap_json — 解析单圈JSON文件，返回 LapTelemetrySummary
2. import_laps_from_directory — 批量加载多圈
3. LapTelemetrySummary.to_telemetry_dict — 输出与engine.py兼容的遥测字典
4. 边界测试：文件不存在、空文件、缺字段、G力全零
5. 冒烟测试：使用真实数据文件
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from setup_tuner.telemetry.importer import (
    LapTelemetrySummary,
    import_lap_json,
    import_laps_from_directory,
)


# ===========================================================================
# 辅助函数：构造测试数据
# ===========================================================================
def _make_sample(
    lap_distance: float = 0.0,
    speed: float = 200.0,
    throttle: float = 0.5,
    brake: float = 0.0,
    steer: float = 0.0,
    tyre_surface_temp: list[float] | None = None,
    tyre_inner_temp: list[float] | None = None,
    tyre_pressure: list[float] | None = None,
    brake_temp: list[float] | None = None,
) -> dict:
    """构造单个采样点字典。"""
    return {
        "lap_distance": lap_distance,
        "speed": speed,
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
        "tyre_surface_temp": tyre_surface_temp or [90.0, 92.0, 88.0, 91.0],
        "tyre_inner_temp": tyre_inner_temp or [85.0, 87.0, 83.0, 86.0],
        "tyre_pressure": tyre_pressure or [23.5, 24.0, 23.0, 23.8],
        "brake_temp": brake_temp or [400.0, 420.0, 380.0, 410.0],
    }


def _make_lap_json(
    lap_number: int = 1,
    samples: list[dict] | None = None,
    weather: dict | None = None,
    tyre: dict | None = None,
    fuel: dict | None = None,
    setup: dict | None = None,
    driver_style: dict | None = None,
) -> dict:
    """构造完整的单圈JSON数据字典。"""
    if samples is None:
        samples = [_make_sample(lap_distance=float(i * 10)) for i in range(10)]
    return {
        "lap_number": lap_number,
        "lap_time_ms": 90000,
        "lap_time_str": "1:30.000",
        "sector_times_ms": [30000, 30000, 30000],
        "lap_valid": True,
        "track_id": 7,
        "track_name": "Abu Dhabi",
        "weather": weather or {"code": 0, "name": "clear", "track_temp": 35.0, "air_temp": 28.0},
        "tyre": tyre or {"compound_name": "Soft", "age_laps": 3},
        "fuel": fuel or {"load_kg": 50.0, "in_tank_kg": 45.0},
        "setup": setup or {"front_wing": 5.0, "rear_wing": 4.0},
        "driver_style": driver_style or {"aggression": 0.7},
        "sample_count": len(samples),
        "samples": samples,
    }


def _write_json_file(tmpdir: Path, filename: str, data: dict) -> Path:
    """将字典写入临时目录下的JSON文件。"""
    filepath = tmpdir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return filepath


# ===========================================================================
# 1. import_lap_json — 正常输入
# ===========================================================================
class TestImportLapJson:
    """import_lap_json 正常输入测试。"""

    def test_returns_lap_telemetry_summary(self, tmp_path: Path) -> None:
        """返回 LapTelemetrySummary 实例。"""
        filepath = _write_json_file(tmp_path, "lap_1.json", _make_lap_json())
        result = import_lap_json(str(filepath))
        assert isinstance(result, LapTelemetrySummary)

    def test_lap_basic_info(self, tmp_path: Path) -> None:
        """圈基本信息正确解析。"""
        data = _make_lap_json(lap_number=5)
        filepath = _write_json_file(tmp_path, "lap_5.json", data)
        result = import_lap_json(str(filepath))
        assert result.lap_number == 5
        assert result.lap_time_ms == 90000
        assert result.lap_time_str == "1:30.000"
        assert result.track_id == 7
        assert result.track_name == "Abu Dhabi"
        assert result.lap_valid is True
        assert result.sample_count == 10

    def test_weather_info(self, tmp_path: Path) -> None:
        """天气信息正确解析。"""
        data = _make_lap_json(weather={"code": 2, "name": "heavy_rain", "track_temp": 20.0, "air_temp": 15.0})
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.weather_code == 2
        assert result.weather_name == "heavy_rain"
        assert result.track_temp == 20.0
        assert result.air_temp == 15.0

    def test_tyre_info(self, tmp_path: Path) -> None:
        """轮胎信息正确解析。"""
        data = _make_lap_json(tyre={"compound_name": "Medium", "age_laps": 10})
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.tyre_compound == "Medium"
        assert result.tyre_age_laps == 10

    def test_speed_statistics(self, tmp_path: Path) -> None:
        """速度统计正确计算。"""
        samples = [
            _make_sample(speed=200.0, lap_distance=0.0),
            _make_sample(speed=300.0, lap_distance=10.0),
            _make_sample(speed=250.0, lap_distance=20.0),
        ]
        data = _make_lap_json(samples=samples)
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.avg_speed == pytest.approx((200 + 300 + 250) / 3)
        assert result.max_speed == 300.0

    def test_throttle_brake_steer_statistics(self, tmp_path: Path) -> None:
        """油门/刹车/转向统计正确计算。"""
        samples = [
            _make_sample(throttle=0.8, brake=0.2, steer=0.1, lap_distance=0.0),
            _make_sample(throttle=0.4, brake=0.6, steer=-0.3, lap_distance=10.0),
        ]
        data = _make_lap_json(samples=samples)
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.avg_throttle == pytest.approx(0.6)
        assert result.avg_brake == pytest.approx(0.4)
        assert result.max_brake == 0.6
        # avg_steer 是绝对值均值: (0.1 + 0.3) / 2 = 0.2
        assert result.avg_steer == pytest.approx(0.2)
        # max_steer 是绝对值最大: 0.3
        assert result.max_steer == pytest.approx(0.3)

    def test_per_wheel_statistics(self, tmp_path: Path) -> None:
        """四轮温度/胎压统计正确计算 [FL, FR, RL, RR]。"""
        samples = [
            _make_sample(
                tyre_surface_temp=[100, 110, 90, 95],
                tyre_pressure=[22.0, 24.0, 23.0, 25.0],
                brake_temp=[500, 550, 450, 480],
                lap_distance=0.0,
            ),
            _make_sample(
                tyre_surface_temp=[102, 108, 92, 97],
                tyre_pressure=[22.5, 24.5, 23.5, 25.5],
                brake_temp=[510, 540, 460, 490],
                lap_distance=10.0,
            ),
        ]
        data = _make_lap_json(samples=samples)
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        # avg_tyre_surface_temp: FL=(100+102)/2=101, FR=(110+108)/2=109
        assert result.avg_tyre_surface_temp[0] == pytest.approx(101.0)
        assert result.avg_tyre_surface_temp[1] == pytest.approx(109.0)
        # max_tyre_surface_temp: FL=max(100,102)=102
        assert result.max_tyre_surface_temp[0] == pytest.approx(102.0)
        # avg_tyre_pressure: FL=(22.0+22.5)/2=22.25
        assert result.avg_tyre_pressure[0] == pytest.approx(22.25)
        # min_tyre_pressure: FL=min(22.0,22.5)=22.0
        assert result.min_tyre_pressure[0] == pytest.approx(22.0)
        # max_tyre_pressure: FL=max(22.0,22.5)=22.5
        assert result.max_tyre_pressure[0] == pytest.approx(22.5)
        # avg_brake_temp
        assert result.avg_brake_temp[0] == pytest.approx(505.0)
        assert result.max_brake_temp[1] == pytest.approx(550.0)

    def test_setup_and_fuel(self, tmp_path: Path) -> None:
        """调教和燃油信息正确解析。"""
        data = _make_lap_json(
            setup={"front_wing": 7.0, "brake_bias": 55.0},
            fuel={"load_kg": 80.0, "in_tank_kg": 60.0},
        )
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.setup_raw == {"front_wing": 7.0, "brake_bias": 55.0}
        assert result.fuel_load_kg == 80.0
        assert result.fuel_in_tank_kg == 60.0

    def test_driver_style(self, tmp_path: Path) -> None:
        """驾驶风格正确解析。"""
        data = _make_lap_json(driver_style={"aggression": 0.8, "consistency": 0.6})
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.driver_style == {"aggression": 0.8, "consistency": 0.6}

    def test_sector_times(self, tmp_path: Path) -> None:
        """扇区时间正确解析。"""
        data = _make_lap_json()
        data["sector_times_ms"] = [28000, 32000, 30000]
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.sector_times_ms == [28000, 32000, 30000]


# ===========================================================================
# 2. import_lap_json — 边界/异常测试
# ===========================================================================
class TestImportLapJsonEdgeCases:
    """import_lap_json 边界与异常测试。"""

    def test_file_not_found(self) -> None:
        """文件不存在抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            import_lap_json("/nonexistent/path/lap_1.json")

    def test_empty_file(self, tmp_path: Path) -> None:
        """空文件抛出 JSONDecodeError。"""
        filepath = tmp_path / "empty.json"
        filepath.write_text("", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            import_lap_json(str(filepath))

    def test_invalid_json(self, tmp_path: Path) -> None:
        """非法JSON抛出 JSONDecodeError。"""
        filepath = tmp_path / "bad.json"
        filepath.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            import_lap_json(str(filepath))

    def test_missing_samples_key(self, tmp_path: Path) -> None:
        """缺 samples 字段时，samples 默认为空列表，统计值为0。"""
        data = {"lap_number": 1, "lap_time_ms": 90000}
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.sample_count == 0
        assert result.avg_speed == 0.0
        assert result.max_speed == 0.0
        assert result.avg_throttle == 0.0
        assert result.avg_tyre_surface_temp == [0.0, 0.0, 0.0, 0.0]

    def test_missing_weather_key(self, tmp_path: Path) -> None:
        """缺 weather 字段时，天气默认值。"""
        data = _make_lap_json()
        del data["weather"]
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.weather_code == 0
        assert result.weather_name == ""
        assert result.track_temp == 0.0

    def test_missing_tyre_key(self, tmp_path: Path) -> None:
        """缺 tyre 字段时，轮胎默认值。"""
        data = _make_lap_json()
        del data["tyre"]
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.tyre_compound == ""
        assert result.tyre_age_laps == 0

    def test_missing_fuel_key(self, tmp_path: Path) -> None:
        """缺 fuel 字段时，燃油默认值。"""
        data = _make_lap_json()
        del data["fuel"]
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.fuel_load_kg == 0.0
        assert result.fuel_in_tank_kg == 0.0

    def test_missing_setup_key(self, tmp_path: Path) -> None:
        """缺 setup 字段时，setup_raw 为空字典。"""
        data = _make_lap_json()
        del data["setup"]
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.setup_raw == {}

    def test_g_force_all_zero(self, tmp_path: Path) -> None:
        """G力数据全零不影响导入（importer不依赖G力）。"""
        samples = [_make_sample(lap_distance=float(i * 10)) for i in range(5)]
        # 添加全零的G力字段
        for s in samples:
            s["lat_g"] = 0.0
            s["long_g"] = 0.0
            s["vert_g"] = 0.0
        data = _make_lap_json(samples=samples)
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        # G力全零不影响其他统计
        assert result.avg_speed > 0
        assert result.avg_throttle > 0

    def test_empty_samples_list(self, tmp_path: Path) -> None:
        """samples 为空列表时统计值为0。"""
        data = _make_lap_json(samples=[])
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.sample_count == 0
        assert result.avg_speed == 0.0
        assert result.max_speed == 0.0
        assert result.avg_tyre_surface_temp == [0.0, 0.0, 0.0, 0.0]
        assert result.avg_tyre_pressure == [0.0, 0.0, 0.0, 0.0]

    def test_samples_with_missing_wheel_fields(self, tmp_path: Path) -> None:
        """samples 中缺少四轮字段时，四轮统计为0。"""
        samples = [{"speed": 200, "throttle": 0.5, "brake": 0.0, "steer": 0.0, "lap_distance": 0.0}]
        data = _make_lap_json(samples=samples)
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        assert result.avg_tyre_surface_temp == [0.0, 0.0, 0.0, 0.0]
        assert result.avg_tyre_pressure == [0.0, 0.0, 0.0, 0.0]


# ===========================================================================
# 3. LapTelemetrySummary.to_telemetry_dict
# ===========================================================================
class TestToTelemetryDict:
    """to_telemetry_dict 输出与 engine.py 兼容的遥测字典。"""

    def test_output_keys_match_engine_expectations(self, tmp_path: Path) -> None:
        """输出字典包含 engine._derive_telemetry_dx 所需的全部 key。"""
        filepath = _write_json_file(tmp_path, "lap_1.json", _make_lap_json())
        result = import_lap_json(str(filepath))
        td = result.to_telemetry_dict()
        # engine._derive_telemetry_dx 读取的 key
        expected_keys = {
            "m_tyresSurfaceTemperature", "m_tyresInnerTemperature",
            "m_brakesTemperature", "m_tyresPressure",
            "m_throttle", "m_brake", "m_speed",
            "weather", "m_weather",
            "track_temp", "air_temp",
            "tyre_compound", "sector", "lap_distance",
            "avg_steer", "max_steer", "max_speed",
        }
        assert expected_keys.issubset(set(td.keys()))

    def test_tyre_lists_are_float_lists(self, tmp_path: Path) -> None:
        """四轮温度/胎压输出为4元素浮点列表。"""
        filepath = _write_json_file(tmp_path, "lap_1.json", _make_lap_json())
        result = import_lap_json(str(filepath))
        td = result.to_telemetry_dict()
        for key in ("m_tyresSurfaceTemperature", "m_tyresInnerTemperature",
                     "m_brakesTemperature", "m_tyresPressure"):
            assert isinstance(td[key], list)
            assert len(td[key]) == 4
            assert all(isinstance(v, float) for v in td[key])

    def test_scalar_values_are_floats(self, tmp_path: Path) -> None:
        """标量值为浮点类型。"""
        filepath = _write_json_file(tmp_path, "lap_1.json", _make_lap_json())
        result = import_lap_json(str(filepath))
        td = result.to_telemetry_dict()
        assert isinstance(td["m_throttle"], float)
        assert isinstance(td["m_brake"], float)
        assert isinstance(td["m_speed"], float)
        assert isinstance(td["max_steer"], float)
        assert isinstance(td["avg_steer"], float)
        assert isinstance(td["max_speed"], float)

    def test_weather_fields(self, tmp_path: Path) -> None:
        """天气字段正确传递。"""
        data = _make_lap_json(weather={"code": 1, "name": "light_rain", "track_temp": 22.0, "air_temp": 18.0})
        filepath = _write_json_file(tmp_path, "lap_1.json", data)
        result = import_lap_json(str(filepath))
        td = result.to_telemetry_dict()
        assert td["weather"] == "light_rain"
        assert td["m_weather"] == 1
        assert td["track_temp"] == 22.0
        assert td["air_temp"] == 18.0

    def test_sector_and_lap_distance_defaults(self, tmp_path: Path) -> None:
        """sector 和 lap_distance 使用默认值0。"""
        filepath = _write_json_file(tmp_path, "lap_1.json", _make_lap_json())
        result = import_lap_json(str(filepath))
        td = result.to_telemetry_dict()
        assert td["sector"] == 0
        assert td["lap_distance"] == 0.0


# ===========================================================================
# 4. import_laps_from_directory
# ===========================================================================
class TestImportLapsFromDirectory:
    """import_laps_from_directory 批量加载测试。"""

    def test_multiple_laps_sorted_by_lap_number(self, tmp_path: Path) -> None:
        """多圈文件按 lap_number 升序排列。"""
        _write_json_file(tmp_path, "lap_3.json", _make_lap_json(lap_number=3))
        _write_json_file(tmp_path, "lap_1.json", _make_lap_json(lap_number=1))
        _write_json_file(tmp_path, "lap_2.json", _make_lap_json(lap_number=2))
        result = import_laps_from_directory(str(tmp_path))
        assert len(result) == 3
        assert result[0].lap_number == 1
        assert result[1].lap_number == 2
        assert result[2].lap_number == 3

    def test_empty_directory(self, tmp_path: Path) -> None:
        """空目录返回空列表。"""
        result = import_laps_from_directory(str(tmp_path))
        assert result == []

    def test_skips_corrupted_json(self, tmp_path: Path) -> None:
        """损坏的JSON文件被跳过，不中断整体导入。"""
        _write_json_file(tmp_path, "lap_1.json", _make_lap_json(lap_number=1))
        # 写一个损坏的JSON
        bad_file = tmp_path / "lap_2.json"
        bad_file.write_text("{invalid", encoding="utf-8")
        _write_json_file(tmp_path, "lap_3.json", _make_lap_json(lap_number=3))
        result = import_laps_from_directory(str(tmp_path))
        assert len(result) == 2
        assert result[0].lap_number == 1
        assert result[1].lap_number == 3

    def test_corrupted_json_skip_is_logged(self, tmp_path: Path, caplog) -> None:
        """跳过的损坏 JSON 必须留痕，否则"圈史少几圈"会被误判为数据缺失。"""
        _write_json_file(tmp_path, "lap_1.json", _make_lap_json(lap_number=1))
        (tmp_path / "lap_2.json").write_text("{invalid", encoding="utf-8")

        with caplog.at_level("WARNING", logger="setup_tuner.telemetry.importer"):
            import_laps_from_directory(str(tmp_path))

        logged = [r for r in caplog.records if r.levelno >= 30]
        assert logged, "损坏 JSON 被静默跳过"
        assert any("lap_2.json" in r.getMessage() for r in logged)

    def test_clean_directory_has_no_warnings(self, tmp_path: Path, caplog) -> None:
        """全部文件正常时不应产生 warning 噪声。"""
        _write_json_file(tmp_path, "lap_1.json", _make_lap_json(lap_number=1))

        with caplog.at_level("WARNING", logger="setup_tuner.telemetry.importer"):
            import_laps_from_directory(str(tmp_path))

        assert not [r for r in caplog.records if r.levelno >= 30]

    def test_only_lap_prefix_files(self, tmp_path: Path) -> None:
        """只加载 lap_*.json 文件，忽略其他文件。"""
        _write_json_file(tmp_path, "lap_1.json", _make_lap_json(lap_number=1))
        _write_json_file(tmp_path, "summary.json", _make_lap_json(lap_number=99))
        _write_json_file(tmp_path, "other.txt", {"not": "json"})
        result = import_laps_from_directory(str(tmp_path))
        assert len(result) == 1
        assert result[0].lap_number == 1

    def test_nonexistent_directory(self) -> None:
        """不存在目录返回空列表（glob不报错）。"""
        result = import_laps_from_directory("/nonexistent/directory")
        assert result == []


# ===========================================================================
# 5. 冒烟测试 — 使用真实数据文件
# ===========================================================================
class TestSmokeRealData:
    """真实数据冒烟测试（H5 修复：路径改为环境变量驱动）。

    设置 ``F1OPT_REAL_DATA_DIR`` 指向含 ``lap_*.json`` 的目录即可启用；
    未设置或目录不存在时干净跳过（CI 中必然跳过，不再指向任何人本机的
    硬编码路径）。
    """

    _REAL_DATA_DIR = os.environ.get("F1OPT_REAL_DATA_DIR", "")

    @pytest.mark.skipif(
        not os.path.isdir(_REAL_DATA_DIR),
        reason="未设置 F1OPT_REAL_DATA_DIR（或目录不存在）",
    )
    def test_import_real_lap_file(self) -> None:
        """冒烟测试：导入真实单圈JSON文件。"""
        json_files = list(Path(self._REAL_DATA_DIR).glob("lap_*.json"))
        if not json_files:
            pytest.skip("无真实数据文件")
        filepath = str(json_files[0])
        result = import_lap_json(filepath)
        assert isinstance(result, LapTelemetrySummary)
        assert result.sample_count > 0
        assert result.avg_speed > 0
        assert len(result.avg_tyre_surface_temp) == 4

    @pytest.mark.skipif(
        not os.path.isdir(_REAL_DATA_DIR),
        reason="未设置 F1OPT_REAL_DATA_DIR（或目录不存在）",
    )
    def test_import_real_directory(self) -> None:
        """冒烟测试：批量导入真实数据目录。"""
        result = import_laps_from_directory(self._REAL_DATA_DIR)
        assert len(result) > 0
        for summary in result:
            assert isinstance(summary, LapTelemetrySummary)
            assert summary.sample_count > 0

    @pytest.mark.skipif(
        not os.path.isdir(_REAL_DATA_DIR),
        reason="未设置 F1OPT_REAL_DATA_DIR（或目录不存在）",
    )
    def test_real_data_to_telemetry_dict(self) -> None:
        """冒烟测试：真实数据转遥测字典。"""
        json_files = list(Path(self._REAL_DATA_DIR).glob("lap_*.json"))
        if not json_files:
            pytest.skip("无真实数据文件")
        filepath = str(json_files[0])
        result = import_lap_json(filepath)
        td = result.to_telemetry_dict()
        # 验证关键字段存在且类型正确
        assert isinstance(td["m_tyresSurfaceTemperature"], list)
        assert len(td["m_tyresSurfaceTemperature"]) == 4
        assert isinstance(td["m_speed"], float)
        assert isinstance(td["m_throttle"], float)