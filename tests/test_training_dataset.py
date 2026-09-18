"""普世化训练数据集构建测试 —— task-82。

普世化要点（逐条锁定）：
1. 赛道特征是**画像**（弯道占比/指数），不含赛道 ID —— 新赛道可推理；
2. 目标是**归一化节奏**（lap_time_ms / track_length_m，数值即 s/km）——
   与赛道长度解耦，跨赛道可比；
3. 工况（轮胎配方类/天气/温度）与风格、路肩特征全部入特征。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_training_dataset import (  # noqa: E402
    build_dataset,
    build_row,
    collect_lap_files,
    compound_class,
)
from setup_tuner.domain.setup import CarSetup  # noqa: E402
from setup_tuner.domain.track import get_track_by_udp_id  # noqa: E402


def _sample(**overrides) -> dict:
    """构造一条真实形态的逐圈样本（Monza，C5，干地）。"""
    s: dict = {
        "session_uid": "123",
        "track_id": 11,               # monza
        "lap_number": 1,
        "lap_time_ms": 85_000,
        "lap_valid": True,
        "weather": 0,
        "track_temp": 30,
        "air_temp": 22,
        "setup": CarSetup.default().to_dict(),
        "style": [0.5] * 12,
        "frames": 8000,
        "lap_agg": {
            "tyre_compound": 16,       # C5（软）
            "kerb_corners": [
                {"corner": 5, "ratio": 3.2, "frames": 200, "side": "left"},
                {"corner": 2, "ratio": 1.9, "frames": 300, "side": "right"},
            ],
            "avg_speed": 210.0,
        },
    }
    s.update(overrides)
    return s


class TestCompoundClass:
    def test_compound_mapping(self) -> None:
        assert compound_class(16) == "soft"
        assert compound_class(18) == "medium"
        assert compound_class(20) == "hard"
        assert compound_class(8) == "wet"
        assert compound_class(None) == "unknown"


class TestWeatherMapping:
    """官方 6 档枚举：只有 ≥3 是湿地（task-82 修正，附实测证据）。"""

    def test_light_cloud_is_dry(self) -> None:
        """weather=1（轻云）→ 非湿地。

        实测证据：车手在 Hungaroring m_weather=1 时用的是 C5/C4 **干胎**；
        此前误按 4 档枚举判为湿地，会污染训练样本工况标签。
        """
        row = build_row(_sample(weather=1))
        assert row["features"]["weather_wet"] == 0.0
        assert row["meta"]["weather_label"] == "轻云"

    def test_overcast_is_dry(self) -> None:
        assert build_row(_sample(weather=2))["features"]["weather_wet"] == 0.0

    def test_storm_is_wet(self) -> None:
        """weather=5（暴雨）→ 湿地（实测：Las Vegas m_weather=5 用雨胎）。"""
        row = build_row(_sample(weather=5))
        assert row["features"]["weather_wet"] == 1.0
        assert row["meta"]["weather_label"] == "暴雨"

    def test_rain_codes_are_wet(self) -> None:
        for w in (3, 4):
            assert build_row(_sample(weather=w))["features"]["weather_wet"] == 1.0

    def test_raw_code_kept_as_feature(self) -> None:
        """原始档位保留（0-5），让模型区分轻云/阴/小雨/暴雨强度。"""
        assert build_row(_sample(weather=2))["features"]["weather_code"] == 2.0

    def test_missing_weather_defaults_dry(self) -> None:
        row = build_row(_sample(weather=None))
        assert row["features"]["weather_wet"] == 0.0
        assert row["features"]["weather_code"] == -1.0


class TestBuildRow:
    def test_valid_row_has_universal_features(self) -> None:
        row = build_row(_sample())
        assert row is not None
        feats = row["features"]

        # 1) 普世化：赛道特征是画像，**不得出现赛道 ID**
        assert "track_slow_share" in feats
        assert "track_traction_index" in feats
        assert "track_id" not in feats
        assert "track_id_monza" not in feats

        # 2) 目标 = 归一化节奏（1 ms/m ≡ 1 s/km）
        t = get_track_by_udp_id(11)
        assert row["target"] == pytest.approx(85_000 / t.length_m, abs=1e-4)

        # 3) 工况与路肩特征齐备
        assert feats["tyre_class_soft"] == 1.0
        assert feats["kerb_max_ratio"] == 3.2
        assert feats["kerb_corner_count"] == 2.0
        assert feats["style_0"] == 0.5

        # 4) 20 项调教全部入特征且归一化到 [0,1]
        setup_keys = [k for k in feats if k.startswith("setup_")]
        assert len(setup_keys) == 20
        assert all(0.0 <= feats[k] <= 1.0 for k in setup_keys)

    def test_meta_records_track_length(self) -> None:
        row = build_row(_sample())
        assert row["meta"]["track_id"] == "monza"
        assert row["meta"]["tyre_class"] == "soft"

    def test_invalid_lap_rejected(self) -> None:
        assert build_row(_sample(lap_valid=False)) is None
        assert build_row(_sample(lap_time_ms=0)) is None
        assert build_row(_sample(lap_time_ms=-5)) is None
        assert build_row(_sample(track_id="monza")) is None  # 非数字 ID

    def test_unknown_track_rejected(self) -> None:
        """未知赛道（无法取画像）跳过——不臆造特征。"""
        assert build_row(_sample(track_id=999)) is None


class TestBuildDataset:
    def test_build_from_files(self, tmp_path: Path) -> None:
        rec = tmp_path / "recordings"
        rec.mkdir(parents=True)
        rows = [_sample(), _sample(lap_number=2, lap_time_ms=84_000),
                _sample(lap_valid=False)]  # 无效圈应被跳过
        (rec / "20260917_213412_laps.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8",
        )
        dataset = build_dataset(rec)
        assert dataset["count"] == 2
        assert dataset["skipped"] == 1
        assert dataset["by_track"] == {"monza": 2}
        assert "track_slow_share" in dataset["feature_keys"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        dataset = build_dataset(tmp_path)
        assert dataset["count"] == 0
        assert dataset["rows"] == []

    def test_collect_lap_files_ignores_other_files(
        self, tmp_path: Path,
    ) -> None:
        rec = tmp_path
        (rec / "a_laps.jsonl").write_text("{}", encoding="utf-8")
        (rec / "b.f1rec").write_bytes(b"x")
        (rec / "c.db").write_bytes(b"x")
        assert [p.name for p in collect_lap_files(rec)] == ["a_laps.jsonl"]
