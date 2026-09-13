"""F1 26 物理仿真引擎 — 遥测校准的第一性原理物理模型。

架构：
    6大子模型 → 仿真器整合 → 遥测校准 → 优化搜索

子模型：
    1. aero_model    — 空气动力学（前翼/后翼 → 下压力 + 阻力）
    2. tyre_model    — Pirelli轮胎（胎温/胎压/外倾 → 抓地力 + 磨耗）
    3. suspension    — 悬挂动力学（悬挂/防倾杆/行驶高度 → 重量转移）
    4. diff_model    — 差速器（开/松油门差速 → 入弯/出弯稳定性）
    5. brake_model   — 制动（刹车压力/偏置 → 制动力 + 锁死阈值）
    6. track_model   — 赛道（弯道分布 → 每弯极限速度）

核心流程：
    setup(20参数) + track + conditions → simulate_lap() → (圈速, 胎耗, 稳定性)
"""

from setup_tuner.physics.aero_model import AeroModel, AeroOutput
from setup_tuner.physics.brake_model import BrakeModel, BrakeOutput
from setup_tuner.physics.calibrator import CalibrationResult, Calibrator, TelemetryBenchmark
from setup_tuner.physics.diff_model import DiffModel, DiffOutput
from setup_tuner.physics.simulator import CornerResult, LapResult, Simulator
from setup_tuner.physics.suspension_model import SuspensionModel, SuspensionOutput
from setup_tuner.physics.track_model import CornerSegment, StraightSegment, TrackModel, TrackOutput
from setup_tuner.physics.tyre_model import TyreModel, TyreOutput

__all__ = [
    "AeroModel", "AeroOutput",
    "BrakeModel", "BrakeOutput",
    "DiffModel", "DiffOutput",
    "SuspensionModel", "SuspensionOutput",
    "TrackModel", "TrackOutput", "CornerSegment", "StraightSegment",
    "TyreModel", "TyreOutput",
    "Simulator", "LapResult", "CornerResult",
    "Calibrator", "CalibrationResult", "TelemetryBenchmark",
]