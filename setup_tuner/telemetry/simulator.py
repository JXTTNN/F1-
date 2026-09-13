"""遥测模拟器 —— 回放模拟遥测数据，通过回调推送。

用于无真实 F1 游戏时测试前端遥测显示和 WebSocket 推送。

设计要点：
- 后台线程按 10fps（默认）回放模拟遥测帧；
- 通过 ``add_handler`` 注册回调（与 :class:`TelemetryListener` 一致）；
- 按赛道/圈数回放，自动循环；
- 帧格式对齐 F1 25 UDP Packet 6 (CarTelemetry) 解析后的字典结构；
- 弯道切换时推送 ``corner`` 事件（含 corner_number / sector）。

使用方式::

    from setup_tuner.telemetry.simulator import TelemetrySimulator

    sim = TelemetrySimulator()
    sim.add_handler(lambda frame: print(frame["speed"]))
    sim.start(track_id="suzuka")
    # ... 运行 ...
    sim.stop()

或与 :class:`TelemetryStream` 集成（供 WebSocket 推送）::

    from setup_tuner.telemetry import TelemetryStream

    stream = TelemetryStream()
    sim = TelemetrySimulator()
    sim.add_handler(stream.update)  # 自动更新最新帧缓存
    sim.start(track_id="suzuka")
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from setup_tuner.domain.setup import CarSetup
from setup_tuner.domain.track import Track, get_track_by_id

logger = logging.getLogger(__name__)

# 默认回放帧率（10fps，足够前端展示，避免 CPU 占用过高）
DEFAULT_FPS = 10
# 默认回放圈数（None = 无限循环）
DEFAULT_LAPS: int | None = None
# 默认随机种子（可复现）
DEFAULT_SEED = 20260911
# 帧间隔下限（避免 FPS 过高导致 CPU 占满）
_MIN_FRAME_INTERVAL_SEC = 0.01


# --------------------------------------------------------------------------- #
# 物理模型辅助
# --------------------------------------------------------------------------- #
def _clamp(value: float, lo: float, hi: float) -> float:
    """裁剪到 [lo, hi]。"""
    return max(lo, min(hi, value))


def _speed_to_gear(speed_kmh: float) -> int:
    """速度 → 档位映射（F1 8 档）。"""
    if speed_kmh < 30:
        return 1
    if speed_kmh < 80:
        return 2
    if speed_kmh < 130:
        return 3
    if speed_kmh < 180:
        return 4
    if speed_kmh < 220:
        return 5
    if speed_kmh < 260:
        return 6
    if speed_kmh < 300:
        return 7
    return 8


def _speed_to_rpm(speed_kmh: float, gear: int) -> int:
    """速度 + 档位 → 转速。"""
    idle_rpm = 2500
    max_rpm = 12500
    if speed_kmh <= 0:
        return idle_rpm
    gear_speed_min = [0, 30, 80, 130, 180, 220, 260, 300, 340]
    lo = gear_speed_min[max(0, gear - 1)]
    hi = gear_speed_min[min(8, gear)]
    if hi <= lo:
        return idle_rpm
    ratio = (speed_kmh - lo) / (hi - lo)
    return int(_clamp(idle_rpm + ratio * (max_rpm - idle_rpm), idle_rpm, max_rpm))


def _estimate_lap_time(track: Track) -> float:
    """根据赛道类型与长度估算圈速基准（秒）。"""
    # 量级核对自 F1 2026 真实圈速
    base_by_type = {
        "high_speed_low_downforce": 85.0,
        "street": 76.0,
        "high_downforce": 78.0,
        "medium": 90.0,
        "mixed": 92.0,
    }
    base = base_by_type.get(track.track_type, 90.0)
    length_factor = track.length_m / 5000.0
    return base * math.sqrt(length_factor)


# --------------------------------------------------------------------------- #
# TelemetrySimulator
# --------------------------------------------------------------------------- #
# handler 类型：接收解析后的 dict（与 TelemetryListener 一致）
SimHandler = Callable[[dict[str, Any]], None]


class TelemetrySimulator:
    """遥测模拟器。

    启动后台线程，按指定赛道回放模拟遥测数据，
    通过回调函数推送（类似 :class:`TelemetryListener` 的 handler 机制）。

    线程安全：handler 列表通过内部锁保护；start/stop 可安全多次调用。
    """

    def __init__(
        self,
        fps: int = DEFAULT_FPS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """初始化模拟器。

        Args:
            fps: 回放帧率（帧/秒），默认 10。
            seed: 随机种子（可复现），默认 20260911。
        """
        self._fps = max(1, fps)
        self._seed = seed
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._track_id: str | None = None
        self._max_laps: int | None = DEFAULT_LAPS
        self._handlers: list[SimHandler] = []
        self._handlers_lock = threading.Lock()
        # 回放状态
        self._current_lap = 0
        self._current_frame = 0
        self._current_corner: int | None = None

    # ------------------------------------------------------------------ #
    # handler 注册
    # ------------------------------------------------------------------ #
    def add_handler(self, handler: SimHandler) -> None:
        """注册遥测数据回调。

        回调签名：``handler(frame: dict) -> None``，
        其中 ``frame`` 含 ``packet_id`` / ``speed`` / ``throttle`` / ``brake`` /
        ``gear`` / ``engine_rpm`` / ``lap_number`` / ``corner_number`` 等字段。
        """
        with self._handlers_lock:
            if handler not in self._handlers:
                self._handlers.append(handler)

    def remove_handler(self, handler: SimHandler) -> None:
        """移除一个已注册的回调。"""
        with self._handlers_lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def _dispatch(self, frame: dict[str, Any]) -> None:
        """将帧分发给所有已注册 handler（handler 异常不中断模拟）。"""
        with self._handlers_lock:
            handlers = list(self._handlers)
        for h in handlers:
            try:
                h(frame)
            except Exception:
                logger.exception("handler %r raised, continuing", h)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(
        self,
        track_id: str,
        max_laps: int | None = None,
    ) -> bool:
        """启动模拟（指定赛道）。

        Args:
            track_id: 赛道标识（如 ``"suzuka"``）。
            max_laps: 最大回放圈数；``None`` 表示无限循环。

        Returns:
            True 表示成功启动；False 表示已在运行或赛道不存在。
        """
        if self._running.is_set():
            logger.warning("simulator already running")
            return False

        track = get_track_by_id(track_id)
        if track is None:
            logger.error("track not found: %s", track_id)
            return False

        self._track_id = track_id
        self._max_laps = max_laps
        self._current_lap = 0
        self._current_frame = 0
        self._current_corner = None
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="f1opt-telemetry-simulator", daemon=True,
        )
        self._thread.start()
        logger.info(
            "telemetry simulator started: track=%s fps=%d max_laps=%s",
            track_id, self._fps, max_laps,
        )
        return True

    def stop(self) -> None:
        """停止模拟（幂等：未运行时直接返回）。"""
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("telemetry simulator stopped")

    @property
    def is_running(self) -> bool:
        """模拟线程是否在运行。"""
        return self._running.is_set()

    @property
    def current_track_id(self) -> str | None:
        """当前回放的赛道标识。"""
        return self._track_id

    @property
    def current_lap(self) -> int:
        """当前回放的圈号（0-based）。"""
        return self._current_lap

    @property
    def current_corner(self) -> int | None:
        """当前所在的弯道编号（1-based，None 表示直道）。"""
        return self._current_corner

    # ------------------------------------------------------------------ #
    # 后台回放循环
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """后台线程：按圈回放遥测数据。

        流程：
        1. 加载赛道数据；
        2. 估算圈速基准；
        3. 循环生成遥测帧（每帧间隔 1/fps 秒）；
        4. 模拟速度/油门/刹车/档位/转速变化；
        5. 检测弯道切换并推送 corner 事件；
        6. 达到 max_laps 时退出。
        """
        assert self._track_id is not None
        track = get_track_by_id(self._track_id)
        if track is None:
            logger.error("track disappeared: %s", self._track_id)
            return

        rng = random.Random(self._seed)
        lap_time = _estimate_lap_time(track)
        n_frames_per_lap = max(60, int(lap_time * self._fps))
        frame_interval = max(_MIN_FRAME_INTERVAL_SEC, 1.0 / self._fps)

        # 弯道位置（按编号均匀分布）
        corners = track.corners
        n_corners = len(corners)
        corner_positions = [(i + 0.5) / n_corners for i in range(n_corners)]
        corner_radius = 0.6 / n_corners

        # 默认调教
        setup = CarSetup.default()

        logger.info(
            "simulator loop: track=%s lap_time=%.2fs frames/lap=%d",
            track.track_id, lap_time, n_frames_per_lap,
        )

        while self._running.is_set():
            if self._max_laps is not None and self._current_lap >= self._max_laps:
                logger.info("reached max_laps=%d, stopping", self._max_laps)
                break
            self._run_one_frame(
                track, lap_time, n_frames_per_lap,
                corner_positions, corner_radius, setup, rng,
            )
            time.sleep(frame_interval)

    def _run_one_frame(
        self, track: Track, lap_time: float, n_frames_per_lap: int,
        corner_positions: list[float], corner_radius: float,
        setup: CarSetup, rng: random.Random,
    ) -> None:
        """生成并派发单帧，处理弯道切换与帧/圈推进。"""
        frame = self._generate_frame(
            track=track, lap_time=lap_time, frame_idx=self._current_frame,
            n_frames_per_lap=n_frames_per_lap,
            corner_positions=corner_positions, corner_radius=corner_radius,
            setup=setup, rng=rng,
        )
        new_corner = frame["corner_number"]
        if new_corner != self._current_corner:
            self._current_corner = new_corner
            self._dispatch(self._build_corner_event(track, frame, new_corner))
        self._dispatch(frame)
        self._advance_frame(n_frames_per_lap, track.track_id)

    def _build_corner_event(
        self, track: Track, frame: dict[str, Any], new_corner: int | None,
    ) -> dict[str, Any]:
        """构造弯道切换事件。"""
        return {
            "packet_id": 2,  # LapData 风格
            "event": "corner",
            "track_id": track.track_id,
            "corner_number": new_corner,
            "sector": frame["sector"],
            "lap_number": self._current_lap + 1,
        }

    def _advance_frame(self, n_frames_per_lap: int, track_id: str) -> None:
        """推进帧索引，到达一圈时推进圈计数。"""
        self._current_frame += 1
        if self._current_frame >= n_frames_per_lap:
            self._current_frame = 0
            self._current_lap += 1
            logger.debug(
                "lap %d completed on track %s", self._current_lap, track_id,
            )

    def _generate_frame(
        self,
        track: Track,
        lap_time: float,
        frame_idx: int,
        n_frames_per_lap: int,
        corner_positions: list[float],
        corner_radius: float,
        setup: CarSetup,
        rng: random.Random,
    ) -> dict[str, Any]:
        """生成单帧遥测数据。

        帧格式对齐 F1 25 UDP Packet 6 (CarTelemetry) 解析后的字典结构，
        额外含 ``lap_number`` / ``corner_number`` / ``sector`` / ``progress``。
        """
        progress = frame_idx / n_frames_per_lap  # 0~1
        lap_distance = progress * track.length_m

        in_corner, corner_number, corner_speed_target = self._compute_corner_influence(
            progress, track, corner_positions, corner_radius,
        )

        speed = _clamp(corner_speed_target + rng.uniform(-5, 5), 45.0, 340.0)
        throttle, brake = self._compute_throttle_brake(in_corner, rng)
        gear = _speed_to_gear(speed)
        rpm = _speed_to_rpm(speed, gear)
        drs = 1 if (not in_corner and speed > 250) else 0
        steer = rng.uniform(-0.3, 0.3) if in_corner else rng.uniform(-0.05, 0.05)
        sector = int(progress * 3) if progress < 1.0 else 2

        return self._build_frame_dict(
            track, lap_time, progress, lap_distance, speed, throttle, brake,
            gear, rpm, drs, steer, sector, in_corner, corner_number, rng,
        )

    def _compute_corner_influence(
        self, progress: float, track: Track,
        corner_positions: list[float], corner_radius: float,
    ) -> tuple[bool, int | None, float]:
        """计算最近弯道对当前进度的影响。"""
        in_corner = False
        corner_number: int | None = None
        corner_speed_target = 340.0  # 直道极速上限
        for idx, cp in enumerate(corner_positions):
            dist = min(abs(progress - cp), 1.0 - abs(progress - cp))
            if dist < corner_radius:
                corner = track.corners[idx]
                factor = 1.0 - (1.0 - dist / corner_radius) ** 2
                target = corner.speed_kmh + (340.0 - corner.speed_kmh) * factor
                if target < corner_speed_target:
                    corner_speed_target = target
                    in_corner = True
                    corner_number = corner.number
        return in_corner, corner_number, corner_speed_target

    @staticmethod
    def _compute_throttle_brake(
        in_corner: bool, rng: random.Random,
    ) -> tuple[float, float]:
        """根据是否在弯道计算油门与刹车。"""
        if in_corner:
            return rng.uniform(0.2, 0.7), rng.uniform(0.0, 0.3)
        return rng.uniform(0.85, 1.0), 0.0

    def _build_frame_dict(
        self, track: Track, lap_time: float, progress: float, lap_distance: float,
        speed: float, throttle: float, brake: float, gear: int, rpm: int,
        drs: int, steer: float, sector: int,
        in_corner: bool, corner_number: int | None, rng: random.Random,
    ) -> dict[str, Any]:
        """构造对齐 Packet 6 解析后字段名的帧字典。"""
        return {
            "packet_id": 6,
            "name": "CarTelemetry",
            "m_speed": int(speed),
            "m_throttle": round(throttle, 3),
            "m_steer": round(steer, 3),
            "m_brake": round(brake, 3),
            "m_clutch": 0,
            "m_gear": gear,
            "m_engineRPM": rpm,
            "m_drs": drs,
            "m_revLightsPercent": int(rpm / 12500 * 100),
            "m_revLightsBitValue": 0,
            "m_brakesTemperature": [rng.randint(100, 600) for _ in range(4)],
            "m_tyresSurfaceTemperature": [rng.randint(80, 110) for _ in range(4)],
            "m_tyresInnerTemperature": [rng.randint(85, 115) for _ in range(4)],
            "m_engineTemperature": rng.randint(95, 115),
            "m_tyresPressure": [
                round(25.5 + rng.uniform(-1, 1), 2) for _ in range(4)
            ],
            "m_surfaceType": [0, 0, 0, 0],
            "speed": round(speed, 2),
            "throttle": round(throttle, 3),
            "brake": round(brake, 3),
            "steer": round(steer, 3),
            "gear": gear,
            "engine_rpm": rpm,
            "lap_number": self._current_lap + 1,
            "lap_time_sec": round(progress * lap_time, 3),
            "lap_distance_m": round(lap_distance, 2),
            "progress": round(progress, 4),
            "sector": sector,
            "corner_number": corner_number,
            "in_corner": in_corner,
            "track_id": track.track_id,
        }


# --------------------------------------------------------------------------- #
# 便捷函数：生成单圈遥测快照（供测试用）
# --------------------------------------------------------------------------- #
def generate_lap_snapshot(
    track_id: str,
    lap_time_sec: float | None = None,
    fps: int = 60,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """生成单圈遥测快照（不启动后台线程，供测试用）。

    Args:
        track_id: 赛道标识。
        lap_time_sec: 圈速（秒）；None 时自动估算。
        fps: 帧率，默认 60。
        seed: 随机种子。

    Returns:
        遥测帧列表。

    Raises:
        ValueError: 赛道不存在。
    """
    track = get_track_by_id(track_id)
    if track is None:
        raise ValueError(f"track not found: {track_id!r}")

    if lap_time_sec is None:
        lap_time_sec = _estimate_lap_time(track)

    rng = random.Random(seed)
    n_frames = max(60, int(lap_time_sec * fps))
    n_corners = len(track.corners)
    corner_positions = [(i + 0.5) / n_corners for i in range(n_corners)]
    corner_radius = 0.6 / n_corners
    setup = CarSetup.default()

    # 复用模拟器的帧生成逻辑
    sim = TelemetrySimulator(fps=fps, seed=seed)
    sim._track_id = track_id  # noqa: SLF001 — 供无线程模式使用

    return _collect_lap_frames(
        sim, track, lap_time_sec, n_frames,
        corner_positions, corner_radius, setup, rng,
    )


def _collect_lap_frames(
    sim: TelemetrySimulator, track: Track, lap_time_sec: float, n_frames: int,
    corner_positions: list[float], corner_radius: float,
    setup: CarSetup, rng: random.Random,
) -> list[dict[str, Any]]:
    """循环生成并收集单圈所有帧。"""
    frames: list[dict[str, Any]] = []
    for i in range(n_frames):
        frame = sim._generate_frame(  # noqa: SLF001
            track=track,
            lap_time=lap_time_sec,
            frame_idx=i,
            n_frames_per_lap=n_frames,
            corner_positions=corner_positions,
            corner_radius=corner_radius,
            setup=setup,
            rng=rng,
        )
        frames.append(frame)
    return frames