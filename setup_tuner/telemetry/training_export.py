"""训练数据导出器 —— 从 .f1rec 录制重放解析，产出逐圈训练样本 JSONL。

数据流（task-81）::

    .f1rec（无损原始字节，全部包类型）
      → ReplayReader 逐包重放
      → parse_packet 解析（7 类核心包）
      → LapAggregator / StyleExtractor 逐圈聚合
      → 每完成一圈落一行 JSONL 训练样本

样本结构（对齐 F1SetupNet 的特征规划）::

    {
      "session_uid": "…",        # UDP 会话 UID（包头）
      "track_id": 3,             # m_trackId（来自 Session 包）
      "lap_number": 2,           # 完成的圈号
      "lap_time_ms": 91234,      # 该圈用时
      "lap_valid": true,         # 近似有效标志（圈末最近一帧 m_currentLapInvalid）
      "setup": {...},            # 当前生效调教（Packet 5 玩家段 21 字段）
      "style": [...],            # 驾驶风格特征（STYLE_DIMS 维，0-1）
      "lap_agg": {...},          # 逐圈聚合明细（油门/刹车/轮胎/ERS/刮底统计）
      "frames": 1200,            # 该圈收到的遥测帧数
      "t_start": 12.5,           # 圈起始（会话内秒）
      "t_end": 103.7,
    }

容错：录制中可能混有非法/截断包（UDP 丢包拼接），解析失败或字段缺失
的包跳过，不中断导出；没有任何完整圈的录制输出空 JSONL 并在返回值中
标注 ``laps=0``。

本模块位于 telemetry 层——**桌面接收器与 Web 系统共用同一实现**，
避免两处导出逻辑分叉。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from setup_tuner.telemetry.lap_aggregator import LapAggregator
from setup_tuner.telemetry.packets import PacketTooShortError, parse_packet
from setup_tuner.telemetry.recorder import ReplayReader
from setup_tuner.telemetry.style_extractor import StyleExtractor

logger = logging.getLogger(__name__)

# Packet 5（CarSetups）玩家段调教字段（对齐 setup_tuner 调教模型的 21 项）
_SETUP_KEYS: tuple[str, ...] = (
    "m_frontWing", "m_rearWing", "m_onThrottleDiff", "m_offThrottleDiff",
    "m_frontCamber", "m_rearCamber", "m_frontToe", "m_rearToe",
    "m_frontSuspension", "m_rearSuspension", "m_frontAntiRollBar",
    "m_rearAntiRollBar", "m_frontSuspensionHeight", "m_rearSuspensionHeight",
    "m_brakePressure", "m_brakeBias", "m_rearLeftTyrePressure",
    "m_rearRightTyrePressure", "m_frontLeftTyrePressure",
    "m_frontRightTyrePressure", "m_ballast", "m_fuelLoad",
)


def _wire_kerb_track(agg: Any, session: dict[str, Any]) -> None:
    """把 Session 包的赛道信息注入聚合器，使训练样本带上「按弯路肩」特征。

    MotionEx 不带圈内距离，只能靠 Session 的 ``m_trackLength`` + 弧长表
    把每一帧归因到弯道。未知赛道静默降级（路肩字段缺省），不影响导出。
    """
    udp_id = session.get("m_trackId")
    length = session.get("m_trackLength")
    if not isinstance(udp_id, int) or isinstance(udp_id, bool):
        return
    if not isinstance(length, (int, float)) or isinstance(length, bool):
        return

    from setup_tuner.domain.corner_locator import locate_corner
    from setup_tuner.domain.track import get_track_by_udp_id

    track = get_track_by_udp_id(udp_id)
    if track is None:
        return

    def _locate(dist: float) -> int | None:
        return locate_corner(dist, track.length_m, track.corners, track.track_id)

    agg.set_track_context(float(length), _locate)


class TrainingExporter:
    """把一份 .f1rec 录制导出为逐圈训练样本 JSONL。

    用法::

        stats = TrainingExporter().export(
            "data/recordings/<session>.f1rec", "training/laps.jsonl")
        # stats = {"laps": 12, "packets": 45678, "parse_errors": 3}
    """

    def export(self, f1rec_path: str | Path, out_path: str | Path) -> dict[str, Any]:
        """重放录制文件并导出训练样本。

        Args:
            f1rec_path: 录制文件路径（.f1rec）。
            out_path: 输出 JSONL 路径（已存在则覆盖；父目录自动创建）。

        Returns:
            统计 dict：``laps``（导出圈数）、``packets``（重放包数）、
            ``parse_errors``（跳过的非法/未知包数）。
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lap_agg = LapAggregator()
        style = StyleExtractor()
        current_setup: dict[str, Any] = {}
        session_uid: str | None = None
        track_id: int | None = None
        kerb_wired: int | None = None
        session_weather: int | None = None
        session_track_temp: float | None = None
        session_air_temp: float | None = None
        # 近似有效性：最近一帧 LapData 的 m_currentLapInvalid（0=有效圈）
        last_seen_invalid: int | None = None

        laps = 0
        packets = 0
        parse_errors = 0
        lap_t_start: float | None = None

        with open(out, "w", encoding="utf-8", newline="\n") as fout:
            with ReplayReader(str(f1rec_path)) as reader:
                while True:
                    item = reader.read_next()
                    if item is None:
                        break
                    t, data = item
                    packets += 1
                    try:
                        parsed = parse_packet(data)
                    except PacketTooShortError:
                        parse_errors += 1
                        continue
                    except Exception:
                        parse_errors += 1
                        continue
                    if parsed is None:
                        continue

                    header = parsed["header"]
                    if session_uid is None:
                        session_uid = str(header.session_uid)
                    pid = header.packet_id

                    if pid == 1 and "m_trackId" in parsed:
                        track_id = parsed["m_trackId"]
                        # 天气/温度：训练样本需要区分干/湿与温度工况（普世化特征）
                        weather = parsed.get("m_weather")
                        if weather is not None:
                            session_weather = weather
                        track_temp = parsed.get("m_trackTemperature")
                        if track_temp is not None:
                            session_track_temp = track_temp
                        air_temp = parsed.get("m_airTemperature")
                        if air_temp is not None:
                            session_air_temp = air_temp
                        # 首次见到该赛道时注入上下文，使样本带上按弯路肩特征
                        if track_id != kerb_wired:
                            _wire_kerb_track(lap_agg, parsed)
                            kerb_wired = track_id
                    elif pid == 2:
                        if "m_currentLapInvalid" in parsed:
                            last_seen_invalid = parsed["m_currentLapInvalid"]
                    elif pid == 5:
                        snap = {
                            k: parsed[k] for k in _SETUP_KEYS if k in parsed
                        }
                        if snap:
                            current_setup = snap

                    # 逐圈聚合器只认 lap_data / telemetry / motion_ex / car_status
                    if pid == 2:
                        lap_agg.on_lap_data(parsed)
                        style.on_lap_data(parsed)
                        if lap_t_start is None:
                            lap_t_start = t
                    elif pid == 6:
                        lap_agg.on_telemetry(parsed)
                        style.on_telemetry(parsed)
                    elif pid == 13:
                        lap_agg.on_motion_ex(parsed)
                    elif pid == 7:
                        lap_agg.on_car_status(parsed)

                    completed = lap_agg.take_completed_lap()
                    if completed is not None:
                        style_vec = style.take_completed()
                        sample = {
                            "session_uid": session_uid,
                            "track_id": track_id,
                            # 天气/温度工况（普世化特征：干/湿与温度影响调教取向）
                            "weather": session_weather,
                            "track_temp": session_track_temp,
                            "air_temp": session_air_temp,
                            "lap_number": completed.get("lap_number"),
                            "lap_time_ms": completed.get("lap_time_ms"),
                            # 近似：取圈末最近一帧的当前圈无效标志
                            "lap_valid": (
                                last_seen_invalid == 0
                                if last_seen_invalid is not None else None
                            ),
                            "setup": dict(current_setup),
                            "style": style_vec,
                            "lap_agg": completed,
                            # 遥测帧数（聚合器口径：Packet 6/13 帧计数）
                            "frames": completed.get("lap_frames", 0),
                            "t_start": lap_t_start,
                            "t_end": t,
                        }
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        laps += 1
                        lap_t_start = None

        stats = {"laps": laps, "packets": packets, "parse_errors": parse_errors}
        logger.info("training export done: %s -> %s (%s)", f1rec_path, out, stats)
        return stats
