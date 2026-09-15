#!/usr/bin/env python3
"""验证真实 F1 25 遥测数据 — 检查 UDP Packet 5 解析正确性。

读取 F1TelemetryCollector 采集的真实 Abu Dhabi 遥测 JSON 文件，
验证其中的调教参数（CarSetups）是否落在 setup_tuner.domain.setup
定义的合法范围内，并检测 engine_braking 字段是否存在。

若 JSON 中包含原始 UDP 包字节（hex/base64），则额外调用
parse_car_setups() 进行二进制解析交叉验证。

用法: python scripts/validate_real_telemetry.py [--data-dir PATH]
默认数据目录: D:\\F1TelemetryCollector\\dist\\data
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便导入 setup_tuner
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field  # noqa: E402
from setup_tuner.telemetry.packets import HEADER_SIZE, parse_car_setups  # noqa: E402

# UDP Packet 5 字段名 → 域参数名映射（用于原始字节解析交叉验证）
UDP_TO_DOMAIN: dict[str, str] = {
    "m_frontWing": "front_wing",
    "m_rearWing": "rear_wing",
    "m_onThrottleDiff": "on_throttle_diff",
    "m_offThrottleDiff": "off_throttle_diff",
    "m_frontCamber": "front_camber",
    "m_rearCamber": "rear_camber",
    "m_frontToe": "front_toe",
    "m_rearToe": "rear_toe",
    "m_frontSuspension": "front_suspension",
    "m_rearSuspension": "rear_suspension",
    "m_frontAntiRollBar": "front_anti_roll_bar",
    "m_rearAntiRollBar": "rear_anti_roll_bar",
    "m_frontSuspensionHeight": "front_ride_height",
    "m_rearSuspensionHeight": "rear_ride_height",
    "m_brakePressure": "brake_pressure",
    "m_brakeBias": "brake_bias",
    "m_engineBraking": "engine_braking",
    "m_frontLeftTyrePressure": "front_left_tyre_pressure",
    "m_frontRightTyrePressure": "front_right_tyre_pressure",
    "m_rearLeftTyrePressure": "rear_left_tyre_pressure",
    "m_rearRightTyrePressure": "rear_right_tyre_pressure",
}


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_lap_data(filepath: Path) -> dict:
    """加载单圈遥测 JSON 数据。

    Raises:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: JSON 解析失败。
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def extract_setup(data: dict) -> dict | None:
    """从遥测数据中提取调教参数字典。

    依次尝试常见字段名: setup / car_setups / packet_5 / car_setup。
    返回找到的第一个字典，若均不存在则返回 None。
    """
    for key in ("setup", "car_setups", "packet_5", "car_setup", "CarSetups"):
        val = data.get(key)
        if isinstance(val, dict):
            return val
    return None


def find_raw_packets(data: dict) -> list[bytes]:
    """在遥测数据中搜索原始 UDP 包字节（hex 或 base64 编码）。

    查找字段名包含 raw/hex/base64/packet 且值为字符串的条目，
    尝试解码为字节。仅保留长度 >= 包头(29) 的有效候选。
    """
    candidates: list[bytes] = []
    _collect_raw_recursive(data, candidates)
    return [b for b in candidates if len(b) >= HEADER_SIZE]


def _collect_raw_recursive(obj: object, out: list[bytes]) -> None:
    """递归搜索字典/列表中疑似原始包字节的字符串值。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, str):
                if any(tag in k.lower() for tag in ("raw", "hex", "base64", "packet_bytes")):
                    decoded = _try_decode_bytes(v)
                    if decoded is not None:
                        out.append(decoded)
            _collect_raw_recursive(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_raw_recursive(item, out)


def _try_decode_bytes(value: str) -> bytes | None:
    """尝试将字符串按 hex 或 base64 解码为字节，失败返回 None。"""
    # 尝试 hex 解码
    try:
        return bytes.fromhex(value)
    except (ValueError, TypeError):
        pass
    # 尝试 base64 解码
    try:
        decoded = base64.b64decode(value, validate=True)
        if len(decoded) > 0:
            return decoded
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# 验证逻辑
# --------------------------------------------------------------------------- #
def validate_setup_values(setup_data: dict) -> list[dict]:
    """验证调教参数值是否在合法范围内。返回违规项列表。

    对 ALL_SETUP_FIELDS 中每个参数，若 setup_data 中存在对应键，
    则检查值是否落在 [min_val, max_val] 区间。违规项含字段名、
    实际值、合法范围与违规原因。
    """
    violations: list[dict] = []
    for spec in ALL_SETUP_FIELDS:
        if spec.name not in setup_data:
            continue
        raw_val = setup_data[spec.name]
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            violations.append({
                "field": spec.name,
                "value": raw_val,
                "range": f"[{spec.min_val:g}, {spec.max_val:g}]",
                "reason": f"值无法转为 float: {raw_val!r}",
            })
            continue
        if val < spec.min_val or val > spec.max_val:
            violations.append({
                "field": spec.name,
                "value": val,
                "range": f"[{spec.min_val:g}, {spec.max_val:g}]",
                "reason": f"超出范围: {val:g} 不在 [{spec.min_val:g}, {spec.max_val:g}]",
            })
    return violations


def check_missing_fields(setup_data: dict) -> list[str]:
    """检查 setup_data 中缺失的域参数名（返回缺失列表）。

    engine_braking 是 F1 24/25 新增字段，缺失时需特别报告。
    """
    return [spec.name for spec in ALL_SETUP_FIELDS if spec.name not in setup_data]


def validate_packet_parsing(raw_bytes: bytes, player_index: int) -> dict:
    """用 parse_car_setups 解析原始 UDP 字节，返回解析结果。

    成功时返回 {"ok": True, "parsed": {...}, "violations": [...]}；
    失败时返回 {"ok": False, "error": "..."}。
    """
    try:
        parsed = parse_car_setups(raw_bytes, player_index)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    # 将 UDP 字段名映射为域参数名并验证范围
    mapped: dict[str, float] = {}
    for udp_name, domain_name in UDP_TO_DOMAIN.items():
        if udp_name in parsed:
            mapped[domain_name] = float(parsed[udp_name])
    violations = validate_setup_values(mapped)
    return {"ok": True, "parsed": parsed, "mapped": mapped, "violations": violations}


def validate_engine_braking(setup_data: dict) -> dict:
    """检查 engine_braking 字段是否存在且取值合法。

    返回 {"present": bool, "value": ..., "valid": bool, "reason": str}。
    """
    spec = get_field("engine_braking")
    if "engine_braking" not in setup_data:
        return {
            "present": False,
            "value": None,
            "valid": False,
            "reason": "engine_braking 字段缺失（F1 24/25 新增字段未采集）",
        }
    val = setup_data["engine_braking"]
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return {"present": True, "value": val, "valid": False,
                "reason": f"值无法转为 float: {val!r}"}
    in_range = spec.min_val <= fval <= spec.max_val
    reason = "合法" if in_range else f"超出范围 [{spec.min_val:g}, {spec.max_val:g}]"
    return {"present": True, "value": fval, "valid": in_range, "reason": reason}


# --------------------------------------------------------------------------- #
# 报告生成
# --------------------------------------------------------------------------- #
def _report_setup_values(setup: dict) -> list[str]:
    """格式化单圈调教参数值列表。"""
    lines: list[str] = [f"  调教参数字段数: {len(setup)}"]
    if not setup:
        return lines
    lines.append("  调教参数值:")
    for spec in ALL_SETUP_FIELDS:
        if spec.name not in setup:
            continue
        val = setup[spec.name]
        in_range = spec.min_val <= float(val) <= spec.max_val if _is_float(val) else False
        mark = "✅" if in_range else "❌"
        lines.append(
            f"    {mark} {spec.name} = {val}  "
            f"(合法范围 [{spec.min_val:g}, {spec.max_val:g}] {spec.unit})",
        )
    return lines


def _report_violations(violations: list[dict]) -> list[str]:
    """格式化单圈违规项列表。"""
    if not violations:
        return ["  ✅ 所有参数均在合法范围内"]
    lines = [f"  ⚠ 超出范围的参数: {len(violations)} 项"]
    for v in violations:
        lines.append(f"    - {v['field']}: {v['value']}  范围 {v['range']}  ({v['reason']})")
    return lines


def _report_engine_braking(eb: dict) -> tuple[list[str], bool, bool]:
    """格式化 engine_braking 检查。返回 (行列表, 是否合格, 是否缺失)。"""
    if not eb:
        return [], False, False
    if eb["present"]:
        mark = "✅" if eb["valid"] else "❌"
        return [f"  {mark} engine_braking: 存在, 值={eb['value']}, {eb['reason']}"], eb["valid"], False
    return [f"  ❌ engine_braking: {eb['reason']}"], False, True


def _report_packet_parse(pkt: list[dict]) -> list[str]:
    """格式化原始 UDP 包解析结果。"""
    if not pkt:
        return ["  原始 UDP 包解析: 无原始字节（仅验证 JSON 解码值）"]
    lines = [f"  原始 UDP 包解析: {len(pkt)} 个"]
    for p in pkt:
        if p["ok"]:
            pv = p.get("violations", [])
            mark = "✅" if not pv else "❌"
            lines.append(f"    {mark} 解析成功, 违规 {len(pv)} 项")
        else:
            lines.append(f"    ❌ 解析失败: {p['error']}")
    return lines


def _report_per_lap(r: dict) -> tuple[list[str], int, int, bool, bool]:
    """格式化单圈报告段。返回 (行列表, 违规数, 缺失数, eb合格, eb缺失)。"""
    lines = ["-" * 72, f"文件: {r['file']}"]
    lines.append(f"  赛道: {r.get('track_name', '未知')} (track_id={r.get('track_id', '?')})")
    lines.append(f"  圈号: {r.get('lap_number', '?')}  用时: {r.get('lap_time_str', '?')}")
    lines.append(f"  采样数: {r.get('sample_count', '?')}  有效圈: {r.get('lap_valid', '?')}")
    lines.extend(_report_setup_values(r.get("setup", {})))
    violations = r.get("violations", [])
    lines.extend(_report_violations(violations))
    missing = r.get("missing_fields", [])
    if missing:
        lines.append(f"  ⚠ 缺失的域参数: {missing}")
    eb_lines, eb_ok, eb_miss = _report_engine_braking(r.get("engine_braking", {}))
    lines.extend(eb_lines)
    lines.extend(_report_packet_parse(r.get("packet_parse", [])))
    return lines, len(violations), len(missing), eb_ok, eb_miss


def generate_report(results: list[dict]) -> str:
    """生成验证报告（纯文本，简体中文）。"""
    lines = ["=" * 72, "F1 25 真实遥测数据验证报告", "=" * 72, f"处理圈数: {len(results)}", ""]
    total_v = total_m = eb_ok = eb_miss = 0
    for r in results:
        lap_lines, nv, nm, ok, miss = _report_per_lap(r)
        lines.extend(lap_lines)
        total_v += nv
        total_m += nm
        eb_ok += 1 if ok else 0
        eb_miss += 1 if miss else 0
    lines.extend(_report_summary(total_v, total_m, eb_ok, eb_miss))
    return "\n".join(lines)


def _report_summary(total_v: int, total_m: int, eb_ok: int, eb_miss: int) -> list[str]:
    """格式化汇总段。"""
    all_valid = total_v == 0 and total_m == 0
    return [
        "",
        "=" * 72,
        "汇总",
        "=" * 72,
        f"总违规参数数: {total_v}",
        f"总缺失参数数: {total_m}",
        f"engine_braking 合格圈数: {eb_ok}",
        f"engine_braking 缺失圈数: {eb_miss}",
        f"总体结论: {'✅ 全部通过' if all_valid else '❌ 存在异常'}",
    ]


def _is_float(val: object) -> bool:
    """判断值是否可转为 float。"""
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def process_lap(lap_file: Path, verbose: bool) -> dict:
    """处理单圈遥测文件，返回验证结果字典。"""
    data = load_lap_data(lap_file)
    setup = extract_setup(data) or {}
    violations = validate_setup_values(setup) if setup else []
    missing = check_missing_fields(setup) if setup else [s.name for s in ALL_SETUP_FIELDS]
    eb = validate_engine_braking(setup) if setup else {
        "present": False, "value": None, "valid": False, "reason": "setup 缺失",
    }
    # 搜索原始 UDP 包字节
    raw_packets = find_raw_packets(data)
    pkt_results: list[dict] = []
    for raw in raw_packets[:5]:  # 最多解析 5 个原始包
        player_idx = data.get("player_car_index", 0)
        pkt_results.append(validate_packet_parsing(raw, player_idx))
    if verbose and raw_packets:
        print(f"    发现 {len(raw_packets)} 个原始包候选")
    return {
        "file": lap_file.name,
        "track_name": data.get("track_name"),
        "track_id": data.get("track_id"),
        "lap_number": data.get("lap_number"),
        "lap_time_str": data.get("lap_time_str"),
        "lap_valid": data.get("lap_valid"),
        "sample_count": data.get("sample_count"),
        "setup": setup,
        "violations": violations,
        "missing_fields": missing,
        "engine_braking": eb,
        "packet_parse": pkt_results,
        "valid": len(violations) == 0 and len(missing) == 0 and eb["valid"],
    }


def main() -> None:
    """脚本主入口：解析参数、处理每圈数据、生成报告。"""
    parser = argparse.ArgumentParser(description="验证真实 F1 25 遥测数据")
    parser.add_argument(
        "--data-dir",
        default=r"D:\F1TelemetryCollector\dist\data",
        help="遥测数据目录",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)

    lap_files = sorted(data_dir.glob("lap_*.json"))
    if not lap_files:
        print(f"错误: 未在 {data_dir} 找到遥测数据文件 (lap_*.json)")
        sys.exit(1)

    print(f"找到 {len(lap_files)} 圈遥测数据")

    results: list[dict] = []
    for lap_file in lap_files:
        print(f"\n处理: {lap_file.name}")
        result = process_lap(lap_file, args.verbose)
        results.append(result)
        status = "✅" if result["valid"] else "❌"
        print(f"  {status} 违规 {len(result['violations'])} 项, 缺失 {len(result['missing_fields'])} 项")

    report = generate_report(results)
    print()
    print(report)

    failed = sum(1 for r in results if not r["valid"])
    if failed:
        print(f"\n❌ {failed} 圈验证失败")
        sys.exit(1)
    else:
        print("\n✅ 全部验证通过")
        sys.exit(0)


if __name__ == "__main__":
    main()