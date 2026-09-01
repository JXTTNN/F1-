#!/usr/bin/env python
"""从真实 F1 26 遥测 JSONL 抽取轻量 pytest fixture 回归集。

输入: UDP 采集器产出的 JSONL（每行一个 UDP 包 JSON，含 ``hex`` 完整包字节 + 头部字段）。
输出: fixture 化 JSONL（保留 ``ts/len/src/packetFormat.../packetId/name/hex`` 等原始
字段以便回溯源文件；合成边界帧额外带 ``synthetic``/``boundary`` 标记）。

抽样策略:
- 每包型等间隔抽 ``--per-type`` 帧（覆盖会话早/中/晚期）;
- Event(packetId=3) 按事件码全覆盖（每码 1 帧）;
- Motion 的 g-force 饱和哨兵 (raw -32768) 帧若存在则强制纳入;
- 合成边界帧: 空帧(len=0) + 截断帧(仅包头)。

用法:
    python scripts/sample_f1_26_fixtures.py --in <input.jsonl> --out <output.jsonl>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter

HEADER_SIZE = 29
NUM_CARS = 24
# Motion per-car 布局 (与 f1opt.telemetry.packets 一致): 6 float + 9 int16 + 3 float = 54B
_MOTION_CAR_FMT = "<" + "f" * 6 + "h" * 9 + "f" * 3
_MOTION_CAR_SIZE = struct.calcsize(_MOTION_CAR_FMT)


def _event_code(hex_str: str) -> str:
    """从 Event 包 hex 提取 4 字节事件码（头 29B 之后）。"""
    code_hex = hex_str[58:66]
    try:
        return bytes.fromhex(code_hex).decode("ascii", "replace").rstrip("\x00")
    except Exception:
        return "?"


def _motion_has_sentinel(hex_str: str) -> bool:
    """Motion 包是否含 g-force raw -32768 饱和哨兵。"""
    try:
        body = bytes.fromhex(hex_str)[HEADER_SIZE:]
    except Exception:
        return False
    for car in range(NUM_CARS):
        base = car * _MOTION_CAR_SIZE
        if base + _MOTION_CAR_SIZE > len(body):
            break
        vals = struct.unpack(_MOTION_CAR_FMT, body[base : base + _MOTION_CAR_SIZE])
        gforce = vals[12:15]  # 9 int16 中末 3 个是 g-force (6 方向在前)
        if any(x == -32768 for x in gforce):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="输入 JSONL 路径")
    ap.add_argument("--out", required=True, help="输出 fixture JSONL 路径")
    ap.add_argument("--per-type", type=int, default=30, help="每包型抽样帧数")
    args = ap.parse_args()

    target = args.per_type

    # ---- Pass 1: 统计包型数量 + 事件码 + 哨兵帧 ts ----
    counts: Counter[int] = Counter()
    event_codes: Counter[str] = Counter()
    sentinel_ts = None
    first_motion_hex = None
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            pid = o.get("packetId")
            counts[pid] += 1
            if pid == 3:
                event_codes[_event_code(o.get("hex", ""))] += 1
            if pid == 0:
                if first_motion_hex is None:
                    first_motion_hex = o.get("hex", "")
                if sentinel_ts is None and _motion_has_sentinel(o.get("hex", "")):
                    sentinel_ts = o.get("ts")

    # 每包型等间隔采样索引集合
    keep_indices: dict[int, set[int]] = {}
    for pid, cnt in counts.items():
        if pid == 3:
            continue  # Event 按码单独处理
        k = min(target, cnt)
        keep_indices[pid] = {int(i * cnt / k) for i in range(k)}

    # ---- Pass 2: 抽样 ----
    sampled: dict[int, list[dict]] = {pid: [] for pid in counts}
    seen_event_codes: set[str] = set()
    type_idx: Counter[int] = Counter()

    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            pid = o.get("packetId")
            idx = type_idx[pid]
            type_idx[pid] += 1

            if pid == 3:
                code = _event_code(o.get("hex", ""))
                if code and code not in seen_event_codes:
                    seen_event_codes.add(code)
                    sampled[3].append(o)
            else:
                if idx in keep_indices.get(pid, set()):
                    sampled[pid].append(o)
            # 哨兵帧强制纳入
            if pid == 0 and sentinel_ts is not None and o.get("ts") == sentinel_ts:
                if not any(x.get("ts") == sentinel_ts for x in sampled[0]):
                    sampled[0].append(o)

    # ---- 合成边界帧 ----
    boundary_frames: list[dict] = []
    # 空帧
    boundary_frames.append(
        {
            "ts": None,
            "len": 0,
            "src": "synthetic",
            "packetFormat": 2026,
            "gameYear": 25,
            "gameMajorVersion": 1,
            "gameMinorVersion": 24,
            "packetVersion": 1,
            "packetId": 0,
            "sessionUID": 0,
            "sessionTime": 0.0,
            "frameIdentifier": 0,
            "overallFrameIdentifier": 0,
            "playerCarIndex": 0,
            "secondaryPlayerCarIndex": 255,
            "name": "Motion",
            "hex": "",
            "synthetic": True,
            "boundary": "empty",
        }
    )
    # 截断帧（仅包头，缺 body）
    if first_motion_hex:
        header_hex = first_motion_hex[: HEADER_SIZE * 2]
        boundary_frames.append(
            {
                "ts": None,
                "len": HEADER_SIZE,
                "src": "synthetic",
                "packetFormat": 2026,
                "gameYear": 25,
                "gameMajorVersion": 1,
                "gameMinorVersion": 24,
                "packetVersion": 1,
                "packetId": 0,
                "sessionUID": 0,
                "sessionTime": 0.0,
                "frameIdentifier": 0,
                "overallFrameIdentifier": 0,
                "playerCarIndex": 0,
                "secondaryPlayerCarIndex": 255,
                "name": "Motion",
                "hex": header_hex,
                "synthetic": True,
                "boundary": "truncated",
            }
        )

    # ---- 写输出 ----
    out_frames: list[dict] = []
    for pid in sorted(sampled):
        out_frames.extend(sampled[pid])
    out_frames.extend(boundary_frames)

    with open(args.out, "w", encoding="utf-8") as f:
        for fr in out_frames:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")

    n_real = sum(len(v) for v in sampled.values())
    print(f"per-type target={target}")
    print(f"packetId counts: {dict(sorted(counts.items()))}")
    print(f"event codes sampled: {sorted(seen_event_codes)}")
    print(f"sentinel frame ts: {sentinel_ts}")
    print(f"sampled real frames: {n_real}; synthetic: {len(boundary_frames)}")
    print(f"total written: {n_real + len(boundary_frames)} → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
