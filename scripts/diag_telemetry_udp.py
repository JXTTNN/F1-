#!/usr/bin/env python3
"""F1OPT 遥测 UDP 链路诊断（纯标准库，无第三方依赖）。

绑定 ``0.0.0.0:20777``（所有网卡，含广播），实时显示收到的 F1 UDP 包
（类型 / 来源地址 / 计数），用于判断「0 包」到底是**接收器问题**还是
**游戏没把数据发出来**。

用法::

    python scripts/diag_telemetry_udp.py [--port 20777] [--seconds 20]

注意：运行前请先关闭「遥测接收器」（否则端口被占用，无法绑定）。
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from collections import Counter

# F1 UDP 包头（29 字节）
HEADER_FMT = "<HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

PACKET_NAMES = {
    0: "Motion", 1: "Session", 2: "LapData", 3: "Event", 4: "Participants",
    5: "CarSetups", 6: "CarTelemetry", 7: "CarStatus", 8: "FinalClassification",
    9: "LobbyInfo", 10: "CarDamage", 11: "SessionHistory", 12: "TyreSets",
    13: "MotionEx", 14: "TimeTrial", 15: "LapPositions", 16: "CarTelemetryData2",
}


def local_ips() -> list[str]:
    """列出本机所有 IPv4 地址（供游戏端填写参考）。"""
    ips: set[str] = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 不发包，只为拿到出口网卡地址
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ips)


def main() -> int:
    ap = argparse.ArgumentParser(description="F1OPT 遥测 UDP 链路诊断")
    ap.add_argument("--port", type=int, default=20777, help="UDP 端口（默认 20777）")
    ap.add_argument("--seconds", type=int, default=20, help="监听时长秒（默认 20）")
    args = ap.parse_args()

    print("=" * 56)
    print("F1OPT 遥测 UDP 链路诊断")
    print("=" * 56)
    print(f"本机 IPv4 : {', '.join(local_ips()) or '(未知)'}")
    print(f"监听       : 0.0.0.0:{args.port}（所有网卡，含广播）")
    print(f"游戏端请填 : UDP 遥测=开启, 端口={args.port}, IP=127.0.0.1 或上面任一 IP")
    print("提示       : 先关掉「遥测接收器」，否则端口被占用")
    print("-" * 56)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", args.port))
    except OSError as exc:
        print(f"[失败] 无法绑定 {args.port}：{exc}")
        print("       多半是「遥测接收器」还开着，请先关闭它。")
        return 1
    sock.settimeout(0.5)

    counter: Counter[str] = Counter()
    senders: Counter[str] = Counter()
    total = 0
    deadline = time.monotonic() + args.seconds
    last_print = 0.0

    while time.monotonic() < deadline:
        try:
            data, addr = sock.recvfrom(2048)
        except TimeoutError:
            pass
        else:
            total += 1
            senders[addr[0]] += 1
            pid = -1
            if len(data) >= HEADER_SIZE:
                try:
                    pid = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])[5]
                except struct.error:
                    pid = -1
            counter[PACKET_NAMES.get(pid, f"未知({pid})")] += 1
        now = time.monotonic()
        if now - last_print >= 1.0:
            last_print = now
            rest = max(0, int(deadline - now))
            brief = "  ".join(f"{k}:{v}" for k, v in counter.most_common(4))
            print(f"\r[{rest:2d}s] 已收 {total} 包   {brief}     ", end="", flush=True)

    sock.close()
    print()
    print("-" * 56)
    if total == 0:
        print("结论：0 包 —— 游戏没有把 UDP 数据发到本机。请逐条检查：")
        print("  1) 游戏内 UDP 遥测是否**真的已开启**（部分版本改完要重启游戏）")
        print(f"  2) 端口是否填 {args.port}（须与接收器/诊断一致）")
        print("  3) 是否关闭了「UDP 广播模式」（广播模式会忽略你填的 IP）")
        print("  4) 是否已**进入赛道**（菜单/车库可能不发遥测）")
        print("  5) 若游戏跑在另一台设备，IP 不能填 127.0.0.1")
        return 2
    print(f"结论：收到 {total} 包，来源 {dict(senders)}")
    print("      链路正常 —— 用接收器按同样设置即可收到数据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
