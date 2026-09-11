#!/usr/bin/env python3
"""F1 2026 UDP telemetry RAW collector (standalone, pure stdlib).

Listens on UDP (default 0.0.0.0:20777), parses the 29-byte F1 packet header
(packetFormat=2026, per MacManley/f1-26-udp), recognizes all 17 packet types
(ID 0-16), and writes EVERY raw frame to a JSONL file:

    {"ts": <unix-float>, "len": <frame bytes>, "packetId": N, "name": "...",
     "packetFormat": 2026, "gameYear": 25, "sessionUID": ...,
     "sessionTime": ..., "frameIdentifier": ..., "playerCarIndex": ...,
     "hex": "<full raw frame hex>"}

Raw hex is always stored so no original bytes are lost; parsed header fields
are a convenience summary. Pure stdlib => small standalone PyInstaller exe.

Usage:
    python udp_telemetry_collector.py [--host 0.0.0.0] [--port 20777]
        [--out-dir <dir>] [--duration <seconds>] [--no-hex]
"""

import argparse
import json
import signal
import socket
import struct
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HEADER_FMT = "<HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 29

PACKET_NAMES = {
    0: "Motion",
    1: "Session",
    2: "LapData",
    3: "Event",
    4: "Participants",
    5: "CarSetups",
    6: "CarTelemetry",
    7: "CarStatus",
    8: "FinalClassification",
    9: "LobbyInfo",
    10: "CarDamage",
    11: "SessionHistory",
    12: "TyreSets",
    13: "MotionEx",
    14: "TimeTrial",
    15: "LapPositions",
    16: "CarTelemetryData2",
}


def parse_header(buf: bytes):
    """Parse the 29-byte packet header; return dict or None if too short."""
    if len(buf) < HEADER_SIZE:
        return None
    (packet_format, game_year, game_major, game_minor, packet_version, packet_id,
     session_uid, session_time, frame_identifier, overall_frame_identifier,
     player_car_index, secondary_player_car_index) = struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])
    return {
        "packetFormat": packet_format,
        "gameYear": game_year,
        "gameMajorVersion": game_major,
        "gameMinorVersion": game_minor,
        "packetVersion": packet_version,
        "packetId": packet_id,
        "sessionUID": session_uid,
        "sessionTime": round(session_time, 6),
        "frameIdentifier": frame_identifier,
        "overallFrameIdentifier": overall_frame_identifier,
        "playerCarIndex": player_car_index,
        "secondaryPlayerCarIndex": secondary_player_car_index,
    }


def default_out_dir() -> Path:
    """Default output dir: <exe/script parent>/F1_26_telemetry."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent
    return base / "F1_26_telemetry"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="F1 2026 UDP telemetry raw collector (writes every frame to JSONL)."
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=20777, help="UDP port (default 20777)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: <exe dir>/F1_26_telemetry)")
    parser.add_argument("--duration", type=float, default=None,
                        help="auto-stop after N seconds (default: run until Ctrl+C)")
    parser.add_argument("--no-hex", action="store_true",
                        help="omit raw hex from output (default: include)")
    args = parser.parse_args()

    out_dir = args.out_dir or default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"f1udp_{stamp}.jsonl"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as exc:
        print(f"[ERROR] cannot bind {args.host}:{args.port} -> {exc}")
        return 1
    sock.settimeout(0.5)  # wake periodically to check stop conditions

    stop = {"flag": False}

    def _on_sig(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    counts = {}
    total = 0
    short = 0
    start = time.time()
    last_report = start

    print(f"[F1-26 collector] listening on {args.host}:{args.port}")
    print(f"[F1-26 collector] writing raw frames to: {out_file}")
    print("[F1-26 collector] press Ctrl+C to stop")

    try:
        with out_file.open("a", encoding="utf-8") as fh:
            while not stop["flag"]:
                if args.duration is not None and (time.time() - start) >= args.duration:
                    break
                try:
                    data, addr = sock.recvfrom(65535)
                except TimeoutError:
                    continue
                except OSError:
                    break

                total += 1
                ts = time.time()
                hdr = parse_header(data)
                if hdr is None:
                    short += 1
                    record = {"ts": round(ts, 6), "len": len(data), "packetId": None,
                              "name": "SHORT_FRAME", "src": f"{addr[0]}:{addr[1]}"}
                else:
                    counts[hdr["packetId"]] = counts.get(hdr["packetId"], 0) + 1
                    record = {"ts": round(ts, 6), "len": len(data), "src": f"{addr[0]}:{addr[1]}"}
                    record.update(hdr)
                    record["name"] = PACKET_NAMES.get(hdr["packetId"], "UNKNOWN")

                # raw bytes always preserved (even for short/truncated frames)
                if not args.no_hex:
                    record["hex"] = data.hex()

                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

                now = time.time()
                if now - last_report >= 5.0:
                    last_report = now
                    print(f"[F1-26 collector] {total} frames, "
                          f"{len(counts)} packet types, "
                          f"out={out_file.name} ({(out_file.stat().st_size / 1024):.0f} KB)")
    finally:
        sock.close()

    elapsed = time.time() - start
    print("\n[F1-26 collector] stopped.")
    print(f"  duration     : {elapsed:.1f}s")
    print(f"  total frames : {total} (short/truncated: {short})")
    print(f"  output file  : {out_file}")
    if counts:
        print("  per-type     :")
        for pid in sorted(counts):
            print(f"    {pid:>2} {PACKET_NAMES.get(pid, 'UNKNOWN'):<22} {counts[pid]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
