"""F1 2026 官方公开专业遥测抓取器（仅 2026，严格校验）。

数据源：TracingInsights 2026 公共仓库（Ergast + F1 官方信号流，经 FastF1 标准化）
    https://github.com/TracingInsights/2026

**硬性约束（用户要求）**：
- 只允许 2026 赛季数据：赛道名必须在 2026 赛历白名单内，且每条圈数据的时间戳
  必须落在 2026 年；任何一条不是 2026 的数据 → 整场丢弃并报错（不静默跳过）。
- 必须"全面"：逐圈逐点遥测（speed/throttle/brake/gear/rpm/drs/acc/x,y,z）
  全部抓下来，不做字段裁剪。

产物（本地缓存，`.ref/` 已在 .gitignore，不入库）::

    .ref/tracing_2026/<Race>/<Session>/
        session_laptimes.json     # 逐圈圈速/扇区/配方/轮胎寿命/天气列
        weather.json              # 会话天气
        manifest.json             # 抓取清单（URL/时间/校验和/年检结果）
        <DRV>/<lap>_tel.json      # 逐圈逐点遥测

用法::

    # 先小规模验证（默认：澳大利亚站正赛，3 位车手各 2 圈）
    python scripts/fetch_official_2026.py --verify

    # 指定比赛/会话/车手/圈数
    python scripts/fetch_official_2026.py --race "Italian Grand Prix" --session Race \
        --drivers VER,NOR --max-laps 5

    # 抓整场正赛（全部车手全部圈）
    python scripts/fetch_official_2026.py --race "Italian Grand Prix" --session Race --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".ref" / "tracing_2026"
RAW = "https://raw.githubusercontent.com/TracingInsights/2026/main"

#: 2026 赛历（官方 24 站；TracingInsights 目录名与官方大奖赛名一致）。
#: 只允许这些比赛 → 杜绝把 2025/2024 赛季文件混进来。
CALENDAR_2026: tuple[str, ...] = (
    "Australian Grand Prix",
    "Chinese Grand Prix",
    "Japanese Grand Prix",
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Miami Grand Prix",
    "Canadian Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Belgian Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "Sao Paulo Grand Prix",
    "Las Vegas Grand Prix",
    "Qatar Grand Prix",
    "Abu Dhabi Grand Prix",
)

#: 允许的会话目录名
SESSIONS: tuple[str, ...] = (
    "Practice 1", "Practice 2", "Practice 3",
    "Qualifying", "Sprint Qualifying", "Sprint", "Race",
)

#: 逐点遥测必须包含的字段（缺任一 → 该圈判定为脏数据，记录但不静默接受）
REQUIRED_TEL_KEYS: tuple[str, ...] = (
    "time", "rpm", "speed", "gear", "throttle", "brake", "drs",
    "distance", "acc_x", "acc_y", "acc_z", "x", "y", "z",
)

_TIMEOUT = 60
_RETRIES = 3


def _get(url: str) -> bytes:
    """带重试的 HTTP GET（GitHub raw 偶发 connection reset）。"""
    last: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "f1opt-2026-fetch"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"download failed after {_RETRIES} tries: {url} ({last})")


def _get_json(url: str) -> Any:
    return json.loads(_get(url).decode("utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _verify_year_2026(laptimes: dict[str, Any], race: str) -> tuple[int, int]:
    """校验每圈时间戳都落在 2026 年。

    Returns:
        (checked, non_2026) —— checked 为检查的圈数，non_2026 为不属于 2026 的圈数。

    Raises:
        RuntimeError: 发现任何非 2026 数据（宁缺勿错，整场丢弃）。
    """
    stamps = laptimes.get("lSD")
    if not isinstance(stamps, list) or not stamps:
        raise RuntimeError(f"{race}: session_laptimes 缺少 lSD 时间列，无法年检")
    # 该列可能含 'None'（未完成/无效圈）——只对真实时间戳做年检，
    # 但要求至少有 1 条真实时间戳，避免"全 None 也能过检"。
    real = [str(s) for s in stamps if s is not None and str(s) != "None"]
    if not real:
        raise RuntimeError(f"{race}: lSD 全为 None，无法年检 → 丢弃")
    bad = [s for s in real if not s.startswith("2026-")]
    if bad:
        raise RuntimeError(
            f"{race}: 发现 {len(bad)} 条非 2026 时间戳（示例 {bad[:2]}）→ 丢弃"
        )
    return len(real), 0


def _drivers_and_laps(laptimes: dict[str, Any]) -> dict[str, list[int]]:
    """从 session_laptimes 列式结构还原 {车手: [圈号...]}。"""
    drv = laptimes.get("drv")
    lap = laptimes.get("lap")
    if not isinstance(drv, list) or not isinstance(lap, list) or len(drv) != len(lap):
        raise RuntimeError("session_laptimes 结构异常（drv/lap 列不齐）")
    out: dict[str, list[int]] = {}
    for d, n in zip(drv, lap):
        if not isinstance(d, str) or not isinstance(n, int):
            continue
        out.setdefault(d, []).append(n)
    for v in out.values():
        v.sort()
    return out


def _validate_tel(tel: dict[str, Any]) -> list[str]:
    """校验逐点遥测字段完整性，返回缺失字段列表（空 = 合格）。"""
    inner = tel.get("tel")
    if not isinstance(inner, dict):
        return ["<tel>"]
    missing = [k for k in REQUIRED_TEL_KEYS if k not in inner]
    lengths = {len(inner[k]) for k in inner if isinstance(inner.get(k), list)}
    if len(lengths) > 1:
        missing.append("<列长不一致>")
    return missing


def fetch_session(
    race: str,
    session: str,
    drivers: list[str] | None,
    max_laps: int | None,
    out_root: Path = CACHE,
) -> dict[str, Any]:
    """抓取一场会话的全部所需数据并落盘，返回抓取清单。"""
    if race not in CALENDAR_2026:
        raise SystemExit(f"拒绝：{race!r} 不在 2026 赛历白名单内（防止混入旧赛季）")
    if session not in SESSIONS:
        raise SystemExit(f"拒绝：会话 {session!r} 非法，合法值 {SESSIONS}")

    sub = f"{race}/{session}"
    out_dir = out_root / race / session
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "race": race, "session": session, "season": 2026,
        "source": f"{RAW}/{urllib.parse.quote(sub)}",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "year_check": None, "laps": [],
    }

    # 1) 会话级文件：laptimes / weather / corners
    for name in ("session_laptimes.json", "weather.json", "corners.json", "corner.json"):
        url = f"{RAW}/{urllib.parse.quote(sub)}/{name}"
        try:
            raw = _get(url)
        except RuntimeError:
            manifest.setdefault("missing", []).append(name)
            continue
        (out_dir / name).write_bytes(raw)
        manifest.setdefault("files", {})[name] = {
            "url": url, "sha256": _sha256(raw), "bytes": len(raw),
        }
        if name == "session_laptimes.json":
            manifest["files"][name]["sha256"] = _sha256(raw)

    lt_path = out_dir / "session_laptimes.json"
    if not lt_path.exists():
        raise SystemExit(f"{race}/{session}: 缺少 session_laptimes.json，无法继续")
    laptimes = json.loads(lt_path.read_text(encoding="utf-8"))

    # 2) 年检：必须全部是 2026
    checked, non_2026 = _verify_year_2026(laptimes, f"{race}/{session}")
    manifest["year_check"] = {"checked_rows": checked, "non_2026_rows": non_2026}

    # 3) 车手 × 圈号
    per_driver = _drivers_and_laps(laptimes)
    if drivers:
        wanted = [d for d in drivers if d in per_driver]
        missing_drivers = sorted(set(drivers) - set(wanted))
        if missing_drivers:
            manifest["missing_drivers"] = missing_drivers
    else:
        wanted = sorted(per_driver)

    # 4) 逐圈逐点遥测
    dirty: list[str] = []
    for drv in wanted:
        laps = per_driver[drv]
        if max_laps is not None:
            laps = laps[:max_laps]
        ddir = out_dir / drv
        ddir.mkdir(parents=True, exist_ok=True)
        for lap in laps:
            url = f"{RAW}/{urllib.parse.quote(sub)}/{drv}/{lap}_tel.json"
            try:
                raw = _get(url)
            except RuntimeError:
                manifest["laps"].append(
                    {"driver": drv, "lap": lap, "status": "missing"}
                )
                continue
            tel = json.loads(raw.decode("utf-8"))
            missing = _validate_tel(tel)
            (ddir / f"{lap}_tel.json").write_bytes(raw)
            rec = {
                "driver": drv, "lap": lap, "status": "ok",
                "sha256": _sha256(raw), "bytes": len(raw),
                "points": len(tel.get("tel", {}).get("time", [])),
            }
            if missing:
                rec["status"] = "dirty"
                rec["missing_keys"] = missing
                dirty.append(f"{drv}/{lap}")
            manifest["laps"].append(rec)

    manifest["dirty_laps"] = dirty
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F1 2026 官方公开遥测抓取（仅 2026）")
    ap.add_argument("--race", default="Australian Grand Prix",
                    help="大奖赛名（须在 2026 赛历白名单内）")
    ap.add_argument("--session", default="Race", choices=SESSIONS)
    ap.add_argument("--drivers", default="", help="逗号分隔车手代码；空=全部")
    ap.add_argument("--max-laps", type=int, default=2, help="每位车手最多抓几圈")
    ap.add_argument("--all", action="store_true", help="全部车手全部圈（忽略 --max-laps）")
    ap.add_argument("--verify", action="store_true", help="小规模验证模式（默认行为）")
    ap.add_argument("--out", default=str(CACHE))
    args = ap.parse_args(argv)

    # --verify 与小规模默认一致；--all 放开圈数限制
    max_laps = None if args.all else args.max_laps
    drivers = [s.strip() for s in args.drivers.split(",") if s.strip()] or None

    manifest = fetch_session(args.race, args.session, drivers, max_laps, Path(args.out))
    ok = sum(1 for r in manifest["laps"] if r.get("status") == "ok")
    dirty = len(manifest.get("dirty_laps") or [])
    missing = sum(1 for r in manifest["laps"] if r.get("status") == "missing")
    print(f"[2026 抓取] {args.race} / {args.session}")
    print(f"  年检: {manifest['year_check']}")
    print(f"  圈数据: ok={ok} dirty={dirty} missing={missing}")
    if manifest.get("missing_drivers"):
        print(f"  未找到车手: {manifest['missing_drivers']}")
    print(f"  输出: {Path(args.out) / args.race / args.session}")
    return 0 if not dirty and not missing and not manifest.get("missing_drivers") else 1


if __name__ == "__main__":
    sys.exit(main())
