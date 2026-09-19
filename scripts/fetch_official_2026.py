"""F1 2026 官方公开专业遥测抓取器（仅 2026，全量可续传）。

数据源：TracingInsights 2026 公共仓库（Ergast + F1 官方信号流，经 FastF1 标准化）
    https://github.com/TracingInsights/2026

**硬性约束（用户要求）**：
- 只允许 2026 赛季数据：赛道名必须在 2026 赛历白名单内；每场会话的
  ``session_laptimes.json`` 时间戳必须全部落在 2026 年。任何一场不是 2026 →
  该场丢弃并在清单里记为 rejected（不静默跳过、不写训练数据）。
- 必须"全面"：逐圈逐点遥测（speed/throttle/brake/gear/rpm/drs/acc/x,y,z）
  全抓，不做字段裁剪。

**可续传全量爬取**：进度记录在 ``.ref/tracing_2026/_crawl_state.json``，
重复运行自动跳过已完成项；支持 ``--max-files`` / ``--max-minutes`` 预算，
中断后重跑继续。

产物（本地缓存，`.ref/` 已在 .gitignore，不入库）::

    .ref/tracing_2026/<Race>/<Session>/
        session_laptimes.json     # 逐圈圈速/扇区/配方/轮胎寿命/天气列
        weather.json              # 会话天气
        manifest.json             # 抓取清单（URL/时间/校验和/年检结果）
        <DRV>/<lap>_tel.json      # 逐圈逐点遥测

用法::

    # 全量爬取（24 站 × 全部会话 × 全部车手全部圈）
    python scripts/fetch_official_2026.py --full

    # 带预算，可反复运行续传
    python scripts/fetch_official_2026.py --full --max-minutes 120 --max-files 5000

    # 单场验证
    python scripts/fetch_official_2026.py --race "Italian Grand Prix" --session Race \
        --drivers VER --max-laps 3

    # 查看进度
    python scripts/fetch_official_2026.py --status
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".ref" / "tracing_2026"
RAW = "https://raw.githubusercontent.com/TracingInsights/2026/main"
STATE_FILE = CACHE / "_crawl_state.json"

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
    "Barcelona Grand Prix",
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

#: 会话目录名，按数据价值排序（Race 优先）
SESSIONS: tuple[str, ...] = (
    "Race", "Qualifying", "Sprint", "Sprint Qualifying",
    "Practice 1", "Practice 2", "Practice 3",
)

#: 逐点遥测必须包含的字段（缺任一 → 该圈判定为脏数据，记录但不静默接受）
REQUIRED_TEL_KEYS: tuple[str, ...] = (
    "time", "rpm", "speed", "gear", "throttle", "brake", "drs",
    "distance", "acc_x", "acc_y", "acc_z", "x", "y", "z",
)

_TIMEOUT = 60
_RETRIES = 3
_POLITE_SLEEP = 0.01


# --------------------------------------------------------------------------- #
# 基础 IO / 状态
# --------------------------------------------------------------------------- #
def _state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "sessions": {}, "files_downloaded": 0, "bytes_downloaded": 0,
        "year_checks": {"2026": 0, "rejected": [], "absent": []},
        "dirty_laps": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _save_state(state: dict[str, Any]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _get(url: str) -> bytes | None:
    """带重试的 HTTP GET；404 返回 None（该场/该圈不存在）。"""
    last: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "f1opt-2026-fetch"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = resp.read()
            time.sleep(_POLITE_SLEEP)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except Exception as exc:  # noqa: BLE001 — 网络层杂类异常统一重试
            last = exc
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"download failed after {_RETRIES} tries: {url} ({last})")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _verify_year_2026(laptimes: dict[str, Any], where: str) -> dict[str, Any]:
    """校验每圈时间戳都落在 2026 年；非 2026 → 抛错（调用方记 rejected）。"""
    stamps = laptimes.get("lSD")
    if not isinstance(stamps, list) or not stamps:
        raise RuntimeError(f"{where}: session_laptimes 缺少 lSD 时间列，无法年检")
    # lSD 可能含 'None'（未完成圈）——只对真实时间戳做年检
    real = [str(s) for s in stamps if s is not None and str(s) != "None"]
    if not real:
        raise RuntimeError(f"{where}: lSD 全为 None，无法年检 → 丢弃")
    bad = [s for s in real if not s.startswith("2026-")]
    if bad:
        raise RuntimeError(
            f"{where}: 发现 {len(bad)} 条非 2026 时间戳（示例 {bad[:2]}）→ 丢弃"
        )
    return {"checked_rows": len(real), "non_2026_rows": 0}


def _drivers_and_laps(laptimes: dict[str, Any]) -> dict[str, list[int]]:
    drv, lap = laptimes.get("drv"), laptimes.get("lap")
    if not isinstance(drv, list) or not isinstance(lap, list) or len(drv) != len(lap):
        raise RuntimeError("session_laptimes 结构异常（drv/lap 列不齐）")
    out: dict[str, list[int]] = {}
    for d, n in zip(drv, lap, strict=True):
        if isinstance(d, str) and isinstance(n, int):
            out.setdefault(d, []).append(n)
    for v in out.values():
        v.sort()
    return out


def _validate_tel(tel: dict[str, Any]) -> list[str]:
    """校验逐点遥测字段完整性，返回缺失项列表（空 = 合格）。"""
    inner = tel.get("tel")
    if not isinstance(inner, dict):
        return ["<tel>"]
    missing = [k for k in REQUIRED_TEL_KEYS if k not in inner]
    lengths = {len(inner[k]) for k in inner if isinstance(inner.get(k), list)}
    if len(lengths) > 1:
        missing.append("<列长不一致>")
    return missing


# --------------------------------------------------------------------------- #
# 单场抓取（可续传）
# --------------------------------------------------------------------------- #
def fetch_session(
    race: str,
    session: str,
    drivers: list[str] | None = None,
    max_laps: int | None = None,
    out_root: Path = CACHE,
    state: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    workers: int = 24,
) -> dict[str, Any]:
    """抓取一场会话（已存在文件自动跳过；budget 可在中途停下）。

    Args:
        workers: 逐点遥测并发下载数（GitHub raw CDN；8 路可把单场从
            分钟级压到十几秒，且不触发限流）。
    """
    if race not in CALENDAR_2026:
        raise SystemExit(f"拒绝：{race!r} 不在 2026 赛历白名单内（防止混入旧赛季）")
    if session not in SESSIONS:
        raise SystemExit(f"拒绝：会话 {session!r} 非法，合法值 {SESSIONS}")

    sub = f"{race}/{session}"
    quoted = urllib.parse.quote(sub)
    out_dir = out_root / race / session
    manifest: dict[str, Any] = {
        "race": race, "session": session, "season": 2026,
        "source": f"{RAW}/{quoted}",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "year_check": None, "laps": [], "missing": [],
    }

    # 1) 会话级文件
    lt_path = out_dir / "session_laptimes.json"
    if not lt_path.exists():
        raw = _get(f"{RAW}/{quoted}/session_laptimes.json")
        if raw is None:
            return {"absent": True, "race": race, "session": session}
        out_dir.mkdir(parents=True, exist_ok=True)
        lt_path.write_bytes(raw)
        if budget is not None:
            budget["files"] += 1
            budget["bytes"] += len(raw)
        manifest["files"] = {"session_laptimes.json": {"sha256": _sha256(raw),
                                                       "bytes": len(raw)}}
    for extra in ("weather.json", "corners.json", "corner.json"):
        p = out_dir / extra
        if p.exists():
            continue
        raw = _get(f"{RAW}/{quoted}/{extra}")
        if raw is None:
            manifest["missing"].append(extra)
            continue
        p.write_bytes(raw)
        if budget is not None:
            budget["files"] += 1
            budget["bytes"] += len(raw)

    laptimes = json.loads(lt_path.read_text(encoding="utf-8"))

    # 2) 年检（全量检查：每一场都检）
    try:
        manifest["year_check"] = _verify_year_2026(laptimes, f"{race}/{session}")
    except RuntimeError as exc:
        manifest["year_check"] = {"rejected": str(exc)}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if state is not None:
            state["year_checks"]["rejected"].append(f"{race}/{session}: {exc}")
        return {"rejected": True, "race": race, "session": session}

    # 3) 车手 × 圈号
    per_driver = _drivers_and_laps(laptimes)
    wanted = ([d for d in drivers if d in per_driver] if drivers
              else sorted(per_driver))

    # 4) 逐圈逐点遥测（并发下载；已存在跳过）
    dirty: list[str] = []
    fetched = skipped = missing = 0
    stopped = False

    pending: list[tuple[str, int, Path]] = []
    for drv in wanted:
        laps = per_driver[drv]
        if max_laps is not None:
            laps = laps[:max_laps]
        for lap in laps:
            f = out_dir / drv / f"{lap}_tel.json"
            if f.exists():
                skipped += 1
                continue
            pending.append((drv, lap, f))
    if budget is not None and budget.get("max_files"):
        room = budget["max_files"] - budget["files"]
        if len(pending) >= room:
            stopped = True
            pending = pending[:max(0, room)]

    def _one(drv: str, lap: int, f: Path) -> tuple[str, int, Path, bytes | None]:
        return drv, lap, f, _get(f"{RAW}/{quoted}/{drv}/{lap}_tel.json")

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_one, d, n, f) for d, n, f in pending]
            for fut in as_completed(futures):
                drv, lap, f, raw = fut.result()
                if budget is not None and budget.get("deadline") \
                        and time.time() >= budget["deadline"]:
                    stopped = True
                if raw is None:
                    missing += 1
                    manifest["laps"].append(
                        {"driver": drv, "lap": lap, "status": "missing"}
                    )
                    manifest.setdefault("missing_laps", []).append(f"{drv}/{lap}")
                    continue
                try:
                    tel = json.loads(raw.decode("utf-8"))
                    bad = _validate_tel(tel)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    bad = ["<非 JSON>"]
                    tel = {}
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(raw)
                fetched += 1
                if budget is not None:
                    budget["files"] += 1
                    budget["bytes"] += len(raw)
                rec = {"driver": drv, "lap": lap,
                       "status": "ok" if not bad else "dirty",
                       "points": len((tel.get("tel") or {}).get("time", []))}
                if bad:
                    rec["missing_keys"] = bad
                    dirty.append(f"{drv}/{lap}")
                manifest["laps"].append(rec)
                if stopped and budget is not None and budget.get("max_files") \
                        and budget["files"] >= budget["max_files"]:
                    for rest in futures:
                        rest.cancel()
                    break

    manifest["counts"] = {"fetched": fetched, "skipped_existing": skipped,
                          "missing": missing, "dirty": len(dirty)}
    manifest["dirty_laps"] = dirty
    manifest["complete"] = not stopped
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if state is not None:
        state["sessions"][sub] = {
            "year_check": manifest["year_check"],
            "fetched": fetched, "skipped_existing": skipped,
            "missing": missing, "dirty": len(dirty), "complete": not stopped,
        }
        state["files_downloaded"] = state.get("files_downloaded", 0) + fetched
        state["year_checks"]["2026"] += 1
        state["dirty_laps"].extend(f"{sub}/{d}" for d in dirty)
    return manifest


# --------------------------------------------------------------------------- #
# 全量爬取
# --------------------------------------------------------------------------- #
def crawl_full(
    races: list[str],
    sessions: tuple[str, ...],
    state: dict[str, Any],
    max_files: int | None,
    max_minutes: float | None,
    drivers: list[str] | None,
    max_laps: int | None,
    force: bool = False,
    workers: int = 24,
) -> None:
    budget = {
        "max_files": max_files, "max_bytes": None,
        "deadline": (time.time() + max_minutes * 60) if max_minutes else None,
        "files": 0, "bytes": 0,
    }
    total = len(races) * len(sessions)
    i = 0
    for race in races:
        for session in sessions:
            i += 1
            sub = f"{race}/{session}"
            prev = state["sessions"].get(sub)
            probe = CACHE / race / session / "session_laptimes.json"
            if not force and prev and (prev.get("complete") or prev.get("absent")) \
                    and (probe.exists() or prev.get("absent")):
                continue
            print(f"[{i}/{total}] {sub} ...", flush=True)
            res = fetch_session(race, session, drivers, max_laps, CACHE,
                                state=state, budget=budget, workers=workers)
            if res.get("absent"):
                state["year_checks"]["absent"].append(sub)
                state["sessions"][sub] = {"absent": True}
                _save_state(state)
                continue
            c = res.get("counts", {})
            print(f"    -> fetched={c.get('fetched')} skipped={c.get('skipped_existing')} "
                  f"dirty={c.get('dirty')} complete={res.get('complete')} "
                  f"| 累计 {budget['files']} 文件 / {budget['bytes'] / 1e6:.1f} MB",
                  flush=True)
            _save_state(state)
            if budget["deadline"] and time.time() >= budget["deadline"]:
                print("达到时间预算，停止（重跑本命令可继续）。", flush=True)
                return
            if max_files and budget["files"] >= max_files:
                print("达到文件上限，停止（重跑本命令可继续）。", flush=True)
                return
    print("全量爬取完成。", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F1 2026 官方公开遥测抓取（仅 2026）")
    ap.add_argument("--race", default=None, help="单场模式：大奖赛名")
    ap.add_argument("--session", default="Race", choices=SESSIONS)
    ap.add_argument("--drivers", default="", help="逗号分隔车手代码；空=全部")
    ap.add_argument("--max-laps", type=int, default=None, help="每位车手最多几圈")
    ap.add_argument("--full", action="store_true", help="全量：24 站 × 全部会话")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="忽略完成状态重抓")
    ap.add_argument("--metadata-only", action="store_true",
                    help="只抓会话级文件并做 2026 全量年检（不抓逐点遥测）")
    ap.add_argument("--workers", type=int, default=8,
                    help="逐点遥测并发下载数（默认 8）")
    ap.add_argument("--status", action="store_true", help="只打印进度")
    ap.add_argument("--out", default=str(CACHE))
    args = ap.parse_args(argv)

    state = _state()
    if args.status:
        yc = state.get("year_checks", {})
        sess = state.get("sessions", {})
        done = sum(1 for s in sess.values() if s.get("complete") or s.get("absent"))
        print(f"会话完成 {done}/{len(CALENDAR_2026) * len(SESSIONS)}；"
              f"2026 年检通过 {yc.get('2026', 0)} 场；"
              f"拒绝 {len(yc.get('rejected', []))}；无数据 {len(yc.get('absent', []))}")
        print(f"已下载文件 {state.get('files_downloaded', 0)}；"
              f"脏圈 {len(state.get('dirty_laps', []))}")
        return 0

    drivers = [s.strip() for s in args.drivers.split(",") if s.strip()] or None
    max_laps = 0 if args.metadata_only else args.max_laps

    if args.full or args.race is None:
        crawl_full(list(CALENDAR_2026), SESSIONS, state,
                   args.max_files, args.max_minutes, drivers, max_laps, args.force,
                   workers=args.workers)
    else:
        res = fetch_session(args.race, args.session, drivers, args.max_laps,
                            Path(args.out), state=state, workers=args.workers)
        if res.get("absent"):
            print(f"{args.race}/{args.session}: 该会话不存在")
            return 0
        print(f"[2026 抓取] {args.race} / {args.session}")
        print(f"  年检: {res.get('year_check')}")
        print(f"  {res.get('counts')}")

    _save_state(state)
    yc = state["year_checks"]
    print(f"[状态] 2026 通过 {yc['2026']} 场；拒绝 {len(yc['rejected'])}；"
          f"无数据 {len(yc['absent'])}；累计文件 {state['files_downloaded']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
