"""从 F1Laps 公开调教库抓取「真实调教 → 真实圈速」配对样本。

数据源
------
``https://www.f1laps.com/f1-26/setups/<track>/`` —— 社区上传的 F1 26 游戏调教，
每套调教带 20 项数值、圈速、车队、会话类型、干湿、日期与全部辅助设置。

合规依据（robots.txt 原文）
--------------------------
::

    User-agent: *
    Allow: /
    Content-Signal: search=yes, ai-input=yes, ai-train=yes
    Disallow: /api/          # 私有接口，禁止
    Disallow: /accounts/     # 私有区域，禁止

本脚本**只走公开 HTML 页面**，不触碰 ``/api/`` 与任何登录区；默认 1.2 秒/请求
的限速，并带明确 UA。抓取结果仅用于本地训练。

为什么需要它
------------
``train_telemetry_surrogate.py`` 的圈级头（``lap_head``）需要
「这套调教跑出这个圈速」的真实配对；2026 官方遥测只有速度/油门/刹车，
**不含调教**，所以 ``lap_head.available`` 长期为 false
（原因：``本地有效带 setup 的圈数不足（0）``）。

产物
----
``data/setup_laps/f1laps_setups.jsonl``，每行::

    {
      "track_id": "melbourne",          # 对齐 setup_tuner.domain.track
      "setup": {"front_wing": 23.0, ...},  # 20 项，单位与项目一致
      "lap_time_ms": 76790,
      "team": "Racing Bulls",
      "session": "Time Trial",
      "conditions": "dry",
      "assists": {"traction_control": "Full", "abs": "On", ...},
      "recorded_date": "2026-10-31",
      "source_url": "https://www.f1laps.com/f1-26/setups/australia/<uuid>/",
      "fetched_at": "2026-11-04T10:20:00",
    }

用法
----
::

    python scripts/fetch_setup_laps.py --check          # 只抓 2 页自检解析
    python scripts/fetch_setup_laps.py --dry            # 全部 24 站干地
    python scripts/fetch_setup_laps.py --dry --wet      # 干 + 湿
    python scripts/fetch_setup_laps.py --max-minutes 20 # 带预算，可续传
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "setup_laps"
OUT_JSONL = OUT_DIR / "f1laps_setups.jsonl"
STATE_FILE = OUT_DIR / "_fetch_state.json"

BASE = "https://www.f1laps.com/f1-26/setups"
UA = "F1OPT-research/0.1 (+https://github.com/JXTTNN/F1-)"
TIMEOUT = 45
RETRIES = 3
DEFAULT_SLEEP = 1.2

#: F1Laps 赛道 slug → 项目 track_id（与 setup_tuner.domain.track 的 24 站一致）
#: 其中 jeddah / las_vegas / lusail / sao_paulo / yas_marina 无 2026 官方遥测，
#: 这五个站的调教-圈速配对是本数据源不可替代的价值。
TRACK_SLUGS: dict[str, str] = {
    "australia": "melbourne",
    "china": "shanghai",
    "japan": "suzuka",
    "bahrain": "sakhir",
    "saudi_arabia": "jeddah",
    "miami": "miami",
    "canada": "montreal",
    "monaco": "monaco",
    "spain": "barcelona",
    "austria": "spielberg",
    "silverstone": "silverstone",
    "spa": "spa",
    "hungary": "hungaroring",
    "netherlands": "zandvoort",
    "monza": "monza",
    "madrid": "madrid",
    "azerbaijan": "baku",
    "singapore": "singapore",
    "usa": "austin",
    "mexico": "mexico_city",
    "brazil": "sao_paulo",
    "las_vegas": "las_vegas",
    "qatar": "lusail",
    "abudhabi": "yas_marina",
}

#: 页面标签 → 项目字段名（20 项，逐条对齐 setup_tuner.domain.setup）
#: 注意 F1Laps 把 Brake 写作 "Break"（站点历史拼写），两种都收。
LABEL_TO_FIELD: dict[str, str] = {
    "front wing": "front_wing",
    "rear wing": "rear_wing",
    "differential adjustment on throttle": "on_throttle_diff",
    "differential adjustment off throttle": "off_throttle_diff",
    "front camber": "front_camber",
    "rear camber": "rear_camber",
    "front toe": "front_toe",
    "rear toe": "rear_toe",
    "front suspension": "front_suspension",
    "rear suspension": "rear_suspension",
    "front anti-roll bar": "front_anti_roll_bar",
    "rear anti-roll bar": "rear_anti_roll_bar",
    "front ride height": "front_ride_height",
    "rear ride height": "rear_ride_height",
    "break pressure": "brake_pressure",
    "brake pressure": "brake_pressure",
    "front break bias": "brake_bias",
    "front brake bias": "brake_bias",
    "front left tyre pressure": "front_left_tyre_pressure",
    "front right tyre pressure": "front_right_tyre_pressure",
    "rear left tyre pressure": "rear_left_tyre_pressure",
    "rear right tyre pressure": "rear_right_tyre_pressure",
}

#: 元数据标签（不属于调教 20 项）
META_LABELS = {
    "team", "session", "lap time", "conditions", "date",
    "traction control", "anti-lock brakes", "steering assist",
    "braking assist", "gearbox", "racing line",
}

DETAIL_ID_RE = re.compile(
    r"/f1-26/setups/([a-z_]+)/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})/",
)


# --------------------------------------------------------------------------- #
# HTTP
def _get(url: str) -> str | None:
    """带重试的 GET；404 返回 None。"""
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except Exception as exc:  # noqa: BLE001 — 网络层杂类异常统一重试
            last = exc
        if attempt < RETRIES:
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"GET failed: {url} ({last})")


def _clean(text: str) -> str:
    """去标签、反转义、压缩空白。"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# 解析
def parse_pairs(page: str) -> dict[str, str]:
    """把页面里的 ``<dt>label</dt> ... <dd>value</dd>`` 抽成 label→value。

    每个参数块有 2 个 ``<dd>``：第一个是进度条（纯标签、无文本），第二个是数值。
    只收「有纯文本」的那个，因此进度条会被自然跳过。
    """
    tokens = re.findall(r"<dt[^>]*>(.*?)</dt>|<dd[^>]*>(.*?)</dd>", page, re.S)
    pairs: dict[str, str] = {}
    pending: str | None = None
    for dt, dd in tokens:
        if dt:
            pending = _clean(dt)
        elif dd and pending:
            value = _clean(dd)
            if value:
                pairs[pending] = value
                pending = None
    return pairs


def parse_lap_time(text: str) -> int | None:
    """``1:16.790`` / ``76.790`` → 毫秒；无法解析返回 None。"""
    text = text.strip()
    m = re.fullmatch(r"(\d+):(\d{2})\.(\d{1,3})", text)
    if m:
        minutes, seconds, frac = m.groups()
        return int(minutes) * 60000 + int(seconds) * 1000 + int(frac.ljust(3, "0"))
    m = re.fullmatch(r"(\d+)\.(\d{1,3})", text)
    if m:
        seconds, frac = m.groups()
        return int(seconds) * 1000 + int(frac.ljust(3, "0"))
    return None


def parse_number(text: str) -> float | None:
    """``-3.50˚`` / ``100%`` / ``29.5`` → float；无法解析返回 None。"""
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def parse_date(text: str) -> str | None:
    """``2026年10月31日`` / ``2026-10-31`` / ``31 Oct 2026`` → ISO 日期。"""
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return m.group(0)
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", text)
    if m:
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        key = m.group(2)[:3].lower()
        if key in months:
            return f"{m.group(3)}-{months[key]:02d}-{int(m.group(1)):02d}"
    return None


def parse_detail(page: str, track_id: str, url: str) -> dict[str, Any] | None:
    """解析单个调教详情页；缺圈速或缺关键字段返回 None。"""
    pairs = parse_pairs(page)
    if not pairs:
        return None

    setup: dict[str, float] = {}
    for label, value in pairs.items():
        field = LABEL_TO_FIELD.get(label.lower())
        if not field:
            continue
        num = parse_number(value)
        if num is not None:
            setup[field] = num

    lap_raw = pairs.get("Lap time") or pairs.get("Lap Time") or ""
    lap_ms = parse_lap_time(lap_raw)
    # 没有圈速的记录对训练无用（无法构成配对）
    if lap_ms is None or len(setup) < 18:
        return None

    assists = {
        "traction_control": pairs.get("Traction Control"),
        "abs": pairs.get("Anti-Lock Brakes"),
        "steering_assist": pairs.get("Steering Assist"),
        "braking_assist": pairs.get("Braking Assist"),
        "gearbox": pairs.get("Gearbox"),
        "racing_line": pairs.get("Racing Line"),
    }
    conditions = (pairs.get("Conditions") or "").lower()

    # 作者名：同一位作者在同一赛道上的多套调教才具备可比性（消除车手差异），
    # 这是把「圈速差异归因到调教」的唯一可用分组键。
    author: str | None = None
    m = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
    if m:
        m2 = re.search(r"\bby\s+(\S.*)$", _clean(m.group(1)), re.I)
        if m2:
            author = m2.group(1).strip()

    return {
        "track_id": track_id,
        "author": author,
        "setup": {k: float(v) for k, v in setup.items()},
        "setup_field_count": len(setup),
        "lap_time_ms": lap_ms,
        "team": pairs.get("Team"),
        "session": pairs.get("Session"),
        "conditions": "wet" if "wet" in conditions else "dry",
        "assists": {k: v for k, v in assists.items() if v},
        "recorded_date": parse_date(pairs.get("Date") or ""),
        "source_url": url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def list_setup_ids(slug: str, wet: bool = False) -> list[str]:
    """列表页上的详情 UUID（每页 20 条，页面无分页参数）。"""
    url = f"{BASE}/{slug}/wet/" if wet else f"{BASE}/{slug}/"
    page = _get(url)
    if not page:
        return []
    ids = [m[1] for m in DETAIL_ID_RE.findall(page) if m[0] == slug]
    return list(dict.fromkeys(ids))


# --------------------------------------------------------------------------- #
# 主流程
def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"done": [], "failed": [], "updated_at": None}


def _save_state(state: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def run(
    tracks: list[str],
    wet: bool,
    sleep_s: float,
    max_minutes: float | None,
    check_only: bool,
) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    done: set[str] = set(state.get("done", []))
    # 以「已落盘数据」为准去重：状态文件可能在换机器 / 换 CI 缓存后丢失，
    # 只靠它会让同一套调教被重复写入（同一 URL 两行），把数据集污染成
    # 重复样本加权。数据文件本身才是唯一可信的去重依据。
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                url = json.loads(line).get("source_url")
            except json.JSONDecodeError:
                continue
            if url:
                done.add(url)
    deadline = time.time() + max_minutes * 60 if max_minutes else None

    written = 0
    skipped = 0
    total_seen = 0
    mode = "a" if OUT_JSONL.exists() else "w"

    with open(OUT_JSONL, mode, encoding="utf-8", newline="\n") as fout:
        for slug in tracks:
            track_id = TRACK_SLUGS[slug]
            ids = list_setup_ids(slug, wet=wet)
            print(f"[{track_id}] 列表页 {len(ids)} 条（{'wet' if wet else 'dry'}）",
                  flush=True)
            time.sleep(sleep_s)
            for uid in ids:
                if deadline and time.time() > deadline:
                    print("[budget] 到达时间预算，停止（可重跑续传）", flush=True)
                    _save_state(state)
                    return 0
                total_seen += 1
                url = f"{BASE}/{slug}/{'wet/' if wet else ''}{uid}/"
                if url in done:
                    skipped += 1
                    continue
                page = _get(url)
                time.sleep(sleep_s)
                if not page:
                    state.setdefault("failed", []).append(url)
                    continue
                rec = parse_detail(page, track_id, url)
                if rec is None:
                    state.setdefault("failed", []).append(url)
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                done.add(url)
                state["done"] = sorted(done)
                if check_only and written >= 2:
                    _save_state(state)
                    print(f"[check] 已解析 {written} 条，字段数 "
                          f"{[r['setup_field_count'] for r in [rec]]}", flush=True)
                    print(json.dumps(rec, ensure_ascii=False, indent=2)[:1200])
                    return 0

    _save_state(state)
    print(f"[done] 新写入 {written} 条；跳过已完成 {skipped}；"
          f"处理 {total_seen} 条；失败 {len(state.get('failed', []))}")
    return 0


def summarize() -> int:
    """打印现有数据集概况（供 CI Step Summary 用）。"""
    if not OUT_JSONL.exists():
        print("[summary] 尚无数据文件")
        return 1
    rows = [json.loads(line) for line in OUT_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("[summary] 空数据集")
        return 1
    by_track: dict[str, int] = {}
    by_session: dict[str, int] = {}
    for r in rows:
        by_track[r["track_id"]] = by_track.get(r["track_id"], 0) + 1
        by_session[r.get("session") or "?"] = by_session.get(r.get("session") or "?", 0) + 1
    fields = [r["setup_field_count"] for r in rows]
    print(f"[summary] 总样本 {len(rows)}；覆盖赛道 {len(by_track)}；"
          f"字段数 min/median/max = {min(fields)}/{sorted(fields)[len(fields)//2]}/{max(fields)}")
    print(f"[summary] 会话分布: {by_session}")
    print(f"[summary] 每站: {json.dumps(by_track, ensure_ascii=False, sort_keys=True)}")
    missing = [t for t in TRACK_SLUGS.values() if t not in by_track]
    print(f"[summary] 缺数据赛道: {missing or '无'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抓取 F1Laps 公开调教-圈速配对")
    ap.add_argument("--tracks", default="", help="逗号分隔 slug；空=全部 24 站")
    ap.add_argument("--dry", action="store_true", help="抓干地（默认）")
    ap.add_argument("--wet", action="store_true", help="抓湿地")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="请求间隔秒")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--check", action="store_true", help="只抓 2 条自检解析")
    ap.add_argument("--summary", action="store_true", help="打印数据集概况")
    args = ap.parse_args(argv)

    if args.summary:
        return summarize()

    tracks = ([s.strip() for s in args.tracks.split(",") if s.strip()]
              if args.tracks else list(TRACK_SLUGS))
    unknown = [t for t in tracks if t not in TRACK_SLUGS]
    if unknown:
        print(f"[error] 未知赛道 slug: {unknown}", file=sys.stderr)
        return 2

    if args.check:
        tracks = tracks[:1]

    rc = run(tracks, wet=False, sleep_s=args.sleep,
             max_minutes=args.max_minutes, check_only=args.check)
    if rc == 0 and not args.check and args.wet:
        rc = run(tracks, wet=True, sleep_s=args.sleep,
                 max_minutes=args.max_minutes, check_only=False)
    if rc == 0 and not args.check:
        summarize()
    return rc


if __name__ == "__main__":
    sys.exit(main())
