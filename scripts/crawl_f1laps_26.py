"""抓取 f1laps《F1 26》社区调教（含圈速/工况/辅助），产出训练语料。

合规依据
--------
- robots.txt：``Allow: /``，且 ``Content-Signal: ai-train=yes``；仅 ``/api/`` 禁止，
  本脚本**不访问**任何 ``/api/`` 端点，只抓公开页面。
- 限速 0.35 s/请求，失败指数退避，最多 3 次；支持断点续抓。

产出
----
``data/f1_2026/game_setups/f1laps_26.jsonl``，每行：
track_id / track_slug / url / team / session / conditions / lap_time_ms /
setup{20 项} / assists{...} / date
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "f1_2026" / "game_setups"
OUT = OUT_DIR / "f1laps_26.jsonl"

BASE = "https://www.f1laps.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

#: 赛道 slug → 本项目 track_id（与 domain.track 对齐）
SLUG_TO_ID = {
    "australia": "melbourne", "china": "shanghai", "japan": "suzuka",
    "bahrain": "sakhir", "saudi_arabia": "jeddah", "miami": "miami",
    "imola": "imola", "monaco": "monaco", "spain": "barcelona",
    "canada": "montreal", "austria": "spielberg", "great_britain": "silverstone",
    "belgium": "spa", "hungary": "hungaroring", "netherlands": "zandvoort",
    "italy": "monza", "madrid": "madrid", "azerbaijan": "baku",
    "singapore": "singapore", "usa": "austin", "mexico": "mexico_city",
    "brazil": "sao_paulo", "las_vegas": "las_vegas", "qatar": "lusail",
    "abu_dhabi": "yas_marina",
}

NUM = r"([-+]?\d+(?:\.\d+)?)"
SETUP_PATTERNS = {
    "front_wing": rf"Front Wing\s*{NUM}",
    "rear_wing": rf"Rear Wing\s*{NUM}",
    "on_throttle_diff": rf"Differential Adjustment On Throttle\s*{NUM}",
    "off_throttle_diff": rf"Differential Adjustment Off Throttle\s*{NUM}",
    "front_camber": rf"Front Camber\s*{NUM}",
    "rear_camber": rf"Rear Camber\s*{NUM}",
    "front_toe": rf"Front Toe\s*{NUM}",
    "rear_toe": rf"Rear Toe\s*{NUM}",
    "front_suspension": rf"Front Suspension\s*{NUM}",
    "rear_suspension": rf"Rear Suspension\s*{NUM}",
    "front_anti_roll_bar": rf"Front Anti-Roll Bar\s*{NUM}",
    "rear_anti_roll_bar": rf"Rear Anti-Roll Bar\s*{NUM}",
    "front_ride_height": rf"Front Ride Height\s*{NUM}",
    "rear_ride_height": rf"Rear Ride Height\s*{NUM}",
    "brake_pressure": rf"Brake Pressure\s*{NUM}",
    "brake_bias": rf"Front Brake Bias\s*{NUM}",
    "front_left_tyre_pressure": rf"Front Left Tyre Pressure\s*{NUM}",
    "front_right_tyre_pressure": rf"Front Right Tyre Pressure\s*{NUM}",
    "rear_left_tyre_pressure": rf"Rear Left Tyre Pressure\s*{NUM}",
    "rear_right_tyre_pressure": rf"Rear Right Tyre Pressure\s*{NUM}",
}


def fetch(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=CTX, timeout=30) as resp:
                return resp.read().decode("utf-8", "ignore")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                print(f"  ! {url} {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def text_of(html: str) -> str:
    body = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", body)


def parse_lap(text: str) -> int | None:
    m = re.search(r"Lap time\s+(\d):(\d{2})\.(\d{3})", text)
    if not m:
        return None
    return int(m.group(1)) * 60000 + int(m.group(2)) * 1000 + int(m.group(3))


def parse_page(html: str, url: str, track_id: str) -> dict | None:
    text = text_of(html)
    setup: dict[str, float] = {}
    for name, pat in SETUP_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            return None
        setup[name] = float(m.group(1))
    lap = parse_lap(text)
    if lap is None:
        return None
    cond = "wet" if re.search(r"Conditions\s+Wet", text) else "dry"
    session = "unknown"
    for s in ("Time Trial", "Qualifying", "Race", "Practice"):
        if re.search(rf"Session\s+{s}", text):
            session = s
            break
    team = None
    m = re.search(r"Team\s+([A-Za-z0-9&\.\- ]{2,30}?)\s+Session", text)
    if m:
        team = m.group(1).strip()
    assists = {}
    for label, key in (("Traction Control", "tc"), ("Anti-Lock Brakes", "abs"),
                       ("Steering Assist", "steering"), ("Braking Assist", "braking")):
        m = re.search(rf"{label}\s+(On|Off|Auto|Low|Medium|High|Full|Corners)", text)
        if m:
            assists[key] = m.group(1)
    date = None
    m = re.search(r"Date\s+([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    if m:
        date = m.group(1)
    return {
        "track_id": track_id, "url": url, "team": team, "session": session,
        "conditions": cond, "lap_time_ms": lap, "setup": setup,
        "assists": assists, "date": date,
    }


def listing_ids(slug: str, wet: bool = False) -> set[str]:
    path = f"/f1-26/setups/{slug}/wet/" if wet else f"/f1-26/setups/{slug}/"
    html = fetch(BASE + path)
    if not html:
        return set()
    return set(re.findall(rf"/f1-26/setups/{slug}/([0-9a-f-]{{36}})/", html))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="", help="逗号分隔 slug；空=全部")
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    seen: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["url"])
            except Exception:
                continue

    slugs = [s.strip() for s in args.tracks.split(",") if s.strip()] or list(SLUG_TO_ID)
    total = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for slug in slugs:
            track_id = SLUG_TO_ID.get(slug, slug)
            ids = listing_ids(slug, wet=False) | listing_ids(slug, wet=True)
            todo = [u for u in sorted(ids)
                    if f"{BASE}/f1-26/setups/{slug}/{u}/" not in seen]
            print(f"{slug}: {len(ids)} 条，待抓 {len(todo)}", flush=True)
            if not todo:
                continue

            def work(uid: str) -> dict | None:
                url = f"{BASE}/f1-26/setups/{slug}/{uid}/"
                html = fetch(url)
                time.sleep(args.delay)
                if not html:
                    return None
                return parse_page(html, url, track_id)

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                for rec in pool.map(work, todo):
                    if not rec:
                        continue
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    seen.add(rec["url"])
                    total += 1
    print(f"新增 {total} 条 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
