"""从 OpenF1 采集 2026 赛季 Race 圈级数据（第二数据源）。

与仓库里的 FastF1 官方计时抓取互补：
- FastF1：官方计时逐点遥测（已由 CI 落成 data/f1_2026/train）
- OpenF1：圈级 sector / 速度陷阱 / 进站 / stint 轮胎信息

输出：data/f1_2026/openf1_raw/<track_id>.laps.json, .stints.json, .pit.json
      data/f1_2026/openf1_manifest.json
支持断点续采：已存在的文件跳过。
"""
from __future__ import annotations

import json
import pathlib
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://api.openf1.org/v1"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "f1_2026" / "openf1_raw"
OUT.mkdir(parents=True, exist_ok=True)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# OpenF1 的 location/country -> 仓库 track_id
TRACK_MAP = {
    ("Australia", "Melbourne"): "melbourne",
    ("China", "Shanghai"): "shanghai",
    ("Japan", "Suzuka"): "suzuka",
    ("Bahrain", "Sakhir"): "sakhir",
    ("Saudi Arabia", "Jeddah"): "jeddah",
    ("United States", "Miami Gardens"): "miami",
    ("Canada", "Montréal"): "montreal",
    ("Monaco", "Monte Carlo"): "monaco",
    ("Spain", "Barcelona"): "barcelona",
    ("Austria", "Spielberg"): "spielberg",
    ("United Kingdom", "Silverstone"): "silverstone",
    ("Belgium", "Spa-Francorchamps"): "spa",
    ("Hungary", "Budapest"): "hungaroring",
    ("Netherlands", "Zandvoort"): "zandvoort",
    ("Italy", "Monza"): "monza",
    ("Spain", "Madrid"): "madrid",
    ("Azerbaijan", "Baku"): "baku",
    ("Singapore", "Marina Bay"): "singapore",
    ("United States", "Austin"): "austin",
    ("Mexico", "Mexico City"): "mexico_city",
    ("Brazil", "São Paulo"): "sao_paulo",
    ("United States", "Las Vegas"): "las_vegas",
    ("Qatar", "Lusail"): "lusail",
    ("United Arab Emirates", "Yas Marina"): "yas_marina",
    ("Bahrain", "Kuala Lumpur"): "kuala_lumpur",  # 已知脏数据，下游丢弃
}


def get(path: str, retries: int = 4) -> list:
    url = f"{BASE}/{path}"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, context=CTX, timeout=180) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2 + attempt * 4)
    raise RuntimeError(f"{url} 连续失败: {last}")


def fetch_session(track: str, session_key: int) -> dict:
    result = {"track_id": track, "session_key": session_key}
    for name in ("laps", "stints", "pit"):
        path = OUT / f"{track}.{name}.json"
        if path.exists() and path.stat().st_size > 2:
            result[name] = len(json.loads(path.read_text(encoding="utf-8")))
            continue
        rows = get(f"{name}?session_key={session_key}")
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        result[name] = len(rows)
        time.sleep(0.4)
    return result


def main() -> int:
    sessions = get("sessions?year=2026")
    races = []
    for s in sessions:
        if s.get("session_name") != "Race":
            continue
        key = (s.get("country_name"), s.get("location"))
        track = TRACK_MAP.get(key)
        if not track:
            print(f"[skip] 未映射场次 {key}", flush=True)
            continue
        races.append((track, int(s["session_key"])))

    seen = {}
    for track, sk in races:
        seen.setdefault(track, sk)

    print(f"发现 {len(races)} 场 Race，去重后 {len(seen)} 站", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fetch_session, track, sk): track for track, sk in seen.items()
        }
        for fut in as_completed(futures):
            track = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"[ok] {track}: laps={r['laps']} stints={r['stints']} pit={r['pit']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[fail] {track}: {exc}", flush=True)

    manifest = {
        "source": "openf1.org",
        "season": 2026,
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tracks": sorted(results, key=lambda x: x["track_id"]),
        "totals": {
            "tracks": len(results),
            "laps": sum(r["laps"] for r in results),
            "stints": sum(r["stints"] for r in results),
            "pit": sum(r["pit"] for r in results),
        },
    }
    (OUT.parent / "openf1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["totals"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
