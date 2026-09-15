#!/usr/bin/env python3
"""批量下载24条赛道的原始SVG到本地缓存目录。"""
import urllib.request
import time
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/julesr0y/f1-circuits-svg/main/"
    "circuits/minimal/white-outline/"
)

TRACK_LAYOUT_MAP = {
    "melbourne": "melbourne-2",
    "shanghai": "shanghai-1",
    "suzuka": "suzuka-2",
    "sakhir": "bahrain-1",
    "jeddah": "jeddah-1",
    "miami": "miami-1",
    "monaco": "monaco-6",
    "montreal": "montreal-6",
    "barcelona": "catalunya-6",
    "silverstone": "silverstone-8",
    "spa": "spa-francorchamps-4",
    "hungaroring": "hungaroring-3",
    "zandvoort": "zandvoort-5",
    "monza": "monza-7",
    "austin": "austin-1",
    "mexico_city": "mexico-city-3",
    "sao_paulo": "interlagos-2",
    "las_vegas": "las-vegas-1",
    "baku": "baku-1",
    "singapore": "marina-bay-4",
    "madrid": "madring-1",
    "spielberg": "spielberg-3",
    "lusail": "lusail-1",
    "yas_marina": "yas-marina-2",
}

CACHE_DIR = Path(__file__).resolve().parent / "raw_svgs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

success = 0
failed = []

for track_id, layout_id in TRACK_LAYOUT_MAP.items():
    cache_path = CACHE_DIR / f"{layout_id}.svg"
    if cache_path.exists():
        print(f"  {track_id:15s} ({layout_id}) -- 已缓存")
        success += 1
        continue

    url = BASE_URL + layout_id + ".svg"
    print(f"  {track_id:15s} ({layout_id}) -- 下载中...", end=" ")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
        cache_path.write_text(data, encoding="utf-8")
        print(f"OK ({len(data)} bytes)")
        success += 1
    except Exception as e:
        print(f"FAILED: {e}")
        failed.append((track_id, str(e)))
    time.sleep(0.5)  # 避免请求过快

print(f"\n{'=' * 60}")
print(f"完成: {success}/{len(TRACK_LAYOUT_MAP)} 条赛道")
if failed:
    print(f"失败: {len(failed)} 条")
    for tid, err in failed:
        print(f"  {tid}: {err}")
else:
    print("全部成功！")