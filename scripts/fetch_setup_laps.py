"""从 F1Laps 公开调教库抓取「真实调教 → 真实圈速」配对样本。

数据源
------
``https://www.f1laps.com/f1-26/setups/<track>/``（干地）
``https://www.f1laps.com/f1-26/setups/<track>/wet/``（湿地）

同一套模板在 F1 25 / F1 24 的版面同样可用，因此每个赛道最多有 **6 个切面**：

===========  ==============  ============================================
游戏年版      干/湿            用途
===========  ==============  ============================================
F1 26         dry / wet       主口径（与项目 2026 赛季对齐）
F1 25         dry / wet       辅助（同一套 20 项参数区间，跨年可比）
F1 24         dry / wet       辅助（同上）
===========  ==============  ============================================

2026 版面每站只展示 20 条且无分页参数（``?page`` / ``?offset`` 均无效，
已验证），所以「多切面」是唯一能成倍扩样的合规方式：6 × 24 × 20 ≈ 2,880 条。

合规依据（robots.txt 原文）
--------------------------
::

    User-agent: *
    Allow: /
    Content-Signal: search=yes, ai-input=yes, ai-train=yes
    Disallow: /api/          # 私有接口，禁止
    Disallow: /accounts/     # 私有区域，禁止

本脚本**只走公开 HTML 页面**，不触碰 ``/api/`` 与任何登录区；默认 1.2 秒/请求
的限速，并带明确 UA。

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
      "surface": "dry",                 # 抓取切面（干/湿）
      "game_year": 2026,                # 抓取切面对应的游戏年版
      "assists": {"traction_control": "Full", "abs": "On", ...},
      "recorded_date": "2026-10-31",
      "source_url": "https://www.f1laps.com/f1-26/setups/australia/<uuid>/",
      "fetched_at": "2026-11-04T10:20:00",
    }

用法
----
::

    python scripts/fetch_setup_laps.py --check          # 只抓 2 页自检解析
    python scripts/fetch_setup_laps.py --dry            # 24 站干地（F1 26）
    python scripts/fetch_setup_laps.py --all-slices     # 6 切面全量（默认 2026）
    python scripts/fetch_setup_laps.py --games 2026,2025,2024 --dry --wet
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "setup_laps"
OUT_JSONL = OUT_DIR / "f1laps_setups.jsonl"
STATE_FILE = OUT_DIR / "_fetch_state.json"

#: 游戏年版 → URL 前缀里的年份代号（F1 2026 版面 = ``/f1-26/``）
GAME_SLUGS: dict[int, str] = {2026: "f1-26", 2025: "f1-25", 2024: "f1-24"}
GAME_YEAR_BY_SLUG: dict[str, int] = {v: k for k, v in GAME_SLUGS.items()}
DEFAULT_GAMES: tuple[int, ...] = (2026,)
ALL_GAMES: tuple[int, ...] = (2026, 2025, 2024)

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


def base_url(game_year: int) -> str:
    """游戏年版 → 调教列表根路径。"""
    return f"https://www.f1laps.com/{GAME_SLUGS[game_year]}/setups"


def detail_id_re(game_year: int) -> re.Pattern[str]:
    """该年版详情页 URL 里的 UUID 正则。"""
    slug = GAME_SLUGS[game_year]
    return re.compile(
        rf"/{slug}/setups/([a-z_]+)/([0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-"
        r"[0-9a-f]{4}-[0-9a-f]{12})/",
    )


# 兼容旧调用（外部脚本可能 import 过）
DETAIL_ID_RE = detail_id_re(2026)


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


def parse_detail(
    page: str, track_id: str, url: str, game_year: int, surface: str,
) -> dict[str, Any] | None:
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
    # 湿地面板只是一个**筛选视图**，详情页 URL 不含 `wet/`（实测 `/wet/<uuid>/`
    # 一律 404）。干湿以页面上的 Conditions 字段为准，缺失时才回退到抓取切面。
    if "wet" in conditions:
        surface_final = "wet"
    elif conditions:
        surface_final = "dry"
    else:
        surface_final = surface

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
        "conditions": surface_final,
        "surface": surface_final,
        "game_year": game_year,
        "assists": {k: v for k, v in assists.items() if v},
        "recorded_date": parse_date(pairs.get("Date") or ""),
        "source_url": url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def list_setup_ids(slug: str, game_year: int = 2026, wet: bool = False) -> list[str]:
    """列表页上的详情 UUID（每页 20 条，页面无分页参数）。"""
    base = base_url(game_year)
    url = f"{base}/{slug}/wet/" if wet else f"{base}/{slug}/"
    page = _get(url)
    if not page:
        return []
    pat = detail_id_re(game_year)
    ids = [m[1] for m in pat.findall(page) if m[0] == slug]
    return list(dict.fromkeys(ids))


# --------------------------------------------------------------------------- #
# 主流程
def _paths(out: Path | None = None) -> tuple[Path, Path]:
    """返回 (数据文件, 状态文件)；`--out` 用于并行分片抓取。"""
    if out is None:
        return OUT_JSONL, STATE_FILE
    return out, out.with_name(out.stem + "_state.json")


def _load_state(state_file: Path) -> dict[str, Any]:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"done": [], "failed": [], "updated_at": None}


def _save_state(state: dict[str, Any], state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _done_urls(data_file: Path, extra: Sequence[Path] = ()) -> set[str]:
    """以「已落盘数据」为准去重。

    状态文件可能在换机器 / 换 CI 缓存后丢失，只靠它会让同一套调教被重复写入
    （同一 URL 两行），把数据集污染成重复样本加权。数据文件本身才是唯一可信
    的去重依据。

    ``extra`` 是并行分片文件：合并前也要互相去重，否则同一套调教会在
    两片里各出现一次。
    """
    done: set[str] = set()
    for path in (data_file, *extra):
        done |= _urls_in(path)
    return done


def _urls_in(data_file: Path) -> set[str]:
    done: set[str] = set()
    if not data_file.exists():
        return done
    for line in data_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            url = json.loads(line).get("source_url")
        except json.JSONDecodeError:
            continue
        if url:
            done.add(url)
    return done


_YEAR_IN_URL = re.compile(r"/(f1-\d{2})/setups/")


def normalize_meta(rec: dict[str, Any]) -> dict[str, Any]:
    """补齐 ``game_year`` / ``surface``，返回同一条记录（原地修改）。

    旧版抓取脚本只写 ``conditions``，不写 ``game_year``/``surface``。这类行
    会自成一个 ``None`` 年份组：去混淆时被单独去均值（K 折指标虚高），跨年
    检验也取不到「2026 干」样本而**整项静默不跑**。年份能从 ``source_url``
    的 ``/f1-NN/`` 反推、干湿能从 ``conditions`` 取，所以不必重新抓取。
    """
    m = _YEAR_IN_URL.search(str(rec.get("source_url") or ""))
    if not rec.get("game_year") and m:
        rec["game_year"] = GAME_YEAR_BY_SLUG.get(m.group(1))
    cond = str(rec.get("conditions") or "").strip().lower()
    if not rec.get("surface") and cond in ("dry", "wet"):
        rec["surface"] = cond
    return rec


def _filled(rec: dict[str, Any]) -> int:
    """非空字段数，用来在重复行里挑「更完整」的那条。"""
    return sum(1 for v in rec.values() if v not in (None, "", [], {}))


def normalize_corpus(path: Path) -> dict[str, int]:
    """对语料做一次自愈式重写：补元数据 + 按 URL 去重。

    抓取是**追加**写；一旦有旧行、缓存回灌或重复写入混进文件，只靠训练期
    归一化治不了产物本身。这里把文件读全、逐行补字段、按 ``source_url``
    去重（保留字段更全的那条），再原样顺序写回，生成稳定的最小 diff。

    Returns:
        统计字典：``total`` / ``repaired`` / ``duplicates`` / ``broken``。
    """
    if not path.exists():
        return {"total": 0, "repaired": 0, "duplicates": 0, "broken": 0}

    rows: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    repaired = duplicates = broken = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            broken += 1
            continue
        lacked = not rec.get("game_year") or not rec.get("surface")
        rec = normalize_meta(rec)
        if lacked and rec.get("game_year") and rec.get("surface"):
            repaired += 1
        url = str(rec.get("source_url") or "")
        if url and url in index:
            duplicates += 1
            i = index[url]
            rows[i] = rows[i] if _filled(rows[i]) >= _filled(rec) else rec
            continue
        if url:
            index[url] = len(rows)
        rows.append(rec)

    if repaired or duplicates or broken:
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
            newline="\n",
        )
    return {
        "total": len(rows),
        "repaired": repaired,
        "duplicates": duplicates,
        "broken": broken,
    }


def verify_corpus(path: Path) -> list[str]:
    """返回语料的问题列表；空列表表示通过。

    这是给 CI 用的**产物侧门禁**：抓取/合并一旦写出缺字段或重复的行，
    立刻在流水线里报错，而不是等训练指标莫名其妙掉下来才回头查。
    """
    if not path.exists():
        return [f"语料文件不存在：{path}"]

    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                problems.append("存在无法解析的 JSON 行")
    if not rows:
        return problems or ["语料为空"]

    no_year = [r.get("source_url") for r in rows if not r.get("game_year")]
    no_surface = [r.get("source_url") for r in rows if not r.get("surface")]
    if no_year:
        problems.append(f"{len(no_year)} 行缺 game_year，例：{no_year[0]}")
    if no_surface:
        problems.append(f"{len(no_surface)} 行缺 surface，例：{no_surface[0]}")

    known_years = set(GAME_SLUGS)
    bad_year = {r["game_year"] for r in rows} - known_years if not no_year else set()
    if bad_year:
        problems.append(f"出现未知年份：{sorted(bad_year)}")

    urls = [r.get("source_url") for r in rows if r.get("source_url")]
    if len(urls) != len(set(urls)):
        problems.append(f"存在重复 source_url：{len(urls) - len(set(urls))} 条")

    return problems


def run(
    tracks: list[str],
    games: tuple[int, ...],
    surfaces: tuple[str, ...],
    sleep_s: float,
    max_minutes: float | None,
    check_only: bool,
    out: Path | None = None,
) -> int:
    data_file, state_file = _paths(out)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_file)
    done: set[str] = _done_urls(OUT_JSONL if out is not None else data_file)
    state["done"] = sorted(done)
    deadline = time.time() + max_minutes * 60 if max_minutes else None

    written = 0
    skipped = 0
    total_seen = 0
    mode = "a" if data_file.exists() else "w"

    def finish() -> None:
        """收尾：落盘状态 + 自愈语料。

        任何退出路径都要走这里 —— 尤其是「到达时间预算」，那是云端 CI 的
        常态路径（55 分钟必然用尽）。早先只在正常结束时自愈，预算一停就
        把历史缺陷行原样留在文件里。
        """
        _save_state(state, state_file)
        print(f"[done] 新写入 {written} 条；跳过已完成 {skipped}；"
              f"处理 {total_seen} 条；失败 {len(state.get('failed', []))}")
        # 追加写会让历史缺陷行一直留在文件里（缺元数据、重复 URL）。收尾做一次
        # 自愈，保证产物**本身**满足不变量，而不是只在训练时被临时修正。
        stats = normalize_corpus(data_file)
        print(f"[normalize] 共 {stats['total']} 条；补元数据 {stats['repaired']} 条；"
              f"去重 {stats['duplicates']} 条；坏行 {stats['broken']} 条")

    with open(data_file, mode, encoding="utf-8", newline="\n") as fout:
        for game_year in games:
            for surface in surfaces:
                for slug in tracks:
                    track_id = TRACK_SLUGS[slug]
                    wet = surface == "wet"
                    ids = list_setup_ids(slug, game_year, wet=wet)
                    print(
                        f"[{track_id}] {GAME_SLUGS[game_year]} {surface} "
                        f"列表页 {len(ids)} 条",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    for uid in ids:
                        if deadline and time.time() > deadline:
                            print("[budget] 到达时间预算，停止（可重跑续传）", flush=True)
                            finish()
                            return 0
                        total_seen += 1
                        # 详情页 URL 与干/湿切面无关（`wet/` 只是列表筛选），
                        # 带上它一定会 404 —— 这是实测过的站点结构。
                        url = f"{base_url(game_year)}/{slug}/{uid}/"
                        if url in done:
                            skipped += 1
                            continue
                        page = _get(url)
                        time.sleep(sleep_s)
                        if not page:
                            state.setdefault("failed", []).append(url)
                            continue
                        rec = parse_detail(page, track_id, url, game_year, surface)
                        if rec is None:
                            state.setdefault("failed", []).append(url)
                            continue
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        written += 1
                        done.add(url)
                        state["done"] = sorted(done)
                        if check_only and written >= 2:
                            _save_state(state, state_file)
                            print(
                                f"[check] 已解析 {written} 条，字段数 "
                                f"{rec['setup_field_count']}",
                                flush=True,
                            )
                            print(json.dumps(rec, ensure_ascii=False, indent=2)[:1200])
                            return 0

    finish()
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
    by_slice: dict[str, int] = {}
    for r in rows:
        by_track[r["track_id"]] = by_track.get(r["track_id"], 0) + 1
        key = f"{r.get('game_year', '?')}-{r.get('surface', '?')}"
        by_slice[key] = by_slice.get(key, 0) + 1
    fields = [r["setup_field_count"] for r in rows]
    print(f"[summary] 总样本 {len(rows)}；覆盖赛道 {len(by_track)}；"
          f"字段数 min/median/max = {min(fields)}/{sorted(fields)[len(fields)//2]}/{max(fields)}")
    print(f"[summary] 切面分布: {json.dumps(by_slice, ensure_ascii=False, sort_keys=True)}")
    print(f"[summary] 每站: {json.dumps(by_track, ensure_ascii=False, sort_keys=True)}")
    missing = [t for t in TRACK_SLUGS.values() if t not in by_track]
    print(f"[summary] 缺数据赛道: {missing or '无'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抓取 F1Laps 公开调教-圈速配对")
    ap.add_argument("--tracks", default="", help="逗号分隔 slug；空=全部 24 站")
    ap.add_argument("--dry", action="store_true", help="抓干地（默认）")
    ap.add_argument("--wet", action="store_true", help="抓湿地")
    ap.add_argument(
        "--games", default="2026",
        help="逗号分隔游戏年版（2026/2025/2024）；默认 2026",
    )
    ap.add_argument(
        "--all-slices", action="store_true",
        help="等价于 --games 2026,2025,2024 --dry --wet（6 切面全量）",
    )
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="请求间隔秒")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--out", default="", help="输出 jsonl 路径（并行分片时用）")
    ap.add_argument("--check", action="store_true", help="只抓 2 条自检解析")
    ap.add_argument("--summary", action="store_true", help="打印数据集概况")
    ap.add_argument(
        "--normalize", action="store_true",
        help="不抓取，只对现有语料做自愈（补 game_year/surface、按 URL 去重）",
    )
    ap.add_argument(
        "--verify", action="store_true",
        help="只校验现有语料（缺元数据/重复 URL 即非零退出），给 CI 当门禁",
    )
    args = ap.parse_args(argv)

    if args.summary:
        return summarize()

    if args.normalize:
        target = Path(args.out) if args.out else OUT_JSONL
        stats = normalize_corpus(target)
        print(f"[normalize] {target}：共 {stats['total']} 条；"
              f"补元数据 {stats['repaired']} 条；去重 {stats['duplicates']} 条；"
              f"坏行 {stats['broken']} 条")
        return 0

    if args.verify:
        target = Path(args.out) if args.out else OUT_JSONL
        problems = verify_corpus(target)
        if problems:
            for p in problems:
                print(f"[verify] ✗ {p}", file=sys.stderr)
            return 1
        print(f"[verify] ✓ 语料通过（{target}）")
        return 0

    tracks = ([s.strip() for s in args.tracks.split(",") if s.strip()]
              if args.tracks else list(TRACK_SLUGS))
    unknown = [t for t in tracks if t not in TRACK_SLUGS]
    if unknown:
        print(f"[error] 未知赛道 slug: {unknown}", file=sys.stderr)
        return 2

    games: tuple[int, ...]
    if args.all_slices:
        games = ALL_GAMES
        surfaces: tuple[str, ...] = ("dry", "wet")
    else:
        games = tuple(int(g.strip()) for g in args.games.split(",") if g.strip())
        bad = [g for g in games if g not in GAME_SLUGS]
        if bad:
            print(f"[error] 不支持的游戏年版: {bad}", file=sys.stderr)
            return 2
        surfaces = tuple(s for s, on in (("dry", True), ("wet", args.wet)) if on)

    if args.check:
        tracks = tracks[:1]
        games = games[:1]
        # 保留用户选的干/湿：`--wet --check` 要验证的就是湿地解析

    return run(
        tracks, games, surfaces,
        sleep_s=args.sleep,
        max_minutes=args.max_minutes,
        check_only=args.check,
        out=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    sys.exit(main())
