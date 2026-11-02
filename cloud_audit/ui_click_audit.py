"""云端真人式 UI 审计 —— 针对现役 UI（index.html）的完整主流程。

2026-09-16 重写：旧版指向已废弃的 /dashboard.html 与 #predict-btn 等
旧 UI 选择器（9/9 不存在），等于空跑。新版对齐现役 UI 的真实主流程：

    选赛道 → 赛道图热区 → 点击弯道 → 反馈面板（17 症状/三档强度）→
    提交 → 生成建议（21 参数报告）→ 遥测面板 → 赛道级全局反馈 → 控制台体检

约束不变：只做「人能做的事」——点击页面元素、在输入框/下拉里输入；
不直接调用任何写 API；结论只依据页面呈现的内容与前端异常。
（HTTP 状态码仅作旁证收集，不参与交互。）

浏览器：CI 由 workflow 安装 chromium；本地可用 AUDIT_CHANNEL=msedge 复用 Edge。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# Windows 控制台默认 cp1252，强制 UTF-8 输出避免中文明细被吞
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

from playwright.sync_api import sync_playwright  # noqa: E402

BASE = os.environ.get("AUDIT_BASE", "http://127.0.0.1:8000").rstrip("/")
ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SHOTS = ROOT / "shots"
REPORTS.mkdir(parents=True, exist_ok=True)
SHOTS.mkdir(parents=True, exist_ok=True)

RESULTS: list[tuple[str, str, str]] = []
CONSOLE_ERRORS: list[str] = []
PAGE_ERRORS: list[str] = []
BAD_RESPONSES: list[str] = []


def _expected_param_count() -> int:
    """报告参数行数 = 领域模型的可调参数量（不写死，避免长期误报）。

    历史教训：这里曾写死 ``21``，而 ``ALL_SETUP_FIELDS`` 是 20 个 —— 审计
    因此长期判红，"④报告参数行"这一条从发现问题的哨兵退化成噪声。
    """
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from setup_tuner.domain.setup import ALL_SETUP_FIELDS

        return len(ALL_SETUP_FIELDS)
    except Exception:  # noqa: BLE001 — 审计脚本必须能独立运行
        return 20


EXPECTED_PARAM_COUNT = _expected_param_count()


def _expected_symptom_groups() -> dict[str, int]:
    """UI 反馈面板应渲染的「分组 → 条目数」，按领域症状表推导。

    不再写死 ``5/3/4/5``：症状表随产品演进（新增症状、把 ``lap_slow`` 从可勾
    列表降级为综合结论），写死的数字只会让审计长期误报 —— 而误报多了，真正的
    回归就没人看了。
    """
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from setup_tuner.domain.symptoms import SYMPTOM_INFO

        out: dict[str, int] = {}
        for name, info in SYMPTOM_INFO.items():
            # app.js 注释明确：lap_slow 已降级为综合结论，不出现在反馈面板
            if str(name) == "lap_slow":
                continue
            cat = str(info.get("category") or "global")
            out[cat] = out.get(cat, 0) + 1
        return out
    except Exception:  # noqa: BLE001 — 审计脚本必须能独立运行
        return {"entry": 6, "apex": 3, "exit": 4, "global": 8}


EXPECTED_SYMPTOM_GROUPS = _expected_symptom_groups()
EXPECTED_SYMPTOM_TOTAL = sum(EXPECTED_SYMPTOM_GROUPS.values())


def item(tag: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((tag, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f" :: {detail}" if detail else ""),
          flush=True)


def group_counts(page) -> dict[str, int]:
    """{category: 组内症状条目数}（与折叠状态无关）。"""
    out: dict[str, int] = {}
    for g in page.locator(".sym-group").all():
        cat = g.get_attribute("data-category")
        out[cat] = g.locator('input[name="symptom"]').count() if g.is_visible() else 0
    return out


def expand_all_groups(page) -> None:
    """像真实用户一样点组头，展开入弯/弯中/出弯三组。"""
    for cat in ("entry", "apex", "exit"):
        for head in page.locator(f'.sym-group[data-category="{cat}"] .sym-group-head').all():
            head.click()
            page.wait_for_timeout(120)


def run(page) -> None:
    # ── ① 首页 ──
    page.goto(BASE + "/", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(1200)
    item("①首页加载", "F1OPT" in page.title(), f"title={page.title()!r}")
    page.screenshot(path=str(SHOTS / "01_home.png"), full_page=True)

    # ── ② 选赛道 + 赛道图 ──
    page.select_option("#track-select", "suzuka")
    page.wait_for_timeout(1500)
    hotzones = page.locator("#track-map-wrap .hotzone").count()
    item("②选择赛道并渲染", hotzones == 18, f"热区={hotzones}（suzuka 应为 18）")
    page.screenshot(path=str(SHOTS / "02_track.png"), full_page=True)

    # ── ③ 弯道反馈（17 症状 / 三档强度）──
    page.locator("#track-map-wrap .hotzone").nth(0).click(force=True)
    page.wait_for_timeout(800)
    panel = page.locator("#feedback-overlay").is_visible()
    item("③反馈面板弹出", panel)
    if panel:
        total = page.locator('#feedback-overlay input[name="symptom"]').count()
        item(f"③症状条目={EXPECTED_SYMPTOM_TOTAL}",
             total == EXPECTED_SYMPTOM_TOTAL, f"实际 {total}（lap_slow 已降级）")
        expand_all_groups(page)
        counts = group_counts(page)
        # 弯道模式：入弯/弯中/出弯三组有内容，全局组存在但不显示
        expected_corner = dict(EXPECTED_SYMPTOM_GROUPS)
        expected_corner["global"] = 0
        item("③分组 " + "/".join(str(expected_corner[k])
                                 for k in ("entry", "apex", "exit", "global")),
             counts == expected_corner, f"{counts}")
        visible = page.locator('#feedback-overlay input[name="symptom"]:visible')
        n_visible = visible.count()
        item("③可见症状>0", n_visible > 0, f"可见 {n_visible}")
        if n_visible:
            visible.nth(0).check()
        page.wait_for_timeout(300)
        rng = page.locator(".fb-strength-item-range")
        item("③强度档位 1-3（默认 2）",
             rng.count() >= 1 and rng.first.get_attribute("min") == "1"
             and rng.first.get_attribute("max") == "3",
             f"min={rng.first.get_attribute('min') if rng.count() else '-'} "
             f"max={rng.first.get_attribute('max') if rng.count() else '-'}")
        page.screenshot(path=str(SHOTS / "03_feedback.png"), full_page=True)
        page.click("#fb-submit")
        page.wait_for_timeout(1200)
        item("③提交后面板关闭", not page.locator("#feedback-overlay").is_visible())

    # ── ④ 生成建议（全部调教参数报告）──
    page.click("#btn-generate-suggest")
    page.wait_for_timeout(2500)
    rows = page.locator("#report-table-wrap table tr").count()
    item("④报告生成", rows > 0, f"表格行数={rows}")
    item(f"④报告参数行={EXPECTED_PARAM_COUNT}",
         rows - 1 == EXPECTED_PARAM_COUNT, f"数据行={rows - 1}")
    page.screenshot(path=str(SHOTS / "04_report.png"), full_page=True)

    # ── ⑤ 遥测面板 ──
    for tid, label in (("tel-rpm", "转速"), ("tel-laptime", "圈速"),
                       ("tel-sector", "扇区"), ("tel-corner", "当前弯")):
        item(f"⑤遥测面板含{label}", page.locator(f"#{tid}").count() == 1)

    # ── ⑥ 赛道级全局反馈 ──
    page.click("#track-feedback-btn")
    page.wait_for_timeout(600)
    counts = group_counts(page)
    expected_track = {k: 0 for k in EXPECTED_SYMPTOM_GROUPS}
    expected_track["global"] = EXPECTED_SYMPTOM_GROUPS["global"]
    # 赛道模式：只有全局组有内容，入弯/弯中/出弯必须为空
    item("⑥赛道模式仅全局组", counts == expected_track, f"{counts}")
    gvals = [
        page.locator('.sym-group[data-category="global"] input[name="symptom"]').nth(i)
        .get_attribute("value")
        for i in range(counts.get("global", 0))
    ]
    item("⑥全局组含胎温过高", "tyre_overheat" in gvals, f"{gvals}")
    page.locator('.sym-group[data-category="global"] input[name="symptom"]').nth(0).check()
    page.click("#fb-submit")
    page.wait_for_timeout(1200)
    item("⑥全局反馈提交", not page.locator("#feedback-overlay").is_visible())
    page.screenshot(path=str(SHOTS / "05_track_feedback.png"), full_page=True)

    # ── ⑦ 前端体检 ──
    item("⑦无 console error", not CONSOLE_ERRORS, "; ".join(CONSOLE_ERRORS[:3]))
    item("⑦无 4xx 响应", not BAD_RESPONSES, "; ".join(BAD_RESPONSES[:5]))
    item("⑦无 page error", not PAGE_ERRORS, "; ".join(PAGE_ERRORS[:3]))


def main() -> int:
    channel = os.environ.get("AUDIT_CHANNEL") or None
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda m: CONSOLE_ERRORS.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: PAGE_ERRORS.append(str(e)))
        page.on("response", lambda r: BAD_RESPONSES.append(f"{r.status} {r.url}")
                 if r.status >= 400 else None)
        try:
            run(page)
        except Exception as e:  # noqa: BLE001
            item("⑧流程异常", False, f"{type(e).__name__}: {e}")
            page.screenshot(path=str(SHOTS / "99_error.png"), full_page=True)
        finally:
            browser.close()

    # 落盘格式必须与 cloud_audit/build_report.py 的读取口径一致
    # （``ui_audit.json`` + total/passed/failed/results/…）。历史上这里写的是
    # ``ui_click_audit_result.json`` 的另一种形状，而汇总脚本 rglob 的是
    # ``ui_audit.json`` —— 于是 UI 审计的每一条结论都被静默丢弃，
    # 报告里永远看不到「哪一项 UI 检查失败了」。
    from collections import Counter
    c = Counter(v for _, v, _ in RESULTS)
    (REPORTS / "ui_audit.json").write_text(
        json.dumps({
            "total": len(RESULTS),
            "passed": c.get("PASS", 0),
            "failed": c.get("FAIL", 0),
            "results": [
                {"name": t, "ok": v == "PASS", "detail": d} for t, v, d in RESULTS
            ],
            "page_errors": PAGE_ERRORS,
            "console_errors": CONSOLE_ERRORS,
            "bad_responses": BAD_RESPONSES,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("汇总：" + " ".join(f"{k}={c.get(k, 0)}" for k in ("PASS", "FAIL")))
    for t, v, d in RESULTS:
        if v == "FAIL":
            print(f"  [FAIL] {t} :: {d}")
    return 1 if c.get("FAIL", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
