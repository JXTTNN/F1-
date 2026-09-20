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
        # 真值来源：setup_tuner/ui/index.html 的 4 组 symptom 输入
        # （22 症状 − lap_slow 已降级 = 21；entry6/apex3/exit4/global8）
        item("③症状条目=21", total == 21, f"实际 {total}（lap_slow 已降级）")
        expand_all_groups(page)
        counts = group_counts(page)
        item("③分组 6/3/4/0", counts == {"entry": 6, "apex": 3, "exit": 4, "global": 0},
             f"{counts}")
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

    # ── ④ 生成建议（21 参数报告）──
    page.click("#btn-generate-suggest")
    page.wait_for_timeout(2500)
    rows = page.locator("#report-table-wrap table tr").count()
    item("④报告生成", rows > 0, f"表格行数={rows}")
    item("④报告参数行=20", rows - 1 == 20, f"数据行={rows - 1}")
    page.screenshot(path=str(SHOTS / "04_report.png"), full_page=True)

    # ── ⑤ 遥测面板 ──
    for tid, label in (("tel-rpm", "转速"), ("tel-laptime", "圈速"),
                       ("tel-sector", "扇区"), ("tel-corner", "当前弯")):
        item(f"⑤遥测面板含{label}", page.locator(f"#{tid}").count() == 1)

    # ── ⑥ 赛道级全局反馈 ──
    page.click("#track-feedback-btn")
    page.wait_for_timeout(600)
    counts = group_counts(page)
    item("⑥赛道模式仅全局组", counts.get("global", 0) == 8
         and counts.get("entry") == 0, f"{counts}")
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

    (REPORTS / "ui_click_audit_result.json").write_text(
        json.dumps([{"tag": t, "verdict": v, "detail": d} for t, v, d in RESULTS],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    from collections import Counter
    c = Counter(v for _, v, _ in RESULTS)
    print("汇总：" + " ".join(f"{k}={c.get(k, 0)}" for k in ("PASS", "FAIL")))
    for t, v, d in RESULTS:
        if v == "FAIL":
            print(f"  [FAIL] {t} :: {d}")
    return 1 if c.get("FAIL", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
