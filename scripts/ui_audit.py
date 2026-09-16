"""真人式 UI 点击审计 —— 用真实浏览器（Edge/Chromium）走完主流程。

这是 ``cloud_audit/ui_click_audit.py`` 的可用替代：后者仍指向已废弃的旧 UI
（``/dashboard.html`` 与 ``#predict-btn`` 等选择器均不存在），等于空跑。

前置：
    pip install playwright        # 复用系统 Edge，无需下载 Chromium
    先启动服务：python -m uvicorn <app> --port 8199

用法：
    AUDIT_BASE=http://127.0.0.1:8199 python scripts/ui_audit.py

输出：逐项 PASS/FAIL + ``shots/*.png`` 截图 + ``shots/ui_audit_result.json``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import os

BASE = os.environ.get("AUDIT_BASE", "http://127.0.0.1:8199").rstrip("/")
SHOTS = Path(os.environ.get("AUDIT_SHOTS", "shots"))
SHOTS.mkdir(parents=True, exist_ok=True)

RESULTS: list[tuple[str, str, str]] = []
CONSOLE_ERRORS: list[str] = []
PAGE_ERRORS: list[str] = []
FAILED_REQUESTS: list[str] = []


def item(tag: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((tag, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f" :: {detail}" if detail else ""), flush=True)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda m: CONSOLE_ERRORS.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: PAGE_ERRORS.append(str(e)))
        page.on("requestfailed", lambda r: FAILED_REQUESTS.append(f"{r.url} :: {r.failure}"))

        # ---------- ① 首页加载 ----------
        page.goto(BASE + "/", wait_until="domcontentloaded")
        page.wait_for_timeout(1200)
        item("①首页加载", page.title() != "", f"title={page.title()!r}")
        page.screenshot(path=str(SHOTS / "01_home.png"), full_page=True)

        # ---------- ② 选择赛道 ----------
        track_id = "suzuka"
        page.select_option("#track-select", track_id)
        page.wait_for_timeout(1500)
        meta = page.inner_text("#track-meta")
        item("②选择赛道", "Suzuka" in meta or "铃鹿" in meta, f"meta={meta!r}")
        svg_count = page.locator("#track-map-wrap svg").count()
        hotzones = page.locator("#track-map-wrap .hotzone").count()
        item("②赛道图渲染", svg_count >= 1, f"SVG 数={svg_count}")
        item("②弯道热区数量", hotzones == 18, f"热区={hotzones}（suzuka 应为 18）")
        page.screenshot(path=str(SHOTS / "02_track_suzuka.png"), full_page=True)

        # ---------- ③ 点击弯道热区 → 反馈面板 ----------
        corners = page.locator("#track-map-wrap .hotzone")
        corners.nth(0).click(force=True)
        page.wait_for_timeout(800)
        panel_visible = page.locator("#feedback-overlay").is_visible()
        item("③点击弯道弹出反馈面板", panel_visible,
             f"corner={page.inner_text('#fb-corner-number') if panel_visible else '-'}")
        if panel_visible:
            boxes = page.locator('#feedback-overlay input[type="checkbox"]')
            n_boxes = boxes.count()
            item("③症状清单条目数", n_boxes == 15, f"checkbox={n_boxes}（15 症状）")
            # 症状按当前弯阶段分组显示：只点「当前可见」的项，并记录可见数
            visible_boxes = page.locator('#feedback-overlay input[type="checkbox"]:visible')
            n_visible = visible_boxes.count()
            item("③当前阶段可见症状数", n_visible > 0, f"可见 checkbox={n_visible}")
            if n_visible >= 2:
                visible_boxes.nth(0).check()
                visible_boxes.nth(1).check()
            elif n_visible == 1:
                visible_boxes.nth(0).check()
            checked = page.locator('#feedback-overlay input[type="checkbox"]:checked').count()
            item("③已勾选症状数", checked >= 1, f"checked={checked}")
            page.screenshot(path=str(SHOTS / "03_feedback_panel.png"), full_page=True)
            page.click("#fb-submit")
            page.wait_for_timeout(1200)
            # 面板应关闭或给出成功提示
            closed = not page.locator("#feedback-overlay").is_visible()
            item("③提交反馈后面板关闭", closed)
            page.screenshot(path=str(SHOTS / "04_after_feedback.png"), full_page=True)

        # ---------- ④ 生成建议 ----------
        page.click("#btn-generate-suggest")
        page.wait_for_timeout(2500)
        rows = page.locator("#report-table-wrap table tr").count()
        summary = page.inner_text("#report-summary")
        item("④建议报告生成", rows > 0, f"表格行数={rows}")
        item("④报告参数行数=21", rows - 1 == 21, f"数据行={rows - 1}")
        item("④报告摘要非空", len(summary.strip()) > 0, f"summary={summary[:60]!r}")
        page.screenshot(path=str(SHOTS / "05_report.png"), full_page=True)

        # ---------- ⑤ 遥测面板字段 ----------
        for tid, label in (("tel-rpm", "转速"), ("tel-laptime", "圈速"),
                           ("tel-sector", "扇区"), ("tel-corner", "当前弯"),
                           ("tel-speed", "速度"), ("tel-gear", "档位")):
            exists = page.locator(f"#{tid}").count() == 1
            item(f"⑤遥测面板含{label}", exists)
        page.screenshot(path=str(SHOTS / "06_final.png"), full_page=True)

        # ---------- ⑥ 控制台 / 网络 ----------
        item("⑥无 console error", not CONSOLE_ERRORS, "; ".join(CONSOLE_ERRORS[:3]))
        item("⑥无 page error", not PAGE_ERRORS, "; ".join(PAGE_ERRORS[:3]))
        item("⑥无失败请求", not FAILED_REQUESTS, "; ".join(FAILED_REQUESTS[:3]))

        browser.close()

    print("\n" + "=" * 70)
    from collections import Counter
    c = Counter(v for _, v, _ in RESULTS)
    print("汇总：" + " ".join(f"{k}={c.get(k, 0)}" for k in ("PASS", "FAIL")))
    (SHOTS / "ui_audit_result.json").write_text(
        json.dumps([{"tag": t, "verdict": v, "detail": d} for t, v, d in RESULTS],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    fails = [(t, d) for t, v, d in RESULTS if v == "FAIL"]
    for t, d in fails:
        print(f"  [FAIL] {t} :: {d}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
