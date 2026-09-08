"""云端真人式 UI 审计。

约束: 只做「人能做的事」——点击页面元素、在输入框里打字。
不直接调用任何 HTTP API，不读服务端日志；结论只依据页面呈现的内容与前端抛出的异常。
"""

from __future__ import annotations

import json
import os
import pathlib

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

BASE = os.environ.get("AUDIT_BASE", "http://127.0.0.1:8000").rstrip("/")
ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SHOTS = ROOT / "shots"
REPORTS.mkdir(parents=True, exist_ok=True)
SHOTS.mkdir(parents=True, exist_ok=True)

RESULTS: list[dict] = []
CONSOLE_ERR: list[str] = []
PAGE_ERR: list[str] = []
NET_FAIL: list[str] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:800]})
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += " :: " + str(detail)[:300]
    print(line, flush=True)


def snap(page, tag: str) -> None:
    try:
        page.screenshot(path=str(SHOTS / f"{tag}.png"), full_page=True)
    except Exception:  # noqa: BLE001
        pass


def guard(name: str, fn):
    try:
        detail = fn()
        rec(name, True, detail if isinstance(detail, str) else "")
    except PWTimeout as exc:
        rec(name, False, "TIMEOUT " + str(exc)[:250])
    except Exception as exc:  # noqa: BLE001
        rec(name, False, f"{type(exc).__name__}: {str(exc)[:250]}")


def err_count() -> int:
    return len(CONSOLE_ERR) + len(PAGE_ERR)


# --------------------------------------------------------------------------
# 审计步骤
# --------------------------------------------------------------------------
def audit_index(page) -> None:
    # --- 1. 首屏加载 ---
    def load():
        page.goto(BASE + "/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("#track-select option", timeout=60000)
        opts = page.eval_on_selector_all("#track-select option", "els => els.length")
        return f"赛道下拉 {opts} 项"

    guard("首页加载 / 赛道下拉填充", load)

    # --- 2. 健康检查徽标 ---
    def health():
        page.wait_for_function(
            "() => { const b = document.getElementById('health-badge');"
            " return b && b.textContent && !b.textContent.includes('…'); }",
            timeout=60000,
        )
        return page.inner_text("#health-badge")

    guard("健康检查徽标显示 API 状态", health)

    # --- 3. 切换赛道 ---
    def pick_track():
        page.select_option("#track-select", index=1)
        page.wait_for_timeout(300)
        return page.inner_text("#track-info")

    guard("切换赛道 → 赛道信息更新", pick_track)

    # --- 4. 预测圈速 ---
    def predict():
        page.click("#predict-btn")
        page.wait_for_function(
            "() => { const e = document.getElementById('predict-out');"
            " return e && e.textContent && !e.textContent.includes('预测中'); }",
            timeout=180000,
        )
        txt = page.inner_text("#predict-out").strip()
        assert txt, "预测输出为空"
        assert "预测不可用" not in txt, f"预测失败: {txt}"
        return txt

    guard("点击「预测圈速」", predict)

    # --- 5. 车手反馈（输入框提问） ---
    def feedback():
        page.fill("#feedback-input", "T1 入弯总推头怎么办？")
        page.click("#feedback-btn")
        page.wait_for_function(
            "() => { const e = document.getElementById('fb-summary');"
            " return e && e.textContent && !e.textContent.startsWith('点击'); }",
            timeout=180000,
        )
        txt = page.inner_text("#fb-summary").strip()
        assert txt, "反馈摘要为空"
        assert "反馈不可用" not in txt, f"反馈失败: {txt}"
        return txt[:160]

    guard("输入框提问 → 点击「获取反馈」", feedback)

    # --- 6. 调教搜索 ---
    def search():
        page.select_option("#driver-style-select", "aggressive")
        page.select_option("#tire-wear-select", "2")
        page.click("#search-btn")
        page.wait_for_selector("#search-results-wrap:not(.hidden)", timeout=300000)
        gain = page.inner_text("#search-gain-val").strip()
        rows = page.eval_on_selector_all("#search-diff-body tr", "els => els.length")
        return f"收益 {gain} / 差异行 {rows}"

    guard("点击「调教搜索」→ 结果面板出现", search)

    # --- 7. 应用推荐调教 ---
    def apply_rec():
        page.click("#apply-recommended-btn")
        page.wait_for_function(
            "() => { const e = document.getElementById('apply-status');"
            " return e && e.textContent.trim(); }",
            timeout=30000,
        )
        txt = page.inner_text("#apply-status").strip()
        assert "无推荐" not in txt, txt
        return txt

    guard("点击「应用推荐调教」", apply_rec)

    # --- 8. 对话（输入框 + 发送） ---
    def chat():
        page.fill("#chat-input", "为什么推头？")
        page.click("#chat-send")
        page.wait_for_function(
            "() => { const ms = document.querySelectorAll('#conv-log .msg.bot');"
            " if (ms.length < 1) return false;"
            " const last = ms[ms.length - 1].textContent || '';"
            " return !last.includes('分析中'); }",
            timeout=180000,
        )
        msgs = page.eval_on_selector_all("#conv-log .msg", "els => els.length")
        last = page.eval_on_selector_all(
            "#conv-log .msg.bot", "els => els.length ? els[els.length-1].textContent : ''"
        )
        assert "反馈不可用" not in last, f"对话返回异常: {last[:200]}"
        return f"消息数 {msgs} / 末条 {last[:80]}"

    guard("对话输入框提问 → 点击「发送」", chat)

    # --- 9. 空输入点击发送（不应崩溃） ---
    def empty_chat():
        before = err_count()
        page.fill("#chat-input", "")
        page.click("#chat-send")
        page.wait_for_timeout(800)
        assert err_count() == before, "空输入触发前端异常"
        return "无异常"

    guard("空输入点击发送（健壮性）", empty_chat)

    # --- 10. 迭代历史面板 ---
    def iters():
        page.click("#iter-link")
        page.wait_for_selector("#iter-panel:not(.hidden)", timeout=30000)
        page.wait_for_function(
            "() => { const l = document.getElementById('iter-list');"
            " return l && l.textContent && !l.textContent.includes('加载中'); }",
            timeout=30000,
        )
        items = page.eval_on_selector_all("#iter-list .iter-item", "els => els.length")
        detail = page.inner_text("#iter-detail").strip()
        assert "加载失败" not in detail, detail[:200]
        return f"迭代 {items} 条 / 详情 {detail[:60]}"

    guard("点击「迭代历史」", iters)
    page.click("#iter-link")
    page.wait_for_timeout(300)

    # --- 11. 极端输入：越界调教值 ---
    def extreme():
        before = err_count()
        inp = page.query_selector('#setup-container input[data-field="front_wing"]')
        if inp is None:
            return "SKIP 未找到调教输入框"
        inp.fill("9999")
        inp.dispatch_event("change")
        page.click("#predict-btn")
        page.wait_for_function(
            "() => { const e = document.getElementById('predict-out');"
            " return e && e.textContent && !e.textContent.includes('预测中'); }",
            timeout=180000,
        )
        txt = page.inner_text("#predict-out").strip()
        inp.fill("5")
        inp.dispatch_event("change")
        assert err_count() == before, "越界输入触发前端异常"
        return f"越界输入响应: {txt[:120]}"

    guard("越界调教值（9999）→ 预测（健壮性）", extreme)

    # --- 12. 导出当前调教（下载） ---
    def export_setup():
        with page.expect_download(timeout=30000) as dl:
            page.click("#export-setup-btn")
        d = dl.value
        return f"下载文件 {d.suggested_filename}"

    guard("点击「导出调教」→ 触发下载", export_setup)

    # --- 13. 导出样本 ---
    def export_samples():
        with page.expect_download(timeout=60000) as dl:
            page.click("#export-samples-btn")
        d = dl.value
        return f"下载文件 {d.suggested_filename}"

    guard("点击「导出样本 (Parquet)」→ 触发下载", export_samples)

    snap(page, "index-final")


def audit_dashboard(page) -> None:
    def load():
        page.goto(BASE + "/dashboard.html", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        return page.title()

    guard("仪表盘页面加载", load)

    # 通用扫描：逐个点击可见按钮，逐个切换下拉框，捕获前端异常
    def sweep():
        before = err_count()
        clicked = 0
        buttons = page.query_selector_all("button")
        for i, btn in enumerate(buttons):
            try:
                if not btn.is_visible():
                    continue
                label = (btn.inner_text() or "").strip()[:40] or f"button#{i}"
                btn.click(timeout=5000)
                page.wait_for_timeout(400)
                clicked += 1
            except PWTimeout:
                RESULTS.append(
                    {"name": f"仪表盘按钮[{i}] 点击超时", "ok": False, "detail": "TIMEOUT"}
                )
                print(f"[FAIL] 仪表盘按钮[{i}] 点击超时", flush=True)
            except Exception as exc:  # noqa: BLE001
                RESULTS.append(
                    {"name": f"仪表盘按钮[{i}] 点击异常", "ok": False, "detail": str(exc)[:200]}
                )
                print(f"[FAIL] 仪表盘按钮[{i}] 点击异常 :: {exc}", flush=True)
        selects = page.query_selector_all("select")
        for sel in selects:
            try:
                if not sel.is_visible():
                    continue
                opts = sel.query_selector_all("option")
                if len(opts) > 1:
                    values = [o.get_attribute("value") for o in opts]
                    sel.select_option(value=values[-1], timeout=5000)
                    page.wait_for_timeout(300)
            except Exception:  # noqa: BLE001
                pass
        new = err_count() - before
        return f"点击 {clicked} 个按钮 / 切换 {len(selects)} 个下拉，新增前端异常 {new} 条"

    guard("仪表盘 全按钮点击扫描", sweep)

    def inputs_sweep():
        before = err_count()
        n = 0
        for inp in page.query_selector_all("input"):
            try:
                if not inp.is_visible():
                    continue
                itype = (inp.get_attribute("type") or "text").lower()
                if itype in ("checkbox", "radio", "button", "submit", "file"):
                    continue
                inp.fill("1", timeout=5000)
                inp.dispatch_event("change")
                n += 1
            except Exception:  # noqa: BLE001
                pass
        page.wait_for_timeout(800)
        return f"填充 {n} 个输入框，新增前端异常 {err_count() - before} 条"

    guard("仪表盘 输入框填充扫描", inputs_sweep)
    snap(page, "dashboard-final")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
        page = ctx.new_page()

        def on_console(msg):
            if msg.type == "error":
                CONSOLE_ERR.append(msg.text[:400])

        def on_pageerror(exc):
            PAGE_ERR.append(str(exc)[:400])

        def on_requestfailed(req):
            try:
                NET_FAIL.append(f"{req.url} :: {req.failure}")
            except Exception:  # noqa: BLE001
                NET_FAIL.append(req.url)

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)
        page.on("requestfailed", on_requestfailed)

        try:
            audit_index(page)
            audit_dashboard(page)
        finally:
            snap(page, "final-state")
            ctx.close()
            browser.close()

    # 前端异常汇总
    rec("无未捕获 JS 异常", not PAGE_ERR, "; ".join(PAGE_ERR[:5]))
    rec("无 console.error", not CONSOLE_ERR, "; ".join(CONSOLE_ERR[:5]))

    payload = {
        "base": BASE,
        "total": len(RESULTS),
        "passed": sum(1 for r in RESULTS if r["ok"]),
        "failed": sum(1 for r in RESULTS if not r["ok"]),
        "results": RESULTS,
        "console_errors": CONSOLE_ERR[:60],
        "page_errors": PAGE_ERR[:60],
        "request_failed": NET_FAIL[:60],
    }
    (REPORTS / "ui_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n== UI AUDIT {payload['passed']}/{payload['total']} passed, "
        f"{payload['failed']} failed ==",
        flush=True,
    )


if __name__ == "__main__":
    main()
