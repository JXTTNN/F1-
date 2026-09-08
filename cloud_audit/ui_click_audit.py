"""云端真人式 UI 审计。

约束: 只做「人能做的事」——点击页面元素、在输入框/下拉里输入。
不直接调用任何 HTTP API，不读服务端日志；结论只依据页面呈现的内容与前端异常。
HTTP 状态码仅作旁证收集（不参与交互）。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

BASE = os.environ.get("AUDIT_BASE", "http://127.0.0.1:8000").rstrip("/")
ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SHOTS = ROOT / "shots"
REPORTS.mkdir(parents=True, exist_ok=True)
SHOTS.mkdir(parents=True, exist_ok=True)

#: 内置 LLM 审计: mock Ollama 返回内容的标记 (见 cloud_audit/mock_ollama.py)。
LLM_MARKER = "[MOCK-LLM]"

RESULTS: list[dict] = []
CONSOLE_ERR: list[str] = []
PAGE_ERR: list[str] = []
NET_FAIL: list[str] = []
HTTP_BAD: list[str] = []


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
    except AssertionError as exc:
        rec(name, False, "ASSERT " + str(exc)[:250])
    except Exception as exc:  # noqa: BLE001
        rec(name, False, f"{type(exc).__name__}: {str(exc)[:250]}")


def err_count() -> int:
    return len(CONSOLE_ERR) + len(PAGE_ERR)


# --------------------------------------------------------------------------
# 内置 LLM 审计夹具: mock Ollama 子进程 (OpenAI 兼容端点 127.0.0.1:11434)。
# 夹具生命周期属于测试环境管理, 不属于「对被测应用的操作」——审计本身
# 仍然只点击 UI / 在输入框输入。
# --------------------------------------------------------------------------
_MOCK_PROC: subprocess.Popen | None = None
_MOCK_EXTERNAL = False  # mock 由工作流提前拉起 (非本脚本子进程)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_mock_llm() -> subprocess.Popen | None:
    """确保 mock Ollama 在 127.0.0.1:11434 可用。

    优先复用工作流已拉起的实例 (preload 时序需要 mock 先于 API server
    启动); 仅当端口无人监听时才由本脚本拉起子进程。
    """
    global _MOCK_PROC, _MOCK_EXTERNAL
    if _port_open(11434):
        _MOCK_EXTERNAL = True
        print("[mock-llm] already running (workflow-started), reusing", flush=True)
        return None
    try:
        _MOCK_PROC = subprocess.Popen(
            [sys.executable, str(ROOT / "cloud_audit" / "mock_ollama.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mock-llm] start failed: {exc}", flush=True)
        return None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _port_open(11434):
            print("[mock-llm] ready on 127.0.0.1:11434", flush=True)
            return _MOCK_PROC
        if _MOCK_PROC.poll() is not None:
            print("[mock-llm] process exited early", flush=True)
            return None
        time.sleep(0.3)
    print("[mock-llm] not ready in 15s", flush=True)
    return None


def stop_mock_llm() -> None:
    """终止 mock Ollama (模拟 LLM 服务掉线)。

    外部实例 (工作流拉起) 通过 pid 文件精准终止; 本脚本的子进程直接
    terminate。终止后等待端口真正关闭, 保证后续回退检查时序正确。
    """
    global _MOCK_PROC, _MOCK_EXTERNAL
    if _MOCK_PROC is not None and _MOCK_PROC.poll() is None:
        _MOCK_PROC.terminate()
        try:
            _MOCK_PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _MOCK_PROC.kill()
    elif _MOCK_EXTERNAL:
        pid_file = ROOT / "reports" / "mock_ollama.pid"
        try:
            pid = int(pid_file.read_text(encoding="ascii").strip())
            os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError) as exc:
            print(f"[mock-llm] external stop failed: {exc}", flush=True)
    _MOCK_PROC = None
    _MOCK_EXTERNAL = False
    deadline = time.time() + 5.0
    while time.time() < deadline and _port_open(11434):
        time.sleep(0.2)


LAPS_PAYLOAD = {
    "reference_lap": {"lap_time": 90.0, "sector_times": [30.0, 30.0, 30.0]},
    "laps": [
        {"lap_time": 89.5, "sector_times": [29.8, 30.0, 29.7]},
        {"lap_time": 90.4, "sector_times": [30.2, 30.1, 30.1]},
        {"lap_time": 89.9, "sector_times": [29.9, 30.0, 30.0]},
    ],
}
DRIVER_LAPS = [
    {"lap_time": 90.0, "sector_times": [30.0, 30.0, 30.0]},
    {"lap_time": 90.3, "sector_times": [30.1, 30.1, 30.1]},
]
TEAMMATE_LAPS = [
    {"lap_time": 90.2, "sector_times": [30.1, 30.0, 30.1]},
    {"lap_time": 90.1, "sector_times": [30.0, 30.0, 30.1]},
]


# --------------------------------------------------------------------------
# 首页（实时面板）
# --------------------------------------------------------------------------
def audit_index(page) -> None:
    def load():
        page.goto(BASE + "/", wait_until="domcontentloaded", timeout=60000)
        # <option> 在折叠的 <select> 内没有可见盒模型，必须用 attached 等。
        page.wait_for_selector("#track-select option", state="attached", timeout=60000)
        opts = page.eval_on_selector_all("#track-select option", "els => els.length")
        assert opts > 1, f"赛道下拉仅 {opts} 项"
        return f"赛道下拉 {opts} 项"

    guard("首页加载 / 赛道下拉填充", load)

    def health():
        page.wait_for_function(
            "() => { const b = document.getElementById('health-badge');"
            " return b && b.textContent && !b.textContent.includes('…'); }",
            timeout=60000,
        )
        return page.inner_text("#health-badge")

    guard("健康检查徽标显示 API 状态", health)

    def pick_track():
        page.select_option("#track-select", index=1)
        page.wait_for_timeout(300)
        return page.inner_text("#track-info")

    guard("切换赛道 → 赛道信息更新", pick_track)

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

    def search():
        page.select_option("#driver-style-select", "aggressive")
        page.select_option("#tire-wear-select", "2")
        page.click("#search-btn")
        page.wait_for_selector("#search-results-wrap:not(.hidden)", timeout=300000)
        gain = page.inner_text("#search-gain-val").strip()
        rows = page.eval_on_selector_all("#search-diff-body tr", "els => els.length")
        return f"收益 {gain} / 差异行 {rows}"

    guard("点击「调教搜索」→ 结果面板出现", search)

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

    def empty_chat():
        before = err_count()
        page.fill("#chat-input", "")
        page.click("#chat-send")
        page.wait_for_timeout(800)
        assert err_count() == before, "空输入触发前端异常"
        return "无异常"

    guard("空输入点击发送（健壮性）", empty_chat)

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

    def extreme():
        """越界输入：应被输入侧钳制，不再打出 400。"""
        before_bad = len(HTTP_BAD)
        inp = page.query_selector('#setup-container input[data-field="front_wing"]')
        if inp is None:
            return "SKIP 未找到调教输入框"
        max_attr = inp.get_attribute("max")
        inp.fill("9999")
        inp.dispatch_event("change")
        page.wait_for_timeout(300)
        clamped = inp.input_value()
        page.click("#predict-btn")
        page.wait_for_function(
            "() => { const e = document.getElementById('predict-out');"
            " return e && e.textContent && !e.textContent.includes('预测中'); }",
            timeout=180000,
        )
        txt = page.inner_text("#predict-out").strip()
        new_bad = [b for b in HTTP_BAD[before_bad:] if "predict" in b]
        assert not new_bad, f"越界输入仍触发 4xx: {new_bad}"
        assert "预测不可用" not in txt, f"预测失败: {txt}"
        return f"max={max_attr} 钳制后={clamped} / {txt[:100]}"

    guard("越界调教值（9999）→ 输入钳制 + 预测", extreme)

    def export_setup():
        with page.expect_download(timeout=30000) as dl:
            page.click("#export-setup-btn")
        return f"下载文件 {dl.value.suggested_filename}"

    guard("点击「导出调教」→ 触发下载", export_setup)

    def export_samples():
        with page.expect_download(timeout=60000) as dl:
            page.click("#export-samples-btn")
        return f"下载文件 {dl.value.suggested_filename}"

    guard("点击「导出样本 (Parquet)」→ 触发下载", export_samples)

    def llm_enhanced():
        """内置 LLM 增强生效: 点击「获取反馈」, 摘要应被 LLM 改写 (含标记)。

        前置: 工作流已以 F1OPT_LLM_BACKEND=local 启动服务并调用
        /api/llm/preload (引擎 _llm_loaded=True), 且 mock Ollama 在线。
        """
        mock_up = _MOCK_EXTERNAL or (
            _MOCK_PROC is not None and _MOCK_PROC.poll() is None
        )
        if not mock_up:
            return "SKIP mock Ollama 不可用"
        page.fill("#feedback-input", "弯中推头，请给一句话建议")
        page.click("#feedback-btn")
        page.wait_for_function(
            "() => { const e = document.getElementById('fb-summary');"
            " return e && e.textContent && !e.textContent.startsWith('点击'); }",
            timeout=180000,
        )
        txt = page.inner_text("#fb-summary").strip()
        assert txt, "反馈摘要为空"
        assert LLM_MARKER in txt, f"LLM 增强未生效, 摘要无标记: {txt[:120]}"
        return txt[:140]

    guard("内置 LLM 增强：点击「获取反馈」→ 摘要含 LLM 改写", llm_enhanced)

    def llm_fallback():
        """LLM 服务掉线: 应静默回退规则引擎, 摘要非空且无标记、无前端异常。"""
        before_err = err_count()
        stop_mock_llm()
        page.fill("#feedback-input", "LLM 掉线后还能用吗？")
        page.click("#feedback-btn")
        page.wait_for_function(
            "() => { const e = document.getElementById('fb-summary');"
            " return e && e.textContent && !e.textContent.startsWith('点击'); }",
            timeout=180000,
        )
        txt = page.inner_text("#fb-summary").strip()
        assert txt, "回退后摘要为空"
        assert LLM_MARKER not in txt, "LLM 已掉线但仍返回了 LLM 内容"
        assert err_count() == before_err, "LLM 掉线触发前端异常"
        return f"回退规则引擎 / 摘要 {txt[:100]}"

    guard("LLM 服务中断 → 静默回退规则引擎（健壮性）", llm_fallback)
    snap(page, "index-final")


# --------------------------------------------------------------------------
# 仪表盘（智能分析中心）—— 逐页签真人操作
# --------------------------------------------------------------------------
def audit_dashboard(page) -> None:
    def load():
        page.goto(BASE + "/dashboard.html", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("#str-track option", state="attached", timeout=60000)
        return page.title()

    guard("仪表盘页面加载", load)

    def strategy():
        page.fill("#str-laps", "58")
        page.fill("#str-fuel", "105.5")
        page.click("#str-run")
        page.wait_for_selector("#str-result-panel:not(.hidden)", timeout=120000)
        txt = page.inner_text("#str-result-panel").strip()
        assert "失败" not in txt[:200], txt[:200]
        return (
            f"策略 {page.inner_text('#str-type')} / 停 {page.inner_text('#str-stops-count')}"
            f" / 总时 {page.inner_text('#str-total')}"
        )

    guard("仪表盘-赛道策略：填表 → 生成策略", strategy)

    def bayes():
        page.click('.tab[data-tab="search"]')
        page.wait_for_timeout(300)
        page.click("#bayes-run")
        page.wait_for_selector("#bayes-result-panel:not(.hidden)", timeout=240000)
        rows = page.eval_on_selector_all("#bayes-setup-table tbody tr", "els => els.length")
        assert rows > 0, "贝叶斯推荐调教表为空"
        return f"收益 {page.inner_text('#bayes-gain')} / 调教行 {rows}"

    guard("仪表盘-调教搜索：贝叶斯", bayes)

    def pareto():
        page.click('.subtab[data-subtab="pareto"]')
        page.wait_for_timeout(300)
        page.click("#pareto-run")
        page.wait_for_selector("#pareto-result-panel:not(.hidden)", timeout=240000)
        dots = page.eval_on_selector_all("#pareto-scatter circle", "els => els.length")
        first_row = page.eval_on_selector(
            "#pareto-setups-table tbody tr",
            "el => el.textContent.replace(/\\s+/g,' ').trim()",
        )
        # 真实数据校验: 散点不应为空, 且「圈速」列不再是恒定的 "—"
        assert dots > 0, "Pareto 散点图无点"
        assert "—" not in first_row.split("关键")[0][:24], f"目标值列仍为空: {first_row}"
        return (
            f"前沿 {page.inner_text('#pareto-front-size')} / 散点 {dots} / 首行 {first_row[:80]}"
        )

    guard("仪表盘-调教搜索：Pareto 真实前沿", pareto)

    def compare():
        page.click('.tab[data-tab="compare"]')
        page.wait_for_timeout(300)
        page.fill("#cmp-input", json.dumps(LAPS_PAYLOAD, ensure_ascii=False))
        page.click("#cmp-run")
        page.wait_for_selector("#cmp-result-panel:not(.hidden)", timeout=120000)
        rows = page.eval_on_selector_all("#cmp-table tbody tr", "els => els.length")
        bars = page.eval_on_selector_all("#cmp-sector-chart rect", "els => els.length")
        strength = page.inner_text("#cmp-strength").strip()
        assert rows == len(LAPS_PAYLOAD["laps"]), f"对比行 {rows}"
        assert bars == 3, f"扇区柱状图应 3 根柱, 实际 {bars}"
        # 后端 strength/weakness 已是 1-based, 展示必须在 S1..S3 内
        assert re.search(r"强 S[1-3] / 弱 S[1-3]", strength), f"强弱扇区越界: {strength}"
        return f"行 {rows} / 扇区柱 {bars} / {strength}"

    guard("仪表盘-圈速对比：扇区 Δ 图与强弱扇区", compare)

    def bad_json():
        before = err_count()
        page.fill("#cmp-input", "{ 这不是 JSON")
        page.click("#cmp-run")
        page.wait_for_timeout(1200)
        toast = page.inner_text("#toast").strip()
        assert err_count() == before, "非法 JSON 触发前端异常"
        assert "JSON" in toast, f"未给出可读提示: {toast}"
        return f"提示: {toast[:80]}"

    guard("仪表盘-圈速对比：非法 JSON 输入提示", bad_json)

    def teammates():
        page.fill("#tm-driver", json.dumps(DRIVER_LAPS))
        page.fill("#tm-teammate", json.dumps(TEAMMATE_LAPS))
        page.click("#tm-run")
        page.wait_for_selector("#tm-result-panel:not(.hidden)", timeout=120000)
        verdict = page.inner_text("#tm-verdict").strip()
        assert verdict and verdict != "—", "队友对比无裁决"
        return f"裁决 {verdict[:90]}"

    guard("仪表盘-队友对比", teammates)

    def weather():
        page.click('.tab[data-tab="weather"]')
        page.wait_for_timeout(300)
        page.click("#w-run")
        page.wait_for_selector("#w-result-panel:not(.hidden)", timeout=120000)
        grip = page.inner_text("#w-grip").strip()
        assert grip and grip != "—", "抓地力无结果"
        return (
            f"抓地 {grip} / Δ {page.inner_text('#w-delta')} / 配方 {page.inner_text('#w-compound')}"
        )

    guard("仪表盘-天气影响", weather)

    def health():
        page.click('.tab[data-tab="health"]')
        page.wait_for_timeout(300)
        page.click("#h-refresh")
        page.wait_for_selector("#h-result-panel:not(.hidden)", timeout=60000)
        st = page.inner_text("#h-status-val").strip()
        assert st == "ok", f"健康状态异常: {st}"
        mods = page.eval_on_selector_all("#h-modules .chip", "els => els.length")
        return f"状态 {st} / 模型 {page.inner_text('#h-model')} / 模块 {mods}"

    guard("仪表盘-系统健康（扩展接口连通性）", health)
    snap(page, "dashboard-final")


def main() -> None:
    start_mock_llm()
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

        def on_response(resp):
            try:
                if resp.status >= 400:
                    HTTP_BAD.append(f"{resp.request.method} {resp.url} -> {resp.status}")
            except Exception:  # noqa: BLE001
                pass

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)
        page.on("requestfailed", on_requestfailed)
        page.on("response", on_response)

        try:
            audit_index(page)
            audit_dashboard(page)
        finally:
            snap(page, "final-state")
            ctx.close()
            browser.close()
            stop_mock_llm()

    rec("无未捕获 JS 异常", not PAGE_ERR, "; ".join(PAGE_ERR[:5]))
    rec("无 console.error", not CONSOLE_ERR, "; ".join(CONSOLE_ERR[:5]))
    rec(
        "无 4xx/5xx 响应",
        not HTTP_BAD,
        "; ".join(sorted(set(HTTP_BAD))[:8]),
    )

    payload = {
        "base": BASE,
        "total": len(RESULTS),
        "passed": sum(1 for r in RESULTS if r["ok"]),
        "failed": sum(1 for r in RESULTS if not r["ok"]),
        "results": RESULTS,
        "console_errors": CONSOLE_ERR[:60],
        "page_errors": PAGE_ERR[:60],
        "request_failed": NET_FAIL[:60],
        "http_bad": sorted(set(HTTP_BAD))[:60],
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
