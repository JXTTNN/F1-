"""汇总云端审计结果：解析 JUnit XML + UI 审计 JSON，生成 Markdown 报告与 Issue 正文。"""

from __future__ import annotations

import json
import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

MAX_ISSUE = 60000


def parse_junit(path: pathlib.Path) -> dict:
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:  # noqa: BLE001
        return {"file": path.name, "error": f"XML 解析失败: {exc}"}

    suites = [root] if root.tag == "testsuite" else list(root)
    total = fail = err = skip = 0.0
    try:
        for s in suites:
            total += int(float(s.get("tests", 0) or 0))
            fail += int(float(s.get("failures", 0) or 0))
            err += int(float(s.get("errors", 0) or 0))
            skip += int(float(s.get("skipped", 0) or 0))
    except Exception:  # noqa: BLE001
        pass

    bad: list[str] = []
    for s in suites:
        for tc in s.iter("testcase"):
            status = "ok"
            msg = ""
            for child in tc:
                if child.tag in ("failure", "error"):
                    status = child.tag
                    msg = (child.get("message") or "").strip().replace("\n", " ")[:220]
                    break
            if status != "ok":
                cls = tc.get("classname") or ""
                name = tc.get("name") or ""
                bad.append(f"- `{cls}::{name}` — {status}: {msg}")
    return {
        "file": path.name,
        "tests": int(total),
        "failures": int(fail),
        "errors": int(err),
        "skipped": int(skip),
        "bad": bad,
    }


def main() -> None:
    groups: list[dict] = []
    for xml in sorted(ART.rglob("*.xml")):
        groups.append(parse_junit(xml))

    ui_json = None
    for cand in sorted(ART.rglob("ui_audit.json")):
        try:
            ui_json = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        break

    t_total = sum(g.get("tests", 0) for g in groups)
    t_fail = sum(g.get("failures", 0) for g in groups)
    t_err = sum(g.get("errors", 0) for g in groups)
    t_skip = sum(g.get("skipped", 0) for g in groups)

    lines: list[str] = []
    lines.append("# F1OPT 云端全量审计报告")
    lines.append("")
    lines.append("由 GitHub Actions 云端执行：全量 pytest + 真人式 UI 点击审计。")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| 项目 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 单元测试用例 | {t_total} |")
    lines.append(f"| 失败 (failures) | {t_fail} |")
    lines.append(f"| 错误 (errors) | {t_err} |")
    lines.append(f"| 跳过 | {t_skip} |")
    if ui_json:
        lines.append(f"| UI 检查项 | {ui_json['total']} |")
        lines.append(f"| UI 通过 | {ui_json['passed']} |")
        lines.append(f"| UI 失败 | {ui_json['failed']} |")
    lines.append("")

    lines.append("## 测试分组明细")
    lines.append("")
    lines.append("| 分组 | 用例 | 失败 | 错误 | 跳过 |")
    lines.append("|---|---:|---:|---:|---:|")
    for g in groups:
        if "error" in g:
            lines.append(f"| {g['file']} | — | — | — | — | {g['error']} |")
            continue
        lines.append(
            f"| {g['file'].replace('.xml', '')} | {g['tests']} | {g['failures']} | "
            f"{g['errors']} | {g['skipped']} |"
        )
    lines.append("")

    any_bad = [g for g in groups if g.get("bad")]
    if any_bad:
        lines.append("## 失败/错误用例清单")
        lines.append("")
        for g in any_bad:
            lines.append(f"### {g['file'].replace('.xml', '')}")
            lines.append("")
            for item in g["bad"][:80]:
                lines.append(item)
            if len(g["bad"]) > 80:
                lines.append(f"- …（另有 {len(g['bad']) - 80} 条，见 artifact）")
            lines.append("")

    if ui_json:
        lines.append("## 真人式 UI 审计")
        lines.append("")
        lines.append("| 检查项 | 结果 | 详情 |")
        lines.append("|---|---|---|")
        for r in ui_json["results"]:
            lines.append(
                f"| {r['name']} | {'✅' if r['ok'] else '❌'} | "
                f"{(r['detail'] or '').replace('|', '/')[:200]} |"
            )
        lines.append("")
        if ui_json.get("page_errors"):
            lines.append("### 未捕获 JS 异常")
            lines.append("")
            lines.append("```")
            lines.extend(ui_json["page_errors"][:30])
            lines.append("```")
            lines.append("")
        if ui_json.get("console_errors"):
            lines.append("### console.error")
            lines.append("")
            lines.append("```")
            lines.extend(ui_json["console_errors"][:30])
            lines.append("```")
            lines.append("")

    (REPORTS / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    failures = t_fail + t_err + (ui_json["failed"] if ui_json else 0)
    title = f"云端全量审计：{'❌ 发现 ' + str(failures) + ' 项问题' if failures else '✅ 全绿'}"
    (REPORTS / "ISSUE_TITLE.txt").write_text(title, encoding="utf-8")

    body = "\n".join(lines)
    if len(body) > MAX_ISSUE:
        body = body[:MAX_ISSUE] + "\n\n…（正文已截断，完整报告见 artifact `cloud-audit-report`）"
    (REPORTS / "ISSUE_BODY.md").write_text(body, encoding="utf-8")

    print(f"report written: {len(body)} chars; failures={failures}", flush=True)


if __name__ == "__main__":
    main()
