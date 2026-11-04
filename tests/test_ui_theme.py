"""界面主题（themes.css）契约测试。

背景
----
v1.6.19 起界面提供 4 套可热切换主题（broadcast / pitwall / carbon / nightrace）。
主题只覆盖 CSS 变量，不碰布局与交互，因此最容易出的不是"崩"，而是**静默残缺**：

1. 新增主题忘了定义某个变量 → 该处回落到 style.css 的深色默认值，
   浅色主题里突然冒出一块深色面板，肉眼很难第一时间定位；
2. app.js 的 THEMES 与 themes.css 的 ``html[data-theme]`` 块、index.html
   首屏内联脚本三处 id 列表各写一遍，改一处忘一处 → 点了没反应或切了没效果；
3. 荧光/霓虹主色配浅色文字 → 按钮上的字看不清。

这组测试把上述三类问题变成提交前的显式失败。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_UI_DIR = Path(__file__).resolve().parent.parent / "setup_tuner" / "ui"
_INDEX = _UI_DIR / "index.html"
_APP_JS = _UI_DIR / "app.js"
_THEMES_CSS = _UI_DIR / "themes.css"
_STYLE_CSS = _UI_DIR / "style.css"

# 默认主题定义在 style.css 的 :root，不需要在 themes.css 重复
_DEFAULT_THEME = "broadcast"

# 每套非默认主题必须覆盖的变量。漏一个就意味着该处在浅色/异色主题下
# 会回落到 style.css 的深色默认值，出现"一块深色补丁"。
_REQUIRED_VARS = {
    "--bg-base",
    "--bg-card",
    "--bg-inset",
    "--bg-elevated",
    "--border",
    "--text",
    "--text-body",
    "--text-dim",
    "--accent",
    "--accent-gradient",
    "--on-accent",
    "--data",
    "--data-gradient",
    "--shadow",
    "--radius",
    "--topbar-bg",
    "--overlay-bg",
    "--toast-bg",
    "--app-bg-image",
    # 赛道图 SVG 内联注入页面 DOM，靠这些变量跟随主题换色
    "--track-bg",
    "--track-line",
    "--track-ring",
    "--track-label",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _theme_blocks() -> dict[str, str]:
    """解析 themes.css 中所有 ``html[data-theme="x"] { ... }`` 块。"""
    css = _read(_THEMES_CSS)
    return {
        name: body
        for name, body in re.findall(
            r'html\[data-theme="([\w-]+)"\]\s*\{(.*?)\n\}', css, re.S
        )
    }


def _block_vars(body: str) -> dict[str, str]:
    """从 CSS 块体中提取自定义属性（同名后者覆盖前者，与浏览器一致）。"""
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body))


def _js_theme_ids() -> list[str]:
    """从 app.js 的 THEMES 常量里取主题 id 顺序。"""
    js = _read(_APP_JS)
    block = re.search(r"const THEMES = \[(.*?)\n  \];", js, re.S)
    assert block, "app.js 中找不到 THEMES 常量"
    return re.findall(r'id:\s*"([\w-]+)"', block.group(1))


def _boot_theme_ids() -> list[str]:
    """从 index.html 首屏内联脚本里取主题 id 白名单。"""
    html = _read(_INDEX)
    m = re.search(
        r'if \(\["([\w", \-]+)"\]\.indexOf\(saved\)', html
    )
    assert m, "index.html 首屏脚本中找不到主题 id 白名单"
    return re.findall(r"[A-Za-z][\w-]*", m.group(1))


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise ValueError(f"不是 6 位十六进制颜色：{value!r}")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _relative_luminance(value: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = _hex_to_rgb(value)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(fg: str, bg: str) -> float:
    """WCAG 2.1 对比度（1.0 ~ 21.0）。"""
    l1, l2 = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


class TestThemeInventory:
    """三处主题 id 列表必须完全一致。"""

    def test_themes_css_defines_expected_versions(self) -> None:
        blocks = _theme_blocks()
        assert set(blocks) == {"pitwall", "carbon", "nightrace"}, (
            "themes.css 的主题块与预期不符：" + ", ".join(sorted(blocks))
        )

    def test_default_theme_is_not_redefined(self) -> None:
        """默认主题定义在 style.css :root，themes.css 重复定义会掩盖真实默认值。"""
        assert _DEFAULT_THEME not in _theme_blocks()

    def test_app_js_matches_css(self) -> None:
        js_ids = _js_theme_ids()
        assert js_ids[0] == _DEFAULT_THEME, "THEMES 第一项必须是默认主题"
        assert set(js_ids) == set(_theme_blocks()) | {_DEFAULT_THEME}

    def test_boot_script_matches_app_js(self) -> None:
        """首屏脚本白名单与 app.js 不一致 → 记住的主题在下次打开时被丢弃。"""
        assert sorted(_boot_theme_ids()) == sorted(_js_theme_ids())

    def test_every_theme_has_a_chip(self) -> None:
        html = _read(_INDEX)
        chips = re.findall(r'data-theme-set="([\w-]+)"', html)
        assert chips == _js_theme_ids(), (
            "顶部栏切换器与 THEMES 顺序/内容不一致：" + ", ".join(chips)
        )

    def test_default_theme_is_applied_in_markup(self) -> None:
        assert f'<html lang="zh-CN" data-theme="{_DEFAULT_THEME}">' in _read(_INDEX)

    def test_themes_css_is_linked_after_style(self) -> None:
        """顺序反了会被 style.css 的 :root 覆盖回去，切换看起来"没生效"。"""
        html = _read(_INDEX)
        i_style = html.index("/static/style.css")
        i_themes = html.index("/static/themes.css")
        assert i_style < i_themes


class TestThemeVariables:
    """每套主题的核心变量必须齐全，否则会出现深色补丁。"""

    @pytest.mark.parametrize("theme", ["pitwall", "carbon", "nightrace"])
    def test_all_required_vars_defined(self, theme: str) -> None:
        defined = set(_block_vars(_theme_blocks()[theme]))
        missing = sorted(_REQUIRED_VARS - defined)
        assert not missing, f"{theme} 缺少变量：" + ", ".join(missing)

    @pytest.mark.parametrize("theme", ["pitwall", "carbon", "nightrace"])
    def test_no_empty_values(self, theme: str) -> None:
        for name, value in _block_vars(_theme_blocks()[theme]).items():
            assert value.strip(), f"{theme} 的 {name} 为空值"

    def test_light_theme_declares_color_scheme(self) -> None:
        """浅色主题必须声明 light，否则原生控件（滚动条/下拉）仍是深色。"""
        assert "color-scheme: light" in _theme_blocks()["pitwall"]

    @pytest.mark.parametrize("theme", ["carbon", "nightrace"])
    def test_dark_themes_declare_color_scheme(self, theme: str) -> None:
        assert "color-scheme: dark" in _theme_blocks()[theme]


class TestThemeContrast:
    """配色可读性：文字看不见的主题等于没有主题。"""

    @pytest.mark.parametrize("theme", sorted(_theme_blocks()))
    def test_body_text_readable_on_card(self, theme: str) -> None:
        v = _block_vars(_theme_blocks()[theme])
        ratio = _contrast(v["--text"], v["--bg-card"])
        assert ratio >= 7.0, f"{theme} 正文对比度仅 {ratio:.2f}:1，低于 7:1"

    @pytest.mark.parametrize("theme", sorted(_theme_blocks()))
    def test_accent_label_readable(self, theme: str) -> None:
        """主色按钮上的文字至少达到 WCAG AA 大字号标准（3:1）。"""
        v = _block_vars(_theme_blocks()[theme])
        ratio = _contrast(v["--on-accent"], v["--accent"])
        assert ratio >= 3.0, (
            f"{theme} 主色按钮文字对比度仅 {ratio:.2f}:1（{v['--on-accent']} on {v['--accent']}）"
        )

    @pytest.mark.parametrize("theme", sorted(_theme_blocks()))
    def test_data_color_readable(self, theme: str) -> None:
        """遥测数值用 --data 色，必须能在卡片底色上读清。"""
        v = _block_vars(_theme_blocks()[theme])
        ratio = _contrast(v["--data"], v["--bg-card"])
        assert ratio >= 3.0, f"{theme} 数据色对比度仅 {ratio:.2f}:1"


class TestNoHardcodedThemeColors:
    """主色不能硬编码，否则换主题后按钮还是旧色。"""

    def test_accent_hover_literals_only_in_tokens(self) -> None:
        """旧 hover 色字面量只能留在 :root 的设计令牌里。

        回归点：``.btn-primary:hover`` 曾写死 ``linear-gradient(#FF3320, #FF5030)``，
        换主题后按钮会从荧光黄绿跳回红色 —— 这正是"换肤只换了一半"的典型残留。
        刹车进度条的红色是语义色（刹车就该红），不在本断言范围内。
        """
        css = _read(_STYLE_CSS)
        root_body_end = css.index("}", css.index(":root"))
        rules = css[root_body_end:]
        # 只查「旧的 hover 专用色」。#FF4020 仍出现在刹车进度条（语义红，
        # 换主题也必须保持红色），不能一起断言。
        for literal in ("#FF3320", "#FF5030"):
            assert literal not in rules, f"规则体里仍硬编码 {literal}，换主题会漏色"

    def test_theme_vars_exist_in_root(self) -> None:
        """主题块只能覆盖 :root 已声明的变量。

        拼错变量名（``--bg-crad``）不会报错，只会静默失效并回落到默认色 ——
        浅色主题里出现一块深色补丁，肉眼极难定位。
        """
        css = _read(_STYLE_CSS)
        root_block = css[css.index(":root"): css.index("}", css.index(":root"))]
        declared = set(re.findall(r"(--[\w-]+)\s*:", root_block))
        assert len(declared) > 50, "style.css :root 变量解析异常"
        unknown: list[str] = []
        for theme, body in _theme_blocks().items():
            for name in _block_vars(body):
                if name not in declared:
                    unknown.append(f"{theme}:{name}")
        assert not unknown, "主题覆盖了 :root 里不存在的变量：" + ", ".join(unknown)

    def test_primary_button_uses_theme_vars(self) -> None:
        css = _read(_STYLE_CSS)
        btn = re.search(r"\.btn-primary\s*\{(.*?)\}", css, re.S)
        assert btn, "style.css 缺少 .btn-primary 规则"
        body = btn.group(1)
        assert "var(--accent-gradient)" in body
        assert "var(--on-accent)" in body
