"""界面布局变体的契约测试（4 套骨架）。

布局很容易「悄悄坏掉」：改个卡片 class 名、新加一张卡忘了归栏、主题文件里
手滑写了个不存在的布局 id —— 页面不会报错，只是某张卡在某个布局下凭空消失。
这里把三条最容易断裂的约定钉住：

1. **三处 id 列表一致**：index.html 的切换器、首屏内联脚本、app.js 的 LAYOUTS；
2. **每套布局都摆满全部卡片**：不多不少、不重复 —— 少一张就是「换布局后
   某个面板不见了」；
3. **layouts.css 只引用真实布局**：写了不存在的 id 等于死代码，而 classic
   是默认布局，不该有任何重排规则（否则「不选布局」也会被改）。
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_UI = _ROOT / "setup_tuner" / "ui"
_INDEX = _UI / "index.html"
_APP = _UI / "app.js"
_LAYOUTS_CSS = _UI / "layouts.css"

# 7 张可重排的卡片
_EXPECTED_CARDS = {
    "selector",
    "map",
    "telemetry",
    "setup",
    "feed",
    "report",
    "history",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _layout_defs() -> dict[str, dict[str, list[str]]]:
    """解析 app.js 里的 LAYOUTS 定义 → {布局 id: {left: [...], mid: [...], right: [...]}}。"""
    src = _read(_APP)
    start = src.index("const LAYOUTS = [")
    end = src.index("\n  ];", start)
    body = src[start:end]
    out: dict[str, dict[str, list[str]]] = {}
    for chunk in re.split(r"\n    \{", body)[1:]:
        lid = re.search(r'id: "([\w-]+)"', chunk)
        if not lid:
            continue
        cols: dict[str, list[str]] = {}
        for col in ("left", "mid", "right"):
            m = re.search(col + r": \[(.*?)\]", chunk, re.S)
            cols[col] = re.findall(r'"(\w+)"', m.group(1)) if m else []
        out[lid.group(1)] = cols
    return out


def _layout_js_ids() -> list[str]:
    """LAYOUTS 的 id 顺序（切换器按钮顺序必须跟它一致）。"""
    src = _read(_APP)
    start = src.index("const LAYOUTS = [")
    end = src.index("\n  ];", start)
    return re.findall(r'id: "([\w-]+)"', src[start:end])


def _layout_cards() -> dict[str, str]:
    """LAYOUT_CARDS：卡片短名 → CSS 类名。"""
    src = _read(_APP)
    start = src.index("const LAYOUT_CARDS = {")
    end = src.index("};", start)
    return dict(re.findall(r'(\w+): "\.([\w-]+)"', src[start:end]))


def _chip_ids() -> list[str]:
    return re.findall(r'data-layout-set="([\w-]+)"', _read(_INDEX))


def _boot_ids() -> list[str]:
    """首屏内联脚本里允许写入 data-layout 的白名单。"""
    html = _read(_INDEX)
    m = re.search(
        r'var layout = localStorage\.getItem\("f1opt\.ui\.layout"\);.*?\[(.*?)\]', html, re.S
    )
    assert m, "index.html 首屏脚本里没找到布局白名单"
    return re.findall(r'"([\w-]+)"', m.group(1))


def _css_layout_ids() -> set[str]:
    return set(re.findall(r'data-layout="([\w-]+)"', _read(_LAYOUTS_CSS)))


class TestLayoutDefinitions:
    def test_exactly_four_layouts(self) -> None:
        assert len(_layout_defs()) == 4, "布局数量变了，文档与切换器都要同步"

    def test_default_is_classic(self) -> None:
        assert 'data-layout="classic"' in _read(_INDEX)
        assert 'const DEFAULT_LAYOUT = "classic";' in _read(_APP)

    def test_every_layout_holds_every_card_exactly_once(self) -> None:
        """少一张 = 换布局后那个面板凭空消失；重复 = 同一张卡被摆两次。"""
        for lid, cols in _layout_defs().items():
            placed: list[str] = []
            for names in cols.values():
                placed.extend(names)
            assert len(placed) == len(set(placed)), f"{lid} 有卡片被重复摆放：{placed}"
            assert set(placed) == _EXPECTED_CARDS, (
                f"{lid} 少了 {sorted(_EXPECTED_CARDS - set(placed))}，"
                f"多了 {sorted(set(placed) - _EXPECTED_CARDS)}"
            )

    def test_card_keys_match_selectors(self) -> None:
        assert set(_layout_cards()) == _EXPECTED_CARDS

    def test_every_card_selector_exists_in_html(self) -> None:
        html = _read(_INDEX)
        for key, cls in _layout_cards().items():
            # class 属性通常还带 .card 等其它类名，按「词」匹配
            assert re.search(r'class="[^"]*\b' + re.escape(cls) + r"\b", html), (
                f"布局里的卡片 {key} 指向 .{cls}，但 index.html 里没有这个 class"
            )


class TestLayoutWiring:
    def test_chips_match_definitions(self) -> None:
        assert _chip_ids() == _layout_js_ids()

    def test_boot_script_whitelist_matches(self) -> None:
        """首屏脚本的白名单若漏一个 id，刷新后那个布局就回落到默认。"""
        assert _boot_ids() == _layout_js_ids()

    def test_layouts_css_loaded_after_style(self) -> None:
        html = _read(_INDEX)
        i_style = html.index("/static/style.css")
        i_layouts = html.index("/static/layouts.css")
        assert i_layouts > i_style, "layouts.css 必须在 style.css 之后加载才能覆盖栏宽"


class TestLayoutsCss:
    def test_only_real_layout_ids(self) -> None:
        known = set(_layout_js_ids())
        unknown = _css_layout_ids() - known
        assert not unknown, f"layouts.css 引用了不存在的布局：{sorted(unknown)}"

    def test_classic_has_no_rules(self) -> None:
        """classic 是默认布局，样式必须完全由 style.css 决定。

        一旦 layouts.css 里出现 ``html[data-layout="classic"]`` 的规则，
        「没选布局」就不再等于「原来的页面」，改动风险会从零变成不可控。
        （``:not([data-layout="classic"])`` 是给非默认布局用的，不算。）
        """
        css = _read(_LAYOUTS_CSS)
        assert 'html[data-layout="classic"]' not in css
        assert ':is([data-layout="classic"]' not in css

    def test_empty_column_collapsed(self) -> None:
        """workbench 只用两栏，空的中间栏必须收起，否则会占掉一列宽度。"""
        css = _read(_LAYOUTS_CSS)
        assert ".is-empty" in css
        assert "display: none" in css

    def test_each_layout_sets_columns(self) -> None:
        """每套非默认布局都得自己定义栅格列数，否则会沿用 classic 的三栏。"""
        css = _read(_LAYOUTS_CSS)
        for lid in _layout_js_ids():
            if lid == "classic":
                continue
            assert f'html[data-layout="{lid}"] .layout' in css, f"{lid} 没有定义 .layout 栏宽"
