"""SVG 内嵌可见圆点必须与 ``_track_anchors`` 锚点一致（task-79 回归防线）。

背景：``ui/tracks/*.svg`` 的 ``<circle>/<text>`` 是用户在地图上实际看到的
层。曾与 ``_track_anchors.py`` 脱节——蒙扎 T1 的可见圆点仍画在起跑线
直道上，而点击热区已用新锚点移到减速弯，同一弯道出现两个错开的标记。
本测试确保两层永不脱节。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config
from setup_tuner.domain._track_anchors import TRACK_ANCHORS

TRACKS_DIR = Path(__file__).resolve().parents[1] / "setup_tuner" / "ui" / "tracks"

# 与 scripts/sync_svg_markers.py 的写入版式严格一致
CIRCLE_RE = re.compile(
    r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6" fill="none" '
    r'stroke="#3B9EFF" stroke-width="1\.5" opacity="0\.85"/>'
)
TEXT_RE = re.compile(
    r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="middle" fill="#9494a8" '
    r'font-size="9" font-family="monospace">(\d+)</text>'
)

# 锚点 1 位小数、SVG 2 位小数，允许舍入误差
TOLERANCE_PX = 0.51
TEXT_Y_OFFSET = 9.0  # 数字标在圆环上方 9px


@pytest.mark.parametrize("track_id", sorted(TRACK_ANCHORS))
def test_svg_visible_dots_match_anchors(track_id: str) -> None:
    svg_path = TRACKS_DIR / f"{track_id}.svg"
    assert svg_path.exists(), f"{track_id}: 缺少 SVG 文件"
    content = svg_path.read_text(encoding="utf-8")
    circles = CIRCLE_RE.findall(content)
    n_expect = len(TRACK_ANCHORS[track_id])
    assert len(circles) == n_expect, (
        f"{track_id}: SVG 圆点数 {len(circles)} != 锚点数 {n_expect}"
    )
    for i, (cx, cy) in enumerate(circles, start=1):
        ax, ay = TRACK_ANCHORS[track_id][i]
        assert abs(float(cx) - ax) <= TOLERANCE_PX, (
            f"{track_id}: 弯道 {i} 圆点 x={cx} 偏离锚点 {ax}"
        )
        assert abs(float(cy) - ay) <= TOLERANCE_PX, (
            f"{track_id}: 弯道 {i} 圆点 y={cy} 偏离锚点 {ay}"
        )


@pytest.mark.parametrize("track_id", sorted(TRACK_ANCHORS))
def test_svg_number_labels_follow_dots(track_id: str) -> None:
    content = (TRACKS_DIR / f"{track_id}.svg").read_text(encoding="utf-8")
    texts = TEXT_RE.findall(content)
    n_expect = len(TRACK_ANCHORS[track_id])
    assert len(texts) == n_expect, (
        f"{track_id}: SVG 标签数 {len(texts)} != 锚点数 {n_expect}"
    )
    for i, (tx, ty, num) in enumerate(texts, start=1):
        assert int(num) == i, f"{track_id}: 第 {i} 个标签序号是 {num}"
        ax, ay = TRACK_ANCHORS[track_id][i]
        assert abs(float(tx) - ax) <= TOLERANCE_PX, (
            f"{track_id}: 弯道 {i} 标签 x={tx} 偏离锚点 {ax}"
        )
        assert abs(float(ty) - (ay - TEXT_Y_OFFSET)) <= TOLERANCE_PX, (
            f"{track_id}: 弯道 {i} 标签 y={ty} 偏离锚点 {ay - TEXT_Y_OFFSET}"
        )


class TestStaticCacheRevalidation:
    """静态资源必须带 ``Cache-Control: no-cache``（task-79）。

    否则浏览器按 Last-Modified 启发式缓存，代码更新后用户仍看到旧
    SVG/app.js（蒙扎 T1 修完、地图不变假的根源之一）。
    """

    def test_track_svg_served_with_no_cache(self, tmp_path: Path) -> None:
        app = create_app(Config(data_dir=str(tmp_path / "data")))
        with TestClient(app) as client:
            resp = client.get("/static/tracks/monza.svg")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"

    def test_app_js_served_with_no_cache(self, tmp_path: Path) -> None:
        app = create_app(Config(data_dir=str(tmp_path / "data")))
        with TestClient(app) as client:
            resp = client.get("/static/app.js")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"

    def test_served_svg_carries_synced_markers(self, tmp_path: Path) -> None:
        """服务端实际返回的 SVG 内容也必须是新锚点（防服务层缓存旧文件）。"""
        app = create_app(Config(data_dir=str(tmp_path / "data")))
        with TestClient(app) as client:
            resp = client.get("/static/tracks/monza.svg")
        circles = CIRCLE_RE.findall(resp.text)
        ax, ay = TRACK_ANCHORS["monza"][1]
        assert circles, "服务端 SVG 未找到圆点标记"
        assert abs(float(circles[0][0]) - ax) <= TOLERANCE_PX
        assert abs(float(circles[0][1]) - ay) <= TOLERANCE_PX
