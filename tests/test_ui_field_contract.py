"""调教参数面板「6 大类分组」的前后端字段契约。

回归背景
--------
``/api/v1/setup/fields`` 返回的类别字段叫 ``category``（中文类别名：空气动力学 /
变速箱 / 悬挂几何 / 悬挂 / 刹车 / 轮胎）。某一版把后端的 ``group`` 改名成
``category`` 时，前端 ``loadSetupFields`` 还在读 ``f.group`` —— 结果 20 项参数
的 ``group`` 全是 ``undefined``，``groupSetupFieldsByGroup`` 把它们塞进同一个
``"undefined"`` 键，面板顶部只剩一个「UNDEFINED 20」页签，6 大类分组静默失效。

这类 bug 不报错、不崩溃，只是"看起来能用的残废界面"，靠肉眼很难在改动时发现。
这里把「后端给了什么字段」和「前端读了什么字段」钉在一起。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config

_ROOT = Path(__file__).resolve().parents[1]
_APP_JS = _ROOT / "setup_tuner" / "ui" / "app.js"

# 后端 /setup/fields 应至少分出 5 个类别（当前 6 类）
_MIN_CATEGORIES = 5


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(Config(data_dir=str(tmp_path / "data")))
    with TestClient(app) as c:
        yield c


def _fields(client: TestClient) -> list[dict]:
    resp = client.get("/api/v1/setup/fields")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    data = body["data"]
    fields = data if isinstance(data, list) else data.get("items", [])
    assert fields, "setup/fields 返回空列表"
    return fields


class TestSetupFieldsContract:
    def test_every_field_has_non_empty_category(self, client: TestClient) -> None:
        for f in _fields(client):
            assert f.get("category"), f"{f.get('name')} 缺少 category"
            assert isinstance(f["category"], str)

    def test_categories_cover_multiple_groups(self, client: TestClient) -> None:
        cats = {f["category"] for f in _fields(client)}
        assert len(cats) >= _MIN_CATEGORIES, (
            f"只有 {len(cats)} 个类别，参数面板的 6 大类分组会塌缩：{sorted(cats)}"
        )

    def test_all_20_parameters_present(self, client: TestClient) -> None:
        assert len(_fields(client)) == 20


class TestFrontendReadsCategory:
    """前端不能再把 group 当作唯一来源。"""

    def test_mapping_falls_back_to_category(self) -> None:
        js = _APP_JS.read_text(encoding="utf-8")
        block = re.search(
            r"state\.setupFields = list\.map\(\(f\) => \(\{(.*?)\}\)\)", js, re.S
        )
        assert block, "app.js 中找不到 setupFields 映射块"
        body = block.group(1)
        assert "f.category" in body, (
            "app.js 的 setupFields 映射没有读 f.category —— "
            "后端字段改名后分组会再次塌缩成 UNDEFINED"
        )

    def test_group_falls_back_not_only_group(self) -> None:
        js = _APP_JS.read_text(encoding="utf-8")
        assert not re.search(r"^\s*group:\s*f\.group,\s*$", js, re.M), (
            "app.js 仍以 f.group 作为分组的唯一来源"
        )
