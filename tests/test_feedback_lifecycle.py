"""反馈生命周期回归测试 —— 生成后"消费"反馈，跨圈不串味。

背景（2026-09-19 用户反馈）：
    "前一圈的车手反馈在点击生成优化后调教要清除，要不然杂着后一圈吗"

``/suggest`` 从 SQLite 读取**累积的**反馈记录。跨圈场景下：第 1 圈录入反馈
F1 → 生成 → 第 2 圈录入 F2 → 生成。若 F1 不清除，第 2 次的优化会同时受
F1 + F2 影响 —— 而 F1 针对的问题应已被上一轮调教修正/替换，混入即是噪声。

修复：生成成功后默认清除本赛道反馈（``clear_feedback_after_suggest``，
默认 True；落库失败时不清除，反馈保留供重试）。设 False 保留累积行为
（连续重新生成、性能基准等场景）。

本文件的用例互为对照（天然的负向验证）：
- 默认路径：生成后反馈消失、第二次生成 400（无输入可用）；
- 关闭路径：反馈保留、第二次生成 200（旧累积语义）；
- 跨圈纯净性：两轮反馈不同时，第二轮结果 == 全新环境仅含第二轮反馈的结果
  （第一轮反馈完全未参与）——清除若失效，该断言必红。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config


def _make_client(tmp_path: Path) -> TestClient:
    """隔离数据目录创建客户端（禁止裸 create_app()，见项目测试硬约定）。"""
    app = create_app(Config(data_dir=str(tmp_path / "data")))
    return TestClient(app)


def _submit(client: TestClient, track: str, corner: int, symptom: str,
            strength: int = 3) -> None:
    resp = client.post(
        "/api/v1/feedback",
        json={
            "track_id": track, "corner_number": corner,
            "symptom": symptom, "strength": strength,
        },
    )
    assert resp.status_code == 200, resp.text


def _suggest(client: TestClient, track: str, **extra) -> dict:
    resp = client.post("/api/v1/suggest", json={"track_id": track, **extra})
    return {"status": resp.status_code, "body": resp.json()}


def _feedback_count(client: TestClient, track: str) -> int:
    resp = client.get("/api/v1/feedback", params={"track_id": track})
    assert resp.status_code == 200, resp.text
    return len(resp.json()["data"])


# ===========================================================================
# 1. API 默认行为：生成成功后清除
# ===========================================================================
class TestSuggestConsumesFeedback:
    """默认 clear_feedback_after_suggest=True：生成即"消费"。"""

    def test_default_clears_feedback_and_blocks_regenerate(
        self, tmp_path: Path,
    ) -> None:
        """生成成功后反馈被清除；无新反馈再次生成 → 400（不是无声复用）。"""
        with _make_client(tmp_path) as client:
            _submit(client, "suzuka", 1, "understeer")
            assert _feedback_count(client, "suzuka") == 1

            first = _suggest(client, "suzuka")
            assert first["status"] == 200, first
            # 响应如实告知清了多少条
            assert "已清除本赛道 1 条反馈" in first["body"]["message"]

            # 生成后：反馈已清空
            assert _feedback_count(client, "suzuka") == 0

            # 无新反馈再次生成 → 400（引导重新录入），而不是复用已被消费的输入
            second = _suggest(client, "suzuka")
            assert second["status"] == 400
            assert "反馈" in second["body"]["message"]

    def test_disabled_keeps_accumulation_semantics(self, tmp_path: Path) -> None:
        """clear_feedback_after_suggest=False：反馈保留，可连续重新生成。"""
        with _make_client(tmp_path) as client:
            _submit(client, "suzuka", 1, "understeer")

            first = _suggest(client, "suzuka", clear_feedback_after_suggest=False)
            assert first["status"] == 200, first
            assert "已清除" not in first["body"]["message"]

            # 反馈保留
            assert _feedback_count(client, "suzuka") == 1

            # 第二次生成仍可用（旧累积语义）
            second = _suggest(client, "suzuka", clear_feedback_after_suggest=False)
            assert second["status"] == 200, second

    def test_next_round_only_sees_its_own_feedback(self, tmp_path: Path) -> None:
        """跨圈纯净性：第二轮结果不受第一轮反馈影响（本文件的核心断言）。

        对照实验：
          A（真实两圈）：F1 生成 → F2 生成 → delta_A
          B（干净环境）：F2 生成 → delta_B
        两者必须逐位一致 —— 即第一轮反馈 F1 对第二轮**零贡献**。
        若"生成后清除"失效，delta_A 会额外携带 F1 的影响，断言变红。
        """
        with _make_client(tmp_path / "a") as client_a:
            _submit(client_a, "monza", 1, "understeer")          # 第 1 圈：推头
            assert _suggest(client_a, "monza")["status"] == 200  # 生成并消费 F1
            _submit(client_a, "monza", 5, "exit_oversteer")      # 第 2 圈：出弯甩尾
            delta_a = _suggest(client_a, "monza")["body"]["data"]["report"]["setup_delta"]

        with _make_client(tmp_path / "b") as client_b:
            _submit(client_b, "monza", 5, "exit_oversteer")      # 只有第 2 圈反馈
            delta_b = _suggest(client_b, "monza")["body"]["data"]["report"]["setup_delta"]

        assert delta_a == delta_b, (
            "第二轮生成混入了第一轮反馈的影响（跨圈串味）："
            f"A={delta_a} B={delta_b}"
        )

    def test_feedback_lifecycle_is_per_track(self, tmp_path: Path) -> None:
        """清除只作用于目标赛道，其他赛道反馈不受影响。"""
        with _make_client(tmp_path) as client:
            _submit(client, "suzuka", 1, "understeer")
            _submit(client, "monza", 1, "oversteer")

            assert _suggest(client, "suzuka")["status"] == 200

            assert _feedback_count(client, "suzuka") == 0
            assert _feedback_count(client, "monza") == 1  # 另一赛道原样保留


# ===========================================================================
# 2. Store 层：clear_track_feedback
# ===========================================================================
class TestStoreClearFeedback:
    """``Store.clear_track_feedback`` 的返回值与作用域。"""

    def test_returns_deleted_count(self, tmp_path: Path) -> None:
        from setup_tuner.db.store import Store

        store = Store(str(tmp_path / "t.db"))
        try:
            store.add_feedback("suzuka", 1, "understeer", "entry", 3)
            store.add_feedback("suzuka", 2, "oversteer", "exit", 2)
            store.add_feedback("monza", 1, "understeer", "entry", 3)

            assert store.clear_track_feedback("suzuka") == 2
            # 幂等：再清一次返回 0（不是报错）
            assert store.clear_track_feedback("suzuka") == 0
            assert store.has_feedback("suzuka") is False
            assert store.has_feedback("monza") is True
        finally:
            store.close()

    def test_empty_track_returns_zero(self, tmp_path: Path) -> None:
        from setup_tuner.db.store import Store

        store = Store(str(tmp_path / "t.db"))
        try:
            assert store.clear_track_feedback("ghost_track") == 0
        finally:
            store.close()


# ===========================================================================
# 3. 边界：遥测-only 模式（无反馈）不误伤
# ===========================================================================
class TestLifecycleEdgeCases:
    """无反馈可清时的行为。"""

    def test_no_feedback_nothing_to_clear(self, tmp_path: Path) -> None:
        """零反馈请求（400 闸门）路径不触发清除，也不报 500。"""
        with _make_client(tmp_path) as client:
            resp = _suggest(client, "suzuka")
            assert resp["status"] == 400
            # 清除逻辑不允许把正常引导流程打成 500
            assert resp["body"]["code"] != 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
