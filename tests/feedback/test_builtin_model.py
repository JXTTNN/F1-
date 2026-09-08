"""内置小模型 (builtin tiny model) 单元 + 引擎集成测试。

覆盖: 意图映射 / 数值钳制 / 在线学习 / 持久化往返 / 引擎全链路
(preload → run → summary) / 流式 / 默认后端离线可用。
"""

from __future__ import annotations

import pytest

from f1opt.config import Settings
from f1opt.data.setup_schema import ALL_SETUP_FIELDS, DEFAULT_SETUP
from f1opt.feedback import builtin_model as bm
from f1opt.feedback.builtin_model import (
    BUILTIN_MARKER,
    BuiltinTinyModel,
    get_builtin_model,
    reset_builtin_model_cache,
)
from f1opt.feedback.engine import FeedbackEngine, llm_enhance, llm_enhance_stream
from f1opt.feedback.engine import TokenUsageTracker


@pytest.fixture()
def model(tmp_path) -> BuiltinTinyModel:
    reset_builtin_model_cache()
    return BuiltinTinyModel(tmp_path / "data_store")


def _settings(tmp_path) -> Settings:
    return Settings(llm_backend="builtin", data_dir=str(tmp_path / "ds"))


def _spec(field: str):
    for s in ALL_SETUP_FIELDS():
        if s.name == field:
            return s
    raise AssertionError(f"unknown field {field}")


class TestSuggest:
    def test_understeer_produces_marker_and_adjustments(self, model) -> None:
        out = model.suggest("T1 老是推头怎么调？", "shanghai")
        assert out["summary"].startswith(BUILTIN_MARKER)
        assert out["sub_intent"] == "understeer"
        fields = [a["field"] for a in out["adjustments"]]
        assert "front_wing" in fields
        fw = next(a for a in out["adjustments"] if a["field"] == "front_wing")
        assert fw["delta"] < 0  # 推头 → 减前翼
        assert "推头" in out["summary"]

    def test_adjustments_clamped_to_spec(self, model) -> None:
        setup = DEFAULT_SETUP.model_dump()
        out = model.suggest("弯中严重甩尾", "monza", setup=setup)
        for adj in out["adjustments"]:
            spec = _spec(adj["field"])
            assert spec.min <= adj["to"] <= spec.max
            assert adj["to"] == adj["from"] + adj["delta"]

    def test_collect_grows_samples_and_confidence(self, model) -> None:
        first = model.suggest("推头怎么办", "suzuka")
        n1 = first["samples_same_context"]
        second = model.suggest("又推头了", "suzuka")
        assert second["samples_same_context"] == n1 + 1
        assert second["confidence"] > first["confidence"]

    def test_unknown_intent_general_no_crash(self, model) -> None:
        out = model.suggest("", "shanghai")
        assert out["summary"].startswith(BUILTIN_MARKER)


class TestLearning:
    def test_learn_outcome_updates_weight(self, model) -> None:
        base = model.suggest("推头", "spa")
        fw = next(a for a in base["adjustments"] if a["field"] == "front_wing")
        d1 = fw["delta"]
        for _ in range(5):
            model.learn_outcome("spa", "understeer", "front_wing", improved=True)
        better = model.suggest("推头", "spa")
        fw2 = next(a for a in better["adjustments"] if a["field"] == "front_wing")
        assert fw2["delta"] <= d1  # 改善 → 同向加权 (幅度不减小)
        for _ in range(20):
            model.learn_outcome("spa", "understeer", "front_wing", improved=False)
        worse = model.suggest("推头", "spa")
        fw3 = next(a for a in worse["adjustments"] if a["field"] == "front_wing")
        assert abs(fw3["delta"]) <= abs(d1)  # 恶化 → 减弱 (不低于 0.5x)

    def test_persistence_roundtrip(self, tmp_path) -> None:
        ds = tmp_path / "ds"
        m1 = BuiltinTinyModel(ds)
        m1.suggest("推头怎么调", "shanghai")
        m1.suggest("推头怎么调", "shanghai")
        m2 = BuiltinTinyModel(ds)  # 新实例从盘上恢复
        assert m2.stats()["samples_total"] == 2
        out = m2.suggest("又推头", "shanghai")
        assert out["samples_same_context"] >= 2


class TestEngineIntegration:
    def setup_method(self) -> None:
        reset_builtin_model_cache()

    def test_preload_then_run_uses_builtin(self, tmp_path) -> None:
        engine = FeedbackEngine(config=_settings(tmp_path))
        pre = engine.preload_llm()
        assert pre["loaded"] is True
        assert pre["backend"] == "builtin"
        out = engine.run(
            [], DEFAULT_SETUP.model_dump(), "shanghai", question="高速弯推头怎么调？"
        )
        assert out["summary"].startswith(BUILTIN_MARKER)
        assert out["builtin_adjustments"], "缺少结构化修正"
        assert out["builtin_meta"]["samples_same_context"] >= 1
        engine.unload_llm()

    def test_stream_builtin_yields_summary(self, tmp_path) -> None:
        cfg = _settings(tmp_path)
        deltas = list(
            llm_enhance_stream(
                {"summary": "规则", "dimensions": [], "sources": []},
                "推头怎么办",
                cfg,
                track_id="shanghai",
            )
        )
        assert deltas
        assert "".join(deltas).startswith(BUILTIN_MARKER)

    def test_llm_enhance_direct_no_network(self, tmp_path) -> None:
        out = llm_enhance(
            {"summary": "规则", "dimensions": [], "sources": []},
            "刹车老锁死",
            _settings(tmp_path),
            track_id="monza",
            tracker=TokenUsageTracker(),
        )
        assert out["summary"].startswith(BUILTIN_MARKER)
        assert out["builtin_meta"]["sub_intent"] in ("brake", "general")

    def test_default_backend_is_builtin(self) -> None:
        assert Settings().llm_backend == "builtin"


class TestSingleton:
    def test_get_builtin_model_cached(self, tmp_path) -> None:
        reset_builtin_model_cache()
        m1 = get_builtin_model(tmp_path)
        m2 = get_builtin_model(tmp_path)
        assert m1 is m2
        reset_builtin_model_cache()
