"""模型证据链测试：训练出来的模型必须**真的**作用在每一条赛道上。

为什么需要这组测试
------------------
现行测试（``tests/test_setup_sim.py``）全部用合成赛道（``alpha`` / ``beta``），
只能证明"模型加载逻辑正确"，无法发现真实模型**只覆盖 17 条赛道**这一事实。

后果是静默的：``sim_optimizer.sim_refine`` 对未覆盖赛道返回
``available=False``，UI 只弹一个 toast，报告的 ``model_type`` 变成 ``rule``
—— 用户在 Jeddah / Austin / Mexico / São Paulo / Las Vegas / Lusail /
Yas Marina 这 7 条赛道上，拿到的其实是纯规则结果，却以为模型在跑。

本组测试把这条链钉死：
    1. 模型覆盖必须 = 24/24（赛历全量）；
    2. 默认调教下模型预测必须接近 0（锚点自洽，证明推理口径没漂）；
    3. UI 页脚的数量/版本必须与代码常量一致（防再次过期）。

逃生开关
--------
覆盖不足默认记为 **xfail**，不判 FAIL：模型是否已重训取决于 CI 执行顺序，
通用 CI 不该拿历史模型判死。真正的硬门禁放在 ``retrain-models.yml`` 的
``Coverage guard (24/24)`` —— 那是重训产物的出口，覆盖不全直接拒绝提交。

重训完成后，设 ``F1OPT_STRICT_COVERAGE=1`` 可把本测试转为强制失败。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.domain.track import ALL_TRACKS
from setup_tuner.engine.setup_sim import get_setup_sim

ROOT = Path(__file__).resolve().parents[1]
#: 严格模式：覆盖不足判 FAIL（默认 xfail，见模块文档）
STRICT_COVERAGE = os.environ.get("F1OPT_STRICT_COVERAGE") == "1"


def test_model_covers_every_calendar_track() -> None:
    """训练后的模型必须覆盖全部 24 条赛道，缺一条就是一条赛道静默失效。"""
    model = get_setup_sim()
    if not model.available:
        pytest.skip(f"模型不可用（{model.reason}），先跑 retrain-models 工作流")

    expected = {t.track_id for t in ALL_TRACKS}
    missing = sorted(expected - set(model.track_ids))
    message = (
        f"模型缺失 {len(missing)} 条赛道：{missing}。"
        f"这些赛道上 sim_refine 会返回 available=False，"
        f"用户拿到的是纯规则结果（报告 model_type=rule）。"
        f"修复：scripts/build_setup_sim_dataset.py --synthesize-missing 后重训。"
    )
    if missing and not STRICT_COVERAGE:
        pytest.xfail(
            f"覆盖未达标（重训前预期，设 F1OPT_STRICT_COVERAGE=1 转为失败）：{message}"
        )
    assert not missing, message


def test_model_reports_anchor_provenance() -> None:
    """锚点来源必须可查：真实遥测与几何迁移要能区分，不能冒充遥测。"""
    model = get_setup_sim()
    if not model.available:
        pytest.skip(f"模型不可用（{model.reason}）")

    # 旧模型没有该字段 → 视为全部来自遥测（向后兼容，不失败）
    sources = model.anchor_sources
    for tid in model.track_ids:
        value = sources.get(tid, "telemetry")
        assert value in ("telemetry", "geometry_transfer"), (
            f"赛道 {tid} 的 anchor_source 非法：{value!r}"
        )

    # describe() 必须把迁移数量说出来（报告里直接展示给用户）
    text = model.describe()
    transferred = sum(1 for v in sources.values() if v != "telemetry")
    if transferred:
        assert f"{transferred} 条为几何迁移" in text, (
            f"describe() 未如实标注迁移赛道：{text}"
        )


@pytest.mark.parametrize("track_id", [t.track_id for t in ALL_TRACKS])
def test_default_setup_predicts_near_zero(track_id: str) -> None:
    """默认调教（全 0 归一化）是仿真锚点基线，模型预测必须接近 0。

    这条能抓出"训练/推理归一化口径漂移"——一旦特征构建与训练不一致，
    默认调教的预测会系统性偏离 0，进而让模拟优化整体偏置。
    """
    model = get_setup_sim()
    if not model.available:
        pytest.skip("模型不可用")
    if not model.covers(track_id):
        pytest.skip(f"{track_id} 不在模型覆盖范围")

    pred = model.predict_delta_s(track_id, {f.name: 0.0 for f in ALL_SETUP_FIELDS})
    assert pred is not None, f"{track_id} 覆盖判定与推理不一致"
    # 允许模型拟合误差量级（MAE ~0.1s，留 3 倍余量）
    assert abs(pred) < 0.35, (
        f"{track_id} 默认调教预测 {pred * 1000:.1f} ms 偏离 0 过多，"
        f"疑似归一化口径漂移"
    )


def test_ui_footer_matches_code_constants() -> None:
    """UI 页脚的版本/参数数/症状数必须与代码真实常量一致。"""
    from setup_tuner import __version__

    html = (ROOT / "setup_tuner" / "ui" / "index.html").read_text(encoding="utf-8")
    footer = re.search(r'<footer.*?</footer>', html, re.S)
    assert footer is not None, "index.html 缺少 footer 区块"
    block = footer.group(0)

    assert f"v{__version__}" in block, (
        f"页脚版本与 setup_tuner.__version__（{__version__}）不一致：{block}"
    )
    assert f"{len(ALL_SETUP_FIELDS)} 调教参数" in block, (
        f"页脚参数数与 ALL_SETUP_FIELDS（{len(ALL_SETUP_FIELDS)}）不一致"
    )
    assert f"{len(list(Symptom))} 症状模型" in block, (
        f"页脚症状数与 Symptom 枚举（{len(list(Symptom))}）不一致"
    )
    assert "F1 2026" in block, "页脚数据基准应标注 F1 2026"
