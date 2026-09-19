"""持续优化闭环：跑一圈 → 反馈 → 出优化调教 → 再跑"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys_path = str(REPO)
import sys

sys.path.insert(0, sys_path)

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.engine.holistic import class_weighted_dx, holistic_coherence
from setup_tuner.engine.optimizer import optimize_setup


def demo_loop(track_id="monza", current_setup=None, feedbacks=None):
    if current_setup is None:
        current_setup = {f.name: f.default for f in ALL_SETUP_FIELDS}
    if feedbacks is None:
        feedbacks = [{"corner_number":1,"symptom":"understeer","strength":3,"category":"global"}]
    # 1. 逐弯加权 Dx
    cdx = class_weighted_dx(feedbacks, track_id)
    # 2. 规则引擎初步 delta
    from setup_tuner.engine.diagnostic import compute_dx
    from setup_tuner.engine.engine import compute_setup_delta
    symptoms = [(fb["symptom"], fb["strength"]) for fb in feedbacks]
    dx0 = compute_dx(symptoms)
    init_delta = compute_setup_delta(dx0, current_setup)
    # 3. 圈级优化
    res = optimize_setup(cdx.dx, current_setup, track_id, feedbacks=feedbacks, initial_delta=init_delta, needs_by_class=cdx.by_class)
    # 4. 整体收口
    closed, notes = holistic_coherence(res.delta, cdx.dx, cdx.demand)
    print("=== 优化结果 ===")
    print(json.dumps(closed, ensure_ascii=False, indent=2))
    print("\n权衡说明:")
    for n in notes:
        print("- ", n)
    return closed

if __name__ == "__main__":
    demo_loop()
