"""一键重训流水线：遥测 → 逐弯样本 → 代理模型 → 调教仿真 → 调教性能 NN。

按顺序跑完五步（任一失败即整体失败，不静默跳过）：

1. ``build_corner_dataset.py``       外部 2026 遥测 → 逐弯样本
2. ``merge_features.py``             外部基准 × 本地样本 → 训练特征
3. ``train_telemetry_surrogate.py``  弯速代理模型（逐弯重要度 + 残差诊断）
4. ``build_setup_sim_dataset.py``    遥测锚定仿真 → 「调教 → 圈速增量」样本
5. ``train_setup_sim_nn.py``         调教性能 NN（**纯标准库，无 PyTorch**）

引擎侧对应两个模型：
- ``setup_tuner/resources/models/telemetry_surrogate.json``：逐弯重要度 / 车手未反馈问题的自动发现
  （``engine.surrogate`` → ``telemetry_diagnosis``）；
- ``setup_tuner/resources/models/setup_sim_nn.json``：调教优化的模型驱动环节
  （``engine.setup_sim`` + ``engine.sim_optimizer``，「矩阵给方向 + 神经网不断模拟优化」）。

跑完即生效，无需改代码。快速路径：``--skip-sim-nn`` 只重训 1~3 步。

用法::

    python scripts/retrain_all.py                  # 默认参数（含调教性能 NN）
    python scripts/retrain_all.py --max-laps 200 --ridge-samples 40000
    python scripts/retrain_all.py --skip-sim-nn    # 只重训代理模型
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(step: str, args: list[str]) -> None:
    print(f"\n{'=' * 72}\n[{step}] {' '.join(args)}\n{'=' * 72}", flush=True)
    t0 = time.time()
    proc = subprocess.run(args, cwd=str(ROOT), check=False)  # noqa: S603
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{step} 失败（退出码 {proc.returncode}）")
    print(f"[{step}] 完成，用时 {dt:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="重训遥测模型全流程（代理模型 + 调教性能 NN）")
    ap.add_argument("--max-laps", type=int, default=120,
                    help="每赛道每会话最多取多少圈（透传给 build_corner_dataset）")
    ap.add_argument("--ridge-samples", type=int, default=25000)
    ap.add_argument("--mlp-samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--sim-setups", type=int, default=1400,
                    help="调教仿真数据集每赛道采样调教数（透传给 build_setup_sim_dataset）")
    ap.add_argument("--sim-epochs", type=int, default=250,
                    help="调教性能 NN 训练轮数（纯标准库训练，250 轮≈20 分钟）")
    ap.add_argument("--skip-sim-nn", action="store_true",
                    help="只重训代理模型，跳过调教性能 NN（快速路径）")
    args = ap.parse_args(argv)

    scripts = ROOT / "scripts"
    _run("1/5 逐弯样本", [
        PY, str(scripts / "build_corner_dataset.py"), "--max-laps", str(args.max_laps),
    ])
    _run("2/5 特征融合", [PY, str(scripts / "merge_features.py")])
    _run("3/5 代理模型训练（弯速/残差诊断）", [
        PY, str(scripts / "train_telemetry_surrogate.py"),
        "--corner-samples", str(args.mlp_samples),
        "--epochs", str(args.epochs),
        "--ridge-samples", str(args.ridge_samples),
    ])
    if args.skip_sim_nn:
        print("\n[跳过] 4/5、5/5 调教性能 NN（--skip-sim-nn）")
    else:
        # 4/5、5/5：调教优化的模型驱动环节（"参数矩阵给方向 + 神经网络不断模拟优化"）
        _run("4/5 调教仿真数据集（遥测锚定）", [
            PY, str(scripts / "build_setup_sim_dataset.py"),
            "--setups", str(args.sim_setups),
        ])
        _run("5/5 调教性能 NN 训练（纯标准库，无 PyTorch）", [
            PY, str(scripts / "train_setup_sim_nn.py"), "--epochs", str(args.sim_epochs),
        ])

    for name in ("telemetry_surrogate.json", "setup_sim_nn.json"):
        model = ROOT / "setup_tuner" / "resources" / "models" / name
        state = (f"{model.stat().st_size / 1e6:.2f} MB"
                 if model.exists() else "缺失（该环节将自动降级）")
        print(f"模型：{model.name} → {state}")
    print("引擎下次调用即自动加载，无需重启代码改动。")
    print("建议接着跑：python scripts/verify_optimization_telemetry.py 复核效果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
