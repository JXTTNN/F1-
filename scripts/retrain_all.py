"""一键重训流水线：遥测 → 逐弯样本 → 代理模型 → 融合特征。

按顺序跑完三步（任一失败即整体失败，不静默跳过）：

1. ``build_corner_dataset.py``   外部 2026 遥测 → 逐弯样本
2. ``merge_features.py``         外部基准 × 本地样本 → 训练特征
3. ``train_telemetry_surrogate.py`` 训练代理模型 → 引擎自动加载

引擎（``setup_tuner.engine.lap_model`` / ``engine.generate_suggestion``）
会在启动时读取 ``data/models/telemetry_surrogate.json``；本脚本跑完即生效，
无需改代码。

用法::

    python scripts/retrain_all.py                  # 默认参数
    python scripts/retrain_all.py --max-laps 200 --ridge-samples 40000
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
    ap = argparse.ArgumentParser(description="重训遥测代理模型全流程")
    ap.add_argument("--max-laps", type=int, default=120,
                    help="每赛道每会话最多取多少圈（透传给 build_corner_dataset）")
    ap.add_argument("--ridge-samples", type=int, default=25000)
    ap.add_argument("--mlp-samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=18)
    args = ap.parse_args(argv)

    scripts = ROOT / "scripts"
    _run("1/3 逐弯样本", [
        PY, str(scripts / "build_corner_dataset.py"), "--max-laps", str(args.max_laps),
    ])
    _run("2/3 特征融合", [PY, str(scripts / "merge_features.py")])
    _run("3/3 代理模型训练", [
        PY, str(scripts / "train_telemetry_surrogate.py"),
        "--corner-samples", str(args.mlp_samples),
        "--epochs", str(args.epochs),
        "--ridge-samples", str(args.ridge_samples),
    ])

    model = ROOT / "data" / "models" / "telemetry_surrogate.json"
    print(f"\n模型就绪：{model}（{model.stat().st_size / 1e6:.2f} MB）")
    print("引擎下次调用即自动加载，无需重启代码改动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
