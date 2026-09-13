"""训练神经网络模型（task-43）。

使用规则引擎生成训练数据（因为暂时没有真实 F1 数据）：

    1. 遍历所有 24 赛道 × 12 症状 × 5 强度 × 随机调教参数
    2. 用规则引擎生成 label（SetupDelta）
    3. 训练神经网络拟合规则引擎的输出
    4. 保存权重到 ``data/nn_weights.pt``

未来有真实 F1 遥测数据后，可以用真实数据替换训练集。

训练流程：
    - 生成训练数据：24 赛道 × 12 症状 × 5 强度 × 100 组随机调教 = 144000 样本
    - 80% 训练 / 20% 验证
    - 损失函数：MSE
    - 优化器：Adam, lr=0.001
    - 训练 100 个 epoch
    - 保存权重到 ``data/nn_weights.pt``

用法::

    python scripts/train_nn_model.py
    python scripts/train_nn_model.py --epochs 50 --samples 50000

来源经验：
    - 2026-09-10-deterministic-advice-engine：参数全集 S + 诊断维度 Dx
    - 2026-09-11-deterministic-engine-unit-test-pattern：确定性断言
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _check_torch() -> bool:
    """检查 PyTorch 是否可用，不可用时打印提示并退出。"""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        print("=" * 60)
        print("错误：PyTorch 未安装！")
        print("请先安装 PyTorch：")
        print("  pip install torch")
        print("或：")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("=" * 60)
        return False


def generate_training_data(
    n_samples_per_combo: int = 100,
    seed: int = 42,
) -> list[dict]:
    """用规则引擎生成训练数据。

    遍历 24 赛道 × 12 症状 × 5 强度 × n_samples_per_combo 组随机调教参数，
    用规则引擎生成 label（SetupDelta）。

    Args:
        n_samples_per_combo: 每个症状×强度组合的随机调教样本数。
        seed: 随机种子（保证可复现）。

    Returns:
        训练样本列表，每项含：
            - symptoms: [(symptom_key, strength), ...]
            - dx: 诊断向量
            - current_setup: 当前调教快照
            - track_id: 赛道 ID
            - label: 规则引擎输出的 SetupDelta
    """

    from setup_tuner.domain.symptoms import Symptom
    from setup_tuner.domain.track import ALL_TRACKS
    from setup_tuner.engine.diagnostic import compute_dx
    from setup_tuner.engine.engine import compute_setup_delta

    rng = random.Random(seed)

    samples: list[dict] = []
    track_ids = [t.track_id for t in ALL_TRACKS]
    symptom_keys = [s.value for s in Symptom]
    intensities = [1, 2, 3, 4, 5]  # 强度 1-5

    total_combos = len(track_ids) * len(symptom_keys) * len(intensities)
    total_samples = total_combos * n_samples_per_combo
    print(f"生成训练数据：{len(track_ids)} 赛道 × {len(symptom_keys)} 症状 × "
          f"{len(intensities)} 强度 × {n_samples_per_combo} 调教 = {total_samples} 样本")

    start_time = time.time()

    for track_id in track_ids:
        for symptom_key in symptom_keys:
            for strength in intensities:
                for _ in range(n_samples_per_combo):
                    # 生成随机调教参数（在合法范围内）
                    current_setup = _random_setup(rng)

                    # 规则引擎生成 label
                    symptoms = [(symptom_key, strength)]
                    dx = compute_dx(symptoms)
                    label = compute_setup_delta(dx, current_setup)

                    samples.append({
                        "symptoms": symptoms,
                        "dx": dx,
                        "current_setup": current_setup,
                        "track_id": track_id,
                        "label": label,
                    })

    elapsed = time.time() - start_time
    print(f"训练数据生成完成：{len(samples)} 样本，耗时 {elapsed:.1f}s")

    return samples


def _random_setup(rng: random.Random) -> dict[str, float]:
    """生成随机调教参数（在合法范围内，对齐到步长）。

    Args:
        rng: 随机数生成器。

    Returns:
        23 参数的调教快照字典。
    """
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    setup: dict[str, float] = {}
    for spec in ALL_SETUP_FIELDS:
        # 在 [min, max] 范围内随机取值，对齐到 step
        n_steps = int(round((spec.max_val - spec.min_val) / spec.step))
        step_idx = rng.randint(0, n_steps)
        value = spec.min_val + step_idx * spec.step
        # 整数参数取整
        if spec.step >= 1.0 and float(spec.step).is_integer():
            value = float(round(value))
        setup[spec.name] = value
    return setup


def build_training_tensors(
    samples: list[dict],
) -> tuple:
    """将训练样本转为 PyTorch tensor。

    Args:
        samples: 训练样本列表。

    Returns:
        (X, y) 二元组，X 为输入 tensor (N, 68)，y 为输出 tensor (N, 23)。
    """
    import torch

    from setup_tuner.domain.setup import ALL_SETUP_FIELDS
    from setup_tuner.engine.nn_model import build_input_vector

    X_list: list[list[float]] = []
    y_list: list[list[float]] = []

    for sample in samples:
        # 构建输入向量
        x_vec = build_input_vector(
            sample["symptoms"],
            sample["dx"],
            sample["current_setup"],
            sample["track_id"],
        )
        X_list.append(x_vec)

        # 构建输出向量（归一化 delta 到 [-1, 1]）
        y_vec: list[float] = []
        for spec in ALL_SETUP_FIELDS:
            delta = sample["label"].get(spec.name, 0.0)
            # 归一化：delta / max_delta → [-1, 1]
            if spec.max_delta > 0:
                normalized = delta / spec.max_delta
            else:
                normalized = 0.0
            # 裁剪到 [-1, 1]
            y_vec.append(max(-1.0, min(1.0, normalized)))
        y_list.append(y_vec)

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)

    return X, y


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 256,
) -> tuple[object, float]:
    """训练神经网络模型。

    Args:
        X_train: 训练集输入 tensor。
        y_train: 训练集输出 tensor。
        X_val: 验证集输入 tensor。
        y_val: 验证集输出 tensor。
        epochs: 训练轮数。
        lr: 学习率。
        batch_size: 批次大小。

    Returns:
        (model, best_val_loss) 二元组，model 为训练好的 F1SetupNet，
        best_val_loss 为最优验证损失。
    """
    import torch
    from torch import nn

    from setup_tuner.engine.nn_model import F1SetupNet

    model = F1SetupNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = X_train.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    print(f"开始训练：{epochs} epochs, {n_train} 样本, {n_batches} 批次/epoch, lr={lr}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # 训练模式
        model.train()
        # 打乱训练集
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            batch_X = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        epoch_loss /= n_train

        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()

        # 保存最优权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                  f"train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

    # 恢复最优权重
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"训练完成，最优验证损失: {best_val_loss:.6f}")

    model.eval()
    return model, best_val_loss


def main() -> int:
    """训练脚本主入口。

    Returns:
        0 表示成功，1 表示失败。
    """
    parser = argparse.ArgumentParser(description="训练 F1 调教优化神经网络模型")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练轮数（默认 100）",
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="每个症状×强度组合的随机调教样本数（默认 100）",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="学习率（默认 0.001）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="批次大小（默认 256）",
    )
    parser.add_argument(
        "--output", type=str, default="data/nn_weights.pt",
        help="权重输出路径（默认 data/nn_weights.pt）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    args = parser.parse_args()

    if not _check_torch():
        return 1

    print("=" * 60)
    print("F1 调教优化神经网络训练")
    print("=" * 60)

    # ① 生成训练数据
    print("\n[1/4] 生成训练数据...")
    samples = generate_training_data(
        n_samples_per_combo=args.samples,
        seed=args.seed,
    )

    # ② 构建训练 tensor
    print("\n[2/4] 构建训练 tensor...")
    X, y = build_training_tensors(samples)
    print(f"  输入维度: {X.shape}, 输出维度: {y.shape}")

    # ③ 划分训练集 / 验证集（80/20）
    import torch
    n_total = X.shape[0]
    n_train = int(n_total * 0.8)
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本")

    # ④ 训练
    print("\n[3/4] 训练神经网络...")
    start_time = time.time()
    model, best_val_loss = train_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )
    elapsed = time.time() - start_time
    print(f"  训练耗时: {elapsed:.1f}s")

    # ⑤ 保存权重
    print("\n[4/4] 保存模型权重...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_path))
    print(f"  权重已保存到: {output_path}")

    # ⑥ 验证：加载权重并推理
    print("\n验证：加载权重并推理...")
    from setup_tuner.engine.nn_model import NNModelManager
    manager = NNModelManager(weights_path=output_path)
    if manager.available:
        # 取一个样本测试推理
        test_sample = samples[0]
        predicted = manager.predict(
            test_sample["symptoms"],
            test_sample["dx"],
            test_sample["current_setup"],
            test_sample["track_id"],
        )
        if predicted is not None:
            label = test_sample["label"]
            # 计算误差
            errors = [abs(predicted[k] - label[k]) for k in label]
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
            print(f"  推理成功！平均误差: {avg_error:.4f}, 最大误差: {max_error:.4f}")
        else:
            print("  警告：推理返回 None")
            return 1
    else:
        print("  警告：模型加载失败")
        return 1

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  权重路径: {output_path}")
    print(f"  样本数: {len(samples)}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  最优验证损失: {best_val_loss:.6f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())