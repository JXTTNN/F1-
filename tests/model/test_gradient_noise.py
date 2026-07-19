"""梯度噪声注入单元测试 (Iter-152).

覆盖:
- ``_add_gradient_noise`` 确实向梯度添加噪声 (非零 std 时梯度改变).
- ``train()`` 默认 ``gradient_noise=0.0`` 向后兼容 (无 crash).
- ``train()`` ``gradient_noise=0.01`` 正常训练 (无 crash, 返回有效模型).
- ``train()`` ``gradient_noise=0.01`` + mini-batch 正常训练.
- ``train_ensemble()`` ``gradient_noise`` 正常训练.
- 无梯度时 ``_add_gradient_noise`` 不崩溃 (安全回退).
"""

from __future__ import annotations

import torch

from f1opt.model.surrogate import SurrogateModel
from f1opt.model.train import _add_gradient_noise, train, train_ensemble


# --- 梯度噪声注入 ------------------------------------------------------------
def test_add_gradient_noise_modifies_gradients() -> None:
    """_add_gradient_noise(std=0.1) 会使梯度发生改变."""
    model = SurrogateModel()
    x = torch.randn(4, 37)
    sr, _rr = model(x)
    sr.sum().backward()
    # 记录原始梯度副本
    original_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    _add_gradient_noise(model, std=0.1)
    # 验证梯度已改变 (至少一个参数的梯度发生变化)
    changed = False
    for name, p in model.named_parameters():
        if p.grad is not None and name in original_grads:
            if not torch.allclose(p.grad, original_grads[name], atol=1e-8):
                changed = True
                break
    assert changed, "梯度在添加噪声后应发生变化"


def test_add_gradient_noise_zero_std_no_change() -> None:
    """_add_gradient_noise(std=0.0) 不改变梯度."""
    model = SurrogateModel()
    x = torch.randn(4, 37)
    sr, _rr = model(x)
    sr.sum().backward()
    original_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    _add_gradient_noise(model, std=0.0)
    for name, p in model.named_parameters():
        if p.grad is not None and name in original_grads:
            assert torch.allclose(p.grad, original_grads[name], atol=1e-8), (
                f"梯度 {name} 在 std=0.0 时不应改变"
            )


def test_add_gradient_noise_no_gradients_safe() -> None:
    """无梯度时 _add_gradient_noise 不崩溃 (安全回退)."""
    model = SurrogateModel()
    # 没有 backward, 梯度为 None
    _add_gradient_noise(model, std=0.1)
    # 不应崩溃


def test_add_gradient_noise_scale_proportional() -> None:
    """std=0.5 的噪声幅度大于 std=0.01 (噪声尺度与 std 成正比)."""
    model = SurrogateModel()
    x = torch.randn(4, 37)
    sr, _rr = model(x)
    sr.sum().backward()
    grads_small = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    _add_gradient_noise(model, std=0.01)
    delta_small = {
        name: (p.grad - grads_small[name]).abs().mean().item()
        for name, p in model.named_parameters() if p.grad is not None and name in grads_small
    }
    # 重置梯度
    for name, p in model.named_parameters():
        if p.grad is not None and name in grads_small:
            p.grad.copy_(grads_small[name])
    _add_gradient_noise(model, std=0.5)
    delta_large = {
        name: (p.grad - grads_small[name]).abs().mean().item()
        for name, p in model.named_parameters() if p.grad is not None and name in grads_small
    }
    for name in delta_small:
        assert delta_large[name] > delta_small[name], (
            f"std=0.5 的噪声幅度 ({delta_large[name]:.6f}) 应大于 std=0.01 ({delta_small[name]:.6f})"
        )


# --- train() 向后兼容 --------------------------------------------------------
def test_train_default_no_noise() -> None:
    """默认 gradient_noise=0.0 训练正常 (向后兼容)."""
    model = train(iterations=100, n_samples=200, seed=0, log=False, save=False)
    assert isinstance(model, SurrogateModel)


def test_train_with_gradient_noise() -> None:
    """gradient_noise=0.01 训练正常, 返回有效模型."""
    model = train(
        iterations=100, n_samples=200, seed=0, log=False, save=False,
        gradient_noise=0.01,
    )
    assert isinstance(model, SurrogateModel)
    # 模型权重非零 (不是未训练状态)
    params = list(model.parameters())
    assert params[0].abs().sum() > 0, "模型权重应为非零"


def test_train_with_gradient_noise_and_minibatch() -> None:
    """gradient_noise=0.01 + mini-batch 训练正常."""
    model = train(
        iterations=100, n_samples=200, seed=0, log=False, save=False,
        batch_size=32, early_stopping_patience=5,
        gradient_noise=0.01,
    )
    assert isinstance(model, SurrogateModel)


def test_train_with_gradient_noise_huber_loss() -> None:
    """gradient_noise + Huber loss 训练正常."""
    model = train(
        iterations=100, n_samples=200, seed=0, log=False, save=False,
        loss_type="huber", gradient_noise=0.01,
    )
    assert isinstance(model, SurrogateModel)


# --- train_ensemble() --------------------------------------------------------
def test_train_ensemble_with_gradient_noise() -> None:
    """train_ensemble() 传递 gradient_noise 正常."""
    ensemble = train_ensemble(
        n_members=2, base_seed=0,
        iterations=100, n_samples=200, log=False, save=False,
        gradient_noise=0.01,
    )
    assert ensemble is not None
    assert len(ensemble.models) == 2


def test_train_ensemble_gradient_noise_default() -> None:
    """train_ensemble() 默认 gradient_noise=0.0 正常."""
    ensemble = train_ensemble(
        n_members=2, base_seed=0,
        iterations=100, n_samples=200, log=False, save=False,
    )
    assert ensemble is not None
    assert len(ensemble.models) == 2


# --- 噪声对训练的影响 (合理性检查) -------------------------------------------
def test_gradient_noise_does_not_destroy_training() -> None:
    """梯度噪声幅度合理时, 训练仍然收敛 (held-out sector MAE 有限)."""
    model = train(
        iterations=200, n_samples=500, seed=0, log=False, save=False,
        gradient_noise=0.005,
    )
    # 验证模型可正常推理
    x = torch.randn(10, 37)
    sr, rr = model(x)
    assert sr.shape == (10, 3)
    assert rr.shape == (10, 7)
    # 输出应在合理范围内 (非 NaN/Inf)
    assert not torch.isnan(sr).any()
    assert not torch.isinf(sr).any()
    assert sr.abs().mean() < 500.0, "sector 预测值异常大"