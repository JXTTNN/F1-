"""Label smoothing tests (Iter-160)."""
from __future__ import annotations

import pytest
import torch

from f1opt.model.surrogate import SurrogateModel
from f1opt.model.train import _LabelSmoothedMSELoss, train


class TestLabelSmoothedMSELoss:
    def test_alpha_zero_equals_mse(self) -> None:
        """alpha=0 时与标准 MSE 相同."""
        loss_fn = _LabelSmoothedMSELoss(alpha=0.0)
        mse = torch.nn.MSELoss()
        pred = torch.randn(10, 3)
        target = torch.randn(10, 3)
        assert torch.allclose(loss_fn(pred, target), mse(pred, target))

    def test_alpha_positive_differs_from_mse(self) -> None:
        """alpha>0 时与标准 MSE 不同 (除非 target 恒等)."""
        loss_fn = _LabelSmoothedMSELoss(alpha=0.1)
        mse = torch.nn.MSELoss()
        pred = torch.randn(10, 3)
        target = torch.randn(10, 3)
        assert not torch.allclose(loss_fn(pred, target), mse(pred, target))

    def test_constant_target_unchanged(self) -> None:
        """当所有 target 相同时, smoothing 不改变 loss (batch_mean = target)."""
        loss_fn = _LabelSmoothedMSELoss(alpha=0.5)
        mse = torch.nn.MSELoss()
        pred = torch.randn(10, 3)
        target = torch.ones(10, 3) * 5.0
        assert torch.allclose(loss_fn(pred, target), mse(pred, target))

    def test_invalid_alpha_negative(self) -> None:
        """alpha < 0 抛出 ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            _LabelSmoothedMSELoss(alpha=-0.1)

    def test_invalid_alpha_one(self) -> None:
        """alpha >= 1 抛出 ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            _LabelSmoothedMSELoss(alpha=1.0)

    def test_alpha_in_range(self) -> None:
        """alpha 在 [0, 1) 内不抛异常."""
        _LabelSmoothedMSELoss(alpha=0.0)
        _LabelSmoothedMSELoss(alpha=0.05)
        _LabelSmoothedMSELoss(alpha=0.15)
        _LabelSmoothedMSELoss(alpha=0.99)

    def test_smoothing_reduces_extreme_targets(self) -> None:
        """smoothing 将极端 target 向 batch_mean 收缩."""
        loss_fn = _LabelSmoothedMSELoss(alpha=0.5)
        pred = torch.zeros(4, 1)
        # Targets: [0, 0, 0, 10] — mean = 2.5
        # Smoothed: [0.5*0+0.5*2.5, ..., 0.5*10+0.5*2.5] = [1.25, 1.25, 1.25, 6.25]
        # MSE(0, [1.25, 1.25, 1.25, 6.25]) = mean([1.5625, 1.5625, 1.5625, 39.0625]) = 10.9375
        target = torch.tensor([[0.0], [0.0], [0.0], [10.0]])
        loss = loss_fn(pred, target)
        assert abs(float(loss) - 10.9375) < 0.01

    def test_forward_with_different_shapes(self) -> None:
        """不同形状的输入正常工作."""
        loss_fn = _LabelSmoothedMSELoss(alpha=0.1)
        pred = torch.randn(20, 7)
        target = torch.randn(20, 7)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # scalar


class TestTrainWithLabelSmoothing:
    def test_default_no_smoothing(self) -> None:
        """默认 label_smoothing=0.0 训练正常 (向后兼容)."""
        model = train(iterations=100, n_samples=200, seed=0, log=False, save=False)
        assert isinstance(model, SurrogateModel)

    def test_with_smoothing(self) -> None:
        """label_smoothing=0.1 训练正常."""
        model = train(
            iterations=100, n_samples=200, seed=0, log=False, save=False,
            label_smoothing=0.1,
        )
        assert isinstance(model, SurrogateModel)
        # Model weights should be non-zero
        params = list(model.parameters())
        assert params[0].abs().sum() > 0

    def test_smoothing_with_minibatch(self) -> None:
        """label_smoothing + mini-batch 训练正常."""
        model = train(
            iterations=100, n_samples=200, seed=0, log=False, save=False,
            batch_size=32, early_stopping_patience=5,
            label_smoothing=0.1,
        )
        assert isinstance(model, SurrogateModel)

    def test_smoothing_with_huber_loss(self) -> None:
        """label_smoothing + Huber loss — smoothing only applies to MSE."""
        # When loss_type="huber", label_smoothing has no effect (Huber doesn't use it)
        model = train(
            iterations=100, n_samples=200, seed=0, log=False, save=False,
            loss_type="huber", label_smoothing=0.1,
        )
        assert isinstance(model, SurrogateModel)

    def test_smoothing_does_not_destroy_training(self) -> None:
        """label_smoothing 不破坏训练 — 模型可正常推理."""
        model = train(
            iterations=200, n_samples=500, seed=0, log=False, save=False,
            label_smoothing=0.1,
        )
        x = torch.randn(10, 41)
        sr, rr = model(x)
        assert sr.shape == (10, 3)
        assert rr.shape == (10, 7)
        assert not torch.isnan(sr).any()
        assert not torch.isinf(sr).any()

    def test_train_ensemble_with_smoothing(self) -> None:
        """train_ensemble 传递 label_smoothing 正常."""
        from f1opt.model.train import train_ensemble
        ensemble = train_ensemble(
            n_members=2, base_seed=0,
            iterations=100, n_samples=200, log=False, save=False,
            label_smoothing=0.1,
        )
        assert ensemble is not None
        assert len(ensemble.models) == 2
