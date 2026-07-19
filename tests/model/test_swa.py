"""Stochastic Weight Averaging tests (Iter-156)."""
from __future__ import annotations

import pytest
import torch

from f1opt.model.surrogate import SurrogateModel
from f1opt.model.train import _SWAWeights, train


class TestSWAWeights:
    def test_should_collect_before_start(self) -> None:
        """swa_start 之前不收集."""
        model = SurrogateModel()
        swa = _SWAWeights(model, swa_start=100, swa_freq=10)
        assert not swa.should_collect(0)
        assert not swa.should_collect(50)
        assert not swa.should_collect(99)

    def test_should_collect_at_start(self) -> None:
        """swa_start 处开始收集."""
        model = SurrogateModel()
        swa = _SWAWeights(model, swa_start=100, swa_freq=10)
        assert swa.should_collect(100)
        assert swa.should_collect(110)
        assert swa.should_collect(120)
        assert not swa.should_collect(105)

    def test_collect_increments_count(self) -> None:
        """collect() 增加 n_collected."""
        model = SurrogateModel()
        swa = _SWAWeights(model, swa_start=0, swa_freq=1)
        assert swa.n_collected == 0
        swa.collect(model)
        assert swa.n_collected == 1
        swa.collect(model)
        assert swa.n_collected == 2

    def test_apply_to_averages_weights(self) -> None:
        """apply_to() 将平均权重写入模型."""
        model = SurrogateModel()
        swa = _SWAWeights(model, swa_start=0, swa_freq=1)
        # Record original weights
        orig_param = next(model.parameters()).detach().clone()
        # Collect current weights twice (n_collected=2)
        swa.collect(model)
        swa.collect(model)
        # Apply SWA (should be same as original since we averaged identical weights)
        swa.apply_to(model)
        avg_param = next(model.parameters()).detach().clone()
        torch.testing.assert_close(avg_param, orig_param)

    def test_apply_to_no_checkpoints_is_noop(self) -> None:
        """无 checkpoint 时 apply_to 是 no-op."""
        model = SurrogateModel()
        swa = _SWAWeights(model, swa_start=100, swa_freq=10)
        orig_param = next(model.parameters()).detach().clone()
        swa.apply_to(model)
        after_param = next(model.parameters()).detach().clone()
        torch.testing.assert_close(after_param, orig_param)

    def test_restore_after_apply(self) -> None:
        """restore() 恢复原始权重."""
        model = SurrogateModel()
        swa = _SWAWeights(model, swa_start=0, swa_freq=1)
        orig_param = next(model.parameters()).detach().clone()
        swa.collect(model)
        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        modified_param = next(model.parameters()).detach().clone()
        swa.collect(model)  # collect modified weights
        swa.apply_to(model)  # apply average (should differ from orig)
        swa.restore(model)  # restore to pre-apply state
        restored_param = next(model.parameters()).detach().clone()
        # After restore, should match the modified weights (state at apply_to time)
        torch.testing.assert_close(restored_param, modified_param)

    def test_swa_averages_different_weights(self) -> None:
        """SWA 正确平均两个不同的权重集."""
        model = SurrogateModel()
        swa = _SWAWeights(model, swa_start=0, swa_freq=1)
        # Collect original weights
        swa.collect(model)
        # Record original for later comparison
        orig_param = next(model.parameters()).detach().clone()
        # Modify weights (add 2.0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(2.0)
        swa.collect(model)
        # Apply SWA — average of w and w+2 is w+1
        swa.apply_to(model)
        avg_param = next(model.parameters()).detach().clone()
        # The average should be orig + 1.0
        assert abs(float(avg_param.mean()) - float(orig_param.mean()) - 1.0) < 0.01

    def test_buffers_averaged(self) -> None:
        """浮点 buffer 也被平均 (BatchNorm running stats)."""
        model = SurrogateModel()
        # Find a floating-point buffer
        float_bufs = [
            (n, b) for n, b in model.named_buffers() if b.is_floating_point()
        ]
        if not float_bufs:
            pytest.skip("No floating-point buffers in SurrogateModel")
        buf_name, buf_val = float_bufs[0]
        swa = _SWAWeights(model, swa_start=0, swa_freq=1)
        swa.collect(model)
        swa.collect(model)
        swa.apply_to(model)
        # Since we collected identical weights twice, buffer should be unchanged
        new_val = dict(model.named_buffers())[buf_name]
        torch.testing.assert_close(new_val, buf_val)

    def test_integer_buffers_not_touched(self) -> None:
        """整数 buffer (num_batches_tracked) 不被 SWA 修改."""
        model = SurrogateModel()
        int_bufs = [
            (n, b) for n, b in model.named_buffers() if not b.is_floating_point()
        ]
        if not int_bufs:
            pytest.skip("No integer buffers in SurrogateModel")
        buf_name, buf_val = int_bufs[0]
        orig_val = buf_val.clone()
        swa = _SWAWeights(model, swa_start=0, swa_freq=1)
        swa.collect(model)
        swa.apply_to(model)
        new_val = dict(model.named_buffers())[buf_name]
        torch.testing.assert_close(new_val, orig_val)


class TestTrainWithSWA:
    def test_train_default_no_swa(self) -> None:
        """默认 swa_start=0 不启用 SWA (向后兼容)."""
        model = train(iterations=100, n_samples=200, seed=0, log=False, save=False)
        assert isinstance(model, SurrogateModel)

    def test_train_with_swa(self) -> None:
        """启用 SWA 训练正常."""
        model = train(
            iterations=100, n_samples=200, seed=0, log=False, save=False,
            swa_start=50, swa_freq=10,
        )
        assert isinstance(model, SurrogateModel)
        # Model weights should be non-zero
        params = list(model.parameters())
        assert params[0].abs().sum() > 0

    def test_train_swa_with_minibatch(self) -> None:
        """SWA + mini-batch 训练正常 (SWA only in full-data path currently)."""
        model = train(
            iterations=100, n_samples=200, seed=0, log=False, save=False,
            batch_size=32, early_stopping_patience=5,
            swa_start=50, swa_freq=10,
        )
        assert isinstance(model, SurrogateModel)

    def test_train_swa_does_not_destroy_training(self) -> None:
        """SWA 不破坏训练 — 模型可正常推理."""
        model = train(
            iterations=200, n_samples=500, seed=0, log=False, save=False,
            swa_start=100, swa_freq=20,
        )
        x = torch.randn(10, 37)
        sr, rr = model(x)
        assert sr.shape == (10, 3)
        assert rr.shape == (10, 7)
        assert not torch.isnan(sr).any()
        assert not torch.isinf(sr).any()

    def test_train_swa_with_ema(self) -> None:
        """SWA + EMA 同时启用 (SWA 覆盖 EMA)."""
        model = train(
            iterations=100, n_samples=200, seed=0, log=False, save=False,
            ema_decay=0.99, swa_start=50, swa_freq=10,
        )
        assert isinstance(model, SurrogateModel)
        # Model produces valid output
        x = torch.randn(5, 37)
        sr, _ = model(x)
        assert not torch.isnan(sr).any()
