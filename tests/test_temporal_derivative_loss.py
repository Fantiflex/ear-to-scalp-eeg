
import pytest
import torch

from eeg_reconstruction.training.losses import (
    masked_temporal_derivative_loss,
)


def test_identical_signals():
    x = torch.randn(2, 29, 80)
    loss = masked_temporal_derivative_loss(x, x)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_known_derivative_error():
    pred = torch.tensor([[[0., 2., 4.]]])
    target = torch.tensor([[[0., 1., 2.]]])

    # Predicted differences: [2, 2]
    # Target differences:    [1, 1]
    # MSE = (1^2 + 1^2) / 2 = 1
    loss = masked_temporal_derivative_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(1.0))


def test_missing_values():
    pred = torch.tensor([[[0., 2., 4., 8.]]])
    target = torch.tensor([[[0., 1., 2., 3.]]])
    mask = torch.tensor([[[True, True, False, True]]])

    # Only the first temporal pair is valid.
    loss = masked_temporal_derivative_loss(pred, target, mask)

    assert torch.isclose(loss, torch.tensor(1.0))


def test_all_invalid():
    pred = torch.randn(2, 29, 80, requires_grad=True)
    target = torch.randn(2, 29, 80)
    mask = torch.zeros_like(target, dtype=torch.bool)

    loss = masked_temporal_derivative_loss(pred, target, mask)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad == 0)


def test_gradients():
    pred = torch.randn(2, 29, 80, requires_grad=True)
    target = torch.randn(2, 29, 80)

    loss = masked_temporal_derivative_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_masked_nan_values():
    pred = torch.tensor([[[0., 1., float("nan"), 3.]]])
    target = torch.tensor([[[0., 0., float("nan"), 0.]]])
    mask = torch.tensor([[[True, True, False, True]]])

    loss = masked_temporal_derivative_loss(pred, target, mask)

    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_invalid_shape():
    pred = torch.randn(2, 29, 80)
    target = torch.randn(2, 28, 80)

    with pytest.raises(ValueError):
        masked_temporal_derivative_loss(pred, target)
