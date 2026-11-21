
"""Tests for hybrid EEG reconstruction loss."""

import torch

from eeg_reconstruction.training.losses import (
    hybrid_loss,
    masked_mse,
    masked_pearson_loss,
)


def test_perfect_correlation():
    target = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]]]
    )
    prediction = 2 * target
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = masked_pearson_loss(
        prediction, target, mask
    )
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_negative_correlation():
    target = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]]]
    )
    prediction = -target
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = masked_pearson_loss(
        prediction, target, mask
    )
    assert torch.allclose(loss, torch.tensor(2.0), atol=1e-6)


def test_missing_values_ignored():
    target = torch.tensor(
        [[[1.0, 2.0, 3.0, 1000.0]]]
    )
    prediction = torch.tensor(
        [[[1.0, 2.0, 3.0, -1000.0]]]
    )
    mask = torch.tensor(
        [[[True, True, True, False]]]
    )

    loss = masked_pearson_loss(
        prediction, target, mask
    )
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_constant_signals_are_safe():
    target = torch.ones(2, 3, 80)
    prediction = torch.zeros(2, 3, 80, requires_grad=True)
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = masked_pearson_loss(
        prediction, target, mask
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_hybrid_zero_weight_equals_mse():
    target = torch.randn(2, 3, 80)
    prediction = torch.randn(2, 3, 80)
    mask = torch.ones_like(target, dtype=torch.bool)

    actual = hybrid_loss(
        prediction, target, mask, pearson_weight=0.0
    )
    expected = masked_mse(prediction, target, mask)

    assert torch.allclose(actual, expected)


def test_hybrid_gradients():
    torch.manual_seed(42)

    target = torch.randn(4, 29, 80)
    prediction = torch.randn(
        4, 29, 80, requires_grad=True
    )
    mask = torch.rand(4, 29, 80) > 0.2

    loss = hybrid_loss(
        prediction, target, mask, pearson_weight=1.0
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
