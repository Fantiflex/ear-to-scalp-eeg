

import torch

from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
    TemporalConformer,
)


def test_forward_shape():
    config = ConformerConfig(
        in_channels=4,
        out_channels=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    model = TemporalConformer(config)
    x = torch.randn(2, 4, 256)

    with torch.no_grad():
        mask = torch.ones_like(x, dtype=torch.bool)
        y = model(x, mask)

    assert y.shape == (2, 8, 256)
    assert torch.isfinite(y).all()


def test_backward():
    config = ConformerConfig(
        in_channels=4,
        out_channels=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    model = TemporalConformer(config)
    x = torch.randn(2, 4, 128)
    
    mask = torch.ones_like(x, dtype=torch.bool)
    prediction = model(x, mask)
    loss = prediction.square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient: {name}"
        )
        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient: {name}"
        )






def test_conformer_with_missing_channels():
    cfg = ConformerConfig(
        in_channels=12,
        out_channels=29,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    model = TemporalConformer(cfg)

    eeg = torch.randn(2, 12, 80)
    mask = torch.ones(2, 12, 80, dtype=torch.bool)

    # Simulate an entirely unavailable electrode.
    eeg[:, 0, :] = float("nan")
    mask[:, 0, :] = False

    predictions = model(eeg, mask)

    assert predictions.shape == (2, 29, 80)
    assert torch.isfinite(predictions).all()

    predictions.mean().backward()
    assert model.stem.weight.grad is not None
    assert torch.isfinite(model.stem.weight.grad).all()
