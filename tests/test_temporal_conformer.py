
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
        y = model(x)

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

    prediction = model(x)
    loss = prediction.square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient: {name}"
        )
        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient: {name}"
        )
