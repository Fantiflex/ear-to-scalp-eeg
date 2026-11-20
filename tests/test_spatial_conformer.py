
import torch

from eeg_reconstruction.models.spatial_conformer import (
    SpatialConformer,
)
from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
)


def make_model():
    cfg = ConformerConfig(
        in_channels=12,
        out_channels=29,
        d_model=64,
        n_heads=4,
        n_layers=2,
        kernel_size=15,
        dropout=0.1,
    )
    return SpatialConformer(cfg)


def test_output_shape():
    model = make_model()

    x = torch.randn(2, 12, 80)
    mask = torch.ones_like(x, dtype=torch.bool)

    prediction = model(x, mask)

    assert prediction.shape == (2, 29, 80)
    assert torch.isfinite(prediction).all()


def test_backward():
    model = make_model()

    x = torch.randn(2, 12, 80)
    mask = torch.ones_like(x, dtype=torch.bool)

    prediction = model(x, mask)
    loss = prediction.square().mean()
    loss.backward()

    assert model.output_head.weight.grad is not None
    assert torch.isfinite(model.output_head.weight.grad).all()


def test_missing_input():
    model = make_model().eval()

    x = torch.randn(2, 12, 80)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 3, :] = False
    x[:, 3, :] = float("nan")

    with torch.no_grad():
        prediction = model(x, mask)

    assert torch.isfinite(prediction).all()
