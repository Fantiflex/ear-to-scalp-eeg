
import torch

from eeg_reconstruction.data.normalization import (
    inverse_normalize,
    normalize_batch,
)
from eeg_reconstruction.training.losses import masked_mse


def test_original_units_mse_after_normalization():
    target = torch.tensor([[[10.0, 12.0, 14.0]]])
    mask = torch.ones_like(target, dtype=torch.bool)

    stats = {
        "scalp": {
            "mode": "channel",
            "mean": [12.0],
            "std": [2.0],
        }
    }

    batch = {
        "scalp": target,
        "target_mask": mask,
    }

    normalized = normalize_batch(batch, stats)

    # A normalized prediction with a constant offset of 1.
    pred_norm = normalized["scalp"] + 1.0

    pred_original = inverse_normalize(
        pred_norm,
        stats,
        modality="scalp",
    )

    loss = masked_mse(pred_original, target, mask)

    # Error is exactly 2 original EEG units per sample.
    assert torch.isclose(
        loss,
        torch.tensor(4.0),
    )


def test_inverse_transform_preserves_gradients():
    pred = torch.randn(2, 29, 80, requires_grad=True)

    stats = {
        "scalp": {
            "mode": "global",
            "mean": 1.0,
            "std": 2.0,
        }
    }

    restored = inverse_normalize(pred, stats)
    restored.square().mean().backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
