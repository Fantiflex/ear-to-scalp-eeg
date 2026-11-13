
"""Loss functions for EEG reconstruction."""

import torch


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE computed only on observed target samples."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")

    if mask.shape != target.shape:
        raise ValueError("Mask and target shapes differ")

    valid = mask.bool()

    if not torch.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")

    if not torch.isfinite(target[valid]).all():
        raise ValueError("Valid targets contain non-finite values")

    if not valid.any().item():
        raise ValueError("No valid target samples")

    # Avoid overflow from invalid-target values if encountered.
    residual = torch.where(
        valid,
        prediction - target,
        torch.zeros_like(prediction),
    )

    return residual.square().sum() / valid.sum()
