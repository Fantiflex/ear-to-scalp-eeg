
"""Loss functions for EEG reconstruction."""

import torch


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE computed only on observed scalp EEG samples."""
    if not (
        prediction.shape == target.shape == observed_mask.shape
    ):
        raise ValueError("Prediction, target and mask must match")

    valid = observed_mask.bool() & torch.isfinite(target)

    if not valid.any():
        raise ValueError("Batch contains no valid scalp targets")

    squared_error = (
        prediction - torch.where(valid, target, 0.0)
    ).square()

    return squared_error[valid].mean()
