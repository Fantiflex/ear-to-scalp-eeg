
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




def masked_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Temporal Pearson loss averaged over valid window-channel pairs.

    Inputs have shape (batch, channels, time).
    Missing samples are excluded using the observation mask.
    Pairs with fewer than two valid samples or zero variance
    contribute no correlation penalty.
    """
    mask = mask.bool()
    weights = mask.to(dtype=prediction.dtype)

    count = weights.sum(dim=-1, keepdim=True)
    safe_count = count.clamp_min(1.0)

    pred_mean = (
        (prediction * weights).sum(dim=-1, keepdim=True)
        / safe_count
    )
    target_mean = (
        (target * weights).sum(dim=-1, keepdim=True)
        / safe_count
    )

    pred_centered = (prediction - pred_mean) * weights
    target_centered = (target - target_mean) * weights

    covariance = (
        pred_centered * target_centered
    ).sum(dim=-1)

    pred_ss = pred_centered.square().sum(dim=-1)
    target_ss = target_centered.square().sum(dim=-1)

    denominator = torch.sqrt(
        pred_ss.clamp_min(eps) * target_ss.clamp_min(eps)
    )

    correlation = covariance / denominator
    correlation = correlation.clamp(-1.0, 1.0)

    valid_pairs = (
        (count.squeeze(-1) >= 2)
        & (pred_ss > eps)
        & (target_ss > eps)
    )

    penalty = 1.0 - correlation

    # Keep the loss differentiable even if every pair is invalid.
    valid_weights = valid_pairs.to(prediction.dtype)
    return (
        (penalty * valid_weights).sum()
        / valid_weights.sum().clamp_min(1.0)
    )


def hybrid_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pearson_weight: float = 1.0,
) -> torch.Tensor:
    """Combine masked reconstruction error and temporal correlation."""
    if pearson_weight < 0:
        raise ValueError("pearson_weight must be non-negative")

    mse = masked_mse(prediction, target, mask)

    if pearson_weight == 0:
        return mse

    pearson_penalty = masked_pearson_loss(
        prediction, target, mask
    )

    return mse + pearson_weight * pearson_penalty





import torch
from torch import Tensor


def masked_temporal_derivative_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Mean squared error between consecutive temporal differences.

    Args:
        pred: Predicted EEG signals, shape (B, C, T).
        target: Ground-truth EEG signals, shape (B, C, T).
        mask: Boolean validity mask, shape (B, C, T).
              True indicates a valid observation.

    Returns:
        Scalar temporal derivative loss.
        Returns zero when no consecutive valid pairs exist.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")

    if pred.ndim != 3:
        raise ValueError("Expected tensors with shape (B, C, T)")

    if pred.shape[-1] < 2:
        raise ValueError("At least 2 time samples are required")

    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    else:
        if mask.shape != target.shape:
            raise ValueError("mask must match target shape")
        mask = mask.to(device=target.device, dtype=torch.bool)

    # A pair is valid only if BOTH samples are observed.
    pair_mask = mask[..., 1:] & mask[..., :-1]

    # Temporal differences along the time axis.
    pred_diff = pred[..., 1:] - pred[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]

    squared_error = (pred_diff - target_diff).square()

    # Exclude invalid pairs before reduction.
    valid_error = torch.where(
        pair_mask,
        squared_error,
        torch.zeros_like(squared_error),
    )

    return valid_error.sum() / pair_mask.sum().clamp(min=1)
