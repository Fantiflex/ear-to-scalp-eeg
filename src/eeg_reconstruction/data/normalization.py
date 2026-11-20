
"""Train-only normalization for masked EEG signals."""

from __future__ import annotations

import torch

MODALITY_MASKS = {
    "in_ear": "in_ear_observed_mask",
    "around_ear": "around_ear_observed_mask",
    "scalp": "target_mask",
}


def fit_normalization(
    train_loader,
    mode="global",
    modalities=("in_ear", "scalp"),
    eps=1e-6,
):
    """Fit normalization statistics using training subjects only.

    Modes:
        none: no normalization
        global: one mean/std per modality
        channel: one mean/std per channel

    Missing observations are excluded from all statistics.
    """
    if mode not in {"none", "global", "channel"}:
        raise ValueError(f"Unknown normalization mode: {mode}")

    if eps <= 0:
        raise ValueError("eps must be positive")

    stats = {}

    for modality in modalities:
        if modality not in MODALITY_MASKS:
            raise ValueError(f"Unknown modality: {modality}")

        mask_key = MODALITY_MASKS[modality]

        count = None
        total = None
        total_sq = None

        for batch in train_loader:
            x = batch[modality].to(torch.float64)
            mask = batch[mask_key].bool()

            # Missing values must not contribute to statistics.
            values = torch.where(
                mask,
                x,
                torch.zeros_like(x),
            )

            if mode == "global":
                dims = (0, 1, 2)
            else:
                dims = (0, 2)

            batch_count = mask.sum(dim=dims).to(torch.float64)
            batch_sum = values.sum(dim=dims)
            batch_sq = values.square().sum(dim=dims)

            if count is None:
                count = batch_count
                total = batch_sum
                total_sq = batch_sq
            else:
                count += batch_count
                total += batch_sum
                total_sq += batch_sq

        if count is None:
            raise ValueError("Empty training loader")

        if torch.any(count == 0):
            raise ValueError(
                f"No observed samples for {modality}"
            )

        mean = total / count
        variance = total_sq / count - mean.square()
        std = torch.sqrt(variance.clamp(min=0))

        # Protect against zero-variance channels.
        std = std.clamp(min=eps)

        stats[modality] = {
            "mode": mode,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    return stats


def normalize_batch(batch, stats, device=None):
    """Normalize EEG tensors while preserving observation masks."""
    result = dict(batch)

    for modality, config in stats.items():
        x = batch[modality]

        if device is not None:
            x = x.to(device)

        mask_key = MODALITY_MASKS[modality]
        mask = batch[mask_key].to(x.device).bool()

        mode = config["mode"]

        if mode == "none":
            normalized = x
        else:
            mean = torch.as_tensor(
                config["mean"],
                dtype=x.dtype,
                device=x.device,
            )
            std = torch.as_tensor(
                config["std"],
                dtype=x.dtype,
                device=x.device,
            )

            if mode == "channel":
                mean = mean.reshape(1, -1, 1)
                std = std.reshape(1, -1, 1)

            normalized = (x - mean) / std

        # Missing values remain exactly zero.
        result[modality] = torch.where(
            mask,
            normalized,
            torch.zeros_like(normalized),
        )

    return result


def inverse_normalize(
    prediction,
    stats,
    modality="scalp",
):
    """Convert normalized predictions back to original EEG units."""
    config = stats[modality]

    if config["mode"] == "none":
        return prediction

    mean = torch.as_tensor(
        config["mean"],
        dtype=prediction.dtype,
        device=prediction.device,
    )
    std = torch.as_tensor(
        config["std"],
        dtype=prediction.dtype,
        device=prediction.device,
    )

    if config["mode"] == "channel":
        mean = mean.reshape(1, -1, 1)
        std = std.reshape(1, -1, 1)

    return prediction * std + mean
