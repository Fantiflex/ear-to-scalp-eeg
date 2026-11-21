
"""Mask-aware EEG re-referencing for EAR-SAAD in-ear channels."""

import torch

VALID_MODES = {"none", "car", "same_ear", "other_ear"}

# Order in the in-ear dataset:
# 0:6   -> ELA, ELB, ELC, ELE, ELI, ELT
# 6:12  -> ERA, ERB, ERC, ERE, ERI, ERT
LEFT = slice(0, 6)
RIGHT = slice(6, 12)


def _masked_reference(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a mean reference using observed channels only.

    Args:
        x: EEG signals, shape (B, C, T).
        mask: Observed-sample mask, shape (B, C, T).

    Returns:
        reference: Shape (B, 1, T).
        valid: Shape (B, 1, T), whether a reference exists.
    """
    count = mask.sum(dim=1, keepdim=True)
    valid = count > 0

    observed = torch.where(
        mask,
        x,
        torch.zeros_like(x),
    )

    reference = observed.sum(
        dim=1, keepdim=True
    ) / count.clamp_min(1)

    return reference, valid


def rereference_in_ear(
    x: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply re-referencing to the 12 in-ear EEG channels.

    Args:
        x: Tensor of shape (B, 12, T).
        mask: Boolean observed mask of the same shape.
        mode: none, car, same_ear, or other_ear.

    Returns:
        rereferenced: EEG with missing samples zero-filled.
        valid_mask: Mask of valid re-referenced samples.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported reference mode: {mode}")

    if x.ndim != 3 or x.shape[1] != 12:
        raise ValueError(
            f"Expected EEG shape (B, 12, T), got {tuple(x.shape)}"
        )

    if mask.shape != x.shape:
        raise ValueError("Mask must have the same shape as EEG")

    mask = mask.bool()

    if not torch.isfinite(x[mask]).all():
        raise ValueError("Observed EEG values must be finite")

    if mode == "none":
        return torch.where(
            mask, x, torch.zeros_like(x)
        ), mask.clone()

    if mode == "car":
        reference, ref_valid = _masked_reference(x, mask)

        valid_mask = mask & ref_valid

        output = torch.where(
            valid_mask,
            x - reference,
            torch.zeros_like(x),
        )

        return output, valid_mask

    # Compute separate left/right reference signals.
    left_reference, left_valid = _masked_reference(
        x[:, LEFT, :],
        mask[:, LEFT, :],
    )

    right_reference, right_valid = _masked_reference(
        x[:, RIGHT, :],
        mask[:, RIGHT, :],
    )

    if mode == "same_ear":
        ref_left, valid_left = left_reference, left_valid
        ref_right, valid_right = right_reference, right_valid
    else:
        # Each ear uses the opposite ear as its reference.
        ref_left, valid_left = right_reference, right_valid
        ref_right, valid_right = left_reference, left_valid

    left_mask = mask[:, LEFT, :] & valid_left
    right_mask = mask[:, RIGHT, :] & valid_right

    output_left = torch.where(
        left_mask,
        x[:, LEFT, :] - ref_left,
        torch.zeros_like(x[:, LEFT, :]),
    )

    output_right = torch.where(
        right_mask,
        x[:, RIGHT, :] - ref_right,
        torch.zeros_like(x[:, RIGHT, :]),
    )

    return (
        torch.cat([output_left, output_right], dim=1),
        torch.cat([left_mask, right_mask], dim=1),
    )


def rereference_batch(
    batch: dict,
    mode: str = "none",
) -> dict:
    """Return a batch with re-referenced in-ear EEG and valid masks.

    The scalp targets and all other modalities remain unchanged.
    """
    result = dict(batch)

    x, mask = rereference_in_ear(
        batch["in_ear"],
        batch["in_ear_observed_mask"],
        mode=mode,
    )

    result["in_ear"] = x

    # Keep the original acquisition mask separately.
    result["in_ear_raw_observed_mask"] = batch[
        "in_ear_observed_mask"
    ]

    # Downstream models must use the new validity mask.
    result["in_ear_observed_mask"] = mask
    result["in_ear_mask"] = mask

    return result
