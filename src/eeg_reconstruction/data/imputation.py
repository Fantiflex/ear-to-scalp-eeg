"""Temporal gap interpolation for EEG windows.

All arrays use (channels, time) or (batch, channels, time).
The original observation mask is never changed. Only short *internal* gaps
are interpolated; long gaps, boundary gaps, and absent channels are zero-filled.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray


def interpolate_short_gaps(
    eeg: ArrayLike,
    observed_mask: ArrayLike,
    max_gap_samples: int = 10,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Interpolate short interior EEG gaps on each channel independently.

    Parameters
    ----------
    eeg:
        Shape (channels, time) or (batch, channels, time). Missing positions
        may already contain zeros from a dataloader; use observed_mask to
        identify them rather than inferring missingness from their values.
    observed_mask:
        Boolean/0-1 array with the exact same shape as eeg. True means
        genuinely observed, not previously interpolated.
    max_gap_samples:
        Maximum number of *consecutive missing samples* to interpolate.
        For 20 Hz EEG, 10 samples correspond to 0.5 seconds.

    Returns
    -------
    filled:
        float32 array, with short internal gaps interpolated and all other
        unobserved entries set to 0; observed entries preserved (up to casting).
    interpolated_mask:
        Boolean array, True only at samples reconstructed by interpolation.
        The input observed_mask is not altered.

    Notes
    -----
    This operation uses observations on both sides of a gap and is therefore
    non-causal. Interpolation is applied independently inside each window:
    gaps across window boundaries are not interpolated.
    """
    if isinstance(max_gap_samples, bool) or not isinstance(max_gap_samples, (int, np.integer)) or max_gap_samples < 0:
        raise ValueError("max_gap_samples must be a nonnegative integer")

    values = np.asarray(eeg, dtype=np.float32)
    mask = np.asarray(observed_mask)
    if values.ndim not in (2, 3):
        raise ValueError("eeg must have shape (channels, time) or (batch, channels, time)")
    if values.shape != mask.shape:
        raise ValueError("observed_mask must have the same shape as eeg")
    if mask.dtype != np.bool_ and not np.all(np.isin(mask, (0, 1))):
        raise ValueError("observed_mask must be boolean or contain only 0 and 1")
    mask = mask.astype(bool)
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed EEG values must be finite; check observed_mask")

    filled = np.where(mask, values, 0.0).astype(np.float32, copy=True)
    interpolated = np.zeros(mask.shape, dtype=bool)
    flat_values = filled.reshape(-1, filled.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1])
    flat_interpolated = interpolated.reshape(-1, interpolated.shape[-1])

    for row, available, imputed in zip(flat_values, flat_mask, flat_interpolated):
        indices = np.flatnonzero(available)
        for left, right in pairwise(indices):
            n_missing = int(right - left - 1)
            if 0 < n_missing <= max_gap_samples:
                # Using neighboring observed samples, not previously imputed values.
                row[left + 1:right] = np.linspace(
                    row[left], row[right], n_missing + 2, dtype=np.float32
                )[1:-1]
                imputed[left + 1:right] = True

    return filled, interpolated
