
"""Missing-data preprocessing for EEG windows."""

import numpy as np


def interpolate_missing_channels(
    signal: np.ndarray,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate missing samples channel-wise.

    Parameters
    ----------
    signal : np.ndarray
        EEG window of shape (channels, time).
    min_valid_fraction : float
        Minimum fraction of observed samples required
        to retain a channel.

    Returns
    -------
    cleaned : np.ndarray
        Interpolated signal. Excluded channels are zero-filled.
    valid_channels : np.ndarray
        Boolean mask of shape (channels,) indicating
        which channels were retained.
    """
    if signal.ndim != 2:
        raise ValueError("Expected shape (channels, time)")

    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("Invalid minimum valid fraction")

    n_channels, n_samples = signal.shape

    cleaned = np.zeros(
        (n_channels, n_samples),
        dtype=np.float32,
    )
    valid_channels = np.zeros(n_channels, dtype=bool)

    time = np.arange(n_samples)

    for ch in range(n_channels):
        values = signal[ch]
        valid = np.isfinite(values)

        if valid.mean() < min_valid_fraction:
            continue

        if valid.sum() == 0:
            continue

        valid_channels[ch] = True

        # np.interp performs linear interpolation;
        # boundary gaps use the nearest valid value.
        cleaned[ch] = np.interp(
            time,
            time[valid],
            values[valid],
        ).astype(np.float32)

    return cleaned, valid_channels
