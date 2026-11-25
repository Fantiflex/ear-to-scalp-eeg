
"""Test whether common-mode removal reveals spatial EEG information."""

import numpy as np
from scipy.io import loadmat

PATH = "data/raw/preprocessedData/dataSubject2.mat"

GROUPS = {
    "in_ear": np.r_[30:36, 45:51],
    "left_ear": np.arange(30, 36),
    "right_ear": np.arange(45, 51),
    "around_ear": np.r_[36:45, 51:61],
    "all_ear": np.arange(30, 61),
}


def spatial_metrics(x):
    x = x - x.mean(axis=1, keepdims=True)

    s = np.linalg.svd(
        x,
        compute_uv=False,
        full_matrices=False,
    )

    energy = np.sum(s**2)

    if energy <= 1e-20:
        return np.nan, np.nan

    p = s**2 / energy

    effective_rank = np.exp(
        -np.sum(p * np.log(p + 1e-12))
    )

    return p[0], effective_rank


data = loadmat(PATH, simplify_cells=True)
trials = np.atleast_1d(data["eegTrials"]).ravel()

for trial_id, trial in enumerate(trials, start=1):
    x = np.asarray(trial, dtype=np.float64)

    if x.shape[0] != 61:
        x = x.T

    if x.shape[0] != 61:
        raise ValueError(f"Unexpected shape: {x.shape}")

    print(f"\n=== Trial {trial_id} ===")

    for name, indices in GROUPS.items():
        signal = x[indices]

        keep = np.isfinite(signal).mean(axis=1) >= 0.80
        signal = signal[keep]

        complete = np.isfinite(signal).all(axis=0)
        signal = signal[:, complete]

        if signal.shape[0] < 2 or signal.shape[1] < 100:
            continue

        signal -= signal.mean(axis=1, keepdims=True)

        # Common-mode component
        common = signal.mean(axis=0, keepdims=True)

        # Common average referencing
        residual = signal - common

        pc1_before, rank_before = spatial_metrics(signal)
        pc1_after, rank_after = spatial_metrics(residual)

        energy_before = np.sum(signal**2)
        energy_after = np.sum(residual**2)

        retained_energy = (
            energy_after / max(energy_before, 1e-20)
        )

        rms_before = np.sqrt(np.mean(signal**2))
        rms_after = np.sqrt(np.mean(residual**2))

        print(
            f"{name:12s} | "
            f"PC1: {pc1_before:.2%} -> {pc1_after:.2%} | "
            f"rank: {rank_before:.2f} -> {rank_after:.2f} | "
            f"retained energy: {retained_energy:.4%} | "
            f"RMS: {rms_before:.4f} -> {rms_after:.4f}"
        )
