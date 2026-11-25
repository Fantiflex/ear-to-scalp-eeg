
"""Identify channels responsible for spatial rank collapse."""

import numpy as np
from scipy.io import loadmat

PATH = "data/raw/preprocessedData/dataSubject2.mat"

data = loadmat(PATH, simplify_cells=True)

channels = np.asarray(data["channels"]).ravel()
trials = np.atleast_1d(data["eegTrials"]).ravel()

GROUPS = {
    "in_ear": np.r_[30:36, 45:51],
    "around_ear": np.r_[36:45, 51:61],
}


def effective_rank(x):
    x = x - x.mean(axis=1, keepdims=True)

    s = np.linalg.svd(
        x, compute_uv=False, full_matrices=False
    )

    energy = np.sum(s**2)

    if energy <= 1e-20:
        return np.nan

    p = s**2 / energy

    return np.exp(
        -np.sum(p * np.log(p + 1e-12))
    )


for trial_id, trial in enumerate(trials, start=1):
    x = np.asarray(trial, dtype=np.float64)

    if x.shape[0] != 61:
        x = x.T

    print(f"\n========== TRIAL {trial_id} ==========")

    for modality, indices in GROUPS.items():

        signal = x[indices]
        names = channels[indices]

        print(f"\n--- {modality} ---")

        for name, channel in zip(names, signal):
            valid = np.isfinite(channel)
            values = channel[valid]

            if len(values) == 0:
                print(f"{name:8s} | missing")
                continue

            centered = values - values.mean()

            rms = np.sqrt(np.mean(centered**2))

            mad = 1.4826 * np.median(
                np.abs(values - np.median(values))
            )

            print(
                f"{name:8s} | "
                f"valid={valid.mean():6.2%} | "
                f"RMS={rms:10.3f} | "
                f"MAD={mad:10.3f}"
            )

        keep = np.isfinite(signal).mean(axis=1) >= 0.80
        signal = signal[keep]
        names = names[keep]

        complete = np.isfinite(signal).all(axis=0)
        signal = signal[:, complete]

        if signal.shape[0] < 3 or signal.shape[1] < 100:
            continue

        signal -= signal.mean(axis=1, keepdims=True)

        u, s, _ = np.linalg.svd(
            signal, full_matrices=False
        )

        print("\nPC1 channel contributions:")

        contributions = u[:, 0] ** 2

        for i in np.argsort(contributions)[::-1]:
            print(
                f"{names[i]:8s} | "
                f"PC1 loading={u[i, 0]:+.4f} | "
                f"contribution={contributions[i]:.2%}"
            )

        print(
            f"\nOriginal effective rank: "
            f"{effective_rank(signal):.3f}"
        )

        print("Leave-one-channel-out diagnostic:")

        for i, name in enumerate(names):
            reduced = np.delete(signal, i, axis=0)

            print(
                f"Without {name:8s} | "
                f"rank={effective_rank(reduced):.3f}"
            )
