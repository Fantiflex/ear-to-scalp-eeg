
"""Diagnose EAR-SAAD shared Fp1 signals and CAR signatures."""

import numpy as np
from scipy.io import loadmat

PATH = "data/raw/preprocessedData/dataSubject2.mat"

GROUPS = {
    "scalp": np.arange(0, 29),
    "in_ear": np.r_[30:36, 45:51],
    "left_ear": np.arange(30, 36),
    "right_ear": np.arange(45, 51),
    "around_ear": np.r_[36:45, 51:61],
}


def channel_first(trial):
    x = np.asarray(trial, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Unexpected shape: {x.shape}")
    if x.shape[0] == 61:
        return x
    if x.shape[1] == 61:
        return x.T
    raise ValueError(f"Expected 61 channels, got {x.shape}")


data = loadmat(PATH, simplify_cells=True)
trials = [
    channel_first(t)
    for t in np.atleast_1d(data["eegTrials"]).ravel()
]

# Flatten for descriptive statistics only.
x = np.concatenate(trials, axis=1)

fp1 = x[14]
fp1_cr = x[29]
valid = np.isfinite(fp1) & np.isfinite(fp1_cr)

a, b = fp1[valid], fp1_cr[valid]

print("=== Shared Fp1 diagnostic ===")
print("Valid samples:", len(a))
print("Fp1 std:", np.std(a))
print("Fp1-cr std:", np.std(b))
print("Pearson:", np.corrcoef(a, b)[0, 1])
print("Difference std:", np.std(a - b))
print("Difference mean:", np.mean(a - b))

print("\n=== CAR signature ===")

for name, indices in GROUPS.items():
    signal = x[indices]
    complete = np.isfinite(signal).all(axis=0)
    selected = signal[:, complete]

    if selected.shape[1] == 0:
        print(f"{name}: no complete samples")
        continue

    channel_mean = selected.mean(axis=0)
    residual = np.mean(np.abs(channel_mean))
    amplitude = np.mean(np.abs(selected))

    print(
        f"{name:12s} | "
        f"n={selected.shape[1]:6d} | "
        f"relative CAR residual={residual / max(amplitude, 1e-12):.5f}"
    )
    print("\n=== Trial-wise Fp1 diagnostic ===")

    for i, trial in enumerate(trials):
        a = trial[14]
        b = trial[29]

        valid = np.isfinite(a) & np.isfinite(b)

        if valid.sum() < 2:
            print(f"Trial {i + 1}: insufficient data")
            continue

        a, b = a[valid], b[valid]

        std_a = np.std(a)
        std_b = np.std(b)

        corr = (
            np.corrcoef(a, b)[0, 1]
            if std_a > 0 and std_b > 0
            else np.nan
        )

        print(
            f"Trial {i + 1} | "
            f"valid={valid.mean():.2%} | "
            f"Pearson={corr:.4f} | "
            f"std Fp1={std_a:.4f} | "
            f"std Fp1-cr={std_b:.4f} | "
            f"std diff={np.std(a-b):.4f}"
        )
        print("\n=== Spatial dimensionality diagnostic ===")

    for trial_id, trial in enumerate(trials, start=1):

        print(f"\n--- Trial {trial_id} ---")

        for name, indices in GROUPS.items():

            signal = trial[indices]

            # Remove channels with insufficient observations
            keep = np.isfinite(signal).mean(axis=1) >= 0.80
            signal = signal[keep]

            if signal.shape[0] < 2:
                print(f"{name}: insufficient channels")
                continue

            # Use jointly valid samples
            complete = np.isfinite(signal).all(axis=0)
            signal = signal[:, complete]

            if signal.shape[1] < 100:
                print(f"{name}: insufficient samples")
                continue

            # Temporal centering
            signal = signal - signal.mean(
                axis=1, keepdims=True
            )

            # Singular value decomposition
            singular_values = np.linalg.svd(
                signal,
                full_matrices=False,
                compute_uv=False,
            )

            eigenvalues = singular_values ** 2

            if eigenvalues.sum() <= 0:
                continue

            p = eigenvalues / eigenvalues.sum()

            cumulative = np.cumsum(p)

            rank_90 = np.searchsorted(
                cumulative, 0.90
            ) + 1

            rank_95 = np.searchsorted(
                cumulative, 0.95
            ) + 1

            effective_rank = np.exp(
                -np.sum(p * np.log(p + 1e-12))
            )

            print(
                f"{name:12s} | "
                f"channels={signal.shape[0]:2d} | "
                f"samples={signal.shape[1]:5d} | "
                f"PC1={p[0]:.2%} | "
                f"rank90={rank_90:2d} | "
                f"rank95={rank_95:2d} | "
                f"effective_rank={effective_rank:.2f}"
            )