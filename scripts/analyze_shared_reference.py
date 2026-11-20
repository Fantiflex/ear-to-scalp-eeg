
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
