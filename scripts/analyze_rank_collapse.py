"""Audit spatial rank collapse across EEG subjects."""

import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATA_DIR = Path("data/raw/preprocessedData")
OUTPUT = Path("diagnostics/rank_collapse.csv")

GROUPS = {
    "scalp": np.arange(0, 29),
    "in_ear": np.r_[30:36, 45:51],
    "around_ear": np.r_[36:45, 51:61],
    "all_ear": np.r_[30:61],
}


def compute_metrics(signal, standardize=False):
    keep = np.isfinite(signal).mean(axis=1) >= 0.80
    signal = signal[keep]

    if signal.shape[0] < 2:
        return None

    complete = np.isfinite(signal).all(axis=0)
    signal = signal[:, complete]

    if signal.shape[1] < 100:
        return None

    signal = signal - signal.mean(
        axis=1, keepdims=True
    )

    std = signal.std(axis=1)
    signal = signal[std > 1e-12]
    std = std[std > 1e-12]

    if signal.shape[0] < 2:
        return None

    if standardize:
        signal = signal / std[:, None]

    s = np.linalg.svd(
        signal,
        compute_uv=False,
        full_matrices=False,
    )

    p = s**2 / np.sum(s**2)

    return {
        "channels": signal.shape[0],
        "samples": signal.shape[1],
        "pc1": float(p[0]),
        "rank90": int(
            np.searchsorted(np.cumsum(p), 0.90) + 1
        ),
        "effective_rank": float(
            np.exp(-np.sum(p * np.log(p + 1e-12)))
        ),
        "median_channel_std": float(np.median(std)),
    }


rows = []

for subject in range(1, 16):
    path = DATA_DIR / f"dataSubject{subject}.mat"

    if not path.exists():
        print(f"Missing subject {subject}")
        continue

    data = loadmat(path, simplify_cells=True)

    trials = np.atleast_1d(data["eegTrials"]).ravel()

    for trial_id, trial in enumerate(trials, start=1):
        x = np.asarray(trial, dtype=np.float64)

        if x.shape[0] != 61:
            x = x.T

        if x.shape[0] != 61:
            raise ValueError(f"Unexpected shape: {x.shape}")

        for modality, indices in GROUPS.items():
            for standardize in (False, True):

                metrics = compute_metrics(
                    x[indices],
                    standardize=standardize,
                )

                if metrics is None:
                    continue

                rows.append({
                    "subject": subject,
                    "trial": trial_id,
                    "modality": modality,
                    "standardized": standardize,
                    **metrics,
                    "rank_collapse": metrics["pc1"] >= 0.99,
                })

    print(f"Subject {subject}: done")


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if not rows:
    raise RuntimeError("No valid results")

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)


print("\n=== Rank collapse summary ===")

for modality in GROUPS:
    for standardize in (False, True):
        selected = [
            r for r in rows
            if r["modality"] == modality
            and r["standardized"] == standardize
        ]

        collapsed = [
            r for r in selected
            if r["rank_collapse"]
        ]

        print(
            f"{modality:12s} | "
            f"standardized={standardize} | "
            f"collapsed={len(collapsed)}/{len(selected)}"
        )

        for r in collapsed:
            print(
                f"  S{r['subject']:02d} "
                f"T{r['trial']} | "
                f"PC1={r['pc1']:.4%}"
            )

print(f"\nSaved to {OUTPUT}")