
"""Inspect the preprocessed Ear-SAAD dataset."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

DATA_DIR = Path("data/raw")


def main():
    files = sorted(DATA_DIR.rglob("dataSubject*.mat"))

    print("=" * 60)
    print("EAR-SAAD DATASET INSPECTION")
    print("=" * 60)

    print(f"\nSubject files found: {len(files)}")

    if not files:
        raise FileNotFoundError(
            "No dataSubject*.mat files found in data/raw/"
        )

    # Inspect one subject before building the DataLoader
    path = files[0]
    print(f"\nInspecting: {path}")

    print("\nMATLAB variables:")
    for name, shape, dtype in whosmat(path):
        print(f"  {name}: shape={shape}, type={dtype}")

    data = loadmat(path, simplify_cells=True)

    print("\nAvailable keys:")
    print([key for key in data if not key.startswith("__")])

    fs = float(np.asarray(data["fs"]).squeeze())
    print(f"\nSampling frequency: {fs} Hz")

    channels = np.asarray(data["channels"]).reshape(-1)
    channels = [str(ch) for ch in channels]

    print(f"\nTotal channel labels: {len(channels)}")
    for i, name in enumerate(channels):
        print(f"  {i:02d}: {name}")

    trials = data["eegTrials"]

    if isinstance(trials, np.ndarray) and trials.dtype == object:
        trials = trials.flat
    elif isinstance(trials, (list, tuple)):
        trials = iter(trials)
    else:
        trials = iter([trials])

    trial = np.asarray(next(trials))

    print("\nFirst trial:")
    print(f"  Shape: {trial.shape}")
    print(f"  Dtype: {trial.dtype}")
    print(f"  Duration: {trial.shape[0] / fs:.1f} s")

    print("\nSignal quality:")
    print(f"  NaN fraction: {np.isnan(trial).mean():.4%}")
    print(f"  Zero fraction: {np.mean(trial == 0):.4%}")
    print(f"  Finite fraction: {np.isfinite(trial).mean():.4%}")

    assert trial.ndim == 2
    assert trial.shape[1] == len(channels), (
        "Channel labels do not match signal dimensions"
    )

    print("\nDataset inspection completed successfully.")


if __name__ == "__main__":
    main()
