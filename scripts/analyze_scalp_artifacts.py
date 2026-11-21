
"""Analyze extreme scalp EEG amplitudes by channel and trial."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_subject(subject_id, data_dir, threshold):
    root = Path(data_dir)
    prefix = root / f"subject_{subject_id:02d}"

    meta = json.loads(
        prefix.with_suffix(".json").read_text()
    )
    signals = np.load(
        prefix.with_suffix(".npy"),
        mmap_mode="r",
    )

    scalp = signals[:, meta["indices"]["scalp"]]
    offsets = meta["trial_offsets"]

    valid = np.isfinite(scalp)
    extreme = valid & (np.abs(scalp) > threshold)

    print(f"\nSubject {subject_id}")
    print(f"Threshold: {threshold}")
    print(f"Total observed values: {valid.sum():,}")
    print(f"Extreme observations: {extreme.sum():,}")
    print(
        f"Fraction of observed values: "
        f"{extreme.sum() / valid.sum():.4%}"
    )

    print("\n=== Channels ===")

    for channel in range(scalp.shape[1]):
        count = int(extreme[:, channel].sum())

        if count:
            values = scalp[:, channel]
            print(
                f"Channel {channel:02d}: "
                f"{count} extreme values, "
                f"max={np.nanmax(np.abs(values)):.2f}"
            )

    print("\n=== Trials ===")

    for trial in range(len(offsets) - 1):
        start = offsets[trial]
        end = offsets[trial + 1]
        segment = scalp[start:end]

        count = int(extreme[start:end].sum())

        if count:
            print(
                f"Trial {trial + 1}: "
                f"{count} extreme values, "
                f"max={np.nanmax(np.abs(segment)):.2f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument(
        "--data-dir",
        default="data/processed/numpy",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1000.0,
    )

    args = parser.parse_args()

    analyze_subject(
        args.subject,
        args.data_dir,
        args.threshold,
    )


if __name__ == "__main__":
    main()
