"""Audit in-ear EEG channel artifacts in processed EAR-SAAD data.

Descriptive only: never modifies EEG signals, masks, or evaluation splits.
"""

import argparse
import csv
import json
import warnings
from itertools import pairwise
from pathlib import Path

import numpy as np

CHANNELS = (
    "ELA", "ELB", "ELC", "ELE", "ELI", "ELT",
    "ERA", "ERB", "ERC", "ERE", "ERI", "ERT",
)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_subject(data_dir, subject_id, threshold, zero_tol):
    stem = Path(data_dir) / f"subject_{subject_id:02d}"
    metadata = json.loads(stem.with_suffix(".json").read_text())
    signals = np.load(stem.with_suffix(".npy"), mmap_mode="r")
    indices = metadata["indices"]["in_ear"]
    offsets = metadata["trial_offsets"]
    if len(indices) != len(CHANNELS):
        raise ValueError(f"Subject {subject_id}: unexpected in-ear channel count")
    if signals.ndim != 2 or signals.shape[1] <= max(indices):
        raise ValueError(f"Subject {subject_id}: unexpected array shape {signals.shape}")
    if offsets[0] != 0 or offsets[-1] != signals.shape[0]:
        raise ValueError(f"Subject {subject_id}: inconsistent trial offsets")

    channel_rows = []
    trial_rows = []
    for trial_idx, (start, end) in enumerate(
        pairwise(offsets), 1
    ):
        # Stored signals are (samples, all_channels), including NaN missingness.
        x = np.asarray(signals[start:end, indices], dtype=np.float64)
        valid = np.isfinite(x)
        observed = np.where(valid, x, np.nan)

        for idx, name in enumerate(CHANNELS):
            values = x[valid[:, idx], idx]
            absolute = np.abs(values)
            n = values.size
            channel_rows.append({
                "subject": subject_id,
                "trial": trial_idx,
                "channel": name,
                "observed": int(n),
                "observed_pct": float(100 * n / len(x)),
                "near_zero_pct_observed": float(100 * np.mean(absolute <= zero_tol)) if n else np.nan,
                "std_observed": float(np.std(values)) if n else np.nan,
                "median_observed": float(np.median(values)) if n else np.nan,
                "p99_abs": float(np.percentile(absolute, 99)) if n else np.nan,
                "max_abs": float(np.max(absolute)) if n else np.nan,
                "n_above_threshold": int(np.count_nonzero(absolute > threshold)),
                "pct_above_threshold_observed": float(100 * np.mean(absolute > threshold)) if n else np.nan,
            })

        # Investigate a common-mode transient without letting the extreme ELB
        # channel dominate. Exclude channels effectively constant throughout
        # this trial, but don't interpret exclusion as an EEG cleaning step.
        candidate = [
            i for i, name in enumerate(CHANNELS)
            if name != "ELB"
            and np.count_nonzero(valid[:, i]) >= 20
            and np.nanstd(observed[:, i]) > zero_tol
        ]
        if candidate:
            selected = observed[:, candidate]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Count simultaneous high-amplitude channels of the same sign.
                positive = np.sum(selected > threshold, axis=1)
                negative = np.sum(selected < -threshold, axis=1)
                available = np.sum(np.isfinite(selected), axis=1)
                coherent = (available >= 5) & (np.maximum(positive, negative) >= 5)
                # Median is robust to single-channel spikes.
                common = np.nanmedian(selected, axis=1)
            common_valid = np.isfinite(common) & (available >= 5)
            n_common = int(np.count_nonzero(common_valid))
            n_common_extreme = int(np.count_nonzero(common_valid & (np.abs(common) > threshold)))
        else:
            coherent = np.zeros(len(x), dtype=bool)
            n_common = 0
            n_common_extreme = 0

        trial_rows.append({
            "subject": subject_id,
            "trial": trial_idx,
            "n_times": len(x),
            "reference_candidate_channels": len(candidate),
            "common_valid_times": n_common,
            "n_common_median_above_threshold": n_common_extreme,
            "pct_common_median_above_threshold": float(100 * n_common_extreme / n_common) if n_common else np.nan,
            "n_simultaneous_5_channels_above_threshold": int(np.count_nonzero(coherent)),
        })

    return channel_rows, trial_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/numpy"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 16)))
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--zero-tol", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=Path("results/channel_artifacts"))
    args = parser.parse_args()
    if args.threshold <= 0 or args.zero_tol < 0:
        parser.error("--threshold must be >0 and --zero-tol must be >=0")

    all_channels, all_trials = [], []
    for subject in args.subjects:
        channels, trials = analyze_subject(args.data_dir, subject, args.threshold, args.zero_tol)
        all_channels.extend(channels)
        all_trials.extend(trials)
        print(f"Subject {subject}: analyzed {len(trials)} trials", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "channel_artifacts.csv", all_channels, list(all_channels[0]))
    write_csv(args.output_dir / "common_mode_by_trial.csv", all_trials, list(all_trials[0]))

    print("\n=== Most extreme channel/trial pairs ===")
    for row in sorted(all_channels, key=lambda r: r["max_abs"] if np.isfinite(r["max_abs"]) else -1, reverse=True)[:12]:
        print(f"S{row['subject']:02d} trial {row['trial']} {row['channel']}: "
              f"max={row['max_abs']:.1f}, >{args.threshold:g}={row['n_above_threshold']}, "
              f"near-zero={row['near_zero_pct_observed']:.1f}%")

    print("\n=== Most pronounced simultaneous multi-channel events ===")
    for row in sorted(all_trials, key=lambda r: r["n_simultaneous_5_channels_above_threshold"], reverse=True)[:10]:
        print(f"S{row['subject']:02d} trial {row['trial']}: "
              f"simultaneous timestamps={row['n_simultaneous_5_channels_above_threshold']}, "
              f"|median|>{args.threshold:g} timestamps={row['n_common_median_above_threshold']}")

    print(f"\nSaved: {args.output_dir / 'channel_artifacts.csv'}")
    print(f"Saved: {args.output_dir / 'common_mode_by_trial.csv'}")
    print("Thresholds are descriptive diagnostics, not automatic rejection rules.")


if __name__ == "__main__":
    main()
