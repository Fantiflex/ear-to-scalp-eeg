
"""Analyze missing EEG segments before temporal interpolation.

Input:
    A .npz file or directory of .npz files containing:
      eeg: (channels, time) or (windows, channels, time)
      observed_mask: same shape, 1 = observed, 0 = missing
      subject_id: optional scalar or one ID per window

Outputs:
    gap_statistics.csv
    subject_statistics.csv
    channel_statistics.csv
    gap_length_distribution.png
    recoverable_missingness.png
    availability_by_subject.png

Example:
    python scripts/analyze_missing_gaps.py \
        --input scratch/ear_sample.npz \
        --output-dir results/missing_gaps \
        --sampling-rate 20
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_windows(path: Path):
    """Yield (subject_id, observed_mask) for each EEG window."""
    files = sorted(path.glob("*.npz")) if path.is_dir() else [path]

    if not files:
        raise FileNotFoundError(f"No NPZ files found in {path}")

    for file in files:
        with np.load(file, allow_pickle=False) as data:
            if "eeg" not in data or "observed_mask" not in data:
                raise ValueError(f"Missing eeg or observed_mask in {file}")

            eeg = data["eeg"]
            mask = data["observed_mask"]

            if eeg.shape != mask.shape:
                raise ValueError(f"Shape mismatch in {file}")

            if eeg.ndim == 2:
                mask = mask[None, ...]
            elif eeg.ndim != 3:
                raise ValueError("Expected (C,T) or (N,C,T)")

            if not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Invalid mask values in {file}")

            n_windows = len(mask)

            if "subject_id" in data:
                ids = np.asarray(data["subject_id"]).reshape(-1)
                if len(ids) == 1:
                    ids = np.repeat(ids, n_windows)
                elif len(ids) != n_windows:
                    raise ValueError("Invalid subject_id shape")
            else:
                ids = np.repeat(file.stem, n_windows)

            for subject_id, window_mask in zip(ids, mask):
                yield str(subject_id), window_mask.astype(bool)


def find_gaps(observed):
    """Identify consecutive missing intervals in one channel."""
    missing = ~observed
    padded = np.pad(missing.astype(np.int8), (1, 1))

    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)

    for start, end in zip(starts, ends):
        length = int(end - start)

        if not observed.any():
            kind = "absent_channel"
        elif start > 0 and end < len(observed):
            kind = "internal"
        else:
            kind = "boundary"

        yield int(start), int(end), length, kind


def analyze(windows, thresholds):
    gaps = []
    subjects = defaultdict(
        lambda: {
            "total": 0,
            "observed": 0,
            "missing": 0,
            "recoverable": defaultdict(int),
            "windows": 0,
        }
    )
    channels = defaultdict(
        lambda: {"total": 0, "observed": 0, "missing": 0}
    )

    total_windows = 0
    expected_channels = None

    for subject_id, mask in windows:
        n_channels, n_times = mask.shape

        if expected_channels is None:
            expected_channels = n_channels
        elif n_channels != expected_channels:
            raise ValueError("Inconsistent number of channels")

        total_windows += 1
        stats = subjects[subject_id]
        stats["windows"] += 1
        stats["total"] += mask.size
        stats["observed"] += int(mask.sum())
        stats["missing"] += int((~mask).sum())

        for channel_idx in range(n_channels):
            observed = mask[channel_idx]
            key = (subject_id, channel_idx)
            ch_stats = channels[key]

            ch_stats["total"] += n_times
            ch_stats["observed"] += int(observed.sum())
            ch_stats["missing"] += int((~observed).sum())

            for start, end, length, kind in find_gaps(observed):
                gaps.append(
                    {
                        "subject_id": subject_id,
                        "window": stats["windows"],
                        "channel": channel_idx,
                        "start": start,
                        "end": end,
                        "length": length,
                        "type": kind,
                    }
                )

                if kind == "internal":
                    for threshold in thresholds:
                        if length <= threshold:
                            stats["recoverable"][threshold] += length

    return gaps, subjects, channels, total_windows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, gaps, subjects, channels, thresholds, sr):
    output_dir.mkdir(parents=True, exist_ok=True)

    gap_rows = [
        {
            **gap,
            "duration_s": gap["length"] / sr,
        }
        for gap in gaps
    ]

    write_csv(
        output_dir / "gap_statistics.csv",
        gap_rows,
        [
            "subject_id", "window", "channel", "start",
            "end", "length", "type", "duration_s",
        ],
    )

    subject_rows = []
    for subject_id, stats in sorted(subjects.items()):
        row = {
            "subject_id": subject_id,
            "windows": stats["windows"],
            "observed_pct": 100 * stats["observed"] / stats["total"],
            "missing_pct": 100 * stats["missing"] / stats["total"],
        }

        for threshold in thresholds:
            recoverable = stats["recoverable"][threshold]
            row[f"recoverable_{threshold}_pct"] = (
                100 * recoverable / stats["missing"]
                if stats["missing"] else 0.0
            )

        subject_rows.append(row)

    write_csv(
        output_dir / "subject_statistics.csv",
        subject_rows,
        list(subject_rows[0]),
    )

    channel_rows = []
    for (subject_id, channel_idx), stats in sorted(channels.items()):
        channel_rows.append(
            {
                "subject_id": subject_id,
                "channel": channel_idx,
                "observed_pct": 100 * stats["observed"] / stats["total"],
                "missing_pct": 100 * stats["missing"] / stats["total"],
            }
        )

    write_csv(
        output_dir / "channel_statistics.csv",
        channel_rows,
        ["subject_id", "channel", "observed_pct", "missing_pct"],
    )

    return subject_rows


def plot_gap_lengths(gaps, sr, output_dir):
    if not gaps:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for kind in ("internal", "boundary", "absent_channel"):
        durations = [
            gap["length"] / sr
            for gap in gaps
            if gap["type"] == kind
        ]

        if durations:
            ax.hist(
                durations,
                bins=20,
                alpha=0.6,
                label=kind,
            )

    ax.set(
        xlabel="Missing gap duration (seconds)",
        ylabel="Number of gaps",
        title="Distribution of missing EEG gap durations",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gap_length_distribution.png", dpi=180)
    plt.close(fig)


def plot_recoverability(gaps, thresholds, output_dir):
    total_missing = sum(gap["length"] for gap in gaps)

    if total_missing == 0:
        return

    recovered = [
        sum(
            gap["length"]
            for gap in gaps
            if gap["type"] == "internal"
            and gap["length"] <= threshold
        )
        for threshold in thresholds
    ]

    percentages = [100 * n / total_missing for n in recovered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(t) for t in thresholds], percentages)

    ax.set(
        xlabel="Maximum interpolated gap length (samples)",
        ylabel="Recoverable missing samples (%)",
        title="Potential recovery using linear interpolation",
        ylim=(0, 100),
    )

    fig.tight_layout()
    fig.savefig(output_dir / "recoverable_missingness.png", dpi=180)
    plt.close(fig)


def plot_subjects(subject_rows, output_dir):
    labels = [row["subject_id"] for row in subject_rows]
    observed = [row["observed_pct"] for row in subject_rows]
    missing = [row["missing_pct"] for row in subject_rows]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x, observed, label="Observed")
    ax.bar(x, missing, bottom=observed, label="Missing")

    ax.set(
        xticks=x,
        xticklabels=labels,
        xlabel="Subject",
        ylabel="Samples (%)",
        title="EEG data availability by subject",
        ylim=(0, 100),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "availability_by_subject.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/missing_gaps"),
    )
    parser.add_argument("--sampling-rate", type=float, default=20.0)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[2, 5, 10],
    )
    args = parser.parse_args()

    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be positive")
    if not args.thresholds or any(t < 0 for t in args.thresholds):
        parser.error("--thresholds must contain nonnegative integers")

    thresholds = sorted(set(args.thresholds))

    gaps, subjects, channels, n_windows = analyze(
        load_windows(args.input),
        thresholds,
    )

    subject_rows = save_results(
        args.output_dir,
        gaps,
        subjects,
        channels,
        thresholds,
        args.sampling_rate,
    )

    plot_gap_lengths(gaps, args.sampling_rate, args.output_dir)
    plot_recoverability(gaps, thresholds, args.output_dir)
    plot_subjects(subject_rows, args.output_dir)

    print(f"Windows analyzed: {n_windows}")
    print(f"Subjects found: {len(subjects)}")
    print(f"Missing gaps: {len(gaps)}")
    print(f"Results saved to: {args.output_dir}")

    total_missing = sum(gap["length"] for gap in gaps)
    for threshold in thresholds:
        recoverable = sum(
            gap["length"]
            for gap in gaps
            if gap["type"] == "internal"
            and gap["length"] <= threshold
        )
        pct = 100 * recoverable / total_missing if total_missing else 0
        print(
            f"Threshold {threshold} samples: "
            f"{pct:.2f}% of missing samples recoverable"
        )


if __name__ == "__main__":
    main()
