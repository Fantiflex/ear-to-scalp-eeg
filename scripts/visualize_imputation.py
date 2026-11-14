"""Visualize EAR-SAAD interpolation and evaluate synthetic temporal gaps.

Input: .npz containing `eeg` and `observed_mask`, each (channels, time)
(or a single (1, channels, time) batch), plus optional `channel_names`.
The original mask must mark *real* observations, not imputed positions.

Example:
    python scripts/visualize_imputation.py --input scratch/ear_sample.npz \
        --output-dir results/imputation --max-gap-samples 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eeg_reconstruction.data.imputation import interpolate_short_gaps


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        if "eeg" not in archive or "observed_mask" not in archive:
            raise ValueError("NPZ requires 'eeg' and 'observed_mask' arrays")
        eeg = np.asarray(archive["eeg"], dtype=np.float32)
        mask = np.asarray(archive["observed_mask"])
        names = archive["channel_names"].astype(str).tolist() if "channel_names" in archive else None
    if eeg.ndim == 3 and eeg.shape[0] == 1:
        eeg, mask = eeg[0], mask[0]
    if eeg.ndim != 2 or mask.shape != eeg.shape:
        raise ValueError("Expected eeg and observed_mask with matching (channels, time) shape")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("observed_mask must be 0/1")
    if not np.isfinite(eeg[mask.astype(bool)]).all():
        raise ValueError("Observed entries must be finite")
    if names is None:
        names = [f"Ch {i + 1}" for i in range(eeg.shape[0])]
    if len(names) != eeg.shape[0]:
        raise ValueError("channel_names length must match channel count")
    return eeg, mask.astype(bool), names


def plot_example(
    eeg: np.ndarray, mask: np.ndarray, filled: np.ndarray, interpolated: np.ndarray,
    names: list[str], sample_rate: float, channel: int, output_dir: Path,
) -> None:
    t = np.arange(eeg.shape[1]) / sample_rate
    fig, ax = plt.subplots(figsize=(11, 4))
    raw = np.where(mask[channel], eeg[channel], np.nan)
    reconstructed = np.where(interpolated[channel], filled[channel], np.nan)
    ax.plot(t, filled[channel], color="0.45", lw=1, alpha=0.65, label="Filled (zero for remaining gaps)")
    ax.plot(t, raw, lw=1.8, label="Observed EEG")
    ax.scatter(t[interpolated[channel]], reconstructed[interpolated[channel]], s=24,
               marker="o", label="Interpolated")
    ax.scatter(t[~(mask[channel] | interpolated[channel])],
               np.zeros(np.count_nonzero(~(mask[channel] | interpolated[channel]))),
               marker="x", s=22, label="Still missing (zero-filled)")
    ax.set(title=f"Temporal interpolation — {names[channel]}", xlabel="Time (s)", ylabel="EEG amplitude (preprocessed)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "before_after.png", dpi=180)
    plt.close(fig)


def plot_availability(mask: np.ndarray, interpolated: np.ndarray, names: list[str], output_dir: Path) -> None:
    observed = mask.mean(axis=1) * 100
    reconstructed = interpolated.mean(axis=1) * 100
    remaining = 100 - observed - reconstructed
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.36)))
    ax.barh(y, observed, label="Observed")
    ax.barh(y, reconstructed, left=observed, label="Interpolated")
    ax.barh(y, remaining, left=observed + reconstructed, label="Still missing")
    ax.set(yticks=y, yticklabels=names, xlim=(0, 100), xlabel="Samples (%)", title="Data availability per channel")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "availability_per_channel.png", dpi=180)
    plt.close(fig)


def synthetic_gap_experiment(
    eeg: np.ndarray, mask: np.ndarray, gap_lengths: list[int],
    max_gap_samples: int, seed: int, repetitions: int,
) -> list[tuple[int, int, float, float]]:
    """Mask wholly observed intervals, never using truly missing samples as truth.

    Non-overlapping synthetic gaps per trial; exclude window edges and require valid
    anchors. Repeated random trials are for descriptive diagnostics, not independent
    subject-level uncertainty estimates.
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, float, float]] = []
    n_time = eeg.shape[1]
    for length in gap_lengths:
        candidates = [
            (ch, start)
            for ch in range(eeg.shape[0])
            for start in range(1, n_time - length)
            if mask[ch, start - 1:start + length + 1].all()
        ]
        if not candidates:
            rows.append((length, 0, float("nan"), float("nan")))
            continue
        squared_errors: list[float] = []
        true_values: list[float] = []
        predictions: list[float] = []
        # Each trial starts from the original mask, so gaps never overlap.
        for _ in range(repetitions):
            ch, start = candidates[int(rng.integers(len(candidates)))]
            synthetic_mask = mask.copy()
            synthetic_mask[ch, start:start + length] = False
            filled, imputed = interpolate_short_gaps(eeg, synthetic_mask, max_gap_samples)
            if not imputed[ch, start:start + length].all():
                continue  # Do not evaluate unreconstructed positions.
            truth = eeg[ch, start:start + length].astype(float)
            prediction = filled[ch, start:start + length].astype(float)
            squared_errors.extend(((prediction - truth) ** 2).tolist())
            true_values.extend(truth.tolist())
            predictions.extend(prediction.tolist())
        n = len(true_values)
        if n < 2:
            rows.append((length, n, float("nan"), float("nan")))
            continue
        rmse = float(np.sqrt(np.mean(squared_errors)))
        pearson = (
            float(np.corrcoef(true_values, predictions)[0, 1])
            if np.std(true_values) > 0 and np.std(predictions) > 0
            else float("nan")
        )
        rows.append((length, n, rmse, pearson))
    return rows


def plot_quality(rows: list[tuple[int, int, float, float]], sample_rate: float, output_dir: Path) -> None:
    lengths = [r[0] / sample_rate for r in rows]
    rmse = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lengths, rmse, marker="o")
    ax.set(xlabel="Artificial gap duration (s)", ylabel="RMSE on hidden observed values",
           title="Temporal interpolation quality vs gap duration")
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_gap_rmse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ file with eeg and observed_mask")
    parser.add_argument("--output-dir", type=Path, default=Path("results/imputation"))
    parser.add_argument("--sampling-rate", type=float, default=20)
    parser.add_argument("--max-gap-samples", type=int, default=10)
    parser.add_argument("--channel", type=int, default=None, help="Zero-based index; default: channel with most interpolated points")
    parser.add_argument("--gap-lengths", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sampling_rate <= 0 or args.repetitions <= 0 or any(g <= 0 for g in args.gap_lengths):
        parser.error("sampling-rate, repetitions and gap-lengths must be positive")
    eeg, mask, names = load_recording(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filled, interpolated = interpolate_short_gaps(eeg, mask, args.max_gap_samples)
    channel = args.channel if args.channel is not None else int(interpolated.sum(axis=1).argmax())
    if not 0 <= channel < eeg.shape[0]:
        parser.error("channel index out of range")
    plot_example(eeg, mask, filled, interpolated, names, args.sampling_rate, channel, args.output_dir)
    plot_availability(mask, interpolated, names, args.output_dir)
    rows = synthetic_gap_experiment(eeg, mask, args.gap_lengths, args.max_gap_samples, args.seed, args.repetitions)
    plot_quality(rows, args.sampling_rate, args.output_dir)
    csv_path = args.output_dir / "synthetic_gap_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write("gap_samples,gap_seconds,evaluated_samples,rmse,pearson\n")
        for gap, n, rmse, corr in rows:
            stream.write(f"{gap},{gap / args.sampling_rate:.4f},{n},{rmse},{corr}\n")
    print(f"Input: {args.input} | shape={eeg.shape} | observed={mask.mean():.2%} | interpolated={interpolated.mean():.2%}")
    print(f"Figures and metrics: {args.output_dir}")
    for gap, n, rmse, corr in rows:
        print(f"  gap={gap} samples | n={n} | RMSE={rmse:.4g} | Pearson={corr:.4g}")


if __name__ == "__main__":
    main()
