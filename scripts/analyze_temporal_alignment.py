
"""Analyze lagged correlations between in-ear and scalp EEG."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.models.train_val_test_split import make_patient_splits


def lagged_correlation(x, y, mx, my, lag, min_samples=200):
    """Correlation between x[t] and y[t + lag]."""
    if lag > 0:
        x, y = x[:-lag], y[lag:]
        mx, my = mx[:-lag], my[lag:]
    elif lag < 0:
        k = -lag
        x, y = x[k:], y[:-k]
        mx, my = mx[k:], my[:-k]

    valid = mx & my & np.isfinite(x) & np.isfinite(y)

    if valid.sum() < min_samples:
        return np.nan

    xv = x[valid]
    yv = y[valid]

    if np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
        return np.nan

    return float(np.corrcoef(xv, yv)[0, 1])


def analyze_subject(data_dir, subject, max_lag=10):
    dataset = EarSAADNumpyDataset(
        data_dir=data_dir,
        subject_ids=[subject],
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    xs, ys, mxs, mys = [], [], [], []

    for batch in loader:
        xs.append(batch["in_ear"].numpy())
        ys.append(batch["scalp"].numpy())
        mxs.append(batch["in_ear_observed_mask"].numpy())
        mys.append(batch["scalp_observed_mask"].numpy())

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mx = np.concatenate(mxs).astype(bool)
    my = np.concatenate(mys).astype(bool)

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.full((len(lags), 12, 29), np.nan)

    for lag_idx, lag in enumerate(lags):
        for i in range(12):
            for j in range(29):
                # Calculate independently inside each 4-second window.
                values = []
                weights = []

                for w in range(len(x)):
                    corr = lagged_correlation(
                        x[w, i],
                        y[w, j],
                        mx[w, i],
                        my[w, j],
                        int(lag),
                        min_samples=20,
                    )
                    if np.isfinite(corr):
                        values.append(corr)
                        weights.append(
                            max(x.shape[-1] - abs(lag) - 3, 1)
                        )

                if values:
                    correlations[lag_idx, i, j] = np.average(
                        values, weights=weights
                    )

    return lags, correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/numpy")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    splits = make_patient_splits(range(1, 16))
    if not 1 <= args.fold <= len(splits):
        parser.error("Fold must be between 1 and 15")
    split = splits[args.fold - 1]

    subjects = [*split["train"], *split["val"]]
    subject_results = []

    for subject in subjects:
        print(f"Subject {subject}...", flush=True)
        lags, corr = analyze_subject(
            args.data_dir, subject, args.max_lag
        )
        subject_results.append(corr)

    all_corr = np.stack(subject_results)
    train_corr = np.nanmedian(
        all_corr[:len(split["train"])], axis=0
    )
    val_corr = np.nanmedian(
        all_corr[len(split["train"]):], axis=0
    )

    # Best absolute correlation across lag values.
    val_strength = np.abs(val_corr)
    best_idx = np.nanargmax(
        np.where(np.isfinite(val_strength), val_strength, -1),
        axis=0,
    )
    best_lags = lags[best_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output_dir / "lagged_correlations.npz",
        lags=lags,
        subjects=np.asarray(subjects),
        correlations=all_corr,
        train=train_corr,
        validation=val_corr,
        best_lags=best_lags,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    zero_idx = np.where(lags == 0)[0][0]

    axes[0].imshow(
        val_corr[zero_idx],
        vmin=-1, vmax=1,
        cmap="RdBu_r", aspect="auto",
    )
    axes[0].set_title("Validation: zero-lag correlation")
    axes[0].set_xlabel("Scalp channel")
    axes[0].set_ylabel("In-ear channel")

    image = axes[1].imshow(
        best_lags * 1000 / 20,
        cmap="coolwarm",
        vmin=-500,
        vmax=500,
        aspect="auto",
    )
    axes[1].set_title("Best lag (ms)")
    axes[1].set_xlabel("Scalp channel")
    axes[1].set_ylabel("In-ear channel")
    fig.colorbar(image, ax=axes[1])

    fig.tight_layout()
    fig.savefig(
        args.output_dir / "temporal_alignment.png",
        dpi=200,
    )
    plt.close(fig)

    zero_r = np.abs(val_corr[zero_idx])
    best_r = np.nanmax(np.abs(val_corr), axis=0)

    summary = {
        "median_zero_lag_abs_r": float(np.nanmedian(zero_r)),
        "median_best_lag_abs_r": float(np.nanmedian(best_r)),
        "median_best_lag_ms": float(
            np.median(best_lags) * 1000 / 20
        ),
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
