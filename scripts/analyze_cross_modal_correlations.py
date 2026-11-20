
"""Analyze in-ear to scalp EEG correlations across subjects."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.models.train_val_test_split import make_patient_splits


def subject_correlations(data_dir, subject, batch_size=64):
    dataset = EarSAADNumpyDataset(
        data_dir=data_dir,
        subject_ids=[subject],
        window_seconds=4.0,
        stride_seconds=4.0,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    inputs, targets = [], []
    input_masks, target_masks = [], []

    for batch in loader:
        inputs.append(batch["in_ear"].numpy())
        targets.append(batch["scalp"].numpy())
        input_masks.append(batch["in_ear_observed_mask"].numpy())
        target_masks.append(batch["scalp_observed_mask"].numpy())

    x = np.concatenate(inputs).transpose(1, 0, 2).reshape(12, -1)
    y = np.concatenate(targets).transpose(1, 0, 2).reshape(29, -1)

    mx = np.concatenate(input_masks).transpose(1, 0, 2).reshape(12, -1)
    my = np.concatenate(target_masks).transpose(1, 0, 2).reshape(29, -1)

    correlations = np.full((12, 29), np.nan)
    pair_counts = np.zeros((12, 29), dtype=int)

    for i in range(12):
        for j in range(29):
            valid = (
                mx[i].astype(bool)
                & my[j].astype(bool)
                & np.isfinite(x[i])
                & np.isfinite(y[j])
            )

            n = int(valid.sum())
            pair_counts[i, j] = n

            if n < 200:
                continue

            xi = x[i, valid]
            yj = y[j, valid]

            if np.std(xi) < 1e-8 or np.std(yj) < 1e-8:
                continue

            correlations[i, j] = np.corrcoef(xi, yj)[0, 1]

    return correlations, pair_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/numpy")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    splits = make_patient_splits(range(1, 16))
    if not 1 <= args.fold <= len(splits):
        parser.error("Fold must be between 1 and 15")

    split = splits[args.fold - 1]
    train_subjects = split["train"]
    val_subjects = split["val"]

    all_correlations = {}
    all_counts = {}

    for subject in [*train_subjects, *val_subjects]:
        print(f"Analyzing subject {subject}...", flush=True)
        corr, counts = subject_correlations(
            args.data_dir, subject
        )
        all_correlations[subject] = corr
        all_counts[subject] = counts

    def aggregate(subjects):
        stack = np.stack([all_correlations[s] for s in subjects])
        # Median across subjects, ignoring unavailable pairs.
        return np.ma.median(
            np.ma.masked_invalid(stack), axis=0
        ).filled(np.nan)

    train_corr = aggregate(train_subjects)
    val_corr = aggregate(val_subjects)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output_dir / "correlations.npz",
        train=train_corr,
        validation=val_corr,
        subject_ids=np.array(list(all_correlations)),
        subject_correlations=np.stack(list(all_correlations.values())),
        subject_pair_counts=np.stack(list(all_counts.values())),
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(19, 6), constrained_layout=True
    )

    images = [
        (train_corr, "Training subjects", -1, 1),
        (val_corr, "Validation subjects", -1, 1),
        (val_corr - train_corr, "Validation - Training", -1, 1),
    ]

    for ax, (matrix, title, vmin, vmax) in zip(axes, images):
        im = ax.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="lower",
        )
        ax.set_title(title)
        ax.set_xlabel("Scalp channel")
        ax.set_ylabel("In-ear channel")
        ax.set_xticks(range(0, 29, 2))
        ax.set_yticks(range(12))
        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.savefig(
        args.output_dir / "cross_modal_correlations.png",
        dpi=200,
    )
    plt.close(fig)

    def stats(matrix):
        finite = np.abs(matrix[np.isfinite(matrix)])
        return {
            "median_absolute_correlation": float(np.median(finite)),
            "fraction_above_0.3": float(np.mean(finite > 0.3)),
            "fraction_above_0.5": float(np.mean(finite > 0.5)),
            "valid_pairs": int(finite.size),
        } if finite.size else {"valid_pairs": 0}

    summary = {
        "fold": args.fold,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train": stats(train_corr),
        "validation": stats(val_corr),
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
