
"""Evaluate zero and channel-wise mean EEG reconstruction baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.models.train_val_test_split import make_patient_splits


def make_loader(data_dir, subject_ids, batch_size):
    dataset = EarSAADNumpyDataset(
        data_dir=data_dir,
        subject_ids=subject_ids,
        window_seconds=4.0,
        stride_seconds=4.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def fit_channel_means(train_loader):
    """Fit one mean per scalp channel using observed training data only."""
    channel_sum = None
    channel_count = None

    for batch in train_loader:
        y = batch["scalp"].double()
        mask = batch["target_mask"].bool()

        current_sum = (y * mask).sum(dim=(0, 2))
        current_count = mask.sum(dim=(0, 2))

        if channel_sum is None:
            channel_sum = current_sum
            channel_count = current_count
        else:
            channel_sum += current_sum
            channel_count += current_count

    if channel_sum is None:
        raise ValueError("Empty training dataset")

    if (channel_count == 0).any():
        raise ValueError("Some scalp channels have no observed training data")

    return (channel_sum / channel_count).numpy()


def evaluate_baseline(loader, channel_means=None):
    """Compute globally masked MSE, RMSE and MAE."""
    squared_error = 0.0
    absolute_error = 0.0
    observed_count = 0

    for batch in loader:
        target = batch["scalp"].double()
        mask = batch["target_mask"].bool()

        if channel_means is None:
            prediction = target.new_zeros(target.shape)
        else:
            prediction = target.new_tensor(
                channel_means
            ).reshape(1, -1, 1).expand_as(target)

        error = prediction - target

        squared_error += (
            error.square() * mask
        ).sum().item()

        absolute_error += (
            error.abs() * mask
        ).sum().item()

        observed_count += mask.sum().item()

    if observed_count == 0:
        raise ValueError("No observed target samples")

    mse = squared_error / observed_count

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(absolute_error / observed_count),
        "observed_values": int(observed_count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/processed/numpy",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        default="results/baselines",
    )
    args = parser.parse_args()

    splits = make_patient_splits(range(1, 16))

    if not 1 <= args.fold <= len(splits):
        raise ValueError("Fold must be between 1 and 15")

    split = splits[args.fold - 1]

    train_loader = make_loader(
        args.data_dir,
        split["train"],
        args.batch_size,
    )
    val_loader = make_loader(
        args.data_dir,
        split["val"],
        args.batch_size,
    )

    print(f"Fold: {args.fold}")
    print(f"Train subjects: {split['train']}")
    print(f"Validation subjects: {split['val']}")
    print(f"Reserved test subjects: {split['test']}")
    print(f"Train windows: {len(train_loader.dataset)}")
    print(f"Val windows: {len(val_loader.dataset)}")

    print("\nFitting channel-wise mean predictor...")
    channel_means = fit_channel_means(train_loader)

    print("Evaluating baselines...")
    zero_results = evaluate_baseline(val_loader)
    mean_results = evaluate_baseline(
        val_loader,
        channel_means=channel_means,
    )

    results = {
        "fold": args.fold,
        "train_subjects": split["train"],
        "val_subjects": split["val"],
        "test_subjects": split["test"],
        "train_windows": len(train_loader.dataset),
        "val_windows": len(val_loader.dataset),
        "zero_predictor": zero_results,
        "channel_mean_predictor": mean_results,
        "channel_means": channel_means.tolist(),
    }

    print("\n=== Validation Results ===")
    print(f"{'Model':<22} {'MSE':>12} {'RMSE':>12} {'MAE':>12}")
    print("-" * 62)

    for name, metrics in [
        ("Zero", zero_results),
        ("Channel mean", mean_results),
    ]:
        print(
            f"{name:<22} "
            f"{metrics['mse']:>12.4f} "
            f"{metrics['rmse']:>12.4f} "
            f"{metrics['mae']:>12.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"fold_{args.fold}.json"
    output_path.write_text(json.dumps(results, indent=2))

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
