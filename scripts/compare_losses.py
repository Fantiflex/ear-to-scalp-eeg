"""Compare two trained EEG reconstruction models on the same subject/window."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
    TemporalConformer,
)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ConformerConfig(**checkpoint["model_config"])
    model = TemporalConformer(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, checkpoint


def predict(model, batch, modality, device):
    x = batch[modality].to(device)
    mask = batch[f"{modality}_observed_mask"].to(device)

    parameters = inspect.signature(model.forward).parameters

    if "observed_mask" in parameters:
        return model(x, observed_mask=mask)
    if len(parameters) >= 2:
        return model(x, mask)
    return model(x)


def compute_metrics(target, prediction, mask):
    valid = mask.astype(bool)
    global_mse = float(np.mean((target[valid] - prediction[valid]) ** 2))
    global_rmse = float(np.sqrt(global_mse))

    pearsons = []
    std_ratios = []

    n_channels = target.shape[1]
    for ch in range(n_channels):
        ch_valid = valid[:, ch, :]
        y = target[:, ch, :][ch_valid]
        p = prediction[:, ch, :][ch_valid]

        if len(y) < 2:
            pearsons.append(np.nan)
            std_ratios.append(np.nan)
            continue

        if np.std(y) > 1e-8 and np.std(p) > 1e-8:
            pearsons.append(float(np.corrcoef(y, p)[0, 1]))
        else:
            pearsons.append(np.nan)

        if np.std(y) > 1e-8:
            std_ratios.append(float(np.std(p) / np.std(y)))
        else:
            std_ratios.append(np.nan)

    return {
        "mse": global_mse,
        "rmse": global_rmse,
        "mean_pearson": float(np.nanmean(pearsons)),
        "mean_std_ratio": float(np.nanmean(std_ratios)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pearson-checkpoint", type=Path, required=True)
    parser.add_argument("--derivative-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", default="data/processed/numpy")
    parser.add_argument("--subject", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/compare_losses"),
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pearson_model, pearson_ckpt = load_model(
        args.pearson_checkpoint, device
    )
    derivative_model, derivative_ckpt = load_model(
        args.derivative_checkpoint, device
    )

    pearson_modality = pearson_ckpt["args"]["modality"]
    derivative_modality = derivative_ckpt["args"]["modality"]

    if pearson_modality != derivative_modality:
        raise ValueError(
            "The two checkpoints do not use the same modality: "
            f"{pearson_modality} vs {derivative_modality}"
        )

    modality = pearson_modality

    dataset = EarSAADNumpyDataset(
        data_dir=args.data_dir,
        subject_ids=[args.subject],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    targets = []
    masks = []
    pearson_predictions = []
    derivative_predictions = []

    with torch.inference_mode():
        for batch in loader:
            pearson_pred = predict(
                pearson_model, batch, modality, device
            )
            derivative_pred = predict(
                derivative_model, batch, modality, device
            )

            targets.append(batch["scalp"].numpy())
            masks.append(batch["target_mask"].numpy())
            pearson_predictions.append(pearson_pred.cpu().numpy())
            derivative_predictions.append(derivative_pred.cpu().numpy())

    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)
    pearson_prediction = np.concatenate(pearson_predictions, axis=0)
    derivative_prediction = np.concatenate(derivative_predictions, axis=0)

    pearson_metrics = compute_metrics(
        target, pearson_prediction, mask
    )
    derivative_metrics = compute_metrics(
        target, derivative_prediction, mask
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "subject": args.subject,
        "pearson_model": pearson_metrics,
        "derivative_model": derivative_metrics,
    }
    (args.output_dir / "metrics_comparison.json").write_text(
        json.dumps(metrics, indent=2)
    )

    # Reproduce the same style as evaluate_conformer.py:
    # use the first window of this subject.
    example_target = target[0]
    example_mask = mask[0]
    example_pearson = pearson_prediction[0]
    example_derivative = derivative_prediction[0]

    channels = [0, 14, 28]
    time = np.arange(example_target.shape[-1]) / dataset.fs

    fig, axes = plt.subplots(
        len(channels),
        1,
        figsize=(10, 8),
        sharex=True,
    )

    for ax, ch in zip(axes, channels):
        valid = example_mask[ch].astype(bool)

        actual = np.where(valid, example_target[ch], np.nan)
        pearson_pred = np.where(valid, example_pearson[ch], np.nan)
        derivative_pred = np.where(valid, example_derivative[ch], np.nan)

        ax.plot(time, actual, label="Ground truth")
        ax.plot(time, pearson_pred, label="Hybrid Pearson", alpha=0.85)
        ax.plot(
            time,
            derivative_pred,
            label="Temporal Derivative",
            alpha=0.85,
        )

        ax.set_ylabel(f"Ch {ch}")
        ax.grid(alpha=0.2)

    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Subject {args.subject} — Reconstruction comparison"
    )
    fig.tight_layout()
    fig.savefig(
        args.output_dir / f"comparison_subject_{args.subject}.png",
        dpi=300,
    )
    plt.close(fig)

    # Optional metrics comparison figure
    labels = ["MSE", "RMSE", "Mean Pearson", "Mean std ratio"]
    pearson_values = [
        pearson_metrics["mse"],
        pearson_metrics["rmse"],
        pearson_metrics["mean_pearson"],
        pearson_metrics["mean_std_ratio"],
    ]
    derivative_values = [
        derivative_metrics["mse"],
        derivative_metrics["rmse"],
        derivative_metrics["mean_pearson"],
        derivative_metrics["mean_std_ratio"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, pearson_values, width, label="Hybrid Pearson")
    ax.bar(
        x + width / 2,
        derivative_values,
        width,
        label="Temporal Derivative",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Subject {args.subject} — Metrics comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / f"metrics_subject_{args.subject}.png",
        dpi=300,
    )
    plt.close(fig)

    print("\n=== Hybrid Pearson ===")
    for key, value in pearson_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== Temporal Derivative ===")
    for key, value in derivative_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nSaved to:", args.output_dir)


if __name__ == "__main__":
    main()