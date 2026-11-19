
"""Evaluate EEG reconstruction quality on held-out validation subjects."""

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


def load_model(checkpoint_path, device, expected_fold):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,  # Only load checkpoints you trust.
    )

    if int(checkpoint["fold"]) != expected_fold:
        raise ValueError(
            f"Wrong checkpoint fold: {checkpoint['fold']}"
        )

    cfg = ConformerConfig(**checkpoint["model_config"])
    model = TemporalConformer(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, checkpoint


def predict(model, batch, modality, device):
    x = batch[modality].to(device)
    mask = batch[f"{modality}_observed_mask"].to(device)

    # Compatible with implementations using forward(x, mask)
    # and older implementations using forward(x).
    parameters = inspect.signature(model.forward).parameters

    if "observed_mask" in parameters:
        return model(x, observed_mask=mask)
    if len(parameters) >= 2:
        return model(x, mask)
    return model(x)


def calculate_metrics(target, prediction, mask):
    n_channels = target.shape[1]
    per_channel = []

    for ch in range(n_channels):
        valid = mask[:, ch, :]
        y = target[:, ch, :][valid]
        p = prediction[:, ch, :][valid]

        if len(y) == 0:
            mse = rmse = pearson = float("nan")
        else:
            mse = float(np.mean((y - p) ** 2))
            rmse = float(np.sqrt(mse))

            if np.std(y) > 0 and np.std(p) > 0:
                pearson = float(np.corrcoef(y, p)[0, 1])
            else:
                pearson = float("nan")

        per_channel.append({
            "channel": ch,
            "mse": mse,
            "rmse": rmse,
            "pearson": pearson,
            "n_observed": int(valid.sum()),
        })

    valid = mask.astype(bool)
    global_mse = float(
        np.mean((target[valid] - prediction[valid]) ** 2)
    )

    correlations = np.array(
        [item["pearson"] for item in per_channel]
    )

    return {
        "global_mse": global_mse,
        "global_rmse": float(np.sqrt(global_mse)),
        "mean_channel_pearson": float(
            np.nanmean(correlations)
        ),
        "per_channel": per_channel,
    }


def plot_metrics(metrics, output_dir):
    channels = np.arange(len(metrics["per_channel"]))
    correlations = [
        x["pearson"] for x in metrics["per_channel"]
    ]
    errors = [x["mse"] for x in metrics["per_channel"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(channels, correlations)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set(
        xlabel="Scalp channel (zero-based index)",
        ylabel="Pearson correlation",
        title="Fold 1 — Temporal Reconstruction Correlation",
    )
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "pearson_per_channel.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(channels, errors)
    ax.set(
        xlabel="Scalp channel (zero-based index)",
        ylabel="MSE",
        title="Fold 1 — Reconstruction Error by Channel",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "mse_per_channel.png", dpi=300)
    plt.close(fig)


def plot_reconstructions(examples, output_dir, fs):
    for subject_id, example in examples.items():
        target, prediction, mask = example

        channels = [0, 14, 28]
        time = np.arange(target.shape[-1]) / fs

        fig, axes = plt.subplots(
            len(channels),
            1,
            figsize=(10, 7),
            sharex=True,
        )

        for ax, ch in zip(axes, channels):
            valid = mask[ch].astype(bool)

            actual = np.where(valid, target[ch], np.nan)
            predicted = np.where(
                valid, prediction[ch], np.nan
            )

            ax.plot(time, actual, label="Ground truth")
            ax.plot(
                time,
                predicted,
                label="Conformer",
                alpha=0.8,
            )
            ax.set_ylabel(f"Ch {ch}")
            ax.grid(alpha=0.2)

        axes[0].legend()
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Subject {subject_id} — Scalp EEG Reconstruction"
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"reconstruction_subject_{subject_id}.png",
            dpi=300,
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/conformer_cleaned/in_ear/fold_1/best_model.pt"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/numpy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/conformer_evaluation/fold_1"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_model(
        args.checkpoint, device, args.fold
    )

    modality = checkpoint["args"]["modality"]
    val_subjects = [
        int(s) for s in checkpoint["val_subjects"]
    ]

    dataset = EarSAADNumpyDataset(
        data_dir=args.data_dir,
        subject_ids=val_subjects,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Device:", device)
    print("Validation subjects:", val_subjects)
    print("Validation windows:", len(dataset))

    targets = []
    predictions = []
    masks = []
    examples = {}

    with torch.inference_mode():
        for batch in loader:
            pred = predict(model, batch, modality, device)

            targets.append(batch["scalp"].numpy())
            predictions.append(pred.cpu().numpy())
            masks.append(batch["target_mask"].numpy())

            for i, subject in enumerate(batch["subject_id"]):
                subject = int(subject)
                if subject not in examples:
                    examples[subject] = (
                        batch["scalp"][i].numpy(),
                        pred[i].cpu().numpy(),
                        batch["target_mask"][i].numpy(),
                    )

    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    mask = np.concatenate(masks)

    metrics = calculate_metrics(target, prediction, mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    plot_metrics(metrics, args.output_dir)
    plot_reconstructions(
        examples,
        args.output_dir,
        dataset.fs,
    )

    print("\n=== Validation Metrics ===")
    print(f"MSE: {metrics['global_mse']:.4f}")
    print(f"RMSE: {metrics['global_rmse']:.4f}")
    print(
        "Mean channel Pearson:",
        f"{metrics['mean_channel_pearson']:.4f}",
    )
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
