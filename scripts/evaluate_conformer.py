"""Evaluate EEG reconstruction quality on held-out validation subjects."""

import argparse
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_reconstruction.data.normalization import (
    inverse_normalize,
    normalize_batch,
)
from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.data.rereferencing import rereference_batch
from eeg_reconstruction.models.spatial_conformer import SpatialConformer
from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
    TemporalConformer,
)


def load_model(checkpoint_path, device, expected_fold):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,  # Only load trusted checkpoints.
    )

    if int(checkpoint["fold"]) != expected_fold:
        raise ValueError(f"Wrong checkpoint fold: {checkpoint['fold']}")

    cfg = ConformerConfig(**checkpoint["model_config"])

    model_type = checkpoint.get(
        "model_type",
        checkpoint.get("args", {}).get("model", "temporal"),
    )

    if model_type == "temporal":
        model = TemporalConformer(cfg)
    elif model_type == "spatial":
        model = SpatialConformer(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, checkpoint


def predict(
    model,
    batch,
    modality,
    device,
    rereference_mode="none",
    normalization_stats=None,
):
    """Predict scalp EEG in its original physical units."""

    # 1. Apply the same reference montage as training.
    if rereference_mode != "none":
        if modality != "in_ear":
            raise ValueError(
                "Re-referencing is currently supported only for in-ear EEG."
            )

        batch = rereference_batch(
            batch,
            mode=rereference_mode,
        )

    # 2. Apply training-only normalization statistics.
    if normalization_stats is not None:
        batch = normalize_batch(
            batch,
            normalization_stats,
            device=device,
        )

    # 3. Prepare the model inputs.
    x = batch[modality].to(device)
    mask = batch[f"{modality}_observed_mask"].to(device)

    # 4. Predict in the model's output space.
    parameters = inspect.signature(model.forward).parameters

    if "observed_mask" in parameters:
        prediction = model(x, observed_mask=mask)
    elif len(parameters) >= 2:
        prediction = model(x, mask)
    else:
        prediction = model(x)

    # 5. Restore scalp EEG to original units.
    if normalization_stats is not None:
        prediction = inverse_normalize(
            prediction,
            normalization_stats,
            modality="scalp",
        )

    return prediction


def calculate_metrics(target, prediction, mask):
    target = target.astype(np.float64)
    prediction = prediction.astype(np.float64)
    mask = mask.astype(bool)

    if not mask.any():
        raise RuntimeError("No valid scalp targets")

    if not np.isfinite(prediction[mask]).all():
        raise RuntimeError("Non-finite predictions")

    n_channels = target.shape[1]
    per_channel = []

    for ch in range(n_channels):
        valid = mask[:, ch, :]
        y = target[:, ch, :][valid]
        p = prediction[:, ch, :][valid]

        mse = rmse = pearson = std_ratio = float("nan")

        if len(y) > 0:
            mse = float(np.mean((y - p) ** 2))
            rmse = float(np.sqrt(mse))

            target_std = float(np.std(y))
            pred_std = float(np.std(p))

            if target_std > 0:
                std_ratio = pred_std / target_std

                if pred_std > 0:
                    pearson = float(np.corrcoef(y, p)[0, 1])

        per_channel.append(
            {
                "channel": ch,
                "mse": mse,
                "rmse": rmse,
                "pearson": pearson,
                "std_ratio": std_ratio,
                "n_observed": int(valid.sum()),
            }
        )

    y = target[mask]
    p = prediction[mask]

    global_mse = float(np.mean((y - p) ** 2))
    zero_mse = float(np.mean(y**2))

    correlations = np.array([item["pearson"] for item in per_channel])

    std_ratios = np.array([item["std_ratio"] for item in per_channel])

    return {
        "global_mse": global_mse,
        "global_rmse": float(np.sqrt(global_mse)),
        "mean_channel_pearson": float(np.nanmean(correlations)),
        "median_std_ratio": float(np.nanmedian(std_ratios)),
        "zero_baseline_mse": zero_mse,
        "skill_vs_zero": (1.0 - global_mse / zero_mse if zero_mse > 0 else None),
        "n_valid_targets": int(mask.sum()),
        "n_windows": int(target.shape[0]),
        "per_channel": per_channel,
    }
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

        per_channel.append(
            {
                "channel": ch,
                "mse": mse,
                "rmse": rmse,
                "pearson": pearson,
                "n_observed": int(valid.sum()),
            }
        )

    valid = mask.astype(bool)
    global_mse = float(np.mean((target[valid] - prediction[valid]) ** 2))

    correlations = np.array([item["pearson"] for item in per_channel])

    return {
        "global_mse": global_mse,
        "global_rmse": float(np.sqrt(global_mse)),
        "mean_channel_pearson": float(np.nanmean(correlations)),
        "per_channel": per_channel,
    }


def plot_metrics(metrics, output_dir):
    channels = np.arange(len(metrics["per_channel"]))
    correlations = [x["pearson"] for x in metrics["per_channel"]]
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
            predicted = np.where(valid, prediction[ch], np.nan)

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
        fig.suptitle(f"Subject {subject_id} — Scalp EEG Reconstruction")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"reconstruction_subject_{subject_id}.png",
            dpi=300,
        )
        plt.close(fig)


def analyze_amplitude(target, prediction, mask, subject_ids, output_dir):
    """Measure prediction amplitude attenuation per subject and channel."""
    import pandas as pd

    records = []

    for subject in np.unique(subject_ids):
        selected = subject_ids == subject

        for ch in range(target.shape[1]):
            valid = mask[selected, ch, :].astype(bool)

            y = target[selected, ch, :][valid]
            p = prediction[selected, ch, :][valid]

            if len(y) < 2:
                continue

            target_std = float(np.std(y))
            pred_std = float(np.std(p))

            ratio = pred_std / target_std if target_std > 1e-8 else float("nan")

            records.append(
                {
                    "subject": int(subject),
                    "channel": ch,
                    "target_std": target_std,
                    "prediction_std": pred_std,
                    "std_ratio": ratio,
                    "n_observed": len(y),
                }
            )

    df = pd.DataFrame(records)

    df.to_csv(
        output_dir / "amplitude_analysis.csv",
        index=False,
    )

    # Figure 1: amplitude ratios by scalp channel and subject
    fig, ax = plt.subplots(figsize=(11, 5))

    for subject, group in df.groupby("subject"):
        ax.plot(
            group["channel"],
            group["std_ratio"],
            marker="o",
            markersize=3,
            label=f"Subject {subject}",
        )

    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        label="Perfect variance ratio",
    )

    ax.set_xlabel("Scalp channel")
    ax.set_ylabel("Prediction std / Target std")
    ax.set_title("EEG Amplitude Attenuation by Channel")
    ax.set_xticks(range(target.shape[1]))
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        output_dir / "amplitude_ratio_per_channel.png",
        dpi=300,
    )
    plt.close(fig)

    # Figure 2: target and prediction standard deviations
    grouped = df.groupby("channel")[["target_std", "prediction_std"]].mean()

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(grouped))
    width = 0.4

    ax.bar(
        x - width / 2,
        grouped["target_std"],
        width,
        label="Ground truth",
    )
    ax.bar(
        x + width / 2,
        grouped["prediction_std"],
        width,
        label="Conformer",
    )

    ax.set_xlabel("Scalp channel")
    ax.set_ylabel("Standard deviation")
    ax.set_title("EEG Signal Variability: Target vs Prediction")
    ax.set_xticks(x)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "amplitude_std_comparison.png",
        dpi=300,
    )
    plt.close(fig)

    print("\n=== Amplitude Analysis ===")
    for subject, group in df.groupby("subject"):
        print(
            f"Subject {subject}: median std ratio = {group['std_ratio'].median():.4f}"
        )

    print(
        "Overall median std ratio:",
        f"{df['std_ratio'].median():.4f}",
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/conformer_cleaned/in_ear/fold_1/best_model.pt"),
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
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model(args.checkpoint, device, args.fold)
    checkpoint_args = checkpoint.get("args", {})

    modality = checkpoint_args.get("modality", "in_ear")
    rereference_mode = checkpoint_args.get("rereference", "none")
    normalization_stats = checkpoint.get("normalization_stats")

    normalization_mode = checkpoint_args.get("normalization", "none")

    if normalization_mode != "none" and normalization_stats is None:
        raise ValueError(
            "Checkpoint requires normalization but "
            "contains no normalization statistics."
        )

    print("Model:", checkpoint.get("model_type", "temporal"))
    print("Modality:", modality)
    print("Re-referencing:", rereference_mode)
    print("Normalization:", normalization_mode)
    normalization_stats = checkpoint.get("normalization_stats")

    print(
        "Normalization:",
        checkpoint.get("args", {}).get("normalization", "none"),
    )

    val_subjects = [int(s) for s in checkpoint["val_subjects"]]

    dataset = EarSAADNumpyDataset(
        data_dir=args.data_dir,
        subject_ids=val_subjects,
        window_seconds=4.0,
        stride_seconds=4.0,
        manifest_path=args.val_manifest,
    )

    if len(dataset) == 0:
        raise RuntimeError("Empty validation dataset")

    print("Validation manifest:", args.val_manifest)
    print(
        "Training validation manifest:",
        checkpoint_args.get("val_manifest"),
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
    all_subject_ids = []

    with torch.inference_mode():
        for batch in loader:
            pred = predict(
                model,
                batch,
                modality,
                device,
                rereference_mode=rereference_mode,
                normalization_stats=normalization_stats,
            )
            targets.append(batch["scalp"].numpy())
            predictions.append(pred.cpu().numpy())
            masks.append(batch["target_mask"].numpy())
            all_subject_ids.append(batch["subject_id"].numpy())
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
    subject_ids = np.concatenate(all_subject_ids)

    metrics = calculate_metrics(target, prediction, mask)
    metrics["val_manifest"] = args.val_manifest
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["metrics_by_subject"] = {}

    for subject in np.unique(subject_ids):
        selected = subject_ids == subject

        subject_metrics = calculate_metrics(
            target[selected],
            prediction[selected],
            mask[selected],
        )

        metrics["metrics_by_subject"][str(subject)] = {
            key: value for key, value in subject_metrics.items() if key != "per_channel"
        }

    metrics["macro_subject_mse"] = float(
        np.mean([item["global_mse"] for item in metrics["metrics_by_subject"].values()])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_metrics(metrics, args.output_dir)
    plot_reconstructions(
        examples,
        args.output_dir,
        dataset.fs,
    )
    analyze_amplitude(
        target=target,
        prediction=prediction,
        mask=mask,
        subject_ids=subject_ids,
        output_dir=args.output_dir,
    )

    print("\n=== Validation Metrics ===")
    print(f"MSE: {metrics['global_mse']:.4f}")
    print(f"RMSE: {metrics['global_rmse']:.4f}")
    print(
        "Mean channel Pearson:",
        f"{metrics['mean_channel_pearson']:.4f}",
    )
    print("Saved to:", args.output_dir)
    print(
        "Median std ratio:",
        metrics["median_std_ratio"],
    )

    print(
        "Zero baseline MSE:",
        metrics["zero_baseline_mse"],
    )

    print(
        "Skill vs zero:",
        metrics["skill_vs_zero"],
    )

    print(
        "Macro subject MSE:",
        metrics["macro_subject_mse"],
    )

    print("\n=== METRICS BY SUBJECT ===")
    print(
        json.dumps(
            metrics["metrics_by_subject"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
