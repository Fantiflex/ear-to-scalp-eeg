"""Evaluate an EAR-SAAD Conformer checkpoint by held-out validation subject.

Usage:
    python scripts/evaluate_eeg_by_subject.py --checkpoint /path/best_model.pt \
        --output-dir results/by_subject --fold 1

Never evaluates reserved test subjects. Checkpoint train/val metadata is verified.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_reconstruction.data.normalization import inverse_normalize, normalize_batch
from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.data.rereferencing import rereference_batch
from eeg_reconstruction.models.spatial_conformer import SpatialConformer
from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
    TemporalConformer,
)
from eeg_reconstruction.models.train_val_test_split import make_patient_splits

N_SCALP = 29


def load_checkpoint(path, device, fold):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint["fold"]) != fold:
        raise ValueError("Checkpoint fold does not match requested fold")
    config = ConformerConfig(**checkpoint["model_config"])
    kind = checkpoint.get("model_type", checkpoint.get("args", {}).get("model", "temporal"))
    if kind == "temporal":
        model = TemporalConformer(config)
    elif kind == "spatial":
        model = SpatialConformer(config)
    else:
        raise ValueError(f"Unknown model type: {kind}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def evaluate_subject(model, subject, args, modality, ref_mode, stats, device):
    dataset = EarSAADNumpyDataset(
        data_dir=args.data_dir,
        subject_ids=[subject],
        window_seconds=4.0,
        stride_seconds=4.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    n = np.zeros(N_SCALP, dtype=np.float64)
    sum_y = np.zeros(N_SCALP, dtype=np.float64)
    sum_p = np.zeros(N_SCALP, dtype=np.float64)
    sum_y2 = np.zeros(N_SCALP, dtype=np.float64)
    sum_p2 = np.zeros(N_SCALP, dtype=np.float64)
    sum_yp = np.zeros(N_SCALP, dtype=np.float64)
    sse = np.zeros(N_SCALP, dtype=np.float64)
    sse_zero = np.zeros(N_SCALP, dtype=np.float64)

    with torch.inference_mode():
        for batch in loader:
            original_y = batch["scalp"].numpy().astype(np.float64)
            valid = batch["target_mask"].numpy().astype(bool)
            if ref_mode != "none":
                if modality != "in_ear":
                    raise ValueError("Re-referencing only supported for in_ear")
                batch = rereference_batch(batch, mode=ref_mode)
            if stats is not None:
                batch = normalize_batch(batch, stats, device=device)
            x = batch[modality].to(device)
            mask = batch[f"{modality}_observed_mask"].to(device)
            prediction = model(x, mask)
            if stats is not None:
                prediction = inverse_normalize(prediction, stats, modality="scalp")
            p = prediction.cpu().numpy().astype(np.float64)
            y = original_y
            good = valid & np.isfinite(y) & np.isfinite(p)
            if (valid & ~good).any():
                raise ValueError(f"Nonfinite target or prediction for subject {subject}")
            yf = np.where(good, y, 0.0)
            pf = np.where(good, p, 0.0)
            n += good.sum(axis=(0, 2))
            sum_y += yf.sum(axis=(0, 2))
            sum_p += pf.sum(axis=(0, 2))
            sum_y2 += (yf * yf).sum(axis=(0, 2))
            sum_p2 += (pf * pf).sum(axis=(0, 2))
            sum_yp += (yf * pf).sum(axis=(0, 2))
            sse += ((pf - yf) ** 2 * good).sum(axis=(0, 2))
            sse_zero += (yf * yf).sum(axis=(0, 2))

    safe = np.maximum(n, 1)
    centered_y2 = np.maximum(sum_y2 - sum_y**2 / safe, 0)
    centered_p2 = np.maximum(sum_p2 - sum_p**2 / safe, 0)
    centered_yp = sum_yp - sum_y * sum_p / safe
    denom = np.sqrt(centered_y2 * centered_p2)
    pearson = np.divide(centered_yp, denom, out=np.full(N_SCALP, np.nan), where=(n >= 2) & (denom > 1e-12))
    ratio = np.sqrt(np.divide(centered_p2, centered_y2, out=np.full(N_SCALP, np.nan), where=(n >= 2) & (centered_y2 > 1e-12)))
    count = n.sum()
    if count == 0:
        raise RuntimeError(f"No valid scalp samples for subject {subject}")
    result = {
        "subject": int(subject),
        "windows": len(dataset),
        "valid_channel_samples": int(count),
        "mse": float(sse.sum() / count),
        "rmse": float(np.sqrt(sse.sum() / count)),
        "zero_mse": float(sse_zero.sum() / count),
        "relative_mse_gain_vs_zero_pct": float(100 * (1 - sse.sum() / sse_zero.sum())),
        "mean_channel_pearson": float(np.nanmean(pearson)),
        "median_std_ratio": float(np.nanmedian(ratio)),
    }
    channel_rows = [
        {
            "subject": subject,
            "scalp_channel": c,
            "n": int(n[c]),
            "mse": float(sse[c] / n[c]) if n[c] else float("nan"),
            "pearson": float(pearson[c]),
            "std_ratio": float(ratio[c]),
        }
        for c in range(N_SCALP)
    ]
    return result, channel_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", default="data/processed/numpy")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device, args.fold)
    split = make_patient_splits(range(1, 16))[args.fold - 1]
    subjects = tuple(int(s) for s in checkpoint["val_subjects"])
    if set(subjects) != set(split["val"]):
        raise ValueError(f"Checkpoint validation subjects {subjects} differ from fold {args.fold}: {split['val']}")
    if set(subjects) & set(checkpoint["test_subjects"]):
        raise ValueError("Validation/test subject overlap")
    ckpt_args = checkpoint.get("args", {})
    modality = ckpt_args.get("modality", "in_ear")
    ref_mode = ckpt_args.get("rereference", "none")
    stats = checkpoint.get("normalization_stats")
    if ckpt_args.get("normalization", "none") != "none" and stats is None:
        raise ValueError("Normalization statistics missing from checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | fold: {args.fold} | validation subjects: {subjects}")
    print(f"Modality: {modality} | reference: {ref_mode} | normalization: {ckpt_args.get('normalization','none')}")
    rows = []
    channel_rows = []
    for subject in subjects:
        row, per_channel = evaluate_subject(model, subject, args, modality, ref_mode, stats, device)
        rows.append(row)
        channel_rows.extend(per_channel)
        print(json.dumps(row, indent=2))

    for filename, records in [("per_subject.csv", rows), ("per_channel.csv", channel_rows)]:
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (args.output_dir / "per_subject.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
