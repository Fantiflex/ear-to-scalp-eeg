
"""Train Stage 1 scalp MAE and Stage 2 around-ear transfer
using 14-fold patient-level cross-validation.
"""

import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_reconstruction.data.mae_data_processing import EarSAADNPYDataset
from eeg_reconstruction.models.MAE import ScalpMAE
from eeg_reconstruction.models.transfer import (
    BTEToScalp,
    load_scalp_pretrained,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/numpy"
OUTPUT_DIR = Path("outputs")

FS = 20
WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 4.0

PATCH_SIZE = 10
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
MASK_RATIO = 0.5

BATCH_SIZE = 32
NUM_WORKERS = 0

STAGE1_EPOCHS = 100
STAGE2_EPOCHS = 100

STAGE1_LR = 1e-4
STAGE2_LR = 1e-4

WEIGHT_DECAY = 1e-4
PATIENCE = 15

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Patient-level splits
# ============================================================

def get_patient_ids(data_dir):
    files = sorted(
        Path(data_dir).glob("subject_*.npy")
    )

    patient_ids = [
        f.stem.replace("subject_", "")
        for f in files
    ]

    if len(patient_ids) != 14:
        print(
            f"Warning: expected 14 patients, "
            f"found {len(patient_ids)}."
        )

    return sorted(patient_ids)


def make_patient_splits(patient_ids):
    """11 train / 2 val / 1 test per fold."""

    patient_ids = sorted(patient_ids)
    n = len(patient_ids)

    splits = []

    for i in range(n):
        test_patient = patient_ids[i]

        val_patients = [
            patient_ids[(i + 1) % n],
            patient_ids[(i + 2) % n],
        ]

        train_patients = [
            p for p in patient_ids
            if p != test_patient
            and p not in val_patients
        ]

        splits.append({
            "fold": i + 1,
            "train": train_patients,
            "val": val_patients,
            "test": [test_patient],
        })

    return splits


# ============================================================
# Dataset / loaders
# ============================================================

def build_dataset(subject_ids):
    return EarSAADNPYDataset(
        data_dir=DATA_DIR,
        subject_ids=subject_ids,
        fs=FS,
        window_seconds=WINDOW_SECONDS,
        stride_seconds=STRIDE_SECONDS,
    )


def build_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ============================================================
# Fold-specific normalization
# ============================================================

@torch.no_grad()
def compute_training_stats(loader):
    """Calculate channel-wise mean and std from train windows only."""

    sums = {}
    sums_sq = {}
    counts = {}

    for batch in loader:
        for key in ("around_ear", "scalp"):
            x = batch[key].double()

            if key not in sums:
                sums[key] = torch.zeros(
                    x.shape[1], dtype=torch.float64
                )
                sums_sq[key] = torch.zeros(
                    x.shape[1], dtype=torch.float64
                )
                counts[key] = 0

            sums[key] += x.sum(dim=(0, 2))
            sums_sq[key] += (x ** 2).sum(dim=(0, 2))
            counts[key] += x.shape[0] * x.shape[2]

    stats = {}

    for key in sums:
        mean = sums[key] / counts[key]

        variance = (
            sums_sq[key] / counts[key]
            - mean.square()
        ).clamp_min(0)

        std = variance.sqrt().clamp_min(1e-6)

        stats[key] = {
            "mean": mean.float(),
            "std": std.float(),
        }

    return stats


def normalize_batch(batch, stats):
    """Normalize a batch using train-fold statistics."""

    normalized = {}

    for key in ("around_ear", "scalp"):
        mean = stats[key]["mean"].to(DEVICE)
        std = stats[key]["std"].to(DEVICE)

        normalized[key] = (
            batch[key].to(DEVICE) - mean[None, :, None]
        ) / std[None, :, None]

    return normalized


# ============================================================
# Stage 1: masked patch reconstruction loss
# ============================================================

def masked_patch_mse(pred, target, mask, patch_size):
    """
    pred, target: (B, C, T)
    mask:         (B, C, P)

    Loss only on masked patches.
    """

    pred_patches = pred.unfold(
        dimension=-1,
        size=patch_size,
        step=patch_size,
    )

    target_patches = target.unfold(
        dimension=-1,
        size=patch_size,
        step=patch_size,
    )

    squared_error = (
        pred_patches - target_patches
    ).square()

    mask = mask.unsqueeze(-1).float()

    numerator = (squared_error * mask).sum()
    denominator = mask.sum() * patch_size

    return numerator / denominator.clamp_min(1)


# ============================================================
# Validation / evaluation
# ============================================================

@torch.no_grad()
def validate_stage1(model, loader, stats):
    model.eval()

    losses = []

    for batch in loader:
        normalized = normalize_batch(batch, stats)
        scalp = normalized["scalp"]

        pred, mask = model(
            scalp,
                mask_ratio=MASK_RATIO,
        )

        loss = masked_patch_mse(
            pred,
            scalp,
            mask,
            PATCH_SIZE,
        )

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_stage2(model, loader, stats):
    """Evaluate normalized scalp reconstruction."""

    model.eval()

    total_squared_error = 0.0
    total_samples = 0

    channel_squared_error = None
    channel_samples = 0

    for batch in loader:
        normalized = normalize_batch(batch, stats)

        around = normalized["around_ear"]
        scalp = normalized["scalp"]

        pred = model(around)

        squared_error = (pred - scalp).square()

        total_squared_error += squared_error.sum().item()
        total_samples += squared_error.numel()

        # Per-channel MSE accumulated over windows/time
        ch_error = squared_error.sum(dim=(0, 2))

        if channel_squared_error is None:
            channel_squared_error = ch_error

        else:
            channel_squared_error += ch_error

        channel_samples += (
            squared_error.shape[0]
            * squared_error.shape[2]
        )

    mse = total_squared_error / total_samples
    rmse = np.sqrt(mse)

    channel_mse = (
        channel_squared_error / channel_samples
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "channel_mse": channel_mse.cpu().numpy(),
    }


# ============================================================
# Early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0

            self.best_state = copy.deepcopy(
                model.state_dict()
            )

            return False

        self.counter += 1

        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# Stage 1 training
# ============================================================

def train_stage1(
    train_loader,
    val_loader,
    stats,
    fold_dir,
):
    model = ScalpMAE(
        n_channels=29,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    stopper = EarlyStopping(PATIENCE)

    history = []

    for epoch in range(STAGE1_EPOCHS):
        model.train()

        train_losses = []

        for batch in train_loader:
            normalized = normalize_batch(batch, stats)
            scalp = normalized["scalp"]

            pred, mask = model(
                scalp,
                mask_ratio=MASK_RATIO,
            )

            loss = masked_patch_mse(
                pred,
                scalp,
                mask,
                PATCH_SIZE,
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_loss = validate_stage1(
            model,
            val_loader,
            stats,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"[Stage 1] Epoch {epoch + 1:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        if stopper.step(val_loss, model):
            print("Stage 1 early stopping.")
            break

    stopper.restore(model)

    torch.save(
        model.state_dict(),
        fold_dir / "stage1_best.pt",
    )

    return model, history


# ============================================================
# Stage 2 training
# ============================================================

def train_stage2(
    scalp_model,
    train_loader,
    val_loader,
    stats,
    fold_dir,
):
    model = BTEToScalp(
        n_around_channels=19,
        n_scalp_channels=29,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    model = load_scalp_pretrained(
        model,
        scalp_model,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY,
    )

    stopper = EarlyStopping(PATIENCE)

    history = []

    for epoch in range(STAGE2_EPOCHS):
        model.train()

        train_losses = []

        for batch in train_loader:
            normalized = normalize_batch(batch, stats)

            around = normalized["around_ear"]
            scalp = normalized["scalp"]

            pred = model(around)

            loss = F.mse_loss(pred, scalp)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        model.eval()

        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                normalized = normalize_batch(batch, stats)

                pred = model(normalized["around_ear"])

                loss = F.mse_loss(
                    pred,
                    normalized["scalp"],
                )

                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"[Stage 2] Epoch {epoch + 1:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        if stopper.step(val_loss, model):
            print("Stage 2 early stopping.")
            break

    stopper.restore(model)

    torch.save(
        model.state_dict(),
        fold_dir / "stage2_best.pt",
    )

    return model, history


# ============================================================
# Save histories
# ============================================================

def save_history(history, filepath):
    if not history:
        return

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# Cross-validation
# ============================================================

def run_cross_validation():
    if FS is None:
        raise ValueError(
            "Set FS to the actual sampling frequency before training."
        )

    if round(WINDOW_SECONDS * FS) % PATCH_SIZE != 0:
        raise ValueError(
            "Window size must be divisible by PATCH_SIZE."
        )

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_ids = get_patient_ids(DATA_DIR)

    splits = make_patient_splits(patient_ids)

    results = []

    for split in splits:
        fold = split["fold"]

        train_patients = split["train"]
        val_patients = split["val"]
        test_patients = split["test"]

        print("\n" + "=" * 70)
        print(f"Fold {fold}/{len(splits)}")
        print("Train:", train_patients)
        print("Val:", val_patients)
        print("Test:", test_patients)
        print("=" * 70)

        fold_dir = OUTPUT_DIR / f"fold_{fold:02d}"

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Separate patient-level datasets
        train_dataset = build_dataset(train_patients)
        val_dataset = build_dataset(val_patients)
        test_dataset = build_dataset(test_patients)

        train_loader = build_loader(
            train_dataset,
            shuffle=True,
        )

        # For normalization, use train windows only
        stats_loader = build_loader(
            train_dataset,
            shuffle=False,
        )

        val_loader = build_loader(
            val_dataset,
            shuffle=False,
        )

        test_loader = build_loader(
            test_dataset,
            shuffle=False,
        )

        # Fold-specific normalization
        stats = compute_training_stats(
            stats_loader
        )

        torch.save(
            stats,
            fold_dir / "normalization_stats.pt",
        )

        # Stage 1
        scalp_model, stage1_history = train_stage1(
            train_loader,
            val_loader,
            stats,
            fold_dir,
        )

        save_history(
            stage1_history,
            fold_dir / "stage1_history.csv",
        )

        # Stage 2
        transfer_model, stage2_history = train_stage2(
            scalp_model,
            train_loader,
            val_loader,
            stats,
            fold_dir,
        )

        save_history(
            stage2_history,
            fold_dir / "stage2_history.csv",
        )

        # Held-out evaluation
        test_metrics = evaluate_stage2(
            transfer_model,
            test_loader,
            stats,
        )

        print(
            f"[Fold {fold}] Test normalized MSE: "
            f"{test_metrics['mse']:.6f}"
        )

        print(
            f"[Fold {fold}] Test normalized RMSE: "
            f"{test_metrics['rmse']:.6f}"
        )

        # Save per-channel metrics
        np.save(
            fold_dir / "channel_mse.npy",
            test_metrics["channel_mse"],
        )

        result = {
            "fold": fold,
            "test_patient": test_patients[0],
            "mse": test_metrics["mse"],
            "rmse": test_metrics["rmse"],
        }

        results.append(result)

        with open(
            OUTPUT_DIR / "cv_results.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=result.keys(),
            )

            writer.writeheader()
            writer.writerows(results)

    # Aggregate across test patients
    mse_values = np.array([
        r["mse"] for r in results
    ])

    rmse_values = np.array([
        r["rmse"] for r in results
    ])

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Normalized MSE: "
        f"{mse_values.mean():.6f} "
        f"+/- {mse_values.std(ddof=1):.6f}"
    )

    print(
        f"Normalized RMSE: "
        f"{rmse_values.mean():.6f} "
        f"+/- {rmse_values.std(ddof=1):.6f}"
    )

    return results


if __name__ == "__main__":
    run_cross_validation()
