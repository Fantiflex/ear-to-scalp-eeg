
"""Train Temporal Conformer on EAR-SAAD with subject-wise CV."""

import argparse
import atexit
import csv
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.models.spatial_conformer import SpatialConformer
from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
    TemporalConformer,
)
from eeg_reconstruction.models.train_val_test_split import make_patient_splits
from eeg_reconstruction.training.losses import (
    hybrid_loss,
    masked_mse,
    masked_temporal_derivative_loss,
)


class Tee:
    """Write console output to both terminal and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def start_file_logging(output_dir):
    log_file = (output_dir / "train.log").open(
        "a", encoding="utf-8", buffering=1
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    def cleanup():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(cleanup)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(split, args):
    train_dataset = EarSAADNumpyDataset(
        data_dir=args.data_dir,
        subject_ids=split.train_subjects,
        window_seconds=4.0,
        stride_seconds=4.0,
    )

    val_dataset = EarSAADNumpyDataset(
        data_dir=args.data_dir,
        subject_ids=split.val_subjects,
        window_seconds=4.0,
        stride_seconds=4.0,
    )

    train_ids = {s[1] for s in train_dataset.samples}
    val_ids = {s[1] for s in val_dataset.samples}

    assert train_ids == set(split.train_subjects)
    assert val_ids == set(split.val_subjects)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(split.test_subjects)
    assert val_ids.isdisjoint(split.test_subjects)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def run_epoch(
    model,
    loader,
    device,
    modality,
    optimizer=None,
    loss_type="mse",
    pearson_weight=1.0,
    derivative_weight=0.1,
):
    training = optimizer is not None
    model.train(training)

    total_squared_error = 0.0
    total_observations = 0

    for batch in loader:
        x = batch[modality].to(device)
        input_mask = batch[
            f"{modality}_observed_mask"
        ].to(device)

        y = batch["scalp"].to(device)
        target_mask = batch["target_mask"].to(device)

        n_valid = int(target_mask.sum().item())
        if n_valid == 0:
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = model(x, input_mask)

            mse_loss = masked_mse(prediction, y, target_mask)

            if training and loss_type == "hybrid":
                loss = hybrid_loss(
                    prediction,
                    y,
                    target_mask,
                    pearson_weight=pearson_weight,
                )

            elif training and loss_type == "derivative":
                diff_loss = masked_temporal_derivative_loss(
                    prediction,
                    y,
                    target_mask,
                )
                loss = mse_loss + derivative_weight * diff_loss

            else:
                loss = mse_loss
            if training and loss.item() > 1000:
                print(
                    "HIGH LOSS |",
                    "subjects:", batch["subject_id"].tolist(),
                    "max_input:", x.abs().max().item(),
                    "max_target:", y[target_mask].abs().max().item(),
                    "max_prediction:", prediction.abs().max().item(),
                    "loss:", loss.item(),
                    flush=True,
                )

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered")

            if training:
                loss.backward()
                optimizer.step()

        total_squared_error += mse_loss.detach().item() * n_valid
        total_observations += n_valid

    if total_observations == 0:
        raise RuntimeError("No observed scalp targets")

    mse = total_squared_error / total_observations

    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "observations": total_observations,
    }


def save_history(history, path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_mse",
                "train_rmse",
                "val_mse",
                "val_rmse",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["temporal", "spatial"],
        default="temporal",
        help="Architecture used for EEG reconstruction.",
    )
    parser.add_argument(
        "--modality",
        choices=["in_ear", "around_ear"],
        default="in_ear",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        default="data/processed/numpy",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "EEG_RESULTS_DIR",
            "/content/drive/MyDrive/ear-to-scalp-eeg/experiments",
        ),
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "hybrid", "derivative"],
        default="mse",
    )

    parser.add_argument(
        "--pearson-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--derivative-weight",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.patience < 1:
        parser.error("patience must be positive")
    if not np.isfinite(args.derivative_weight) or args.derivative_weight < 0:
        parser.error("derivative-weight must be finite and non-negative")
    if args.loss == "mse":
        experiment_name = "mse"

    elif args.loss == "hybrid":
        experiment_name = f"mse_pearson_w{args.pearson_weight:g}"

    elif args.loss == "derivative":
        experiment_name = (
            f"mse_derivative_w{args.derivative_weight:g}"
        )

    else:
        parser.error(f"Unsupported loss: {args.loss}")

    
    set_seed(args.seed)
    device = get_device()

    splits = make_patient_splits(range(1, 16))

    if not 1 <= args.fold <= len(splits):
        parser.error("Fold must be between 1 and 15")

    raw_split = splits[args.fold - 1]

    split = SimpleNamespace(
        fold=raw_split["fold"],
        train_subjects=tuple(raw_split["train"]),
        val_subjects=tuple(raw_split["val"]),
        test_subjects=tuple(raw_split["test"]),
    )

    # Ensure the persistent storage is available.
    results_root = Path(args.output_dir).expanduser()

    if (
        str(results_root).startswith("/content/drive/")
        and not Path("/content/drive/MyDrive").is_dir()
    ):
        parser.error(
            "Google Drive is not mounted. "
            "Run drive.mount('/content/drive') first."
        )

    # Unique timestamp prevents overwriting previous experiments.
    timestamp = datetime.now(UTC).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_name = f"seed_{args.seed}_{timestamp}"

    output_dir = (
        results_root
        / args.modality
        / f"fold_{args.fold:02d}"
        / f"{args.model}_conformer"
        / experiment_name
        / run_name
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    # Save the complete run configuration.
    run_config = {
        **vars(args),
        "experiment_name": experiment_name,
        "run_name": run_name,
        "train_subjects": list(split.train_subjects),
        "val_subjects": list(split.val_subjects),
        "test_subjects": list(split.test_subjects),
    }

    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2)
    )

    # Save console logs automatically.
    start_file_logging(output_dir)

    print(f"Experiment: {experiment_name}")
    print(f"Results directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Modality: {args.modality}")
    print(f"Train subjects: {split.train_subjects}")
    print(f"Val subjects: {split.val_subjects}")
    print(f"Reserved test subjects: {split.test_subjects}")

    train_loader, val_loader = make_loaders(split, args)

    print(f"Train windows: {len(train_loader.dataset)}")
    print(f"Val windows: {len(val_loader.dataset)}")

    in_channels = 12 if args.modality == "in_ear" else 19

    config = ConformerConfig(
        in_channels=in_channels,
        out_channels=29,
        d_model=64,
        n_heads=4,
        n_layers=2,
        kernel_size=15,
        dropout=0.1,
    )

    if args.model == "temporal":
        model = TemporalConformer(config)
    elif args.model == "spatial":
        model = SpatialConformer(config)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.modality,
            optimizer,
            loss_type=args.loss,
            pearson_weight=args.pearson_weight,
            derivative_weight=args.derivative_weight,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.modality,
            optimizer=None,
            loss_type="mse",
        )

        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train MSE: {row['train_mse']:.6f} | "
            f"Val MSE: {row['val_mse']:.6f} | "
            f"Val RMSE: {row['val_rmse']:.6f}",
            flush=True,
        )

        save_history(history, output_dir / "history.csv")

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": vars(config),
                    "model_type": args.model,
                    "args": vars(args),
                    "fold": args.fold,
                    "train_subjects": split.train_subjects,
                    "val_subjects": split.val_subjects,
                    "test_subjects": split.test_subjects,
                    "best_epoch": best_epoch,
                    "best_val_mse": best_val_mse,
                },
                output_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    summary = {
        "fold": int(args.fold),
        "modality": args.modality,
        "train_subjects": [int(s) for s in split.train_subjects],
        "val_subjects": [int(s) for s in split.val_subjects],
        "test_subjects": [int(s) for s in split.test_subjects],
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val_mse),
        "best_val_rmse": float(best_val_mse**0.5),
        "model_type": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation MSE: {best_val_mse:.6f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
