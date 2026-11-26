"""Lagged Ridge baseline for in-ear to scalp EEG reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.models.train_val_test_split import make_patient_splits


class MaskedRidge:
    """Fit one Ridge regressor per scalp channel."""

    def __init__(self, alpha=1000.0, standardize=True):
        self.alpha = alpha
        self.standardize = standardize
        self.models = []
        self.n_train_per_channel = []

    def fit(self, x, y, valid):
        self.models = []
        self.n_train_per_channel = []

        for channel in range(y.shape[1]):
            observed = valid[:, channel]
            n_valid = int(observed.sum())

            self.n_train_per_channel.append(n_valid)

            if n_valid == 0:
                raise RuntimeError(f"No training targets for channel {channel}")

            ridge = Ridge(
                alpha=self.alpha,
                solver="lsqr",
            )

            if self.standardize:
                model = make_pipeline(
                    StandardScaler(),
                    ridge,
                )
            else:
                model = ridge

            model.fit(
                x[observed],
                y[observed, channel],
            )

            self.models.append(model)

            print(
                f"Channel {channel:02d} | training samples={n_valid:,}",
                flush=True,
            )

        return self

    def predict(self, x):
        return np.column_stack([model.predict(x) for model in self.models])


def features(batch, lag):
    x = batch["in_ear"].numpy().astype(np.float32)
    mask = batch["in_ear_observed_mask"].numpy().astype(bool)

    x = np.where(mask, x, 0.0)
    b, _, t = x.shape

    padded_x = np.pad(x, ((0, 0), (0, 0), (lag, lag)))
    padded_m = np.pad(
        mask.astype(np.float32),
        ((0, 0), (0, 0), (lag, lag)),
    )

    parts = []
    for offset in range(-lag, lag + 1):
        start = lag + offset
        parts.append(padded_x[:, :, start : start + t])
        parts.append(padded_m[:, :, start : start + t])

    z = np.concatenate(parts, axis=1)
    return z.transpose(0, 2, 1).reshape(b * t, -1)


def targets(batch):
    y = batch["scalp"].numpy()
    mask = batch["target_mask"].numpy().astype(bool)

    return (
        y.transpose(0, 2, 1).reshape(-1, 29),
        mask.transpose(0, 2, 1).reshape(-1, 29),
    )


def make_loader(
    subject_ids,
    data_dir,
    batch_size,
    shuffle,
    manifest_path=None,
):
    dataset = EarSAADNumpyDataset(
        data_dir=data_dir,
        subject_ids=subject_ids,
        window_seconds=4.0,
        stride_seconds=4.0,
        manifest_path=manifest_path,
    )

    if len(dataset) == 0:
        raise RuntimeError("Empty EEG dataset")

    print(f"Dataset: {len(dataset):,} windows | manifest={manifest_path}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train(loader, lag, seed):
    xs, ys, masks = [], [], []

    for batch in loader:
        z = features(batch, lag)
        y, valid = targets(batch)

        b = batch["in_ear"].shape[0]
        t = batch["in_ear"].shape[-1]

        subjects = batch["subject_id"].numpy()
        trials = batch["trial_id"].numpy()
        starts = batch["start_sample"].numpy()

        positions = []

        for j in range(b):
            rng = np.random.default_rng(
                [
                    seed,
                    int(subjects[j]),
                    int(trials[j]),
                    int(starts[j]),
                ]
            )

            positions.append(rng.choice(t, size=10, replace=False))

        positions = np.stack(positions)

        indices = (np.arange(b)[:, None] * t + positions).reshape(-1)

        # Keep timestamps with at least one valid target.
        indices = indices[valid[indices].any(axis=1)]

        xs.append(z[indices])
        ys.append(y[indices])
        masks.append(valid[indices])

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    valid_train = np.concatenate(masks)

    if len(x_train) == 0:
        raise RuntimeError("No valid training samples")

    print("Training matrix:", x_train.shape)
    print("Target matrix:", y_train.shape)
    print("Target mask:", valid_train.shape)

    print(
        "Valid target fraction:",
        valid_train.mean(),
    )

    print(
        "Valid samples per scalp channel:",
        valid_train.sum(axis=0),
    )

    return x_train, y_train, valid_train


def evaluate(model, loader, lag):
    """Evaluate all valid scalp samples and locate extreme error windows."""
    squared_error = 0.0
    count = 0
    zero_squared_error = 0.0
    channel_squared_error = np.zeros(29, dtype=np.float64)
    n = np.zeros(29, dtype=np.float64)
    sy = np.zeros(29, dtype=np.float64)
    sp = np.zeros(29, dtype=np.float64)
    syy = np.zeros(29, dtype=np.float64)
    spp = np.zeros(29, dtype=np.float64)
    syp = np.zeros(29, dtype=np.float64)

    subject_errors = {}
    worst_windows = []

    for batch_idx, batch in enumerate(loader):
        z = features(batch, lag)
        y, valid = targets(batch)
        pred = model.predict(z)

        # Accumulate in float64 for numerical robustness.
        y64 = y.astype(np.float64)
        p64 = pred.astype(np.float64)
        error = np.where(valid, p64 - y64, 0.0)
        squared_error += float(np.square(error).sum())
        count += int(valid.sum())

        yf = np.where(valid, y64, 0.0)
        pf = np.where(valid, p64, 0.0)
        zero_squared_error += float(np.square(yf).sum())
        channel_squared_error += np.square(error).sum(axis=0)
        n += valid.sum(axis=0)
        sy += yf.sum(axis=0)
        sp += pf.sum(axis=0)
        syy += np.square(yf).sum(axis=0)
        spp += np.square(pf).sum(axis=0)
        syp += (yf * pf).sum(axis=0)

        b, _, t = batch["in_ear"].shape
        z3 = z.reshape(b, t, -1)
        p3 = p64.reshape(b, t, 29)
        y3 = y64.reshape(b, t, 29)
        v3 = valid.reshape(b, t, 29)
        subject_ids = batch["subject_id"].numpy()
        trial_ids = batch["trial_id"].numpy()
        starts = batch["start_sample"].numpy()

        for j in range(b):
            subject = int(subject_ids[j])
            valid_count = int(v3[j].sum())
            if valid_count == 0:
                continue
            window_error = np.where(v3[j], p3[j] - y3[j], 0.0)
            sse = float(np.square(window_error).sum())
            stats = subject_errors.setdefault(subject, [0.0, 0])
            stats[0] += sse
            stats[1] += valid_count
            worst_windows.append(
                {
                    "subject": subject,
                    "trial": int(trial_ids[j]),
                    "start_sample": int(starts[j]),
                    "batch": batch_idx,
                    "window": j,
                    "mse": sse / valid_count,
                    "max_abs_prediction": float(np.abs(p3[j]).max()),
                    "max_abs_target_valid": float(np.abs(y3[j][v3[j]]).max()),
                    "max_abs_input_feature": float(np.abs(z3[j]).max()),
                    "valid_fraction": valid_count / (t * 29),
                }
            )

    if count == 0:
        raise RuntimeError("No valid scalp targets in validation")

    mse = squared_error / count
    var_y = syy - sy**2 / np.maximum(n, 1)
    var_p = spp - sp**2 / np.maximum(n, 1)
    covariance = syp - sy * sp / np.maximum(n, 1)
    denominator = np.sqrt(np.maximum(var_y * var_p, 0))
    pearson = np.divide(
        covariance,
        denominator,
        out=np.full(29, np.nan),
        where=denominator > 1e-12,
    )
    std_ratio = np.sqrt(
        np.divide(
            var_p,
            var_y,
            out=np.full(29, np.nan),
            where=var_y > 1e-12,
        )
    )

    print("\n=== MSE BY SUBJECT ===")
    for subject, (sse, n_valid) in sorted(subject_errors.items()):
        print(
            f"Subject {subject}: MSE={sse / n_valid:.6f} "
            f"({n_valid:,} valid channel-samples)"
        )

    print("\n=== WORST VALIDATION WINDOWS ===")
    for item in sorted(worst_windows, key=lambda row: row["mse"], reverse=True)[:10]:
        print(item)

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mean_channel_pearson": float(np.nanmean(pearson)),
        "median_std_ratio": float(np.nanmedian(std_ratio)),
        "mse_by_subject": {
            str(subject): float(sse / n_valid)
            for subject, (sse, n_valid) in sorted(subject_errors.items())
        },
        "worst_windows": sorted(
            worst_windows, key=lambda row: row["mse"], reverse=True
        )[:10],
        "n_valid_targets": int(count),
        "n_windows": len(loader.dataset),
        "zero_baseline_mse": float(zero_squared_error / count),
        "skill_vs_zero": float(1 - squared_error / zero_squared_error)
        if zero_squared_error > 0
        else None,
        "mse_by_channel": np.divide(
            channel_squared_error,
            n,
            out=np.full(29, np.nan),
            where=n > 0,
        ).tolist(),
        "pearson_by_channel": pearson.tolist(),
        "std_ratio_by_channel": std_ratio.tolist(),
        "valid_targets_by_channel": n.astype(int).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument(
        "--data-dir",
        default="data/processed/numpy",
    )

    parser.add_argument(
        "--output-dir",
        default="results/masked_lagged_ridge",
    )

    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--no-standardize",
        action="store_true",
    )

    args = parser.parse_args()

    split = make_patient_splits(range(1, 16))[args.fold - 1]

    print("\n=== EXPERIMENT CONFIGURATION ===")
    print(json.dumps(vars(args), indent=2))

    print("\n=== SUBJECT SPLIT ===")
    print(split)

    train_loader = make_loader(
        split["train"],
        args.data_dir,
        args.batch_size,
        shuffle=False,
        manifest_path=args.train_manifest,
    )

    val_loader = make_loader(
        split["val"],
        args.data_dir,
        args.batch_size,
        shuffle=False,
        manifest_path=args.val_manifest,
    )

    print("\n=== BUILDING TRAINING MATRIX ===")

    x_train, y_train, valid_train = train(
        train_loader,
        args.lag,
        args.seed,
    )

    print("\n=== FITTING MASKED RIDGE ===")

    model = MaskedRidge(
        alpha=args.alpha,
        standardize=not args.no_standardize,
    )

    model.fit(
        x_train,
        y_train,
        valid_train,
    )

    print("\n=== EVALUATION ===")

    metrics = evaluate(
        model,
        val_loader,
        args.lag,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = {
        **vars(args),
        "train_subjects": list(split["train"]),
        "val_subjects": list(split["val"]),
        "test_subjects": list(split["test"]),
        "n_train_windows": len(train_loader.dataset),
        "n_val_windows": len(val_loader.dataset),
        "n_train_timestamps": len(x_train),
        "n_train_per_channel": model.n_train_per_channel,
        **metrics,
    }

    (output / "metrics.json").write_text(json.dumps(results, indent=2))

    print("\n=== FINAL METRICS ===")

    for key in (
        "mse",
        "rmse",
        "mean_channel_pearson",
        "median_std_ratio",
        "zero_baseline_mse",
        "skill_vs_zero",
    ):
        print(f"{key}: {metrics[key]}")

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
