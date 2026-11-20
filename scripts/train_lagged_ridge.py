
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
        parts.append(padded_x[:, :, start:start + t])
        parts.append(padded_m[:, :, start:start + t])

    z = np.concatenate(parts, axis=1)
    return z.transpose(0, 2, 1).reshape(b * t, -1)


def targets(batch):
    y = batch["scalp"].numpy()
    mask = batch["target_mask"].numpy().astype(bool)

    return (
        y.transpose(0, 2, 1).reshape(-1, 29),
        mask.transpose(0, 2, 1).reshape(-1, 29),
    )


def make_loader(subject_ids, data_dir, batch_size, shuffle):
    dataset = EarSAADNumpyDataset(
        data_dir=data_dir,
        subject_ids=subject_ids,
        window_seconds=4.0,
        stride_seconds=4.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train(loader, lag, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []

    for batch in loader:
        z = features(batch, lag)
        y, valid = targets(batch)

        b = batch["in_ear"].shape[0]
        t = batch["in_ear"].shape[-1]

        # Ten randomly sampled timestamps per training window.
        positions = rng.integers(0, t, size=(b, 10))
        indices = (
            np.arange(b)[:, None] * t + positions
        ).reshape(-1)

        indices = indices[valid[indices].all(axis=1)]

        xs.append(z[indices])
        ys.append(y[indices])

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    print("Training matrix:", x_train.shape)
    print("Target matrix:", y_train.shape)

    return x_train, y_train


def evaluate(model, loader, lag):
    squared_error = 0.0
    count = 0

    # Per-channel sufficient statistics.
    n = np.zeros(29)
    sy = np.zeros(29)
    sp = np.zeros(29)
    syy = np.zeros(29)
    spp = np.zeros(29)
    syp = np.zeros(29)

    for batch in loader:
        z = features(batch, lag)
        y, valid = targets(batch)

        pred = model.predict(z)

        error = np.where(valid, pred - y, 0.0)
        squared_error += np.square(error).sum()
        count += valid.sum()

        yf = np.where(valid, y, 0.0)
        pf = np.where(valid, pred, 0.0)

        n += valid.sum(axis=0)
        sy += yf.sum(axis=0)
        sp += pf.sum(axis=0)
        syy += np.square(yf).sum(axis=0)
        spp += np.square(pf).sum(axis=0)
        syp += (yf * pf).sum(axis=0)

    mse = squared_error / count

    var_y = syy - sy**2 / np.maximum(n, 1)
    var_p = spp - sp**2 / np.maximum(n, 1)
    covariance = syp - sy * sp / np.maximum(n, 1)

    denominator = np.sqrt(
        np.maximum(var_y * var_p, 0)
    )

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

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mean_channel_pearson": float(np.nanmean(pearson)),
        "median_std_ratio": float(np.nanmedian(std_ratio)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--data-dir", default="data/processed/numpy"
    )
    parser.add_argument(
        "--output-dir", default="results/lagged_ridge"
    )
    args = parser.parse_args()

    split = make_patient_splits(range(1, 16))[args.fold - 1]

    train_loader = make_loader(
        split["train"], args.data_dir,
        args.batch_size, shuffle=True,
    )
    val_loader = make_loader(
        split["val"], args.data_dir,
        args.batch_size, shuffle=False,
    )

    x_train, y_train = train(
        train_loader, args.lag, args.seed
    )

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=args.alpha, solver="lsqr"),
    )
    model.fit(x_train, y_train)

    metrics = evaluate(model, val_loader, args.lag)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = {
        **vars(args),
        "train_subjects": list(split["train"]),
        "val_subjects": list(split["val"]),
        "test_subjects": list(split["test"]),
        **metrics,
    }

    (output / "metrics.json").write_text(
        json.dumps(results, indent=2)
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
