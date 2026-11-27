"""Evaluate within-trial EEG reconstruction using Masked Ridge."""

import argparse
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Subset
from train_lagged_ridge import MaskedRidge, evaluate, train

from eeg_reconstruction.data.numpy_dataset import (
    EarSAADNumpyDataset,
)

SAMPLING_RATE = 20
TRIAL_DURATION = 600
WINDOW_SECONDS = 4
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS


def make_loader(dataset, indices, batch_size):
    """Create a deterministic loader for a temporal partition."""

    if not indices:
        raise RuntimeError("Empty temporal partition")

    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def split_trial(dataset, indices, gap_seconds):
    """Split a trial chronologically with temporal exclusion gaps."""

    trial_samples = TRIAL_DURATION * SAMPLING_RATE
    gap = int(gap_seconds * SAMPLING_RATE)

    boundary_1 = int(0.6 * trial_samples)
    boundary_2 = int(0.8 * trial_samples)

    partitions = {
        "train": [],
        "val": [],
        "test": [],
    }

    for idx in indices:
        sample = dataset[idx]

        start = int(sample["start_sample"])
        end = start + WINDOW_SAMPLES

        if end <= boundary_1 - gap // 2:
            partitions["train"].append(idx)

        elif start >= boundary_1 + gap // 2 and end <= boundary_2 - gap // 2:
            partitions["val"].append(idx)

        elif start >= boundary_2 + gap // 2 and end <= trial_samples:
            partitions["test"].append(idx)

    train_set = set(partitions["train"])
    val_set = set(partitions["val"])
    test_set = set(partitions["test"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    return partitions


def describe_partition(dataset, indices):
    """Report the actual temporal coverage of a partition."""

    if not indices:
        return {
            "n_windows": 0,
            "start_seconds": None,
            "end_seconds": None,
        }

    starts = [int(dataset[idx]["start_sample"]) for idx in indices]

    return {
        "n_windows": len(indices),
        "start_seconds": min(starts) / SAMPLING_RATE,
        "end_seconds": (max(starts) + WINDOW_SAMPLES) / SAMPLING_RATE,
    }


def run_subject(subject_id, args):
    """Train and evaluate independent models for each trial."""

    dataset = EarSAADNumpyDataset(
        data_dir=args.data_dir,
        subject_ids=[subject_id],
        window_seconds=WINDOW_SECONDS,
        stride_seconds=WINDOW_SECONDS,
        manifest_path=args.manifest,
    )

    trials = {}

    for idx, sample in enumerate(dataset.samples):
        trial_id = sample[2] + 1
        trials.setdefault(trial_id, []).append(idx)

    print(f"\n=== SUBJECT {subject_id:02d} ===")
    print(
        "Available trials:",
        {trial_id: len(indices) for trial_id, indices in trials.items()},
    )

    records = []

    for trial_id, indices in sorted(trials.items()):
        print(f"\n--- Subject {subject_id:02d} | Trial {trial_id} ---")

        partitions = split_trial(
            dataset,
            indices,
            args.gap_seconds,
        )

        descriptions = {
            name: describe_partition(dataset, subset)
            for name, subset in partitions.items()
        }

        counts = {name: len(subset) for name, subset in partitions.items()}

        for name, description in descriptions.items():
            print(f"{name.upper()}: {description}")

        excluded = len(indices) - sum(counts.values())

        print("Excluded boundary windows:", excluded)

        if min(counts.values()) < args.min_windows:
            print("Skipping trial: insufficient windows in at least one partition")
            continue

        # Verify temporal separation.
        for left, right in [
            ("train", "val"),
            ("val", "test"),
        ]:
            actual_gap = (
                descriptions[right]["start_seconds"] - descriptions[left]["end_seconds"]
            )

            assert actual_gap >= args.gap_seconds

            print(f"Temporal gap {left}/{right}: {actual_gap:.1f} seconds")

        train_loader = make_loader(
            dataset,
            partitions["train"],
            args.batch_size,
        )

        val_loader = make_loader(
            dataset,
            partitions["val"],
            args.batch_size,
        )

        test_loader = make_loader(
            dataset,
            partitions["test"],
            args.batch_size,
        )

        # Extract training features and valid target masks.
        x, y, valid = train(
            train_loader,
            args.lag,
            args.seed,
        )

        print("Training features:", x.shape)
        print("Training targets:", y.shape)

        # Fit exclusively on the training segment.
        model = MaskedRidge(
            alpha=args.alpha,
            standardize=False,
        )

        model.fit(x, y, valid)

        # Evaluate on independent temporal segments.
        evaluation_loaders = {
            "val": val_loader,
        }

        if args.evaluate_test:
            evaluation_loaders["test"] = test_loader

        for partition, loader in evaluation_loaders.items():
            metrics = evaluate(
                model,
                loader,
                args.lag,
            )

            record = {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "partition": partition,
                "lag": args.lag,
                "alpha": args.alpha,
                "seed": args.seed,
                "gap_seconds": args.gap_seconds,
                "n_train_windows": counts["train"],
                "n_eval_windows": counts[partition],
                "n_excluded_windows": excluded,
                "train_start_seconds": (descriptions["train"]["start_seconds"]),
                "train_end_seconds": (descriptions["train"]["end_seconds"]),
                "eval_start_seconds": (descriptions[partition]["start_seconds"]),
                "eval_end_seconds": (descriptions[partition]["end_seconds"]),
                **metrics,
            }

            records.append(record)

            print(
                f"{partition.upper()} | "
                f"MSE: {metrics['mse']:.4f} | "
                f"RMSE: {metrics['rmse']:.4f} | "
                f"Pearson: "
                f"{metrics['mean_channel_pearson']:.4f} | "
                f"Std ratio: "
                f"{metrics['median_std_ratio']:.4f} | "
                f"Skill: "
                f"{100 * metrics['skill_vs_zero']:.2f}%"
            )

    return records


def save_results(records, args):
    """Save trial-level and aggregated results."""

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "metrics.json").write_text(json.dumps(records, indent=2))

    (output / "config.json").write_text(json.dumps(vars(args), indent=2))

    columns = [
        "subject_id",
        "trial_id",
        "partition",
        "lag",
        "alpha",
        "n_train_windows",
        "n_eval_windows",
        "n_excluded_windows",
        "mse",
        "rmse",
        "mean_channel_pearson",
        "median_std_ratio",
        "zero_baseline_mse",
        "skill_vs_zero",
    ]

    df = pd.DataFrame(records)[columns]

    df.to_csv(
        output / "trial_metrics.csv",
        index=False,
    )

    # Equal weighting of trials within each participant.
    summary = (
        df.groupby(["subject_id", "partition"])
        .agg(
            n_trials=("trial_id", "count"),
            mean_trial_mse=("mse", "mean"),
            mean_trial_pearson=(
                "mean_channel_pearson",
                "mean",
            ),
            std_trial_pearson=(
                "mean_channel_pearson",
                lambda x: x.std(ddof=0),
            ),
            median_trial_std_ratio=(
                "median_std_ratio",
                "median",
            ),
            mean_trial_skill=("skill_vs_zero", "mean"),
        )
        .reset_index()
    )

    summary["lag"] = args.lag
    summary["alpha"] = args.alpha

    summary.to_csv(
        output / "subject_summary.csv",
        index=False,
    )

    # Equal weighting of participants.
    macro = (
        summary.groupby("partition")
        .agg(
            n_subjects=("subject_id", "nunique"),
            macro_mse=("mean_trial_mse", "mean"),
            macro_pearson=("mean_trial_pearson", "mean"),
            macro_skill=("mean_trial_skill", "mean"),
            macro_std_ratio=(
                "median_trial_std_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    macro["lag"] = args.lag
    macro["alpha"] = args.alpha

    macro.to_csv(
        output / "macro_summary.csv",
        index=False,
    )

    print("\n=== TRIAL RESULTS ===")
    print(df.to_string(index=False))

    print("\n=== SUBJECT SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== MACRO SUMMARY ===")
    print(macro.to_string(index=False))

    print("\nResults saved to:", output)


def main():
    parser = argparse.ArgumentParser(
        description="Within-trial Masked Ridge reconstruction"
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=[2, 3],
    )

    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gap-seconds", type=float, default=8)
    parser.add_argument("--min-windows", type=int, default=10)

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the reserved temporal test segment",
    )

    parser.add_argument(
        "--data-dir",
        default="data/processed/numpy",
    )

    parser.add_argument(
        "--manifest",
        default=("data/processed/manifests/fold_01/d3_manifest.csv"),
    )

    parser.add_argument(
        "--output-dir",
        default="results/within_trial_ridge/fold_01",
    )

    args = parser.parse_args()

    if 1 in args.subjects:
        parser.error("Subject 1 is reserved for final evaluation")

    if args.lag < 0 or args.lag >= WINDOW_SAMPLES:
        parser.error("Invalid temporal lag")

    if args.alpha <= 0:
        parser.error("Alpha must be positive")

    if not 0 <= args.gap_seconds < 120:
        parser.error("Invalid temporal gap")

    if not Path(args.manifest).is_file():
        parser.error(f"Manifest not found: {args.manifest}")

    print("=== WITHIN-TRIAL RIDGE ===")
    print(json.dumps(vars(args), indent=2))

    records = []

    for subject_id in args.subjects:
        records.extend(run_subject(subject_id, args))

    if not records:
        raise RuntimeError("No eligible trials found")

    save_results(records, args)


if __name__ == "__main__":
    main()
