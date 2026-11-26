
"""Build a fold-specific quality manifest for no-video EEG."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset
from eeg_reconstruction.models.train_val_test_split import make_patient_splits


def channel_stats(x):
    """Return robust scale, centered RMS and observed fraction."""
    observed = np.isfinite(x).mean(axis=0)
    mad = np.full(x.shape[1], np.nan)
    rms = np.full(x.shape[1], np.nan)

    for c in range(x.shape[1]):
        v = x[np.isfinite(x[:, c]), c]

        if len(v) < 20:
            continue

        mad[c] = 1.4826 * np.median(
            np.abs(v - np.median(v))
        )
        rms[c] = np.sqrt(np.mean((v - v.mean()) ** 2))

    return mad, rms, observed


def pc1_fraction(x):
    """Estimate spatial variance concentration."""
    x = x[np.isfinite(x).all(axis=1)]

    if x.shape[0] < 100 or x.shape[1] < 2:
        return np.nan

    x = x - x.mean(axis=0, keepdims=True)

    s = np.linalg.svd(
        x, compute_uv=False, full_matrices=False
    )

    energy = np.sum(s**2)

    if energy <= 1e-20:
        return np.nan

    return float(s[0] ** 2 / energy)


def load_subject(root, subject):
    stem = root / f"subject_{subject:02d}"

    x = np.load(
        stem.with_suffix(".npy"),
        mmap_mode="r",
    )

    meta = json.loads(
        stem.with_suffix(".json").read_text()
    )

    return x, meta


def load_video(mat_root, subject):
    data = loadmat(
        mat_root / f"dataSubject{subject}.mat",
        simplify_cells=True,
        variable_names=["videoCondition"],
    )

    return np.asarray(data["videoCondition"]).ravel()


def trials(x, meta):
    offsets = meta["trial_offsets"]

    for i, (start, end) in enumerate(
        zip(offsets[:-1], offsets[1:])
    ):
        yield i + 1, np.asarray(
            x[start:end], dtype=np.float64
        )


def calibrate(root, mat_root, subjects, no_video):
    """Calibrate robust channel references on training subjects."""
    values = {"in_ear": [], "scalp": []}

    for subject in subjects:
        x, meta = load_subject(root, subject)
        video = load_video(mat_root, subject)

        for trial_id, trial in trials(x, meta):
            if int(video[trial_id - 1]) != no_video:
                continue

            for modality in values:
                signal = trial[:, meta["indices"][modality]]

                mad, rms, observed = channel_stats(signal)

                valid = (
                    (mad > 0.01)
                    & (observed >= 0.80)
                )

                values[modality].append(
                    np.where(valid, rms, np.nan)
                )

    refs = {}

    for modality, rows in values.items():
        if not rows:
            raise RuntimeError(
                f"No calibration trials for {modality}"
            )

        refs[modality] = np.nanmedian(
            np.stack(rows), axis=0
        )

    return refs


def good_channels(x, ref, args):
    mad, rms, observed = channel_stats(x)

    return (
        (observed >= args.min_channel_observed)
        & (mad > args.min_mad)
        & np.isfinite(ref)
        & np.isfinite(rms)
        & (rms <= args.rms_factor * ref)
    )


def window_quality(x, ref, good, args):
    """Return observed fraction and extreme amplitude fraction."""
    if not good.any():
        return 0.0, 1.0

    x = x[:, good]
    ref = ref[good]

    observed = np.isfinite(x)

    extreme = observed & (
        np.abs(x) > args.peak_factor * ref[None, :]
    )

    extreme_fraction = (
        extreme.sum() / max(observed.sum(), 1)
    )

    return float(observed.mean()), float(extreme_fraction)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--no-video", type=int, required=True)

    parser.add_argument(
        "--data-dir", default="data/processed/numpy"
    )
    parser.add_argument(
        "--mat-dir", default="data/raw/preprocessedData"
    )
    parser.add_argument(
        "--output-dir", default="data/processed/manifests"
    )

    parser.add_argument("--min-mad", type=float, default=0.01)
    parser.add_argument("--rms-factor", type=float, default=5.0)
    parser.add_argument("--peak-factor", type=float, default=8.0)
    parser.add_argument("--max-pc1", type=float, default=0.95)

    parser.add_argument(
        "--min-channel-observed", type=float, default=0.80
    )
    parser.add_argument(
        "--min-ear-channels", type=int, default=8
    )
    parser.add_argument(
        "--min-scalp-channels", type=int, default=27
    )
    parser.add_argument(
        "--min-ear-observed", type=float, default=0.90
    )
    parser.add_argument(
        "--min-scalp-observed", type=float, default=0.95
    )
    parser.add_argument(
        "--max-extreme-fraction", type=float, default=0.01
    )

    args = parser.parse_args()

    root = Path(args.data_dir)
    mat_root = Path(args.mat_dir)

    output = (
        Path(args.output_dir) / f"fold_{args.fold:02d}"
    )
    output.mkdir(parents=True, exist_ok=True)

    split = make_patient_splits(
        range(1, 16)
    )[args.fold - 1]

    refs = calibrate(
        root,
        mat_root,
        split["train"],
        args.no_video,
    )

    # Preserve the existing dataset's window exclusions.
    base = EarSAADNumpyDataset(
        data_dir=root,
        subject_ids=range(1, 16),
        window_seconds=4.0,
        stride_seconds=4.0,
    )

    allowed = {
        (subject, trial_idx + 1, start)
        for _, subject, trial_idx, start, _ in base.samples
    }

    trial_rows = []
    window_rows = []

    for subject in range(1, 16):
        x, meta = load_subject(root, subject)
        video = load_video(mat_root, subject)

        if len(video) != len(meta["trial_offsets"]) - 1:
            raise ValueError(
                f"Video/trial mismatch for S{subject:02d}"
            )

        partition = next(
            key for key in ("train", "val", "test")
            if subject in split[key]
        )

        size = round(4 * float(meta["fs"]))

        for trial_id, trial in trials(x, meta):
            condition = int(video[trial_id - 1])

            ear = trial[:, meta["indices"]["in_ear"]]
            scalp = trial[:, meta["indices"]["scalp"]]

            good_ear = good_channels(
                ear, refs["in_ear"], args
            )
            good_scalp = good_channels(
                scalp, refs["scalp"], args
            )

            ear_pc1 = pc1_fraction(ear[:, good_ear])
            scalp_pc1 = pc1_fraction(scalp[:, good_scalp])

            reasons = []

            if condition != args.no_video:
                reasons.append("video")

            if good_ear.sum() < args.min_ear_channels:
                reasons.append("ear_channels")

            if good_scalp.sum() < args.min_scalp_channels:
                reasons.append("scalp_channels")

            if not np.isfinite(ear_pc1):
                reasons.append("ear_rank_unavailable")
            elif ear_pc1 >= args.max_pc1:
                reasons.append("ear_rank_collapse")

            if not np.isfinite(scalp_pc1):
                reasons.append("scalp_rank_unavailable")
            elif scalp_pc1 >= args.max_pc1:
                reasons.append("scalp_rank_collapse")

            trial_ok = not reasons

            trial_rows.append({
                "subject_id": subject,
                "partition": partition,
                "trial_id": trial_id,
                "video_condition": condition,
                "good_ear_channels": int(good_ear.sum()),
                "good_scalp_channels": int(good_scalp.sum()),
                "ear_pc1": ear_pc1,
                "scalp_pc1": scalp_pc1,
                "accepted": trial_ok,
                "reason": ";".join(reasons),
            })

            retained = 0

            for start in range(0, len(trial) - size + 1, size):
                end = start + size
                window_reasons = reasons.copy()

                ew = ear[start:end]
                sw = scalp[start:end]

                eo, ee = window_quality(
                    ew, refs["in_ear"], good_ear, args
                )
                so, se = window_quality(
                    sw, refs["scalp"], good_scalp, args
                )

                ear_mad, _, ear_obs = channel_stats(ew)

                window_good_ear = (
                    good_ear
                    & (ear_obs >= args.min_ear_observed)
                    & (ear_mad > args.min_mad)
                )

                if window_good_ear.sum() < args.min_ear_channels:
                    window_reasons.append("window_ear_channels")

                if eo < args.min_ear_observed:
                    window_reasons.append("ear_missingness")

                if so < args.min_scalp_observed:
                    window_reasons.append("scalp_missingness")

                if ee > args.max_extreme_fraction:
                    window_reasons.append("ear_artifact")

                if se > args.max_extreme_fraction:
                    window_reasons.append("scalp_artifact")

                absolute_start = (
                    meta["trial_offsets"][trial_id - 1] + start
                )

                if (subject, trial_id, absolute_start) not in allowed:
                    window_reasons.append("existing_qc")

                accepted = not window_reasons
                retained += int(accepted)

                window_rows.append({
                    "subject_id": subject,
                    "partition": partition,
                    "trial_id": trial_id,
                    "start_sample": start,
                    "end_sample": end,
                    "video_condition": condition,
                    "ear_observed": eo,
                    "scalp_observed": so,
                    "ear_extreme": ee,
                    "scalp_extreme": se,
                    "bad_ear_channels": json.dumps(
                        np.flatnonzero(
                            ~window_good_ear
                        ).tolist()
                    ),
                    "bad_scalp_channels": json.dumps(
                        np.flatnonzero(~good_scalp).tolist()
                    ),
                    "accepted": accepted,
                    "reason": ";".join(window_reasons),
                })

            print(
                f"S{subject:02d} T{trial_id} "
                f"video={condition} "
                f"ear={good_ear.sum()}/12 "
                f"scalp={good_scalp.sum()}/29 "
                f"PC1={ear_pc1:.3f} "
                f"windows={retained}/{len(trial) // size} "
                f"{';'.join(reasons)}"
            )

    trial_df = pd.DataFrame(trial_rows)
    window_df = pd.DataFrame(window_rows)
    manifest = window_df[window_df["accepted"]].copy()

    trial_df.to_csv(output / "trial_quality.csv", index=False)
    window_df.to_csv(output / "window_quality.csv", index=False)
    manifest.to_csv(output / "d3_manifest.csv", index=False)

    parameters = {
        **vars(args),
        "train_subjects": list(split["train"]),
        "reference_rms": {
            k: v.tolist() for k, v in refs.items()
        },
    }

    (output / "qc_parameters.json").write_text(
        json.dumps(parameters, indent=2)
    )

    print("\n=== COVERAGE BY PARTITION ===")

    print(
        window_df.groupby("partition")["accepted"]
        .agg(["sum", "count", "mean"])
    )

    print("\n=== COVERAGE BY SUBJECT ===")

    print(
        window_df.groupby("subject_id")["accepted"]
        .agg(["sum", "count", "mean"])
    )

    print("\n=== REJECTION REASONS ===")

    print(
        window_df.loc[~window_df["accepted"], "reason"]
        .str.split(";")
        .explode()
        .value_counts()
    )

    print(f"\nRetained {len(manifest)}/{len(window_df)} windows")
    print(f"Duration: {len(manifest) * 4 / 3600:.2f} hours")
    print(f"Manifest saved to {output}")


if __name__ == "__main__":
    main()
