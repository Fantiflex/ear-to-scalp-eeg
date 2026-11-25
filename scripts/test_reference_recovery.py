
"""Investigate recovery from suspected reference contamination."""

import numpy as np
from scipy.io import loadmat

PATH = "data/raw/preprocessedData/dataSubject2.mat"

data = loadmat(PATH, simplify_cells=True)

channels = np.asarray(data["channels"]).ravel()
trials = np.atleast_1d(data["eegTrials"]).ravel()

IN_EAR = np.r_[30:36, 45:51]
AROUND_EAR = np.r_[36:45, 51:61]


def metrics(x):
    x = x - x.mean(axis=1, keepdims=True)

    s = np.linalg.svd(
        x, compute_uv=False, full_matrices=False
    )

    energy = np.sum(s**2)

    if energy <= 1e-20:
        return np.nan, np.nan, 0.0

    p = s**2 / energy

    rank = np.exp(
        -np.sum(p * np.log(p + 1e-12))
    )

    rms = np.sqrt(np.mean(x**2))

    return p[0], rank, rms


def report(name, x):
    pc1, rank, rms = metrics(x)

    print(
        f"{name:28s} | "
        f"PC1={pc1:8.2%} | "
        f"rank={rank:6.3f} | "
        f"RMS={rms:10.4f}"
    )


def robust_std(x):
    values = x[np.isfinite(x)]

    if len(values) == 0:
        return 0.0

    return 1.4826 * np.median(
        np.abs(values - np.median(values))
    )


for trial_id, trial in enumerate(trials, start=1):

    x = np.asarray(trial, dtype=np.float64)

    if x.shape[0] != 61:
        x = x.T

    if x.shape[0] != 61:
        raise ValueError(f"Unexpected shape: {x.shape}")

    ear = x[IN_EAR]
    around = x[AROUND_EAR]

    names = channels[IN_EAR]

    # Keep sufficiently observed channels.
    keep = np.isfinite(ear).mean(axis=1) >= 0.80

    ear = ear[keep]
    names = names[keep]

    around = around[
        np.isfinite(around).mean(axis=1) >= 0.80
    ]

    if len(around) < 2:
        continue

    # Use identical time samples for all methods.
    complete = (
        np.isfinite(ear).all(axis=0)
        & np.isfinite(around).all(axis=0)
    )

    ear = ear[:, complete]
    around = around[:, complete]

    if ear.shape[1] < 100:
        continue

    ear -= ear.mean(axis=1, keepdims=True)
    around -= around.mean(axis=1, keepdims=True)

    print(f"\n========== TRIAL {trial_id} ==========")
    print(f"Common samples: {ear.shape[1]}")

    report("P0 Original", ear)

    # Remove ELB.
    no_elb = names != "ELB"

    reduced = ear[no_elb]
    reduced_names = names[no_elb]

    report("P1 Without ELB", reduced)

    # Remove flat or nearly flat channels.
    nonflat = np.array([
        robust_std(ch) > 0.01
        for ch in reduced
    ])

    clean = reduced[nonflat]
    clean_names = reduced_names[nonflat]

    if clean.shape[0] < 2:
        continue

    print("Retained channels:", clean_names.tolist())

    report("P2 Without ELB/flatlines", clean)

    # Common average reference.
    car = clean - clean.mean(axis=0, keepdims=True)

    report("P3 CAR", car)

    # Robust common reference.
    median_ref = clean - np.median(
        clean, axis=0, keepdims=True
    )

    report("P4 Median reference", median_ref)

    # Estimate common mode using around-ear.
    common = np.median(around, axis=0)
    common -= common.mean()

    corrected = clean - common[None, :]

    report("P5 Around-ear reference", corrected)

    # Test the predicted -31 relationship.
    if "ELB" in names:

        elb = ear[names == "ELB"][0]

        denom = np.dot(common, common)

        if denom > 1e-12:

            slope = np.dot(elb, common) / denom

            if np.std(elb) > 1e-12:
                corr = np.corrcoef(elb, common)[0, 1]
            else:
                corr = np.nan

            residual = elb + 31 * common

            print("\nReference contamination test:")
            print(f"ELB/common slope: {slope:.4f}")
            print(f"ELB/common Pearson: {corr:.6f}")
            print(
                "RMS(ELB + 31*common): "
                f"{np.sqrt(np.mean(residual**2)):.4f}"
            )
