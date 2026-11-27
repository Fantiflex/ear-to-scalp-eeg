"""Analyze within-subject Ridge temporal context ablations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("results/within_subject_ridge/fold_01")
OUTPUT = ROOT / "analysis"

EXPECTED_LAGS = [0, 1, 2, 5, 10]
EXPECTED_ALPHAS = [10, 100, 1000, 10000]

METRICS = {
    "mean_trial_pearson": "Mean Pearson correlation",
    "mean_trial_skill": "Mean skill vs zero",
    "mean_trial_mse": "Mean MSE",
    "median_trial_std_ratio": "Median std ratio",
}


def load_results():
    records = []

    for path in sorted(ROOT.glob("lag_*_alpha_*/subject_summary.csv")):
        parts = path.parent.name.split("_")

        df = pd.read_csv(path)
        df["lag"] = int(parts[1])
        df["alpha"] = float(parts[3])

        records.append(df)

    if not records:
        raise RuntimeError("No ablation results found")

    results = pd.concat(records, ignore_index=True)

    if results.duplicated(["subject_id", "lag", "alpha"]).any():
        raise RuntimeError("Duplicate configurations found")

    return results


def check_completeness(results):
    expected = {(lag, alpha) for lag in EXPECTED_LAGS for alpha in EXPECTED_ALPHAS}

    for subject, group in results.groupby("subject_id"):
        observed = set(zip(group["lag"], group["alpha"]))

        missing = expected - observed

        print(f"\nSubject {subject:02d}")
        print(f"Configurations: {len(observed)}/{len(expected)}")

        if missing:
            print("Missing configurations:", sorted(missing))


def plot_heatmap(group, subject, metric, title):
    pivot = group.pivot(
        index="lag",
        columns="alpha",
        values=metric,
    ).reindex(
        index=EXPECTED_LAGS,
        columns=EXPECTED_ALPHAS,
    )

    values = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    if metric == "mean_trial_mse":
        cmap = "viridis_r"
    elif metric == "mean_trial_skill":
        cmap = "RdYlGn"
    else:
        cmap = "viridis"

    kwargs = {}

    if metric == "mean_trial_skill":
        limit = np.nanmax(np.abs(values))
        limit = max(limit, 1e-8)
        kwargs = {"vmin": -limit, "vmax": limit}

    im = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        **kwargs,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{alpha:g}" for alpha in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Ridge regularization (alpha)")
    ax.set_ylabel("Temporal lag (samples)")
    ax.set_title(f"Subject {subject:02d} — {title}")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = values[i, j]

            if np.isnan(value):
                ax.text(
                    j,
                    i,
                    "N/A",
                    ha="center",
                    va="center",
                )
                continue

            rgba = im.cmap(im.norm(value))

            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]

            text_color = "black" if luminance > 0.5 else "white"

            label = (
                f"{100 * value:.1f}%"
                if metric == "mean_trial_skill"
                else f"{value:.3f}"
            )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    # Highlight the best configuration.
    # Std ratio is excluded because maximizing amplitude
    # does not necessarily improve reconstruction.
    if metric != "median_trial_std_ratio":
        if metric == "mean_trial_mse":
            best_idx = np.nanargmin(values)
        else:
            best_idx = np.nanargmax(values)

        i, j = np.unravel_index(
            best_idx,
            values.shape,
        )

        rectangle = plt.Rectangle(
            (j - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
        )

        ax.add_patch(rectangle)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"s{subject:02d}_{metric}.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_lag_curves(group, subject, metric, title):
    fig, ax = plt.subplots(figsize=(7, 5))

    for alpha, subset in group.groupby("alpha"):
        subset = subset.sort_values("lag")

        ax.plot(
            subset["lag"],
            subset[metric],
            marker="o",
            label=f"alpha={alpha:g}",
        )

    ax.set_xlabel("Temporal lag (samples)")
    ax.set_ylabel(title)
    ax.set_title(f"Subject {subject:02d} — {title}")
    ax.set_xticks(EXPECTED_LAGS)

    if metric == "mean_trial_skill":
        ax.axhline(
            0,
            linestyle="--",
            color="gray",
            label="Zero predictor",
        )

    if metric == "median_trial_std_ratio":
        ax.axhline(
            1,
            linestyle="--",
            color="gray",
            label="Target amplitude",
        )

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"s{subject:02d}_{metric}_curves.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def summarize_best(results):
    records = []

    for subject, group in results.groupby("subject_id"):
        criteria = {
            "best_mse": group["mean_trial_mse"].idxmin(),
            "best_pearson": (group["mean_trial_pearson"].idxmax()),
            "best_skill": (group["mean_trial_skill"].idxmax()),
        }

        for criterion, idx in criteria.items():
            record = results.loc[idx].to_dict()
            record["criterion"] = criterion
            records.append(record)

    return pd.DataFrame(records)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = load_results()

    check_completeness(results)

    columns = [
        "subject_id",
        "lag",
        "alpha",
        "n_trials",
        "mean_trial_mse",
        "mean_trial_pearson",
        "std_trial_pearson",
        "median_trial_std_ratio",
        "mean_trial_skill",
    ]

    results = results.sort_values(["subject_id", "lag", "alpha"]).reset_index(drop=True)

    results[columns].to_csv(
        OUTPUT / "ablation_summary.csv",
        index=False,
    )

    for subject, group in results.groupby("subject_id"):
        print(f"\n=== SUBJECT {subject:02d} ===")

        print(
            group[columns]
            .sort_values(
                "mean_trial_skill",
                ascending=False,
            )
            .to_string(index=False)
        )

        for metric, title in METRICS.items():
            plot_heatmap(group, subject, metric, title)

            plot_lag_curves(group, subject, metric, title)

    best = summarize_best(results)

    best.to_csv(
        OUTPUT / "best_configurations.csv",
        index=False,
    )

    print("\n=== BEST CONFIGURATIONS ===")

    print(best[["criterion"] + columns].to_string(index=False))

    # Equal weighting of subjects.
    # Only retain configurations available for every subject.
    n_subjects = results["subject_id"].nunique()

    macro = (
        results.groupby(["lag", "alpha"])
        .agg(
            n_subjects=("subject_id", "nunique"),
            macro_pearson=("mean_trial_pearson", "mean"),
            macro_mse=("mean_trial_mse", "mean"),
            macro_skill=("mean_trial_skill", "mean"),
            macro_std_ratio=(
                "median_trial_std_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    macro = macro[macro["n_subjects"] == n_subjects].sort_values(
        "macro_skill",
        ascending=False,
    )

    macro.to_csv(
        OUTPUT / "macro_summary.csv",
        index=False,
    )

    print("\n=== MACRO SUBJECT RESULTS ===")
    print(macro.to_string(index=False))

    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    main()
