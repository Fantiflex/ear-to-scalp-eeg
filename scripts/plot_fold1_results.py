"""Plot baseline and Conformer results for fold 1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    baseline_path = Path("results/baselines/fold_1.json")
    history_path = Path(
        "results/conformer_cleaned/in_ear/fold_1/history.csv"
    )
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline results
    baseline_results = json.loads(baseline_path.read_text())

    zero_mse = baseline_results["zero_predictor"]["mse"]
    zero_rmse = baseline_results["zero_predictor"]["rmse"]

    mean_mse = baseline_results["channel_mean_predictor"]["mse"]
    mean_rmse = baseline_results["channel_mean_predictor"]["rmse"]

    # Load Conformer training history
    history = pd.read_csv(history_path)

    best_idx = history["val_mse"].idxmin()
    best_epoch = int(history.loc[best_idx, "epoch"])
    conformer_mse = float(history.loc[best_idx, "val_mse"])
    conformer_rmse = float(history.loc[best_idx, "val_rmse"])

    # --------------------------------------------------
    # Figure 1: validation MSE comparison
    # --------------------------------------------------
    labels = ["Zero", "Channel mean", "Conformer"]
    mse_values = [zero_mse, mean_mse, conformer_mse]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, mse_values)
    plt.ylabel("Validation MSE")
    plt.title("Fold 1: Validation MSE Comparison")

    for i, value in enumerate(mse_values):
        plt.text(i, value + 0.05, f"{value:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_dir / "fold1_val_mse_comparison.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 2: validation RMSE comparison
    # --------------------------------------------------
    rmse_values = [zero_rmse, mean_rmse, conformer_rmse]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, rmse_values)
    plt.ylabel("Validation RMSE")
    plt.title("Fold 1: Validation RMSE Comparison")

    for i, value in enumerate(rmse_values):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_dir / "fold1_val_rmse_comparison.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 3: Conformer learning curves + baselines
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_mse"], label="Train MSE")
    plt.plot(history["epoch"], history["val_mse"], label="Validation MSE")
    plt.axhline(
        zero_mse,
        linestyle="--",
        label=f"Zero baseline ({zero_mse:.3f})",
    )
    plt.axhline(
        mean_mse,
        linestyle=":",
        label=f"Channel mean baseline ({mean_mse:.3f})",
    )
    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch ({best_epoch})",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Fold 1: Conformer Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fold1_learning_curve.png", dpi=300)
    plt.close()

    print("Saved figures to:", output_dir)
    print("Best epoch:", best_epoch)
    print("Best Conformer validation MSE:", f"{conformer_mse:.4f}")


if __name__ == "__main__":
    main()