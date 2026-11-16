
"""Analyze missing EEG observations by subject and channel."""

from collections import defaultdict

import numpy as np

from eeg_reconstruction.data.channels import (
    IN_EAR_CHANNELS,
    SCALP_CHANNELS,
)
from eeg_reconstruction.data.dataset import EarSAADDataset


def main():
    dataset = EarSAADDataset(
        data_dir="data/raw/preprocessedData",
        window_seconds=4.0,
        stride_seconds=4.0,
    )

    stats = defaultdict(
        lambda: {
            "in_ear_valid": np.zeros(len(IN_EAR_CHANNELS)),
            "in_ear_total": np.zeros(len(IN_EAR_CHANNELS)),
            "scalp_valid": np.zeros(len(SCALP_CHANNELS)),
            "scalp_total": np.zeros(len(SCALP_CHANNELS)),
            "windows": 0,
        }
    )

    for sample in dataset:
        sid = sample["subject_id"]
        record = stats[sid]
        record["windows"] += 1

        for modality in ("in_ear", "scalp"):
            mask = sample[
                "target_mask"
                if modality == "scalp"
                else "in_ear_mask"
            ].numpy()

            record[f"{modality}_valid"] += mask.sum(axis=1)
            record[f"{modality}_total"] += mask.shape[1]

    for sid, record in sorted(stats.items()):
        print(f"\nSubject {sid} ({record['windows']} windows)")

        for modality, names in (
            ("in_ear", IN_EAR_CHANNELS),
            ("scalp", SCALP_CHANNELS),
        ):
            valid = record[f"{modality}_valid"]
            total = record[f"{modality}_total"]

            fraction = np.divide(
                valid,
                total,
                out=np.zeros_like(valid),
                where=total > 0,
            )

            print(
                f"  {modality} overall: "
                f"{valid.sum() / total.sum():.2%}"
            )

            for name, value in zip(names, fraction, strict=True):
                if value < 0.90:
                    print(f"    {name}: {value:.2%} valid")


if __name__ == "__main__":
    main()
