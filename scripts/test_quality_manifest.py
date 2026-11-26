
"""Validate D3 manifest integration with the NumPy dataset."""

import csv
from pathlib import Path

from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset

MANIFEST = Path(
    "data/processed/manifests/fold_01/d3_manifest.csv"
)

PARTITIONS = {
    "train": list(range(4, 16)),
    "val": [2, 3],
    "test": [1],
}

EXPECTED = {
    "train": 4315,
    "val": 630,
    "test": 310,
}


def main():
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))

    import json

    manifest = {
        (
            int(row["subject_id"]),
            int(row["trial_id"]),
            int(row["start_sample"]),
        ): row
        for row in rows
    }

    for partition, subjects in PARTITIONS.items():
        dataset = EarSAADNumpyDataset(
            subject_ids=subjects,
            manifest_path=MANIFEST,
        )

        print(f"\n=== {partition.upper()} ===")
        print("Windows:", len(dataset))

        assert len(dataset) == EXPECTED[partition]

        masked_ear = 0
        masked_scalp = 0

        for sample in dataset:
            key = (
                int(sample["subject_id"]),
                int(sample["trial_id"]),
                int(sample["start_sample"]),
            )

            row = manifest[key]

            assert int(row["video_condition"]) == 0
            assert sample["in_ear"].shape == (12, 80)
            assert sample["scalp"].shape == (29, 80)

            for modality, column in (
                ("in_ear", "bad_ear_channels"),
                ("scalp", "bad_scalp_channels"),
            ):
                bad = json.loads(row[column])
                mask = sample[f"{modality}_observed_mask"]

                for c in bad:
                    assert not mask[c].any()
                    assert (sample[modality][c] == 0).all()

                if modality == "in_ear":
                    masked_ear += len(bad)
                else:
                    masked_scalp += len(bad)

            assert (
                sample["target_mask"]
                == sample["scalp_observed_mask"]
            ).all()

        print("Masked ear channel-windows:", masked_ear)
        print("Masked scalp channel-windows:", masked_scalp)
        print("All checks passed.")

    print("\nD3 MANIFEST VALIDATION PASSED")


if __name__ == "__main__":
    main()
