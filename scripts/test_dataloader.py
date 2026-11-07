
"""Smoke test for synchronized Ear-SAAD windows."""

import torch
from torch.utils.data import DataLoader

from eeg_reconstruction.data.dataset import EarSAADDataset


def main():
    dataset = EarSAADDataset(
        data_dir="data/raw/preprocessedData",
        subject_ids=[1],
        window_seconds=4,
        stride_seconds=4,
    )

    print(f"Sampling frequency: {dataset.fs} Hz")
    print(f"Total windows: {len(dataset)}")

    sample = dataset[0]

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)}")
        else:
            print(f"{key}: {value}")

    assert sample["in_ear"].shape == (12, 80)
    assert sample["around_ear"].shape == (19, 80)
    assert sample["scalp"].shape == (29, 80)

    assert sample["in_ear_mask"].shape == (12, 80)
    assert sample["around_ear_mask"].shape == (19, 80)
    assert sample["target_mask"].shape == (29, 80)

    for modality in ("in_ear", "around_ear", "scalp"):
        assert torch.isfinite(sample[modality]).all()

    print("\nValid sample fractions:")
    for key in ("in_ear_mask", "around_ear_mask", "target_mask"):
        fraction = sample[key].float().mean().item()
        print(f"  {key}: {fraction:.2%}")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["in_ear"].shape == (8, 12, 80)
    assert batch["around_ear"].shape == (8, 19, 80)
    assert batch["scalp"].shape == (8, 29, 80)

    print("\nBatch shapes:")
    print("In-ear:", tuple(batch["in_ear"].shape))
    print("Around-ear:", tuple(batch["around_ear"].shape))
    print("Scalp:", tuple(batch["scalp"].shape))

    print("\nDataLoader test passed.")


if __name__ == "__main__":
    main()
