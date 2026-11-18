
"""Validate subject-wise cross-validation DataLoaders."""

from torch.utils.data import DataLoader

from eeg_reconstruction.data.dataset import EarSAADDataset
from eeg_reconstruction.training.splits import make_subject_splits


def main():
    split = make_subject_splits()[0]

    train_ids = set(split.train_subjects)
    val_ids = set(split.val_subjects)
    test_ids = set(split.test_subjects)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_dataset = EarSAADDataset(
        subject_ids=split.train_subjects,
        window_seconds=4.0,
        stride_seconds=4.0,
    )

    val_dataset = EarSAADDataset(
        subject_ids=split.val_subjects,
        window_seconds=4.0,
        stride_seconds=4.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    actual_train_ids = {s[1] for s in train_dataset.samples}
    actual_val_ids = {s[1] for s in val_dataset.samples}

    assert actual_train_ids == train_ids
    assert actual_val_ids == val_ids
    assert actual_train_ids.isdisjoint(actual_val_ids)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    for batch in (train_batch, val_batch):
        assert batch["in_ear"].shape[1:] == (12, 80)
        assert batch["around_ear"].shape[1:] == (19, 80)
        assert batch["scalp"].shape[1:] == (29, 80)

        for modality in ("in_ear", "around_ear", "scalp"):
            x = batch[modality]
            mask = batch[f"{modality}_observed_mask"]

            assert x.shape == mask.shape
            assert (x[~mask] == 0).all()
            assert x.isfinite().all()

    print(f"Fold: {split.fold}")
    print(f"Train subjects: {sorted(actual_train_ids)}")
    print(f"Val subjects: {sorted(actual_val_ids)}")
    print(f"Test subjects: {sorted(test_ids)}")
    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")
    print(f"Train batch: {tuple(train_batch['in_ear'].shape)}")
    print(f"Val batch: {tuple(val_batch['in_ear'].shape)}")
    print("No subject leakage detected.")
    print("CV DataLoader validation passed.")


if __name__ == "__main__":
    main()
