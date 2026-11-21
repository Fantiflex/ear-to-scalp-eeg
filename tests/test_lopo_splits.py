
"""Validate shared leave-one-participant-out splits."""

from eeg_reconstruction.models.train_val_test_split import (
    make_patient_splits,
)


def test_lopo_splits():
    subjects = set(range(1, 16))
    splits = make_patient_splits(subjects)

    assert len(splits) == 15

    test_subjects = []

    for split in splits:
        train = set(split["train"])
        val = set(split["val"])
        test = set(split["test"])

        assert len(train) == 12
        assert len(val) == 2
        assert len(test) == 1

        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

        assert train | val | test == subjects

        test_subjects.extend(test)

    assert sorted(test_subjects) == sorted(subjects)


def test_deterministic_splits():
    subjects = range(1, 16)

    assert make_patient_splits(subjects) == make_patient_splits(
        subjects
    )
