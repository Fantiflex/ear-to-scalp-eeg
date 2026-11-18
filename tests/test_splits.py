
"""Tests for subject-wise cross-validation splits."""

import pytest

from eeg_reconstruction.training.splits import make_subject_splits


def test_split_sizes():
    splits = make_subject_splits()

    assert len(splits) == 5

    for split in splits:
        assert len(split.train_subjects) == 10
        assert len(split.val_subjects) == 4
        assert len(split.test_subjects) == 1


def test_no_subject_leakage():
    for split in make_subject_splits():
        train = set(split.train_subjects)
        val = set(split.val_subjects)
        test = set(split.test_subjects)

        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

        assert train | val | test == set(range(1, 16))


def test_reproducibility():
    splits_a = make_subject_splits(seed=42)
    splits_b = make_subject_splits(seed=42)

    assert splits_a == splits_b


def test_unique_validation_sets():
    splits = make_subject_splits(n_folds=5)

    val_sets = [split.val_subjects for split in splits]

    assert len(val_sets) == len(set(val_sets))


def test_fixed_test_subject():
    splits = make_subject_splits(test_subject=15)

    for split in splits:
        assert split.test_subjects == (15,)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        make_subject_splits(subject_ids=range(1, 10))

    with pytest.raises(ValueError):
        make_subject_splits(test_subject=20)

    with pytest.raises(ValueError):
        make_subject_splits(n_folds=0)
