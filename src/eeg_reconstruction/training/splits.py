
"""Reproducible subject-wise splits for EAR-SAAD experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    """Subject-level train/validation/test partition."""

    fold: int
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]


def make_subject_splits(
    subject_ids=range(1, 16),
    test_subject: int = 15,
    n_folds: int = 5,
    seed: int = 42,
) -> list[SubjectSplit]:
    """Generate unique repeated holdout splits.

    Each fold has 10 training subjects, 4 validation
    subjects, and one fixed held-out test subject.
    """
    subjects = sorted(set(subject_ids))

    if len(subjects) != 15:
        raise ValueError("Expected exactly 15 subjects")

    if test_subject not in subjects:
        raise ValueError("Test subject must belong to the dataset")

    if not 1 <= n_folds <= 1001:
        raise ValueError("n_folds must be between 1 and 1001")

    candidates = np.array(
        [s for s in subjects if s != test_subject]
    )

    rng = np.random.default_rng(seed)
    seen = set()
    splits = []

    while len(splits) < n_folds:
        val = tuple(sorted(
            rng.choice(candidates, size=4, replace=False).tolist()
        ))

        if val in seen:
            continue

        seen.add(val)

        train = tuple(
            s for s in candidates if s not in val
        )

        splits.append(
            SubjectSplit(
                fold=len(splits),
                train_subjects=train,
                val_subjects=val,
                test_subjects=(test_subject,),
            )
        )

    return splits
