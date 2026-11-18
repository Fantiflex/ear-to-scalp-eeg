
"""Check that MATLAB and NumPy datasets yield equivalent EEG windows."""

import numpy as np

from eeg_reconstruction.data.dataset import EarSAADDataset
from eeg_reconstruction.data.numpy_dataset import EarSAADNumpyDataset


def test_numpy_matches_matlab():
    for subject in (1, 8, 15):
        original = EarSAADDataset(subject_ids=[subject])
        converted = EarSAADNumpyDataset(subject_ids=[subject])

        assert len(original) == len(converted)

        indices = sorted({
            0,
            len(original) // 2,
            len(original) - 1,
        })

        for idx in indices:
            a = original[idx]
            b = converted[idx]

            for key in ("subject_id", "trial_id", "start_sample"):
                assert a[key] == b[key]

            for modality in ("in_ear", "around_ear", "scalp"):
                np.testing.assert_allclose(
                    a[modality].numpy(),
                    b[modality].numpy(),
                    rtol=1e-5,
                    atol=1e-6,
                )

                assert (
                    a[f"{modality}_observed_mask"]
                    == b[f"{modality}_observed_mask"]
                ).all()
