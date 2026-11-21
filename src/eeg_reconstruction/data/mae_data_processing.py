
"""Dataset for preprocessed Ear-SAAD EEG stored in .npy files."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Channel indices
# ============================================================

N_SCALP = 29
N_IN_EAR = 12
N_AROUND_EAR = 19


SCALP_INDICES = np.arange(0, 29)
IN_EAR_INDICES = np.arange(29, 41)
AROUND_EAR_INDICES = np.arange(41, 60)


class EarSAADNPYDataset(Dataset):
    """Load preprocessed EEG and extract synchronized EEG windows.

    Each subject has one .npy file with shape (72000, 61).

    Returns:
        around_ear: (19, window_size)
        in_ear:    (12, window_size)
        scalp:      (29, window_size)
        subject_id:  str
        start_sample: int
    """

    def __init__(
        self,
        data_dir="data/numpy",
        subject_ids=None,
        fs=20,
        window_seconds=4.0,
        stride_seconds=4.0,
    ):
        self.data_dir = Path(data_dir)

        self.fs = fs
        self.window_size = round(window_seconds * fs)
        self.stride_size = round(stride_seconds * fs)

        if self.window_size <= 0 or self.stride_size <= 0:
            raise ValueError("Window size and stride must be positive.")

        # Discover subject files
        files = sorted(self.data_dir.glob("subject_*.npy"))

        if subject_ids is not None:
            subject_ids = set(subject_ids)
            files = [
                f for f in files
                if f.stem.replace("subject_", "") in subject_ids
            ]

        if not files:
            raise FileNotFoundError(
                f"No subject files found in {self.data_dir}"
            )

        # Index windows without loading full arrays into memory
        self.windows = []

        for path in files:
            subject_id = path.stem.replace("subject_", "")

            arr = np.load(path, mmap_mode="r")

            if arr.ndim != 2:
                raise ValueError(
                    f"{path.name}: expected 2D array (time, channels), "
                    f"got {arr.shape}"
                )

            if arr.shape[1] < 60:
                raise ValueError(
                    f"{path.name}: expected at least 60 columns, "
                    f"got {arr.shape[1]}"
                )

            n_samples = arr.shape[0]

            for start in range(
                0,
                n_samples - self.window_size + 1,
                self.stride_size,
            ):
                self.windows.append(
                    (path, subject_id, start)
                )

        print(
            f"Loaded {len(files)} subjects | "
            f"{len(self.windows)} windows | "
            f"window={self.window_size} samples | "
            f"stride={self.stride_size} samples"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, subject_id, start = self.windows[idx]

        arr = np.load(path, mmap_mode="r")

        end = start + self.window_size
        window = np.asarray(arr[start:end])

        # Extract channels
        scalp = window[:, SCALP_INDICES].T
        around_ear = window[:, AROUND_EAR_INDICES].T

        # Convert to float32 tensors
        scalp = torch.from_numpy(
            np.array(scalp, dtype=np.float32, copy=True)
        )

        around_ear = torch.from_numpy(
            np.array(around_ear, dtype=np.float32, copy=True)
        )

        in_ear = torch.from_numpy(
            np.array(window[:, IN_EAR_INDICES].T, dtype=np.float32, copy=True)
        )
        return {
            "around_ear": around_ear,
            "in_ear": in_ear,
            "scalp": scalp,
            "subject_id": subject_id,
            "start_sample": start,
        }
