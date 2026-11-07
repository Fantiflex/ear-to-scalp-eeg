
"""Synchronized PyTorch dataset for Ear-SAAD."""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from eeg_reconstruction.data.channels import (
    CHANNEL_GROUPS,
    get_channel_indices,
)


def _read_trials(value):
    """Convert MATLAB cell arrays into a list of trials."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [np.asarray(item) for item in value.flat]

    if isinstance(value, (list, tuple)):
        return [np.asarray(item) for item in value]

    return [np.asarray(value)]


@lru_cache(maxsize=2)
def _load_subject(path):
    """Load one subject, caching up to two subject files."""
    mat = loadmat(path, simplify_cells=True)

    channels = [
        str(name)
        for name in np.asarray(mat["channels"]).reshape(-1)
    ]

    indices = {
        modality: get_channel_indices(channels, modality)
        for modality in CHANNEL_GROUPS
    }

    fs = float(np.asarray(mat["fs"]).squeeze())
    trials = _read_trials(mat["eegTrials"])

    for trial in trials:
        if trial.ndim != 2 or trial.shape[1] != len(channels):
            raise ValueError(f"Invalid trial dimensions in {path}")

    return trials, indices, fs


class EarSAADDataset(Dataset):
    """Return synchronized windows of all three EEG modalities."""

    def __init__(
        self,
        data_dir="data/raw/preprocessedData",
        subject_ids=None,
        window_seconds=4.0,
        stride_seconds=4.0,
    ):
        self.data_dir = Path(data_dir)
        self.samples = []

        files = sorted(
            self.data_dir.glob("dataSubject*.mat"),
            key=lambda p: int(
                re.search(r"dataSubject(\d+)\.mat$", p.name).group(1)
            ),
        )

        if subject_ids is not None:
            selected = set(subject_ids)
            files = [
                p for p in files
                if int(re.search(
                    r"dataSubject(\d+)\.mat$", p.name
                ).group(1)) in selected
            ]

        if not files:
            raise FileNotFoundError(
                f"No subject files found in {self.data_dir}"
            )

        self.fs = None

        for path in files:
            subject_id = int(
                re.search(
                    r"dataSubject(\d+)\.mat$", path.name
                ).group(1)
            )

            trials, _, fs = _load_subject(str(path))

            if self.fs is None:
                self.fs = fs
            elif not np.isclose(self.fs, fs):
                raise ValueError("Inconsistent sampling rates")

            window_size = round(window_seconds * fs)
            stride_size = round(stride_seconds * fs)

            if window_size <= 0 or stride_size <= 0:
                raise ValueError("Window and stride must be positive")

            for trial_idx, trial in enumerate(trials):
                for start in range(
                    0,
                    trial.shape[0] - window_size + 1,
                    stride_size,
                ):
                    self.samples.append(
                        (str(path), subject_id, trial_idx, start,
                         window_size)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, subject_id, trial_idx, start, size = self.samples[idx]

        trials, indices, _ = _load_subject(path)
        window = trials[trial_idx][start:start + size, :]

        result = {}

        for modality in ("in_ear", "around_ear", "scalp"):
            # MATLAB: (time, channels)
            # PyTorch: (channels, time)
            signal = window[:, indices[modality]].T.copy()
            valid = np.isfinite(signal)

            # Replace missing observations only after masking.
            clean = np.nan_to_num(
                signal,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            result[modality] = torch.from_numpy(
                clean.astype(np.float32)
            )

            mask_name = (
                "target_mask"
                if modality == "scalp"
                else f"{modality}_mask"
            )
            result[mask_name] = torch.from_numpy(valid)

        result["subject_id"] = subject_id
        result["trial_id"] = trial_idx + 1
        result["start_sample"] = start

        return result
