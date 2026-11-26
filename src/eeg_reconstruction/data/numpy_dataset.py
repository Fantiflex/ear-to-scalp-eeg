
"""Memory-mapped EAR-SAAD dataset using converted NumPy recordings."""

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg_reconstruction.data.quality import is_valid_eeg_window


@lru_cache(maxsize=15)
def _load_numpy(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class EarSAADNumpyDataset(Dataset):
    def __init__(
        self,
        data_dir="data/processed/numpy",
        subject_ids=None,
        window_seconds=4.0,
        stride_seconds=4.0,
        manifest_path=None,
    ):
        self.data_dir = Path(data_dir)
        self.samples = []
        self.metadata = {}
        self.fs = None
        self.manifest = None

        if manifest_path is not None:
            if window_seconds != 4.0 or stride_seconds != 4.0:
                raise ValueError(
                    "D3 manifest requires 4-second non-overlapping windows"
                )

            self.manifest = {}

            with Path(manifest_path).open(newline="") as f:
                for row in csv.DictReader(f):
                    key = (
                        int(row["subject_id"]),
                        int(row["trial_id"]),
                        int(row["start_sample"]),
                    )

                    if key in self.manifest:
                        raise ValueError(f"Duplicate manifest entry: {key}")

                    self.manifest[key] = {
                        "end_sample": int(row["end_sample"]),
                        "in_ear": json.loads(row["bad_ear_channels"]),
                        "scalp": json.loads(row["bad_scalp_channels"]),
                    }

        if subject_ids is None:
            files = sorted(self.data_dir.glob("subject_*.json"))
        else:
            ids = sorted({int(s) for s in subject_ids})
            files = [
                self.data_dir / f"subject_{s:02d}.json"
                for s in ids
            ]

        if not files:
            raise FileNotFoundError("No NumPy subjects found")

        for metadata_path in files:
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)

            meta = json.loads(metadata_path.read_text())
            subject_id = int(meta["subject_id"])
            fs = float(meta["fs"])

            if self.fs is None:
                self.fs = fs
            elif not np.isclose(self.fs, fs):
                raise ValueError("Inconsistent sampling rates")

            signal_path = self.data_dir / (
                f"subject_{subject_id:02d}.npy"
            )
            if not signal_path.is_file():
                raise FileNotFoundError(signal_path)

            self.metadata[subject_id] = meta

            window_size = round(window_seconds * fs)
            stride_size = round(stride_seconds * fs)

            if window_size <= 0 or stride_size <= 0:
                raise ValueError("Invalid window or stride")

            offsets = meta["trial_offsets"]

            for trial_idx in range(len(offsets) - 1):
                trial_start = offsets[trial_idx]
                trial_end = offsets[trial_idx + 1]

                for window_idx, start in enumerate(
                    range(
                        trial_start,
                        trial_end - window_size + 1,
                        stride_size,
                    )
                ):
                    if not is_valid_eeg_window(
                        subject_id=subject_id,
                        trial_id=trial_idx + 1,
                        window_idx=window_idx,
                    ):
                        continue
                    if self.manifest is not None:
                        key = (
                            subject_id,
                            trial_idx + 1,
                            start - trial_start,
                        )

                        if key not in self.manifest:
                            continue

                        if (
                            self.manifest[key]["end_sample"]
                            != start - trial_start + window_size
                        ):
                            raise ValueError(f"Manifest window size mismatch: {key}")
                    self.samples.append(
                        (
                            str(signal_path),
                            subject_id,
                            trial_idx,
                            start,
                            window_size,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, idx):
        path, subject_id, trial_idx, start, size = self.samples[idx]

        signals = _load_numpy(path)
        window = signals[start : start + size, :]

        meta = self.metadata[subject_id]
        result = {}

        trial_start = meta["trial_offsets"][trial_idx]

        # Retrieve quality-control masks when a manifest is provided.
        qc = {}

        if self.manifest is not None:
            key = (
                subject_id,
                trial_idx + 1,
                start - trial_start,
            )

            qc = self.manifest[key]

        for modality in ("in_ear", "around_ear", "scalp"):
            channel_indices = meta["indices"][modality]

            signal = window[:, channel_indices].T.copy()

            # Mask channels rejected by quality control.
            bad_channels = qc.get(modality, [])

            if bad_channels:
                signal[bad_channels, :] = np.nan

            observed = np.isfinite(signal)

            clean = np.where(
                observed, signal, 0.0
            ).astype(np.float32)

            result[modality] = torch.from_numpy(clean)

            mask = torch.from_numpy(observed.copy())
            result[f"{modality}_observed_mask"] = mask

            legacy_key = (
                "target_mask"
                if modality == "scalp"
                else f"{modality}_mask"
            )

            result[legacy_key] = mask.clone()

        result["subject_id"] = subject_id
        result["trial_id"] = trial_idx + 1
        result["start_sample"] = start - trial_start

        return result
