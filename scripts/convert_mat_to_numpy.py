
"""Convert EAR-SAAD MATLAB recordings into memory-mappable NumPy files."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eeg_reconstruction.data.dataset import _load_subject


def convert_subject(mat_path: Path, output_dir: Path):
    match = re.fullmatch(r"dataSubject(\d+)\.mat", mat_path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {mat_path.name}")

    subject_id = int(match.group(1))
    trials, indices, fs = _load_subject(str(mat_path))

    # Preserve trial boundaries while storing one contiguous array.
    offsets = [0]
    for trial in trials:
        offsets.append(offsets[-1] + trial.shape[0])

    signals = np.concatenate(
        [np.asarray(trial, dtype=np.float32) for trial in trials],
        axis=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f"subject_{subject_id:02d}.npy"
    json_path = output_dir / f"subject_{subject_id:02d}.json"

    np.save(npy_path, signals, allow_pickle=False)

    metadata = {
        "subject_id": subject_id,
        "fs": float(fs),
        "trial_offsets": offsets,
        "n_trials": len(trials),
        "n_samples": int(signals.shape[0]),
        "n_channels": int(signals.shape[1]),
        "indices": {
            modality: [int(i) for i in channel_indices]
            for modality, channel_indices in indices.items()
        },
        "dtype": "float32",
    }

    json_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"Subject {subject_id:02d}: "
        f"{signals.shape}, "
        f"{len(trials)} trials, "
        f"{npy_path.stat().st_size / 1024**2:.1f} MiB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/raw/preprocessedData",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/numpy",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = sorted(
        input_dir.glob("dataSubject*.mat"),
        key=lambda p: int(
            re.search(r"dataSubject(\d+)\.mat$", p.name).group(1)
        ),
    )

    if len(files) != 15:
        raise RuntimeError(
            f"Expected 15 MATLAB files, found {len(files)}"
        )

    for path in files:
        convert_subject(path, output_dir)

    print(f"\nConversion complete: {len(files)} subjects")


if __name__ == "__main__":
    main()
