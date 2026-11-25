
"""Inspect experimental conditions across EAR-SAAD subjects."""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path("data/raw/preprocessedData")


def main():
    counts = Counter()

    for subject in range(1, 16):
        path = ROOT / f"dataSubject{subject}.mat"

        data = loadmat(
            path,
            simplify_cells=True,
            variable_names=[
                "videoCondition",
                "attendedEar",
                "attSpeaker",
            ],
        )

        print(f"\n{'=' * 60}")
        print(f"SUBJECT {subject:02d}")
        print(f"{'=' * 60}")

        video = np.asarray(
            data["videoCondition"]
        ).ravel()

        ear = np.asarray(
            data["attendedEar"]
        ).ravel()

        speaker = np.asarray(
            data["attSpeaker"]
        ).ravel()

        print("Video dtype:", video.dtype)
        print("Video shape:", video.shape)

        if not len(video) == len(ear) == len(speaker) == 6:
            raise ValueError(
                f"Unexpected metadata shape for S{subject:02d}"
            )

        for trial in range(6):
            condition = str(video[trial]).strip()

            counts[condition] += 1

            print(
                f"T{trial + 1} | "
                f"video={video[trial]!r} | "
                f"ear={ear[trial]!r} | "
                f"speaker={speaker[trial]!r}"
            )

    print("\n=== VIDEO CONDITION DISTRIBUTION ===")

    for condition, count in sorted(counts.items()):
        print(f"{condition!r}: {count} trials")
        print(f"  Duration: {count * 10 / 60:.2f} hours")
        print(f"  Windows: {count * 150}")


if __name__ == "__main__":
    main()
