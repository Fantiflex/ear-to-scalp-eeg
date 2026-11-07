
"""Generate presentation-ready Ear-SAAD dataset visualizations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
from scipy.signal import welch

from eeg_reconstruction.data.channels import CHANNEL_GROUPS

COLORS = {
    "scalp": "#263B80",
    "in_ear": "#DC7456",
    "around_ear": "#299B92",
}

LABELS = {
    "scalp": "Scalp EEG",
    "in_ear": "In-ear EEG",
    "around_ear": "Around-ear EEG",
}

MODALITIES = ("scalp", "in_ear", "around_ear")

OUT = Path("reports/figures")


def save_figure(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "svg"):
        fig.savefig(
            OUT / f"{name}.{extension}",
            dpi=250,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)
    print(f"Saved: {name}.png and {name}.svg")


def load_subject(subject_id):
    path = Path(
        f"data/raw/preprocessedData/dataSubject{subject_id}.mat"
    )

    mat = loadmat(path, simplify_cells=True)

    channels = [
        str(name)
        for name in np.asarray(mat["channels"]).reshape(-1)
    ]

    raw_trials = mat["eegTrials"]

    if isinstance(raw_trials, np.ndarray) and raw_trials.dtype == object:
        trials = [np.asarray(item) for item in raw_trials.flat]
    elif isinstance(raw_trials, (list, tuple)):
        trials = [np.asarray(item) for item in raw_trials]
    else:
        trials = [np.asarray(raw_trials)]

    fs = float(np.asarray(mat["fs"]).squeeze())

    indices = {
        modality: [
            channels.index(name)
            for name in CHANNEL_GROUPS[modality]
        ]
        for modality in MODALITIES
    }

    return trials, channels, indices, fs


def zscore_valid(x):
    """Normalize each channel for visualization only."""
    y = np.full(x.shape, np.nan, dtype=float)

    for i, channel in enumerate(x):
        valid = np.isfinite(channel)
        if valid.sum() < 2:
            continue

        mean = channel[valid].mean()
        std = channel[valid].std()

        if std > 0:
            y[i, valid] = (channel[valid] - mean) / std

    return y


def longest_valid_segment(x):
    """Return the longest contiguous finite interval."""
    valid = np.isfinite(x)
    boundaries = np.diff(
        np.concatenate(([False], valid, [False])).astype(int)
    )

    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)

    if len(starts) == 0:
        return np.array([])

    idx = np.argmax(ends - starts)
    return x[starts[idx]:ends[idx]]


def plot_waveforms(trial, indices, fs, subject_id, trial_id):
    """Three synchronized multichannel EEG traces."""
    duration = 12
    n = min(int(duration * fs), trial.shape[0])
    time = np.arange(n) / fs

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True,
        constrained_layout=True,
    )

    for ax, modality in zip(axes, MODALITIES):
        channel_names = CHANNEL_GROUPS[modality]

        # Show six representative channels per modality.
        select = np.linspace(
            0, len(channel_names) - 1, 6, dtype=int
        )

        data = trial[:n, indices[modality]].T
        data = zscore_valid(data)

        for row, channel_idx in enumerate(select):
            ax.plot(
                time,
                data[channel_idx] + row * 5,
                color=COLORS[modality],
                linewidth=0.8,
            )

        ax.set_yticks(np.arange(6) * 5)
        ax.set_yticklabels(
            [channel_names[i] for i in select]
        )
        ax.invert_yaxis()
        ax.set_title(
            f"{LABELS[modality]} "
            f"({len(channel_names)} electrodes)",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        f"Synchronized EEG recordings — "
        f"Subject {subject_id}, Trial {trial_id}",
        fontsize=16,
        fontweight="bold",
    )

    save_figure(fig, "01_eeg_waveforms")


def plot_spectra(trial, indices, fs):
    """Compare normalized spectral shapes, not absolute power."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for modality in MODALITIES:
        data = trial[:, indices[modality]].T
        psds = []

        for channel in data:
            segment = longest_valid_segment(channel)

            if len(segment) < fs * 10:
                continue

            frequencies, psd = welch(
                segment,
                fs=fs,
                nperseg=min(int(fs * 10), len(segment)),
            )

            # Normalize spectral area to compare shape.
            area = np.trapezoid(psd, frequencies)

            if area > 0:
                psds.append(psd / area)

        if not psds:
            print(f"No usable PSD channels for {modality}")
            continue

        median_psd = np.median(psds, axis=0)

        ax.plot(
            frequencies,
            10 * np.log10(np.maximum(median_psd, 1e-12)),
            linewidth=2.3,
            label=f"{LABELS[modality]} (n={len(psds)})",
            color=COLORS[modality],
        )

    ax.axvspan(1, 4, color="#A0ADCF", alpha=0.12)
    ax.axvspan(4, 8, color="#E5BD80", alpha=0.12)

    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD (dB/Hz)")
    ax.set_title(
        "Spectral characteristics of the three EEG modalities",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save_figure(fig, "02_frequency_spectra")


def plot_missingness(trial, channels, fs):
    """Visualize missing observations across all 61 channels."""
    missing = ~np.isfinite(trial.T)

    # One pixel represents one second with any missing sample.
    samples_per_bin = int(fs)
    n_bins = missing.shape[1] // samples_per_bin

    missing_binned = missing[:, :n_bins * samples_per_bin]
    missing_binned = missing_binned.reshape(
        len(channels), n_bins, samples_per_bin
    ).any(axis=2)

    fig, ax = plt.subplots(figsize=(14, 11))

    cmap = ListedColormap(["#E7EFF3", "#C64F5A"])

    ax.imshow(
        missing_binned,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[0, n_bins, len(channels) - 0.5, -0.5],
    )

    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels, fontsize=7)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Electrodes")
    ax.set_title(
        "Missing EEG observations across electrodes",
        fontweight="bold",
    )

    for boundary in (29.5, 35.5, 44.5, 50.5):
        ax.axhline(boundary, color="#263238", linewidth=1)

    fig.tight_layout()
    save_figure(fig, "03_missingness_heatmap")


def plot_scalp_layout():
    """Visualize approximate standard montage electrode positions."""
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]

    fig, ax = plt.subplots(figsize=(8, 8))

    found = []
    missing = []

    for channel in CHANNEL_GROUPS["scalp"]:
        if channel not in positions:
            missing.append(channel)
            continue

        xyz = positions[channel]
        found.append((channel, xyz))

    if not found:
        raise ValueError("No scalp channel coordinates found.")

    # Top-down 2D projection of standard 3D coordinates.
    for name, xyz in found:
        ax.scatter(
            xyz[0], xyz[1],
            s=75,
            color=COLORS["scalp"],
            zorder=3,
        )
        ax.annotate(
            name,
            (xyz[0], xyz[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )

    from matplotlib.patches import Circle

    ax.add_patch(
        Circle(
            (0, 0),
            0.105,
            fill=False,
            linewidth=1.5,
            color="#555555",
        )
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Scalp EEG electrode positions\n"
        "(standard 10-20 montage approximation)",
        fontweight="bold",
    )

    if missing:
        print(f"Channels without standard positions: {missing}")

    fig.tight_layout()
    save_figure(fig, "04_scalp_electrode_layout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    trials, channels, indices, fs = load_subject(args.subject)

    if not 1 <= args.trial <= len(trials):
        raise ValueError("Invalid trial number.")

    trial = trials[args.trial - 1]

    print(f"Subject: {args.subject}")
    print(f"Trial: {args.trial}")
    print(f"Shape: {trial.shape}")
    print(f"Sampling frequency: {fs} Hz")

    plot_waveforms(trial, indices, fs, args.subject, args.trial)
    plot_spectra(trial, indices, fs)
    plot_missingness(trial, channels, fs)
    plot_scalp_layout()

    print("\nAll visualizations generated successfully.")


if __name__ == "__main__":
    main()
