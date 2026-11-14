
"""Smoke tests for the shared EEG reconstruction environment."""



import mne
import numpy as np
import pytest
import scipy.signal
import torch


def test_numpy():
    """Create a synthetic 12-channel EEG recording."""
    rng = np.random.default_rng(42)
    eeg = rng.standard_normal((12, 512))

    assert eeg.shape == (12, 512)
    assert np.isfinite(eeg).all()


def test_scipy():
    """Apply a temporal filter to synthetic EEG."""
    rng = np.random.default_rng(42)
    eeg = rng.standard_normal((12, 512))

    sos = scipy.signal.butter(
        4,
        [1, 40],
        btype="bandpass",
        fs=128,
        output="sos",
    )
    filtered = scipy.signal.sosfiltfilt(sos, eeg, axis=-1)

    assert filtered.shape == eeg.shape
    assert np.isfinite(filtered).all()


def test_mne():
    """Create an MNE EEG recording."""
    info = mne.create_info(
        ch_names=[f"EEG_{i}" for i in range(12)],
        sfreq=128,
        ch_types="eeg",
    )

    data = np.zeros((12, 512))
    raw = mne.io.RawArray(data, info, verbose=False)

    assert len(raw.ch_names) == 12
    assert raw.info["sfreq"] == 128
    assert raw.n_times == 512


@pytest.mark.parametrize("input_channels", [12, 19])
def test_pytorch_reconstruction_shape(input_channels):
    """Check both modalities can reconstruct 29 scalp channels."""
    model = torch.nn.Conv1d(
        in_channels=input_channels,
        out_channels=29,
        kernel_size=3,
        padding=1,
    )

    x = torch.randn(2, input_channels, 512)
    y = model(x)

    assert y.shape == (2, 29, 512)


def test_pytorch_backward():
    """Verify that reconstruction models can be trained."""
    model = torch.nn.Conv1d(12, 29, kernel_size=3, padding=1)

    x = torch.randn(2, 12, 512)
    target = torch.randn(2, 29, 512)

    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, target)

    loss.backward()

    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()


def test_device():
    """Verify the available computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    x = torch.randn(2, 12, 512, device=device)

    assert x.device.type == device.type
