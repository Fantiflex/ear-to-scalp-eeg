
import numpy as np

from eeg_reconstruction.data.preprocessing import (
    interpolate_missing_channels,
)


def test_linear_interpolation():
    signal = np.array(
        [[1.0, np.nan, 3.0, 4.0]],
        dtype=float,
    )

    cleaned, valid = interpolate_missing_channels(signal)

    np.testing.assert_allclose(
        cleaned[0],
        [1.0, 2.0, 3.0, 4.0],
    )
    assert valid.tolist() == [True]


def test_exclude_channel_over_50_percent_missing():
    signal = np.array(
        [[1.0, np.nan, np.nan, np.nan]],
        dtype=float,
    )

    cleaned, valid = interpolate_missing_channels(signal)

    assert not valid[0]
    assert np.all(cleaned == 0)


def test_keep_channel_at_50_percent_valid():
    signal = np.array(
        [[1.0, np.nan, 3.0, np.nan]],
        dtype=float,
    )

    cleaned, valid = interpolate_missing_channels(signal)

    assert valid[0]
    np.testing.assert_allclose(
        cleaned[0],
        [1.0, 2.0, 3.0, 3.0],
    )


def test_output_is_finite():
    signal = np.array(
        [
            [1.0, np.nan, 3.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    cleaned, valid = interpolate_missing_channels(signal)

    assert np.isfinite(cleaned).all()
    assert valid.tolist() == [True, False]
