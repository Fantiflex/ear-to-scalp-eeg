
import numpy as np
import pytest

from eeg_reconstruction.data.imputation import interpolate_short_gaps


def test_internal_short_gap_and_preserves_original_mask():
    x = np.array([[1., 99., 99., 4., 5.]], dtype=np.float32)
    mask = np.array([[1, 0, 0, 1, 1]], dtype=bool)

    old_mask = mask.copy()

    filled, reconstructed = interpolate_short_gaps(
        x, mask, max_gap_samples=2
    )

    np.testing.assert_allclose(filled, [[1, 2, 3, 4, 5]])

    np.testing.assert_array_equal(
        reconstructed,
        [[False, True, True, False, False]]
    )

    np.testing.assert_array_equal(mask, old_mask)


def test_long_gap_and_boundary_gaps_zero_filled():
    x = np.array(
        [[8., 1., 5., 5., 5., 5., 6., 7.]],
        dtype=np.float32,
    )

    mask = np.array(
        [[0, 1, 0, 0, 0, 0, 1, 0]],
        dtype=bool,
    )

    filled, imputed = interpolate_short_gaps(
        x, mask, max_gap_samples=2
    )

    np.testing.assert_array_equal(
        filled,
        [[0, 1, 0, 0, 0, 0, 6, 0]]
    )

    assert not imputed.any()


def test_fully_missing_channel_and_batch():
    x = np.array(
        [[[7., 0., 9.], [5., 5., 5.]]],
        dtype=np.float32,
    )

    mask = np.array(
        [[[1, 0, 1], [0, 0, 0]]],
        dtype=bool,
    )

    filled, imputed = interpolate_short_gaps(x, mask)

    np.testing.assert_allclose(filled[0, 0], [7, 8, 9])
    np.testing.assert_array_equal(filled[0, 1], [0, 0, 0])

    assert imputed[0, 0, 1]
    assert not imputed[0, 1].any()


def test_nan_missing_values_and_zero_length_limit():
    x = np.array([[1., np.nan, 3.]])

    filled, imputed = interpolate_short_gaps(
        x,
        [[1, 0, 1]],
        max_gap_samples=0,
    )

    np.testing.assert_array_equal(filled, [[1, 0, 3]])
    assert not imputed.any()


def test_invalid_mask_or_observation_rejected():
    with pytest.raises(ValueError):
        interpolate_short_gaps([[1., np.nan]], [[1, 1]])

    with pytest.raises(ValueError):
        interpolate_short_gaps([[1., 2.]], [[1, 2]])

    with pytest.raises(ValueError):
        interpolate_short_gaps([[1., 2.]], [[1]])
