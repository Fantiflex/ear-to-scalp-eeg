
"""Tests for mask-aware EEG re-referencing."""

import pytest
import torch

from eeg_reconstruction.data.rereferencing import (
    rereference_batch,
    rereference_in_ear,
)


@pytest.fixture
def example():
    # Left ear: 1, 2, 3, 4, 5, 6
    # Right ear: 11, 12, 13, 14, 15, 16
    left = torch.arange(1, 7, dtype=torch.float32)
    right = torch.arange(11, 17, dtype=torch.float32)

    x = torch.cat([left, right]).reshape(1, 12, 1)
    mask = torch.ones_like(x, dtype=torch.bool)
    return x, mask


def test_none_preserves_values(example):
    x, mask = example
    output, valid = rereference_in_ear(x, mask, "none")

    torch.testing.assert_close(output, x)
    assert torch.equal(valid, mask)


def test_global_car_has_zero_channel_mean(example):
    x, mask = example
    output, valid = rereference_in_ear(x, mask, "car")

    torch.testing.assert_close(
        output.mean(dim=1),
        torch.zeros(1, 1),
        atol=1e-6,
        rtol=0,
    )
    assert valid.all()


def test_same_ear_car_has_zero_mean_per_ear(example):
    x, mask = example
    output, _ = rereference_in_ear(x, mask, "same_ear")

    assert torch.allclose(
        output[:, :6, :].mean(dim=1),
        torch.zeros(1, 1),
        atol=1e-6,
    )
    assert torch.allclose(
        output[:, 6:, :].mean(dim=1),
        torch.zeros(1, 1),
        atol=1e-6,
    )


def test_other_ear_reference(example):
    x, mask = example
    output, _ = rereference_in_ear(x, mask, "other_ear")

    # Right ear mean = 13.5; left ear mean = 3.5
    assert output[0, 0, 0].item() == pytest.approx(-12.5)
    assert output[0, 6, 0].item() == pytest.approx(7.5)


def test_missing_channels_not_included_in_reference(example):
    x, mask = example
    mask[0, 0, 0] = False

    output, valid = rereference_in_ear(x, mask, "same_ear")

    # Mean of remaining left channels = (2+3+4+5+6)/5 = 4
    assert output[0, 1, 0].item() == pytest.approx(-2.0)
    assert output[0, 0, 0].item() == 0.0
    assert not valid[0, 0, 0]


def test_missing_opposite_ear_invalidates_output(example):
    x, mask = example
    mask[:, 6:, :] = False

    output, valid = rereference_in_ear(x, mask, "other_ear")

    assert not valid.any()
    assert torch.count_nonzero(output) == 0


def test_batch_preserves_scalp_targets(example):
    x, mask = example
    scalp = torch.randn(1, 29, 1)

    batch = {
        "in_ear": x,
        "in_ear_observed_mask": mask,
        "scalp": scalp,
    }

    result = rereference_batch(batch, "car")

    torch.testing.assert_close(result["scalp"], scalp)
    assert result["in_ear"].shape == x.shape
    assert result["in_ear_mask"].shape == mask.shape


def test_invalid_mode(example):
    x, mask = example
    with pytest.raises(ValueError):
        rereference_in_ear(x, mask, "invalid")
