
import torch

from eeg_reconstruction.data.normalization import (
    fit_normalization,
    inverse_normalize,
    normalize_batch,
)


def make_batch():
    x = torch.tensor([
        [[1., 2., 3.], [10., 20., 30.]],
        [[4., 5., 6.], [40., 50., 60.]],
    ])

    mask = torch.ones_like(x, dtype=torch.bool)

    return {
        "in_ear": x,
        "scalp": x.clone(),
        "in_ear_observed_mask": mask,
        "target_mask": mask.clone(),
    }


def test_channel_normalization():
    batch = make_batch()
    stats = fit_normalization([batch], mode="channel")

    result = normalize_batch(batch, stats)

    means = result["in_ear"].mean(dim=(0, 2))
    stds = result["in_ear"].std(
        dim=(0, 2), unbiased=False
    )

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-6)


def test_global_normalization():
    batch = make_batch()
    stats = fit_normalization([batch], mode="global")
    result = normalize_batch(batch, stats)

    assert abs(result["scalp"].mean().item()) < 1e-6
    assert abs(
        result["scalp"].std(unbiased=False).item() - 1
    ) < 1e-6


def test_inverse_transform():
    batch = make_batch()
    stats = fit_normalization([batch], mode="channel")

    normalized = normalize_batch(batch, stats)
    restored = inverse_normalize(
        normalized["scalp"], stats
    )

    assert torch.allclose(restored, batch["scalp"], atol=1e-5)


def test_missing_values():
    batch = make_batch()
    batch["in_ear"][0, 0, 1] = float("nan")
    batch["in_ear_observed_mask"][0, 0, 1] = False

    stats = fit_normalization([batch], mode="channel")
    result = normalize_batch(batch, stats)

    assert torch.isfinite(result["in_ear"]).all()
    assert result["in_ear"][0, 0, 1].item() == 0.0


def test_no_data_leakage():
    batch = make_batch()
    stats = fit_normalization([batch], mode="channel")

    val_batch = make_batch()
    val_batch["in_ear"] += 100.0

    normalized = normalize_batch(val_batch, stats)

    # Validation is transformed using training statistics.
    assert normalized["in_ear"].mean().item() > 1.0


def test_none_mode():
    batch = make_batch()
    stats = fit_normalization([batch], mode="none")

    result = normalize_batch(batch, stats)

    assert torch.allclose(result["in_ear"], batch["in_ear"])
