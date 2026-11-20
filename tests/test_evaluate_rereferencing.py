
import torch
from torch import nn

from eeg_reconstruction.data.rereferencing import rereference_in_ear
from scripts.evaluate_conformer import predict


class IdentityModel(nn.Module):
    def forward(self, x, observed_mask):
        return x


def make_batch():
    x = torch.arange(12, dtype=torch.float32)
    x = x.reshape(1, 12, 1).expand(1, 12, 80).clone()

    return {
        "in_ear": x,
        "in_ear_observed_mask": torch.ones_like(
            x, dtype=torch.bool
        ),
    }


def test_predict_applies_same_ear_reference():
    batch = make_batch()
    model = IdentityModel()

    prediction = predict(
        model,
        batch,
        modality="in_ear",
        device=torch.device("cpu"),
        rereference_mode="same_ear",
    )

    expected, _ = rereference_in_ear(
        batch["in_ear"],
        batch["in_ear_observed_mask"],
        mode="same_ear",
    )

    torch.testing.assert_close(prediction, expected)


def test_predict_preserves_legacy_behavior():
    batch = make_batch()
    model = IdentityModel()

    prediction = predict(
        model,
        batch,
        modality="in_ear",
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        prediction,
        batch["in_ear"],
    )
