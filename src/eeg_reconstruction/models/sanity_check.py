
import torch

from eeg_reconstruction.models.MAE import ScalpMAE
from eeg_reconstruction.models.transfer import (
    BTEToScalp,
    load_scalp_pretrained,
)


def main():
    patch_size = 128
    T = 1024

    # Stage 1
    scalp_model = ScalpMAE(
        n_channels=29,
        patch_size=patch_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
    )

    scalp_input = torch.randn(2, 29, T)

    scalp_pred, mask = scalp_model(
        scalp_input,
        mask_ratio=0.5,
    )

    print("Stage 1 input:", scalp_input.shape)
    print("Stage 1 output:", scalp_pred.shape)
    print("Stage 1 mask:", mask.shape)

    # Stage 2
    transfer_model = BTEToScalp(
        n_around_channels=19,
        n_scalp_channels=29,
        patch_size=patch_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
    )

    transfer_model = load_scalp_pretrained(
        transfer_model,
        scalp_model,
    )

    around_input = torch.randn(2, 19, T)

    scalp_reconstruction = transfer_model(around_input)

    print("Stage 2 input:", around_input.shape)
    print("Stage 2 output:", scalp_reconstruction.shape)

    # Backprop test
    loss = scalp_reconstruction.square().mean()
    loss.backward()

    print("Backward pass successful!")


if __name__ == "__main__":
    main()
