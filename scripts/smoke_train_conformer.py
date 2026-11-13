
"""One forward/backward pass on real Ear-SAAD signals."""

import torch
from torch.utils.data import DataLoader

from eeg_reconstruction.data.dataset import EarSAADDataset
from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
    TemporalConformer,
)
from eeg_reconstruction.training.losses import masked_mse


def main():
    torch.manual_seed(42)

    dataset = EarSAADDataset(
        data_dir="data/raw/preprocessedData",
        subject_ids=[1],
        window_seconds=4.0,
        stride_seconds=4.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    x = batch["in_ear"]
    y = batch["scalp"]
    input_mask = batch["in_ear_mask"]
    target_mask = batch["target_mask"]

    print(f"Sampling rate: {dataset.fs} Hz")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Target shape: {tuple(y.shape)}")
    print(
        "Valid input fraction:",
        f"{input_mask.float().mean().item():.2%}",
    )
    print(
        "Valid target fraction:",
        f"{target_mask.float().mean().item():.2%}",
    )

    # This is a diagnostic, not a preprocessing policy.
    if not input_mask.all().item():
        print(
            "WARNING: Missing ear-EEG observations "
            "are currently represented by zeros."
        )

    model = TemporalConformer(
        ConformerConfig(
            in_channels=x.shape[1],
            out_channels=y.shape[1],
            d_model=32,
            n_heads=4,
            n_layers=1,
            kernel_size=7,
            dropout=0.0,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = model(x)
    loss = masked_mse(prediction, y, target_mask)

    print(f"Initial masked MSE: {loss.item():.6f}")

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Invalid gradient: {name}")

    optimizer.step()

    print("Real EEG forward/backward pass successful.")


if __name__ == "__main__":
    main()
