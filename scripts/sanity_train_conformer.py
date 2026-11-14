
"""Check that the Temporal Conformer can learn a known mapping."""

import torch
from torch import nn

from eeg_reconstruction.models.temporal_conformer import (
    ConformerConfig,
    TemporalConformer,
)


def main():
    torch.manual_seed(42)

    # Small model to keep the test inexpensive on CPU.
    config = ConformerConfig(
        in_channels=4,
        out_channels=8,
        d_model=32,
        n_heads=4,
        n_layers=1,
        kernel_size=7,
        dropout=0.0,
    )

    model = TemporalConformer(config)
    model.train()

    # Fixed synthetic mini-batch:
    # (batch, ear channels, time samples)
    x = torch.randn(4, 4, 64)

    # Known linear mapping from 4 to 8 channels.
    mixing = torch.randn(8, 4) * 0.3
    target = torch.einsum("oc,bct->bot", mixing, x)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )
    criterion = nn.MSELoss()

    losses = []

    for step in range(201):
        optimizer.zero_grad(set_to_none=True)

        mask = torch.ones_like(x, dtype=torch.bool)
        prediction = model(x, mask)
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 25 == 0:
            print(
                f"Step {step:03d} | "
                f"MSE: {loss.item():.6f}"
            )

    improvement = losses[0] / max(losses[-1], 1e-12)
    print(f"\nLoss reduction factor: {improvement:.2f}x")

    if improvement < 5:
        raise RuntimeError(
            "Insufficient loss reduction; inspect the model."
        )


if __name__ == "__main__":
    main()
