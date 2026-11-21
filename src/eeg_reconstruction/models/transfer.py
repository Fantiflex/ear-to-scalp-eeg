
"""Stage 2: transfer scalp MAE to around-ear EEG."""

import torch
import torch.nn as nn

from eeg_reconstruction.models.MAE import (
    TemporalPatchEmbedding,
    EEGTransformerEncoder,
)


class BTEToScalp(nn.Module):
    """Around-ear EEG to scalp EEG using transferred encoder."""

    def __init__(
        self,
        n_around_channels=19,
        n_scalp_channels=29,
        patch_size=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.n_around_channels = n_around_channels
        self.n_scalp_channels = n_scalp_channels
        self.patch_size = patch_size
        self.d_model = d_model

        # Shared temporal patch embedding
        self.patch_embed = TemporalPatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
        )

        # New around-ear channel embeddings
        self.channel_embed = nn.Parameter(
            torch.randn(
                1, n_around_channels, 1, d_model
            ) * 0.02
        )

        # Temporal position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1, 512, d_model) * 0.02
        )

        # Transferred Transformer encoder
        self.encoder = EEGTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Map around-ear channels to scalp channels
        self.channel_mapping = nn.Linear(
            n_around_channels,
            n_scalp_channels,
        )

        # Reconstruct scalp waveform patches
        self.reconstruction_head = nn.Linear(
            d_model,
            patch_size,
        )

    def forward(self, x):
        """
        x: (B, 19, T)

        Returns:
            reconstruction: (B, 29, T)
        """
        B, C, T = x.shape

        if C != self.n_around_channels:
            raise ValueError(
                f"Expected {self.n_around_channels} channels, "
                f"got {C}"
            )

        # (B, 19, P, D)
        tokens = self.patch_embed(x)
        P = tokens.shape[2]

        if P > self.pos_embed.shape[2]:
            raise ValueError("Too many temporal patches.")

        tokens = (
            tokens
            + self.channel_embed
            + self.pos_embed[:, :, :P, :]
        )

        # (B, 19*P, D)
        tokens = tokens.reshape(
            B, C * P, self.d_model
        )

        # Transformer encoder
        encoded = self.encoder(tokens)

        # (B, 19, P, D)
        encoded = encoded.reshape(
            B, C, P, self.d_model
        )

        # Map channel axis: 19 -> 29
        # (B, P, D, 19)
        encoded = encoded.permute(
            0, 2, 3, 1
        )

        # (B, P, D, 29)
        encoded = self.channel_mapping(encoded)

        # (B, 29, P, D)
        encoded = encoded.permute(
            0, 3, 1, 2
        )

        # (B, 29, P, patch_size)
        pred_patches = self.reconstruction_head(encoded)

        # (B, 29, T)
        reconstruction = pred_patches.reshape(
            B,
            self.n_scalp_channels,
            P * self.patch_size,
        )

        return reconstruction


def load_scalp_pretrained(
    transfer_model,
    scalp_model,
):
    """Transfer shared pretrained weights from Stage 1."""

    transfer_model.patch_embed.load_state_dict(
        scalp_model.patch_embed.state_dict()
    )

    transfer_model.encoder.load_state_dict(
        scalp_model.encoder.state_dict()
    )

    # Transfer temporal positional embeddings.
    with torch.no_grad():
        transfer_model.pos_embed.copy_(
            scalp_model.pos_embed
        )

    print("Transferred patch embedding, encoder, and temporal positions.")

    return transfer_model
