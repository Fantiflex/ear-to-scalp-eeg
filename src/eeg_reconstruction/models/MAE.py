
"""Masked Autoencoder for scalp EEG."""

import torch
import torch.nn as nn


class TemporalPatchEmbedding(nn.Module):
    """Convert EEG temporal patches into embedding vectors."""

    def __init__(self, patch_size, d_model):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x):
        """
        x: (B, C, T)

        Returns:
            (B, C, P, D)
        """
        B, C, T = x.shape

        if T % self.patch_size != 0:
            raise ValueError(
                f"T={T} must be divisible by patch_size="
                f"{self.patch_size}"
            )

        # (B, C, P, patch_size)
        patches = x.unfold(
            dimension=-1,
            size=self.patch_size,
            step=self.patch_size,
        )

        return self.proj(patches)


class EEGTransformerEncoder(nn.Module):
    """Shared transformer encoder for EEG channel-patch tokens."""

    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        x = self.transformer(x)
        return self.norm(x)


class ScalpMAE(nn.Module):
    """Stage 1: masked reconstruction of 29-channel scalp EEG."""

    def __init__(
        self,
        n_channels=29,
        patch_size=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embed = TemporalPatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
        )

        # Channel identity embeddings
        self.channel_embed = nn.Parameter(
            torch.randn(1, n_channels, 1, d_model) * 0.02
        )

        # Temporal patch position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1, 512, d_model) * 0.02
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, d_model)
        )

        self.encoder = EEGTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Reconstruct waveform samples within each patch
        self.reconstruction_head = nn.Linear(
            d_model,
            patch_size,
        )

    def forward(self, x, mask_ratio=0.5):
        """
        x: (B, 29, T)

        Returns:
            reconstruction: (B, 29, T)
            mask:           (B, 29, P)
        """
        B, C, T = x.shape

        if C != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {C}"
            )

        # (B, C, P, D)
        tokens = self.patch_embed(x)
        P = tokens.shape[2]

        if P > self.pos_embed.shape[2]:
            raise ValueError("Too many temporal patches.")

        tokens = (
            tokens
            + self.channel_embed
            + self.pos_embed[:, :, :P, :]
        )

        # Random patch masking
        mask = torch.rand(
            B, C, P,
            device=x.device,
        ) < mask_ratio

        # Replace masked tokens
        tokens = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.expand(B, C, P, self.d_model),
            tokens,
        )

        # Flatten channel-patch tokens
        tokens = tokens.reshape(
            B, C * P, self.d_model
        )

        encoded = self.encoder(tokens)

        # Restore dimensions
        encoded = encoded.reshape(
            B, C, P, self.d_model
        )

        # (B, C, P, patch_size)
        pred_patches = self.reconstruction_head(encoded)

        # (B, C, T)
        reconstruction = pred_patches.reshape(
            B, C, P * self.patch_size
        )

        return reconstruction, mask
