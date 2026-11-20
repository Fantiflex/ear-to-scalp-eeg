
"""Spatial cross-attention Conformer for ear-to-scalp EEG."""

import torch
from torch import nn

from eeg_reconstruction.models.temporal_conformer import (
    ConformerBlock,
    ConformerConfig,
)


class SpatialConformer(nn.Module):
    def __init__(self, cfg: ConformerConfig):
        super().__init__()

        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        d = cfg.d_model

        # Original multichannel temporal backbone.
        self.stem = nn.Conv1d(
            2 * cfg.in_channels, d, kernel_size=1
        )
        self.temporal_blocks = nn.Sequential(
            *[ConformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # Shared temporal encoder for individual ear electrodes.
        self.electrode_encoder = nn.Sequential(
            nn.Conv1d(2, d, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=5, padding=2),
        )

        # Learnable identities, not anatomical coordinates.
        self.ear_embeddings = nn.Embedding(cfg.in_channels, d)
        self.scalp_embeddings = nn.Embedding(cfg.out_channels, d)

        # Each scalp electrode attends to all ear electrodes.
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d)
        self.output_head = nn.Linear(d, 1)

    def forward(self, x, observed_mask):
        if x.ndim != 3 or x.shape != observed_mask.shape:
            raise ValueError("Expected matching (B, C, T) tensors")
        if x.shape[1] != self.in_channels:
            raise ValueError("Incorrect number of input channels")

        batch_size, channels, length = x.shape
        mask = observed_mask.bool()

        x = torch.where(mask, x, torch.zeros_like(x))
        masked_input = torch.cat([x, mask.to(x.dtype)], dim=1)

        # Global temporal representation: (B, T, D).
        global_features = self.stem(masked_input)
        global_features = self.temporal_blocks(
            global_features.transpose(1, 2)
        )

        # Preserve individual electrode representations.
        electrode_input = torch.stack(
            [x, mask.to(x.dtype)], dim=2
        )  # (B, C, 2, T)

        electrode_input = electrode_input.reshape(
            batch_size * channels, 2, length
        )

        ear_features = self.electrode_encoder(
            electrode_input
        )

        ear_features = ear_features.reshape(
            batch_size, channels, -1, length
        ).permute(0, 3, 1, 2)  # (B, T, C, D)

        ear_ids = torch.arange(channels, device=x.device)
        ear_features = (
            ear_features
            + self.ear_embeddings(ear_ids)[None, None, :, :]
        )

        # Each output electrode has its own learned query.
        scalp_ids = torch.arange(
            self.out_channels, device=x.device
        )
        scalp_queries = self.scalp_embeddings(scalp_ids)
        scalp_queries = scalp_queries[None, None, :, :]
        scalp_queries = scalp_queries.expand(
            batch_size, length, -1, -1
        )

        scalp_queries = (
            scalp_queries + global_features[:, :, None, :]
        )

        # Attention across ELECTRODES at each time point.
        queries = scalp_queries.reshape(
            batch_size * length, self.out_channels, -1
        )
        keys_values = ear_features.reshape(
            batch_size * length, channels, -1
        )

        spatial_features, _ = self.spatial_attention(
            queries,
            keys_values,
            keys_values,
            need_weights=False,
        )

        spatial_features = self.norm(
            spatial_features + queries
        )

        output = self.output_head(spatial_features)

        return output.reshape(
            batch_size, length, self.out_channels
        ).transpose(1, 2)
