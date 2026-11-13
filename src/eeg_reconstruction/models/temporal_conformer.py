
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ConformerConfig:
    in_channels: int
    out_channels: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    kernel_size: int = 15
    dropout: float = 0.1


class ConformerBlock(nn.Module):
    """Temporal convolution + self-attention block."""

    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        d = cfg.d_model

        self.ff1 = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * d, d),
        )

        self.norm_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.norm_conv = nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(
                d,
                d,
                kernel_size=cfg.kernel_size,
                padding=cfg.kernel_size // 2,
                groups=d,
            ),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=1),
            nn.Dropout(cfg.dropout),
        )

        self.ff2 = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * d, d),
        )

        self.final_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)

        z = self.norm_attn(x)
        z, _ = self.attn(z, z, z, need_weights=False)
        x = x + z

        z = self.norm_conv(x).transpose(1, 2)
        x = x + self.conv(z).transpose(1, 2)

        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


class TemporalConformer(nn.Module):
    """Map ear-EEG signals to reconstructed scalp-EEG."""

    def __init__(self, cfg: ConformerConfig):
        super().__init__()

        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        if cfg.kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd"
            )

        self.stem = nn.Conv1d(
            cfg.in_channels,
            cfg.d_model,
            kernel_size=1,
        )

        self.blocks = nn.Sequential(
            *[
                ConformerBlock(cfg)
                for _ in range(cfg.n_layers)
            ]
        )

        self.head = nn.Conv1d(
            cfg.d_model,
            cfg.out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, ear_channels, time)

        z = self.stem(x)
        z = z.transpose(1, 2)

        z = self.blocks(z)

        z = z.transpose(1, 2)
        return self.head(z)
