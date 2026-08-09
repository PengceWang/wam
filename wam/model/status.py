"""Scalar HUD state -> tokens in LLM space.

A handful of floats, so this is a projection rather than anything clever. It
exists as its own block instead of being concatenated onto the visual tokens so
that the backbone can attend to "am I in a menu" without going through the
frame, and so a missing reading is a zero in a known slot rather than noise
smeared across the visual latents.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import StatusConfig


class StatusEncoder(nn.Module):
    def __init__(self, cfg: StatusConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(cfg.n_fields, d_model),
            nn.GELU(),
            nn.Linear(d_model, cfg.n_tokens * d_model),
        )

    def forward(self, status: torch.Tensor) -> torch.Tensor:
        """(*batch, n_fields) -> (*batch, n_tokens, d_model)."""
        x = self.proj(status.to(self.proj[0].weight.dtype))
        return x.unflatten(-1, (self.cfg.n_tokens, -1))
