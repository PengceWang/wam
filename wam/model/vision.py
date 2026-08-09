"""RGB frame -> a handful of semantic tokens living in the LLM's embedding space.

Two pieces, matching the readme:

    Minecraft RGB -> Pretrained Vision Encoder -> Visual Resampler (16-32 tokens)

The resampler is a Perceiver-style cross-attention stack: a fixed set of learned
queries attends over the (many) patch features and returns ``n_visual_tokens``
vectors. Those are then projected into ``d_model`` so they can be concatenated
with the LLM's own token embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import VisionConfig


class _RandomCNNEncoder(nn.Module):
    """Tiny conv trunk used when ``encoder_name == "random"``.

    Exists so the whole model can be built and smoke-tested offline, with no
    HuggingFace download. Swap in a real tower (SigLIP / DINOv2) for training.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        c = dim // 4
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, stride=4),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c * 2, 4, stride=4),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, dim, 2, stride=2),
        )
        self.out_dim = dim

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        feat = self.net(pixels)  # (B, dim, h, w)
        return feat.flatten(2).transpose(1, 2)  # (B, h*w, dim)


class _HFEncoder(nn.Module):
    """Wraps a HuggingFace vision tower and returns its patch features."""

    def __init__(self, name: str) -> None:
        super().__init__()
        from transformers import AutoModel

        model = AutoModel.from_pretrained(name)
        # CLIP/SigLIP checkpoints carry a text tower we do not need.
        self.model = getattr(model, "vision_model", model)
        self.out_dim = self.model.config.hidden_size

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixels)
        return out.last_hidden_state  # (B, n_patches, out_dim)


class PerceiverResampler(nn.Module):
    """Compress a variable number of patch features into ``n_queries`` tokens."""

    def __init__(self, in_dim: int, dim: int, n_queries: int, depth: int, heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        nn.LayerNorm(dim),
                        nn.MultiheadAttention(dim, heads, batch_first=True),
                        nn.LayerNorm(dim),
                        nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                    ]
                )
            )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        b = feats.shape[0]
        kv = self.in_proj(feats)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        for q_norm, kv_norm, attn, ff_norm, ff in self.layers:
            # Attend over patches *and* the queries themselves, as in Flamingo.
            ctx = kv_norm(torch.cat([kv, q], dim=1))
            delta, _ = attn(q_norm(q), ctx, ctx, need_weights=False)
            q = q + delta
            q = q + ff(ff_norm(q))
        return self.out_norm(q)


class VisionFrontend(nn.Module):
    """Encoder + resampler + projection into the LLM embedding space."""

    def __init__(self, cfg: VisionConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.encoder_name == "random":
            self.encoder: nn.Module = _RandomCNNEncoder(cfg.random_encoder_dim)
        else:
            self.encoder = _HFEncoder(cfg.encoder_name)

        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.resampler = PerceiverResampler(
            in_dim=self.encoder.out_dim,
            dim=d_model,
            n_queries=cfg.n_visual_tokens,
            depth=cfg.resampler_depth,
            heads=cfg.resampler_heads,
        )
        # Separate projection used only by the grounding loss, so that the
        # alignment target does not have to distort the tokens fed to the LLM.
        self.align_proj = nn.Linear(d_model, d_model)

    def train(self, mode: bool = True) -> VisionFrontend:  # keep a frozen tower in eval
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) or (B, 3, H, W) -> (B, T, n_visual_tokens, d_model)."""
        squeeze_time = pixels.dim() == 4
        if squeeze_time:
            pixels = pixels.unsqueeze(1)
        b, t = pixels.shape[:2]
        flat = pixels.flatten(0, 1)
        if flat.shape[-1] != self.cfg.image_size or flat.shape[-2] != self.cfg.image_size:
            flat = F.interpolate(
                flat,
                size=(self.cfg.image_size, self.cfg.image_size),
                mode="bilinear",
                align_corners=False,
            )
        ctx = torch.no_grad() if self.cfg.freeze_encoder else torch.enable_grad()
        with ctx:
            feats = self.encoder(flat)
        feats = feats.to(self.resampler.queries.dtype)
        tokens = self.resampler(feats)
        tokens = tokens.view(b, t, -1, tokens.shape[-1])
        return tokens.squeeze(1) if squeeze_time else tokens
