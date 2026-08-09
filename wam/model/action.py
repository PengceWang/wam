"""Action tokenisation and the factorized action head.

The readme rejects "LLM emits JSON" in favour of native action tokens, and
rejects one-token-per-frame in favour of chunks::

    a_t = (a_move, a_camera, a_attack, a_use, a_hotbar)   emitted chunk_size steps at a time

So a single forward pass of the backbone produces ~0.5s of low-level keyboard and
mouse control, and the same encoding is fed back in as the ``prev_action`` block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import ActionConfig


@dataclass
class ActionChunk:
    """A chunk of ``chunk_size`` low-level actions, in discrete form.

    Shapes carry a leading ``*batch`` that is either ``(B,)`` or ``(B, T)``.
    """

    buttons: torch.Tensor  # (*batch, chunk, n_buttons) float 0/1
    camera: torch.Tensor  # (*batch, chunk, 2) long, bin indices for (yaw, pitch)
    hotbar: torch.Tensor  # (*batch, chunk) long, 0 = keep current slot

    def to(self, *args, **kwargs) -> ActionChunk:
        return ActionChunk(
            self.buttons.to(*args, **kwargs),
            self.camera.to(*args, **kwargs),
            self.hotbar.to(*args, **kwargs),
        )


class CameraBinner:
    """Maps camera deltas in degrees to/from discrete bins.

    Bins are spaced non-uniformly (mu-law): small adjustments get fine
    resolution, large flicks get coarse resolution, which is what mouse-look
    data actually looks like.
    """

    def __init__(self, n_bins: int, max_delta: float, mu: float = 8.0) -> None:
        if n_bins % 2 == 0:
            raise ValueError("camera_bins must be odd so that a zero-motion bin exists")
        self.n_bins = n_bins
        self.max_delta = max_delta
        self.mu = mu
        half = n_bins // 2
        # Bin centres in degrees, symmetric around 0.
        lin = torch.linspace(-1.0, 1.0, n_bins)
        centres = torch.sign(lin) * (torch.expm1(torch.abs(lin) * torch.log1p(torch.tensor(mu))))
        centres = centres / mu * max_delta
        centres[half] = 0.0
        self.centres = centres

    def to_bins(self, degrees: torch.Tensor) -> torch.Tensor:
        centres = self.centres.to(degrees.device, degrees.dtype)
        return (degrees.unsqueeze(-1) - centres).abs().argmin(dim=-1)

    def to_degrees(self, bins: torch.Tensor) -> torch.Tensor:
        return self.centres.to(bins.device)[bins]


class ActionEncoder(nn.Module):
    """Encode the previous action chunk into ``n_action_tokens`` LLM-space tokens."""

    def __init__(self, cfg: ActionConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.button_emb = nn.Linear(cfg.n_buttons, d_model)
        self.camera_emb = nn.Embedding(cfg.camera_bins, d_model)
        self.hotbar_emb = nn.Embedding(cfg.n_hotbar + 1, d_model)
        # Which slot inside the chunk a low-level action came from.
        self.step_emb = nn.Parameter(torch.randn(cfg.chunk_size, d_model) * 0.02)
        self.to_tokens = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * cfg.n_action_tokens),
        )

    def forward(self, action: ActionChunk) -> torch.Tensor:
        """ActionChunk -> (*batch, n_action_tokens, d_model)."""
        x = self.button_emb(action.buttons.to(self.button_emb.weight.dtype))
        x = x + self.camera_emb(action.camera).sum(dim=-2)
        x = x + self.hotbar_emb(action.hotbar)
        x = x + self.step_emb
        x = x.mean(dim=-2)  # pool over the chunk
        tokens = self.to_tokens(x)
        return tokens.unflatten(-1, (self.cfg.n_action_tokens, -1))


class ActionHead(nn.Module):
    """Predict the next chunk of low-level actions from a single hidden state."""

    def __init__(self, cfg: ActionConfig, d_model: int, hidden_mult: int = 2) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = d_model * hidden_mult
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
        )
        k = cfg.chunk_size
        self.buttons = nn.Linear(hidden, k * cfg.n_buttons)
        self.camera = nn.Linear(hidden, k * 2 * cfg.camera_bins)
        self.hotbar = nn.Linear(hidden, k * (cfg.n_hotbar + 1))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """(*batch, d_model) -> logits for each factor."""
        z = self.trunk(h)
        k = self.cfg.chunk_size
        return {
            "buttons": self.buttons(z).unflatten(-1, (k, self.cfg.n_buttons)),
            "camera": self.camera(z).unflatten(-1, (k, 2, self.cfg.camera_bins)),
            "hotbar": self.hotbar(z).unflatten(-1, (k, self.cfg.n_hotbar + 1)),
        }

    @torch.no_grad()
    def sample(self, h: torch.Tensor, temperature: float = 1.0) -> ActionChunk:
        logits = self(h)
        if temperature <= 0:
            buttons = (logits["buttons"] > 0).float()
            camera = logits["camera"].argmax(-1)
            hotbar = logits["hotbar"].argmax(-1)
        else:
            buttons = torch.bernoulli(torch.sigmoid(logits["buttons"] / temperature))
            camera = _sample_categorical(logits["camera"] / temperature)
            hotbar = _sample_categorical(logits["hotbar"] / temperature)
        return ActionChunk(buttons=buttons, camera=camera, hotbar=hotbar)


def _sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(-1)
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return idx.view(probs.shape[:-1])


def zero_action(cfg: ActionConfig, *batch: int, device=None) -> ActionChunk:
    """A no-op chunk, used to seed ``prev_action`` at t=0."""
    k = cfg.chunk_size
    return ActionChunk(
        buttons=torch.zeros(*batch, k, cfg.n_buttons, device=device),
        camera=torch.full((*batch, k, 2), cfg.camera_bins // 2, dtype=torch.long, device=device),
        hotbar=torch.zeros(*batch, k, dtype=torch.long, device=device),
    )
