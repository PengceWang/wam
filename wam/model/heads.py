"""Heads hanging off the shared hidden state H_t.

    World Head    next semantic visual latent + event prediction
    Value Head    long-horizon exploration value
    Goal Head     the next self-proposed objective, as goal tokens
    Drive Head    novelty / learning-progress / danger (intrinsic reward)
    Language Head verbalise what was learned, reusing the LLM vocabulary

The action head lives in ``action.py`` next to the action encoding it mirrors.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import EventConfig, HeadsConfig


def _mlp(d_model: int, out_dim: int, hidden_mult: int) -> nn.Sequential:
    hidden = d_model * hidden_mult
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class EventEmbedding(nn.Module):
    """Concept tokens (``<object: oak log>``, ``<event: collected>``, ...).

    These are what tie the game's symbolic side to the LLM's semantic space.
    ``init_from_text`` seeds them with the LLM's own embedding of the matching
    phrase, so "oak log" starts out near the language model's notion of wood
    rather than at a random point.
    """

    PAD = 0

    def __init__(self, cfg: EventConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, d_model, padding_idx=self.PAD)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.PAD].zero_()

    @torch.no_grad()
    def init_from_text(self, concept_ids: list[int], vectors: torch.Tensor) -> None:
        idx = torch.tensor(concept_ids, device=self.embedding.weight.device)
        self.embedding.weight[idx] = vectors.to(self.embedding.weight.dtype)

    def forward(self, event_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(event_ids)


class WorldHead(nn.Module):
    """Predict the next step of the world in latent + symbolic space.

    Deliberately *not* pixel reconstruction: the target is the next semantic
    visual latent produced by the frozen vision frontend, plus the discrete
    things the readme lists (inventory change, block broken, health, contact,
    new area, episode end).
    """

    DISCRETE_FLAGS = (
        "block_removed",
        "inventory_changed",
        "contact",
        "new_area",
        "done",
    )

    def __init__(
        self, cfg: HeadsConfig, event_cfg: EventConfig, d_model: int, n_visual_tokens: int
    ) -> None:
        super().__init__()
        self.n_visual_tokens = n_visual_tokens
        self.latent = _mlp(d_model, n_visual_tokens * d_model, cfg.hidden_mult)
        self.events = _mlp(d_model, event_cfg.vocab_size, cfg.hidden_mult)
        self.flags = _mlp(d_model, len(self.DISCRETE_FLAGS), cfg.hidden_mult)
        self.health = _mlp(d_model, 1, cfg.hidden_mult)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.latent(h).unflatten(-1, (self.n_visual_tokens, -1))
        return {
            "next_latent": latent,
            # Multi-label over the concept vocabulary: several events can fire.
            "event_logits": self.events(h),
            "flag_logits": self.flags(h),
            "health_delta": self.health(h).squeeze(-1),
        }


class ValueHead(nn.Module):
    """Long-horizon exploration value. Optionally distributional (HL-Gauss)."""

    def __init__(self, cfg: HeadsConfig, d_model: int, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__()
        self.n_bins = max(1, cfg.value_bins)
        self.net = _mlp(d_model, self.n_bins, cfg.hidden_mult)
        if self.n_bins > 1:
            self.register_buffer("support", torch.linspace(v_min, v_max, self.n_bins))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns raw logits when distributional, otherwise a scalar value."""
        out = self.net(h)
        return out.squeeze(-1) if self.n_bins == 1 else out

    def expectation(self, out: torch.Tensor) -> torch.Tensor:
        if self.n_bins == 1:
            return out
        return (out.softmax(-1) * self.support.to(out.dtype)).sum(-1)


class GoalHead(nn.Module):
    """Emit the next self-proposed objective as goal tokens in LLM space.

    The tokens are fed back into the sequence, so the model states its own
    intention in the same representation it reasons in.
    """

    def __init__(self, cfg: HeadsConfig, d_model: int) -> None:
        super().__init__()
        self.n_goal_tokens = cfg.n_goal_tokens
        self.net = _mlp(d_model, cfg.n_goal_tokens * d_model, cfg.hidden_mult)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).unflatten(-1, (self.n_goal_tokens, -1))


class DriveHead(nn.Module):
    """Intrinsic reward: novelty, learning progress, danger."""

    COMPONENTS = ("novelty", "progress", "danger")

    def __init__(self, cfg: HeadsConfig, d_model: int) -> None:
        super().__init__()
        self.net = _mlp(d_model, len(self.COMPONENTS), cfg.hidden_mult)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(h)
        return {name: out[..., i] for i, name in enumerate(self.COMPONENTS)}

    def intrinsic_reward(self, drives: dict[str, torch.Tensor], w=(1.0, 1.0, -0.5)) -> torch.Tensor:
        novelty, progress, danger = w
        return (
            novelty * drives["novelty"] + progress * drives["progress"] + danger * drives["danger"]
        )


class LanguageHead(nn.Module):
    """Summarise regularities into readable text, reusing the LLM's vocabulary.

    Owns only a projection; the actual unembedding is the backbone's LM head, so
    the head cannot drift away from the pretrained output space.
    """

    def __init__(self, cfg: HeadsConfig, d_model: int) -> None:
        super().__init__()
        self.proj = _mlp(d_model, d_model, cfg.hidden_mult)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
