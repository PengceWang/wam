"""External goals, encoded into the same slot the model writes its own into.

The backbone is a pretrained LLM, so "chop oak wood" needs no new modality and no
alignment loss -- it is already a sequence of vectors the model can read. That is
the whole reason an external goal is cheap here.

The two sources have to be *interchangeable*, not merely both present. If
supplied goals and self-proposed goals differ in norm or distribution, the model
learns to tell them apart, and withdrawing the scaffolding in stage 3 becomes a
distribution shift instead of a handover. Both paths therefore pass through the
same normalisation before they reach the sequence -- see ``goal_norm`` in
``WorldActionModel``.
"""

from __future__ import annotations

import torch
from torch import nn


class GoalEncoder(nn.Module):
    """Text -> ``n_goal_tokens`` vectors, plus the "no goal given" default."""

    def __init__(self, n_goal_tokens: int, d_model: int) -> None:
        super().__init__()
        self.n_goal_tokens = n_goal_tokens
        self.d_model = d_model
        # What fills the slot when nobody has set a goal yet: a learned token
        # rather than zeros, so "no goal" is a state the model can represent
        # instead of an absence it has to infer from a dead input.
        self.null_goal = nn.Parameter(torch.randn(n_goal_tokens, d_model) * 0.02)

    def null(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        out = self.null_goal.expand(batch_size, -1, -1)
        return out.to(device=device or out.device, dtype=dtype or out.dtype)

    def from_text(self, backbone, texts: list[str]) -> torch.Tensor:
        """(B strings) -> (B, n_goal_tokens, d).

        Uses the backbone's own input embeddings, keeping per-token structure
        rather than mean-pooling: "chop oak wood" is already about as many tokens
        as the slot holds, and the ordering is information.
        """
        if backbone.tokenizer is None:
            raise RuntimeError("external goals need a tokenizer; backbone.model_name is 'random'")
        emb = backbone.get_input_embeddings()
        rows = []
        for text in texts:
            if not text:
                rows.append(self.null_goal.to(emb.weight.device))
                continue
            ids = backbone.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            vectors = emb(ids["input_ids"].to(emb.weight.device))[0]  # (L, d)
            rows.append(self._fit(vectors))
        return torch.stack(rows)

    def _fit(self, vectors: torch.Tensor) -> torch.Tensor:
        """(L, d) -> (n_goal_tokens, d) by truncating or padding with the mean."""
        n, length = self.n_goal_tokens, vectors.shape[0]
        if length >= n:
            return vectors[:n]
        pad = vectors.mean(dim=0, keepdim=True).expand(n - length, -1)
        return torch.cat([vectors, pad], dim=0)
