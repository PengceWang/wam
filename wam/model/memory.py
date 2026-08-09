"""Persistent memory tokens: M_{t+1} = f_memory(M_t, H_t).

A KV cache cannot grow forever if the agent is supposed to play Minecraft
indefinitely, so the readme keeps a fixed-size memory state that the model
rewrites every step:

    16 scene | 16 long-term world knowledge | 8 skill | 8 drive

The drive slots are held in the same tensor but are sliced out and placed in
their own block of the input sequence, matching ``x_t`` in the readme.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import MemoryConfig


class PersistentMemory(nn.Module):
    def __init__(self, cfg: MemoryConfig, d_model: int, n_heads: int = 8) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_slots = cfg.n_memory + cfg.n_drive
        self.d_model = d_model

        # Learned initial state, plus a per-slot identity embedding so the model
        # can tell "scene slot 3" from "skill slot 3".
        self.init_state = nn.Parameter(torch.randn(self.n_slots, d_model) * 0.02)
        self.slot_type = nn.Parameter(torch.randn(self.n_slots, d_model) * 0.02)

        heads = n_heads if d_model % n_heads == 0 else 1
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.read = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.cand_norm = nn.LayerNorm(d_model)
        self.candidate = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, cfg.update_gate_bias)

    def initial(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        state = self.init_state.unsqueeze(0).expand(batch_size, -1, -1)
        return state.to(device=device, dtype=dtype).contiguous()

    def as_input(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the memory state into the (memory, drive) blocks of ``x_t``."""
        tagged = state + self.slot_type.to(state.dtype)
        return tagged[:, : self.cfg.n_memory], tagged[:, self.cfg.n_memory :]

    def forward(self, state: torch.Tensor, step_hidden: torch.Tensor) -> torch.Tensor:
        """(B, n_slots, d) x (B, tokens_per_step, d) -> (B, n_slots, d).

        Gated so that at init the memory mostly carries over unchanged; the model
        has to learn to overwrite a slot.
        """
        delta, _ = self.read(
            self.q_norm(state),
            self.kv_norm(step_hidden),
            self.kv_norm(step_hidden),
            need_weights=False,
        )
        merged = state + delta
        cand = self.candidate(self.cand_norm(merged))
        gate = torch.sigmoid(self.gate(merged))
        return (1.0 - gate) * state + gate * cand
