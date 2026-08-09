"""The seam between the model and Minecraft.

Nothing above this file knows how frames are produced. When you have a real
Minecraft running, implement ``MinecraftEnv`` against it (Malmo, MineRL, a mod
that streams frames over a socket, or a screen-capture + keyboard-injection
bridge to the vanilla client) and the rest of the codebase is unchanged.

The contract is deliberately chunk-shaped: ``step`` receives ``chunk_size``
low-level actions and returns the observation after all of them have been
executed, matching the readme's "one model call ≈ half a second of control".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import StatusConfig
from ..model.action import ActionChunk

# Divisor that brings each status field into roughly [0, 1]. Anything absent is
# already a 0/1 flag. Keeping the scales here rather than in each env means two
# environments cannot disagree about what "health" means numerically.
STATUS_SCALE = {"health": 20.0, "food": 20.0, "light_level": 15.0}


def status_vector(cfg: StatusConfig, values: dict) -> np.ndarray:
    """Named readings -> the fixed-order float vector the model consumes.

    Missing keys become 0.0 rather than raising: an env that cannot measure
    something should read as "nothing there", not crash the rollout.
    """
    out = np.zeros(cfg.n_fields, dtype=np.float32)
    for i, name in enumerate(cfg.fields):
        raw = values.get(name)
        if raw is None:
            continue
        out[i] = float(raw) / STATUS_SCALE.get(name, 1.0)
    return out


@dataclass
class Observation:
    """What one environment step hands back."""

    rgb: np.ndarray  # (H, W, 3) uint8
    # Concept-vocabulary ids of what just happened, padded with 0.
    event_ids: np.ndarray  # (n_event_tokens,) int64
    # Human-readable description of the same events, used by the grounding loss.
    event_text: str = ""
    health: float = 20.0
    done: bool = False
    # Scalar HUD state, in StatusConfig.fields order. None means the env cannot
    # report it; the model then sees zeros.
    status: np.ndarray | None = None  # (n_fields,) float32
    info: dict = field(default_factory=dict)


@runtime_checkable
class MinecraftEnv(Protocol):
    """Minimal environment protocol. Chunked, not per-tick."""

    def reset(self) -> Observation: ...

    def step(self, action: ActionChunk) -> Observation:
        """Execute a whole action chunk, then return the resulting observation."""
        ...

    def close(self) -> None: ...
