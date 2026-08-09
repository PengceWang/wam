"""A fake environment so the whole stack can be exercised without Minecraft.

It is not a world model of anything -- it just produces frames that react to the
action in a visible way, so shape bugs, dtype bugs and "the loss never moves"
bugs surface before a real client is attached.
"""

from __future__ import annotations

import numpy as np

from ..config import WAMConfig
from ..model.action import ActionChunk
from .base import Observation, status_vector


class DummyMinecraftEnv:
    def __init__(self, cfg: WAMConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.size = cfg.vision.image_size
        self._yaw = 0.0
        self._t = 0

    def reset(self) -> Observation:
        self._yaw = 0.0
        self._t = 0
        return self._observe()

    def step(self, action: ActionChunk) -> Observation:
        # Camera bins shift the "view", so consecutive frames are correlated with
        # the action -- enough signal for a smoke test to be meaningful.
        centre = self.cfg.action.camera_bins // 2
        yaw_bins = action.camera[..., 0].reshape(-1).float().mean().item() - centre
        self._yaw += yaw_bins * 4.0
        self._t += 1
        return self._observe()

    def _observe(self) -> Observation:
        x = np.linspace(0, 4 * np.pi, self.size, dtype=np.float32) + np.deg2rad(self._yaw)
        band = (np.sin(x)[None, :] * 0.5 + 0.5) * 255
        frame = np.repeat(band[None, :, :], 3, axis=0).transpose(1, 2, 0)
        frame = np.repeat(frame[:1], self.size, axis=0)
        frame = frame.astype(np.uint8)

        n = self.cfg.event.n_event_tokens
        event_ids = np.zeros(n, dtype=np.int64)
        if self._t % 5 == 4:  # pretend something happened
            event_ids[0] = int(self.rng.integers(1, self.cfg.event.vocab_size))

        # Status wobbles with time so a shape bug shows up as a moving number
        # rather than a constant that any encoder can fit.
        return Observation(
            rgb=frame,
            event_ids=event_ids,
            event_text="collected oak log" if event_ids[0] else "",
            health=20.0,
            done=self._t >= 64,
            status=status_vector(
                self.cfg.status,
                {
                    "health": 20.0,
                    "food": 20 - (self._t % 5),
                    "is_gui_open": False,
                    "light_level": 15,
                    "can_see_sky": True,
                    "is_raining": self._t % 11 == 0,
                },
            ),
        )

    def close(self) -> None:  # nothing to release
        return
