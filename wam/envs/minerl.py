"""MineStudio's MineRL simulator behind the :class:`MinecraftEnv` protocol.

The browser rig in :mod:`wam.envs.eaglercraft` can see the game but cannot read
it: no inventory, no block breaks, so ``event_ids`` is always empty and
``L_align`` has nothing to align against. MineRL exposes exactly those symbols,
which is the reason this adapter exists.

Runs on Linux only (JDK 8 + MCP-Reborn), so ``minestudio`` is imported lazily --
a Windows checkout must still be able to import :mod:`wam.envs`.

Three things here were measured against a live simulator rather than assumed,
because each fails silently if guessed wrong. They are marked MEASURED below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import WAMConfig
from ..model.action import ActionChunk, CameraBinner
from .base import Observation, status_vector

# MEASURED: these arrive as running totals for the episode, not per-step deltas.
# Swinging at a tree, ``mine_block`` sat at ``{'spruce_leaves': 1}`` for twelve
# consecutive steps, stepped to 2, then held at 2 for the remaining sixty. Used
# raw, every event would re-fire on every step for the rest of the episode.
EVENT_KINDS = (
    "mine_block",
    "pickup",
    "craft_item",
    "break_item",
    "kill_entity",
    "damage_dealt",
    "use_item",
)

# Phrases, not identifiers: these strings are embedded by the LLM to seed the
# concept vectors, so "mined spruce log" is worth more than "mine_block".
_EVENT_VERB = {
    "mine_block": "mined",
    "pickup": "picked up",
    "craft_item": "crafted",
    "break_item": "broke",
    "kill_entity": "killed",
    "damage_dealt": "damaged",
    "use_item": "used",
}


class ConceptVocab:
    """Append-only ``phrase -> id`` map over the concept vocabulary.

    ``EventEmbedding.init_from_text`` seeds row *i* from the LLM's embedding of
    phrase *i*, so an id that moves between runs silently repoints a concept at
    someone else's meaning. Hence: assigned once, never reordered, persisted.

    Id 0 is ``EventEmbedding.PAD`` and is never handed out.
    """

    PAD = 0

    def __init__(self, size: int, path: str | Path | None = None) -> None:
        self.size = size
        self.path = Path(path) if path else None
        self.phrases: dict[str, int] = {}
        if self.path is not None and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.phrases = {k: int(v) for k, v in raw.items()}
        # Concepts seen after the vocabulary filled up, kept for reporting.
        self.overflow: set[str] = set()

    def get(self, phrase: str) -> int:
        if phrase in self.phrases:
            return self.phrases[phrase]
        next_id = len(self.phrases) + 1  # 0 stays PAD
        if next_id >= self.size:
            # Dropping a rare concept costs one training signal; aliasing it onto
            # an occupied id would teach the grounding loss a wrong pairing.
            self.overflow.add(phrase)
            return self.PAD
        self.phrases[phrase] = next_id
        return next_id

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.phrases, indent=2, sort_keys=True), encoding="utf-8")

    def id_to_phrase(self) -> dict[int, str]:
        """The table ``EventEmbedding.init_from_text`` is initialised from."""
        return {i: p for p, i in self.phrases.items()}


def _unbatch(action: ActionChunk) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(1, chunk, ...)`` as produced by ``ActionHead.sample`` -> ``(chunk, ...)``."""
    buttons, camera, hotbar = action.buttons, action.camera, action.hotbar
    while buttons.dim() > 2:
        if buttons.shape[0] != 1:
            raise ValueError(f"a real environment steps one agent, got batch {buttons.shape[0]}")
        buttons, camera, hotbar = buttons[0], camera[0], hotbar[0]
    # .float()/.long() before .numpy(): the model runs in bfloat16, which numpy
    # has no dtype for, so the conversion raises. Hand-built chunks are float32
    # and hide this -- which is why it survived a dry run and failed the first
    # time a real policy drove the env.
    return (
        buttons.detach().float().cpu().numpy(),
        camera.detach().long().cpu().numpy(),
        hotbar.detach().long().cpu().numpy(),
    )


class MineRLEnv:
    """``MinecraftSim`` adapted to the chunked, symbol-carrying protocol.

    One ``step`` executes a whole ``ActionChunk`` tick by tick and returns the
    frame after the last one, matching the readme's "one model call is about half
    a second of control".
    """

    def __init__(
        self,
        cfg: WAMConfig,
        seed: int = 0,
        sim=None,
        vocab_path: str | Path | None = None,
        **sim_kwargs,
    ) -> None:
        self.cfg = cfg
        if sim is None:
            from minestudio.simulator import MinecraftSim  # JVM + 445MB of assets

            size = cfg.vision.image_size
            # MineRL renders 640x360 and squashes to the observation size, so the
            # world's aspect ratio is not preserved. That is the convention VPT
            # and the contractor data were recorded under, so keep it.
            sim = MinecraftSim(action_type="env", obs_size=(size, size), seed=seed, **sim_kwargs)
        self.sim = sim
        self.binner = CameraBinner(
            cfg.action.camera_bins, cfg.action.camera_max_delta, cfg.action.camera_mu
        )
        self.vocab = ConceptVocab(cfg.event.vocab_size, vocab_path or cfg.event.concept_vocab_path)
        self._totals: dict[str, dict[str, int]] = {}
        self._prev_health = 20.0
        self._prev_biome: int | None = None
        self._prev_inventory: tuple = ()

    # -- protocol ---------------------------------------------------------

    def reset(self) -> Observation:
        obs, info = self.sim.reset()
        # Baselines, so the first step's diff is against spawn rather than zero.
        self._totals = self._snapshot(info)
        self._prev_health = float(info.get("health", 20.0))
        self._prev_biome = self._biome(info)
        self._prev_inventory = self._inventory(info)
        return self._observe(obs, info, phrases=[], done=False)

    def step(self, action: ActionChunk) -> Observation:
        obs = info = None
        done = False
        for tick in self._to_ticks(action):
            obs, _reward, terminated, truncated, info = self.sim.step(tick)
            done = bool(terminated or truncated)
            if done:
                break
        if info is None:  # chunk_size == 0; nothing was executed
            raise ValueError("action chunk was empty")
        phrases = self._drain_events(info)
        return self._observe(obs, info, phrases=phrases, done=done)

    def close(self) -> None:
        self.vocab.save()  # ids must survive the process that minted them
        self.sim.close()

    # -- action translation ----------------------------------------------

    def _to_ticks(self, action: ActionChunk) -> list[dict]:
        buttons, camera, hotbar = _unbatch(action)
        import torch

        degrees = self.binner.to_degrees(torch.from_numpy(camera)).numpy()

        ticks = []
        for k in range(buttons.shape[0]):
            tick = self.sim.env.action_space.no_op()
            for i, name in enumerate(self.cfg.action.buttons):
                tick[name] = int(buttons[k, i] > 0.5)
            # MEASURED: MineRL's camera vector is [pitch, yaw]; ActionChunk's bins
            # are (yaw, pitch). camera=[10,0] moved pitch by 30 degrees over three
            # ticks and left yaw at 0; camera=[0,10] did the reverse. Without this
            # swap every learned turn becomes a look, and nothing in the loss
            # curves would ever show it.
            tick["camera"] = [float(degrees[k, 1]), float(degrees[k, 0])]
            slot = int(hotbar[k])
            if slot:  # 0 means "keep the current slot"
                tick[f"hotbar.{slot}"] = 1
            ticks.append(tick)
        return ticks

    # -- symbols ----------------------------------------------------------

    @staticmethod
    def _snapshot(info: dict) -> dict[str, dict[str, int]]:
        return {
            kind: {name: int(n) for name, n in (info.get(kind) or {}).items()}
            for kind in EVENT_KINDS
        }

    def _drain_events(self, info: dict) -> list[str]:
        """Totals -> the phrases for what happened during this chunk."""
        current = self._snapshot(info)
        phrases: list[str] = []
        for kind in EVENT_KINDS:
            before = self._totals.get(kind, {})
            for name, count in current.get(kind, {}).items():
                if count > before.get(name, 0):
                    phrases.append(f"{_EVENT_VERB[kind]} {name.replace('_', ' ')}")
        self._totals = current
        return phrases

    @staticmethod
    def _inventory(info: dict) -> tuple:
        slots = info.get("inventory") or {}
        return tuple((str(s.get("type")), int(s.get("quantity", 0))) for s in slots.values())

    @staticmethod
    def _biome(info: dict):
        stats = info.get("location_stats") or {}
        biome = stats.get("biome_id")
        return None if biome is None else int(biome)

    def _observe(self, obs: dict | None, info: dict, phrases: list[str], done: bool) -> Observation:
        n = self.cfg.event.n_event_tokens
        event_ids = np.zeros(n, dtype=np.int64)
        for i, phrase in enumerate(phrases[:n]):
            event_ids[i] = self.vocab.get(phrase)

        health = float(info.get("health", self._prev_health))
        biome = self._biome(info)
        inventory = self._inventory(info)
        gui_open = bool(info.get("is_gui_open", info.get("isGuiOpen", False)))

        stats = info.get("location_stats") or {}
        status = status_vector(
            self.cfg.status,
            {
                "health": health,
                "food": info.get("food_level", 20),
                "is_gui_open": gui_open,
                "light_level": stats.get("light_level", 15),
                "can_see_sky": stats.get("can_see_sky", True),
                "is_raining": stats.get("is_raining", False),
            },
        )

        # The flags WorldHead predicts, derived here where the symbols live.
        flags = {
            "block_removed": any(p.startswith("mined") for p in phrases),
            "inventory_changed": inventory != self._prev_inventory,
            "contact": health < self._prev_health,
            "new_area": biome is not None and biome != self._prev_biome,
            "done": done,
        }
        self._prev_health, self._prev_biome, self._prev_inventory = health, biome, inventory

        image = (
            obs["image"]
            if obs is not None
            else np.zeros(
                (self.cfg.vision.image_size, self.cfg.vision.image_size, 3), dtype=np.uint8
            )
        )
        return Observation(
            rgb=np.ascontiguousarray(image, dtype=np.uint8),
            event_ids=event_ids,
            event_text=", ".join(phrases),
            health=health,
            done=done,
            status=status,
            info={
                "flags": flags,
                # While the inventory GUI is up, the camera drives a cursor
                # instead of the head, and `use` clicks a slot instead of the
                # world. Same action numbers, different meaning -- this is the
                # only thing that tells the two apart.
                "is_gui_open": gui_open,
                "player_pos": info.get("player_pos", {}),
                "food_level": info.get("food_level"),
                "mainhand": (info.get("equipped_items", {}).get("mainhand", {})).get("type"),
                "n_events": len(phrases),
                "events_dropped": max(0, len(phrases) - n),
            },
        )
