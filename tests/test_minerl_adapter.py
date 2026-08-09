"""Adapter tests against a fake MinecraftSim -- no JVM, so these run anywhere.

Each test here pins down something that was measured against the live simulator
and that fails *silently* if it regresses: a swapped camera axis trains a turn as
a look, and undifferenced event totals re-fire forever. Neither shows up in a
loss curve.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wam.config import ActionConfig, EventConfig, VisionConfig, WAMConfig
from wam.envs.minerl import ConceptVocab, MineRLEnv
from wam.model.action import ActionChunk

# MineRL's SIMPLE_KEYBOARD_ACTION, copied verbatim from
# minestudio/simulator/minerl/herobraine/hero/mc.py. The adapter writes button
# names straight into a MineRL tick, so a rename on either side must fail loudly
# here rather than turn into a button that quietly never fires.
BUTTONS = (
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
    "use",
    "pickItem",
    "inventory",
)


class FakeActionSpace:
    def no_op(self) -> dict:
        act = {b: 0 for b in BUTTONS}
        act.update({f"hotbar.{i}": 0 for i in range(1, 10)})
        act["camera"] = [0.0, 0.0]
        return act


class FakeSim:
    """Records the ticks it was given and replays a scripted info stream."""

    def __init__(self, infos: list[dict]) -> None:
        self.env = type("E", (), {"action_space": FakeActionSpace()})()
        self.infos = infos
        self.ticks: list[dict] = []
        self.i = 0
        self.closed = False

    def _frame(self):
        return {"image": np.zeros((8, 8, 3), dtype=np.uint8)}

    def reset(self):
        self.i = 0
        return self._frame(), self.infos[0]

    def step(self, action):
        self.ticks.append(action)
        self.i = min(self.i + 1, len(self.infos) - 1)
        return self._frame(), 0.0, False, False, self.infos[self.i]

    def close(self):
        self.closed = True


def base_info(**overrides) -> dict:
    info = {
        "health": 20.0,
        "food_level": 20,
        "inventory": {},
        "equipped_items": {"mainhand": {"type": "air"}},
        "location_stats": {"biome_id": 1},
        "player_pos": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0},
        **{
            k: {}
            for k in (
                "mine_block",
                "pickup",
                "craft_item",
                "break_item",
                "kill_entity",
                "damage_dealt",
                "use_item",
            )
        },
    }
    info.update(overrides)
    return info


def cfg_for(chunk_size: int = 2, n_event_tokens: int = 8) -> WAMConfig:
    return WAMConfig(
        vision=VisionConfig(image_size=8),
        action=ActionConfig(chunk_size=chunk_size),
        event=EventConfig(n_event_tokens=n_event_tokens),
    )


def chunk(cfg: WAMConfig, *, buttons=None, camera=None, hotbar=None) -> ActionChunk:
    k = cfg.action.chunk_size
    centre = cfg.action.camera_bins // 2
    return ActionChunk(
        buttons=torch.zeros(1, k, cfg.action.n_buttons) if buttons is None else buttons,
        camera=torch.full((1, k, 2), centre) if camera is None else camera,
        hotbar=torch.zeros(1, k, dtype=torch.long) if hotbar is None else hotbar,
    )


def test_button_names_are_exactly_minerl_keyboard_actions():
    """The config's names are the wire format, not labels."""
    assert cfg_for().action.buttons == BUTTONS
    assert "chat" not in BUTTONS, "chat can /give anything and shorts out intrinsic reward"
    assert "drop" not in BUTTONS, "MineRL leaves drop commented out; there is no channel"
    assert "swapHands" not in BUTTONS


def test_gui_buttons_reach_the_tick():
    """pickItem and inventory are the two that were missing; crafting needs them."""
    cfg = cfg_for(chunk_size=2)
    env = MineRLEnv(cfg, sim=FakeSim([base_info(), base_info()]))
    env.reset()

    buttons = torch.zeros(1, 2, cfg.action.n_buttons)
    buttons[0, 0, cfg.action.buttons.index("inventory")] = 1.0
    buttons[0, 1, cfg.action.buttons.index("pickItem")] = 1.0
    env.step(chunk(cfg, buttons=buttons))

    assert env.sim.ticks[0]["inventory"] == 1 and env.sim.ticks[0]["pickItem"] == 0
    assert env.sim.ticks[1]["pickItem"] == 1 and env.sim.ticks[1]["inventory"] == 0


def test_camera_axes_are_swapped_into_minerl_order():
    """ActionChunk bins are (yaw, pitch); MineRL's vector is [pitch, yaw]."""
    cfg = cfg_for(chunk_size=1)
    env = MineRLEnv(cfg, sim=FakeSim([base_info(), base_info()]))
    env.reset()

    centre = cfg.action.camera_bins // 2
    # Yaw to the outermost bin, pitch held at centre.
    camera = torch.tensor([[[cfg.action.camera_bins - 1, centre]]])
    env.step(chunk(cfg, camera=camera))

    pitch, yaw = env.sim.ticks[-1]["camera"]
    assert pitch == pytest.approx(0.0), "pitch must not move when only yaw was asked for"
    assert yaw > 0.0, "yaw belongs in slot 1 of MineRL's camera vector"


def test_event_totals_are_differenced_not_replayed():
    """The live sim reports running totals; the same event must fire once."""
    cfg = cfg_for(chunk_size=1)
    sim = FakeSim(
        [
            base_info(),
            base_info(mine_block={"spruce_log": 1}),  # fires
            base_info(mine_block={"spruce_log": 1}),  # unchanged -> silent
            base_info(mine_block={"spruce_log": 2}),  # fires again
        ]
    )
    env = MineRLEnv(cfg, sim=sim)
    env.reset()

    first = env.step(chunk(cfg))
    second = env.step(chunk(cfg))
    third = env.step(chunk(cfg))

    assert first.event_text == "mined spruce log"
    assert second.event_text == "", "a flat total must not re-fire the event"
    assert third.event_text == "mined spruce log"
    assert first.event_ids[0] == third.event_ids[0], "same concept, same id"
    assert second.event_ids.sum() == 0


def test_buttons_and_hotbar_translate_per_tick():
    cfg = cfg_for(chunk_size=2)
    env = MineRLEnv(cfg, sim=FakeSim([base_info(), base_info()]))
    env.reset()

    buttons = torch.zeros(1, 2, cfg.action.n_buttons)
    buttons[0, 0, cfg.action.buttons.index("attack")] = 1.0
    buttons[0, 1, cfg.action.buttons.index("jump")] = 1.0
    hotbar = torch.tensor([[0, 3]])
    env.step(chunk(cfg, buttons=buttons, hotbar=hotbar))

    assert len(env.sim.ticks) == 2, "every action in the chunk is executed"
    assert env.sim.ticks[0]["attack"] == 1 and env.sim.ticks[0]["jump"] == 0
    assert env.sim.ticks[1]["jump"] == 1 and env.sim.ticks[1]["attack"] == 0
    # hotbar 0 means "keep the current slot", so nothing is pressed.
    assert all(env.sim.ticks[0][f"hotbar.{i}"] == 0 for i in range(1, 10))
    assert env.sim.ticks[1]["hotbar.3"] == 1


def test_flags_track_world_changes():
    cfg = cfg_for(chunk_size=1)
    sim = FakeSim(
        [
            base_info(),
            base_info(
                mine_block={"dirt": 1},
                health=18.0,
                location_stats={"biome_id": 4},
                inventory={"0": {"type": "dirt", "quantity": 1}},
            ),
        ]
    )
    env = MineRLEnv(cfg, sim=sim)
    env.reset()
    obs = env.step(chunk(cfg))

    flags = obs.info["flags"]
    assert flags["block_removed"] and flags["inventory_changed"]
    assert flags["contact"], "health dropped"
    assert flags["new_area"], "biome changed"
    assert not flags["done"]
    assert obs.health == pytest.approx(18.0)


def test_event_budget_is_reported_not_silently_truncated():
    cfg = cfg_for(chunk_size=1, n_event_tokens=2)
    sim = FakeSim([base_info(), base_info(mine_block={"a": 1, "b": 1, "c": 1, "d": 1})])
    env = MineRLEnv(cfg, sim=sim)
    env.reset()
    obs = env.step(chunk(cfg))

    assert (obs.event_ids != 0).sum() == 2
    assert obs.info["n_events"] == 4
    assert obs.info["events_dropped"] == 2


def test_vocab_ids_are_stable_across_reload(tmp_path):
    path = tmp_path / "concepts.json"
    first = ConceptVocab(64, path)
    a, b = first.get("mined spruce log"), first.get("picked up dirt")
    first.save()

    reloaded = ConceptVocab(64, path)
    assert reloaded.get("mined spruce log") == a
    assert reloaded.get("picked up dirt") == b
    assert reloaded.get("crafted stick") not in (a, b, ConceptVocab.PAD)


def test_vocab_overflow_pads_rather_than_aliasing():
    vocab = ConceptVocab(3)  # ids 1 and 2 only
    assert vocab.get("one") == 1
    assert vocab.get("two") == 2
    assert vocab.get("three") == ConceptVocab.PAD
    assert "three" in vocab.overflow


def test_batched_chunk_is_rejected():
    cfg = cfg_for(chunk_size=1)
    env = MineRLEnv(cfg, sim=FakeSim([base_info(), base_info()]))
    env.reset()
    k = cfg.action.chunk_size
    batched = ActionChunk(
        buttons=torch.zeros(4, k, cfg.action.n_buttons),
        camera=torch.zeros(4, k, 2, dtype=torch.long),
        hotbar=torch.zeros(4, k, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="one agent"):
        env.step(batched)


def test_bfloat16_chunk_from_a_real_policy_survives():
    """ActionHead.sample() returns the backbone's dtype, and numpy has no bfloat16.

    Hand-built chunks are float32, so a dry run with scripted actions passes
    while the first real policy rollout dies on `.numpy()`. That is exactly what
    happened; this pins it.
    """
    cfg = cfg_for(chunk_size=2)
    env = MineRLEnv(cfg, sim=FakeSim([base_info(), base_info()]))
    env.reset()

    k = cfg.action.chunk_size
    chunk_bf16 = ActionChunk(
        buttons=torch.zeros(1, k, cfg.action.n_buttons, dtype=torch.bfloat16),
        camera=torch.full((1, k, 2), cfg.action.camera_bins // 2),
        hotbar=torch.zeros(1, k, dtype=torch.long),
    )
    chunk_bf16.buttons[0, 0, cfg.action.buttons.index("forward")] = 1.0

    obs = env.step(chunk_bf16)  # must not raise
    assert env.sim.ticks[0]["forward"] == 1
    assert obs.rgb.dtype == np.uint8
