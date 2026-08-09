"""The goal slot and the status block.

The goal slot is the piece that makes "follows an instruction" and "decides for
itself" the same mechanism, so the properties worth pinning are about the
*handover*: an external goal has to persist while it is set, stop persisting
when it is withdrawn, and be indistinguishable in scale from a self-proposed one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wam.config import (
    ActionConfig,
    BackboneConfig,
    StatusConfig,
    TrainConfig,
    VisionConfig,
    WAMConfig,
)
from wam.data import rollout, trajectory_to_batch
from wam.envs import DummyMinecraftEnv
from wam.envs.base import status_vector
from wam.model import WorldActionModel
from wam.training.losses import goal_loss


def tiny_config(**overrides) -> WAMConfig:
    cfg = WAMConfig(
        backbone=BackboneConfig(
            model_name="random",
            dtype="float32",
            random_hidden_size=64,
            random_n_layers=2,
            random_n_heads=4,
        ),
        vision=VisionConfig(
            encoder_name="random",
            image_size=64,
            n_visual_tokens=4,
            resampler_depth=1,
            resampler_heads=4,
            random_encoder_dim=64,
        ),
        action=ActionConfig(chunk_size=2, n_action_tokens=2),
        train=TrainConfig(seq_len=3, batch_size=2, device="cpu", max_steps=2, log_every=1),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# -- status ---------------------------------------------------------------


def test_status_vector_is_scaled_and_ordered():
    cfg = StatusConfig()
    vec = status_vector(
        cfg,
        {
            "health": 10.0,
            "food": 20,
            "is_gui_open": True,
            "light_level": 15,
            "can_see_sky": False,
            "is_raining": True,
        },
    )
    assert vec.dtype == np.float32
    assert vec.shape == (cfg.n_fields,)
    order = {name: i for i, name in enumerate(cfg.fields)}
    assert vec[order["health"]] == pytest.approx(0.5)  # 10/20
    assert vec[order["food"]] == pytest.approx(1.0)
    assert vec[order["light_level"]] == pytest.approx(1.0)  # 15/15
    assert vec[order["is_gui_open"]] == pytest.approx(1.0)
    assert vec[order["can_see_sky"]] == pytest.approx(0.0)


def test_missing_readings_become_zero_not_an_error():
    """An env that cannot measure something must not kill the rollout."""
    vec = status_vector(StatusConfig(), {"health": 20.0})
    assert vec.sum() == pytest.approx(1.0), "only health is set, and it scales to 1"


def test_status_block_is_in_the_sequence():
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    state = model.initial_state(1)
    pixels = torch.rand(1, 3, cfg.vision.image_size, cfg.vision.image_size)

    out, _ = model.step(state, pixels, status=torch.zeros(1, cfg.status.n_fields))
    assert out.hidden.shape[1] == cfg.tokens_per_step


def test_status_changes_the_readout():
    """If status were dropped on the floor the model would be blind to the GUI."""
    cfg = tiny_config()
    model = WorldActionModel(cfg).eval()
    pixels = torch.rand(1, 3, cfg.vision.image_size, cfg.vision.image_size)

    with torch.no_grad():
        a, _ = model.step(
            model.initial_state(1), pixels, status=torch.zeros(1, cfg.status.n_fields)
        )
        b, _ = model.step(model.initial_state(1), pixels, status=torch.ones(1, cfg.status.n_fields))
    assert not torch.allclose(a.readout, b.readout)


# -- goal -----------------------------------------------------------------


def test_goal_slot_starts_from_the_null_goal():
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    state = model.initial_state(2)
    assert state.goal.shape == (2, cfg.heads.n_goal_tokens, model.d_model)
    assert not state.goal_is_external


def test_self_proposed_goal_is_carried_to_the_next_step():
    cfg = tiny_config()
    model = WorldActionModel(cfg).eval()
    pixels = torch.rand(1, 3, cfg.vision.image_size, cfg.vision.image_size)

    with torch.no_grad():
        out, state = model.step(model.initial_state(1), pixels)
    assert torch.allclose(state.goal, out.goal), "the head writes the slot it reads next step"


def test_external_goal_survives_steps_and_the_head_does_not_overwrite_it():
    """A handed-in instruction has to outlive the step, or it is not an instruction."""
    cfg = tiny_config()
    model = WorldActionModel(cfg).eval()
    pixels = torch.rand(1, 3, cfg.vision.image_size, cfg.vision.image_size)

    state = model.initial_state(1)
    given = torch.randn(1, cfg.heads.n_goal_tokens, model.d_model)
    state = type(state)(
        memory=state.memory,
        prev_action=state.prev_action,
        goal=given,
        goal_is_external=True,
    )

    with torch.no_grad():
        for _ in range(3):
            _, state = model.step(state, pixels)
    assert torch.allclose(state.goal, given), "external goal must not be overwritten"

    # Withdrawing it hands the slot back to the head -- the stage-3 handover.
    state = model.set_goal(state, None)
    with torch.no_grad():
        out, state = model.step(state, pixels)
    assert torch.allclose(state.goal, out.goal)
    assert not torch.allclose(state.goal, given)


def test_goal_source_config_rejects_nonsense():
    from wam.config import GoalConfig

    with pytest.raises(ValueError, match="unknown goal source"):
        GoalConfig(source="whatever")


def test_set_goal_needs_a_tokenizer():
    """The random backbone has none; the failure has to be loud, not a zero goal."""
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    with pytest.raises(RuntimeError, match="tokenizer"):
        model.set_goal(model.initial_state(1), ["chop oak wood"])


# -- hindsight relabelling -------------------------------------------------


def test_goal_loss_falls_back_to_a_regulariser_without_targets():
    tokens = torch.randn(2, 3, 4, 8)
    assert goal_loss(tokens).item() > 0


def test_goal_loss_rewards_matching_the_achieved_embedding():
    """Cosine to what was actually achieved: aligned should beat opposed."""
    d = 8
    target = torch.randn(1, 2, d)
    aligned = target.unsqueeze(-2).expand(1, 2, 4, d)
    opposed = -aligned
    assert goal_loss(aligned, target) < goal_loss(opposed, target)


def test_goal_loss_masks_steps_where_nothing_was_achieved():
    d = 8
    target = torch.randn(1, 2, d)
    tokens = torch.randn(1, 2, 4, d)
    all_off = torch.zeros(1, 2)
    assert goal_loss(tokens, target, all_off).item() == pytest.approx(0.0)


def test_rollout_records_status_and_reaches_the_batch():
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    env = DummyMinecraftEnv(cfg)
    traj = rollout(model, env, n_steps=4)

    assert len(traj.status) == len(traj)
    batch = trajectory_to_batch(traj, cfg)
    assert batch.status is not None
    assert batch.status.shape == (1, len(traj), cfg.status.n_fields)


def test_batch_to_device_moves_fields_added_later():
    """A hand-written .to() silently drops new fields; this is the guard."""
    from dataclasses import fields as dc_fields

    from wam.training.losses import Batch

    cfg = tiny_config()
    model = WorldActionModel(cfg)
    traj = rollout(model, DummyMinecraftEnv(cfg), n_steps=3)
    batch = trajectory_to_batch(traj, cfg).to("cpu")

    for f in dc_fields(Batch):
        if getattr(batch, f.name) is None and f.name in ("status",):
            pytest.fail(f"{f.name} was dropped by Batch.to")


# -- camera quantisation ---------------------------------------------------


def test_small_camera_moves_survive_quantisation():
    """Measured on 192k contractor frames: half of all nonzero camera deltas are
    under ~1.2 degrees. A scheme that rounds those to the zero bin teaches the
    model that the human held still, so this pins the resolution near zero.

    The old default (11 bins, +/-15, mu=8) put only the zero bin inside +/-1
    degree and destroyed 36% of real turns.
    """
    from wam.model.action import CameraBinner

    cfg = WAMConfig().action
    binner = CameraBinner(cfg.camera_bins, cfg.camera_max_delta, cfg.camera_mu)
    zero_bin = cfg.camera_bins // 2

    fine = torch.tensor([0.3, 0.6, 0.75, 1.2, 2.0])
    assert (binner.to_bins(fine) != zero_bin).all(), "real human turns must not read as no motion"

    within_one = (binner.centres.abs() <= 1.0).sum().item()
    assert within_one >= 3, "need resolution where the data actually lives"


def test_camera_precision_is_spent_where_the_data_is():
    """Fine near zero, deliberately coarse in the tail -- that is the mu-law's job.

    99% of contractor camera deltas are under 13 degrees, so precision belongs
    there. A big flick reconstructed 1.6 degrees off still means "flick"; a
    0.75-degree turn reconstructed as 0 means "did not move", which is a
    different action.
    """
    from wam.model.action import CameraBinner

    cfg = WAMConfig().action
    binner = CameraBinner(cfg.camera_bins, cfg.camera_max_delta, cfg.camera_mu)

    small = torch.tensor([-3.0, -1.2, -0.5, 0.0, 0.5, 1.2, 3.0])
    err = (binner.to_degrees(binner.to_bins(small)) - small).abs()
    assert err.max() < 0.5, "the bulk of the distribution needs sub-half-degree accuracy"

    large = torch.tensor([-15.0, 15.0])
    rel = (binner.to_degrees(binner.to_bins(large)) - large).abs() / large.abs()
    assert (rel < 0.2).all(), "the tail may be coarse, but not wildly wrong"


def test_zero_stays_exactly_zero():
    """Holding still has to be representable, not approximated."""
    from wam.model.action import CameraBinner

    cfg = WAMConfig().action
    binner = CameraBinner(cfg.camera_bins, cfg.camera_max_delta, cfg.camera_mu)
    assert binner.to_degrees(binner.to_bins(torch.tensor([0.0]))).item() == 0.0
