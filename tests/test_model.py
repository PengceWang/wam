"""Shape and plumbing tests. All run on the tiny random backbone, no downloads."""

from __future__ import annotations

import pytest
import torch

from wam.config import ActionConfig, BackboneConfig, TrainConfig, VisionConfig, WAMConfig
from wam.data import random_batches, rollout, trajectory_to_batch
from wam.envs import DummyMinecraftEnv
from wam.model import CameraBinner, WorldActionModel
from wam.training import Trainer


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


def test_tokens_per_step_matches_sequence():
    cfg = tiny_config()
    expected = (
        cfg.memory.n_memory
        + cfg.memory.n_drive
        + cfg.heads.n_goal_tokens
        + cfg.vision.n_visual_tokens
        + cfg.action.n_action_tokens
        + cfg.event.n_event_tokens
        + cfg.status.n_tokens
        + 1
    )
    assert cfg.tokens_per_step == expected

    model = WorldActionModel(cfg)
    state = model.initial_state(1)
    pixels = torch.rand(1, 3, cfg.vision.image_size, cfg.vision.image_size)
    out, _ = model.step(state, pixels)
    assert out.hidden.shape == (1, cfg.tokens_per_step, model.d_model)


def test_forward_shapes():
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    batch = next(random_batches(cfg, n=1))
    out = model(batch.pixels, batch.actions, batch.event_ids)

    b, t = cfg.train.batch_size, cfg.train.seq_len
    d = model.d_model
    assert out.readout.shape == (b, t, d)
    assert out.visual_latent.shape == (b, t, cfg.vision.n_visual_tokens, d)
    assert out.action["buttons"].shape == (b, t, cfg.action.chunk_size, cfg.action.n_buttons)
    assert out.action["camera"].shape == (b, t, cfg.action.chunk_size, 2, cfg.action.camera_bins)
    assert out.world["next_latent"].shape == (b, t, cfg.vision.n_visual_tokens, d)
    assert out.value.shape == (b, t)
    assert out.goal.shape == (b, t, cfg.heads.n_goal_tokens, d)
    assert out.state.memory.shape == (b, cfg.memory.n_memory + cfg.memory.n_drive, d)


def test_memory_is_carried_and_updated():
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    batch = next(random_batches(cfg, n=1))
    initial = model.initial_state(cfg.train.batch_size).memory.clone()
    out = model(batch.pixels, batch.actions, batch.event_ids)
    assert not torch.allclose(initial, out.state.memory), "memory never changed"


def test_backbone_frozen_in_stage_one():
    cfg = tiny_config()
    cfg.train.stage = 1
    trainer = Trainer(cfg)
    assert not any(p.requires_grad for p in trainer.model.backbone.parameters())
    assert any(p.requires_grad for p in trainer.model.vision.resampler.parameters())


def test_stage_two_unfreezes_top_layers():
    cfg = tiny_config()
    cfg.train.stage = 2
    cfg.backbone.n_trainable_top_layers = 1
    trainer = Trainer(cfg)
    layers = trainer.model.backbone.layers
    assert not any(p.requires_grad for p in layers[0].parameters())
    assert all(p.requires_grad for p in layers[-1].parameters())


def test_train_step_moves_trainable_params():
    cfg = tiny_config()
    trainer = Trainer(cfg)
    before = trainer.model.vision.resampler.queries.detach().clone()
    logs = trainer.train_step(next(random_batches(cfg, n=1)))
    after = trainer.model.vision.resampler.queries.detach()
    assert torch.isfinite(torch.tensor(logs["total"]))
    assert not torch.allclose(before, after), "no parameter update happened"


def test_backbone_stays_frozen_after_a_step():
    cfg = tiny_config()
    trainer = Trainer(cfg)
    layer = trainer.model.backbone.layers[0]
    before = next(layer.parameters()).detach().clone()
    trainer.train_step(next(random_batches(cfg, n=1)))
    assert torch.allclose(before, next(layer.parameters()).detach())


def test_rollout_and_imagination():
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    env = DummyMinecraftEnv(cfg)
    traj = rollout(model, env, n_steps=4)
    assert len(traj) == 4

    batch = trajectory_to_batch(traj, cfg, model)
    assert batch.pixels.shape[1] == 4
    assert batch.actions.buttons.shape[1] == 4

    with torch.no_grad():
        state, latent = model.prime(batch.pixels[:, :2])
        imagined = model.imagine(state, latent, horizon=3)
    assert len(imagined) == 3


def test_kv_cache_grows_across_timesteps_while_training():
    """Cross-timestep context lives in the KV cache, so it must actually grow.

    Regression guard: transformers drops the cache when gradient checkpointing is
    on and the module is training, which would leave each step blind to history.
    """
    cfg = tiny_config()
    model = WorldActionModel(cfg)
    model.train()
    state = model.initial_state(1)
    pixels = torch.rand(1, 3, cfg.vision.image_size, cfg.vision.image_size)

    lengths = []
    for _ in range(3):
        _, state = model.step(state, pixels)
        assert state.cache is not None, "KV cache was dropped in training mode"
        lengths.append(state.cache.get_seq_length())

    assert lengths == [cfg.tokens_per_step * i for i in (1, 2, 3)]


def test_gradient_checkpointing_is_rejected():
    cfg = tiny_config()
    cfg.backbone.gradient_checkpointing = True
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        WorldActionModel(cfg)


def test_camera_binner_roundtrip():
    binner = CameraBinner(n_bins=11, max_delta=15.0)
    degrees = torch.tensor([0.0, 15.0, -15.0])
    bins = binner.to_bins(degrees)
    assert bins[0].item() == 5  # centre bin is exactly zero motion
    recovered = binner.to_degrees(bins)
    assert torch.allclose(recovered, degrees, atol=1e-4)


def test_config_roundtrip(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    assert WAMConfig.from_yaml(path).to_dict() == cfg.to_dict()
