"""End-to-end check that the whole stack is wired up.

    python scripts/smoke_test.py                        # tiny random backbone, CPU
    python scripts/smoke_test.py --config configs/qwen3-0.6b.yaml

Exercises, in order: model construction, a teacher-forced training sequence,
a few optimisation steps, an environment rollout, and imagination.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.config import WAMConfig
from wam.data import random_batches, rollout, trajectory_to_batch
from wam.envs import DummyMinecraftEnv
from wam.model import WorldActionModel
from wam.training import Trainer


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = WAMConfig.from_yaml(root / args.config)
    print(f"config: {args.config}")
    print(f"tokens per timestep: {cfg.tokens_per_step}")
    print(f"context for seq_len={cfg.train.seq_len}: {cfg.tokens_per_step * cfg.train.seq_len}")

    print("\n[1/5] building model ...")
    model = WorldActionModel(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = sum(p.numel() for p in model.backbone.parameters())
    vision = sum(p.numel() for p in model.vision.parameters())
    print(f"      d_model={model.d_model}")
    print(f"      total={human(total)}  backbone={human(backbone)}  vision={human(vision)}")
    print(f"      trainable={human(trainable)} ({100 * trainable / total:.1f}%)")

    trainer = Trainer(cfg, model)
    device = trainer.device
    print(f"      device={device}")

    print("\n[2/5] forward pass ...")
    batch = next(random_batches(cfg, n=1)).to(device)
    with torch.no_grad():
        out = model(batch.pixels, batch.actions, batch.event_ids)
    print(f"      readout        {tuple(out.readout.shape)}")
    print(f"      visual_latent  {tuple(out.visual_latent.shape)}")
    print(f"      action.buttons {tuple(out.action['buttons'].shape)}")
    print(f"      action.camera  {tuple(out.action['camera'].shape)}")
    print(f"      world.next_lat {tuple(out.world['next_latent'].shape)}")
    print(f"      value          {tuple(out.value.shape)}")
    print(f"      goal           {tuple(out.goal.shape)}")
    print(f"      memory         {tuple(out.state.memory.shape)}")

    print(f"\n[3/5] {args.steps} optimisation steps ...")
    for logs in map(trainer.train_step, random_batches(cfg, n=args.steps)):
        print("      " + " ".join(f"{k}={v:.4f}" for k, v in sorted(logs.items())))

    print("\n[4/5] env rollout ...")
    env = DummyMinecraftEnv(cfg)
    traj = rollout(model, env, n_steps=6, device=device)
    rollout_batch = trajectory_to_batch(traj, cfg, model)
    print(f"      collected {len(traj)} steps")
    print(f"      batch pixels {tuple(rollout_batch.pixels.shape)}")
    print(f"      mean intrinsic reward {sum(traj.rewards) / max(len(traj), 1):.4f}")

    print("\n[5/5] imagination ...")
    with torch.no_grad():
        state, latent = model.prime(batch.pixels[:, :2])
        imagined = model.imagine(state, latent, horizon=4)
    print(f"      imagined {len(imagined)} steps with no env")
    print(f"      last predicted latent {tuple(imagined[-1].world['next_latent'].shape)}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
