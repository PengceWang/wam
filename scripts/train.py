"""Training entrypoint.

    python scripts/train.py --config configs/qwen3-0.6b.yaml
    python scripts/train.py --config configs/qwen3-0.6b.yaml --source env --stage 2

``--source random`` trains on synthetic batches (plumbing check only).
``--source env`` collects rollouts from the environment registered in
``build_env`` -- swap ``DummyMinecraftEnv`` for the real bridge once Minecraft is
running, and nothing else here changes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.config import WAMConfig
from wam.data import random_batches, rollout, trajectory_to_batch
from wam.envs import DummyMinecraftEnv
from wam.training import Batch, Trainer


def build_env(cfg: WAMConfig):
    """Replace this with the real Minecraft bridge."""
    return DummyMinecraftEnv(cfg)


def env_batches(trainer: Trainer, cfg: WAMConfig) -> Iterator[Batch]:
    """Collect a fresh rollout, train on it, repeat. The online loop."""
    env = build_env(cfg)
    while True:
        traj = rollout(trainer.model, env, n_steps=cfg.train.seq_len, device=trainer.device)
        if len(traj) < 2:
            continue
        yield trajectory_to_batch(traj, cfg, trainer.model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", choices=("random", "env"), default="random")
    ap.add_argument("--stage", type=int, default=None, help="override train.stage")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/wam.pt")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = WAMConfig.from_yaml(root / args.config)
    if args.stage is not None:
        cfg.train.stage = args.stage
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps

    trainer = Trainer(cfg)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(
        f"stage {cfg.train.stage} | device {trainer.device} | "
        f"{trainable / 1e6:.1f}M trainable | source={args.source}"
    )

    batches = random_batches(cfg) if args.source == "random" else env_batches(trainer, cfg)
    try:
        trainer.fit(batches)
    except KeyboardInterrupt:
        print("\ninterrupted, saving ...")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out))
    print(f"saved {out} at step {trainer.step_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
