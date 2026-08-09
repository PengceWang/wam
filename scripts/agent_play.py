"""Hand the controls to the model and let it play.

    python scripts/agent_play.py --steps 20
    python scripts/agent_play.py --config configs/qwen3-0.6b.yaml --steps 50
    python scripts/agent_play.py --steps 200 --train

This is the real loop from ``readme.md``, closed for the first time: frames off
the browser -> vision encoder -> LLM backbone -> action head -> keyboard and
mouse -> new frames. With ``--train`` each rollout is also fed back through the
stage-1 objective.

An untrained model plays like static, which is the point -- it proves the wiring
before any of it means anything. Ctrl-C is safe; input is always released.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.config import WAMConfig
from wam.data import frames_to_tensor, trajectory_to_batch
from wam.data.collect import Trajectory
from wam.envs.eaglercraft import EaglercraftEnv
from wam.model import WorldActionModel
from wam.training import Trainer


def describe(action, cfg) -> str:
    """One-line summary of what the model just decided to do."""
    names = cfg.action.buttons
    pressed = {names[i] for i in range(len(names)) if action.buttons[0, :, i].float().mean() > 0.5}
    centre = cfg.action.camera_bins // 2
    yaw = action.camera[0, :, 0].float().mean().item() - centre
    pitch = action.camera[0, :, 1].float().mean().item() - centre
    slot = int(action.hotbar[0].mode().values)
    return f"{','.join(sorted(pressed)) or '-':<28} yaw{yaw:+5.1f} pitch{pitch:+5.1f} slot{slot}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--window", default="Eaglercraft")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--train", action="store_true", help="also run a training step per rollout")
    ap.add_argument("--tick", type=float, default=0.05, help="seconds per low-level action")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = WAMConfig.from_yaml(root / args.config)

    trainer = Trainer(cfg) if args.train else None
    model = trainer.model if trainer else WorldActionModel(cfg)
    device = trainer.device if trainer else next(model.parameters()).device
    model = model.to(device)
    model.eval()

    env = EaglercraftEnv(cfg, window=args.window, tick_seconds=args.tick)
    traj = Trajectory()

    try:
        print(f"resuming game: {env.ensure_playing()}")
        obs = env.reset()
        state = model.initial_state(1, device=device)
        print(f"{'step':>4}  {'buttons':<28} camera            fps")

        t0 = time.time()
        for i in range(args.steps):
            pixels = frames_to_tensor([obs.rgb]).to(device)[:, 0]
            events = torch.from_numpy(obs.event_ids).long().unsqueeze(0).to(device)

            with torch.no_grad():
                out, state = model.step(state, pixels, events)
                action = model.action_head.sample(out.readout, args.temperature)
            state = replace(state, prev_action=action)

            traj.frames.append(obs.rgb)
            traj.event_ids.append(obs.event_ids)
            traj.event_texts.append(obs.event_text)
            traj.buttons.append(action.buttons[0].cpu())
            traj.camera.append(action.camera[0].cpu())
            traj.hotbar.append(action.hotbar[0].cpu())
            traj.rewards.append(float(model.drive_head.intrinsic_reward(out.drive)[0]))

            obs = env.step(action)
            rate = (i + 1) / (time.time() - t0)
            print(f"{i:>4}  {describe(action, cfg)}  {rate:4.1f}/s")

            if obs.done:
                break

        if trainer is not None and len(traj) > 2:
            print("\ntraining on the rollout ...")
            batch = trajectory_to_batch(traj, cfg, model)
            logs = trainer.train_step(batch)
            print("  " + " ".join(f"{k}={v:.4f}" for k, v in sorted(logs.items())))

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.close()
        print("released all input")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
