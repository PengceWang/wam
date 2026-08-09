"""Give the model each instruction in a live game and measure what happens.

    python scripts/eval_stage1a.py --checkpoint /home/wangp/stage1a_run1.pt

Two independent questions, because they fail independently:

  obeyed?   did the world change the way the instruction asked -- measured from
            player_pos, not from what the model claims it pressed
  human?    is the camera trace correlated in time like a person's, or is it
            per-tick white noise with the right marginal distribution

A model can score well on the first while failing the second completely, and
neither shows up in the training loss.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.config import WAMConfig
from wam.data.contractor import INSTRUCTIONS
from wam.data.stage1a import instruction_goal_tokens
from wam.model.wam import WorldActionModel
from wam.training.eval_stage1a import humanness, obeyed, run_instruction


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=3, help="repeats per instruction")
    ap.add_argument("--steps", type=int, default=20, help="model calls per episode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = WAMConfig.from_dict(blob["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WorldActionModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    print(f"loaded {args.checkpoint}  |  device {device}")

    from wam.envs.minerl import MineRLEnv

    env = MineRLEnv(cfg, seed=args.seed)
    goal_table = instruction_goal_tokens(model, device)

    centre = cfg.action.camera_bins // 2
    results = defaultdict(list)
    traces = []

    for instruction in INSTRUCTIONS:
        for _ in range(args.episodes):
            res, camera = run_instruction(model, env, instruction, goal_table, steps=args.steps)
            results[instruction].append(res)
            traces.append(camera)

    print()
    print(f"{'instruction':<26} {'obeyed':>8}  measurement")
    print("-" * 78)
    total_ok = total_n = 0
    for instruction in INSTRUCTIONS:
        runs = results[instruction]
        ok = sum(obeyed(r) for r in runs)
        total_ok += ok
        total_n += len(runs)
        fwd = np.mean([r.forward_distance for r in runs])
        lat = np.mean([r.lateral_distance for r in runs])
        yaw = np.mean([r.yaw_delta for r in runs])
        mined = np.mean([r.blocks_mined for r in runs])
        print(
            f"{instruction:<26} {ok}/{len(runs):<6}  "
            f"fwd={fwd:+6.2f} lat={lat:+6.2f} dyaw={yaw:+7.1f} mined={mined:.1f}"
        )

    print("-" * 78)
    print(f"{'OBEYED':<26} {total_ok}/{total_n}  ({100 * total_ok / max(1, total_n):.0f}%)")

    print()
    stats = humanness(np.concatenate(traces, axis=0), centre)
    print("=== camera time structure (human reference in brackets) ===")
    for name in ("yaw", "pitch"):
        print(
            f"  {name:<6} autocorr {stats[f'{name}_autocorr']:+.3f}  "
            f"[{stats[f'{name}_autocorr_human']:+.3f}]     "
            f"still {100 * stats[f'{name}_still']:5.1f}%  "
            f"[{100 * stats[f'{name}_still_human']:5.1f}%]"
        )
    print(f"  frames: {stats['n']:,}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
