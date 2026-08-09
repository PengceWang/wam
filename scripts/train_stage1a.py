"""Stage 1a: can an instruction in the goal slot drive the action head?

    python scripts/train_stage1a.py --overfit 32      # sanity: can it memorise?
    python scripts/train_stage1a.py --max-steps 2000  # the real thing

The overfit run comes first and is not optional. It takes minutes and catches
the failures that otherwise surface after a day of training: labels off by a
chunk, camera axes swapped, the goal never reaching the sequence. If a model
cannot memorise 32 windows, nothing about the 40,000-window run will be
informative.

Losses are cut down to what stage 1a actually has labels for. Event, align and
value are off: this stage loads only the image and action trees, so their
targets do not exist, and a term computed against absent labels is not a
regulariser -- it is noise with a weight.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.config import WAMConfig
from wam.data.stage1a import Stage1aData, goal_for, instruction_goal_tokens
from wam.model.wam import WorldActionModel
from wam.training.losses import action_loss, next_latent_loss
from wam.training.trainer import configure_stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3-0.6b.yaml")
    ap.add_argument("--root", default="/home/wangp/data/6xx")
    ap.add_argument("--index", default="/home/wangp/data/6xx_index_seq8.npz")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="reuse N fixed batches instead of sampling fresh ones",
    )
    ap.add_argument("--out", default="checkpoints/stage1a.pt")
    ap.add_argument(
        "--wandb", default=None, metavar="PROJECT", help="log to this Weights & Biases project"
    )
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="a run that only checkpoints at the end loses everything to one crash",
    )
    args = ap.parse_args()

    cfg = WAMConfig.from_yaml(args.config)
    cfg.train.stage = 1
    if args.batch_size:
        cfg.train.batch_size = args.batch_size

    # The index fixes the window layout, so it -- not the YAML -- decides
    # seq_len. Keeping them in sync by hand is how you end up training on
    # windows whose boundaries do not match the labels they were given.
    import numpy as np

    index_seq_len = int(np.load(Path(args.index), allow_pickle=True)["seq_len"])
    if cfg.train.seq_len != index_seq_len:
        print(f"seq_len {cfg.train.seq_len} -> {index_seq_len} (from the index)")
        cfg.train.seq_len = index_seq_len

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = WorldActionModel(cfg).to(device)
    configure_stage(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"device {device} | params {total / 1e6:.1f}M, trainable {trainable / 1e6:.1f}M")
    print(
        f"seq_len {cfg.train.seq_len} ({cfg.train.seq_len * cfg.action.chunk_size / 20:.1f}s), "
        f"batch {cfg.train.batch_size}, tokens/step {cfg.tokens_per_step}"
    )

    data = Stage1aData(cfg, args.root, args.index, episodes=None)
    print(f"index: {len(data.label_of):,} labelled windows")

    # Eight fixed instructions, embedded once with the backbone's own tokenizer.
    goal_table = instruction_goal_tokens(model, device)
    print(f"goal table {tuple(goal_table.shape)}")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb,
            name=args.run_name,
            config={
                **cfg.to_dict(),
                "trainable_params": trainable,
                "total_params": total,
                "labelled_windows": len(data.label_of),
                # The loss a model gets by reproducing the marginal action
                # distribution and nothing else, measured on the contractor
                # data. Anything at or above this has learned nothing, so it
                # belongs on the chart rather than in a comment.
                "actor_trivial_floor": 2.4719,
            },
        )
        wandb.define_metric("actor", summary="min")

    optimiser = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.train.weight_decay)

    fixed = None
    if args.overfit:
        fixed = [data.batch(cfg.train.batch_size) for _ in range(args.overfit)]
        print(
            f"overfit mode: {args.overfit} fixed batches, "
            f"{args.overfit * cfg.train.batch_size} sequences"
        )

    model.train()
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        batch, texts = fixed[step % len(fixed)] if fixed else data.batch(cfg.train.batch_size)
        batch = batch.to(device)

        state = model.initial_state(cfg.train.batch_size, device=device)
        state = type(state)(
            memory=state.memory,
            prev_action=state.prev_action,
            goal=goal_for(texts, goal_table),
            goal_is_external=True,  # the instruction must survive every step
        )

        out = model(batch.pixels, batch.actions, state=state)

        actor = action_loss(out.action, batch.actions)
        loss = cfg.loss.actor * actor
        parts = {"actor": actor.detach()}
        if out.visual_latent.shape[1] > 1:
            latent = next_latent_loss(out.world["next_latent"][:, :-1], out.visual_latent[:, 1:])
            loss = loss + 0.5 * latent
            parts["latent"] = latent.detach()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
        )
        optimiser.step()

        if run is not None:
            run.log(
                {**{k: float(v) for k, v in parts.items()},
                 "total": float(loss.detach()),
                 "grad_norm": float(grad),
                 "instructions": len(set(texts)),
                 "lr": optimiser.param_groups[0]["lr"]},
                step=step,
            )

        if step % args.save_every == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "step": step}, args.out)

        if step % args.log_every == 0 or step == 1:
            rate = step / (time.time() - t0)
            shown = " ".join(f"{k}={v:.4f}" for k, v in sorted(parts.items()))
            print(
                f"{step:>6} {shown} grad={float(grad):.2f} total={float(loss):.4f} {rate:.2f} it/s"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, args.out)
    print(f"saved {args.out}")
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
