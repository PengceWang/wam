"""Staged trainer.

The readme is explicit that everything must not be unfrozen at once:

    stage 1  freeze the LLM; train the vision projector, action tokenizer and
             world/event heads until Minecraft state activates the right concepts
    stage 2  unfreeze the top blocks; add persistent memory + imagined RL
    stage 3  full joint training at a small LR, with prior-retention KL on

``Trainer`` only differs across stages by which parameters require grad and
which loss terms are switched on, which is what ``configure_stage`` encodes.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import torch

from ..config import WAMConfig
from ..model.backbone import FrozenReference
from ..model.wam import WorldActionModel
from .losses import Batch, compute_losses


def configure_stage(model: WorldActionModel, cfg: WAMConfig) -> None:
    """Set which parameters train, per the staged schedule."""
    stage = cfg.train.stage
    if stage == 1:
        model.backbone.set_trainable_top_layers(0)
    elif stage == 2:
        k = cfg.backbone.n_trainable_top_layers or len(model.backbone.layers) // 2
        model.backbone.set_trainable_top_layers(k)
    elif stage == 3:
        model.backbone.set_trainable_top_layers(len(model.backbone.layers))
    else:
        raise ValueError(f"unknown training stage {stage}")

    # The vision tower stays frozen in every stage; only the resampler learns.
    if cfg.vision.freeze_encoder:
        for p in model.vision.encoder.parameters():
            p.requires_grad = False


class Trainer:
    def __init__(self, cfg: WAMConfig, model: WorldActionModel | None = None) -> None:
        torch.manual_seed(cfg.train.seed)
        self.cfg = cfg
        self.device = torch.device(
            cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"
        )
        self.model = (model or WorldActionModel(cfg)).to(self.device)
        configure_stage(self.model, cfg)

        self.optimizer = torch.optim.AdamW(
            self.model.param_groups(), weight_decay=cfg.train.weight_decay
        )
        # Only stage 3 needs the frozen reference; building it doubles LLM memory.
        self.reference: FrozenReference | None = None
        if cfg.loss.prior_retention > 0 and cfg.train.stage >= 3:
            self.reference = FrozenReference(cfg.backbone).to(self.device)

        self.step_idx = 0

    # -- one optimisation step ----------------------------------------------

    def train_step(self, batch: Batch) -> dict[str, float]:
        self.model.train()
        batch = batch.to(self.device)

        out = self.model(batch.pixels, batch.actions, batch.event_ids, status=batch.status)

        language_logits = None
        if batch.language_targets is not None:
            language_logits = self.model.backbone.lm_logits(out.language)

        student_text, ref_text = None, None
        if self.reference is not None and batch.text_ids is not None:
            emb = self.model.backbone.get_input_embeddings()(batch.text_ids)
            hidden, _ = self.model.backbone(emb)
            student_text = self.model.backbone.lm_logits(hidden)
            ref_text = self.reference.logits(batch.text_ids)

        loss, logs = compute_losses(
            out,
            batch,
            self.cfg.loss,
            language_logits=language_logits,
            student_text_logits=student_text,
            ref_text_logits=ref_text,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.trainable_parameters(), self.cfg.train.grad_clip
        )
        self.optimizer.step()

        logs["grad_norm"] = float(grad_norm)
        self.step_idx += 1
        return logs

    # -- loop ----------------------------------------------------------------

    def fit(self, batches: Iterable[Batch]) -> None:
        it: Iterator[Batch] = iter(batches)
        t0 = time.time()
        while self.step_idx < self.cfg.train.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(batches)
                continue
            logs = self.train_step(batch)
            if self.step_idx % self.cfg.train.log_every == 0:
                rate = self.step_idx / max(time.time() - t0, 1e-6)
                parts = " ".join(f"{k}={v:.4f}" for k, v in sorted(logs.items()))
                print(f"[{self.step_idx:>6}] {parts} ({rate:.2f} it/s)", flush=True)

    # -- checkpointing --------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "step": self.step_idx,
                "config": self.cfg.to_dict(),
                # Frozen backbone weights are reloadable from HF; only keep what trains.
                "model": {
                    k: v
                    for k, v in self.model.state_dict().items()
                    if not k.startswith("backbone.") or self.cfg.train.stage >= 2
                },
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, strict: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_idx = ckpt.get("step", 0)
