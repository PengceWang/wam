"""The joint objective.

    L = λ_w L_next_latent + λ_e L_event + λ_a L_actor + λ_v L_value
        + λ_g L_goal + λ_l L_language + λ_p L_prior_retention  (+ λ L_align)

Every term is computed on the *same* hidden states, which is the whole point of
the architecture: perception, prediction, control and value are not separate
networks that happen to be trained together.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from ..config import LossConfig
from ..model.action import ActionChunk
from ..model.wam import SequenceOutput


@dataclass
class Batch:
    """One training sequence. Shapes are (B, T, ...) unless noted."""

    pixels: torch.Tensor  # (B, T, 3, H, W)
    actions: ActionChunk  # chunk executed at each step
    event_ids: torch.Tensor | None = None  # (B, T, n_event_tokens) long
    # World-head targets
    event_targets: torch.Tensor | None = None  # (B, T, vocab) multi-hot for step t+1
    flags: torch.Tensor | None = None  # (B, T, n_flags) float 0/1 for step t+1
    health_delta: torch.Tensor | None = None  # (B, T)
    # Value / drive targets
    returns: torch.Tensor | None = None  # (B, T) bootstrapped exploration return
    # Scalar HUD state per step, in StatusConfig.fields order
    status: torch.Tensor | None = None  # (B, T, n_fields)
    # Semantic grounding: LLM text embedding of the event description at step t
    align_targets: torch.Tensor | None = None  # (B, T, d_model)
    align_mask: torch.Tensor | None = None  # (B, T) 1 where a description exists
    # Hindsight goals: embedding of what was actually achieved after step t
    goal_targets: torch.Tensor | None = None  # (B, T, d_model)
    # 同一件事的多热形式：(B, T, vocab)。乘上短语嵌入表就得到 goal_targets。
    # 分开存是因为那张表在模型上，取数 worker 里没有。
    hindsight: torch.Tensor | None = None
    goal_mask: torch.Tensor | None = None  # (B, T) 1 where something was achieved
    # Language head supervision (token ids of the written-out summary)
    language_targets: torch.Tensor | None = None  # (B, T) long, -100 to ignore
    # Prior retention: a batch of plain text run through both models
    text_ids: torch.Tensor | None = None  # (B2, L) long

    def to(self, device) -> Batch:
        # Walk the fields rather than listing them: a hand-written list drops any
        # field added later in silence, and the symptom is a target that is None
        # on GPU and fine on CPU.
        def move(x):
            return x.to(device) if torch.is_tensor(x) or isinstance(x, ActionChunk) else x

        return Batch(**{f.name: move(getattr(self, f.name)) for f in fields(self)})


def action_loss(pred: dict[str, torch.Tensor], target: ActionChunk) -> torch.Tensor:
    """Behaviour-cloning / actor loss over the factorized action chunk."""
    buttons = F.binary_cross_entropy_with_logits(
        pred["buttons"].float(), target.buttons.float().to(pred["buttons"].device)
    )
    camera = F.cross_entropy(pred["camera"].float().flatten(0, -2), target.camera.reshape(-1))
    hotbar = F.cross_entropy(pred["hotbar"].float().flatten(0, -2), target.hotbar.reshape(-1))
    return buttons + camera + hotbar


def next_latent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Predict the next *semantic* latent, not pixels.

    Cosine + MSE: cosine keeps the direction (what the frame means) honest, MSE
    keeps the scale from collapsing.
    """
    p, t = pred.float(), target.float().detach()
    cos = 1.0 - F.cosine_similarity(p, t, dim=-1).mean()
    return cos + F.mse_loss(p, t)


def event_loss(
    logits: torch.Tensor, targets: torch.Tensor, flags: torch.Tensor | None, flag_logits=None
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    if flags is not None and flag_logits is not None:
        loss = loss + F.binary_cross_entropy_with_logits(flag_logits.float(), flags.float())
    return loss


def value_loss(pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.float(), returns.float())


def align_loss(
    projected: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """L_align = || P_vision(o_t) - E_LLM(event description) ||.

    Pools the visual tokens of a step and pulls that vector towards the LLM's own
    embedding of the matching phrase, so seeing wood activates the language
    model's existing "wood" region instead of an arbitrary direction.
    """
    pooled = projected.float().mean(dim=-2)  # (B, T, d)
    tgt = targets.float().detach()
    per_step = 1.0 - F.cosine_similarity(pooled, tgt, dim=-1)
    if mask is None:
        return per_step.mean()
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_step * mask).sum() / denom


def goal_loss(
    goal_tokens: torch.Tensor,
    targets: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hindsight relabelling: what the agent went on to achieve *was* its goal.

    ``targets`` is the LLM's embedding of whatever actually happened in the next
    few steps ("mined oak log"), so no goal labels have to be authored -- the
    event stream already says what was accomplished, and any trajectory is a
    successful demonstration of reaching the state it ended in.

    Without targets this falls back to the old regulariser, which only keeps the
    tokens from blowing up. Stage 1 has nothing to relabel against, so that is
    still the right behaviour there -- but it is not an objective, and a run that
    never supplies targets is not training the goal head to mean anything.
    """
    if targets is None:
        return goal_tokens.float().pow(2).mean()

    pooled = goal_tokens.float().mean(dim=-2)  # (B, T, d)
    per_step = 1.0 - F.cosine_similarity(pooled, targets.float().detach(), dim=-1)
    if mask is None:
        return per_step.mean()
    mask = mask.float()
    return (per_step * mask).sum() / mask.sum().clamp_min(1.0)


def language_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().flatten(0, -2), targets.reshape(-1), ignore_index=-100)


def prior_retention_loss(student_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """KL( p_theta || p_pretrained ) on ordinary text.

    Without this, online Minecraft RL quietly deletes the language knowledge that
    made the pretrained backbone worth using.
    """
    p = F.log_softmax(student_logits.float(), dim=-1)
    q = F.log_softmax(ref_logits.float(), dim=-1)
    return F.kl_div(q, p, log_target=True, reduction="batchmean")


def compute_losses(
    out: SequenceOutput,
    batch: Batch,
    cfg: LossConfig,
    *,
    language_logits: torch.Tensor | None = None,
    student_text_logits: torch.Tensor | None = None,
    ref_text_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble whichever terms the batch actually has labels for."""
    parts: dict[str, torch.Tensor] = {}

    parts["actor"] = cfg.actor * action_loss(out.action, batch.actions)

    # World head at step t predicts step t+1, so targets are shifted.
    if out.visual_latent.shape[1] > 1:
        parts["next_latent"] = cfg.next_latent * next_latent_loss(
            out.world["next_latent"][:, :-1], out.visual_latent[:, 1:]
        )

    if batch.event_targets is not None:
        parts["event"] = cfg.event * event_loss(
            out.world["event_logits"],
            batch.event_targets,
            batch.flags,
            out.world["flag_logits"],
        )

    if batch.health_delta is not None:
        parts["health"] = cfg.event * F.mse_loss(
            out.world["health_delta"].float(), batch.health_delta.float()
        )

    if batch.returns is not None:
        parts["value"] = cfg.value * value_loss(out.value, batch.returns)

    if batch.align_targets is not None:
        parts["align"] = cfg.align * align_loss(
            out.align_latent, batch.align_targets, batch.align_mask
        )

    parts["goal"] = cfg.goal * goal_loss(out.goal, batch.goal_targets, batch.goal_mask)

    if batch.language_targets is not None and language_logits is not None:
        parts["language"] = cfg.language * language_loss(language_logits, batch.language_targets)

    if student_text_logits is not None and ref_text_logits is not None:
        parts["prior_retention"] = cfg.prior_retention * prior_retention_loss(
            student_text_logits, ref_text_logits
        )

    total = torch.stack(list(parts.values())).sum()
    logs = {k: float(v.detach()) for k, v in parts.items()}
    logs["total"] = float(total.detach())
    return total, logs
