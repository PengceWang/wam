"""The unified Vision-World-Action Transformer.

One shared trunk, one shared hidden state, many heads::

    x_t = [memory | drive | goal | visual_t | prev_action_t | event_t | status_t | readout]
    H_t = LLM(x_<=t)
    -> action / world / value / goal / drive / language

The goal block is both an output and an input: ``GoalHead`` writes it at step t
and it is read back at t+1, so the model states its intention in the same
representation it reasons in. The same slot accepts a goal handed in from
outside, which is what makes instruction-following and autonomy the same
mechanism rather than two architectures.

Because the memory state is recurrent (``M_{t+1} = f(M_t, H_t)``) the sequence
cannot be flattened into a single parallel forward pass: step ``t`` needs the
memory produced at ``t-1``. So training rolls the backbone forward one step at a
time over a KV cache. Each step is only ``tokens_per_step`` tokens wide, so this
is far cheaper than it sounds, and it is exactly the same code path used at
inference and during imagination.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from ..config import WAMConfig
from .action import ActionChunk, ActionEncoder, ActionHead, zero_action
from .backbone import LLMBackbone
from .goal import GoalEncoder
from .heads import DriveHead, EventEmbedding, GoalHead, LanguageHead, ValueHead, WorldHead
from .memory import PersistentMemory
from .status import StatusEncoder
from .vision import VisionFrontend


@dataclass
class RolloutState:
    """Everything carried from one timestep to the next."""

    memory: torch.Tensor  # (B, n_slots, d)
    prev_action: ActionChunk
    # What fills the goal slot on the next step: either what GoalHead just wrote
    # or a goal handed in from outside, which then sticks until it is replaced.
    goal: torch.Tensor | None = None  # (B, n_goal_tokens, d)
    # True while `goal` came from outside, so the head's own proposal does not
    # quietly overwrite the instruction it was given.
    goal_is_external: bool = False
    cache: object | None = None  # transformers Cache

    def detach(self) -> RolloutState:
        goal = None if self.goal is None else self.goal.detach()
        return replace(self, memory=self.memory.detach(), goal=goal)


@dataclass
class StepOutput:
    hidden: torch.Tensor  # (B, tokens_per_step, d) full block
    readout: torch.Tensor  # (B, d) the summary token all heads read
    visual_latent: torch.Tensor  # (B, n_visual_tokens, d) encoding of the current frame
    align_latent: torch.Tensor  # (B, n_visual_tokens, d) projection used by L_align
    action: dict[str, torch.Tensor]
    world: dict[str, torch.Tensor]
    value: torch.Tensor
    goal: torch.Tensor  # (B, n_goal_tokens, d)
    drive: dict[str, torch.Tensor]
    language: torch.Tensor  # (B, d), unembed with backbone.lm_logits


@dataclass
class SequenceOutput:
    """``StepOutput`` fields stacked along a time axis, plus the final state."""

    readout: torch.Tensor  # (B, T, d)
    visual_latent: torch.Tensor  # (B, T, n_vis, d)
    align_latent: torch.Tensor  # (B, T, n_vis, d)
    action: dict[str, torch.Tensor]
    world: dict[str, torch.Tensor]
    value: torch.Tensor
    goal: torch.Tensor
    drive: dict[str, torch.Tensor]
    language: torch.Tensor
    state: RolloutState


class WorldActionModel(nn.Module):
    def __init__(self, cfg: WAMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = LLMBackbone(cfg.backbone)
        d = self.backbone.d_model
        self.d_model = d

        self.vision = VisionFrontend(cfg.vision, d)
        self.memory = PersistentMemory(cfg.memory, d)
        self.action_encoder = ActionEncoder(cfg.action, d)
        self.event_embedding = EventEmbedding(cfg.event, d)
        self.status_encoder = StatusEncoder(cfg.status, d)
        self.goal_encoder = GoalEncoder(cfg.heads.n_goal_tokens, d)
        # Both goal sources go through this before entering the sequence. An MLP
        # head's output and a raw token embedding have no reason to share a
        # scale, and if they do not, withdrawing the external goal in stage 3 is
        # a distribution shift rather than a handover.
        self.goal_norm = nn.LayerNorm(d)

        self.action_head = ActionHead(cfg.action, d, cfg.heads.hidden_mult)
        self.world_head = WorldHead(cfg.heads, cfg.event, d, cfg.vision.n_visual_tokens)
        self.value_head = ValueHead(cfg.heads, d)
        self.goal_head = GoalHead(cfg.heads, d)
        self.drive_head = DriveHead(cfg.heads, d)
        self.language_head = LanguageHead(cfg.heads, d)

        # Learned position/type embedding inside one timestep block, and the
        # readout token whose hidden state every head consumes.
        self.block_pos = nn.Parameter(torch.randn(cfg.tokens_per_step, d) * 0.02)
        self.readout_token = nn.Parameter(torch.randn(d) * 0.02)
        self.input_norm = nn.LayerNorm(d)

        self.to(self.backbone.dtype)

    # -- state ---------------------------------------------------------------

    def initial_state(self, batch_size: int, device=None) -> RolloutState:
        device = device or self.block_pos.device
        return RolloutState(
            memory=self.memory.initial(batch_size, device=device, dtype=self.backbone.dtype),
            prev_action=zero_action(self.cfg.action, batch_size, device=device),
            goal=self.goal_encoder.null(batch_size, device=device, dtype=self.backbone.dtype),
            goal_is_external=False,
            cache=None,
        )

    def set_goal(self, state: RolloutState, texts: list[str] | None) -> RolloutState:
        """Hand the agent a goal, or take it away again.

        ``texts=None`` returns the slot to ``GoalHead``: this is the stage-3
        withdrawal, and it is the only thing that separates "follows
        instructions" from "decides for itself" in this architecture.
        """
        b = state.memory.shape[0]
        if texts is None:
            return replace(state, goal_is_external=False)
        if len(texts) != b:
            raise ValueError(f"expected {b} goal strings, got {len(texts)}")
        goal = self.goal_encoder.from_text(self.backbone, texts).to(self.backbone.dtype)
        return replace(state, goal=goal, goal_is_external=True)

    # -- one timestep --------------------------------------------------------

    def step(
        self,
        state: RolloutState,
        pixels: torch.Tensor,
        event_ids: torch.Tensor | None = None,
        visual_latent: torch.Tensor | None = None,
        status: torch.Tensor | None = None,
    ) -> tuple[StepOutput, RolloutState]:
        """Advance one timestep.

        ``pixels``: (B, 3, H, W) real frame, or ``None``-like when a precomputed
        ``visual_latent`` is supplied instead (imagination mode).
        ``status``: (B, n_fields) HUD scalars; zeros when the env cannot report.
        """
        b = state.memory.shape[0]
        device = state.memory.device
        dtype = self.backbone.dtype

        if visual_latent is None:
            visual_latent = self.vision(pixels)
        visual_latent = visual_latent.to(dtype)

        if event_ids is None:
            event_ids = torch.zeros(
                b, self.cfg.event.n_event_tokens, dtype=torch.long, device=device
            )
        if status is None:
            status = torch.zeros(b, self.cfg.status.n_fields, device=device, dtype=dtype)

        mem_block, drive_block = self.memory.as_input(state.memory)
        act_block = self.action_encoder(state.prev_action.to(device))
        evt_block = self.event_embedding(event_ids).to(dtype)
        sta_block = self.status_encoder(status.to(device, dtype))
        goal_in = state.goal
        if goal_in is None:
            goal_in = self.goal_encoder.null(b, device=device, dtype=dtype)
        goal_block = self.goal_norm(goal_in.to(device, dtype))
        readout = self.readout_token.expand(b, 1, -1)

        x = torch.cat(
            [
                mem_block,
                drive_block,
                goal_block,
                visual_latent,
                act_block,
                evt_block,
                sta_block,
                readout,
            ],
            1,
        )
        x = self.input_norm(x + self.block_pos)

        hidden, cache = self.backbone(x, past_key_values=state.cache, use_cache=True)
        h = hidden[:, -1]  # readout position

        goal = self.goal_head(h)
        out = StepOutput(
            hidden=hidden,
            readout=h,
            visual_latent=visual_latent,
            align_latent=self.vision.align_proj(visual_latent),
            action=self.action_head(h),
            world=self.world_head(h),
            value=self.value_head(h),
            goal=goal,
            drive=self.drive_head(h),
            language=self.language_head(h),
        )

        # The goal the model just set for itself becomes context for the memory
        # update, so intention actually feeds back into the drive slots.
        memory = self.memory(state.memory, torch.cat([hidden, goal], dim=1))
        # An externally set goal outlives the step; a self-proposed one is
        # replaced every step by whatever the head now intends.
        next_goal = state.goal if state.goal_is_external else goal
        new_state = RolloutState(
            memory=memory,
            prev_action=state.prev_action,
            goal=next_goal,
            goal_is_external=state.goal_is_external,
            cache=cache,
        )
        return out, new_state

    # -- a training sequence -------------------------------------------------

    def forward(
        self,
        pixels: torch.Tensor,
        actions: ActionChunk,
        event_ids: torch.Tensor | None = None,
        state: RolloutState | None = None,
        status: torch.Tensor | None = None,
    ) -> SequenceOutput:
        """Teacher-forced rollout over (B, T, 3, H, W) frames and ground-truth actions.

        ``actions[:, t]`` is the chunk *executed at* step t, so it enters the
        sequence as ``prev_action`` at step t+1 and is the target of the action
        head at step t.
        """
        b, t = pixels.shape[:2]
        state = state or self.initial_state(b, device=pixels.device)

        steps: list[StepOutput] = []
        for i in range(t):
            evt = None if event_ids is None else event_ids[:, i]
            sta = None if status is None else status[:, i]
            out, state = self.step(state, pixels[:, i], evt, status=sta)
            steps.append(out)
            # Feed the executed action forward.
            state = replace(
                state,
                prev_action=ActionChunk(
                    buttons=actions.buttons[:, i],
                    camera=actions.camera[:, i],
                    hotbar=actions.hotbar[:, i],
                ),
            )

        return SequenceOutput(
            readout=_stack([s.readout for s in steps]),
            visual_latent=_stack([s.visual_latent for s in steps]),
            align_latent=_stack([s.align_latent for s in steps]),
            action=_stack_dict([s.action for s in steps]),
            world=_stack_dict([s.world for s in steps]),
            value=_stack([s.value for s in steps]),
            goal=_stack([s.goal for s in steps]),
            drive=_stack_dict([s.drive for s in steps]),
            language=_stack([s.language for s in steps]),
            state=state,
        )

    # -- imagination ---------------------------------------------------------

    def imagine(
        self,
        state: RolloutState,
        latent: torch.Tensor,
        horizon: int,
        temperature: float = 1.0,
    ) -> list[StepOutput]:
        """Roll forward without the environment, starting from ``prime()``.

        The world head's predicted next latent is fed straight back in as the
        next step's visual tokens, so actor and value can be trained on imagined
        trajectories instead of real Minecraft steps. Gradients are *not*
        disabled here: stage-2 imagined RL needs to backprop through the rollout.
        Wrap the call in ``torch.no_grad()`` when you only want a preview.
        """
        outs: list[StepOutput] = []
        for _ in range(horizon):
            out, state = self.step(state, pixels=None, visual_latent=latent)
            action = self.action_head.sample(out.readout, temperature)
            state = replace(state, prev_action=action)
            latent = out.world["next_latent"]
            outs.append(out)
        return outs

    @torch.no_grad()
    def prime(
        self, pixels: torch.Tensor, event_ids: torch.Tensor | None = None
    ) -> tuple[RolloutState, torch.Tensor]:
        """Run real frames through the model and return (state, next latent)."""
        b, t = pixels.shape[:2]
        state = self.initial_state(b, device=pixels.device)
        out = None
        for i in range(t):
            evt = None if event_ids is None else event_ids[:, i]
            out, state = self.step(state, pixels[:, i], evt)
            state = replace(state, prev_action=self.action_head.sample(out.readout))
        assert out is not None
        return state, out.world["next_latent"]

    # -- utilities -----------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self) -> list[dict]:
        """Backbone weights get their own (smaller) learning rate once unfrozen."""
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        backbone, rest = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (backbone if id(p) in backbone_ids else rest).append(p)
        groups = [{"params": rest, "lr": self.cfg.train.lr}]
        if backbone:
            groups.append({"params": backbone, "lr": self.cfg.train.backbone_lr})
        return groups


def _stack(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs, dim=1)


def _stack_dict(ds: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([d[k] for d in ds], dim=1) for k in ds[0]}
