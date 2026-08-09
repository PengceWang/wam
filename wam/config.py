"""Configuration for the Pretrained Developmental World-Action Model.

The whole model is described by nested dataclasses so that a run is reproducible
from a single YAML file. See ``configs/default.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class VisionConfig:
    """Frontend that turns an RGB frame into a handful of semantic tokens."""

    # HF vision tower. "random" builds a tiny CNN instead (no download, for tests).
    encoder_name: str = "random"
    image_size: int = 224
    freeze_encoder: bool = True
    # Perceiver-style resampler: how many tokens survive per frame.
    n_visual_tokens: int = 16
    resampler_depth: int = 2
    resampler_heads: int = 8
    # Width of the tiny CNN used when encoder_name == "random".
    random_encoder_dim: int = 256


@dataclass
class MemoryConfig:
    """Persistent memory tokens carried across timesteps (readme: 16/16/8/8)."""

    n_scene: int = 16  # what the agent is looking at right now
    n_world: int = 16  # long-term world knowledge
    n_skill: int = 8  # what it is currently getting good at
    n_drive: int = 8  # exploration motivation ("drive tokens" in x_t)
    # Leak of the update gate at init: 0 -> memory starts as a pure carry-over.
    update_gate_bias: float = -2.0

    @property
    def n_memory(self) -> int:
        """Memory block that sits at the front of every step (drive is separate)."""
        return self.n_scene + self.n_world + self.n_skill


@dataclass
class ActionConfig:
    """Factorized keyboard/mouse action space, emitted in chunks."""

    # Low-level actions predicted per forward pass (readme: ~0.5s of control).
    chunk_size: int = 8
    # Binary buttons, in a fixed order. Index = bit position in the packed action.
    # These are exactly MineRL's SIMPLE_KEYBOARD_ACTION, spelled the same way, so
    # the adapter can use the name as the key it writes into a MineRL tick.
    # `drop` and `swapHands` are absent because MineRL leaves them commented out;
    # there is no channel to press them through.
    buttons: tuple[str, ...] = (
        "forward",
        "back",
        "left",
        "right",
        "jump",
        "sneak",
        "sprint",
        "attack",
        "use",
        "pickItem",  # middle click: copy the targeted block into the hotbar
        # The only way into crafting: MineRL's HumanSurvival exposes no craft,
        # place or equip action, so recipes have to be clicked out in the GUI.
        # While it is open the camera moves a cursor instead of the head --
        # `is_gui_open` in the observation is what tells the two apart.
        "inventory",
    )
    # Camera is discretised into bins per axis (yaw / pitch).
    #
    # Chosen against the contractor data rather than by taste. Measured over
    # 192k frames of 6xx: half of all camera values are exactly 0, and half the
    # nonzero ones are under 1.2 degrees. The number that matters is how much
    # genuine motion collapses into the no-motion bin, because a turn quantised
    # to zero is a label that says the human did not move:
    #
    #   11 bins, +/-15, mu=8   ->  36.3% of real turns lost, MAE 0.350
    #   11 bins, +/-10, mu=10  ->  28.6% lost, MAE 0.402   (VPT's setting)
    #   21 bins, +/-20, mu=40  ->   1.8% lost, MAE 0.208   <- this
    #
    # A large turn clipped at the outermost bin still says "turn a lot"; a small
    # turn rounded to zero says "hold still". The two errors are not equivalent,
    # so mu is aggressive and the range is wide rather than tight.
    camera_bins: int = 21
    camera_max_delta: float = 20.0  # degrees at the outermost bin
    camera_mu: float = 40.0  # mu-law curvature: higher = finer near zero
    n_hotbar: int = 9
    # How many tokens the previous action occupies in the input sequence.
    n_action_tokens: int = 4

    @property
    def n_buttons(self) -> int:
        return len(self.buttons)


@dataclass
class EventConfig:
    """Symbolic game events, used both as input tokens and as grounding targets."""

    n_event_tokens: int = 8  # fixed budget per step, padded
    vocab_size: int = 1024  # concept vocabulary (<object: oak log>, <event: collected>, ...)
    # Path to a json mapping concept-id -> natural language, used to initialise the
    # concept embeddings from the LLM's own text embeddings.
    concept_vocab_path: str | None = None


@dataclass
class StatusConfig:
    """Scalar game state fed in alongside the frame.

    The rule for what belongs here: things a player reads off the HUD at a
    glance, which 128px of downscaled frame cannot resolve. Deliberately *not*
    here: the 36 inventory slots and the xyz position. A player has to open the
    inventory to know what they carry and press F3 to know where they stand, and
    handing those over for free would delete both the reason to press
    ``inventory`` and the need for any spatial memory. Inventory reaches the
    model as ``pickup`` / ``craft_item`` events instead, which is what the
    persistent memory slots are for.
    """

    n_tokens: int = 2
    fields: tuple[str, ...] = (
        "health",
        "food",
        "is_gui_open",
        "light_level",
        "can_see_sky",
        "is_raining",
    )

    @property
    def n_fields(self) -> int:
        return len(self.fields)


@dataclass
class GoalConfig:
    """Where the goal tokens in ``x_t`` come from.

    The readme's agent proposes its own goals, but pure novelty-seeking in
    Minecraft parks itself in front of flowing water: the noisy-TV failure, where
    the least learnable thing is the most rewarding to watch. So the slot takes
    either source. ``external`` is scaffolding -- train with goals handed in,
    then withdraw them and let ``GoalHead`` fill the same slot. The autonomy
    claim survives, and "can it still act when nobody tells it what to do?"
    becomes a measurable question rather than an assumption.

    Token count lives in ``HeadsConfig.n_goal_tokens``: the head that writes the
    slot and the slot itself have to agree, and the head owns the number.
    """

    # "self": GoalHead fills the slot. "external": a supplied goal does.
    # "mixed": external with probability `external_prob`, self otherwise.
    source: str = "self"
    external_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.source not in ("self", "external", "mixed"):
            raise ValueError(f"unknown goal source {self.source!r}")


@dataclass
class BackboneConfig:
    """The pretrained LLM that everything shares."""

    # "random" builds a small randomly-initialised Llama for tests (no download).
    model_name: str = "random"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    # Stage 1 freezes everything; stage 2 unfreezes the top-k blocks.
    n_trainable_top_layers: int = 0
    # Only used when model_name == "random".
    random_hidden_size: int = 256
    random_n_layers: int = 4
    random_n_heads: int = 4
    random_vocab_size: int = 2048


@dataclass
class HeadsConfig:
    hidden_mult: int = 2  # width of the MLP inside each head, relative to d_model
    n_goal_tokens: int = 4
    value_bins: int = 1  # >1 switches the value head to a distributional (HL-Gauss) head


@dataclass
class LossConfig:
    """Weights for L = Σ λ_i L_i (readme)."""

    next_latent: float = 1.0
    event: float = 1.0
    actor: float = 1.0
    value: float = 0.5
    goal: float = 0.1
    language: float = 0.1
    align: float = 1.0  # semantic grounding
    prior_retention: float = 0.1  # KL against the frozen pretrained LLM


@dataclass
class TrainConfig:
    stage: int = 1  # 1: grounding, 2: top layers + online, 3: full joint
    seq_len: int = 16  # timesteps per training sequence
    batch_size: int = 2
    lr: float = 1e-4
    backbone_lr: float = 1e-5  # smaller LR for LLM weights once unfrozen
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 1000
    log_every: int = 10
    device: str = "cuda"
    seed: int = 0


@dataclass
class WAMConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    event: EventConfig = field(default_factory=EventConfig)
    status: StatusConfig = field(default_factory=StatusConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def tokens_per_step(self) -> int:
        """Length of one timestep block in the unified sequence.

        [memory | drive | goal | visual | prev_action | event | status | readout]
        """
        return (
            self.memory.n_memory
            + self.memory.n_drive
            + self.heads.n_goal_tokens
            + self.vision.n_visual_tokens
            + self.action.n_action_tokens
            + self.event.n_event_tokens
            + self.status.n_tokens
            + 1  # readout token that all heads read from
        )

    # -- (de)serialisation -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> WAMConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WAMConfig:
        return _build(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively instantiate nested dataclasses, rejecting unknown keys."""
    # `from __future__ import annotations` turns field.type into a string, so the
    # real classes have to be resolved before we can recurse into them.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        ftype = hints[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
