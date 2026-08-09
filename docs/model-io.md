# What goes in, what comes out

One pretrained LLM is the trunk. Frames and symbols go in, keyboard and mouse
come out, and every head reads the same hidden state.

```
              ┌──────────── one timestep, every 0.4 s ────────────┐

  Minecraft ──┤ 1 frame  224x224x3                                │
              │ 6 HUD scalars                                     │
              │ event ids (what just happened)                    │──┐
              └───────────────────────────────────────────────────┘  │
                                                                     ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  x_t  =  83 tokens                                                   │
   │                                                                      │
   │  memory 40 │ drive 8 │ goal 4 │ visual 16 │ prev_act 4 │ evt 8 │ st 2 │ ro 1
   │  ▲                     ▲                                             │
   │  └── carried from t-1  └── self-proposed OR handed in                │
   └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │   Qwen3-0.6B   28 layers, d=1024, KV cache holds every past step      │
   └──────────────────────────────────────────────────────────────────────┘
                                    │
                       readout token's hidden state h
                                    │
        ┌──────────┬──────────┬─────┴─────┬──────────┬──────────┐
        ▼          ▼          ▼           ▼          ▼          ▼
     Action     World       Value       Goal       Drive     Language
      head       head        head       head        head       head
        │          │           │          │           │          │
   8 ticks of   next latent  long-term  next      novelty     readable
   key + mouse  + events     value      objective  progress   summary
        │                                 │        danger
        ▼                                 └──► back into x_{t+1}
    Minecraft
```

## The frame rate question

**The model sees one frame per decision, and decides 2.5 times a second.**

```
Minecraft ticks      │▌│▌│▌│▌│▌│▌│▌│▌│▌│▌│▌│▌│▌│▌│▌│▌│   20 per second
model sees           └───────────────┘└───────────────┘   1 frame each
model decides        ▲               ▲                    every 8 ticks
                     └─ 8 actions ───┘└─ 8 actions ───┘   = 0.4 s
```

| | |
| --- | --- |
| Minecraft tick rate | 20 Hz |
| `chunk_size` | 8 ticks |
| One forward pass covers | 0.4 s of gameplay |
| Model call rate | **2.5 Hz** |
| Frames the model actually sees | **2.5 per second** |
| Frames rendered but discarded | 7 out of every 8 |

This is deliberate — the readme's "one model call is about half a second of
control". A 0.6B backbone cannot run at 20 Hz, and it does not need to: the
action head emits eight low-level actions at once, so control stays at tick
resolution even though perception does not.

**What it costs.** The seven skipped frames are genuinely gone. Anything that
happens and finishes inside 0.4 s — an arrow in flight, a creeper's fuse, a mob
swinging — is invisible unless its consequence survives into the next frame or
shows up in the event stream. If that turns out to matter, the knob is
`chunk_size`, not the architecture.

## Context: how long can it run?

Every timestep appends 83 tokens to the KV cache and nothing evicts them.

```
83 tokens/step  x  2.5 steps/s  =  ~208 tokens per second of gameplay
Qwen3-0.6B max_position_embeddings = 40960
40960 / 83  =  493 steps  =  ~3.3 minutes of continuous play
```

**Three minutes, then it hits the wall.** There is no truncation, no sliding
window and no eviction anywhere in the codebase yet, so this is a hard stop
rather than a graceful degradation. The persistent memory tokens exist precisely
so that history can be *compressed* instead of *retained* — but nothing yet
drops old steps out of the cache to make that pay off. This is the next real
architectural gap, and it is bigger than it looks: Minecraft progress is measured
in hours.

## Input, block by block

| Block | Tokens | Where it comes from |
| --- | --- | --- |
| memory | 40 | recurrent: 16 scene + 16 world + 8 skill, updated each step from `H_t` |
| drive | 8 | recurrent: exploration motivation, same update |
| goal | 4 | `GoalHead` at t-1, **or** an instruction encoded by the LLM's own embeddings |
| visual | 16 | one frame → frozen vision tower → Perceiver resampler |
| prev_action | 4 | the 8 actions actually executed last step |
| event | 8 | concept ids: `mined spruce log`, `picked up dirt`, … |
| status | 2 | health, food, `is_gui_open`, light level, sky visible, raining |
| readout | 1 | learned token; every head reads *this* position's hidden state |

### What the model is not given

**The 36 inventory slots and the xyz position.** A player has to open the
inventory to know what they carry, and press F3 to know where they stand.
Handing those over free would delete the reason to press `inventory` — which is
the only route into crafting — and remove any need for spatial memory. Inventory
arrives as `pickup` / `craft_item` events instead; keeping track of it is the
memory slots' job.

## Output, head by head

| Head | Shape | Meaning |
| --- | --- | --- |
| Action | `buttons (8,11)`, `camera (8,2)`, `hotbar (8,)` | 0.4 s of keyboard and mouse |
| World | next visual latent + event logits + 5 flags + health delta | what it expects to happen |
| Value | scalar (or a distribution) | long-horizon exploration value |
| Goal | `(4, d)` | the next objective, written back into `x_{t+1}` |
| Drive | novelty, progress, danger | intrinsic reward — why it bothers |
| Language | `(d,)` → unembed | a readable summary of what it learned |

### The action space

11 buttons, all named exactly as MineRL names them:

```
forward  back  left  right  jump  sneak  sprint  attack  use  pickItem  inventory
```

plus `camera` (11 bins per axis, mu-law) and `hotbar` (0 = keep current slot,
1-9 = switch). `inventory` is the only way into crafting, and while the GUI is
open the camera moves a cursor rather than the head — which is what the
`is_gui_open` status bit is for.

## Two ways to run the same model

```
REAL          frame ──► LLM ──► action ──► Minecraft ──► frame
IMAGINED      latent ──► LLM ──► action ──► World head ──► predicted latent ──┐
                ▲                                                            │
                └────────────────────────────────────────────────────────────┘
```

Same weights, same code path, same `step()`. In imagination the World head's
prediction is fed back in place of a real frame, so the actor and value heads can
train on trajectories that never touched the simulator.

## Goals: instruction and autonomy are one slot

```
model.set_goal(state, ["chop oak wood"])   # external: sticks until replaced
model.set_goal(state, None)                # withdrawn: GoalHead fills it again
```

Both sources pass through one `LayerNorm` before entering the sequence. If they
differed in scale the model would learn to tell them apart, and withdrawing the
scaffolding would be a distribution shift rather than a handover.

The point of the scaffolding: pure novelty-seeking in Minecraft parks itself in
front of flowing water — the least learnable thing is the most rewarding to
watch. Train with goals handed in, then take them away and see whether it still
acts. That turns "is it autonomous?" into something measurable.

## Numbers, current config

| | |
| --- | --- |
Measured, not estimated — `scripts/smoke_test.py --config configs/qwen3-0.6b.yaml`
on the RTX 3060:

| | |
| --- | --- |
| Backbone | Qwen3-0.6B — 28 layers, d=1024, **596.0M** |
| Vision tower | SigLIP base patch16-224, frozen — **119.9M** |
| Resampler + heads | **82.9M** |
| Total | **798.8M** |
| Trainable in stage 1 | **109.9M (13.8%)** |
| Frame | 224x224x3 → 16 tokens |
| Sequence | **83** tokens per timestep |

Shapes from the same run, confirming the layout end to end:

```
action.buttons (1, 16, 8, 11)      8 ticks x 11 buttons
action.camera  (1, 16, 8, 2, 11)   8 ticks x (pitch, yaw) x 11 bins
goal           (1, 16, 4, 1024)
memory         (1, 48, 1024)       40 memory + 8 drive
```
