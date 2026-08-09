# Getting started

Implementation of the design in [`readme.md`](../readme.md): a pretrained ~0.6B
LLM used as the **central backbone** of a Minecraft agent, with vision, action,
world prediction, value, goals and memory all sharing one transformer and one
hidden state.

> The folder is called `MineStudio` for historical reasons, but there are now two
> ways into the game and they run on different operating systems:
>
> | Env | Where | Symbols |
> | --- | --- | --- |
> | `EaglercraftEnv` (browser + screen capture) | Windows | none |
> | `MineRLEnv` (MineStudio / MCP-Reborn) | WSL2 Ubuntu | inventory, events, biome |
>
> See [wsl-minerl.md](wsl-minerl.md) for the WSL side.

## Setup

Already done in this checkout, but to reproduce:

```bash
uv sync --extra dev          # creates .venv, pulls torch 2.13.0+cu130
.venv/Scripts/python.exe -m pytest tests -q
```

`pyproject.toml` pins the CUDA wheel index explicitly — PyPI's default torch on
Windows is CPU-only.

## Check it works

```bash
# tiny random backbone, CPU, no downloads, ~20s
.venv/Scripts/python.exe scripts/smoke_test.py

# real Qwen3-0.6B + SigLIP on the GPU (~1.5GB download on first run)
.venv/Scripts/python.exe scripts/smoke_test.py --config configs/qwen3-0.6b.yaml
```

## Train

```bash
# plumbing check on synthetic batches
.venv/Scripts/python.exe scripts/train.py --config configs/qwen3-0.6b.yaml --max-steps 50

# online, against whatever build_env() returns
.venv/Scripts/python.exe scripts/train.py --config configs/qwen3-0.6b.yaml --source env
```

## Layout

| Path | What it is |
| --- | --- |
| `wam/config.py` | every knob, one nested dataclass tree, loadable from YAML |
| `wam/model/vision.py` | frozen vision tower + Perceiver resampler → 16 tokens/frame |
| `wam/model/action.py` | factorized action space, chunk encoder, chunk head |
| `wam/model/memory.py` | persistent memory tokens, `M_{t+1} = f(M_t, H_t)` |
| `wam/model/backbone.py` | the LLM, plus the frozen reference for prior retention |
| `wam/model/heads.py` | world / value / goal / drive / language heads |
| `wam/model/wam.py` | assembles `x_t`, runs the rollout, does imagination |
| `wam/training/losses.py` | the `L = Σ λ_i L_i` objective |
| `wam/training/trainer.py` | the 3-stage schedule |
| `wam/envs/base.py` | **the seam to Minecraft** — implement this next |
| `wam/data/collect.py` | rollout → `Batch` |

## The token layout

Per timestep, `tokens_per_step = 83`:

```
[ memory 40 | drive 8 | goal 4 | visual 16 | prev_action 4 | event 8 | status 2 | readout 1 ]
   ^ 16 scene + 16 world + 8 skill                                                ^ heads read here
```

### The goal slot is read as well as written

`GoalHead` writes the goal block at step *t* and the model reads it back at
*t+1*, so intention travels in the same representation everything else does. The
same slot also accepts a goal handed in from outside (`model.set_goal(state,
["chop oak wood"])`), encoded with the backbone's own token embeddings — the
backbone is an LLM, so an instruction needs no new modality and no alignment
loss.

That makes instruction-following and autonomy **one mechanism, not two
architectures**. It matters because pure novelty-seeking in Minecraft parks
itself in front of flowing water: the noisy-TV failure, where the least
learnable thing is the most rewarding to watch. External goals are scaffolding —
train with them, then withdraw them (`set_goal(state, None)`) and let the head
fill the same slot. Both sources pass through one `LayerNorm` first, so the
withdrawal is a handover rather than a distribution shift.

### What the model is allowed to see

`status` carries what a player reads off the HUD at a glance and 128px cannot
resolve: health, food, `is_gui_open`, light level, sky visibility, rain.

Deliberately absent: **the 36 inventory slots and the xyz position**. A player
has to open the inventory to know what they carry and press F3 to know where they
stand. Handing those over free would delete the reason to press `inventory` —
which is the only route into crafting — and remove any need for spatial memory.
Inventory reaches the model as `pickup` / `craft_item` events instead, and
keeping track of it is what the persistent memory slots are for.

Memory is recurrent, so step `t` needs the memory produced at `t-1`. The rollout
therefore steps the backbone one timestep at a time over a growing KV cache
rather than flattening the sequence into one pass. Same code path for training,
inference and imagination.

**Do not enable `backbone.gradient_checkpointing`.** transformers disables
`use_cache` whenever checkpointing is on *and* the module is training, which
silently drops all cross-timestep attention (and only in training mode, so eval
still looks fine). The backbone raises rather than allowing it; a test locks the
cache growth in. Lower `seq_len`/`batch_size` if VRAM is tight.

## Measured on the RTX 3060 (12GB), stage 1, bf16

| config | s/step | peak VRAM |
| --- | --- | --- |
| batch 1 × seq 8 | 1.7 | 3.7 GiB |
| batch 1 × seq 16 | 3.3 | 6.2 GiB |
| batch 2 × seq 16 | 8.7 | 10.5 GiB |

Parameter counts with `configs/qwen3-0.6b.yaml`: 797M total — 596M backbone,
120M vision tower, 81M resampler + heads. Stage 1 trains 108M of them (13.5%).

## The browser rig

The game is Eaglercraft 1.8.8 in Edge, so there is no Java process and no MineRL
hook. Frames come off the window; control goes in as OS-level input.

| Piece | File |
| --- | --- |
| Frame capture (WGC ~56fps, mss / PrintWindow fallbacks) | `wam/envs/browser.py` |
| `SendInput` keyboard + mouse | `wam/envs/control.py` |
| The env: pause handling, action execution, safety | `wam/envs/eaglercraft.py` |

```bash
.venv/Scripts/python.exe scripts/capture_probe.py     # is the stream live?
.venv/Scripts/python.exe scripts/control_test.py      # does each switch work?
.venv/Scripts/python.exe scripts/agent_play.py --steps 20   # model drives
```

### Your window is yours

Position and size are never touched — no moving, resizing or maximising. The one
exception is un-minimising, since a minimised window cannot be captured at all,
and even that restores *your* previous geometry (`restore_if_minimised=False`
turns it off).

All geometry is read fresh on every use, so you can drag or resize the browser
mid-run and nothing cares. It comes from Chromium's own viewport child window
(`Chrome_RenderWidgetHostHWND`), which is exact and free:

- the clickable region is the page, never the tab strip or address bar;
- observations are cropped to the page, so no browser UI reaches the encoder;
- frames are **letterboxed**, not stretched, so changing the window aspect ratio
  does not silently change the world geometry the model learned on.

An earlier version guessed the chrome height as 12% of window height. Measured
against the real window that was 163px versus a true 80px — and being a
*fraction* it was wrong at every window size, since chrome is a fixed pixel
height. That mistake let a click reach the tab strip and close the page.

### Safety rails

Injected input goes wherever the cursor is, and relative mouse moves physically
drag the cursor whenever Pointer Lock is not held. Four guards, all covered by
`tests/test_click_safety.py`:

1. **`safe_click`** refuses any click outside the page viewport.
2. **`keep_cursor_inside`** parks a drifted cursor back on the canvas and skips
   that tick's attack/use rather than clicking somewhere untrusted.
3. **`assert_window_alive`** stops if the window is gone *or* if the game has
   fallen into a background tab. Tabs are never switched for you.
4. **`max_mouse_step`** caps one tick's mouse delta.

`Observation.info` reports `cursor_escapes` and `geometry_changes`. A rising
`cursor_escapes` means Pointer Lock is not held and camera actions are going
nowhere.

### Pointer Lock

There is no cheap reliable way to ask the browser whether the pointer is locked.
Both obvious tests lie: Chromium does **not** pin `GetCursorPos` while locked,
and whole-frame correlation reads drifting clouds as motion while a steeply
pitched camera turns yaw into image rotation rather than translation.

So `ensure_playing` does not test — it always performs the gesture that grants
the lock: ESC to open the pause menu, then click "Back to Game". Clicking the
canvas of an already-running game does *not* work; measured, it just swings the
arm. Cost is a menu flash per reset, which is cheap because resets are rare.

### Camera sensitivity is not calibrated, on purpose

`pixels_per_degree` converts a camera bin to a mouse delta. It does not need to
be accurate — the model only ever sees frames its own actions produced, so any
consistent scale is learnable. It is a sensitivity knob, not a constant to
measure.

## What is deliberately still a stub

These are placeholders with the right shape, not finished work:

- **Events, on the browser rig only.** `EaglercraftEnv` still returns empty
  `event_ids` and a constant health: nothing in the browser exposes inventory
  changes or block breaks, and the game is TeaVM-compiled Java with no hook to
  attach to, so these are left empty rather than faked. Pixel reading
  (health/hunger/hotbar sit at fixed positions, F3 gives x/y/z) is the realistic
  route. **On `MineRLEnv` this is solved** — see [wsl-minerl.md](wsl-minerl.md).
- **Seeding the concept embeddings.** `MineRLEnv` now mints and persists the
  id → phrase table (`ConceptVocab`, `vocab.id_to_phrase()`), so
  `concept_vocab_path` has something real to point at. What is still missing is
  the call site: nothing yet feeds those phrases through the LLM's embedding
  layer into `EventEmbedding.init_from_text` at model construction. Until that
  is wired, concept ids are learned from scratch and `L_align` carries no
  pretrained prior — it is no longer empty, but it is not yet grounded.
- **Goal supervision is wired but unexercised.** `goal_loss` now takes hindsight
  targets — `hindsight_goals()` embeds whatever the agent achieved in the next
  8 steps and treats that as what it had been aiming for, so no goal labels have
  to be authored. It still falls back to the L2 regulariser when targets are
  absent, which is the honest stage-1 behaviour but is *not* an objective: a run
  that never supplies targets is not teaching the goal head to mean anything.
- **Value targets.** `discounted_returns` uses intrinsic reward with no
  bootstrapping and no λ-returns.
- **Imagined RL.** `imagine()` produces the trajectory; nothing trains the actor
  on it yet.

## Staged schedule (from the readme)

1. **Grounding** — LLM frozen. Train the resampler, action tokenizer and
   world/event heads until game state activates the right concepts.
2. **Top layers** — unfreeze the top half, add persistent memory and imagined RL.
3. **Joint** — everything at a small LR, with prior-retention KL against the
   frozen pretrained model so Minecraft does not wash the language knowledge out.

`--stage N` switches; `configure_stage` in `trainer.py` is the whole difference.
