# MineStudio on WSL2

The symbol-carrying environment. Windows can run the model but not the
simulator (JDK 8 + MCP-Reborn is Linux-only), so the game lives in WSL.

## Running it

```bash
wsl                       # Ubuntu 26.04, default user wangp
conda activate minestudio
python -m minestudio.simulator.entry -y      # smoke test, ~100 steps
```

The repo is reachable from inside WSL at
`/mnt/c/Users/wangp/OneDrive/Desktop/dreamer/MineStudio`; put it on `PYTHONPATH`
to use the adapter:

```bash
PYTHONPATH=/mnt/c/Users/wangp/OneDrive/Desktop/dreamer/MineStudio python your_script.py
```

## What is installed

| Piece | Version |
| --- | --- |
| WSL / WSLg | 2.7.11 / 1.0.73, Ubuntu 26.04 |
| Python / JDK | 3.10 (miniforge, env `minestudio`) / Zulu 1.8.0_472 |
| MineStudio | 1.1.6 (+ `psutil`, which it needs but does not declare) |
| torch | 2.8.0+cu128, `cuda True` on the RTX 3060 |

`MINESTUDIO_DIR=/home/wangp/.minestudio` is set in `.bashrc`. It matters:
MineStudio otherwise puts its 445MB engine under `tempfile.gettempdir()`, and
`/tmp` in WSL is tmpfs — the engine would sit in RAM and be re-downloaded on
every restart.

## Rendering: leave it on llvmpipe

Measured here, MineStudio speed test, 100 steps, runs interleaved to control for
ordering:

| Renderer | FPS |
| --- | --- |
| Mesa llvmpipe (CPU, the WSLg default) | **42.3 / 44.3** |
| Mesa d3d12 (RTX 3060 via GPU-PV) | 31.7 / 27.0 / 29.1 |

`GALLIUM_DRIVER=d3d12` does genuinely reach the GPU — `glxinfo` reports
`D3D12 (NVIDIA GeForce RTX 3060)`, OpenGL 4.6 — and is still the slower option.
MineRL reads every frame back to the CPU each tick; at 640x360 that readback
across GPU-PV costs more than the GPU saves. It also leaves the 3060 entirely to
torch, which is where it is actually needed.

MineStudio's documented GPU path, `MINESTUDIO_GPU_RENDER=1`, cannot work in WSL
at all: it resolves `/dev/dri/by-path/pci-*-card`, and WSL exposes no `/dev/dri`,
only `/dev/dxg`. VirtualGL is a dead end here for the same reason — the WSL
NVIDIA driver provides CUDA, not GLX.

## Three things the adapter had to measure

Each of these fails silently — no exception, no visible change in a loss curve.

**Event counters are running totals, not per-step deltas.** Swinging at a tree,
`mine_block` sat at `{'spruce_leaves': 1}` for twelve consecutive steps, went to
2, then held at 2 for sixty more. Used raw, every event re-fires forever.
`MineRLEnv` diffs against the previous chunk.

**MineRL's camera vector is `[pitch, yaw]`.** `ActionChunk.camera` bins are
`(yaw, pitch)`. Measured: `camera=[10,0]` moved pitch by 30° over three ticks and
left yaw at 0; `camera=[0,10]` did the reverse. Without the swap every learned
turn becomes a look.

**Mining and picking up are different events.** `mined grass block` leaves the
inventory unchanged; the `picked up dirt` a chunk later is what changes it. The
`inventory_changed` flag follows the inventory, not the block break.

## The action space, and what is not in it

`ActionConfig.buttons` is MineRL's `SIMPLE_KEYBOARD_ACTION` verbatim — the names
are the wire format, written straight into a MineRL tick:

| Channel | Shape | Semantics |
| --- | --- | --- |
| forward / back / left / right | 0/1 | held |
| jump / sneak / sprint | 0/1 | held |
| attack / use | 0/1 | held — mining needs a sustained press |
| pickItem | 0/1 | middle click, momentary |
| inventory | 0/1 | **toggles** the GUI |
| hotbar.1–9 | 0/1 | momentary |
| camera | `[pitch, yaw]` float | 11 bins per axis |

**There is no craft, place or equip action.** `HumanSurvival.create_actionables`
is keyboard + camera + chat and nothing else; `CraftAction`, `PlaceBlock` and
`EquipAction` live on `SimpleHumanEmbodimentEnvSpec`, which this env is not. So
recipes have to be clicked out in the GUI, VPT-style. While it is open the camera
moves a cursor rather than the head — `is_gui_open` in the observation is the
only thing that distinguishes them.

**`drop` and `swapHands` do not exist here.** Both are commented out in MineRL
(`mc.py`, `human_survival_specs.py`). Nothing can drop an item or use the
offhand; do not design around them.

**`chat` is deliberately not wired.** It reaches `/give` and `/tp`, which shorts
out any intrinsic-reward story.

`inventory` is edge-triggered: measured, holding it for all eight ticks of a
chunk toggles the GUI exactly once, the same as pressing it on one tick. The
action head needs no special case for it.

`voxels` and `mobs` look like actions in `no_op()` but are observation requests.

## What comes out

```
chunk  1  ids=[1]  'mined spruce leaves'  block_removed=True
chunk  8  ids=[2]  'mined grass block'    block_removed=True
chunk  9  ids=[3]  'picked up dirt'       inventory_changed=True
concept vocab: {'mined spruce leaves': 1, 'mined grass block': 2, 'picked up dirt': 3}
```

`ConceptVocab` is append-only and persisted. `EventEmbedding.init_from_text`
seeds row *i* from the LLM's embedding of phrase *i*, so an id that moves between
runs silently repoints a concept at another meaning. When the vocabulary fills
up, new concepts get `PAD` rather than an alias: losing a rare concept costs one
signal, aliasing teaches a wrong pairing.
