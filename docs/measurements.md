# Measurements, and the failures that were silent

Everything here was measured on one machine — RTX 3060 12GB, 16GB RAM, WSL2
Ubuntu 26.04 — rather than estimated. The second half is the more useful half:
a list of things that broke without raising, without a bad loss curve, and
without any other symptom until something much later went wrong.

## Hardware and throughput

| | |
| --- | --- |
| GPU | RTX 3060 12GB, driver 595.95, CUDA 13.2 |
| WSL2 | 2.7.11, WSLg 1.0.73, kernel 6.18.33 |
| WSL allocation | 10GB RAM, 12 cores, 8GB swap (`~/.wslconfig`) |
| Torch | 2.8.0+cu128, `cuda True` |

| Config | s/step | Peak VRAM | Windows/s |
| --- | --- | --- | --- |
| batch 1 × seq 16 | 3.3 | 6.2 GiB | — |
| batch 2 × seq 16 | 8.7 | 10.5 GiB | — |
| batch 2 × seq 8 | 1.14 | 7.6 GiB | 1.76 |

At batch 2 × seq 8 one pass over the 151,517 labelled windows takes about
**24 hours**. Batch 2 is superlinearly slower than batch 1, which is a memory
wall rather than a compute one — the per-step forward is only 83 tokens wide, so
the GPU is under-fed and the constraint is activations.

Sizing a larger backbone was attempted and **failed on a DNS error** before it
measured anything. That question is open.

## Rendering: leave it on llvmpipe

MineStudio speed test, 100 steps, runs interleaved to control for ordering:

| Renderer | FPS |
| --- | --- |
| Mesa llvmpipe (CPU, the WSLg default) | **42.3 / 44.3** |
| Mesa d3d12 (RTX 3060 via GPU-PV) | 31.7 / 27.0 / 29.1 |

`GALLIUM_DRIVER=d3d12` really does reach the GPU — `glxinfo` reports
`D3D12 (NVIDIA GeForce RTX 3060)`, OpenGL 4.6 — and is still slower. MineRL
reads every frame back to the CPU each tick; at 640×360 that readback across
GPU-PV costs more than the GPU saves. Leaving it off also keeps the 3060 for
torch.

MineStudio's documented GPU path, `MINESTUDIO_GPU_RENDER=1`, cannot work in WSL
at all: it resolves `/dev/dri/by-path/pci-*-card`, and WSL exposes no `/dev/dri`,
only `/dev/dxg`. VirtualGL is a dead end for the same reason.

## The dataset

`CraftJarvis/minestudio-data-6xx-v110`, 248GB total. `segmentation/` (61GB) is
for segmentation-conditioned policies and has no input path in this model, so
187GB is enough.

| Tree | Size | Contents |
| --- | --- | --- |
| `image` | 144G | **MP4**, 32 frames per chunk, already 224×224 |
| `meta_info` | 13G | inventory, `isGuiOpen`, `cursor_x/y`, position, per-frame |
| `event` | 12G | an **index**: `(event name, n) → (episode, frame, count)` |
| `action` | 6.6G | 21 keys, per frame |

3,658 episodes exist in both `image` and `action`; 27,267,072 usable frames =
378.7 hours. **Join on episode name, not shard position** — the trees are sharded
differently (8 image shards vs 4 action shards), and 15 episodes disagree about
their own length, one by 9×.

MP4 decode runs at **1,789 frames/s single-threaded**, so a 16-step window costs
72ms against 3,300ms of GPU. Decoding is not the bottleneck.

## The action space

`ActionConfig.buttons` is MineRL's `SIMPLE_KEYBOARD_ACTION` verbatim: the names
are the wire format.

```
forward  back  left  right  jump  sneak  sprint  attack  use  pickItem  inventory
```

- **No craft, place or equip action exists.** `HumanSurvival` exposes keyboard +
  camera + chat only. Recipes have to be clicked out in the GUI, VPT-style.
- **`drop` and `swapHands` are commented out** in MineRL and cannot be pressed.
- **`pickItem` is absent from the contractor recordings**, so offline training
  gives it no supervision at all.
- **`inventory` toggles and is edge-triggered.** Holding it for all eight ticks
  of a chunk flips the GUI exactly once, same as pressing it on one tick.

Press rates over 128,000 contractor frames:

| Button | Pressed | | Button | Pressed |
| --- | --- | --- | --- | --- |
| forward | 43.6% | | use | 6.1% |
| sneak | 13.7% | | sprint | 4.3% |
| attack | 12.6% | | back | 1.8% |
| jump | 10.0% | | hotbar (any) | 1.13% |
| left / right | 4.8 / 5.2% | | **inventory** | **0.52%** |

## Two reference values

**Actor loss 2.4719 means "learned nothing".** That is the loss of a predictor
that reproduces the marginal action distribution, computed from the real label
statistics. Its breakdown also says where the gradient goes:

| Component | Floor | Share of actor loss |
| --- | --- | --- |
| camera | 2.1399 | **86.6%** |
| buttons (11, averaged) | 0.2467 | 10.0% |
| hotbar | 0.0853 | 3.5% |

`inventory` alone carries 1.3% of the actor gradient, which is why crafting is
not reachable from behaviour cloning at this data scale.

**Human camera has lag-1 autocorrelation 0.78.** Measured over 96,000 frames:

| | yaw | pitch |
| --- | --- | --- |
| human | **+0.782** | **+0.757** |
| independent per-tick sampling | −0.002 | +0.005 |
| frames completely still | 46.1% | 52.4% |

The baseline is the same values shuffled — identical marginals, no time
structure. That is exactly what the current `ActionHead.sample` produces, since
all eight ticks are drawn independently from one hidden state. A policy can match
the marginal distribution perfectly and still visibly shake.

## Camera binning, chosen against the data

Half of all camera values are exactly 0, and half the nonzero ones are under
1.2°. The number that matters is how much genuine motion collapses into the
no-motion bin, because a turn quantised to zero is a label saying the human held
still.

| Bins / range / mu | Real turns lost | Round-trip MAE |
| --- | --- | --- |
| 11 / ±15° / 8 (the original) | **36.27%** | 0.350° |
| 11 / ±10° / 10 (VPT's setting) | 28.55% | 0.402° |
| **21 / ±20° / 40 (chosen)** | **1.78%** | 0.208° |

A large turn clipped at the outermost bin still says "turn a lot"; a small turn
rounded to zero says "hold still". The two errors are not equivalent, so mu is
aggressive and the range is wide rather than tight.

## Instruction labels, derived from actions

A window is labelled only when one behaviour dominates it; ambiguous windows are
dropped. 378 hours are available and a week of training sees ~1%, so selectivity
is free and a noisy instruction is worse than no window.

Window length trades against rare instructions:

| Instruction | seq 16 (6.4s) | **seq 8 (3.2s)** | seq 2 (0.8s) |
| --- | --- | --- | --- |
| move forward | 40,951 | **77,910** | 347,475 |
| mine | 19,246 | **44,183** | 198,962 |
| turn left / right | 7.7k / 6.7k | **11.4k / 12.2k** | 40k / 37k |
| jump | 1,475 | **2,362** | 15,871 |
| move backward | 561 | **1,868** | 9,069 |
| strafe left | **70** | **814** | 7,935 |

`seq_len=8` was chosen: long enough to show a behaviour, and every instruction
clears 800 windows. Rare classes are handled by **balanced sampling, not loss
reweighting** — reweighting distorts the loss surface, sampling only changes what
the model looks at.

Total: **151,517 labelled windows** out of 423,316 candidates (35.8%).

---

# Failures that were silent

Each of these ran without an exception and without a suspicious loss curve.

**Event counters are running totals, not per-step deltas.** Swinging at a tree,
`mine_block` sat at `{'spruce_leaves': 1}` for twelve consecutive steps, went to
2, then held at 2 for sixty more. Used raw, every event re-fires forever and the
grounding loss is trained on noise. `MineRLEnv` diffs against the previous chunk.

*And the contractor data is the opposite*: `meta_info[...]["events"]` is per-frame.
The same code cannot treat both the same way.

**MineRL's camera vector is `[pitch, yaw]`; `ActionChunk` bins are `(yaw, pitch)`.**
Measured: `camera=[10,0]` moved pitch 30° over three ticks and left yaw at 0;
`camera=[0,10]` did the reverse. Without the swap, every learned turn becomes a
look — and nothing in the loss would say so.

**A bfloat16 tensor that only breaks under a real policy.** `ActionChunk` from
`ActionHead.sample()` carries the backbone's dtype, and numpy has no bfloat16.
Hand-built test chunks are float32, so the adapter passed a 100-chunk dry run and
then died the first time a trained model drove it. The dry run's conclusion was
true but narrower than it appeared. There is now a test that builds the chunk in
bfloat16.

**A dead thread that keeps serving.** The browser server's sim loop was a daemon
thread with no `try`; the first exception killed it while HTTP kept returning the
last snapshot. The page looked alive and simply stopped responding. Failures now
hand control back and surface the error on the page.

**Frame/action alignment off by one chunk.** Timestep *t* must observe the frame
at the *start* of its chunk and predict the actions that follow. Shifted by one,
the model is shown the consequences of the actions it is asked to output; the
loss falls *faster* and the policy is useless. Verified by cross-correlating
consecutive frames to recover the turn direction from the image and comparing it
against the yaw in each chunk: 77.8% agreed with the current-step pairing.

**`Batch.to(device)` listed its fields by hand**, so any field added later was
dropped in silence — present on CPU, `None` on GPU. It now walks the dataclass.

**`set -u` versus conda's openjdk hook.** `openjdk_activate.sh` dereferences an
unset `$target_platform`, so a script with `set -euo pipefail` aborts right
before Minecraft starts. This cost two separate debugging sessions in one day.

**`pgrep -f` matching its own command line.** A wait loop whose pattern appears
in its own `bash -lc` string never exits; `pkill -f` with the same problem kills
the shell issuing it. Both happened.

**Processes dying with their launcher.** WSL terminates a session's processes
when `wsl.exe` exits, so a server started as a background task disappears when
that task ends. `setsid` is required for anything meant to outlive the command
that started it.

**MineStudio's own gaps**, for what they are worth: `psutil` is imported but not
declared as a dependency, and `MinecraftGUI` calls `TextLayout.update(x=, y=)`,
a pyglet 2.x signature, against the pyglet 1.4 it pins. Both are patched around
rather than fixed upstream — the GUI patch lives in site-packages and will not
survive a reinstall.
