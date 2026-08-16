# WAM — a pretrained LLM as the trunk of a Minecraft agent

A ~0.8B Vision-Language-Action model where a pretrained language model is not a
planner calling out to other components, but the **shared trunk**: vision,
action, world prediction, value, goals and memory all enter the same transformer
as tokens and all read out of the same hidden state.

The design this implements is in [docs/design.md](docs/design.md). What is
actually built, running, and measured is below — the two are not the same thing,
and the gap is stated rather than blurred.

```
              ┌──────────── one timestep, every 0.4 s ────────────┐
  Minecraft ──┤ 1 frame 224²  ·  6 HUD scalars  ·  event ids      │──┐
              └───────────────────────────────────────────────────┘  ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ x_t = 83 tokens                                                       │
   │ memory 40 │ drive 8 │ goal 4 │ visual 16 │ prev_act 4 │ evt 8 │ st 2 │ rd 1
   └───────────────────────────────────────────────────────────────────────┘
                                    ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ Qwen3-0.6B · 28 layers · d=1024 · frozen in stage 1 · growing KV cache │
   └───────────────────────────────────────────────────────────────────────┘
                                    ▼  hidden state at the readout token
     Action · World · Value · Goal · Drive · Language
        │
        ▼  8 ticks of keyboard and mouse
```

Full structure, including which parts are dormant:
[docs/model-io.md](docs/model-io.md).

## Status

| | |
| --- | --- |
| Environment | MineRL / MCP-Reborn via MineStudio, on WSL2 |
| Data | VPT contractor `6xx`, 3,658 episodes, 27.3M frames, 378 hours |
| Stage 1a | instruction → action, trained 3,000 steps (≈4% of the labelled set) |
| Instruction following | **demonstrated**: `forward` held 89% of ticks under "move forward" vs 4% under "turn right" |
| Camera control | poor — drifts 157° over 12s of "walk forward" |
| Recovery from stuck states | absent; the model walks into obstacles and stays there |
| Stages 2–3 | not started |

The stage-1a question — *does an instruction in the goal slot reach the action
head?* — is answered yes. Everything downstream of that is early.

## What online RL actually did

The most load-bearing results in this repo are negative ones, so they are stated
before anything else. Full record: [docs/online-rl-log.md](docs/online-rl-log.md).

**Context length is a capability floor, not a memory knob.** Same weights, same
seeds, only the KV-cache reset interval changed:

| reset every | 8 | 16 | 32 | 64 | never |
| --- | --- | --- | --- | --- | --- |
| logs / 1000 steps | **0** | **0** | **0** | 1.11 | **5.00** |

Fast blocks (grass, dirt, leaves — one step each) are unaffected. Only logs
(~3 s ≈ 7.5 model steps) go to zero. An agent whose context is cut every 8 steps
**cannot finish chopping a tree**, and PPO was updating in 8-step segments.
Note that BC trained at `seq_len=8`, so its training windows almost never
contained a complete chop either.

**Single-task RL changed behaviour, but not the target behaviour.** 6000 steps
per checkpoint on held-out seeds:

| checkpoint | mined | logs | **logs / mined** |
| --- | --- | --- | --- |
| BC start | 198 | 12 | **6.1%** |
| U25 | 373 | 24 | **6.4%** |
| U100 | 356 | 18 | **5.1%** |

Mining doubled — 7.3σ, p < 1e-12, the online loop demonstrably works. But the
**ratio never moved**. The extra activity was `mined dirt`, `mined grass block`.
Reward went only to logs; what was learned was "hold attack". Under a single
task every rewarded trajectory shares one common component, and diffuse credit
(γ=0.999, λ=0.98 over 150 steps) converges to exactly that component.

That is what [docs/multitask-design.md](docs/multitask-design.md) is built to fix.

**Judge by ratios, not counts.** An earlier round of four experiments was
concluded on evaluations of 900 steps in which the BC baseline scored **2 log
events total** (`0,0,2,0,0,0` across six episodes). Those conclusions were
Poisson noise and have been withdrawn. Evaluation is now 12 envs × 500 steps and
reports absolute counts so that "2 in total" cannot hide inside a mean.

## Install

Linux only for the simulator (JDK 8 + MCP-Reborn). On Windows this means WSL2;
the full walkthrough, including the parts where MineStudio's own instructions do
not work, is in [docs/wsl-minerl.md](docs/wsl-minerl.md).

```bash
conda create -n minestudio python=3.10 -y && conda activate minestudio
conda install --channel=conda-forge openjdk=8 -y
pip install MineStudio psutil          # psutil is required but undeclared
python -m minestudio.simulator.entry -y
```

Then this package:

```bash
uv sync --extra dev
```

## Run

```bash
# no downloads, no Minecraft: shapes, losses, a rollout, imagination
python scripts/smoke_test.py

# play it yourself in a browser, or hand an instruction to the model
python scripts/play_server.py --checkpoint ~/stage1a_run1.pt   # → localhost:8080

# multi-task goal-conditioned online RL (2x H100: one run per card, no DDP)
CUDA_VISIBLE_DEVICES=0 python scripts/train_multitask.py \
    --config configs/h100-multitask.yaml --init-from CKPT --envs 16 --tag A

# evaluate: 12 envs x 500 steps, absolute counts, ratio is the criterion
python scripts/eval_parallel.py --ckpt CKPT --envs 12 --steps 500 --seed0 300

# side-by-side video of two checkpoints in the same world, same RNG
python scripts/compare_video.py --ckpts a b --labels A B --seeds 300

# stage 1a
python scripts/train_stage1a.py --overfit 4 --max-steps 80     # sanity first
python scripts/train_stage1a.py --max-steps 3000 --wandb wam-stage1a

# does it obey, and does it move like a person?
python scripts/eval_stage1a.py --checkpoint ~/stage1a_run1.pt
```

The overfit run is not optional. It takes minutes and catches the failures that
otherwise surface after a day of training.

## Layout

| Path | What it is |
| --- | --- |
| `wam/config.py` | every knob, one dataclass tree, loadable from YAML |
| `wam/model/wam.py` | assembles `x_t`, runs the rollout, does imagination |
| `wam/model/vision.py` | frozen tower + Perceiver resampler → 16 tokens/frame |
| `wam/model/action.py` | factorized action space, chunk encoder, chunk head |
| `wam/model/goal.py` | text → goal tokens; the "no goal yet" default |
| `wam/model/memory.py` | persistent memory, `M_{t+1} = f(M_t, H_t)` |
| `wam/model/heads.py` | world / value / goal / drive / language heads |
| `wam/envs/minerl.py` | **the seam to Minecraft** — MineStudio behind a chunked protocol |
| `wam/envs/eaglercraft.py` | the older browser rig: sees the game, cannot read it |
| `wam/data/contractor.py` | contractor LMDB → windows, instructions derived from actions |
| `wam/data/stage1a.py` | balanced sampling, goal token table |
| `wam/training/losses.py` | `L = Σ λ_i L_i` |
| `wam/training/online.py` | PPO in the live env; long-horizon GAE, cache detach vs free |
| `wam/training/multitask.py` | tasks, base-rate-normalised reward, hindsight relabel, SIL |
| `wam/training/eval_stage1a.py` | obedience and human-likeness, both as numbers |

## Two environments, deliberately

| | `EaglercraftEnv` | `MineRLEnv` |
| --- | --- | --- |
| Platform | Windows | WSL2 / Linux |
| How frames arrive | screen capture of a browser | simulator API |
| Symbols | **none** | inventory, events, biome, GUI state |

The browser rig came first and can see the game but not read it, so `event_ids`
is always empty and the grounding loss has nothing to align against. That is why
the MineRL adapter exists. Both are kept; both are lazily imported so a checkout
on either OS still imports `wam.envs`.

## What the numbers mean

Two reference values are worth knowing before reading any training curve:

- **actor loss 2.4719** — what a model scores by reproducing the marginal action
  distribution and nothing else. At or above this, it has learned nothing. It is
  logged to wandb as `actor_trivial_floor` so the chart carries its own baseline.
- **camera lag-1 autocorrelation 0.78** — human mouse-look, measured. Independent
  per-tick sampling gives 0.00 with an identical marginal distribution, which is
  what "moves like a bot" means numerically.

Everything else measured on this hardware is in
[docs/measurements.md](docs/measurements.md), along with the failures that were
silent — a swapped camera axis, undifferenced event counters, a bfloat16 tensor
that only breaks once a real policy drives the env. Those cost the most time and
none of them show up in a loss curve.

## Not done

- **`L_align` is off.** The loss that ties vision to the LLM's semantics needs
  the `event` and `meta_info` trees, which stage 1a does not load. Until it runs,
  the project's central claim is untested.
- **Language is doing no work.** The LLM is frozen, no text is generated, and
  instructions enter as eight fixed strings. Eight random vectors would probably
  work as well; that experiment has not been run.
- **Context ends at ~3.3 minutes.** 83 tokens/step × 2.5 steps/s against 40,960
  positions, with no eviction anywhere. Minecraft progress is measured in hours.
- **Crafting is unreachable from behaviour cloning.** `inventory` is pressed in
  0.52% of contractor frames and carries ~1.3% of the actor gradient.
- **Model sizing is still unknown.** The VRAM sweep for larger backbones failed
  on a DNS error before it measured anything. A controlled ablation is running:
  `seq_len 8 → 32` on one card, `Qwen3-0.6B → 1.7B` on the other, each a
  single-variable change against `stage1b_cl`, judged by logs/mined.
- **20% of the model trains.** 596M of 865M parameters are the frozen LLM
  (`n_trainable_top_layers: 0`). Before adding parameters, unfreeze the ones
  that are already there — `configs/h100-multitask.yaml` sets 14.
- **Goals are not used yet.** Every run so far passed the null goal token, so
  the agent free-runs and the reward exists only in the weights. Goal
  conditioning has failed three times under BC (no search in BC); the
  multi-task RL design attacks it from the other side, with a
  `goal_sensitivity` metric that says on update 1 whether the goal input is
  being read at all.

## Licence and provenance

Built on [MineStudio](https://github.com/CraftJarvis/MineStudio) (CraftJarvis)
and the VPT contractor recordings. Model weights are Qwen3 and SigLIP.
