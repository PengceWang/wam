"""并行评测 —— 因为串行评测的样本量根本不够，之前四轮结论全在噪声里。

    python scripts/eval_parallel.py --ckpt /data/wam/checkpoints/stage1b_cl.pt

**为什么要重做评测。** 旧的 ``eval_online.py`` 是 6 局 × 150 步 = 900 步。
BC 起点在这 900 步里一共砍到 **2 次**原木，6 局中 5 局是零。而在线 RL 的四轮
比较（0.833 / 0.667 / 0.500 / 0.000）换算成事件数是 5 / 4 / 3 / 0 次 ——
这个量级上任何差异都是泊松噪声，**四轮"RL 把砍树变差了"的结论都不成立**。

这里用和训练同一套并行 worker：12 个环境 × 500 步 = 6000 步，样本量 ×6.7。
同时分开统计三类事件，因为诊断发现了一个关键事实：

    挖到: grass×32, oak leaves×26, spruce leaves×11, jungle leaves×5 ...  原木 ×0

四种树叶天天挖、原木一次没有 —— **agent 一直站在树旁边，只是砍在树叶上而不是
树干上**。所以 leaves 是"到了树跟前"的可靠代理指标，把它和 log 分开报，
才能区分"找不到树"和"找到了但瞄错地方"这两种完全不同的失败。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_online import _boot  # noqa: E402  复用同一套 worker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=300, help="留出的评测种子，训练用 0..N")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache-reset", type=int, default=8,
                    help="每这么多步清一次 KV cache，0 = 永不清。"
                         "注意实际上下文是**锯齿形**的：清完只剩 1 个时间步，"
                         "涨到 N 再归零，均值 (N+1)/2。")
    ap.add_argument("--keep-cache", action="store_true",
                    help="不清 KV cache，让上下文一路长满。**这是分布外推理** —— "
                         "BC 的 seq_len=8，模型没见过更长的上下文。旧的串行评测"
                         "无意中一直是这个模式（900 步出 2 次原木），"
                         "而分布内评测 6000 步出 0 次，差异 p≈2e-6。加这个开关"
                         "就是为了把\"上下文长度\"这一个变量单独摘出来验。"
                         "注意显存：150 步 × 12 环境的 cache 约 16 GB。")
    args = ap.parse_args()

    from wam.config import WAMConfig
    from wam.model.wam import WorldActionModel

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = WAMConfig.from_dict(blob["config"])

    import multiprocessing as mp

    ctx = mp.get_context("spawn")            # 环境必须在碰 CUDA 之前起
    N = args.envs
    conns, procs = [], []
    for i in range(N):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_boot, args=(child, args.seed0 + i, cfg.to_dict()), daemon=True)
        p.start()
        conns.append(parent)
        procs.append(p)
    frames = [c.recv() for c in conns]
    if any(isinstance(f[0], str) for f in frames):
        print("环境启动失败", flush=True)
        return 1

    dev = torch.device("cuda")
    model = WorldActionModel(cfg).to(dev)
    model.load_state_dict(blob["model"])
    model.eval()

    shape0 = frames[0][0].shape
    last_good = list(frames)
    counts: dict[str, int] = {}
    st = model.initial_state(N, device=dev)
    t0 = time.time()

    with torch.no_grad():
        for t in range(args.steps):
            # cache 每 8 步清一次，和训练时一致 —— 上下文长度不同会让两边不可比
            R = 0 if args.keep_cache else args.cache_reset
            if R and t % R == 0:
                st = type(st)(memory=st.memory, prev_action=st.prev_action, goal=st.goal,
                              goal_is_external=st.goal_is_external, cache=None)
            px = torch.from_numpy(np.stack([f[0] for f in frames]).astype(np.float32) / 255.0)
            px = px.permute(0, 3, 1, 2).to(dev)
            ev = torch.from_numpy(np.stack([f[1] for f in frames])).long().to(dev)
            out, st = model.step(st, px, ev)
            a = model.action_head.sample(out.readout, temperature=1.0)
            st = type(st)(memory=st.memory, prev_action=a, goal=st.goal,
                          goal_is_external=st.goal_is_external, cache=st.cache)

            b = a.buttons.float().cpu().numpy()
            c = a.camera.long().cpu().numpy()
            h = a.hotbar.long().cpu().numpy()
            dead = []
            for i, conn in enumerate(conns):
                try:
                    conn.send((b[i], c[i], h[i]))
                except (BrokenPipeError, OSError):
                    dead.append(i)
            new = []
            for i, conn in enumerate(conns):
                if i in dead:
                    new.append(last_good[i])
                    continue
                try:
                    f = conn.recv()
                except (EOFError, OSError):
                    f = last_good[i]
                if isinstance(f[0], np.ndarray) and f[0].shape == shape0:
                    last_good[i] = f
                else:
                    f = last_good[i]
                new.append(f)
            frames = new
            for f in frames:
                for part in (f[2] or "").split(", "):
                    if part:
                        counts[part] = counts.get(part, 0) + 1

    for c in conns:
        try:
            c.send(None)
        except Exception:
            pass

    total = N * args.steps
    mine = sum(v for k, v in counts.items() if k.startswith("mined"))
    log = sum(v for k, v in counts.items() if "log" in k)
    leaf = sum(v for k, v in counts.items() if "leaves" in k)
    per = lambda n: n / total * 1000                                   # noqa: E731

    print(f"\n=== {args.tag or Path(args.ckpt).stem} | {total} 步 "
          f"({N}×{args.steps}, 种子 {args.seed0}-{args.seed0+N-1}, "
          f"{time.time()-t0:.0f}s) ===", flush=True)
    print(f"EVAL {args.tag or Path(args.ckpt).stem} steps={total} "
          f"reset={0 if args.keep_cache else args.cache_reset} "
          f"mine={mine} log={log} leaf={leaf} "
          f"mine_k={per(mine):.2f} log_k={per(log):.2f} leaf_k={per(leaf):.2f}", flush=True)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
    print("  最常见事件: " + ", ".join(f"{k}×{v}" for k, v in top), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
