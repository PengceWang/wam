"""把几个 checkpoint 放进**同一个世界**跑同样的步数，并排录成一段视频。

    python scripts/compare_video.py --ckpts stage1b_cl online_lh_u25 online_lh_u75 \
        --labels BC U25 U75 --seeds 300 301 --steps 300

数字判据（原木事件）在这个量级上噪声很大 —— BC 起点 900 步只砍到 2 次，
6 局中 5 局为零。所以肉眼看行为是有价值的补充：站着挖坑、绕着树转、还是真的
对着树干砍，这些数字表里看不出来。

**公平性靠三件事**，缺一不可：

* **同一个种子** —— 不同世界的树密度差好几倍，换个种子就能造出任意结论
* **同一批随机数** —— 策略是按 temperature=1.0 采样的，同一个模型跑两遍都不一样。
  每个 (checkpoint, seed) 组合前都把 torch 的种子重置成同一个值
* **同样的上下文长度** —— 每 150 步清一次 KV cache，和训练时一致。这一条是硬要求：
  实测每 8 步清一次会让原木事件直接归零（docs/online-rl-log.md §3），
  上下文长度不一样等于在比两个不同的策略

一个模型步 = 8 tick = 0.4 秒，所以原始帧率是 2.5 fps。默认按 5 fps 写出去，
也就是 2 倍速 —— 2.5 fps 看着太顿，再快就看不清在按什么了。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def label_frame(rgb: np.ndarray, lines: list[str], scale: int = 2) -> np.ndarray:
    """放大并在左上角压一块半透明底，写上标签/步数/事件数。"""
    import cv2

    img = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    h = 16 * len(lines) + 8
    band = img[:h].astype(np.float32) * 0.35
    img[:h] = band.astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_one(ckpt: Path, seed: int, steps: int, reset: int, rng_seed: int, dev):
    """跑一个 checkpoint，返回 (帧列表, 每步累计原木数, 每步累计挖掘数)。"""
    from wam.config import WAMConfig
    from wam.envs.minerl import MineRLEnv
    from wam.model.wam import WorldActionModel

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = WAMConfig.from_dict(blob["config"])
    model = WorldActionModel(cfg).to(dev)
    model.load_state_dict(blob["model"])
    model.eval()

    env = MineRLEnv(cfg, seed=seed, vocab_path=None)
    obs = env.reset()
    st = model.initial_state(1, device=dev)
    torch.manual_seed(rng_seed)               # 同一批随机数，见模块说明

    frames, logs, mines = [], [], []
    n_log = n_mine = 0
    with torch.no_grad():
        for t in range(steps):
            if reset and t % reset == 0:
                st = type(st)(memory=st.memory, prev_action=st.prev_action, goal=st.goal,
                              goal_is_external=st.goal_is_external, cache=None)
            frames.append(obs.rgb.copy())
            px = torch.from_numpy(obs.rgb.astype(np.float32) / 255.0)
            px = px.permute(2, 0, 1).unsqueeze(0).to(dev)
            ev = torch.from_numpy(obs.event_ids).long().unsqueeze(0).to(dev)
            out, st = model.step(st, px, ev)
            a = model.action_head.sample(out.readout, temperature=1.0)
            st = type(st)(memory=st.memory, prev_action=a, goal=st.goal,
                          goal_is_external=st.goal_is_external, cache=st.cache)
            obs = env.step(a)
            for part in (obs.event_text or "").split(", "):
                if not part:
                    continue
                if "log" in part:
                    n_log += 1
                if part.startswith("mined"):
                    n_mine += 1
            logs.append(n_log)
            mines.append(n_mine)
    env.close()
    del model
    torch.cuda.empty_cache()
    return frames, logs, mines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint 名（不含 .pt）")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[300])
    ap.add_argument("--steps", type=int, default=300, help="模型步，一步 0.4 秒")
    ap.add_argument("--cache-reset", type=int, default=150,
                    help="每这么多步清一次 KV cache。默认 150 = 训练的 horizon。")
    ap.add_argument("--ckpt-dir", default="/data/wam/checkpoints")
    ap.add_argument("--out-dir", default="/home/pengce/MineWorld/wam/video")
    ap.add_argument("--fps", type=int, default=5, help="2.5 fps 是真实速度，默认 2 倍速")
    ap.add_argument("--rng-seed", type=int, default=1234)
    args = ap.parse_args()

    assert len(args.ckpts) == len(args.labels), "--ckpts 和 --labels 数量要一致"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    for seed in args.seeds:
        panels, summary = [], []
        for name, label in zip(args.ckpts, args.labels):
            ck = Path(args.ckpt_dir) / f"{name}.pt"
            print(f"[seed {seed}] {label} ({ck.name}) ...", flush=True)
            fr, logs, mines = run_one(ck, seed, args.steps, args.cache_reset,
                                      args.rng_seed, dev)
            panels.append([
                label_frame(f, [f"{label}  seed {seed}",
                                f"step {i+1}/{args.steps}  ({(i+1)*0.4:.0f}s)",
                                f"log {logs[i]}   mined {mines[i]}"])
                for i, f in enumerate(fr)
            ])
            summary.append(f"{label}: 原木 {logs[-1]}, 挖掘 {mines[-1]}")
            print(f"  -> {summary[-1]}", flush=True)

        # 并排：高度相同，横向拼接，中间留一条分隔线
        n = min(len(p) for p in panels)
        h, w = panels[0][0].shape[:2]
        sep = np.full((h, 4, 3), 60, np.uint8)
        merged = []
        for i in range(n):
            row = []
            for j, p in enumerate(panels):
                if j:
                    row.append(sep)
                row.append(p[i])
            merged.append(np.concatenate(row, axis=1))

        out = out_dir / f"compare_seed{seed}.mp4"
        # 直接喂给 ffmpeg —— imageio-ffmpeg 没装，但系统 ffmpeg 在
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{merged[0].shape[1]}x{merged[0].shape[0]}", "-r", str(args.fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            stdin=subprocess.PIPE)
        for f in merged:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        print(f"写出 {out}  ({n} 帧 @ {args.fps}fps = {n/args.fps:.0f}s)")
        print("  " + " | ".join(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
