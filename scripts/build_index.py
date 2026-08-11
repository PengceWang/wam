"""扫 contractor 全量，切窗口、反推指令、落一个索引 npz。

    python scripts/build_index.py --root /data/wam/6xx --out /data/wam/6xx_index_seq8.npz

索引固定窗口布局，训练时由它（而不是 YAML）决定 seq_len —— 手工保持两者同步，
迟早会在窗口边界和标签对不上的数据上训练。
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.data.contractor import (  # noqa: E402
    CHUNK_FRAMES, INSTRUCTIONS, WINDOW_FRAMES, episode_table, label_window,
    n_windows, read_actions, window_features,
)


def do_episode(job):
    ep, img_shard, img_idx, act_shard, act_idx, n_frames, window = job
    nw = n_windows(n_frames, window)
    if nw == 0:
        return []
    try:
        act = read_actions(act_shard, act_idx, 0, nw * window)
    except Exception:
        return []
    if len(act["forward"]) < nw * window:
        return []
    rows = []
    for w in range(nw):
        s = slice(w * window, (w + 1) * window)
        f = window_features({k: v[s] for k, v in act.items()})
        rows.append((ep, img_shard, img_idx, act_shard, act_idx, w * window, label_window(f)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/wam/6xx")
    ap.add_argument("--out", default="/data/wam/6xx_index_seq8.npz")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    window = args.seq_len * args.chunk_size
    if window != WINDOW_FRAMES:
        print(f"注意：窗口 {window} 帧，contractor.py 的常量是 {WINDOW_FRAMES}")

    table = episode_table(args.root)
    print(f"{len(table):,} 个 episode（image ∩ action）")

    jobs = [(ep, v["image"].shard, v["image"].episode_idx, v["action"].shard,
             v["action"].episode_idx, v["image"].n_frames, window)
            for ep, v in table.items()]

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(do_episode, jobs, chunksize=8):
            rows.extend(out)
            done += 1
            if done % 500 == 0:
                print(f"  {done:,}/{len(jobs):,} episode, {len(rows):,} 窗口", flush=True)

    print(f"\n候选窗口 {len(rows):,}  (文档 423,316)")
    label = np.array([r[6] for r in rows], dtype=np.int8)
    n_lab = int((label >= 0).sum())
    print(f"标注窗口 {n_lab:,}  ({100 * n_lab / max(1, len(rows)):.1f}%)   文档 151,517 (35.8%)")
    print()
    print(f"{'指令':<26} {'本次':>9} {'文档':>9} {'差':>9}")
    print("-" * 56)
    DOC = {"move forward": 77910, "mine the block in front": 44183, "turn right": 12200,
           "turn left": 11400, "jump": 2362, "move backward": 1868, "strafe left": 814,
           "strafe right": 780}
    for i, name in enumerate(INSTRUCTIONS):
        c = int((label == i).sum())
        d = DOC.get(name)
        print(f"{name:<26} {c:>9,} {d:>9,} {c - d:>+9,}" if d else f"{name:<26} {c:>9,}")

    np.savez_compressed(
        args.out,
        seq_len=np.int64(args.seq_len),
        chunk_size=np.int64(args.chunk_size),
        episode=np.array([r[0] for r in rows]),
        img_shard=np.array([r[1] for r in rows]),
        img_idx=np.array([r[2] for r in rows], dtype=np.int32),
        act_shard=np.array([r[3] for r in rows]),
        act_idx=np.array([r[4] for r in rows], dtype=np.int32),
        start=np.array([r[5] for r in rows], dtype=np.int64),
        label=label,
        instructions=np.array(INSTRUCTIONS),
    )
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
