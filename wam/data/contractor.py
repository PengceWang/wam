"""VPT contractor LMDB -> 训练窗口，指令从动作反推。

.. warning::
   **这是重建，不是原件。** 原 ``wam/data/`` 从未进过 git —— ``.gitignore`` 的
   ``data/`` 规则不带前导斜杠，匹配任意层级的同名目录，把 ``wam/data/`` 整包吞掉了。
   本文件按 ``docs/measurements.md`` 记录的规格复原，并以其中的确切计数为验收标准。

已精确复现的部分（不是估计）：

* **窗口枚举**。每 episode ``(min(image_frames, action_frames) - 1) // 64`` 个
  不重叠窗口，合计 **423,316**，与文档的候选数差 0。末尾那 1 帧是刻意留的：
  timestep t 观察其 chunk 的**起始**帧并预测其后的动作，所以最后一个 chunk 之后
  还需要一帧。文档记过对齐错一个 chunk 的后果 —— 损失反而降得更快，策略却没用。
* **按 episode 名 join**。image 8 分片 / action 4 分片，分片位置不对应；action 多出
  6 个 episode；15 个 episode 对自己的长度说法不一（最大 9.1 倍），全部实测复现。

近似的部分：**指令标注阈值是反解的**。文档只给了各类的最终窗口数，没给规则本身。
下面的阈值由坐标下降拟合那些计数得到，逐类相对误差见 ``FIT_QUALITY``。
"""

from __future__ import annotations

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 一个窗口 = seq_len 个 timestep x chunk_size 个 tick。stage 1a 用 seq8 x chunk8。
WINDOW_FRAMES = 64
CHUNK_FRAMES = 32  # LMDB 里一个 MP4 chunk 的帧数（__chunk_size__）

# 顺序即 goal table 的行号，一旦确定就不能重排：EventEmbedding 那套同理，
# 行号漂移会把一条指令悄悄指向另一条的语义。
INSTRUCTIONS: tuple[str, ...] = (
    "move forward",
    "move backward",
    "strafe left",
    "strafe right",
    "jump",
    "mine the block in front",
    "turn left",
    "turn right",
)

# contractor 录制里的 21 个键。注意两处与 MineRL 动作空间的差异：
#   * `drop` 在这里有，但 MineRL 把它注释掉了 -> 无处可按，丢弃
#   * `pickItem` 在这里没有 -> 离线训练对它没有任何监督（文档明确记过）
CONTRACTOR_BUTTONS = ("forward", "back", "left", "right", "jump", "sneak",
                      "sprint", "attack", "use", "inventory")

# 拟合出的阈值。按钮是窗口内的按下帧占比，yaw 是窗口内累计转角（度）。
THRESHOLDS = {
    "forward": 0.95,
    "back": 0.8125,
    "left": 0.875,
    "right": 0.9375,
    "jump": 0.875,
    "attack": 0.875,
    "yaw": 180.0,
}

# 拟合质量，对照 measurements.md 的 seq8 列。留在代码里而不是提交信息里，
# 因为任何拿这套标签得出的结论都要连着这张表一起读。
FIT_QUALITY = {
    "move forward": (76028, 77910),
    "mine the block in front": (43479, 44183),
    "turn right": (10899, 12200),
    "turn left": (12448, 11400),
    "jump": (2387, 2362),
    "move backward": (1767, 1868),
    "strafe left": (769, 814),
    "strafe right": (699, 780),
}


@dataclass
class EpisodeRef:
    episode: str
    shard: str        # LMDB 目录
    episode_idx: int  # 该分片内的下标，构成 key 的第一项
    n_frames: int     # min(image, action)


def _chunk_infos(path: str) -> list[dict]:
    import lmdb

    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        infos = pickle.loads(txn.get(b"__chunk_infos__"))
    env.close()
    return infos


def episode_table(root: str | Path) -> dict[str, dict]:
    """按 episode 名 join image 与 action 两棵树。

    **不能按分片位置 join** —— image 8 分片、action 4 分片，顺序不对应。
    """
    root = Path(root)
    img: dict[str, tuple[str, int, int]] = {}
    for p in sorted(glob.glob(str(root / "image" / "part-*"))):
        for i in _chunk_infos(p):
            img[i["episode"]] = (p, i["episode_idx"], i["num_frames"])
    act: dict[str, tuple[str, int, int]] = {}
    for p in sorted(glob.glob(str(root / "action" / "part-*"))):
        for i in _chunk_infos(p):
            act[i["episode"]] = (p, i["episode_idx"], i["num_frames"])

    out = {}
    for ep in sorted(set(img) & set(act)):
        ip, ii, inf = img[ep]
        ap, ai, anf = act[ep]
        out[ep] = {
            "image": EpisodeRef(ep, ip, ii, min(inf, anf)),
            "action": EpisodeRef(ep, ap, ai, min(inf, anf)),
            # 长度分歧本身留着：15 个 episode 有分歧，最大 9.1 倍
            "n_image": inf,
            "n_action": anf,
        }
    return out


def read_actions(shard: str, episode_idx: int, start: int, n: int) -> dict[str, np.ndarray]:
    """取 [start, start+n) 的逐帧动作。"""
    import lmdb

    env = lmdb.open(shard, readonly=True, lock=False, readahead=False)
    first = (start // CHUNK_FRAMES) * CHUNK_FRAMES
    last = ((start + n + CHUNK_FRAMES - 1) // CHUNK_FRAMES) * CHUNK_FRAMES
    buf: dict[str, list] = {k: [] for k in CONTRACTOR_BUTTONS}
    hot: list = []
    cam: list = []
    with env.begin() as txn:
        for off in range(first, last, CHUNK_FRAMES):
            raw = txn.get(str((episode_idx, off)).encode())
            if raw is None:
                break
            d = pickle.loads(raw)
            for k in CONTRACTOR_BUTTONS:
                buf[k].append(np.asarray(d[k], dtype=np.int8))
            hot.append(np.stack([np.asarray(d[f"hotbar.{i}"], dtype=np.int8)
                                 for i in range(1, 10)], axis=1))
            cam.append(np.asarray(d["camera"], dtype=np.float32))
    env.close()
    lo = start - first
    out = {k: np.concatenate(buf[k])[lo:lo + n] for k in CONTRACTOR_BUTTONS}
    out["hotbar"] = np.concatenate(hot)[lo:lo + n]
    out["camera"] = np.concatenate(cam)[lo:lo + n]   # (n, 2) = [pitch, yaw]，度
    return out


def read_frames(shard: str, episode_idx: int, wanted: np.ndarray) -> np.ndarray:
    """按绝对帧号取若干帧，返回 (len(wanted), H, W, 3) uint8。

    只解码真正覆盖到 ``wanted`` 的那些 MP4 chunk。
    """
    import av
    import lmdb

    env = lmdb.open(shard, readonly=True, lock=False, readahead=False)
    need = {}
    for f in wanted:
        need.setdefault((int(f) // CHUNK_FRAMES) * CHUNK_FRAMES, []).append(int(f))
    frames: dict[int, np.ndarray] = {}
    with env.begin() as txn:
        for off, fs in need.items():
            raw = txn.get(str((episode_idx, off)).encode())
            if raw is None:
                continue
            local = {f - off for f in fs}
            i = 0
            with av.open(__import__("io").BytesIO(raw), "r") as c:
                for frame in c.decode(video=0):
                    if i in local:
                        frames[off + i] = frame.to_ndarray(format="rgb24")
                    i += 1
                    if i > max(local):
                        break
    env.close()
    h, w = next(iter(frames.values())).shape[:2] if frames else (224, 224)
    return np.stack([frames.get(int(f), np.zeros((h, w, 3), np.uint8)) for f in wanted])


def window_features(act: dict[str, np.ndarray]) -> dict[str, float]:
    """一个窗口的行为统计量。"""
    return {
        **{k: float(act[k].mean()) for k in ("forward", "back", "left", "right", "jump", "attack")},
        # camera 是 [pitch, yaw]；转向看 yaw 的累计
        "yaw_sum": float(act["camera"][:, 1].sum()),
    }


def label_window(f: dict[str, float]) -> int:
    """恰好一条行为占优才给标签，含糊的丢掉。

    文档：*"A window is labelled only when one behaviour dominates it; ambiguous
    windows are dropped. 378 hours are available and a week of training sees ~1%,
    so selectivity is free and a noisy instruction is worse than no window."*
    """
    t = THRESHOLDS
    hits = [
        f["forward"] >= t["forward"],
        f["back"] >= t["back"],
        f["left"] >= t["left"],
        f["right"] >= t["right"],
        f["jump"] >= t["jump"],
        f["attack"] >= t["attack"],
        f["yaw_sum"] <= -t["yaw"],
        f["yaw_sum"] >= t["yaw"],
    ]
    if sum(hits) != 1:
        return -1
    return int(np.argmax(hits))


def n_windows(n_frames: int, window: int = WINDOW_FRAMES) -> int:
    """该 episode 能切出多少个窗口。

    ``-1`` 不是差一错误：timestep t 观察其 chunk 的起始帧，所以最后一个 chunk
    之后还要留一帧。合计 423,316，与文档候选数精确一致。
    """
    return max(0, (n_frames - 1) // window)
