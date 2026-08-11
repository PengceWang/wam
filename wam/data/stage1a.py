"""Stage 1a 的取数：平衡采样 + goal token 表。

罕见指令走**平衡采样，不走损失加权** —— 文档的理由是加权会扭曲损失曲面，采样只改变
模型看到什么。seq8 下每条指令都过了 800 个窗口（strafe left 最少，814 个）。

解码在独立进程里并行做。文档实测单线程 MP4 解码 1,789 帧/s，一个 8 步窗口只要 8 帧
（每个 timestep 取其 chunk 的起始帧），所以瓶颈从来不是解码 —— 但 batch 16 时如果串行
仍会和 0.7s 的 GPU 步时同量级，所以这里预取。
"""

from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..config import WAMConfig
from ..model.action import ActionChunk, CameraBinner
from ..training.losses import Batch
from .contractor import INSTRUCTIONS, CONTRACTOR_BUTTONS, read_actions, read_frames

# 每个 worker 一份，避免每次取数都重建（LMDB 句柄不能跨 fork 共享）
_CTX: dict = {}


def _init_worker(cfg_dict: dict) -> None:
    from ..config import WAMConfig as _C

    cfg = _C.from_dict(cfg_dict)
    _CTX["cfg"] = cfg
    _CTX["binner"] = CameraBinner(cfg.action.camera_bins, cfg.action.camera_max_delta,
                                  cfg.action.camera_mu)


def _load_window(job: tuple) -> tuple:
    """一个窗口 -> (pixels, buttons, camera_bins, hotbar)。在 worker 进程里跑。"""
    img_shard, img_idx, act_shard, act_idx, start, seq_len = job
    cfg = _CTX["cfg"]
    binner = _CTX["binner"]
    k = cfg.action.chunk_size

    # timestep t 观察它那个 chunk 的**起始**帧，并预测其后的 k 个动作。
    # 错开一个 chunk 的话，模型看到的是它被要求输出的那些动作的结果 ——
    # 损失反而降得更快，策略却没用（docs/measurements.md 记过，用逐帧互相关验的）。
    want = np.array([start + t * k for t in range(seq_len)], dtype=np.int64)
    frames = read_frames(img_shard, img_idx, want)               # (T, H, W, 3)
    act = read_actions(act_shard, act_idx, start, seq_len * k)

    size = cfg.vision.image_size
    if frames.shape[1] != size or frames.shape[2] != size:
        import cv2

        frames = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
                           for f in frames])
    pixels = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0

    names = list(cfg.action.buttons)
    buttons = np.zeros((seq_len, k, len(names)), dtype=np.float32)
    for i, n in enumerate(names):
        if n in CONTRACTOR_BUTTONS:                    # pickItem 在录制里不存在 -> 恒 0
            buttons[:, :, i] = act[n].reshape(seq_len, k)

    cam = act["camera"].reshape(seq_len, k, 2)         # [pitch, yaw]，度
    # MineRL 的 camera 是 [pitch, yaw]，ActionChunk 的 bin 是 (yaw, pitch)。
    # 不交换的话每个学到的转身都会变成抬头，而且损失曲线上看不出来。
    deg = np.stack([cam[..., 1], cam[..., 0]], axis=-1)
    camera = binner.to_bins(torch.from_numpy(deg)).numpy().astype(np.int64)

    hb = act["hotbar"].reshape(seq_len, k, 9)
    hotbar = np.where(hb.any(-1), hb.argmax(-1) + 1, 0).astype(np.int64)  # 0 = 保持当前槽
    return pixels, buttons, camera, hotbar


class Stage1aData:
    """索引里的标注窗口 -> 平衡采样的 ``Batch``。"""

    def __init__(self, cfg: WAMConfig, root: str | Path, index: str | Path,
                 episodes=None, workers: int = 32, prefetch: int = 6) -> None:
        self.cfg = cfg
        self.root = Path(root)
        z = np.load(Path(index), allow_pickle=True)
        self.seq_len = int(z["seq_len"])

        labels = z["label"]
        keep = labels >= 0
        if episodes is not None:
            keep &= np.isin(z["episode"], list(episodes))

        self.img_shard = z["img_shard"][keep]
        self.img_idx = z["img_idx"][keep]
        self.act_shard = z["act_shard"][keep]
        self.act_idx = z["act_idx"][keep]
        self.start = z["start"][keep]
        self.label = labels[keep]

        # label_of: 训练脚本打印 len(data.label_of) 作为「标注窗口数」
        self.label_of = {i: int(l) for i, l in enumerate(self.label)}
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for i, l in self.label_of.items():
            self.by_label[l].append(i)
        self.present = sorted(self.by_label)

        # spawn，不是 fork：这个池是在 model.to(cuda) 之后建的，fork 会把已初始化的
        # CUDA 上下文复制进子进程，是典型的挂死来源。worker 只做 CPU 活，spawn 的
        # 启动开销一次性付掉即可。
        import multiprocessing as mp

        self._pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                         initargs=(cfg.to_dict(),),
                                         mp_context=mp.get_context("spawn"))
        self._prefetch = prefetch
        self._queue: list = []

    # -- 采样 -------------------------------------------------------------

    def _sample_ids(self, batch_size: int) -> list[int]:
        """平衡采样：先均匀挑指令，再从该类里挑窗口。"""
        out = []
        for _ in range(batch_size):
            l = random.choice(self.present)
            out.append(random.choice(self.by_label[l]))
        return out

    def _submit(self, batch_size: int):
        ids = self._sample_ids(batch_size)
        jobs = [(str(self.img_shard[i]), int(self.img_idx[i]), str(self.act_shard[i]),
                 int(self.act_idx[i]), int(self.start[i]), self.seq_len) for i in ids]
        return ids, [self._pool.submit(_load_window, j) for j in jobs]

    def batch(self, batch_size: int) -> tuple[Batch, list[str]]:
        while len(self._queue) < self._prefetch:
            self._queue.append(self._submit(batch_size))
        ids, futures = self._queue.pop(0)
        parts = [f.result() for f in futures]

        pixels = torch.from_numpy(np.stack([p[0] for p in parts]))
        buttons = torch.from_numpy(np.stack([p[1] for p in parts]))
        camera = torch.from_numpy(np.stack([p[2] for p in parts]))
        hotbar = torch.from_numpy(np.stack([p[3] for p in parts]))
        texts = [INSTRUCTIONS[self.label_of[i]] for i in ids]
        return Batch(pixels=pixels, actions=ActionChunk(buttons, camera, hotbar)), texts

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# -- goal slot ------------------------------------------------------------


def instruction_goal_tokens(model, device) -> torch.Tensor:
    """八条指令，用主干自己的 token embedding 编码一次。

    (n_instructions, n_goal_tokens, d_model)。主干本来就是 LLM，所以一条指令不需要
    新模态、也不需要额外的对齐损失。
    """
    with torch.no_grad():
        table = model.goal_encoder.from_text(model.backbone, list(INSTRUCTIONS))
    return table.to(device)


def goal_for(texts: list[str], goal_table: torch.Tensor) -> torch.Tensor:
    """指令字符串 -> (B, n_goal_tokens, d)，从表里取行。"""
    idx = torch.tensor([INSTRUCTIONS.index(t) for t in texts], device=goal_table.device)
    return goal_table[idx]
