"""Stage 1b：在 1a 的基础上，加上事件与世界标志的监督。

相对 :class:`wam.data.stage1a.Stage1aData` 只多做两件事：
读 ``meta_info``，产出 ``event_ids`` / ``event_targets`` / ``flags``。
损失从 2 项（actor + next_latent）变成 4 项。

窗口数会掉到约 74.8% —— ``meta_info`` 只覆盖 2,944 / 3,658 个 episode。
这是数据本身的缺口，不是筛选策略，所以 :class:`Stage1bData` 直接把没有
meta_info 的窗口丢掉，而不是给它们编标签。
"""

from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..config import WAMConfig
from ..model.action import ActionChunk
from ..training.losses import Batch
from .contractor import INSTRUCTIONS
from .events import FLAG_SUPERVISED, load_concepts, meta_table, read_meta, window_events
from .stage1a import _load_window

_CTX: dict = {}


def _init_worker(cfg_dict: dict, concepts_path: str) -> None:
    from ..config import WAMConfig as _C
    from .stage1a import _init_worker as _base

    _base(cfg_dict)
    _CTX["cfg"] = _C.from_dict(cfg_dict)
    _CTX["key_to_id"] = load_concepts(concepts_path)


def _load_window_1b(job: tuple) -> tuple:
    (img_shard, img_idx, act_shard, act_idx, start, seq_len,
     meta_shard, meta_idx, is_last) = job
    cfg = _CTX["cfg"]
    k = cfg.action.chunk_size

    pixels, buttons, camera, hotbar = _load_window(
        (img_shard, img_idx, act_shard, act_idx, start, seq_len))

    frames = read_meta(meta_shard, meta_idx, start, seq_len * k)
    if len(frames) < seq_len * k:
        raise ValueError("meta_info 帧数不足")
    ids, targets, flags, hind = window_events(
        frames, _CTX["key_to_id"], seq_len, k,
        cfg.event.n_event_tokens, cfg.event.vocab_size, is_last)
    return pixels, buttons, camera, hotbar, ids, targets, flags, hind


class Stage1bData:
    def __init__(self, cfg: WAMConfig, root: str | Path, index: str | Path,
                 concepts: str | Path, workers: int = 32, prefetch: int = 6,
                 all_windows: bool = False) -> None:
        """``all_windows=True`` 时不再要求"有指令标签"。

        事件预测是自监督的 —— 每个窗口都有事件，不需要"一条行为占优"这个条件。
        那个筛选是 stage 1a 为了填 goal 槽才需要的，对世界模型是纯损失：它砍掉了
        65% 的数据。没有标签的窗口，goal 槽填 ``GoalEncoder.null``，actor 于是学到
        "没人指挥时人会怎么做" —— 这正是自主 agent 需要的先验。
        """
        self.cfg = cfg
        self.all_windows = all_windows
        z = np.load(Path(index), allow_pickle=True)
        self.seq_len = int(z["seq_len"])

        meta = meta_table(root)
        ep = z["episode"]
        label = z["label"]
        has_meta = np.array([e in meta for e in ep])
        keep = has_meta if all_windows else ((label >= 0) & has_meta)

        self.img_shard, self.img_idx = z["img_shard"][keep], z["img_idx"][keep]
        self.act_shard, self.act_idx = z["act_shard"][keep], z["act_idx"][keep]
        self.start, self.label = z["start"][keep], label[keep]
        eps = ep[keep]
        self.meta_shard = np.array([meta[e][0] for e in eps])
        self.meta_idx = np.array([meta[e][1] for e in eps], dtype=np.int32)
        # 本集最后一个窗口 -> done 标志为真
        window = self.seq_len * cfg.action.chunk_size
        last_start = {e: 0 for e in eps}
        for e, s in zip(eps, self.start):
            last_start[e] = max(last_start[e], int(s))
        self.is_last = np.array([int(s) == last_start[e] for e, s in zip(eps, self.start)])

        n_lab = int((label >= 0).sum())
        n_inst = int((self.label >= 0).sum())
        print(f"候选 {len(label):,} | 指令标注 {n_lab:,} | 有 meta_info {int(has_meta.sum()):,} "
              f"-> 训练用 {int(keep.sum()):,}（其中 {n_inst:,} 个带指令，"
              f"{int(keep.sum()) - n_inst:,} 个用空目标）")

        self.label_of = {i: int(l) for i, l in enumerate(self.label)}
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for i, l in self.label_of.items():
            self.by_label[l].append(i)
        self.present = sorted(k for k in self.by_label if k >= 0)

        import multiprocessing as mp

        self._pool = ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(cfg.to_dict(), str(concepts)),
            mp_context=mp.get_context("spawn"))
        self._prefetch = prefetch
        self._queue: list = []
        self.flag_mask = torch.from_numpy(FLAG_SUPERVISED)

    def _submit(self, batch_size: int):
        if self.all_windows:
            # 自然分布。全量模式下多数窗口没有指令类别，平衡采样无从谈起；
            # 而且世界模型本来就该在真实分布上学，过采样罕见指令会扭曲它。
            ids = [random.randrange(len(self.start)) for _ in range(batch_size)]
        else:
            ids = [random.choice(self.by_label[random.choice(self.present)])
                   for _ in range(batch_size)]
        jobs = [(str(self.img_shard[i]), int(self.img_idx[i]), str(self.act_shard[i]),
                 int(self.act_idx[i]), int(self.start[i]), self.seq_len,
                 str(self.meta_shard[i]), int(self.meta_idx[i]), bool(self.is_last[i]))
                for i in ids]
        return ids, [self._pool.submit(_load_window_1b, j) for j in jobs]

    def batch(self, batch_size: int) -> tuple[Batch, list[str]]:
        while len(self._queue) < self._prefetch:
            self._queue.append(self._submit(batch_size))
        ids, futures = self._queue.pop(0)
        parts = []
        for f in futures:
            try:
                parts.append(f.result())
            except Exception:
                continue
        if not parts:
            return self.batch(batch_size)

        st = lambda i: torch.from_numpy(np.stack([p[i] for p in parts]))  # noqa: E731
        # -1 表示这个窗口没有指令标签 -> goal 槽用 null
        labels = [self.label_of[i] for i in ids[: len(parts)]]
        return Batch(
            pixels=st(0),
            actions=ActionChunk(st(1), st(2), st(3)),
            event_ids=st(4),
            event_targets=st(5),
            flags=st(6),
            # hindsight 多热。转成 goal 空间的向量要用短语嵌入表，那张表在
            # 模型上（GPU），所以留到训练循环里做一次矩阵乘，worker 保持轻量。
            hindsight=st(7),
        ), labels

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
