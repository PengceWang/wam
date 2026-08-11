"""取数：合成 batch、在线 rollout、contractor 离线窗口。

.. warning::
   **整个 ``wam/data/`` 是重建的。** 原版从未进过 git —— ``.gitignore`` 的 ``data/``
   规则不带前导斜杠，匹配任意层级的同名目录，把这个包整包吞掉了
   （``git check-ignore -v wam/data/__init__.py`` -> ``.gitignore:25:data/``）。
   按 ``docs/measurements.md`` 的规格复原；窗口枚举与文档的 423,316 精确一致，
   指令标注阈值是反解的，逐类偏差见 :data:`wam.data.contractor.FIT_QUALITY`。
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import WAMConfig
from ..model.action import ActionChunk
from ..training.losses import Batch
from .collect import Trajectory, frames_to_tensor, rollout, trajectory_to_batch

__all__ = [
    "Batch",
    "Trajectory",
    "frames_to_tensor",
    "random_batches",
    "rollout",
    "trajectory_to_batch",
]


def random_batch(cfg: WAMConfig, generator: torch.Generator | None = None) -> Batch:
    """一个形状正确的合成 batch。用于冒烟测试和管路检查，不下载任何东西。"""
    b, t = cfg.train.batch_size, cfg.train.seq_len
    k, s = cfg.action.chunk_size, cfg.vision.image_size
    g = generator

    pixels = torch.rand(b, t, 3, s, s, generator=g)
    actions = ActionChunk(
        torch.randint(0, 2, (b, t, k, cfg.action.n_buttons), generator=g).float(),
        torch.randint(0, cfg.action.camera_bins, (b, t, k, 2), generator=g),
        torch.randint(0, cfg.action.n_hotbar + 1, (b, t, k), generator=g),
    )
    event_ids = torch.randint(0, cfg.event.vocab_size, (b, t, cfg.event.n_event_tokens), generator=g)
    event_targets = torch.zeros(b, t, cfg.event.vocab_size)
    event_targets.scatter_(2, event_ids, 1.0)

    return Batch(
        pixels=pixels,
        actions=actions,
        event_ids=event_ids,
        event_targets=event_targets,
        flags=torch.randint(0, 2, (b, t, 5), generator=g).float(),
        health_delta=torch.zeros(b, t),
        returns=torch.randn(b, t, generator=g),
        status=torch.rand(b, t, cfg.status.n_fields, generator=g),
    )


def random_batches(cfg: WAMConfig, n: int | None = None, seed: int | None = None):
    """合成 batch 的迭代器。``n=None`` 表示无限。"""
    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    i = 0
    while n is None or i < n:
        yield random_batch(cfg, g)
        i += 1
