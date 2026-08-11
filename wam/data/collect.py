"""在线采集：rollout -> ``Batch``。

用于 ``scripts/train.py --source env`` 和 ``scripts/agent_play.py``。离线的
contractor 取数走 :mod:`wam.data.stage1a`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import WAMConfig
from ..model.action import ActionChunk
from ..training.losses import Batch


@dataclass
class Trajectory:
    """一段 rollout 里逐步记下来的东西。"""

    frames: list = field(default_factory=list)        # (H, W, 3) uint8
    event_ids: list = field(default_factory=list)
    event_texts: list = field(default_factory=list)
    buttons: list = field(default_factory=list)
    camera: list = field(default_factory=list)
    hotbar: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    status: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)


def frames_to_tensor(frames: list, size: int | None = None) -> torch.Tensor:
    """RGB 帧序列 -> (1, T, 3, H, W) float，0..1。"""
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    if size is not None and (arr.shape[1] != size or arr.shape[2] != size):
        import cv2

        arr = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in arr])
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.permute(0, 3, 1, 2).unsqueeze(0)


@torch.no_grad()
def rollout(model, env, n_steps: int, device=None) -> Trajectory:
    """让当前策略跑 ``n_steps`` 个模型调用（每次 = chunk_size 个 tick）。"""
    device = device or next(model.parameters()).device
    cfg = model.cfg
    obs = env.reset()
    state = model.initial_state(1, device=device)
    traj = Trajectory()

    for _ in range(n_steps):
        pixels = frames_to_tensor([obs.rgb], cfg.vision.image_size).to(device)[:, 0]
        events = torch.from_numpy(np.asarray(obs.event_ids)).long().unsqueeze(0).to(device)
        status = None
        if obs.status is not None:
            status = torch.from_numpy(np.asarray(obs.status)).float().unsqueeze(0).to(device)

        out, state = model.step(state, pixels, events, status=status)
        action = model.action_head.sample(out.readout, temperature=1.0)
        state = type(state)(memory=state.memory, prev_action=action, goal=state.goal,
                            goal_is_external=state.goal_is_external, cache=state.cache)

        traj.frames.append(obs.rgb)
        traj.event_ids.append(np.asarray(obs.event_ids))
        traj.event_texts.append(obs.event_text)
        traj.buttons.append(action.buttons[0].detach().float().cpu())
        traj.camera.append(action.camera[0].detach().long().cpu())
        traj.hotbar.append(action.hotbar[0].detach().long().cpu())
        traj.rewards.append(float(model.drive_head.intrinsic_reward(out.drive)[0]))
        if obs.status is not None:
            traj.status.append(np.asarray(obs.status))
        traj.flags.append((obs.info or {}).get("flags", {}))

        obs = env.step(action)
        if obs.done:
            break
    return traj


def trajectory_to_batch(traj: Trajectory, cfg: WAMConfig, model=None) -> Batch:
    """``Trajectory`` -> 一条长度为 T 的训练序列（batch 维为 1）。"""
    t = len(traj)
    pixels = frames_to_tensor(traj.frames, cfg.vision.image_size)
    actions = ActionChunk(
        torch.stack(traj.buttons).unsqueeze(0),
        torch.stack(traj.camera).unsqueeze(0),
        torch.stack(traj.hotbar).unsqueeze(0),
    )
    event_ids = torch.from_numpy(np.stack(traj.event_ids)).long().unsqueeze(0) if traj.event_ids else None
    status = None
    if traj.status:
        status = torch.from_numpy(np.stack(traj.status)).float().unsqueeze(0)

    returns = None
    if traj.rewards:
        returns = discounted_returns(traj.rewards).unsqueeze(0)

    flags = None
    if traj.flags and any(traj.flags):
        keys = ("block_removed", "inventory_changed", "contact", "new_area", "done")
        flags = torch.tensor(
            [[float(bool(f.get(k, False))) for k in keys] for f in traj.flags]
        ).unsqueeze(0)

    return Batch(pixels=pixels, actions=actions, event_ids=event_ids,
                 status=status, returns=returns, flags=flags)


def discounted_returns(rewards: list, gamma: float = 0.99) -> torch.Tensor:
    """折扣回报。没有 bootstrap、没有 λ-return —— 文档把这条列为未完成项。"""
    out, run = [], 0.0
    for r in reversed(rewards):
        run = r + gamma * run
        out.append(run)
    return torch.tensor(list(reversed(out)), dtype=torch.float32)
