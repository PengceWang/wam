"""Does it follow the instruction, and does it move like a person?

Both questions get numbers, because both fail in ways that look fine on a loss
curve. A policy can drive the actor loss down by reproducing the marginal
action distribution while ignoring the goal slot entirely, and it can match that
distribution frame-by-frame while shaking the view in a way no human ever does.

Reference values measured over 96k contractor frames:

    lag-1 autocorrelation   yaw +0.782   pitch +0.757
    independent sampling    yaw -0.002   pitch +0.005

The second row is the same values shuffled -- identical marginals, no time
structure. That is exactly what per-tick independent sampling produces, so the
autocorrelation is the statistic that separates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

HUMAN_AUTOCORR = {"yaw": 0.782, "pitch": 0.757}
HUMAN_STILL_FRACTION = {"yaw": 0.461, "pitch": 0.524}


@dataclass
class InstructionResult:
    instruction: str
    steps: int
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0
    forward_distance: float = 0.0  # along the facing direction
    lateral_distance: float = 0.0
    height_gain: float = 0.0
    blocks_mined: int = 0
    buttons: dict = field(default_factory=dict)


def _facing(yaw_degrees: float) -> tuple[float, float]:
    """Minecraft yaw -> unit (dx, dz) of the direction the player faces."""
    rad = np.deg2rad(yaw_degrees)
    return -float(np.sin(rad)), float(np.cos(rad))


@torch.no_grad()
def run_instruction(model, env, instruction: str, goal_table, steps: int = 20):
    """Give one instruction, execute, and measure what physically happened."""
    from ..data.stage1a import goal_for

    obs = env.reset()
    device = next(model.parameters()).device
    state = model.initial_state(1, device=device)
    state = type(state)(
        memory=state.memory,
        prev_action=state.prev_action,
        goal=goal_for([instruction], goal_table),
        goal_is_external=True,
    )

    start = dict(obs.info.get("player_pos") or {})
    result = InstructionResult(instruction=instruction, steps=steps)
    cameras, pressed = [], np.zeros(len(model.cfg.action.buttons))

    for _ in range(steps):
        pixels = torch.from_numpy(obs.rgb.astype(np.float32) / 255.0)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device)
        events = torch.from_numpy(obs.event_ids).long().unsqueeze(0).to(device)
        status = None
        if obs.status is not None:
            status = torch.from_numpy(obs.status).float().unsqueeze(0).to(device)

        out, state = model.step(state, pixels, events, status=status)
        action = model.action_head.sample(out.readout, temperature=1.0)
        state = type(state)(
            memory=state.memory,
            prev_action=action,
            goal=state.goal,
            goal_is_external=True,
            cache=state.cache,
        )

        # Bin indices, not degrees: the autocorrelation is about whether
        # successive decisions are related, and bins keep that while being
        # immune to the mu-law's nonlinearity.
        cameras.append(action.camera[0].cpu().numpy())
        pressed += action.buttons[0].cpu().numpy().mean(axis=0)

        obs = env.step(action)
        if obs.event_text and "mined" in obs.event_text:
            result.blocks_mined += 1
        if obs.done:
            break

    end = dict(obs.info.get("player_pos") or {})
    if start and end:
        result.yaw_delta = float(end.get("yaw", 0) - start.get("yaw", 0))
        result.pitch_delta = float(end.get("pitch", 0) - start.get("pitch", 0))
        dx = float(end.get("x", 0) - start.get("x", 0))
        dz = float(end.get("z", 0) - start.get("z", 0))
        fx, fz = _facing(start.get("yaw", 0.0))
        result.forward_distance = dx * fx + dz * fz
        result.lateral_distance = dx * fz - dz * fx
        result.height_gain = float(end.get("y", 0) - start.get("y", 0))
    result.buttons = {
        n: float(pressed[i] / max(1, len(cameras))) for i, n in enumerate(model.cfg.action.buttons)
    }
    return result, np.concatenate(cameras, axis=0) if cameras else np.zeros((0, 2))


def humanness(camera_bins: np.ndarray, centre: int) -> dict:
    """Time-structure statistics of a camera trace, against the human numbers.

    Works on bin indices rather than degrees: what matters is whether successive
    decisions are correlated, and the bin index preserves that while being
    immune to the mu-law's nonlinearity.
    """
    out = {}
    if len(camera_bins) < 3:
        return {"n": 0}
    for axis, name in ((0, "yaw"), (1, "pitch")):
        v = camera_bins[:, axis].astype(np.float64)
        still = float((v == centre).mean())
        if v.std() < 1e-9:
            ac = 0.0  # a frozen view is not human either, and correlates with nothing
        else:
            ac = float(np.corrcoef(v[:-1], v[1:])[0, 1])
        out[f"{name}_autocorr"] = ac
        out[f"{name}_autocorr_human"] = HUMAN_AUTOCORR[name]
        out[f"{name}_still"] = still
        out[f"{name}_still_human"] = HUMAN_STILL_FRACTION[name]
    out["n"] = len(camera_bins)
    return out


# What each instruction is supposed to do to the world. Sign, not magnitude:
# the point is whether the instruction was obeyed at all.
EXPECTED = {
    "move forward": lambda r: r.forward_distance > 1.0,
    "move backward": lambda r: r.forward_distance < -1.0,
    "strafe left": lambda r: r.lateral_distance > 1.0,
    "strafe right": lambda r: r.lateral_distance < -1.0,
    "turn left": lambda r: r.yaw_delta < -20.0,
    "turn right": lambda r: r.yaw_delta > 20.0,
    "jump": lambda r: r.buttons.get("jump", 0) > 0.2,
    "mine the block in front": lambda r: r.blocks_mined > 0 or r.buttons.get("attack", 0) > 0.5,
}


def obeyed(result: InstructionResult) -> bool:
    check = EXPECTED.get(result.instruction)
    return bool(check(result)) if check else False
