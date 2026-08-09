"""Check every basic switch reaches the game, one at a time.

    python scripts/control_test.py

Movement is verified against the F3 debug overlay (turn F3 on in game), because
a frame delta only proves "something changed". The keyboard is always released
on exit, including on crash -- a stuck W walks the player into a ravine.

It takes over the mouse and keyboard for about 15 seconds. Do not type.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.config import WAMConfig
from wam.envs.eaglercraft import EaglercraftEnv, find_menu_buttons

# key/mouse switch -> how long to hold it
CHECKS: list[tuple[str, dict]] = [
    ("W forward", {"keys": ["w"], "secs": 1.2}),
    ("S back", {"keys": ["s"], "secs": 1.2}),
    ("A strafe", {"keys": ["a"], "secs": 0.9}),
    ("D strafe", {"keys": ["d"], "secs": 0.9}),
    ("space jump", {"keys": ["space"], "secs": 0.3}),
    ("ctrl+W sprint", {"keys": ["ctrl", "w"], "secs": 1.0}),
    ("shift sneak", {"keys": ["shift", "w"], "secs": 1.0}),
    ("mouse yaw +", {"mouse": (400, 0)}),
    ("mouse yaw -", {"mouse": (-400, 0)}),
    ("mouse pitch +", {"mouse": (0, 250)}),
    ("mouse pitch -", {"mouse": (0, -250)}),
    ("hotbar 3", {"tap": "3"}),
]


def hud_crop(env, scale: float = 1.5):
    import cv2

    frame = env.capture.latest()
    h = frame.shape[0]
    crop = frame[int(h * 0.03) : int(h * 0.10), 0:560]
    return cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def run_check(env, spec: dict) -> None:
    if "keys" in spec:
        env.input.set_keys(set(spec["keys"]))
        time.sleep(spec["secs"])
        env.input.set_keys(set())
    elif "mouse" in spec:
        dx, dy = spec["mouse"]
        for _ in range(10):
            env.input.mouse_move(dx // 10, dy // 10)
            time.sleep(0.02)
    elif "tap" in spec:
        env.input.tap(spec["tap"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="Eaglercraft")
    ap.add_argument("--out", default="control_test.png")
    args = ap.parse_args()

    cfg = WAMConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    env = EaglercraftEnv(cfg, window=args.window)
    strip = []

    try:
        print(f"window: {env.title[:60]}")
        print(f"menu buttons: {len(find_menu_buttons(env.capture.latest()))}")
        print(f"ensure_playing: {env.ensure_playing()}")
        time.sleep(0.5)
        strip.append(("start", hud_crop(env)))

        for label, spec in CHECKS:
            print(f"  {label}")
            run_check(env, spec)
            time.sleep(0.45)
            strip.append((label, hud_crop(env)))

        import cv2

        labelled = []
        for name, img in strip:
            canvas = np.zeros((img.shape[0] + 24, img.shape[1], 3), np.uint8)
            canvas[24:] = img
            cv2.putText(canvas, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            labelled.append(canvas)
        cv2.imwrite(args.out, np.vstack(labelled)[:, :, ::-1])
        print(f"wrote {args.out} -- x/y/z should change on the movement rows")

    finally:
        env.close()
        print("released all input")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
