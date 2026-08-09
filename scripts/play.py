"""Play the same Minecraft the agent trains on, in a real window.

    python scripts/play.py                    # just play
    python scripts/play.py --task craft_table # start with a task's items and setup

Runs inside WSL and draws on Windows through WSLg. This is the *same*
``MinecraftSim`` the model sees, at the same 640x360 render and the same 20Hz
tick, so what feels awkward here is what the model is up against -- notably that
the camera moves a cursor rather than the head whenever the inventory is open.

Key bindings, on top of normal Minecraft controls:

    C               capture the mouse (click into the window first)
    Left Ctrl + C   close
    Esc             command mode

Prints the same observation fields the model receives, so you can watch
``is_gui_open`` flip and events fire while you play.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument(
        "--task", default=None, help="a MineStudio benchmark task yaml name, e.g. craft_table"
    )
    ap.add_argument(
        "--inventory",
        default=None,
        help="comma-separated items to start with, e.g. oak_planks:64,stone_axe:1",
    )
    args = ap.parse_args()

    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        CommandsCallback,
        PlayCallback,
        RecordCallback,
    )

    callbacks = [PlayCallback()]

    commands = ["/time set day", "/gamemode survival"]
    if args.task:
        import yaml
        from minestudio.benchmark import prepare_task_configs

        root = Path(prepare_task_configs("simple")) / f"{args.task}.yaml"
        data = yaml.safe_load(root.read_text())
        commands += list(data.get("custom_init_commands", []))
        print(f"task: {args.task} -- {data.get('text', '').strip()}")
    if args.inventory:
        for spec in args.inventory.split(","):
            name, _, count = spec.partition(":")
            commands.append(f"/give @s minecraft:{name} {count or 1}")
    callbacks.append(CommandsCallback(commands))

    # Recording is cheap and the footage is the only way to look at something
    # afterwards; the window itself is gone the moment it closes.
    out = Path.home() / "play_recordings"
    callbacks.append(RecordCallback(record_path=str(out), fps=20, frame_type="pov"))
    print(f"recording to {out}")

    sim = MinecraftSim(
        action_type="env",
        obs_size=(224, 224),  # what the model would see
        render_size=(args.width, args.height),
        seed=args.seed,
        callbacks=callbacks,
    )

    _obs, _info = sim.reset()
    print("\nplaying -- click the window, press C to capture the mouse\n")

    terminated = False
    step = 0
    while not terminated:
        action = None  # PlayCallback fills this from the keyboard
        _obs, _reward, terminated, truncated, info = sim.step(action)
        step += 1
        if step % 40 == 0:  # roughly every 2 seconds
            pos = info.get("player_pos", {})
            print(
                f"step {step:5d}  gui={info.get('is_gui_open')}  "
                f"health={info.get('health')}  food={info.get('food_level')}  "
                f"yaw={pos.get('yaw', 0):7.1f} pitch={pos.get('pitch', 0):6.1f}  "
                f"mined={info.get('mine_block') or {}}"
            )
        if truncated:
            break

    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
