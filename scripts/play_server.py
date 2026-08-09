"""Drive the simulator from a browser -- by hand, or by telling the model what to do.

    python scripts/play_server.py --checkpoint ~/stage1a_run1.pt
    then open  http://localhost:8080

The point of stage 1a is whether an instruction in the goal slot actually
reaches the action head, so the page has a button per instruction: press one and
the model takes over with that goal, press "you" and control comes back. Being
able to switch mid-episode, in the same world, is what makes the difference
visible -- "move forward" and "turn left" from the same starting frame either
produce different behaviour or the goal slot is decorative.

No X server, no VNC: the GUI route needs a window WSLg would not show, and an
Xvfb that cannot bind because /tmp/.X11-unix is mounted read-only. Frames go out
as MJPEG, input comes back as JSON.

The model thinks in chunks of ``chunk_size`` ticks, so one model call fills a
queue that the 20Hz loop then drains -- the same cadence it was trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Filled in once a checkpoint is loaded; empty means the page offers manual play
# only, rather than buttons that would silently do nothing.
INSTRUCTION_NAMES: list[str] = []

PAGE = """<!doctype html>
<meta charset="utf-8"><title>MineStudio</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px ui-monospace,monospace;display:flex;gap:16px;padding:16px}
 #game{border:1px solid #333;image-rendering:pixelated;cursor:pointer;width:960px}
 #side{width:340px;line-height:1.7}
 h2{font-size:13px;color:#8ab;margin:16px 0 4px;text-transform:uppercase;letter-spacing:.08em}
 .k{color:#666} .v{color:#eee} .on{color:#6d6} .off{color:#555}
 table{border-collapse:collapse;width:100%} td{padding:1px 0}
 td:last-child{text-align:right}
 #hint{color:#c96}
 button{display:block;width:100%;margin:3px 0;padding:6px 8px;background:#222;color:#ccc;
   border:1px solid #444;border-radius:3px;font:12px ui-monospace,monospace;cursor:pointer;text-align:left}
 button:hover{background:#2c2c2c}
 button.sel{background:#264;border-color:#5a8;color:#dfd}
 button.you{background:#443;border-color:#885;color:#ffd}
 #who{font-size:15px;margin:6px 0}
</style>
<img id="game" src="/stream">
<div id="side">
  <div id="hint">click the picture to capture the mouse &middot; Esc releases</div>
  <h2>who is playing</h2>
  <div id="who">&mdash;</div>
  <div id="cmds"></div>
  <h2>what the model sees</h2>
  <table id="obs"></table>
  <h2>keys going in</h2>
  <table id="act"></table>
  <h2>events</h2>
  <div id="ev" class="v">&mdash;</div>
</div>
<script>
const KEYS = {KeyW:'forward',KeyS:'back',KeyA:'left',KeyD:'right',Space:'jump',
  ShiftLeft:'sneak',ControlLeft:'sprint',KeyE:'inventory',
  Digit1:'hotbar.1',Digit2:'hotbar.2',Digit3:'hotbar.3',Digit4:'hotbar.4',Digit5:'hotbar.5',
  Digit6:'hotbar.6',Digit7:'hotbar.7',Digit8:'hotbar.8',Digit9:'hotbar.9'};
const held = new Set(); let dx = 0, dy = 0, locked = false;
const game = document.getElementById('game');

game.onclick = () => game.requestPointerLock();
document.addEventListener('pointerlockchange', () => { locked = document.pointerLockElement === game; });
document.addEventListener('mousemove', e => { if (locked) { dx += e.movementX; dy += e.movementY; } });
document.addEventListener('mousedown', e => { if (locked) held.add(e.button === 0 ? 'attack' : 'use'); });
document.addEventListener('mouseup', e => { if (locked) held.delete(e.button === 0 ? 'attack' : 'use'); });
document.addEventListener('keydown', e => { if (KEYS[e.code]) { held.add(KEYS[e.code]); e.preventDefault(); } });
document.addEventListener('keyup', e => { if (KEYS[e.code]) { held.delete(KEYS[e.code]); e.preventDefault(); } });

function row(k, v, cls) { return `<tr><td class="k">${k}</td><td class="${cls||'v'}">${v}</td></tr>`; }

let driver = 'human', instruction = null, INSTRUCTIONS = [];

function setDriver(who, instr) {
  driver = who; instruction = instr;
  fetch('/control', {method:'POST', body: JSON.stringify({driver: who, instruction: instr})});
  render();
}

function render() {
  const c = document.getElementById('cmds');
  c.innerHTML = '';
  const rst = document.createElement('button');
  rst.textContent = '↻  reset world';
  rst.style.background = '#422'; rst.style.borderColor = '#855'; rst.style.color = '#fcc';
  rst.onclick = async () => {
    rst.textContent = '↻  resetting...';
    await fetch('/reset', {method:'POST'});
    driver = 'human'; instruction = null;
    document.getElementById('hint').textContent =
      'world reset -- give it a moment, then pick an instruction';
    render();
  };
  c.appendChild(rst);
  const you = document.createElement('button');
  you.textContent = 'you play';
  you.className = driver === 'human' ? 'you sel' : 'you';
  you.onclick = () => setDriver('human', null);
  c.appendChild(you);
  for (const name of INSTRUCTIONS) {
    const b = document.createElement('button');
    b.textContent = '▶ ' + name;
    if (driver === 'agent' && instruction === name) b.className = 'sel';
    b.onclick = () => setDriver('agent', name);
    c.appendChild(b);
  }
  document.getElementById('who').innerHTML = driver === 'human'
    ? '<span class="off">you</span>'
    : '<span class="on">model</span> &rarr; "' + instruction + '"';
}

async function boot() {
  const r = await fetch('/meta');
  const m = await r.json();
  INSTRUCTIONS = m.instructions || [];
  if (!m.has_model) {
    document.getElementById('hint').textContent =
      'no checkpoint loaded -- start with --checkpoint to let the model play';
  }
  render();
}
boot();

async function tick() {
  const payload = {keys: [...held], dx: dx, dy: dy};
  dx = 0; dy = 0;
  try {
    const r = await fetch('/input', {method:'POST', body: JSON.stringify(payload)});
    const s = await r.json();
    document.getElementById('obs').innerHTML =
      row('is_gui_open', s.is_gui_open, s.is_gui_open ? 'on' : 'off') +
      row('health', s.health) + row('food', s.food) +
      row('mainhand', s.mainhand) +
      row('yaw', (s.yaw||0).toFixed(1)) + row('pitch', (s.pitch||0).toFixed(1)) +
      row('x / y / z', `${(s.x||0).toFixed(1)} ${(s.y||0).toFixed(1)} ${(s.z||0).toFixed(1)}`) +
      row('step', s.step) + row('fps', (s.fps||0).toFixed(1));
    const keys = s.driver === 'agent' ? (s.model_keys || []) : [...held];
    document.getElementById('act').innerHTML =
      keys.map(k => row(k, 'held', 'on')).join('') || row('&mdash;', 'idle', 'off');
    if (s.events) document.getElementById('ev').textContent = s.events;
    if (s.error) { document.getElementById('hint').textContent = 'model error: ' + s.error;
                   driver = 'human'; instruction = null; render(); }
  } catch (e) {}
}
setInterval(tick, 100);
</script>
"""


class World:
    """The sim, plus the latest frame and whoever is currently driving it."""

    def __init__(self, sim, agent=None, sensitivity: float = 0.15) -> None:
        self.sim = sim
        self.agent = agent
        self.sensitivity = sensitivity
        self.lock = threading.Lock()
        self.keys: set[str] = set()
        self.dx = 0.0
        self.dy = 0.0
        self.frame = np.zeros((360, 640, 3), np.uint8)
        self.info: dict = {}
        self.step_count = 0
        self.fps = 0.0
        self.events = ""
        self.running = True
        self.driver = "human"
        self.instruction: str | None = None
        self.model_keys: list[str] = []
        self.error: str | None = None
        self._queue: list[dict] = []  # ticks the model has already decided on
        self._reset_wanted = False

    def set_input(self, keys, dx, dy) -> None:
        with self.lock:
            self.keys = set(keys)
            self.dx += float(dx)
            self.dy += float(dy)

    def request_reset(self) -> None:
        """Ask the loop to reset at its next tick.

        The reset has to happen on the sim thread -- MineRL is not safe to call
        from two threads, and a reset from the HTTP handler would race the step
        that is already in flight.
        """
        with self.lock:
            self._reset_wanted = True
            self._queue.clear()
            self.driver = "human"
            self.instruction = None
            self.error = None
        self.model_keys = []
        if self.agent is not None:
            self.agent.set_instruction(None)

    def set_control(self, driver: str, instruction: str | None) -> None:
        with self.lock:
            if driver == "agent" and self.agent is None:
                driver = "human"  # nothing to hand over to
            self.driver = driver
            self.instruction = instruction
            self.error = None
            # Drop whatever the model had already planned: those ticks were for
            # the previous instruction, and executing them would make the switch
            # look like the model ignored you for half a second.
            self._queue.clear()
            if driver == "agent":
                self.agent.set_instruction(instruction)
            else:
                self.model_keys = []

    def _take_action(self) -> dict:
        action = self.sim.env.action_space.no_op()
        with self.lock:
            for k in self.keys:
                if k in action:
                    action[k] = 1
            # Mouse delta is consumed, not sampled: dropping it would make fast
            # flicks vanish, and reusing it would make the view drift forever.
            dx, dy = self.dx, self.dy
            self.dx = self.dy = 0.0
        action["camera"] = [
            float(np.clip(dy * self.sensitivity, -180, 180)),  # pitch first
            float(np.clip(dx * self.sensitivity, -180, 180)),
        ]
        return action

    def _agent_action(self) -> dict:
        """One tick from the model, refilling the chunk queue when it runs dry.

        A failure here hands control back to the keyboard instead of taking the
        loop down with it: a dead sim thread keeps serving its last snapshot, so
        the page looks alive and simply stops responding -- which is a far worse
        way to find out something broke.
        """
        try:
            if not self._queue:
                self._queue = self.agent.plan(self.frame, self.info)
                self.model_keys = self.agent.last_keys
            return self._queue.pop(0)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            with self.lock:
                self.driver = "human"
                self.error = f"{type(exc).__name__}: {exc}"
                self._queue.clear()
            self.model_keys = []
            return self.sim.env.action_space.no_op()

    def _next_action(self) -> dict:
        with self.lock:
            driving = self.driver
        if driving == "agent" and self.agent is not None:
            return self._agent_action()
        return self._take_action()

    def loop(self) -> None:
        _obs, info = self.sim.reset()
        self.info = info
        last = time.time()
        while self.running:
            with self.lock:
                wanted = self._reset_wanted
                self._reset_wanted = False
            if wanted:
                _obs, info = self.sim.reset()
                self.info = info
                self.events = ""
                continue
            _obs, _r, terminated, truncated, info = self.sim.step(self._next_action())
            self.step_count += 1
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(1e-6, now - last)
            last = now

            pov = info.get("pov")
            if pov is not None:
                self.frame = pov
            self.info = info
            fired = {k: v for k, v in (info.get("mine_block") or {}).items()}
            if fired:
                self.events = ", ".join(f"mined {k} x{v}" for k, v in fired.items())

            if terminated or truncated:
                _obs, info = self.sim.reset()
                self.info = info

    def snapshot(self) -> dict:
        info = self.info
        pos = info.get("player_pos") or {}
        mainhand = (info.get("equipped_items") or {}).get("mainhand") or {}
        return {
            "is_gui_open": bool(info.get("is_gui_open", False)),
            "health": info.get("health"),
            "food": info.get("food_level"),
            "mainhand": mainhand.get("type", "air"),
            "yaw": pos.get("yaw", 0.0),
            "pitch": pos.get("pitch", 0.0),
            "x": pos.get("x", 0.0),
            "y": pos.get("y", 0.0),
            "z": pos.get("z", 0.0),
            "step": self.step_count,
            "fps": self.fps,
            "events": self.events,
            "driver": self.driver,
            "instruction": self.instruction,
            "model_keys": list(self.model_keys),
            "error": self.error,
        }


class Agent:
    """The trained policy, kept in the cadence it was trained on.

    One ``model.step`` produces ``chunk_size`` ticks, and the goal tokens for the
    current instruction are fed in every step -- an external goal has to survive,
    or 'move forward' would only apply to the first 0.4s after you pressed it.
    """

    def __init__(self, model, goal_table, instructions) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.goal_table = goal_table
        self.instructions = list(instructions)
        self.device = next(model.parameters()).device
        self.cfg = model.cfg
        self.state = model.initial_state(1, device=self.device)
        self.last_keys: list[str] = []
        self._goal = None

    def set_instruction(self, instruction: str | None) -> None:
        if instruction is None or instruction not in self.instructions:
            self._goal = None
            return
        idx = self.instructions.index(instruction)
        self._goal = self.goal_table[idx : idx + 1]
        # Fresh memory per instruction: the point is what this instruction
        # produces, not what the previous one left lying around in the state.
        self.state = self.model.initial_state(1, device=self.device)

    def plan(self, frame: np.ndarray, info: dict) -> list[dict]:
        torch = self.torch
        size = self.cfg.vision.image_size
        small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        pixels = torch.from_numpy(small.astype(np.float32) / 255.0)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(self.device)

        state = self.state
        if self._goal is not None:
            state = type(state)(
                memory=state.memory,
                prev_action=state.prev_action,
                goal=self._goal,
                goal_is_external=True,
                cache=state.cache,
            )
        with torch.no_grad():
            out, state = self.model.step(state, pixels)
            action = self.model.action_head.sample(out.readout, temperature=1.0)
        self.state = type(state)(
            memory=state.memory,
            prev_action=action,
            goal=state.goal,
            goal_is_external=state.goal_is_external,
            cache=state.cache,
        )

        # .float() first: the model runs in bfloat16 and numpy has no such dtype,
        # so the conversion raises rather than silently rounding.
        buttons = action.buttons[0].float().cpu().numpy()
        camera = action.camera[0].cpu()
        hotbar = action.hotbar[0].long().cpu().numpy()
        from wam.model.action import CameraBinner

        binner = CameraBinner(
            self.cfg.action.camera_bins, self.cfg.action.camera_max_delta, self.cfg.action.camera_mu
        )
        degrees = binner.to_degrees(camera).numpy()

        names = self.cfg.action.buttons
        ticks = []
        for k in range(buttons.shape[0]):
            tick = {n: 0 for n in names}
            tick.update({f"hotbar.{i}": 0 for i in range(1, 10)})
            for i, n in enumerate(names):
                tick[n] = int(buttons[k, i] > 0.5)
            slot = int(hotbar[k])
            if slot:
                tick[f"hotbar.{slot}"] = 1
            # ActionChunk bins are (yaw, pitch); MineRL wants [pitch, yaw].
            tick["camera"] = [float(degrees[k, 1]), float(degrees[k, 0])]
            ticks.append(tick)

        pressed = {names[i] for i in range(len(names)) if buttons[:, i].mean() > 0.5}
        yaw = float(degrees[:, 0].mean())
        if abs(yaw) > 0.5:
            pressed.add(f"camera yaw {yaw:+.1f}deg")
        self.last_keys = sorted(pressed)
        return ticks


def make_handler(world: World):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # the request log would drown the game log
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        ok, buf = cv2.imencode(
                            ".jpg", world.frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(buf.tobytes())
                            self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path == "/meta":
                self._json(
                    {
                        "instructions": list(INSTRUCTION_NAMES),
                        "has_model": world.agent is not None,
                    }
                )
            else:
                self.send_error(404)

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_POST(self):
            if self.path == "/input":
                p = self._payload()
                world.set_input(p.get("keys", []), p.get("dx", 0), p.get("dy", 0))
                self._json(world.snapshot())
            elif self.path == "/reset":
                world.request_reset()
                self._json(world.snapshot())
            elif self.path == "/control":
                p = self._payload()
                world.set_control(p.get("driver", "human"), p.get("instruction"))
                self._json(world.snapshot())
            else:
                self.send_error(404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inventory", default="oak_planks:64,stone_axe:1,crafting_table:1")
    ap.add_argument("--sensitivity", type=float, default=0.15)
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="a stage-1a checkpoint; without it the page is manual-only",
    )
    args = ap.parse_args()

    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback

    agent = None
    if args.checkpoint:
        import torch

        from wam.config import WAMConfig
        from wam.data.contractor import INSTRUCTIONS
        from wam.data.stage1a import instruction_goal_tokens
        from wam.model.wam import WorldActionModel

        print(f"loading {args.checkpoint} ...")
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = WAMConfig.from_dict(blob["config"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WorldActionModel(cfg).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        agent = Agent(model, instruction_goal_tokens(model, device), INSTRUCTIONS)
        INSTRUCTION_NAMES[:] = list(INSTRUCTIONS)
        print(f"model on {device}; {len(INSTRUCTIONS)} instructions available")
    else:
        print("no --checkpoint: manual play only")

    commands = ["/time set day", "/gamemode survival"]
    for spec in filter(None, args.inventory.split(",")):
        name, _, count = spec.partition(":")
        commands.append(f"/give @s minecraft:{name} {count or 1}")

    print("starting Minecraft (this takes ~30s)...")
    sim = MinecraftSim(
        action_type="env",
        obs_size=(224, 224),
        render_size=(640, 360),
        seed=args.seed,
        callbacks=[CommandsCallback(commands)],
    )

    world = World(sim, agent=agent, sensitivity=args.sensitivity)
    threading.Thread(target=world.loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(world))
    print(f"\n  open  http://localhost:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    world.running = False
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
