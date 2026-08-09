"""A ``MinecraftEnv`` backed by Eaglercraft 1.8.8 running in a browser.

Frames come off the window (``browser.py``), control goes in as OS-level input
(``control.py``). The awkward part in between is Pointer Lock:

* Eaglercraft pauses whenever the window loses focus or the pointer unlocks --
  which is exactly what happens every time the training script takes focus back.
* A locked pointer can only be re-acquired from a genuine user gesture, so
  resuming means finding the "Back to Game" button and clicking it. Pressing ESC
  does the opposite: on an already-running game it opens the menu.

``ensure_playing`` handles that. It locates the menu buttons by colour, clicks
only the topmost one, and verifies the menu actually went away rather than
clicking blindly again -- the bottom button is "Save and Quit to Title", so a
mis-aimed click is not a harmless mistake.

Beyond that this layer stays dumb on purpose. It exposes the same switches a
human has -- keys down/up, mouse delta, clicks -- and does not try to be clever
about what a "correct" camera movement is. The agent learns that mapping from
what the frames do in response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Self

import numpy as np

from ..config import ActionConfig, WAMConfig
from ..model.action import ActionChunk, CameraBinner
from .base import Observation
from .browser import BrowserCapture, Crop, find_window, window_geometry
from .control import KeyboardMouse, focus_window

# Our button names -> what to press. Attack/use are mouse buttons.
KEY_FOR_BUTTON = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d",
    "jump": "space",
    "sneak": "shift",
    "sprint": "ctrl",
    "inventory": "e",
}
MOUSE_FOR_BUTTON = {"attack": "left", "use": "right", "pickItem": "middle"}


@dataclass
class MenuButton:
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def find_menu_buttons(frame: np.ndarray, min_width: int = 200) -> list[MenuButton]:
    """Locate Minecraft GUI buttons by their flat mid-grey fill.

    Returns them top-to-bottom. An empty list means no menu is open, which is
    how ``ensure_playing`` decides whether the game is already running.
    """
    g = frame.astype(np.int16)
    r, gr, b = g[:, :, 0], g[:, :, 1], g[:, :, 2]
    grey = (np.abs(r - gr) < 12) & (np.abs(gr - b) < 12) & (r > 100) & (r < 180)

    h, w = grey.shape
    # Ignore the top strip: browser chrome is grey too.
    grey[: int(h * 0.10)] = False

    rows = grey.sum(axis=1)
    hot = np.where(rows > w * 0.10)[0]
    if len(hot) == 0:
        return []

    bands: list[tuple[int, int]] = []
    start = prev = hot[0]
    for y in hot[1:]:
        if y - prev > 5:
            bands.append((start, prev))
            start = y
        prev = y
    bands.append((start, prev))

    buttons: list[MenuButton] = []
    for y0, y1 in bands:
        if y1 - y0 < 15:  # too thin to be a button
            continue
        cols = np.where(grey[y0 : y1 + 1].any(axis=0))[0]
        if len(cols) == 0:
            continue
        # Split a row into separate buttons wherever there is a real gap.
        seg_start = prev_c = cols[0]
        segs = []
        for c in cols[1:]:
            if c - prev_c > 20:
                segs.append((seg_start, prev_c))
                seg_start = c
            prev_c = c
        segs.append((seg_start, prev_c))
        for x0, x1 in segs:
            if x1 - x0 >= min_width:
                buttons.append(MenuButton(int(x0), int(x1), int(y0), int(y1)))

    buttons.sort(key=lambda b: b.y0)
    return buttons


class EaglercraftEnv:
    """Chunked env over a browser Minecraft. Implements ``MinecraftEnv``."""

    def __init__(
        self,
        cfg: WAMConfig,
        window: str = "Eaglercraft",
        backend: str = "wgc",
        crop: Crop | None = None,
        tick_seconds: float = 0.05,
        pixels_per_degree: float = 4.0,
        edge_margin: int = 6,
        max_mouse_step: int = 120,
        require_active_tab: bool = True,
    ) -> None:
        """``pixels_per_degree`` converts a camera bin into a mouse delta.

        It does not need to be accurate. The agent only ever sees the frames its
        own actions produced, so any consistent scale is learnable -- this is a
        sensitivity knob, not a calibration constant. Raise it for faster turns.

        ``edge_margin`` insets the clickable region from the viewport edge.
        ``max_mouse_step`` caps a single tick's mouse delta.

        Window position and size are never touched. All geometry is read from
        the window on every use, so the browser can be moved, resized, or
        maximised at any point during a run.
        """
        self.cfg = cfg
        self.action_cfg: ActionConfig = cfg.action
        self.tick_seconds = tick_seconds
        self.pixels_per_degree = pixels_per_degree
        self.edge_margin = edge_margin
        self.max_mouse_step = max_mouse_step
        self.window_substr = window
        self.require_active_tab = require_active_tab
        self.binner = CameraBinner(cfg.action.camera_bins, cfg.action.camera_max_delta, cfg.action.camera_mu)

        self.hwnd, self.title = find_window(window)
        self.capture = BrowserCapture(window, backend, crop)
        self.input = KeyboardMouse()
        self._steps = 0
        self._cursor_escapes = 0
        self._geometry_changes = 0
        self._last_geometry = window_geometry(self.hwnd)

    # -- keeping input inside the game ---------------------------------------

    def _client_origin_and_scale(self) -> tuple[int, int, float, float]:
        """Map capture-frame pixels onto screen pixels."""
        return self.capture.frame_to_screen()

    def canvas_rect(self) -> tuple[int, int, int, int]:
        """Screen rect that input is allowed to touch: the page, not the browser.

        Taken from Chromium's own viewport child window, so it is exact at any
        window position and size and costs nothing to recompute. It is
        deliberately re-read on every call rather than cached: the user is free
        to move or resize the browser mid-run and nothing here should care.

        An earlier version guessed this as a fixed *fraction* of window height,
        which measured 163px against a true chrome height of 80px -- wrong in
        both directions as soon as the window is resized.
        """
        vx0, vy0, vx1, vy1 = window_geometry(self.hwnd)["viewport"]
        m = self.edge_margin
        return vx0 + m, vy0 + m, vx1 - m, vy1 - m

    def assert_window_alive(self) -> None:
        """Fail loudly if the game is not the thing on screen.

        Two distinct failures, both of which would otherwise send input to the
        wrong place:

        * the window is gone -- a closed tab or window leaves a stale HWND, and
          clicks land on whatever inherited that screen area;
        * the window is alive but the game sits in a *background* tab. A browser
          window is titled after its active tab, and Chromium's visible viewport
          belongs to that tab, so frames would show a different page and clicks
          would go to it.

        The tab is never switched automatically -- which tab is in front is the
        user's business, not this rig's.
        """
        import win32gui

        if not win32gui.IsWindow(self.hwnd) or not win32gui.IsWindowVisible(self.hwnd):
            raise RuntimeError("the browser window is gone -- stopping rather than clicking blind")
        if self.require_active_tab:
            title = win32gui.GetWindowText(self.hwnd)
            if self.window_substr.lower() not in title.lower():
                raise RuntimeError(
                    f"the game is not the active tab (window title is {title[:60]!r}). "
                    f"Switch back to the {self.window_substr!r} tab -- this will not "
                    f"do it for you."
                )

    def safe_click(self, x: int, y: int, **kw) -> None:
        """Click, but only inside the game canvas of the focused game window.

        Injected input goes wherever the cursor is, and relative mouse moves
        physically drag the cursor whenever Pointer Lock is *not* held. That
        combination once walked the cursor onto the tab strip and closed the
        page. So every click is checked against the canvas rect first.
        """
        import win32gui

        self.assert_window_alive()
        if win32gui.GetForegroundWindow() != self.hwnd:
            focus_window(self.hwnd)
        x0, y0, x1, y1 = self.canvas_rect()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise RuntimeError(
                f"refusing to click ({x},{y}): outside the game canvas {(x0, y0, x1, y1)}"
            )
        self.input.click_at(x, y, **kw)

    def keep_cursor_inside(self) -> bool:
        """Park a drifted cursor back in the canvas. Returns True if it moved.

        Under Pointer Lock the cursor stays put and this never fires. Without
        the lock it fires constantly -- which is also a useful signal that the
        camera is not actually being driven.
        """
        import win32api

        self.assert_window_alive()
        x0, y0, x1, y1 = self.canvas_rect()
        x, y = win32api.GetCursorPos()
        if x0 <= x <= x1 and y0 <= y <= y1:
            return False
        self.input.mouse_move_to((x0 + x1) // 2, (y0 + y1) // 2)
        return True

    # -- pause handling ------------------------------------------------------

    def is_paused(self) -> bool:
        return len(find_menu_buttons(self.capture.latest())) >= 3

    def dismiss_menu(self, attempts: int = 3) -> bool:
        """Click "Back to Game" until the pause menu is gone.

        Clicks the *topmost* button only, and re-checks instead of retrying
        blindly -- the bottom button is "Save and Quit to Title".
        """
        for _ in range(attempts):
            buttons = find_menu_buttons(self.capture.latest())
            if len(buttons) < 3:
                return True
            ox, oy, sx, sy = self._client_origin_and_scale()
            cx, cy = buttons[0].centre
            self.safe_click(int(ox + cx * sx), int(oy + cy * sy))
            time.sleep(0.6)
        return not self.is_paused()

    def camera_responds(self, probe_px: int = 400) -> bool:
        """Best-effort check that yaw reaches the camera. Diagnostics only.

        Do not gate control on this. Every cheap way of asking "is Pointer Lock
        held?" turned out to lie:

        * cursor confinement -- Chromium does *not* pin ``GetCursorPos`` while
          locked, so an injected move always appears to move the cursor and the
          test reads "unlocked" even when the camera is turning fine;
        * whole-frame correlation -- drifting clouds count as motion, and a
          steeply pitched camera turns yaw into image *rotation*, which phase
          correlation reports as zero shift.

        This version correlates only the lower, textured part of the frame, and
        is still fooled by a featureless view. ``ensure_playing`` sidesteps the
        whole question by always performing the gesture that grants the lock.
        """
        import cv2

        def ground():
            f = self.capture.latest()
            h = f.shape[0]
            return cv2.cvtColor(f[int(h * 0.55) : int(h * 0.92)], cv2.COLOR_RGB2GRAY).astype(
                np.float32
            )

        self.keep_cursor_inside()
        before = ground()
        for _ in range(10):
            self.input.mouse_move(probe_px // 10, 0)
            time.sleep(0.02)
        time.sleep(0.4)
        after = ground()
        for _ in range(10):
            self.input.mouse_move(-probe_px // 10, 0)
            time.sleep(0.02)
        # If the lock is not held these probes just dragged the cursor across the
        # desktop; put it back before anything can click.
        self.keep_cursor_inside()

        (shift_x, _), response = cv2.phaseCorrelate(before, after)
        return response > 0.05 and abs(shift_x) > 3.0

    def ensure_playing(self, attempts: int = 2) -> bool:
        """Focus the window and put the game into a controllable state.

        Rather than trying to detect Pointer Lock -- which has no reliable cheap
        test from outside the browser -- this always performs the gesture that
        is known to grant it: open the pause menu, then click "Back to Game".
        That click is a genuine user gesture on the canvas, which is the only
        thing the browser will re-lock the pointer for. Clicking the canvas of
        an already-running game does *not* work; measured, it just swings.

        The cost is a brief menu flash on every reset, which is cheap because
        resets are rare.
        """
        focus_window(self.hwnd)
        time.sleep(0.3)

        for _ in range(attempts):
            if not self.is_paused():
                self.input.tap("esc")
                time.sleep(0.6)
            if self.dismiss_menu():
                return True
        return not self.is_paused()

    # -- action execution ----------------------------------------------------

    def _apply(self, buttons: np.ndarray, camera: np.ndarray, hotbar: int) -> None:
        """Apply one low-level action (one entry of a chunk).

        Mouse buttons are only pressed once the cursor is known to be over the
        canvas. If Pointer Lock is not held the cursor drifts with every camera
        move, and an unguarded attack/use lands on whatever the cursor happened
        to reach -- the tab strip, and its close button, included.
        """
        names = self.action_cfg.buttons
        active = {names[i] for i, on in enumerate(buttons) if on > 0.5}
        wanted_mouse = {MOUSE_FOR_BUTTON[n] for n in active if n in MOUSE_FOR_BUTTON}

        self.input.set_keys({KEY_FOR_BUTTON[n] for n in active if n in KEY_FOR_BUTTON})

        if wanted_mouse and self.keep_cursor_inside():
            # The cursor had escaped, so the lock is not held and the camera is
            # not really being driven. Recentred now; skip the click this tick
            # rather than clicking at a position we do not trust.
            self._cursor_escapes += 1
            self.input.set_buttons(set())
        else:
            self.input.set_buttons(wanted_mouse)

        if hotbar > 0:
            self.input.tap(str(int(hotbar)), hold=0.01)

        yaw_deg = float(self.binner.to_degrees(camera[0]))
        pitch_deg = float(self.binner.to_degrees(camera[1]))
        # Clamp per tick: a runaway delta would fling an unlocked cursor across
        # the desktop in a single step.
        limit = self.max_mouse_step
        dx = int(np.clip(round(yaw_deg * self.pixels_per_degree), -limit, limit))
        dy = int(np.clip(round(pitch_deg * self.pixels_per_degree), -limit, limit))
        self.input.mouse_move(dx, dy)

    def step(self, action: ActionChunk) -> Observation:
        """Execute a whole chunk, then observe. One call ~= chunk_size ticks."""
        self.assert_window_alive()
        buttons = action.buttons.detach().cpu().numpy().reshape(-1, self.action_cfg.n_buttons)
        camera = action.camera.detach().cpu().numpy().reshape(-1, 2)
        hotbar = action.hotbar.detach().cpu().numpy().reshape(-1)

        for i in range(len(buttons)):
            self._apply(buttons[i], camera[i], int(hotbar[i]))
            time.sleep(self.tick_seconds)

        self.keep_cursor_inside()
        self._steps += 1
        return self._observe()

    def reset(self) -> Observation:
        self.input.release_all()
        if not self.ensure_playing():
            raise RuntimeError(
                "could not resume the game -- is the pause menu showing something "
                "other than the normal Game menu?"
            )
        self._steps = 0
        return self._observe()

    def _note_geometry(self) -> bool:
        """Track window moves/resizes. We adapt to them; we never cause them."""
        geo = window_geometry(self.hwnd)
        if geo != self._last_geometry:
            self._last_geometry = geo
            self._geometry_changes += 1
            return True
        return False

    def _observe(self) -> Observation:
        self._note_geometry()
        # Cropped to Chromium's viewport and letterboxed, so what the model sees
        # is the game and nothing else, at a stable aspect ratio.
        rgb = self.capture.observe(self.cfg.vision.image_size)
        # No event extraction yet: nothing in the browser exposes inventory or
        # block-break events, so these stay empty rather than being faked.
        return Observation(
            rgb=rgb,
            event_ids=np.zeros(self.cfg.event.n_event_tokens, dtype=np.int64),
            event_text="",
            health=20.0,
            done=False,
            info={
                "steps": self._steps,
                "paused": False,
                # Non-zero means the cursor keeps leaving the canvas, i.e. the
                # pointer is not locked and camera actions are going nowhere.
                "cursor_escapes": self._cursor_escapes,
                # The window was moved or resized this many times; harmless,
                # but it explains a discontinuity in the frames.
                "geometry_changes": self._geometry_changes,
                "viewport": self._last_geometry["viewport"],
            },
        )

    def close(self) -> None:
        self.input.release_all()
        self.capture.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
