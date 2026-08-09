"""The click guard, tested without touching a real window or a real mouse.

Regression cover for an actual incident: relative mouse moves physically drag
the cursor whenever Pointer Lock is not held, the cursor walked up onto the
browser tab strip, and the next attack action clicked a tab shut.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from wam.config import WAMConfig
from wam.envs.eaglercraft import KEY_FOR_BUTTON, MOUSE_FOR_BUTTON, EaglercraftEnv

WINDOW = (100, 50, 1100, 850)  # a 1000x800 client area at screen (100, 50)
CHROME_PX = 80  # tab strip + address bar: a fixed pixel height, not a fraction
VIEWPORT = (WINDOW[0], WINDOW[1] + CHROME_PX, WINDOW[2], WINDOW[3])


class FakeInput:
    """Records what would have been sent, and moves a fake cursor."""

    def __init__(self, cursor):
        self.cursor = cursor
        self.clicks: list[tuple[int, int]] = []
        self.buttons: set[str] = set()
        self.keys: set[str] = set()
        self.moves: list[tuple[int, int]] = []

    def click_at(self, x, y, **kw):
        self.cursor[:] = [x, y]
        self.clicks.append((x, y))

    def mouse_move_to(self, x, y):
        self.cursor[:] = [x, y]

    def mouse_move(self, dx, dy):
        self.moves.append((dx, dy))
        self.cursor[0] += dx  # unlocked pointer: the cursor really moves
        self.cursor[1] += dy

    def set_buttons(self, wanted):
        self.buttons = set(wanted)

    def set_keys(self, wanted):
        self.keys = set(wanted)

    def tap(self, name, hold=0.0):
        pass

    def release_all(self):
        self.buttons.clear()
        self.keys.clear()


class FakeWindow:
    """A movable, resizable stand-in for the browser window."""

    def __init__(self, rect=WINDOW, chrome_px=CHROME_PX):
        self.rect = rect
        self.chrome_px = chrome_px

    @property
    def client_size(self):
        return self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]

    @property
    def viewport(self):
        return (self.rect[0], self.rect[1] + self.chrome_px, self.rect[2], self.rect[3])


@pytest.fixture
def env(monkeypatch):
    """An EaglercraftEnv with every OS call stubbed out."""
    cursor = [600, 500]  # starts inside the canvas
    win = FakeWindow()

    win32gui = types.SimpleNamespace(
        ClientToScreen=lambda hwnd, pt: (win.rect[0], win.rect[1]),
        GetClientRect=lambda hwnd: (0, 0, *win.client_size),
        GetWindowRect=lambda hwnd: win.rect,
        GetForegroundWindow=lambda: 1234,
        IsWindow=lambda hwnd: True,
        IsWindowVisible=lambda hwnd: True,
        GetClassName=lambda hwnd: "Chrome_RenderWidgetHostHWND",
        EnumWindows=lambda cb, extra: None,
        EnumChildWindows=lambda hwnd, cb, extra: cb(999, extra),
        GetWindowText=lambda hwnd: "Eaglercraft",
    )
    # The viewport child window is what the real code reads geometry from.
    win32gui.GetWindowRect = lambda hwnd: win.viewport if hwnd == 999 else win.rect
    win32api = types.SimpleNamespace(GetCursorPos=lambda: tuple(cursor))
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)
    monkeypatch.setitem(sys.modules, "win32api", win32api)

    cfg = WAMConfig.from_dict({"vision": {"image_size": 64}})
    e = EaglercraftEnv.__new__(EaglercraftEnv)  # skip __init__: no window, no capture
    e.cfg = cfg
    e.action_cfg = cfg.action
    e.hwnd = 1234
    e.edge_margin = 6
    e.max_mouse_step = 120
    e.window_substr = "Eaglercraft"
    e.require_active_tab = True
    e.pixels_per_degree = 4.0
    e.tick_seconds = 0.0
    e._steps = 0
    e._cursor_escapes = 0
    e._geometry_changes = 0
    e.input = FakeInput(cursor)
    from wam.envs.browser import window_geometry
    from wam.model.action import CameraBinner

    e._last_geometry = window_geometry(e.hwnd)

    e.binner = CameraBinner(cfg.action.camera_bins, cfg.action.camera_max_delta, cfg.action.camera_mu)
    e.capture = types.SimpleNamespace(latest=lambda: np.zeros((800, 1000, 3), np.uint8))
    e.window = win  # tests mutate this to move/resize the browser
    return e


def test_canvas_rect_tracks_the_real_chrome_height(env):
    """Must come from the viewport, not a fraction of the window height."""
    x0, y0, x1, y1 = env.canvas_rect()
    assert (x0, y0, x1, y1) == (
        VIEWPORT[0] + 6,
        VIEWPORT[1] + 6,
        VIEWPORT[2] - 6,
        VIEWPORT[3] - 6,
    )


def test_canvas_rect_follows_a_moved_window(env):
    env.window.rect = (1500, 300, 2500, 1100)  # user dragged it to another spot
    x0, y0, x1, y1 = env.canvas_rect()
    assert (x0, y0) == (1500 + 6, 300 + CHROME_PX + 6)
    assert (x1, y1) == (2500 - 6, 1100 - 6)


def test_chrome_stays_a_fixed_height_when_the_window_shrinks(env):
    """The bug this replaced: a fraction of height is wrong at every other size.

    At a 300px-tall window, 12% of height is 36px while the real chrome is
    80px -- so the old guard would have declared the address bar clickable.
    """
    env.window.rect = (100, 50, 700, 350)  # 600x300
    _, y0, _, _ = env.canvas_rect()
    assert y0 == 50 + CHROME_PX + 6
    assert y0 > 50 + 0.12 * 300, "a height fraction would have allowed clicks on chrome"


def test_geometry_change_is_noticed_not_prevented(env):
    assert env._note_geometry() is False
    env.window.rect = (200, 100, 1200, 900)
    assert env._note_geometry() is True
    assert env._geometry_changes == 1
    assert env.window.rect == (200, 100, 1200, 900), "the env must not move the window back"


def test_click_inside_canvas_is_allowed(env):
    x0, y0, x1, y1 = env.canvas_rect()
    env.safe_click((x0 + x1) // 2, (y0 + y1) // 2)
    assert len(env.input.clicks) == 1


@pytest.mark.parametrize(
    "point",
    [
        (600, 60),  # the tab strip -- this is the one that closed the page
        (600, 120),  # address bar, just above the viewport top at y=130
        (50, 500),  # left of the window entirely
        (600, 900),  # below the window
    ],
)
def test_click_outside_canvas_is_refused(env, point):
    with pytest.raises(RuntimeError, match="refusing to click"):
        env.safe_click(*point)
    assert env.input.clicks == []


def test_dead_window_stops_everything(env, monkeypatch):
    import win32gui

    monkeypatch.setattr(win32gui, "IsWindow", lambda hwnd: False)
    with pytest.raises(RuntimeError, match="window is gone"):
        env.assert_window_alive()


def test_background_tab_stops_everything(env, monkeypatch):
    """The window survives a tab switch, but the viewport is then someone else's page."""
    import win32gui

    monkeypatch.setattr(win32gui, "GetWindowText", lambda hwnd: "Some other page - Edge")
    with pytest.raises(RuntimeError, match="not the active tab"):
        env.assert_window_alive()

    # And no click can slip through while that is true.
    x0, y0, x1, y1 = env.canvas_rect()
    with pytest.raises(RuntimeError, match="not the active tab"):
        env.safe_click((x0 + x1) // 2, (y0 + y1) // 2)
    assert env.input.clicks == []


def test_drifted_cursor_is_recentred(env):
    env.input.cursor[:] = [600, 55]  # up on the tab strip
    assert env.keep_cursor_inside() is True
    x0, y0, x1, y1 = env.canvas_rect()
    x, y = env.input.cursor
    assert x0 <= x <= x1 and y0 <= y <= y1


def test_attack_is_skipped_while_the_cursor_is_outside(env):
    """The actual incident: an attack fired with the cursor on the tab strip."""
    names = env.action_cfg.buttons
    buttons = np.zeros(len(names), dtype=np.float32)
    buttons[names.index("attack")] = 1.0
    centre = env.action_cfg.camera_bins // 2

    env.input.cursor[:] = [600, 55]  # drifted onto the tab strip
    env._apply(buttons, np.array([centre, centre]), 0)

    assert env.input.buttons == set(), "must not click while the cursor is off-canvas"
    assert env._cursor_escapes == 1


def test_attack_fires_normally_when_the_cursor_is_safe(env):
    names = env.action_cfg.buttons
    buttons = np.zeros(len(names), dtype=np.float32)
    buttons[names.index("attack")] = 1.0
    centre = env.action_cfg.camera_bins // 2

    env._apply(buttons, np.array([centre, centre]), 0)
    assert env.input.buttons == {"left"}
    assert env._cursor_escapes == 0


def test_mouse_delta_is_clamped(env):
    """A max-magnitude camera bin must not fling the cursor across the desktop."""
    names = env.action_cfg.buttons
    buttons = np.zeros(len(names), dtype=np.float32)
    env.pixels_per_degree = 1000.0  # absurd on purpose
    env._apply(buttons, np.array([env.action_cfg.camera_bins - 1, 0]), 0)

    dx, dy = env.input.moves[-1]
    assert abs(dx) <= env.max_mouse_step
    assert abs(dy) <= env.max_mouse_step


def test_every_button_is_mapped_to_an_input():
    """A button the rig cannot press is a capability the model cannot learn.

    `_apply` filters with `if n in KEY_FOR_BUTTON`, so an unmapped name is
    dropped in silence -- the head keeps emitting it and nothing ever happens.
    """
    names = WAMConfig().action.buttons
    mapped = set(KEY_FOR_BUTTON) | set(MOUSE_FOR_BUTTON)
    assert set(names) - mapped == set(), "unmapped buttons would be silently dropped"


def test_pick_item_is_a_guarded_click(env):
    """pickItem is a middle click, so it needs the same cursor guard as attack."""
    names = env.action_cfg.buttons
    buttons = np.zeros(len(names), dtype=np.float32)
    buttons[names.index("pickItem")] = 1.0
    centre = env.action_cfg.camera_bins // 2

    env._apply(buttons, np.array([centre, centre]), 0)
    assert env.input.buttons == {"middle"}
