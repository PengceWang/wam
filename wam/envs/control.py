"""Keyboard and mouse injection via Win32 ``SendInput``.

Eaglercraft is a web page, so there is no game API to call -- control has to go
in as real OS-level input events, which the browser then turns into DOM key
events and (under Pointer Lock) raw mouse deltas.

Two consequences worth knowing:

* ``SendInput`` delivers to whatever window has focus, so the browser must be in
  the foreground. Frame capture via WGC does not need focus, but control does.
* Mouse look only works while the canvas holds Pointer Lock. Chromium feeds
  pointer-locked ``movementX/Y`` from raw input, which injected moves do reach.

``KeyboardMouse`` tracks which keys it is currently holding and can always let
go of everything -- a crashed agent must not leave the player sprinting into a
ravine with W stuck down.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Self

# -- Win32 plumbing ---------------------------------------------------------

PUL = ctypes.POINTER(ctypes.c_ulong)
INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
MOUSE_BUTTON_FLAGS = {
    "left": (0x0002, 0x0004),  # down, up
    "right": (0x0008, 0x0010),
    "middle": (0x0020, 0x0040),
}


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", PUL),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", PUL),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [  # noqa: RUF012 - ctypes requires a plain mutable list here
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUTUNION),
    ]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _send(*inputs: _INPUT) -> None:
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    sent = _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
    if sent != n:
        raise OSError(f"SendInput sent {sent}/{n}: {ctypes.get_last_error()}")


# -- key table --------------------------------------------------------------
# (virtual key, scan code). Both are supplied: the browser derives `event.key`
# from the VK and `event.code` from the scan code, and Eaglercraft reads code.

KEYS: dict[str, tuple[int, int]] = {
    "w": (0x57, 0x11),
    "a": (0x41, 0x1E),
    "s": (0x53, 0x1F),
    "d": (0x44, 0x20),
    "e": (0x45, 0x12),
    "q": (0x51, 0x10),
    "f": (0x46, 0x21),
    "space": (0x20, 0x39),
    "shift": (0xA0, 0x2A),  # left shift  -> sneak
    "ctrl": (0xA2, 0x1D),  # left control -> sprint
    "esc": (0x1B, 0x01),
    "f3": (0x72, 0x3D),
    "f11": (0x7A, 0x57),
    **{str(i): (0x30 + i, 0x01 + i) for i in range(1, 10)},  # hotbar 1-9
}


class KeyboardMouse:
    """Stateful input injector that never loses track of what it is holding."""

    def __init__(self) -> None:
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()

    # -- keys ---------------------------------------------------------------

    def key_down(self, name: str) -> None:
        if name in self._held_keys:
            return
        vk, scan = KEYS[name]
        _send(_INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=_KEYBDINPUT(vk, scan, 0, 0, None))))
        self._held_keys.add(name)

    def key_up(self, name: str) -> None:
        if name not in self._held_keys:
            return
        vk, scan = KEYS[name]
        _send(
            _INPUT(
                INPUT_KEYBOARD,
                _INPUTUNION(ki=_KEYBDINPUT(vk, scan, KEYEVENTF_KEYUP, 0, None)),
            )
        )
        self._held_keys.discard(name)

    def tap(self, name: str, hold: float = 0.03) -> None:
        self.key_down(name)
        time.sleep(hold)
        self.key_up(name)

    def set_keys(self, wanted: set[str]) -> None:
        """Make the held set exactly ``wanted``, pressing/releasing the delta."""
        for name in self._held_keys - wanted:
            self.key_up(name)
        for name in wanted - self._held_keys:
            self.key_down(name)

    # -- mouse ---------------------------------------------------------------

    def mouse_move(self, dx: int, dy: int) -> None:
        """Relative move. Under Pointer Lock this becomes movementX/movementY."""
        if dx == 0 and dy == 0:
            return
        _send(
            _INPUT(
                INPUT_MOUSE,
                _INPUTUNION(mi=_MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, None)),
            )
        )

    def mouse_move_to(self, x: int, y: int) -> None:
        """Absolute move to a screen pixel. Only needed for clicking menu buttons.

        In-game look control must use the relative ``mouse_move``: under Pointer
        Lock the cursor does not move, so absolute positioning is meaningless.
        """
        vx = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vy = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vw = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        vh = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        nx = int((x - vx) * 65535 / max(vw - 1, 1))
        ny = int((y - vy) * 65535 / max(vh - 1, 1))
        flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        _send(_INPUT(INPUT_MOUSE, _INPUTUNION(mi=_MOUSEINPUT(nx, ny, 0, flags, 0, None))))

    def click_at(
        self, x: int, y: int, button: str = "left", settle: float = 0.08, hold: float = 0.05
    ) -> None:
        self.mouse_move_to(x, y)
        time.sleep(settle)
        self.click(button, hold=hold)

    def button_down(self, button: str) -> None:
        if button in self._held_buttons:
            return
        down, _ = MOUSE_BUTTON_FLAGS[button]
        _send(_INPUT(INPUT_MOUSE, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, down, 0, None))))
        self._held_buttons.add(button)

    def button_up(self, button: str) -> None:
        if button not in self._held_buttons:
            return
        _, up = MOUSE_BUTTON_FLAGS[button]
        _send(_INPUT(INPUT_MOUSE, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, up, 0, None))))
        self._held_buttons.discard(button)

    def set_buttons(self, wanted: set[str]) -> None:
        for b in self._held_buttons - wanted:
            self.button_up(b)
        for b in wanted - self._held_buttons:
            self.button_down(b)

    def click(self, button: str = "left", hold: float = 0.05) -> None:
        self.button_down(button)
        time.sleep(hold)
        self.button_up(button)

    # -- safety --------------------------------------------------------------

    def release_all(self) -> None:
        """Let go of everything. Call this in a ``finally``, always."""
        for name in list(self._held_keys):
            self.key_up(name)
        for button in list(self._held_buttons):
            self.button_up(button)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.release_all()


# -- window focus -----------------------------------------------------------


def focus_window(hwnd: int, settle: float = 0.25, restore_if_minimised: bool = True) -> bool:
    """Bring a window to the foreground so injected input lands on it.

    Focus and z-order only -- the window is never moved, resized, maximised or
    restored down. The one exception is un-minimising, which is opt-out via
    ``restore_if_minimised``: a minimised window cannot be captured or clicked
    at all, and ``SW_RESTORE`` puts back the user's own previous geometry rather
    than imposing one.

    ``SetForegroundWindow`` is refused when the caller does not own the current
    foreground window, so borrow its input queue for the duration of the call.
    """
    import win32con
    import win32gui
    import win32process

    if win32gui.GetForegroundWindow() == hwnd:
        return True

    before = win32gui.GetWindowRect(hwnd)

    if win32gui.IsIconic(hwnd):
        if not restore_if_minimised:
            return False
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    current = _kernel32.GetCurrentThreadId()
    foreground = win32gui.GetForegroundWindow()
    other = win32process.GetWindowThreadProcessId(foreground)[0] if foreground else 0

    attached = bool(other) and bool(_user32.AttachThreadInput(current, other, True))
    try:
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception as exc:  # noqa: BLE001 - the shell refuses this sometimes
            # Not fatal on its own: BringWindowToTop below often still works, and
            # the caller gets the truth from the verified return value.
            focus_window.last_error = exc
        # Raises the window without touching its position or size.
        win32gui.BringWindowToTop(hwnd)
    finally:
        if attached:
            _user32.AttachThreadInput(current, other, False)

    time.sleep(settle)
    focus_window.last_geometry_change = (
        None if win32gui.GetWindowRect(hwnd) == before else (before, win32gui.GetWindowRect(hwnd))
    )
    return win32gui.GetForegroundWindow() == hwnd


focus_window.last_error = None
focus_window.last_geometry_change = None
