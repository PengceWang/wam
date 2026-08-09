"""Frame capture from a browser-hosted Minecraft (Eaglercraft) window.

Eaglercraft renders 1.8.8 into a WebGL canvas, so there is no Java process and no
MineRL hook to attach to -- the picture has to come off the window itself.

Three backends, in preference order:

    wgc          Windows Graphics Capture. ~50 fps, GPU path, and it keeps
                 working when the window is occluded or behind other windows.
    mss          Screen-region grab. ~33 fps, but only correct while the window
                 is actually visible on screen.
    printwindow  PW_RENDERFULLCONTENT. ~30 fps, last-resort compatibility.

Capture only. Sending input back into the game is a separate concern -- see
``wam/envs/base.py`` for the contract an actual env has to satisfy.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

DEFAULT_WINDOW = "Eaglercraft"


@dataclass
class Crop:
    """Region of the window that is actually the game canvas.

    Browser chrome (tab strip, address bar) is not part of the world and would
    otherwise be fed to the vision encoder as if it were. Press F11 in the
    browser to go fullscreen and this can stay ``None``.
    """

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        return frame[self.top : h - self.bottom, self.left : w - self.right]


def find_viewport(hwnd: int) -> tuple[int, tuple[int, int, int, int]]:
    """Screen rect of the page content area, excluding all browser chrome.

    Chromium puts the web contents in a child window of class
    ``Chrome_RenderWidgetHostHWND``, one per tab, of which only the foreground
    tab's is visible. Its rect is the viewport -- exact, live, and free.

    This is what makes the rig indifferent to where the user puts the window and
    how big they make it. The alternative, assuming chrome is some fraction of
    the window height, is wrong the moment the window is resized: the tab strip
    and address bar are a fixed pixel height, not a fixed proportion.
    """
    import win32gui

    hits: list[tuple[int, tuple[int, int, int, int]]] = []

    def cb(child, _):
        if win32gui.GetClassName(child) != "Chrome_RenderWidgetHostHWND":
            return
        if not win32gui.IsWindowVisible(child):
            return  # a background tab
        hits.append((child, win32gui.GetWindowRect(child)))

    win32gui.EnumChildWindows(hwnd, cb, None)
    if not hits:
        raise RuntimeError(
            "no visible Chrome_RenderWidgetHostHWND child -- is this actually a "
            "Chromium browser window?"
        )
    return max(hits, key=lambda h: (h[1][2] - h[1][0]) * (h[1][3] - h[1][1]))


def window_geometry(hwnd: int) -> dict:
    """Everything needed to map between screen, client and capture-frame pixels."""
    import win32gui

    ox, oy = win32gui.ClientToScreen(hwnd, (0, 0))
    _, _, cw, ch = win32gui.GetClientRect(hwnd)
    _, viewport = find_viewport(hwnd)
    return {
        "window": win32gui.GetWindowRect(hwnd),
        "client_origin": (ox, oy),
        "client_size": (cw, ch),
        "viewport": viewport,
    }


def list_windows() -> list[tuple[int, str]]:
    """Every visible top-level window with a title."""
    import win32gui

    hits: list[tuple[int, str]] = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            hits.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(cb, None)
    return hits


def find_window(substr: str = DEFAULT_WINDOW) -> tuple[int, str]:
    """Locate a visible top-level window whose title contains ``substr``.

    A browser window's title is the *active tab's* title, so this stops matching
    the moment the user switches tabs even though the window and the game are
    both still there. The error therefore lists what was actually on screen
    instead of just saying "not found".
    """
    hits = [(h, t) for h, t in list_windows() if substr.lower() in t.lower()]
    if hits:
        return hits[0]

    seen = "\n".join(f"    {t[:70]}" for _, t in list_windows())
    raise RuntimeError(
        f"no visible window titled like {substr!r}. A browser window is titled after "
        f"its active tab, so this also happens when the game is open but in a "
        f"background tab -- switch to it, or pass an explicit window=... substring.\n"
        f"  visible windows:\n{seen}"
    )


class BrowserCapture:
    """Latest-frame-wins capture of a browser window.

    The capture thread runs free and overwrites a single slot, so ``latest()``
    never blocks and never hands back a stale queue backlog -- an agent that
    thinks slowly should see the *current* world, not the world from 2s ago.
    """

    def __init__(
        self,
        window: str = DEFAULT_WINDOW,
        backend: str = "wgc",
        crop: Crop | None = None,
    ) -> None:
        self.window = window
        self.backend = backend
        self.crop = crop or Crop()
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._count = 0
        self._control = None
        self.last_error: Exception | None = None
        self.error_count = 0
        self._hwnd, self.title = find_window(window)
        self._start()

    # -- backends -----------------------------------------------------------

    def _start(self) -> None:
        if self.backend == "wgc":
            self._start_wgc()
        elif self.backend in ("mss", "printwindow"):
            self._start_polling()
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

    def _start_wgc(self) -> None:
        from windows_capture import Frame, InternalCaptureControl, WindowsCapture

        cap = WindowsCapture(cursor_capture=False, draw_border=False, window_name=self.title)

        @cap.event
        def on_frame_arrived(frame: Frame, control: InternalCaptureControl):
            self._publish(frame.frame_buffer)

        @cap.event
        def on_closed():
            pass

        self._control = cap.start_free_threaded()

    def _start_polling(self) -> None:
        self._stop = threading.Event()

        def loop():
            grab = self._grab_mss if self.backend == "mss" else self._grab_printwindow
            while not self._stop.is_set():
                try:
                    self._publish(grab())
                except Exception as exc:  # noqa: BLE001 - the thread must survive
                    # Swallowing this silently would let a permanently broken
                    # capture look like a merely idle one, so keep it visible.
                    self.last_error = exc
                    self.error_count += 1
                    time.sleep(0.05)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def _grab_mss(self) -> np.ndarray:
        import mss
        import win32gui

        if not hasattr(self, "_sct"):
            self._sct = mss.mss()
        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        region = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        return np.asarray(self._sct.grab(region))

    def _grab_printwindow(self) -> np.ndarray:
        from ctypes import windll

        import win32gui
        import win32ui

        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        w, h = right - left, bottom - top
        hwnd_dc = win32gui.GetWindowDC(self._hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        try:
            windll.user32.PrintWindow(self._hwnd, save_dc.GetSafeHdc(), 3)
            info = bmp.GetInfo()
            arr = np.frombuffer(bmp.GetBitmapBits(True), dtype=np.uint8)
            return arr.reshape(info["bmHeight"], info["bmWidth"], 4).copy()
        finally:
            win32gui.DeleteObject(bmp.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self._hwnd, hwnd_dc)

    # -- frames --------------------------------------------------------------

    def _publish(self, buf: np.ndarray) -> None:
        # Every backend hands back BGRA; the rest of the codebase speaks RGB.
        rgb = np.ascontiguousarray(buf[:, :, :3][:, :, ::-1])
        with self._lock:
            self._frame = rgb
            self._count += 1

    def latest(self, timeout: float = 5.0) -> np.ndarray:
        """Most recent frame of the whole window, as (H, W, 3) uint8 RGB."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                frame = self._frame
            if frame is not None:
                return self.crop.apply(frame)
            if time.time() > deadline:
                raise TimeoutError(f"no frame from {self.backend} within {timeout}s")
            time.sleep(0.005)

    def frame_to_screen(self) -> tuple[int, int, float, float]:
        """(origin_x, origin_y, scale_x, scale_y) mapping frame px -> screen px."""
        geo = window_geometry(self._hwnd)
        ox, oy = geo["client_origin"]
        cw, ch = geo["client_size"]
        fh, fw = self.latest().shape[:2]
        return ox, oy, cw / fw, ch / fh

    def viewport_box(self) -> tuple[int, int, int, int]:
        """The page content area in *frame* coordinates, recomputed live."""
        geo = window_geometry(self._hwnd)
        ox, oy = geo["client_origin"]
        cw, ch = geo["client_size"]
        fh, fw = self.latest().shape[:2]
        sx, sy = fw / max(cw, 1), fh / max(ch, 1)
        vx0, vy0, vx1, vy1 = geo["viewport"]
        return (
            max(0, int((vx0 - ox) * sx)),
            max(0, int((vy0 - oy) * sy)),
            min(fw, int((vx1 - ox) * sx)),
            min(fh, int((vy1 - oy) * sy)),
        )

    def viewport_frame(self) -> np.ndarray:
        """Latest frame cropped to the page, so no browser chrome reaches the model."""
        x0, y0, x1, y1 = self.viewport_box()
        frame = self.latest()
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise RuntimeError(f"viewport crop is degenerate: {(x0, y0, x1, y1)}")
        return frame[y0:y1, x0:x1]

    def observe(self, size: int = 224, letterbox: bool = True) -> np.ndarray:
        """Model-ready frame: (size, size, 3) uint8 RGB of the page content.

        Letterboxed rather than stretched. A squashed frame would mean the model
        sees a different world geometry whenever the window aspect ratio changes,
        so every resize would silently invalidate what it had learned.
        """
        import cv2

        frame = self.viewport_frame()
        if not letterbox:
            return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out = np.zeros((size, size, 3), dtype=np.uint8)
        top, left = (size - new_h) // 2, (size - new_w) // 2
        out[top : top + new_h, left : left + new_w] = resized
        return out

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._count

    def close(self) -> None:
        if self.backend == "wgc":
            if self._control is not None:
                self._control.stop()
        else:
            self._stop.set()
            self._thread.join(timeout=1.0)


def measure(cap: BrowserCapture, seconds: float = 3.0) -> dict:
    """Sample the stream and report rate plus how much it is actually moving."""
    cap.latest()  # wait for the first frame
    start_count = cap.frame_count
    t0 = time.time()
    prev = cap.latest().astype(np.int16)
    deltas, samples = [], 0
    while time.time() - t0 < seconds:
        time.sleep(0.02)
        cur = cap.latest().astype(np.int16)
        if cur.shape == prev.shape:
            deltas.append(float(np.abs(cur - prev).mean()))
        prev = cur
        samples += 1
    dt = time.time() - t0
    return {
        "capture_fps": (cap.frame_count - start_count) / dt,
        "sampled": samples,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "max_delta": float(np.max(deltas)) if deltas else 0.0,
        "shape": cap.latest().shape,
        "errors": cap.error_count,
        "last_error": repr(cap.last_error) if cap.last_error else None,
    }
