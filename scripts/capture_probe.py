"""Check the live stream off the browser window.

    python scripts/capture_probe.py                    # measure the stream
    python scripts/capture_probe.py --save out.png     # dump what the model sees
    python scripts/capture_probe.py --backend mss --seconds 5

Prints capture rate and how much the picture is actually changing, so a paused
game (delta ~0) is distinguishable from a broken capture (also delta ~0, but no
frames either).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.envs.browser import (
    DEFAULT_WINDOW,
    BrowserCapture,
    Crop,
    measure,
    window_geometry,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=DEFAULT_WINDOW)
    ap.add_argument("--backend", default="wgc", choices=("wgc", "mss", "printwindow"))
    ap.add_argument("--seconds", type=float, default=3.0)
    # Browser chrome is cropped automatically from Chromium's own viewport
    # window; this is only an escape hatch for a non-Chromium host.
    ap.add_argument("--crop-top", type=int, default=0, help="extra pixels to drop from the top")
    ap.add_argument("--save", default=None, help="write the model-input frame here")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    cap = BrowserCapture(args.window, args.backend, Crop(top=args.crop_top))
    print(f"window : {cap.title[:70]}")
    print(f"backend: {args.backend}")

    geo = window_geometry(cap._hwnd)
    wx0, wy0, wx1, wy1 = geo["window"]
    vx0, vy0, vx1, vy1 = geo["viewport"]
    print(f"window rect  : {geo['window']}  ({wx1 - wx0}x{wy1 - wy0})")
    print(f"page viewport: {geo['viewport']}  ({vx1 - vx0}x{vy1 - vy0})")
    print(f"browser chrome above the page: {vy0 - wy0}px")
    print(f"viewport in frame coords: {cap.viewport_box()}")

    stats = measure(cap, args.seconds)
    print(f"raw frame   : {stats['shape']}")
    print(f"capture rate: {stats['capture_fps']:.1f} fps")
    print(f"motion      : mean delta {stats['mean_delta']:.3f}, peak {stats['max_delta']:.3f}")
    if stats["capture_fps"] < 1:
        print("  -> capture is NOT running")
    elif stats["max_delta"] < 0.5:
        print("  -> frames are flowing but the picture is static (game paused?)")
    else:
        print("  -> live stream with motion")

    if args.save:
        import cv2

        frame = cap.observe(args.size)
        cv2.imwrite(args.save, frame[:, :, ::-1])
        print(f"saved {args.save} ({frame.shape}) -- this is exactly what the encoder sees")

    cap.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
