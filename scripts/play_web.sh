#!/usr/bin/env bash
# Play the simulator from a browser tab.
#
#   bash scripts/play_web.sh            then open  http://localhost:6080/vnc.html
#
# WSLg's own display works for GUI apps in principle, but a window that never
# appears is impossible to debug from the terminal side, and a browser tab is
# reachable no matter what the desktop is doing. So the game gets its own
# virtual display, x11vnc exports it, and noVNC serves that over HTTP -- WSL2
# forwards localhost to Windows, so the tab just works.
# No `set -u`: conda-forge's openjdk activate script dereferences an unset
# $target_platform and would abort the run right before Minecraft starts.
set -eo pipefail

DISPLAY_NUM=${DISPLAY_NUM:-1}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-760}
PORT=${PORT:-6080}
REPO=/mnt/c/Users/wangp/OneDrive/Desktop/dreamer/MineStudio
INVENTORY=${INVENTORY:-oak_planks:64,stone_axe:1,crafting_table:1}

cleanup() {
  echo "shutting down..."
  pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
  pkill -f "x11vnc.*:${DISPLAY_NUM}" 2>/dev/null || true
  pkill -f "websockify.*${PORT}" 2>/dev/null || true
  pkill -f "scripts/play.py" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
cleanup
sleep 1

echo "[1/4] virtual display :${DISPLAY_NUM} at ${WIDTH}x${HEIGHT}"
# WSLg creates /tmp/.X11-unix with a mode Xvfb refuses to bind under, and the
# failure is only visible as "failed to create listener" several lines up from
# where things actually break.
if [ "$(stat -c %a /tmp/.X11-unix 2>/dev/null)" != "1777" ]; then
  sudo chmod 1777 /tmp/.X11-unix
fi
Xvfb ":${DISPLAY_NUM}" -screen 0 "${WIDTH}x${HEIGHT}x24" -nolisten tcp &
sleep 3
if ! DISPLAY=":${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; then
  echo "Xvfb did not come up on :${DISPLAY_NUM}" >&2
  exit 1
fi

echo "[2/4] x11vnc"
# -forever so closing the tab does not kill the session; -shared so you can
# reconnect without booting the previous viewer.
x11vnc -display ":${DISPLAY_NUM}" -forever -shared -nopw -quiet -rfbport 5900 &
sleep 2

echo "[3/4] noVNC on http://localhost:${PORT}/vnc.html"
websockify --web=/usr/share/novnc "${PORT}" localhost:5900 &
sleep 2

echo "[4/4] Minecraft"
# shellcheck disable=SC1091
source /home/wangp/miniforge3/etc/profile.d/conda.sh
conda activate minestudio
export DISPLAY=":${DISPLAY_NUM}"
export MINESTUDIO_DIR=/home/wangp/.minestudio
export PYTHONPATH="${REPO}"
export PYTHONUNBUFFERED=1
cd "${REPO}"

echo
echo "================================================================"
echo "  open in Windows:   http://localhost:${PORT}/vnc.html"
echo "  click Connect, then click the game and press C to grab the mouse"
echo "================================================================"
echo

python scripts/play.py --inventory "${INVENTORY}" 2>&1 |
  grep -avE "^(ERROR|WARNING):minestudio|Gym has been|Please upgrade|Users of this|See the migration"
