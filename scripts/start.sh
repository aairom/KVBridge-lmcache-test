#!/usr/bin/env bash
# Launch the KVBridge GUI detached and print its URL.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -x .venv/bin/python ] || scripts/setup.sh
mkdir -p output/logs
PIDFILE=output/kvbridge.pid
LOG=output/logs/kvbridge.log

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "KVBridge already running (pid $(cat "$PIDFILE"))."
else
  # Start in a NEW session so the GUI survives the calling shell/terminal closing.
  .venv/bin/python - "$LOG" "$PIDFILE" <<'PY'
import subprocess, sys
log, pidfile = sys.argv[1:3]
p = subprocess.Popen([sys.executable, "-m", "kvbridge"], stdout=open(log, "ab"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
open(pidfile, "w").write(str(p.pid))
PY
fi

# Resolve host/port from .env / environment, falling back to the defaults.
get() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- || true; }
HOST="${KVB_GUI_HOST:-$(get KVB_GUI_HOST)}"; HOST="${HOST:-127.0.0.1}"
PORT="${KVB_GUI_PORT:-$(get KVB_GUI_PORT)}"; PORT="${PORT:-7871}"

for _ in $(seq 1 60); do
  if curl -fs -o /dev/null "http://$HOST:$PORT/"; then
    echo "KVBridge is running:  http://$HOST:$PORT   (pid $(cat "$PIDFILE"), log: $LOG)"
    exit 0
  fi
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "KVBridge exited early; see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
  sleep 0.5
done
echo "Timed out waiting for the GUI; see $LOG" >&2
exit 1
