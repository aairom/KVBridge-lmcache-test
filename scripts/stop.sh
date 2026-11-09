#!/usr/bin/env bash
# Gracefully stop the GUI and any managed llama-server it started.
set -uo pipefail
cd "$(dirname "$0")/.."

stop_pid() {  # $1=pidfile $2=name
  [ -f "$1" ] || return 0
  pid="$(cat "$1")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null; echo "$2 force-killed"; else echo "$2 stopped"; fi
  fi
  rm -f "$1"
}

stop_pid output/kvbridge.pid "KVBridge GUI"
# The GUI stops its managed llama-server on exit; clean up if it was orphaned.
stop_pid output/llama_server.pid "managed llama-server"
