#!/usr/bin/env bash
# Run the unit tests; pass --live to also run tests against real engines.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
if [ "${1:-}" = "--live" ]; then export KVB_LIVE=1; shift; fi
exec .venv/bin/python -m pytest "$@"
