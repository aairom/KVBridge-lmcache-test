#!/usr/bin/env bash
# Remove regenerable artefacts: .venv, node_modules and __pycache__.
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/stop.sh || true
rm -rf .venv .pytest_cache
find . -name node_modules -type d -prune -not -path "./_*" -exec rm -rf {} + 2>/dev/null || true
find . -name __pycache__ -type d -prune -not -path "./_*" -exec rm -rf {} + 2>/dev/null || true
echo "Cleaned .venv, node_modules and __pycache__ (input/, output/ and sources are untouched)."
