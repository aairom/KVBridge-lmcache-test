#!/usr/bin/env bash
# Create/refresh the virtual environment (uv) and install dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
[ -f .env ] || { cp .env.example .env && echo "Created .env from .env.example"; }
mkdir -p input output
echo "Setup complete. Start the GUI with: scripts/start.sh"
