#!/usr/bin/env bash
# OPTIONAL: build the real LMCache native token hasher on macOS (no GPU) so that
# KVB_USE_LMCACHE_HASHER=true uses lmcache.TokenHasher instead of the built-in copy.
# Usage: scripts/install_lmcache.sh /path/to/LMCache-source   (an unpacked LMCache release/dev tree)
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?usage: $0 /path/to/LMCache-source}"
[ -f "$SRC/pyproject.toml" ] || { echo "$SRC does not look like an LMCache checkout" >&2; exit 1; }
uv pip install --python .venv/bin/python torch numba cachetools blake3 grpcio-tools numpy setuptools-scm packaging
NO_GPU_EXT=1 SETUPTOOLS_SCM_PRETEND_VERSION="${LMCACHE_VERSION:-0.5.7}" \
  uv pip install --python .venv/bin/python --no-deps --no-build-isolation "$SRC"
.venv/bin/python -c "from lmcache.v1.multiprocess.token_hasher import TokenHasher; print('real LMCache TokenHasher OK')"
