# Quickstart

Get KVBridge running in about five minutes.

## 1. Prerequisites

| Need | Notes |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) + Python 3.12 | `uv` creates the virtual environment |
| One engine | **Ollama** with a model (`ollama pull llama3.2`) **or** a `llama-server` binary and a GGUF |
| macOS / Linux | Tested on macOS (Apple Silicon) |

Ollama.app ships a `llama-server` binary. KVBridge finds it automatically and
resolves the GGUF blob of your Ollama model, so a plain Ollama install is enough
for the full llama.cpp integration.

## 2. Install

```bash
scripts/setup.sh
```

This creates `.venv` (uv), installs `requirements.txt` and copies `.env.example`
to `.env`. Edit `.env` to change ports, model, chunk size or tier sizes. Re-run
`scripts/setup.sh` after changing `requirements.txt`.

## 3. Start (single command)

```bash
scripts/start.sh
# KVBridge is running: http://127.0.0.1:7871
```

The GUI runs detached; logs go to `output/logs/kvbridge.log`.

## 4. First tour

1. **1 - Engine & Cache**: choose *llama.cpp (full integration)*, click **Connect**.
   The status line shows model, context size, integration level and hasher.
2. **4 - Scenarios**: keep *Document Q&A*, click **Run scenario**. Each question runs
   in `cold`, `native` and `lmcache` mode and the summary reports the TTFT speed-up.
3. **2 - Dashboard**: hit rate, token reuse, TTFT charts and tier usage.
4. **5 - Cache Inspector**: see the stored snapshot, click **Simulate engine restart**
   (L1 RAM cleared, L2 disk kept), then run the **Playground** with `lmcache` mode:
   the prefix is restored from disk in tens of milliseconds.
5. Switch to **Ollama (shadow mode)** to see the accounting-only integration.

## 5. Use your own documents

Put `.txt` / `.md` files in `input/` (subfolders are scanned). They replace the
built-in sample as the shared context.

## 6. Stop and clean up

```bash
scripts/stop.sh      # graceful stop, also the managed llama-server
scripts/cleanup.sh   # removes .venv, node_modules and __pycache__
```

No container images are pulled, so there is nothing else to remove.

## 7. Tests

```bash
scripts/test.sh           # 125 unit tests, no engine required
scripts/test.sh --live    # adds tests against real llama.cpp and Ollama
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "llama-server not found" | Set `KVB_LLAMA_SERVER_BIN`, or `KVB_LLAMA_URL` to attach to a running one |
| "model not found" | Set `KVB_MODEL_PATH` to a GGUF, or check `KVB_OLLAMA_MODEL` with `ollama list` |
| Port in use | Change `KVB_GUI_PORT` / `KVB_MANAGED_PORT` (avoid 5000, 8080, 9931, 11434) |
| Cold and lmcache look the same | Keep `KVB_CACHE_RAM_MB=0`; use a larger context (Scenarios slider) |
| Ollama shows no byte savings | Expected: shadow mode never moves KV; see `Docs/Research.md` |
| Cache ignored after changing chunk size | Intended: the persisted index is invalidated (see log warning) |
