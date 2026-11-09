# Quickstart

Get KVBridge running in about five minutes.

## 1. Prerequisites

| Need | Notes |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) + Python 3.12 | `uv` creates the virtual environment |
| One engine | **Ollama** with a model (`ollama pull llama3.2`), **or** a `llama-server` binary and a GGUF, **or** an OpenAI-compatible **cloud** endpoint with an API key |
| macOS / Linux | Tested on macOS (Apple Silicon) |

Ollama.app ships a `llama-server` binary. KVBridge finds it automatically and
resolves the GGUF blob of your Ollama model, so a plain Ollama install is enough
for the full llama.cpp integration. The cloud engine needs no local software at all.

## 2. Install

```bash
scripts/setup.sh
```

This creates `.venv` (uv), installs `requirements.txt` and copies `.env.example`
to `.env`. Edit `.env` to choose the engine (`KVB_ENGINE=llamacpp|ollama|cloud`) and change
its settings, ports, chunk size or tier sizes - or skip editing and use the GUI (section 5).
Re-run `scripts/setup.sh` after changing `requirements.txt`.

## 3. Start (single command)

```bash
scripts/start.sh
# KVBridge is running: http://127.0.0.1:7871
```

The GUI runs detached; logs go to `output/logs/kvbridge.log`.

## 4. First tour

1. **1 - Engine & Cache**: choose an engine (*llama.cpp*, *Ollama* or *Cloud*), review the
   settings group that appears for it, click **Connect**. The status line shows model,
   context size, integration level, hasher and whether prompts stay on this machine.
2. **4 - Scenarios**: keep *Document Q&A*, click **Run scenario**. Each question runs
   in `cold`, `native` and `lmcache` mode and the summary reports the TTFT speed-up.
3. **2 - Dashboard**: hit rate, token reuse, TTFT charts and tier usage.
4. **5 - Cache Inspector**: see the stored snapshot, click **Simulate engine restart**
   (L1 RAM cleared, L2 disk kept), then run the **Playground** with `lmcache` mode:
   the prefix is restored from disk in tens of milliseconds.
5. Switch to **Ollama** or **Cloud** to see the accounting-only (shadow) integration.

## 5. Configure engines: GUI or `.env`

Precedence, highest first: **GUI > process environment > `.env` > defaults**. Every setting
is documented in [`.env.example`](../.env.example); the full table is in the README.

**In the GUI** (tab 1): pick the engine, edit its fields, **Connect**. Invalid values are
reported per field and nothing changes until they are fixed. **Show effective configuration**
prints the active settings with secrets masked; **Save settings to .env** persists them
(comments preserved, file mode `0600`). The API key is written only if you tick
*Also write the API key*.

**In `.env`**, pick one of these blocks:

```bash
# A. llama.cpp - full integration
KVB_ENGINE=llamacpp
KVB_OLLAMA_MODEL=llama3.2:latest

# B. Ollama - shadow mode
KVB_ENGINE=ollama
KVB_OLLAMA_URL=http://127.0.0.1:11434
KVB_OLLAMA_MODEL=llama3.2:latest

# C. Hosted provider - key read from your shell, with a spend guard
KVB_ENGINE=cloud
KVB_CLOUD_PROVIDER=openai          # presets: openai groq openrouter together mistral deepseek
KVB_CLOUD_MODEL=gpt-4o-mini        #          vllm-local lmstudio-local ollama-local-openai
KVB_CLOUD_API_KEY_ENV=OPENAI_API_KEY
KVB_CLOUD_TOKEN_BUDGET=200000

# D. Free smoke test of the cloud path (no account needed)
KVB_ENGINE=cloud
KVB_CLOUD_PROVIDER=ollama-local-openai
KVB_CLOUD_MODEL=llama3.2:latest
```

For any other service set `KVB_CLOUD_BASE_URL` (ending in `/v1`) instead of a provider.
Use `KVB_CLOUD_API_STYLE=completions` for legacy endpoints and
`KVB_CLOUD_MAX_TOKENS_FIELD=max_completion_tokens` for newer OpenAI models.

**Cloud caveats.** Prompts leave your machine and are billed. KVBridge cannot move KV
tensors on a hosted API, so it runs in shadow mode and reports the provider's own
`cached_tokens`; providers cache only long prompts (about 1,024+ tokens) for a limited time,
so use the long sample document. `cold` mode adds a random nonce to defeat the provider cache.
Set `KVB_CLOUD_TOKEN_BUDGET` to cap spend.

## 6. Use your own documents

Put `.txt` / `.md` files in `input/` (subfolders are scanned). They replace the
built-in sample as the shared context.

## 7. Stop and clean up

```bash
scripts/stop.sh      # graceful stop, also the managed llama-server
scripts/cleanup.sh   # removes .venv, node_modules and __pycache__
```

No container images are pulled, so there is nothing else to remove.

## 8. Tests

```bash
scripts/test.sh           # 188 tests (5 skipped without LMCache/live engines); cloud is faked
scripts/test.sh --live    # adds tests against real llama.cpp, Ollama and the cloud path
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "llama-server not found" | Set `KVB_LLAMA_SERVER_BIN`, or `KVB_LLAMA_URL` to attach to a running one |
| "model not found" | Set `KVB_MODEL_PATH` to a GGUF, or check `KVB_OLLAMA_MODEL` with `ollama list` |
| "cloud: missing API key / base URL / model" | Fill the fields in the GUI or set `KVB_CLOUD_*`; `KVB_CLOUD_API_KEY_ENV` must name an exported variable |
| HTTP 400 "unsupported parameter" | Set `KVB_CLOUD_MAX_TOKENS_FIELD=max_completion_tokens` and/or `KVB_CLOUD_SEND_TEMPERATURE=false` |
| HTTP 429 | Retried automatically (honours `Retry-After`); lower the load or change provider |
| Cloud reports 0 cached tokens | Prefix shorter than the provider minimum, cache expired, or `cold` mode; use a long shared prefix |
| "http with API key refused" | Use `https://`; plain http is only allowed for loopback hosts |
| Port in use | Change `KVB_GUI_PORT` / `KVB_MANAGED_PORT` (avoid 5000, 8080, 9931, 11434) |
| Cold and lmcache look the same | Keep `KVB_CACHE_RAM_MB=0`; use a larger context (Scenarios slider) |
| Ollama shows no byte savings | Expected: shadow mode never moves KV; see `Docs/Research.md` |
| Cache ignored after changing chunk size | Intended: the persisted index is invalidated (see log warning) |
