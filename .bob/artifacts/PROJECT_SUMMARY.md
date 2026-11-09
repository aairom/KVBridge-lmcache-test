# KVBridge - Project Summary (artifact)

**Goal:** research LMCache, assess Ollama / llama.cpp compatibility, and build an end-to-end
demo with a GUI showing KV-cache management and inference gains. Engines are pluggable and fully
configurable from the GUI or `.env`: local (llama.cpp, Ollama) and cloud (OpenAI-compatible).

## Verdict

| Engine | Official LMCache | Result in this project |
| --- | --- | --- |
| llama.cpp | no | **Full integration** via slot save/restore (real KV tensors) |
| Ollama | no (issue #923, "not planned") | **Shadow mode** (index + accounting; Ollama's cache does the work) |
| Cloud (OpenAI-compatible) | no | **Shadow mode** (index + accounting; provider `cached_tokens` is ground truth; token budget, retries, key redaction) |

Stock LMCache cannot run here (Linux/CUDA wheel, no connector). KVBridge re-implements its
concepts: 256-token chunks, rolling blake3 prefix hashes (bit-compatible), tiered RAM/disk
store with LRU, persistence across engine restarts, and the same metrics vocabulary.

## Deliverables

* `kvbridge/` - config, hashing, store, backends (engine registry), engine, metrics, scenarios, charts, Gradio app
* `tests/` - 188 tests (184 unit incl. faked cloud + 4 opt-in live: llama.cpp, Ollama, cloud path); 5 skipped without LMCache/live engines
* `scripts/` - setup, start (detached, prints URL), stop (graceful), cleanup, test, install_lmcache
* `README.md`, `Docs/Quickstart.md`, `Docs/Architecture.md`, `Docs/Research.md`
* `.env.example` documents every `KVB_*` setting (engine selection, llama.cpp, Ollama, cloud, shared, cache, GUI)
* Screenshots in this folder: dashboard, scenarios, playground, cache inspector, cloud engine settings (`gui_engine_cloud.png`)

## Configuration model

Priority: GUI > process env > `.env` > defaults. `KVB_ENGINE=llamacpp|ollama|cloud`. No engine value is
hard-coded (search paths, timeouts, reserved ports, keep-alive, seed are settings). Secrets are masked,
redacted from errors, and written to `.env` only on request (mode 0600).

## Measured (Apple Silicon, llama3.2 3B Q4_K_M)

| Metric | Value |
| --- | --- |
| Doc Q&A avg TTFT: cold / lmcache | 4,554 ms / 999 ms (4.6x); 44-53 ms once cached |
| Chunk hit rate / token reuse (12 requests) | 75 % / 58 % |
| Restart persistence (~800 tokens): cold vs L2 hit | 495 ms vs 13 ms |
| Snapshot size / save-restore time | ~70 MB per 600 tokens / 5-14 ms |

## Known limitations

Whole-slot snapshots (not per-chunk sharing), Ollama and cloud observational only (cloud: prompts leave
the machine, cost and provider cache TTL/minimum length apply), no remote tiers,
snapshots valid per model file and context size.
