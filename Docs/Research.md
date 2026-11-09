# Research: LMCache x Ollama x llama.cpp x cloud engines

This page records what was researched, what is technically feasible, which
challenges came up, and how KVBridge solves them. It is also shown in the
**About** tab of the GUI.

## 1. What LMCache is

[LMCache](https://github.com/LMCache/LMCache) (Apache-2.0) is a *KV cache
management layer for LLM inference*. Instead of every request re-computing the
attention keys/values (the "KV cache") for a prompt prefix, LMCache stores them
outside the engine and re-injects them on demand.

| Capability | How LMCache does it |
| --- | --- |
| Reuse | Chunk-level prefix reuse: the token stream is split into fixed chunks (default **256 tokens**), each identified by a **rolling prefix hash** (blake3). A hit on chunk *i* proves the whole prefix up to *i* is identical. |
| Tiered storage | GPU/CPU RAM -> local disk -> Redis / S3 / Mooncake / NIXL / GDS. LRU eviction between tiers. |
| Engine independence | A cache daemon / connector that does not share fate with the engine: the cache survives engine restarts. |
| Scale-out | Cross-instance sharing, prefill/decode disaggregation. |
| Observability | Hit rate, token reuse, tier usage, eviction counters. |

Value proposition: lower time-to-first-token (TTFT) and GPU cycles for
workloads with shared prefixes (RAG documents, system prompts, multi-turn chat,
agents).

## 2. Compatibility matrix

| Engine | Official LMCache support | KV tensors accessible from outside? | What KVBridge does |
| --- | --- | --- | --- |
| vLLM | Yes (connector) | Yes | Out of scope (use upstream LMCache) |
| SGLang / TRT-LLM | Yes | Yes | Out of scope |
| **llama.cpp** (`llama-server`) | **No** | **Yes** - slot save/restore API | **Full integration**: real KV tensors are saved to / restored from the cache layer |
| **Ollama** | **No** (issue #923 closed "not planned") | **No** - no KV export/import, no tokenizer endpoint | **Shadow mode**: the layer indexes chunks, predicts reuse, and uses Ollama's `prompt_eval_cached_count` as ground truth |
| **Cloud APIs** (OpenAI, Groq, OpenRouter, Together, Mistral, DeepSeek...; also vLLM / LM Studio servers) | **No** (vLLM itself is supported, but only when you control the server) | **No** - hosted APIs never expose KV tensors | **Shadow mode** through the OpenAI-compatible REST API; the provider's `cached_tokens` is the ground truth |

Platform finding: the LMCache PyPI wheel is `manylinux`-only and its build
needs CUDA. On macOS `pip install lmcache` fails with *"CUDA_HOME not set"*.
With `NO_GPU_EXT=1` plus extra dependencies the pure-Python / native
`TokenHasher` can be built (`scripts/install_lmcache.sh`), which is how hash
compatibility is verified.

## 3. Feasibility verdict

| Question | Answer |
| --- | --- |
| Can stock LMCache run in front of llama.cpp or Ollama? | **No.** There is no connector, and LMCache targets GPU engines on Linux. |
| Can the *concepts* be reproduced faithfully? | **Yes** for llama.cpp, thanks to `/slots/{id}?action=save|restore`. |
| Can Ollama benefit? | Only partially. Ollama already has an in-process prefix cache, but it hides KV state, so external persistence or sharing is impossible. KVBridge can only *observe and account*. |

| Can a hosted API benefit? | Only by *observation*. Providers run their own prefix cache (automatic above a minimum prompt length, limited lifetime) and report `cached_tokens`; KVBridge shows how well your prompts exploit it and can cap spend with a token budget. |

KVBridge is therefore an **LMCache-style layer** (same chunking, same hashing,
same tier model, same metrics vocabulary) built for local and cloud engines. It is not a
fork or a drop-in replacement of LMCache.

## 4. Approach

1. **Hashing** (`kvbridge/hashing.py`): re-implements LMCache's rolling prefix
   hash (blake3, big-endian `uint32` token packing, partial tail chunk
   discarded). A test checks digests against the real `TokenHasher` when
   `lmcache` is installed, and the app can delegate to it
   (`KVB_USE_LMCACHE_HASHER=true`).
2. **Store** (`kvbridge/store.py`): L1 (RAM) + L2 (disk) with LRU eviction,
   write-through, persisted index and live resizing.
3. **Backends and engine registry** (`kvbridge/backends.py`): `LlamaCppBackend`
   (real tensors), `OllamaBackend` and `CloudBackend` (shadow mode). Engines register
   themselves (`register_engine`), so the GUI choices, config validation and factory
   are all generated from one registry.
3b. **Configuration** (`kvbridge/config.py`): every engine setting lives in `Settings`
   and can come from the GUI, `KVB_*` variables or `.env` (priority in that order).
   Updates are atomic and validated before use; secrets are masked and never logged.
4. **Pipeline** (`kvbridge/engine.py`): tokenize -> chunk-hash -> longest-prefix
   lookup -> restore -> prefill only the missing suffix -> save.
5. **GUI** (`kvbridge/app.py`, Gradio): engine selection and per-engine settings,
   dashboard, playground, scenarios, cache inspector.

Three modes make the benefit measurable: `cold` (no reuse), `native` (engine's
own cache) and `lmcache` (the KV layer).

## 5. Measured results (Apple Silicon, llama3.2 3B Q4_K_M)

Document Q&A scenario, ~1,900-token context, 4 questions per mode:

| Mode | Average TTFT | Notes |
| --- | --- | --- |
| cold | **4,554 ms** | Full prefill for every question |
| native (slot already warm) | 64 ms | Best case: the slot still holds the prefix |
| lmcache | **999 ms** average (first request 3,848 ms, then **44-53 ms**) | Chunk hit rate 75 %, token reuse 58 % over 12 requests |

Other measurements:

* Snapshot of ~600 tokens is about 70 MB; save/restore takes **5-14 ms**.
  A 1,956-token snapshot is 214 MB.
* ~805-token prompt: cold 495 ms, LMCache L1 hit 11 ms, L2 hit **after an
  engine restart** 13 ms.
* Playground after a simulated restart: 1,423 of 1,440 tokens reused from L2,
  TTFT 75 ms.
* Live Ollama shadow-mode test: Ollama's own cache reports the reuse;
  KVBridge accounts for it.

Interpretation: LMCache's advantage over `native` is not raw speed on a single
warm slot. It is that the cache **persists across restarts, survives slot
eviction and can be shared**; `native` only wins while the exact slot is still
resident.

## 6. Challenges and solutions

| Challenge | Solution |
| --- | --- |
| LMCache cannot be installed on macOS (CUDA, manylinux wheel) | Re-implement the algorithm; optional `NO_GPU_EXT=1` build only for hash verification |
| No official llama.cpp / Ollama connector | Use llama.cpp's slot save/restore as the connector; shadow mode for Ollama |
| Ollama has no tokenizer API | `pseudo_tokenize` (CRC32 per word piece) for stable chunk hashing; real counts come from Ollama's `prompt_eval_cached_count` |
| Ollama has no KV export | Shadow entries in L1 use `virtual_size` (accounting only) |
| llama.cpp saves a whole slot, not chunks | One entry = one slot snapshot addressed by its chunk-hash chain; partial-prefix hits are matched by the longest common chunk prefix |
| llama.cpp's own host-RAM prompt cache would hide LMCache's effect | Managed server runs with `--cache-ram 0` (`KVB_CACHE_RAM_MB`) |
| `llama-server` binary and model location | Auto-detect binary (PATH, brew, Ollama.app); resolve the Ollama GGUF blob from its manifest |
| Cache validity after config change | Persisted index is ignored when the chunk size differs (logged warning) |
| Gradio sends analytics | Disabled (`GRADIO_ANALYTICS_ENABLED=False`); only the cloud engine talks to a remote host, and only to the URL you configure |
| Hard-coded engine settings (search paths, timeouts, reserved ports, keep-alive, seed) | Moved into `Settings` with `KVB_*` variables and GUI fields; a test asserts `.env.example` lists every field |
| Supporting many engines without `if engine == ...` chains | Engine registry + `Backend.kind` / `supports_kv_export` / `check_settings`; scenarios and GUI read capabilities, not names |
| Cloud providers differ (`max_tokens` vs `max_completion_tokens`, no `temperature`, completions vs chat) | Configurable API style, max-tokens field name and temperature switch; provider presets for the common services |
| Cloud has no KV export and may cache silently | Shadow mode; `cold` prepends a random nonce so the provider cannot reuse its cache; `cached_tokens` is reported as ground truth |
| Cost and rate limits | `KVB_CLOUD_TOKEN_BUDGET` soft cap; 429/5xx retried with `Retry-After` |
| API-key leakage | Password field never pre-filled; key can come from a named env var; masked in `to_display()`; redacted from errors; excluded from `.env` writes unless explicitly requested; `.env` written atomically with mode 0600; plain `http://` + key refused for non-loopback hosts |
| Partial or invalid GUI edits | `Settings.apply()` validates a candidate copy and commits only if valid |
| Port clashes (5000 AirDrop, 8080, 9931, 11434) | GUI on 7871, managed server on 8089; both configurable |
| Background process dying with the shell | `start.sh` launches via `Popen(start_new_session=True)` |

## 7. Limitations (honest summary)

* KVBridge is **not** the LMCache codebase; it re-implements its concepts and
  hashing.
* Snapshots are whole-slot, not per-chunk, so storage is less deduplicated than
  real LMCache.
* Ollama and cloud integrations are observational only (shadow mode).
* Cloud use sends prompts off the machine, costs money, and is subject to the provider's cache minimum length, TTL and rate limits; KVBridge cannot make the provider cache more.
* Cross-instance sharing, remote tiers (Redis/S3) and GPU memory tiers are not
  implemented.
* A snapshot is only valid for the same model file and context size.

## 8. References

* LMCache repository: https://github.com/LMCache/LMCache
* LMCache issue #923 (Ollama / LM Studio support, closed "not planned")
* OpenAI prompt caching (`usage.prompt_tokens_details.cached_tokens`): https://platform.openai.com/docs/guides/prompt-caching
* llama.cpp server slot API: `POST /slots/{id}?action=save|restore|erase`
