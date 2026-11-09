# Architecture

KVBridge puts an LMCache-style KV cache layer between a UI and an inference engine.
The engine is **pluggable**: local (llama.cpp, Ollama) or cloud (any OpenAI-compatible
API), selected and configured in the GUI or in `.env`. No engine setting is hard-coded. Details of the research behind it are in [`Research.md`](Research.md).

## Component flow

```mermaid
flowchart TB
    subgraph GUI["Gradio GUI (kvbridge/app.py)"]
        T1[Engine and Cache]
        T2[Dashboard]
        T3[Playground]
        T4[Scenarios]
        T5[Cache Inspector]
    end

    subgraph CORE["Cache layer"]
        CE[CachedEngine<br/>engine.py]
        HS[ChunkHasher<br/>hashing.py]
        ST[TieredStore<br/>store.py]
        MT[Metrics<br/>metrics.py]
        SC[Scenarios<br/>scenarios.py]
        CH[Charts<br/>charts.py]
    end

    subgraph TIERS["Storage tiers"]
        L1[(L1 RAM<br/>LRU)]
        L2[(L2 Disk<br/>persisted index)]
    end

    subgraph CFG["Configuration (config.py)"]
        SET["Settings<br/>GUI > env > .env > defaults"]
        ENVF[".env / KVB_* variables"]
        PRE["PROVIDER_PRESETS"]
    end

    subgraph ENG["Engines (backends.py)"]
        REG{{"Engine registry<br/>register_engine / resolve_engine"}}
        LC[LlamaCppBackend<br/>kind=local, KV export]
        OL[OllamaBackend<br/>kind=local, shadow]
        CL[CloudBackend<br/>kind=cloud, shadow]
    end

    T1 -->|"apply(form) / save to .env"| SET
    ENVF --> SET
    PRE --> SET
    SET -->|"validate() via check_settings"| REG
    REG -->|"make_backend(engine)"| CE
    T1 --> CE
    T3 --> CE
    T4 --> SC --> CE
    T5 --> ST
    CE --> HS
    CE --> ST
    CE --> MT
    ST --> L1
    ST --> L2
    CE -->|"tokenize, complete, slots save/restore"| LC
    CE -->|"/api/generate, shadow accounting"| OL
    CE -->|"/v1/chat/completions, cached_tokens"| CL
    MT --> CH --> T2
    SC --> IN["input/ documents"]
    MT --> OUT["output/ JSON exports"]
```

## Request pipeline (llama.cpp, full integration)

```mermaid
flowchart TD
    A[Prompt] --> B["Tokenize (POST /tokenize)"]
    B --> E{Mode?}
    E -->|cold| F["Erase slot, cache_prompt=false<br/>full prefill"]
    E -->|native| G["Keep slot, cache_prompt=true<br/>engine prefix reuse only"]
    E -->|lmcache| C["Split into 256-token chunks,<br/>drop partial tail, rolling blake3 hashes"]
    C --> R[Erase slot: engine lost its own KV]
    R --> H[Longest-prefix lookup in store]
    H -->|hit L1 or L2| I["Fetch snapshot and restore slot<br/>(on failure: recompute)"]
    H -->|miss| K
    I --> K["Generate (POST /completion)<br/>prefills only the missing suffix"]
    F --> L[RequestRecord -> Metrics]
    G --> L
    K --> M{More chunks than were hit?}
    M -->|yes| N["Save slot -> put in L1, write-through L2"]
    M -->|no| L
    N --> L
```

`cold` and `native` never read or write the store; they exist as baselines.

## Shadow mode (Ollama and cloud)

Ollama and hosted APIs expose no tokenizer and no KV import/export, so the layer cannot move
tensors. Instead it indexes chunk hashes and compares its prediction with the engine's own
report (`prompt_eval_cached_count` for Ollama, `usage.prompt_tokens_details.cached_tokens`
or `prompt_cache_hit_tokens` for cloud):

```mermaid
flowchart LR
    P[Prompt] --> T[pseudo_tokenize<br/>CRC32 pieces]
    T --> H[Chunk hashes]
    H --> S{In index?}
    S -->|yes| Pred[Predicted reuse]
    S -->|no| New[Index chunks<br/>virtual_size entries in L1]
    P --> O["Ollama /api/generate<br/>or cloud /v1/chat/completions"]
    O --> GT["prompt_eval_cached_count / cached_tokens<br/>(ground truth)"]
    Pred --> M[Metrics]
    New --> M
    GT --> M
```

## Engine registry and configuration flow

```mermaid
sequenceDiagram
    participant U as GUI tab 1 / .env
    participant C as Settings
    participant R as Engine registry
    participant B as Backend
    U->>C: form_to_values -> apply(dict)
    C->>C: coerce types, build candidate
    C->>R: resolve_engine(engine).cls.check_settings
    R-->>C: problems (no network I/O)
    alt problems
        C-->>U: per-field messages, settings unchanged
    else valid
        C-->>U: committed (atomic)
        U->>R: make_backend(engine, settings)
        R->>B: instantiate and start()
        B-->>U: describe(): kind, location, supports_kv_export
    end
    U->>C: Save settings to .env (update_env_file, 0600, atomic)
```

* **Registry.** `EngineSpec(name, label, cls, aliases)`; `register_engine` adds one, `resolve_engine` normalises names (`Llama.cpp` = `llamacpp`), `engine_choices()` feeds the GUI Radio. A new engine is one subclass plus one `register_engine` call.
* **Backend contract.** `kind` (`local`/`cloud`), `supports_kv_export`, `describe()`, `check_settings()` classmethod, plus `start/stop/tokenize/generate`. The GUI and scenarios read these flags instead of testing engine names.
* **Priority.** GUI values beat process env, which beats `.env`, which beats defaults. `apply()` is atomic: coercion plus `validate()` run on a candidate copy.
* **Secrets.** `SECRET_FIELDS` are masked by `to_display()`, excluded from `env_values()` by default, redacted from backend error text, and written to `.env` only on explicit request.

### Cloud engine behaviour

* OpenAI-compatible `chat` or `completions` style; the max-tokens field name is configurable.
* Streams the answer to measure TTFT; requests usage in the stream to read `cached_tokens`.
* Retries 429/502/503/504 (3 attempts, honours `Retry-After`).
* `cold` mode prepends a one-shot random nonce so the provider cannot reuse its prefix cache.
* `cloud_token_budget` is a soft cap on prompt tokens per connection; `http://` plus a key is refused for non-loopback hosts.
* Privacy: prompts leave the machine; the status line says so.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | `Settings` from `KVB_*` env vars / `.env` / GUI (`apply`, `validate`, `to_display`, `update_env_file`); provider presets; no hard-coded values |
| `hashing.py` | LMCache-compatible rolling prefix hash; optional adapter to real `TokenHasher` |
| `store.py` | Two-tier snapshot store, LRU, write-through, persistence, resize, eviction counters |
| `backends.py` | Engine registry; `LlamaCppBackend` (slot API), `OllamaBackend` (shadow), `CloudBackend` (OpenAI-compatible, shadow); binary/model discovery |
| `engine.py` | `CachedEngine`: modes `cold`/`native`/`lmcache`, restart simulation, snapshots |
| `metrics.py` | Thread-safe counters, `RequestRecord`, JSON export |
| `scenarios.py` | Document Q&A, multi-turn, shared prefix, restart; reads `input/` recursively |
| `charts.py` | Plotly figures and tables (no Gradio dependency, unit-testable) |
| `app.py` | Gradio Blocks UI generated from the registry; per-engine settings groups; one `AppState` shared by callbacks |

## Design decisions

* **Namespace per model.** Hashes are seeded with `engine|model`, and the store lives in a per-model directory, so snapshots never leak between models.
* **Whole-slot snapshots.** llama.cpp can only serialise a complete slot; an entry is indexed by its chunk-hash chain, so a request sharing the first *k* chunks still hits.
* **Graceful degradation.** A failed restore falls back to full prefill; a failed save is logged and skipped; request errors are recorded on the `RequestRecord` instead of raising into the UI.
* **Locking.** A single `RLock` serialises engine access, because there is one llama.cpp slot.
* **Local GUI.** The GUI binds to `127.0.0.1` and Gradio analytics are disabled. Only the cloud engine sends data off-machine, and only to the URL you configure.
* **Configuration over code.** Engine choice, paths, ports, timeouts, keep-alive and seed are settings, not constants; `.env.example` is tested to list every field.
