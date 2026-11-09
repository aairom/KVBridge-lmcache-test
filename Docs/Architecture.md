# Architecture

KVBridge puts an LMCache-style KV cache layer between a UI and a local inference
engine. Details of the research behind it are in [`Research.md`](Research.md).

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

    subgraph ENG["Engines (backends.py)"]
        LC[LlamaCppBackend<br/>llama-server :8089]
        OL[OllamaBackend<br/>:11434]
    end

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

## Shadow mode (Ollama)

Ollama exposes no tokenizer and no KV import/export, so the layer cannot move
tensors. Instead:

```mermaid
flowchart LR
    P[Prompt] --> T[pseudo_tokenize<br/>CRC32 pieces]
    T --> H[Chunk hashes]
    H --> S{In index?}
    S -->|yes| Pred[Predicted reuse]
    S -->|no| New[Index chunks<br/>virtual_size entries in L1]
    P --> O[Ollama /api/generate]
    O --> GT["prompt_eval_cached_count<br/>(ground truth)"]
    Pred --> M[Metrics]
    New --> M
    GT --> M
```

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | `Settings` from `KVB_*` env vars / `.env`; no hard-coded values |
| `hashing.py` | LMCache-compatible rolling prefix hash; optional adapter to real `TokenHasher` |
| `store.py` | Two-tier snapshot store, LRU, write-through, persistence, resize, eviction counters |
| `backends.py` | `LlamaCppBackend` (slot API), `OllamaBackend` (shadow), binary/model discovery |
| `engine.py` | `CachedEngine`: modes `cold`/`native`/`lmcache`, restart simulation, snapshots |
| `metrics.py` | Thread-safe counters, `RequestRecord`, JSON export |
| `scenarios.py` | Document Q&A, multi-turn, shared prefix, restart; reads `input/` recursively |
| `charts.py` | Plotly figures and tables (no Gradio dependency, unit-testable) |
| `app.py` | Gradio Blocks UI; one `AppState` shared by callbacks |

## Design decisions

* **Namespace per model.** Hashes are seeded with `engine|model`, and the store lives in a per-model directory, so snapshots never leak between models.
* **Whole-slot snapshots.** llama.cpp can only serialise a complete slot; an entry is indexed by its chunk-hash chain, so a request sharing the first *k* chunks still hits.
* **Graceful degradation.** A failed restore falls back to full prefill; a failed save is logged and skipped; request errors are recorded on the `RequestRecord` instead of raising into the UI.
* **Locking.** A single `RLock` serialises engine access, because there is one llama.cpp slot.
* **Local only.** Binds to `127.0.0.1`; Gradio analytics are disabled.
