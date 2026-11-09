"""The cache-aware inference pipeline.

Request flow (llama.cpp, full integration)::

    prompt --tokenize--> tokens --chunk+hash--> [h0 h1 h2 ...]
                                                   |
                    store.lookup (longest prefix) <-+
                          |hit                |miss
            fetch snapshot (L1 RAM / L2 disk)  |
            restore into engine slot           |
                          +---------+----------+
                                    v
                 engine.generate (prefills only the *missing* suffix)
                                    v
                 save slot -> store.put (only when new chunks were computed)

Three execution modes make the benefit measurable:

``cold``    prefix reuse disabled - every request prefills the whole prompt.
``native``  the engine's own slot/prefix cache only (baseline a user has today).
``lmcache`` native cache is dropped first (modelling eviction by another
            session, a restart, ...) and the KV layer restores state instead.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .backends import Backend, BackendError, GenResult
from .config import Settings
from .hashing import Hasher, make_hasher
from .metrics import Metrics, RequestRecord
from .store import TieredStore

log = logging.getLogger(__name__)

MODES = ("cold", "native", "lmcache")


class CachedEngine:
    """Binds an engine :class:`Backend`, a :class:`TieredStore` and metrics."""

    def __init__(self, settings: Settings, backend: Backend,
                 store: Optional[TieredStore] = None, metrics: Optional[Metrics] = None) -> None:
        self.s = settings
        self.backend = backend
        self.metrics = metrics or Metrics()
        self._lock = threading.RLock()
        self.hasher: Hasher = self._new_hasher()
        self.store = store or TieredStore(
            settings.l1_bytes, settings.l2_bytes, settings.store_dir / self._namespace_dir(),
            settings.chunk_size, self.metrics, settings.write_through)

    # ----------------------------------------------------------- construction
    def _namespace(self) -> str:
        return f"{self.backend.name}|{self.backend.model_id}"

    def _namespace_dir(self) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in self._namespace())
        return safe[:60]

    def _new_hasher(self) -> Hasher:
        return make_hasher(self.s.chunk_size, self._namespace(), self.s.use_lmcache_hasher)

    def start(self) -> None:
        """Start/attach the engine and refresh namespace-dependent state."""
        self.backend.start()
        self.hasher = self._new_hasher()
        self.store.dir = self.s.store_dir / self._namespace_dir()
        if self.store.l2_cap:
            self.store.dir.mkdir(parents=True, exist_ok=True)
            if self.store.persist:
                self.store._load_index()

    def stop(self) -> None:
        self.backend.stop()

    # -------------------------------------------------------------- inference
    def infer(self, prompt: str, mode: str = "lmcache", label: str = "",
              n_predict: Optional[int] = None, temperature: Optional[float] = None) -> RequestRecord:
        """Run one generation in *mode* and return a populated record."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n_predict = n_predict if n_predict is not None else self.s.n_predict
        temperature = temperature if temperature is not None else self.s.temperature
        rec = RequestRecord(label=label, mode=mode, engine=self.backend.name)
        with self._lock:
            try:
                if self.backend.supports_kv_export:
                    self._infer_full(prompt, rec, n_predict, temperature)
                else:
                    self._infer_shadow(prompt, rec, n_predict, temperature)
            except BackendError as exc:
                rec.error = str(exc)
                log.error("request '%s' failed: %s", label, exc)
            self.metrics.record_request(rec)
        return rec

    @staticmethod
    def _fill(rec: RequestRecord, res: GenResult) -> None:
        rec.prompt_tokens = res.prompt_tokens
        rec.computed_tokens = res.computed_tokens
        rec.server_cached_tokens = res.cached_tokens
        rec.ttft_ms, rec.total_ms = res.ttft_ms, res.total_ms
        rec.prefill_ms, rec.decode_tps = res.prefill_ms, res.decode_tps
        rec.text = res.text.strip()

    def _infer_full(self, prompt: str, rec: RequestRecord, n_predict: int, temperature: float) -> None:
        """llama.cpp path: real KV save/restore."""
        b = self.backend
        tokens = b.tokenize(prompt)
        if rec.mode == "cold":
            b.reset_native_cache()
            self._fill(rec, b.generate(prompt, tokens, n_predict, False, temperature))
            return
        if rec.mode == "native":
            self._fill(rec, b.generate(prompt, tokens, n_predict, True, temperature))
            return

        # ---- lmcache mode
        hashes = self.hasher.chunk_hashes(tokens)
        rec.chunks_total = len(hashes)
        rec.chunk_status = self.store.chunk_status(hashes)
        b.reset_native_cache()                       # engine lost its own KV
        match = self.store.lookup(hashes)
        rec.lmcache_match_tokens, rec.chunks_hit, rec.tier = match.tokens, match.chunks, match.tier
        if match.entry is not None:
            t0 = time.perf_counter()
            try:
                blob = self.store.fetch(match.entry)
                n = b.restore_state(blob)
                rec.restore_ms = (time.perf_counter() - t0) * 1000
                self.metrics.inc("restores")
                self.metrics.inc("restore_bytes", len(blob))
                log.debug("restored %d tokens from %s", n, match.tier)
            except (BackendError, KeyError, OSError) as exc:
                log.warning("restore failed (%s); falling back to recompute", exc)
                rec.chunks_hit, rec.lmcache_match_tokens, rec.tier = 0, 0, "none"
                b.reset_native_cache()
        res = b.generate(prompt, tokens, n_predict, True, temperature)
        self._fill(rec, res)
        rec.lmcache_restored_tokens = res.cached_tokens if rec.chunks_hit else 0

        if len(hashes) > rec.chunks_hit:             # new chunks were computed -> store
            t0 = time.perf_counter()
            try:
                blob, n_saved = b.save_state()
                self.store.put(hashes, len(hashes) * self.s.chunk_size, blob,
                               note=f"{n_saved} tok snapshot")
            except BackendError as exc:
                log.warning("could not store KV snapshot: %s", exc)
            rec.save_ms = (time.perf_counter() - t0) * 1000

    def _infer_shadow(self, prompt: str, rec: RequestRecord, n_predict: int, temperature: float) -> None:
        """Ollama path: the KV layer indexes/predicts; Ollama's cache does the work."""
        b = self.backend
        if rec.mode == "cold":
            b.reset_native_cache()
        tokens = b.tokenize(prompt)
        hashes = self.hasher.chunk_hashes(tokens)
        rec.shadow = True
        rec.chunks_total = len(hashes)
        rec.chunk_status = self.store.chunk_status(hashes)
        match = self.store.lookup(hashes) if rec.mode == "lmcache" else None
        res = b.generate(prompt, None, n_predict, True, temperature)
        self._fill(rec, res)
        if match is not None:
            rec.chunks_hit, rec.tier = match.chunks, match.tier
            scale = (res.prompt_tokens / len(tokens)) if tokens else 1.0
            rec.lmcache_match_tokens = int(match.tokens * scale)
            rec.lmcache_restored_tokens = min(res.cached_tokens, rec.lmcache_match_tokens)
            if match.entry is not None:
                match.entry.hits += 1
                match.entry.last_used = time.time()
        if rec.mode == "lmcache" and len(hashes) > rec.chunks_hit:
            scale = (res.prompt_tokens / len(tokens)) if tokens else 1.0
            size = int(len(hashes) * self.s.chunk_size * scale * max(b.kv_bytes_per_token, 1))
            self.store.put(hashes, len(hashes) * self.s.chunk_size, None,
                           note="shadow (accounting only)", virtual_size=size)

    # ---------------------------------------------------------------- control
    def simulate_engine_restart(self) -> str:
        """Wipe the engine's KV state and the RAM tier, keep the disk tier."""
        with self._lock:
            if hasattr(self.backend, "restart") and not getattr(self.backend, "external", False):
                self.backend.restart()
                how = "llama-server restarted"
            else:
                self.backend.reset_native_cache()
                how = "engine KV cache dropped"
            self.store.clear("l1")
            return f"{how}; L1 (RAM) cleared, L2 (disk) kept"

    def reset(self, wipe_store: bool = True) -> None:
        with self._lock:
            self.metrics.reset()
            if wipe_store:
                self.store.clear("all")
            try:
                self.backend.reset_native_cache()
            except BackendError as exc:
                log.warning("engine reset failed: %s", exc)

    def snapshot(self) -> dict:
        """Everything the dashboard needs in one dict."""
        st = self.store.stats()
        return {"metrics": self.metrics.snapshot(), "store": st,
                "engine": self.backend.describe(), "hasher": self.hasher.impl,
                "chunk_size": self.s.chunk_size}
