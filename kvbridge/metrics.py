"""Thread-safe metrics collection for the KV cache layer.

The counters intentionally mirror the vocabulary used by LMCache's own
observability stack (chunk-level hits, token-level reuse, tier usage,
evictions) so the dashboard reads the same way as a production deployment.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional

COUNTERS = (
    "requests",
    "chunks_requested",
    "chunks_hit",
    "tokens_prompt",
    "tokens_reused_lmcache",     # tokens injected from LMCache tiers
    "tokens_reused_native",      # tokens the engine reused on its own
    "tokens_computed",           # tokens actually prefetched (prefilled)
    "stores",
    "store_bytes",
    "restores",
    "restore_bytes",
    "evictions_l1",              # L1 -> L2 demotions / RAM drops
    "evictions_l2",              # entries dropped from the cache entirely
    "rejected",                  # entries too large for any tier
    "errors",
)


@dataclass
class RequestRecord:
    """One generation request, as seen by the cache layer."""

    ts: float = field(default_factory=time.time)
    label: str = ""
    mode: str = "lmcache"                 # cold | native | lmcache
    engine: str = ""
    prompt_tokens: int = 0
    computed_tokens: int = 0              # prefilled by the engine
    server_cached_tokens: int = 0         # engine-reported cache hits
    lmcache_match_tokens: int = 0         # chunk-aligned match found in tiers
    lmcache_restored_tokens: int = 0      # tokens actually injected
    chunks_total: int = 0
    chunks_hit: int = 0
    tier: str = "none"                    # L1 | L2 | none
    shadow: bool = False                  # simulated (engine cannot ingest KV)
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    restore_ms: float = 0.0
    save_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_tps: float = 0.0
    chunk_status: list[str] = field(default_factory=list)
    text: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class Metrics:
    """Counters plus a bounded history of request records."""

    def __init__(self, history: int = 1000) -> None:
        self._lock = threading.RLock()
        self._c: dict[str, float] = {k: 0 for k in COUNTERS}
        self._records: Deque[RequestRecord] = deque(maxlen=history)
        self.started_at = time.time()

    # -- counters -------------------------------------------------------
    def inc(self, name: str, value: float = 1) -> None:
        with self._lock:
            self._c[name] = self._c.get(name, 0) + value

    def get(self, name: str) -> float:
        with self._lock:
            return self._c.get(name, 0)

    def reset(self) -> None:
        with self._lock:
            self._c = {k: 0 for k in COUNTERS}
            self._records.clear()
            self.started_at = time.time()

    # -- requests -------------------------------------------------------
    def record_request(self, rec: RequestRecord) -> None:
        """Store *rec* and update the aggregate counters."""
        with self._lock:
            self._records.append(rec)
            self._c["requests"] += 1
            if rec.error:
                self._c["errors"] += 1
                return
            self._c["tokens_prompt"] += rec.prompt_tokens
            self._c["tokens_computed"] += rec.computed_tokens
            self._c["tokens_reused_lmcache"] += rec.lmcache_restored_tokens
            self._c["tokens_reused_native"] += max(
                rec.server_cached_tokens - rec.lmcache_restored_tokens, 0
            )
            if rec.mode == "lmcache":
                self._c["chunks_requested"] += rec.chunks_total
                self._c["chunks_hit"] += rec.chunks_hit

    def records(self, mode: Optional[str] = None) -> list[RequestRecord]:
        with self._lock:
            recs = list(self._records)
        return [r for r in recs if mode is None or r.mode == mode]

    # -- derived --------------------------------------------------------
    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return counters plus derived KPIs (hit rate, reuse ratio, TTFT)."""
        with self._lock:
            c = dict(self._c)
            recs = [r for r in self._records if not r.error]
        req_chunks = c["chunks_requested"]
        prompt = c["tokens_prompt"]
        by_mode: dict[str, dict[str, float]] = {}
        for mode in ("cold", "native", "lmcache"):
            sel = [r for r in recs if r.mode == mode]
            by_mode[mode] = {
                "n": len(sel),
                "avg_ttft_ms": self._avg([r.ttft_ms for r in sel]),
                "avg_total_ms": self._avg([r.total_ms for r in sel]),
            }
        cold, lm = by_mode["cold"]["avg_ttft_ms"], by_mode["lmcache"]["avg_ttft_ms"]
        return {
            **c,
            "chunk_hit_rate": (c["chunks_hit"] / req_chunks) if req_chunks else 0.0,
            "token_reuse_ratio": (
                (c["tokens_reused_lmcache"] + c["tokens_reused_native"]) / prompt
                if prompt else 0.0
            ),
            "lmcache_token_share": (c["tokens_reused_lmcache"] / prompt) if prompt else 0.0,
            "by_mode": by_mode,
            "ttft_speedup_vs_cold": (cold / lm) if (cold and lm) else 0.0,
            "uptime_s": time.time() - self.started_at,
        }

    def export(self, directory: Path) -> Path:
        """Write counters and request history as timestamped JSON."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "snapshot": self.snapshot(),
            "requests": [r.to_dict() for r in self.records()],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
