"""Two-tier (RAM + disk) KV snapshot store with LMCache semantics.

Concepts borrowed from LMCache
------------------------------
* **Chunk-hash addressing** - every cached snapshot is addressed by the chain
  of rolling prefix hashes of its prompt chunks.  Because each hash covers the
  entire prefix, a hit on chunk *i* proves the first ``(i+1) * chunk_size``
  tokens are identical, so the stored KV tensors can be reused for them.
* **Tiering** - L1 is host RAM (LMCache's "CPU memory" tier), L2 is local disk
  (LMCache's "local storage" tier).  Entries are evicted LRU; L1 evictions are
  demoted to L2 when write-through is off.
* **Persistence** - the L2 index is written to disk so the cache survives an
  engine (or application) restart: LMCache's "no fate-sharing with the
  engine" property.

Granularity note: llama.cpp can only save/restore a *whole slot*, so one
stored entry is one slot snapshot which is registered in the index under all
of its chunk hashes (a longer snapshot supersedes shorter ones it contains).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from .metrics import Metrics

log = logging.getLogger(__name__)

INDEX_FILE = "index.json"


@dataclass
class Entry:
    """One stored KV snapshot."""

    id: str
    hashes: list[bytes]
    n_tokens: int            # tokens contained in the snapshot
    size: int                # bytes
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0
    in_l1: bool = False
    in_l2: bool = False
    data: Optional[bytes] = None     # RAM copy (L1)
    note: str = ""
    virtual: bool = False            # accounting-only entry (no payload, RAM only)

    @property
    def tier(self) -> str:
        return "L1" if self.in_l1 else ("L2" if self.in_l2 else "none")


@dataclass
class Match:
    """Result of a prefix lookup."""

    entry: Optional[Entry]
    chunks: int               # number of leading chunks matched
    tokens: int               # chunks * chunk_size
    tier: str = "none"


class TieredStore:
    """Thread-safe LRU store with RAM (L1) and disk (L2) tiers."""

    def __init__(
        self,
        l1_bytes: int,
        l2_bytes: int,
        directory: Path,
        chunk_size: int,
        metrics: Optional[Metrics] = None,
        write_through: bool = True,
        persist: bool = True,
    ) -> None:
        self.l1_cap = max(int(l1_bytes), 0)
        self.l2_cap = max(int(l2_bytes), 0)
        self.dir = Path(directory)
        self.chunk_size = chunk_size
        self.metrics = metrics or Metrics()
        self.write_through = write_through
        self.persist = persist
        self._lock = threading.RLock()
        self._entries: dict[str, Entry] = {}
        self._index: dict[bytes, str] = {}      # chunk hash -> entry id
        self._l1_used = 0
        self._l2_used = 0
        if self.l2_cap:
            self.dir.mkdir(parents=True, exist_ok=True)
            if persist:
                self._load_index()

    # ------------------------------------------------------------------ paths
    def _blob_path(self, entry_id: str) -> Path:
        return self.dir / f"{entry_id}.kv"

    # ----------------------------------------------------------------- lookup
    def lookup(self, hashes: Sequence[bytes]) -> Match:
        """Find the entry covering the longest prefix of ``hashes``."""
        with self._lock:
            for i in range(len(hashes) - 1, -1, -1):
                eid = self._index.get(hashes[i])
                if eid is None:
                    continue
                entry = self._entries.get(eid)
                if entry is None:
                    continue
                n_chunks = i + 1
                return Match(entry, n_chunks, n_chunks * self.chunk_size, entry.tier)
        return Match(None, 0, 0, "none")

    def chunk_status(self, hashes: Sequence[bytes]) -> list[str]:
        """Per-chunk tier (``L1``/``L2``/``miss``) for visualisation."""
        with self._lock:
            out = []
            for h in hashes:
                e = self._entries.get(self._index.get(h, ""))
                out.append(e.tier if e else "miss")
            return out

    # ------------------------------------------------------------------ fetch
    def fetch(self, entry: Entry) -> bytes:
        """Return the snapshot bytes, promoting L2 hits into L1 when they fit."""
        with self._lock:
            entry.hits += 1
            entry.last_used = time.time()
            if entry.in_l1 and entry.data is not None:
                return entry.data   # also covers virtual entries (empty payload)
            if not entry.in_l2:
                raise KeyError(f"entry {entry.id} has no backing tier")
            data = self._blob_path(entry.id).read_bytes()
            if entry.size <= self.l1_cap:
                self._make_room_l1(entry.size)
                entry.data, entry.in_l1 = data, True
                self._l1_used += entry.size
            return data

    # -------------------------------------------------------------------- put
    def put(
        self,
        hashes: Sequence[bytes],
        n_tokens: int,
        data: Optional[bytes],
        note: str = "",
        virtual_size: Optional[int] = None,
    ) -> Optional[Entry]:
        """Store a snapshot addressed by ``hashes``; returns ``None`` if rejected.

        With ``virtual_size`` the entry only *accounts* for that many bytes in
        L1 (no payload, never written to disk).  It is used for engines such
        as Ollama that cannot export their KV tensors.
        """
        if not hashes:
            return None
        virtual = virtual_size is not None
        size = virtual_size if virtual else len(data or b"")
        with self._lock:
            fits_l1 = size <= self.l1_cap
            fits_l2 = size <= self.l2_cap and not virtual
            if virtual:
                data = b""
            if not (fits_l1 or fits_l2):
                self.metrics.inc("rejected")
                log.warning("snapshot of %d bytes exceeds both tier capacities; dropped", size)
                return None
            self._drop_superseded(list(hashes))
            entry = Entry(uuid.uuid4().hex[:12], list(hashes), n_tokens, size, note=note,
                          virtual=virtual)

            if fits_l1:
                self._make_room_l1(size)
                entry.data, entry.in_l1 = data, True
                self._l1_used += size
            if fits_l2 and (self.write_through or not fits_l1):
                self._write_l2(entry, data)

            self._entries[entry.id] = entry
            for h in entry.hashes:
                self._index[h] = entry.id
            self.metrics.inc("stores")
            self.metrics.inc("store_bytes", size)
            self._save_index()
            return entry

    # --------------------------------------------------------------- eviction
    def _drop_superseded(self, hashes: list[bytes]) -> None:
        """Remove stored entries whose hash chain is a prefix of ``hashes``."""
        victims = set()
        for h in hashes:
            eid = self._index.get(h)
            if eid and eid in self._entries:
                old = self._entries[eid].hashes
                if len(old) <= len(hashes) and hashes[: len(old)] == old:
                    victims.add(eid)
        for eid in victims:
            self._remove(eid, count_eviction=False)

    def _remove(self, entry_id: str, count_eviction: bool = True) -> None:
        entry = self._entries.pop(entry_id, None)
        if not entry:
            return
        if entry.in_l1:
            self._l1_used -= entry.size
        if entry.in_l2:
            self._l2_used -= entry.size
            try:
                self._blob_path(entry.id).unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover
                log.warning("could not delete %s: %s", entry.id, exc)
        for h in entry.hashes:
            if self._index.get(h) == entry_id:
                del self._index[h]
        if count_eviction:
            self.metrics.inc("evictions_l2")

    def _write_l2(self, entry: Entry, data: bytes) -> None:
        self._make_room_l2(entry.size, keep=entry.id)
        tmp = self._blob_path(entry.id).with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self._blob_path(entry.id))
        entry.in_l2 = True
        self._l2_used += entry.size

    def _make_room_l1(self, need: int) -> None:
        """Free L1 space by dropping/demoting least-recently-used entries."""
        while self._l1_used + need > self.l1_cap:
            victim = min(
                (e for e in self._entries.values() if e.in_l1),
                key=lambda e: e.last_used,
                default=None,
            )
            if victim is None:
                break
            if (not victim.in_l2 and not victim.virtual and victim.size <= self.l2_cap
                    and victim.data is not None):
                self._write_l2(victim, victim.data)          # demote
            victim.data, victim.in_l1 = None, False
            self._l1_used -= victim.size
            self.metrics.inc("evictions_l1")
            if not victim.in_l2:
                self._remove(victim.id)

    def _make_room_l2(self, need: int, keep: str = "") -> None:
        while self._l2_used + need > self.l2_cap:
            victim = min(
                (e for e in self._entries.values() if e.in_l2 and e.id != keep),
                key=lambda e: e.last_used,
                default=None,
            )
            if victim is None:
                break
            self._remove(victim.id)

    # ------------------------------------------------------------ persistence
    def _save_index(self) -> None:
        if not (self.persist and self.l2_cap):
            return
        rows = [
            {
                "id": e.id, "hashes": [h.hex() for h in e.hashes], "n_tokens": e.n_tokens,
                "size": e.size, "created": e.created, "last_used": e.last_used,
                "hits": e.hits, "note": e.note,
            }
            for e in self._entries.values() if e.in_l2
        ]
        tmp = self.dir / (INDEX_FILE + ".tmp")
        tmp.write_text(json.dumps({"chunk_size": self.chunk_size, "entries": rows}), encoding="utf-8")
        os.replace(tmp, self.dir / INDEX_FILE)

    def _load_index(self) -> None:
        path = self.dir / INDEX_FILE
        if not path.exists():
            return
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("chunk_size") != self.chunk_size:
                log.warning("persisted cache uses another chunk size; ignoring it")
                return
            for r in doc["entries"]:
                blob = self._blob_path(r["id"])
                if not blob.exists() or blob.stat().st_size != r["size"]:
                    continue
                e = Entry(r["id"], [bytes.fromhex(h) for h in r["hashes"]], r["n_tokens"],
                          r["size"], r["created"], r["last_used"], r["hits"], in_l2=True,
                          note=r.get("note", ""))
                self._entries[e.id] = e
                self._l2_used += e.size
                for h in e.hashes:
                    self._index[h] = e.id
            log.info("restored %d cached snapshot(s) from %s", len(self._entries), self.dir)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("could not load persisted cache index: %s", exc)

    # ------------------------------------------------------------------ admin
    def clear(self, tiers: str = "all") -> None:
        """Drop everything (``all``) or only the RAM tier (``l1``)."""
        with self._lock:
            if tiers == "l1":
                for e in self._entries.values():
                    if e.in_l1:
                        e.data, e.in_l1 = None, False
                        self._l1_used -= e.size
                for eid in [i for i, e in self._entries.items() if not e.in_l2]:
                    self._remove(eid, count_eviction=False)
            else:
                for eid in list(self._entries):
                    self._remove(eid, count_eviction=False)
            self._save_index()

    def resize(self, l1_bytes: int, l2_bytes: int) -> None:
        """Change capacities at runtime (evicting as needed)."""
        with self._lock:
            self.l1_cap, self.l2_cap = max(int(l1_bytes), 0), max(int(l2_bytes), 0)
            if self.l2_cap:
                self.dir.mkdir(parents=True, exist_ok=True)
            self._make_room_l1(0)
            self._make_room_l2(0)

    def stats(self) -> dict:
        with self._lock:
            return {
                "l1_used": self._l1_used, "l1_cap": self.l1_cap,
                "l2_used": self._l2_used, "l2_cap": self.l2_cap,
                "entries": len(self._entries),
                "chunks_indexed": len(self._index),
                "tokens_cached": sum(e.n_tokens for e in self._entries.values()),
            }

    def entries(self) -> list[Entry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: e.last_used, reverse=True)
