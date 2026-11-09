"""Chunk-level rolling prefix hashing, compatible with LMCache.

LMCache splits a token sequence into fixed-size chunks (default 256 tokens)
and identifies each chunk by a *rolling prefix hash*::

    h_0 = blake3(prefix_seed        || pack(chunk_0))
    h_i = blake3(h_{i-1}            || pack(chunk_i))

where ``pack`` is the big-endian ``uint32`` encoding of the token ids
(``lmcache/v1/multiprocess/token_codec.py``).  Only *complete* chunks are
hashed; a trailing partial chunk is discarded - exactly like
``lmcache.v1.multiprocess.token_hasher.TokenHasher``.

This module re-implements that algorithm (so the app runs on macOS where the
full LMCache wheel cannot be installed) and can optionally delegate to the
real ``TokenHasher`` when the ``lmcache`` package is importable.  A unit test
verifies both produce identical digests.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Protocol, Sequence

log = logging.getLogger(__name__)

try:  # blake3 is the algorithm LMCache uses
    import blake3 as _blake3

    def _digest(data_parts: Sequence[bytes]) -> bytes:
        h = _blake3.blake3()
        for part in data_parts:
            h.update(part)
        return h.digest()

    ALGORITHM = "blake3"
except ImportError:  # pragma: no cover - fallback keeps the app usable
    def _digest(data_parts: Sequence[bytes]) -> bytes:
        h = hashlib.sha256()
        for part in data_parts:
            h.update(part)
        return h.digest()

    ALGORITHM = "sha256 (blake3 missing; NOT LMCache compatible)"


def pack_tokens(tokens: Sequence[int]) -> bytes:
    """Pack token ids as big-endian ``uint32`` (LMCache wire format)."""
    return struct.pack(f">{len(tokens)}I", *tokens)


def namespace_seed(namespace: str) -> bytes:
    """Derive the initial prefix hash from a model/config namespace.

    KV tensors are only valid for the exact model, quantisation and context
    configuration that produced them, so the namespace is mixed into the
    root of every hash chain (LMCache keys also embed the model name).
    """
    return _digest([namespace.encode("utf-8")])


class Hasher(Protocol):
    chunk_size: int
    impl: str

    def chunk_hashes(self, tokens: Sequence[int]) -> list[bytes]: ...


class ChunkHasher:
    """Pure-Python implementation of LMCache's rolling chunk hashing."""

    impl = "kvbridge (LMCache-compatible blake3)"

    def __init__(self, chunk_size: int = 256, namespace: str = "") -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = chunk_size
        self.namespace = namespace
        self.seed = namespace_seed(namespace)

    def chunk_hashes(self, tokens: Sequence[int]) -> list[bytes]:
        """Return one rolling hash per *complete* chunk of ``tokens``."""
        n_complete = len(tokens) - len(tokens) % self.chunk_size
        prefix = self.seed
        out: list[bytes] = []
        for i in range(0, n_complete, self.chunk_size):
            prefix = _digest([prefix, pack_tokens(tokens[i : i + self.chunk_size])])
            out.append(prefix)
        return out


class LMCacheHasherAdapter:
    """Delegates to ``lmcache.v1.multiprocess.token_hasher.TokenHasher``."""

    def __init__(self, chunk_size: int = 256, namespace: str = "") -> None:
        from lmcache.v1.multiprocess.token_hasher import TokenHasher  # type: ignore

        self.chunk_size = chunk_size
        self.namespace = namespace
        self.seed = namespace_seed(namespace)
        self._hasher = TokenHasher(chunk_size=chunk_size, hash_algorithm="blake3")
        self.impl = "lmcache.TokenHasher (real LMCache)"

    def chunk_hashes(self, tokens: Sequence[int]) -> list[bytes]:
        return self._hasher.compute_chunk_hashes(list(tokens), prefix_hash=self.seed)


def make_hasher(chunk_size: int, namespace: str, prefer_lmcache: bool = False) -> Hasher:
    """Create a hasher, preferring the real LMCache one when requested."""
    if prefer_lmcache:
        try:
            return LMCacheHasherAdapter(chunk_size, namespace)
        except Exception as exc:  # ImportError or LMCache init problems
            log.warning("LMCache TokenHasher unavailable (%s); using built-in hasher", exc)
    return ChunkHasher(chunk_size, namespace)


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
