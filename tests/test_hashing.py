"""Tests for kvbridge.hashing."""

import struct

import pytest

from kvbridge.hashing import (ChunkHasher, common_prefix_len, make_hasher, namespace_seed,
                              pack_tokens)


def test_pack_tokens_big_endian_uint32():
    assert pack_tokens([1, 258]) == struct.pack(">II", 1, 258)
    assert pack_tokens([]) == b""


def test_only_complete_chunks_are_hashed():
    h = ChunkHasher(4)
    assert h.chunk_hashes(list(range(3))) == []
    assert len(h.chunk_hashes(list(range(4)))) == 1
    assert len(h.chunk_hashes(list(range(11)))) == 2       # partial tail dropped


def test_deterministic():
    h1, h2 = ChunkHasher(4, "ns"), ChunkHasher(4, "ns")
    toks = list(range(16))
    assert h1.chunk_hashes(toks) == h2.chunk_hashes(toks)


def test_rolling_prefix_property():
    h = ChunkHasher(4)
    a = h.chunk_hashes([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9])
    b = h.chunk_hashes([1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0])
    assert a[:2] == b[:2] and a[2] != b[2]


def test_hash_covers_whole_prefix():
    """Identical second chunk but different first chunk must give another hash."""
    h = ChunkHasher(4)
    a = h.chunk_hashes([1, 1, 1, 1, 5, 6, 7, 8])
    b = h.chunk_hashes([2, 2, 2, 2, 5, 6, 7, 8])
    assert a[1] != b[1]


def test_namespace_separation():
    toks = list(range(8))
    assert ChunkHasher(4, "model-a").chunk_hashes(toks) != ChunkHasher(4, "model-b").chunk_hashes(toks)
    assert namespace_seed("a") != namespace_seed("b")


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        ChunkHasher(0)


def test_digest_length_is_32_bytes():
    assert all(len(x) == 32 for x in ChunkHasher(2).chunk_hashes([1, 2, 3, 4]))


def test_make_hasher_falls_back_when_lmcache_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("lmcache"):
            raise ImportError("no lmcache")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    h = make_hasher(4, "ns", prefer_lmcache=True)
    assert isinstance(h, ChunkHasher)


def test_matches_real_lmcache_token_hasher():
    """Bit-for-bit compatibility with LMCache (skipped when it is not installed)."""
    pytest.importorskip("lmcache.v1.multiprocess.token_hasher")
    from kvbridge.hashing import LMCacheHasherAdapter

    toks = list(range(1000, 1100))
    mine = ChunkHasher(16, "ns").chunk_hashes(toks)
    real = LMCacheHasherAdapter(16, "ns").chunk_hashes(toks)
    assert mine == real


def test_common_prefix_len():
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2
