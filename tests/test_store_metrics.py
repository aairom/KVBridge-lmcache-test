"""Tests for kvbridge.store (TieredStore) and kvbridge.metrics."""

import json

import pytest

from kvbridge.metrics import Metrics, RequestRecord
from kvbridge.store import TieredStore

KB = 1024


def mk(tmp_path, l1=10 * KB, l2=100 * KB, chunk=4, **kw):
    return TieredStore(l1, l2, tmp_path / "s", chunk, kw.pop("metrics", None), **kw)


def H(*ids):
    """Fake 'hash' chain helper."""
    return [bytes([i]) * 4 for i in ids]


# ------------------------------------------------------------------ lookup
def test_lookup_miss_on_empty(tmp_path):
    m = mk(tmp_path).lookup(H(1, 2))
    assert m.entry is None and m.chunks == 0 and m.tier == "none"


def test_put_then_longest_prefix_lookup(tmp_path):
    st = mk(tmp_path)
    st.put(H(1, 2, 3), 12, b"x" * KB)
    m = st.lookup(H(1, 2, 9, 9))
    assert m.chunks == 2 and m.tokens == 8 and m.tier == "L1"
    assert st.lookup(H(9)).entry is None


def test_chunk_status(tmp_path):
    st = mk(tmp_path)
    st.put(H(1, 2), 8, b"x" * KB)
    assert st.chunk_status(H(1, 2, 3)) == ["L1", "L1", "miss"]


def test_put_empty_hashes_rejected(tmp_path):
    assert mk(tmp_path).put([], 0, b"abc") is None


# ------------------------------------------------------------------- fetch
def test_fetch_from_l1_counts_hit(tmp_path):
    st = mk(tmp_path)
    e = st.put(H(1), 4, b"data")
    assert st.fetch(e) == b"data" and e.hits == 1


def test_fetch_from_l2_promotes_to_l1(tmp_path):
    st = mk(tmp_path)
    e = st.put(H(1), 4, b"data")
    # drop only the RAM copy
    e.data, e.in_l1 = None, False
    st._l1_used = 0
    assert e.tier == "L2"
    assert st.fetch(e) == b"data"
    assert e.in_l1 and st.stats()["l1_used"] == 4


def test_fetch_without_backing_raises(tmp_path):
    st = mk(tmp_path)
    e = st.put(H(1), 4, b"data")
    e.data, e.in_l1, e.in_l2 = None, False, False
    with pytest.raises(KeyError):
        st.fetch(e)


# ---------------------------------------------------------- tiering / evict
def test_write_through_populates_both_tiers(tmp_path):
    st = mk(tmp_path, write_through=True)
    e = st.put(H(1), 4, b"x" * KB)
    assert e.in_l1 and e.in_l2
    assert (tmp_path / "s" / f"{e.id}.kv").exists()


def test_no_write_through_only_l1_until_demoted(tmp_path):
    st = mk(tmp_path, l1=3 * KB, write_through=False)
    a = st.put(H(1), 4, b"a" * 2 * KB)
    assert a.in_l1 and not a.in_l2
    b = st.put(H(2), 4, b"b" * 2 * KB)           # forces `a` out of L1 -> demoted
    assert not a.in_l1 and a.in_l2
    assert b.in_l1
    assert st.metrics.get("evictions_l1") == 1
    assert st.fetch(a) == b"a" * 2 * KB          # still reachable from disk


def test_l1_lru_order(tmp_path):
    st = mk(tmp_path, l1=4 * KB, write_through=True)
    a = st.put(H(1), 4, b"a" * 2 * KB)
    b = st.put(H(2), 4, b"b" * 2 * KB)
    st.fetch(a)                                   # a is now more recent than b
    st.put(H(3), 4, b"c" * 2 * KB)
    assert a.in_l1 and not b.in_l1 and b.in_l2


def test_l2_eviction_drops_entry_entirely(tmp_path):
    st = mk(tmp_path, l1=0, l2=3 * KB)
    a = st.put(H(1), 4, b"a" * 2 * KB)
    st.put(H(2), 4, b"b" * 2 * KB)
    assert st.lookup(H(1)).entry is None
    assert not (tmp_path / "s" / f"{a.id}.kv").exists()
    assert st.metrics.get("evictions_l2") == 1


def test_oversize_rejected(tmp_path):
    st = mk(tmp_path, l1=KB, l2=KB)
    assert st.put(H(1), 4, b"x" * 2 * KB) is None
    assert st.metrics.get("rejected") == 1


def test_oversize_for_l1_goes_to_l2_only(tmp_path):
    st = mk(tmp_path, l1=KB, l2=10 * KB)
    e = st.put(H(1), 4, b"x" * 2 * KB)
    assert e.tier == "L2" and not e.in_l1


def test_longer_snapshot_supersedes_prefix(tmp_path):
    st = mk(tmp_path)
    short = st.put(H(1, 2), 8, b"s" * KB)
    long = st.put(H(1, 2, 3, 4), 16, b"l" * 2 * KB)
    assert st.stats()["entries"] == 1
    assert st.lookup(H(1, 2)).entry is long
    assert not (tmp_path / "s" / f"{short.id}.kv").exists()
    assert st.stats()["l1_used"] == 2 * KB


def test_divergent_branch_is_kept_separately(tmp_path):
    st = mk(tmp_path)
    st.put(H(1, 2, 3), 12, b"a" * KB)
    st.put(H(1, 2, 7), 12, b"b" * KB)
    assert st.stats()["entries"] == 2
    assert st.lookup(H(1, 2, 7)).chunks == 3


# ----------------------------------------------------------- virtual entries
def test_virtual_entry_accounts_ram_only(tmp_path):
    st = mk(tmp_path)
    e = st.put(H(1, 2), 8, None, virtual_size=5 * KB)
    assert e.virtual and e.in_l1 and not e.in_l2
    assert st.stats()["l1_used"] == 5 * KB
    assert st.fetch(e) == b""
    assert not list((tmp_path / "s").glob("*.kv"))


def test_virtual_entry_evicted_not_demoted(tmp_path):
    st = mk(tmp_path, l1=6 * KB)
    v = st.put(H(1), 4, None, virtual_size=5 * KB)
    st.put(H(2), 4, None, virtual_size=5 * KB)
    assert st.lookup(H(1)).entry is None and v.id not in {e.id for e in st.entries()}


def test_virtual_entry_too_big_rejected(tmp_path):
    st = mk(tmp_path, l1=KB)
    assert st.put(H(1), 4, None, virtual_size=2 * KB) is None


# --------------------------------------------------------------- persistence
def test_persistence_roundtrip(tmp_path):
    a = mk(tmp_path)
    e = a.put(H(1, 2), 8, b"persist-me", note="n")
    b = mk(tmp_path)                                # "restart"
    m = b.lookup(H(1, 2, 3))
    assert m.chunks == 2 and m.tier == "L2" and m.entry.note == "n"
    assert b.fetch(m.entry) == b"persist-me"
    assert m.entry.id == e.id


def test_persistence_ignores_other_chunk_size(tmp_path):
    mk(tmp_path, chunk=4).put(H(1), 4, b"x")
    assert mk(tmp_path, chunk=8).stats()["entries"] == 0


def test_persistence_skips_missing_or_truncated_blobs(tmp_path):
    a = mk(tmp_path)
    e = a.put(H(1), 4, b"abcdef")
    (tmp_path / "s" / f"{e.id}.kv").write_bytes(b"ab")      # corrupt size
    assert mk(tmp_path).stats()["entries"] == 0


def test_corrupt_index_is_survivable(tmp_path):
    mk(tmp_path).put(H(1), 4, b"x")
    (tmp_path / "s" / "index.json").write_text("{not json")
    assert mk(tmp_path).stats()["entries"] == 0


def test_persist_false_writes_no_index(tmp_path):
    st = mk(tmp_path, persist=False)
    st.put(H(1), 4, b"x")
    assert not (tmp_path / "s" / "index.json").exists()


def test_index_file_is_valid_json(tmp_path):
    mk(tmp_path).put(H(1, 2), 8, b"x")
    doc = json.loads((tmp_path / "s" / "index.json").read_text())
    assert doc["chunk_size"] == 4 and len(doc["entries"]) == 1


# ---------------------------------------------------------------- admin
def test_clear_l1_keeps_disk(tmp_path):
    st = mk(tmp_path)
    st.put(H(1), 4, b"x" * KB)
    st.clear("l1")
    m = st.lookup(H(1))
    assert m.tier == "L2" and st.stats()["l1_used"] == 0


def test_clear_l1_drops_virtual_and_l1_only_entries(tmp_path):
    st = mk(tmp_path, write_through=False)
    st.put(H(1), 4, b"x")
    st.put(H(2), 4, None, virtual_size=KB)
    st.clear("l1")
    assert st.stats()["entries"] == 0


def test_clear_all(tmp_path):
    st = mk(tmp_path)
    st.put(H(1), 4, b"x")
    st.clear("all")
    assert st.stats()["entries"] == 0 and not list((tmp_path / "s").glob("*.kv"))
    assert st.stats()["l1_used"] == 0 and st.stats()["l2_used"] == 0


def test_resize_evicts(tmp_path):
    st = mk(tmp_path, l1=10 * KB, l2=10 * KB)
    st.put(H(1), 4, b"a" * 4 * KB)
    st.put(H(2), 4, b"b" * 4 * KB)
    st.resize(5 * KB, 5 * KB)
    s = st.stats()
    assert s["l1_used"] <= 5 * KB and s["l2_used"] <= 5 * KB and s["entries"] == 1


def test_stats_and_entries_sorted_by_recency(tmp_path):
    st = mk(tmp_path)
    a = st.put(H(1), 4, b"a")
    b = st.put(H(2), 4, b"b")
    st.fetch(a)
    assert [e.id for e in st.entries()][0] == a.id
    s = st.stats()
    assert s["entries"] == 2 and s["chunks_indexed"] == 2 and s["tokens_cached"] == 8
    assert b.id in {e.id for e in st.entries()}


# ----------------------------------------------------------------- metrics
def rec(**kw):
    base = dict(mode="lmcache", prompt_tokens=100, computed_tokens=20, server_cached_tokens=80,
                lmcache_restored_tokens=80, chunks_total=10, chunks_hit=8, ttft_ms=10.0)
    base.update(kw)
    return RequestRecord(**base)


def test_metrics_counters_and_kpis():
    m = Metrics()
    m.record_request(rec())
    m.record_request(rec(mode="cold", ttft_ms=100.0, server_cached_tokens=0,
                         lmcache_restored_tokens=0, computed_tokens=100, chunks_total=0, chunks_hit=0))
    s = m.snapshot()
    assert s["requests"] == 2
    assert s["chunk_hit_rate"] == pytest.approx(0.8)
    assert s["tokens_prompt"] == 200
    assert s["tokens_reused_lmcache"] == 80
    assert s["token_reuse_ratio"] == pytest.approx(0.4)
    assert s["ttft_speedup_vs_cold"] == pytest.approx(10.0)
    assert s["by_mode"]["cold"]["n"] == 1


def test_native_reuse_is_not_double_counted():
    m = Metrics()
    m.record_request(rec(mode="native", server_cached_tokens=70, lmcache_restored_tokens=0,
                         chunks_total=0, chunks_hit=0))
    s = m.snapshot()
    assert s["tokens_reused_native"] == 70 and s["tokens_reused_lmcache"] == 0


def test_error_records_counted_but_excluded():
    m = Metrics()
    m.record_request(RequestRecord(error="x"))
    s = m.snapshot()
    assert s["errors"] == 1 and s["requests"] == 1 and s["tokens_prompt"] == 0


def test_empty_snapshot_has_no_division_errors():
    s = Metrics().snapshot()
    assert s["chunk_hit_rate"] == 0 and s["token_reuse_ratio"] == 0 and s["ttft_speedup_vs_cold"] == 0


def test_reset_and_records_filter_and_history_bound():
    m = Metrics(history=3)
    for _ in range(5):
        m.record_request(rec())
    assert len(m.records()) == 3
    assert len(m.records("cold")) == 0
    m.reset()
    assert m.get("requests") == 0 and m.records() == []


def test_export_writes_json(tmp_path):
    m = Metrics()
    m.record_request(rec())
    p = m.export(tmp_path / "out")
    doc = json.loads(p.read_text())
    assert p.name.startswith("metrics_") and doc["snapshot"]["requests"] == 1
    assert len(doc["requests"]) == 1


def test_inc_unknown_counter():
    m = Metrics()
    m.inc("custom", 3)
    assert m.get("custom") == 3
