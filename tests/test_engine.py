"""Tests for kvbridge.engine using in-memory fake engines."""

import pytest

from conftest import FakeLlama, FakeOllama, words
from kvbridge.backends import BackendError
from kvbridge.engine import CachedEngine

LONG = words(80)                   # ~80 pseudo tokens -> 20 chunks of 4 (+BOS)


def prompt(q: str) -> str:
    return f"{LONG}\n\nQ: {q}\nA:"


@pytest.fixture
def eng(settings, fake):
    e = CachedEngine(settings, fake)
    e.start()
    return e


def test_invalid_mode(eng):
    with pytest.raises(ValueError):
        eng.infer("x", "bogus")


def test_cold_recomputes_everything(eng):
    r1 = eng.infer(prompt("a"), "cold")
    r2 = eng.infer(prompt("b"), "cold")
    assert r1.server_cached_tokens == 0 and r2.server_cached_tokens == 0
    assert r2.computed_tokens == r2.prompt_tokens


def test_native_reuses_slot_prefix(eng):
    eng.infer(prompt("a"), "native")
    r = eng.infer(prompt("b"), "native")
    assert r.server_cached_tokens > 60 and r.computed_tokens < 20


def test_lmcache_miss_then_store_then_hit(eng, fake):
    r1 = eng.infer(prompt("a"), "lmcache")
    assert r1.chunks_hit == 0 and r1.chunks_total > 10 and r1.tier == "none"
    assert eng.store.stats()["entries"] == 1 and r1.save_ms >= 0

    r2 = eng.infer(prompt("b"), "lmcache")       # native cache was erased, store restores it
    assert r2.chunks_hit > 10 and r2.tier == "L1"
    assert r2.lmcache_restored_tokens > 40
    assert r2.computed_tokens < r1.computed_tokens
    assert "restore" in fake.calls
    # the question differs, so the last chunk(s) diverge: two branches are kept
    assert eng.store.stats()["entries"] == 2


def test_extended_prompt_supersedes_shorter_snapshot(eng):
    eng.infer(LONG, "lmcache")
    eng.infer(LONG + " " + words(40, "x"), "lmcache")   # same prefix, longer
    assert eng.store.stats()["entries"] == 1


def test_identical_prompt_is_not_stored_twice(eng):
    eng.infer(prompt("a"), "lmcache")
    n_saves = eng.backend.calls.count("save")
    eng.infer(prompt("a"), "lmcache")             # every chunk already hit
    assert eng.backend.calls.count("save") == n_saves


def test_metrics_aggregate(eng):
    eng.infer(prompt("a"), "cold")
    eng.infer(prompt("a"), "lmcache")
    eng.infer(prompt("b"), "lmcache")
    s = eng.snapshot()
    assert s["metrics"]["requests"] == 3
    assert s["metrics"]["chunk_hit_rate"] > 0
    assert s["metrics"]["restores"] == 1
    assert s["store"]["entries"] == 2
    assert s["hasher"].startswith("kvbridge")


def test_restore_failure_falls_back_to_recompute(eng, fake):
    eng.infer(prompt("a"), "lmcache")
    fake.fail_restore = True
    r = eng.infer(prompt("b"), "lmcache")
    assert r.error == "" and r.chunks_hit == 0 and r.tier == "none"
    assert r.computed_tokens == r.prompt_tokens         # recomputed everything


def test_save_failure_is_nonfatal(eng, fake):
    fake.fail_save = True
    r = eng.infer(prompt("a"), "lmcache")
    assert r.error == "" and eng.store.stats()["entries"] == 0


def test_backend_error_recorded(eng, fake):
    fake.fail_generate = True
    r = eng.infer(prompt("a"), "native")
    assert r.error == "boom"
    assert eng.metrics.get("errors") == 1


def test_namespace_isolation_between_models(settings):
    a, b = FakeLlama(), FakeLlama()
    b.model_id = "other-model"
    ea, eb = CachedEngine(settings, a), CachedEngine(settings, b)
    ea.start(), eb.start()
    ea.infer(prompt("a"), "lmcache")
    assert ea.store.dir != eb.store.dir
    assert eb.infer(prompt("a"), "lmcache").chunks_hit == 0


def test_simulate_restart_keeps_disk_tier(eng, fake):
    eng.infer(prompt("a"), "lmcache")
    msg = eng.simulate_engine_restart()
    assert "L2" in msg and "restart" in fake.calls
    assert eng.store.stats()["l1_used"] == 0 and eng.store.stats()["l2_used"] > 0
    r = eng.infer(prompt("b"), "lmcache")
    assert r.tier == "L2" and r.chunks_hit > 10


def test_simulate_restart_external_engine(settings, fake):
    fake.external = True
    e = CachedEngine(settings, fake)
    e.start()
    assert "dropped" in e.simulate_engine_restart()


def test_persistence_across_engine_instances(settings):
    e1 = CachedEngine(settings, FakeLlama())
    e1.start()
    e1.infer(prompt("a"), "lmcache")
    e2 = CachedEngine(settings, FakeLlama())          # new process, same store dir
    e2.start()
    assert e2.infer(prompt("z"), "lmcache").tier == "L2"


def test_reset(eng):
    eng.infer(prompt("a"), "lmcache")
    eng.reset(wipe_store=True)
    assert eng.metrics.get("requests") == 0 and eng.store.stats()["entries"] == 0
    eng.infer(prompt("a"), "lmcache")
    eng.reset(wipe_store=False)
    assert eng.store.stats()["entries"] == 1


def test_reset_tolerates_backend_error(eng, fake, monkeypatch):
    monkeypatch.setattr(fake, "reset_native_cache", lambda: (_ for _ in ()).throw(BackendError("x")))
    eng.reset()


def test_stop_stops_backend(eng, fake):
    eng.stop()
    assert not fake.started


# ---------------------------------------------------------------- shadow mode
@pytest.fixture
def shadow(settings, fake_ollama):
    e = CachedEngine(settings, fake_ollama)
    e.start()
    return e


def test_shadow_mode_flags_and_virtual_entries(shadow):
    r1 = shadow.infer(prompt("a"), "lmcache")
    assert r1.shadow and r1.chunks_hit == 0
    st = shadow.store.entries()
    assert len(st) == 1 and st[0].virtual and st[0].size > 0
    assert not list(shadow.store.dir.glob("*.kv"))      # nothing written to disk


def test_shadow_predicts_hit_and_uses_engine_counts(shadow):
    shadow.infer(prompt("a"), "lmcache")
    r2 = shadow.infer(prompt("b"), "lmcache")
    assert r2.shadow and r2.chunks_hit > 10 and r2.tier == "L1"
    assert r2.lmcache_match_tokens > 0
    assert r2.lmcache_restored_tokens <= r2.server_cached_tokens


def test_shadow_cold_resets_engine(shadow, fake_ollama):
    shadow.infer(prompt("a"), "native")
    r = shadow.infer(prompt("a"), "cold")
    assert r.server_cached_tokens == 0 and "reset" in fake_ollama.calls


def test_shadow_never_calls_kv_export(shadow, fake_ollama):
    shadow.infer(prompt("a"), "lmcache")
    shadow.infer(prompt("b"), "lmcache")
    assert "save" not in fake_ollama.calls and "restore" not in fake_ollama.calls


def test_shadow_restart_simulation(shadow):
    shadow.infer(prompt("a"), "lmcache")
    msg = shadow.simulate_engine_restart()
    assert "dropped" in msg
    assert shadow.store.stats()["entries"] == 0      # virtual entries lived in RAM only
