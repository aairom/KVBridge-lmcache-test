"""Live integration tests against real engines.

Skipped unless ``KVB_LIVE=1``.  Needs Ollama with the model from ``KVB_OLLAMA_MODEL``
(default ``llama3.2:latest``) and a ``llama-server`` binary::

    KVB_LIVE=1 .venv/bin/python -m pytest tests/test_live.py -v -s
"""

import os

import pytest

from kvbridge.backends import BackendError, make_backend
from kvbridge.config import Settings
from kvbridge.engine import CachedEngine
from kvbridge.scenarios import SCENARIOS

pytestmark = pytest.mark.skipif(os.environ.get("KVB_LIVE") != "1", reason="set KVB_LIVE=1 to run live tests")


@pytest.fixture(scope="module")
def live_settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("live")
    s = Settings.from_env(load_dotenv=False, env=dict(os.environ))
    s.chunk_size, s.ctx_size = 64, 4096
    s.store_dir, s.slot_dir, s.log_dir = root / "store", root / "slots", root / "logs"
    s.output_dir, s.input_dir = root / "out", root / "in"
    return s


@pytest.fixture(scope="module")
def llama_engine(live_settings):
    e = CachedEngine(live_settings, make_backend("llamacpp", live_settings))
    try:
        e.start()
    except BackendError as exc:
        pytest.skip(f"llama.cpp unavailable: {exc}")
    yield e
    e.stop()


@pytest.fixture(scope="module")
def ollama_engine(live_settings):
    e = CachedEngine(live_settings, make_backend("ollama", live_settings))
    try:
        e.start()
    except BackendError as exc:
        pytest.skip(f"Ollama unavailable: {exc}")
    yield e


def test_llamacpp_lmcache_beats_cold(llama_engine):
    recs = SCENARIOS["doc_qa"].run(llama_engine, context_tokens=700, n_questions=3)
    assert not [r.error for r in recs if r.error]
    cold = [r for r in recs if r.mode == "cold"]
    lm = [r for r in recs if r.mode == "lmcache"]
    assert all(r.computed_tokens > 500 for r in cold)
    assert lm[0].chunks_hit == 0                                  # first request: miss
    assert all(r.chunks_hit > 0 and r.computed_tokens < 100 for r in lm[1:])
    avg = lambda rs: sum(r.ttft_ms for r in rs) / len(rs)
    assert avg(lm[1:]) < avg(cold)


def test_llamacpp_restart_restores_from_disk(llama_engine):
    recs = SCENARIOS["restart"].run(llama_engine, context_tokens=700)
    assert recs[2].tier == "L2" and recs[2].chunks_hit > 0
    assert recs[2].computed_tokens < recs[1].computed_tokens


def test_ollama_shadow_mode(ollama_engine):
    recs = SCENARIOS["shared_prefix"].run(ollama_engine, users=3, system_tokens=700)
    assert not [r.error for r in recs if r.error]
    lm = [r for r in recs if r.mode == "lmcache"]
    assert all(r.shadow for r in lm)
    assert lm[1].chunks_hit > 0 and lm[1].server_cached_tokens > 0
