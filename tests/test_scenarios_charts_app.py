"""Tests for kvbridge.scenarios, kvbridge.charts and kvbridge.app."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from conftest import FakeLlama, FakeOllama
from kvbridge import charts
from kvbridge.engine import CachedEngine
from kvbridge.metrics import RequestRecord
from kvbridge.scenarios import SCENARIOS, build_context, load_documents


@pytest.fixture
def eng(settings, fake):
    e = CachedEngine(settings, fake)
    e.start()
    return e


# --------------------------------------------------------------- documents
def test_load_documents_recursive_and_filtered(tmp_path):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub" / "b.md").write_text("beta")
    (tmp_path / "sub" / "deep" / "c.rst").write_text("gamma")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    (tmp_path / "empty.txt").write_text("   ")
    docs = dict(load_documents(tmp_path))
    assert set(docs) == {"a.txt", "sub/b.md", "sub/deep/c.rst"}


def test_load_documents_missing_dir(tmp_path):
    assert load_documents(tmp_path / "nope") == []


def test_build_context_builtin_hits_target(eng, settings):
    text, src = build_context(eng, 300, settings.input_dir)
    n = len(eng.backend.tokenize(text))
    assert "built-in" in src and 200 <= n <= 300


def test_build_context_uses_input_files_and_clamps(eng, settings):
    settings.input_dir.mkdir(parents=True)
    (settings.input_dir / "d.txt").write_text("custom document text " * 20)
    text, src = build_context(eng, 10_000_000, settings.input_dir)
    assert "1 file" in src and "custom document" in text
    assert len(eng.backend.tokenize(text)) <= eng.backend.n_ctx - 600


# --------------------------------------------------------------- scenarios
@pytest.mark.parametrize("key,kwargs", [
    ("doc_qa", {"context_tokens": 300, "n_questions": 3}),
    ("multi_turn", {"turns": 3, "system_tokens": 300}),
    ("shared_prefix", {"users": 3, "system_tokens": 300}),
    ("restart", {"context_tokens": 300}),
])
def test_scenarios_run_and_show_benefit(eng, key, kwargs):
    msgs = []
    recs = SCENARIOS[key].run(eng, on_progress=msgs.append, **kwargs)
    assert recs and not [r.error for r in recs if r.error] and msgs
    cold = [r for r in recs if r.mode == "cold"]
    hits = [r for r in recs if r.mode == "lmcache" and r.chunks_hit]
    assert cold and hits
    assert min(r.computed_tokens for r in hits) < min(r.computed_tokens for r in cold)


def test_restart_scenario_sequence(eng):
    recs = SCENARIOS["restart"].run(eng, context_tokens=300)
    assert [r.mode for r in recs] == ["lmcache", "cold", "lmcache", "lmcache"]
    assert recs[0].chunks_hit == 0 and recs[2].tier == "L2" and recs[3].tier == "L1"


def test_scenarios_work_in_shadow_mode(settings, fake_ollama):
    e = CachedEngine(settings, fake_ollama)
    e.start()
    recs = SCENARIOS["shared_prefix"].run(e, users=3, system_tokens=300)
    assert all(r.shadow for r in recs if r.mode == "lmcache")
    assert any(r.chunks_hit for r in recs if r.mode == "lmcache")


def test_registry_complete():
    assert set(SCENARIOS) == {"doc_qa", "multi_turn", "shared_prefix", "restart"}
    assert all(s.title and s.blurb for s in SCENARIOS.values())


# ------------------------------------------------------------------ charts
def sample_records():
    return [
        RequestRecord(mode="cold", ttft_ms=500, prompt_tokens=800, computed_tokens=800, label="a"),
        RequestRecord(mode="native", ttft_ms=20, prompt_tokens=800, computed_tokens=5,
                      server_cached_tokens=795),
        RequestRecord(mode="lmcache", ttft_ms=25, prompt_tokens=800, computed_tokens=5,
                      server_cached_tokens=795, chunks_total=10, chunks_hit=8, tier="L1",
                      chunk_status=["L1"] * 8 + ["miss"] * 2),
        RequestRecord(mode="lmcache", error="boom"),
    ]


@pytest.mark.parametrize("fn", [charts.ttft_figure, charts.tokens_figure, charts.hit_rate_figure])
def test_record_figures_empty_and_filled(fn):
    assert isinstance(fn([]), go.Figure)
    fig = fn(sample_records())
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_tier_chunkmap_mode_figures():
    st = {"l1_used": 10, "l1_cap": 100, "l2_used": 5, "l2_cap": 50}
    assert len(charts.tier_figure(st).data) == 4
    assert len(charts.tier_figure({}).data) == 4
    assert len(charts.chunk_map_figure(["L1", "L2", "miss"] * 20).data) == 1
    assert isinstance(charts.chunk_map_figure([]), go.Figure)
    snap = {"by_mode": {"cold": {"n": 1, "avg_ttft_ms": 100}, "lmcache": {"n": 1, "avg_ttft_ms": 10}}}
    assert len(charts.mode_compare_figure(snap).data[0].x) == 2
    assert isinstance(charts.mode_compare_figure({}), go.Figure)


def test_tables(eng):
    df = charts.records_table(sample_records())
    assert isinstance(df, pd.DataFrame) and len(df) == 4 and "TTFT ms" in df.columns
    assert list(charts.records_table([]).columns)[:3] == ["#", "label", "mode"]
    eng.infer("one two three four five six seven eight nine", "lmcache")
    et = charts.entries_table(eng.store.entries())
    assert len(et) == 1 and et.iloc[0]["tier"].startswith("L1")
    assert list(charts.entries_table([]).columns)[0] == "id"


def test_fmt_bytes():
    assert charts.fmt_bytes(0) == "0 B"
    assert charts.fmt_bytes(1536) == "1.5 KB"
    assert charts.fmt_bytes(5 * 1024 ** 3) == "5.0 GB"
    assert charts.fmt_bytes(3 * 1024 ** 5).endswith("TB")


def test_kpi_markdown(eng):
    eng.infer("a b c d e f g h i j", "cold")
    md = charts.kpi_markdown(eng.snapshot())
    assert "Chunk hit rate" in md and "(n/a)" in md
    assert "x**" in md


# --------------------------------------------------------------------- app
def test_build_app_constructs_and_state_requires_connection(settings):
    gr = pytest.importorskip("gradio")
    from kvbridge.app import AppState, build_app

    state = AppState(settings)
    demo = build_app(state)
    assert isinstance(demo, gr.Blocks)
    with pytest.raises(gr.Error):
        state.require()
    state.engine = object()
    assert state.require() is state.engine


def test_chunk_map_has_readable_height_for_single_row():
    fig = charts.chunk_map_figure(["L1"] * 5)
    assert fig.layout.height >= 220


def test_sigterm_runs_cleanup(tmp_path):
    """main() must turn SIGTERM into a normal exit so ``finally`` stops the engine."""
    import os
    import signal
    import subprocess
    import sys
    import textwrap
    import time

    marker = tmp_path / "cleaned"
    script = textwrap.dedent(f"""
        import sys, time
        from unittest import mock
        import kvbridge.app as app

        class Eng:
            def stop(self): open({str(marker)!r}, "w").write("ok")

        class Demo:
            def queue(self, **k): return self
            def launch(self, **k):
                print("READY", flush=True)
                time.sleep(60)

        with mock.patch.object(app, "build_app", lambda state: Demo()), \\
             mock.patch.object(app, "AppState") as st:
            st.return_value.engine = Eng()
            app.main()
    """)
    env = {**os.environ, "KVB_OUTPUT_DIR": str(tmp_path), "KVB_STORE_DIR": str(tmp_path / "s"),
           "KVB_SLOT_DIR": str(tmp_path / "sl"), "KVB_LOG_DIR": str(tmp_path / "l")}
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if "READY" in (p.stdout.readline() or ""):
                break
        else:
            pytest.fail("child never became ready")
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=15)
    finally:
        if p.poll() is None:
            p.kill()
    assert marker.exists(), "engine.stop() was not called on SIGTERM"


# ------------------------------------------------- configurable-engine GUI helpers
def _handlers(demo):
    return {f.fn.__name__: f.fn for f in demo.fns.values() if getattr(f, "fn", None)}


def test_form_to_values_keeps_existing_key_when_blank():
    from kvbridge.app import form_to_values
    assert "cloud_api_key" not in form_to_values({"cloud_api_key": "  ", "engine": "cloud"})
    assert "cloud_api_key" not in form_to_values({"cloud_api_key": None})
    assert form_to_values({"cloud_api_key": "k"})["cloud_api_key"] == "k"


def test_engine_visibility_per_engine():
    from kvbridge.app import engine_visibility
    assert engine_visibility("llamacpp") == {"llamacpp": True, "ollama": False, "cloud": False,
                                             "local_model": True, "warning": False}
    assert engine_visibility("ollama")["ollama"] and not engine_visibility("ollama")["warning"]
    cl = engine_visibility("openai")
    assert cl["cloud"] and cl["warning"] and not cl["local_model"]
    assert not any(engine_visibility("nope").values())


def test_describe_connection_is_engine_agnostic():
    from kvbridge.app import describe_connection
    base = {"engine": "cloud", "model": "m@h", "n_ctx": 8192, "kv_export": False, "location": "cloud",
            "kv_bytes_per_token": 0}
    txt = describe_connection(base, "blake3")
    assert "prompts leave this machine" in txt and "SHADOW" in txt and "KV size" not in txt
    full = describe_connection({**base, "engine": "llama.cpp", "location": "local", "kv_export": True,
                                "kv_bytes_per_token": 114688, "token_budget": 1000}, "blake3")
    assert "FULL" in full and "this machine" in full and "KV size" in full and "1,000" in full


def test_gui_connect_cloud_with_fake_backend(settings, fake_cloud, monkeypatch):
    from kvbridge import app as A
    monkeypatch.setattr(A, "make_backend", lambda kind, s: fake_cloud)
    state = A.AppState(settings)
    h = _handlers(A.build_app(state))
    vals = dict(settings.to_display())
    vals.update(engine="cloud", cloud_base_url="https://h.example/v1", cloud_model="m", cloud_api_key="sk-abc12345")
    out = h["connect"](*[vals[n] for n in A.FORM_FIELDS])
    assert "Connected" in out and "prompts leave this machine" in out and state.engine is not None
    assert settings.engine == "cloud" and settings.cloud_api_key == "sk-abc12345"
    assert "sk-abc12345" not in out
    # blank key box keeps the stored key
    vals["cloud_api_key"] = ""
    h["connect"](*[vals[n] for n in A.FORM_FIELDS])
    assert settings.cloud_api_key == "sk-abc12345"
    assert "Disconnected" in h["disconnect"]() and state.engine is None


def test_gui_connect_rejects_invalid_atomically(settings):
    from kvbridge import app as A
    state = A.AppState(settings)
    h = _handlers(A.build_app(state))
    vals = dict(settings.to_display())
    vals.update(engine="cloud", cloud_base_url="", cloud_model="")
    before = dict(settings.__dict__)
    out = h["connect"](*[vals[n] for n in A.FORM_FIELDS])
    assert "Invalid configuration" in out and "nothing was changed" in out
    assert settings.__dict__ == before and state.engine is None


def test_gui_save_env_never_writes_key_unless_asked(settings, tmp_path, monkeypatch):
    from kvbridge import app as A
    env = tmp_path / ".env"
    monkeypatch.setattr(A, "ENV_FILE", env)
    state = A.AppState(settings)
    h = _handlers(A.build_app(state))
    vals = dict(settings.to_display())
    vals.update(engine="cloud", cloud_base_url="https://h.example/v1", cloud_model="m", cloud_api_key="sk-abc12345")
    args = [vals[n] for n in A.FORM_FIELDS]
    msg = h["save_env"](False, *args)
    text = env.read_text()
    assert "API key NOT written" in msg and "KVB_ENGINE=cloud" in text and "sk-abc12345" not in text
    assert settings.engine != "cloud"                       # saving does not change the live settings
    msg = h["save_env"](True, *args)
    assert "including the API key" in msg and "KVB_CLOUD_API_KEY=sk-abc12345" in env.read_text()
    bad = dict(vals, cloud_model="")
    assert "Not saved" in h["save_env"](False, *[bad[n] for n in A.FORM_FIELDS])


def test_gui_show_effective_masks_key(settings):
    from kvbridge import app as A
    settings.cloud_api_key = "sk-abc12345"
    h = _handlers(A.build_app(A.AppState(settings)))
    out = h["show_effective"]()
    assert "sk-abc12345" not in out and "********" in out


def test_gui_provider_and_engine_change(settings):
    from kvbridge import app as A
    h = _handlers(A.build_app(A.AppState(settings)))
    assert h["on_provider_change"]("groq") == "https://api.groq.com/openai/v1"
    assert isinstance(h["on_provider_change"](""), dict)
    assert len(h["on_engine_change"]("cloud")) == 5
