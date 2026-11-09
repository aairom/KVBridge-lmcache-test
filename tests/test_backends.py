"""Tests for kvbridge.backends with mocked HTTP and file systems."""

import json
from pathlib import Path

import pytest

from kvbridge import backends as B
from kvbridge.backends import (BackendError, LlamaCppBackend, OllamaBackend, find_llama_server,
                               make_backend, pseudo_tokenize, resolve_ollama_gguf)
from kvbridge.config import Settings


# ------------------------------------------------------------- fake requests
class Resp:
    def __init__(self, status=200, js=None, lines=None, text=""):
        self.status_code, self._js, self._lines, self.text = status, js, lines or [], text

    def json(self):
        if self._js is None:
            raise ValueError("no json")
        return self._js

    def iter_lines(self):
        return iter(self._lines)


class FakeHTTP:
    """Records calls and answers from a route table: {(METHOD, path-substring): Resp | callable}."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _answer(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for (m, frag), r in self.routes.items():
            if m == method and frag in url:
                if isinstance(r, Exception):
                    raise r
                return r(url, kw) if callable(r) else r
        raise B.requests.ConnectionError(f"no route for {method} {url}")

    def get(self, url, **kw):
        return self._answer("GET", url, **kw)

    def post(self, url, **kw):
        return self._answer("POST", url, **kw)


def sse(*msgs):
    return [b"data: " + json.dumps(m).encode() for m in msgs] + [b"", b"data: [DONE]"]


# ------------------------------------------------------------------ helpers
def test_pseudo_tokenize_is_stable_and_prefix_preserving():
    a = pseudo_tokenize("Hello, world! This is a test.")
    assert a == pseudo_tokenize("Hello, world! This is a test.")
    b = pseudo_tokenize("Hello, world! This is a test. And more.")
    assert b[: len(a)] == a and len(b) > len(a)
    assert pseudo_tokenize("") == []


def make_manifest(base: Path, name="llama3.2", tag="latest", ns="library", digest="sha256:abc"):
    mdir = base / "manifests" / "registry.ollama.ai" / ns / name
    mdir.mkdir(parents=True)
    (mdir / tag).write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:zzz"},
        {"mediaType": "application/vnd.ollama.image.model", "digest": digest}]}))
    (base / "blobs").mkdir(exist_ok=True)
    blob = base / "blobs" / digest.replace(":", "-")
    blob.write_bytes(b"GGUF")
    return blob


def test_resolve_ollama_gguf(tmp_path):
    blob = make_manifest(tmp_path)
    assert resolve_ollama_gguf("llama3.2:latest", tmp_path) == blob
    assert resolve_ollama_gguf("llama3.2", tmp_path) == blob          # default tag


def test_resolve_ollama_gguf_namespaced(tmp_path):
    blob = make_manifest(tmp_path, name="foo", ns="acme", tag="7b", digest="sha256:def")
    assert resolve_ollama_gguf("acme/foo:7b", tmp_path) == blob


def test_resolve_ollama_gguf_errors(tmp_path):
    with pytest.raises(BackendError, match="manifest not found"):
        resolve_ollama_gguf("nope:latest", tmp_path)
    blob = make_manifest(tmp_path)
    blob.unlink()
    with pytest.raises(BackendError, match="blob missing"):
        resolve_ollama_gguf("llama3.2", tmp_path)
    mdir = tmp_path / "manifests/registry.ollama.ai/library/bad"
    mdir.mkdir(parents=True)
    (mdir / "latest").write_text("{ not json")
    with pytest.raises(BackendError, match="cannot parse"):
        resolve_ollama_gguf("bad", tmp_path)


def test_find_llama_server(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert find_llama_server(str(exe)) == str(exe)
    monkeypatch.setattr(B.shutil, "which", lambda _: str(exe))
    assert find_llama_server() == str(exe)
    monkeypatch.setattr(B.shutil, "which", lambda _: None)
    monkeypatch.setattr(B.Path, "exists", lambda self: False)
    with pytest.raises(BackendError, match="not found"):
        find_llama_server()


def test_make_backend(settings):
    assert isinstance(make_backend("llamacpp", settings), LlamaCppBackend)
    assert isinstance(make_backend("llama.cpp", settings), LlamaCppBackend)
    assert isinstance(make_backend("Ollama", settings), OllamaBackend)
    with pytest.raises(BackendError):
        make_backend("vllm", settings)


# ------------------------------------------------------------------ llama.cpp
@pytest.fixture
def llama(settings):
    b = LlamaCppBackend(settings)
    return b


def test_build_command(settings, tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    settings.model_path, settings.n_gpu_layers, settings.extra_args = str(model), 20, "--threads 4"
    monkeypatch.setattr(B, "find_llama_server", lambda cfg="": "/bin/llama-server")
    b = LlamaCppBackend(settings)
    cmd = b.build_command()
    assert cmd[0] == "/bin/llama-server" and "-m" in cmd and str(model) in cmd
    assert cmd[cmd.index("--port") + 1] == str(settings.managed_port)
    assert "--slot-save-path" in cmd and cmd[cmd.index("--cache-ram") + 1] == "0"
    assert cmd[cmd.index("-ngl") + 1] == "20" and cmd[-2:] == ["--threads", "4"]
    assert settings.slot_dir.exists()


def test_build_command_missing_model(settings):
    settings.model_path = "/definitely/not/here.gguf"
    with pytest.raises(BackendError, match="KVB_MODEL_PATH"):
        LlamaCppBackend(settings).build_command()


def test_external_url_used_and_unreachable_raises(settings):
    settings.llama_url = "http://127.0.0.1:9/"
    b = LlamaCppBackend(settings)
    assert b.external and b.base == "http://127.0.0.1:9"
    b.http = FakeHTTP({})
    with pytest.raises(BackendError, match="not reachable"):
        b.start()


def test_start_attaches_to_healthy_server_and_probes_ctx(llama):
    llama.http = FakeHTTP({("GET", "/health"): Resp(js={"status": "ok"}),
                           ("GET", "/slots"): Resp(js=[{"n_ctx": 1234}])})
    llama.start()
    assert llama.n_ctx == 1234 and llama.proc is None


def test_start_refuses_busy_port(llama, monkeypatch):
    llama.http = FakeHTTP({})
    monkeypatch.setattr(B, "port_open", lambda h, p: True)
    with pytest.raises(BackendError, match="busy"):
        llama.start()


def test_is_ready_handles_errors(llama):
    llama.http = FakeHTTP({("GET", "/health"): Resp(503, js={"status": "loading"})})
    assert not llama.is_ready()
    llama.http = FakeHTTP({})
    assert not llama.is_ready()


def test_tokenize_and_http_error(llama):
    llama.http = FakeHTTP({("POST", "/tokenize"): Resp(js={"tokens": [1, 2, 3]})})
    assert llama.tokenize("hi") == [1, 2, 3]
    assert llama.http.calls[0][2]["json"]["add_special"] is True
    llama.http = FakeHTTP({("POST", "/tokenize"): Resp(500, text="boom")})
    with pytest.raises(BackendError, match="HTTP 500"):
        llama.tokenize("hi")
    llama.http = FakeHTTP({})
    with pytest.raises(BackendError, match="failed"):
        llama.tokenize("hi")


def test_generate_parses_stream_and_timings(llama):
    stream = sse({"content": "Hel", "stop": False}, {"content": "lo", "stop": False},
                 {"content": "", "stop": True, "timings": {
                     "cache_n": 90, "prompt_n": 10, "prompt_ms": 12.5, "predicted_n": 2,
                     "predicted_per_second": 40.0}})
    llama.http = FakeHTTP({("POST", "/completion"): Resp(lines=stream)})
    r = llama.generate("p", [1, 2, 3], 8, True, 0.1)
    body = llama.http.calls[0][2]["json"]
    assert body["prompt"] == [1, 2, 3] and body["cache_prompt"] is True and body["id_slot"] == 0
    assert r.text == "Hello" and r.cached_tokens == 90 and r.computed_tokens == 10
    assert r.prompt_tokens == 100 and r.decode_tps == 40.0 and r.ttft_ms > 0 and r.prefill_ms == 12.5


def test_generate_without_tokens_sends_text(llama):
    llama.http = FakeHTTP({("POST", "/completion"): Resp(lines=sse({"content": "", "stop": True}))})
    llama.generate("raw text", None, 4, False, 0.0)
    assert llama.http.calls[0][2]["json"]["prompt"] == "raw text"


def test_reset_native_cache_calls_erase(llama):
    llama.http = FakeHTTP({("POST", "action=erase"): Resp(js={})})
    llama.reset_native_cache()
    assert "action=erase" in llama.http.calls[0][1]


def test_save_and_restore_state_roundtrip_via_slot_dir(llama, settings):
    def on_save(url, kw):
        (settings.slot_dir / kw["json"]["filename"]).write_bytes(b"KVDATA")
        return Resp(js={"n_saved": 7})

    seen = {}

    def on_restore(url, kw):
        seen["data"] = (settings.slot_dir / kw["json"]["filename"]).read_bytes()
        return Resp(js={"n_restored": 7})

    settings.slot_dir.mkdir(parents=True)
    llama.http = FakeHTTP({("POST", "action=save"): on_save, ("POST", "action=restore"): on_restore})
    blob, n = llama.save_state()
    assert blob == b"KVDATA" and n == 7
    assert llama.restore_state(b"KVDATA") == 7 and seen["data"] == b"KVDATA"
    assert not list(settings.slot_dir.iterdir())                    # staging files cleaned up


def test_save_state_missing_file_is_backend_error(llama, settings):
    settings.slot_dir.mkdir(parents=True)
    llama.http = FakeHTTP({("POST", "action=save"): Resp(js={"n_saved": 1})})
    with pytest.raises(BackendError, match="slot-save-path"):
        llama.save_state()


def test_restart_external_not_allowed(settings):
    settings.llama_url = "http://x"
    with pytest.raises(BackendError):
        LlamaCppBackend(settings).restart()


def test_stop_without_process_is_noop(llama):
    llama.stop()
    assert llama.proc is None


def test_metrics_text(llama):
    llama.http = FakeHTTP({("GET", "/metrics"): Resp(text="llamacpp:prompt_tokens_total 5")})
    assert "prompt_tokens_total" in llama.metrics_text()
    llama.http = FakeHTTP({})
    assert llama.metrics_text() == ""


# --------------------------------------------------------------------- Ollama
SHOW = {"model_info": {"general.architecture": "llama", "llama.block_count": 28,
                       "llama.attention.head_count": 24, "llama.attention.head_count_kv": 8,
                       "llama.embedding_length": 3072}}


@pytest.fixture
def ollama(settings):
    return OllamaBackend(settings)


def test_ollama_start_computes_kv_size(ollama):
    ollama.http = FakeHTTP({("GET", "/api/version"): Resp(js={"version": "0"}),
                            ("POST", "/api/show"): Resp(js=SHOW)})
    ollama.start()
    assert ollama.kv_bytes_per_token == 2 * 28 * 8 * 128 * 2          # 114,688 B / token
    assert ollama.describe()["kv_export"] is False


def test_ollama_start_errors(ollama):
    ollama.http = FakeHTTP({})
    with pytest.raises(BackendError, match="not reachable"):
        ollama.start()
    ollama.http = FakeHTTP({("GET", "/api/version"): Resp(js={}), ("POST", "/api/show"): Resp(404)})
    with pytest.raises(BackendError, match="ollama pull"):
        ollama.start()


def test_ollama_generate_maps_counts(ollama):
    final = {"done": True, "response": "", "prompt_eval_count": 528, "prompt_eval_cached_count": 521,
             "prompt_eval_duration": 20_000_000, "eval_count": 10, "eval_duration": 500_000_000}
    lines = [json.dumps({"response": "Hi", "done": False}).encode(), json.dumps(final).encode()]
    ollama.http = FakeHTTP({("POST", "/api/generate"): Resp(lines=lines)})
    r = ollama.generate("p", None, 8, True, 0.0)
    assert r.text == "Hi" and r.prompt_tokens == 528 and r.cached_tokens == 521
    assert r.computed_tokens == 7 and r.prefill_ms == 20.0 and r.decode_tps == pytest.approx(20.0)
    body = ollama.http.calls[0][2]["json"]
    assert body["raw"] is True and body["options"]["num_predict"] == 8


def test_ollama_generate_errors(ollama):
    ollama.http = FakeHTTP({("POST", "/api/generate"): Resp(500, text="x")})
    with pytest.raises(BackendError, match="HTTP 500"):
        ollama.generate("p", None, 1, True, 0)
    ollama.http = FakeHTTP({("POST", "/api/generate"): Resp(lines=[b'{"error": "oom"}'])})
    with pytest.raises(BackendError, match="oom"):
        ollama.generate("p", None, 1, True, 0)
    ollama.http = FakeHTTP({("POST", "/api/generate"): B.requests.ConnectionError("down")})
    with pytest.raises(BackendError, match="request failed"):
        ollama.generate("p", None, 1, True, 0)


def test_ollama_reset_unloads_and_reloads(ollama, monkeypatch):
    monkeypatch.setattr(B.time, "sleep", lambda _: None)
    ollama.http = FakeHTTP({("POST", "/api/generate"): Resp(js={}),
                            ("GET", "/api/ps"): Resp(js={"models": []})})
    ollama.reset_native_cache()
    posts = [c[2]["json"] for c in ollama.http.calls if c[0] == "POST"]
    assert posts[0]["keep_alive"] == 0 and posts[1]["keep_alive"] == "10m"


def test_ollama_reset_error(ollama):
    ollama.http = FakeHTTP({})
    with pytest.raises(BackendError):
        ollama.reset_native_cache()


def test_ollama_tokenize_is_pseudo(ollama):
    assert ollama.tokenize("a b") == pseudo_tokenize("a b")


def test_base_backend_cannot_export():
    class Dummy(B.Backend):
        start = stop = is_ready = tokenize = generate = reset_native_cache = lambda *a, **k: None

    d = Dummy()
    with pytest.raises(BackendError):
        d.save_state()
    with pytest.raises(BackendError):
        d.restore_state(b"")
