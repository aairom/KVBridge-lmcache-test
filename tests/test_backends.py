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
    def __init__(self, status=200, js=None, lines=None, text="", headers=None):
        self.status_code, self._js, self._lines, self.text = status, js, lines or [], text
        self.headers = headers or {}

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
    monkeypatch.setattr(B, "find_llama_server", lambda *a: "/bin/llama-server")
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


# ---------------------------------------------------------------- engine registry
def test_registry_resolves_keys_and_aliases():
    assert B.resolve_engine("llamacpp").key == "llamacpp"
    assert B.resolve_engine("llama.cpp").key == "llamacpp"
    assert B.resolve_engine("LLAMA").key == "llamacpp"
    assert B.resolve_engine("OpenAI").key == "cloud"
    assert B.resolve_engine("openai-compatible").key == "cloud"
    assert {k for _, k in B.engine_choices()} >= {"llamacpp", "ollama", "cloud"}
    assert B.resolve_engine("cloud").location == "cloud" and B.resolve_engine("ollama").location == "local"


def test_registry_unknown_engine():
    with pytest.raises(BackendError, match="unknown engine"):
        B.resolve_engine("vllm")


def test_register_custom_engine(settings):
    class Mine(B.Backend):
        name = "mine"
        start = stop = is_ready = tokenize = generate = reset_native_cache = lambda *a, **k: None

        def __init__(self, s):
            self.s = s

    spec = B.register_engine(B.EngineSpec("mine-engine", "Mine", Mine, "demo", ("mine",)))
    try:
        assert isinstance(make_backend("mine", settings), Mine)
        assert spec in B.engine_specs()
        assert Settings(engine="mine-engine").validate() == []
    finally:
        B._REGISTRY.pop(B._norm("mine-engine"), None)
    with pytest.raises(BackendError):
        make_backend("mine", settings)


def test_find_llama_server_uses_configured_search_paths(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "llama-server").write_text("x")
    monkeypatch.setattr(B.shutil, "which", lambda _: None)
    assert find_llama_server("", str(d)).endswith("llama-server")


def test_ollama_honours_keep_alive_and_seed(settings, monkeypatch):
    settings.ollama_keep_alive, settings.seed = "30m", 7
    o = OllamaBackend(settings)
    monkeypatch.setattr(B.time, "sleep", lambda _: None)
    o.http = FakeHTTP({("POST", "/api/generate"): Resp(js={}), ("GET", "/api/ps"): Resp(js={"models": []})})
    o.reset_native_cache()
    posts = [c[2]["json"] for c in o.http.calls if c[0] == "POST"]
    assert posts[1]["keep_alive"] == "30m"
    o.http = FakeHTTP({("POST", "/api/generate"): Resp(lines=[b'{"done": true, "response": ""}'])})
    o.generate("p", None, 1, True, 0)
    assert o.http.calls[0][2]["json"]["options"]["seed"] == 7


# ---------------------------------------------------------------- cloud backend
SECRET = "sk-test-SECRET-123456"


@pytest.fixture
def cloud_settings(settings):
    settings.engine, settings.cloud_base_url = "cloud", "https://api.example.com/v1/"
    settings.cloud_model, settings.cloud_api_key = "gpt-test", SECRET
    return settings


@pytest.fixture
def cloud(cloud_settings, monkeypatch):
    monkeypatch.setattr(B.time, "sleep", lambda _: None)
    return B.CloudBackend(cloud_settings)


def chat_stream(text="Hello world", prompt=100, cached=0, completion=2):
    chunks = [{"choices": [{"delta": {"content": w + " "}}]} for w in text.split()]
    usage = {"usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                       "prompt_tokens_details": {"cached_tokens": cached}}, "choices": []}
    return [b": keep-alive"] + sse(*chunks, usage)


def test_cloud_properties_and_describe(cloud):
    assert cloud.kind == "cloud" and cloud.supports_kv_export is False
    assert cloud.base == "https://api.example.com/v1"
    assert cloud.http.headers["Authorization"] == f"Bearer {SECRET}"
    d = cloud.describe()
    assert d["location"] == "cloud" and d["kv_export"] is False and d["endpoint"] == "api.example.com"
    assert SECRET not in json.dumps(d)


def test_cloud_start_ok_and_warns_on_unknown_model(cloud, caplog):
    cloud.http = FakeHTTP({("GET", "/models"): Resp(js={"data": [{"id": "other"}]})})
    with caplog.at_level("WARNING"):
        cloud.start()
    assert "not in the provider's model list" in caplog.text
    assert cloud.is_ready() is True
    cloud.stop()                      # no-op


@pytest.mark.parametrize("status,msg", [(401, "authentication failed"), (403, "authentication failed"),
                                         (500, "HTTP 500")])
def test_cloud_start_http_errors(cloud, status, msg):
    cloud.http = FakeHTTP({("GET", "/models"): Resp(status, text="nope")})
    with pytest.raises(BackendError, match=msg):
        cloud.start()


def test_cloud_start_unreachable_redacts_key(cloud):
    cloud.http = FakeHTTP({("GET", "/models"): B.requests.ConnectionError(f"failed with {SECRET}")})
    with pytest.raises(BackendError) as ei:
        cloud.start()
    assert "not reachable" in str(ei.value) and SECRET not in str(ei.value)
    assert cloud.is_ready() is False


def test_cloud_start_tolerates_missing_models_endpoint(cloud):
    cloud.http = FakeHTTP({("GET", "/models"): Resp(404)})
    cloud.start()
    assert cloud.is_ready() is True
    cloud.http = FakeHTTP({("GET", "/models"): Resp(js=None)})
    cloud.start()


def test_cloud_start_requires_endpoint(settings):
    with pytest.raises(BackendError, match="no cloud endpoint"):
        B.CloudBackend(settings).start()


def test_cloud_generate_chat_maps_usage_and_cache(cloud):
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(lines=chat_stream(prompt=200, cached=150))})
    r = cloud.generate("hi", None, 16, True, 0.3)
    assert r.text.strip() == "Hello world" and r.prompt_tokens == 200
    assert r.cached_tokens == 150 and r.computed_tokens == 50 and r.ttft_ms > 0
    body = cloud.http.calls[0][2]["json"]
    assert body["messages"] == [{"role": "user", "content": "hi"}] and body["max_tokens"] == 16
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.3 and body["model"] == "gpt-test"
    assert cloud.tokens_sent == 200


def test_cloud_generate_deepseek_style_cache_field(cloud):
    usage = {"usage": {"prompt_tokens": 40, "completion_tokens": 1, "prompt_cache_hit_tokens": 32}, "choices": []}
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(lines=sse({"choices": [{"delta": {"content": "x"}}]}, usage))})
    assert cloud.generate("p", None, 4, True, 0).cached_tokens == 32


def test_cloud_generate_without_usage_estimates_tokens(cloud):
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(lines=sse({"choices": [{"delta": {"content": "ok"}}]}))})
    r = cloud.generate("one two three", None, 4, True, 0)
    assert r.prompt_tokens == len(pseudo_tokenize("one two three")) and r.cached_tokens == 0


def test_cloud_generate_completions_style_and_custom_fields(cloud_settings):
    cloud_settings.cloud_api_style = "completions"
    cloud_settings.cloud_max_tokens_field = "max_completion_tokens"
    cloud_settings.cloud_send_temperature = False
    cloud_settings.cloud_stream_usage = False
    c = B.CloudBackend(cloud_settings)
    c.http = FakeHTTP({("POST", "/completions"): Resp(lines=sse({"choices": [{"text": "yo"}]}))})
    r = c.generate("prompt", None, 5, True, 0.9)
    body = c.http.calls[0][2]["json"]
    assert r.text == "yo" and body["prompt"] == "prompt" and body["max_completion_tokens"] == 5
    assert "messages" not in body and "temperature" not in body and "stream_options" not in body
    assert c.http.calls[0][1].endswith("/v1/completions")


def test_cloud_cold_mode_adds_unique_nonce(cloud):
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(lines=chat_stream())})
    cloud.reset_native_cache()
    cloud.generate("same prompt", None, 4, True, 0)
    cloud.generate("same prompt", None, 4, True, 0)
    first, second = (c[2]["json"]["messages"][0]["content"] for c in cloud.http.calls)
    assert first.startswith("[nonce ") and first.endswith("same prompt")
    assert second == "same prompt"                       # the nonce is one-shot


def test_cloud_retries_429_then_succeeds(cloud):
    answers = iter([Resp(429, headers={"Retry-After": "1"}), Resp(lines=chat_stream())])
    cloud.http = FakeHTTP({("POST", "/chat/completions"): lambda url, kw: next(answers)})
    assert cloud.generate("p", None, 4, True, 0).text
    assert len(cloud.http.calls) == 2


def test_cloud_gives_up_after_retries(cloud):
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(503, text="busy")})
    with pytest.raises(BackendError, match="HTTP 503"):
        cloud.generate("p", None, 4, True, 0)
    assert len(cloud.http.calls) == 3


@pytest.mark.parametrize("status,msg", [(401, "authentication failed"), (404, "not found"),
                                         (400, "HTTP 400")])
def test_cloud_http_errors_have_clear_messages(cloud, status, msg):
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(status, text=f"bad {SECRET}")})
    with pytest.raises(BackendError, match=msg) as ei:
        cloud.generate("p", None, 4, True, 0)
    assert SECRET not in str(ei.value)


def test_cloud_rate_limit_message_after_retries(cloud):
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(429)})
    with pytest.raises(BackendError, match="rate limit"):
        cloud.generate("p", None, 4, True, 0)


def test_cloud_stream_error_event_and_connection_error(cloud):
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(lines=sse({"error": {"message": f"x {SECRET}"}}))})
    with pytest.raises(BackendError, match="cloud error") as ei:
        cloud.generate("p", None, 4, True, 0)
    assert SECRET not in str(ei.value)
    cloud.http = FakeHTTP({("POST", "/chat/completions"): B.requests.ConnectionError(f"down {SECRET}")})
    with pytest.raises(BackendError, match="request failed") as ei:
        cloud.generate("p", None, 4, True, 0)
    assert SECRET not in str(ei.value)


def test_cloud_ignores_malformed_stream_lines(cloud):
    lines = [b"data: {not json", b"event: ping", b'data: {"choices": [{"delta": {"content": "ok"}}]}', b"data: [DONE]"]
    cloud.http = FakeHTTP({("POST", "/chat/completions"): Resp(lines=lines)})
    assert cloud.generate("p", None, 4, True, 0).text == "ok"


def test_cloud_token_budget(cloud_settings):
    cloud_settings.cloud_token_budget = 150
    c = B.CloudBackend(cloud_settings)
    c.http = FakeHTTP({("POST", "/chat/completions"): Resp(lines=chat_stream(prompt=100))})
    c.generate("p", None, 4, True, 0)
    c.generate("p", None, 4, True, 0)                   # 100 < 150: still allowed
    with pytest.raises(BackendError, match="budget exhausted"):
        c.generate("p", None, 4, True, 0)
    assert c.describe()["token_budget"] == 150 and c.describe()["tokens_sent"] == 200


def test_cloud_extra_headers_applied(cloud_settings):
    cloud_settings.cloud_extra_headers = '{"X-Org": "acme", "HTTP-Referer": "https://e.com"}'
    c = B.CloudBackend(cloud_settings)
    assert c.http.headers["X-Org"] == "acme" and c.http.headers["HTTP-Referer"] == "https://e.com"
    cloud_settings.cloud_extra_headers = "{bad"
    with pytest.raises(BackendError, match="not valid JSON"):
        B.CloudBackend(cloud_settings)


def test_cloud_key_from_env_var(settings, monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_KEY", "from-env-key-1")
    settings.cloud_base_url, settings.cloud_model = "https://h.example/v1", "m"
    settings.cloud_api_key_env = "MY_PROVIDER_KEY"
    assert B.CloudBackend(settings).http.headers["Authorization"] == "Bearer from-env-key-1"


def test_cloud_no_key_means_no_auth_header(settings):
    settings.cloud_base_url, settings.cloud_model = "http://127.0.0.1:11434/v1", "m"
    assert "Authorization" not in B.CloudBackend(settings).http.headers


def test_cloud_tls_verification_follows_setting(cloud_settings):
    assert B.CloudBackend(cloud_settings).http.verify is True
    cloud_settings.cloud_verify_tls = False
    assert B.CloudBackend(cloud_settings).http.verify is False


def test_cloud_check_settings_problems():
    chk = B.CloudBackend.check_settings
    assert any("cloud_base_url" in p for p in chk(Settings(engine="cloud", cloud_model="m")))
    assert any("cloud_model" in p for p in chk(Settings(cloud_base_url="https://h/v1")))
    assert any("http(s)" in p for p in chk(Settings(cloud_base_url="ftp://h", cloud_model="m")))
    assert any("plain http" in p for p in chk(Settings(cloud_base_url="http://remote.example/v1",
                                                       cloud_model="m", cloud_api_key="k")))
    assert chk(Settings(cloud_base_url="http://127.0.0.1:8000/v1", cloud_model="m", cloud_api_key="k")) == []
    assert any("cloud_api_style" in p for p in chk(Settings(cloud_base_url="https://h", cloud_model="m",
                                                            cloud_api_style="x")))
    assert any("max_tokens" in p for p in chk(Settings(cloud_base_url="https://h", cloud_model="m",
                                                       cloud_max_tokens_field="")))
    assert any("budget" in p for p in chk(Settings(cloud_base_url="https://h", cloud_model="m",
                                                   cloud_token_budget=-1)))
    assert any("JSON object" in p for p in chk(Settings(cloud_base_url="https://h", cloud_model="m",
                                                        cloud_extra_headers="[1]")))


def test_cloud_provider_preset_supplies_url(settings):
    settings.cloud_provider, settings.cloud_model = "groq", "llama"
    assert B.CloudBackend(settings).base == "https://api.groq.com/openai/v1"
    settings.cloud_base_url = "https://proxy.example/v1"
    assert B.CloudBackend(settings).base == "https://proxy.example/v1"          # explicit URL wins


def test_make_backend_cloud(cloud_settings):
    assert isinstance(make_backend("openai", cloud_settings), B.CloudBackend)
