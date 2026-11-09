"""Inference-engine adapters.

Two adapters share one small interface (:class:`Backend`):

``LlamaCppBackend``
    Talks to ``llama-server`` (llama.cpp).  llama.cpp exposes the **slot
    save/restore API** (``POST /slots/{id}?action=save|restore|erase``) which
    serialises the *real KV tensors* of a sequence.  That is the integration
    point that lets an external cache layer (LMCache semantics) take KV out of
    the engine and inject it back -> **full integration**.

``OllamaBackend``
    Talks to the Ollama REST API.  Ollama deliberately hides KV state: there
    is no KV export/import and no tokenizer endpoint.  The adapter therefore
    works in **shadow mode** - the LMCache layer indexes/accounts for chunks
    and predicts reuse, while Ollama's own prefix cache does the real work and
    its ``prompt_eval_cached_count`` is used as ground truth.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import requests

from .config import Settings

log = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """Raised for any engine communication or lifecycle problem."""


@dataclass
class GenResult:
    """Normalised result of one generation call."""

    text: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0          # total tokens in the prompt
    cached_tokens: int = 0          # reused by the engine's own KV state
    computed_tokens: int = 0        # actually prefilled
    prefill_ms: float = 0.0
    generated_tokens: int = 0
    decode_tps: float = 0.0


class Backend(ABC):
    """Minimal engine interface used by :class:`kvbridge.engine.CachedEngine`."""

    name = "backend"
    supports_kv_export = False      # can KV tensors be saved/restored?
    model_id = ""
    n_ctx = 0
    kv_bytes_per_token = 0

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]: ...

    @abstractmethod
    def generate(self, prompt: str, tokens: Optional[Sequence[int]], n_predict: int,
                 cache_prompt: bool, temperature: float) -> GenResult: ...

    @abstractmethod
    def reset_native_cache(self) -> None:
        """Drop whatever prefix cache the engine keeps on its own."""

    def save_state(self) -> tuple[bytes, int]:
        raise BackendError(f"{self.name} cannot export KV state")

    def restore_state(self, blob: bytes) -> int:
        raise BackendError(f"{self.name} cannot import KV state")

    def describe(self) -> dict:
        return {"engine": self.name, "model": self.model_id, "n_ctx": self.n_ctx,
                "kv_export": self.supports_kv_export,
                "kv_bytes_per_token": self.kv_bytes_per_token}


# --------------------------------------------------------------------------
# Helpers shared by the adapters
# --------------------------------------------------------------------------

OLLAMA_MODELS_DIR = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def resolve_ollama_gguf(model: str, models_dir: Optional[Path] = None) -> Path:
    """Locate the GGUF blob of an Ollama model so llama.cpp can load it too.

    ``llama3.2:latest`` -> ``manifests/registry.ollama.ai/library/llama3.2/latest``
    whose ``application/vnd.ollama.image.model`` layer digest names the blob.
    """
    base = Path(models_dir or OLLAMA_MODELS_DIR)
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    ns = "library"
    if "/" in name:
        ns, name = name.rsplit("/", 1)
    manifest = base / "manifests" / "registry.ollama.ai" / ns / name / tag
    if not manifest.exists():
        raise BackendError(f"Ollama manifest not found for '{model}' ({manifest}). "
                           "Pull it with `ollama pull` or set KVB_MODEL_PATH.")
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        layer = next(l for l in doc["layers"] if l["mediaType"] == "application/vnd.ollama.image.model")
    except (OSError, ValueError, StopIteration, KeyError) as exc:
        raise BackendError(f"cannot parse Ollama manifest {manifest}: {exc}") from exc
    blob = base / "blobs" / layer["digest"].replace(":", "-")
    if not blob.exists():
        raise BackendError(f"model blob missing: {blob}")
    return blob


def find_llama_server(configured: str = "") -> str:
    """Find a ``llama-server`` binary (env > PATH > Homebrew > Ollama.app)."""
    candidates = [configured] if configured else []
    which = shutil.which("llama-server")
    if which:
        candidates.append(which)
    candidates += [
        "/opt/homebrew/bin/llama-server",
        "/usr/local/bin/llama-server",
        "/Applications/Ollama.app/Contents/Resources/llama-server",
    ]
    for c in candidates:
        if c and Path(c).exists() and os.access(c, os.X_OK):
            return c
    raise BackendError("llama-server not found. Install llama.cpp (`brew install llama.cpp`) "
                       "or set KVB_LLAMA_SERVER_BIN.")


def port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


# --------------------------------------------------------------------------
# llama.cpp
# --------------------------------------------------------------------------

class LlamaCppBackend(Backend):
    """llama-server adapter with real KV save/restore."""

    name = "llama.cpp"
    supports_kv_export = True

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.proc: Optional[subprocess.Popen] = None
        self.external = bool(settings.llama_url)
        self.base = settings.llama_url.rstrip("/") if self.external else (
            f"http://{settings.managed_host}:{settings.managed_port}")
        self.http = requests.Session()
        self.n_ctx = settings.ctx_size
        self._seq = 0
        self.pidfile = settings.output_dir / "llama_server.pid"

    # -- lifecycle -------------------------------------------------------
    def _model_path(self) -> Path:
        if self.s.model_path:
            p = Path(self.s.model_path).expanduser()
            if not p.exists():
                raise BackendError(f"KVB_MODEL_PATH does not exist: {p}")
            return p
        return resolve_ollama_gguf(self.s.ollama_model)

    def build_command(self) -> list[str]:
        model = self._model_path()
        self.model_id = model.name[:24] if self.s.model_path else self.s.ollama_model
        self.s.slot_dir.mkdir(parents=True, exist_ok=True)
        cmd = [find_llama_server(self.s.llama_server_bin), "-m", str(model),
               "--host", self.s.managed_host, "--port", str(self.s.managed_port),
               "-c", str(self.s.ctx_size), "-np", "1",
               "--slot-save-path", str(self.s.slot_dir.resolve()),
               "--cache-ram", str(self.s.cache_ram_mb), "--metrics", "--no-webui"]
        if self.s.n_gpu_layers >= 0:
            cmd += ["-ngl", str(self.s.n_gpu_layers)]
        if self.s.extra_args:
            cmd += self.s.extra_args.split()
        return cmd

    def start(self) -> None:
        if self.is_ready():
            log.info("attaching to llama-server at %s", self.base)
            self._probe()
            return
        if self.external:
            raise BackendError(f"external llama-server not reachable at {self.base}")
        if port_open(self.s.managed_host, self.s.managed_port):
            raise BackendError(f"port {self.s.managed_port} is busy but not a healthy llama-server; "
                               "change KVB_MANAGED_PORT")
        cmd = self.build_command()
        self.s.log_dir.mkdir(parents=True, exist_ok=True)
        logf = open(self.s.log_dir / "llama_server.log", "ab")
        log.info("starting: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.s.output_dir.mkdir(parents=True, exist_ok=True)
        self.pidfile.write_text(str(self.proc.pid))
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise BackendError(f"llama-server exited with code {self.proc.returncode}; "
                                   f"see {self.s.log_dir / 'llama_server.log'}")
            if self.is_ready():
                self._probe()
                return
            time.sleep(0.5)
        self.stop()
        raise BackendError("llama-server did not become ready within 180 s")

    def _probe(self) -> None:
        """Discover context size and KV bytes/token from the running server."""
        try:
            slots = self.http.get(f"{self.base}/slots", timeout=5).json()
            if slots:
                self.n_ctx = int(slots[0].get("n_ctx", self.n_ctx))
        except (requests.RequestException, ValueError):
            pass

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None
        self.pidfile.unlink(missing_ok=True)

    def restart(self) -> None:
        """Hard restart - wipes every bit of engine-side KV state."""
        if self.external:
            raise BackendError("cannot restart an external llama-server")
        self.stop()
        time.sleep(0.5)
        self.start()

    def is_ready(self) -> bool:
        try:
            r = self.http.get(f"{self.base}/health", timeout=1.5)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except (requests.RequestException, ValueError):
            return False

    # -- tokens & generation ----------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        r = self._post("/tokenize", {"content": text, "add_special": True}, timeout=60)
        return r.json()["tokens"]

    def _post(self, path: str, body: dict, timeout: Optional[float] = None, **kw) -> requests.Response:
        try:
            r = self.http.post(self.base + path, json=body, timeout=timeout or self.s.request_timeout, **kw)
        except requests.RequestException as exc:
            raise BackendError(f"llama-server request {path} failed: {exc}") from exc
        if r.status_code >= 400:
            raise BackendError(f"llama-server {path} -> HTTP {r.status_code}: {r.text[:200]}")
        return r

    def generate(self, prompt, tokens, n_predict, cache_prompt, temperature) -> GenResult:
        body = {"prompt": list(tokens) if tokens is not None else prompt, "n_predict": n_predict,
                "cache_prompt": cache_prompt, "id_slot": 0, "stream": True,
                "temperature": temperature, "seed": 1}
        t0 = time.perf_counter()
        res, ttft, final = GenResult(), 0.0, {}
        resp = self._post("/completion", body, stream=True)
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            try:
                msg = json.loads(line[6:])
            except ValueError:          # e.g. an OpenAI-style "[DONE]" sentinel
                continue
            if not ttft:
                ttft = (time.perf_counter() - t0) * 1000
            res.text += msg.get("content", "")
            if msg.get("stop"):
                final = msg
        res.total_ms = (time.perf_counter() - t0) * 1000
        tm = final.get("timings", {})
        res.ttft_ms = ttft
        res.cached_tokens = int(tm.get("cache_n", 0))
        res.computed_tokens = int(tm.get("prompt_n", 0))
        res.prompt_tokens = res.cached_tokens + res.computed_tokens
        res.prefill_ms = float(tm.get("prompt_ms", 0.0))
        res.generated_tokens = int(tm.get("predicted_n", 0))
        res.decode_tps = float(tm.get("predicted_per_second", 0.0))
        return res

    # -- KV state ------------------------------------------------------------
    def reset_native_cache(self) -> None:
        self._post("/slots/0?action=erase", {}, timeout=30)

    def _snap_name(self) -> str:
        self._seq += 1
        return f"stage_{os.getpid()}_{self._seq}.bin"

    def save_state(self) -> tuple[bytes, int]:
        name = self._snap_name()
        path = self.s.slot_dir / name
        try:
            info = self._post("/slots/0?action=save", {"filename": name}, timeout=120).json()
            return path.read_bytes(), int(info.get("n_saved", 0))
        except OSError as exc:
            raise BackendError(f"cannot read KV snapshot {path} (is KVB_SLOT_DIR the server's "
                               f"--slot-save-path?): {exc}") from exc
        finally:
            path.unlink(missing_ok=True)

    def restore_state(self, blob: bytes) -> int:
        name = self._snap_name()
        path = self.s.slot_dir / name
        try:
            path.write_bytes(blob)
            info = self._post("/slots/0?action=restore", {"filename": name}, timeout=120).json()
            return int(info.get("n_restored", 0))
        except OSError as exc:
            raise BackendError(f"cannot stage KV snapshot {path}: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)

    def metrics_text(self) -> str:
        try:
            return self.http.get(f"{self.base}/metrics", timeout=3).text
        except requests.RequestException:
            return ""


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

_PIECE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def pseudo_tokenize(text: str) -> list[int]:
    """Deterministic stand-in tokenizer for engines without a tokenizer API.

    Each word / punctuation / whitespace run becomes one id (CRC32).  It is
    NOT the model's vocabulary - it only has to be *stable*, so that two
    prompts sharing a text prefix share a pseudo-token prefix.
    """
    return [zlib.crc32(p.encode("utf-8")) for p in _PIECE.findall(text)]


class OllamaBackend(Backend):
    """Ollama adapter (shadow mode - Ollama cannot export its KV cache)."""

    name = "ollama"
    supports_kv_export = False

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.base = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_model
        self.model_id = self.model
        self.http = requests.Session()
        self.n_ctx = settings.ctx_size

    def start(self) -> None:
        if not self.is_ready():
            raise BackendError(f"Ollama not reachable at {self.base}. Start it with `ollama serve` "
                               "(or open Ollama.app).")
        try:
            info = self.http.post(f"{self.base}/api/show", json={"model": self.model}, timeout=10)
            if info.status_code == 404:
                raise BackendError(f"model '{self.model}' not found; run `ollama pull {self.model}`")
            mi = info.json().get("model_info", {})
            arch = mi.get("general.architecture", "llama")
            blocks = mi.get(f"{arch}.block_count", 0)
            heads = mi.get(f"{arch}.attention.head_count", 1)
            kv_heads = mi.get(f"{arch}.attention.head_count_kv", heads)
            emb = mi.get(f"{arch}.embedding_length", 0)
            if blocks and emb:
                # K and V, fp16 (2 bytes) per layer per KV head
                self.kv_bytes_per_token = int(2 * blocks * kv_heads * (emb // heads) * 2)
        except requests.RequestException as exc:
            raise BackendError(f"cannot query Ollama model info: {exc}") from exc

    def stop(self) -> None:  # Ollama is an externally managed daemon
        pass

    def is_ready(self) -> bool:
        try:
            return self.http.get(f"{self.base}/api/version", timeout=1.5).status_code == 200
        except requests.RequestException:
            return False

    def tokenize(self, text: str) -> list[int]:
        return pseudo_tokenize(text)

    def generate(self, prompt, tokens, n_predict, cache_prompt, temperature) -> GenResult:
        body = {"model": self.model, "prompt": prompt, "raw": True, "stream": True,
                "keep_alive": "10m",
                "options": {"num_predict": n_predict, "temperature": temperature, "seed": 1,
                            "num_ctx": self.s.ctx_size}}
        t0 = time.perf_counter()
        res, ttft, final = GenResult(), 0.0, {}
        try:
            resp = self.http.post(f"{self.base}/api/generate", json=body, stream=True,
                                  timeout=self.s.request_timeout)
        except requests.RequestException as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"Ollama /api/generate -> HTTP {resp.status_code}: {resp.text[:200]}")
        for line in resp.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if "error" in msg:
                raise BackendError(f"Ollama error: {msg['error']}")
            if not ttft and msg.get("response"):
                ttft = (time.perf_counter() - t0) * 1000
            res.text += msg.get("response", "")
            if msg.get("done"):
                final = msg
        res.total_ms = (time.perf_counter() - t0) * 1000
        res.ttft_ms = ttft or res.total_ms
        res.prompt_tokens = int(final.get("prompt_eval_count", 0))
        res.cached_tokens = int(final.get("prompt_eval_cached_count", 0))
        res.computed_tokens = max(res.prompt_tokens - res.cached_tokens, 0)
        res.prefill_ms = final.get("prompt_eval_duration", 0) / 1e6
        res.generated_tokens = int(final.get("eval_count", 0))
        ev = final.get("eval_duration", 0)
        res.decode_tps = res.generated_tokens / (ev / 1e9) if ev else 0.0
        return res

    def reset_native_cache(self) -> None:
        """Unload the model (drops its KV cache) and load it again, empty."""
        try:
            self.http.post(f"{self.base}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=60)
            for _ in range(40):  # wait until /api/ps no longer lists the model
                ps = self.http.get(f"{self.base}/api/ps", timeout=5).json().get("models", [])
                if not any(m.get("name", "").startswith(self.model.split(":")[0]) for m in ps):
                    break
                time.sleep(0.25)
            self.http.post(f"{self.base}/api/generate",
                           json={"model": self.model, "keep_alive": "10m",
                                 "options": {"num_ctx": self.s.ctx_size}}, timeout=120)
        except requests.RequestException as exc:
            raise BackendError(f"cannot reset Ollama cache: {exc}") from exc


def make_backend(kind: str, settings: Settings) -> Backend:
    """Factory: ``llamacpp`` or ``ollama``."""
    k = kind.lower().replace(".", "").replace("-", "").replace("_", "")
    if k in ("llamacpp", "llama"):
        return LlamaCppBackend(settings)
    if k == "ollama":
        return OllamaBackend(settings)
    raise BackendError(f"unknown engine '{kind}' (expected 'llamacpp' or 'ollama')")
