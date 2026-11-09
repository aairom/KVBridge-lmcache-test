"""Runtime configuration for KVBridge.

Every value that influences an inference engine lives in :class:`Settings`.
Values come from (highest priority first):

1. the GUI ("Engine & Cache" tab) - applied with :meth:`Settings.apply`,
2. process environment variables ``KVB_<FIELD>``,
3. a ``.env`` file in the project root,
4. the defaults declared below.

Nothing engine-related is hard-coded in the adapters: they only read this
object.  See ``.env.example`` for the full list of ``KVB_*`` variables.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

_TRUE = {"1", "true", "yes", "on"}

#: Settings that must never be logged, exported or written without consent.
SECRET_FIELDS = frozenset({"cloud_api_key"})

#: Convenience presets for the *cloud* engine (OpenAI-compatible base URLs).
#: They are only suggestions for the GUI drop-down: an explicit
#: ``KVB_CLOUD_BASE_URL`` always wins, and any other URL can be typed in.
PROVIDER_PRESETS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "vllm-local": "http://127.0.0.1:8000/v1",
    "lmstudio-local": "http://127.0.0.1:1234/v1",
    "ollama-local-openai": "http://127.0.0.1:11434/v1",
}

CLOUD_API_STYLES = ("chat", "completions")

#: Default places probed for ``llama-server`` (extended by ``KVB_LLAMA_SEARCH_PATHS``).
DEFAULT_LLAMA_SEARCH_PATHS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/Applications/Ollama.app/Contents/Resources",
)


def _bool(value: str) -> bool:
    return value.strip().lower() in _TRUE


def _get(env: Mapping[str, str], name: str, default: Any) -> Any:
    """Read ``KVB_<name>`` from *env*, casting to the type of *default*."""
    raw = env.get(f"KVB_{name}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return _bool(raw)
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def _coerce(default: Any, value: Any) -> Any:
    """Cast a GUI/dict *value* to the type of the field's *default*."""
    if isinstance(default, bool):
        return value if isinstance(value, bool) else _bool(str(value))
    if isinstance(default, int):
        if value is None or value == "":
            raise ValueError("a number is required")
        return int(float(value))
    if isinstance(default, float):
        if value is None or value == "":
            raise ValueError("a number is required")
        return float(value)
    if isinstance(default, Path):
        return Path(str(value)).expanduser()
    return "" if value is None else str(value).strip()


@dataclass
class Settings:
    """Application settings (mutable so the GUI can apply changes)."""

    # --- engine selection --------------------------------------------------
    #: Key of the engine selected at start-up: ``llamacpp``, ``ollama`` or
    #: ``cloud`` (see :func:`kvbridge.backends.engine_choices`).
    engine: str = "llamacpp"

    # --- local engines: llama.cpp ------------------------------------------
    llama_server_bin: str = ""        # empty -> auto-detect
    llama_search_paths: str = ""      # extra dirs/files probed (os.pathsep separated)
    llama_url: str = ""               # empty -> start a managed llama-server
    llama_start_timeout: float = 180.0  # seconds to wait for a managed server
    managed_host: str = "127.0.0.1"
    managed_port: int = 8089
    reserved_ports: str = "5000,8080,9931,11434"  # ports the managed server must avoid
    model_path: str = ""              # explicit GGUF path (overrides ollama_model)
    ctx_size: int = 8192
    n_gpu_layers: int = -1            # -1 -> llama.cpp default
    cache_ram_mb: int = 0             # llama.cpp built-in host prompt cache (0=off)
    extra_args: str = ""              # extra llama-server flags

    # --- local engines: Ollama ---------------------------------------------
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_models_dir: str = ""       # empty -> $OLLAMA_MODELS or ~/.ollama/models
    ollama_keep_alive: str = "10m"

    # --- cloud engine (any OpenAI-compatible HTTP API) ---------------------
    cloud_provider: str = ""          # key of PROVIDER_PRESETS (optional)
    cloud_base_url: str = ""          # overrides the preset, e.g. https://host/v1
    cloud_model: str = ""
    cloud_api_key: str = field(default="", repr=False)   # secret - never logged
    cloud_api_key_env: str = ""       # name of an env var holding the key instead
    cloud_extra_headers: str = ""     # JSON object, e.g. {"HTTP-Referer": "..."}
    cloud_api_style: str = "chat"     # "chat" (/chat/completions) or "completions"
    cloud_max_tokens_field: str = "max_tokens"  # some APIs want max_completion_tokens
    cloud_send_temperature: bool = True
    cloud_stream_usage: bool = True   # ask for token usage in the stream
    cloud_verify_tls: bool = True
    cloud_kv_bytes_per_token: int = 0  # only used for the capacity accounting (0 = unknown)
    cloud_token_budget: int = 0       # soft cap on prompt tokens sent per session (0 = off)

    # --- shared by all engines ---------------------------------------------
    request_timeout: float = 300.0
    seed: int = 1

    # --- KV cache layer ----------------------------------------------------
    chunk_size: int = 256
    l1_mb: int = 2048
    l2_mb: int = 8192
    write_through: bool = True
    use_lmcache_hasher: bool = False  # use the real lmcache TokenHasher if installed
    store_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "kv_store")
    # Directory shared with llama-server (--slot-save-path); KV snapshots are
    # staged here for the few milliseconds of a save/restore call.
    slot_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "slot_stage")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "logs")
    input_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "input")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    # --- generation / GUI --------------------------------------------------
    n_predict: int = 48
    temperature: float = 0.2
    gui_host: str = "127.0.0.1"
    gui_port: int = 7871

    # ------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, load_dotenv: bool = True) -> "Settings":
        """Build settings from ``env`` (defaults to ``os.environ`` + ``.env``)."""
        if env is None:
            if load_dotenv:
                try:
                    from dotenv import load_dotenv as _load

                    _load(ENV_FILE, override=False)
                except ImportError:  # python-dotenv is optional at runtime
                    pass
            env = os.environ
        d = cls()
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            default = getattr(d, f.name)
            if isinstance(default, Path):
                raw = env.get(f"KVB_{f.name.upper()}")
                kwargs[f.name] = Path(raw).expanduser() if raw else default
            else:
                kwargs[f.name] = _get(env, f.name.upper(), default)
        return cls(**kwargs)

    def apply(self, values: Mapping[str, Any]) -> list[str]:
        """Atomically update fields from *values* (e.g. from the GUI).

        Returns the list of problems; when it is not empty nothing was changed.
        """
        known = {f.name for f in fields(self)}
        coerced: dict[str, Any] = {}
        problems: list[str] = []
        for name, raw in values.items():
            if name not in known:
                problems.append(f"unknown setting '{name}'")
                continue
            try:
                coerced[name] = _coerce(getattr(self, name), raw)
            except (TypeError, ValueError) as exc:
                problems.append(f"{name}: {exc}")
        if problems:
            return problems
        candidate = replace(self, **coerced)
        problems = candidate.validate()
        if not problems:
            self.__dict__.update(candidate.__dict__)
        return problems

    # -------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""
        from .backends import BackendError, resolve_engine   # lazy: avoids an import cycle

        problems: list[str] = []
        if self.chunk_size < 1:
            problems.append("chunk_size must be >= 1")
        if self.ctx_size < self.chunk_size:
            problems.append("ctx_size must be >= chunk_size")
        if self.l1_mb < 0 or self.l2_mb < 0:
            problems.append("tier capacities must be >= 0")
        if self.l1_mb == 0 and self.l2_mb == 0:
            problems.append("at least one cache tier (L1 or L2) must have capacity")
        if self.request_timeout <= 0:
            problems.append("request_timeout must be > 0")
        try:
            problems += resolve_engine(self.engine).cls.check_settings(self)
        except BackendError as exc:
            problems.append(str(exc))
        return problems

    # ------------------------------------------------------- derived / helpers
    @property
    def l1_bytes(self) -> int:
        return self.l1_mb * 1024 * 1024

    @property
    def l2_bytes(self) -> int:
        return self.l2_mb * 1024 * 1024

    @property
    def reserved_port_set(self) -> set[int]:
        out: set[int] = set()
        for part in self.reserved_ports.replace(";", ",").split(","):
            part = part.strip()
            if part.isdigit():
                out.add(int(part))
        return out

    def resolved_cloud_url(self) -> str:
        """Explicit ``cloud_base_url`` first, else the preset's URL, else ``""``."""
        return (self.cloud_base_url or PROVIDER_PRESETS.get(self.cloud_provider, "")).rstrip("/")

    def resolved_cloud_key(self) -> str:
        """The API key: ``cloud_api_key`` first, else the env var named by ``cloud_api_key_env``."""
        if self.cloud_api_key:
            return self.cloud_api_key
        return os.environ.get(self.cloud_api_key_env, "") if self.cloud_api_key_env else ""

    def cloud_headers(self) -> dict[str, str]:
        """Parsed ``cloud_extra_headers`` (raises ``ValueError`` when malformed)."""
        if not self.cloud_extra_headers:
            return {}
        data = json.loads(self.cloud_extra_headers)
        if not isinstance(data, dict):
            raise ValueError("cloud_extra_headers must be a JSON object")
        return {str(k): str(v) for k, v in data.items()}

    # ---------------------------------------------------------- display / .env
    def to_display(self) -> dict[str, Any]:
        """Settings as plain JSON-able values with secrets masked."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in SECRET_FIELDS:
                out[f.name] = "********" if (v or self.resolved_cloud_key()) else ""
            else:
                out[f.name] = str(v) if isinstance(v, Path) else v
        return out

    def env_values(self, include_secrets: bool = False) -> dict[str, str]:
        """``KVB_*`` -> string mapping suitable for a ``.env`` file.

        Folder settings are skipped (they stay relative to the project root) and
        secrets are only included on request.
        """
        out: dict[str, str] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Path) or (f.name in SECRET_FIELDS and not include_secrets):
                continue
            out[f"KVB_{f.name.upper()}"] = ("true" if v else "false") if isinstance(v, bool) else str(v)
        return out


# ----------------------------------------------------------------------------
# .env writer
# ----------------------------------------------------------------------------

def _env_quote(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("newlines are not allowed in .env values")
    if value == "" or any(c in value for c in ' #"\'$\\'):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def update_env_file(path: Path, values: Mapping[str, str]) -> Path:
    """Create/update *path*, changing only the given ``KEY=value`` lines.

    Comments, ordering and unrelated keys are preserved; missing keys are
    appended.  The file is written atomically with mode ``0600`` because it may
    contain an API key.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(values)
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
        if key in remaining:
            out.append(f"{key}={_env_quote(remaining.pop(key))}")
        else:
            out.append(line)
    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.append("# ---- saved from the KVBridge GUI ----")
        out += [f"{k}={_env_quote(v)}" for k, v in remaining.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
