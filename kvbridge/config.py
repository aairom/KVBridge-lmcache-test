"""Runtime configuration for KVBridge.

All values come from environment variables (optionally loaded from a ``.env``
file) so that nothing is hard-coded in the application.  See ``.env.example``
for the full list of ``KVB_*`` variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "yes", "on"}


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


@dataclass
class Settings:
    """Application settings (mutable so the GUI can apply changes)."""

    # --- engines -----------------------------------------------------------
    llama_server_bin: str = ""        # empty -> auto-detect
    llama_url: str = ""               # empty -> start a managed llama-server
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:latest"
    model_path: str = ""              # explicit GGUF path (overrides ollama_model)
    managed_host: str = "127.0.0.1"
    managed_port: int = 8089
    ctx_size: int = 8192
    n_gpu_layers: int = -1            # -1 -> llama.cpp default
    cache_ram_mb: int = 0             # llama.cpp built-in host prompt cache (0=off)
    extra_args: str = ""              # extra llama-server flags
    request_timeout: float = 300.0

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

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, load_dotenv: bool = True) -> "Settings":
        """Build settings from ``env`` (defaults to ``os.environ`` + ``.env``)."""
        if env is None:
            if load_dotenv:
                try:
                    from dotenv import load_dotenv as _load

                    _load(PROJECT_ROOT / ".env", override=False)
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

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""
        problems: list[str] = []
        if self.chunk_size < 1:
            problems.append("chunk_size must be >= 1")
        if self.ctx_size < self.chunk_size:
            problems.append("ctx_size must be >= chunk_size")
        if self.l1_mb < 0 or self.l2_mb < 0:
            problems.append("tier capacities must be >= 0")
        if self.l1_mb == 0 and self.l2_mb == 0:
            problems.append("at least one cache tier (L1 or L2) must have capacity")
        if not (1 <= self.managed_port <= 65535):
            problems.append("managed_port must be a valid TCP port")
        if self.managed_port in (5000, 8080, 9931, 11434):
            problems.append(f"managed_port {self.managed_port} collides with a reserved/known service")
        return problems

    @property
    def l1_bytes(self) -> int:
        return self.l1_mb * 1024 * 1024

    @property
    def l2_bytes(self) -> int:
        return self.l2_mb * 1024 * 1024
