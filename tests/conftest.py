"""Shared pytest fixtures: in-memory fake engines so no real LLM is required."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Optional, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvbridge.backends import Backend, BackendError, GenResult, pseudo_tokenize  # noqa: E402
from kvbridge.config import Settings  # noqa: E402


class FakeLlama(Backend):
    """Simulates llama.cpp: one slot, prefix reuse and KV save/restore."""

    name = "fake-llama"
    supports_kv_export = True
    model_id = "fake-model"
    n_ctx = 4096
    kv_bytes_per_token = 100

    def __init__(self) -> None:
        self.slot: list[int] = []
        self.calls: list[str] = []
        self.fail_restore = False
        self.fail_save = False
        self.fail_generate = False
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def is_ready(self) -> bool:
        return self.started

    def tokenize(self, text: str) -> list[int]:
        return [1] + pseudo_tokenize(text)          # leading "BOS" like llama.cpp

    def generate(self, prompt, tokens, n_predict, cache_prompt, temperature) -> GenResult:
        self.calls.append("generate")
        if self.fail_generate:
            raise BackendError("boom")
        tokens = list(tokens if tokens is not None else self.tokenize(prompt))
        cached = 0
        if cache_prompt:
            while cached < min(len(self.slot), len(tokens)) and self.slot[cached] == tokens[cached]:
                cached += 1
            cached = min(cached, len(tokens) - 1)    # llama.cpp always re-evaluates >=1 token
        self.slot = tokens
        computed = len(tokens) - cached
        return GenResult(text=" answer ", ttft_ms=1.0 + computed * 0.1, total_ms=5.0,
                         prompt_tokens=len(tokens), cached_tokens=cached, computed_tokens=computed,
                         prefill_ms=computed * 0.1, generated_tokens=4, decode_tps=50.0)

    def reset_native_cache(self) -> None:
        self.calls.append("reset")
        self.slot = []

    def save_state(self) -> tuple[bytes, int]:
        self.calls.append("save")
        if self.fail_save:
            raise BackendError("save failed")
        return struct.pack(f">{len(self.slot)}I", *self.slot) + b"\0" * 64, len(self.slot)

    def restore_state(self, blob: bytes) -> int:
        self.calls.append("restore")
        if self.fail_restore:
            raise BackendError("restore failed")
        n = (len(blob) - 64) // 4
        self.slot = list(struct.unpack(f">{n}I", blob[: n * 4]))
        return n

    def restart(self) -> None:
        self.calls.append("restart")
        self.slot = []


class FakeOllama(FakeLlama):
    """Simulates Ollama: hidden KV state, no tokenizer, approximate token counts."""

    name = "fake-ollama"
    supports_kv_export = False
    kv_bytes_per_token = 1000

    def tokenize(self, text: str) -> list[int]:
        return pseudo_tokenize(text)

    def save_state(self):
        raise BackendError("cannot export")

    def restore_state(self, blob):
        raise BackendError("cannot import")

    def __getattribute__(self, item):
        if item == "restart":               # the real OllamaBackend has no restart()
            raise AttributeError(item)
        return super().__getattribute__(item)


class FakeCloud(FakeOllama):
    """Simulates a hosted OpenAI-compatible engine: shadow mode, prompts leave the machine."""

    name = "fake-cloud"
    kind = "cloud"
    kv_bytes_per_token = 0


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        chunk_size=4, ctx_size=4096, l1_mb=8, l2_mb=16, n_predict=4,
        store_dir=tmp_path / "store", slot_dir=tmp_path / "slots",
        log_dir=tmp_path / "logs", input_dir=tmp_path / "input", output_dir=tmp_path / "output",
    )


@pytest.fixture
def fake() -> FakeLlama:
    b = FakeLlama()
    b.start()
    return b


@pytest.fixture
def fake_ollama() -> FakeOllama:
    b = FakeOllama()
    b.start()
    return b


@pytest.fixture
def fake_cloud() -> FakeCloud:
    b = FakeCloud()
    b.start()
    return b


def words(n: int, prefix: str = "w") -> str:
    """Deterministic text with *n* whitespace-separated words."""
    return " ".join(f"{prefix}{i}" for i in range(n))
