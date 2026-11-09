"""Demonstration scenarios that make the value of KV cache reuse visible.

Each scenario returns a list of :class:`~kvbridge.metrics.RequestRecord`.
Documents placed in ``input/`` (``.txt``/``.md``, scanned recursively) are used
as the long shared context; a built-in corpus is the fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .engine import CachedEngine
from .metrics import RequestRecord

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".rst"}

_BUILTIN = (
    "LMCache is a KV cache management layer for large language model inference. "
    "During prefill an LLM turns every prompt token into key and value tensors, the KV cache, "
    "which later tokens attend to. Recomputing that state for text the model has already seen "
    "wastes GPU or CPU cycles and inflates time to first token. LMCache splits the token stream "
    "into fixed size chunks, names every chunk with a rolling prefix hash and stores the tensors "
    "in a tiered hierarchy made of device memory, host memory, local disk and remote backends. "
    "When a later request begins with the same tokens the stored tensors are loaded instead of "
    "being recomputed, even if the first copy was produced by another session or another engine "
    "instance. Typical beneficiaries are multi turn chat, retrieval augmented generation with "
    "recurring documents, agent loops with long tool descriptions and code assistants that resend "
    "the same repository context. "
)

QUESTIONS = [
    "In one sentence, what problem does the KV cache layer solve?",
    "Name two workloads that benefit most from reuse.",
    "What identifies a chunk in the cache?",
    "Which storage tiers are mentioned?",
    "Why does reuse reduce time to first token?",
    "Summarise the text in five words.",
]


def load_documents(input_dir: Path) -> list[tuple[str, str]]:
    """Recursively read text documents from *input_dir*."""
    docs: list[tuple[str, str]] = []
    if input_dir.exists():
        for p in sorted(input_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace").strip()
                except OSError as exc:
                    log.warning("skipping %s: %s", p, exc)
                    continue
                if txt:
                    docs.append((str(p.relative_to(input_dir)), txt))
    return docs


def build_context(engine: CachedEngine, target_tokens: int, input_dir: Path) -> tuple[str, str]:
    """Return ``(text, source)`` of roughly ``target_tokens`` tokens.

    The context is clamped so that prompt + answer always fit in the engine's
    context window.
    """
    limit = max(engine.backend.n_ctx - 600, 256) if engine.backend.n_ctx else target_tokens
    target = min(target_tokens, limit)
    docs = load_documents(input_dir)
    base, source = ("\n\n".join(t for _, t in docs), f"{len(docs)} file(s) from input/") if docs \
        else (_BUILTIN, "built-in sample text")
    text = base
    guard = 0
    while len(engine.backend.tokenize(text)) < target and guard < 200:
        text += "\n" + base
        guard += 1
    # trim by characters until it fits
    while len(engine.backend.tokenize(text)) > target and len(text) > 200:
        text = text[: int(len(text) * 0.95)]
    return text, source


@dataclass
class Scenario:
    key: str
    title: str
    blurb: str
    run: Callable[..., list[RequestRecord]]


def _progress(cb: Optional[Callable[[str], None]], msg: str) -> None:
    log.info(msg)
    if cb:
        cb(msg)


def scenario_document_qa(engine: CachedEngine, context_tokens: int = 2000, n_questions: int = 4,
                         on_progress: Optional[Callable[[str], None]] = None, **_) -> list[RequestRecord]:
    """Same long document, different questions: classic RAG / doc-QA reuse.

    Runs the three modes back to back so the TTFT gap is directly comparable.
    """
    ctx, src = build_context(engine, context_tokens, engine.s.input_dir)
    _progress(on_progress, f"Document QA over {src}")
    recs: list[RequestRecord] = []
    engine.reset(wipe_store=True)
    for mode in ("cold", "native", "lmcache"):
        for i in range(n_questions):
            q = QUESTIONS[i % len(QUESTIONS)]
            prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
            _progress(on_progress, f"[{mode}] question {i + 1}/{n_questions}")
            recs.append(engine.infer(prompt, mode, label=f"docQA q{i + 1}"))
    return recs


def scenario_multi_turn(engine: CachedEngine, turns: int = 5, system_tokens: int = 1200,
                        on_progress: Optional[Callable[[str], None]] = None, **_) -> list[RequestRecord]:
    """A growing conversation: every turn re-sends the full history."""
    system, _src = build_context(engine, system_tokens, engine.s.input_dir)
    recs: list[RequestRecord] = []
    engine.reset(wipe_store=True)
    for mode in ("cold", "lmcache"):
        history = f"System: You are a concise assistant. Reference material:\n{system}\n"
        for t in range(turns):
            history += f"User: {QUESTIONS[t % len(QUESTIONS)]}\nAssistant:"
            _progress(on_progress, f"[{mode}] turn {t + 1}/{turns}")
            r = engine.infer(history, mode, label=f"chat turn {t + 1}", n_predict=24)
            recs.append(r)
            history += f" {r.text[:120]}\n"
    return recs


def scenario_shared_prefix(engine: CachedEngine, users: int = 4, system_tokens: int = 1500,
                           on_progress: Optional[Callable[[str], None]] = None, **_) -> list[RequestRecord]:
    """Many users share one system prompt (e.g. an agent's tool catalogue)."""
    system, _src = build_context(engine, system_tokens, engine.s.input_dir)
    recs: list[RequestRecord] = []
    engine.reset(wipe_store=True)
    for mode in ("cold", "lmcache"):
        for u in range(users):
            prompt = f"{system}\n\nUser #{u + 1} asks: {QUESTIONS[u % len(QUESTIONS)]}\nAnswer:"
            _progress(on_progress, f"[{mode}] user {u + 1}/{users}")
            recs.append(engine.infer(prompt, mode, label=f"user {u + 1}"))
    return recs


def scenario_restart(engine: CachedEngine, context_tokens: int = 2000,
                     on_progress: Optional[Callable[[str], None]] = None, **_) -> list[RequestRecord]:
    """Engine restart: native cache is lost, LMCache's disk tier survives."""
    ctx, _src = build_context(engine, context_tokens, engine.s.input_dir)
    engine.reset(wipe_store=True)
    recs: list[RequestRecord] = []
    prompt = lambda q: f"{ctx}\n\nQuestion: {q}\nAnswer:"
    _progress(on_progress, "warming the cache")
    recs.append(engine.infer(prompt(QUESTIONS[0]), "lmcache", label="warm-up (miss)"))
    _progress(on_progress, "simulating an engine restart")
    msg = engine.simulate_engine_restart()
    _progress(on_progress, msg)
    recs.append(engine.infer(prompt(QUESTIONS[1]), "cold", label="after restart, no cache"))
    recs.append(engine.infer(prompt(QUESTIONS[2]), "lmcache", label="after restart, L2 restore"))
    recs.append(engine.infer(prompt(QUESTIONS[3]), "lmcache", label="then served from L1"))
    return recs


SCENARIOS: dict[str, Scenario] = {
    s.key: s for s in (
        Scenario("doc_qa", "Document Q&A (RAG-style reuse)",
                 "One long document, several questions - cold vs native vs LMCache.", scenario_document_qa),
        Scenario("multi_turn", "Multi-turn chat",
                 "A conversation whose history grows each turn.", scenario_multi_turn),
        Scenario("shared_prefix", "Shared system prompt",
                 "Several users share the same long system prompt.", scenario_shared_prefix),
        Scenario("restart", "Engine restart / persistence",
                 "Engine memory is wiped; the disk tier restores the KV state.", scenario_restart),
    )
}
