"""Gradio dashboard for KVBridge.

Run with ``python -m kvbridge.app`` (or ``scripts/start.sh``).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from typing import Optional

# Switch off Gradio's anonymous usage telemetry (nothing is sent anywhere by the GUI itself).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402  (import after the environment is configured)

from . import __version__
from .backends import BackendError, engine_choices, make_backend, resolve_engine
from .charts import (chunk_map_figure, entries_table, fmt_bytes, hit_rate_figure, kpi_markdown,
                     mode_compare_figure, records_table, tier_figure, tokens_figure, ttft_figure)
from .config import CLOUD_API_STYLES, ENV_FILE, PROVIDER_PRESETS, Settings, update_env_file
from .engine import CachedEngine
from .scenarios import SCENARIOS, build_context

log = logging.getLogger("kvbridge.app")

#: Settings editable in the "Engine & Cache" tab, in the order of the form components.
FORM_FIELDS = (
    "engine", "ctx_size", "chunk_size", "l1_mb", "l2_mb", "write_through", "use_lmcache_hasher",
    "seed", "request_timeout",
    "ollama_model", "reserved_ports",
    "llama_server_bin", "llama_search_paths", "llama_url", "model_path", "managed_port",
    "llama_start_timeout", "n_gpu_layers", "cache_ram_mb", "extra_args",
    "ollama_url", "ollama_keep_alive", "ollama_models_dir",
    "cloud_provider", "cloud_base_url", "cloud_model", "cloud_api_key", "cloud_api_key_env",
    "cloud_extra_headers", "cloud_api_style", "cloud_max_tokens_field", "cloud_send_temperature",
    "cloud_stream_usage", "cloud_verify_tls", "cloud_token_budget", "cloud_kv_bytes_per_token",
)

CLOUD_WARNING = ("**Cloud engine selected - your prompts and documents are sent to the remote service "
                 "and may be billed.** Use a key with a spending limit. The API key is kept in memory "
                 "only; it is written to `.env` solely when you tick the box below.")


def form_to_values(raw: dict) -> dict:
    """Turn raw form values into a ``Settings.apply`` mapping.

    A blank API-key box means "keep the current key": the secret is never echoed
    back into the form, so an empty value must not wipe it.
    """
    values = dict(raw)
    if not str(values.get("cloud_api_key") or "").strip():
        values.pop("cloud_api_key", None)
    return values


def engine_visibility(engine_key: str) -> dict[str, bool]:
    """Which setting groups to show for *engine_key* (unknown engines show none)."""
    try:
        key = resolve_engine(engine_key).key
        kind = resolve_engine(engine_key).location
    except BackendError:
        return {"llamacpp": False, "ollama": False, "cloud": False, "local_model": False, "warning": False}
    return {"llamacpp": key == "llamacpp", "ollama": key == "ollama", "cloud": key == "cloud",
            "local_model": key in ("llamacpp", "ollama"), "warning": kind == "cloud"}


def describe_connection(d: dict, hasher_impl: str) -> str:
    """Human-readable status for a freshly started backend (engine-agnostic)."""
    place = ("remote service - prompts leave this machine" if d.get("location") == "cloud"
             else "this machine")
    mode = ("FULL integration: real KV tensors are saved / restored through llama.cpp slots."
            if d["kv_export"] else
            "SHADOW mode: the engine hides its KV cache, so LMCache-style chunk indexing and "
            "accounting run next to the engine's own prefix cache.")
    lines = [f"Connected to {d['engine']} ({place}) - model {d['model']} - ctx {d['n_ctx']}", mode,
             f"Hasher: {hasher_impl}"]
    if d.get("kv_bytes_per_token"):
        lines.append(f"KV size: ~{fmt_bytes(d['kv_bytes_per_token'])}/token")
    if d.get("token_budget"):
        lines.append(f"Prompt-token budget: {d['token_budget']:,}")
    return "\n".join(lines)

CSS = """
.gradio-container {font-family: 'IBM Plex Sans', Helvetica, Arial, sans-serif !important;}
.kpi table {width:100%; text-align:center; font-size:1.15rem}
.kpi th {background:#f4f4f4; color:#525252; font-weight:600}
.note {color:#525252; font-size:0.9rem}
"""


class AppState:
    """Process-wide mutable state shared by all GUI callbacks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: Optional[CachedEngine] = None
        self.last_status: list[str] = []
        self.last_label = ""
        self.lock = threading.RLock()

    def require(self) -> CachedEngine:
        if self.engine is None:
            raise gr.Error("Not connected. Open the 'Engine & Cache' tab and click Connect.")
        return self.engine


def build_app(state: AppState) -> gr.Blocks:
    s = state.settings

    # ------------------------------------------------------------- handlers
    def connect(*vals):
        """Apply the whole form atomically, then (re)start the selected engine."""
        values = form_to_values(dict(zip(FORM_FIELDS, vals)))
        with state.lock:
            if state.engine is not None:
                state.engine.stop()
                state.engine = None
            problems = s.apply(values)
            if problems:
                return "Invalid configuration (nothing was changed):\n- " + "\n- ".join(problems)
            try:
                eng = CachedEngine(s, make_backend(s.engine, s))
                eng.start()
            except BackendError as exc:
                return f"Could not start the engine:\n{exc}"
            except Exception as exc:  # pragma: no cover - defensive, surfaces in GUI
                log.exception("connect failed")
                return f"Unexpected error: {type(exc).__name__}"
            state.engine = eng
            return describe_connection(eng.backend.describe(), eng.hasher.impl)

    def on_engine_change(engine_key):
        vis = engine_visibility(engine_key)
        return (gr.update(visible=vis["llamacpp"]), gr.update(visible=vis["ollama"]),
                gr.update(visible=vis["cloud"]), gr.update(visible=vis["local_model"]),
                gr.update(visible=vis["warning"]))

    def on_provider_change(provider):
        # Fill the URL box with the preset; the user can still overwrite it.
        return PROVIDER_PRESETS.get(provider or "", gr.update())

    def save_env(include_key, *vals):
        """Write the form's values to ``.env`` without touching the live settings."""
        import dataclasses
        values = form_to_values(dict(zip(FORM_FIELDS, vals)))
        candidate = dataclasses.replace(s)
        problems = candidate.apply(values)
        if problems:
            return "Not saved - invalid configuration:\n- " + "\n- ".join(problems)
        env = candidate.env_values(include_secrets=bool(include_key))
        env = {k: v for k, v in env.items() if v != "" or k != "KVB_CLOUD_API_KEY"}
        try:
            path = update_env_file(ENV_FILE, env)
        except (OSError, ValueError) as exc:
            return f"Could not write {ENV_FILE.name}: {exc}"
        note = "including the API key" if include_key and candidate.cloud_api_key else "API key NOT written"
        return f"Saved {len(env)} setting(s) to {path.name} ({note}). Restart-safe, mode 0600."

    def show_effective():
        import json
        return json.dumps(s.to_display(), indent=2, default=str)

    def disconnect():
        with state.lock:
            if state.engine:
                state.engine.stop()
                state.engine = None
        return "Disconnected (a managed llama-server was stopped)."

    def refresh():
        eng = state.engine
        if eng is None:
            empty = ttft_figure([])
            return ("*Not connected.*", empty, tokens_figure([]), hit_rate_figure([]),
                    mode_compare_figure({}), tier_figure({}), records_table([]))
        snap = eng.snapshot()
        recs = eng.metrics.records()
        return (kpi_markdown(snap), ttft_figure(recs), tokens_figure(recs), hit_rate_figure(recs),
                mode_compare_figure(snap["metrics"]), tier_figure(snap["store"]), records_table(recs[-25:]))

    def play(prompt, context_tokens, mode, n_predict, temperature):
        eng = state.require()
        if not prompt.strip():
            raise gr.Error("Enter a question or prompt.")
        ctx, _src = build_context(eng, int(context_tokens), s.input_dir) if context_tokens else ("", "")
        full = f"{ctx}\n\nQuestion: {prompt}\nAnswer:" if ctx else prompt
        rec = eng.infer(full, mode, label="playground", n_predict=int(n_predict), temperature=float(temperature))
        if rec.error:
            raise gr.Error(rec.error)
        state.last_status, state.last_label = rec.chunk_status, f"({mode})"
        info = (f"mode **{rec.mode}** - tier **{rec.tier}** - prompt {rec.prompt_tokens} tok, "
                f"prefilled {rec.computed_tokens}, reused {rec.server_cached_tokens} - "
                f"chunks hit {rec.chunks_hit}/{rec.chunks_total} - TTFT **{rec.ttft_ms:.0f} ms** - "
                f"restore {rec.restore_ms:.0f} ms - save {rec.save_ms:.0f} ms - "
                f"decode {rec.decode_tps:.1f} tok/s" + (" - *shadow*" if rec.shadow else ""))
        return rec.text, info, chunk_map_figure(rec.chunk_status, rec.mode)

    def run_scenario(key, ctx_tokens, n_items, progress=gr.Progress()):
        eng = state.require()
        sc = SCENARIOS[key]
        log_lines: list[str] = []

        def cb(msg):
            log_lines.append(msg)
            progress(None, desc=msg)

        kwargs = {"context_tokens": int(ctx_tokens), "system_tokens": int(ctx_tokens), "on_progress": cb}
        if key == "doc_qa":
            kwargs["n_questions"] = int(n_items)
        elif key == "multi_turn":
            kwargs["turns"] = int(n_items)
        elif key == "shared_prefix":
            kwargs["users"] = int(n_items)
        recs = sc.run(eng, **kwargs)
        errs = [r.error for r in recs if r.error]
        if errs:
            raise gr.Error(errs[0])
        lm = [r for r in recs if r.mode == "lmcache" and r.chunks_hit]
        state.last_status = (lm[-1].chunk_status if lm else (recs[-1].chunk_status if recs else []))
        snap = eng.snapshot()["metrics"]
        sp = snap.get("ttft_speedup_vs_cold", 0)
        cold = snap["by_mode"]["cold"]["avg_ttft_ms"]
        lmc = snap["by_mode"]["lmcache"]["avg_ttft_ms"]
        summary = (f"### {sc.title}\n{len(recs)} requests. Average TTFT: cold **{cold:.0f} ms** vs LMCache "
                   f"**{lmc:.0f} ms** -> **{sp:.1f}x** faster. Chunk hit rate **{snap['chunk_hit_rate'] * 100:.0f}%**, "
                   f"token reuse **{snap['token_reuse_ratio'] * 100:.0f}%**.")
        if not eng.backend.supports_kv_export:
            summary += ("\n\n*Shadow mode: the engine's own prefix cache provides the reuse, KVBridge "
                        "indexes it.*")
        if eng.backend.kind == "cloud":
            summary += "\n\n*Cloud engine: these runs sent prompts to the remote service and consumed tokens.*"
        return (summary, records_table(recs), ttft_figure(recs), tokens_figure(recs),
                chunk_map_figure(state.last_status, "(last LMCache request)"))

    def inspector():
        eng = state.engine
        if eng is None:
            return entries_table([]), chunk_map_figure([]), "Not connected."
        st = eng.store.stats()
        txt = (f"L1 RAM {fmt_bytes(st['l1_used'])} / {fmt_bytes(st['l1_cap'])} - "
               f"L2 disk {fmt_bytes(st['l2_used'])} / {fmt_bytes(st['l2_cap'])} - "
               f"{st['entries']} snapshot(s), {st['chunks_indexed']} chunk hashes, {st['tokens_cached']:,} tokens - "
               f"evictions L1 {int(eng.metrics.get('evictions_l1'))} / L2 {int(eng.metrics.get('evictions_l2'))}")
        return entries_table(eng.store.entries()), chunk_map_figure(state.last_status, state.last_label), txt

    def clear_l1():
        state.require().store.clear("l1")
        return inspector()

    def clear_all():
        state.require().store.clear("all")
        return inspector()

    def restart_sim():
        eng = state.require()
        try:
            msg = eng.simulate_engine_restart()
        except BackendError as exc:
            raise gr.Error(str(exc))
        return (msg,) + inspector()

    def resize(l1, l2):
        eng = state.require()
        eng.store.resize(int(l1) * 1024 * 1024, int(l2) * 1024 * 1024)
        return inspector()

    def reset_metrics():
        eng = state.require()
        eng.reset(wipe_store=False)
        return refresh()

    def export_metrics():
        eng = state.require()
        return str(eng.metrics.export(s.output_dir))

    # ------------------------------------------------------------------ UI
    with gr.Blocks(title="KVBridge - LMCache-style KV cache demo", fill_width=True) as demo:
        gr.Markdown(
            f"# KVBridge `v{__version__}`  -  *A KV Cache Management Layer for Scalable LLM Inference*\n"
            "LMCache-style chunk hashing + tiered KV storage in front of a **configurable inference "
            "engine**: local (**llama.cpp**, **Ollama**) or any cloud **OpenAI-compatible** API. "
            "Pick the engine below, or set it in `.env` (`KVB_ENGINE`).")

        with gr.Tab("1 - Engine & Cache"):
            try:
                start_engine = resolve_engine(s.engine).key
            except BackendError:
                start_engine = engine_choices()[0][1]
            vis0 = engine_visibility(start_engine)
            f: dict = {}                                 # form components by setting name
            f["engine"] = gr.Radio(engine_choices(), value=start_engine, label="Inference engine",
                                   info="llama.cpp = full KV save/restore - Ollama and cloud = shadow mode.")
            warn = gr.Markdown(CLOUD_WARNING, visible=vis0["warning"])
            with gr.Row():
                with gr.Column():
                    with gr.Group(visible=vis0["local_model"]) as grp_model:
                        f["ollama_model"] = gr.Textbox(
                            value=s.ollama_model, label="Local model (Ollama name)",
                            info="llama.cpp also loads this model's GGUF unless 'GGUF path' is set.")
                    f["ctx_size"] = gr.Slider(1024, 131072, value=s.ctx_size, step=1024,
                                              label="Context window (tokens)")
                    f["seed"] = gr.Number(value=s.seed, precision=0, label="Sampling seed")
                    f["request_timeout"] = gr.Number(value=s.request_timeout, label="Request timeout (s)")
                with gr.Column():
                    f["chunk_size"] = gr.Slider(16, 512, value=s.chunk_size, step=16, label="Chunk size (tokens)",
                                                info="LMCache default is 256. Smaller = finer reuse, more index entries.")
                    f["l1_mb"] = gr.Slider(0, 16384, value=s.l1_mb, step=64, label="L1 RAM tier (MB)")
                    f["l2_mb"] = gr.Slider(0, 65536, value=s.l2_mb, step=256, label="L2 disk tier (MB)")
                    f["write_through"] = gr.Checkbox(value=s.write_through,
                                                     label="Write-through to L2 (survives restarts)")
                    f["use_lmcache_hasher"] = gr.Checkbox(
                        value=s.use_lmcache_hasher,
                        label="Use the real lmcache TokenHasher (needs the lmcache package)")

            with gr.Group(visible=vis0["llamacpp"]) as grp_llama:
                gr.Markdown("**llama.cpp settings**  (`KVB_LLAMA_*`, `KVB_MANAGED_*`, `KVB_MODEL_PATH` ...)")
                with gr.Row():
                    f["llama_url"] = gr.Textbox(value=s.llama_url, label="Existing llama-server URL",
                                                info="Empty = start a managed llama-server.")
                    f["model_path"] = gr.Textbox(value=s.model_path, label="GGUF path (optional)")
                with gr.Row():
                    f["llama_server_bin"] = gr.Textbox(value=s.llama_server_bin, label="llama-server binary",
                                                       info="Empty = auto-detect.")
                    f["llama_search_paths"] = gr.Textbox(value=s.llama_search_paths,
                                                         label="Extra search paths (':'-separated)")
                with gr.Row():
                    f["managed_port"] = gr.Number(value=s.managed_port, precision=0, label="Managed server port")
                    f["reserved_ports"] = gr.Textbox(value=s.reserved_ports, label="Reserved ports (avoid)")
                    f["llama_start_timeout"] = gr.Number(value=s.llama_start_timeout, label="Start timeout (s)")
                with gr.Row():
                    f["n_gpu_layers"] = gr.Number(value=s.n_gpu_layers, precision=0,
                                                  label="GPU layers (-1 = default)")
                    f["cache_ram_mb"] = gr.Number(value=s.cache_ram_mb, precision=0,
                                                  label="llama.cpp --cache-ram (MB, 0 = off)")
                    f["extra_args"] = gr.Textbox(value=s.extra_args, label="Extra llama-server flags")

            with gr.Group(visible=vis0["ollama"]) as grp_ollama:
                gr.Markdown("**Ollama settings**  (`KVB_OLLAMA_*`)")
                with gr.Row():
                    f["ollama_url"] = gr.Textbox(value=s.ollama_url, label="Ollama URL")
                    f["ollama_keep_alive"] = gr.Textbox(value=s.ollama_keep_alive, label="keep_alive")
                    f["ollama_models_dir"] = gr.Textbox(value=s.ollama_models_dir, label="Models directory",
                                                        info="Empty = $OLLAMA_MODELS or ~/.ollama/models")

            with gr.Group(visible=vis0["cloud"]) as grp_cloud:
                gr.Markdown("**Cloud engine settings**  (`KVB_CLOUD_*`) - any OpenAI-compatible endpoint")
                with gr.Row():
                    f["cloud_provider"] = gr.Dropdown(
                        [""] + list(PROVIDER_PRESETS), value=s.cloud_provider, allow_custom_value=True,
                        label="Provider preset", info="Fills the base URL; you can overwrite it.")
                    f["cloud_base_url"] = gr.Textbox(value=s.cloud_base_url, label="Base URL",
                                                     placeholder="https://api.example.com/v1")
                    f["cloud_model"] = gr.Textbox(value=s.cloud_model, label="Model name")
                with gr.Row():
                    f["cloud_api_key"] = gr.Textbox(
                        value="", type="password", label="API key",
                        placeholder="stored" if s.resolved_cloud_key() else "paste a key",
                        info="Never displayed or logged. Leave blank to keep the current key.")
                    f["cloud_api_key_env"] = gr.Textbox(value=s.cloud_api_key_env, label="...or env var holding the key",
                                                        placeholder="OPENAI_API_KEY")
                f["cloud_extra_headers"] = gr.Textbox(value=s.cloud_extra_headers,
                                                      label="Extra HTTP headers (JSON object)",
                                                      placeholder='{"HTTP-Referer": "https://example.com"}')
                with gr.Row():
                    f["cloud_api_style"] = gr.Dropdown(list(CLOUD_API_STYLES), value=s.cloud_api_style,
                                                       label="API style")
                    f["cloud_max_tokens_field"] = gr.Textbox(value=s.cloud_max_tokens_field,
                                                             label="max-tokens field name",
                                                             info="max_tokens or max_completion_tokens")
                    f["cloud_token_budget"] = gr.Number(value=s.cloud_token_budget, precision=0,
                                                        label="Prompt-token budget (0 = off)")
                    f["cloud_kv_bytes_per_token"] = gr.Number(value=s.cloud_kv_bytes_per_token, precision=0,
                                                              label="KV bytes/token (accounting, 0 = unknown)")
                with gr.Row():
                    f["cloud_send_temperature"] = gr.Checkbox(value=s.cloud_send_temperature,
                                                              label="Send temperature")
                    f["cloud_stream_usage"] = gr.Checkbox(value=s.cloud_stream_usage,
                                                          label="Request token usage in the stream")
                    f["cloud_verify_tls"] = gr.Checkbox(value=s.cloud_verify_tls, label="Verify TLS certificates")

            form = [f[name] for name in FORM_FIELDS]
            f["engine"].change(on_engine_change, f["engine"], [grp_llama, grp_ollama, grp_cloud, grp_model, warn])
            f["cloud_provider"].change(on_provider_change, f["cloud_provider"], f["cloud_base_url"])

            with gr.Row():
                btn_connect = gr.Button("Connect / (re)start", variant="primary")
                btn_disc = gr.Button("Disconnect")
                btn_show = gr.Button("Show effective configuration")
            status = gr.Textbox(label="Status", lines=5, interactive=False)
            with gr.Row():
                save_key = gr.Checkbox(value=False, label="Also write the API key to .env (plain text, mode 0600)")
                btn_save = gr.Button("Save settings to .env")
            effective = gr.Code(label="Effective configuration (secrets masked)", language="json", interactive=False)
            btn_connect.click(connect, form, status)
            btn_disc.click(disconnect, None, status)
            btn_save.click(save_env, [save_key] + form, status)
            btn_show.click(show_effective, None, effective)

        with gr.Tab("2 - Dashboard"):
            kpi = gr.Markdown(elem_classes="kpi")
            with gr.Row():
                p_ttft = gr.Plot(label="TTFT")
                p_tokens = gr.Plot(label="Tokens")
            with gr.Row():
                p_hit = gr.Plot(label="Hit rate")
                p_mode = gr.Plot(label="Modes")
                p_tier = gr.Plot(label="Tiers")
            t_recent = gr.Dataframe(label="Most recent requests", interactive=False, wrap=True)
            with gr.Row():
                b_refresh = gr.Button("Refresh now")
                b_reset = gr.Button("Reset metrics")
                b_export = gr.Button("Export metrics JSON")
                exp_path = gr.Textbox(label="Exported to", interactive=False)
            dash_out = [kpi, p_ttft, p_tokens, p_hit, p_mode, p_tier, t_recent]
            b_refresh.click(refresh, None, dash_out)
            b_reset.click(reset_metrics, None, dash_out)
            b_export.click(export_metrics, None, exp_path)
            timer = gr.Timer(3.0)
            timer.tick(refresh, None, dash_out)
            demo.load(refresh, None, dash_out)

        with gr.Tab("3 - Playground"):
            gr.Markdown("Send a prompt through the pipeline. Add a **shared context** (taken from `input/` or the "
                        "built-in sample) and ask several questions: the second one should hit the cache.",
                        elem_classes="note")
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Question", lines=3, value="In one sentence, what does the KV cache store?")
                    with gr.Row():
                        pg_ctx = gr.Slider(0, 8000, value=1500, step=100, label="Shared context size (tokens, 0 = none)")
                        pg_mode = gr.Radio(["cold", "native", "lmcache"], value="lmcache", label="Mode",
                                           info="cold: no reuse - native: engine cache only - lmcache: KV layer")
                    with gr.Row():
                        pg_n = gr.Slider(8, 256, value=s.n_predict, step=8, label="Max new tokens")
                        pg_t = gr.Slider(0.0, 1.5, value=s.temperature, step=0.05, label="Temperature")
                    go_btn = gr.Button("Run", variant="primary")
                with gr.Column(scale=3):
                    answer = gr.Textbox(label="Answer", lines=6, interactive=False)
                    pg_info = gr.Markdown()
                    pg_map = gr.Plot(label="Chunk map")
            go_btn.click(play, [prompt, pg_ctx, pg_mode, pg_n, pg_t], [answer, pg_info, pg_map])

        with gr.Tab("4 - Scenarios"):
            gr.Markdown("Pre-built workloads that compare **cold** vs **LMCache** (and **native** where relevant). "
                        "Put your own `.txt`/`.md` files in `input/` to use them as context.", elem_classes="note")
            sc_key = gr.Radio([(v.title, k) for k, v in SCENARIOS.items()], value="doc_qa", label="Scenario")
            sc_blurb = gr.Markdown(SCENARIOS["doc_qa"].blurb)
            sc_key.change(lambda k: SCENARIOS[k].blurb, sc_key, sc_blurb)
            with gr.Row():
                sc_ctx = gr.Slider(500, 8000, value=2000, step=100, label="Context size (tokens)")
                sc_n = gr.Slider(2, 8, value=4, step=1, label="Questions / turns / users")
            sc_run = gr.Button("Run scenario", variant="primary")
            sc_sum = gr.Markdown()
            with gr.Row():
                sc_ttft = gr.Plot()
                sc_tok = gr.Plot()
            sc_map = gr.Plot()
            sc_tab = gr.Dataframe(interactive=False, wrap=True)
            sc_run.click(run_scenario, [sc_key, sc_ctx, sc_n], [sc_sum, sc_tab, sc_ttft, sc_tok, sc_map])

        with gr.Tab("5 - Cache Inspector"):
            insp_txt = gr.Markdown()
            insp_map = gr.Plot(label="Chunk map of the last request")
            insp_tab = gr.Dataframe(label="Stored KV snapshots", interactive=False)
            with gr.Row():
                i_ref = gr.Button("Refresh")
                i_l1 = gr.Button("Clear L1 (RAM)")
                i_all = gr.Button("Clear everything")
                i_rst = gr.Button("Simulate engine restart")
            rst_msg = gr.Textbox(label="Last action", interactive=False)
            with gr.Row():
                rs_l1 = gr.Slider(0, 16384, value=s.l1_mb, step=64, label="Resize L1 (MB)")
                rs_l2 = gr.Slider(0, 65536, value=s.l2_mb, step=256, label="Resize L2 (MB)")
                rs_btn = gr.Button("Apply (evicts LRU entries if needed)")
            insp_out = [insp_tab, insp_map, insp_txt]
            i_ref.click(inspector, None, insp_out)
            i_l1.click(clear_l1, None, insp_out)
            i_all.click(clear_all, None, insp_out)
            i_rst.click(restart_sim, None, [rst_msg] + insp_out)
            rs_btn.click(resize, [rs_l1, rs_l2], insp_out)

        with gr.Tab("About"):
            gr.Markdown((s.output_dir.parent / "Docs" / "Research.md").read_text(encoding="utf-8")
                        if (s.output_dir.parent / "Docs" / "Research.md").exists()
                        else "See `Docs/Research.md`.")
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    problems = settings.validate()
    if problems:
        raise SystemExit("Invalid configuration: " + "; ".join(problems))
    state = AppState(settings)
    demo = build_app(state)
    demo.queue(default_concurrency_limit=4)
    print(f"\n KVBridge GUI:  http://{settings.gui_host}:{settings.gui_port}\n", flush=True)
    # Turn SIGTERM (scripts/stop.sh) into a normal exit so ``finally`` stops the
    # managed llama-server instead of leaving it orphaned.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        demo.launch(server_name=settings.gui_host, server_port=settings.gui_port, show_error=True,
                    css=CSS, theme=gr.themes.Soft(primary_hue="blue"), inbrowser=False)
    finally:
        if state.engine:
            state.engine.stop()


if __name__ == "__main__":
    main()
