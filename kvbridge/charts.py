"""Plotly figure builders and table helpers for the dashboard.

Kept free of any Gradio import so they can be unit-tested headlessly.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go

from .metrics import RequestRecord

COLORS = {"cold": "#da1e28", "native": "#f1c21b", "lmcache": "#0f62fe",
          "L1": "#0f62fe", "L2": "#8a3ffc", "miss": "#c6c6c6", "free": "#e0e0e0"}
FONT = dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", size=13)


def fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _layout(fig: go.Figure, title: str, height: int = 330) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.01), height=height, font=FONT,
                      margin=dict(l=50, r=20, t=50, b=45), plot_bgcolor="white",
                      paper_bgcolor="white", legend=dict(orientation="h", y=-0.22))
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig


def _empty(title: str, msg: str = "No data yet - run a scenario or send a prompt") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#6f6f6f", size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title)


def ttft_figure(records: Sequence[RequestRecord]) -> go.Figure:
    """Time-to-first-token for each request, coloured by execution mode."""
    recs = [r for r in records if not r.error]
    if not recs:
        return _empty("Time to first token (ms)")
    fig = go.Figure()
    for mode in ("cold", "native", "lmcache"):
        pts = [(i + 1, r) for i, r in enumerate(recs) if r.mode == mode]
        if pts:
            fig.add_trace(go.Bar(
                name=mode, x=[i for i, _ in pts], y=[r.ttft_ms for _, r in pts],
                marker_color=COLORS[mode],
                customdata=[[r.label, r.computed_tokens, r.prompt_tokens, r.tier] for _, r in pts],
                hovertemplate="#%{x} %{customdata[0]}<br>TTFT %{y:.0f} ms<br>"
                              "prefilled %{customdata[1]}/%{customdata[2]} tok<br>tier %{customdata[3]}<extra></extra>"))
    fig.update_layout(barmode="group", xaxis_title="request #", yaxis_title="ms")
    return _layout(fig, "Time to first token (lower is better)")


def tokens_figure(records: Sequence[RequestRecord]) -> go.Figure:
    """Stacked prefilled vs reused tokens per request."""
    recs = [r for r in records if not r.error]
    if not recs:
        return _empty("Prompt tokens: computed vs reused")
    x = list(range(1, len(recs) + 1))
    reused = [max(r.server_cached_tokens, 0) for r in recs]
    computed = [r.computed_tokens for r in recs]
    fig = go.Figure([
        go.Bar(name="reused from cache", x=x, y=reused, marker_color=COLORS["lmcache"]),
        go.Bar(name="computed (prefill)", x=x, y=computed, marker_color=COLORS["cold"]),
    ])
    fig.update_layout(barmode="stack", xaxis_title="request #", yaxis_title="tokens")
    return _layout(fig, "Prompt tokens: reused vs computed")


def hit_rate_figure(records: Sequence[RequestRecord]) -> go.Figure:
    """Rolling chunk hit-rate across LMCache-mode requests."""
    recs = [r for r in records if r.mode == "lmcache" and not r.error and r.chunks_total]
    if not recs:
        return _empty("Chunk hit rate (LMCache requests)")
    per, cum, h, t = [], [], 0, 0
    for r in recs:
        h += r.chunks_hit
        t += r.chunks_total
        per.append(100.0 * r.chunks_hit / r.chunks_total)
        cum.append(100.0 * h / t)
    x = list(range(1, len(recs) + 1))
    fig = go.Figure([
        go.Bar(name="per request", x=x, y=per, marker_color="#78a9ff"),
        go.Scatter(name="cumulative", x=x, y=cum, mode="lines+markers", line=dict(color="#0f62fe", width=3)),
    ])
    fig.update_layout(yaxis=dict(range=[0, 105], title="% of chunks"), xaxis_title="LMCache request #")
    return _layout(fig, "Chunk hit rate")


def tier_figure(store_stats: dict) -> go.Figure:
    """L1/L2 capacity usage gauges (two stacked horizontal bars)."""
    fig = go.Figure()
    for name, used, cap in (("L1  RAM", store_stats.get("l1_used", 0), store_stats.get("l1_cap", 0)),
                            ("L2  Disk", store_stats.get("l2_used", 0), store_stats.get("l2_cap", 0))):
        free = max(cap - used, 0)
        fig.add_trace(go.Bar(y=[name], x=[used], orientation="h", name="used", showlegend=name.startswith("L1"),
                             marker_color=COLORS["lmcache"], text=[fmt_bytes(used)], textposition="inside"))
        fig.add_trace(go.Bar(y=[name], x=[free], orientation="h", name="free", showlegend=name.startswith("L1"),
                             marker_color=COLORS["free"], text=[f"{fmt_bytes(free)} free"], textposition="inside"))
    fig.update_layout(barmode="stack", xaxis_title="bytes")
    return _layout(fig, "Cache tier usage", height=240)


def chunk_map_figure(status: Sequence[str], label: str = "") -> go.Figure:
    """Grid of prompt chunks coloured by where they were found."""
    if not status:
        return _empty("Chunk map", "Prompt shorter than one chunk, or nothing run yet")
    cols = min(24, max(len(status), 1))
    rows = (len(status) + cols - 1) // cols
    z, text = [], []
    code = {"miss": 0, "L2": 1, "L1": 2}
    for r in range(rows):
        row, trow = [], []
        for c in range(cols):
            i = r * cols + c
            if i < len(status):
                row.append(code.get(status[i], 0))
                trow.append(f"chunk {i}: {status[i]}")
            else:
                row.append(None)
                trow.append("")
        z.append(row)
        text.append(trow)
    fig = go.Figure(go.Heatmap(
        z=z, text=text, hoverinfo="text", showscale=False, xgap=3, ygap=3,
        colorscale=[[0, COLORS["miss"]], [0.5, COLORS["L2"]], [1, COLORS["L1"]]], zmin=0, zmax=2))
    fig.update_yaxes(autorange="reversed", visible=False)
    fig.update_xaxes(visible=False)
    return _layout(fig, f"Chunk map {label}  (blue=L1 RAM, purple=L2 disk, grey=miss)",
                   height=max(220, 150 + 40 * rows))


def mode_compare_figure(snapshot: dict) -> go.Figure:
    """Average TTFT per mode."""
    bm = snapshot.get("by_mode", {})
    modes = [m for m in ("cold", "native", "lmcache") if bm.get(m, {}).get("n")]
    if not modes:
        return _empty("Average TTFT by mode")
    fig = go.Figure(go.Bar(x=modes, y=[bm[m]["avg_ttft_ms"] for m in modes],
                           marker_color=[COLORS[m] for m in modes],
                           text=[f"{bm[m]['avg_ttft_ms']:.0f} ms" for m in modes], textposition="outside"))
    fig.update_layout(yaxis_title="ms")
    return _layout(fig, "Average TTFT by mode")


def records_table(records: Iterable[RequestRecord]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(records, 1):
        rows.append({
            "#": i, "label": r.label, "mode": r.mode, "tier": r.tier,
            "prompt tok": r.prompt_tokens, "prefilled": r.computed_tokens,
            "reused": r.server_cached_tokens,
            "chunks hit": f"{r.chunks_hit}/{r.chunks_total}" if r.chunks_total else "-",
            "TTFT ms": round(r.ttft_ms, 1), "restore ms": round(r.restore_ms, 1),
            "save ms": round(r.save_ms, 1), "decode tok/s": round(r.decode_tps, 1),
            "error": r.error,
        })
    cols = ["#", "label", "mode", "tier", "prompt tok", "prefilled", "reused", "chunks hit",
            "TTFT ms", "restore ms", "save ms", "decode tok/s", "error"]
    return pd.DataFrame(rows, columns=cols)


def entries_table(entries) -> pd.DataFrame:
    import time

    rows = [{
        "id": e.id, "tier": e.tier + ("+L2" if e.in_l1 and e.in_l2 else ""),
        "tokens": e.n_tokens, "chunks": len(e.hashes), "size": fmt_bytes(e.size),
        "hits": e.hits, "age s": int(time.time() - e.created),
        "idle s": int(time.time() - e.last_used), "note": e.note,
    } for e in entries]
    return pd.DataFrame(rows, columns=["id", "tier", "tokens", "chunks", "size", "hits", "age s", "idle s", "note"])


def kpi_markdown(snap: dict) -> str:
    """Headline KPIs as Markdown (rendered in a Gradio Markdown block)."""
    m, s = snap["metrics"], snap["store"]
    sp = m.get("ttft_speedup_vs_cold", 0)
    return (
        f"| Chunk hit rate | Token reuse | TTFT speed-up vs cold | Requests | Cached snapshots | Cached tokens |\n"
        f"|:--:|:--:|:--:|:--:|:--:|:--:|\n"
        f"| **{m['chunk_hit_rate'] * 100:.0f}%** | **{m['token_reuse_ratio'] * 100:.0f}%** | "
        f"**{sp:.1f}x**" + ("" if sp else " (n/a)") + f" | **{int(m['requests'])}** | **{s['entries']}** | "
        f"**{s['tokens_cached']:,}** |\n"
    )
