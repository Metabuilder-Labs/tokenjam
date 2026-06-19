"""
HTML + Markdown report bundle for the Reuse analyzer (#116).

`tj report --reuse` (and the `tj optimize reuse --export-templates` shortcut)
turn a `ReuseFinding` into:

- one static HTML page (`reuse-<timestamp>.html`) summarizing every cluster,
  with the skeleton rendered and variable slots highlighted, and
- one Markdown sidecar per cluster (`reuse-<cluster_id>.md`) that's directly
  copy-paste-usable as a slash command / saved prompt.

The Markdown filename keys off the deterministic `cluster_id` (no timestamp) so
re-running over the same data overwrites the same files instead of piling up
duplicates (#116 AC6). The HTML carries a timestamp, mirroring `tj report
--trim`.

Skeleton rendering needs the planning call's *completion* text, which only
lives in the spans table when `[capture] completions = true`. Without it the
cluster still renders (numbers + signature) but the skeleton is replaced by a
one-line hint and no sidecar is written (#116 AC7). No network calls, no JS.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tokenjam.core.export.reuse_skeleton import is_weak_match, render_skeleton
from tokenjam.core.optimize.analyzers.plan_reuse import (
    _SpanRow,
    _identify_planning_call,
)
from tokenjam.core.optimize.types import ReuseCluster, ReuseFinding
from tokenjam.otel.semconv import GenAIAttributes

REUSE_REPORT_CAVEAT = (
    "Structural analysis only — review the templates before reusing them."
)

_SLOT_RE = re.compile(r"\{\{(slot_\d+)\}\}")


@dataclass
class ClusterRender:
    """Everything the HTML/Markdown needs for one cluster."""
    cluster:            ReuseCluster
    skeleton_available: bool
    skeleton:           str = ""
    slot_map:           dict[str, list[str]] = field(default_factory=dict)
    weak:               bool = False
    md_filename:        str | None = None
    md_path:            Path | None = None


# --------------------------------------------------------------------------
# Planning-text fetch (re-derives the planner per session, same rule as the
# analyzer, then pulls its completion content).
# --------------------------------------------------------------------------

def _completion_text(row: _SpanRow | None) -> str | None:
    if row is None:
        return None
    attrs = row.attributes
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except Exception:
            return None
    if not isinstance(attrs, dict):
        return None
    text = attrs.get(GenAIAttributes.COMPLETION_CONTENT)
    if text is None or text == "":
        return None
    if not isinstance(text, str):
        text = json.dumps(text, sort_keys=True)
    return text


def _fetch_planning_texts(conn, session_ids: list[str]) -> dict[str, str | None]:
    """Return {session_id: planning completion text or None} for the given ids."""
    if not session_ids:
        return {}
    # Parameterized SQL (CLAUDE.md Rule 7): the f-string interpolates only the
    # positional placeholder *indices* ($1, $2, …); every value is bound via
    # the params list below. Same placeholders-only pattern as
    # plan_reuse.run and runner.summarize_window — no user data in the string.
    placeholders = ",".join(f"${i + 1}" for i in range(len(session_ids)))
    rows = conn.execute(
        f"SELECT session_id, start_time, model, tool_name, attributes "
        f"FROM spans WHERE session_id IN ({placeholders}) "
        f"ORDER BY session_id, start_time",
        list(session_ids),
    ).fetchall()

    per_session: dict[str, list[_SpanRow]] = {}
    for sid, start_time, model, tool_name, attrs in rows:
        per_session.setdefault(str(sid), []).append(_SpanRow(
            session_id=str(sid), start_time=start_time, model=model,
            tool_name=tool_name, input_tokens=None, output_tokens=None,
            cache_tokens=None, cache_write_tokens=None, cost_usd=None,
            attributes=attrs,
        ))
    return {
        sid: _completion_text(_identify_planning_call(srows))
        for sid, srows in per_session.items()
    }


# --------------------------------------------------------------------------
# Markdown sidecar
# --------------------------------------------------------------------------

def _render_markdown(
    cluster: ReuseCluster,
    skeleton: str,
    slot_map: dict[str, list[str]],
    *,
    version: str,
    generated_at_iso: str,
) -> str:
    sig = ", ".join(cluster.tool_signature)
    slot_lines = "\n".join(
        f"- `{{{{{name}}}}}`: examples = {json.dumps(values)}"
        for name, values in slot_map.items()
    ) or "- (no variable slots — the plans were identical)"

    return f"""---
cluster_id: {cluster.cluster_id}
tool_signature: [{sig}]
repetitions: {cluster.repetitions}
generated_at: {generated_at_iso}
tokenjam_version: {version}
---

# Reuse skeleton

{skeleton}

## Variable slots

{slot_lines}

## How to use this

Review the skeleton. If it captures a workflow you can deterministically
reproduce, copy it into ~/.claude/commands/<name>.md as a Claude Code
slash command, or save it as a prompt template in your tool of choice.
The variable slots show you what changes across runs.

This skeleton was extracted by `tj report --reuse`. It's a structural
pattern match, not a guarantee that the underlying plans were
interchangeable. Review before reusing.
"""


# --------------------------------------------------------------------------
# Render preparation (shared by report + --export-templates)
# --------------------------------------------------------------------------

def gather_planning_texts(conn, finding: ReuseFinding) -> dict[str, str | None]:
    """{session_id: planning completion text or None} across ALL clusters.

    The dedicated `/api/v1/reuse/clusters` endpoint (#154) calls this so the
    skeleton-rendering text travels with the finding over HTTP — letting
    `tj report --reuse` render without a direct DB connection when the daemon
    holds the write lock. One batched query for every example session in the
    finding.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for c in finding.clusters:
        for sid in c.example_session_ids:
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return _fetch_planning_texts(conn, ids)


def prepare_renders(
    finding: ReuseFinding,
    *,
    config,
    out_dir: Path,
    version: str,
    generated_at_iso: str,
    write_md: bool = True,
    conn=None,
    planning_texts: dict[str, str | None] | None = None,
) -> list[ClusterRender]:
    """
    Build per-cluster render data, writing Markdown sidecars as a side effect
    (when capture.completions allow). Returns one ClusterRender per cluster in
    the finding's existing (recoverable-ranked) order.

    Planning text comes from one of two sources:
      - `planning_texts` (HTTP path, #154): a pre-fetched {session_id: text}
        map from `/api/v1/reuse/clusters`. Used as-is — the daemon already
        applied its own capture gating, so a skeleton renders iff text is
        present. No DB connection needed.
      - `conn` (local path): fetched per cluster, gated on local
        `capture.completions` (no text is in the DB unless it was captured).
    """
    capture = getattr(config, "capture", None)
    completions_on = bool(capture and getattr(capture, "completions", False))

    renders: list[ClusterRender] = []
    for c in finding.clusters:
        if planning_texts is not None:
            texts = {sid: planning_texts.get(sid) for sid in c.example_session_ids}
        elif completions_on:
            texts = _fetch_planning_texts(conn, list(c.example_session_ids))
        else:
            renders.append(ClusterRender(cluster=c, skeleton_available=False))
            continue

        skel_text = texts.get(c.skeleton_session_id)
        if not skel_text:
            # Fall back to any example that does have text.
            skel_text = next((t for t in texts.values() if t), None)
        if not skel_text:
            renders.append(ClusterRender(cluster=c, skeleton_available=False))
            continue

        example_texts = [
            t for sid, t in texts.items()
            if t and sid != c.skeleton_session_id
        ]
        skeleton, slot_map = render_skeleton(skel_text, example_texts)
        weak = is_weak_match(slot_map)

        md_filename = f"reuse-{c.cluster_id}.md"
        md_path: Path | None = None
        if write_md:
            md_path = out_dir / md_filename
            md_path.write_text(_render_markdown(
                c, skeleton, slot_map,
                version=version, generated_at_iso=generated_at_iso,
            ))

        renders.append(ClusterRender(
            cluster=c, skeleton_available=True, skeleton=skeleton,
            slot_map=slot_map, weak=weak, md_filename=md_filename,
            md_path=md_path,
        ))
    return renders


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _recoverable_pair(c: ReuseCluster, pricing_mode: str) -> tuple[str, str]:
    """(cache-reuse, script-replacement) display strings, framed per mode."""
    if pricing_mode in ("subscription", "local"):
        return (
            f"~{c.cache_reuse_recoverable_tokens:,} tokens",
            f"~{c.script_replacement_recoverable_tokens:,} tokens",
        )
    return (
        f"${c.cache_reuse_recoverable_usd:,.2f}",
        f"${c.script_replacement_recoverable_usd:,.2f}",
    )


def _highlight_slots(skeleton: str) -> str:
    """Escape skeleton text, then wrap {{slot_N}} / {{…}} markers in spans."""
    escaped = html.escape(skeleton)
    escaped = _SLOT_RE.sub(
        r"<span class='slot'>{{\1}}</span>", escaped
    )
    return escaped.replace("{{…}}", "<span class='slot overflow'>{{…}}</span>")


def render_html(
    finding: ReuseFinding,
    renders: list[ClusterRender],
    *,
    agent_scope: str | None,
    since: str,
    pricing_mode: str,
) -> str:
    scope_label = f"agent={html.escape(agent_scope)}, " if agent_scope else ""
    cache_total, script_total = (
        (f"~{finding.estimated_recoverable_tokens or 0:,} tokens", "")
        if pricing_mode in ("subscription", "local")
        else (f"${finding.estimated_recoverable_usd or 0.0:,.2f}", "")
    )

    mode_hint = ""
    if finding.capture_mode == "tool_sequence_only":
        mode_hint = (
            "<div class='hint'>Clustered on tool sequences only. Set "
            "<code>[capture] prompts = true</code> for narrower, more accurate "
            "clusters.</div>"
        )

    sections: list[str] = []
    for idx, r in enumerate(renders, start=1):
        c = r.cluster
        cache_str, script_str = _recoverable_pair(c, pricing_mode)
        sig_items = "".join(
            f"<li>{html.escape(t)}</li>" for t in c.tool_signature
        ) or "<li><em>(no tools after the plan)</em></li>"

        if r.skeleton_available:
            weak_flag = (
                "<span class='weak'>weak match — many divergences</span>"
                if r.weak else ""
            )
            slot_rows = "".join(
                f"<tr><td><code>{{{{{html.escape(name)}}}}}</code></td>"
                f"<td>{html.escape(', '.join(values))}</td></tr>"
                for name, values in r.slot_map.items()
            ) or "<tr><td colspan='2'><em>No variable slots.</em></td></tr>"
            skeleton_block = (
                f"<h4>Skeleton {weak_flag}</h4>"
                f"<pre class='skeleton'>{_highlight_slots(r.skeleton)}</pre>"
                f"<details><summary>Variable slots ({len(r.slot_map)})</summary>"
                f"<table class='slots'>{slot_rows}</table></details>"
            )
            if r.md_filename:
                skeleton_block += (
                    f"<p class='sidecar'>Copy-paste template: "
                    f"<a href='{html.escape(r.md_filename)}'>"
                    f"{html.escape(r.md_filename)}</a></p>"
                )
        else:
            skeleton_block = (
                "<p class='degraded'>Enable <code>[capture] completions = "
                "true</code> to render the skeleton text for this cluster.</p>"
            )

        examples = "".join(
            f"<li><code>tj traces</code> · session "
            f"<code>{html.escape(sid)}</code></li>"
            for sid in c.example_session_ids
        )

        sections.append(
            f"<section>"
            f"<h2>#{idx} · {c.repetitions}× repeated planning</h2>"
            f"<p class='recoverable'>Recoverable by reusing "
            f"<b>{html.escape(cache_str)}</b> · by scripting "
            f"<b>{html.escape(script_str)}</b> "
            f"<span class='avg'>(avg planning {c.avg_planning_tokens:,} tokens)"
            f"</span></p>"
            f"<h4>Tool sequence</h4><ol class='sig'>{sig_items}</ol>"
            f"{skeleton_block}"
            f"<h4>Example sessions</h4><ul class='examples'>{examples}</ul>"
            f"<p class='caveat-inline'>{html.escape(REUSE_REPORT_CAVEAT)}</p>"
            f"</section>"
        )

    title = f"TokenJam — Reuse report ({scope_label}{html.escape(since)})"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
           max-width: 920px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.4em; }}
    h2 {{ margin-top: 2em; color: #444; }}
    .summary {{ background: #f4f4f8; padding: 1em; border-radius: 6px; }}
    .summary b {{ color: #258; }}
    .hint, .degraded {{ background: #eef6ff; padding: 0.6em 1em; border-left: 3px solid #69c;
            border-radius: 0 4px 4px 0; margin: 0.6em 0; font-size: 0.92em; }}
    .recoverable {{ font-size: 1.05em; }}
    .recoverable b {{ color: #182; }}
    .avg {{ color: #888; font-weight: normal; font-size: 0.85em; }}
    ol.sig li, ul.examples li {{ font-family: ui-monospace, "SF Mono", Menlo, monospace;
            font-size: 0.9em; }}
    pre.skeleton {{ background: #fafafa; padding: 1em; border-left: 3px solid #ccc;
            border-radius: 0 4px 4px 0; white-space: pre-wrap; word-break: break-word;
            font-size: 0.9em; }}
    .slot {{ background: #fff3cd; color: #8a6d00; padding: 0 2px; border-radius: 3px; }}
    .slot.overflow {{ background: #f0d6d6; color: #a33; }}
    .weak {{ color: #a33; font-size: 0.75em; font-weight: normal; margin-left: 0.6em; }}
    table.slots {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.88em; }}
    table.slots td {{ border: 1px solid #ddd; padding: 0.3em 0.6em; }}
    .sidecar {{ font-size: 0.9em; }}
    .caveat-inline {{ background: #fff8d6; padding: 0.6em 1em; border-left: 3px solid #cb3;
            border-radius: 0 4px 4px 0; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>TokenJam — Reuse report</h1>
  <div class='summary'>
    Scope: <b>{scope_label}{html.escape(since)}</b><br>
    <b>{len(renders)}</b> cluster(s) of repeated planning ·
    Recoverable by reusing: <b>{html.escape(cache_total)}</b><br>
    <span style='font-size:0.9em;color:#555'>{html.escape(finding.estimate_basis)}</span>
  </div>
  {mode_hint}
  {''.join(sections)}
</body>
</html>
"""


# --------------------------------------------------------------------------
# Orchestrators used by the CLI
# --------------------------------------------------------------------------

def write_reuse_report(
    finding: ReuseFinding,
    *,
    config,
    out_dir: Path,
    agent_scope: str | None,
    since: str,
    pricing_mode: str,
    version: str,
    generated_at_iso: str,
    html_filename: str,
    conn=None,
    planning_texts: dict[str, str | None] | None = None,
) -> tuple[Path, list[Path]]:
    """Write the HTML page + Markdown sidecars. Returns (html_path, md_paths).

    Pass either `conn` (local DB) or `planning_texts` (pre-fetched over HTTP,
    #154) — see `prepare_renders`.
    """
    renders = prepare_renders(
        finding, conn=conn, config=config, out_dir=out_dir,
        version=version, generated_at_iso=generated_at_iso, write_md=True,
        planning_texts=planning_texts,
    )
    html_path = out_dir / html_filename
    html_path.write_text(render_html(
        finding, renders, agent_scope=agent_scope,
        since=since, pricing_mode=pricing_mode,
    ))
    md_paths = [r.md_path for r in renders if r.md_path]
    return html_path, md_paths


def export_templates(
    finding: ReuseFinding,
    *,
    conn,
    config,
    out_dir: Path,
    version: str,
    generated_at_iso: str,
) -> list[Path]:
    """Write only the Markdown sidecars (no HTML). Returns the written paths."""
    renders = prepare_renders(
        finding, conn=conn, config=config, out_dir=out_dir,
        version=version, generated_at_iso=generated_at_iso, write_md=True,
    )
    return [r.md_path for r in renders if r.md_path]
