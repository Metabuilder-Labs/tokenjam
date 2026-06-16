"""
`tj report` — generate detailed HTML reports for analyzer findings.

Currently supports:
  - `tj report --trim [<agent_id>]`: open an HTML visualization of the
    Trim analyzer's findings, with high-significance tokens in
    bold and low-significance regions dimmed.

The HTML is written to a local file and opened in the user's default
browser via `webbrowser.open()`. Reports live under
`~/.cache/tokenjam/reports/` and are overwritten on each run.
"""
from __future__ import annotations

import html
import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.markup import escape as _rich_escape

from tokenjam.utils.formatting import console


def _report_dir() -> Path:
    base = Path(os.environ.get(
        "TOKENJAM_REPORT_DIR",
        os.path.expanduser("~/.cache/tokenjam/reports"),
    ))
    base.mkdir(parents=True, exist_ok=True)
    return base


@click.command("report")
@click.option("--trim", "trim_agent", default=None, flag_value="",
              is_flag=False, help="Generate the Trim HTML report. "
                                   "Optional agent_id scopes the report.")
@click.option("--since", default="30d", help="Window for the report (default 30d).")
@click.option("--no-open", "no_open", is_flag=True, default=False,
              help="Write the HTML file without opening it in a browser.")
@click.pass_context
def cmd_report(ctx: click.Context, trim_agent: str | None, since: str,
               no_open: bool) -> None:
    """Generate detailed HTML reports for analyzer findings."""
    if trim_agent is None:
        raise click.UsageError(
            "Specify a report type. Currently supported: --trim [<agent_id>]"
        )
    _render_trim_report(ctx, trim_agent or None, since, no_open)


def _render_trim_report(
    ctx: click.Context,
    agent_id: str | None,
    since: str,
    no_open: bool,
) -> None:
    """Run the Trim analyzer and write its findings as HTML."""
    from tokenjam.core.optimize import build_report
    from tokenjam.utils.time_parse import parse_since, utcnow

    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    if db is None or config is None:
        raise click.ClickException("tj report requires a database connection.")

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    report = build_report(
        db=db, config=config,
        since=since_dt, until=utcnow(),
        agent_id=agent_id,
        findings=["trim"],
    )
    finding = report.findings.get("trim")
    if finding is None:
        raise click.ClickException("Trim analyzer didn't produce a finding.")

    if not finding.enabled:
        # Show the hint inline rather than producing an empty HTML file.
        console.print(f"[yellow]Trim analyzer not ready:[/yellow]\n{_rich_escape(finding.hint)}")
        return

    if not finding.per_prompt:
        console.print(
            "[dim]No bloat regions found in the window. Either prompts are "
            "tight already or the captured sample is too small.[/dim]"
        )
        return

    out_path = _report_dir() / f"trim-{datetime.now(tz=timezone.utc):%Y%m%d-%H%M%S}.html"
    out_path.write_text(_render_html(finding, agent_id, since))
    console.print(f"[green]✓[/green] Trim report written to [bold]{out_path}[/bold]")
    if not no_open:
        webbrowser.open(f"file://{out_path.resolve()}")


def _render_html(finding, agent_scope: str | None, since: str) -> str:
    """
    Render the Trim finding as a standalone HTML page.

    Per-prompt detail: the prompt text is split into segments — high-sig
    (bold) vs flagged bloat regions (dimmed + struck through). Each region
    is annotated with its average significance score.
    """
    scope_label = f"agent={html.escape(agent_scope)}, " if agent_scope else ""
    title = f"TokenJam — Trim report ({scope_label}{html.escape(since)})"

    sections: list[str] = []
    for i, p in enumerate(finding.per_prompt, start=1):
        # The original full text isn't stored on the finding;
        # we reconstruct visible portions from sample_chars and regions.
        # The HTML preview uses the prompt's first 120 chars (sample) plus
        # region samples — full text rendering happens when the user opens
        # the source span in the web UI. For v1 this is sufficient to
        # identify bloat patterns.
        preview = html.escape(p.sample_chars)
        region_blocks = []
        for r in p.regions:
            region_blocks.append(
                f"<div class='region'>"
                f"<span class='meta'>chars {r.start_char}–{r.end_char} "
                f"({r.char_length} chars, avg score {r.avg_score:.2f})</span>"
                f"<div class='sample'>{html.escape(r.sample_chars)}…</div>"
                f"</div>"
            )

        sections.append(
            f"<section>"
            f"<h2>Prompt #{i} <span class='agent'>{html.escape(p.agent_id)}</span></h2>"
            f"<p class='preview'><strong>Preview:</strong> {preview}…</p>"
            f"<p class='stats'>"
            f"<b>{p.prompt_chars}</b> chars · "
            f"<b>{p.significant_chars}</b> significant · "
            f"<b class='bloat'>{p.bloat_chars}</b> in flagged regions · "
            f"~<b>{p.estimated_token_reduction}</b> tokens potentially trimmable"
            f"</p>"
            f"<h3>Bloat regions ({len(p.regions)})</h3>"
            f"{''.join(region_blocks) if region_blocks else '<p>None flagged.</p>'}"
            f"</section>"
        )

    pct = (
        finding.total_bloat_chars / finding.total_chars * 100.0
        if finding.total_chars > 0 else 0.0
    )

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
    h2 .agent {{ font-weight: normal; color: #777; font-size: 0.7em; margin-left: 1em; }}
    .summary {{ background: #f4f4f8; padding: 1em; border-radius: 6px; }}
    .summary b {{ color: #c25; }}
    .preview {{ background: #fafafa; padding: 0.6em 1em; border-left: 3px solid #ccc;
                font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.9em; }}
    .stats {{ color: #555; }}
    .stats b {{ color: #222; }}
    .stats b.bloat {{ color: #c25; }}
    .region {{ background: #fff3f3; padding: 0.6em 1em; margin: 0.5em 0;
               border-left: 3px solid #c25; border-radius: 0 4px 4px 0; }}
    .region .meta {{ color: #777; font-size: 0.85em; }}
    .region .sample {{ font-family: ui-monospace, "SF Mono", Menlo, monospace;
                        font-size: 0.9em; color: #888; text-decoration: line-through;
                        margin-top: 0.3em; }}
    .caveat {{ background: #fff8d6; padding: 0.8em 1em; border-left: 3px solid #cb3;
               margin-top: 1em; border-radius: 0 4px 4px 0; }}
  </style>
</head>
<body>
  <h1>TokenJam — Trim report</h1>
  <div class='summary'>
    Scope: <b>{scope_label}{html.escape(since)}</b><br>
    Scored <b>{finding.prompts_scored}</b> prompts; skipped <b>{finding.prompts_skipped}</b>.<br>
    Total chars: <b>{finding.total_chars}</b> · Flagged as bloat: <b class='bloat'>{finding.total_bloat_chars}</b>
    (<b>{pct:.1f}%</b>).
  </div>
  <div class='caveat'>
    <strong>Caveat.</strong> Trim flags regions LLMLingua-2 predicts the model gives little weight to.
    Subtle context the model does rely on can score low — review each region before editing your
    prompt template, and re-run after the change to confirm the bloat is gone without behavior regression.
  </div>
  {''.join(sections)}
</body>
</html>
"""
