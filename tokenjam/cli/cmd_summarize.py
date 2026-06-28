"""`tj summarize` — structure-aware prompt summarization (advisory in v1).

`tj summarize list` finds prompt files worth summarizing and estimates the
per-call token saving. Bare = the known-location catalog (globals + this dir).
A scope-widening input — a PATH, `--repo`, `--recursive`, or `--ext` — opens it
to all `*.md`; the scanned location is shown first, the catalog globals after a
divider. `prep` wraps a prompt's structure and emits it for you to rewrite; `check`
verifies the rewrite preserved every structure block (a hard gate) and stages it for
review. Reads, reports, and stages — never rewrites a file (apply lands in a later PR).
See DEC-020/021/024.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.markup import escape

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize.candidates import list_candidates
from tokenjam.core.summarize.estimate import DEFAULT_TARGET_RATIO
from tokenjam.core.summarize.session import CheckVerdict, SummarizeRefused, check, prepare, results_dir
from tokenjam.utils.formatting import console, format_tokens

# Honesty discipline (CLAUDE.md Rule 14): every candidate is a suggestion to
# review, never an assertion the rewrite is safe — and the saving is estimated.
CANDIDATE_NOTE = (
    "Candidates only — review the summary before adopting. The figure is the "
    "estimated per-call token reduction, which amortizes across every reuse of "
    "the (cached) prompt."
)


def _print_verdict(config: TjConfig, verdict: CheckVerdict) -> None:
    """Human-readable check verdict: the ✓/✗ line and, when staged, where it landed.
    (must-keep word movement is recorded on the staged result for later metrics — collected,
    never surfaced to the user here.)"""
    if verdict.structure_ok:
        console.print(f"[green]✓[/green] {escape(verdict.path)} — structure preserved, "
                      f"~{format_tokens(verdict.est_tokens_saved)} tok_out/call "
                      f"({verdict.words_before}→{verdict.words_after} words)")
    else:
        console.print(f"[red]✗[/red] {escape(verdict.path)} — {escape(verdict.reason)} (not staged)")
    if verdict.staged:
        console.print(f"[dim]staged for review in {escape(str(results_dir(config)))}[/dim]")


@click.group("summarize", invoke_without_command=False)
def cmd_summarize() -> None:
    """Structure-aware prompt summarization (advisory preview)."""


@cmd_summarize.command("list")
@click.argument("path", required=False, default=None)
@click.option("-r", "--recursive", is_flag=True,
              help="Walk the repo subtree (or PATH) — opens to all .md.")
@click.option("--repo", "repo", is_flag=True,
              help="Check the git-repo root (no walk) — opens to all .md.")
@click.option("--no-global", "no_global", is_flag=True,
              help="Skip the global/system locations (project only).")
@click.option("--ext", "ext", default=None,
              help="Also scan these comma-separated extensions, e.g. txt,rst "
                   "(opens beyond the catalog).")
@click.option("--json", "output_json_flag", is_flag=True,
              help="Emit machine-readable JSON.")
@click.option("--min-prose", "min_prose", default=None, type=int,
              help="Minimum prose words to flag a file (default 100).")
@click.pass_context
def cmd_summarize_list(
    ctx: click.Context, path: str | None, recursive: bool, repo: bool,
    no_global: bool, ext: str | None, output_json_flag: bool, min_prose: int | None,
) -> None:
    """List prompt files worth summarizing (bare = catalog; a PATH/--repo/--recursive/--ext opens to .md)."""
    config: TjConfig = ctx.obj["config"]
    output_json: bool = output_json_flag or ctx.obj.get("output_json", False)

    if repo and recursive:
        raise click.UsageError("--repo and --recursive are mutually exclusive.")
    if repo and path is not None:
        raise click.UsageError("--repo cannot be combined with an explicit PATH.")

    extra_exts = tuple(e for e in (ext.split(",") if ext else []) if e.strip())
    kwargs: dict = {}
    if min_prose is not None:
        kwargs["min_prose_words"] = min_prose
    result = list_candidates(
        path, config=config, recursive=recursive, repo=repo,
        include_global=not no_global, extra_exts=extra_exts, **kwargs,
    )

    if output_json:
        payload = result.to_dict()
        payload["note"] = result.note or CANDIDATE_NOTE
        click.echo(json.dumps(payload, indent=2))
        return

    # Transparency — what was scanned (DEC-020).
    scanned: list[str] = []
    if result.root:
        scanned.append(f"{escape(result.root)}{' (recursive)' if result.recursive else ''}")
    if result.globals_checked:
        scanned.append(f"{result.globals_checked} global location(s)")
    if scanned:
        console.print(f"[dim]Scanned: {' + '.join(scanned)}[/dim]")
    if result.walk_capped:
        console.print("[yellow]Walk hit the file cap — results truncated; "
                      "narrow with a PATH.[/yellow]")
    if result.note:
        console.print(f"[yellow]{escape(result.note)}[/yellow]")

    if not result.candidates:
        console.print("[dim]No summarize candidates found.[/dim]")
        return

    from rich.table import Table

    def _new_table(show_header: bool) -> Table:
        t = Table(show_header=show_header, header_style="bold", box=None, padding=(0, 2))
        t.add_column("FILE")
        t.add_column("KIND", style="dim")
        t.add_column("PROSE WORDS", justify="right")
        t.add_column("EST. TOKENS/CALL", justify="right")
        return t

    def _add_rows(t: Table, items: list) -> None:
        for c in items:
            t.add_row(escape(c.path), "prompt" if c.is_prompt else "other",
                      str(c.prose_words), f"~{format_tokens(c.est_tokens_saved)}")

    # What the user asked for (the scanned location) prints first; the always-on catalog
    # globals follow a divider — supplementary, not the focus (DEC-021). Kind orders within.
    requested = [c for c in result.candidates if c.scope != "global"]
    catalog_globals = [c for c in result.candidates if c.scope == "global"]
    if requested:
        t = _new_table(show_header=True)
        _add_rows(t, requested)
        console.print(t)
    if catalog_globals:
        if requested:
            console.print("[dim]── global / catalog defaults (always included) ──[/dim]")
        t = _new_table(show_header=not requested)
        _add_rows(t, catalog_globals)
        console.print(t)
    console.print()
    console.print(f"[dim]{escape(CANDIDATE_NOTE)}[/dim]")


@cmd_summarize.command("prep")
@click.argument("path")
@click.option("--ratio", default=DEFAULT_TARGET_RATIO, show_default=True, type=float,
              help="Target prose ratio (0.5 = keep ~half the prose words).")
@click.option("--json", "output_json_flag", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def cmd_summarize_prep(ctx: click.Context, path: str, ratio: float, output_json_flag: bool) -> None:
    """Wrap a prompt's structure → emit the wrapped prompt + rules + hash for you to rewrite, then `check`."""
    output_json: bool = output_json_flag or ctx.obj.get("output_json", False)
    try:
        result = prepare(path=path, ratio=ratio)
    except SummarizeRefused as e:
        raise click.ClickException(str(e)) from e   # e.g. a symlink — house-voice refuse
    if output_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return
    if not result.wrapped_prompt:                   # below the worth-it prose gate
        console.print(f"[yellow]{escape(result.note)}[/yellow]")
        return
    console.print(f"[dim]{escape(result.path)}[/dim] · prose {result.prose_words} → "
                  f"~{result.target_prose_words} words · "
                  f"{result.protected_blocks} block(s) kept verbatim")
    console.print(f"hash: [bold]{result.source_sha256}[/bold]")
    # The manual/copy path: emit the actual payload so the user can rewrite in any model
    # without needing --json (a JSON form is still available via --json for tooling).
    console.print()
    console.print("[bold]── rewrite rules (system prompt for the model) ──[/bold]")
    console.print(escape(result.system_rules))
    console.print()
    console.print("[bold]── wrapped prompt (summarize the prose; keep every <tj-keep> marker verbatim) ──[/bold]")
    console.print(escape(result.wrapped_prompt))
    console.print()
    console.print("[dim]Save the rewrite to a file, then: tj summarize check "
                  f"{escape(result.path)} --summary <file> --prepped-hash {result.source_sha256}[/dim]")


@cmd_summarize.command("check")
@click.argument("path")
@click.option("--summary", "summary_path", required=True,
              help="File holding the model's summary ('-' for stdin).")
@click.option("--prepped-hash", "prepped_hash", required=True,
              help="The source_sha256 returned by `prep`.")
@click.option("--json", "output_json_flag", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def cmd_summarize_check(
    ctx: click.Context, path: str, summary_path: str, prepped_hash: str, output_json_flag: bool,
) -> None:
    """Verify a summary (hash-guards the file) and stage it for review."""
    config: TjConfig = ctx.obj["config"]
    output_json: bool = output_json_flag or ctx.obj.get("output_json", False)
    summary_text = (
        click.get_text_stream("stdin").read() if summary_path == "-"
        else Path(summary_path).expanduser().read_text(encoding="utf-8")
    )
    try:
        verdict = check(config, path, summary_text, prepped_hash)
    except SummarizeRefused as e:
        raise click.ClickException(str(e)) from e   # file changed/missing — house-voice refuse
    if output_json:
        click.echo(json.dumps(verdict.to_dict(), indent=2))
        return
    _print_verdict(config, verdict)
