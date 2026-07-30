"""`tj summarize` — structure-aware prompt summarization (advisory in v1).

`tj summarize list` finds prompt files worth summarizing and estimates the
per-call token saving. Bare = the known-location catalog (globals + this dir).
A scope-widening input — a PATH, `--repo`, `--recursive`, or `--ext` — opens it
to all `*.md`; the scanned location is shown first, the catalog globals after a
divider. `prep` wraps a prompt's structure and emits it for you to rewrite (or `--via
claude-p`/`--via api` to have a model do it in one shot); `check` verifies the rewrite
preserved every structure block (a hard gate) and stages it; `apply`
writes a staged result (taking a backup first), `undo` reverts — both default to a dry-run,
`--go` writes. `calibrate` samples real rewrites so the savings estimate can use a MEASURED
prose ratio instead of the unenforced target it is otherwise assuming; it too defaults to a
dry-run, because every sample is a billed model call. See DEC-020/021/024/025.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.markup import escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.config import TjConfig
from tokenjam.core.summarize.apply import apply_staged, undo
from tokenjam.core.summarize.calibrate import (
    DEFAULT_SAMPLES,
    MAX_SAMPLES,
    CalibrationReport,
    run_calibration,
)
from tokenjam.core.summarize.candidates import list_candidates
from tokenjam.core.summarize.delivery import Amortization, DeliveryError, summarize_via
from tokenjam.core.summarize.relocate import (
    DEFAULT_TARGET,
    apply_relocation,
    plan_relocation,
)
from tokenjam.core.summarize.estimate import (
    DEFAULT_TARGET_RATIO,
    UNMEASURED_PRIOR_RANGE,
    UNMEASURED_PRIOR_RATIO,
    UNMEASURED_PRIOR_SAMPLES,
    observed_prose_ratio,
)
from tokenjam.core.summarize.session import CheckVerdict, SummarizeRefused, check, prepare
from tokenjam.utils.formatting import console, format_tokens

# Honesty discipline (CLAUDE.md Rule 14): every candidate is a suggestion to
# review, never an assertion the rewrite is safe — and the saving is estimated.
CANDIDATE_NOTE = (
    "Candidates only — review the summary before adopting. The figure is the "
    "estimated per-call token reduction, which amortizes across every reuse of "
    "the (cached) prompt."
)


def _print_verdict(verdict: CheckVerdict) -> None:
    """Human-readable check verdict: the ✓/✗ line and, when staged, how to review it.
    (must-keep word movement is recorded on the staged result for later metrics — collected,
    never surfaced to the user here.)"""
    if verdict.structure_ok:
        console.print(f"[green]✓[/green] {escape(verdict.path)} — structure preserved, "
                      f"~{format_tokens(verdict.est_tokens_saved or 0)} prompt tok/call "
                      f"({verdict.words_before}→{verdict.words_after} words)")
    else:
        console.print(f"[red]✗[/red] {escape(verdict.path)} — {escape(verdict.reason)} (not staged)")
    if verdict.staged:
        console.print(f"[dim]review it: tj summarize apply {escape(verdict.path)} "
                      f"(dry-run shows the diff), then --go to write.[/dim]")


def _print_diff(diff: str) -> None:
    """Render a unified diff with +/- coloring — the dry-run preview of a staged rewrite."""
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            console.print(f"[dim]{escape(line)}[/dim]")
        elif line.startswith("+"):
            console.print(f"[green]{escape(line)}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{escape(line)}[/red]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{escape(line)}[/cyan]")
        else:
            console.print(f"[dim]{escape(line)}[/dim]")


def _print_amortization(amort: Amortization) -> None:
    """The api 'pays for itself' line — real charge (or default-rate estimate) ÷ Estimate saving (DEC-029)."""
    charge = "real charge" if amort.rates_known else "estimated, default rates"
    line = f"[dim]~${amort.rewrite_usd:.4f} to rewrite ({charge})"
    if amort.saving_usd_per_call > 0:
        line += f" · ~${amort.saving_usd_per_call:.4f}/call saved (Estimate)"
    else:
        line += " · no staged saving to amortize"
    if amort.break_even_calls is not None:
        if amort.rates_known:
            line += f" · pays for itself in ~{amort.break_even_calls} use(s) at {escape(amort.model)} rates"
        else:
            line += (f" · pays for itself in ~{amort.break_even_calls} use(s) — "
                     f"add pricing for {escape(amort.model)} for the real charge")
    console.print(line + "[/dim]")


@click.group("summarize", invoke_without_command=False)
def cmd_summarize() -> None:
    """Summarize prompts (structure-aware, advisory)."""


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
@json_option
@click.option("--min-prose", "min_prose", default=None, type=int,
              help="Minimum prose words to flag a file (default 100).")
@click.pass_context
def cmd_summarize_list(
    ctx: click.Context, path: str | None, recursive: bool, repo: bool,
    no_global: bool, ext: str | None, output_json_flag: bool, min_prose: int | None,
) -> None:
    """List prompt files worth summarizing (bare = catalog; a PATH/--repo/--recursive/--ext opens to .md)."""
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)

    if repo and recursive:
        raise click.UsageError("--repo and --recursive are mutually exclusive.")
    if repo and path is not None:
        raise click.UsageError("--repo cannot be combined with an explicit PATH.")

    extra_exts = tuple(e for e in (ext.split(",") if ext else []) if e.strip())
    kwargs: dict = {}
    if min_prose is not None:
        kwargs["min_prose_words"] = min_prose
    # Forecast on what rewrites have ACTUALLY delivered here when that is known,
    # and on the measured prior otherwise — never on the target the rewriter is
    # merely asked for. `list` forecasting at the ask is what let a file
    # advertised at 286 tokens deliver 81.
    measured_ratio, ratio_samples = observed_prose_ratio(config)
    result = list_candidates(
        path, config=config, recursive=recursive, repo=repo,
        include_global=not no_global, extra_exts=extra_exts,
        ratio=measured_ratio if measured_ratio is not None else UNMEASURED_PRIOR_RATIO,
        **kwargs,
    )
    ratio_note = (
        f"Reduction assumes prose compresses to {(measured_ratio or 0) * 100:.0f}% "
        f"of its words, measured across {ratio_samples:,} verified rewrite(s) here."
        if measured_ratio is not None else
        f"Reduction assumes prose compresses to {UNMEASURED_PRIOR_RATIO * 100:.0f}% of "
        f"its words — tokenjam's measurement on other machines, not yours "
        f"({UNMEASURED_PRIOR_SAMPLES:,} rewrites spanning "
        f"{UNMEASURED_PRIOR_RANGE[0]:.0%}-{UNMEASURED_PRIOR_RANGE[1]:.0%}). "
        f"Run `tj summarize calibrate --via claude-p --go` to measure your own."
    )

    if output_json:
        payload = result.to_dict()
        payload["note"] = result.note or CANDIDATE_NOTE
        payload["ratio_basis"] = ratio_note
        payload["prose_ratio_observed"] = measured_ratio is not None
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
    console.print(f"[dim]{escape(ratio_note)}[/dim]")


@cmd_summarize.command("prep")
@click.argument("path")
@click.option("--via", "via", type=click.Choice(["claude-p", "api"]), default=None,
              help="Let TJ run the rewrite for you: 'claude-p' drives your local Claude Code "
                   "(headless `claude -p`); 'api' calls Anthropic with your TJ_ANTHROPIC_API_KEY "
                   "(needs [summarize] api_model). Omit it to rewrite the prompt yourself, then `check`.")
@click.option("--ratio", default=DEFAULT_TARGET_RATIO, show_default=True, type=float,
              help="Target prose ratio (0.5 = keep ~half the prose words).")
@json_option
@click.pass_context
def cmd_summarize_prep(
    ctx: click.Context, path: str, via: str | None, ratio: float, output_json_flag: bool,
) -> None:
    """Wrap a prompt's structure. Bare: emit the wrapped prompt + rules + hash for you to rewrite,
    then `check`. With --via: TJ runs the rewrite, verifies, and stages it in one shot."""
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)

    if via is not None:                             # automated: wrap → rewrite → check → stage
        on_progress = None if output_json else (
            lambda m: console.print(f"[dim]{escape(m)}…[/dim]"))
        try:
            outcome = summarize_via(config, path, via, ratio=ratio, on_progress=on_progress)
        except (DeliveryError, SummarizeRefused) as e:
            raise click.ClickException(str(e)) from e
        if outcome.verdict is None:                 # below the worth-it prose gate (note from the one prep)
            if output_json:
                click.echo(json.dumps(
                    {"path": path, "staged": False, "note": outcome.skipped_note}, indent=2))
            else:
                console.print(f"[yellow]{escape(outcome.skipped_note or '')}[/yellow]")
            return
        if output_json:
            payload = outcome.verdict.to_dict()
            if outcome.amortization is not None:
                payload["amortization"] = outcome.amortization.to_dict()
            elif outcome.cost_unknown:
                payload["cost"] = "unknown"          # api call was billed, but the response had no usage
            click.echo(json.dumps(payload, indent=2))
            return
        _print_verdict(outcome.verdict)
        if outcome.amortization is not None:
            _print_amortization(outcome.amortization)
        elif outcome.cost_unknown:
            console.print("[dim]api rewrite — cost unknown (the response carried no usage)[/dim]")
        return

    try:
        result = prepare(path=path, ratio=ratio)    # manual: emit for the user to rewrite
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
    if result.target_basis:                         # whose target this is, and why
        console.print(f"[dim]{escape(result.target_basis)}[/dim]")
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


def _print_calibration(report: CalibrationReport) -> None:
    """The calibration verdict: what was sampled, what it cost, what it showed."""
    if report.dry_run:
        for t in report.planned:
            console.print(f"[dim]would sample[/dim] {escape(t.path)} "
                          f"({t.prose_words:,} prose words)")
        console.print(f"[yellow]{escape(report.note)}[/yellow]")
        return

    for s in report.samples:
        if s.achieved_ratio is not None:
            console.print(
                f"[green]✓[/green] {escape(s.path)} — prose to "
                f"{s.achieved_ratio * 100:.0f}% of its words "
                f"({s.words_before}→{s.words_after} words)")
        else:
            console.print(f"[red]✗[/red] {escape(s.path)} — "
                          f"{escape(s.error or 'no usable outcome')} (recorded, not staged)")
    if not report.samples:
        console.print("[dim]Nothing was sampled.[/dim]")
        return
    if report.rewrite_usd is not None:
        console.print(f"[dim]{len(report.samples):,} rewrite(s) via {escape(report.via)}; "
                      f"~${report.rewrite_usd:.4f} billed.[/dim]")
    else:
        # Not "free": claude-p spends the user's Claude Code quota, it just
        # reports no per-token price. Saying $0.00 would be a quiet lie.
        console.print(f"[dim]{len(report.samples):,} rewrite(s) via {escape(report.via)}; "
                      f"per-token cost not reported on this path.[/dim]")
    console.print(escape(report.note))


@cmd_summarize.command("calibrate")
@click.option("--via", "via", type=click.Choice(["claude-p", "api"]), required=True,
              help="How to run the sample rewrites: 'claude-p' drives your local Claude Code "
                   "(headless `claude -p`); 'api' calls Anthropic with your TJ_ANTHROPIC_API_KEY "
                   "(needs [summarize] api_model).")
@click.option("--limit", "limit", default=DEFAULT_SAMPLES, show_default=True, type=int,
              help=f"How many files to sample (hard cap {MAX_SAMPLES}).")
@click.option("--go", is_flag=True,
              help="Actually run the rewrites (default is a dry-run that spends nothing).")
@click.argument("path", required=False, default=None)
@json_option
@click.pass_context
def cmd_summarize_calibrate(
    ctx: click.Context, via: str, limit: int, go: bool, path: str | None,
    output_json_flag: bool,
) -> None:
    """Measure what a rewrite actually delivers here, instead of assuming the target.

    The savings estimate assumes prose compresses to the ratio the rewriter is
    ASKED for, which nothing enforces. This samples the largest prompt files with
    real rewrites and records what they achieved, so the estimate can use a
    measured ratio. Each sample is a billed model call; default is a dry-run.
    """
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)
    on_progress = None if output_json else (lambda m: console.print(f"[dim]{escape(m)}…[/dim]"))
    try:
        report = run_calibration(
            config, via=via, limit=limit, go=go, path=path, on_progress=on_progress)
    except (DeliveryError, SummarizeRefused) as e:
        raise click.ClickException(str(e)) from e
    if output_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
        return
    _print_calibration(report)


@cmd_summarize.command("check")
@click.argument("path")
@click.option("--summary", "summary_path", required=True,
              help="File holding the model's summary ('-' for stdin).")
@click.option("--prepped-hash", "prepped_hash", required=True,
              help="The source_sha256 returned by `prep`.")
@json_option
@click.pass_context
def cmd_summarize_check(
    ctx: click.Context, path: str, summary_path: str, prepped_hash: str, output_json_flag: bool,
) -> None:
    """Verify a summary (hash-guards the file) and stage it for review."""
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)
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
    _print_verdict(verdict)


@cmd_summarize.command("apply")
@click.argument("path", required=False, default=None)
@click.option("--go", is_flag=True,
              help="Write the files (default is dry-run; can't combine with --dry-run).")
@click.option("--dry-run", "dry_run", is_flag=True,
              help="Preview only; the default (can't combine with --go).")
@json_option
@click.pass_context
def cmd_summarize_apply(
    ctx: click.Context, path: str | None, go: bool, dry_run: bool, output_json_flag: bool,
) -> None:
    """Apply staged results to their files — take-all, or one PATH. Default dry-run; --go writes."""
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)
    if dry_run and go:
        raise click.UsageError("Choose one of --dry-run or --go (--dry-run is the default with neither).")
    if path is not None and Path(path).expanduser().is_dir():
        raise click.UsageError("PATH is a directory — accept all (no PATH) or specify one file.")

    report = apply_staged(config, path, go=go)          # default dry-run; --go writes (both rejected above)
    if output_json:
        click.echo(json.dumps(report, indent=2))
        return

    verb = "applied" if go else "would apply"
    for a in report["applied"]:
        if not go and a["diff"]:                        # dry-run = preview the actual change
            _print_diff(a["diff"])
        console.print(f"[green]✓[/green] {verb} {escape(a['path'])} "
                      f"(~{format_tokens(a['est_tokens_saved'])} prompt tok/call)")
    for s in report["skipped"]:
        console.print(f"[yellow]skip[/yellow] {escape(s['path'])} — {escape(s['reason'])}")
    if not report["applied"] and not report["skipped"]:
        console.print("[dim]nothing staged.[/dim]")
    elif report["dry_run"]:
        console.print("[dim]dry-run — nothing written. Re-run with --go to apply.[/dim]")


@cmd_summarize.command("undo")
@click.argument("path")
@click.option("--go", is_flag=True,
              help="Restore the file (default is dry-run; can't combine with --dry-run).")
@click.option("--dry-run", "dry_run", is_flag=True,
              help="Preview only; the default (can't combine with --go).")
@json_option
@click.pass_context
def cmd_summarize_undo(
    ctx: click.Context, path: str, go: bool, dry_run: bool, output_json_flag: bool,
) -> None:
    """Restore a file from its summarize backup. Default dry-run; --go writes. Refuses on drift."""
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)
    if dry_run and go:
        raise click.UsageError("Choose one of --dry-run or --go (--dry-run is the default with neither).")
    if Path(path).expanduser().is_dir():
        raise click.UsageError("PATH is a directory — undo takes one file.")

    try:
        result = undo(config, path, go=go)              # default dry-run; --go writes (both rejected above)
    except SummarizeRefused as e:
        raise click.ClickException(str(e)) from e        # missing backup / changed since apply

    if output_json:
        click.echo(json.dumps(result, indent=2))
        return
    if result["dry_run"]:
        console.print(f"[dim]would restore {escape(result['path'])} from backup — re-run with --go.[/dim]")
    else:
        console.print(f"[green]✓[/green] restored {escape(result['path'])} from backup")


@cmd_summarize.command("relocate")
@click.argument("path")
@click.option("--to", "target", default=None,
              help=f"Where the reference material goes (default: {DEFAULT_TARGET} beside PATH).")
@click.option("--section", "sections", multiple=True,
              help="Only this section (repeatable). Still subject to the classifier.")
@click.option("--go", is_flag=True,
              help="Write the files (default is dry-run; can't combine with --dry-run).")
@click.option("--dry-run", "dry_run", is_flag=True,
              help="Preview only; the default (can't combine with --go).")
@json_option
@click.pass_context
def cmd_summarize_relocate(
    ctx: click.Context, path: str, target: str | None, sections: tuple[str, ...],
    go: bool, dry_run: bool, output_json_flag: bool,
) -> None:
    """Move REFERENCE sections out of PATH into a linked file, leaving a pointer.

    Nothing is rewritten and nothing is deleted: the text moves and a pointer
    stays behind, so unlike a summary this cannot change what any surviving
    instruction says. Only sections a classifier is confident describe what
    EXISTS are moved; anything that might be an instruction is left alone and
    the reason is printed. Default dry-run; --go writes.
    """
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)
    if dry_run and go:
        raise click.UsageError("Choose one of --dry-run or --go (--dry-run is the default with neither).")
    source = Path(path).expanduser()
    if source.is_dir():
        raise click.UsageError("PATH is a directory; relocate takes one file.")
    if not source.is_file():
        raise click.UsageError(f"{path} is not a file.")

    target_path = Path(target).expanduser() if target else source.parent / DEFAULT_TARGET
    target_text = target_path.read_text(encoding="utf-8") if target_path.is_file() else ""
    try:
        plan = plan_relocation(
            source_path=str(source), source_text=source.read_text(encoding="utf-8"),
            target_path=str(target_path), target_text=target_text,
            titles=list(sections) or None,
        )
    except SummarizeRefused as e:
        raise click.ClickException(str(e)) from e

    if plan is None:
        if output_json:
            click.echo(json.dumps({"plan": None, "applied": False}, indent=2))
            return
        console.print(
            f"[muted]No section of {escape(str(source))} is confidently reference "
            f"material, so nothing is offered. Leaving a section in place costs a "
            f"saving; moving an instruction out of an always-loaded file costs "
            f"correctness, so the ambiguous cases stay put.[/muted]"
        )
        return

    result = apply_relocation(config, plan, go=go)
    if output_json:
        click.echo(json.dumps(result, indent=2))
        return

    for s in plan.sections:
        verb = "moved" if result["applied"] else "would move"
        console.print(
            f"[ok]✓[/ok] {verb} [accent]{escape(s.title)}[/accent] to "
            f"[accent]{escape(str(target_path))}[/accent] "
            f"(~{format_tokens(s.tokens_freed)} always-resident tok/read)"
        )
        console.print(f"  [muted]{escape(s.classification.reason)}[/muted]")
    for title, verdict in plan.declined:
        console.print(f"[muted]left in place: {escape(title)}; {escape(verdict.reason)}[/muted]")
    for skip in result["skipped"]:
        console.print(f"[warn]skip[/warn] {escape(skip['path'])}; {escape(skip['reason'])}")
    if result["dry_run"]:
        console.print("[muted]dry-run; nothing written. Re-run with --go to apply.[/muted]")
    elif result["applied"]:
        console.print(
            f"[muted]Both files are backed up; `tj summarize undo {escape(str(source))} --go` "
            f"restores the original.[/muted]"
        )
