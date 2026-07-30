"""Shared streaming-progress UI for Claude Code backfill (#443).

`tj backfill claude-code` and `tj onboard --claude-code` both ingest through
`ingest_claude_code`'s `progress=` hook (called once per parsed session) — this
module gives both callers the same live counter instead of each rolling its
own. On a real terminal it renders one line that updates in place (Rich
`Progress`); on a non-terminal (piped output, CI, redirected logs — anywhere
Rich's live redraw can't work) it degrades to periodic plain `console.print`
lines so a long backfill still shows signs of life without spamming a line per
session.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from tokenjam.core.backfill import BackfillResult, ParsedSession
from tokenjam.utils.formatting import console as _default_console
from tokenjam.utils.humanize import format_tokens

ProgressCallback = Callable[[ParsedSession, BackfillResult], None]

# Non-TTY print cadence: one line every Nth session, so a redirected/CI run
# shows progress without a line per session on a large history.
_PLAIN_PRINT_EVERY = 100


def _noop(_parsed: ParsedSession, _result: BackfillResult) -> None:
    pass


@contextmanager
def backfill_progress(
    total: int | None, *, quiet: bool = False, console: Console | None = None,
) -> Iterator[ProgressCallback]:
    """Yield a `progress(parsed, result)` callback for `ingest_claude_code`.

    `total` is the cheap pre-count of in-scope transcript FILES
    (`count_claude_code_sessions_in_scope`), or `None` when unknown — the
    counter then shows a running count with no "/total". `quiet=True` yields a
    no-op callback (mirrors `tj backfill claude-code --quiet`).

    The counter says "transcripts", not "sessions", and the distinction is the
    point. Both the numerator (`BackfillResult.sessions_seen`) and the total
    count `.jsonl` FILES walked, and a Claude Code session is more than one
    file: every `Task` dispatch writes its own `subagents/agent-*.jsonl`
    sharing the parent's `session_id`. On a real corpus roughly half the files
    under the projects root are subagent transcripts, so a file count reads
    about twice the session count, and calling it "sessions" put two different
    answers to one question on the same screen. Nothing about what gets
    ingested changed: subagent transcripts are still read, and they must be.

    `console` overrides where the counter renders (default: the shared stdout
    console) — `tj quickstart --json` passes the stderr console so the
    counter never contaminates its machine-readable stdout.
    """
    if quiet:
        yield _noop
        return

    target_console = console if console is not None else _default_console
    tokens_seen = 0

    def _line(result: BackfillResult) -> str:
        count = (
            f"{result.sessions_seen:,}/{total:,}" if total is not None
            else f"{result.sessions_seen:,}"
        )
        return (f"Backfilling {count} transcripts · "
                f"{format_tokens(tokens_seen)} tokens read")

    def _accumulate(parsed: ParsedSession) -> None:
        nonlocal tokens_seen
        tokens_seen += (
            parsed.total_input_tokens
            + parsed.total_output_tokens
            + parsed.total_cache_tokens
        )

    if target_console.is_terminal:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]◆[/bold] {task.description}"),
            console=target_console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Backfilling…", total=None)

            def _tick(parsed: ParsedSession, result: BackfillResult) -> None:
                _accumulate(parsed)
                progress.update(task_id, description=_line(result))

            yield _tick
        return

    def _tick_plain(parsed: ParsedSession, result: BackfillResult) -> None:
        _accumulate(parsed)
        if result.sessions_seen % _PLAIN_PRINT_EVERY == 0:
            target_console.print(f"  [dim]{_line(result)}[/dim]")

    yield _tick_plain


@contextmanager
def transient_status(message: str, *, console: Console | None = None) -> Iterator[None]:
    """Hold a single self-erasing status line for the duration of the block.

    For a slow stretch that has no per-item tick to count. Built from the SAME
    `Progress` construction as `backfill_progress` above (same spinner, same
    `◆` prefix, same `transient=True`), so a command that runs one after the
    other reads as one continuous process rather than two different UIs.

    `transient=True` is the contract: Rich erases the line on exit, so whatever
    renders next starts on a clean screen with no residue above it.

    On a NON-terminal (piped output, CI, redirected logs) this prints nothing
    at all, deliberately. Rich's live redraw cannot erase there, so the only
    options are permanent residue above the output or silence, and residue in a
    machine-read or scrolled-back log is the worse of the two. A caller whose
    non-TTY runs need a sign of life has one already: `backfill_progress`
    degrades to periodic plain lines rather than going quiet.
    """
    target_console = console if console is not None else _default_console
    if not target_console.is_terminal:
        yield
        return

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]◆[/bold] {task.description}"),
        console=target_console,
        transient=True,
    ) as progress:
        progress.add_task(message, total=None)
        yield


__all__ = ["backfill_progress", "transient_status", "ProgressCallback"]
