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

    `total` is the cheap pre-count of in-scope sessions
    (`count_claude_code_sessions_in_scope`), or `None` when unknown — the
    counter then shows a running count with no "/total". `quiet=True` yields a
    no-op callback (mirrors `tj backfill claude-code --quiet`).

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
        return f"Backfilling {count} sessions · {format_tokens(tokens_seen)} tokens read"

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


__all__ = ["backfill_progress", "ProgressCallback"]
