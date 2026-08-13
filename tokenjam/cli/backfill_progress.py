"""Shared streaming-progress UI for slow CLI stretches.

Three shapes, all built on the same Rich `Progress` spinner so a command that
runs one after another reads as one continuous process rather than switching
UIs mid-command:

- `backfill_progress` — a per-item ticking counter ("N/total transcripts").
  Built for `ingest_claude_code`'s `progress=` hook (called once per parsed
  session); `tj backfill claude-code` and `tj onboard --claude-code` share it
  instead of each rolling its own.
- `transient_status` — one static self-erasing line for a slow stretch with
  no per-item count to show. The Click-layer `TjCommand` (`cli/tj_status.py`)
  is built on this.
- `phase_status` — like `transient_status`, but for a stretch with more than
  one named phase (`tj summarize prep --via ...`: wrap → rewrite → verify).
  Yields an `update(message)` callable that replaces the line in place
  instead of printing a new one per phase.

On a real terminal all three render via Rich `Progress` (one line that
updates or replaces in place); on a non-terminal (piped output, CI,
redirected logs — anywhere Rich's live redraw can't work) they degrade to
plain `console.print` lines instead: `backfill_progress` prints periodically
(bounded call count is per-item, so it would spam a huge history otherwise),
`phase_status` and `transient_status` print once per phase/stretch (their
call count is already small — a handful of phases, or one).
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
StatusUpdate = Callable[[str], None]

# Non-TTY print cadence: one line every Nth session, so a redirected/CI run
# shows progress without a line per session on a large history.
_PLAIN_PRINT_EVERY = 100


def _spinner(console: Console) -> Progress:
    """The one Rich `Progress` construction every shape in this module
    renders through — same spinner, same `◆` prefix, same erase-on-exit.
    Change the loading UI once here and every command picks it up.

    `redirect_stdout=False, redirect_stderr=False`: Rich's `Live` defaults to
    BOTH true, which installs a proxy over `sys.stdout`/`sys.stderr` for the
    life of the display and routes ANY raw write on either stream through
    THIS console's `.print()` instead. `click.echo()` on a real terminal
    turns out NOT to be at risk in practice — it resolves the stream's
    underlying binary buffer and writes straight to it, sidestepping the
    proxy entirely (verified: forcing the proxy active around a
    `click.echo()` call with a realistic stdout still produced byte-clean
    output). But that protection is Click's implementation detail, not a
    contract this module should depend on — a bare `print()`, or any future
    caller that writes to the stream directly, would NOT get it. Turning the
    redirect off entirely removes the whole class of risk instead of relying
    on an incidental side effect elsewhere, at zero cost: nothing in this
    module needs Rich to intercept prints on a stream it isn't rendering to.
    """
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]◆[/bold] {task.description}"),
        console=console,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    )


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
        with _spinner(target_console) as progress:
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

    with _spinner(target_console) as progress:
        progress.add_task(message, total=None)
        yield


@contextmanager
def phase_status(
    message: str, *, console: Console | None = None,
) -> Iterator[StatusUpdate]:
    """Hold one self-erasing status line across SEVERAL named phases.

    For a multi-step stretch that has no per-item count but does have more
    than one thing worth naming as it happens — `tj summarize prep --via
    ...` (wrap → rewrite → verify) and `tj summarize calibrate --go` (one
    line per sampled file) are the callers. Built from the same `_spinner`
    construction as `transient_status` above, so it looks identical; the
    difference is the caller gets an `update(message)` callable back instead
    of a bare block, and each call REPLACES the line rather than adding one —
    unlike a static `console.print()` per phase, nothing accumulates on
    screen, and the last phase erases with everything else on exit.

    On a non-terminal this degrades to a plain `console.print` per `update()`
    call rather than going silent — the call count here is bounded by phase
    count (a handful), never per-item, so it can't spam a large history the
    way an uncapped `backfill_progress`-shaped counter would. Like
    `transient_status`, messages are printed exactly as given — callers own
    their own trailing punctuation (an ellipsis for "in progress" phases).
    """
    target_console = console if console is not None else _default_console

    if not target_console.is_terminal:
        def _update_plain(msg: str) -> None:
            target_console.print(f"[dim]{msg}[/dim]")
        yield _update_plain
        return

    with _spinner(target_console) as progress:
        task_id = progress.add_task(message, total=None)

        def _update_live(msg: str) -> None:
            progress.update(task_id, description=msg)

        yield _update_live


__all__ = ["backfill_progress", "transient_status", "phase_status", "ProgressCallback",
           "StatusUpdate"]
