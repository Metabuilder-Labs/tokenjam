"""Reusable delayed-start progress indicator for slow CLI commands.

The CLI went silent during long operations (`tj optimize` runs 40-90s of
analyzers against a real Claude Code history and prints nothing the whole
time) which reads as "hung", not "working". This module is the one
cross-cutting primitive every slow command wraps its work in, instead of each
command bolting on its own spinner.

Design:

  * DELAYED START — nothing renders for the first
    `PROGRESS_START_DELAY_SECONDS`, so the overwhelming majority of commands
    (which finish well under that) stay exactly as silent as they are today.
  * STDERR ONLY — stdout is the data channel; the indicator never touches it.
  * SUPPRESSED entirely (see `progress_disabled`) when stderr isn't a real
    terminal, `--json` is in effect, a quiet/no-progress flag is set, or the
    conventional `CI` env var is present — the exact cases where a stray
    spinner frame could corrupt a script's captured output.
  * CLEAN TEARDOWN on success, on exception, and on Ctrl-C — the spinner line
    is always erased and the cursor restored.

`tj backfill claude-code` / `tj onboard --claude-code` already have their own
purpose-built counter (`tokenjam.cli.backfill_progress`) driven by a
per-session callback from `ingest_claude_code` — that one stays as-is. This
module is for everything else: a labelled spinner around a call site, with an
optional `.update(label)` for callers (like `tj optimize`'s analyzer loop)
that can say what they're doing as they go.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Callable, Iterator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

from tokenjam.utils.formatting import err_console as _default_console

# How long a command gets to finish before the spinner is allowed to render
# anything. Named so tuning it isn't a hunt for a bare literal.
PROGRESS_START_DELAY_SECONDS = 1.0


def progress_disabled(
    *, output_json: bool = False, quiet: bool = False, console: Console | None = None,
) -> bool:
    """Whether the progress indicator must stay silent for this invocation.

    True when any of: `--json` (or another machine-readable mode) is in
    effect, an explicit quiet/no-progress flag is set, the conventional `CI`
    env var is present, or stderr isn't a real terminal (piped, redirected,
    captured). Machine-readable output must stay byte-identical regardless of
    timing, so callers should compute this once, up front, and pass it
    through as `disabled=` rather than let the indicator guess.
    """
    target = console if console is not None else _default_console
    return bool(output_json) or bool(quiet) or bool(os.environ.get("CI")) or not target.is_terminal


def _noop_update(_label: str) -> None:
    pass


class ProgressHandle:
    """Handle yielded by `progress_indicator` — update the visible label."""

    def __init__(self, update_fn: Callable[[str], None]) -> None:
        self._update_fn = update_fn

    def update(self, label: str) -> None:
        """Change the rendered label (no-op before the delayed start fires,
        and no-op entirely when the indicator is disabled)."""
        self._update_fn(label)


class _DelayedSpinner:
    """Owns the Rich `Progress` instance, started only after `delay` seconds.

    A background `threading.Timer` fires `_start`; if the caller's work
    finishes (and `end()` runs) before the timer fires, the timer is
    cancelled and nothing is ever rendered. `update()` is safe to call at any
    time — before the spinner exists it just records the latest label for
    whenever (if ever) `_start` renders it.
    """

    def __init__(self, label: str, console: Console, delay: float) -> None:
        self._console = console
        self._lock = threading.Lock()
        self._label = label
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._stopped = False
        self._timer = threading.Timer(delay, self._start)
        self._timer.daemon = True

    def _start(self) -> None:
        with self._lock:
            if self._stopped:
                return
            progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold]{task.description}[/bold]"),
                console=self._console,
                transient=True,
            )
            progress.start()
            self._progress = progress
            self._task_id = progress.add_task(self._label, total=None)

    def update(self, label: str) -> None:
        with self._lock:
            self._label = label
            if self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, description=label)

    def begin(self) -> None:
        self._timer.start()

    def end(self) -> None:
        """Stop the timer (if it hasn't fired) and the spinner (if it has).

        Safe to call on every exit path — success, exception, KeyboardInterrupt
        — since it never raises and always leaves the terminal clean.
        """
        self._timer.cancel()
        with self._lock:
            self._stopped = True
            if self._progress is not None:
                self._progress.stop()
                self._progress = None


@contextmanager
def progress_indicator(
    label: str,
    *,
    disabled: bool = False,
    console: Console | None = None,
    delay: float = PROGRESS_START_DELAY_SECONDS,
) -> Iterator[ProgressHandle]:
    """Show a spinner + `label` on stderr, but only after `delay` seconds.

    `disabled` should be the result of `progress_disabled(...)` — callers
    compute it once against the suppression matrix (non-TTY, --json, quiet,
    CI) and pass it through, so a disabled indicator costs nothing (no timer,
    no thread) rather than starting and immediately no-opping.

    Yields a `ProgressHandle` so callers doing multi-step work (e.g. `tj
    optimize` running one analyzer after another) can `.update(...)` the
    label as they go — a user watching a long wait benefits far more from
    "Scanning transcripts..." than from an anonymous twirl.
    """
    if disabled:
        yield ProgressHandle(_noop_update)
        return

    target = console if console is not None else _default_console
    spinner = _DelayedSpinner(label, target, delay)
    spinner.begin()
    try:
        yield ProgressHandle(spinner.update)
    finally:
        spinner.end()


__all__ = [
    "PROGRESS_START_DELAY_SECONDS",
    "ProgressHandle",
    "progress_disabled",
    "progress_indicator",
]
