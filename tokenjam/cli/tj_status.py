"""The universal CLI loading-state choke point.

`tj optimize` and `tj status` used to run in total silence — 96s and 48s on a
real corpus — which reads as a hang. `backfill_progress.py` already had a
progress primitive (`transient_status`, `backfill_progress`) but it was wired
into four call sites out of ~40 commands, because nothing forced a NEW
command to consider it. This module is the fix for that: `TjCommand` /
`TjGroup` are drop-in `click.Command` / `click.Group` subclasses that every
registered command in `cli/main.py` (and every group's own subcommands) is
built from, so the choke point is structural rather than a convention
someone has to remember. Changing the loading UI is a one-file edit to
`backfill_progress.py`'s `_spinner()`; changing WHICH commands render one is
a one-line edit here or at the call site — never a new bespoke spinner.

A status line renders only when a command explicitly sets `status_message` —
by design there is no timing heuristic anywhere in this file. "Is this
command slow" is a one-time authoring decision, not something inferred from
a clock. `tests/unit/test_cli_status_registry.py` is the guard that keeps
a newly added command from shipping without that decision being made.

Simple, single-phase, fully non-interactive commands opt in with one kwarg
on their `@click.command`:

    @click.command("status", cls=TjCommand, status_message="Scanning your sessions…")

**Commands that read from stdin anywhere in their body (`click.prompt`,
`click.confirm`, `click.edit`) must NOT set `status_message`.** Rich's
spinner is a live, auto-refreshing display; a background thread repainting
it collides with a blocking terminal read for the same reason two people
typing on one keyboard collide. `tj optimize --validate`, `tj uninstall`,
`tj reset`, and `tj onboard` all have a confirmation gate or interactive
prompt somewhere in their body, so all four leave `status_message` unset and
instead call `tj_status()` directly around just the silent stretch, after
the gate has already resolved. `tj summarize prep --via ...` and `calibrate
--go` have more than one named phase, so they use `tj_status_stream()`
(built on `phase_status`) instead of a single fixed message.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import click

from tokenjam.cli.backfill_progress import StatusUpdate, phase_status, transient_status
from tokenjam.utils.formatting import console as _stdout_console
from tokenjam.utils.formatting import err_console as _stderr_console

# Local `--json` parameter names this module checks generically, so it never
# needs to know which spelling a given command used: `json_option`'s shared
# decorator always names its parameter `output_json_flag`; a couple of
# commands with their own ad hoc `--json` option (e.g. `backfill status`)
# name it `output_json`. Either being True routes the status line to stderr.
_JSON_PARAM_NAMES = ("output_json_flag", "output_json")


def _json_requested(ctx: click.Context) -> bool:
    """Whether JSON output was requested, globally (`tj --json <cmd>`) or
    locally (`tj <cmd> --json`) — same two-flag contract as
    `json_option.resolve_output_json`, generalised across commands rather
    than tied to one parameter name.
    """
    if ctx.obj and ctx.obj.get("output_json"):
        return True
    return any(ctx.params.get(name) for name in _JSON_PARAM_NAMES)


def _status_console(ctx: click.Context):
    """stderr under `--json` — stdout must stay byte-clean for a machine
    reader — else the normal shared stdout console."""
    return _stderr_console if _json_requested(ctx) else _stdout_console


def tj_status(message: str, ctx: click.Context) -> AbstractContextManager[None]:
    """One self-erasing status line for the duration of a `with` block,
    console-routed for `ctx`. `TjCommand` calls this for you when
    `status_message` is set; call it directly for a slow stretch that is
    NOT the whole command body (see module docstring for when that's
    required)."""
    return transient_status(message, console=_status_console(ctx))


def tj_status_stream(message: str, ctx: click.Context) -> AbstractContextManager[StatusUpdate]:
    """Like `tj_status`, but for a stretch with more than one named phase.
    Yields an `update(message)` callable — see `phase_status`."""
    return phase_status(message, console=_status_console(ctx))


class TjCommand(click.Command):
    """The class every `tj` command is built from, whether or not it ever
    renders anything. `status_message` is the single switch that decides
    that (Rule: no time-based heuristic — see module docstring for why).

    `no_status` carries no runtime behavior of its own — `not status_message`
    already skips the wrap either way. It exists purely so a command can be
    marked as a considered, deliberate silence (`mcp`, `serve`, `statusline`,
    `session-end` — each commented at its call site) rather than one that
    simply hasn't been decided yet, and so the registry test can tell those
    apart.
    """

    def __init__(self, *args: Any, status_message: str | None = None,
                 no_status: bool = False, **kwargs: Any) -> None:
        self.status_message = status_message
        self.no_status = no_status
        super().__init__(*args, **kwargs)

    def _tokens_request_help(self, ctx: click.Context) -> bool:
        names: set[str] = set()
        node: click.Context | None = ctx
        while node is not None:
            names.update(node.help_option_names)
            node = node.parent
        tokens = list(ctx.args)
        return any(tok in names for tok in tokens)

    def invoke(self, ctx: click.Context) -> Any:
        ctx.ensure_object(dict)
        if self._tokens_request_help(ctx):
            # Root/nested group callbacks probe DuckDB before Click renders
            # ``--help`` on a leaf subcommand (#580).
            ctx.obj["_skip_db_for_help"] = True
        if not self.status_message:
            return super().invoke(ctx)
        with tj_status(self.status_message, ctx):
            return super().invoke(ctx)


class TjGroup(TjCommand, click.Group):
    """The matching `click.Group` subclass. `command_class` makes every
    subcommand attached with a plain `@group.command(...)` (no explicit
    `cls=`) a `TjCommand` automatically — that's what lets a leaf like
    `tj backfill status` opt into a status with nothing but a
    `status_message=` kwarg, no import of this class needed at that call
    site. `group_class = type` makes a further-nested `@group.group(...)`
    keep using whichever class `self` already is, rather than falling back
    to plain `click.Group` and losing the inheritance one level down.

    **Never set `status_message` on a Group.** `click.Group.invoke` builds
    the chosen subcommand's context and calls ITS `invoke` — a second
    `TjCommand.invoke` — from inside its own; a group-level spinner would
    still be live when the subcommand tried to open its own, and Rich allows
    only one live display at a time (raises `LiveError`). Every group in
    this codebase leaves `status_message` unset for exactly this reason —
    only leaf commands render.
    """
    command_class = TjCommand
    group_class = type


__all__ = ["TjCommand", "TjGroup", "tj_status", "tj_status_stream"]
