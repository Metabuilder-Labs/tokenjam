"""Guard: every registered `tj` command has explicitly considered the
universal loading state, and nothing constructs its own spinner.

Two failure modes this catches that ruff/mypy cannot:

1. A new command added with a bare `click.command(...)` (no `cls=`) — it
   would silently skip the central `TjCommand`/`TjGroup` machinery and, if
   it ever grew a spinner, would have to roll its own.
2. A new command added WITH `cls=TjCommand` but no thought given to whether
   it should render a status — the four buckets below must partition the
   ENTIRE registered command graph exactly; a command that isn't in any of
   them fails the set-equality assertion and forces it to be added to one,
   which is the "considered, one way or the other" property this test
   exists for.
"""
from __future__ import annotations

import inspect

import click

from tokenjam.cli.main import cli
from tokenjam.cli.tj_status import TjCommand, TjGroup

# --------------------------------------------------------------------------- #
# The four buckets. A command appears in exactly one.
# --------------------------------------------------------------------------- #

#: Groups only dispatch to subcommands — never set `status_message` on one
#: (see `TjGroup`'s docstring for why: a live spinner would still be open
#: when the chosen subcommand tried to open its own).
GROUPS = {
    "cli",
    "cli route",
    "cli backfill",
    "cli policy",
    "cli pricing",
    "cli proxy",
    "cli loop",
    "cli relearn",
    "cli summarize",
    "cli summarize quarantine",
    "cli rules",
}

#: Leaf commands that render a status automatically, via `status_message=`
#: on their `@click.command`. Each does real work of at least a few seconds
#: with nothing interactive anywhere in its body.
AUTO_STATUS = {
    "cli status",
    "cli cost",
    "cli export",
    "cli stop",
    "cli upgrade",
    "cli doctor",
    "cli route export",
    "cli tokenmaxx",
    "cli backfill status",
    "cli backfill codex",
    "cli backfill langfuse",
    "cli backfill helicone",
    "cli backfill otlp",
    "cli report",
    "cli context",
    "cli quota-audit",
    "cli ping",
    "cli summarize list",
}

#: Leaf commands that call `tj_status` / `tj_status_stream` / `transient_status`
#: directly, scoped to just their silent stretch, instead of a class-level
#: `status_message` — because something else in the command body reads from
#: stdin (a `click.confirm`/`click.prompt`) and a live spinner active at the
#: same time would corrupt it. `status_message` must stay unset (falsy) on
#: all of these; the module-source check below is what proves each one
#: didn't just forget to wire anything.
MANUAL_STATUS = {
    "cli optimize",           # --validate has its own confirm() gate
    "cli uninstall",          # confirmation prompt before teardown
    "cli reset",              # confirmation prompt before teardown
    "cli onboard",            # interactive prompts throughout
    "cli summarize prep",     # multi-phase (wrap/rewrite/verify) — tj_status_stream
    "cli summarize calibrate",  # multi-phase (per-sample) — tj_status_stream
}

#: Structural opt-outs: `no_status=True`, commented at their own
#: `@click.command` about why a status must never be wired in, even later.
STRUCTURAL_OPT_OUT = {
    "cli mcp",           # stdio JSON-RPC — any stray stdout write corrupts it
    "cli serve",          # prints its own banner, then blocks forever
    "cli statusline",     # every-turn, must stay near-instant
    "cli session-end",    # best-effort, intentionally silent unless -v
}

#: Fast / config-only / already-self-narrating commands: a deliberate "no" —
#: considered, not forgotten. Most read a small cache or local config.
FAST_NO_STATUS = {
    "cli traces",
    "cli trace",
    "cli alerts",
    "cli tools",
    "cli budget",
    "cli backfill claude-code",  # already has its own backfill_progress counter
    "cli policy list",
    "cli policy decisions",
    "cli pricing list",
    "cli proxy enable",
    "cli proxy disable",
    "cli proxy killswitch",
    "cli proxy status",
    "cli session-story",
    "cli otel-resource-attrs",
    "cli loop annotate",
    "cli loop annotations",
    "cli loop expect",
    "cli loop expectations",
    "cli loop record",
    "cli loop history",
    "cli resume-brief",
    "cli relearn eval-case",
    "cli relearn list",
    "cli relearn apply",
    "cli relearn enable",
    "cli relearn revert",
    "cli relearn cost-proposals",
    "cli relearn cost-apply",
    "cli relearn cost-mark-applied",
    "cli relearn cost-revert",
    "cli drift",
    "cli demo",
    "cli summarize check",
    "cli summarize apply",
    "cli summarize undo",
    "cli summarize relocate",
    "cli summarize prune",
    "cli summarize expire",
    "cli summarize quarantine list",
    "cli summarize quarantine show",
    "cli summarize restore",
    "cli rules list",
    "cli rules show",
    "cli rules stage",
    "cli rules check",
    "cli rules apply",
    "cli rules undo",
    "cli rules applied",
    "cli rules dismiss",
    "cli rules undismiss",
}

#: Substrings proving a MANUAL_STATUS command's own module actually calls
#: into the shared primitive somewhere, rather than having its manual
#: placement quietly deleted in a later edit.
_MANUAL_STATUS_CALL_MARKERS = {
    "cli optimize": "tj_status(",
    "cli uninstall": "tj_status(",
    "cli reset": "tj_status(",
    "cli onboard": "transient_status(",
    "cli summarize prep": "tj_status_stream(",
    "cli summarize calibrate": "tj_status_stream(",
}


def _walk(cmd: click.Command, path: str = ""):
    full = f"{path} {cmd.name}".strip()
    yield full, cmd
    if isinstance(cmd, click.Group):
        for sub in cmd.commands.values():
            yield from _walk(sub, full)


def _all_commands() -> dict[str, click.Command]:
    return dict(_walk(cli))


def test_every_registered_command_routes_through_the_central_class():
    """No command may construct its own spinner: every leaf is a
    `TjCommand`, every group is a `TjGroup` — including one attached via
    `group.add_command(...)` rather than the `@group.command(...)` shortcut
    (which does NOT auto-inherit `command_class`)."""
    for path, cmd in _all_commands().items():
        if isinstance(cmd, click.Group):
            assert isinstance(cmd, TjGroup), (
                f"{path!r} is a click.Group but not a TjGroup — its "
                f"subcommands won't inherit TjCommand automatically."
            )
        else:
            assert isinstance(cmd, TjCommand), (
                f"{path!r} is registered as a plain click.Command — add "
                f"cls=TjCommand (and classify it in one of the buckets in "
                f"this file)."
            )


def test_status_buckets_partition_the_whole_registry_exactly():
    """The four buckets above, unioned, must equal the live command graph
    exactly — not a superset, not a subset. A new command that isn't added
    to one of them fails here, which is the whole point: it forces someone
    to decide whether it renders a status before it ships."""
    declared = GROUPS | AUTO_STATUS | MANUAL_STATUS | STRUCTURAL_OPT_OUT | FAST_NO_STATUS
    actual = set(_all_commands())

    missing_from_test = actual - declared
    assert not missing_from_test, (
        f"New command(s) not classified in test_cli_status_registry.py: "
        f"{sorted(missing_from_test)}. Add each to exactly one bucket."
    )
    stale_in_test = declared - actual
    assert not stale_in_test, (
        f"test_cli_status_registry.py names command(s) no longer registered: "
        f"{sorted(stale_in_test)}. Remove them from their bucket."
    )


def test_auto_status_commands_declare_a_message():
    commands = _all_commands()
    for path in AUTO_STATUS:
        cmd = commands[path]
        assert not isinstance(cmd, click.Group)
        assert cmd.status_message, f"{path!r} is in AUTO_STATUS but has no status_message"
        assert not cmd.no_status


def test_manual_status_commands_have_no_class_level_message_but_call_the_primitive():
    commands = _all_commands()
    for path in MANUAL_STATUS:
        cmd = commands[path]
        assert not cmd.status_message, (
            f"{path!r} is in MANUAL_STATUS (interactive prompt in its body) "
            f"but has a class-level status_message set — a live spinner "
            f"active for the whole command will collide with the prompt."
        )
        assert not cmd.no_status
        module = inspect.getmodule(cmd.callback)
        assert module is not None and module.__file__
        source = open(module.__file__, encoding="utf-8").read()
        marker = _MANUAL_STATUS_CALL_MARKERS[path]
        assert marker in source, (
            f"{path!r} is in MANUAL_STATUS but {module.__file__!r} has no "
            f"call to {marker!r} — the manual placement looks like it was "
            f"removed without updating this test."
        )


def test_structural_opt_out_commands_are_marked_and_silent():
    commands = _all_commands()
    for path in STRUCTURAL_OPT_OUT:
        cmd = commands[path]
        assert cmd.no_status is True, f"{path!r} must set no_status=True"
        assert not cmd.status_message


def test_fast_no_status_commands_render_nothing():
    commands = _all_commands()
    for path in FAST_NO_STATUS:
        cmd = commands[path]
        assert not cmd.status_message, f"{path!r} unexpectedly has a status_message"
        assert not cmd.no_status, (
            f"{path!r} sets no_status=True but isn't in STRUCTURAL_OPT_OUT — "
            f"move it there or drop the flag."
        )


def test_groups_never_set_a_status_message():
    """A group-level spinner would still be live when the chosen subcommand
    tried to open its own (see `TjGroup`'s docstring)."""
    commands = _all_commands()
    for path in GROUPS:
        cmd = commands[path]
        assert isinstance(cmd, click.Group)
        assert not getattr(cmd, "status_message", None), (
            f"{path!r} is a group with a status_message set — this will "
            f"raise a Rich LiveError the first time a subcommand runs."
        )
