"""Every command string these screens print must actually run.

`tj status`'s trailer and `tj optimize`'s Next block are the escape hatch from
a capped view: the cap is only defensible because the full data is one named
command away. So a next-step string that does not parse is worse than the
uncapped screen it replaced, and it fails silently — the renderer happily
prints any string, and the defect only surfaces when a user types it.

That is exactly what shipped: both screens advertised `-v`, which was declared
on the top-level group and NOT on the subcommand, so `tj status -v` and
`tj optimize -v` exited 2 with "No such option '-v'" while `tj -v status`
worked. Reading `ctx.obj["verbose"]` is not the same as accepting the flag.

These tests parse the command strings out of the RENDERED output rather than
from a hand-kept list, so new copy is covered the moment it is printed, and
check each one against the real Click command tree: the subcommand must
resolve, and every option and positional value in it must be one the command
actually declares. A rename can therefore no longer turn a call to action
back into a lie.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.optimize.types import OptimizeReport, WindowSummary

UTC = timezone.utc

#: A printed placeholder the user is expected to substitute (`<id>`).
_PLACEHOLDER = re.compile(r"^<[^>]+>$")

#: Command strings as they appear in prose: `tj` plus subcommand words,
#: options, and `<placeholder>` arguments, stopping at sentence punctuation.
_COMMAND = re.compile(r"\btj(?: (?:[a-z][a-z0-9-]*|--?[a-z][a-z0-9-]*|<[^>]+>))+")


def advertised_commands(rendered: str) -> list[str]:
    """Every `tj ...` command string in a rendered screen, de-duplicated.

    Scanned line by line, and a single space is the only token separator: the
    Next block puts its hint in a second column padded with several spaces, so
    a run of whitespace is what ends a command. Prose that continues past a
    command on the SAME line with single spaces will be over-captured and fail
    loudly, which is the safe direction for a guard to be wrong in.
    """
    found: list[str] = []
    for line in rendered.splitlines():
        for match in _COMMAND.finditer(line):
            command = match.group(0).rstrip(".")
            if command not in found:
                found.append(command)
    return found


def assert_invocable(command: str) -> None:
    """Fail unless Click parses `command` as a real command with real options.

    Placeholders are resolved the way a user would: an option's placeholder
    value becomes a dummy string, a positional placeholder is dropped (its
    value is the user's choice, and a wrong one is a different error than the
    one under test). `--help` short-circuits before any DB or analyzer work,
    so this checks the PARSE, which is the thing that was broken.
    """
    tokens = command.split()[1:]  # drop the `tj`
    resolved: list[str] = []
    for i, token in enumerate(tokens):
        if not _PLACEHOLDER.match(token):
            resolved.append(token)
            continue
        if i and tokens[i - 1].startswith("-"):
            resolved.append("x")  # a value for the preceding option

    result = CliRunner().invoke(cli, [*resolved, "--help"])
    assert result.exit_code == 0, (
        f"advertised command does not parse: `{command}` "
        f"(ran: tj {' '.join(resolved)} --help) -> {result.output}"
    )


# --------------------------------------------------------------------------- #
# The guard itself has to be able to fail
# --------------------------------------------------------------------------- #

def test_the_guard_rejects_an_option_the_command_does_not_declare():
    """`--help` is eager, so this pins that it does NOT mask a bad option —
    without which every assertion below would pass vacuously."""
    with pytest.raises(AssertionError):
        assert_invocable("tj status --no-such-option")


def test_the_guard_rejects_a_subcommand_that_does_not_exist():
    with pytest.raises(AssertionError):
        assert_invocable("tj no-such-command")


def test_the_guard_finds_commands_in_prose():
    found = advertised_commands(
        "  +294 more agents. See one with tj status --agent <id>.\n"
        "  All of them: tj status -v.\n"
    )
    assert found == ["tj status --agent <id>", "tj status -v"]


# --------------------------------------------------------------------------- #
# The screens
# --------------------------------------------------------------------------- #

def test_every_command_the_status_overview_advertises_runs(capsys):
    from tokenjam.cli.cmd_status import _print_overview

    entries = []
    for i in range(25):
        entries.append((
            {
                "agent_id": f"agent-{i:02d}", "status": "completed",
                "session_id": None, "cost_today": 1.0, "daily_limit": None,
                "input_tokens": 0, "output_tokens": 0, "tool_call_count": 0,
                "error_count": 0, "active_alerts": 0,
                "duration_seconds": None, "active_seconds": None,
            },
            [],
            SimpleNamespace(started_at=datetime(2026, 5, 1, tzinfo=UTC), ended_at=None),
        ))
    _print_overview(entries)
    rendered = capsys.readouterr().out

    commands = advertised_commands(rendered)
    assert "tj status --agent <id>" in commands
    assert "tj status -v" in commands
    for command in commands:
        assert_invocable(command)


def test_every_command_the_optimize_scoreboard_advertises_runs(capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    since, until = datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 5, 30, tzinfo=UTC)
    report = OptimizeReport(
        window=WindowSummary(
            since=since, until=until, days=29.0, sessions=12, spans=100,
            total_tokens=5_000_000, total_cost_usd=120.0, thin_data=False,
        ),
        findings={
            "summarize": SimpleNamespace(
                candidates=[object()],
                past_overspend_usd=48.55, past_overspend_tokens=9_000_000,
            ),
            "deadweight": SimpleNamespace(
                unused_servers=["some-mcp"],
                past_overspend_usd=296.74, past_overspend_tokens=1_000,
            ),
        },
    )
    _render_scoreboard(report, agent=None, pricing_mode="api", cost_proposal_count=3)
    rendered = capsys.readouterr().out

    commands = advertised_commands(rendered)
    assert "tj optimize -v" in commands
    assert "tj optimize deadweight" in commands
    assert "tj relearn cost-proposals" in commands
    for command in commands:
        assert_invocable(command)


@pytest.mark.parametrize("command", [
    "tj status -v",
    "tj -v status",
    "tj status --agent <id>",
    "tj optimize -v",
    "tj -v optimize",
    "tj optimize <analyzer>",
    "tj relearn cost-proposals",
])
def test_both_verbose_positions_and_every_drill_down_parse(command):
    """`-v` is accepted on the group AND on the subcommand, so whichever the
    user types works. Both are booleans meaning the same thing; the command
    ORs them, so there is no precedence question."""
    assert_invocable(command)
