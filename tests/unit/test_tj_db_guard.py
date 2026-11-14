"""Regression tests for the shared DB-required guard (#727, #733)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_backfill import cmd_backfill
from tokenjam.cli.cmd_context import cmd_context
from tokenjam.cli.cmd_optimize import cmd_optimize
from tokenjam.cli.cmd_quota_audit import cmd_quota_audit
from tokenjam.cli.cmd_report import cmd_report
from tokenjam.cli.cmd_route import _resolve_finding, cmd_route
from tokenjam.cli.cmd_session_story import _run_via_serve, cmd_session_story
from tokenjam.cli.cmd_tokenmaxx import cmd_tokenmaxx
from tokenjam.cli.tj_status import (
    TjCommand,
    TjGroup,
    config_required_message,
    db_required_message,
    help_consumed_as_option_value_message,
    resolve_command_path,
)
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend


def test_db_required_message_names_command_path() -> None:
    msg = db_required_message("tj loop annotate")
    assert "tj loop annotate requires either a direct DuckDB connection" in msg
    assert "api.{host,port}" in msg


def test_help_consumed_as_option_value_message_names_command_path() -> None:
    msg = help_consumed_as_option_value_message("tj loop annotate")
    assert "tj loop annotate cannot run" in msg
    assert "`--help` was consumed as an option value" in msg
    assert "Give that option a value" in msg
    assert "Put `--help` before other flags" not in msg


def test_tj_command_guard_raises_click_exception_not_traceback() -> None:
    @click.command("demo-cmd", cls=TjCommand)
    def demo_cmd() -> None:
        raise AssertionError("must not reach command body")

    ctx = click.Context(demo_cmd)
    ctx.obj = {"db": None, "requires_db": True}
    with pytest.raises(click.ClickException) as exc:
        demo_cmd.invoke(ctx)
    assert "demo-cmd requires either a direct DuckDB connection" in str(exc.value)
    assert "AssertionError" not in str(exc.value)


def test_tj_command_guard_help_skip_message() -> None:
    @click.command("demo-cmd", cls=TjCommand)
    def demo_cmd() -> None:
        raise AssertionError("must not reach command body")

    ctx = click.Context(demo_cmd)
    ctx.obj = {"db": None, "requires_db": True, "_skip_db_for_help": True}
    with pytest.raises(click.ClickException) as exc:
        demo_cmd.invoke(ctx)
    assert "`--help` was consumed as an option value" in str(exc.value)
    assert "direct DuckDB connection" not in str(exc.value)


def test_tj_command_guard_fail_open_without_requires_db_key() -> None:
    """``requires_db`` absent from ``ctx.obj`` must not fail-closed (#729)."""
    @click.command("demo-cmd", cls=TjCommand)
    def demo_cmd() -> None:
        return "ok"

    ctx = click.Context(demo_cmd)
    ctx.obj = {"db": None}
    assert demo_cmd.invoke(ctx) == "ok"


def test_tj_command_guard_allows_live_backend() -> None:
    """A command with a non-None db passes the guard (regression for is-not-None check)."""
    @click.command("demo-cmd", cls=TjCommand)
    def demo_cmd() -> None:
        return "ok"

    ctx = click.Context(demo_cmd)
    ctx.obj = {"db": MagicMock(), "requires_db": True}
    # Guard runs inside invoke; should not raise before the body.
    assert demo_cmd.invoke(ctx) == "ok"


def test_tj_group_skips_db_guard() -> None:
    # Groups never enforce the leaf guard — only subcommand leaves do.
    group = TjGroup("demo-group")
    ctx = click.Context(group)
    ctx.obj = {"db": None, "requires_db": True}
    group._ensure_db_available(ctx)  # does not raise


def test_config_required_message_names_command_path() -> None:
    assert config_required_message("tj route export") == (
        "tj route export could not load configuration."
    )


def test_resolve_command_path_handles_direct_contexts_and_fallbacks() -> None:
    group = TjGroup("route")

    @group.command("export")
    def export_cmd() -> None:
        pass

    # Explicit info_name → ctx.command_path wins.
    parent_named = click.Context(group, info_name="tj route")
    leaf_named = click.Context(export_cmd, info_name="export", parent=parent_named)
    assert resolve_command_path(leaf_named) == "tj route export"

    # Bare click.Context(cmd) without info_name → walks command.name chain.
    parent_bare = click.Context(group)
    leaf_bare = click.Context(export_cmd, parent=parent_bare)
    assert resolve_command_path(leaf_bare) == "route export"

    # Non-Click stand-in or None → fallback.
    assert resolve_command_path(SimpleNamespace(obj={}), "custom") == "custom"
    assert resolve_command_path(None) == "this command"


@pytest.mark.parametrize(
    ("cmd", "args", "expected_path"),
    [
        (cmd_backfill, ["claude-code"], "backfill claude-code"),
        (cmd_backfill, ["status"], "backfill status"),
        (cmd_backfill, ["codex"], "backfill codex"),
        (cmd_backfill, ["langfuse", "--source-url", "https://example.com"], "backfill langfuse"),
        (cmd_backfill, ["helicone", "--source-url", "https://example.com"], "backfill helicone"),
        (cmd_backfill, ["otlp", "--source-url", "https://example.com"], "backfill otlp"),
        (cmd_report, ["--trim", "--no-open"], "report"),
        (cmd_report, ["--reuse", "--no-open"], "report"),
        (cmd_route, ["export", "--target", "ccr"], "route export"),
        (cmd_context, [], "context"),
        (cmd_quota_audit, [], "quota-audit"),
        (cmd_session_story, [], "session-story"),
        (cmd_tokenmaxx, [], "tokenmaxx"),
    ],
)
def test_commands_bypassing_leaf_guard_emit_shared_db_required_message(
    cmd: click.Command, args: list[str], expected_path: str
) -> None:
    """When invoked without ``requires_db=True`` in ``ctx.obj`` (so
    ``TjCommand._ensure_db_available`` fails open), every command's internal
    ``db is None`` guard still routes through ``db_required_message`` (#733)."""
    runner = CliRunner()
    result = runner.invoke(
        cmd,
        args,
        obj={"db": None, "config": TjConfig(version="1")},
    )
    assert result.exit_code == 1, result.output
    assert db_required_message(expected_path) in result.output
    assert "requires a database connection" not in result.output


def test_non_api_backend_fallbacks_emit_shared_db_required_message() -> None:
    """Dual-path commands whose ``conn is None`` branch checks ``ApiBackend``
    must also emit ``db_required_message`` rather than bespoke copies (#733)."""
    class _UnusableBackend:
        pass

    cfg = TjConfig(version="1")
    runner = CliRunner()

    for cmd, args, expected_path in [
        (cmd_route, ["export", "--target", "ccr"], "route export"),
        (cmd_report, ["--reuse", "--no-open"], "report"),
        (cmd_session_story, [], "session-story"),
        (cmd_optimize, [], "optimize"),
    ]:
        result = runner.invoke(cmd, args, obj={"db": _UnusableBackend(), "config": cfg})
        assert result.exit_code == 1, result.output
        assert db_required_message(expected_path) in result.output

    # Direct helper call on cmd_route._resolve_finding and cmd_session_story._run_via_serve
    route_ctx = click.Context(cmd_route, info_name="tj route")
    route_ctx.obj = {"db": _UnusableBackend(), "config": cfg}
    with pytest.raises(click.ClickException) as exc:
        _resolve_finding(route_ctx, agent=None, since="7d")
    assert str(exc.value) == db_required_message("tj route")

    with pytest.raises(click.ClickException) as exc:
        _run_via_serve(_UnusableBackend(), session_id=None, output_json=False)
    assert str(exc.value) == db_required_message("tj session-story")


@pytest.mark.parametrize(
    ("cmd", "args", "expected_path"),
    [
        (cmd_route, ["export", "--target", "ccr"], "route export"),
        (cmd_context, [], "context"),
        (cmd_report, ["--trim", "--no-open"], "report"),
        (cmd_report, ["--reuse", "--no-open"], "report"),
        (cmd_quota_audit, [], "quota-audit"),
        (cmd_tokenmaxx, [], "tokenmaxx"),
        (cmd_optimize, [], "optimize"),
    ],
)
def test_config_none_reports_configuration_error_not_db_error(
    cmd: click.Command, args: list[str], expected_path: str
) -> None:
    """A ``config is None`` failure must report a configuration error rather
    than blaming the database connection (#733)."""
    db = InMemoryBackend()
    try:
        runner = CliRunner()
        result = runner.invoke(
            cmd,
            args,
            obj={"db": db, "config": None, "requires_db": True},
        )
        assert result.exit_code == 1, result.output
        assert config_required_message(expected_path) in result.output
        assert "database" not in result.output.lower()
        assert "duckdb" not in result.output.lower()
    finally:
        db.close()
