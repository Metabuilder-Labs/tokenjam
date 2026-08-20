"""Regression tests for the shared DB-required guard (#727)."""
from __future__ import annotations

from unittest.mock import MagicMock

import click
import pytest

from tokenjam.cli.tj_status import (
    TjCommand,
    TjGroup,
    db_required_message,
    help_consumed_as_option_value_message,
)


def test_db_required_message_names_command() -> None:
    msg = db_required_message("annotate")
    assert "annotate requires either a direct DuckDB connection" in msg
    assert "api.{host,port}" in msg


def test_help_consumed_as_option_value_message_names_command() -> None:
    msg = help_consumed_as_option_value_message("annotate")
    assert "annotate cannot run" in msg
    assert "`--help` was consumed as an option value" in msg


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
