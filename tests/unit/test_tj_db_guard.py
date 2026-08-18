"""Regression tests for the shared DB-required guard (#727)."""
from __future__ import annotations

import click
import pytest

from tokenjam.cli.tj_status import TjCommand, TjGroup, db_required_message


def test_db_required_message_names_command() -> None:
    msg = db_required_message("annotate")
    assert "annotate requires either a direct DuckDB connection" in msg
    assert "api.{host,port}" in msg


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


def test_tj_group_skips_db_guard() -> None:
    group = TjGroup("demo-group")
    ctx = click.Context(group)
    ctx.obj = {"db": None, "requires_db": True}
    group._ensure_db_available(ctx)
