"""Regression coverage for the global `tj --json <cmd>` flag.

The root `tj` group defines a `--json` option (`tj --json status`), but until
now most subcommands only honored their OWN local `--json` (`tj status
--json`) — they read a local flag straight into their render branch and never
looked at `ctx.obj["output_json"]`, so the two spellings silently diverged:
one printed Rich text, the other JSON. `tokenjam/cli/json_option.py` fixes
this with a shared `json_option` decorator + `resolve_output_json()` helper
that ORs the local flag with the global one; every command below is wired
through it (or, for the handful with no local flag at all, reads
`ctx.obj["output_json"]` directly).

This file walks every registered command that supports JSON output and
asserts `tj --json <cmd>` and `tj <cmd> --json` are equivalent: same exit
code, and both stdouts parse as JSON (proving the global spelling actually
renders the JSON branch, not the human-text one — the exact failure mode of
the bug this locks in). That single parametrized walk is the regression
test; the rest of the suite (test_cli.py, test_cmd_policy.py, test_ping.py,
test_demos.py, ...) covers each command's actual JSON *content*.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from tokenjam.cli.main import cli
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.models import AgentRecord
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config():
    return TjConfig(version="1")


def _no_setup(db: InMemoryBackend) -> list[str]:
    """No seeding needed; the command must render its empty/no-data JSON
    branch cleanly for this test to be meaningful."""
    return []


def _seed_trace(db: InMemoryBackend) -> list[str]:
    """One span so `tj trace <trace_id>` has something to resolve, returning
    the trace_id as the extra positional arg."""
    span = make_llm_span(agent_id="test-agent")
    db.upsert_agent(AgentRecord(
        agent_id="test-agent", first_seen=utcnow(), last_seen=utcnow(),
    ))
    db.upsert_session(make_session(agent_id="test-agent"))
    db.insert_span(span)
    return [span.trace_id]


# (subcommand path, setup callable returning extra args appended before --json)
JSON_COMMAND_CASES: list[tuple[str, tuple[str, ...], Callable[[InMemoryBackend], list[str]]]] = [
    ("status", (), _no_setup),
    ("cost", (), _no_setup),
    ("alerts", (), _no_setup),
    ("tools", (), _no_setup),
    ("traces", (), _no_setup),
    ("trace", (), _seed_trace),
    ("drift", (), _no_setup),
    ("budget", (), _no_setup),
    ("context", (), _no_setup),
    ("tokenmaxx", (), _no_setup),
    ("quota-audit", (), _no_setup),
    ("optimize", (), _no_setup),
    ("session-story", (), _no_setup),
    ("policy", ("list",), _no_setup),
    ("pricing", ("list",), _no_setup),
    ("summarize", ("list",), _no_setup),
    ("route", ("export", "--check"), _no_setup),
    ("demo", ("retry-loop",), _no_setup),
]


@pytest.mark.parametrize(
    "root, base_args, setup",
    JSON_COMMAND_CASES,
    ids=[c[0] if not c[1] else f"{c[0]}-{'-'.join(c[1])}" for c in JSON_COMMAND_CASES],
)
def test_global_json_flag_matches_local_json_flag(
    runner, db, config, root, base_args, setup,
):
    """`tj --json <cmd>` must render the same JSON branch as `tj <cmd> --json`.

    A regression here means the global flag went back to being silently
    dropped for this command — the exact bug class this test locks in.
    """
    extra = setup(db)
    argv = [root, *base_args, *extra]

    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db):
        global_result = runner.invoke(cli, ["--json", *argv])
        local_result = runner.invoke(cli, [*argv, "--json"])

    assert global_result.exit_code == local_result.exit_code, (
        f"tj --json {' '.join(argv)} exited {global_result.exit_code} but "
        f"tj {' '.join(argv)} --json exited {local_result.exit_code}:\n"
        f"global: {global_result.output}\nlocal: {local_result.output}"
    )
    # The core assertion: both must actually be JSON. If the global flag were
    # silently ignored, `global_result.output` would be Rich human text and
    # this json.loads would raise.
    global_payload = json.loads(global_result.output)
    local_payload = json.loads(local_result.output)

    # Same shape: for dict payloads, the same top-level keys came back either
    # way (a handful of commands embed a fresh `utcnow()` timestamp per call,
    # so full deep-equality isn't reliable — the key set is).
    if isinstance(global_payload, dict) and isinstance(local_payload, dict):
        assert set(global_payload) == set(local_payload)
    else:
        assert type(global_payload) is type(local_payload)


def test_global_json_flag_matches_local_json_flag_for_ping(monkeypatch):
    """`ping` needs its own harness (a real SDK TracerProvider + stubbed
    bootstrap/delivery-confirmation) — mirrors test_ping.py's `_run` helper,
    so it's kept out of the main parametrize above rather than forcing every
    case through ping's setup."""
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        trace.set_tracer_provider(TracerProvider())

    monkeypatch.setattr("tokenjam.sdk.bootstrap.ensure_initialised", lambda: None)
    monkeypatch.setattr("tokenjam.sdk.bootstrap.get_mode", lambda: "http")
    monkeypatch.setattr(
        "tokenjam.cli.cmd_ping._confirm_delivery", lambda *a, **k: (True, None)
    )

    runner = CliRunner()
    global_result = runner.invoke(cli, ["--json", "ping"])
    local_result = runner.invoke(cli, ["ping", "--json"])

    assert global_result.exit_code == local_result.exit_code == 0
    global_payload = json.loads(global_result.output)
    local_payload = json.loads(local_result.output)
    assert set(global_payload) == set(local_payload)
    assert global_payload["delivery_mode"] == local_payload["delivery_mode"] == "http"


def test_command_with_no_json_support_rejects_json_flag(runner, db, config):
    """`tj stop` never claimed --json support; the fix must not accidentally
    add a --json option to commands that don't render one."""
    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db):
        result = runner.invoke(cli, ["stop", "--json"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()
