"""`tj init --cloud` and the surfaces that report the bridge (ledger W5;
contracts §6, §9; Critical Rules 20, 44).

The command is exercised through Click against an isolated HOME, a real
config file and a real DuckDB store, with a recorded fake Cloud behind
`httpx.MockTransport` injected at the one seam (`cloud_sync.CloudClient`),
so the emission list, the confirmation, the config write, the state file
and the initial push are all the real path. The daemon stop / restart
helpers are stubbed: a unit test never touches launchd.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from click.testing import CliRunner

import tokenjam.core.config as cfg_mod
from tests.factories import make_llm_span, make_session
from tests.unit.test_advertised_commands_are_invocable import (
    advertised_commands,
    assert_invocable,
)
from tokenjam.cli.cmd_onboard import cmd_onboard
from tokenjam.cli.main import cli
from tokenjam.core import cloud_sync as cs
from tokenjam.core.config import (
    CloudConfig,
    ProviderBudget,
    StorageConfig,
    TjConfig,
    load_config,
    write_config,
)
from tokenjam.core.db import DuckDBBackend

KEY = "tj_live_" + "k" * 43
ORG = "org_abc123"
T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    # The daemon helpers: never launchd from a test.
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._daemon_already_running", lambda: False)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._restart_tj_server",
                        lambda *a, **k: "restarted (stub)")
    return h


@pytest.fixture
def config_path(home, monkeypatch) -> Path:
    path = home / ".config" / "tj" / "config.toml"
    path.parent.mkdir(parents=True)
    write_config(TjConfig(
        version="1", budgets={"anthropic": ProviderBudget(plan="max_20x")},
        storage=StorageConfig(path=str(home / "t.duckdb")),
    ), path)
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [Path(".tj/config.toml"), path])
    return path


@pytest.fixture
def seeded(config_path) -> Path:
    db = DuckDBBackend(load_config(str(config_path)).storage)
    db.upsert_session(replace(
        make_session(session_id="s1", agent_id="claude-code-w", started_at=T0, total_cost_usd=2.0),
        ended_at=T0 + timedelta(minutes=5), repo_remote="https://github.com/Acme/w",
        branch_start="main", user_email="dev@example.com", developer_id="deadbeef00000000",
    ))
    for i in range(3):
        db.insert_span(make_llm_span(agent_id="claude-code-w", session_id="s1",
                                     start_time=T0 + timedelta(seconds=i),
                                     extra_attributes={"gen_ai.prompt.content": "SECRET"}))
    db.close()
    return config_path


class FakeCloud:
    def __init__(self, accept: str = KEY):
        self.requests: list[httpx.Request] = []
        self.accept = accept

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("Authorization") != f"Bearer {self.accept}":
            return httpx.Response(401, json={"detail": "Invalid organization or ingest key"})
        return httpx.Response(200, json={"ingested": 1, "rejected": 0})


@pytest.fixture
def cloud(monkeypatch) -> FakeCloud:
    fake = FakeCloud()
    real = cs.CloudClient
    monkeypatch.setattr(cs, "CloudClient", lambda cfg, transport=None, **kw: real(
        cfg, transport=transport or httpx.MockTransport(fake.handler), sleep=lambda _s: None,
        **{k: v for k, v in kw.items() if k != "sleep"},
    ))
    return fake


def _init(*args: str, input: str | None = None):
    return CliRunner().invoke(cmd_onboard, list(args), obj={}, input=input)


# --- Invocable (Critical Rule 44) ------------------------------------------------------

def test_tj_init_cloud_flags_parse():
    for args in (["init", "--cloud", KEY, "--org", ORG], ["init", "--cloud", "off"],
                 ["init", "--cloud", KEY, "--org", ORG, "--cloud-endpoint", "http://x", "--yes"],
                 ["onboard", "--cloud", KEY, "--org", ORG, "-y"], ["init", "--hooks", "--cloud", KEY]):
        result = CliRunner().invoke(cli, [*args, "--help"])
        assert result.exit_code == 0, (args, result.output)


def test_every_command_the_cloud_surfaces_advertise_runs(tmp_path):
    from tokenjam.cli.ledger_cloud import cloud_summary_line

    lines = [cloud_summary_line(None), cloud_summary_line(TjConfig(version="1"))]
    config = TjConfig(version="1", cloud=CloudConfig(org_id=ORG, ingest_key=KEY, enabled=False))
    lines.append(cs.status_line(config, state_file=tmp_path / "s.json"))
    state = cs.SyncState(disabled_reason="Cloud rejected the ingest key (401). Copy a current key "
                                         "from Cloud's Connect screen, then: tj init --cloud <key> --org <org>.")
    cs.save_state(state, tmp_path / "s.json")
    config.cloud.enabled = True
    lines.append(cs.status_line(config, state_file=tmp_path / "s.json"))
    commands = advertised_commands("\n".join(line for line in lines if line))
    assert "tj init --cloud <key> --org <org>" in commands
    for command in commands:
        assert_invocable(command)


# --- The command --------------------------------------------------------------------------

def test_emission_list_is_printed_and_nothing_leaves_or_is_written_without_consent(seeded, cloud):
    result = _init("--cloud", KEY, "--org", ORG, input="n\n")
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "What leaves this machine when Cloud forwarding is on" in out
    for item in cs.EMISSION_LEAVES:
        assert item in out
    assert "What never leaves by default" in out
    for item in cs.EMISSION_NEVER_BY_DEFAULT:
        assert item in out
    assert "forward_content = false" in out
    assert "Connect and forward?" in out and "Nothing written" in out
    assert cloud.requests == []
    assert "[cloud]" not in seeded.read_text()
    assert not cs.state_path().exists()
    # The whole key is never echoed.
    assert KEY not in result.output and "tj_live_kkkk" in result.output


def test_yes_writes_the_block_prints_the_list_first_and_pushes_history(seeded, cloud):
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    # The list precedes the first byte: printed before the config write line.
    assert out.index("What leaves this machine") < out.index("[cloud] written to")
    assert "Forwarded 3 spans, 1 sessions, 0 commits" in out
    assert "every 5 minutes" in out

    cloud_cfg = load_config(str(seeded)).cloud
    assert cloud_cfg == CloudConfig(enabled=True, endpoint=CloudConfig.endpoint,
                                    org_id=ORG, ingest_key=KEY, forward_content=False)
    state = cs.load_state()
    assert (state.spans_sent, state.sessions_sent, state.org_id) == (3, 1, ORG)
    paths = [r.url.path for r in cloud.requests]
    assert paths == [cs.LEDGER_SESSIONS_PATH, cs.SPANS_PATH]
    for r in cloud.requests:
        assert r.url.host == "tokenjam-cloud-api.onrender.com"
        assert r.headers["X-TokenJam-Org"] == ORG
        assert r.headers["Authorization"] == f"Bearer {KEY}"
    assert b"SECRET" not in b"".join(r.content for r in cloud.requests)
    for command in advertised_commands(result.output):
        assert_invocable(command)

    # Again: idempotent (same org keeps the marks; nothing re-sent).
    cloud.requests.clear()
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert "Forwarded 0 spans, 0 sessions, 0 commits" in _flat(result.output)


def test_org_is_required_unless_the_config_already_names_it(seeded, cloud):
    result = _init("--cloud", KEY, "--yes")
    assert result.exit_code == 2 and "--org" in result.output
    assert "[cloud]" not in seeded.read_text() and cloud.requests == []

    _init("--cloud", KEY, "--org", ORG, "--yes")
    cloud.requests.clear()
    new_key = "tj_live_" + "n" * 43
    cloud.accept = new_key
    result = _init("--cloud", new_key, "--yes", "--cloud-endpoint", "https://staging.test")
    assert result.exit_code == 0, result.output
    back = load_config(str(seeded)).cloud
    assert back.org_id == ORG and back.ingest_key == new_key and back.endpoint == "https://staging.test"
    assert all(r.url.host == "staging.test" for r in cloud.requests)


def test_a_tracked_config_never_receives_the_key(seeded, cloud, tmp_path, monkeypatch):
    """Critical Rule 20: the block carries a live secret, so a config git
    tracks is refused rather than written."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    tracked = repo / "tokenjam.toml"
    write_config(TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "x.duckdb"))), tracked)
    subprocess.run(["git", "add", "tokenjam.toml"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "x"],
                   cwd=repo, check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [Path("tokenjam.toml"), seeded])
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    assert "tracked by git" in _flat(result.output)
    assert "[cloud]" not in tracked.read_text() and KEY not in tracked.read_text()
    assert cloud.requests == []
    # An untracked sibling in the same repo is fine.
    untracked = repo / ".tj" / "config.toml"
    write_config(TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "x.duckdb"))), untracked)
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [Path(".tj/config.toml"), seeded])
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert "[cloud] written to" in _flat(result.output)
    assert KEY in untracked.read_text()


def test_a_malformed_key_is_refused_before_anything_happens(seeded, cloud):
    result = _init("--cloud", "sk-ant-not-a-tj-key", "--org", ORG, "--yes")
    assert result.exit_code == 2 and "tj_live_" in result.output
    assert "[cloud]" not in seeded.read_text() and cloud.requests == []


def test_a_rejected_key_disables_the_bridge_and_every_surface_says_so(seeded, cloud, monkeypatch):
    bad = "tj_live_" + "b" * 43
    result = _init("--cloud", bad, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    assert "Cloud rejected the key" in _flat(result.output)
    assert len(cloud.requests) == 1
    state = cs.load_state()
    assert state.disabled_reason and "401" in state.disabled_reason

    # tj status: the reason, in prose and in --json.
    from tokenjam.core.db import open_db

    config = load_config(str(seeded))
    db = open_db(config.storage)
    try:
        with_db = {"config": config, "db": db, "config_path_override": None}
        result = CliRunner().invoke(cli, ["--config", str(seeded), "status"], obj=with_db)
        assert "Cloud: disabled: Cloud rejected the ingest key" in _flat(result.output)
        result = CliRunner().invoke(cli, ["--config", str(seeded), "status", "--json"], obj=with_db)
        payload = json.loads(result.output)
        assert payload["cloud"]["state"] == "disabled_by_receiver"
        assert "ingest_key" not in payload["cloud"]
    finally:
        db.close()

    # tj doctor: a warning naming the fix, and no probe while disabled.
    from tokenjam.cli.cmd_doctor import _check_cloud_bridge

    cloud.requests.clear()
    check = _check_cloud_bridge(config)
    assert check["level"] == "warning" and "tj init --cloud" in check["message"]
    assert cloud.requests == []

    # A fresh key clears it and forwarding resumes.
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert "Forwarded 3 spans, 1 sessions" in _flat(result.output)
    assert cs.load_state().disabled_reason is None


def test_cloud_off_keeps_the_key_and_turns_forwarding_off(seeded, cloud):
    _init("--cloud", KEY, "--org", ORG, "--yes")
    cloud.requests.clear()
    result = _init("--cloud", "off")
    assert result.exit_code == 0, result.output
    assert "Cloud forwarding off" in _flat(result.output)
    back = load_config(str(seeded)).cloud
    assert back.enabled is False and back.ingest_key == KEY and back.configured
    assert not back.is_active
    # The forwarder honours it: a daemon pass sends nothing.
    db = DuckDBBackend(load_config(str(seeded)).storage)
    try:
        report = cs.run_sync(db, load_config(str(seeded)))
    finally:
        db.close()
    assert report.skipped_reason and cloud.requests == []
    assert cs.status_line(load_config(str(seeded))).startswith("Cloud: off")
    # And again is a no-op that says so.
    assert "already off" in _init("--cloud", "off").output


def test_cloud_off_without_a_block_says_so(config_path, cloud):
    result = _init("--cloud", "off")
    assert result.exit_code == 0 and "never configured" in result.output
    assert cloud.requests == []


def test_without_a_config_the_wizard_runs_first(home, monkeypatch, cloud):
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [Path(".tj/config.toml"), home / "nope.toml"])
    wizard = MagicMock(return_value=None)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._run_onboard_wizard", wizard)
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    wizard.assert_called_once()
    assert "no tj config found" in _flat(result.output)
    assert cloud.requests == []


def test_with_a_config_the_wizard_is_skipped(seeded, cloud, monkeypatch):
    wizard = MagicMock()
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._run_onboard_wizard", wizard)
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    wizard.assert_not_called()


def test_the_running_daemon_is_stopped_for_the_push_and_restarted(seeded, cloud, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._stop_serve_for_db_write",
                        lambda: calls.append("stop") or True)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._restart_tj_server",
                        lambda path, no_daemon, **kw: calls.append(f"restart:{Path(path).name}") or "restarted")
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    assert calls == ["stop", "restart:config.toml"]
    assert "Daemon: restarted" in _flat(result.output)


# --- Surfaces ------------------------------------------------------------------------------

def test_status_and_init_summary_show_the_cloud_line(seeded, cloud, capsys):
    from tokenjam.cli.cmd_onboard import _print_setup_complete_home
    from tokenjam.core.db import open_db

    _init("--cloud", KEY, "--org", ORG, "--yes")
    config = load_config(str(seeded))
    db = open_db(config.storage)
    try:
        obj = {"config": config, "db": db, "config_path_override": None}
        result = CliRunner().invoke(cli, ["--config", str(seeded), "status"], obj=obj)
        out = _flat(result.output)
        assert "Cloud: connected · 3 spans, 1 sessions, 0 commits sent · last just now" in out
        result = CliRunner().invoke(cli, ["--config", str(seeded), "status", "--json"], obj=obj)
        assert json.loads(result.output)["cloud"]["state"] == "connected"
    finally:
        db.close()

    _print_setup_complete_home()
    rendered = _flat(capsys.readouterr().out)
    assert "Cloud: connected · 3 spans, 1 sessions, 0 commits sent" in rendered


def test_status_is_silent_about_cloud_on_a_machine_that_never_opted_in(config_path):
    from tokenjam.core.db import open_db

    config = load_config(str(config_path))
    db = open_db(config.storage)
    try:
        obj = {"config": config, "db": db, "config_path_override": None}
        result = CliRunner().invoke(cli, ["--config", str(config_path), "status"], obj=obj)
        assert "Cloud:" not in result.output
        result = CliRunner().invoke(cli, ["--config", str(config_path), "status", "--json"], obj=obj)
        assert json.loads(result.output)["cloud"] == {
            "configured": False, "enabled": False, "state": "not_configured"}
    finally:
        db.close()


def test_doctor_probes_reachability_and_the_key(seeded, cloud):
    from tokenjam.cli.cmd_doctor import _check_cloud_bridge

    assert _check_cloud_bridge(load_config(str(seeded)))["level"] == "info"
    _init("--cloud", KEY, "--org", ORG, "--yes")
    cloud.requests.clear()
    config = load_config(str(seeded))

    ok = _check_cloud_bridge(config, transport=httpx.MockTransport(cloud.handler))
    assert ok["level"] == "ok" and "3 spans, 1 sessions" in ok["message"]
    assert json.loads(cloud.requests[-1].content) == {"resourceSpans": []}

    config.cloud.ingest_key = "tj_live_" + "w" * 43
    bad = _check_cloud_bridge(config, transport=httpx.MockTransport(cloud.handler))
    assert bad["level"] == "warning" and "rejected the ingest key" in bad["message"]

    def _down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = _check_cloud_bridge(config, transport=httpx.MockTransport(_down))
    assert down["level"] == "warning" and "unreachable" in down["message"]

    config.cloud.enabled = False
    assert _check_cloud_bridge(config)["level"] == "info"


# --- The first push on a fresh machine (issue #770) --------------------------------------

def _ledger_bodies(cloud: FakeCloud) -> list[dict]:
    return [json.loads(r.content) for r in cloud.requests if r.url.path == cs.LEDGER_SESSIONS_PATH]


@pytest.fixture
def fresh_machine(config_path, monkeypatch, tmp_path):
    """What the first real run looked like: a session ingested by a daemon
    that never derived repo context (nulls), its transcript still on disk in
    a real checkout, and the commit its Bash tool made. No install id yet."""
    import subprocess

    from tests.ledger_fixtures import T0 as START
    from tests.ledger_fixtures import git_repo, write_transcript
    from tests.factories import make_tool_span

    repo = next(git_repo(tmp_path, monkeypatch))
    # `git_repo` re-points HOME under tmp_path; the config fixture's home is
    # what `tj init` must keep using.
    home = config_path.parents[2]
    monkeypatch.setenv("HOME", str(home))
    projects = home / ".claude" / "projects"
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(projects))
    write_transcript(projects, "live-1", str(repo), at=START)

    db = DuckDBBackend(load_config(str(config_path)).storage)
    db.upsert_session(replace(
        make_session(session_id="live-1", agent_id="claude-code-widgets", started_at=START,
                     total_cost_usd=2.0, input_tokens=200, output_tokens=40, tool_call_count=1),
        ended_at=START + timedelta(minutes=10), source="claude-code",
    ))
    db.insert_span(make_llm_span(agent_id="claude-code-widgets", session_id="live-1",
                                 start_time=START, cost_usd=2.0, input_tokens=200, output_tokens=40))
    db.insert_span(make_tool_span(agent_id="claude-code-widgets", tool_name="Bash",
                                  session_id="live-1", start_time=START + timedelta(minutes=5),
                                  tool_input={"command": "git commit -m feat"}))
    db.close()
    (repo / "a.py").write_text("1\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    stamp = str(int((START + timedelta(minutes=5, seconds=2)).timestamp()))
    subprocess.run(["git", "commit", "-q", "-m", "feat"], cwd=repo, check=True,
                   env={**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp,
                        "GIT_AUTHOR_EMAIL": "dev@example.com",
                        "GIT_COMMITTER_EMAIL": "dev@example.com",
                        "GIT_AUTHOR_NAME": "Dev", "GIT_COMMITTER_NAME": "Dev"})
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert not (home / ".tj" / "install_id").exists()
    return {"config_path": config_path, "home": home, "repo": repo, "sha": sha}


def test_first_push_carries_context_install_id_and_commits(fresh_machine, cloud):
    """The acceptance case for issue #770: `tj init --cloud` on a machine whose
    sessions were ingested without context sends, on the FIRST push, sessions
    with repo context and this machine's install id, and the commits joined
    to them. Before: 25k spans, 11 sessions, 0 commits, no developer."""
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    out = _flat(result.output)

    install_id = (fresh_machine["home"] / ".tj" / "install_id").read_text().strip()
    assert install_id

    bodies = _ledger_bodies(cloud)
    sessions = [s for b in bodies for s in b["sessions"]]
    commits = [c for b in bodies for c in b["session_commits"]]
    assert [s["session_id"] for s in sessions] == ["live-1"]
    session = sessions[0]
    # Fix 1: context filled BEFORE the send.
    assert session["repo_remote"] == "https://github.com/Acme/widgets"
    assert session["repo_root"] == str(fresh_machine["repo"])
    assert session["branch_start"] == "main"
    assert session["user_email"] == "dev@example.com"
    assert session["developer_id"]
    # Fix 2: the install id, from ~/.tj/install_id, on every session row
    # and on the spans' resource attributes.
    assert session["install_id"] == install_id
    spans_body = json.loads(next(r.content for r in cloud.requests if r.url.path == cs.SPANS_PATH))
    resource_attrs = {a["key"]: a["value"] for a in spans_body["resourceSpans"][0]["resource"]["attributes"]}
    assert resource_attrs["tokenjam.install_id"] == {"stringValue": install_id}
    # Fix 1: the join was made before the send, and sessions went first.
    assert [(c["session_id"], c["commit_sha"], c["confidence"], c["source"]) for c in commits] == [
        ("live-1", fresh_machine["sha"], "deterministic", "tool_span_git_log"),
    ]
    paths = [r.url.path for r in cloud.requests]
    assert paths.index(cs.SPANS_PATH) > paths.index(cs.LEDGER_SESSIONS_PATH)
    # And the screen says what happened, in order.
    assert "repo context filled on 1 of 1" in out
    assert "1 new session-commit join" in out
    assert "Forwarded 2 spans, 1 sessions, 1 commits" in out
    assert out.index("Before sending") < out.index("Forwarded 2 spans")
    for command in advertised_commands(result.output):
        assert_invocable(command)


def test_the_status_line_stops_and_the_result_prints_before_the_daemon_restarts(
    fresh_machine, cloud, monkeypatch,
):
    """Fix 5. The spinner used to sit on screen until the restart finished.
    Rendered as a real terminal so the live display exists, then proved: no
    live display is active when the restart runs, and the forwarded line is
    already on screen by then."""
    from rich.console import Console

    from tokenjam.utils.formatting import console

    monkeypatch.setattr(Console, "is_terminal", property(lambda self: True))
    seen: dict[str, object] = {}

    def _restart(path, no_daemon, **kw):
        # Rich keeps every active live display (a status spinner is one) on
        # the console's live stack; an empty stack is "nothing is spinning".
        seen["live_at_restart"] = list(console._live_stack)
        return "restarted (stub)"

    monkeypatch.setattr("tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: True)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._restart_tj_server", _restart)
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    assert "live_at_restart" in seen, "the restart never ran"
    assert seen["live_at_restart"] == [], "a status line was still live during the restart"
    out = _flat(result.output)
    assert out.index("Forwarded 2 spans, 1 sessions, 1 commits") < out.index("Daemon: restarted")


def test_a_refill_or_match_failure_never_blocks_the_push(fresh_machine, cloud, monkeypatch):
    monkeypatch.setattr("tokenjam.core.shipped.match_sessions_to_commits",
                        lambda db, config=None, **kw: (_ for _ in ()).throw(RuntimeError("git exploded")))
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "could not run (git exploded)" in out
    assert "Forwarded 2 spans, 1 sessions, 0 commits" in out
    # The refill ran before the failing step: the session still left with context.
    assert _ledger_bodies(cloud)[0]["sessions"][0]["repo_remote"] == "https://github.com/Acme/widgets"


def test_a_missing_install_id_is_said_out_loud_and_never_blocks_the_push(
    fresh_machine, cloud, monkeypatch,
):
    """Fix 2, the failure half: an unwritable ~/.tj/install_id still lets the
    history leave, and the user is told Cloud will not count the machine."""
    monkeypatch.setattr("tokenjam.cli.ledger_cloud.ensure_install_id", lambda: None)
    monkeypatch.setattr("tokenjam.core.cloud_sync.ensure_install_id", lambda: None)
    result = _init("--cloud", KEY, "--org", ORG, "--yes")
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "No install id could be written" in out
    assert "without an install id" in out
    assert "Forwarded 2 spans, 1 sessions, 1 commits" in out
    assert _ledger_bodies(cloud)[0]["sessions"][0]["install_id"] is None
    for command in advertised_commands(result.output):
        assert_invocable(command)
