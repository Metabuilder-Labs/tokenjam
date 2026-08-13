"""First-run onboarding UX (#240): welcome banner, plan-first prompt order,
post-onboard next-steps nudge, and the bare `tj` home screen."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

import tokenjam.core.backfill as backfill_mod
from tokenjam import __version__
from tokenjam.cli.banner import print_welcome_banner
from tokenjam.cli.cmd_onboard import _print_next_steps_nudge, cmd_onboard
from tokenjam.cli.home import print_home


# --- Welcome banner ---------------------------------------------------------


def test_welcome_banner_shows_brand_version_and_value_prop(capsys):
    print_welcome_banner()
    out = capsys.readouterr().out
    assert "TokenJam" in out
    assert __version__ in out
    # Trimmed one-line tagline (#643); honest framing, no promised savings (Rule 14).
    assert "token efficiency for AI agents" in out
    assert "saves you" not in out.lower()


# --- Next-steps nudge -------------------------------------------------------


def test_nudge_leads_with_no_restart_wins(capsys):
    _print_next_steps_nudge(has_data=True, days=30)
    out = capsys.readouterr().out
    # The high-wow, no-restart commands + the bench "prove" nudge.
    for cmd in ("tj tokenmaxx", "tj optimize", "tjb", "tj serve"):
        assert cmd in out, out
    # Prove step: honest framing (holds, never "guaranteed") + install hint.
    assert "prove a cheaper model still holds" in out
    assert "pip install tokenjam-bench" in out
    assert "already loaded" in out
    assert "last 30 days" in out
    # tokenmaxx is the efficiency card, never a spend brag.
    assert "spend tier" not in out


def test_nudge_without_data_omits_already_loaded_claim(capsys):
    # Honesty: don't claim "already loaded" when nothing was backfilled.
    _print_next_steps_nudge(has_data=False)
    out = capsys.readouterr().out
    assert "already loaded" not in out
    for cmd in ("tj tokenmaxx", "tj optimize", "tj serve"):
        assert cmd in out, out


def test_nudge_with_data_but_unknown_days_avoids_bogus_count(capsys):
    _print_next_steps_nudge(has_data=True, days=None)
    out = capsys.readouterr().out
    assert "already loaded" in out
    assert "last None days" not in out


def test_nudge_claude_code_persona_is_exactly_two_entries(capsys):
    """The CC list is exactly two action rows (founder review, demo trim):
    "View the Web UI (Lens)" then "View your Token Efficiency Card". The
    per-command diagnosis rows (`tj context`, `tj quota-audit`), `tj optimize`,
    and the `tjb` SDK workflow must not appear here."""
    _print_next_steps_nudge(
        has_data=True, days=30, persona="claude-code", daemon_running=True,
    )
    out = capsys.readouterr().out
    assert "View the Web UI (Lens)" in out
    assert "View your Token Efficiency Card" in out
    assert "tj tokenmaxx" in out
    assert out.index("View the Web UI (Lens)") < out.index("View your Token Efficiency Card")
    assert "tj context" not in out
    assert "tj quota-audit" not in out
    assert "tj optimize" not in out
    assert "tjb" not in out
    assert "tokenjam-bench" not in out
    # Exactly two command rows: the indented block that follows the
    # "Next steps" header, up to the blank line that closes it. (Anything
    # after that blank line is a separate advisory, e.g. the PATH warning.)
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if "Next steps" in ln) + 1
    rows = []
    for ln in lines[start:]:
        if not ln.strip():
            if rows:
                break
            continue
        rows.append(ln)
    assert len(rows) == 2, rows


def test_nudge_claude_code_header_is_plain(capsys):
    """The CC "Next steps" header is bare — no trailing "already loaded /
    these work right now" muted text (founder review, demo trim)."""
    _print_next_steps_nudge(
        has_data=True, days=30, persona="claude-code", daemon_running=True,
    )
    out = capsys.readouterr().out
    assert "Next steps" in out
    assert "already loaded" not in out
    assert "these work right now" not in out


def test_nudge_claude_code_points_at_dashboard_url(capsys):
    """The CC list names the dashboard URL as a concrete instruction and runs
    `tj tokenmaxx` — regardless of daemon state (no daemon branch on this
    list anymore)."""
    for daemon in (True, False):
        _print_next_steps_nudge(
            has_data=True, days=30, persona="claude-code", daemon_running=daemon,
        )
        out = capsys.readouterr().out
        assert "http://127.0.0.1:7391/" in out
        assert "tj tokenmaxx" in out
        # No `tj serve` / "already running" churn on this trimmed list.
        assert "tj serve" not in out
        assert "already running" not in out


# --- Bare `tj` home screen --------------------------------------------------


def test_home_when_not_configured_points_at_onboarding(monkeypatch, capsys):
    # No config discoverable AND no populated DB → truly a fresh user (#506).
    monkeypatch.setattr("tokenjam.cli.home.find_config_file", lambda *a, **k: None)
    monkeypatch.setattr("tokenjam.cli.home._db_has_data", lambda: False)
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    print_home()
    out = capsys.readouterr().out
    assert "TokenJam" in out                     # banner
    assert "Not set up yet" in out
    # Get Started section (#643): all three onboard variants, each with a
    # "run this if..." one-liner, plus a Learn more section.
    assert "Get Started:" in out
    assert "tj onboard" in out
    assert "tj onboard --claude-code" in out
    assert "tj onboard --codex" in out
    assert "run this if you only use Claude Code" in out
    assert "run this if you only use Codex" in out
    assert "more than one agent client" in out
    assert "Learn more:" in out
    assert "tj --help" in out
    assert "tokenjam.dev/docs" in out
    # #643: the stale "run it once inside each project" line is gone —
    # onboarding aggregates all projects automatically.
    assert "inside each project" not in out


def test_home_when_configured_shows_next_best_actions(monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("version = '1'\n")
    monkeypatch.setattr("tokenjam.cli.home.find_config_file", lambda *a, **k: cfg)
    # Clear TJ_CONFIG so a stray env var on the runner can't short-circuit
    # _is_set_up() via its env branch and bypass the patched find_config_file
    # this test means to exercise.
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    print_home()
    out = capsys.readouterr().out
    assert "You're set up" in out
    for cmd in ("tj status", "tj optimize", "tj serve"):
        assert cmd in out, out


def test_home_when_db_populated_but_no_config_is_set_up(monkeypatch, capsys, tmp_path):
    """#506: a user who backfilled (populated DB) but never ran full `tj onboard`
    has no config file, yet must NOT be told "Not set up yet"."""
    # No config discoverable anywhere...
    monkeypatch.setattr("tokenjam.cli.home.find_config_file", lambda *a, **k: None)
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    # ...but a non-empty DB file exists at the default path.
    db = tmp_path / ".tj" / "telemetry.duckdb"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"\x00" * 4096)
    # `_db_has_data()` resolves the default "~/.tj/telemetry.duckdb" via
    # `Path.expanduser()`, which reads the `HOME` env var directly rather
    # than going through `pathlib.Path.home()` — so patching `Path.home`
    # (as this test used to) is a no-op here and the assertion only passed
    # by accident, off a DB left behind under the session-wide fake HOME by
    # an earlier test. Set `HOME` itself so this test's own db is what gets
    # found, independent of suite order.
    monkeypatch.setenv("HOME", str(tmp_path))
    print_home()
    out = capsys.readouterr().out
    assert "You're set up" in out, out
    assert "Not set up yet" not in out


def test_bare_tj_renders_home_without_opening_db(monkeypatch):
    """`tj` with no subcommand prints the home screen and must NOT open the DB
    (so it works while `tj serve` holds the write lock)."""
    monkeypatch.setattr("tokenjam.cli.home.find_config_file", lambda *a, **k: None)
    monkeypatch.setattr("tokenjam.cli.home._db_has_data", lambda: False)
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    monkeypatch.setattr(
        "tokenjam.cli.main.open_db",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bare tj opened the DB")),
    )
    from tokenjam.cli.main import cli

    res = CliRunner().invoke(cli, [], obj={})
    assert res.exit_code == 0, res.output
    assert "TokenJam" in res.output
    assert "tj onboard" in res.output


# --- Plan-first prompt order (#240 part 3) ----------------------------------


@pytest.fixture
def _isolated_claude_code(monkeypatch, tmp_path):
    """Make `tj onboard --claude-code` safe + deterministic to drive in-process:
    isolate HOME, skip backfill, and stub out daemon / external side effects."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    # No Claude Code history → backfill is skipped (has_data == False).
    monkeypatch.setattr(
        backfill_mod, "CLAUDE_CODE_PROJECTS_ROOT", tmp_path / "no-such-claude",
    )
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _x: None)
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._finish_onboard_serve", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._try_apply_declared_plans", lambda *a, **k: None,
    )
    # #643: the restart panel is now conditional on `claude` actually running.
    # These tests assert panel CONTENT/ORDER, so force it on deterministically
    # (whether the machine running the suite has a `claude` process is
    # irrelevant to what the panel says). The conditional behavior itself is
    # covered by test_restart_panel_conditional_on_claude_running below.
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._claude_code_is_running", lambda: True,
    )


def _run_claude_code(tmp_path, plan_choice: str):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # The project name is derived automatically (no prompt); the input
        # sequence is just plan_choice then daily budget "0".
        return runner.invoke(
            cmd_onboard, ["--claude-code", "--no-daemon"],
            input=f"{plan_choice}\n0\n", obj={},
        )


def test_claude_code_asks_plan_before_budget(_isolated_claude_code, tmp_path):
    # Choice 1 == api: the only tier that still gets a daily-budget prompt
    # (#128 — subscription tiers have $0 marginal cost and skip it silently).
    # Answers: plan(1=api) → API spend ceiling(0) → daily budget(0).
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(
            cmd_onboard, ["--claude-code", "--no-daemon"],
            input="1\n0\n0\n", obj={},
        )
    assert res.exit_code == 0, res.output
    out = res.output
    assert "How do you pay for Claude?" in out
    assert "Daily budget in USD" in out
    assert out.index("How do you pay for Claude?") < out.index("Daily budget in USD"), out


def test_claude_code_restart_banner_is_third_next_step(_isolated_claude_code, tmp_path):
    """#675: the optional restart note is now the THIRD item UNDER Next steps
    (it used to print above the Next-steps header). Still one OPTIONAL line —
    the daemon catch-up ingests running sessions regardless, so a restart is not
    required. #643: connection details moved behind --verbose."""
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    out = res.output
    assert "Next steps" in out
    assert "restart Claude Code" in out
    assert "Action required" not in out
    # Now inside Next steps, so it follows the header rather than preceding it.
    assert out.index("Next steps") < out.index("restart Claude Code"), out
    # #643: connection details are no longer on the default success screen.
    assert "Connection details" not in out
    # No backfill happened here, so no "already loaded" over-claim.
    assert "already loaded" not in out


def test_restart_panel_conditional_on_claude_running(
    _isolated_claude_code, tmp_path, monkeypatch,
):
    """#643: the optional restart line only shows when a `claude` process is
    actually running. A fresh terminal with none open needs no restart, so the
    line is suppressed."""
    # Force "not running" — the line must be absent.
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._claude_code_is_running", lambda: False,
    )
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    assert "restart Claude Code" not in res.output
    # ... but the rest of the success screen is intact.
    assert "Claude Code observability configured" in res.output
    assert "Next steps" in res.output


def test_verbose_shows_connection_details(_isolated_claude_code, tmp_path):
    """#643: --verbose restores the connection-details block on the success
    screen for debugging/harness setups."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(
            cmd_onboard,
            ["--claude-code", "--no-daemon", "--verbose"],
            input="3\n0\n", obj={},
        )
    assert res.exit_code == 0, res.output
    assert "Connection details" in res.output
    assert "Agent ID:" in res.output


def test_claude_code_restart_note_is_one_optional_line(
    _isolated_claude_code, tmp_path,
):
    """Demo trim (2026-07-30): the old "Action required" panel (numbered steps +
    resume semantics) is replaced by ONE optional line, because the daemon's
    transcript catch-up ingests running sessions from disk regardless — a
    restart only affects live-stream immediacy, so it is not an action gate."""
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    flat = " ".join(res.output.split())
    assert "Optional: restart Claude Code" in flat
    assert "picks them up from disk" in flat
    # The old scary panel, numbered steps, resume copy and footnote are gone.
    assert "Action required" not in flat
    assert "Quit Claude Code in every terminal" not in flat
    assert "claude --resume" not in flat
    assert "own dashboard tile" not in flat
    assert "claude --as <name>" not in flat


def test_claude_code_no_pre_restart_verify_prompt(_isolated_claude_code, tmp_path):
    """The interactive 'verify now?' poll can only time out before the restart
    it depends on, so it must never appear. The demo trim (2026-07-30) also
    dropped the --verify-only pointer — the restart is one optional line now."""
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    out = res.output
    assert "Verify tj is receiving telemetry now?" not in out
    assert "Verify after restarting:" not in out


def test_claude_code_never_prompts_for_a_project_name(
    _isolated_claude_code, tmp_path,
):
    """--project (and the interactive prompt it fed) were removed — the
    project/dashboard-namespace name is now always derived from the repo/
    folder name, never asked for."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # plan(3=max_5x, no ceiling/budget prompts) — nothing else to answer.
        res = runner.invoke(
            cmd_onboard, ["--claude-code", "--no-daemon"], input="3\n", obj={},
        )
    assert res.exit_code == 0, res.output
    out = res.output
    assert "How do you pay for Claude?" in out
    assert "Project name" not in out


def test_claude_code_shows_welcome_banner(_isolated_claude_code, tmp_path):
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    assert "TokenJam" in res.output
