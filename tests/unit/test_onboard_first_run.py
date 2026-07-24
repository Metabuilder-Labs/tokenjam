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
    # One-line value prop; honest framing, no promised savings (Rule 14).
    assert "cost-saving utility for AI agents" in out
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


def test_nudge_claude_code_persona_leads_with_quota_diagnosis(capsys):
    """CC users get the quota-diagnosis commands; tjb is an SDK workflow and
    must not appear on the Claude Code list."""
    _print_next_steps_nudge(
        has_data=True, days=30, persona="claude_code", daemon_running=True,
    )
    out = capsys.readouterr().out
    assert "tj context" in out
    assert "tj quota-audit" in out
    assert "tj tokenmaxx" in out
    assert "tjb" not in out
    assert "tokenjam-bench" not in out
    # Lead with the diagnosis pair.
    assert out.index("tj context") < out.index("tj quota-audit") < out.index("tj tokenmaxx")


def test_nudge_daemon_running_never_suggests_tj_serve(capsys):
    """Onboarding just installed the daemon — Lens is already up; suggesting
    `tj serve` invites a port conflict."""
    _print_next_steps_nudge(
        has_data=True, days=30, persona="claude_code", daemon_running=True,
    )
    out = capsys.readouterr().out
    assert "tj serve" not in out
    assert "already running" in out
    assert "http://127.0.0.1:7391/" in out


def test_nudge_no_daemon_still_points_at_tj_serve(capsys):
    _print_next_steps_nudge(
        has_data=False, persona="claude_code", daemon_running=False,
    )
    out = capsys.readouterr().out
    assert "tj serve" in out
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
    assert "tj onboard" in out
    # No longer implies `--claude-code` is a separate/recommended setup.
    assert "tj onboard --claude-code" not in out


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
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
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


def _run_claude_code(tmp_path, plan_choice: str):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # --project skips the interactive project-name prompt; then plan_choice
        # then daily budget "0".
        return runner.invoke(
            cmd_onboard, ["--claude-code", "--no-daemon", "--project", "testproj"],
            input=f"{plan_choice}\n0\n", obj={},
        )


def test_claude_code_asks_plan_before_budget(_isolated_claude_code, tmp_path):
    # Choice 1 == api: the only tier that still gets a daily-budget prompt
    # (#128 — subscription tiers have $0 marginal cost and skip it silently).
    # Answers: plan(1=api) → API spend ceiling(0) → daily budget(0).
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(
            cmd_onboard, ["--claude-code", "--no-daemon", "--project", "testproj"],
            input="1\n0\n0\n", obj={},
        )
    assert res.exit_code == 0, res.output
    out = res.output
    assert "How do you pay for Claude?" in out
    assert "Daily budget in USD" in out
    assert out.index("How do you pay for Claude?") < out.index("Daily budget in USD"), out


def test_claude_code_restart_banner_precedes_nudge(_isolated_claude_code, tmp_path):
    """Founder-review order (2026-07): the one REQUIRED action (restart) is the
    visually primary element, next steps come after it, connection details
    are a dim footer at the end."""
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    out = res.output
    assert "Next steps" in out
    assert "Action required" in out
    assert out.index("Action required") < out.index("Next steps"), out
    assert out.index("Next steps") < out.index("Connection details"), out
    # No backfill happened here, so no "already loaded" over-claim.
    assert "already loaded" not in out


def test_claude_code_restart_panel_is_why_first_and_consolidated(
    _isolated_claude_code, tmp_path,
):
    """One consolidated Action-required panel (2026-07 restructure): the WHY
    (stale telemetry endpoint) leads, then numbered steps; no more scattering
    across a panel + a stray "open a new terminal" paragraph + an "after
    restarting, run" pointer + a duplicate verify line near Connection
    details."""
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    out = res.output
    flat = " ".join(out.split())
    assert "Action required" in flat
    assert "restart Claude Code" in flat
    # Why leads the panel, before the numbered steps.
    assert flat.index("old endpoint") < flat.index("Quit Claude Code"), flat
    assert "Quit Claude Code in every terminal" in flat
    assert "Relaunch claude in the same folder" in flat
    # The now-redundant scattered pieces are gone.
    assert "Open a new terminal" not in out
    assert "After restarting, run:" not in out
    assert "tj status --agent" not in out
    # Folded into one dim footnote instead.
    assert "own dashboard tile" in flat
    assert "claude --as <name>" in flat


def test_claude_code_restart_panel_states_honest_resume_semantics(
    _isolated_claude_code, tmp_path,
):
    """The old copy claimed `claude --resume` "picks up exactly where you left
    off", which is inaccurate: --resume opens a picker you must pick from; -c
    only reopens the CURRENT project's latest conversation. Pin the corrected,
    honest phrasing (Critical Rule 14: no promised outcomes)."""
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    flat = " ".join(res.output.split())
    assert "claude -c" in flat
    assert "reopen this project's latest conversation" in flat
    assert "claude --resume" in flat
    assert "pick any earlier one from a list" in flat
    assert "resuming is optional" in flat
    # The inaccurate old claims must not resurface.
    assert "pick up exactly where you left off" not in flat
    assert "conversation survives" not in flat


def test_claude_code_no_pre_restart_verify_prompt(_isolated_claude_code, tmp_path):
    """The interactive 'verify now?' poll can only time out before the restart
    it depends on; CC gets the verify pointer as step 3 of the restart panel
    instead, and only there (no duplicate near Connection details)."""
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    out = res.output
    assert "Verify tj is receiving telemetry now?" not in out
    assert "--verify-only" in out
    assert out.count("--verify-only") == 1
    assert "Verify after restarting:" not in out


def test_claude_code_asks_project_name_after_agent_questions(
    _isolated_claude_code, tmp_path,
):
    """Prompt order: usage/plan questions first, THEN project name (founder
    direction 2026-07 — the project-name question wedged between the two
    agent questions broke their grouping)."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # plan(3=max_5x, no ceiling/budget prompts) → project name (default).
        res = runner.invoke(
            cmd_onboard, ["--claude-code", "--no-daemon"], input="3\n\n", obj={},
        )
    assert res.exit_code == 0, res.output
    out = res.output
    assert "How do you pay for Claude?" in out
    assert "Project name" in out
    assert out.index("How do you pay for Claude?") < out.index("Project name"), out


def test_claude_code_shows_welcome_banner(_isolated_claude_code, tmp_path):
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    assert "TokenJam" in res.output
