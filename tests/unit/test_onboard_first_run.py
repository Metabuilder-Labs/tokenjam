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
    # One-line value prop; honest framing — no promised savings (Rule 14).
    assert "cost-optimization for AI agents" in out
    assert "saves you" not in out.lower()


# --- Next-steps nudge -------------------------------------------------------


def test_nudge_leads_with_no_restart_wins(capsys):
    _print_next_steps_nudge(has_data=True, days=30)
    out = capsys.readouterr().out
    # The three high-wow, no-restart commands.
    for cmd in ("tj tokenmaxx", "tj optimize", "tj serve"):
        assert cmd in out, out
    assert "already loaded" in out
    assert "last 30 days" in out


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


# --- Bare `tj` home screen --------------------------------------------------


def test_home_when_not_configured_points_at_onboarding(monkeypatch, capsys):
    monkeypatch.setattr("tokenjam.cli.home.find_config_file", lambda: None)
    print_home()
    out = capsys.readouterr().out
    assert "TokenJam" in out                     # banner
    assert "Not set up yet" in out
    assert "tj onboard --claude-code" in out


def test_home_when_configured_shows_next_best_actions(monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("version = '1'\n")
    monkeypatch.setattr("tokenjam.cli.home.find_config_file", lambda: cfg)
    print_home()
    out = capsys.readouterr().out
    assert "You're set up" in out
    for cmd in ("tj status", "tj optimize", "tj serve"):
        assert cmd in out, out


def test_bare_tj_renders_home_without_opening_db(monkeypatch):
    """`tj` with no subcommand prints the home screen and must NOT open the DB
    (so it works while `tj serve` holds the write lock)."""
    monkeypatch.setattr("tokenjam.cli.home.find_config_file", lambda: None)
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
    # Choice 3 == max_5x (subscription) → no API-ceiling prompt to interleave.
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    out = res.output
    assert "How do you pay for Claude?" in out
    assert "Daily budget in USD" in out
    assert out.index("How do you pay for Claude?") < out.index("Daily budget in USD"), out


def test_claude_code_nudge_precedes_restart_banner(_isolated_claude_code, tmp_path):
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    out = res.output
    assert "Next steps" in out
    assert "Restart" in out
    # Lead with the no-restart wins, THEN the restart note (#240).
    assert out.index("Next steps") < out.index("Restart"), out
    # No backfill happened here, so no "already loaded" over-claim.
    assert "already loaded" not in out


def test_claude_code_shows_welcome_banner(_isolated_claude_code, tmp_path):
    res = _run_claude_code(tmp_path, "3")
    assert res.exit_code == 0, res.output
    assert "TokenJam" in res.output
