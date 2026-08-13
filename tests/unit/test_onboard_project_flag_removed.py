"""`tj onboard --project` was removed — the project/dashboard-namespace name
is now ALWAYS derived from the repo (git remote) or folder name
(`_derive_project_name`), never typed in. A script that still passes
`--project` must get a clear, specific error explaining why it stopped
working, not Click's generic "no such option"."""
from __future__ import annotations

from click.testing import CliRunner

from tokenjam.cli.cmd_onboard import cmd_onboard


def test_project_flag_fails_with_a_clear_removal_message():
    runner = CliRunner()
    res = runner.invoke(
        cmd_onboard, ["--claude-code", "--project", "aquanode"], obj={},
    )
    assert res.exit_code != 0
    assert "--project has been removed" in res.output
    assert "repo/folder name" in res.output
    # Not Click's generic rejection.
    assert "no such option" not in res.output.lower()


def test_project_flag_fails_before_any_other_onboarding_work():
    """The check fires first — before the ephemeral-runner guard, the
    welcome banner, or any prompt — so a script passing the removed flag
    gets the error immediately, not buried after other output."""
    runner = CliRunner()
    res = runner.invoke(cmd_onboard, ["--project", "x"], obj={})
    assert res.exit_code != 0
    assert "--project has been removed" in res.output
