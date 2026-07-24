"""`tj onboard --add-project` — register another repo's namespace without
re-running the full onboarding wizard.

Registering the Nth repo previously meant re-running `--claude-code` end to
end: plan prompt, budget prompt, backfill-scope prompt, a full backfill pass
over ALL of `~/.claude/projects`, a relearn scan, and a daemon reinstall — all
to set one config key (`agents.<id>.project`). `--add-project` writes just
that key against an already-onboarded global config.
"""
from __future__ import annotations

from click.testing import CliRunner

from tokenjam.cli.cmd_onboard import cmd_onboard
from tokenjam.cli.home import print_home
from tokenjam.core.config import AgentConfig, ProviderBudget, TjConfig, load_config, write_config


def _existing_global_config(home, *, agents=None, budgets=None):
    """Write a minimal already-onboarded global config under `home`."""
    config_path = home / ".config" / "tj" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_config(
        TjConfig(
            version="1",
            agents=agents or {},
            budgets=budgets or {"anthropic": ProviderBudget(plan="api")},
        ),
        config_path,
    )
    return config_path


def _run(home, repo_dir, args):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=repo_dir):
        return runner.invoke(cmd_onboard, args, obj={})


class TestAddProjectRequiresExistingConfig:
    def test_fails_clearly_when_no_global_config(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "my-repo",
        )

        res = _run(home, repo, ["--add-project", "--project", "myproj"])

        assert res.exit_code != 0
        assert "tj onboard" in res.output
        assert not (home / ".config" / "tj" / "config.toml").exists()


class TestAddProjectRegistersAgent:
    def test_first_registration_writes_namespace(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        config_path = _existing_global_config(home)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )

        res = _run(home, repo, ["--add-project", "--project", "widgets"])

        assert res.exit_code == 0, res.output
        cfg = load_config(str(config_path))
        assert "claude-code-widgets-api" in cfg.agents
        assert cfg.agents["claude-code-widgets-api"].project == "widgets"
        assert "widgets" in res.output
        assert "claude-code-widgets-api" in res.output

    def test_no_plan_budget_or_backfill_prompt(self, tmp_path, monkeypatch):
        """The whole point: none of the heavy --claude-code prompts fire."""
        home = tmp_path / "home"
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        _existing_global_config(home)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )

        res = _run(home, repo, ["--add-project", "--project", "widgets"])

        assert res.exit_code == 0, res.output
        for banned in (
            "Daily budget", "Monthly Anthropic API spend ceiling",
            "backfill", "Backfill", "How do you pay",
        ):
            assert banned not in res.output, f"unexpected prompt text: {banned!r}"

    def test_does_not_touch_backfill_or_relearn(self, tmp_path, monkeypatch):
        """Belt-and-suspenders: patch the heavy paths so a call would explode
        the test, proving --add-project never reaches them."""
        home = tmp_path / "home"
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        _existing_global_config(home)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )

        def _boom(*a, **k):
            raise AssertionError("--add-project must not run backfill/relearn")

        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._resolve_backfill_scope", _boom, raising=False,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._run_relearn_first_fix", _boom, raising=False,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._install_claude_wrapper", _boom, raising=False,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._install_daemon", _boom, raising=False,
        )

        res = _run(home, repo, ["--add-project", "--project", "widgets"])
        assert res.exit_code == 0, res.output


class TestAddProjectIdempotent:
    def test_rerun_updates_namespace_in_place(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        config_path = _existing_global_config(
            home,
            agents={
                "claude-code-widgets-api": AgentConfig(project="old-namespace"),
            },
        )
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )

        res = _run(home, repo, ["--add-project", "--project", "new-namespace"])

        assert res.exit_code == 0, res.output
        cfg = load_config(str(config_path))
        assert cfg.agents["claude-code-widgets-api"].project == "new-namespace"
        # Only one agent entry — re-registering doesn't duplicate it.
        assert len(cfg.agents) == 1

    def test_rerun_preserves_other_agent_fields(self, tmp_path, monkeypatch):
        """Re-registering must not clobber budget/description already set on
        the agent by an earlier --claude-code onboard."""
        home = tmp_path / "home"
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        from tokenjam.core.config import BudgetConfig

        config_path = _existing_global_config(
            home,
            agents={
                "claude-code-widgets-api": AgentConfig(
                    description="existing agent",
                    budget=BudgetConfig(daily_usd=5.0),
                    project="old-namespace",
                ),
            },
        )
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )

        res = _run(home, repo, ["--add-project", "--project", "new-namespace"])

        assert res.exit_code == 0, res.output
        cfg = load_config(str(config_path))
        agent = cfg.agents["claude-code-widgets-api"]
        assert agent.project == "new-namespace"
        assert agent.description == "existing agent"
        assert agent.budget.daily_usd == 5.0


class TestAddProjectPromptsOnlyForNamespace:
    def test_project_flag_skips_interactive_prompt(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        _existing_global_config(home)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )

        def _boom_prompt(*a, **k):
            raise AssertionError("click.prompt should not be called with --project set")

        monkeypatch.setattr("click.prompt", _boom_prompt)

        res = _run(home, repo, ["--add-project", "--project", "widgets"])
        assert res.exit_code == 0, res.output


class TestHomeScreenNudge:
    """The bare `tj` home screen (already-configured branch) nudges
    `--add-project` when the CURRENT repo's agent has no namespace yet — the
    exact moment a user has cd'd into a new, unregistered repo."""

    def test_nudges_when_current_repo_unregistered(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / "home"
        _existing_global_config(home)  # global config exists, but no agents
        monkeypatch.setattr("tokenjam.cli.home.Path.home", lambda: home)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "new-repo",
        )
        monkeypatch.setattr(
            "tokenjam.cli.home.find_config_file",
            lambda *a, **k: home / ".config" / "tj" / "config.toml",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        print_home()
        out = capsys.readouterr().out
        assert "tj onboard --add-project" in out

    def test_no_nudge_when_current_repo_already_registered(
        self, tmp_path, monkeypatch, capsys,
    ):
        home = tmp_path / "home"
        _existing_global_config(
            home,
            agents={"claude-code-new-repo": AgentConfig(project="already-set")},
        )
        monkeypatch.setattr("tokenjam.cli.home.Path.home", lambda: home)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "new-repo",
        )
        monkeypatch.setattr(
            "tokenjam.cli.home.find_config_file",
            lambda *a, **k: home / ".config" / "tj" / "config.toml",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        print_home()
        out = capsys.readouterr().out
        assert "tj onboard --add-project" not in out

    def test_no_nudge_when_no_global_config(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr("tokenjam.cli.home.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.home.find_config_file", lambda *a, **k: None,
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        print_home()
        out = capsys.readouterr().out
        assert "tj onboard --add-project" not in out
