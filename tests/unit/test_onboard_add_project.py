"""`tj onboard --add-project` — register another repo's namespace without
re-running the full onboarding wizard.

Registering the Nth repo previously meant re-running `--claude-code` end to
end: plan prompt, budget prompt, backfill-scope prompt, a full backfill pass
over ALL of `~/.claude/projects`, a relearn scan, and a daemon reinstall — all
to set one config key (`agents.<id>.project`). `--add-project` writes just
that key against an already-onboarded config, resolved the same way the rest
of the CLI resolves it (TJ_CONFIG, then project-local, then global —
`load_config`'s `SEARCH_PATHS` order), so it targets whatever config actually
governs this repo rather than always the default global path.
"""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import tokenjam.core.config as cfg_mod
from tokenjam.cli.cmd_onboard import cmd_onboard
from tokenjam.cli.home import print_home
from tokenjam.core.config import AgentConfig, ProviderBudget, TjConfig, load_config, write_config


def _normalize_output(text):
    """Normalize output for wrap-independent matching.

    Rich wraps long lines at console width, breaking strings across newlines.
    This collapses whitespace to match substrings regardless of where wrapping occurs.
    """
    return " ".join(text.split())


def _write_config(path, *, agents=None, budgets=None):
    """Write a minimal already-onboarded config to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_config(
        TjConfig(
            version="1",
            agents=agents or {},
            budgets=budgets or {"anthropic": ProviderBudget(plan="api")},
        ),
        path,
    )
    return path


def _patch_search_paths(monkeypatch, global_config_path):
    """`SEARCH_PATHS` is a module-level constant built at import time from the
    REAL home dir, so patching `Path.home` has no effect on it.
    Patch it directly, same pattern as `tests/unit/test_config.py`, keeping
    the relative project-local candidates so `.tj/config.toml` /
    `tokenjam.toml` in the isolated cwd still take priority when present.
    """
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
        Path("tokenjam.toml"),
        Path(".tj/config.toml"),
        global_config_path,
    ])


def _run(repo_dir, args, env=None):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=repo_dir):
        return runner.invoke(cmd_onboard, args, obj={}, env=env or {})


class TestAddProjectRequiresExistingConfig:
    def test_fails_clearly_when_no_config_anywhere(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        _patch_search_paths(monkeypatch, tmp_path / "home" / ".config" / "tj" / "config.toml")
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "my-repo",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        res = _run(repo, ["--add-project", "--project", "myproj"])

        assert res.exit_code != 0
        assert "tj onboard" in res.output


class TestAddProjectRegistersAgent:
    def test_first_registration_writes_namespace(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        global_config = _write_config(tmp_path / "home" / ".config" / "tj" / "config.toml")
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        res = _run(repo, ["--add-project", "--project", "widgets"])

        assert res.exit_code == 0, res.output
        cfg = load_config(str(global_config))
        assert "claude-code-widgets-api" in cfg.agents
        assert cfg.agents["claude-code-widgets-api"].project == "widgets"
        assert "widgets" in res.output
        assert "claude-code-widgets-api" in _normalize_output(res.output)

    def test_no_plan_budget_or_backfill_prompt(self, tmp_path, monkeypatch):
        """The whole point: none of the heavy --claude-code prompts fire."""
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        global_config = _write_config(tmp_path / "home" / ".config" / "tj" / "config.toml")
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        res = _run(repo, ["--add-project", "--project", "widgets"])

        assert res.exit_code == 0, res.output
        normalized_output = _normalize_output(res.output)
        for banned in (
            "Daily budget", "Monthly Anthropic API spend ceiling",
            "backfill", "Backfill", "How do you pay",
        ):
            assert banned not in normalized_output, f"unexpected prompt text: {banned!r}"

    def test_does_not_touch_backfill_or_relearn(self, tmp_path, monkeypatch):
        """Belt-and-suspenders: patch the heavy paths so a call would explode
        the test, proving --add-project never reaches them."""
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        global_config = _write_config(tmp_path / "home" / ".config" / "tj" / "config.toml")
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

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

        res = _run(repo, ["--add-project", "--project", "widgets"])
        assert res.exit_code == 0, res.output


class TestAddProjectIdempotent:
    def test_rerun_updates_namespace_in_place(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        global_config = _write_config(
            tmp_path / "home" / ".config" / "tj" / "config.toml",
            agents={
                "claude-code-widgets-api": AgentConfig(project="old-namespace"),
            },
        )
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        res = _run(repo, ["--add-project", "--project", "new-namespace"])

        assert res.exit_code == 0, res.output
        cfg = load_config(str(global_config))
        assert cfg.agents["claude-code-widgets-api"].project == "new-namespace"
        # Only one agent entry — re-registering doesn't duplicate it.
        assert len(cfg.agents) == 1

    def test_rerun_preserves_other_agent_fields(self, tmp_path, monkeypatch):
        """Re-registering must not clobber budget/description already set on
        the agent by an earlier --claude-code onboard."""
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        from tokenjam.core.config import BudgetConfig

        global_config = _write_config(
            tmp_path / "home" / ".config" / "tj" / "config.toml",
            agents={
                "claude-code-widgets-api": AgentConfig(
                    description="existing agent",
                    budget=BudgetConfig(daily_usd=5.0),
                    project="old-namespace",
                ),
            },
        )
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        res = _run(repo, ["--add-project", "--project", "new-namespace"])

        assert res.exit_code == 0, res.output
        cfg = load_config(str(global_config))
        agent = cfg.agents["claude-code-widgets-api"]
        assert agent.project == "new-namespace"
        assert agent.description == "existing agent"
        assert agent.budget.daily_usd == 5.0


class TestAddProjectPromptsOnlyForNamespace:
    def test_project_flag_skips_interactive_prompt(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        global_config = _write_config(tmp_path / "home" / ".config" / "tj" / "config.toml")
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        def _boom_prompt(*a, **k):
            raise AssertionError("click.prompt should not be called with --project set")

        monkeypatch.setattr("click.prompt", _boom_prompt)

        res = _run(repo, ["--add-project", "--project", "widgets"])
        assert res.exit_code == 0, res.output


class TestAddProjectHonorsTjConfig:
    """P1 fix: --add-project must resolve the config the same way the rest of
    the CLI does — honoring TJ_CONFIG — instead of always hardcoding the
    default global path regardless of an active override."""

    def test_writes_to_tj_config_path_when_set(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        custom_config = _write_config(tmp_path / "custom" / "tj-config.toml")
        # SEARCH_PATHS points somewhere else entirely — TJ_CONFIG must win.
        _patch_search_paths(
            monkeypatch, tmp_path / "unused-home" / ".config" / "tj" / "config.toml",
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )

        res = _run(
            repo, ["--add-project", "--project", "widgets"],
            env={"TJ_CONFIG": str(custom_config)},
        )

        assert res.exit_code == 0, res.output
        cfg = load_config(str(custom_config))
        assert cfg.agents["claude-code-widgets-api"].project == "widgets"
        # Rich may line-wrap the path, so normalize output to make matching
        # wrap-independent rather than relying on lucky console width.
        assert "tj-config.toml" in _normalize_output(res.output)

    def test_fails_clearly_when_tj_config_points_nowhere(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        _patch_search_paths(
            monkeypatch, tmp_path / "unused-home" / ".config" / "tj" / "config.toml",
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        missing = tmp_path / "does-not-exist.toml"

        res = _run(
            repo, ["--add-project", "--project", "widgets"],
            env={"TJ_CONFIG": str(missing)},
        )

        assert res.exit_code != 0


class TestAddProjectUsesProjectLocalConfigWhenPresent:
    """P1 fix: a project-local `.tj/config.toml` in THIS repo (e.g. from an
    earlier bare `tj onboard`) is the config actually loaded here at runtime
    — --add-project must write into it rather than a stale/irrelevant global
    file, or the mapping would silently never take effect for this repo."""

    def test_local_config_wins_over_global(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)
        global_config = _write_config(tmp_path / "home" / ".config" / "tj" / "config.toml")
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "widgets-api",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=repo) as cwd:
            local_config_path = Path(cwd) / ".tj" / "config.toml"
            _write_config(local_config_path)
            monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
                Path("tokenjam.toml"),
                Path(".tj/config.toml"),
                global_config,
            ])
            res = runner.invoke(
                cmd_onboard, ["--add-project", "--project", "widgets"], obj={},
            )

            assert res.exit_code == 0, res.output
            local_cfg = load_config(str(local_config_path))
            assert local_cfg.agents["claude-code-widgets-api"].project == "widgets"
            # The global file must be untouched.
            global_cfg = load_config(str(global_config))
            assert "claude-code-widgets-api" not in global_cfg.agents


class TestHomeScreenNudge:
    """The bare `tj` home screen (already-configured branch) nudges
    `--add-project` when the CURRENT repo's agent has no namespace yet — the
    exact moment a user has cd'd into a new, unregistered repo."""

    def test_nudges_when_current_repo_unregistered(self, tmp_path, monkeypatch, capsys):
        global_config = _write_config(
            tmp_path / "home" / ".config" / "tj" / "config.toml",
            # A different repo's claude-code agent already exists, proving
            # this user is a Claude Code user -- just not registered HERE yet.
            agents={"claude-code-other-repo": AgentConfig(project="other")},
        )
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "new-repo",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)

        print_home()
        out = capsys.readouterr().out
        assert "tj onboard --add-project" in out

    def test_no_nudge_when_current_repo_already_registered(
        self, tmp_path, monkeypatch, capsys,
    ):
        global_config = _write_config(
            tmp_path / "home" / ".config" / "tj" / "config.toml",
            agents={"claude-code-new-repo": AgentConfig(project="already-set")},
        )
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "new-repo",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)

        print_home()
        out = capsys.readouterr().out
        assert "tj onboard --add-project" not in out

    def test_no_nudge_when_no_config(self, tmp_path, monkeypatch, capsys):
        _patch_search_paths(
            monkeypatch, tmp_path / "home" / ".config" / "tj" / "config.toml",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)

        print_home()
        out = capsys.readouterr().out
        assert "tj onboard --add-project" not in out

    def test_no_nudge_for_codex_only_user(self, tmp_path, monkeypatch, capsys):
        """P2 fix: a user who only ever ran `tj onboard --codex` has no
        `claude-code-*` agent at all. --add-project only ever writes a
        claude-code-* entry, so nudging them would be irrelevant nagging."""
        global_config = _write_config(
            tmp_path / "home" / ".config" / "tj" / "config.toml",
            agents={"codex-new-repo": AgentConfig(project="already-set")},
        )
        _patch_search_paths(monkeypatch, global_config)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._derive_project_name", lambda: "new-repo",
        )
        monkeypatch.delenv("TJ_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)

        print_home()
        out = capsys.readouterr().out
        assert "tj onboard --add-project" not in out
