"""Path-branched first run (#448).

`tj onboard` (no flag) opens with "How do you use AI agents?" and routes to the
matching flow — Claude Code / Codex / SDK / combination — so a Claude Code user
(the common case) gets a backfill + statusline instead of an SDK snippet and a
live-span verify that can never succeed. `--claude-code` / `--codex` stay as
shortcuts that skip the question. A non-tty bare invocation keeps the historical
generic SDK behavior (scripts / CI).
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

import tokenjam.core.backfill as backfill_mod
from tokenjam.cli.cmd_onboard import (
    _onboard_combination,
    _print_setup_complete_home,
    _prompt_usage_path,
    _try_backfill_codex,
    cmd_onboard,
)


# --- The path question -------------------------------------------------------


class TestUsagePathPrompt:
    def test_lists_all_four_paths(self, capsys, monkeypatch):
        import click as _click

        monkeypatch.setattr(_click, "prompt", lambda *a, **k: 1)
        _prompt_usage_path()
        out = capsys.readouterr().out
        assert "How do you use AI agents?" in out
        assert "Claude Code" in out
        assert "Codex" in out
        assert "Your own agents" in out
        assert "combination" in out.lower()

    @pytest.mark.parametrize(
        "choice,expected",
        [(1, "claude_code"), (2, "codex"), (3, "sdk"), (4, "combination")],
    )
    def test_returns_selected_key(self, choice, expected, monkeypatch):
        import click as _click

        monkeypatch.setattr(_click, "prompt", lambda *a, **k: choice)
        assert _prompt_usage_path() == expected


# --- Routing: bare onboard dispatches on the choice --------------------------


@pytest.fixture
def _routing_stubs(monkeypatch):
    """Stub the per-path onboarders so we can assert which one bare onboard
    dispatched to, without running their heavy side effects. Force a tty so the
    path question is asked."""
    called: dict[str, bool] = {}

    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._is_interactive", lambda: True,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._onboard_claude_code",
        lambda *a, **k: called.__setitem__("claude_code", True),
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._onboard_codex",
        lambda *a, **k: called.__setitem__("codex", True),
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._onboard_combination",
        lambda *a, **k: called.__setitem__("combination", True),
    )
    return called


@pytest.mark.parametrize(
    "answer,key",
    [("1", "claude_code"), ("2", "codex"), ("4", "combination")],
)
def test_bare_onboard_routes_to_selected_path(_routing_stubs, answer, key):
    res = CliRunner().invoke(cmd_onboard, [], input=f"{answer}\n", obj={})
    assert res.exit_code == 0, res.output
    assert _routing_stubs.get(key) is True, res.output
    # It must NOT have run any other path.
    assert set(_routing_stubs) == {key}


def test_bare_onboard_sdk_choice_falls_through_to_generic(_routing_stubs, monkeypatch):
    """Choice 3 (SDK) falls through to the historical generic path, not a
    per-path onboarder."""
    # The generic path writes a real config; keep it from touching the daemon /
    # DB apply. It will still write .tj/config.toml in the isolated fs.
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._finish_onboard_serve", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._try_apply_declared_plans", lambda *a, **k: None,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        # path=3(sdk) → plan prompt(1=api) → api ceiling(0) → daily budget(0)
        res = runner.invoke(
            cmd_onboard, ["--no-daemon"], input="3\n1\n0\n", obj={},
        )
    assert res.exit_code == 0, res.output
    # No per-path onboarder fired.
    assert _routing_stubs == {}
    # Generic path prints the instrument snippet.
    assert "@watch" in res.output


# --- Non-tty keeps the historical generic behavior ---------------------------


def test_non_tty_bare_onboard_skips_path_question(monkeypatch):
    """A non-interactive bare `tj onboard` must NOT ask the path question — it
    falls straight through to the generic SDK path (scripts / CI)."""
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._is_interactive", lambda: False,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._finish_onboard_serve", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._try_apply_declared_plans", lambda *a, **k: None,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Non-interactive: no plan prompt; the budget prompt still reads stdin.
        res = runner.invoke(cmd_onboard, ["--no-daemon"], input="0\n", obj={})
    assert res.exit_code == 0, res.output
    assert "How do you use AI agents?" not in res.output


# --- --claude-code / --codex still skip the question -------------------------


def test_flag_shortcut_skips_path_question(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._is_interactive", lambda: True,
    )
    fired = {}
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._onboard_claude_code",
        lambda *a, **k: fired.__setitem__("cc", True),
    )
    res = CliRunner().invoke(cmd_onboard, ["--claude-code"], obj={})
    assert res.exit_code == 0, res.output
    assert fired.get("cc") is True
    assert "How do you use AI agents?" not in res.output


# --- Defensive Codex backfill ------------------------------------------------


class TestDefensiveCodexBackfill:
    def test_missing_adapter_is_forward_only(self, monkeypatch):
        """When the Codex backfill adapter hasn't shipped, we report nothing and
        claim no data (honesty) — never crash."""
        import builtins

        real_import = builtins.__import__

        def _no_codex(name, *a, **k):
            if name == "tokenjam.core.ingest_adapters.codex":
                raise ImportError("not shipped yet")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_codex)
        msg, has_data, total = _try_backfill_codex(object())
        assert msg is None
        assert has_data is False
        assert total == 0

    def test_reports_when_adapter_ingests(self, monkeypatch):
        """When the adapter exists and ingests, we report the count and mark
        has_data True."""
        import sys
        import types

        fake = types.ModuleType("tokenjam.core.ingest_adapters.codex")

        class _Result:
            sessions_total = 3
            sessions_new = 2
            total_cost_usd = 4.0

        fake.ingest_codex = lambda db, config=None: _Result()  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules, "tokenjam.core.ingest_adapters.codex", fake,
        )

        class _DB:
            def close(self):
                pass

        monkeypatch.setattr(
            "tokenjam.core.db.open_db", lambda storage: _DB(),
        )

        class _Cfg:
            storage = object()

        msg, has_data, total = _try_backfill_codex(_Cfg())
        assert has_data is True
        assert total == 3
        assert "3 total session" in msg
        assert "2 new" in msg


# --- Shared closing banner ---------------------------------------------------


class TestSetupCompleteHome:
    def test_reports_backfill_count_when_present(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "tokenjam.cli.home.find_config_file", lambda: "/tmp/x.toml",
        )
        _print_setup_complete_home(
            sessions_backfilled=7, has_data=True, days=30,
        )
        out = capsys.readouterr().out
        assert "You're set up." in out
        assert "7 sessions backfilled" in out
        assert "last 30 days" in out
        # No second command list here: the onboard flows print their own
        # curated next-steps block just above this close — a duplicate
        # next-best-actions list was founder-flagged (2026-07). Just the
        # help pointer.
        assert "tj optimize" not in out
        assert "tj --help" in out

    def test_no_count_claim_without_data(self, monkeypatch, capsys):
        """Honesty: never claim a backfill count when nothing was ingested."""
        monkeypatch.setattr(
            "tokenjam.cli.home.find_config_file", lambda: "/tmp/x.toml",
        )
        _print_setup_complete_home(sessions_backfilled=0, has_data=False)
        out = capsys.readouterr().out
        assert "You're set up." in out
        assert "backfilled" not in out.lower()


# --- Combination path: backfill + banner run exactly once (#432) -------------


class TestCombinationPathNoDoubleRun:
    """The combination flow (#432) delegates to `_onboard_claude_code` and
    `_onboard_codex` for wiring, then runs the Codex backfill once and prints the
    closing home banner once at the very end. Before the fix, `_onboard_codex`
    ran its own backfill AND printed the banner, `_onboard_claude_code` printed
    the banner too, and combination did both again — the Codex backfill ran twice
    and the banner printed up to three times.
    """

    def _stub(self, monkeypatch):
        counters: dict[str, int] = {
            "banner": 0, "codex_backfill": 0, "cc_standalone_true": 0,
            "codex_standalone_true": 0,
        }

        def _cc(*a, **k):
            if k.get("standalone", True):
                counters["cc_standalone_true"] += 1

        def _codex(*a, **k):
            if k.get("standalone", True):
                counters["codex_standalone_true"] += 1

        def _backfill(_cfg):
            counters["codex_backfill"] += 1
            return ("1 new (0 already present) · 1 total session", True, 1)

        def _banner(*a, **k):
            counters["banner"] += 1

        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_claude_code", _cc,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_codex", _codex,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._try_backfill_codex", _backfill,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_setup_complete_home", _banner,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_instrument_agent_snippet",
            lambda *a, **k: None,
        )
        # The Codex backfill leg loads the global config; make it a no-op path so
        # the stubbed `_try_backfill_codex` (not disk) is what runs.
        monkeypatch.setattr(
            "tokenjam.core.config.load_config", lambda *a, **k: object(),
        )
        return counters

    def _run(self, monkeypatch, answers, tmp_path):
        import click as _click

        counters = self._stub(monkeypatch)
        replies = iter(answers)
        monkeypatch.setattr(
            _click, "confirm", lambda *a, **k: next(replies),
        )
        # The Codex backfill leg only fires when the global config exists; point
        # HOME at a tmp dir with the file present so it runs (stubbed) once.
        home = tmp_path / "home"
        (home / ".config" / "tj").mkdir(parents=True)
        (home / ".config" / "tj" / "config.toml").write_text('version = "1"\n')
        monkeypatch.setattr("pathlib.Path.home", lambda: home)

        class _Ctx:
            def exit(self, code=0):
                raise SystemExit(code)

        _onboard_combination(
            _Ctx(), None, True, False,
            plan_override=None, project_override=None, verify=False,
        )
        return counters

    def test_banner_prints_exactly_once_all_surfaces(self, monkeypatch, tmp_path):
        # Answer yes to Claude Code, Codex, and SDK.
        counters = self._run(monkeypatch, [True, True, True], tmp_path)
        assert counters["banner"] == 1

    def test_codex_backfill_runs_exactly_once(self, monkeypatch, tmp_path):
        counters = self._run(monkeypatch, [True, True, True], tmp_path)
        assert counters["codex_backfill"] == 1

    def test_sub_onboarders_invoked_non_standalone(self, monkeypatch, tmp_path):
        """The per-path onboarders must be called with standalone=False so they
        skip their own backfill + banner on the combination path."""
        counters = self._run(monkeypatch, [True, True, False], tmp_path)
        # Neither sub-onboarder was called with standalone=True (the default).
        assert counters["cc_standalone_true"] == 0
        assert counters["codex_standalone_true"] == 0

    def test_banner_once_even_codex_only(self, monkeypatch, tmp_path):
        # Claude Code no, Codex yes, SDK no — still exactly one banner, one
        # backfill.
        counters = self._run(monkeypatch, [False, True, False], tmp_path)
        assert counters["banner"] == 1
        assert counters["codex_backfill"] == 1
