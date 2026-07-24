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


def test_bare_onboard_sdk_choice_falls_through_to_generic(_routing_stubs, monkeypatch, tmp_path):
    """Choice 3 (SDK) falls through to the historical generic path, not a
    per-path onboarder."""
    from pathlib import Path

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
    # Isolate the config search paths to a tmp dir so the config lookup doesn't find
    # the developer's real ~/.config/tj/config.toml. This makes the test independent
    # from the host's home directory state. SEARCH_PATHS is evaluated at module import
    # time, so we monkeypatch the global list directly.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [
            Path("tokenjam.toml"),
            Path(".tj/config.toml"),
            home / ".config" / "tj" / "config.toml",
        ],
    )
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
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
    def test_reports_when_adapter_ingests(self, monkeypatch):
        """When the adapter ingests, we report the count and mark
        has_data True."""

        class _Result:
            sessions_total = 3
            sessions_new = 2
            total_cost_usd = 4.0

        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.ingest_codex",
            lambda db, config=None: _Result(),
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
    def test_reports_backfill_count_when_present(self, capsys):
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

    def test_no_count_claim_without_data(self, capsys):
        """Honesty: never claim a backfill count when nothing was ingested."""
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
        # The billing questions are now collected up front (before the legs run);
        # these tests drive the legs as stubs and only assert backfill/banner
        # counts, so short-circuit the up-front collection to a fixed answer
        # instead of reading stdin for the plan prompts.
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._collect_combination_billing",
            lambda *a, **k: ("max_5x", None, 0.0),
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


# --- Combination prompt order: all questions before any long-running work ----


class TestCombinationBillingHoistedUpFront:
    """A combination run must ask EVERY question before any leg's long-running
    backfill starts: Claude billing → Codex billing → project → backfill scope →
    execution. The old shape ran each leg to completion in turn, so the Codex
    plan prompt fired only AFTER the Claude leg's backfill had already run — a
    question stranded behind minutes of ingest. Both legs' billing is now hoisted
    to the top and threaded in, so no leg re-asks.
    """

    def _fresh_home(self, monkeypatch, tmp_path):
        """Isolate HOME to a config-less dir so both legs read as fresh (their
        billing gets collected up front)."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._is_interactive", lambda: True,
        )
        return home

    class _Ctx:
        def exit(self, code=0):
            raise SystemExit(code)

    def test_codex_billing_asked_before_claude_leg_runs(
        self, monkeypatch, tmp_path, capsys,
    ):
        """The core fix: the Codex plan question is asked before the Claude leg
        (and its backfill) executes — proven by sentinels the stubbed legs
        print."""
        import click as _click

        self._fresh_home(monkeypatch, tmp_path)

        from tokenjam.utils.formatting import console

        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_claude_code",
            lambda *a, **k: console.print("::CC_LEG_RAN::"),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_codex",
            lambda *a, **k: console.print("::CODEX_LEG_RAN::"),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._try_backfill_codex",
            lambda _cfg: (None, False, 0),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_setup_complete_home",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_instrument_agent_snippet",
            lambda *a, **k: None,
        )

        # Which do you use? → Claude Code yes, Codex yes, SDK no.
        confirms = iter([True, True, False])
        monkeypatch.setattr(_click, "confirm", lambda *a, **k: next(confirms))
        # Plan prompts: Claude=3 (max_5x, subscription — no ceiling/budget),
        # OpenAI=2 (plus, subscription — no ceiling/budget).
        prompts = iter([3, 2])
        monkeypatch.setattr(_click, "prompt", lambda *a, **k: next(prompts))

        _onboard_combination(
            self._Ctx(), None, True, False,
            plan_override=None, project_override=None, verify=False,
        )
        out = capsys.readouterr().out
        assert "How do you pay for Claude?" in out
        assert "How do you pay for OpenAI / Codex?" in out
        # Both billing prompts precede either leg's execution.
        i_claude = out.index("How do you pay for Claude?")
        i_openai = out.index("How do you pay for OpenAI / Codex?")
        i_cc = out.index("::CC_LEG_RAN::")
        i_codex = out.index("::CODEX_LEG_RAN::")
        assert i_claude < i_openai, out
        # The reported bug: Codex billing came AFTER the Claude backfill. It must
        # now come before the Claude leg runs at all.
        assert i_openai < i_cc, out
        assert i_cc < i_codex, out

    def test_collected_answers_threaded_into_legs(
        self, monkeypatch, tmp_path,
    ):
        """The hoisted billing answers (plan tier, API ceiling, daily budget) are
        threaded into each leg via plan_override / plan_usd_override / budget, so
        the legs re-ask nothing."""
        import click as _click

        self._fresh_home(monkeypatch, tmp_path)

        seen: dict[str, dict] = {}
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_claude_code",
            lambda *a, **k: seen.__setitem__("cc", {"args": a, "kw": k}),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_codex",
            lambda *a, **k: seen.__setitem__("codex", {"args": a, "kw": k}),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._try_backfill_codex",
            lambda _cfg: (None, False, 0),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_setup_complete_home",
            lambda *a, **k: None,
        )

        confirms = iter([True, True, False])
        monkeypatch.setattr(_click, "confirm", lambda *a, **k: next(confirms))
        # Claude=1 (api) → ceiling 50 → daily budget 0 ; OpenAI=2 (plus).
        prompts = iter([1, 50.0, 0.0, 2])
        monkeypatch.setattr(_click, "prompt", lambda *a, **k: next(prompts))

        _onboard_combination(
            self._Ctx(), None, True, False,
            plan_override=None, project_override=None, verify=False,
        )

        cc_kw = seen["cc"]["kw"]
        assert cc_kw["plan_override"] == "api"
        assert cc_kw["plan_usd_override"] == 50.0
        # Daily budget threaded via the positional budget arg (2nd positional).
        assert seen["cc"]["args"][1] == 0.0
        assert cc_kw["standalone"] is False
        codex_kw = seen["codex"]["kw"]
        assert codex_kw["plan_override"] == "plus"
        # Subscription tier → no API ceiling collected.
        assert codex_kw["plan_usd_override"] is None

    def test_existing_stored_plan_not_reprompted(self, monkeypatch, tmp_path):
        """An existing-config re-run keeps a stored plan without re-asking — the
        up-front hoist only fires for a leg the standalone flow would prompt (a
        fresh config or an explicit --plan). No click.prompt is patched here, so
        any stray plan prompt would raise."""
        import click as _click

        home = tmp_path / "home"
        (home / ".config" / "tj").mkdir(parents=True)
        # A config that already stores an anthropic plan → CC billing must NOT be
        # re-collected up front.
        (home / ".config" / "tj" / "config.toml").write_text(
            'version = "1"\n\n[budget.anthropic]\nplan = "max_5x"\n'
        )
        monkeypatch.setattr("pathlib.Path.home", lambda: home)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._is_interactive", lambda: True,
        )

        fired: dict[str, dict] = {}
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_claude_code",
            lambda *a, **k: fired.__setitem__("cc", k),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_setup_complete_home",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_instrument_agent_snippet",
            lambda *a, **k: None,
        )

        # Claude Code yes, Codex no, SDK no.
        confirms = iter([True, False, False])
        monkeypatch.setattr(_click, "confirm", lambda *a, **k: next(confirms))

        _onboard_combination(
            self._Ctx(), None, True, False,
            plan_override=None, project_override=None, verify=False,
        )
        # No up-front plan override was collected — the leg keeps its stored plan.
        assert fired["cc"]["plan_override"] is None
        assert fired["cc"]["plan_usd_override"] is None

    def test_full_combined_order_with_real_claude_leg(
        self, monkeypatch, tmp_path,
    ):
        """End-to-end order with the REAL Claude leg running (driven through the
        CLI with real stdin, so every prompt's text renders): Claude billing →
        Codex billing → project → backfill scope → execution. Project name and
        backfill scope stay inside the Claude leg but now follow BOTH billing
        prompts and precede the (sentinel) Codex leg."""
        from contextlib import contextmanager

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: home)
        # Force the interactive routing (bare onboard → path question →
        # combination) and the interactive backfill-scope menu.
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._is_interactive", lambda: True,
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which", lambda _x: None,
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
        # Real Claude leg, but a no-op backfill: the scope menu still fires
        # (root exists) while the ingest itself touches nothing.
        cc_root = home / "claude-projects"
        cc_root.mkdir()
        monkeypatch.setattr(backfill_mod, "CLAUDE_CODE_PROJECTS_ROOT", cc_root)
        monkeypatch.setattr(
            backfill_mod, "count_claude_code_sessions_in_scope", lambda **k: 0,
        )

        class _Result:
            limit_reached = False
            sessions_total = 0

        monkeypatch.setattr(
            backfill_mod, "ingest_claude_code", lambda *a, **k: _Result(),
        )

        class _DB:
            def close(self):
                pass

        monkeypatch.setattr("tokenjam.core.db.open_db", lambda storage: _DB())

        @contextmanager
        def _fake_progress(total):
            yield lambda *a, **k: None

        monkeypatch.setattr(
            "tokenjam.cli.backfill_progress.backfill_progress", _fake_progress,
        )

        from tokenjam.utils.formatting import console

        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._onboard_codex",
            lambda *a, **k: console.print("::CODEX_LEG_RAN::"),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._try_backfill_codex",
            lambda _cfg: (None, False, 0),
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard._print_setup_complete_home",
            lambda *a, **k: None,
        )

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # path=4 (combination) → Claude yes → Codex yes → SDK no →
            # Claude plan 3 (max_5x) → OpenAI plan 2 (plus) → project name →
            # backfill scope 1 (recent). Subscription tiers skip ceiling/budget.
            res = runner.invoke(
                cmd_onboard, ["--no-daemon"],
                input="4\ny\ny\nn\n3\n2\nmyproj\n1\n", obj={},
            )
        assert res.exit_code == 0, res.output
        out = res.output
        order = [
            "How do you pay for Claude?",
            "How do you pay for OpenAI / Codex?",
            "Project name",
            "Backfill your Claude Code history",
            "::CODEX_LEG_RAN::",
        ]
        indices = [out.index(s) for s in order]
        assert indices == sorted(indices), (order, indices, out)
