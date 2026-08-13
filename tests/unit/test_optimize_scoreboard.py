"""The default `tj optimize` screen — the scoreboard.

`tj optimize` used to print every finding card in full on every run:
candidate lists, verbatim caveats, multi-hundred-word methodology
paragraphs. Nothing in it was wrong; it simply asserted no hierarchy, so the
headline numbers, the per-area findings and the next command all sat at equal
weight. The default view is now a scoreboard, the cards are reached by naming
an analyzer (`tj optimize resend`) or by asking for everything (`-v`).

These tests pin the honesty rails that make a summary legitimate:

* method prose (`estimate_basis`, `coverage_note`) and verbatim caveats are
  never paraphrased onto the scoreboard — they stay on the card;
* a finding with no priced figure shows the `—` null marker, never `0` and
  never an empty cell, because zero reads as "no waste";
* an analyzer that ran and found nothing gets no row, but is still counted in
  the `N analyzers` header, so silence is never ambiguous.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
from tests.factories import make_llm_span, make_session

UTC = timezone.utc


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _flat(out: str) -> str:
    """Collapse Rich's terminal-width wrapping so a phrase can be asserted
    whole regardless of the runner's column count."""
    return " ".join(out.split())


def _window():
    return datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 5, 30, tzinfo=UTC)


def _seed_heavy_resend(db, sessions=6, turns=8):
    """A window whose turns re-send most of their prompt — enough to give the
    resend analyzer a real, priced finding to summarise."""
    start = datetime(2026, 5, 10, tzinfo=UTC)
    for s in range(sessions):
        sid = f"s-{s}"
        db.upsert_session(make_session(session_id=sid, plan_tier="api"))
        for i in range(turns):
            db.insert_span(make_llm_span(
                session_id=sid, provider="anthropic", model="claude-sonnet-5",
                input_tokens=20_000 + i * 500, cache_tokens=0, output_tokens=50,
                cost_usd=0.30, start_time=start + timedelta(minutes=i),
            ))


def _real_report(db):
    since, until = _window()
    return build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["resend"],
    )


def _synthetic_report(**findings):
    """A report carrying hand-built findings, so a single attribute (an
    unpriced estimate, an empty candidate list) can be pinned in isolation."""
    since, until = _window()
    return OptimizeReport(
        window=WindowSummary(
            since=since, until=until, days=29.0, sessions=12, spans=100,
            total_tokens=5_000_000, total_cost_usd=120.0, thin_data=False,
        ),
        findings=dict(findings),
    )


# --------------------------------------------------------------------------- #
# What the scoreboard shows
# --------------------------------------------------------------------------- #

def test_scoreboard_prints_header_counts_a_row_and_the_next_block(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    _seed_heavy_resend(db)
    report = _real_report(db)
    _render_scoreboard(report, agent=None, pricing_mode="api", cost_proposal_count=3)
    out = _flat(capsys.readouterr().out)

    assert "12 sessions" not in out  # the real window, not the synthetic one
    assert "sessions ·" in out
    assert "analyzers ·" in out and "finding" in out
    assert "ANALYZERS" in out and "FINDING" in out and "RECOVERABLE" in out
    assert "resend" in out
    # The Next block names literal runnable commands, not advice.
    assert "Next:" in out
    assert "tj relearn cost-proposals" in out
    assert "3 copy-paste fixes" in out
    assert "tj optimize -v" in out


def test_scoreboard_row_reports_the_measured_resend_share(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    _seed_heavy_resend(db)
    report = _real_report(db)
    finding = report.findings["resend"]
    assert finding.repeat_share is not None

    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert f"{finding.repeat_share * 100:.0f}% of prompt tokens re-sent" in out


def test_scoreboard_never_paraphrases_method_prose_or_caveats(db, capsys):
    """The caveat / estimate basis is the card's job and must render there
    verbatim. Summarising it onto the scoreboard would be a paraphrase of the
    one string that is never allowed to be paraphrased."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    _seed_heavy_resend(db)
    report = _real_report(db)
    finding = report.findings["resend"]
    assert finding.caveat and finding.estimate_basis

    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert _flat(finding.caveat) not in out
    assert _flat(finding.estimate_basis) not in out
    # ...but the user is told where they live, by a command they can type.
    assert "method notes and caveats" in out
    assert "tj optimize <analyzer>" in out


def test_scoreboard_omits_the_candidate_detail_the_card_carries(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_report, _render_scoreboard

    _seed_heavy_resend(db)
    report = _real_report(db)

    _render_scoreboard(report, agent=None, pricing_mode="api")
    scoreboard = _flat(capsys.readouterr().out)
    _render_report(report, agent=None, requested=["resend"], pricing_mode="api")
    card = _flat(capsys.readouterr().out)

    assert "Heaviest sessions" in card
    assert "Heaviest sessions" not in scoreboard
    assert len(scoreboard) < len(card)


# --------------------------------------------------------------------------- #
# The two honesty rails
# --------------------------------------------------------------------------- #

def test_an_unpriced_finding_renders_the_null_marker_never_zero(capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(deadweight=SimpleNamespace(
        unused_servers=["some-mcp"],
        past_overspend_usd=None,
        past_overspend_tokens=None,
    ))
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "1 MCP server injected, never invoked" in out
    assert "—" in out
    assert "$0" not in out
    assert " 0 " not in out


def test_stream_usage_is_never_priced_into_the_recoverable_column(capsys):
    """`stream-usage` measures spend that already happened and was never
    recorded. It is the one finding on this screen that is not a saving, so it
    must not borrow the savings column even though it carries a dollar
    figure."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(**{"stream-usage": SimpleNamespace(
        call_sites=[object()],
        streams_missing_usage=12,
        streams_observed=40,
        undercounted_usd=88.0,
        past_overspend_usd=88.0,
        past_overspend_tokens=1_000_000,
    )})
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "12 of 40 streams reported no usage" in out
    assert "—" in out
    assert "$88" not in out


def test_a_clean_analyzer_gets_no_row_but_is_still_counted(capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        verbosity=SimpleNamespace(
            candidates=[], total_candidates=0,
            past_overspend_usd=None, past_overspend_tokens=None,
        ),
        relearn=SimpleNamespace(
            clusters=["a", "b"],
            past_overspend_usd=41.0, past_overspend_tokens=2_000,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    # Ran, found nothing → no row, and no "no waste" claim of any kind.
    assert "verbosity" not in out.split("Next:")[0]
    # ...but the did-it-run signal survives in the header count: three
    # analyzers ran (downsize's empty state, verbosity, relearn), one has a
    # finding.
    assert "3 analyzers · 1 finding" in out
    assert "relearn" in out


def test_no_findings_at_all_says_so_without_a_savings_figure(capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(verbosity=SimpleNamespace(
        candidates=[], total_candidates=0,
        past_overspend_usd=None, past_overspend_tokens=None,
    ))
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "No candidates flagged in this window" in out
    assert "RECOVERABLE" not in out


# --------------------------------------------------------------------------- #
# Framing + degraded terminals
# --------------------------------------------------------------------------- #

def test_subscription_header_uses_implied_api_value_not_spend(capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report()
    _render_scoreboard(
        report, agent=None, plan_mix={"max_20x": 12},
        dominant_plan="max_20x", pricing_mode="subscription",
    )
    out = _flat(capsys.readouterr().out)

    assert "Max 20x" in out
    assert "Implied API value" in out
    assert "spend (last" not in out


def test_unknown_plan_suppresses_dollar_figures(capsys):
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report()
    _render_scoreboard(
        report, agent=None, plan_mix={"unknown": 12}, pricing_mode="unknown",
    )
    out = _flat(capsys.readouterr().out)

    assert "dollar figures suppressed" in out
    assert "$120" not in out


def test_scoreboard_carries_no_em_dash_prose(db, capsys):
    """The `—` in RECOVERABLE is a null marker, not punctuation. Nothing else
    on this screen may use one (see the copy rules in tokenjam/CLAUDE.md)."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    _seed_heavy_resend(db)
    _render_scoreboard(_real_report(db), agent=None, pricing_mode="api",
                       cost_proposal_count=2)
    out = capsys.readouterr().out

    for line in out.splitlines():
        if "—" in line:
            # Only a table row may carry it, and only as a lone cell value.
            assert line.split().count("—") == 1, line


def test_scoreboard_degrades_to_plain_aligned_text_without_colour(db):
    """`--no-color` / a non-TTY must leave the columns aligned and every
    figure readable — the shared console handles the ANSI, the layout must
    not depend on it."""
    from rich.console import Console

    import tokenjam.cli.cmd_optimize as mod
    from tokenjam.utils.theme import TJ_THEME

    _seed_heavy_resend(db)
    report = _real_report(db)

    plain = Console(highlight=False, theme=TJ_THEME, no_color=True,
                    width=100, force_terminal=False, record=True)
    original = mod.console
    mod.console = plain
    try:
        mod._render_scoreboard(report, agent=None, pricing_mode="api")
    finally:
        mod.console = original
    out = plain.export_text()

    assert "\x1b[" not in out
    header = next(ln for ln in out.splitlines() if "ANALYZERS" in ln)
    body = next(ln for ln in out.splitlines() if "resend" in ln and "%" in ln)
    # The FINDING column starts at the same offset in the header and the row.
    assert body.index("resend") == header.index("ANALYZERS")


# --------------------------------------------------------------------------- #
# Row order
# --------------------------------------------------------------------------- #

def _row_order(out: str, names) -> list[str]:
    """The analyzer names in the order their rows render."""
    seen = []
    for line in out.splitlines():
        for name in names:
            if line.strip().startswith(name) and name not in seen:
                seen.append(name)
    return seen


def test_rows_are_ordered_by_the_recoverable_column_they_print(capsys):
    """The rows came out of `_rank_findings`, which orders by reclaimable
    TOKEN share — the right order for the card path it was written for, and
    the wrong one for a table whose visible column is dollars. It rendered
    $48.55 above $296.74, which reads as unsorted."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        summarize=SimpleNamespace(
            candidates=[object()],
            past_overspend_usd=48.55, past_overspend_tokens=9_000_000,
        ),
        deadweight=SimpleNamespace(
            unused_servers=["some-mcp"],
            past_overspend_usd=296.74, past_overspend_tokens=1_000,
        ),
        relearn=SimpleNamespace(
            clusters=[object()],
            past_overspend_usd=253.74, past_overspend_tokens=5_000,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = capsys.readouterr().out

    assert _row_order(out, ("summarize", "deadweight", "relearn")) == [
        "deadweight", "relearn", "summarize",
    ]


def test_unpriced_rows_sort_last_and_keep_the_null_marker(capsys):
    """An unpriced finding is one with no figure, not one worth nothing: it
    goes to the bottom rather than being sorted as a zero, and the cell stays
    the null marker."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        deadweight=SimpleNamespace(
            unused_servers=["some-mcp"],
            past_overspend_usd=None, past_overspend_tokens=None,
        ),
        relearn=SimpleNamespace(
            clusters=[object()],
            past_overspend_usd=12.0, past_overspend_tokens=5_000,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = capsys.readouterr().out

    assert _row_order(out, ("deadweight", "relearn")) == ["relearn", "deadweight"]
    dead_row = next(ln for ln in out.splitlines() if ln.strip().startswith("deadweight"))
    assert "—" in dead_row
    assert "$0" not in dead_row


def test_local_framing_orders_by_the_token_figure_it_prints(capsys):
    """`local` pricing renders tokens, not dollars, so the ordering has to
    follow the same figure — a sort key read from a field the column does not
    show is the same defect wearing a different hat."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        summarize=SimpleNamespace(
            candidates=[object()],
            past_overspend_usd=1.0, past_overspend_tokens=9_000_000,
        ),
        relearn=SimpleNamespace(
            clusters=[object()],
            past_overspend_usd=900.0, past_overspend_tokens=5_000,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="local")
    out = capsys.readouterr().out

    assert _row_order(out, ("summarize", "relearn")) == ["summarize", "relearn"]


def test_next_block_points_at_the_largest_recoverable_finding(capsys):
    """The Next block names `rows[0]`, so an unsorted table also mis-aimed the
    one command it tells the user to run."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        summarize=SimpleNamespace(
            candidates=[object()],
            past_overspend_usd=48.55, past_overspend_tokens=9_000_000,
        ),
        deadweight=SimpleNamespace(
            unused_servers=["some-mcp"],
            past_overspend_usd=296.74, past_overspend_tokens=1_000,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "tj optimize deadweight" in out
    assert "tj optimize summarize" not in out


# --------------------------------------------------------------------------- #
# No total, and a disclosure that says why
# --------------------------------------------------------------------------- #

def test_scoreboard_prints_no_summed_total(capsys):
    """The analyzers price overlapping angles on the same sessions (`downsize`
    excludes `sub_agent_id IS NOT NULL` spans precisely because `subagent`
    already prices the identical swap over them), so a summed headline would
    add waste that was measured twice. The largest single line is the honest
    standalone figure, and sorting puts it on top; nothing is summed."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        summarize=SimpleNamespace(
            candidates=[object()],
            past_overspend_usd=48.55, past_overspend_tokens=9_000_000,
        ),
        deadweight=SimpleNamespace(
            unused_servers=["some-mcp"],
            past_overspend_usd=296.74, past_overspend_tokens=1_000,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "$48.55" in out and "$296.74" in out
    # 48.55 + 296.74, in every rendering `_fmt_usd` could produce.
    assert "345" not in out
    assert "Total" not in out and "total" not in out


def test_two_or_more_priced_rows_carry_the_overlap_disclosure(capsys):
    """A column of dollar figures invites the reader to add it up. The
    disclosure is what stops the sum happening in their head instead of in
    our code."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        summarize=SimpleNamespace(
            candidates=[object()],
            past_overspend_usd=48.55, past_overspend_tokens=9_000_000,
        ),
        deadweight=SimpleNamespace(
            unused_servers=["some-mcp"],
            past_overspend_usd=296.74, past_overspend_tokens=1_000,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "These 2 estimates are computed from overlapping angles on the same sessions" in out
    assert "do not add up" in out
    assert "largest single line is the one to act on first" in out


def test_unpriced_rows_do_not_count_toward_the_disclosure(capsys):
    """One priced row and one `—` row is not two overlapping estimates, so
    there is nothing to disclaim — mirroring `_recoverable_overlap_note`,
    which returns an empty string below two entries."""
    from tokenjam.cli.cmd_optimize import _render_scoreboard

    report = _synthetic_report(
        summarize=SimpleNamespace(
            candidates=[object()],
            past_overspend_usd=48.55, past_overspend_tokens=9_000_000,
        ),
        deadweight=SimpleNamespace(
            unused_servers=["some-mcp"],
            past_overspend_usd=None, past_overspend_tokens=None,
        ),
    )
    _render_scoreboard(report, agent=None, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "overlapping angles" not in out
    assert "—" in out
