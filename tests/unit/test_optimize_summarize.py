"""Track A summarize analyzer — filesystem-derived recoverable finding.

The analyzer reasons over the summarize scan (filesystem), not telemetry, so the
scan is monkeypatched to a controlled ScanResult. Asserts the #111 recoverable
contract: `estimated_recoverable_tokens` and `estimated_recoverable_usd` are on
the SAME window-priced basis — both None when no loading session is
observed, both populated together when one is — while the one-time per-call
aggregate lives in `file_reduction_tokens`. Also checks an explicit basis and a
clean report_to_dict/report_from_dict round-trip.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
from tokenjam.core.summarize.candidates import Candidate, ScanResult
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


def _cand(path: str, saved: int, *, is_prompt: bool = True, scope: str = "repo") -> Candidate:
    return Candidate(
        path=path, prose_words=saved * 3, total_chars=saved * 12,
        protected_blocks=0, est_tokens_saved=saved, pricing_mode="api",
        scope=scope, is_prompt=is_prompt,
    )


def _patch_scan(monkeypatch, cands: list[Candidate]) -> None:
    result = ScanResult(candidates=cands, root=".", recursive=False,
                        globals_checked=0, walk_capped=False, note="")
    monkeypatch.setattr(
        "tokenjam.core.summarize.candidates.list_candidates",
        lambda **kw: result,
    )


def _seed_window(db) -> None:
    """One qualifying LLM call in the window so the analyzer runs — it
    window-guards on a dead window (no telemetry → no per-call saving to attach).
    Content is irrelevant; the finding is filesystem-derived."""
    db.upsert_session(make_session(session_id="s0"))
    db.insert_span(make_llm_span(session_id="s0", start_time=utcnow() - timedelta(days=1)))


def _run(db) -> object:
    _seed_window(db)
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    return report.findings["summarize"]


def test_sums_per_call_tokens_drops_zero_saving(db, monkeypatch):
    _patch_scan(monkeypatch, [
        _cand("./CLAUDE.md", 410),
        _cand("./AGENTS.md", 300),
        _cand("./docs/x.md", 0, is_prompt=False),   # no saving → dropped
    ])
    f = _run(db)
    assert f.files == 2
    # One-time, per-call aggregate — always available regardless of session
    # evidence; feeds the curate/diff UI and reduction_pct.
    assert f.file_reduction_tokens == 710
    # No session in this window is observed loading a repo-scoped "./CLAUDE.md"
    # (its ancestor names match no agent_id-derived repo), so NEITHER window
    # figure is attached — tokens and dollars are on the same basis:
    # never a zero, never a rate borrowed from a file that did resolve.
    assert f.estimated_recoverable_tokens is None
    assert f.estimated_recoverable_usd is None
    assert f.estimate_confidence == "heuristic"
    assert f.estimate_basis                          # explicit basis required by contract
    assert {c.path for c in f.candidates} == {"./CLAUDE.md", "./AGENTS.md"}
    # Mandatory honesty caveat carried as the field default (Rule 14).
    assert "meaning may change" in f.caveat
    # Prose reduction %s computed server-side (_cand sets total_chars = saved*12,
    # so source tokens = saved*3 → every file reduces ~33%). Per-file + aggregate.
    assert all(c.reduction_pct == 33 for c in f.candidates)
    assert f.reduction_pct == 33
    assert f.avg_reduction_pct == 33


def test_dead_window_contributes_nothing(db, monkeypatch):
    # No telemetry in the window → no per-call saving to attach; the analyzer must
    # NOT scan the filesystem and must emit no recoverable figure (#211 invariant).
    def must_not_run(**kw):
        raise AssertionError("summarize scan ran on a dead telemetry window")
    monkeypatch.setattr("tokenjam.core.summarize.candidates.list_candidates", must_not_run)
    since, until = _window()   # empty db → total_tokens == 0
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    f = report.findings["summarize"]
    assert f.files == 0
    assert f.estimated_recoverable_tokens is None
    assert f.reduction_pct is None


def test_empty_scan_yields_no_tokens_but_keeps_basis(db, monkeypatch):
    _patch_scan(monkeypatch, [])
    f = _run(db)
    assert f.files == 0
    assert f.estimated_recoverable_tokens is None
    assert f.estimated_recoverable_usd is None
    assert f.estimate_basis


def test_scan_error_never_breaks_the_report(db, monkeypatch, caplog):
    def boom(**kw):
        raise OSError("disk gone")
    monkeypatch.setattr("tokenjam.core.summarize.candidates.list_candidates", boom)
    with caplog.at_level(logging.DEBUG, logger="tokenjam.core.optimize.analyzers.summarize"):
        f = _run(db)
    assert f.files == 0 and f.estimated_recoverable_tokens is None
    # the swallow must leave a trail (not silent) so a real regression is diagnosable
    assert any("scan failed" in r.message for r in caplog.records)


# --- CLI text-view rendering regression --------------------------------------
# Same class of defect deadweight/relearn hit before: `summarize` was registered
# in ANALYZER_REGISTRY and ran on every report, but had no entry in cmd_optimize's
# _FINDING_RENDERERS dispatch table, so `_rank_findings` silently dropped it and
# plain-text `tj optimize` never showed it -- only `--json` did.

def test_summarize_in_click_choices_and_renderer():
    from tokenjam.cli.cmd_optimize import (
        _FINDING_RENDERERS,
        _MINOR_FINDING_LABELS,
        cmd_optimize,
    )

    findings_param = next(
        p for p in cmd_optimize.params if getattr(p, "name", None) == "findings"
    )
    assert "summarize" in findings_param.type.choices
    assert "summarize" in _FINDING_RENDERERS
    assert "summarize" in _MINOR_FINDING_LABELS


def test_render_summarize_lists_candidates_and_points_to_summarize_list(db, monkeypatch, capsys):
    """The finding renders through the CLI dispatch path, names the files, the
    per-call token saving, and points at `tj summarize`.

    These candidates resolve to no loading session, so there is no dollar
    figure to show and none is fabricated. The inverse — that a resolvable file
    DOES surface its dollars — is pinned by
    `test_render_summarize_shows_the_window_dollar_figure_when_priced`.
    """
    from tokenjam.cli.cmd_optimize import _render_summarize

    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410), _cand("./AGENTS.md", 300)])
    finding = _run(db)
    assert finding.estimated_recoverable_usd is None

    for mode in ("api", "subscription", "local", "unknown"):
        _render_summarize(finding, pricing_mode=mode, marker="①")
    out = capsys.readouterr().out

    assert "CLAUDE.md" in out
    assert "AGENTS.md" in out
    assert "710" in out or "710" in out.replace(",", "")  # aggregate tokens saved
    assert "tj summarize list" in out
    assert "tj summarize prep" in out
    assert "$" not in out  # no fabricated dollar figure


# --- Always-on files save REPEATEDLY, so the figure is priced per window ------

def test_global_scope_file_is_priced_across_every_session_in_the_window(
    db, monkeypatch,
):
    """A `~/.claude` file is loaded by every session and re-read on every call
    within one, so its reduction is worth reduction x sessions x (input rate +
    later calls at the cache-read rate) — not a one-time figure."""
    from tokenjam.core.pricing import get_rates

    db.upsert_session(make_session(session_id="s1"))
    db.upsert_session(make_session(session_id="s2"))
    for sid in ("s1", "s2"):
        for _ in range(3):
            db.insert_span(make_llm_span(
                session_id=sid, provider="anthropic", model="claude-haiku-4-5",
                input_tokens=100, output_tokens=10,
                start_time=utcnow() - timedelta(days=1),
            ))
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 1_000, scope="global")])

    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    f = report.findings["summarize"]

    rates = get_rates("anthropic", "claude-haiku-4-5")
    # 2 sessions x 3 calls each: one send at the input rate, two re-reads.
    expected = (
        1_000
        * (rates.input_per_mtok / 1_000_000)
        * (1 + (rates.cache_read_per_mtok / rates.input_per_mtok) * 2)
        * 2
    )
    assert f.sessions_examined == 2
    assert f.calls_per_session == 3.0
    assert f.candidates[0].sessions_loading == 2
    assert f.estimated_recoverable_usd == pytest.approx(round(expected, 6))
    assert f.rate_basis
    # Strictly more than a single send: the saving recurs, it is not one-time.
    assert f.estimated_recoverable_usd > 1_000 * rates.input_per_mtok / 1_000_000


def test_token_and_dollar_aggregates_describe_the_same_quantity(db, monkeypatch):
    """Basis-coherence guard: dividing the dollar aggregate by the token
    aggregate must land on the blended per-token rate the basis advertises.

    Before the fix `estimated_recoverable_tokens` was the un-multiplied one-time
    file reduction while `estimated_recoverable_usd` was per-call-multiplied, so
    the implied rate came out orders of magnitude above any real price — a card
    whose own tokens and dollars described different quantities. Both fields are
    now the same event count (reduction x reads x loading sessions), one counted
    and one priced.
    """
    from tokenjam.core.pricing import get_rates

    for sid in ("s1", "s2"):
        db.upsert_session(make_session(session_id=sid))
        for _ in range(3):
            db.insert_span(make_llm_span(
                session_id=sid, provider="anthropic", model="claude-haiku-4-5",
                input_tokens=100, output_tokens=10,
                start_time=utcnow() - timedelta(days=1),
            ))
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 1_000, scope="global")])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    # 1,000 tokens x 3 reads per session (one send + two re-reads) x 2 sessions.
    assert f.estimated_recoverable_tokens == 6_000
    # The one-time per-call reduction keeps its own field and its own basis.
    assert f.file_reduction_tokens == 1_000

    rates = get_rates("anthropic", "claude-haiku-4-5")
    input_per_token = rates.input_per_mtok / 1_000_000
    cache_read_per_token = rates.cache_read_per_mtok / 1_000_000
    implied = f.estimated_recoverable_usd / f.estimated_recoverable_tokens
    # One send at the input rate + two re-reads at the cache-read rate, over 3
    # token-events: the exact blend the basis string states.
    assert implied == pytest.approx((input_per_token + 2 * cache_read_per_token) / 3)
    # And it stays inside the real price band — the property that fails loudly if
    # either field ever drifts back onto its own basis.
    assert cache_read_per_token <= implied <= input_per_token


def test_project_scope_file_is_priced_against_its_own_repos_calls_per_session(
    db, monkeypatch,
):
    """A project-scope candidate must price against its OWN repo's average
    calls-per-session -- never the window-wide blend across every other
    repo/agent in the window, which can differ sharply from this repo's own
    session behavior.

    Repo A: one session, 5 calls (a heavy call-count repo). Repo B: four
    sessions, one call each (a light one). The window-wide average
    (9 calls / 5 sessions = 1.8, rounds to 2) is nowhere near repo A's own
    average (5.0) -- pricing repo A's file against the blend would understate
    its recoverable figure.
    """
    from tokenjam.core.pricing import get_rates

    db.upsert_session(make_session(session_id="a1", agent_id="claude-code-repo-a"))
    for i in range(5):
        db.insert_span(make_llm_span(
            session_id="a1", agent_id="claude-code-repo-a",
            provider="anthropic", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=10,
            start_time=utcnow() - timedelta(days=1, minutes=i),
        ))
    for i in range(4):
        sid = f"b{i}"
        db.upsert_session(make_session(session_id=sid, agent_id="claude-code-repo-b"))
        db.insert_span(make_llm_span(
            session_id=sid, agent_id="claude-code-repo-b",
            provider="anthropic", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=10,
            start_time=utcnow() - timedelta(days=1),
        ))

    _patch_scan(monkeypatch, [_cand("repo-a/CLAUDE.md", 1_000, scope="repo")])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])

    f = report.findings["summarize"]
    candidate = f.candidates[0]
    rates = get_rates("anthropic", "claude-haiku-4-5")
    ratio = rates.cache_read_per_mtok / rates.input_per_mtok

    assert candidate.sessions_loading == 1  # only repo-a's own session matches
    # Repo A's own average is 5 calls/session -> 4 rereads, never the
    # window-wide blend's rounded 2 calls/session -> 1 reread.
    expected = 1_000 * (rates.input_per_mtok / 1_000_000) * (1 + ratio * 4)
    wrong_window_blend = 1_000 * (rates.input_per_mtok / 1_000_000) * (1 + ratio * 1)
    assert candidate.est_usd_saved == pytest.approx(round(expected, 6))
    assert candidate.est_usd_saved != pytest.approx(round(wrong_window_blend, 6))
    assert candidate.est_usd_saved > wrong_window_blend


def test_render_summarize_shows_the_window_dollar_figure_when_priced(
    db, monkeypatch, capsys,
):
    from tokenjam.cli.cmd_optimize import _render_summarize

    db.upsert_session(make_session(session_id="s1"))
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-haiku-4-5",
        input_tokens=100, output_tokens=10, start_time=utcnow() - timedelta(days=1),
    ))
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 5_000_000, scope="global")])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    finding = report.findings["summarize"]
    assert finding.estimated_recoverable_usd

    _render_summarize(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out
    assert "$" in out
    # Rich soft-wraps, so match on a fragment that survives a line break.
    assert "session(s) in this" in out


def test_render_summarize_empty_state_names_the_reason(db, monkeypatch, capsys):
    from tokenjam.cli.cmd_optimize import _render_summarize

    _patch_scan(monkeypatch, [])
    finding = _run(db)

    _render_summarize(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "No catalog prompt files" in out
    assert "tj summarize list" not in out  # no remedy pointer on an empty finding


def test_render_report_surfaces_summarize_instead_of_generic_empty(db, monkeypatch, capsys):
    """End-to-end: a report whose only finding is a populated summarize set
    must not fall through to the generic "No candidates flagged" empty state."""
    from tokenjam.cli.cmd_optimize import _render_report

    _seed_window(db)
    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410)])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    assert report.findings["summarize"].candidates  # sanity

    _render_report(report, agent=None, requested=["summarize"], pricing_mode="local")
    out = capsys.readouterr().out

    assert "No candidates flagged" not in out
    assert "CLAUDE.md" in out


def test_finding_round_trips(db, monkeypatch):
    _seed_window(db)
    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410)])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    payload = report_to_dict(report)
    sd = payload["findings"]["summarize"]
    # No session in this window loads a repo-scoped "./CLAUDE.md", so neither
    # window figure resolves; the one-time aggregate still does.
    assert sd["file_reduction_tokens"] == 410
    assert sd["estimated_recoverable_tokens"] is None
    assert sd["estimated_recoverable_usd"] is None
    assert "meaning may change" in sd["caveat"]       # caveat survives serialization
    assert sd["reduction_pct"] == 33 and sd["avg_reduction_pct"] == 33
    back = report_from_dict(payload).findings["summarize"]
    assert back.files == 1
    assert back.file_reduction_tokens == 410
    assert back.estimated_recoverable_tokens is None
    assert back.candidates[0].path == "./CLAUDE.md"
    assert back.candidates[0].reduction_pct == 33     # per-file % survives the round-trip
    assert "meaning may change" in back.caveat        # and the caveat survives the ctor
    assert back.reduction_pct == 33 and back.avg_reduction_pct == 33
