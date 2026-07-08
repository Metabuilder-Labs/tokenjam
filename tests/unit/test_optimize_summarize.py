"""Track A summarize analyzer — filesystem-derived recoverable finding.

The analyzer reasons over the summarize scan (filesystem), not telemetry, so the
scan is monkeypatched to a controlled ScanResult. Asserts the #111 recoverable
contract: tokens summed from candidates, usd deliberately None (tokens-only), an
explicit basis, and a clean report_to_dict/report_from_dict round-trip.
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
    assert f.estimated_recoverable_tokens == 710
    assert f.estimated_recoverable_usd is None       # tokens-only by design
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


def test_finding_round_trips(db, monkeypatch):
    _seed_window(db)
    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410)])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    payload = report_to_dict(report)
    sd = payload["findings"]["summarize"]
    assert sd["estimated_recoverable_tokens"] == 410
    assert sd["estimated_recoverable_usd"] is None
    assert "meaning may change" in sd["caveat"]       # caveat survives serialization
    assert sd["reduction_pct"] == 33 and sd["avg_reduction_pct"] == 33
    back = report_from_dict(payload).findings["summarize"]
    assert back.files == 1
    assert back.estimated_recoverable_tokens == 410
    assert back.candidates[0].path == "./CLAUDE.md"
    assert back.candidates[0].reduction_pct == 33     # per-file % survives the round-trip
    assert "meaning may change" in back.caveat        # and the caveat survives the ctor
    assert back.reduction_pct == 33 and back.avg_reduction_pct == 33
