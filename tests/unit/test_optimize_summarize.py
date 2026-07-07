"""Track A summarize analyzer — filesystem-derived recoverable finding.

The analyzer reasons over the summarize scan (filesystem), not telemetry, so the
scan is monkeypatched to a controlled ScanResult. Asserts the #111 recoverable
contract: tokens summed from candidates, usd deliberately None (tokens-only), an
explicit basis, and a clean report_to_dict/report_from_dict round-trip.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
from tokenjam.core.summarize.candidates import Candidate, ScanResult
from tokenjam.utils.time_parse import utcnow


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


def _run(db) -> object:
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


def test_empty_scan_yields_no_tokens_but_keeps_basis(db, monkeypatch):
    _patch_scan(monkeypatch, [])
    f = _run(db)
    assert f.files == 0
    assert f.estimated_recoverable_tokens is None
    assert f.estimated_recoverable_usd is None
    assert f.estimate_basis


def test_scan_error_never_breaks_the_report(db, monkeypatch):
    def boom(**kw):
        raise OSError("disk gone")
    monkeypatch.setattr("tokenjam.core.summarize.candidates.list_candidates", boom)
    f = _run(db)
    assert f.files == 0 and f.estimated_recoverable_tokens is None


def test_finding_round_trips(db, monkeypatch):
    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410)])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    payload = report_to_dict(report)
    assert payload["findings"]["summarize"]["estimated_recoverable_tokens"] == 410
    assert payload["findings"]["summarize"]["estimated_recoverable_usd"] is None
    back = report_from_dict(payload).findings["summarize"]
    assert back.files == 1
    assert back.estimated_recoverable_tokens == 410
    assert back.candidates[0].path == "./CLAUDE.md"
