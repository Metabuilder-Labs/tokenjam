"""Unit tests for `tj summarize calibrate` — producing the rewrite evidence.

Every test routes the staging dir at a tmp dir via `config.storage.path` and
patches the delivery handler, so nothing here touches the developer's real
`~/.tj` and nothing ever issues a model call.
"""
from __future__ import annotations

import re

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize import calibrate, estimate, session
from tokenjam.core.summarize.delivery import DeliveryError, DeliveryResult

PROSE = "Always act carefully and never drop a required step when you respond. " * 40
_MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """Three real CLAUDE.md-class files of decreasing prose size, in a fake scan."""
    from tokenjam.core.summarize.candidates import Candidate, ScanResult

    made = []
    for name, mult in (("small.md", 2), ("huge.md", 8), ("mid.md", 4)):
        f = tmp_path / name
        # A protected block is deliberately present: structure survival is the
        # only gate `check` applies, so a fixture with none could never fail it.
        f.write_text(PROSE * mult + "\n```\nkeep = 'me'\n```\n", encoding="utf-8")
        made.append(Candidate(
            path=str(f), prose_words=len(PROSE.split()) * mult,
            total_chars=len(PROSE) * mult, protected_blocks=0,
            est_tokens_saved=100 * mult, pricing_mode="api", scope="project",
            is_prompt=True,
        ))

    def fake_scan(path=None, **kw):
        return ScanResult(candidates=list(made), root=str(tmp_path), recursive=False,
                          globals_checked=0, walk_capped=False, note="")

    monkeypatch.setattr("tokenjam.core.summarize.calibrate.list_candidates", fake_scan)
    return made


def _shrinking_delivery(keep_fraction: float):
    """A delivery handler that returns a structurally-valid, shorter rewrite."""
    def _deliver(config, mode, wrapped_prompt, system_rules):
        markers = _MARKER_RE.findall(wrapped_prompt)
        words = [w for w in wrapped_prompt.split() if not w.startswith("<tj-keep")]
        kept = words[: max(1, int(len(words) * keep_fraction))]
        return DeliveryResult(summary=" ".join(kept + markers))
    return _deliver


# --------------------------------------------------------------------------- #
# Bounded and consented: a dry run spends nothing, and the cap is a hard cap.
# --------------------------------------------------------------------------- #

def test_dry_run_makes_no_model_call_and_says_what_it_would_cost(cfg, catalog, monkeypatch):
    def _explode(*a, **k):                      # any delivery here is a bug
        raise AssertionError("a dry run must never call a model")

    monkeypatch.setattr("tokenjam.core.summarize.delivery.deliver", _explode)

    report = calibrate.run_calibration(cfg, via="claude-p")

    assert report.dry_run is True
    assert report.samples == []
    assert len(report.planned) == calibrate.DEFAULT_SAMPLES
    assert "billed to you" in report.note
    assert "--go" in report.note
    assert report.rewrite_usd is None


def test_sample_size_is_capped_however_large_the_limit(cfg, catalog):
    """A typo in --limit must not turn into an unbounded spend."""
    planned = calibrate.plan_calibration(cfg, limit=9_999)
    assert len(planned) <= calibrate.MAX_SAMPLES
    assert calibrate.plan_calibration(cfg, limit=0) != []       # floored at one


def test_the_largest_prose_candidates_are_sampled_first(cfg, catalog):
    """The ratio is weighted by prose volume, so the biggest files buy the most
    evidence per call spent."""
    planned = calibrate.plan_calibration(cfg, limit=2)

    assert [p.path.rsplit("/", 1)[-1] for p in planned] == ["huge.md", "mid.md"]
    assert planned[0].prose_words > planned[1].prose_words


def test_an_already_sampled_file_is_not_paid_for_twice(cfg, catalog, monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.deliver", _shrinking_delivery(0.4))
    calibrate.run_calibration(cfg, via="claude-p", limit=1, go=True)

    planned = calibrate.plan_calibration(cfg, limit=3)

    assert all("huge.md" not in p.path for p in planned)


# --------------------------------------------------------------------------- #
# What the run learns, and how honestly it reports having learned it.
# --------------------------------------------------------------------------- #

def test_a_run_produces_the_measured_ratio_the_estimate_was_waiting_for(
        cfg, catalog, monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.deliver", _shrinking_delivery(0.4))

    assert estimate.observed_prose_ratio(cfg) == (None, 0)     # nothing to go on

    report = calibrate.run_calibration(cfg, via="claude-p", limit=3, go=True)

    assert report.dry_run is False
    assert len(report.samples) == 3
    assert all(s.staged for s in report.samples)
    assert report.ratio_before is None and report.samples_before == 0
    assert report.ratio_after is not None and report.samples_after == 3
    assert 0.0 <= report.ratio_after <= 1.0
    assert estimate.observed_prose_ratio(cfg)[0] == pytest.approx(report.ratio_after)
    assert "measured ratio" in report.note


def test_insufficient_evidence_falls_back_without_zeroing_or_hiding(
        cfg, catalog, monkeypatch):
    """One usable sample is below the credibility gate. The ratio stays None —
    it does NOT become a zero, and the run still reports what it found."""
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.deliver", _shrinking_delivery(0.4))

    report = calibrate.run_calibration(cfg, via="claude-p", limit=1, go=True)

    assert len(report.samples) == 1
    assert report.samples[0].achieved_ratio is not None        # the sample is real
    assert report.ratio_after is None                          # ...but not yet credible
    assert report.samples_after == 1
    assert "Not enough evidence yet" in report.note
    assert "more structure-checked rewrite" in report.note


def test_a_structure_gate_failure_is_recorded_as_evidence_not_dropped(
        cfg, catalog, monkeypatch):
    def _mangle(config, mode, wrapped_prompt, system_rules):
        return DeliveryResult(summary="Short prose with every marker discarded.")

    monkeypatch.setattr("tokenjam.core.summarize.delivery.deliver", _mangle)

    report = calibrate.run_calibration(cfg, via="claude-p", limit=2, go=True)

    assert report.gate_failures == 2
    assert all(not s.structure_ok and not s.staged for s in report.samples)
    # None, not 0.0: "we learned no ratio here" is not "it compressed to nothing".
    assert all(s.achieved_ratio is None for s in report.samples)
    assert all(s.error for s in report.samples)
    assert len(session.list_attempts(cfg)) == 2                # the finding persists
    assert session.list_staged(cfg) == []
    assert report.ratio_after is None


def test_one_delivery_failure_does_not_abandon_the_rest_of_the_sample(
        cfg, catalog, monkeypatch):
    """The run is already paid for; the next candidate may well succeed."""
    calls = {"n": 0}
    ok = _shrinking_delivery(0.4)

    def _flaky(config, mode, wrapped_prompt, system_rules):
        calls["n"] += 1
        if calls["n"] == 1:
            raise DeliveryError("`claude -p` returned nothing.")
        return ok(config, mode, wrapped_prompt, system_rules)

    monkeypatch.setattr("tokenjam.core.summarize.delivery.deliver", _flaky)

    report = calibrate.run_calibration(cfg, via="claude-p", limit=3, go=True)

    assert len(report.samples) == 3
    assert report.samples[0].achieved_ratio is None
    assert "returned nothing" in report.samples[0].error
    assert sum(1 for s in report.samples if s.staged) == 2


def test_an_unpriced_path_reports_unknown_cost_rather_than_zero(
        cfg, catalog, monkeypatch):
    """`claude -p` spends the user's quota. A $0.00 would read as "this was
    free", which is a quiet lie in our favour."""
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.deliver", _shrinking_delivery(0.4))

    report = calibrate.run_calibration(cfg, via="claude-p", limit=1, go=True)

    assert report.rewrite_usd is None
    assert report.to_dict()["rewrite_usd"] is None
