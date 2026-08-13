"""`tj status` recoverable teaser (`_recoverable_teaser`, cmd_status.py).

Nothing in `tj status`, `tj doctor`, `tj statusline`, or the banner ever
pointed at `tj optimize` — a user could run tokenjam for months without
learning the command exists.

The teaser READS THE STORED ANALYZER REPORT; it never runs an analyzer. It
used to call `build_report` inline on every plain `tj status`, which cost
roughly a minute of CPU on a real corpus before the status table's own
figures could be printed. `test_status_command_cannot_reach_build_report`
below is the regression pin for that, and
`core/optimize/report_store.py`'s module docstring is the rule it enforces.

`report_store.stored_report` is monkeypatched in the aggregation tests rather
than seeded through real analyzer thresholds: what's under test is the
aggregation + the honesty/silence gates around it, not the analyzers
themselves (those have their own tests). The cold-store test uses a real
config and a real, genuinely empty store directory.

The teaser used to gate on the user's plan tier (`plan_tier_mix` +
`pricing_mode_for`), staying silent for subscription/local plans. That gate
was removed by product decision: dollars are always legitimate, so tj no
longer differentiates its output between subscription and API users — the
teaser now fires for any plan tier with a large-enough recoverable figure.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tokenjam.cli.cmd_status import _recoverable_teaser
from tokenjam.core.config import TjConfig


def _report(*, downgrade_usd=None, finding_usd=None, findings=None):
    built = {}
    if finding_usd is not None:
        built["cache"] = SimpleNamespace(past_overspend_usd=finding_usd)
    if findings:
        built.update(findings)
    downgrade = None
    if downgrade_usd is not None:
        downgrade = SimpleNamespace(past_overspend_usd=downgrade_usd)
    return SimpleNamespace(downgrade=downgrade, findings=built)


def _stored(monkeypatch, report):
    monkeypatch.setattr(
        "tokenjam.core.optimize.report_store.stored_report",
        lambda config=None, **kw: report,
    )


def _config(tmp_path: Path) -> TjConfig:
    """A real config whose store directory is a real, empty tmp dir."""
    cfg = TjConfig(version="1")
    cfg.storage.path = str(tmp_path / "telemetry.duckdb")
    return cfg


def test_teaser_prints_largest_single_estimate_not_a_sum(monkeypatch):
    """The teaser must mirror `largest_recoverable_usd` in api/routes/cost.py:
    downsize and cache price OVERLAPPING angles on the same spans, so
    summing 2.0 + 3.5 into "$5.50" would print an inflated, non-additive
    headline the product's own API disclaims via `recoverable_additive: False`
    and `_recoverable_overlap_note`. The honest figure is the larger of the
    two on its own: $3.50."""
    _stored(monkeypatch, _report(downgrade_usd=2.0, finding_usd=3.5))

    out = _recoverable_teaser(config=object())

    assert out is not None
    assert "$3.50" in out
    assert "$5.50" not in out
    assert "tj optimize" in out


def test_teaser_silent_below_minimum_threshold(monkeypatch):
    _stored(monkeypatch, _report(finding_usd=0.42))

    assert _recoverable_teaser(config=object()) is None


def test_teaser_reads_store_without_a_direct_db_connection(monkeypatch):
    """The daemon holding the write lock (API-shim mode, no `.conn`) used to
    silence the teaser, because it had no connection to run analyzers over.
    Reading a stored figure needs no connection at all — and that mode is
    precisely when the store IS warm, since the daemon is what fills it. The
    teaser takes no `db` argument any more; this pins that the figure still
    renders when there is no local DB handle in play."""
    _stored(monkeypatch, _report(finding_usd=12.0))

    out = _recoverable_teaser(config=object())

    assert out is not None
    assert "$12.00" in out


def test_teaser_shows_for_subscription_plan_too(monkeypatch):
    """The `tj optimize` pointer no longer differentiates by billing mode
    (product decision, reversing the prior subscription/local silence):
    dollars are always legitimate, so a subscription-tier user with a large
    recoverable total gets the same dollar teaser an API user would."""
    _stored(monkeypatch, _report(finding_usd=50.0))

    out = _recoverable_teaser(config=object())
    assert out is not None
    assert "$50.00" in out
    assert "tj optimize" in out


def test_teaser_silent_on_store_read_failure(monkeypatch):
    """Never let the teaser computation break `tj status` itself."""
    def boom(config=None, **kw):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr("tokenjam.core.optimize.report_store.stored_report", boom)

    assert _recoverable_teaser(config=object()) is None


def test_teaser_scopes_the_figure_to_cost_analyzers(monkeypatch):
    """The store holds EVERY analyzer's finding, not just the cost rail's, so
    the `COST_ANALYZERS` scoping the teaser has always claimed now has to be
    applied on the read. A finding under a name outside that set must not be
    able to become the headline."""
    _stored(monkeypatch, _report(
        finding_usd=4.0,
        findings={"not-a-cost-analyzer": SimpleNamespace(past_overspend_usd=999.0)},
    ))

    out = _recoverable_teaser(config=object())

    assert out is not None
    assert "$4.00" in out
    assert "999" not in out


def test_cold_store_renders_a_command_and_no_figure(tmp_path):
    """A store that has never been written is COLD, not zero. `$0.00
    recoverable` off a scan that never ran reads as "you have no waste",
    which is a reassurance the data does not support (root anti-pattern 22).
    The line may name the command that would produce a figure; it may not
    state a quantity."""
    out = _recoverable_teaser(config=_config(tmp_path))

    assert out is not None
    assert "tj optimize" in out
    assert "$" not in out
    assert "0" not in out
    assert "not been scanned" in out


def test_status_command_cannot_reach_build_report(monkeypatch, tmp_path):
    """THE regression pin. `build_report` dispatches every cost analyzer over
    the whole corpus, including the unbounded `relearn` scan; running it on a
    plain `tj status` cost about a minute of CPU per invocation. The status
    module must not be able to reach it at all — so it is neither named in
    the source nor callable from the teaser, even with a warm store."""
    from tokenjam.cli import cmd_status

    # The docstring explains WHY the call is gone, so match the CALL form
    # rather than the bare name.
    source = Path(cmd_status.__file__).read_text(encoding="utf-8")
    assert "build_report(" not in source

    def boom(*a, **kw):
        raise AssertionError("tj status must never dispatch an analyzer")

    monkeypatch.setattr("tokenjam.core.optimize.build_report", boom)
    monkeypatch.setattr("tokenjam.core.optimize.runner.build_report", boom)
    _stored(monkeypatch, _report(finding_usd=7.0))

    out = _recoverable_teaser(config=object())
    assert out is not None
    assert "$7.00" in out

    # And cold, with the same trap armed.
    monkeypatch.undo()
    monkeypatch.setattr("tokenjam.core.optimize.build_report", boom)
    monkeypatch.setattr("tokenjam.core.optimize.runner.build_report", boom)
    assert "$" not in (_recoverable_teaser(config=_config(tmp_path)) or "")
