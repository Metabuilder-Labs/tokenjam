"""An unpriced model must be VISIBLE in the cost views, not just in a log line.

When `get_rates` finds no row, `calculate_cost` prices the span at a flat
default rate and logs one warning per process. The dollar figure that reaches
the dashboard is indistinguishable from a real one — which is how a benchmark
replay could be wrong by 5-30x for most of its models while every surface
looked fine. The stored `pricing_source` column already records the provenance;
this pins that the cost views actually read it.
"""

from __future__ import annotations

import pytest

from tokenjam.core.pricing_coverage import (
    PricingCoverage,
    coverage_note,
    summarize_pricing_coverage,
)


class _FakeConn:
    """Minimal stand-in returning one canned row for the coverage query."""

    def __init__(self, row):
        self._row = row
        self.executed: list[tuple[str, list]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params or []))
        return self

    def fetchall(self):
        return self._row


def test_no_conn_reports_nothing_rather_than_a_clean_bill():
    cov = summarize_pricing_coverage(None, None, None, None)
    assert cov.measured is False
    assert cov.unpriced_call_count == 0
    assert coverage_note(cov) is None


def test_a_fully_priced_window_produces_no_note():
    conn = _FakeConn([])
    cov = summarize_pricing_coverage(conn, None, None, None)
    assert cov.measured is True
    assert cov.unpriced_models == ()
    assert cov.unpriced_call_count == 0
    assert coverage_note(cov) is None


def test_unpriced_models_are_named_with_their_call_share():
    conn = _FakeConn([("anthropic", "claude-mystery-9", 120, 3.5)])
    cov = summarize_pricing_coverage(conn, None, None, None)
    assert cov.measured is True
    assert cov.unpriced_call_count == 120
    assert cov.unpriced_models == (("anthropic", "claude-mystery-9", 120),)
    note = coverage_note(cov)
    assert note is not None
    assert "claude-mystery-9" in note
    # The claim the user needs: the number is a default-rate estimate, not a
    # quoted price. It must not read as a $0 or as a confirmed figure.
    assert "default rate" in note
    assert "$0" not in note


def test_the_note_reports_every_unpriced_model_it_was_given():
    conn = _FakeConn([
        ("anthropic", "model-a", 10, 1.0),
        ("openai", "model-b", 5, 0.5),
    ])
    cov = summarize_pricing_coverage(conn, None, None, None)
    assert cov.unpriced_call_count == 15
    note = coverage_note(cov)
    assert note is not None
    assert "model-a" in note and "model-b" in note


def test_filters_are_parameterised_never_interpolated():
    conn = _FakeConn([])
    summarize_pricing_coverage(conn, "agent-1", None, None)
    sql, params = conn.executed[0]
    assert "agent-1" not in sql
    assert "agent-1" in params


def test_a_dataclass_instance_is_immutable():
    cov = PricingCoverage(measured=True, unpriced_models=(), unpriced_call_count=0,
                          unpriced_cost_usd=0.0)
    with pytest.raises(Exception):
        cov.measured = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The CLI must render the warning on BOTH paths
# ---------------------------------------------------------------------------
#
# `tj cost` reaches the DB two ways: directly, and — whenever `tj serve` holds
# the DuckDB lock, which is the mode most installs actually run in — through
# ApiBackend. The coverage check only ever consulted `db.conn`, so on the
# daemon path a default-rate cost figure rendered with no warning at all,
# even though the API response already carried a fully-computed block.

class _NoConnBackend:
    """An ApiBackend-shaped stand-in: no `.conn`, fetches over HTTP."""

    def __init__(self, block, *, raises=False):
        self._block = block
        self._raises = raises
        self.calls: list[dict] = []

    def fetch_pricing_coverage(self, *, since="7d", agent_id=None):
        self.calls.append({"since": since, "agent_id": agent_id})
        if self._raises:
            raise RuntimeError("daemon went away")
        return self._block


def _render(db, capsys, agent=None):
    from tokenjam.cli.cmd_cost import _print_pricing_coverage

    _print_pricing_coverage(db, agent, "7d", None)
    return capsys.readouterr().out


def test_daemon_path_renders_the_unpriced_warning(capsys):
    """The regression: this path printed nothing at all."""
    db = _NoConnBackend({
        "measured": True,
        "unpriced_call_count": 12,
        "unpriced_cost_usd": 3.5,
        "unpriced_models": [
            {"provider": "acme", "model": "mystery-1", "call_count": 12},
        ],
        "note": "acme/mystery-1 is not in the pricing table: 12 calls here "
                "are estimated at tokenjam's default rate.",
    })

    out = _render(db, capsys)

    assert "acme/mystery-1" in out
    assert db.calls == [{"since": "7d", "agent_id": None}]


def test_daemon_path_stays_silent_when_everything_was_priced(capsys):
    """Measured and clean is the one case that legitimately says nothing."""
    db = _NoConnBackend({
        "measured": True, "unpriced_call_count": 0, "unpriced_cost_usd": 0.0,
        "unpriced_models": [], "note": None,
    })

    assert _render(db, capsys).strip() == ""


@pytest.mark.parametrize(
    "db",
    [
        _NoConnBackend(None),
        _NoConnBackend(None, raises=True),
        _NoConnBackend({"measured": False, "note": None}),
    ],
    ids=["no-block", "fetch-failed", "explicitly-unmeasured"],
)
def test_an_unmeasured_window_says_so_instead_of_going_quiet(db, capsys):
    """Silence and a clean bill must not look the same.

    An older daemon, a failed call and a window the server could not measure
    all mean "nobody checked" — and staying quiet there is indistinguishable
    from "checked, all good", which is the false all-clear this whole module
    exists to prevent.
    """
    out = _render(db, capsys)

    assert "not checked" in out


def test_the_agent_filter_reaches_the_daemon(capsys):
    """A scoped table must get a scoped coverage answer, not a global one."""
    db = _NoConnBackend({"measured": True, "note": None, "unpriced_models": []})

    _render(db, capsys, agent="chat-service")

    assert db.calls == [{"since": "7d", "agent_id": "chat-service"}]
