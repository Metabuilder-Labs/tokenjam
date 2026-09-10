"""The statusline's top-driver suffix must not go dark, or go dark silently.

Two halves of one defect (#754), and each half needs its own pin because
either one alone still loses the driver:

1. **Nothing refreshed the cache but a backfill.** `refresh_attribution_cache`
   had exactly one caller — `ingest_claude_code` — so on a machine that never
   runs `tj backfill` the entry aged past its TTL and the suffix vanished for
   good. The daemon's analyzer pass already holds a DuckDB connection and the
   `[capture]` flags at the end of every cycle, so it is where the refresh
   belongs; `test_the_daemon_pass_refreshes_the_attribution_cache` pins that it
   actually runs there rather than merely existing as a callable.
2. **Stale and absent rendered identically.** A cache past the TTL was dropped,
   so the line degraded to a bare percentage with nothing saying a richer one
   existed — the figure stayed true and the reader could not tell they were
   seeing less. Stale is the ONE case where we hold the answer and choose not
   to show it, so it renders age-marked and must be DISTINGUISHABLE from a
   missing cache.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from tests.factories import (
    make_claude_transcript_assistant_line,
    write_claude_transcript,
)
from tokenjam.cli.cmd_statusline import REREAD_CRIT, render_line
from tokenjam.core.attribution_cache import MAX_CACHE_AGE_SECONDS
from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import report_store, scan_cycle
from tokenjam.utils.time_parse import utcnow


@pytest.fixture(autouse=True)
def _isolate_attribution_cache(tmp_path, monkeypatch):
    """Never the real ~/.local/share/tj/attribution_cache.json — the host's own
    backfill state must not decide whether these pass."""
    monkeypatch.setattr(
        "tokenjam.core.attribution_cache._cache_path",
        lambda: tmp_path / "attribution_cache.json",
    )


def _write_cache(tmp_path, *, age_days: float, label: str = "CLAUDE.md") -> None:
    (tmp_path / "attribution_cache.json").write_text(json.dumps({
        "top_label": label, "occurrences": 14, "sessions": 3,
        "inclusion_type": "file_read",
        "computed_at": (utcnow() - timedelta(days=age_days)).isoformat(),
    }))


def _line(tmp_path, reread_pct: float = REREAD_CRIT + 1) -> str:
    """A statusline rendered past the WARN threshold (where the driver shows)."""
    reread = int(round(1000 * reread_pct / 100.0))
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=1000 - reread, output_tokens=0,
            cache_read_input_tokens=reread, cache_creation_input_tokens=0,
        ),
    ])
    return render_line({"model": "Opus 4.8", "transcript_path": path})


# --------------------------------------------------------------------------- #
# 1. The daemon pass is a writer of the cache
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_leaked_cycle_state():
    """`_CYCLE_COMPUTING` and the watermark are module globals; a leak here makes
    an unrelated test in another file believe a scan is running or decline the
    next scheduled one. Cleared on both sides."""
    scan_cycle._CYCLE_COMPUTING.clear()
    scan_cycle._last_pass_watermark = None
    scan_cycle._last_pass_at = None
    yield
    scan_cycle._CYCLE_COMPUTING.clear()
    scan_cycle._last_pass_watermark = None
    scan_cycle._last_pass_at = None


def test_the_daemon_pass_refreshes_the_attribution_cache(monkeypatch, tmp_path):
    """THE DEFECT: the pass held the connection and the flags and never used
    them for this, so the driver's only writer was a command nothing tells the
    user to run."""
    cfg = TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))
    backend = type("_B", (), {"conn": object(), "close": lambda self: None})()
    calls: list[tuple] = []

    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(
        report_store, "recompute_now",
        lambda b, c, until=None, provenance=None: {"computed_at": "now"},
    )
    monkeypatch.setattr(report_store, "stored_report", lambda config: object())
    monkeypatch.setattr(
        scan_cycle, "_write_relearn_from",
        lambda report, config, factory, provenance=None: None,
    )
    monkeypatch.setattr(scan_cycle, "_refresh_rule_presence", lambda config: None)

    import tokenjam.core.optimize.cost_proposals as cp

    monkeypatch.setattr(
        cp, "recompute_cost_proposals",
        lambda b, c, until=None, report=None, provenance=None: None,
    )
    monkeypatch.setattr(
        "tokenjam.core.attribution_cache.refresh_attribution_cache",
        lambda conn, capture, path=None: calls.append((conn, capture)),
    )

    threads: list = []

    def _fake_thread(target=None, name=None, daemon=None):
        return type("_T", (), {"start": lambda _s: threads.append(target)})()

    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread)
    scan_cycle._trigger_analyzer_pass(lambda: backend, cfg, None)
    threads[0]()   # run the job body inline

    assert calls, "the daemon pass never refreshed the attribution cache"
    conn, capture = calls[0]
    assert conn is backend.conn, "the refresh must reuse the pass's own connection"
    assert capture is cfg.capture, "the refresh must be handed the [capture] flags"


def test_a_backend_without_a_connection_is_skipped_not_crashed(monkeypatch):
    """An in-memory / API backend has no `conn`; the leg is a no-op, and a
    no-op must not become an exception on the daemon's only scan thread."""
    called: list = []
    monkeypatch.setattr(
        "tokenjam.core.attribution_cache.refresh_attribution_cache",
        lambda *a, **k: called.append(a),
    )
    scan_cycle._refresh_attribution_cache(object(), TjConfig(version="1"))
    assert called == []


# --------------------------------------------------------------------------- #
# 2. Stale renders, and renders DIFFERENTLY from absent
# --------------------------------------------------------------------------- #
def test_a_stale_cache_still_names_the_driver_with_an_age_marker(tmp_path):
    _write_cache(tmp_path, age_days=20)
    line = _line(tmp_path)
    assert "CLAUDE.md ×14" in line, (
        "a driver we still hold must be shown, not silently dropped"
    )
    assert "20d old" in line, "a dated driver must be marked as dated"


def test_stale_and_absent_are_distinguishable_renders(tmp_path):
    """The pin the issue asks for. Both used to collapse to the same bare
    percentage, so the reader could not tell one from the other."""
    _write_cache(tmp_path, age_days=20)
    stale_line = _line(tmp_path)
    (tmp_path / "attribution_cache.json").unlink()
    absent_line = _line(tmp_path)

    assert stale_line != absent_line
    assert "CLAUDE.md" not in absent_line
    assert "d old" not in absent_line


def test_a_fresh_cache_renders_unmarked(tmp_path):
    """The non-degraded case is byte-for-byte as shipped: no marker, no noise."""
    _write_cache(tmp_path, age_days=1)
    line = _line(tmp_path)
    assert "(CLAUDE.md ×14)" in line
    assert "old" not in line


def test_a_stale_static_driver_does_not_condition_the_remedy(tmp_path):
    """A driver too old to present as current is too old to pick a remedy from,
    so the nudge falls back to its driver-agnostic default. (The nudge copy
    itself is `_nudge_for`'s; this only pins that the stale path stops feeding
    it an inclusion type.)"""
    _write_cache(tmp_path, age_days=20)          # inclusion_type "file_read"
    stale_line = _line(tmp_path)
    (tmp_path / "attribution_cache.json").unlink()
    absent_line = _line(tmp_path)
    stale_nudge = stale_line.split("  ")[-1]
    assert stale_nudge == absent_line.split("  ")[-1]


def test_the_boundary_is_the_documented_ttl(tmp_path):
    """Just inside the window is fresh, just outside is stale — pinned against
    the constant rather than a hard-coded 7 so a TTL change stays coherent."""
    ttl_days = MAX_CACHE_AGE_SECONDS / 86400.0
    _write_cache(tmp_path, age_days=ttl_days - 0.05)
    assert "old" not in _line(tmp_path)
    _write_cache(tmp_path, age_days=ttl_days + 0.05)
    assert "old" in _line(tmp_path)


def test_an_unprovable_age_is_absent_not_stale(tmp_path):
    """No usable `computed_at` means we cannot say how dated it is, so there is
    nothing honest to render — the one case that stays hidden."""
    (tmp_path / "attribution_cache.json").write_text(json.dumps({
        "top_label": "CLAUDE.md", "occurrences": 14, "sessions": 3,
    }))
    line = _line(tmp_path)
    assert "CLAUDE.md" not in line
    assert "old" not in line
